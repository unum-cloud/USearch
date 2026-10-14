//! The dense modality - Vector Search over equidimensional embeddings.
//!
//! `DenseIndex` is the handle; `RowsView` is the batch currency every verb reads, one vector being
//! a one-row run, and every output whose length scales with the call lands in an `Arena` the verb
//! requires and comes back as a view over it. `DenseCursor` walks the contents in storage order,
//! and `DenseSearchStream` streams nearest neighbors of one query without a "k" chosen upfront.
//!
//! The receiver chooses where a verb runs and how it completes, and the verb stays the same: the
//! index runs it on the calling thread, `index.on(executor)` blocks on an executor, and
//! `index.deferred(executor, &call)` submits it there into a `Deferred` the caller holds. Optional
//! knobs, the progress handler and the filtering predicate among them, ride one config struct per verb.

const std = @import("std");

const types = @import("types.zig");
const Arena = types.Arena;
const BytesView = types.BytesView;
const Completion = types.Completion;
const Deferred = types.Deferred;
const Distance = types.Distance;
const Error = types.Error;
const Executor = types.Executor;
const Key = types.Key;
const Matches = types.Matches;
const MatchesRow = types.MatchesRow;
const Outcome = types.Outcome;
const Predicate = types.Predicate;
const PredicateFunction = types.PredicateFunction;
const ProgressCallback = types.ProgressCallback;
const ProgressFunction = types.ProgressFunction;
const Residency = types.Residency;
const RowsView = types.RowsView;
const Scalar = types.Scalar;
const Status = types.Status;
const Traversal = types.Traversal;
const landed = types.landed;
const refuse = types.refuse;
const refuseInto = types.refuseInto;
const usearch_keys_view_t = types.usearch_keys_view_t;
const usearch_matches_tape_view_t = types.usearch_matches_tape_view_t;
const usearch_matches_view_t = types.usearch_matches_view_t;
const usearch_nvme_options_t = types.usearch_nvme_options_t;
const usearch_progress_t = types.usearch_progress_t;
const usearch_ram_options_t = types.usearch_ram_options_t;
const usearch_rows_view_t = types.usearch_rows_view_t;
const usearch_vram_options_t = types.usearch_vram_options_t;

/// Mirrors `usearch_value_lookup_t`.
const ValueLookupFunction = *const fn (?*anyopaque, [*]const Key, usize, Scalar, usize, [*]?*const anyopaque) callconv(.c) void;

/// Where `attend` reads value rows, as the C interface carries it: a `usearch_value_lookup_t` and
/// the state forwarded into it. Build one with `ValueLookup.of`.
pub const ValueLookup = struct {
    callback: ?ValueLookupFunction = null,
    state: ?*anyopaque = null,

    /// Wraps `resolve` in a trampoline generated per call site, which answers one key's value row
    /// of `Element`, or null where it has none. Pass `{}` for a lookup needing no context, and a
    /// single-item pointer otherwise.
    ///
    /// A row must be as wide as the keys, stored in the index's quantization, and alive until the
    /// call is final. The trampoline checks the scalar and columns each batch states, so a null
    /// row, or one of another kind or width, refuses the call with `Error.ShapeMismatch`. It runs
    /// concurrently from the executor's threads.
    pub fn of(
        comptime Element: type,
        context: anytype,
        comptime resolve: fn (@TypeOf(context), Key) ?[]const Element,
    ) ValueLookup {
        const Context = @TypeOf(context);
        types.checkContext(Context);
        return .{
            .callback = struct {
                fn lookup(state: ?*anyopaque, keys: [*]const Key, count: usize, scalar: Scalar, columns: usize, rows: [*]?*const anyopaque) callconv(.c) void {
                    const per_value = comptime Scalar.of(Element).dimensionsPerValue();
                    const values: ?usize = if (scalar == comptime Scalar.of(Element)) (columns + per_value - 1) / per_value else null;
                    for (keys[0..count], rows[0..count]) |key, *row| {
                        const found = if (Context == void) resolve({}, key) else resolve(@ptrCast(@alignCast(state.?)), key);
                        row.* = if (found != null and values == found.?.len) found.?.ptr else null;
                    }
                }
            }.lookup,
            .state = types.contextPointer(Context, context),
        };
    }
};

extern fn usearch_dense_metric_name(metric: DenseMetric) [*:0]const u8;
extern fn usearch_dense_metric_from_name(name: BytesView) DenseMetric;

/// The dense metric kinds shipped out of the box, all derived from one inner-product kernel with
/// pre-computed magnitudes. The bit-level pair reads its rows as `Scalar.u1x8` and refuses any
/// other quantization; the first three refuse that one. Mirrors `usearch_dense_metric_t`.
///
/// Non-exhaustive, so a kind a newer engine reports arrives intact rather than coerced.
pub const DenseMetric = enum(c_int) {
    /// No kind named - an unset field, refused by every constructor.
    unknown = 0,
    /// Sum of elementwise products, unnormalized - the only kind the linear-algebra verbs accept.
    inner = 1,
    /// Cosine distance in [0, 2], blind to magnitude.
    angular = 2,
    /// L2 distance, zero only for identical rows.
    euclidean = 3,
    /// Count of differing bits, so its range is the dimensionality.
    hamming = 4,
    /// Differing bits over union bits in [0, 1], ignoring the zeros both rows share.
    jaccard = 5,
    _,

    /// The canonical spelling, as `.inner` spells "inner". Calls `usearch_dense_metric_name`.
    pub fn name(self: DenseMetric) [:0]const u8 {
        return std.mem.span(usearch_dense_metric_name(self));
    }

    /// Reads a spelling back, or null when no kind spells it. Calls `usearch_dense_metric_from_name`.
    pub fn fromName(text: []const u8) ?DenseMetric {
        const value = usearch_dense_metric_from_name(.of(text));
        return if (std.mem.eql(u8, value.name(), text)) value else null;
    }
};

/// The geometry and metric an empty index freezes - what `DenseIndex.initRam` and `initVram` read
/// beside their placement. Binding-only.
///
/// Stored rows round up to whole steps of the distance kit the build picked, so the memory is
/// slightly larger than the logical rows, and that is not a knob.
pub const DenseOptions = struct {
    /// The metric kind used for distance calculations between vectors.
    metric: DenseMetric = .angular,
    /// The scalar type vectors are stored and quantized into.
    quantization: Scalar = .bf16,
    /// Dimensions of the indexed vectors.
    dimensions: usize = 0,
    /// The optional connections-per-node bound; zero derives.
    connectivity: usize = 0,
    /// Upper bound on entries sharing one key; zero derives to one - unique keys.
    values_per_key: usize = 0,
};

/// The geometry and metric a checkpoint froze - everything a reader learns without opening it, and
/// nothing a reader must choose for itself. Mirrors `usearch_dense_metadata_t`.
///
/// `values_per_key` is absent because it bounds a live handle's mutable state rather than anything
/// the serialized form carries, and the placement is absent because a checkpoint records where it
/// was written and never where a reader must keep it. Both stay the caller's to set when loading.
pub const DenseMetadata = extern struct {
    /// The metric kind the checkpoint scores with.
    metric: DenseMetric = .unknown,
    /// The scalar type its rows are stored and quantized into.
    quantization: Scalar = .unknown,
    /// Scalars per stored vector.
    dimensions: usize = 0,
    /// Connections per node the graph was built with.
    connectivity: usize = 0,
};

/// A live index's state, reported by `DenseIndex.describe`. Each counter is read once, so under
/// concurrent insertions they may come from slightly different moments.
/// Mirrors `usearch_dense_index_description_t`.
pub const DenseIndexDescription = struct {
    /// The frozen metric, quantization, dimensions and connectivity.
    metadata: DenseMetadata,
    /// Stored entries.
    size: usize,
    /// Entries hosted without growing again.
    capacity: usize,
    /// Bytes the handle holds; traversal scratch is the executor's.
    memory_usage: usize,
    /// Codename of the kernels the index dispatched to on this machine; empty where none was named.
    capabilities: [:0]const u8,
};

/// Traversal telemetry, fully overwritten by the call it is passed to.
/// Mirrors `usearch_dense_traversal_stats_t`.
pub const DenseTraversalStats = extern struct {
    /// Graph members visited while resolving the batch.
    visited_members: usize = 0,
    /// Distances computed while resolving the batch, diversity re-selections included.
    computed_distances: usize = 0,
    /// Starvation restarts - a graph-connectivity health signal rather than a cost counter.
    restarts: usize = 0,
};

/// One reachability audit: the population it judged, and how much of it a search reaches.
/// Mirrors `usearch_dense_reachability_t`.
pub const DenseReachability = extern struct {
    /// Entries published when the audit started; compare `reachable` against this, never a later size.
    audited: usize = 0,
    /// Of those, the entries a search reaches from the landmarks; the rest wait for a `refine`.
    reachable: usize = 0,
    /// Levels the sweep took to converge - a diameter proxy.
    sweeps: usize = 0,
};

/// The online-softmax carry of one query row - Flash Attention's running maximum and partition.
/// Mirrors `usearch_dense_softmax_state_t`.
///
/// The two are one concept: a maximum without its partition is a number nothing can rescale, so
/// they share an element rather than travelling as two arrays a caller could supply half of.
pub const DenseSoftmaxState = extern struct {
    /// The largest logit seen so far - the recurrence's `m`.
    maximum_score: f32 = 0,
    /// The sum of exponentials taken against that maximum - the recurrence's `l`.
    partition: f32 = 0,
};

/// Optional settings of `DenseIndex.loadRam`.
pub const DenseLoadRamConfig = struct {
    /// The executor the read fans out over; null reads on the calling thread.
    executor: ?Executor = null,
    /// The memory domains to place the parts on; empty derives every node the process may use.
    memory_domains: []const i32 = &.{},
    /// Upper bound on entries sharing one key, which no checkpoint records; zero derives to one.
    values_per_key: usize = 0,
    /// Called with shards placed; a stopped load creates no index.
    progress: ProgressCallback = .{},
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    progress_steps: usize = 0,
    /// Where the call's final outcome, its static message included, is copied; null copies nothing.
    diagnostics: ?*Outcome = null,
};

/// Optional settings of `DenseIndex.loadVram`.
pub const DenseLoadVramConfig = struct {
    /// The executor the upload runs on; null runs on the calling thread and the default stream.
    executor: ?Executor = null,
    /// The devices to place the parts on; empty derives one free device.
    devices: []const i32 = &.{},
    /// Upper bound on entries sharing one key, which no checkpoint records; zero derives to one.
    values_per_key: usize = 0,
    /// Called with shards placed; a stopped load creates no index.
    progress: ProgressCallback = .{},
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    progress_steps: usize = 0,
    /// Where the call's final outcome, its static message included, is copied; null copies nothing.
    diagnostics: ?*Outcome = null,
};

/// Optional settings of `DenseIndex.loadRamBuffer`.
pub const DenseLoadRamBufferConfig = struct {
    /// The executor the copy fans out over; null copies on the calling thread.
    executor: ?Executor = null,
    /// The memory domains to place shards on; the monolith lands whole on the first.
    memory_domains: []const i32 = &.{},
    /// Upper bound on entries sharing one key, which no checkpoint records; zero derives to one.
    values_per_key: usize = 0,
    /// Called with bytes placed; a stopped load creates no index.
    progress: ProgressCallback = .{},
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    progress_steps: usize = 0,
    /// Where the call's final outcome, its static message included, is copied; null copies nothing.
    diagnostics: ?*Outcome = null,
};

/// Optional settings of `DenseIndex.loadVramBuffer`.
pub const DenseLoadVramBufferConfig = struct {
    /// The executor the upload runs on; null runs on the calling thread and the default stream.
    executor: ?Executor = null,
    /// The devices to place shards on; the monolith lands whole on the first.
    devices: []const i32 = &.{},
    /// Upper bound on entries sharing one key, which no checkpoint records; zero derives to one.
    values_per_key: usize = 0,
    /// Called with bytes placed; a stopped load creates no index.
    progress: ProgressCallback = .{},
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    progress_steps: usize = 0,
    /// Where the call's final outcome, its static message included, is copied; null copies nothing.
    diagnostics: ?*Outcome = null,
};

/// Optional settings of `DenseIndex.viewNvme`.
pub const DenseViewNvmeConfig = struct {
    /// The NVMe executor whose rings read the checkpoint's headers; null reads synchronously.
    executor: ?Executor = null,
    /// The devices the parts are served from; empty serves each where its manifest points.
    paths: []const BytesView = &.{},
    /// Where the call's final outcome, its static message included, is copied; null copies nothing.
    diagnostics: ?*Outcome = null,
};

/// Optional settings of `DenseIndex.metadataRead`.
pub const DenseMetadataReadConfig = struct {
    /// The executor the read runs on; null reads on the calling thread.
    executor: ?Executor = null,
    /// Where the call's final outcome, its static message included, is copied; null copies nothing.
    diagnostics: ?*Outcome = null,
};

/// Optional settings of `DenseIndex.metadataReadBuffer`.
pub const DenseMetadataReadBufferConfig = struct {
    /// The executor the read runs on; null reads on the calling thread.
    executor: ?Executor = null,
    /// Where the call's final outcome, its static message included, is copied; null copies nothing.
    diagnostics: ?*Outcome = null,
};

/// Optional settings of `DenseIndex.insert`.
pub const DenseInsertConfig = struct {
    /// The candidate-pool breadth the build walk explores; zero derives.
    expansion_factor: usize = 0,
    /// Called with rows admitted; it reports and cannot cancel, since every row is claimed first.
    progress: ProgressCallback = .{},
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    progress_steps: usize = 0,
    stats: ?*DenseTraversalStats = null,
};

/// Optional settings of `DenseIndex.refine`.
pub const DenseRefineConfig = struct {
    /// The candidate-pool breadth the re-selection walk explores; zero derives.
    expansion_factor: usize = 0,
    /// Called with nodes swept; a stop is a time budget rather than a failure.
    progress: ProgressCallback = .{},
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    progress_steps: usize = 0,
    stats: ?*DenseTraversalStats = null,
};

/// Optional settings of `DenseIndex.search`.
pub const DenseSearchConfig = struct {
    /// The candidate-pool breadth the walk explores; zero derives.
    expansion_factor: usize = 0,
    /// A walk, or an exact scan of every row on RAM and NVMe handles; VRAM answers `Error.Unsupported`.
    traversal: Traversal = .walk,
    /// Admits only the keys it approves; non-matching entries still route the walk. Refused on VRAM.
    predicate: Predicate = .{},
    /// Called with queries resolved; a stopped search answers `Error.Cancelled` and no matches.
    progress: ProgressCallback = .{},
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    progress_steps: usize = 0,
    stats: ?*DenseTraversalStats = null,
};

/// Optional settings of `DenseIndex.merge`.
pub const DenseMergeConfig = struct {
    /// Called with donors absorbed; a stop keeps the donors already absorbed.
    progress: ProgressCallback = .{},
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    progress_steps: usize = 0,
};

/// Optional settings of `DenseIndex.multiply`.
pub const DenseMultiplyConfig = struct {
    /// Bound on pairs kept per row, the first ones its walk scored; zero keeps every pair.
    wanted: usize = 0,
    /// The candidate-pool breadth, which widens the emitted pattern; zero derives.
    expansion_factor: usize = 0,
    progress: ProgressCallback = .{},
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    progress_steps: usize = 0,
    stats: ?*DenseTraversalStats = null,
};

/// Optional settings of `DenseIndex.attend`.
pub const DenseAttendConfig = struct {
    /// The query heads sharing the index's key-value head, and with it one walk per token: row
    /// `t · head_count + h` is head `h` of token `t`. It must divide the rows; zero means one.
    head_count: usize = 0,
    /// The softmax state, one per query row, lent rather than landed in the arena; null skips it.
    carry: ?[]DenseSoftmaxState = null,
    /// The scalar type the attended rows land as; `.unknown` takes the queries' own.
    outputs_scalar: Scalar = .unknown,
    /// The candidate-pool breadth; zero derives. A wider pool walks further, folding more of the tail.
    expansion_factor: usize = 0,
    /// The logit scale, conventionally the inverse square root of the head dimensionality; zero
    /// derives it from the index's frozen geometry.
    score_scale: f32 = 0,
    progress: ProgressCallback = .{},
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    progress_steps: usize = 0,
    stats: ?*DenseTraversalStats = null,
};

/// Optional settings of `DenseIndex.maxsim`.
pub const DenseMaxsimConfig = struct {
    /// The candidate-pool breadth; zero derives.
    expansion_factor: usize = 0,
    progress: ProgressCallback = .{},
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    progress_steps: usize = 0,
    stats: ?*DenseTraversalStats = null,
};

/// Optional settings of `DenseIndex.save`.
pub const DenseSaveConfig = struct {
    /// Called with shards written; a stopped save leaves leftovers every reader rejects.
    progress: ProgressCallback = .{},
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    progress_steps: usize = 0,
};

/// Optional settings of `DenseIndex.saveBuffer`.
pub const DenseSaveBufferConfig = struct {
    /// Called with bytes written; a stopped save leaves no bytes.
    progress: ProgressCallback = .{},
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    progress_steps: usize = 0,
};

/// Optional settings of `DenseSearchStream.init`.
pub const DenseSearchStreamInitConfig = struct {
    /// The candidate-pool breadth the walk explores; zero derives.
    expansion_factor: usize = 0,
    /// Where the call's final outcome, its static message included, is copied; null copies nothing.
    diagnostics: ?*Outcome = null,
};

const usearch_dense_init_ram_t = extern struct {
    index: *?*anyopaque,
    outcome: *Outcome,
    metric: DenseMetric,
    quantization: Scalar,
    dimensions: usize,
    connectivity: usize,
    values_per_key: usize,
    ram: usearch_ram_options_t,
};

const usearch_dense_init_vram_t = extern struct {
    index: *?*anyopaque,
    outcome: *Outcome,
    metric: DenseMetric,
    quantization: Scalar,
    dimensions: usize,
    connectivity: usize,
    values_per_key: usize,
    vram: usearch_vram_options_t,
};

const usearch_dense_load_ram_t = extern struct {
    index: *?*anyopaque,
    executor: ?*anyopaque,
    outcome: *Outcome,
    completion: Completion,
    path: BytesView,
    ram: usearch_ram_options_t,
    values_per_key: usize,
    progress: usearch_progress_t,
};

const usearch_dense_load_vram_t = extern struct {
    index: *?*anyopaque,
    executor: ?*anyopaque,
    outcome: *Outcome,
    completion: Completion,
    path: BytesView,
    vram: usearch_vram_options_t,
    values_per_key: usize,
    progress: usearch_progress_t,
};

const usearch_dense_load_ram_buffer_t = extern struct {
    index: *?*anyopaque,
    executor: ?*anyopaque,
    outcome: *Outcome,
    completion: Completion,
    buffer: BytesView,
    ram: usearch_ram_options_t,
    values_per_key: usize,
    progress: usearch_progress_t,
};

const usearch_dense_load_vram_buffer_t = extern struct {
    index: *?*anyopaque,
    executor: ?*anyopaque,
    outcome: *Outcome,
    completion: Completion,
    buffer: BytesView,
    vram: usearch_vram_options_t,
    values_per_key: usize,
    progress: usearch_progress_t,
};

const usearch_dense_view_ram_t = extern struct {
    index: *?*anyopaque,
    outcome: *Outcome,
    path: BytesView,
};

const usearch_dense_view_nvme_t = extern struct {
    index: *?*anyopaque,
    executor: ?*anyopaque,
    outcome: *Outcome,
    completion: Completion,
    path: BytesView,
    nvme: usearch_nvme_options_t,
};

const usearch_dense_reserve_t = extern struct {
    index: ?*anyopaque,
    executor: ?*anyopaque,
    outcome: *Outcome,
    completion: Completion,
    capacity: usize,
};

const usearch_dense_clear_t = extern struct {
    index: ?*anyopaque,
    executor: ?*anyopaque,
    outcome: *Outcome,
    completion: Completion,
};

const usearch_dense_insert_t = extern struct {
    index: ?*anyopaque,
    executor: ?*anyopaque,
    outcome: *Outcome,
    completion: Completion,
    count: usize,
    keys: usearch_keys_view_t,
    vectors: usearch_rows_view_t,
    expansion_factor: usize,
    progress: usearch_progress_t,
    stats: ?*DenseTraversalStats,
};

const usearch_dense_refine_t = extern struct {
    index: ?*anyopaque,
    executor: ?*anyopaque,
    outcome: *Outcome,
    completion: Completion,
    changed: ?*usize,
    expansion_factor: usize,
    progress: usearch_progress_t,
    stats: ?*DenseTraversalStats,
};

const usearch_dense_health_t = extern struct {
    index: ?*anyopaque,
    executor: ?*anyopaque,
    outcome: *Outcome,
    completion: Completion,
    reachability: *DenseReachability,
};

const usearch_dense_search_t = extern struct {
    index: ?*anyopaque,
    executor: ?*anyopaque,
    outcome: *Outcome,
    completion: Completion,
    count: usize,
    queries: usearch_rows_view_t,
    matches: *usearch_matches_view_t,
    arena: ?*anyopaque,
    wanted: usize,
    expansion_factor: usize,
    traversal: Traversal,
    predicate: Predicate,
    progress: usearch_progress_t,
    stats: ?*DenseTraversalStats,
};

const usearch_dense_count_t = extern struct {
    index: ?*anyopaque,
    executor: ?*anyopaque,
    outcome: *Outcome,
    completion: Completion,
    key: Key,
    entries: *usize,
};

const usearch_dense_get_t = extern struct {
    index: ?*anyopaque,
    executor: ?*anyopaque,
    outcome: *Outcome,
    completion: Completion,
    key: Key,
    vectors: *usearch_rows_view_t,
    written: ?*usize,
    arena: ?*anyopaque,
    outputs_scalar: Scalar,
};

const usearch_dense_merge_t = extern struct {
    index: ?*anyopaque,
    executor: ?*anyopaque,
    outcome: *Outcome,
    completion: Completion,
    count: usize,
    donors: [*]?*anyopaque,
    progress: usearch_progress_t,
};

const usearch_dense_cursor_init_t = extern struct {
    cursor: *?*anyopaque,
    index: ?*anyopaque,
    outcome: *Outcome,
};

const usearch_dense_cursor_next_t = extern struct {
    index: ?*anyopaque,
    executor: ?*anyopaque,
    outcome: *Outcome,
    completion: Completion,
    cursor: ?*anyopaque,
    keys: *usearch_keys_view_t,
    vectors: ?*usearch_rows_view_t,
    written: *usize,
    arena: ?*anyopaque,
    wanted: usize,
    outputs_scalar: Scalar,
};

const usearch_dense_search_stream_init_t = extern struct {
    stream: *?*anyopaque,
    index: ?*anyopaque,
    outcome: *Outcome,
    query: usearch_rows_view_t,
    expansion_factor: usize,
};

const usearch_dense_search_stream_next_t = extern struct {
    index: ?*anyopaque,
    executor: ?*anyopaque,
    outcome: *Outcome,
    completion: Completion,
    stream: ?*anyopaque,
    matches: *usearch_matches_view_t,
    arena: ?*anyopaque,
    wanted: usize,
};

const usearch_dense_save_t = extern struct {
    index: ?*anyopaque,
    executor: ?*anyopaque,
    outcome: *Outcome,
    completion: Completion,
    path: BytesView,
    progress: usearch_progress_t,
};

const usearch_dense_save_buffer_t = extern struct {
    index: ?*anyopaque,
    executor: ?*anyopaque,
    outcome: *Outcome,
    completion: Completion,
    monolith: *BytesView,
    arena: ?*anyopaque,
    progress: usearch_progress_t,
};

const usearch_dense_metadata_read_t = extern struct {
    executor: ?*anyopaque,
    outcome: *Outcome,
    completion: Completion,
    path: BytesView,
    metadata: *DenseMetadata,
};

const usearch_dense_metadata_read_buffer_t = extern struct {
    executor: ?*anyopaque,
    outcome: *Outcome,
    completion: Completion,
    buffer: BytesView,
    metadata: *DenseMetadata,
};

const usearch_dense_index_description_t = extern struct {
    metadata: DenseMetadata = .{},
    size: usize = 0,
    capacity: usize = 0,
    memory_usage: usize = 0,
    capabilities: ?[*:0]const u8 = null,
};

const usearch_dense_describe_t = extern struct {
    index: ?*anyopaque,
    executor: ?*anyopaque,
    outcome: *Outcome,
    completion: Completion,
    description: *usearch_dense_index_description_t,
};

const usearch_dense_multiply_t = extern struct {
    index: ?*anyopaque,
    executor: ?*anyopaque,
    outcome: *Outcome,
    completion: Completion,
    count: usize,
    left: usearch_rows_view_t,
    products: *usearch_matches_tape_view_t,
    written: ?*usize,
    required: ?*usize,
    arena: ?*anyopaque,
    wanted: usize,
    expansion_factor: usize,
    progress: usearch_progress_t,
    stats: ?*DenseTraversalStats,
};

const usearch_dense_attend_t = extern struct {
    index: ?*anyopaque,
    executor: ?*anyopaque,
    outcome: *Outcome,
    completion: Completion,
    count: usize,
    queries: usearch_rows_view_t,
    head_count: usize,
    value_lookup: ?ValueLookupFunction,
    value_lookup_state: ?*anyopaque,
    carry: ?[*]DenseSoftmaxState,
    outputs: *usearch_rows_view_t,
    arena: ?*anyopaque,
    outputs_scalar: Scalar,
    expansion_factor: usize,
    score_scale: f32,
    progress: usearch_progress_t,
    stats: ?*DenseTraversalStats,
};

const usearch_dense_maxsim_t = extern struct {
    index: ?*anyopaque,
    executor: ?*anyopaque,
    outcome: *Outcome,
    completion: Completion,
    count: usize,
    queries: usearch_rows_view_t,
    rows_per_query: usize,
    matches: *usearch_matches_view_t,
    arena: ?*anyopaque,
    wanted: usize,
    expansion_factor: usize,
    progress: usearch_progress_t,
    stats: ?*DenseTraversalStats,
};

extern fn usearch_dense_init_ram(arguments: *usearch_dense_init_ram_t) void;
extern fn usearch_dense_init_vram(arguments: *usearch_dense_init_vram_t) void;
extern fn usearch_dense_load_ram(arguments: *usearch_dense_load_ram_t) void;
extern fn usearch_dense_load_vram(arguments: *usearch_dense_load_vram_t) void;
extern fn usearch_dense_load_ram_buffer(arguments: *usearch_dense_load_ram_buffer_t) void;
extern fn usearch_dense_load_vram_buffer(arguments: *usearch_dense_load_vram_buffer_t) void;
extern fn usearch_dense_view_ram(arguments: *usearch_dense_view_ram_t) void;
extern fn usearch_dense_view_nvme(arguments: *usearch_dense_view_nvme_t) void;
extern fn usearch_dense_free(index: ?*anyopaque) void;
extern fn usearch_dense_reserve(arguments: *usearch_dense_reserve_t) void;
extern fn usearch_dense_clear(arguments: *usearch_dense_clear_t) void;
extern fn usearch_dense_insert(arguments: *usearch_dense_insert_t) void;
extern fn usearch_dense_refine(arguments: *usearch_dense_refine_t) void;
extern fn usearch_dense_health(arguments: *usearch_dense_health_t) void;
extern fn usearch_dense_search(arguments: *usearch_dense_search_t) void;
extern fn usearch_dense_count(arguments: *usearch_dense_count_t) void;
extern fn usearch_dense_get(arguments: *usearch_dense_get_t) void;
extern fn usearch_dense_merge(arguments: *usearch_dense_merge_t) void;
extern fn usearch_dense_cursor_init(arguments: *usearch_dense_cursor_init_t) void;
extern fn usearch_dense_cursor_next(arguments: *usearch_dense_cursor_next_t) void;
extern fn usearch_dense_cursor_free(cursor: ?*anyopaque) void;
extern fn usearch_dense_search_stream_init(arguments: *usearch_dense_search_stream_init_t) void;
extern fn usearch_dense_search_stream_next(arguments: *usearch_dense_search_stream_next_t) void;
extern fn usearch_dense_search_stream_free(stream: ?*anyopaque) void;
extern fn usearch_dense_save(arguments: *usearch_dense_save_t) void;
extern fn usearch_dense_save_buffer(arguments: *usearch_dense_save_buffer_t) void;
extern fn usearch_dense_metadata_read(arguments: *usearch_dense_metadata_read_t) void;
extern fn usearch_dense_metadata_read_buffer(arguments: *usearch_dense_metadata_read_buffer_t) void;
extern fn usearch_dense_describe(arguments: *usearch_dense_describe_t) void;
extern fn usearch_dense_multiply(arguments: *usearch_dense_multiply_t) void;
extern fn usearch_dense_attend(arguments: *usearch_dense_attend_t) void;
extern fn usearch_dense_maxsim(arguments: *usearch_dense_maxsim_t) void;

/// Vector Search Index over equidimensional dense representations.
///
/// A handle is shareable across threads: searches never write to the graph, and insertions
/// synchronize through per-node locks and publish in order. It is the `Executor` that orders,
/// never the index.
///
/// ```zig
/// const usearch = @import("usearch");
///
/// var index = try usearch.DenseIndex.initRam(.{ .dimensions = 4, .metric = .angular, .quantization = .i8 }, &.{});
/// defer index.deinit();
/// try index.reserve(1_000);
///
/// const keys = [_]usearch.Key{ .fromInteger(0), .fromInteger(1) };
/// const slab = [_]f32{ 0.2, 0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0.2 };
/// try index.insert(&keys, try .of(f32, &slab, 4), .{});
///
/// const query = [_]f32{ 0.2, 0.1, 0.2, 0.1 };
/// var arena = try usearch.Arena.init(null);
/// defer arena.deinit();
/// const matches = try index.search(try .ofRow(f32, &query), 10, &arena, .{});
/// const nearest = (try matches.row(0)).keys[0];
/// ```
pub const DenseIndex = extern struct {
    /// The opaque `usearch_dense_index_t` the ABI hands back.
    handle: ?*anyopaque,

    /// Every verb of the index, blocking on an executor after every call submitted there before it.
    pub const On = Receiver(.blocking, void);

    /// Every verb of the index, submitted to an executor into a `Deferred(Call)` the caller holds,
    /// borrowing the arena a verb lands in as a blocking call does; only the verb `Call` describes compiles.
    pub fn DeferredOn(comptime Call: type) type {
        return Receiver(.deferred, Call);
    }

    /// Runs every verb on `executor`, whose kind must match the index's residency. Binding-only.
    pub fn on(self: DenseIndex, executor: Executor) On {
        return .{ .index = self.handle, .executor = executor.handle, .into = {}, .diagnostics = null };
    }

    /// Submits a verb to `executor`, holding it in `call` - a `*Deferred(Call)` the caller places
    /// anywhere, which stays put until its `deinit`. Binding-only.
    pub fn deferred(self: DenseIndex, executor: Executor, call: anytype) DeferredOn(@FieldType(@TypeOf(call.*), "call")) {
        return .{ .index = self.handle, .executor = executor.handle, .into = call, .diagnostics = {} };
    }

    /// Runs every verb on the calling thread, copying each call's final outcome into `outcome`. Binding-only.
    pub fn diagnosed(self: DenseIndex, outcome: *Outcome) On {
        return self.here().diagnosed(outcome);
    }

    /// The receiver every verb called on the index itself runs through: no executor, the calling thread.
    fn here(self: DenseIndex) On {
        return .{ .index = self.handle, .executor = null, .into = {}, .diagnostics = null };
    }

    /// Creates an empty, mutable index in host memory, one shard per memory domain; empty derives
    /// every node the process may allocate on. Calls `usearch_dense_init_ram`.
    pub fn initRam(options: DenseOptions, memory_domains: []const i32) Error!DenseIndex {
        var handle: ?*anyopaque = null;
        var outcome: Outcome = .{};
        var arguments: usearch_dense_init_ram_t = .{
            .index = &handle,
            .outcome = &outcome,
            .metric = options.metric,
            .quantization = options.quantization,
            .dimensions = options.dimensions,
            .connectivity = options.connectivity,
            .values_per_key = options.values_per_key,
            .ram = .of(memory_domains),
        };
        usearch_dense_init_ram(&arguments);
        return adopt(handle, outcome, null);
    }

    /// Creates an empty, mutable index in device memory, one shard per device and never a replica;
    /// empty derives one free device. Calls `usearch_dense_init_vram`.
    pub fn initVram(options: DenseOptions, devices: []const i32) Error!DenseIndex {
        var handle: ?*anyopaque = null;
        var outcome: Outcome = .{};
        var arguments: usearch_dense_init_vram_t = .{
            .index = &handle,
            .outcome = &outcome,
            .metric = options.metric,
            .quantization = options.quantization,
            .dimensions = options.dimensions,
            .connectivity = options.connectivity,
            .values_per_key = options.values_per_key,
            .vram = .of(devices),
        };
        usearch_dense_init_vram(&arguments);
        return adopt(handle, outcome, null);
    }

    /// Copies a checkpoint into a new, mutable index in host memory, shard by shard.
    ///
    /// Point `path` at the `.safegraphs.json` manifest, never at one of the parts it enumerates.
    /// The geometry and metric come from the checkpoint's own headers. Calls `usearch_dense_load_ram`.
    pub fn loadRam(path: []const u8, config: DenseLoadRamConfig) Error!DenseIndex {
        var handle: ?*anyopaque = null;
        var outcome: Outcome = .{};
        var arguments: usearch_dense_load_ram_t = .{
            .index = &handle,
            .executor = handleOf(config.executor),
            .outcome = &outcome,
            .completion = .blocking,
            .path = .of(path),
            .ram = .of(config.memory_domains),
            .values_per_key = config.values_per_key,
            .progress = .of(config.progress, config.progress_steps),
        };
        usearch_dense_load_ram(&arguments);
        return adopt(handle, outcome, config.diagnostics);
    }

    /// Copies a checkpoint into a new, mutable index in device memory, with the placement rules of
    /// `loadRam`. Calls `usearch_dense_load_vram`.
    pub fn loadVram(path: []const u8, config: DenseLoadVramConfig) Error!DenseIndex {
        var handle: ?*anyopaque = null;
        var outcome: Outcome = .{};
        var arguments: usearch_dense_load_vram_t = .{
            .index = &handle,
            .executor = handleOf(config.executor),
            .outcome = &outcome,
            .completion = .blocking,
            .path = .of(path),
            .vram = .of(config.devices),
            .values_per_key = config.values_per_key,
            .progress = .of(config.progress, config.progress_steps),
        };
        usearch_dense_load_vram(&arguments);
        return adopt(handle, outcome, config.diagnostics);
    }

    /// Copies a monolith from a memory buffer into a new index in host memory, as `loadRam` does a
    /// path. Calls `usearch_dense_load_ram_buffer`.
    pub fn loadRamBuffer(buffer: []const u8, config: DenseLoadRamBufferConfig) Error!DenseIndex {
        var handle: ?*anyopaque = null;
        var outcome: Outcome = .{};
        var arguments: usearch_dense_load_ram_buffer_t = .{
            .index = &handle,
            .executor = handleOf(config.executor),
            .outcome = &outcome,
            .completion = .blocking,
            .buffer = .of(buffer),
            .ram = .of(config.memory_domains),
            .values_per_key = config.values_per_key,
            .progress = .of(config.progress, config.progress_steps),
        };
        usearch_dense_load_ram_buffer(&arguments);
        return adopt(handle, outcome, config.diagnostics);
    }

    /// Copies a monolith from a host or device buffer into a new index in device memory.
    /// Calls `usearch_dense_load_vram_buffer`.
    pub fn loadVramBuffer(buffer: []const u8, config: DenseLoadVramBufferConfig) Error!DenseIndex {
        var handle: ?*anyopaque = null;
        var outcome: Outcome = .{};
        var arguments: usearch_dense_load_vram_buffer_t = .{
            .index = &handle,
            .executor = handleOf(config.executor),
            .outcome = &outcome,
            .completion = .blocking,
            .buffer = .of(buffer),
            .vram = .of(config.devices),
            .values_per_key = config.values_per_key,
            .progress = .of(config.progress, config.progress_steps),
        };
        usearch_dense_load_vram_buffer(&arguments);
        return adopt(handle, outcome, config.diagnostics);
    }

    /// Serves a checkpoint in place through a memory mapping, as a read-only index whose mutating
    /// verbs report `Error.ReadOnly`; the files must stay unmodified while it lives.
    /// Calls `usearch_dense_view_ram`.
    pub fn viewRam(path: []const u8) Error!DenseIndex {
        var handle: ?*anyopaque = null;
        var outcome: Outcome = .{};
        var arguments: usearch_dense_view_ram_t = .{ .index = &handle, .outcome = &outcome, .path = .of(path) };
        usearch_dense_view_ram(&arguments);
        return adopt(handle, outcome, null);
    }

    /// Serves a checkpoint in place from NVMe drives, read-only and sized past RAM; until NVMe
    /// checkpoints carry keys it answers `Error.Unsupported`. Calls `usearch_dense_view_nvme`.
    pub fn viewNvme(path: []const u8, config: DenseViewNvmeConfig) Error!DenseIndex {
        var handle: ?*anyopaque = null;
        var outcome: Outcome = .{};
        var arguments: usearch_dense_view_nvme_t = .{
            .index = &handle,
            .executor = handleOf(config.executor),
            .outcome = &outcome,
            .completion = .blocking,
            .path = .of(path),
            .nvme = .of(config.paths),
        };
        usearch_dense_view_nvme(&arguments);
        return adopt(handle, outcome, config.diagnostics);
    }

    /// Reads a checkpoint's geometry and metric, without loading it. Calls `usearch_dense_metadata_read`.
    pub fn metadataRead(path: []const u8, config: DenseMetadataReadConfig) Error!DenseMetadata {
        var found: DenseMetadata = .{};
        var outcome: Outcome = .{};
        var arguments: usearch_dense_metadata_read_t = .{
            .executor = handleOf(config.executor),
            .outcome = &outcome,
            .completion = .blocking,
            .path = .of(path),
            .metadata = &found,
        };
        usearch_dense_metadata_read(&arguments);
        try outcome.checkInto(config.diagnostics);
        return found;
    }

    /// Reads geometry and metric out of a serialized monolith buffer, without loading it.
    /// Calls `usearch_dense_metadata_read_buffer`.
    pub fn metadataReadBuffer(buffer: []const u8, config: DenseMetadataReadBufferConfig) Error!DenseMetadata {
        var found: DenseMetadata = .{};
        var outcome: Outcome = .{};
        var arguments: usearch_dense_metadata_read_buffer_t = .{
            .executor = handleOf(config.executor),
            .outcome = &outcome,
            .completion = .blocking,
            .buffer = .of(buffer),
            .metadata = &found,
        };
        usearch_dense_metadata_read_buffer(&arguments);
        try outcome.checkInto(config.diagnostics);
        return found;
    }

    fn adopt(handle: ?*anyopaque, outcome: Outcome, diagnostics: ?*Outcome) Error!DenseIndex {
        errdefer usearch_dense_free(handle);
        try outcome.checkInto(diagnostics);
        return .{ .handle = handle };
    }

    /// Frees the resources associated with the index, waiting for every deferred call naming it;
    /// the handle is invalid afterwards. Calls `usearch_dense_free`.
    pub fn deinit(self: *DenseIndex) void {
        usearch_dense_free(self.handle);
        self.* = undefined;
    }

    /// Reserves room for a total number of entries, the current size included.
    /// Calls `usearch_dense_reserve`.
    pub fn reserve(self: DenseIndex, capacity: usize) Error!void {
        return self.here().reserve(capacity);
    }

    /// Erases all entries, keeping the reserved capacity and frozen geometry. Never runs
    /// concurrently with any other verb on the same handle. Calls `usearch_dense_clear`.
    pub fn clear(self: DenseIndex) Error!void {
        return self.here().clear();
    }

    /// Reports the frozen metadata and the current counters in one call. Calls `usearch_dense_describe`.
    pub fn describe(self: DenseIndex) Error!DenseIndexDescription {
        return self.here().describe();
    }

    /// The entries stored under `key`; zero means the index holds none. Calls `usearch_dense_count`.
    pub fn count(self: DenseIndex, key: Key) Error!usize {
        return self.here().count(key);
    }

    /// Whether the index holds at least one entry under `key`. Binding-only; calls `usearch_dense_count`.
    pub fn contains(self: DenseIndex, key: Key) Error!bool {
        return self.here().contains(key);
    }

    /// Every vector stored under `key` as `scalar`, one packed row per entry in insertion order,
    /// landed in `arena`; an absent key yields no rows. Calls `usearch_dense_get`.
    pub fn get(self: DenseIndex, key: Key, scalar: Scalar, arena: *Arena) Error!DenseVectors {
        return self.here().get(key, scalar, arena);
    }

    /// Adds a run of vectors under their keys - the native batch operation, one key per row.
    ///
    /// The batch is all-or-nothing: every fallible step runs before the first entry identity is
    /// claimed, so a handler reports progress but cannot stop it. Overlaps safely with concurrent
    /// searches on the same handle, and the input scalar type is independent of the stored
    /// quantization. Calls `usearch_dense_insert`.
    pub fn insert(self: DenseIndex, keys: []const Key, vectors: RowsView, config: DenseInsertConfig) Error!void {
        return self.here().insert(keys, vectors, config);
    }

    /// Re-selects the neighbors of every stored entry against the finished graph, returning the
    /// nodes whose edges changed; zero means the graph has settled.
    ///
    /// A build step rather than a repair: a cheap insertion pass followed by one refinement
    /// reaches a better graph, in less total time, than one expensive insertion pass - so a wide
    /// expansion factor belongs here rather than in `insert`. A stopped sweep leaves the graph
    /// correct and merely less improved, and reports no `Error.Cancelled`. Calls `usearch_dense_refine`.
    pub fn refine(self: DenseIndex, config: DenseRefineConfig) Error!usize {
        return self.here().refine(config);
    }

    /// Audits how much of the graph a search reaches from its landmarks; the entries it misses stay
    /// unfound until a `refine` re-links them. Calls `usearch_dense_health`.
    pub fn health(self: DenseIndex) Error!DenseReachability {
        return self.here().health();
    }

    /// Resolves a run of queries into up to `wanted` matches apiece, landed in `arena` - the native
    /// batch operation.
    ///
    /// Never writes to the graph, so any number of these may overlap with concurrent insertions.
    /// Calls `usearch_dense_search`.
    pub fn search(self: DenseIndex, queries: RowsView, wanted: usize, arena: *Arena, config: DenseSearchConfig) Error!Matches {
        return self.here().search(queries, wanted, arena, config);
    }

    /// Absorbs every donor into the index. Each donor it absorbs is consumed and nulled in place, so
    /// `deinit` on every donor afterwards frees only those a stop left live. Calls `usearch_dense_merge`.
    pub fn merge(self: DenseIndex, donors: []DenseIndex, config: DenseMergeConfig) Error!void {
        return self.here().merge(donors, config);
    }

    /// Copies up to `wanted` more entries out of `cursor` into `arena` - their keys, and their rows
    /// as `vectors` unless null - answering an empty batch once the walk is exhausted.
    /// Calls `usearch_dense_cursor_next`.
    pub fn cursorNext(self: DenseIndex, cursor: DenseCursor, wanted: usize, vectors: ?Scalar, arena: *Arena) Error!DenseCursorBatch {
        return self.here().cursorNext(cursor, wanted, vectors, arena);
    }

    /// Pulls up to `wanted` more matches out of `stream` into `arena`, as one query's row, answering
    /// an empty row once the walk is drained.
    ///
    /// Ordering beyond the configured expansion horizon is best-effort nearest rather than
    /// globally exact, so pulling far past that breadth degrades the ranking rather than ending
    /// the stream. Calls `usearch_dense_search_stream_next`.
    pub fn searchStreamNext(self: DenseIndex, stream: DenseSearchStream, wanted: usize, arena: *Arena) Error!Matches {
        return self.here().searchStreamNext(stream, wanted, arena);
    }

    /// Saves the index under `path` in the layout its extension picks - a `.safegraphs` monolith,
    /// or a `.safegraphs.json` manifest beside its shards.
    ///
    /// Demands a mutation-free window; concurrent readers keep searching throughout. A failed
    /// save leaves a file without its header, which every reader rejects, and that leftover is
    /// the caller's to remove. Calls `usearch_dense_save`.
    pub fn save(self: DenseIndex, path: []const u8, config: DenseSaveConfig) Error!void {
        return self.here().save(path, config);
    }

    /// Saves the index as one monolith - the same bytes `save` writes to a `.safegraphs` path -
    /// landed in `arena`. Calls `usearch_dense_save_buffer`.
    pub fn saveBuffer(self: DenseIndex, arena: *Arena, config: DenseSaveBufferConfig) Error!Monolith {
        return self.here().saveBuffer(arena, config);
    }

    /// Approximate sparse product of the left-hand rows against the indexed rows, reporting
    /// every pair the traversal scored instead of only the nearest, as a ragged output landed in
    /// `arena` with Apache Arrow offsets.
    ///
    /// Pairs arrive in traversal order rather than sorted, and carry no duplicates. Cost is
    /// roughly `rows × expansion × connectivity × dimensions`, sublinear in the indexed rows.
    /// Runs on the inner-product kind only, refusing others with `Error.ConfigMismatch`.
    /// Calls `usearch_dense_multiply`.
    pub fn multiply(self: DenseIndex, left: RowsView, arena: *Arena, config: DenseMultiplyConfig) Error!DenseProducts {
        return self.here().multiply(left, arena, config);
    }

    /// Softmax attention of a run of query heads over every indexed key their graph walks score,
    /// folded as Flash Attention folds key tiles - no top-k cut, so the long tail the walk scores
    /// brings the result toward exact dense attention.
    ///
    /// One walk per token serves its `config.head_count` heads, rows `t · head_count + h`. The
    /// index stores no values: `value_lookup` resolves the keys of every scored batch to their
    /// rows. The attended rows land in `arena` as wide as the keys and divided by the partition,
    /// and `carry` is worth asking for only to fold this branch into another over the same
    /// queries. Runs on the inner-product kind only. Calls `usearch_dense_attend`.
    pub fn attend(self: DenseIndex, queries: RowsView, value_lookup: ValueLookup, arena: *Arena, config: DenseAttendConfig) Error!DenseVectors {
        return self.here().attend(queries, value_lookup, arena, config);
    }

    /// Late-interaction scoring of multi-row queries against multi-row entries: every query row
    /// takes its best match within a key, and those maxima sum to that key's score.
    ///
    /// `queries` carries the queries laid end to end, `rows_per_query` rows apiece and uniform, so
    /// ColBERT's padding applies as is, and the matches land in `arena`, up to `wanted` keys per
    /// query. Approximate downward only, and runs on the inner-product kind only.
    /// Calls `usearch_dense_maxsim`.
    pub fn maxsim(self: DenseIndex, queries: RowsView, rows_per_query: usize, wanted: usize, arena: *Arena, config: DenseMaxsimConfig) Error!Matches {
        return self.here().maxsim(queries, rows_per_query, wanted, arena, config);
    }
};

fn handleOf(executor: ?Executor) ?*anyopaque {
    return if (executor) |chosen| chosen.handle else null;
}

/// The verbs of `DenseIndex` once more, for a receiver that completes them as `completion` says:
/// blocking answers the result, deferred submits the verb `Into` describes into the caller's
/// `Deferred(Into)`. Each verb is written once.
fn Receiver(comptime completion: Completion, comptime Into: type) type {
    return struct {
        const Self = @This();

        index: ?*anyopaque,
        executor: ?*anyopaque,
        /// Where a deferred call is held until final.
        into: if (completion == .deferred) *Deferred(Into) else void,
        /// Where a blocking call copies its final outcome; a `Deferred` keeps its own instead.
        diagnostics: if (completion == .blocking) ?*Outcome else void,

        fn Answer(comptime Call: type) type {
            return if (completion == .deferred) Error!void else Error!Call.Result;
        }

        fn answer(self: Self, call: anytype) Answer(@TypeOf(call)) {
            return self.answerIn(call, {});
        }

        /// Runs or submits `call` landing in `arena`, which a verb without outputs passes as `{}`.
        fn answerIn(self: Self, call: anytype, arena: anytype) Answer(@TypeOf(call)) {
            var local = call;
            if (@TypeOf(arena) != void) local.stamp.arena = arena;
            switch (completion) {
                .blocking => {
                    local.launch(self.index, self.executor, .blocking);
                    if (@TypeOf(arena) != void) local.stamp = types.served(arena, local.outcome.status);
                    if (self.diagnostics) |lent| lent.* = local.outcome;
                    return local.finish();
                },
                .deferred => {
                    self.into.submit(local, self.index, self.executor);
                    if (@TypeOf(arena) != void) self.into.call.stamp = types.served(arena, self.into.status());
                },
            }
        }

        /// Fails a call the binding refused before it crossed, as `refuseInto` does.
        fn refused(self: Self, status: Status, message: [*:0]const u8) Error {
            return refuseInto(if (completion == .blocking) self.diagnostics else null, status, message);
        }

        /// See `DenseIndex.diagnosed`; a `Deferred` answers its own `message` instead. Binding-only.
        pub const diagnosed = if (completion == .blocking) diagnosedBlocking else {};

        /// See `DenseIndex.contains`; a `Deferred` answers `count` instead. Calls `usearch_dense_count`.
        pub const contains = if (completion == .blocking) containsBlocking else {};

        fn diagnosedBlocking(self: Self, outcome: *Outcome) Self {
            var lending = self;
            lending.diagnostics = outcome;
            return lending;
        }

        fn containsBlocking(self: Self, key: Key) Error!bool {
            return try self.count(key) != 0;
        }

        /// See `DenseIndex.reserve`. Calls `usearch_dense_reserve`.
        pub fn reserve(self: Self, capacity: usize) Answer(DenseReserve) {
            return self.answer(DenseReserve{ .capacity = capacity });
        }

        /// See `DenseIndex.clear`. Calls `usearch_dense_clear`.
        pub fn clear(self: Self) Answer(DenseClear) {
            return self.answer(DenseClear{});
        }

        /// See `DenseIndex.describe`. Calls `usearch_dense_describe`.
        pub fn describe(self: Self) Answer(DenseDescribe) {
            return self.answer(DenseDescribe{});
        }

        /// See `DenseIndex.count`. Calls `usearch_dense_count`.
        pub fn count(self: Self, key: Key) Answer(DenseCount) {
            return self.answer(DenseCount{ .key = key });
        }

        /// See `DenseIndex.get`. Calls `usearch_dense_get`.
        pub fn get(self: Self, key: Key, scalar: Scalar, arena: *Arena) Answer(DenseGet) {
            return self.answerIn(DenseGet{ .key = key, .scalar = scalar }, arena);
        }

        /// See `DenseIndex.insert`. Calls `usearch_dense_insert`.
        pub fn insert(self: Self, keys: []const Key, vectors: RowsView, config: DenseInsertConfig) Answer(DenseInsert) {
            // C takes one count for both runs, so only the binding sees the two lengths disagree.
            if (keys.len != vectors.rows) return self.refused(.shape_mismatch, "The key run and the rows disagree in length");
            return self.answer(DenseInsert{ .keys = keys, .vectors = vectors, .config = config });
        }

        /// See `DenseIndex.refine`. Calls `usearch_dense_refine`.
        pub fn refine(self: Self, config: DenseRefineConfig) Answer(DenseRefine) {
            return self.answer(DenseRefine{ .config = config });
        }

        /// See `DenseIndex.health`. Calls `usearch_dense_health`.
        pub fn health(self: Self) Answer(DenseHealth) {
            return self.answer(DenseHealth{});
        }

        /// See `DenseIndex.search`. Calls `usearch_dense_search`.
        pub fn search(self: Self, queries: RowsView, wanted: usize, arena: *Arena, config: DenseSearchConfig) Answer(DenseSearch) {
            return self.answerIn(DenseSearch{ .queries = queries, .wanted = wanted, .config = config }, arena);
        }

        /// See `DenseIndex.merge`. Calls `usearch_dense_merge`.
        pub fn merge(self: Self, donors: []DenseIndex, config: DenseMergeConfig) Answer(DenseMerge) {
            return self.answer(DenseMerge{ .donors = donors, .config = config });
        }

        /// See `DenseIndex.cursorNext`. Calls `usearch_dense_cursor_next`.
        pub fn cursorNext(self: Self, cursor: DenseCursor, wanted: usize, vectors: ?Scalar, arena: *Arena) Answer(DenseCursorNext) {
            return self.answerIn(DenseCursorNext{ .cursor = cursor, .wanted = wanted, .scalar = vectors }, arena);
        }

        /// See `DenseIndex.searchStreamNext`. Calls `usearch_dense_search_stream_next`.
        pub fn searchStreamNext(self: Self, stream: DenseSearchStream, wanted: usize, arena: *Arena) Answer(DenseSearchStreamNext) {
            return self.answerIn(DenseSearchStreamNext{ .stream = stream, .wanted = wanted }, arena);
        }

        /// See `DenseIndex.save`. Calls `usearch_dense_save`.
        pub fn save(self: Self, path: []const u8, config: DenseSaveConfig) Answer(DenseSave) {
            return self.answer(DenseSave{ .path = path, .config = config });
        }

        /// See `DenseIndex.saveBuffer`. Calls `usearch_dense_save_buffer`.
        pub fn saveBuffer(self: Self, arena: *Arena, config: DenseSaveBufferConfig) Answer(DenseSaveBuffer) {
            return self.answerIn(DenseSaveBuffer{ .config = config }, arena);
        }

        /// See `DenseIndex.multiply`. Calls `usearch_dense_multiply`.
        pub fn multiply(self: Self, left: RowsView, arena: *Arena, config: DenseMultiplyConfig) Answer(DenseMultiply) {
            return self.answerIn(DenseMultiply{ .left = left, .config = config }, arena);
        }

        /// See `DenseIndex.attend`. Calls `usearch_dense_attend`.
        pub fn attend(self: Self, queries: RowsView, value_lookup: ValueLookup, arena: *Arena, config: DenseAttendConfig) Answer(DenseAttend) {
            // C reads one carry per query and has no length to check it against.
            if (config.carry) |carry| if (carry.len < queries.rows)
                return self.refused(.shape_mismatch, "The carry holds fewer states than there are query rows");
            return self.answerIn(DenseAttend{ .queries = queries, .value_lookup = value_lookup, .config = config }, arena);
        }

        /// See `DenseIndex.maxsim`. Calls `usearch_dense_maxsim`.
        pub fn maxsim(self: Self, queries: RowsView, rows_per_query: usize, wanted: usize, arena: *Arena, config: DenseMaxsimConfig) Answer(DenseMaxsim) {
            // C sees only the query count, never the rows left over from dividing by it.
            if (rows_per_query == 0 or queries.rows % rows_per_query != 0)
                return self.refused(.shape_mismatch, "The rows do not divide into queries of `rows_per_query`");
            return self.answerIn(DenseMaxsim{ .queries = queries, .rows_per_query = rows_per_query, .wanted = wanted, .config = config }, arena);
        }
    };
}

/// One `DenseIndex.reserve`, as a `Deferred` holds it.
pub const DenseReserve = struct {
    pub const Result = void;
    capacity: usize,
    outcome: Outcome = .{},

    pub fn launch(self: *DenseReserve, index: ?*anyopaque, executor: ?*anyopaque, completion: Completion) void {
        var arguments: usearch_dense_reserve_t = .{ .index = index, .executor = executor, .outcome = &self.outcome, .completion = completion, .capacity = self.capacity };
        usearch_dense_reserve(&arguments);
    }

    pub fn finish(self: *DenseReserve) Error!void {
        return self.outcome.check();
    }
};

/// One `DenseIndex.clear`, as a `Deferred` holds it.
pub const DenseClear = struct {
    pub const Result = void;
    outcome: Outcome = .{},

    pub fn launch(self: *DenseClear, index: ?*anyopaque, executor: ?*anyopaque, completion: Completion) void {
        var arguments: usearch_dense_clear_t = .{ .index = index, .executor = executor, .outcome = &self.outcome, .completion = completion };
        usearch_dense_clear(&arguments);
    }

    pub fn finish(self: *DenseClear) Error!void {
        return self.outcome.check();
    }
};

/// One `DenseIndex.describe`, as a `Deferred` holds it.
pub const DenseDescribe = struct {
    pub const Result = DenseIndexDescription;
    description: usearch_dense_index_description_t = .{},
    outcome: Outcome = .{},

    pub fn launch(self: *DenseDescribe, index: ?*anyopaque, executor: ?*anyopaque, completion: Completion) void {
        var arguments: usearch_dense_describe_t = .{ .index = index, .executor = executor, .outcome = &self.outcome, .completion = completion, .description = &self.description };
        usearch_dense_describe(&arguments);
    }

    pub fn finish(self: *DenseDescribe) Error!DenseIndexDescription {
        try self.outcome.check();
        const raw = self.description;
        return .{
            .metadata = raw.metadata,
            .size = raw.size,
            .capacity = raw.capacity,
            .memory_usage = raw.memory_usage,
            .capabilities = if (raw.capabilities) |capabilities| std.mem.span(capabilities) else "",
        };
    }
};

/// One `DenseIndex.count`, as a `Deferred` holds it.
pub const DenseCount = struct {
    pub const Result = usize;
    key: Key,
    entries: usize = 0,
    outcome: Outcome = .{},

    pub fn launch(self: *DenseCount, index: ?*anyopaque, executor: ?*anyopaque, completion: Completion) void {
        var arguments: usearch_dense_count_t = .{ .index = index, .executor = executor, .outcome = &self.outcome, .completion = completion, .key = self.key, .entries = &self.entries };
        usearch_dense_count(&arguments);
    }

    pub fn finish(self: *DenseCount) Error!usize {
        try self.outcome.check();
        return self.entries;
    }
};

/// One `DenseIndex.get`, as a `Deferred` holds it.
pub const DenseGet = struct {
    pub const Result = DenseVectors;
    key: Key,
    scalar: Scalar,
    /// The call the borrowed arena serves, which every view over the outputs carries.
    stamp: Arena.Stamp = undefined,
    vectors: usearch_rows_view_t = .{},
    written: usize = 0,
    outcome: Outcome = .{},

    pub fn launch(self: *DenseGet, index: ?*anyopaque, executor: ?*anyopaque, completion: Completion) void {
        var arguments: usearch_dense_get_t = .{
            .index = index,
            .executor = executor,
            .outcome = &self.outcome,
            .completion = completion,
            .key = self.key,
            .vectors = &self.vectors,
            .written = &self.written,
            .arena = self.stamp.arena.handle,
            .outputs_scalar = self.scalar,
        };
        usearch_dense_get(&arguments);
    }

    pub fn finish(self: *DenseGet) Error!DenseVectors {
        try self.outcome.check();
        return .of(self.stamp, self.vectors, self.written);
    }
};

/// One `DenseIndex.insert`, as a `Deferred` holds it.
pub const DenseInsert = struct {
    pub const Result = void;
    keys: []const Key,
    vectors: RowsView,
    config: DenseInsertConfig,
    outcome: Outcome = .{},

    pub fn launch(self: *DenseInsert, index: ?*anyopaque, executor: ?*anyopaque, completion: Completion) void {
        var arguments: usearch_dense_insert_t = .{
            .index = index,
            .executor = executor,
            .outcome = &self.outcome,
            .completion = completion,
            .count = self.vectors.rows,
            .keys = .of(self.keys),
            .vectors = .of(self.vectors),
            .expansion_factor = self.config.expansion_factor,
            .progress = .of(self.config.progress, self.config.progress_steps),
            .stats = self.config.stats,
        };
        usearch_dense_insert(&arguments);
    }

    pub fn finish(self: *DenseInsert) Error!void {
        return self.outcome.check();
    }
};

/// One `DenseIndex.refine`, as a `Deferred` holds it.
pub const DenseRefine = struct {
    pub const Result = usize;
    config: DenseRefineConfig,
    changed: usize = 0,
    outcome: Outcome = .{},

    pub fn launch(self: *DenseRefine, index: ?*anyopaque, executor: ?*anyopaque, completion: Completion) void {
        var arguments: usearch_dense_refine_t = .{
            .index = index,
            .executor = executor,
            .outcome = &self.outcome,
            .completion = completion,
            .changed = &self.changed,
            .expansion_factor = self.config.expansion_factor,
            .progress = .of(self.config.progress, self.config.progress_steps),
            .stats = self.config.stats,
        };
        usearch_dense_refine(&arguments);
    }

    pub fn finish(self: *DenseRefine) Error!usize {
        try self.outcome.check();
        return self.changed;
    }
};

/// One `DenseIndex.health`, as a `Deferred` holds it.
pub const DenseHealth = struct {
    pub const Result = DenseReachability;
    reachability: DenseReachability = .{},
    outcome: Outcome = .{},

    pub fn launch(self: *DenseHealth, index: ?*anyopaque, executor: ?*anyopaque, completion: Completion) void {
        var arguments: usearch_dense_health_t = .{ .index = index, .executor = executor, .outcome = &self.outcome, .completion = completion, .reachability = &self.reachability };
        usearch_dense_health(&arguments);
    }

    pub fn finish(self: *DenseHealth) Error!DenseReachability {
        try self.outcome.check();
        return self.reachability;
    }
};

/// One `DenseIndex.search`, as a `Deferred` holds it.
pub const DenseSearch = struct {
    pub const Result = Matches;
    queries: RowsView,
    wanted: usize,
    config: DenseSearchConfig,
    /// The call the borrowed arena serves, which every view over the outputs carries.
    stamp: Arena.Stamp = undefined,
    matches: usearch_matches_view_t = .{},
    outcome: Outcome = .{},

    pub fn launch(self: *DenseSearch, index: ?*anyopaque, executor: ?*anyopaque, completion: Completion) void {
        var arguments: usearch_dense_search_t = .{
            .index = index,
            .executor = executor,
            .outcome = &self.outcome,
            .completion = completion,
            .count = self.queries.rows,
            .queries = .of(self.queries),
            .matches = &self.matches,
            .arena = self.stamp.arena.handle,
            .wanted = self.wanted,
            .expansion_factor = self.config.expansion_factor,
            .traversal = self.config.traversal,
            .predicate = self.config.predicate,
            .progress = .of(self.config.progress, self.config.progress_steps),
            .stats = self.config.stats,
        };
        usearch_dense_search(&arguments);
    }

    pub fn finish(self: *DenseSearch) Error!Matches {
        try self.outcome.check();
        return .{ .stamp = self.stamp, .view = self.matches, .queries = self.queries.rows };
    }
};

/// One `DenseIndex.merge`, as a `Deferred` holds it.
pub const DenseMerge = struct {
    pub const Result = void;
    donors: []DenseIndex,
    config: DenseMergeConfig,
    outcome: Outcome = .{},

    pub fn launch(self: *DenseMerge, index: ?*anyopaque, executor: ?*anyopaque, completion: Completion) void {
        var arguments: usearch_dense_merge_t = .{
            .index = index,
            .executor = executor,
            .outcome = &self.outcome,
            .completion = completion,
            .count = self.donors.len,
            .donors = @ptrCast(self.donors.ptr),
            .progress = .of(self.config.progress, self.config.progress_steps),
        };
        usearch_dense_merge(&arguments);
    }

    pub fn finish(self: *DenseMerge) Error!void {
        return self.outcome.check();
    }
};

/// One `DenseIndex.cursorNext`, as a `Deferred` holds it.
pub const DenseCursorNext = struct {
    pub const Result = DenseCursorBatch;
    cursor: DenseCursor,
    wanted: usize,
    scalar: ?Scalar,
    /// The call the borrowed arena serves, which every view over the outputs carries.
    stamp: Arena.Stamp = undefined,
    keys: usearch_keys_view_t = .{},
    vectors: usearch_rows_view_t = .{},
    written: usize = 0,
    outcome: Outcome = .{},

    pub fn launch(self: *DenseCursorNext, index: ?*anyopaque, executor: ?*anyopaque, completion: Completion) void {
        var arguments: usearch_dense_cursor_next_t = .{
            .index = index,
            .executor = executor,
            .outcome = &self.outcome,
            .completion = completion,
            .cursor = self.cursor.handle,
            .keys = &self.keys,
            .vectors = if (self.scalar != null) &self.vectors else null,
            .written = &self.written,
            .arena = self.stamp.arena.handle,
            .wanted = self.wanted,
            .outputs_scalar = self.scalar orelse .unknown,
        };
        usearch_dense_cursor_next(&arguments);
    }

    pub fn finish(self: *DenseCursorNext) Error!DenseCursorBatch {
        try self.outcome.check();
        return .{
            .stamp = self.stamp,
            .data = self.keys.data,
            .written = self.written,
            .vectors = if (self.scalar != null) .of(self.stamp, self.vectors, self.written) else null,
        };
    }
};

/// One `DenseIndex.searchStreamNext`, as a `Deferred` holds it.
pub const DenseSearchStreamNext = struct {
    pub const Result = Matches;
    stream: DenseSearchStream,
    wanted: usize,
    /// The call the borrowed arena serves, which every view over the outputs carries.
    stamp: Arena.Stamp = undefined,
    matches: usearch_matches_view_t = .{},
    outcome: Outcome = .{},

    pub fn launch(self: *DenseSearchStreamNext, index: ?*anyopaque, executor: ?*anyopaque, completion: Completion) void {
        var arguments: usearch_dense_search_stream_next_t = .{
            .index = index,
            .executor = executor,
            .outcome = &self.outcome,
            .completion = completion,
            .stream = self.stream.handle,
            .matches = &self.matches,
            .arena = self.stamp.arena.handle,
            .wanted = self.wanted,
        };
        usearch_dense_search_stream_next(&arguments);
    }

    pub fn finish(self: *DenseSearchStreamNext) Error!Matches {
        try self.outcome.check();
        return .{ .stamp = self.stamp, .view = self.matches, .queries = 1 };
    }
};

/// One `DenseIndex.save`, as a `Deferred` holds it.
pub const DenseSave = struct {
    pub const Result = void;
    path: []const u8,
    config: DenseSaveConfig,
    outcome: Outcome = .{},

    pub fn launch(self: *DenseSave, index: ?*anyopaque, executor: ?*anyopaque, completion: Completion) void {
        var arguments: usearch_dense_save_t = .{
            .index = index,
            .executor = executor,
            .outcome = &self.outcome,
            .completion = completion,
            .path = .of(self.path),
            .progress = .of(self.config.progress, self.config.progress_steps),
        };
        usearch_dense_save(&arguments);
    }

    pub fn finish(self: *DenseSave) Error!void {
        return self.outcome.check();
    }
};

/// One `DenseIndex.saveBuffer`, as a `Deferred` holds it.
pub const DenseSaveBuffer = struct {
    pub const Result = Monolith;
    config: DenseSaveBufferConfig,
    /// The call the borrowed arena serves, which every view over the outputs carries.
    stamp: Arena.Stamp = undefined,
    monolith: BytesView = .{ .data = null, .length = 0 },
    outcome: Outcome = .{},

    pub fn launch(self: *DenseSaveBuffer, index: ?*anyopaque, executor: ?*anyopaque, completion: Completion) void {
        var arguments: usearch_dense_save_buffer_t = .{
            .index = index,
            .executor = executor,
            .outcome = &self.outcome,
            .completion = completion,
            .monolith = &self.monolith,
            .arena = self.stamp.arena.handle,
            .progress = .of(self.config.progress, self.config.progress_steps),
        };
        usearch_dense_save_buffer(&arguments);
    }

    pub fn finish(self: *DenseSaveBuffer) Error!Monolith {
        try self.outcome.check();
        return .{ .stamp = self.stamp, .view = self.monolith };
    }
};

/// One `DenseIndex.multiply`, as a `Deferred` holds it.
pub const DenseMultiply = struct {
    pub const Result = DenseProducts;
    left: RowsView,
    config: DenseMultiplyConfig,
    /// The call the borrowed arena serves, which every view over the outputs carries.
    stamp: Arena.Stamp = undefined,
    products: usearch_matches_tape_view_t = .{},
    written: usize = 0,
    required: usize = 0,
    outcome: Outcome = .{},

    pub fn launch(self: *DenseMultiply, index: ?*anyopaque, executor: ?*anyopaque, completion: Completion) void {
        var arguments: usearch_dense_multiply_t = .{
            .index = index,
            .executor = executor,
            .outcome = &self.outcome,
            .completion = completion,
            .count = self.left.rows,
            .left = .of(self.left),
            .products = &self.products,
            .written = &self.written,
            .required = &self.required,
            .arena = self.stamp.arena.handle,
            .wanted = self.config.wanted,
            .expansion_factor = self.config.expansion_factor,
            .progress = .of(self.config.progress, self.config.progress_steps),
            .stats = self.config.stats,
        };
        usearch_dense_multiply(&arguments);
    }

    pub fn finish(self: *DenseMultiply) Error!DenseProducts {
        try self.outcome.check();
        return .{
            .stamp = self.stamp,
            .view = self.products,
            .rows = self.left.rows,
            .written = self.written,
            .required = self.required,
        };
    }
};

/// One `DenseIndex.attend`, as a `Deferred` holds it.
pub const DenseAttend = struct {
    pub const Result = DenseVectors;
    queries: RowsView,
    value_lookup: ValueLookup,
    config: DenseAttendConfig,
    /// The call the borrowed arena serves, which every view over the outputs carries.
    stamp: Arena.Stamp = undefined,
    outputs: usearch_rows_view_t = .{},
    outcome: Outcome = .{},

    pub fn launch(self: *DenseAttend, index: ?*anyopaque, executor: ?*anyopaque, completion: Completion) void {
        var arguments: usearch_dense_attend_t = .{
            .index = index,
            .executor = executor,
            .outcome = &self.outcome,
            .completion = completion,
            .count = self.queries.rows,
            .queries = .of(self.queries),
            .head_count = self.config.head_count,
            .value_lookup = self.value_lookup.callback,
            .value_lookup_state = self.value_lookup.state,
            .carry = if (self.config.carry) |carry| carry.ptr else null,
            .outputs = &self.outputs,
            .arena = self.stamp.arena.handle,
            .outputs_scalar = self.config.outputs_scalar,
            .expansion_factor = self.config.expansion_factor,
            .score_scale = self.config.score_scale,
            .progress = .of(self.config.progress, self.config.progress_steps),
            .stats = self.config.stats,
        };
        usearch_dense_attend(&arguments);
    }

    pub fn finish(self: *DenseAttend) Error!DenseVectors {
        try self.outcome.check();
        return .of(self.stamp, self.outputs, self.queries.rows);
    }
};

/// One `DenseIndex.maxsim`, as a `Deferred` holds it.
pub const DenseMaxsim = struct {
    pub const Result = Matches;
    queries: RowsView,
    rows_per_query: usize,
    wanted: usize,
    config: DenseMaxsimConfig,
    /// The call the borrowed arena serves, which every view over the outputs carries.
    stamp: Arena.Stamp = undefined,
    matches: usearch_matches_view_t = .{},
    outcome: Outcome = .{},

    pub fn launch(self: *DenseMaxsim, index: ?*anyopaque, executor: ?*anyopaque, completion: Completion) void {
        var arguments: usearch_dense_maxsim_t = .{
            .index = index,
            .executor = executor,
            .outcome = &self.outcome,
            .completion = completion,
            .count = self.queries.rows / self.rows_per_query,
            .queries = .of(self.queries),
            .rows_per_query = self.rows_per_query,
            .matches = &self.matches,
            .arena = self.stamp.arena.handle,
            .wanted = self.wanted,
            .expansion_factor = self.config.expansion_factor,
            .progress = .of(self.config.progress, self.config.progress_steps),
            .stats = self.config.stats,
        };
        usearch_dense_maxsim(&arguments);
    }

    pub fn finish(self: *DenseMaxsim) Error!Matches {
        try self.outcome.check();
        return .{ .stamp = self.stamp, .view = self.matches, .queries = self.queries.rows / self.rows_per_query };
    }
};

/// Rows a call wrote packed into an arena - `get` one per entry under the key, `attend` one per
/// query, a cursor pull one per entry - `stride` bytes apiece, which sub-byte kinds round up to
/// whole bytes. Mirrors `usearch_rows_view_t`.
///
/// It reads the arena, so every read is refused once that arena serves another call or is freed.
pub const DenseVectors = struct {
    stamp: Arena.Stamp,
    data: ?[*]const u8,
    /// Rows written.
    rows: usize,
    /// Bytes per row, as the call reported them.
    stride: usize,
    /// Dimensions per row, as the call reported them.
    columns: usize,
    /// The scalar type the rows were written as.
    scalar: Scalar,

    fn of(stamp: Arena.Stamp, view: usearch_rows_view_t, rows: usize) DenseVectors {
        return .{ .stamp = stamp, .data = @ptrCast(view.data), .rows = rows, .stride = view.stride, .columns = view.columns, .scalar = view.scalar };
    }

    /// One row as a run of `Element`, the type the rows were written as; any other type and a
    /// `position` past `rows` are refused with `Error.InvalidArgument`, a stale view with
    /// `Error.Uninitialized`, and device memory with `Error.Device`.
    pub fn row(self: DenseVectors, comptime Element: type, position: usize) Error![]const Element {
        try self.stamp.check();
        if (Scalar.of(Element) != self.scalar or position >= self.rows) return refuse(.invalid_argument);
        const first: [*]const Element = @ptrCast(@alignCast(self.data.? + position * self.stride));
        return first[0 .. self.stride / @sizeOf(Element)];
    }

    /// Every row, back to back, refused as `row` refuses a stale view or device memory.
    pub fn bytes(self: DenseVectors) Error![]const u8 {
        try self.stamp.check();
        return landed(u8, self.data, 0, self.rows * self.stride);
    }

    /// Where the rows live, which the host reads only as `.ram` or `.nvme`.
    pub fn residency(self: DenseVectors) Residency {
        return self.stamp.arena.residency;
    }
};

/// The monolith `DenseIndex.saveBuffer` serialized into an arena.
///
/// It reads the arena, so every read is refused once that arena serves another call or is freed.
pub const Monolith = struct {
    stamp: Arena.Stamp,
    view: BytesView,

    /// The serialized bytes, refusing a stale view with `Error.Uninitialized` and device memory with `Error.Device`.
    pub fn bytes(self: Monolith) Error![]const u8 {
        try self.stamp.check();
        return landed(u8, self.view.data, 0, self.view.length);
    }

    /// Where the bytes live, which the host reads only as `.ram` or `.nvme`.
    pub fn residency(self: Monolith) Residency {
        return self.stamp.arena.residency;
    }
};

/// The ragged output of `DenseIndex.multiply` in an arena, laid out as Apache Arrow does: row `i`
/// spans pairs `offsets[i]` up to `offsets[i + 1]` of `keys` and `scores`.
/// Mirrors `usearch_matches_tape_view_t`.
///
/// It reads the arena, so every read is refused once that arena serves another call or is freed.
pub const DenseProducts = struct {
    stamp: Arena.Stamp,
    view: usearch_matches_tape_view_t,
    /// Left-hand rows the call multiplied.
    rows: usize,
    /// Pairs written across every row.
    written: usize,
    /// Pairs scored across every row, before `DenseMultiplyConfig.wanted` cut any.
    required: usize,

    /// The pairs of one left-hand row, in traversal order, refusing a `position` past `rows` with
    /// `Error.InvalidArgument`, a stale view with `Error.Uninitialized`, and device memory with `Error.Device`.
    pub fn row(self: DenseProducts, position: usize) Error!MatchesRow {
        try self.stamp.check();
        if (position >= self.rows) return refuse(.invalid_argument);
        const begin = self.view.offsets.?[position];
        const length = self.view.offsets.?[position + 1] - begin;
        return .{ .keys = landed(Key, self.view.keys, begin, length), .scores = landed(Distance, self.view.scores, begin, length) };
    }

    /// The `rows + 1` offsets, refused as `row` refuses a stale view or device memory.
    pub fn offsets(self: DenseProducts) Error![]const usize {
        try self.stamp.check();
        return landed(usize, self.view.offsets, 0, self.rows + 1);
    }

    /// Every pair's indexed key, end to end, refused as `row` refuses a stale view or device memory.
    pub fn keys(self: DenseProducts) Error![]const Key {
        try self.stamp.check();
        return landed(Key, self.view.keys, 0, self.written);
    }

    /// Every pair's product, positionally paired with `keys`, refused as `keys` is.
    pub fn scores(self: DenseProducts) Error![]const Distance {
        try self.stamp.check();
        return landed(Distance, self.view.scores, 0, self.written);
    }

    /// Where the pairs live, which the host reads only as `.ram` or `.nvme`.
    pub fn residency(self: DenseProducts) Residency {
        return self.stamp.arena.residency;
    }
};

/// One batch a `DenseCursor` pull wrote into an arena: the keys, and the rows beside them where
/// asked for.
///
/// It reads the arena, so every read is refused once that arena serves another call or is freed.
pub const DenseCursorBatch = struct {
    stamp: Arena.Stamp,
    data: ?[*]const Key,
    /// Entries the pull wrote; zero once the walk is exhausted.
    written: usize,
    /// The rows, one per key, or null where the pull exported keys alone.
    vectors: ?DenseVectors,

    /// The keys, refusing a stale view with `Error.Uninitialized` and device memory with `Error.Device`.
    pub fn keys(self: DenseCursorBatch) Error![]const Key {
        try self.stamp.check();
        return landed(Key, self.data, 0, self.written);
    }
};

/// A sequential row scan over the index contents - export, migration, and visualization feed
/// off it. Every pull names the index again, through `DenseIndex.cursorNext`, so free the cursor
/// before the index. Mirrors `usearch_dense_cursor_t`.
pub const DenseCursor = struct {
    /// The opaque `usearch_dense_cursor_t` the ABI hands back.
    handle: ?*anyopaque,

    /// Opens a row scan over the current contents; each shard's publication watermark is read as
    /// the walk enters it, so a row once passed is never revisited. Calls `usearch_dense_cursor_init`.
    pub fn init(index: DenseIndex) Error!DenseCursor {
        var handle: ?*anyopaque = null;
        var outcome: Outcome = .{};
        var arguments: usearch_dense_cursor_init_t = .{ .cursor = &handle, .index = index.handle, .outcome = &outcome };
        usearch_dense_cursor_init(&arguments);
        errdefer usearch_dense_cursor_free(handle);
        try outcome.check();
        return .{ .handle = handle };
    }

    /// Releases the cursor, waiting for every deferred pull naming it. Calls `usearch_dense_cursor_free`.
    pub fn deinit(self: *DenseCursor) void {
        usearch_dense_cursor_free(self.handle);
        self.* = undefined;
    }
};

/// A resumable nearest-neighbor stream over one query.
///
/// Best-first traversal is naturally pausable, so the stream keeps a live frontier and visited
/// set and yields matches lazily - callers that post-filter aggressively pull until satisfied
/// instead of guessing "k" upfront. It keeps walk state rather than a thread, and every pull names
/// the index again through `DenseIndex.searchStreamNext`, so free the stream before the index.
/// Mirrors `usearch_dense_search_stream_t`.
pub const DenseSearchStream = struct {
    /// The opaque `usearch_dense_search_stream_t` the ABI hands back.
    handle: ?*anyopaque,

    /// Opens a stream for one query row, refusing a run of several with `Error.ShapeMismatch`.
    /// Calls `usearch_dense_search_stream_init`.
    pub fn init(index: DenseIndex, query: RowsView, config: DenseSearchStreamInitConfig) Error!DenseSearchStream {
        // The stream's query carries no count, so only the binding sees a run of several.
        if (query.rows != 1) return refuseInto(config.diagnostics, .shape_mismatch, "A stream takes exactly one query row");
        var handle: ?*anyopaque = null;
        var outcome: Outcome = .{};
        var arguments: usearch_dense_search_stream_init_t = .{
            .stream = &handle,
            .index = index.handle,
            .outcome = &outcome,
            .query = .of(query),
            .expansion_factor = config.expansion_factor,
        };
        usearch_dense_search_stream_init(&arguments);
        errdefer usearch_dense_search_stream_free(handle);
        try outcome.checkInto(config.diagnostics);
        return .{ .handle = handle };
    }

    /// Releases the stream and the walk state it keeps, waiting for every deferred pull naming it.
    /// Calls `usearch_dense_search_stream_free`.
    pub fn deinit(self: *DenseSearchStream) void {
        usearch_dense_search_stream_free(self.handle);
        self.* = undefined;
    }
};

test "a slab reads as rows, and one vector as a single row" {
    const slab = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const rows = try RowsView.of(f32, &slab, 3);
    // A packed slab spells its stride out, since the interface reads zero as "repeat the first row".
    try std.testing.expectEqual(RowsView{ .data = &slab, .rows = 2, .columns = 3, .stride = 12, .scalar = .f32 }, rows);

    const single = try RowsView.ofRow(f32, &slab);
    try std.testing.expectEqual(RowsView{ .data = &slab, .rows = 1, .columns = 6, .stride = 24, .scalar = .f32 }, single);
}

test "a shape the lengths cannot support is refused rather than divided by" {
    const slab = [_]f32{ 1, 2, 3, 4, 5, 6 };
    try std.testing.expectError(Error.ShapeMismatch, RowsView.of(f32, &slab, 0));
    try std.testing.expectError(Error.ShapeMismatch, RowsView.of(f32, &slab, 4));
    try std.testing.expectError(Error.ShapeMismatch, RowsView.ofRow(f32, slab[0..0]));
}

test "left alone, an index is an angular one quantized to bf16" {
    try std.testing.expectEqual(DenseOptions{
        .metric = .angular,
        .quantization = .bf16,
        .dimensions = 0,
        .connectivity = 0,
        .values_per_key = 0,
    }, DenseOptions{});
    try std.testing.expectEqual(@as(?ProgressFunction, null), (DenseSearchConfig{}).progress.callback);
    try std.testing.expectEqual(@as(?PredicateFunction, null), (DenseSearchConfig{}).predicate.callback);
    try std.testing.expectEqual(Traversal.walk, (DenseSearchConfig{}).traversal);
}

test "metrics keep the values the linked library spells them by" {
    inline for (.{ "inner", "angular", "euclidean", "hamming", "jaccard" }, 1..) |spelling, value| {
        const metric: DenseMetric = @enumFromInt(value);
        try std.testing.expectEqualStrings(spelling, metric.name());
        try std.testing.expectEqual(@as(?DenseMetric, metric), DenseMetric.fromName(spelling));
    }
    try std.testing.expectEqual(@as(?DenseMetric, null), DenseMetric.fromName("cosine"));
}

test "callbacks wrap with and without a context" {
    const Tenant = struct {
        allowed: u8,
        fn admits(self: *const @This(), key: Key) types.Admission {
            return if (key.bytes[0] == self.allowed) .admit else .reject;
        }
    };
    const tenant: Tenant = .{ .allowed = 7 };
    const filter = Predicate.of(&tenant, Tenant.admits);
    try std.testing.expectEqual(types.Admission.admit, filter.callback.?(.{ .bytes = .{7} ++ .{0} ** 15 }, filter.state));
    try std.testing.expectEqual(types.Admission.reject, filter.callback.?(.nil, filter.state));

    var tally: usize = 0;
    const counting = ProgressCallback.of(&tally, struct {
        fn saw(seen: *usize, completed: usize, _: usize) types.Continuation {
            seen.* = completed;
            return .@"continue";
        }
    }.saw);
    _ = counting.callback.?(5, 9, counting.state);
    try std.testing.expectEqual(@as(usize, 5), tally);

    // A handler needing no context travels with a null state and is never handed one.
    const stateless = ProgressCallback.of({}, struct {
        fn saw(_: void, _: usize, _: usize) types.Continuation {
            return .cancel;
        }
    }.saw);
    try std.testing.expectEqual(@as(?*anyopaque, null), stateless.state);
    try std.testing.expectEqual(types.Continuation.cancel, stateless.callback.?(0, 1, null));
}

test "a value lookup answers only rows of the kind and width each batch states" {
    const rows = [_]f32{ 1, 2, 3, 4 };
    const lookup = ValueLookup.of(f32, &rows, struct {
        fn resolve(table: *const [4]f32, key: Key) ?[]const f32 {
            const entry = key.asInteger() orelse return null;
            return if (entry < 2) table[entry * 2 ..][0..2] else null;
        }
    }.resolve);
    const keys = [_]Key{ .fromInteger(1), .fromInteger(5) };
    var found: [2]?*const anyopaque = undefined;

    lookup.callback.?(lookup.state, &keys, 2, .f32, 2, &found);
    try std.testing.expectEqual([2]?*const anyopaque{ &rows[2], null }, found);

    // Another kind or width than the lookup's rows leaves every row null, which C refuses.
    lookup.callback.?(lookup.state, &keys, 2, .f16, 2, &found);
    try std.testing.expectEqual([2]?*const anyopaque{ null, null }, found);
    lookup.callback.?(lookup.state, &keys, 2, .f32, 3, &found);
    try std.testing.expectEqual([2]?*const anyopaque{ null, null }, found);
}

test { // analyze every declaration, the methods on each handle included
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(DenseIndex);
    std.testing.refAllDecls(DenseIndex.On);
    inline for (.{
        DenseReserve,  DenseClear,  DenseDescribe, DenseCount,      DenseGet,              DenseInsert,
        DenseRefine,   DenseHealth, DenseSearch,   DenseMerge,      DenseSave,             DenseSaveBuffer,
        DenseMultiply, DenseAttend, DenseMaxsim,   DenseCursorNext, DenseSearchStreamNext,
    }) |Call| std.testing.refAllDecls(Deferred(Call));
    std.testing.refAllDecls(DenseCursor);
    std.testing.refAllDecls(DenseSearchStream);
    std.testing.refAllDecls(DenseProducts);
    std.testing.refAllDecls(DenseVectors);
    std.testing.refAllDecls(DenseCursorBatch);
    std.testing.refAllDecls(Monolith);
    std.testing.refAllDecls(Matches);
    std.testing.refAllDecls(RowsView);
}

test "a batch crosses the ABI and comes back as views over the arena it lands in" {
    const dimensions = 3;
    var executor = try Executor.forRam(.{ .threads = 2 });
    defer executor.deinit();
    var index = try DenseIndex.initRam(.{ .dimensions = dimensions, .metric = .euclidean }, &.{});
    defer index.deinit();
    try index.on(executor).reserve(3);

    const slab = [_]f32{ 1, 0, 0, 0, 1, 0, 0, 0, 1 };
    const keys = [_]Key{ .fromInteger(1), .fromInteger(2), .fromInteger(3) };
    var stats: DenseTraversalStats = .{};
    try index.on(executor).insert(&keys, try .of(f32, &slab, dimensions), .{ .stats = &stats });
    const description = try index.on(executor).describe();
    // The engine derives the connectivity, so it is read back rather than predicted.
    const frozen: DenseMetadata = .{ .metric = .euclidean, .quantization = .bf16, .dimensions = dimensions, .connectivity = description.metadata.connectivity };
    try std.testing.expectEqual(frozen, description.metadata);
    try std.testing.expectEqual(@as(usize, 3), description.size);

    // A lent outcome carries the message an error cannot, the binding's own refusals included.
    var outcome: Outcome = .{};
    try std.testing.expectError(Error.ShapeMismatch, index.diagnosed(&outcome).insert(keys[0..2], try .of(f32, &slab, dimensions), .{}));
    try std.testing.expectEqual(types.Status.shape_mismatch, outcome.status);
    try std.testing.expectEqualStrings("The key run and the rows disagree in length", outcome.text());
    try index.on(executor).diagnosed(&outcome).reserve(3);
    try std.testing.expectEqual(Outcome{}, outcome);

    var arena: Arena = try .init(executor);
    defer arena.deinit();
    try std.testing.expectEqual(Residency.ram, arena.residency);
    const matches = try index.on(executor).search(try .ofRow(f32, slab[0..dimensions]), 2, &arena, .{ .stats = &stats });
    try std.testing.expectEqual(@as(usize, 2), (try matches.row(0)).keys.len);
    try std.testing.expectEqual(keys[0], (try matches.row(0)).keys[0]);
    try std.testing.expectError(Error.InvalidArgument, matches.row(1));

    // The caller's own thread answers the same search, and its call rewinds the arena, so the
    // previous view refuses rather than reads what the next call wrote.
    const again = try index.search(try .ofRow(f32, slab[0..dimensions]), 2, &arena, .{});
    try std.testing.expectError(Error.Uninitialized, matches.row(0));
    try std.testing.expectEqual(keys[0], (try again.row(0)).keys[0]);

    // A predicate narrows the matches, and still finds the admitted row behind rejected ones.
    const Only = struct {
        key: Key,
        fn admits(self: *const @This(), key: Key) types.Admission {
            return if (std.meta.eql(key, self.key)) .admit else .reject;
        }
    };
    const only: Only = .{ .key = keys[2] };
    const admitted = try index.on(executor).search(try .ofRow(f32, slab[0..dimensions]), 2, &arena, .{
        .predicate = .of(&only, Only.admits),
    });
    try std.testing.expectEqualSlices(Key, keys[2..3], (try admitted.row(0)).keys);

    const exported = try index.on(executor).get(keys[1], .f32, &arena);
    try std.testing.expectEqualSlices(f32, &.{ 0, 1, 0 }, try exported.row(f32, 0));
    try std.testing.expectEqual(@as(usize, 1), exported.rows);
    try std.testing.expectEqual(@as(usize, dimensions), exported.columns);
    try std.testing.expectEqual(@as(usize, 1), try index.count(keys[1]));
    try std.testing.expectEqual(true, try index.contains(keys[1]));
    try std.testing.expectEqual(false, try index.on(executor).contains(.fromInteger(99)));
    // Rows exported as one kind are never reinterpreted as another.
    try std.testing.expectError(Error.InvalidArgument, exported.row(f16, 0));
    try std.testing.expectError(Error.InvalidArgument, exported.row(f32, 1));
    const absent = try index.get(.fromInteger(99), .f32, &arena);
    try std.testing.expectEqual(@as(usize, 0), absent.rows);
    try std.testing.expectEqual(@as(usize, 0), (try absent.bytes()).len);

    // Sub-byte rows report their own stride, so no width arithmetic happens here.
    const bits = try index.on(executor).get(keys[2], .u1x8, &arena);
    try std.testing.expectEqual(@as(usize, 1), bits.stride);
    try std.testing.expectEqual(@as(usize, dimensions), bits.columns);
    try std.testing.expectEqual(@as(usize, 1), (try bits.row(Scalar.U1x8, 0)).len);

    // A failed call rewinds the arena too, and leaves it the caller's, ready for the next one.
    try std.testing.expectError(Error.ConfigMismatch, index.multiply(try .ofRow(f32, slab[0..dimensions]), &arena, .{}));
    try std.testing.expectError(Error.Uninitialized, bits.row(Scalar.U1x8, 0));
    try std.testing.expectEqualSlices(f32, &.{ 0, 0, 1 }, try (try index.get(keys[2], .f32, &arena)).row(f32, 0));

    // A freed arena leaves its views refusing, and device memory is never read as the host's.
    var short: Arena = try .init(null);
    const doomed = try index.get(keys[1], .f32, &short);
    short.deinit();
    try std.testing.expectError(Error.Uninitialized, doomed.row(f32, 0));
    var device: Arena = try .init(null);
    defer device.deinit();
    device.residency = .vram;
    const remote = try index.get(keys[1], .f32, &device);
    try std.testing.expectEqual(Residency.vram, remote.residency());
    try std.testing.expectError(Error.Device, remote.bytes());

    // The width travels with the rows, and the engine refuses one the geometry disagrees with.
    const narrow = [_]f32{ 1, 0 };
    try std.testing.expectError(
        Error.ShapeMismatch,
        index.on(executor).diagnosed(&outcome).search(try .ofRow(f32, &narrow), 2, &arena, .{}),
    );
    try std.testing.expectEqual(types.Status.shape_mismatch, outcome.status);
    try std.testing.expect(outcome.text().len != 0);

    // A constructor's config lends one the same way.
    try std.testing.expectError(Error.Io, DenseIndex.loadRam("absent.safegraphs", .{ .diagnostics = &outcome }));
    try std.testing.expectEqual(types.Status.io, outcome.status);
    try std.testing.expect(outcome.text().len != 0);

    // A stream yields one query's matches nearest first, and refuses a run carrying more; a
    // residency that streams no matches yet refuses the stream itself.
    try std.testing.expectError(Error.ShapeMismatch, DenseSearchStream.init(index, try .of(f32, &slab, dimensions), .{}));
    if (DenseSearchStream.init(index, try .ofRow(f32, slab[0..dimensions]), .{})) |opened| {
        var stream = opened;
        defer stream.deinit();
        const pulled = try index.on(executor).searchStreamNext(stream, 2, &arena);
        try std.testing.expectEqualSlices(Key, keys[0..2], (try pulled.row(0)).keys);
        try std.testing.expectEqual(@as(usize, 1), (try (try index.searchStreamNext(stream, 2, &arena)).row(0)).keys.len);
        // A drained walk answers an empty row, as a pull of zero does.
        try std.testing.expectEqual(@as(usize, 0), (try (try index.searchStreamNext(stream, 2, &arena)).row(0)).keys.len);
    } else |failure| try std.testing.expectEqual(Error.Unsupported, failure);

    // A cursor walks the rows, `wanted` at a time, and exports the rows only when asked.
    var walk = try DenseCursor.init(index);
    defer walk.deinit();
    const batch = try index.on(executor).cursorNext(walk, 2, .f32, &arena);
    try std.testing.expectEqualSlices(Key, keys[0..2], try batch.keys());
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(slab[0..6]), try batch.vectors.?.bytes());
    const rest = try index.cursorNext(walk, 2, null, &arena);
    try std.testing.expectEqualSlices(Key, keys[2..3], try rest.keys());
    try std.testing.expectEqual(@as(?DenseVectors, null), rest.vectors);
    const drained = try index.cursorNext(walk, 2, .f32, &arena);
    try std.testing.expectEqual(@as(usize, 0), (try drained.keys()).len);
    try std.testing.expectEqual(@as(usize, 0), (try drained.vectors.?.bytes()).len);

    // A monolith round-trips through memory, and its metadata reads without loading it.
    const monolith = try (try index.on(executor).saveBuffer(&arena, .{})).bytes();
    try std.testing.expectEqual(frozen, try DenseIndex.metadataReadBuffer(monolith, .{ .executor = executor }));
    var copy = try DenseIndex.loadRamBuffer(monolith, .{ .executor = executor });
    defer copy.deinit();
    try std.testing.expectEqual(@as(usize, 3), (try copy.describe()).size);

    // A donor is consumed and nulled in place, and one a residency cannot absorb yet stays live,
    // so freeing every donor afterwards is safe either way.
    var donors = [_]DenseIndex{try DenseIndex.loadRamBuffer(monolith, .{})};
    defer for (&donors) |*donor| donor.deinit();
    if (copy.on(executor).merge(&donors, .{})) {
        try std.testing.expectEqual(DenseIndex{ .handle = null }, donors[0]);
        try std.testing.expectEqual(@as(usize, 2), try copy.count(keys[0]));
    } else |failure| {
        try std.testing.expectEqual(Error.Unsupported, failure);
        try std.testing.expectEqual(@as(usize, 1), try copy.count(keys[0]));
    }

    const reachability = try index.on(executor).health();
    try std.testing.expectEqual([2]usize{ 3, 3 }, [2]usize{ reachability.audited, reachability.reachable });
}

test "a scan scores every row, so every query finds itself first" {
    const dimensions = 4;
    var index = try DenseIndex.initRam(.{ .dimensions = dimensions }, &.{});
    defer index.deinit();
    try index.reserve(4);
    const slab = [_]f32{ 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0.5, 0.5, 0.5, 0.5 };
    const keys = [_]Key{ .fromInteger(0), .fromInteger(1), .fromInteger(2), .fromInteger(3) };
    try index.insert(&keys, try .of(f32, &slab, dimensions), .{});

    var arena: Arena = try .init(null);
    defer arena.deinit();
    const matches = try index.search(try .of(f32, &slab, dimensions), 1, &arena, .{ .traversal = .scan });
    for (keys, 0..) |key, query| try std.testing.expectEqual(key, (try matches.row(query)).keys[0]);
}

test "a deferred call is the same verb, final only once waited on" {
    const dimensions = 2;
    var executor = try Executor.forRam(.{});
    defer executor.deinit();
    var index = try DenseIndex.initRam(.{ .dimensions = dimensions, .metric = .euclidean }, &.{});
    defer index.deinit();

    var reserving: Deferred(DenseReserve) = undefined;
    try index.deferred(executor, &reserving).reserve(4);
    defer reserving.deinit();
    const slab = [_]f32{ 1, 0, 0, 1 };
    const keys = [_]Key{ .fromInteger(1), .fromInteger(2) };
    var inserting: Deferred(DenseInsert) = undefined;
    try index.deferred(executor, &inserting).insert(&keys, try .of(f32, &slab, dimensions), .{});
    defer inserting.deinit();

    // Both are submitted rather than run, and a wait finishes each, in submission order.
    try std.testing.expectEqual(types.Status.deferred, reserving.status());
    try std.testing.expectEqual(types.Status.deferred, inserting.status());
    try reserving.wait();
    try inserting.wait();
    try std.testing.expectEqual(types.Status.success, inserting.status());
    try std.testing.expectEqualStrings("", inserting.message());

    // Waiting again answers the same result rather than a second call's.
    var arena: Arena = try .init(executor);
    defer arena.deinit();
    var getting: Deferred(DenseGet) = undefined;
    try index.deferred(executor, &getting).get(keys[1], .f32, &arena);
    defer getting.deinit();
    const vectors = try getting.wait();
    try std.testing.expectEqualSlices(f32, &.{ 0, 1 }, try vectors.row(f32, 0));
    try std.testing.expectEqual(vectors, try getting.wait());

    // The arena is borrowed as a blocking call borrows it, so a second call while one holds it is
    // refused as contended and leaves the holder's views alone, and the next call retires them.
    var holding: Deferred(DenseGet) = undefined;
    try index.deferred(executor, &holding).get(keys[0], .f32, &arena);
    defer holding.deinit();
    try std.testing.expectError(Error.Uninitialized, vectors.row(f32, 0));
    try std.testing.expectError(Error.Contended, index.get(keys[1], .f32, &arena));
    var contending: Deferred(DenseGet) = undefined;
    try index.deferred(executor, &contending).get(keys[1], .f32, &arena);
    defer contending.deinit();
    try std.testing.expectEqual(types.Status.contended, contending.status());
    try std.testing.expectError(Error.Contended, contending.wait());
    const held = try holding.wait();
    try std.testing.expectEqualSlices(f32, &.{ 1, 0 }, try held.row(f32, 0));
    _ = try index.get(keys[1], .f32, &arena);
    try std.testing.expectError(Error.Uninitialized, held.row(f32, 0));

    // Storage is the caller's, so a batch of calls is an array and a reused slot is resubmitted.
    var counting: [2]Deferred(DenseCount) = undefined;
    for (&counting, keys) |*slot, key| try index.deferred(executor, slot).count(key);
    for (&counting) |*slot| try std.testing.expectEqual(@as(usize, 1), try slot.wait());
    for (&counting) |*slot| slot.deinit();
    try index.deferred(executor, &counting[0]).count(.fromInteger(99));
    defer counting[0].deinit();
    try std.testing.expectEqual(@as(usize, 0), try counting[0].wait());

    // A failure the engine reports arrives through the same wait, as often as it is asked.
    const narrow = [_]f32{1};
    var refused: Deferred(DenseSearch) = undefined;
    try index.deferred(executor, &refused).search(try .ofRow(f32, &narrow), 1, &arena, .{});
    defer refused.deinit();
    try std.testing.expectError(Error.ShapeMismatch, refused.wait());
    try std.testing.expectError(Error.ShapeMismatch, refused.wait());

    // Its message is the one the same call reports blocking.
    var outcome: Outcome = .{};
    try std.testing.expectError(Error.ShapeMismatch, index.diagnosed(&outcome).search(try .ofRow(f32, &narrow), 1, &arena, .{}));
    try std.testing.expectEqualStrings(outcome.text(), refused.message());

    // A shape only the binding sees is refused before anything is submitted.
    var mismatched: Deferred(DenseInsert) = undefined;
    try std.testing.expectError(Error.ShapeMismatch, index.deferred(executor, &mismatched).insert(keys[0..1], try .of(f32, &slab, dimensions), .{}));
}

test "a product lands ragged, and a softmax weighs every key the walk scores" {
    const dimensions = 2;
    var index = try DenseIndex.initRam(.{ .dimensions = dimensions, .metric = .inner }, &.{});
    defer index.deinit();
    try index.reserve(3);
    // Every value is exact in bf16, so the oracle below scores what the index stores.
    const slab = [_]f32{ 1, 0, 0, 1, 0.5, 0.75 };
    const keys = [_]Key{ .fromInteger(0), .fromInteger(1), .fromInteger(2) };
    try index.insert(&keys, try .of(f32, &slab, dimensions), .{});

    const left = [_]f32{ 1, 0, 0, 1 };
    var arena: Arena = try .init(null);
    defer arena.deinit();
    const products = try index.multiply(try .of(f32, &left, dimensions), &arena, .{});
    const offsets = try products.offsets();
    try std.testing.expectEqual(products.written, offsets[2]);
    try std.testing.expectEqual(products.written, (try products.keys()).len);
    try std.testing.expectEqual(products.written, (try products.scores()).len);
    try std.testing.expectEqual(offsets[2] - offsets[1], (try products.row(1)).keys.len);
    try std.testing.expect(products.required >= products.written);

    // A second arena keeps the first product alive past the next one.
    var second: Arena = try .init(null);
    defer second.deinit();
    const cut = try index.multiply(try .of(f32, &left, dimensions), &second, .{ .wanted = 1 });
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 2 }, try cut.offsets());
    try std.testing.expect((try products.keys()).len >= (try cut.keys()).len);

    // A graph this small walks every key, so attention is exact over all of them, whether the two
    // query rows are two tokens of one head or one token of two; a key without a row refuses.
    const Table = struct {
        rows: []const Scalar.BF16,
        resolvable: usize,
        fn resolve(self: *const @This(), key: Key) ?[]const Scalar.BF16 {
            const entry: usize = @intCast(key.asInteger().?);
            return if (entry < self.resolvable) self.rows[entry * dimensions ..][0..dimensions] else null;
        }
    };
    const values = [_]f32{ 10, 1, 20, 2, 30, 3 };
    // Value rows come in the index's own quantization, and a small integer's bf16 bits are its f32's high half.
    var stored: [values.len]Scalar.BF16 = undefined;
    for (&stored, values) |*bits, value| bits.* = .{ .bits = @truncate(@as(u32, @bitCast(value)) >> 16) };
    var table: Table = .{ .rows = &stored, .resolvable = keys.len };
    var carry: [2]DenseSoftmaxState = undefined;
    for ([_]usize{ 1, 2 }) |head_count| {
        const outputs = try index.attend(try .of(f32, &left, dimensions), .of(Scalar.BF16, &table, Table.resolve), &arena, .{
            .head_count = head_count,
            .carry = &carry,
            .expansion_factor = 8,
        });
        for (0..2) |query| {
            var logits: [keys.len]f64 = undefined;
            var maximum: f64 = -std.math.inf(f64);
            for (&logits, 0..) |*logit, key| {
                const product = left[query * 2] * slab[key * 2] + left[query * 2 + 1] * slab[key * 2 + 1];
                logit.* = @as(f64, product) / @sqrt(2.0);
                maximum = @max(maximum, logit.*);
            }
            var partition: f64 = 0;
            var mixed: [dimensions]f64 = @splat(0);
            for (logits, 0..) |logit, key| {
                const weight = @exp(logit - maximum);
                partition += weight;
                for (&mixed, 0..) |*column, position| column.* += weight * values[key * dimensions + position];
            }
            const attended = try outputs.row(f32, query);
            for (mixed, attended) |column, landed_column|
                try std.testing.expectApproxEqAbs(column / partition, landed_column, 1e-3 * @max(@abs(column / partition), 1));
            try std.testing.expectApproxEqAbs(maximum, carry[query].maximum_score, 1e-3 * @max(@abs(maximum), 1));
            try std.testing.expectApproxEqAbs(partition, carry[query].partition, 1e-3 * @max(@abs(partition), 1));
        }
    }
    table.resolvable = keys.len - 1;
    try std.testing.expectError(Error.ShapeMismatch, index.attend(try .of(f32, &left, dimensions), .of(Scalar.BF16, &table, Table.resolve), &arena, .{}));
    // A lookup over another kind than the index stores answers no row, which the call refuses alike.
    table.resolvable = keys.len;
    const Halves = struct {
        fn resolve(_: void, _: Key) ?[]const f16 {
            return &.{ 1, 2 };
        }
    };
    try std.testing.expectError(Error.ShapeMismatch, index.attend(try .of(f32, &left, dimensions), .of(f16, {}, Halves.resolve), &arena, .{}));

    const documents = try index.maxsim(try .of(f32, &left, dimensions), 2, 4, &arena, .{});
    try std.testing.expect((try documents.row(0)).keys.len != 0);
    try std.testing.expectError(Error.ShapeMismatch, index.maxsim(try .of(f32, &left, dimensions), 3, 4, &arena, .{}));
}

test "a reporting batch counts what it published, and a stopped sweep is no failure" {
    const dimensions = 3;
    var executor = try Executor.forRam(.{ .threads = 2 });
    defer executor.deinit();
    var index = try DenseIndex.initRam(.{ .dimensions = dimensions, .metric = .euclidean }, &.{});
    defer index.deinit();
    try index.on(executor).reserve(6);

    const slab = [_]f32{ 1, 0, 0, 0, 1, 0, 0, 0, 1 };
    const keys = [_]Key{ .fromInteger(1), .fromInteger(2), .fromInteger(3) };

    // How often a verb reports is the increment's size rather than a promise, so the counts are
    // asserted and their number never is.
    const Watch = struct {
        last: usize = 0,
        total: usize = 0,
        ordered: bool = true,

        fn record(self: *@This(), completed: usize, total: usize) void {
            self.ordered = self.ordered and completed >= self.last;
            self.last = completed;
            self.total = total;
        }
        fn carryOn(self: *@This(), completed: usize, total: usize) types.Continuation {
            self.record(completed, total);
            return .@"continue";
        }
        fn stop(self: *@This(), completed: usize, total: usize) types.Continuation {
            self.record(completed, total);
            return .cancel;
        }
    };

    var watched: Watch = .{};
    try index.on(executor).insert(&keys, try .of(f32, &slab, dimensions), .{ .progress = .of(&watched, Watch.carryOn) });
    try std.testing.expectEqual(Watch{ .last = 3, .total = 3, .ordered = true }, watched);

    // Every row is claimed before the first report, so an insertion's handler cannot stop it.
    var ignored: Watch = .{};
    const more = [_]Key{ .fromInteger(4), .fromInteger(5), .fromInteger(6) };
    try index.on(executor).insert(&more, try .of(f32, &slab, dimensions), .{ .progress = .of(&ignored, Watch.stop) });
    try std.testing.expectEqual(@as(usize, 6), (try index.on(executor).describe()).size);

    // Stopping a sweep is the time budget rather than a failure, so no status comes back.
    var swept: Watch = .{};
    _ = try index.on(executor).refine(.{ .progress = .of(&swept, Watch.stop) });
    try std.testing.expect(swept.ordered);
    try std.testing.expectEqual(@as(usize, 6), swept.total);
}
