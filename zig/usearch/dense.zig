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
//!
//! File: zig/usearch/dense.zig
//! Author: Ash Vardanian
//! Date: September 7, 2026

const std = @import("std");
const build_options = @import("build_options");

const types = @import("types.zig");
const Arena = types.Arena;
const BytesView = types.BytesView;
const Completion = types.Completion;
const Distance = types.Distance;
const Error = types.Error;
const Executor = types.Executor;
const Key = types.Key;
const Matches = types.Matches;
const QueryMatches = types.QueryMatches;
const Outcome = types.Outcome;
const Predicate = types.Predicate;
const PhaseProgress = types.PhaseProgress;
const PredicateFunction = types.PredicateFunction;
const ProgressCallback = types.ProgressCallback;
const ProgressFunction = types.ProgressFunction;
const RowsView = types.RowsView;
const Scalar = types.Scalar;
const Status = types.Status;
const Traversal = types.Traversal;
const outputSlice = types.outputSlice;
const ArenaWrite = types.ArenaWrite;
const refuseInto = types.refuseInto;
const usearch_keys_view_t = types.abi.keys_view_t;
const usearch_matches_tape_view_t = types.abi.matches_tape_view_t;
const usearch_keys_tape_view_t = types.abi.keys_tape_view_t;
const usearch_matches_view_t = types.abi.matches_view_t;
const usearch_nvme_options_t = types.abi.nvme_options_t;
const usearch_progress_t = types.abi.progress_t;
const usearch_ram_options_t = types.abi.ram_options_t;
const usearch_rows_view_t = types.abi.rows_view_t;
const usearch_vram_options_t = types.abi.vram_options_t;

/// Mirrors `usearch_value_lookup_t`.
const ValueLookupFunction = *const fn (?*anyopaque, [*]const Key, usize, Scalar, usize, [*]?*const anyopaque) callconv(.c) void;

/// Where `attend` reads value rows, as the C interface holds it: a `usearch_value_lookup_t` and
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
    ///
    /// The trampoline is a host function, so it serves RAM and NVMe handles only: a VRAM walk calls
    /// `callback` on the device, so a VRAM handle needs a device-callable one set directly.
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

extern fn usearch_outcome_wait(outcome: *Outcome) Status;
extern fn usearch_outcome_status(outcome: *Outcome) Status;
extern fn usearch_dense_metric_name(metric: DenseMetric) [*:0]const u8;
extern fn usearch_dense_metric_from_name(name: BytesView, metric: *DenseMetric) Status;
extern fn usearch_refine_verdict_name(verdict: DenseRefineVerdict) [*:0]const u8;
extern fn usearch_refine_verdict_from_name(name: BytesView, verdict: *DenseRefineVerdict) Status;

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
    /// L2 distance, zero only for identical vectors.
    euclidean = 3,
    /// Count of differing bits, so its range is the dimensionality.
    hamming = 4,
    /// Differing bits over union bits in [0, 1], ignoring the zeros both vectors share.
    jaccard = 5,
    _,

    /// The canonical spelling, as `.inner` spells "inner". Calls `usearch_dense_metric_name`.
    pub fn name(self: DenseMetric) [:0]const u8 {
        return std.mem.span(usearch_dense_metric_name(self));
    }

    /// Reads a spelling back, or null when no kind spells it. Calls `usearch_dense_metric_from_name`.
    pub fn fromName(text: []const u8) ?DenseMetric {
        var metric: DenseMetric = undefined;
        return if (usearch_dense_metric_from_name(.of(text), &metric) == .success) metric else null;
    }
};

/// The geometry and metric an empty index freezes - what `DenseIndex.initRam` and `initVram` read
/// beside their placement. Binding-only.
///
/// Stored vectors round up to whole steps of the distance kit the build picked, so the memory is
/// slightly larger than the logical vectors, and that is not a knob.
pub const DenseOptions = struct {
    /// The metric kind used for distance calculations between vectors.
    metric: DenseMetric = .angular,
    /// The scalar type vectors are stored and quantized into.
    quantization: Scalar = .bf16,
    /// Dimensions of the indexed vectors.
    dimensions: usize = 0,
    /// The optional connections-per-node bound; zero derives.
    connectivity: usize = 0,
    /// Upper bound on entries sharing one key; zero is unbounded.
    values_per_key: usize = 0,
};

/// The geometry and metric a checkpoint froze - everything a reader learns without opening it, and
/// nothing a reader must choose for itself. Mirrors `usearch_dense_metadata_t`.
///
/// `values_per_key` is absent because it bounds a live handle's mutable state rather than anything
/// the serialized form holds, and the placement is absent because a checkpoint records where it
/// was written and never where a reader must keep it. Both stay the caller's to set when loading.
pub const DenseMetadata = extern struct {
    /// The metric kind the checkpoint scores with.
    metric: DenseMetric = .unknown,
    /// The scalar type its vectors are stored and quantized into.
    quantization: Scalar = .unknown,
    /// Scalars per stored vector.
    dimensions: usize = 0,
    /// Connections per node the graph was built with.
    connectivity: usize = 0,
};

/// A live index's state, reported by `DenseIndex.describe`. Each counter is read once, so under
/// concurrent insertions they may come from slightly different moments.
/// Mirrors `usearch_dense_index_description_t`.
pub const DenseIndexDescription = extern struct {
    /// The frozen metric, quantization, dimensions and connectivity.
    metadata: DenseMetadata = .{},
    /// Stored entries.
    size: usize = 0,
    /// Entries hosted without growing again.
    capacity: usize = 0,
    /// Bytes the handle holds; traversal scratch is the executor's.
    memory_usage: usize = 0,
    /// The CPU class dispatch picked on this machine.
    cpu_class: types.abi.cpu_class_t = 0,
    /// The GPU class dispatch picked on the index's devices; zero for a host index.
    gpu_class: types.abi.gpu_class_t = 0,
    /// Live shards.
    shards_count: usize = 0,
    /// How the last load from a path read the parts.
    load_path: types.abi.load_path_t = 0,

    /// The CPU class's spelling. Calls `usearch_cpu_class_name`.
    pub fn cpuClassName(self: DenseIndexDescription) [:0]const u8 {
        return std.mem.span(types.abi.cpu_class_name(self.cpu_class));
    }

    /// The GPU class's spelling, "unknown" for a host index. Calls `usearch_gpu_class_name`.
    pub fn gpuClassName(self: DenseIndexDescription) [:0]const u8 {
        return std.mem.span(types.abi.gpu_class_name(self.gpu_class));
    }

    /// The load path's spelling: "none", "host-copy" or "cufile". Calls `usearch_load_path_name`.
    pub fn loadPathName(self: DenseIndexDescription) [:0]const u8 {
        return std.mem.span(types.abi.load_path_name(self.load_path));
    }
};

/// Traversal telemetry, fully overwritten by the call it is passed to.
/// Mirrors `usearch_traversal_stats_t`.
pub const DenseTraversalStats = extern struct {
    /// Graph members visited while resolving the batch.
    visited_members: usize = 0,
    /// Distances computed while resolving the batch, diversity re-selections included.
    computed_distances: usize = 0,
    /// Starvation restarts - a graph-connectivity health signal rather than a cost counter.
    restarts: usize = 0,
    /// Hint keys no shard held an entry under; zero unless a search was handed hints.
    missing_hints: usize = 0,
};

/// One reachability audit: the population it judged, and how much of it a search reaches.
/// Mirrors `usearch_reachability_t`.
pub const DenseReachability = extern struct {
    /// Entries published when the audit started; compare `reachable` against this, never a later size.
    audited: usize = 0,
    /// Of those, the entries a search reaches from the root; the rest wait for a `refine`.
    reachable: usize = 0,
    /// Depths the walk from the root took to converge; the deepest shard's on a sharded handle.
    depth_from_root: usize = 0,
    /// Depths the walk back toward the root took, where a health check walks both ways; the deepest
    /// shard's on a sharded handle.
    depth_to_root: usize = 0,
    /// Entries no reached entry links to.
    unlinked: usize = 0,
    /// Entries linked to by fewer entries than they link to.
    underlinked: usize = 0,
    /// Entries deeper than the depth holding the most entries.
    beyond_most_populated_depth: usize = 0,
    /// Islands no entry outside them links into.
    unreached_islands: usize = 0,
    /// Islands with no link out.
    stranded_islands: usize = 0,
    /// Components when direction is ignored.
    components: usize = 0,
};

/// What a round concluded, so no caller picks a change rate. Mirrors `usearch_refine_verdict_t`.
///
/// Non-exhaustive, so a verdict a newer engine reports arrives intact rather than coerced.
pub const DenseRefineVerdict = enum(c_int) {
    /// No round concluded: a report no round filled yet.
    none = 0,
    /// The round changed some entry's link set, so another round can help.
    improved = 1,
    /// The round changed no entry's link set.
    converged = 2,
    /// The round lowered nothing while something stands, as full entries leave no slot.
    stalled = 3,
    /// The caller stopped the round; the phase names where.
    stopped = 4,
    _,

    /// The canonical spelling, as `.converged` spells "converged".
    /// Calls `usearch_refine_verdict_name`.
    pub fn name(self: DenseRefineVerdict) [:0]const u8 {
        return std.mem.span(usearch_refine_verdict_name(self));
    }

    /// Reads a spelling back, or null when no verdict spells it.
    /// Calls `usearch_refine_verdict_from_name`.
    pub fn fromName(text: []const u8) ?DenseRefineVerdict {
        var verdict: DenseRefineVerdict = undefined;
        return if (usearch_refine_verdict_from_name(.of(text), &verdict) == .success) verdict else null;
    }
};

/// A refine a cancel left live, which the next `refine` naming it continues and
/// `refinePlanFinish` closes. Mirrors `usearch_dense_refine_plan_t`; the default holds none.
pub const DenseRefinePlan = extern struct {
    handle: ?*anyopaque = null,
};

/// A reshard a cancel left live, which the next `reshard` naming it continues and
/// `reshardPlanFinish` closes. Mirrors `usearch_dense_reshard_plan_t`; the default holds none.
pub const DenseReshardPlan = extern struct {
    handle: ?*anyopaque = null,
};

/// What one `refine` or `reshard` did to the graph. Mirrors `usearch_refine_report_t`.
pub const DenseRefineReport = extern struct {
    /// Entries whose neighbors were selected again, the islands of every heal round included.
    reselected: usize = 0,
    /// Of those, the entries whose neighbor set changed; never above `reselected`.
    changed: usize = 0,
    /// Entries the first audit found not reached both ways, counted once.
    islands_found: usize = 0,
    /// Links written between an island and an entry the audit reached both ways.
    islands_reconnected: usize = 0,
    /// Neighborhoods the round re-stitched.
    neighborhoods: usize = 0,
    /// Entries of the widest neighborhood.
    widest_neighborhood: usize = 0,
    /// Links the round added.
    added_links: usize = 0,
    /// Links the round dropped.
    dropped_links: usize = 0,
    /// What the round concluded.
    verdict: DenseRefineVerdict = .none,
    /// The health check opening the round.
    opening: DenseReachability = .{},
    /// The health check closing the heal, as a fresh `health` right after would take it.
    reachability: DenseReachability = .{},
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
    /// Upper bound on entries sharing one key, which no checkpoint records; zero is unbounded.
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
    /// Upper bound on entries sharing one key, which no checkpoint records; zero is unbounded.
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
    /// Upper bound on entries sharing one key, which no checkpoint records; zero is unbounded.
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
    /// Upper bound on entries sharing one key, which no checkpoint records; zero is unbounded.
    values_per_key: usize = 0,
    /// Called with bytes placed; a stopped load creates no index.
    progress: ProgressCallback = .{},
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    progress_steps: usize = 0,
    /// Where the call's final outcome, its static message included, is copied; null copies nothing.
    diagnostics: ?*Outcome = null,
};

/// Optional settings of `DenseIndex.viewNvme` and `DenseIndex.loadNvme`.
pub const DenseViewNvmeConfig = struct {
    /// The NVMe executor whose rings read the checkpoint's headers; null reads synchronously.
    executor: ?Executor = null,
    /// The devices the parts are served from; empty serves each where its manifest points.
    paths: []const BytesView = &.{},
    /// Memory domains of a key table held in RAM, which `count`, `get`, `sample`, MaxSim and hinted
    /// searches read; empty derives them, and null holds none.
    key_table_domains: ?[]const i32 = null,
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
    /// The breadth of the candidates the build walk explores; zero derives.
    expansion_factor: usize = 0,
    /// Called with entries admitted; it reports and cannot cancel, since every entry is claimed first.
    progress: ProgressCallback = .{},
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    progress_steps: usize = 0,
    stats: ?*DenseTraversalStats = null,
};

/// Optional settings of `DenseIndex.refine`.
pub const DenseRefineConfig = struct {
    /// The breadth of the candidates the re-selection walk explores; zero derives.
    expansion_factor: usize = 0,
    /// Bytes a loaded NVMe handle holds for the one part it rewrites; zero derives what rounds the
    /// part in RAM, and RAM and VRAM handles never read it.
    memory_limit: usize = 0,
    /// Called at every barrier of every phase with that phase's units; a stop answers
    /// `Error.Cancelled`.
    progress: ProgressCallback = .{},
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    progress_steps: usize = 0,
    stats: ?*DenseTraversalStats = null,
    /// Null runs one round to its end; an empty plan opens one a cancel leaves live, and a live
    /// one continues.
    plan: ?*DenseRefinePlan = null,
    /// Where the call stopped, `.done` once its round ended; null skips it.
    phase: ?*PhaseProgress = null,
};

/// Optional settings of `DenseIndex.search`.
pub const DenseSearchConfig = struct {
    /// The breadth of the candidates the walk explores; zero derives.
    expansion_factor: usize = 0,
    /// A walk, or an exact scan of every entry on RAM and NVMe handles; VRAM answers `Error.Unsupported`.
    traversal: Traversal = .walk,
    /// Admits only the keys it approves; non-matching entries still route the walk. Refused on VRAM.
    predicate: Predicate = .{},
    /// Called with queries resolved; a stopped search answers `Error.Cancelled` and no matches.
    progress: ProgressCallback = .{},
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    progress_steps: usize = 0,
    stats: ?*DenseTraversalStats = null,
};

/// Optional placements and settings of `DenseIndex.reshard`; a null placement keeps the
/// handle's own.
pub const DenseReshardConfig = struct {
    /// NUMA node IDs of a RAM placement recut in place.
    memory_domains: ?[]const i32 = null,
    /// CUDA ordinals of a VRAM placement recut in place.
    devices: ?[]const i32 = null,
    /// Drives of an NVMe placement recut in place.
    paths: ?[]const BytesView = null,
    /// Entries per shard of the RAM or VRAM placement; zero keeps the handle's own bound.
    shard_capacity: usize = 0,
    /// The breadth of the candidates of the closing rounds; zero derives.
    expansion_factor: usize = 0,
    /// Bytes a loaded NVMe target holds for the one part it rewrites; zero derives what rounds the
    /// part in RAM, and RAM and VRAM targets never read it.
    memory_limit: usize = 0,
    /// Picks the part roots of a source split among several targets: equal seeds split equally.
    seed: u64 = 0,
    /// Called at every barrier of the closing rounds' phases; a stop never undoes a move.
    progress: ProgressCallback = .{},
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    progress_steps: usize = 0,
    stats: ?*DenseTraversalStats = null,
    /// Null moves and runs to the end; an empty plan opens one a cancel leaves live, and a live
    /// one continues.
    plan: ?*DenseReshardPlan = null,
    /// Where the call stopped, `.done` once its rounds ended; null skips it.
    phase: ?*PhaseProgress = null,
};

/// Optional settings of `DenseIndex.sample`.
pub const DenseSampleConfig = struct {
    /// The keys already shown: never drawn again, and the paths reach toward them.
    shown: []const Key = &.{},
    /// Where each `shown` key sits, one point apiece.
    shown_coordinates: []const [3]f32 = &.{},
    /// The keys to look around; empty looks around the whole index.
    focus: []const Key = &.{},
    /// Equal seeds, shown keys and focus over an unchanged index draw equally.
    seed: u64 = 0,
    /// The breadth of the candidates of the walks behind the draws; zero derives.
    expansion_factor: usize = 0,
    progress: ProgressCallback = .{},
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    progress_steps: usize = 0,
    stats: ?*DenseTraversalStats = null,
};

/// Optional settings of `DenseIndex.multiply`.
pub const DenseMultiplyConfig = struct {
    /// Bound on pairs kept per left row, the first ones its walk scored; zero keeps every pair.
    wanted: usize = 0,
    /// The breadth of the candidates, which widens the emitted pattern; zero derives.
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
    /// The softmax state, one per query row, passed rather than written in the arena; null skips
    /// it.
    carry: ?[]DenseSoftmaxState = null,
    /// The breadth of the candidates; zero derives. Wider candidates walk further, folding more of the tail.
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
    /// The breadth of the candidates; zero derives.
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
    /// The breadth of the candidates the walk explores; zero derives.
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
    connectivity: usize,
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
    connectivity: usize,
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
    connectivity: usize,
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
    connectivity: usize,
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
    ram: ?*const usearch_ram_options_t = null,
};

const usearch_dense_load_nvme_t = extern struct {
    index: *?*anyopaque,
    executor: ?*anyopaque,
    outcome: *Outcome,
    completion: Completion,
    path: BytesView,
    nvme: usearch_nvme_options_t,
    ram: ?*const usearch_ram_options_t = null,
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

const usearch_refine_options_t = extern struct {
    expansion_factor: usize,
    memory_limit: usize,
};

const usearch_dense_refine_t = extern struct {
    index: ?*anyopaque,
    executor: ?*anyopaque,
    outcome: *Outcome,
    completion: Completion,
    report: ?*DenseRefineReport,
    refine: usearch_refine_options_t,
    progress: usearch_progress_t,
    stats: ?*DenseTraversalStats,
    plan: ?*?*anyopaque,
    phase: ?*PhaseProgress,
};

const usearch_dense_refine_plan_finish_t = extern struct {
    index: ?*anyopaque,
    executor: ?*anyopaque,
    outcome: *Outcome,
    completion: Completion,
    plan: ?*anyopaque,
    report: ?*DenseRefineReport,
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
    hints: ?*const usearch_matches_view_t = null,
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

const usearch_dense_reshard_t = extern struct {
    sources_count: usize,
    sources: [*]const ?*anyopaque,
    targets_count: usize,
    targets: [*]const ?*anyopaque,
    executor: ?*anyopaque,
    outcome: *Outcome,
    completion: Completion,
    ram: ?*const usearch_ram_options_t,
    vram: ?*const usearch_vram_options_t,
    nvme: ?*const usearch_nvme_options_t,
    report: ?*DenseRefineReport,
    refine: usearch_refine_options_t,
    seed: u64,
    progress: usearch_progress_t,
    stats: ?*DenseTraversalStats,
    plan: ?*?*anyopaque,
    phase: ?*PhaseProgress,
};

const usearch_dense_reshard_plan_finish_t = extern struct {
    executor: ?*anyopaque,
    outcome: *Outcome,
    completion: Completion,
    plan: ?*anyopaque,
    report: ?*DenseRefineReport,
};

const usearch_dense_sample_t = extern struct {
    index: ?*anyopaque,
    executor: ?*anyopaque,
    outcome: *Outcome,
    completion: Completion,
    shown_count: usize,
    shown: usearch_keys_view_t,
    shown_coordinates: usearch_rows_view_t,
    focus_count: usize,
    focus: usearch_keys_view_t,
    keys: *usearch_keys_view_t,
    entry_ordinals: *?[*]const usize,
    parents: *usearch_matches_view_t,
    coordinates: *usearch_rows_view_t,
    written: *usize,
    arena: ?*anyopaque,
    wanted: usize,
    seed: u64,
    expansion_factor: usize,
    progress: usearch_progress_t,
    stats: ?*DenseTraversalStats,
};

const usearch_dense_cursor_init_t = extern struct {
    index: ?*anyopaque,
    executor: ?*anyopaque = null,
    outcome: *Outcome,
    starts_count: usize,
    starts: ?[*]const Key,
    cursor: *?*anyopaque,
};

const usearch_dense_cursor_next_t = extern struct {
    index: ?*anyopaque,
    executor: ?*anyopaque,
    outcome: *Outcome,
    completion: Completion,
    cursor: ?*anyopaque,
    keys: *usearch_keys_view_t,
    links: ?*usearch_keys_tape_view_t,
    written: *usize,
    arena: ?*anyopaque,
    wanted: usize,
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

const usearch_dense_describe_t = extern struct {
    index: ?*anyopaque,
    executor: ?*anyopaque,
    outcome: *Outcome,
    completion: Completion,
    description: *DenseIndexDescription,
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
    traversal: Traversal = .walk,
    predicate: Predicate = .{},
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
    traversal: Traversal = .walk,
    predicate: Predicate = .{},
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
    query_offsets: ?[*]const usize = null,
    rows_per_query: usize,
    matches: *usearch_matches_tape_view_t,
    written: ?*usize,
    required: ?*usize,
    arena: ?*anyopaque,
    wanted: usize,
    expansion_factor: usize,
    traversal: Traversal = .walk,
    predicate: Predicate = .{},
    hints: ?*const usearch_matches_view_t = null,
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
extern fn usearch_dense_load_nvme(arguments: *usearch_dense_load_nvme_t) void;
extern fn usearch_dense_free(index: ?*anyopaque) void;
extern fn usearch_dense_reserve(arguments: *usearch_dense_reserve_t) void;
extern fn usearch_dense_clear(arguments: *usearch_dense_clear_t) void;
extern fn usearch_dense_insert(arguments: *usearch_dense_insert_t) void;
extern fn usearch_dense_refine(arguments: *usearch_dense_refine_t) void;
extern fn usearch_dense_health(arguments: *usearch_dense_health_t) void;
extern fn usearch_dense_search(arguments: *usearch_dense_search_t) void;
extern fn usearch_dense_count(arguments: *usearch_dense_count_t) void;
extern fn usearch_dense_get(arguments: *usearch_dense_get_t) void;
extern fn usearch_dense_refine_plan_finish(arguments: *usearch_dense_refine_plan_finish_t) void;
extern fn usearch_dense_reshard(arguments: *usearch_dense_reshard_t) void;
extern fn usearch_dense_reshard_plan_finish(arguments: *usearch_dense_reshard_plan_finish_t) void;
extern fn usearch_dense_sample(arguments: *usearch_dense_sample_t) void;
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
/// const nearest = (try matches.query(0)).keys[0];
/// ```
pub const DenseIndex = extern struct {
    /// The opaque `usearch_dense_index_t` the ABI hands back.
    handle: ?*anyopaque,

    /// Every verb of the index, blocking on an executor, refused while another call holds it.
    pub const On = Receiver(.blocking, void);

    /// Every verb of the index, submitted to an executor into a `Deferred(Result)` the caller
    /// holds; only the verbs answering `Result` compile.
    pub fn DeferredOn(comptime Result: type) type {
        return Receiver(.deferred, Result);
    }

    /// Runs every verb on `executor`, whose kind must match the index's residency. Binding-only.
    pub fn on(self: DenseIndex, executor: Executor) On {
        return .{ .index = self.handle, .executor = executor.handle, .into = {}, .diagnostics = null };
    }

    /// Submits a verb to `executor`, holding it in `call`, a `*Deferred(Result)` the caller places
    /// anywhere, which stays put until its `deinit`. Binding-only.
    pub fn deferred(self: DenseIndex, executor: Executor, call: anytype) DeferredOn(@FieldType(@TypeOf(call.*), "result")) {
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
            .connectivity = 0,
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
            .connectivity = 0,
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
            .connectivity = 0,
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
            .connectivity = 0,
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

    /// Serves a checkpoint in place from NVMe drives, read-only and sized past RAM. Calls
    /// `usearch_dense_view_nvme`.
    pub fn viewNvme(path: []const u8, config: DenseViewNvmeConfig) Error!DenseIndex {
        var handle: ?*anyopaque = null;
        var outcome: Outcome = .{};
        const key_table: ?usearch_ram_options_t = if (config.key_table_domains) |domains| .of(domains) else null;
        var arguments: usearch_dense_view_nvme_t = .{
            .index = &handle,
            .executor = handleOf(config.executor),
            .outcome = &outcome,
            .completion = .blocking,
            .path = .of(path),
            .nvme = .of(config.paths),
            .ram = if (key_table) |*ram| ram else null,
        };
        usearch_dense_view_nvme(&arguments);
        return adopt(handle, outcome, config.diagnostics);
    }

    /// Opens a checkpoint's manifest from NVMe drives as a writable index: it answers every read a
    /// view answers, and `refine` rewrites its parts one at a time. A lone monolith has no
    /// manifest to replace and answers `Error.Unsupported`, as do `paths` naming any.
    /// Calls `usearch_dense_load_nvme`.
    pub fn loadNvme(path: []const u8, config: DenseViewNvmeConfig) Error!DenseIndex {
        var handle: ?*anyopaque = null;
        var outcome: Outcome = .{};
        const key_table: ?usearch_ram_options_t = if (config.key_table_domains) |domains| .of(domains) else null;
        var arguments: usearch_dense_load_nvme_t = .{
            .index = &handle,
            .executor = handleOf(config.executor),
            .outcome = &outcome,
            .completion = .blocking,
            .path = .of(path),
            .nvme = .of(config.paths),
            .ram = if (key_table) |*ram| ram else null,
        };
        usearch_dense_load_nvme(&arguments);
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

    /// Every vector stored under `key` as `Element`, one packed row per entry in insertion order,
    /// written in `arena`; an absent key yields no rows. Calls `usearch_dense_get`.
    pub fn get(self: DenseIndex, comptime Element: type, key: Key, arena: *Arena) Error!DenseVectors(Element) {
        return self.here().get(Element, key, arena);
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

    /// Runs one round over the finished graph: a health check, a heal of its islands, and a gather
    /// and restitch of every neighborhood, closed by a second health check, reporting what changed.
    ///
    /// A build step rather than a repair: a cheap insertion pass followed by refinement reaches a
    /// better graph, in less total time, than one expensive insertion pass. A stop at a barrier
    /// answers `Error.Cancelled`, and a `plan` in the config keeps the round to continue later.
    /// Calls `usearch_dense_refine`.
    pub fn refine(self: DenseIndex, config: DenseRefineConfig) Error!DenseRefineReport {
        return self.here().refine(config);
    }

    /// Closes a live refine plan, reporting what the whole plan did; the plan is freed and emptied
    /// whatever the call answers. Calls `usearch_dense_refine_plan_finish`.
    pub fn refinePlanFinish(self: DenseIndex, plan: *DenseRefinePlan) Error!DenseRefineReport {
        return self.here().refinePlanFinish(plan);
    }

    /// Measures how much of the graph a search reaches from the root; the entries it misses stay
    /// unfound until a `refine` re-links them. Calls `usearch_dense_health`.
    pub fn health(self: DenseIndex) Error!DenseReachability {
        return self.here().health();
    }

    /// Resolves a run of queries into up to `wanted` matches apiece, written in `arena` - the native
    /// batch operation.
    ///
    /// Never writes to the graph, so any number of these may overlap with concurrent insertions.
    /// A zero `wanted` walks nothing and lands only the counts, every one zero.
    /// Calls `usearch_dense_search`.
    pub fn search(self: DenseIndex, queries: RowsView, wanted: usize, arena: *Arena, config: DenseSearchConfig) Error!Matches {
        return self.here().search(queries, wanted, arena, config);
    }

    /// Draws up to `wanted` keys around `config.focus`, or across the whole index, each hanging
    /// from a shown or earlier draw and placed in three dimensions beside the shown ones, written
    /// in `arena`; a larger `wanted` draws a superset. Only RAM samples yet. Calls `usearch_dense_sample`.
    pub fn sample(self: DenseIndex, wanted: usize, arena: *Arena, config: DenseSampleConfig) Error!DenseDraws {
        return self.here().sample(wanted, arena, config);
    }

    /// Copies up to `wanted` more entries out of `cursor` into `arena` - their keys, and the keys
    /// each links to when `links` asks - answering an empty batch once the walk is exhausted. The
    /// vectors themselves come from `get`. Calls `usearch_dense_cursor_next`.
    pub fn cursorNext(self: DenseIndex, cursor: DenseCursor, wanted: usize, links: bool, arena: *Arena) Error!DenseCursorBatch {
        return self.here().cursorNext(cursor, wanted, links, arena);
    }

    /// Pulls up to `wanted` more matches out of `stream` into `arena`, as one query's matches,
    /// answering an empty query once the walk is drained.
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
    /// written in `arena`. Calls `usearch_dense_save_buffer`.
    pub fn saveBuffer(self: DenseIndex, arena: *Arena, config: DenseSaveBufferConfig) Error!Monolith {
        return self.here().saveBuffer(arena, config);
    }

    /// Approximate sparse product of the left-hand rows against the indexed vectors, reporting
    /// every pair the traversal scored instead of only the nearest, as a ragged output written in
    /// `arena` with Apache Arrow offsets.
    ///
    /// Pairs arrive in traversal order rather than sorted, and hold no duplicates. Cost is
    /// roughly `rows × expansion × connectivity × dimensions`, sublinear in the indexed entries.
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
    /// rows, a host function on RAM and NVMe and a device one on VRAM. The attended rows land in
    /// `arena` as `Element`, as wide as the keys and divided by the partition, and `carry` is worth
    /// asking for only to fold this branch into another over the same queries. Runs on the
    /// inner-product kind only. Calls `usearch_dense_attend`.
    pub fn attend(self: DenseIndex, comptime Element: type, queries: RowsView, value_lookup: ValueLookup, arena: *Arena, config: DenseAttendConfig) Error!DenseVectors(Element) {
        return self.here().attend(Element, queries, value_lookup, arena, config);
    }

    /// Late-interaction scoring of multi-row queries against multi-vector documents: every query
    /// row takes its best match within a key, and those maxima sum to that key's score.
    ///
    /// `queries` holds the queries laid end to end, `rows_per_query` rows apiece and uniform, so
    /// ColBERT's padding applies as is, and each query's keys land in `arena` largest first, up to
    /// `wanted` of them, or every key it scored when `wanted` is zero. Approximate downward only,
    /// and runs on the inner-product kind only. Calls `usearch_dense_maxsim`.
    pub fn maxsim(self: DenseIndex, queries: RowsView, rows_per_query: usize, wanted: usize, arena: *Arena, config: DenseMaxsimConfig) Error!DenseProducts {
        return self.here().maxsim(queries, rows_per_query, wanted, arena, config);
    }

    /// Moves the entries of `sources` into `targets` on the calling thread, keeping their links
    /// rather than re-inserting their vectors, then runs a round on every target that took entries.
    ///
    /// Several sources into one target, itself among them, gather into it; one source into
    /// several targets cuts it into graph-local parts; one handle as its only source and target
    /// recuts its own shards in place. A source the targets do not name ends empty and live, and
    /// every handle stays the caller's. Calls `usearch_dense_reshard`.
    pub fn reshard(sources: []const DenseIndex, targets: []const DenseIndex, config: DenseReshardConfig) Error!DenseRefineReport {
        return (DenseIndex{ .handle = null }).here().reshard(sources, targets, config);
    }

    /// Closes a live reshard plan on the calling thread, reporting what the whole plan did; the
    /// plan is freed and emptied whatever the call answers.
    /// Calls `usearch_dense_reshard_plan_finish`.
    pub fn reshardPlanFinish(plan: *DenseReshardPlan) Error!DenseRefineReport {
        return (DenseIndex{ .handle = null }).here().reshardPlanFinish(plan);
    }
};

fn handleOf(executor: ?Executor) ?*anyopaque {
    return if (executor) |chosen| chosen.handle else null;
}

/// The verbs of `DenseIndex` once more, for a receiver that completes them as `completion` says:
/// blocking answers the result, deferred submits into the caller's `Deferred(Into)`. Each verb is
/// written once, filling its C structure with the result as its output slot.
fn Receiver(comptime completion: Completion, comptime Into: type) type {
    return struct {
        const Self = @This();

        index: ?*anyopaque,
        executor: ?*anyopaque,
        /// Where a deferred call is held until final.
        into: if (completion == .deferred) *Deferred(Into) else void,
        /// Where a blocking call copies its final outcome; a `Deferred` keeps its own instead.
        diagnostics: if (completion == .blocking) ?*Outcome else void,

        fn Answer(comptime Result: type) type {
            return if (completion == .deferred) Error!void else Error!Result;
        }

        /// The storage a call reports into until final: the caller's `Deferred`, or `local`.
        fn hold(self: Self, local: anytype, result: @TypeOf(local.result)) @TypeOf(local) {
            const held = if (completion == .deferred) self.into else local;
            held.* = .{ .result = result };
            return held;
        }

        /// The result of a blocking call, final on return; nothing for a submitted one.
        fn end(self: Self, held: anytype) Answer(@TypeOf(held.result)) {
            if (completion == .deferred) return;
            if (self.diagnostics) |passed| passed.* = held.outcome;
            try held.outcome.check();
            return held.result;
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
            var passing = self;
            passing.diagnostics = outcome;
            return passing;
        }

        fn containsBlocking(self: Self, key: Key) Error!bool {
            return try self.count(key) != 0;
        }

        /// See `DenseIndex.reserve`. Calls `usearch_dense_reserve`.
        pub fn reserve(self: Self, capacity: usize) Answer(void) {
            var local: Deferred(void) = undefined;
            const held = self.hold(&local, {});
            var arguments: usearch_dense_reserve_t = .{ .index = self.index, .executor = self.executor, .outcome = &held.outcome, .completion = completion, .capacity = capacity };
            usearch_dense_reserve(&arguments);
            return self.end(held);
        }

        /// See `DenseIndex.clear`. Calls `usearch_dense_clear`.
        pub fn clear(self: Self) Answer(void) {
            var local: Deferred(void) = undefined;
            const held = self.hold(&local, {});
            var arguments: usearch_dense_clear_t = .{ .index = self.index, .executor = self.executor, .outcome = &held.outcome, .completion = completion };
            usearch_dense_clear(&arguments);
            return self.end(held);
        }

        /// See `DenseIndex.describe`. Calls `usearch_dense_describe`.
        pub fn describe(self: Self) Answer(DenseIndexDescription) {
            var local: Deferred(DenseIndexDescription) = undefined;
            const held = self.hold(&local, .{});
            var arguments: usearch_dense_describe_t = .{ .index = self.index, .executor = self.executor, .outcome = &held.outcome, .completion = completion, .description = &held.result };
            usearch_dense_describe(&arguments);
            return self.end(held);
        }

        /// See `DenseIndex.count`. Calls `usearch_dense_count`.
        pub fn count(self: Self, key: Key) Answer(usize) {
            var local: Deferred(usize) = undefined;
            const held = self.hold(&local, 0);
            var arguments: usearch_dense_count_t = .{ .index = self.index, .executor = self.executor, .outcome = &held.outcome, .completion = completion, .key = key, .entries = &held.result };
            usearch_dense_count(&arguments);
            return self.end(held);
        }

        /// See `DenseIndex.get`. Calls `usearch_dense_get`.
        pub fn get(self: Self, comptime Element: type, key: Key, arena: *Arena) Answer(DenseVectors(Element)) {
            var local: Deferred(DenseVectors(Element)) = undefined;
            const held = self.hold(&local, .{});
            held.result.write = .stamp(arena.handle);
            var arguments: usearch_dense_get_t = .{
                .index = self.index,
                .executor = self.executor,
                .outcome = &held.outcome,
                .completion = completion,
                .key = key,
                .vectors = &held.result.view,
                .written = &held.result.rows,
                .arena = arena.handle,
                .outputs_scalar = comptime Scalar.of(Element),
            };
            usearch_dense_get(&arguments);
            return self.end(held);
        }

        /// See `DenseIndex.insert`. Calls `usearch_dense_insert`.
        pub fn insert(self: Self, keys: []const Key, vectors: RowsView, config: DenseInsertConfig) Answer(void) {
            // C takes one count for both runs, so only the binding sees the two lengths disagree.
            if (keys.len != vectors.rows) return self.refused(.shape_mismatch, "The key run and the rows disagree in length");
            var local: Deferred(void) = undefined;
            const held = self.hold(&local, {});
            var arguments: usearch_dense_insert_t = .{
                .index = self.index,
                .executor = self.executor,
                .outcome = &held.outcome,
                .completion = completion,
                .count = vectors.rows,
                .keys = .of(keys),
                .vectors = .of(vectors),
                .expansion_factor = config.expansion_factor,
                .progress = .of(config.progress, config.progress_steps),
                .stats = config.stats,
            };
            usearch_dense_insert(&arguments);
            return self.end(held);
        }

        /// See `DenseIndex.refine`. Calls `usearch_dense_refine`.
        pub fn refine(self: Self, config: DenseRefineConfig) Answer(DenseRefineReport) {
            var local: Deferred(DenseRefineReport) = undefined;
            const held = self.hold(&local, .{});
            var arguments: usearch_dense_refine_t = .{
                .index = self.index,
                .executor = self.executor,
                .outcome = &held.outcome,
                .completion = completion,
                .report = &held.result,
                .refine = .{ .expansion_factor = config.expansion_factor, .memory_limit = config.memory_limit },
                .progress = .of(config.progress, config.progress_steps),
                .stats = config.stats,
                .plan = if (config.plan) |plan| &plan.handle else null,
                .phase = config.phase,
            };
            usearch_dense_refine(&arguments);
            return self.end(held);
        }

        /// See `DenseIndex.refinePlanFinish`. Calls `usearch_dense_refine_plan_finish`.
        pub fn refinePlanFinish(self: Self, plan: *DenseRefinePlan) Answer(DenseRefineReport) {
            var local: Deferred(DenseRefineReport) = undefined;
            const held = self.hold(&local, .{});
            var arguments: usearch_dense_refine_plan_finish_t = .{
                .index = self.index,
                .executor = self.executor,
                .outcome = &held.outcome,
                .completion = completion,
                .plan = plan.handle,
                .report = &held.result,
            };
            plan.* = .{};
            usearch_dense_refine_plan_finish(&arguments);
            return self.end(held);
        }

        /// See `DenseIndex.health`. Calls `usearch_dense_health`.
        pub fn health(self: Self) Answer(DenseReachability) {
            var local: Deferred(DenseReachability) = undefined;
            const held = self.hold(&local, .{});
            var arguments: usearch_dense_health_t = .{ .index = self.index, .executor = self.executor, .outcome = &held.outcome, .completion = completion, .reachability = &held.result };
            usearch_dense_health(&arguments);
            return self.end(held);
        }

        /// See `DenseIndex.search`. Calls `usearch_dense_search`.
        pub fn search(self: Self, queries: RowsView, wanted: usize, arena: *Arena, config: DenseSearchConfig) Answer(Matches) {
            var local: Deferred(Matches) = undefined;
            const held = self.hold(&local, .{ .queries = queries.rows });
            held.result.write = .stamp(arena.handle);
            var arguments: usearch_dense_search_t = .{
                .index = self.index,
                .executor = self.executor,
                .outcome = &held.outcome,
                .completion = completion,
                .count = queries.rows,
                .queries = .of(queries),
                .matches = &held.result.view,
                .arena = arena.handle,
                .wanted = wanted,
                .expansion_factor = config.expansion_factor,
                .traversal = config.traversal,
                .predicate = config.predicate,
                .progress = .of(config.progress, config.progress_steps),
                .stats = config.stats,
            };
            usearch_dense_search(&arguments);
            return self.end(held);
        }

        /// See `DenseIndex.reshard`; the handles come from the two lists, the executor and the
        /// completion from this receiver. Calls `usearch_dense_reshard`.
        pub fn reshard(self: Self, sources: []const DenseIndex, targets: []const DenseIndex, config: DenseReshardConfig) Answer(DenseRefineReport) {
            var local: Deferred(DenseRefineReport) = undefined;
            const held = self.hold(&local, .{});
            if (config.memory_domains) |domains| held.placements.ram = .of(domains);
            if (config.devices) |devices| held.placements.vram = .of(devices);
            if (config.paths) |paths| held.placements.nvme = .of(paths);
            held.placements.ram.shard_capacity = config.shard_capacity;
            held.placements.vram.shard_capacity = config.shard_capacity;
            var arguments: usearch_dense_reshard_t = .{
                .sources_count = sources.len,
                .sources = @ptrCast(sources.ptr),
                .targets_count = targets.len,
                .targets = @ptrCast(targets.ptr),
                .executor = self.executor,
                .outcome = &held.outcome,
                .completion = completion,
                .ram = if (config.memory_domains != null) &held.placements.ram else null,
                .vram = if (config.devices != null) &held.placements.vram else null,
                .nvme = if (config.paths != null) &held.placements.nvme else null,
                .report = &held.result,
                .refine = .{ .expansion_factor = config.expansion_factor, .memory_limit = config.memory_limit },
                .seed = config.seed,
                .progress = .of(config.progress, config.progress_steps),
                .stats = config.stats,
                .plan = if (config.plan) |plan| &plan.handle else null,
                .phase = config.phase,
            };
            usearch_dense_reshard(&arguments);
            return self.end(held);
        }

        /// See `DenseIndex.reshardPlanFinish`. Calls `usearch_dense_reshard_plan_finish`.
        pub fn reshardPlanFinish(self: Self, plan: *DenseReshardPlan) Answer(DenseRefineReport) {
            var local: Deferred(DenseRefineReport) = undefined;
            const held = self.hold(&local, .{});
            var arguments: usearch_dense_reshard_plan_finish_t = .{
                .executor = self.executor,
                .outcome = &held.outcome,
                .completion = completion,
                .plan = plan.handle,
                .report = &held.result,
            };
            plan.* = .{};
            usearch_dense_reshard_plan_finish(&arguments);
            return self.end(held);
        }

        /// See `DenseIndex.sample`. Calls `usearch_dense_sample`.
        pub fn sample(self: Self, wanted: usize, arena: *Arena, config: DenseSampleConfig) Answer(DenseDraws) {
            // C reads one count for both runs, so only the binding sees the two lengths disagree.
            if (config.shown.len != config.shown_coordinates.len) return self.refused(.shape_mismatch, "The shown keys and coordinates disagree in length");
            var local: Deferred(DenseDraws) = undefined;
            const held = self.hold(&local, .{});
            held.result.write = .stamp(arena.handle);
            var arguments: usearch_dense_sample_t = .{
                .index = self.index,
                .executor = self.executor,
                .outcome = &held.outcome,
                .completion = completion,
                .shown_count = config.shown.len,
                .shown = .of(config.shown),
                .shown_coordinates = .{ .data = config.shown_coordinates.ptr, .stride = @sizeOf([3]f32), .columns = 3, .scalar = .f32 },
                .focus_count = config.focus.len,
                .focus = .of(config.focus),
                .keys = &held.result.keys_view,
                .entry_ordinals = &held.result.entry_ordinals_view,
                .parents = &held.result.parents_view,
                .coordinates = &held.result.coordinates_view,
                .written = &held.result.written,
                .arena = arena.handle,
                .wanted = wanted,
                .seed = config.seed,
                .expansion_factor = config.expansion_factor,
                .progress = .of(config.progress, config.progress_steps),
                .stats = config.stats,
            };
            usearch_dense_sample(&arguments);
            return self.end(held);
        }

        /// See `DenseIndex.cursorNext`. Calls `usearch_dense_cursor_next`.
        pub fn cursorNext(self: Self, cursor: DenseCursor, wanted: usize, links: bool, arena: *Arena) Answer(DenseCursorBatch) {
            var local: Deferred(DenseCursorBatch) = undefined;
            const held = self.hold(&local, .{});
            held.result.write = .stamp(arena.handle);
            var arguments: usearch_dense_cursor_next_t = .{
                .index = self.index,
                .executor = self.executor,
                .outcome = &held.outcome,
                .completion = completion,
                .cursor = cursor.handle,
                .keys = &held.result.keys_view,
                .links = if (links) &held.result.links_view else null,
                .written = &held.result.written,
                .arena = arena.handle,
                .wanted = wanted,
            };
            usearch_dense_cursor_next(&arguments);
            return self.end(held);
        }

        /// See `DenseIndex.searchStreamNext`. Calls `usearch_dense_search_stream_next`.
        pub fn searchStreamNext(self: Self, stream: DenseSearchStream, wanted: usize, arena: *Arena) Answer(Matches) {
            var local: Deferred(Matches) = undefined;
            const held = self.hold(&local, .{ .queries = 1 });
            held.result.write = .stamp(arena.handle);
            var arguments: usearch_dense_search_stream_next_t = .{
                .index = self.index,
                .executor = self.executor,
                .outcome = &held.outcome,
                .completion = completion,
                .stream = stream.handle,
                .matches = &held.result.view,
                .arena = arena.handle,
                .wanted = wanted,
            };
            usearch_dense_search_stream_next(&arguments);
            return self.end(held);
        }

        /// See `DenseIndex.save`. Calls `usearch_dense_save`.
        pub fn save(self: Self, path: []const u8, config: DenseSaveConfig) Answer(void) {
            var local: Deferred(void) = undefined;
            const held = self.hold(&local, {});
            var arguments: usearch_dense_save_t = .{
                .index = self.index,
                .executor = self.executor,
                .outcome = &held.outcome,
                .completion = completion,
                .path = .of(path),
                .progress = .of(config.progress, config.progress_steps),
            };
            usearch_dense_save(&arguments);
            return self.end(held);
        }

        /// See `DenseIndex.saveBuffer`. Calls `usearch_dense_save_buffer`.
        pub fn saveBuffer(self: Self, arena: *Arena, config: DenseSaveBufferConfig) Answer(Monolith) {
            var local: Deferred(Monolith) = undefined;
            const held = self.hold(&local, .{});
            held.result.write = .stamp(arena.handle);
            var arguments: usearch_dense_save_buffer_t = .{
                .index = self.index,
                .executor = self.executor,
                .outcome = &held.outcome,
                .completion = completion,
                .monolith = &held.result.view,
                .arena = arena.handle,
                .progress = .of(config.progress, config.progress_steps),
            };
            usearch_dense_save_buffer(&arguments);
            return self.end(held);
        }

        /// See `DenseIndex.multiply`. Calls `usearch_dense_multiply`.
        pub fn multiply(self: Self, left: RowsView, arena: *Arena, config: DenseMultiplyConfig) Answer(DenseProducts) {
            var local: Deferred(DenseProducts) = undefined;
            const held = self.hold(&local, .{ .queries = left.rows });
            held.result.write = .stamp(arena.handle);
            var arguments: usearch_dense_multiply_t = .{
                .index = self.index,
                .executor = self.executor,
                .outcome = &held.outcome,
                .completion = completion,
                .count = left.rows,
                .left = .of(left),
                .products = &held.result.view,
                .written = &held.result.written,
                .required = &held.result.required,
                .arena = arena.handle,
                .wanted = config.wanted,
                .expansion_factor = config.expansion_factor,
                .progress = .of(config.progress, config.progress_steps),
                .stats = config.stats,
            };
            usearch_dense_multiply(&arguments);
            return self.end(held);
        }

        /// See `DenseIndex.attend`. Calls `usearch_dense_attend`.
        pub fn attend(self: Self, comptime Element: type, queries: RowsView, value_lookup: ValueLookup, arena: *Arena, config: DenseAttendConfig) Answer(DenseVectors(Element)) {
            // C reads one carry per query and has no length to check it against.
            if (config.carry) |carry| if (carry.len < queries.rows)
                return self.refused(.shape_mismatch, "The carry holds fewer states than there are query rows");
            var local: Deferred(DenseVectors(Element)) = undefined;
            const held = self.hold(&local, .{ .rows = queries.rows });
            held.result.write = .stamp(arena.handle);
            var arguments: usearch_dense_attend_t = .{
                .index = self.index,
                .executor = self.executor,
                .outcome = &held.outcome,
                .completion = completion,
                .count = queries.rows,
                .queries = .of(queries),
                .head_count = config.head_count,
                .value_lookup = value_lookup.callback,
                .value_lookup_state = value_lookup.state,
                .carry = if (config.carry) |carry| carry.ptr else null,
                .outputs = &held.result.view,
                .arena = arena.handle,
                .outputs_scalar = comptime Scalar.of(Element),
                .expansion_factor = config.expansion_factor,
                .score_scale = config.score_scale,
                .progress = .of(config.progress, config.progress_steps),
                .stats = config.stats,
            };
            usearch_dense_attend(&arguments);
            return self.end(held);
        }

        /// See `DenseIndex.maxsim`. Calls `usearch_dense_maxsim`.
        pub fn maxsim(self: Self, queries: RowsView, rows_per_query: usize, wanted: usize, arena: *Arena, config: DenseMaxsimConfig) Answer(DenseProducts) {
            // C sees only the query count, never the rows left over from dividing by it.
            if (rows_per_query == 0 or queries.rows % rows_per_query != 0)
                return self.refused(.shape_mismatch, "The rows do not divide into queries of `rows_per_query`");
            var local: Deferred(DenseProducts) = undefined;
            const held = self.hold(&local, .{ .queries = queries.rows / rows_per_query });
            held.result.write = .stamp(arena.handle);
            var arguments: usearch_dense_maxsim_t = .{
                .index = self.index,
                .executor = self.executor,
                .outcome = &held.outcome,
                .completion = completion,
                .count = queries.rows / rows_per_query,
                .queries = .of(queries),
                .rows_per_query = rows_per_query,
                .matches = &held.result.view,
                .written = &held.result.written,
                .required = &held.result.required,
                .arena = arena.handle,
                .wanted = wanted,
                .expansion_factor = config.expansion_factor,
                .progress = .of(config.progress, config.progress_steps),
                .stats = config.stats,
            };
            usearch_dense_maxsim(&arguments);
            return self.end(held);
        }
    };
}

/// The placements a reshard passes until its call is final.
const Placements = struct {
    ram: usearch_ram_options_t = .{},
    vram: usearch_vram_options_t = .{},
    nvme: usearch_nvme_options_t = .{},
};

/// A submitted call, held in storage the caller places - on the stack, in a struct, or from an
/// allocator - which owns its outcome and result slot until the call is final, while the arena,
/// the handles and the slices it was handed stay allocated and unchanged until then.
///
/// `status` polls without blocking, `wait` blocks until the call is final and answers its result or
/// error, and `deinit` waits the same way. A submitted `Deferred` stays put until `deinit`, and has
/// one owner, while `status` reads from any thread.
pub fn Deferred(comptime Result: type) type {
    return struct {
        const Self = @This();

        outcome: Outcome = .{},
        result: Result,
        placements: if (Result == DenseRefineReport) Placements else void = if (Result == DenseRefineReport) .{} else {},

        /// `.deferred` until the call is final, then its final status.
        /// Calls `usearch_outcome_status`.
        pub fn status(self: *Self) Status {
            return usearch_outcome_status(&self.outcome);
        }

        /// The static detail of the final status; empty until the call is final, and on success.
        pub fn message(self: *Self) [:0]const u8 {
            return if (self.status() == .deferred) "" else self.outcome.text();
        }

        /// Blocks until the call is final, then answers its result, the same views a blocking call
        /// answers over the arena it viewed. Calls `usearch_outcome_wait`.
        pub fn wait(self: *Self) Error!Result {
            _ = usearch_outcome_wait(&self.outcome);
            try self.outcome.check();
            return self.result;
        }

        /// Blocks until the call is final; the storage is reusable afterwards, and the result's
        /// views stay valid until the arena's next call. Calls `usearch_outcome_wait`.
        pub fn deinit(self: *Self) void {
            _ = usearch_outcome_wait(&self.outcome);
            self.* = undefined;
        }
    };
}

/// Rows a call wrote packed into an arena as `Element` - `get` one per entry under the key,
/// `attend` one per query - `view.stride` bytes apiece, which sub-byte kinds round up to whole
/// bytes, readable until the arena's next call. Mirrors `usearch_rows_view_t`.
pub fn DenseVectors(comptime Element: type) type {
    return struct {
        const Self = @This();

        write: ArenaWrite = .{},
        view: usearch_rows_view_t = .{},
        /// Rows written.
        rows: usize = 0,

        /// One row, `position` below `rows`.
        pub fn row(self: Self, position: usize) Error![]const Element {
            try self.write.open();
            defer self.write.close();
            std.debug.assert(position < self.rows);
            const base: [*]const u8 = @ptrCast(self.view.data.?);
            const first: [*]const Element = @ptrCast(@alignCast(base + position * self.view.stride));
            return first[0 .. self.view.stride / @sizeOf(Element)];
        }

        /// Every row, back to back.
        pub fn bytes(self: Self) Error![]const u8 {
            try self.write.open();
            defer self.write.close();
            return outputSlice(u8, @ptrCast(self.view.data), 0, self.rows * self.view.stride);
        }
    };
}

/// The monolith `DenseIndex.saveBuffer` serialized into an arena, readable until its next call.
pub const Monolith = struct {
    write: ArenaWrite = .{},
    view: BytesView = .{ .data = null, .length = 0 },

    /// The serialized bytes.
    pub fn bytes(self: Monolith) Error![]const u8 {
        try self.write.open();
        defer self.write.close();
        return outputSlice(u8, self.view.data, 0, self.view.length);
    }
};

/// The ragged output of `DenseIndex.multiply` or `DenseIndex.maxsim` in an arena, laid out as
/// Apache Arrow does: query `i` spans pairs `offsets[i]` up to `offsets[i + 1]` of `keys` and
/// `scores`, readable until the arena's next call. Mirrors `usearch_matches_tape_view_t`.
pub const DenseProducts = struct {
    write: ArenaWrite = .{},
    view: usearch_matches_tape_view_t = .{},
    /// Left-hand rows the call multiplied, or queries it scored.
    queries: usize,
    /// Pairs written across every query.
    written: usize = 0,
    /// Pairs scored across every left row, or keys across every query for a `maxsim`, before
    /// `wanted` cut any.
    required: usize = 0,

    /// The pairs of one left row in traversal order, or one query's keys largest first for a
    /// `maxsim`, `position` below `queries`.
    pub fn query(self: DenseProducts, position: usize) Error!QueryMatches {
        try self.write.open();
        defer self.write.close();
        const tape = try self.offsets();
        const begin = tape[position];
        const length = tape[position + 1] - begin;
        return .{ .keys = outputSlice(Key, self.view.keys, begin, length), .scores = outputSlice(Distance, self.view.scores, begin, length) };
    }

    /// The `queries + 1` offsets.
    pub fn offsets(self: DenseProducts) Error![]const usize {
        try self.write.open();
        defer self.write.close();
        return outputSlice(usize, self.view.offsets, 0, self.queries + 1);
    }

    /// Every pair's indexed key, end to end.
    pub fn keys(self: DenseProducts) Error![]const Key {
        try self.write.open();
        defer self.write.close();
        return outputSlice(Key, self.view.keys, 0, self.written);
    }

    /// Every pair's product, positionally paired with `keys`.
    pub fn scores(self: DenseProducts) Error![]const Distance {
        try self.write.open();
        defer self.write.close();
        return outputSlice(Distance, self.view.scores, 0, self.written);
    }
};

/// One batch a `DenseCursor` pull wrote into an arena: the keys, and the keys each links to where
/// asked for, readable until the arena's next call; the vectors come from `DenseIndex.get`.
pub const DenseCursorBatch = struct {
    write: ArenaWrite = .{},
    keys_view: usearch_keys_view_t = .{},
    links_view: usearch_keys_tape_view_t = .{},
    /// Entries the pull wrote; zero once the walk is exhausted.
    written: usize = 0,

    /// The keys.
    pub fn keys(self: DenseCursorBatch) Error![]const Key {
        try self.write.open();
        defer self.write.close();
        return outputSlice(Key, self.keys_view.data, 0, self.written);
    }

    /// The keys one entry links to, `position` below `written`, of a pull that asked for links.
    pub fn links(self: DenseCursorBatch, position: usize) Error![]const Key {
        try self.write.open();
        defer self.write.close();
        const offsets = self.links_view.offsets.?[0 .. self.written + 1];
        return outputSlice(Key, self.links_view.keys, offsets[position], offsets[position + 1] - offsets[position]);
    }
};

/// The draws `DenseIndex.sample` wrote in an arena, one per draw: its key, which of that key's
/// entries it is, the shown or earlier draw it hangs from, and where it is placed, readable until
/// the arena's next call.
pub const DenseDraws = struct {
    write: ArenaWrite = .{},
    keys_view: usearch_keys_view_t = .{},
    entry_ordinals_view: ?[*]const usize = null,
    parents_view: usearch_matches_view_t = .{},
    coordinates_view: usearch_rows_view_t = .{},
    /// Draws written; fewer than `wanted` only once the region is exhausted.
    written: usize = 0,

    /// The keys drawn.
    pub fn keys(self: DenseDraws) Error![]const Key {
        try self.write.open();
        defer self.write.close();
        return outputSlice(Key, self.keys_view.data, 0, self.written);
    }

    /// Which of its key's entries each draw is, in the order `get` writes them.
    pub fn entryOrdinals(self: DenseDraws) Error![]const usize {
        try self.write.open();
        defer self.write.close();
        return outputSlice(usize, self.entry_ordinals_view, 0, self.written);
    }

    /// Each draw's parent and its distance, as a one-column run per draw; a root holds none.
    pub fn parents(self: DenseDraws) Matches {
        return .{ .view = self.parents_view, .queries = self.written };
    }

    /// Where each draw is placed, three `f32` apiece.
    pub fn coordinates(self: DenseDraws) Error![]const [3]f32 {
        try self.write.open();
        defer self.write.close();
        return outputSlice([3]f32, @ptrCast(@alignCast(self.coordinates_view.data)), 0, self.written);
    }
};

/// A sequential walk over the index contents - export, migration, and visualization feed
/// off it. Every pull names the index again, through `DenseIndex.cursorNext`, so free the cursor
/// before the index. Mirrors `usearch_dense_cursor_t`.
pub const DenseCursor = struct {
    /// The opaque `usearch_dense_cursor_t` the ABI hands back.
    handle: ?*anyopaque,

    /// Opens a walk over the current contents; each shard's publication watermark is read as the
    /// walk enters it, so an entry once passed is never revisited. Without `starts` it yields every
    /// entry as stored; with them, the start keys' own entries first in order, then the entries
    /// those link to breadth-first, each once. Calls `usearch_dense_cursor_init`.
    pub fn init(index: DenseIndex, starts: []const Key) Error!DenseCursor {
        var handle: ?*anyopaque = null;
        var outcome: Outcome = .{};
        var arguments: usearch_dense_cursor_init_t = .{
            .index = index.handle,
            .outcome = &outcome,
            .starts_count = starts.len,
            .starts = starts.ptr,
            .cursor = &handle,
        };
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
        // The stream's query holds no count, so only the binding sees a run of several.
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

test "metrics and verdicts keep the values the linked library spells them by" {
    inline for (.{ "inner", "angular", "euclidean", "hamming", "jaccard" }, 1..) |spelling, value| {
        const metric: DenseMetric = @enumFromInt(value);
        try std.testing.expectEqualStrings(spelling, metric.name());
        try std.testing.expectEqual(@as(?DenseMetric, metric), DenseMetric.fromName(spelling));
    }
    try std.testing.expectEqual(@as(?DenseMetric, null), DenseMetric.fromName("cosine"));
    try std.testing.expectEqualStrings("converged", DenseRefineVerdict.converged.name());
    try std.testing.expectEqual(@as(?DenseRefineVerdict, .stopped), DenseRefineVerdict.fromName("stopped"));
    try std.testing.expectEqual(@as(?DenseRefineVerdict, null), DenseRefineVerdict.fromName("settled"));
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
    const judged = [_]Key{ .{ .bytes = .{7} ++ .{0} ** 15 }, .nil };
    var verdicts: [judged.len]types.Admission = undefined;
    filter.callback.?(&judged, &verdicts, judged.len, filter.state);
    try std.testing.expectEqual([judged.len]types.Admission{ .admit, .reject }, verdicts);

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
        void,              usize,         DenseIndexDescription, DenseReachability, DenseRefineReport, Matches,
        DenseVectors(f32), DenseProducts, DenseCursorBatch,      DenseDraws,        Monolith,
    }) |Result| std.testing.refAllDecls(Deferred(Result));
    std.testing.refAllDecls(DenseCursor);
    std.testing.refAllDecls(DenseSearchStream);
    std.testing.refAllDecls(DenseProducts);
    std.testing.refAllDecls(DenseVectors(f32));
    std.testing.refAllDecls(DenseCursorBatch);
    std.testing.refAllDecls(DenseDraws);
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

    // A passed outcome holds the message an error cannot, the binding's own refusals included.
    var outcome: Outcome = .{};
    try std.testing.expectError(Error.ShapeMismatch, index.diagnosed(&outcome).insert(keys[0..2], try .of(f32, &slab, dimensions), .{}));
    try std.testing.expectEqual(types.Status.shape_mismatch, outcome.status);
    try index.on(executor).diagnosed(&outcome).reserve(3);
    try std.testing.expectEqual(Outcome{}, outcome);

    var arena: Arena = try .init(executor);
    defer arena.deinit();
    try std.testing.expectEqual(types.ArenaDescription{ .residency = .ram, .device = null, .vendor = .unknown, .vendor_ordinal = null, .host_access = .readable }, try arena.describe());
    const matches = try index.on(executor).search(try .ofRow(f32, slab[0..dimensions]), 2, &arena, .{ .stats = &stats });
    try std.testing.expectEqual(@as(usize, 2), (try matches.query(0)).keys.len);
    try std.testing.expectEqual(keys[0], (try matches.query(0)).keys[0]);

    // The caller's own thread answers the same search.
    const again = try index.search(try .ofRow(f32, slab[0..dimensions]), 2, &arena, .{});
    try std.testing.expectEqual(keys[0], (try again.query(0)).keys[0]);
    try std.testing.expectError(Error.Contended, matches.query(0));

    // A zero `wanted` lands the counts alone, so a row reads empty rather than through absent keys.
    const none = try index.search(try .ofRow(f32, slab[0..dimensions]), 0, &arena, .{});
    try std.testing.expectEqual(@as(usize, 0), (try none.query(0)).keys.len);

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
    try std.testing.expectEqualSlices(Key, keys[2..3], (try admitted.query(0)).keys);

    const exported = try index.on(executor).get(f32, keys[1], &arena);
    try std.testing.expectEqualSlices(f32, &.{ 0, 1, 0 }, try exported.row(0));
    try std.testing.expectEqual(@as(usize, 1), exported.rows);
    try std.testing.expectEqual(@as(usize, dimensions), exported.view.columns);
    try std.testing.expectEqual(@as(usize, 1), try index.count(keys[1]));
    try std.testing.expectEqual(true, try index.contains(keys[1]));
    try std.testing.expectEqual(false, try index.on(executor).contains(.fromInteger(99)));
    const absent = try index.get(f32, .fromInteger(99), &arena);
    try std.testing.expectEqual(@as(usize, 0), absent.rows);
    try std.testing.expectEqual(@as(usize, 0), (try absent.bytes()).len);

    // A failed call leaves the arena the caller's, ready for the next one.
    try std.testing.expectError(Error.ConfigMismatch, index.multiply(try .ofRow(f32, slab[0..dimensions]), &arena, .{}));
    try std.testing.expectEqualSlices(f32, &.{ 0, 0, 1 }, try (try index.get(f32, keys[2], &arena)).row(0));

    // The width travels with the rows, and the engine refuses one the geometry disagrees with.
    const narrow = [_]f32{ 1, 0 };
    try std.testing.expectError(
        Error.ShapeMismatch,
        index.on(executor).diagnosed(&outcome).search(try .ofRow(f32, &narrow), 2, &arena, .{}),
    );
    try std.testing.expectEqual(types.Status.shape_mismatch, outcome.status);

    // A constructor's config passes one the same way.
    try std.testing.expectError(Error.Io, DenseIndex.loadRam("absent.safegraphs", .{ .diagnostics = &outcome }));
    try std.testing.expectEqual(types.Status.io, outcome.status);

    // A stream yields one query's matches nearest first, and refuses a run holding more; a
    // residency that streams no matches yet refuses the stream itself.
    try std.testing.expectError(Error.ShapeMismatch, DenseSearchStream.init(index, try .of(f32, &slab, dimensions), .{}));
    if (DenseSearchStream.init(index, try .ofRow(f32, slab[0..dimensions]), .{})) |opened| {
        var stream = opened;
        defer stream.deinit();
        const pulled = try index.on(executor).searchStreamNext(stream, 2, &arena);
        try std.testing.expectEqualSlices(Key, keys[0..2], (try pulled.query(0)).keys);
        try std.testing.expectEqual(@as(usize, 1), (try (try index.searchStreamNext(stream, 2, &arena)).query(0)).keys.len);
        // A drained walk answers an empty row, as a pull of zero does.
        try std.testing.expectEqual(@as(usize, 0), (try (try index.searchStreamNext(stream, 2, &arena)).query(0)).keys.len);
    } else |failure| try std.testing.expectEqual(Error.Unsupported, failure);

    // A cursor walks the keys, `wanted` at a time, and exports their links only when asked.
    var walk = try DenseCursor.init(index, &.{});
    defer walk.deinit();
    const batch = try index.on(executor).cursorNext(walk, 2, false, &arena);
    try std.testing.expectEqualSlices(Key, keys[0..2], try batch.keys());
    const rest = try index.cursorNext(walk, 2, false, &arena);
    try std.testing.expectEqualSlices(Key, keys[2..3], try rest.keys());
    const drained = try index.cursorNext(walk, 2, false, &arena);
    try std.testing.expectEqual(@as(usize, 0), (try drained.keys()).len);

    // A monolith round-trips through memory, and its metadata reads without loading it.
    const monolith = try (try index.on(executor).saveBuffer(&arena, .{})).bytes();
    try std.testing.expectEqual(frozen, try DenseIndex.metadataReadBuffer(monolith, .{ .executor = executor }));
    var copy = try DenseIndex.loadRamBuffer(monolith, .{ .executor = executor });
    defer copy.deinit();
    try std.testing.expectEqual(@as(usize, 3), (try copy.describe()).size);

    // A recut in place keeps every entry, and the one shard left saves as a monolith.
    _ = try index.on(executor).reshard(&.{index}, &.{index}, .{ .memory_domains = &.{}, .shard_capacity = 8 });
    try std.testing.expectEqual(@as(usize, 3), (try index.describe()).size);

    // An NVMe view resolves keys only through the key table it was opened with.
    const checkpoint = ".zig-cache/usearch-zig-nvme.safegraphs";
    try index.on(executor).save(checkpoint, .{});
    defer std.Io.Dir.cwd().deleteFile(std.testing.io, checkpoint) catch {};
    var drives = try Executor.forNvme(.{ .threads = 1 });
    defer drives.deinit();
    var keyed = try DenseIndex.viewNvme(checkpoint, .{ .executor = drives, .key_table_domains = &.{} });
    defer keyed.deinit();
    try std.testing.expectEqual(@as(usize, 1), try keyed.on(drives).count(keys[0]));
    var keyless = try DenseIndex.viewNvme(checkpoint, .{ .executor = drives });
    defer keyless.deinit();
    try std.testing.expectError(Error.Unsupported, keyless.on(drives).count(keys[0]));

    // A lone monolith has no manifest a loaded handle could replace.
    try std.testing.expectError(Error.Unsupported, DenseIndex.loadNvme(checkpoint, .{ .executor = drives }));

    // A cancel stops only the closing round, so every source has moved and ends empty and live;
    // the plan it leaves closes with the whole gather's report.
    var shards = [_]DenseIndex{ try DenseIndex.loadRamBuffer(monolith, .{}), try DenseIndex.loadRamBuffer(monolith, .{}) };
    defer for (&shards) |*shard| shard.deinit();
    const Stop = struct {
        fn afterFirst(_: void, completed: usize, _: usize) types.Continuation {
            return if (completed == 0) .@"continue" else .cancel;
        }
    };
    var gathering: DenseReshardPlan = .{};
    var stopped: types.PhaseProgress = .{};
    try std.testing.expectError(Error.Cancelled, copy.on(executor).reshard(&.{ copy, shards[0], shards[1] }, &.{copy}, .{
        .progress = .of({}, Stop.afterFirst),
        .plan = &gathering,
        .phase = &stopped,
    }));
    try std.testing.expect(gathering.handle != null);
    try std.testing.expect(stopped.phase != .none and stopped.phase != .done);
    for (shards) |shard| try std.testing.expectEqual(@as(usize, 0), (try shard.describe()).size);
    try std.testing.expectEqual(@as(usize, 3), try copy.count(keys[0]));
    _ = try copy.on(executor).reshardPlanFinish(&gathering);
    try std.testing.expectEqual(DenseReshardPlan{}, gathering);

    const reachability = try index.on(executor).health();
    try std.testing.expectEqual([2]usize{ 3, 3 }, [2]usize{ reachability.audited, reachability.reachable });

    // A refine reports what it did, and closes on the audit a fresh `health` would take.
    const report = try copy.on(executor).refine(.{});
    try std.testing.expect(report.changed <= report.reselected);
    try std.testing.expectEqual(@as(usize, 9), report.reachability.audited);

    // One source into several targets hands them every entry, leaving the source empty but live.
    var receivers = [_]DenseIndex{ try DenseIndex.loadRamBuffer(monolith, .{}), try DenseIndex.loadRamBuffer(monolith, .{}) };
    defer for (&receivers) |*receiver| receiver.deinit();
    for (receivers) |receiver| try receiver.clear();
    _ = try copy.on(executor).reshard(&.{copy}, &receivers, .{ .seed = 7 });
    try std.testing.expectEqual(@as(usize, 0), (try copy.describe()).size);
    try std.testing.expectEqual(@as(usize, 9), (try receivers[0].describe()).size + (try receivers[1].describe()).size);

    // A sample draws held keys, each placed in three dimensions, and refuses shown keys without places.
    const draws = try index.on(executor).sample(2, &arena, .{ .seed = 42 });
    try std.testing.expect(draws.written <= 2);
    for (try draws.keys()) |drawn| try std.testing.expect(drawn.asInteger().? >= 1 and drawn.asInteger().? <= 3);
    try std.testing.expectEqual(draws.written, (try draws.coordinates()).len);
    try std.testing.expectError(Error.ShapeMismatch, index.sample(2, &arena, .{ .shown = keys[0..1] }));
}

test "a scan scores every entry, so every query finds itself first" {
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
    for (keys, 0..) |key, query| try std.testing.expectEqual(key, (try matches.query(query)).keys[0]);

    // Sub-byte rows report their own stride, so no width arithmetic happens here.
    const nibbles = try index.get(Scalar.U4x2, keys[3], &arena);
    try std.testing.expectEqual(@as(usize, 2), nibbles.view.stride);
    try std.testing.expectEqual(@as(usize, dimensions), nibbles.view.columns);
    try std.testing.expectEqual(@as(usize, 2), (try nibbles.row(0)).len);
}

test "a deferred call is the same verb, final once waited on" {
    const dimensions = 2;
    var executor = try Executor.forRam(.{});
    defer executor.deinit();
    var index = try DenseIndex.initRam(.{ .dimensions = dimensions, .metric = .euclidean }, &.{});
    defer index.deinit();

    // One call holds the executor at a time, so each is waited on before the next is submitted.
    var reserving: Deferred(void) = undefined;
    try index.deferred(executor, &reserving).reserve(4);
    defer reserving.deinit();
    try reserving.wait();
    const slab = [_]f32{ 1, 0, 0, 1 };
    const keys = [_]Key{ .fromInteger(1), .fromInteger(2) };
    var inserting: Deferred(void) = undefined;
    try index.deferred(executor, &inserting).insert(&keys, try .of(f32, &slab, dimensions), .{});
    defer inserting.deinit();
    try inserting.wait();
    try std.testing.expectEqual(types.Status.success, inserting.status());
    try std.testing.expectEqualStrings("", inserting.message());

    // Waiting again answers the same result rather than a second call's.
    var arena: Arena = try .init(executor);
    defer arena.deinit();
    var getting: Deferred(DenseVectors(f32)) = undefined;
    try index.deferred(executor, &getting).get(f32, keys[1], &arena);
    defer getting.deinit();
    const vectors = try getting.wait();
    try std.testing.expectEqualSlices(f32, &.{ 0, 1 }, try vectors.row(0));
    try std.testing.expectEqual(vectors, try getting.wait());

    // Storage is the caller's, so a batch of calls is an array and a reused slot is resubmitted;
    // an executor runs one call at a time, so each call running at once takes its own.
    var counting: [2]Deferred(usize) = undefined;
    var executors: [2]Executor = .{ try Executor.forRam(.{}), try Executor.forRam(.{}) };
    defer for (&executors) |*own| own.deinit();
    for (&counting, &executors, keys) |*slot, own, key| try index.deferred(own, slot).count(key);
    for (&counting) |*slot| try std.testing.expectEqual(@as(usize, 1), try slot.wait());
    for (&counting) |*slot| slot.deinit();
    try index.deferred(executor, &counting[0]).count(.fromInteger(99));
    defer counting[0].deinit();
    try std.testing.expectEqual(@as(usize, 0), try counting[0].wait());

    // A failure the engine reports arrives through the same wait, as often as it is asked.
    const narrow = [_]f32{1};
    var refused: Deferred(Matches) = undefined;
    try index.deferred(executor, &refused).search(try .ofRow(f32, &narrow), 1, &arena, .{});
    defer refused.deinit();
    try std.testing.expectError(Error.ShapeMismatch, refused.wait());
    try std.testing.expectError(Error.ShapeMismatch, refused.wait());

    // Its message is the one the same call reports blocking.
    var outcome: Outcome = .{};
    try std.testing.expectError(Error.ShapeMismatch, index.diagnosed(&outcome).search(try .ofRow(f32, &narrow), 1, &arena, .{}));
    try std.testing.expectEqualStrings(outcome.text(), refused.message());

    // A shape only the binding sees is refused before anything is submitted.
    var mismatched: Deferred(void) = undefined;
    try std.testing.expectError(Error.ShapeMismatch, index.deferred(executor, &mismatched).insert(keys[0..1], try .of(f32, &slab, dimensions), .{}));
}

test "a running call holds its arena, so another call naming it is contended" {
    if (build_options.stub) return error.SkipZigTest;
    const dimensions = 2;
    var executor = try Executor.forRam(.{});
    defer executor.deinit();
    var index = try DenseIndex.initRam(.{ .dimensions = dimensions, .metric = .euclidean }, &.{});
    defer index.deinit();
    try index.reserve(4);
    const slab = [_]f32{ 1, 0, 0, 1 };
    const keys = [_]Key{ .fromInteger(1), .fromInteger(2) };
    try index.insert(&keys, try .of(f32, &slab, dimensions), .{});
    var arena: Arena = try .init(executor);
    defer arena.deinit();

    // The arena is viewed as a blocking call views it, so a second call while one holds it is
    // refused as contended and leaves the holder's views alone.
    // The search waits at its last report until released, so the test meets it holding the arena.
    const Gate = struct {
        entered: std.atomic.Value(bool) = .init(false),
        released: std.atomic.Value(bool) = .init(false),

        fn holdUntilReleased(gate: *@This(), completed: usize, total: usize) types.Continuation {
            if (completed != total) return .@"continue";
            gate.entered.store(true, .release);
            _ = awaitSoon(&gate.released);
            return .@"continue";
        }

        fn awaitSoon(flag: *const std.atomic.Value(bool)) bool {
            const deadline = std.Io.Clock.awake.now(std.testing.io).addDuration(.fromSeconds(3));
            while (!flag.load(.acquire) and std.Io.Clock.awake.now(std.testing.io).nanoseconds < deadline.nanoseconds) {}
            return flag.load(.acquire);
        }
    };
    var gate: Gate = .{};
    var holding: Deferred(Matches) = undefined;
    try index.deferred(executor, &holding).search(try .ofRow(f32, slab[0..dimensions]), 1, &arena, .{
        .progress = .of(&gate, Gate.holdUntilReleased),
        .progress_steps = 1,
    });
    defer holding.deinit();
    try std.testing.expect(Gate.awaitSoon(&gate.entered));
    try std.testing.expectError(Error.Contended, index.get(f32, keys[1], &arena));
    var contending: Deferred(DenseVectors(f32)) = undefined;
    try index.deferred(executor, &contending).get(f32, keys[1], &arena);
    defer contending.deinit();
    try std.testing.expectEqual(types.Status.contended, contending.status());
    try std.testing.expectError(Error.Contended, contending.wait());
    gate.released.store(true, .release);
    const held = try holding.wait();
    try std.testing.expectEqualSlices(Key, keys[0..1], (try held.query(0)).keys);
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
    try std.testing.expectEqual(offsets[2] - offsets[1], (try products.query(1)).keys.len);
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
        const outputs = try index.attend(f32, try .of(f32, &left, dimensions), .of(Scalar.BF16, &table, Table.resolve), &arena, .{
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
            const attended = try outputs.row(query);
            for (mixed, attended) |column, attended_column|
                try std.testing.expectApproxEqAbs(column / partition, attended_column, 1e-3 * @max(@abs(column / partition), 1));
            try std.testing.expectApproxEqAbs(maximum, carry[query].maximum_score, 1e-3 * @max(@abs(maximum), 1));
            try std.testing.expectApproxEqAbs(partition, carry[query].partition, 1e-3 * @max(@abs(partition), 1));
        }
    }
    table.resolvable = keys.len - 1;
    try std.testing.expectError(Error.ShapeMismatch, index.attend(f32, try .of(f32, &left, dimensions), .of(Scalar.BF16, &table, Table.resolve), &arena, .{}));
    // A lookup over another kind than the index stores answers no row, which the call refuses alike.
    table.resolvable = keys.len;
    const Halves = struct {
        fn resolve(_: void, _: Key) ?[]const f16 {
            return &.{ 1, 2 };
        }
    };
    try std.testing.expectError(Error.ShapeMismatch, index.attend(f32, try .of(f32, &left, dimensions), .of(f16, {}, Halves.resolve), &arena, .{}));

    const documents = try index.maxsim(try .of(f32, &left, dimensions), 2, 4, &arena, .{});
    try std.testing.expect((try documents.query(0)).keys.len != 0);
    const every = try index.maxsim(try .of(f32, &left, dimensions), 2, 0, &second, .{});
    try std.testing.expect((try every.query(0)).keys.len >= (try documents.query(0)).keys.len);
    try std.testing.expectError(Error.ShapeMismatch, index.maxsim(try .of(f32, &left, dimensions), 3, 4, &arena, .{}));
}

test "a reporting batch counts what it published, and a stopped sweep answers the cancel" {
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

    // Every entry is claimed before the first report, so an insertion's handler cannot stop it.
    var ignored: Watch = .{};
    const more = [_]Key{ .fromInteger(4), .fromInteger(5), .fromInteger(6) };
    try index.on(executor).insert(&more, try .of(f32, &slab, dimensions), .{ .progress = .of(&ignored, Watch.stop) });
    try std.testing.expectEqual(@as(usize, 6), (try index.on(executor).describe()).size);

    // A stopped round answers the cancel; with a plan it stays live, the next call ends its round,
    // and finishing reports the whole plan.
    var swept: Watch = .{};
    try std.testing.expectError(Error.Cancelled, index.on(executor).refine(.{ .progress = .of(&swept, Watch.stop) }));
    try std.testing.expect(swept.ordered);
    try std.testing.expect(swept.total != 0);
    var plan: DenseRefinePlan = .{};
    var phase: types.PhaseProgress = .{};
    try std.testing.expectError(Error.Cancelled, index.on(executor).refine(.{ .progress = .of(&swept, Watch.stop), .plan = &plan, .phase = &phase }));
    try std.testing.expect(plan.handle != null and phase.phase != .done);
    _ = try index.on(executor).refine(.{ .plan = &plan, .phase = &phase });
    try std.testing.expectEqual(types.Phase.done, phase.phase);
    const whole = try index.on(executor).refinePlanFinish(&plan);
    try std.testing.expectEqual(DenseRefinePlan{}, plan);
    try std.testing.expect(whole.changed <= whole.reselected);
}
