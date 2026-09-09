//! The dense modality - Vector Search over equidimensional embeddings, the Zig twin of
//! `include/usearch/dense_index.h`.
//!
//! `DenseIndex` is the handle; `MatrixView` and `MatrixSpan` are the batch currency every verb
//! reads and writes, one vector being a one-row matrix. `Cursor` walks a consistent snapshot of
//! the contents, and `Finder` streams nearest neighbors of one query without a "k" chosen upfront.
//!
//! Every fanned-out verb takes an `Executor`, whose kind must match the handle's residency; there
//! is no null to pass, since the engine reads a missing one as a residency mismatch.
//!
//! Every long verb has a reporting sibling - `addReporting` and friends - taking a handler called
//! with the units finished so far, whose answer is what stops the call.

const std = @import("std");

const types = @import("types.zig");
const BytesSpan = types.BytesSpan;
const BytesView = types.BytesView;
const Distance = types.Distance;
const Error = types.Error;
const Executor = types.Executor;
const Key = types.Key;
const KeysSpan = types.KeysSpan;
const KeysView = types.KeysView;
const MatchesSpan = types.MatchesSpan;
const MatrixSpan = types.MatrixSpan;
const MatrixView = types.MatrixView;
const Outcome = types.Outcome;
const PlacementOptions = types.PlacementOptions;
const Placement = types.Placement;
const Residency = types.Residency;
const Scalar = types.Scalar;
const refuse = types.refuse;

/// Whether a filtered search admits one candidate key or passes over it.
pub const Admission = enum(c_int) {
    /// Skip the key; the walk still routes through it.
    reject = 0,
    /// Report the key among the matches.
    admit = 1,
};

/// The `usearch_predicate_t` a filtered search calls once per scored candidate.
const Predicate = *const fn (Key, ?*anyopaque) callconv(.c) Admission;

/// Whether the operation a handler is watching carries on or unwinds.
pub const Continuation = enum(c_int) {
    /// Carry on to the next unit of work.
    @"continue" = 0,
    /// Stop; what the finished increments leave behind is each verb's own contract.
    cancel = 1,
};

/// The `usearch_progress_callback_t` a reporting verb calls at every increment boundary.
const Progress = *const fn (usize, usize, ?*anyopaque) callconv(.c) Continuation;

/// The `usearch_dense_metric_callback_t` a custom metric supplies.
pub const MetricCallback = *const fn (?*const anyopaque, ?*const anyopaque, usize, ?*anyopaque) callconv(.c) Distance;

extern fn usearch_dense_init(options: *const Options, outcome: *Outcome) ?*anyopaque;
extern fn usearch_dense_free(index: *anyopaque) void;
extern fn usearch_dense_reserve(index: *anyopaque, capacity: usize, outcome: *Outcome) void;
extern fn usearch_dense_clear(index: *anyopaque, outcome: *Outcome) void;

extern fn usearch_dense_size(index: *anyopaque, outcome: *Outcome) usize;
extern fn usearch_dense_capacity(index: *anyopaque, outcome: *Outcome) usize;
extern fn usearch_dense_dimensions(index: *anyopaque, outcome: *Outcome) usize;
extern fn usearch_dense_connectivity(index: *anyopaque, outcome: *Outcome) usize;
extern fn usearch_dense_memory_usage(index: *anyopaque, outcome: *Outcome) usize;
extern fn usearch_dense_capabilities(index: *anyopaque, outcome: *Outcome) ?[*:0]const u8;
extern fn usearch_dense_storage_requirements(index: *anyopaque, outcome: *Outcome) usize;

extern fn usearch_dense_add(
    index: *anyopaque,
    executor: *anyopaque,
    keys: KeysView,
    vectors: *const MatrixView,
    options: *const InsertionOptions,
    progress: ?Progress,
    progress_state: ?*anyopaque,
    stats: ?*TraversalStats,
    outcome: *Outcome,
) usize;

extern fn usearch_dense_refine(
    index: *anyopaque,
    executor: *anyopaque,
    options: *const InsertionOptions,
    progress: ?Progress,
    progress_state: ?*anyopaque,
    stats: ?*TraversalStats,
    outcome: *Outcome,
) usize;

extern fn usearch_dense_search(
    index: *anyopaque,
    executor: *anyopaque,
    queries: *const MatrixView,
    matches: *const MatchesSpan,
    options: *const SearchOptions,
    progress: ?Progress,
    progress_state: ?*anyopaque,
    stats: ?*TraversalStats,
    outcome: *Outcome,
) void;

extern fn usearch_dense_search_if(
    index: *anyopaque,
    executor: *anyopaque,
    queries: *const MatrixView,
    matches: *const MatchesSpan,
    options: *const SearchOptions,
    predicate: ?Predicate,
    predicate_state: ?*anyopaque,
    progress: ?Progress,
    progress_state: ?*anyopaque,
    stats: ?*TraversalStats,
    outcome: *Outcome,
) void;

extern fn usearch_dense_multiply(
    index: *anyopaque,
    executor: *anyopaque,
    left: *const MatrixView,
    products: *const MatchesSpan,
    options: *const SearchOptions,
    progress: ?Progress,
    progress_state: ?*anyopaque,
    stats: ?*TraversalStats,
    outcome: *Outcome,
) void;

extern fn usearch_dense_maxsim(
    index: *anyopaque,
    executor: *anyopaque,
    queries: *const MatrixView,
    rows_per_query: usize,
    matches: *const MatchesSpan,
    options: *const SearchOptions,
    progress: ?Progress,
    progress_state: ?*anyopaque,
    stats: ?*TraversalStats,
    outcome: *Outcome,
) void;

extern fn usearch_dense_attend(
    index: *anyopaque,
    executor: *anyopaque,
    queries: *const MatrixView,
    values: *const MatrixView,
    outputs: *const MatrixSpan,
    carry: ?[*]SoftmaxState,
    options: *const AttentionOptions,
    progress: ?Progress,
    progress_state: ?*anyopaque,
    stats: ?*TraversalStats,
    outcome: *Outcome,
) void;

extern fn usearch_dense_count(index: *anyopaque, key: Key, outcome: *Outcome) usize;
extern fn usearch_dense_get(index: *anyopaque, key: Key, vectors: ?*const MatrixSpan, outcome: *Outcome) usize;

extern fn usearch_dense_cursor(index: *anyopaque, outcome: *Outcome) ?*anyopaque;
extern fn usearch_dense_cursor_next(
    cursor: *anyopaque,
    keys: KeysSpan,
    vectors: ?*const MatrixSpan,
    outcome: *Outcome,
) usize;
extern fn usearch_dense_cursor_free(cursor: *anyopaque) void;

extern fn usearch_dense_find(
    index: *anyopaque,
    executor: *anyopaque,
    query: *const MatrixView,
    outcome: *Outcome,
) ?*anyopaque;
extern fn usearch_dense_find_next(
    finder: *anyopaque,
    keys: KeysSpan,
    distances: [*]Distance,
    outcome: *Outcome,
) usize;
extern fn usearch_dense_find_free(finder: *anyopaque) void;

extern fn usearch_dense_save(
    index: *anyopaque,
    executor: *anyopaque,
    path: BytesView,
    progress: ?Progress,
    progress_state: ?*anyopaque,
    outcome: *Outcome,
) void;
extern fn usearch_dense_load(
    index: *anyopaque,
    executor: *anyopaque,
    path: BytesView,
    progress: ?Progress,
    progress_state: ?*anyopaque,
    outcome: *Outcome,
) void;
extern fn usearch_dense_metadata(path: BytesView, found: *Metadata, outcome: *Outcome) void;

extern fn usearch_dense_save_buffer(
    index: *anyopaque,
    executor: *anyopaque,
    buffer: BytesSpan,
    progress: ?Progress,
    progress_state: ?*anyopaque,
    outcome: *Outcome,
) void;
extern fn usearch_dense_load_buffer(
    index: *anyopaque,
    executor: *anyopaque,
    buffer: BytesView,
    progress: ?Progress,
    progress_state: ?*anyopaque,
    outcome: *Outcome,
) void;
extern fn usearch_dense_metadata_buffer(buffer: BytesView, found: *Metadata, outcome: *Outcome) void;

/// The dense metric kinds shipped out of the box, all derived from one inner-product kernel with
/// pre-computed magnitudes. The bit-level pair reads its rows as `Scalar.u1x8` and refuses any
/// other quantization; the first three refuse that one.
///
/// Non-exhaustive, so a kind a newer engine reports arrives intact rather than coerced.
pub const Metric = enum(c_int) {
    /// No built-in kind - `Options.scoredBy` supplies the scoring and forfeits the SIMD
    /// kernel chains.
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
};

/// Construction and loading options of a dense index. Mirrors `usearch_dense_options_t`.
///
/// Stored rows always round up to the SIMD step quantum, so rows stay cache-line aligned and
/// every kernel runs tail-free; the arena is slightly larger for it, and that is not a knob.
pub const Options = extern struct {
    /// The metric kind used for distance calculations between vectors.
    metric: Metric = .angular,
    /// The optional custom metric; must be set when `metric` is `.unknown`, which `scoredBy` does.
    metric_callback: ?MetricCallback = null,
    /// The optional opaque state forwarded into every `metric_callback` call.
    metric_callback_state: ?*anyopaque = null,
    /// The scalar type vectors are stored and quantized into.
    quantization: Scalar = .bf16,
    /// Dimensions of the indexed vectors; zero defers the geometry to a later `load`.
    dimensions: usize = 0,
    /// The optional connections-per-node bound; zero derives.
    connectivity: usize = 0,
    /// Upper bound on entries sharing one key; zero derives to one - unique keys.
    values_per_key: usize = 0,
    /// Selects the active member of `placement`, and the meaning of "load".
    residency: Residency = .ram,
    /// Where the payload lives, read according to `residency`.
    placement: PlacementOptions = .{ .ram = .{} },

    /// Points the options at a placement, setting the tag and the union together so the two
    /// cannot disagree. The placement's slices are borrowed for the duration of the `init` call.
    pub fn placed(self: Options, placement: Placement) Options {
        var options = self;
        options.residency = std.meta.activeTag(placement);
        options.placement = placement.asOptions();
        return options;
    }

    /// Points the options at a caller-supplied metric over rows stored as `Element`, setting the
    /// kind, the quantization, and the callback together so the four cannot disagree.
    ///
    /// The trampoline is generated per call site, so `score` reads two typed slices rather than
    /// two `anyopaque` pointers and nothing is boxed. Pass `{}` for a metric needing no context
    /// and a single-item pointer otherwise; the context is borrowed for the life of every index
    /// built from these options, and is read concurrently from the executor's threads. A custom
    /// metric forfeits the SIMD kernel chains and the magnitude cache both.
    pub fn scoredBy(
        self: Options,
        comptime Element: type,
        context: anytype,
        comptime score: fn (@TypeOf(context), []const Element, []const Element) Distance,
    ) Options {
        const Context = @TypeOf(context);
        checkContext(Context);
        const trampoline = struct {
            fn evaluate(
                first: ?*const anyopaque,
                second: ?*const anyopaque,
                dimensions: usize,
                state: ?*anyopaque,
            ) callconv(.c) Distance {
                const left: [*]const Element = @ptrCast(@alignCast(first.?));
                const right: [*]const Element = @ptrCast(@alignCast(second.?));
                if (Context == void) return score({}, left[0..dimensions], right[0..dimensions]);
                return score(@ptrCast(@alignCast(state.?)), left[0..dimensions], right[0..dimensions]);
            }
        }.evaluate;

        var options = self;
        options.metric = .unknown;
        options.quantization = Scalar.of(Element);
        options.metric_callback = trampoline;
        options.metric_callback_state = contextPointer(Context, context);
        return options;
    }
};

/// The geometry and metric a checkpoint froze - everything a reader learns without opening it, and
/// nothing a reader must choose for itself. Mirrors `usearch_dense_metadata_t`.
///
/// `values_per_key` is absent because it bounds a live handle's mutable state rather than anything
/// the serialized form carries, and the placement is absent because a checkpoint records where it
/// was written and never where a reader must keep it. Both stay the caller's to set in `Options`.
pub const Metadata = extern struct {
    /// The metric kind the checkpoint scores with.
    metric: Metric = .unknown,
    /// The scalar type its rows are stored and quantized into.
    quantization: Scalar = .unknown,
    /// Scalars per stored vector.
    dimensions: usize = 0,
    /// Connections per node the graph was built with.
    connectivity: usize = 0,
};

/// Optional per-call settings of `DenseIndex.add` and `DenseIndex.refine`. A zeroed value derives
/// every setting, which is what almost every caller wants.
pub const InsertionOptions = extern struct {
    /// The optional candidate-pool breadth the build walk explores; zero derives.
    expansion_factor: usize = 0,
};

/// Optional per-call settings of the read-only walks - both searches, `multiply` and `maxsim`. A
/// zeroed value derives every setting, which is what almost every caller wants.
pub const SearchOptions = extern struct {
    /// The optional candidate-pool breadth the walk explores; zero derives.
    expansion_factor: usize = 0,
};

/// The online-softmax carry of one query row - Flash Attention's running maximum and partition.
///
/// The two are one concept: a maximum without its partition is a number nothing can rescale, so
/// they share an element rather than travelling as two arrays a caller could supply half of.
/// Carrying them is what folds one attention call into another over the same queries.
pub const SoftmaxState = extern struct {
    /// The largest logit seen so far - the recurrence's `m`.
    maximum_score: f32 = 0,
    /// The sum of exponentials taken against that maximum - the recurrence's `l`.
    partition: f32 = 0,
};

/// Optional per-call settings of `DenseIndex.attend`. A zeroed value derives every setting.
pub const AttentionOptions = extern struct {
    /// The optional candidate-pool breadth; zero derives.
    expansion_factor: usize = 0,
    /// The optional logit scale, conventionally the inverse square root of the head
    /// dimensionality; zero derives it from the index's frozen geometry.
    score_scale: f32 = 0,
    /// The optional residual-mass threshold the walk stops at; zero derives.
    mass_tolerance: f32 = 0,
};

/// Traversal telemetry, fully overwritten by the call it is passed to. Passing null skips it.
pub const TraversalStats = extern struct {
    /// Graph members visited while resolving the batch.
    visited_members: usize = 0,
    /// Distances computed while resolving the batch, diversity re-selections included.
    computed_distances: usize = 0,
    /// Starvation restarts - a graph-connectivity health signal rather than a cost counter.
    restarts: usize = 0,
};

// Every descriptor this module declares by hand, field by field, against the layout the C
// compiler gives its twin - the check that makes writing the interface out as safe as translating
// it. The offsets are written in pointer widths, so a 32-bit target holds them too.
comptime {
    const pointer = @sizeOf(usize);
    const tag = @sizeOf(c_int);

    std.debug.assert(@sizeOf(Admission) == tag);
    std.debug.assert(@sizeOf(Metric) == tag);

    // The nine fields the whole tagged-union contract rests on: `residency` selects the arm of
    // `placement`, so a drift in either one silently reads the wrong member.
    std.debug.assert(@sizeOf(Options) == 10 * pointer);
    std.debug.assert(@offsetOf(Options, "metric") == 0);
    std.debug.assert(@offsetOf(Options, "metric_callback") == pointer);
    std.debug.assert(@offsetOf(Options, "metric_callback_state") == 2 * pointer);
    std.debug.assert(@offsetOf(Options, "quantization") == 3 * pointer);
    std.debug.assert(@offsetOf(Options, "dimensions") == 4 * pointer);
    std.debug.assert(@offsetOf(Options, "connectivity") == 5 * pointer);
    std.debug.assert(@offsetOf(Options, "values_per_key") == 6 * pointer);
    std.debug.assert(@offsetOf(Options, "residency") == 7 * pointer);
    std.debug.assert(@offsetOf(Options, "placement") == 8 * pointer);

    // The four fields a checkpoint answers with, the two tags packing ahead of the two widths.
    std.debug.assert(@sizeOf(Metadata) == 2 * tag + 2 * pointer);
    std.debug.assert(@offsetOf(Metadata, "metric") == 0);
    std.debug.assert(@offsetOf(Metadata, "quantization") == tag);
    std.debug.assert(@offsetOf(Metadata, "dimensions") == 2 * tag);
    std.debug.assert(@offsetOf(Metadata, "connectivity") == 2 * tag + pointer);

    std.debug.assert(@sizeOf(InsertionOptions) == pointer);
    std.debug.assert(@offsetOf(InsertionOptions, "expansion_factor") == 0);

    std.debug.assert(@sizeOf(SearchOptions) == pointer);
    std.debug.assert(@offsetOf(SearchOptions, "expansion_factor") == 0);

    std.debug.assert(@sizeOf(AttentionOptions) == pointer + 8);
    std.debug.assert(@offsetOf(AttentionOptions, "expansion_factor") == 0);
    std.debug.assert(@offsetOf(AttentionOptions, "score_scale") == pointer);
    std.debug.assert(@offsetOf(AttentionOptions, "mass_tolerance") == pointer + 4);

    std.debug.assert(@sizeOf(SoftmaxState) == 8);
    std.debug.assert(@offsetOf(SoftmaxState, "maximum_score") == 0);
    std.debug.assert(@offsetOf(SoftmaxState, "partition") == 4);

    std.debug.assert(@sizeOf(TraversalStats) == 3 * pointer);
    std.debug.assert(@offsetOf(TraversalStats, "visited_members") == 0);
    std.debug.assert(@offsetOf(TraversalStats, "computed_distances") == pointer);
    std.debug.assert(@offsetOf(TraversalStats, "restarts") == 2 * pointer);
}

/// Rejects a context that cannot cross the FFI boundary. The predicate travels as one
/// `*anyopaque`, so a slice or a many-item pointer has no single address to pass.
inline fn checkContext(comptime Context: type) void {
    if (Context == void) return;
    const info = @typeInfo(Context);
    if (info != .pointer or info.pointer.size != .one)
        @compileError(std.fmt.comptimePrint(
            "Context must be `void` or a single-item pointer like `&my_context`, got `{s}`",
            .{@typeName(Context)},
        ));
}

/// The pointer a predicate travels with. A `void` context travels as null and is never read.
inline fn contextPointer(comptime Context: type, context: Context) ?*anyopaque {
    return if (Context == void) null else @ptrCast(@constCast(context));
}

/// The trampoline a reporting verb hands the ABI, generated per call site so `reports` reads a
/// typed context rather than an `anyopaque`.
inline fn progressTrampoline(
    comptime Context: type,
    comptime reports: fn (Context, usize, usize) Continuation,
) Progress {
    return struct {
        fn advance(completed: usize, total: usize, state: ?*anyopaque) callconv(.c) Continuation {
            if (Context == void) return reports({}, completed, total);
            return reports(@ptrCast(@alignCast(state.?)), completed, total);
        }
    }.advance;
}

/// Vector Search Index over equidimensional dense representations.
///
/// A handle is shareable across threads: searches never write to the graph, and insertions
/// synchronize through per-node locks and publish in order. It is the `Executor` that serializes,
/// never the index.
///
/// ```zig
/// const usearch = @import("usearch");
///
/// const index = try usearch.DenseIndex.init(&.{ .dimensions = 4, .metric = .angular, .quantization = .i8 });
/// defer index.deinit();
/// try index.reserve(1_000);
///
/// const executor = try usearch.Executor.forRam(&.{}, 0);
/// defer executor.deinit();
///
/// const keys = [_]usearch.Key{ .fromInteger(0), .fromInteger(1) };
/// const slab = [_]f32{ 0.2, 0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0.2 };
/// _ = try index.add(executor, &keys, try .of(f32, &slab, 4), .{}, null);
///
/// const query = [_]f32{ 0.2, 0.1, 0.2, 0.1 };
/// var found: [10]usearch.Key = undefined;
/// var scores: [10]usearch.Distance = undefined;
/// var counts: [1]usize = undefined;
/// try index.search(executor, try .ofRow(f32, &query), try .of(&found, &scores, &counts), .{}, null);
/// ```
pub const DenseIndex = struct {
    /// The opaque `usearch_dense_index_t` the ABI hands back.
    handle: *anyopaque,

    /// Initializes a new index for the residency the options select.
    pub fn init(options: *const Options) Error!DenseIndex {
        var outcome: Outcome = .{};
        const handle = usearch_dense_init(options, &outcome);
        errdefer if (handle) |live| usearch_dense_free(live);
        try outcome.check();
        return .{ .handle = handle orelse return refuse(.internal) };
    }

    /// Frees the resources associated with the index.
    pub fn deinit(self: DenseIndex) void {
        usearch_dense_free(self.handle);
    }

    /// Reserves room for a total number of entries, the current size included.
    pub fn reserve(self: DenseIndex, entries: usize) Error!void {
        var outcome: Outcome = .{};
        usearch_dense_reserve(self.handle, entries, &outcome);
        try outcome.check();
    }

    /// Erases all entries, keeping the reserved capacity and frozen geometry. Never runs
    /// concurrently with any other verb on the same handle.
    pub fn clear(self: DenseIndex) Error!void {
        var outcome: Outcome = .{};
        usearch_dense_clear(self.handle, &outcome);
        try outcome.check();
    }

    /// The number of stored entries.
    pub fn size(self: DenseIndex) Error!usize {
        return self.readIntegerProperty(usearch_dense_size);
    }

    /// The reserved capacity in entries.
    pub fn capacity(self: DenseIndex) Error!usize {
        return self.readIntegerProperty(usearch_dense_capacity);
    }

    /// The frozen dimensionality of the stored vectors.
    pub fn dimensions(self: DenseIndex) Error!usize {
        return self.readIntegerProperty(usearch_dense_dimensions);
    }

    /// The frozen connections-per-node bound of the graph.
    pub fn connectivity(self: DenseIndex) Error!usize {
        return self.readIntegerProperty(usearch_dense_connectivity);
    }

    /// The memory the index occupies, in bytes.
    pub fn memoryUsage(self: DenseIndex) Error!usize {
        return self.readIntegerProperty(usearch_dense_memory_usage);
    }

    /// The bytes one monolith of the index takes - what `saveBuffer` needs.
    pub fn storageRequirements(self: DenseIndex) Error!usize {
        return self.readIntegerProperty(usearch_dense_storage_requirements);
    }

    /// The capability codename the index dispatched to on this machine.
    pub fn capabilities(self: DenseIndex) Error![:0]const u8 {
        var outcome: Outcome = .{};
        const named = usearch_dense_capabilities(self.handle, &outcome);
        try outcome.check();
        return std.mem.span(named orelse return refuse(.internal));
    }

    /// Reads one of the handle's integer properties, which every geometry and footprint accessor
    /// above shares the shape of; `capabilities` is the one property that answers text instead.
    ///
    /// The entry point is comptime, so each accessor compiles to its own direct call rather than
    /// routing through a shared function pointer.
    fn readIntegerProperty(
        self: DenseIndex,
        comptime entry_point: fn (*anyopaque, *Outcome) callconv(.c) usize,
    ) Error!usize {
        var outcome: Outcome = .{};
        const reading = entry_point(self.handle, &outcome);
        try outcome.check();
        return reading;
    }

    /// Adds a matrix of vectors under their keys - the native batch operation, one key per row.
    /// Returns the rows published, which is every row on success.
    ///
    /// The batch is all-or-nothing: every fallible step runs before the first entry identity is
    /// claimed, so a rejected batch leaves the index untouched. Overlaps safely with concurrent
    /// searches on the same handle, and the input scalar type is independent of the stored
    /// quantization - the engine converts on ingest.
    pub fn add(
        self: DenseIndex,
        executor: Executor,
        keys: []const Key,
        vectors: MatrixView,
        options: InsertionOptions,
        stats: ?*TraversalStats,
    ) Error!usize {
        return self.addDispatch(executor, keys, vectors, options, stats, null, null);
    }

    /// Adds a matrix of vectors like `add`, reporting the rows admitted as they publish and
    /// stopping wherever `reports` asks it to.
    ///
    /// A handler makes the batch incremental rather than all-or-nothing: a stop lands on an
    /// increment boundary, the rows already published stand, and the call reports
    /// `Error.Cancelled` - so the count the handler last saw is what the index kept.
    ///
    /// The handler arrives on whichever thread advanced the work and never on two at once, so it
    /// needs no synchronization of its own. Pass `{}` for a handler needing no context, and a
    /// single-item pointer otherwise.
    pub fn addReporting(
        self: DenseIndex,
        executor: Executor,
        keys: []const Key,
        vectors: MatrixView,
        options: InsertionOptions,
        stats: ?*TraversalStats,
        context: anytype,
        comptime reports: fn (@TypeOf(context), usize, usize) Continuation,
    ) Error!usize {
        const Context = @TypeOf(context);
        checkContext(Context);
        return self.addDispatch(
            executor,
            keys,
            vectors,
            options,
            stats,
            progressTrampoline(Context, reports),
            contextPointer(Context, context),
        );
    }

    fn addDispatch(
        self: DenseIndex,
        executor: Executor,
        keys: []const Key,
        vectors: MatrixView,
        options: InsertionOptions,
        stats: ?*TraversalStats,
        progress: ?Progress,
        progress_state: ?*anyopaque,
    ) Error!usize {
        var outcome: Outcome = .{};
        const added = usearch_dense_add(
            self.handle,
            executor.handle,
            .of(keys),
            &vectors,
            &options,
            progress,
            progress_state,
            stats,
            &outcome,
        );
        try outcome.check();
        return added;
    }

    /// Re-selects the neighbors of every stored entry against the finished graph, returning the
    /// nodes whose edges changed; zero means the graph has settled.
    ///
    /// A build step rather than a repair: a cheap insertion pass followed by one refinement
    /// reaches a better graph, in less total time, than one expensive insertion pass - so a wide
    /// expansion factor belongs here rather than in `add`. Searches may run throughout.
    pub fn refine(
        self: DenseIndex,
        executor: Executor,
        options: InsertionOptions,
        stats: ?*TraversalStats,
    ) Error!usize {
        return self.refineDispatch(executor, options, stats, null, null);
    }

    /// Re-selects neighbors like `refine`, reporting the nodes swept and stopping wherever
    /// `reports` asks it to.
    ///
    /// Stopping here is a time budget rather than a failure: re-selecting a node's edges is
    /// idempotent and claims no identity, so a stopped sweep leaves the graph correct and merely
    /// less improved, reports no `Error.Cancelled`, and resumes where it stopped on the next call.
    /// A deadline check is the handler this asks for.
    ///
    /// The handler arrives on whichever thread advanced the work and never on two at once, so it
    /// needs no synchronization of its own. Pass `{}` for a handler needing no context, and a
    /// single-item pointer otherwise.
    pub fn refineReporting(
        self: DenseIndex,
        executor: Executor,
        options: InsertionOptions,
        stats: ?*TraversalStats,
        context: anytype,
        comptime reports: fn (@TypeOf(context), usize, usize) Continuation,
    ) Error!usize {
        const Context = @TypeOf(context);
        checkContext(Context);
        return self.refineDispatch(
            executor,
            options,
            stats,
            progressTrampoline(Context, reports),
            contextPointer(Context, context),
        );
    }

    fn refineDispatch(
        self: DenseIndex,
        executor: Executor,
        options: InsertionOptions,
        stats: ?*TraversalStats,
        progress: ?Progress,
        progress_state: ?*anyopaque,
    ) Error!usize {
        var outcome: Outcome = .{};
        const changed = usearch_dense_refine(
            self.handle,
            executor.handle,
            &options,
            progress,
            progress_state,
            stats,
            &outcome,
        );
        try outcome.check();
        return changed;
    }

    /// Resolves a matrix of queries into caller-owned outputs - the native batch operation.
    ///
    /// Never writes to the graph, so any number of these may overlap with concurrent insertions.
    pub fn search(
        self: DenseIndex,
        executor: Executor,
        queries: MatrixView,
        matches: MatchesSpan,
        options: SearchOptions,
        stats: ?*TraversalStats,
    ) Error!void {
        return self.searchDispatch(executor, queries, matches, options, stats, null, null);
    }

    /// Resolves a matrix of queries like `search`, reporting the queries resolved and stopping
    /// wherever `reports` asks it to.
    ///
    /// A stopped batch reports `Error.Cancelled` and leaves every output row indeterminate, a
    /// query resolved before the stop being indistinguishable from one that never ran.
    ///
    /// The handler arrives on whichever thread advanced the work and never on two at once, so it
    /// needs no synchronization of its own. Pass `{}` for a handler needing no context, and a
    /// single-item pointer otherwise.
    pub fn searchReporting(
        self: DenseIndex,
        executor: Executor,
        queries: MatrixView,
        matches: MatchesSpan,
        options: SearchOptions,
        stats: ?*TraversalStats,
        context: anytype,
        comptime reports: fn (@TypeOf(context), usize, usize) Continuation,
    ) Error!void {
        const Context = @TypeOf(context);
        checkContext(Context);
        return self.searchDispatch(
            executor,
            queries,
            matches,
            options,
            stats,
            progressTrampoline(Context, reports),
            contextPointer(Context, context),
        );
    }

    fn searchDispatch(
        self: DenseIndex,
        executor: Executor,
        queries: MatrixView,
        matches: MatchesSpan,
        options: SearchOptions,
        stats: ?*TraversalStats,
        progress: ?Progress,
        progress_state: ?*anyopaque,
    ) Error!void {
        var outcome: Outcome = .{};
        usearch_dense_search(
            self.handle,
            executor.handle,
            &queries,
            &matches,
            &options,
            progress,
            progress_state,
            stats,
            &outcome,
        );
        try outcome.check();
    }

    /// The filtered sibling of `search`, admitting only the keys `admits` approves.
    ///
    /// Non-matching entries still route the traversal rather than terminating it, so recall does
    /// not collapse at low selectivity. The cost is that `admits` runs once per scored candidate,
    /// far more often than `wanted`, and concurrently from the executor's threads - so it belongs
    /// on the caller's fast path and must tolerate that. Pass `{}` for a predicate needing no
    /// context, and a single-item pointer otherwise.
    pub fn searchIf(
        self: DenseIndex,
        executor: Executor,
        queries: MatrixView,
        matches: MatchesSpan,
        options: SearchOptions,
        stats: ?*TraversalStats,
        context: anytype,
        comptime admits: fn (@TypeOf(context), Key) Admission,
    ) Error!void {
        const Context = @TypeOf(context);
        checkContext(Context);
        const trampoline = struct {
            fn admit(key: Key, state: ?*anyopaque) callconv(.c) Admission {
                if (Context == void) return admits({}, key);
                return admits(@ptrCast(@alignCast(state.?)), key);
            }
        }.admit;

        var outcome: Outcome = .{};
        usearch_dense_search_if(
            self.handle,
            executor.handle,
            &queries,
            &matches,
            &options,
            trampoline,
            contextPointer(Context, context),
            null,
            null,
            stats,
            &outcome,
        );
        try outcome.check();
    }

    /// The entries stored under `key`; zero means the index holds none.
    pub fn count(self: DenseIndex, key: Key) Error!usize {
        var outcome: Outcome = .{};
        const found = usearch_dense_count(self.handle, key, &outcome);
        try outcome.check();
        return found;
    }

    /// Copies the vectors stored under `key` into `vectors`, one row per entry sharing the key,
    /// in insertion order. Returns the true number of entries, never the number written.
    ///
    /// A return above `vectors.capacity` means the copy was truncated, so allocate that many rows
    /// and call again. `MatrixSpan.none` probes the count without copying.
    pub fn get(self: DenseIndex, key: Key, vectors: MatrixSpan) Error!usize {
        var outcome: Outcome = .{};
        const stored = usearch_dense_get(self.handle, key, if (vectors.data == null) null else &vectors, &outcome);
        try outcome.check();
        return stored;
    }

    /// Opens a consistent snapshot cursor over the current contents.
    pub fn cursor(self: DenseIndex) Error!Cursor {
        var outcome: Outcome = .{};
        const handle = usearch_dense_cursor(self.handle, &outcome);
        errdefer if (handle) |live| usearch_dense_cursor_free(live);
        try outcome.check();
        return .{ .handle = handle orelse return refuse(.internal) };
    }

    /// Opens a resumable nearest-neighbor stream for one query row.
    ///
    /// A live finder holds one of the executor's threads for as long as it exists, so an executor
    /// with few threads and many open finders starves. Free each one the moment its caller stops
    /// pulling, rather than at the end of the enclosing scope.
    pub fn find(self: DenseIndex, executor: Executor, query: MatrixView) Error!Finder {
        if (query.rows != 1) return refuse(.shape_mismatch);
        var outcome: Outcome = .{};
        const handle = usearch_dense_find(self.handle, executor.handle, &query, &outcome);
        errdefer if (handle) |live| usearch_dense_find_free(live);
        try outcome.check();
        return .{ .handle = handle orelse return refuse(.internal) };
    }

    /// Saves the index under `path` in the layout its extension picks - a `.safegraphs` monolith,
    /// or a `.safegraphs.json` manifest beside its shards.
    ///
    /// Demands a mutation-free window; concurrent readers keep searching throughout. A failed
    /// save leaves a file without its header, which every reader rejects, and that leftover is
    /// the caller's to remove.
    pub fn save(self: DenseIndex, executor: Executor, path: []const u8) Error!void {
        return self.saveDispatch(executor, path, null, null);
    }

    /// Saves the index like `save`, reporting the shards written and stopping wherever `reports`
    /// asks it to.
    ///
    /// A stopped save leaves a monolith without its header, or the parts of a manifest layout
    /// without the manifest - leftovers every reader rejects and the caller is left to remove.
    ///
    /// The handler arrives on whichever thread advanced the work and never on two at once, so it
    /// needs no synchronization of its own. Pass `{}` for a handler needing no context, and a
    /// single-item pointer otherwise.
    pub fn saveReporting(
        self: DenseIndex,
        executor: Executor,
        path: []const u8,
        context: anytype,
        comptime reports: fn (@TypeOf(context), usize, usize) Continuation,
    ) Error!void {
        const Context = @TypeOf(context);
        checkContext(Context);
        return self.saveDispatch(
            executor,
            path,
            progressTrampoline(Context, reports),
            contextPointer(Context, context),
        );
    }

    fn saveDispatch(
        self: DenseIndex,
        executor: Executor,
        path: []const u8,
        progress: ?Progress,
        progress_state: ?*anyopaque,
    ) Error!void {
        var outcome: Outcome = .{};
        usearch_dense_save(self.handle, executor.handle, BytesView.of(path), progress, progress_state, &outcome);
        try outcome.check();
    }

    /// Loads a checkpoint into this handle, shard by shard, with the residency deciding the
    /// behavior: materialize into RAM, upload to the devices, or open in place on disk.
    ///
    /// Point `path` at the manifest, never at one of the parts it enumerates. A handle whose
    /// geometry was left zeroed freezes it here, from the checkpoint's own headers.
    pub fn load(self: DenseIndex, executor: Executor, path: []const u8) Error!void {
        return self.loadDispatch(executor, path, null, null);
    }

    /// Loads a checkpoint like `load`, reporting the shards placed and stopping wherever `reports`
    /// asks it to.
    ///
    /// A stopped load publishes nothing: the handle is empty afterwards and loads cleanly again.
    ///
    /// The handler arrives on whichever thread advanced the work and never on two at once, so it
    /// needs no synchronization of its own. Pass `{}` for a handler needing no context, and a
    /// single-item pointer otherwise.
    pub fn loadReporting(
        self: DenseIndex,
        executor: Executor,
        path: []const u8,
        context: anytype,
        comptime reports: fn (@TypeOf(context), usize, usize) Continuation,
    ) Error!void {
        const Context = @TypeOf(context);
        checkContext(Context);
        return self.loadDispatch(
            executor,
            path,
            progressTrampoline(Context, reports),
            contextPointer(Context, context),
        );
    }

    fn loadDispatch(
        self: DenseIndex,
        executor: Executor,
        path: []const u8,
        progress: ?Progress,
        progress_state: ?*anyopaque,
    ) Error!void {
        var outcome: Outcome = .{};
        usearch_dense_load(self.handle, executor.handle, BytesView.of(path), progress, progress_state, &outcome);
        try outcome.check();
    }

    /// Saves the index into a caller-owned buffer of at least `storageRequirements` bytes, as one
    /// monolith - the same bytes the path-taking sibling writes.
    pub fn saveBuffer(self: DenseIndex, executor: Executor, buffer: []u8) Error!void {
        var outcome: Outcome = .{};
        usearch_dense_save_buffer(self.handle, executor.handle, BytesSpan.of(buffer), null, null, &outcome);
        try outcome.check();
    }

    /// Loads a monolith from a memory buffer into this handle, the bytes materializing into its
    /// own shards.
    pub fn loadBuffer(self: DenseIndex, executor: Executor, buffer: []const u8) Error!void {
        var outcome: Outcome = .{};
        usearch_dense_load_buffer(self.handle, executor.handle, BytesView.of(buffer), null, null, &outcome);
        try outcome.check();
    }

    /// Approximate sparse product of the left-hand matrix against the indexed rows, reporting
    /// every pair the traversal scored instead of only the nearest.
    ///
    /// Pairs arrive in traversal order rather than sorted, and carry no duplicates. Each entry of
    /// `matches.counts` reports the true number of pairs scored, so a value above `wanted` means
    /// that row was truncated to its nearest pairs. Cost is roughly
    /// `rows × expansion × connectivity × dimensions`, sublinear in the indexed rows.
    ///
    /// Runs on the inner-product kinds only: a graph frozen under a distance kind scored a
    /// different quantity, and the call refuses with `Error.ConfigMismatch`.
    pub fn multiply(
        self: DenseIndex,
        executor: Executor,
        left: MatrixView,
        matches: MatchesSpan,
        options: SearchOptions,
        stats: ?*TraversalStats,
    ) Error!void {
        var outcome: Outcome = .{};
        usearch_dense_multiply(self.handle, executor.handle, &left, &matches, &options, null, null, stats, &outcome);
        try outcome.check();
    }

    /// Softmax-weighted attention of a query matrix over the indexed keys, folded into the graph
    /// walk in a single pass - no match list is materialized, and each value row is gathered
    /// exactly once, at the moment its key was scored.
    ///
    /// The index holds the keys and `values` holds the value rows, addressed by key: an entry's
    /// key is its row offset into `values`, parked in the low bytes. `outputs` comes back already
    /// divided by its partition, so `maximum_scores` and `partitions` - the online softmax's `m`
    /// and `l` - are worth asking for only together, to fold this branch into another over the
    /// same queries.
    ///
    /// Runs on the inner-product kinds only, since a softmax needs logits.
    pub fn attend(
        self: DenseIndex,
        executor: Executor,
        queries: MatrixView,
        values: MatrixView,
        outputs: MatrixSpan,
        carry: ?[]SoftmaxState,
        options: AttentionOptions,
        stats: ?*TraversalStats,
    ) Error!void {
        if (outputs.capacity < queries.rows) return refuse(.shape_mismatch);
        if (outputs.columns != values.columns) return refuse(.shape_mismatch);
        if (carry) |state| if (state.len < queries.rows) return refuse(.shape_mismatch);
        var outcome: Outcome = .{};
        usearch_dense_attend(
            self.handle,
            executor.handle,
            &queries,
            // The value table carries its own row count now, so a key past it is refused rather
            // than silently skipped - a missing row is a wrong softmax, not a smaller one.
            &values,
            &outputs,
            if (carry) |state| state.ptr else null,
            &options,
            null,
            null,
            stats,
            &outcome,
        );
        try outcome.check();
    }

    /// Late-interaction scoring of multi-row queries against multi-row entries: every query row
    /// takes its best match within a key, and those maxima sum to that key's score.
    ///
    /// `queries` carries the queries laid end to end, `rows_per_query` rows apiece and uniform, so
    /// ColBERT's padding applies as is. `matches.wanted` is the "k" over documents, not over rows.
    /// Approximate downward only: a token whose neighborhood the walk never entered contributes no
    /// maximum, so a key can score below its exact value but never above it.
    ///
    /// Runs on the inner-product kinds only, since a maximum over logits is what it sums.
    pub fn maxsim(
        self: DenseIndex,
        executor: Executor,
        queries: MatrixView,
        rows_per_query: usize,
        matches: MatchesSpan,
        options: SearchOptions,
        stats: ?*TraversalStats,
    ) Error!void {
        if (rows_per_query == 0 or queries.rows % rows_per_query != 0) return refuse(.shape_mismatch);
        var outcome: Outcome = .{};
        usearch_dense_maxsim(
            self.handle,
            executor.handle,
            &queries,
            rows_per_query,
            &matches,
            &options,
            null,
            null,
            stats,
            &outcome,
        );
        try outcome.check();
    }
};

/// Reads a checkpoint's geometry and metric, without loading it.
pub fn metadata(path: []const u8) Error!Metadata {
    var found: Metadata = .{};
    var outcome: Outcome = .{};
    usearch_dense_metadata(BytesView.of(path), &found, &outcome);
    try outcome.check();
    return found;
}

/// Reads geometry and metric out of a serialized monolith buffer, without loading it.
pub fn metadataBuffer(buffer: []const u8) Error!Metadata {
    var found: Metadata = .{};
    var outcome: Outcome = .{};
    usearch_dense_metadata_buffer(BytesView.of(buffer), &found, &outcome);
    try outcome.check();
    return found;
}

/// A consistent snapshot cursor over the index contents.
///
/// Captures the publication watermark at creation, so concurrent inserts never appear
/// mid-iteration, and the walk is a sequential row scan - export, migration, and visualization
/// feed off it.
pub const Cursor = struct {
    /// The opaque `usearch_dense_cursor_t` the ABI hands back.
    handle: *anyopaque,

    /// Copies the next batch into `keys` and `vectors`, returning the rows written; zero means
    /// the snapshot is exhausted. `MatrixSpan.none` skips the vector export.
    ///
    /// The key run bounds the batch, so a vector export must have room for as many rows.
    pub fn next(self: Cursor, keys: []Key, vectors: MatrixSpan) Error!usize {
        var outcome: Outcome = .{};
        const written = usearch_dense_cursor_next(
            self.handle,
            .of(keys),
            if (vectors.data == null) null else &vectors,
            &outcome,
        );
        try outcome.check();
        return written;
    }

    /// Releases the cursor.
    pub fn deinit(self: Cursor) void {
        usearch_dense_cursor_free(self.handle);
    }
};

/// A resumable nearest-neighbor stream over one query.
///
/// Best-first traversal is naturally pausable, so the finder owns a live frontier and visited set
/// and yields matches lazily - callers that post-filter aggressively pull until satisfied instead
/// of guessing "k" upfront, at the price of pinning one of the executor's threads until freed.
pub const Finder = struct {
    /// The opaque `usearch_dense_finder_t` the ABI hands back.
    handle: *anyopaque,

    /// Pulls the next matches into `keys` and `distances`, returning how many were written; zero
    /// means the traversal is drained.
    ///
    /// Ordering beyond the configured expansion horizon is best-effort nearest rather than
    /// globally exact, so pulling far past that breadth degrades the ranking rather than ending
    /// the stream.
    pub fn next(self: Finder, keys: []Key, distances: []Distance) Error!usize {
        if (keys.len != distances.len) return refuse(.shape_mismatch);
        var outcome: Outcome = .{};
        const written = usearch_dense_find_next(self.handle, .of(keys), distances.ptr, &outcome);
        try outcome.check();
        return written;
    }

    /// Releases the finder and the traversal context it pinned.
    pub fn deinit(self: Finder) void {
        usearch_dense_find_free(self.handle);
    }
};

test "a slab reads as rows, and one vector as a single row" {
    const slab = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const matrix = try MatrixView.of(f32, &slab, 3);
    try std.testing.expectEqual(@as(usize, 2), matrix.rows);
    try std.testing.expectEqual(@as(usize, 3), matrix.columns);
    try std.testing.expectEqual(@as(usize, 0), matrix.stride_bytes);
    try std.testing.expectEqual(Scalar.f32, matrix.scalar_kind);

    const single = try MatrixView.ofRow(f32, &slab);
    try std.testing.expectEqual(@as(usize, 1), single.rows);
    try std.testing.expectEqual(@as(usize, 6), single.columns);

    // An absent input is a named value rather than a null the caller assembles.
    try std.testing.expectEqual(@as(?*const anyopaque, null), MatrixView.none.data);
    try std.testing.expectEqual(@as(?*anyopaque, null), MatrixSpan.none.data);
}

test "a shape the lengths cannot support is refused rather than divided by" {
    const slab = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var writable = slab;
    try std.testing.expectError(Error.ShapeMismatch, MatrixView.of(f32, &slab, 0));
    try std.testing.expectError(Error.ShapeMismatch, MatrixView.of(f32, &slab, 4));
    try std.testing.expectError(Error.ShapeMismatch, MatrixView.ofRow(f32, slab[0..0]));
    try std.testing.expectError(Error.ShapeMismatch, MatrixSpan.of(f32, &writable, 0));
    try std.testing.expectError(Error.ShapeMismatch, MatrixSpan.of(f32, &writable, 4));

    var keys: [30]Key = undefined;
    var scores: [30]Distance = undefined;
    var counts: [3]usize = undefined;
    try std.testing.expectError(Error.ShapeMismatch, MatchesSpan.of(&keys, &scores, counts[0..0]));
    try std.testing.expectError(Error.ShapeMismatch, MatchesSpan.of(keys[0..29], &scores, &counts));
    try std.testing.expectError(Error.ShapeMismatch, MatchesSpan.of(keys[0..8], scores[0..8], &counts));
}

test "packed outputs derive their width from the counts" {
    var keys: [30]Key = undefined;
    var distances: [30]Distance = undefined;
    var counts: [3]usize = undefined;
    const matches = try MatchesSpan.of(&keys, &distances, &counts);
    try std.testing.expectEqual(@as(usize, 10), matches.wanted);
    // A packed run leaves both strides zero, which the interface reads as the natural width.
    try std.testing.expectEqual(@as(usize, 0), matches.keys_stride_bytes);
    try std.testing.expectEqual(@as(usize, 0), matches.scores_stride_bytes);
}

test "options tag their placement and their union together" {
    const devices = [_]i32{ 0, 1, 2, 3 };
    const options = (Options{ .dimensions = 96, .metric = .inner }).placed(.{ .vram = &devices });
    try std.testing.expectEqual(Residency.vram, options.residency);
    try std.testing.expectEqual(@as(usize, 4), options.placement.vram.devices_count);
    try std.testing.expectEqual(@as(usize, 96), options.dimensions);

    // Left alone, an index is a RAM index across every domain, quantized to bf16.
    const defaults: Options = .{};
    try std.testing.expectEqual(Residency.ram, defaults.residency);
    try std.testing.expectEqual(Scalar.bf16, defaults.quantization);
    try std.testing.expectEqual(Metric.angular, defaults.metric);
}

test "a predicate compiles with and without a context" {
    // Only the trampoline's shape is asserted here - calling one needs a live index.
    const Tenant = struct {
        allowed: u8,
        fn admits(self: *const @This(), key: Key) Admission {
            return if (key.bytes[0] == self.allowed) .admit else .reject;
        }
    };
    const tenant: Tenant = .{ .allowed = 7 };
    checkContext(*const Tenant);
    checkContext(void);
    try std.testing.expectEqual(Admission.admit, Tenant.admits(&tenant, .{ .bytes = .{7} ++ .{0} ** 15 }));
    try std.testing.expect(contextPointer(*const Tenant, &tenant) != null);
    try std.testing.expectEqual(@as(?*anyopaque, null), contextPointer(void, {}));
}

test { // analyze every declaration, the methods on each handle included
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(DenseIndex);
    std.testing.refAllDecls(Cursor);
    std.testing.refAllDecls(Finder);
    std.testing.refAllDecls(Options);
    std.testing.refAllDecls(MatchesSpan);
    std.testing.refAllDecls(MatrixView);
    std.testing.refAllDecls(MatrixSpan);
}

test "the generic verbs typecheck" {
    // `searchIf`, the reporting siblings and the matrix constructors are generic, so
    // `refAllDecls` above names them without analyzing them - only an instantiation does that, and
    // this is the only one in the tree. The branch never runs; the compiler still walks it.
    var never = false;
    _ = &never;
    if (never) {
        const index: DenseIndex = undefined;
        const executor: Executor = undefined;
        const slab: []const f32 = undefined;
        const keys: []Key = undefined;
        const distances: []Distance = undefined;
        const counts: []usize = undefined;
        const matches = try MatchesSpan.of(keys, distances, counts);
        const Tenant = struct {
            allowed: u8,
            fn admits(self: *const @This(), key: Key) Admission {
                return if (key.bytes[0] == self.allowed) .admit else .reject;
            }
        };
        const tenant: Tenant = .{ .allowed = 7 };
        try index.searchIf(executor, try .of(f32, slab, 4), matches, .{}, null, &tenant, Tenant.admits);
        try index.searchIf(executor, try .of(f32, slab, 4), matches, .{}, null, {}, struct {
            fn admits(_: void, _: Key) Admission {
                return .admit;
            }
        }.admits);
        _ = try index.get(.nil, try .of(f32, @constCast(slab), 4));

        // Every reporting sibling, once with a context and once without, since the trampoline is
        // generated per call site and only an instantiation compiles it.
        const path: []const u8 = undefined;
        var tally: usize = 0;
        const counting = struct {
            fn saw(seen: *usize, completed: usize, _: usize) Continuation {
                seen.* = completed;
                return .@"continue";
            }
        }.saw;
        const stateless = struct {
            fn saw(_: void, _: usize, _: usize) Continuation {
                return .@"continue";
            }
        }.saw;
        _ = try index.addReporting(executor, keys, try .of(f32, slab, 4), .{}, null, &tally, counting);
        _ = try index.addReporting(executor, keys, try .of(f32, slab, 4), .{}, null, {}, stateless);
        _ = try index.refineReporting(executor, .{}, null, &tally, counting);
        _ = try index.refineReporting(executor, .{}, null, {}, stateless);
        try index.searchReporting(executor, try .of(f32, slab, 4), matches, .{}, null, &tally, counting);
        try index.searchReporting(executor, try .of(f32, slab, 4), matches, .{}, null, {}, stateless);
        try index.saveReporting(executor, path, &tally, counting);
        try index.saveReporting(executor, path, {}, stateless);
        try index.loadReporting(executor, path, &tally, counting);
        try index.loadReporting(executor, path, {}, stateless);
    }
}

test "a custom metric tags the kind, the quantization, and the callback together" {
    const Weights = struct {
        scale: f32,
        fn score(self: *const @This(), left: []const f32, right: []const f32) Distance {
            var total: f32 = 0;
            for (left, right) |first, second| total += first * second;
            return @as(Distance, total * self.scale);
        }
    };
    const weights: Weights = .{ .scale = 2 };
    const options = (Options{ .dimensions = 4 }).scoredBy(f32, &weights, Weights.score);
    try std.testing.expectEqual(Metric.unknown, options.metric);
    try std.testing.expectEqual(Scalar.f32, options.quantization);
    try std.testing.expect(options.metric_callback != null);
    try std.testing.expectEqual(@as(?*anyopaque, @ptrCast(@constCast(&weights))), options.metric_callback_state);

    // The trampoline is what the engine calls, so it must read the two rows back as typed slices.
    const left = [_]f32{ 1, 2, 3, 4 };
    const right = [_]f32{ 1, 1, 1, 1 };
    try std.testing.expectEqual(
        @as(Distance, 20),
        options.metric_callback.?(&left, &right, left.len, options.metric_callback_state),
    );

    // A metric needing no context travels with a null state and is never handed one.
    const stateless = (Options{}).scoredBy(f32, {}, struct {
        fn score(_: void, first: []const f32, second: []const f32) Distance {
            return @as(Distance, @floatFromInt(first.len + second.len));
        }
    }.score);
    try std.testing.expectEqual(@as(?*anyopaque, null), stateless.metric_callback_state);
    try std.testing.expectEqual(@as(Distance, 8), stateless.metric_callback.?(&left, &right, left.len, null));
}

test "a batch crosses the ABI and comes back through the spans the caller lends" {
    const dimensions = 3;
    const executor = try Executor.forRam(&.{}, 2);
    defer executor.deinit();
    const index = try DenseIndex.init(&.{ .dimensions = dimensions, .metric = .euclidean, .quantization = .f32 });
    defer index.deinit();
    try index.reserve(3);

    const slab = [_]f32{ 1, 0, 0, 0, 1, 0, 0, 0, 1 };
    const keys = [_]Key{ .fromInteger(1), .fromInteger(2), .fromInteger(3) };
    var stats: TraversalStats = .{};
    const added = try index.add(executor, &keys, try .of(f32, &slab, dimensions), .{}, &stats);
    try std.testing.expectEqual(@as(usize, 3), added);
    try std.testing.expectEqual(@as(usize, 3), try index.size());
    try std.testing.expectEqual(@as(usize, dimensions), try index.dimensions());

    var found: [2]Key = undefined;
    var scores: [2]Distance = undefined;
    var counts: [1]usize = undefined;
    try index.search(executor, try .ofRow(f32, slab[0..dimensions]), try .of(&found, &scores, &counts), .{}, &stats);
    try std.testing.expectEqual(@as(usize, 2), counts[0]);
    try std.testing.expect(stats.computed_distances != 0);

    // A stored row comes back through the span the caller lends, value for value.
    var exported = [_]f32{ -1, -1, -1 };
    try std.testing.expectEqual(@as(usize, 1), try index.get(keys[1], try .of(f32, &exported, dimensions)));
    try std.testing.expectEqualSlices(f32, &.{ 0, 1, 0 }, &exported);
    try std.testing.expectEqual(@as(usize, 1), try index.count(keys[1]));

    // A row the frozen geometry disagrees with is refused, and the status crosses back intact.
    const narrow = [_]f32{ 1, 0 };
    try std.testing.expectError(
        Error.ShapeMismatch,
        index.search(executor, try .ofRow(f32, &narrow), try .of(&found, &scores, &counts), .{}, null),
    );

    // A finder streams one query row and refuses a matrix carrying more.
    try std.testing.expectError(Error.ShapeMismatch, index.find(executor, try .of(f32, &slab, dimensions)));
    const finder = try index.find(executor, try .ofRow(f32, slab[0..dimensions]));
    defer finder.deinit();
    try std.testing.expect(try finder.next(&found, &scores) != 0);

    // A cursor walks the snapshot, and its key run bounds the batch.
    const walk = try index.cursor();
    defer walk.deinit();
    var walked: [3]Key = undefined;
    try std.testing.expectEqual(@as(usize, 3), try walk.next(&walked, .none));
    try std.testing.expectEqual(@as(usize, 0), try walk.next(&walked, .none));
}

test "a reporting batch counts what it published, and a stop keeps it" {
    const dimensions = 3;
    const executor = try Executor.forRam(&.{}, 2);
    defer executor.deinit();
    const index = try DenseIndex.init(&.{ .dimensions = dimensions, .metric = .euclidean, .quantization = .f32 });
    defer index.deinit();
    try index.reserve(6);

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
        fn carryOn(self: *@This(), completed: usize, total: usize) Continuation {
            self.record(completed, total);
            return .@"continue";
        }
        fn stop(self: *@This(), completed: usize, total: usize) Continuation {
            self.record(completed, total);
            return .cancel;
        }
    };

    var watched: Watch = .{};
    const added = try index.addReporting(
        executor,
        &keys,
        try .of(f32, &slab, dimensions),
        .{},
        null,
        &watched,
        Watch.carryOn,
    );
    try std.testing.expectEqual(@as(usize, 3), added);
    try std.testing.expect(watched.ordered);
    try std.testing.expectEqual(@as(usize, 3), watched.last);
    try std.testing.expectEqual(@as(usize, 3), watched.total);

    // What a stop leaves behind is the increment's size, not this test's to name, so the index is
    // checked against the count the handler last saw.
    var stopped: Watch = .{};
    const more = [_]Key{ .fromInteger(4), .fromInteger(5), .fromInteger(6) };
    if (index.addReporting(executor, &more, try .of(f32, &slab, dimensions), .{}, null, &stopped, Watch.stop)) |published| {
        try std.testing.expectEqual(stopped.last, published);
    } else |failure| {
        try std.testing.expectEqual(Error.Cancelled, failure);
    }
    try std.testing.expectEqual(3 + stopped.last, try index.size());

    // Stopping a sweep is the time budget rather than a failure, so no status comes back.
    var swept: Watch = .{};
    _ = try index.refineReporting(executor, .{}, null, &swept, Watch.stop);
    try std.testing.expect(swept.ordered);
    try std.testing.expectEqual(try index.size(), swept.total);
}
