//! The sequence modality - Sequence Search over byte strings, UTF-8 text, DNA, RNA and proteins,
//! under edit distances and alignments.
//!
//! `SequenceIndex` is the handle; `TapeView` is the batch currency every verb reads, the bytes of
//! every sequence end to end around one more offset than entries, as an Apache Arrow `Binary`
//! column lays them out. Every output whose length scales with the call lands in an `Arena` the
//! verb requires and comes back as a view over it, and `SequenceCursor` walks the contents.
//!
//! The receiver chooses where a verb runs and how it completes, as for the dense modality: the
//! index runs it on the calling thread, `index.on(executor)` blocks on an executor, and
//! `index.deferred(executor, &call)` submits it there into a `Deferred` the caller holds.
//!
//! File: zig/usearch/sequence.zig
//! Author: Ash Vardanian
//! Date: November 16, 2026

const std = @import("std");

const types = @import("types.zig");
const basic = @import("basic.zig");
const dense = @import("dense.zig");
const Arena = types.Arena;
const BytesView = types.BytesView;
const Completion = types.Completion;
const Error = types.Error;
const Executor = types.Executor;
const Key = types.Key;
const Matches = types.Matches;
const Outcome = types.Outcome;
const Predicate = types.Predicate;
const PhaseProgress = types.PhaseProgress;
const ProgressCallback = types.ProgressCallback;
const Status = types.Status;
const TapeEntries = basic.TapeEntries;
const TapeView = basic.TapeView;
const Traversal = types.Traversal;
const refuseInto = types.refuseInto;
const Deferred = dense.Deferred;
const DenseCursorBatch = dense.DenseCursorBatch;
const DenseDraws = dense.DenseDraws;
const DenseMetadataReadBufferConfig = dense.DenseMetadataReadBufferConfig;
const DenseMetadataReadConfig = dense.DenseMetadataReadConfig;
const DenseReachability = dense.DenseReachability;
const DenseRefineReport = dense.DenseRefineReport;
const DenseSampleConfig = dense.DenseSampleConfig;
const DenseSaveBufferConfig = dense.DenseSaveBufferConfig;
const DenseSaveConfig = dense.DenseSaveConfig;
const DenseTraversalStats = dense.DenseTraversalStats;
const Monolith = dense.Monolith;
const usearch_keys_tape_view_t = types.abi.keys_tape_view_t;
const usearch_keys_view_t = types.abi.keys_view_t;
const usearch_matches_view_t = types.abi.matches_view_t;
const usearch_nvme_options_t = types.abi.nvme_options_t;
const usearch_progress_t = types.abi.progress_t;
const usearch_ram_options_t = types.abi.ram_options_t;
const usearch_rows_view_t = types.abi.rows_view_t;
const usearch_tape_view_t = basic.usearch_tape_view_t;
const usearch_vram_options_t = types.abi.vram_options_t;

extern fn usearch_sequence_metric_name(metric: SequenceMetric) [*:0]const u8;
extern fn usearch_sequence_metric_from_name(name: BytesView, metric: *SequenceMetric) Status;

/// The sequence kinds: edit distances over bytes or code points, a weighted alignment, and the
/// overlap of fixed-width windows. Mirrors `usearch_sequence_metric_t`.
///
/// Non-exhaustive, so a kind a newer engine reports arrives intact rather than coerced.
pub const SequenceMetric = enum(c_int) {
    /// No kind named - an unset field, refused by every constructor.
    unknown = 0,
    /// Edits between two byte strings, each insertion, deletion or substitution counting one.
    levenshtein_bytes = 1,
    /// Edits between two UTF-8 strings, counted over code points rather than bytes.
    levenshtein_utf8 = 2,
    /// The global alignment under `SequenceWeights`, every unaligned symbol charged the gap cost.
    needleman_wunsch = 3,
    /// The share of the larger window set the other side misses, over the frozen window widths.
    overlap = 4,
    _,

    /// The canonical spelling, as `.levenshtein_bytes` spells "levenshtein".
    /// Calls `usearch_sequence_metric_name`.
    pub fn name(self: SequenceMetric) [:0]const u8 {
        return std.mem.span(usearch_sequence_metric_name(self));
    }

    /// Reads a spelling back, or null when no kind spells it.
    /// Calls `usearch_sequence_metric_from_name`.
    pub fn fromName(text: []const u8) ?SequenceMetric {
        var metric: SequenceMetric = undefined;
        return if (usearch_sequence_metric_from_name(.of(text), &metric) == .success) metric else null;
    }
};

/// The weights `.needleman_wunsch` scores with; the Levenshtein kinds refuse any.
/// Mirrors `usearch_sequence_weights_t`.
pub const SequenceWeights = extern struct {
    /// Signed scores of every byte against every byte, row-major.
    substitution_matrix: *const [256][256]i8,
    /// Cost of one unaligned symbol.
    gap: i8 = 0,
};

/// The kind and graph an empty index freezes - what `SequenceIndex.initRam` and `initVram` read
/// beside their placement. Binding-only.
pub const SequenceOptions = struct {
    /// The kind every pair is scored under.
    metric: SequenceMetric = .levenshtein_bytes,
    /// The weights an alignment kind needs; null for the Levenshtein kinds, which refuse any.
    weights: ?*const SequenceWeights = null,
    /// Window widths in bytes for the overlap kind, which needs at least one; every other kind
    /// refuses any.
    window_widths: []const usize = &.{},
    /// The optional connections-per-node bound; zero derives.
    connectivity: usize = 0,
    /// Upper bound on entries sharing one key; zero is unbounded.
    values_per_key: usize = 0,
};

/// The kind and graph a checkpoint froze - everything a reader learns without opening it.
/// Mirrors `usearch_sequence_metadata_t`.
///
/// The weights are absent because a checkpoint records costs rather than the matrix, so every
/// load and view of an alignment checkpoint names them again.
pub const SequenceMetadata = extern struct {
    /// The kind the checkpoint scores with.
    metric: SequenceMetric = .unknown,
    /// Connections per node the graph was built with.
    connectivity: usize = 0,
};

/// A live index's state, reported by `SequenceIndex.describe`. Each counter is read once, so under
/// concurrent insertions they may come from slightly different moments.
/// Mirrors `usearch_sequence_index_description_t`.
pub const SequenceIndexDescription = extern struct {
    /// The frozen metric and connectivity.
    metadata: SequenceMetadata = .{},
    /// Stored entries.
    size: usize = 0,
    /// Entries hosted without growing again.
    capacity: usize = 0,
    /// Bytes the handle holds, payload storage included; traversal scratch is the executor's.
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
    pub fn cpuClassName(self: SequenceIndexDescription) [:0]const u8 {
        return std.mem.span(types.abi.cpu_class_name(self.cpu_class));
    }

    /// The GPU class's spelling, "unknown" for a host index. Calls `usearch_gpu_class_name`.
    pub fn gpuClassName(self: SequenceIndexDescription) [:0]const u8 {
        return std.mem.span(types.abi.gpu_class_name(self.gpu_class));
    }

    /// The load path's spelling: "none", "host-copy" or "cufile". Calls `usearch_load_path_name`.
    pub fn loadPathName(self: SequenceIndexDescription) [:0]const u8 {
        return std.mem.span(types.abi.load_path_name(self.load_path));
    }
};

/// What one key holds, enough to size a buffer before retrieving any of it.
/// Mirrors `usearch_sequence_key_stats_t`.
pub const SequenceKeyStats = extern struct {
    /// Entries stored under the key; zero means the index holds none.
    entries: usize = 0,
    /// Bytes summed across those entries.
    total_bytes: usize = 0,
    /// Bytes of the longest single entry.
    largest_bytes: usize = 0,
};

/// A refine a cancel left live, which the next `refine` naming it continues and
/// `refinePlanFinish` closes. Mirrors `usearch_sequence_refine_plan_t`; the default holds none.
pub const SequenceRefinePlan = extern struct {
    handle: ?*anyopaque = null,
};

/// A reshard a cancel left live, which the next `reshard` naming it continues and
/// `reshardPlanFinish` closes. Mirrors `usearch_sequence_reshard_plan_t`; the default holds none.
pub const SequenceReshardPlan = extern struct {
    handle: ?*anyopaque = null,
};

/// Optional settings of `SequenceIndex.loadRam`.
pub const SequenceLoadRamConfig = struct {
    /// The executor the read fans out over; null reads on the calling thread.
    executor: ?Executor = null,
    /// The memory domains to place the parts on; empty derives every node the process may use.
    memory_domains: []const i32 = &.{},
    /// Upper bound on entries sharing one key, which no checkpoint records; zero is unbounded.
    values_per_key: usize = 0,
    /// The weights an alignment checkpoint was built with; null for the Levenshtein kinds.
    weights: ?*const SequenceWeights = null,
    /// Called with shards placed; a stopped load creates no index.
    progress: ProgressCallback = .{},
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    progress_steps: usize = 0,
    /// Where the call's final outcome, its static message included, is copied; null copies nothing.
    diagnostics: ?*Outcome = null,
};

/// Optional settings of `SequenceIndex.loadVram`.
pub const SequenceLoadVramConfig = struct {
    /// The executor the upload runs on; null runs on the calling thread and the default stream.
    executor: ?Executor = null,
    /// The devices to place the parts on; empty derives one free device.
    devices: []const i32 = &.{},
    /// Upper bound on entries sharing one key, which no checkpoint records; zero is unbounded.
    values_per_key: usize = 0,
    /// The weights an alignment checkpoint was built with; null for the Levenshtein kinds.
    weights: ?*const SequenceWeights = null,
    /// Called with shards placed; a stopped load creates no index.
    progress: ProgressCallback = .{},
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    progress_steps: usize = 0,
    /// Where the call's final outcome, its static message included, is copied; null copies nothing.
    diagnostics: ?*Outcome = null,
};

/// Optional settings of `SequenceIndex.loadRamBuffer`.
pub const SequenceLoadRamBufferConfig = struct {
    /// The executor the copy fans out over; null copies on the calling thread.
    executor: ?Executor = null,
    /// The memory domains to place shards on; the monolith lands whole on the first.
    memory_domains: []const i32 = &.{},
    /// Upper bound on entries sharing one key, which no checkpoint records; zero is unbounded.
    values_per_key: usize = 0,
    /// The weights an alignment checkpoint was built with; null for the Levenshtein kinds.
    weights: ?*const SequenceWeights = null,
    /// Called with bytes placed; a stopped load creates no index.
    progress: ProgressCallback = .{},
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    progress_steps: usize = 0,
    /// Where the call's final outcome, its static message included, is copied; null copies nothing.
    diagnostics: ?*Outcome = null,
};

/// Optional settings of `SequenceIndex.loadVramBuffer`.
pub const SequenceLoadVramBufferConfig = struct {
    /// The executor the upload runs on; null runs on the calling thread and the default stream.
    executor: ?Executor = null,
    /// The devices to place shards on; the monolith lands whole on the first.
    devices: []const i32 = &.{},
    /// Upper bound on entries sharing one key, which no checkpoint records; zero is unbounded.
    values_per_key: usize = 0,
    /// The weights an alignment checkpoint was built with; null for the Levenshtein kinds.
    weights: ?*const SequenceWeights = null,
    /// Called with bytes placed; a stopped load creates no index.
    progress: ProgressCallback = .{},
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    progress_steps: usize = 0,
    /// Where the call's final outcome, its static message included, is copied; null copies nothing.
    diagnostics: ?*Outcome = null,
};

/// Optional settings of `SequenceIndex.viewNvme`.
pub const SequenceViewNvmeConfig = struct {
    /// The NVMe executor whose rings read the checkpoint's headers; null reads synchronously.
    executor: ?Executor = null,
    /// The devices the parts are served from; empty serves each where its manifest points.
    paths: []const BytesView = &.{},
    /// The weights an alignment checkpoint was built with; null for the Levenshtein kinds.
    weights: ?*const SequenceWeights = null,
    /// Where the call's final outcome, its static message included, is copied; null copies nothing.
    diagnostics: ?*Outcome = null,
};

/// Optional settings of `SequenceIndex.insert`.
pub const SequenceInsertConfig = struct {
    /// The breadth of the candidates the build walk explores; zero derives.
    expansion_factor: usize = 0,
    /// Called with entries admitted; it reports but cannot cancel, as every entry is claimed first.
    progress: ProgressCallback = .{},
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    progress_steps: usize = 0,
    stats: ?*DenseTraversalStats = null,
};

/// Optional settings of `SequenceIndex.refine`.
pub const SequenceRefineConfig = struct {
    /// The breadth of the candidates the re-selection walk explores; zero derives.
    expansion_factor: usize = 0,
    /// Bytes a loaded NVMe handle holds for the one part it rewrites; zero derives, and RAM and
    /// VRAM handles never read it.
    memory_limit: usize = 0,
    /// Called at every barrier of every phase with that phase's units; a stop answers
    /// `Error.Cancelled`.
    progress: ProgressCallback = .{},
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    progress_steps: usize = 0,
    stats: ?*DenseTraversalStats = null,
    /// Null runs one round to its end; an empty plan opens one a cancel leaves live, and a live
    /// one continues.
    plan: ?*SequenceRefinePlan = null,
    /// Where the call stopped, `.done` once its round ended; null skips it.
    phase: ?*PhaseProgress = null,
};

/// Optional settings of `SequenceIndex.search`.
pub const SequenceSearchConfig = struct {
    /// The breadth of the candidates the walk explores; zero derives.
    expansion_factor: usize = 0,
    /// A walk, or an exact scan of every entry on RAM and NVMe; VRAM answers `Error.Unsupported`.
    traversal: Traversal = .walk,
    /// Admits only the keys it approves; the rest still route the walk. Refused on VRAM.
    predicate: Predicate = .{},
    /// Called with queries resolved; a stopped search answers `Error.Cancelled` and no matches.
    progress: ProgressCallback = .{},
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    progress_steps: usize = 0,
    stats: ?*DenseTraversalStats = null,
};

/// Optional settings of `SequenceIndex.reshard`.
pub const SequenceReshardConfig = struct {
    /// The breadth of the candidates of the closing rounds; zero derives.
    expansion_factor: usize = 0,
    /// Bytes a loaded NVMe target holds for the one part it rewrites; zero derives, and RAM and
    /// VRAM targets never read it.
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
    plan: ?*SequenceReshardPlan = null,
    /// Where the call stopped, `.done` once its rounds ended; null skips it.
    phase: ?*PhaseProgress = null,
};

const usearch_sequence_init_ram_t = extern struct {
    index: *?*anyopaque,
    outcome: *Outcome,
    metric: SequenceMetric,
    weights: ?*const SequenceWeights,
    window_widths: ?[*]const usize,
    window_widths_count: usize,
    connectivity: usize,
    values_per_key: usize,
    ram: usearch_ram_options_t,
};

const usearch_sequence_init_vram_t = extern struct {
    index: *?*anyopaque,
    outcome: *Outcome,
    metric: SequenceMetric,
    weights: ?*const SequenceWeights,
    window_widths: ?[*]const usize,
    window_widths_count: usize,
    connectivity: usize,
    values_per_key: usize,
    vram: usearch_vram_options_t,
};

const usearch_sequence_load_ram_t = extern struct {
    index: *?*anyopaque,
    executor: ?*anyopaque,
    outcome: *Outcome,
    completion: Completion,
    path: BytesView,
    ram: usearch_ram_options_t,
    values_per_key: usize,
    weights: ?*const SequenceWeights,
    progress: usearch_progress_t,
};

const usearch_sequence_load_ram_buffer_t = extern struct {
    index: *?*anyopaque,
    executor: ?*anyopaque,
    outcome: *Outcome,
    completion: Completion,
    buffer: BytesView,
    ram: usearch_ram_options_t,
    values_per_key: usize,
    weights: ?*const SequenceWeights,
    progress: usearch_progress_t,
};

const usearch_sequence_load_vram_t = extern struct {
    index: *?*anyopaque,
    executor: ?*anyopaque,
    outcome: *Outcome,
    completion: Completion,
    path: BytesView,
    vram: usearch_vram_options_t,
    values_per_key: usize,
    weights: ?*const SequenceWeights,
    progress: usearch_progress_t,
};

const usearch_sequence_load_vram_buffer_t = extern struct {
    index: *?*anyopaque,
    executor: ?*anyopaque,
    outcome: *Outcome,
    completion: Completion,
    buffer: BytesView,
    vram: usearch_vram_options_t,
    values_per_key: usize,
    weights: ?*const SequenceWeights,
    progress: usearch_progress_t,
};

const usearch_sequence_view_ram_t = extern struct {
    index: *?*anyopaque,
    outcome: *Outcome,
    path: BytesView,
    weights: ?*const SequenceWeights,
    ram: usearch_ram_options_t = .{},
};

const usearch_sequence_view_nvme_t = extern struct {
    index: *?*anyopaque,
    executor: ?*anyopaque,
    outcome: *Outcome,
    completion: Completion,
    path: BytesView,
    nvme: usearch_nvme_options_t,
    weights: ?*const SequenceWeights,
};

const usearch_sequence_reserve_t = extern struct {
    index: ?*anyopaque,
    executor: ?*anyopaque,
    outcome: *Outcome,
    completion: Completion,
    entries: usize,
    payload_bytes: usize,
};

const usearch_sequence_clear_t = extern struct {
    index: ?*anyopaque,
    executor: ?*anyopaque,
    outcome: *Outcome,
    completion: Completion,
};

const usearch_sequence_insert_t = extern struct {
    index: ?*anyopaque,
    executor: ?*anyopaque,
    outcome: *Outcome,
    completion: Completion,
    count: usize,
    keys: usearch_keys_view_t,
    sequences: usearch_tape_view_t,
    expansion_factor: usize,
    progress: usearch_progress_t,
    stats: ?*DenseTraversalStats,
};

const usearch_refine_options_t = extern struct {
    expansion_factor: usize,
    memory_limit: usize,
};

const usearch_sequence_refine_t = extern struct {
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

const usearch_sequence_refine_plan_finish_t = extern struct {
    index: ?*anyopaque,
    executor: ?*anyopaque,
    outcome: *Outcome,
    completion: Completion,
    plan: ?*anyopaque,
    report: ?*DenseRefineReport,
};

const usearch_sequence_health_t = extern struct {
    index: ?*anyopaque,
    executor: ?*anyopaque,
    outcome: *Outcome,
    completion: Completion,
    reachability: *DenseReachability,
};

const usearch_sequence_search_t = extern struct {
    index: ?*anyopaque,
    executor: ?*anyopaque,
    outcome: *Outcome,
    completion: Completion,
    count: usize,
    queries: usearch_tape_view_t,
    matches: *usearch_matches_view_t,
    arena: ?*anyopaque,
    wanted: usize,
    expansion_factor: usize,
    traversal: Traversal,
    predicate: Predicate,
    progress: usearch_progress_t,
    stats: ?*DenseTraversalStats,
};

const usearch_sequence_count_t = extern struct {
    index: ?*anyopaque,
    executor: ?*anyopaque,
    outcome: *Outcome,
    completion: Completion,
    key: Key,
    key_stats: *SequenceKeyStats,
};

const usearch_sequence_get_t = extern struct {
    index: ?*anyopaque,
    executor: ?*anyopaque,
    outcome: *Outcome,
    completion: Completion,
    key: Key,
    sequences: *usearch_tape_view_t,
    written: ?*usize,
    arena: ?*anyopaque,
};

const usearch_sequence_reshard_t = extern struct {
    sources_count: usize,
    sources: [*]const ?*anyopaque,
    targets_count: usize,
    targets: [*]const ?*anyopaque,
    executor: ?*anyopaque,
    outcome: *Outcome,
    completion: Completion,
    report: ?*DenseRefineReport,
    refine: usearch_refine_options_t,
    seed: u64,
    progress: usearch_progress_t,
    stats: ?*DenseTraversalStats,
    plan: ?*?*anyopaque,
    phase: ?*PhaseProgress,
};

const usearch_sequence_reshard_plan_finish_t = extern struct {
    executor: ?*anyopaque,
    outcome: *Outcome,
    completion: Completion,
    plan: ?*anyopaque,
    report: ?*DenseRefineReport,
};

const usearch_sequence_cursor_init_t = extern struct {
    index: ?*anyopaque,
    outcome: *Outcome,
    starts_count: usize,
    starts: ?[*]const Key,
    cursor: *?*anyopaque,
};

const usearch_sequence_cursor_next_t = extern struct {
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

const usearch_sequence_sample_t = extern struct {
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

const usearch_sequence_save_t = extern struct {
    index: ?*anyopaque,
    executor: ?*anyopaque,
    outcome: *Outcome,
    completion: Completion,
    path: BytesView,
    progress: usearch_progress_t,
};

const usearch_sequence_save_buffer_t = extern struct {
    index: ?*anyopaque,
    executor: ?*anyopaque,
    outcome: *Outcome,
    completion: Completion,
    monolith: *BytesView,
    arena: ?*anyopaque,
    progress: usearch_progress_t,
};

const usearch_sequence_metadata_read_t = extern struct {
    executor: ?*anyopaque,
    outcome: *Outcome,
    completion: Completion,
    path: BytesView,
    metadata: *SequenceMetadata,
};

const usearch_sequence_metadata_read_buffer_t = extern struct {
    executor: ?*anyopaque,
    outcome: *Outcome,
    completion: Completion,
    buffer: BytesView,
    metadata: *SequenceMetadata,
};

const usearch_sequence_describe_t = extern struct {
    index: ?*anyopaque,
    executor: ?*anyopaque,
    outcome: *Outcome,
    completion: Completion,
    description: *SequenceIndexDescription,
};

extern fn usearch_sequence_init_ram(arguments: *usearch_sequence_init_ram_t) void;
extern fn usearch_sequence_init_vram(arguments: *usearch_sequence_init_vram_t) void;
extern fn usearch_sequence_load_ram(arguments: *usearch_sequence_load_ram_t) void;
extern fn usearch_sequence_load_ram_buffer(arguments: *usearch_sequence_load_ram_buffer_t) void;
extern fn usearch_sequence_load_vram(arguments: *usearch_sequence_load_vram_t) void;
extern fn usearch_sequence_load_vram_buffer(arguments: *usearch_sequence_load_vram_buffer_t) void;
extern fn usearch_sequence_view_ram(arguments: *usearch_sequence_view_ram_t) void;
extern fn usearch_sequence_view_nvme(arguments: *usearch_sequence_view_nvme_t) void;
extern fn usearch_sequence_free(index: ?*anyopaque) void;
extern fn usearch_sequence_reserve(arguments: *usearch_sequence_reserve_t) void;
extern fn usearch_sequence_clear(arguments: *usearch_sequence_clear_t) void;
extern fn usearch_sequence_insert(arguments: *usearch_sequence_insert_t) void;
extern fn usearch_sequence_refine(arguments: *usearch_sequence_refine_t) void;
extern fn usearch_sequence_refine_plan_finish(arguments: *usearch_sequence_refine_plan_finish_t) void;
extern fn usearch_sequence_health(arguments: *usearch_sequence_health_t) void;
extern fn usearch_sequence_search(arguments: *usearch_sequence_search_t) void;
extern fn usearch_sequence_count(arguments: *usearch_sequence_count_t) void;
extern fn usearch_sequence_get(arguments: *usearch_sequence_get_t) void;
extern fn usearch_sequence_reshard(arguments: *usearch_sequence_reshard_t) void;
extern fn usearch_sequence_reshard_plan_finish(arguments: *usearch_sequence_reshard_plan_finish_t) void;
extern fn usearch_sequence_sample(arguments: *usearch_sequence_sample_t) void;
extern fn usearch_sequence_cursor_init(arguments: *usearch_sequence_cursor_init_t) void;
extern fn usearch_sequence_cursor_next(arguments: *usearch_sequence_cursor_next_t) void;
extern fn usearch_sequence_cursor_free(cursor: ?*anyopaque) void;
extern fn usearch_sequence_save(arguments: *usearch_sequence_save_t) void;
extern fn usearch_sequence_save_buffer(arguments: *usearch_sequence_save_buffer_t) void;
extern fn usearch_sequence_metadata_read(arguments: *usearch_sequence_metadata_read_t) void;
extern fn usearch_sequence_metadata_read_buffer(arguments: *usearch_sequence_metadata_read_buffer_t) void;
extern fn usearch_sequence_describe(arguments: *usearch_sequence_describe_t) void;

/// Sequence Search Index over variable-length byte strings.
///
/// A handle is shareable across threads: searches never write to the graph, and insertions
/// synchronize through per-node locks and publish in order. It is the `Executor` that orders,
/// never the index.
///
/// ```zig
/// const usearch = @import("usearch");
///
/// var index = try usearch.SequenceIndex.initRam(.{ .metric = .levenshtein_utf8 }, &.{});
/// defer index.deinit();
///
/// const keys = [_]usearch.Key{ .fromInteger(1), .fromInteger(2) };
/// const offsets = [_]u64{ 0, 6, 13 };
/// try index.insert(&keys, try .of(u8, u64, "kittensitting", &offsets), .{});
///
/// var arena = try usearch.Arena.init(null);
/// defer arena.deinit();
/// const query = [_]u64{ 0, 6 };
/// const matches = try index.search(try .of(u8, u64, "sittin", &query), 1, &arena, .{});
/// const nearest = matches.query(0).keys[0];
/// ```
pub const SequenceIndex = extern struct {
    /// The opaque `usearch_sequence_index_t` the ABI hands back.
    handle: ?*anyopaque,

    /// Every verb of the index, blocking on an executor, refused while another call holds it.
    pub const On = Receiver(.blocking, void);

    /// Every verb of the index, submitted to an executor into a `Deferred(Result)` the caller
    /// holds; only the verbs answering `Result` compile.
    pub fn DeferredOn(comptime Result: type) type {
        return Receiver(.deferred, Result);
    }

    /// Runs every verb on `executor`, whose kind must match the index's residency. Binding-only.
    pub fn on(self: SequenceIndex, executor: Executor) On {
        return .{ .index = self.handle, .executor = executor.handle, .into = {}, .diagnostics = null };
    }

    /// Submits a verb to `executor`, holding it in `call`, a `*Deferred(Result)` the caller places
    /// anywhere, which stays put until its `deinit`. Binding-only.
    pub fn deferred(self: SequenceIndex, executor: Executor, call: anytype) DeferredOn(@FieldType(@TypeOf(call.*), "result")) {
        return .{ .index = self.handle, .executor = executor.handle, .into = call, .diagnostics = {} };
    }

    /// Runs every verb on the calling thread, copying each call's final outcome into `outcome`.
    /// Binding-only.
    pub fn diagnosed(self: SequenceIndex, outcome: *Outcome) On {
        return self.here().diagnosed(outcome);
    }

    /// The receiver the index's own verbs run through: no executor, the calling thread.
    fn here(self: SequenceIndex) On {
        return .{ .index = self.handle, .executor = null, .into = {}, .diagnostics = null };
    }

    /// Creates an empty, mutable index in host memory, one shard per memory domain; empty derives
    /// every node the process may allocate on. Calls `usearch_sequence_init_ram`.
    pub fn initRam(options: SequenceOptions, memory_domains: []const i32) Error!SequenceIndex {
        var handle: ?*anyopaque = null;
        var outcome: Outcome = .{};
        var arguments: usearch_sequence_init_ram_t = .{
            .index = &handle,
            .outcome = &outcome,
            .metric = options.metric,
            .weights = options.weights,
            .window_widths = options.window_widths.ptr,
            .window_widths_count = options.window_widths.len,
            .connectivity = options.connectivity,
            .values_per_key = options.values_per_key,
            .ram = .of(memory_domains),
        };
        usearch_sequence_init_ram(&arguments);
        return adopt(handle, outcome, null);
    }

    /// Creates an empty, mutable index in device memory, one shard per device and never a replica;
    /// empty derives one free device. Calls `usearch_sequence_init_vram`.
    pub fn initVram(options: SequenceOptions, devices: []const i32) Error!SequenceIndex {
        var handle: ?*anyopaque = null;
        var outcome: Outcome = .{};
        var arguments: usearch_sequence_init_vram_t = .{
            .index = &handle,
            .outcome = &outcome,
            .metric = options.metric,
            .weights = options.weights,
            .window_widths = options.window_widths.ptr,
            .window_widths_count = options.window_widths.len,
            .connectivity = options.connectivity,
            .values_per_key = options.values_per_key,
            .vram = .of(devices),
        };
        usearch_sequence_init_vram(&arguments);
        return adopt(handle, outcome, null);
    }

    /// Copies a checkpoint into a new, mutable index in host memory, shard by shard.
    ///
    /// Point `path` at the `.safegraphs.json` manifest, never at one of the parts it enumerates.
    /// The kind comes from the checkpoint's own headers. Calls `usearch_sequence_load_ram`.
    pub fn loadRam(path: []const u8, config: SequenceLoadRamConfig) Error!SequenceIndex {
        var handle: ?*anyopaque = null;
        var outcome: Outcome = .{};
        var arguments: usearch_sequence_load_ram_t = .{
            .index = &handle,
            .executor = handleOf(config.executor),
            .outcome = &outcome,
            .completion = .blocking,
            .path = .of(path),
            .ram = .of(config.memory_domains),
            .values_per_key = config.values_per_key,
            .weights = config.weights,
            .progress = .of(config.progress, config.progress_steps),
        };
        usearch_sequence_load_ram(&arguments);
        return adopt(handle, outcome, config.diagnostics);
    }

    /// Copies a checkpoint into a new, mutable index in device memory, with the placement rules of
    /// `loadRam`. Calls `usearch_sequence_load_vram`.
    pub fn loadVram(path: []const u8, config: SequenceLoadVramConfig) Error!SequenceIndex {
        var handle: ?*anyopaque = null;
        var outcome: Outcome = .{};
        var arguments: usearch_sequence_load_vram_t = .{
            .index = &handle,
            .executor = handleOf(config.executor),
            .outcome = &outcome,
            .completion = .blocking,
            .path = .of(path),
            .vram = .of(config.devices),
            .values_per_key = config.values_per_key,
            .weights = config.weights,
            .progress = .of(config.progress, config.progress_steps),
        };
        usearch_sequence_load_vram(&arguments);
        return adopt(handle, outcome, config.diagnostics);
    }

    /// Copies a monolith from a memory buffer into a new index in host memory, as `loadRam` does a
    /// path. Calls `usearch_sequence_load_ram_buffer`.
    pub fn loadRamBuffer(buffer: []const u8, config: SequenceLoadRamBufferConfig) Error!SequenceIndex {
        var handle: ?*anyopaque = null;
        var outcome: Outcome = .{};
        var arguments: usearch_sequence_load_ram_buffer_t = .{
            .index = &handle,
            .executor = handleOf(config.executor),
            .outcome = &outcome,
            .completion = .blocking,
            .buffer = .of(buffer),
            .ram = .of(config.memory_domains),
            .values_per_key = config.values_per_key,
            .weights = config.weights,
            .progress = .of(config.progress, config.progress_steps),
        };
        usearch_sequence_load_ram_buffer(&arguments);
        return adopt(handle, outcome, config.diagnostics);
    }

    /// Copies a monolith from a host or device buffer into a new index in device memory.
    /// Calls `usearch_sequence_load_vram_buffer`.
    pub fn loadVramBuffer(buffer: []const u8, config: SequenceLoadVramBufferConfig) Error!SequenceIndex {
        var handle: ?*anyopaque = null;
        var outcome: Outcome = .{};
        var arguments: usearch_sequence_load_vram_buffer_t = .{
            .index = &handle,
            .executor = handleOf(config.executor),
            .outcome = &outcome,
            .completion = .blocking,
            .buffer = .of(buffer),
            .vram = .of(config.devices),
            .values_per_key = config.values_per_key,
            .weights = config.weights,
            .progress = .of(config.progress, config.progress_steps),
        };
        usearch_sequence_load_vram_buffer(&arguments);
        return adopt(handle, outcome, config.diagnostics);
    }

    /// Serves a checkpoint in place through a memory mapping, as a read-only index whose mutating
    /// verbs report `Error.ReadOnly`; an alignment checkpoint names its `weights` again.
    /// Calls `usearch_sequence_view_ram`.
    pub fn viewRam(path: []const u8, weights: ?*const SequenceWeights) Error!SequenceIndex {
        var handle: ?*anyopaque = null;
        var outcome: Outcome = .{};
        var arguments: usearch_sequence_view_ram_t = .{ .index = &handle, .outcome = &outcome, .path = .of(path), .weights = weights };
        usearch_sequence_view_ram(&arguments);
        return adopt(handle, outcome, null);
    }

    /// Serves a checkpoint in place from NVMe drives, read-only and sized past RAM.
    /// Calls `usearch_sequence_view_nvme`.
    pub fn viewNvme(path: []const u8, config: SequenceViewNvmeConfig) Error!SequenceIndex {
        var handle: ?*anyopaque = null;
        var outcome: Outcome = .{};
        var arguments: usearch_sequence_view_nvme_t = .{
            .index = &handle,
            .executor = handleOf(config.executor),
            .outcome = &outcome,
            .completion = .blocking,
            .path = .of(path),
            .nvme = .of(config.paths),
            .weights = config.weights,
        };
        usearch_sequence_view_nvme(&arguments);
        return adopt(handle, outcome, config.diagnostics);
    }

    /// Reads a checkpoint's kind and graph, without loading it.
    /// Calls `usearch_sequence_metadata_read`.
    pub fn metadataRead(path: []const u8, config: DenseMetadataReadConfig) Error!SequenceMetadata {
        var found: SequenceMetadata = .{};
        var outcome: Outcome = .{};
        var arguments: usearch_sequence_metadata_read_t = .{
            .executor = handleOf(config.executor),
            .outcome = &outcome,
            .completion = .blocking,
            .path = .of(path),
            .metadata = &found,
        };
        usearch_sequence_metadata_read(&arguments);
        try outcome.checkInto(config.diagnostics);
        return found;
    }

    /// Reads the kind and graph out of a serialized monolith buffer, without loading it.
    /// Calls `usearch_sequence_metadata_read_buffer`.
    pub fn metadataReadBuffer(buffer: []const u8, config: DenseMetadataReadBufferConfig) Error!SequenceMetadata {
        var found: SequenceMetadata = .{};
        var outcome: Outcome = .{};
        var arguments: usearch_sequence_metadata_read_buffer_t = .{
            .executor = handleOf(config.executor),
            .outcome = &outcome,
            .completion = .blocking,
            .buffer = .of(buffer),
            .metadata = &found,
        };
        usearch_sequence_metadata_read_buffer(&arguments);
        try outcome.checkInto(config.diagnostics);
        return found;
    }

    fn adopt(handle: ?*anyopaque, outcome: Outcome, diagnostics: ?*Outcome) Error!SequenceIndex {
        errdefer usearch_sequence_free(handle);
        try outcome.checkInto(diagnostics);
        return .{ .handle = handle };
    }

    /// Frees the resources associated with the index, waiting for every deferred call naming it;
    /// the handle is invalid afterwards. Calls `usearch_sequence_free`.
    pub fn deinit(self: *SequenceIndex) void {
        usearch_sequence_free(self.handle);
        self.* = undefined;
    }

    /// Reserves room for a total number of entries, the current size included, and for
    /// `payload_bytes` of sequences in all; zero lets the payload grow lazily.
    /// Calls `usearch_sequence_reserve`.
    pub fn reserve(self: SequenceIndex, entries: usize, payload_bytes: usize) Error!void {
        return self.here().reserve(entries, payload_bytes);
    }

    /// Erases all entries, keeping the reserved capacity and frozen kind. Never runs concurrently
    /// with any other verb on the same handle. Calls `usearch_sequence_clear`.
    pub fn clear(self: SequenceIndex) Error!void {
        return self.here().clear();
    }

    /// Reports the frozen metadata and the current counters in one call.
    /// Calls `usearch_sequence_describe`.
    pub fn describe(self: SequenceIndex) Error!SequenceIndexDescription {
        return self.here().describe();
    }

    /// What `key` holds: its entries, and their byte lengths in total and at most.
    /// Calls `usearch_sequence_count`.
    pub fn count(self: SequenceIndex, key: Key) Error!SequenceKeyStats {
        return self.here().count(key);
    }

    /// Whether the index holds an entry under `key`. Binding-only; calls `usearch_sequence_count`.
    pub fn contains(self: SequenceIndex, key: Key) Error!bool {
        return self.here().contains(key);
    }

    /// Every sequence stored under `key`, one tape entry per value in insertion order, written in
    /// `arena`; an absent key yields no entries. Calls `usearch_sequence_get`.
    pub fn get(self: SequenceIndex, key: Key, arena: *Arena) Error!TapeEntries(u8) {
        return self.here().get(key, arena);
    }

    /// Adds a tape of sequences under their keys - the native batch operation, one key per entry.
    ///
    /// The batch is all-or-nothing: every fallible step runs before the first entry identity is
    /// claimed, so a handler reports progress but cannot stop it. Overlaps safely with concurrent
    /// searches on the same handle. Calls `usearch_sequence_insert`.
    pub fn insert(self: SequenceIndex, keys: []const Key, sequences: TapeView, config: SequenceInsertConfig) Error!void {
        return self.here().insert(keys, sequences, config);
    }

    /// Runs one round over the finished graph: a health check, a heal of its islands, and a gather
    /// and restitch of every neighborhood, closed by a second health check, reporting what changed.
    /// A stop at a barrier answers `Error.Cancelled`, and a `plan` in the config keeps the round to
    /// continue later. Calls `usearch_sequence_refine`.
    pub fn refine(self: SequenceIndex, config: SequenceRefineConfig) Error!DenseRefineReport {
        return self.here().refine(config);
    }

    /// Closes a live refine plan, reporting what the whole plan did; the plan is freed and emptied
    /// whatever the call answers. Calls `usearch_sequence_refine_plan_finish`.
    pub fn refinePlanFinish(self: SequenceIndex, plan: *SequenceRefinePlan) Error!DenseRefineReport {
        return self.here().refinePlanFinish(plan);
    }

    /// Measures how much of the graph a search reaches from the root; the entries it misses stay
    /// unfound until a `refine` re-links them. Calls `usearch_sequence_health`.
    pub fn health(self: SequenceIndex) Error!DenseReachability {
        return self.here().health();
    }

    /// Resolves a tape of queries into up to `wanted` matches apiece, written in `arena` - the
    /// native batch operation.
    ///
    /// Never writes to the graph, so any number of these may overlap with concurrent insertions.
    /// A zero `wanted` walks nothing and lands only the counts, every one zero.
    /// Calls `usearch_sequence_search`.
    pub fn search(self: SequenceIndex, queries: TapeView, wanted: usize, arena: *Arena, config: SequenceSearchConfig) Error!Matches {
        return self.here().search(queries, wanted, arena, config);
    }

    /// Draws up to `wanted` keys around `config.focus`, or across the whole index, each hanging
    /// from a shown or earlier draw and placed in three dimensions beside the shown ones, written
    /// in `arena`. Calls `usearch_sequence_sample`.
    pub fn sample(self: SequenceIndex, wanted: usize, arena: *Arena, config: DenseSampleConfig) Error!DenseDraws {
        return self.here().sample(wanted, arena, config);
    }

    /// Copies up to `wanted` more entries out of `cursor` into `arena` - their keys, and the keys
    /// each links to when `links` asks - answering an empty batch once the walk is exhausted. The
    /// sequences themselves come from `get`. Calls `usearch_sequence_cursor_next`.
    pub fn cursorNext(self: SequenceIndex, cursor: SequenceCursor, wanted: usize, links: bool, arena: *Arena) Error!DenseCursorBatch {
        return self.here().cursorNext(cursor, wanted, links, arena);
    }

    /// Saves the index under `path` in the layout its extension picks - a `.safegraphs` monolith,
    /// or a `.safegraphs.json` manifest beside its shards.
    ///
    /// Demands a mutation-free window; concurrent readers keep searching throughout. A failed
    /// save leaves a file without its header, which every reader rejects, and that leftover is
    /// the caller's to remove. Calls `usearch_sequence_save`.
    pub fn save(self: SequenceIndex, path: []const u8, config: DenseSaveConfig) Error!void {
        return self.here().save(path, config);
    }

    /// Saves the index as one monolith - the same bytes `save` writes to a `.safegraphs` path -
    /// written in `arena`. Calls `usearch_sequence_save_buffer`.
    pub fn saveBuffer(self: SequenceIndex, arena: *Arena, config: DenseSaveBufferConfig) Error!Monolith {
        return self.here().saveBuffer(arena, config);
    }

    /// Moves the entries of `sources` into `targets` on the calling thread, keeping their links
    /// rather than re-inserting the sequences, and runs a round on each target that took any.
    ///
    /// Several sources into one target, itself among them, gather into it; one source into several
    /// targets cuts it into graph-local parts. A source the targets do not name ends empty and
    /// live, and every handle stays the caller's. Calls `usearch_sequence_reshard`.
    pub fn reshard(sources: []const SequenceIndex, targets: []const SequenceIndex, config: SequenceReshardConfig) Error!DenseRefineReport {
        return (SequenceIndex{ .handle = null }).here().reshard(sources, targets, config);
    }

    /// Closes a live reshard plan on the calling thread, reporting what the whole plan did; the
    /// plan is freed and emptied whatever the call answers.
    /// Calls `usearch_sequence_reshard_plan_finish`.
    pub fn reshardPlanFinish(plan: *SequenceReshardPlan) Error!DenseRefineReport {
        return (SequenceIndex{ .handle = null }).here().reshardPlanFinish(plan);
    }
};

fn handleOf(executor: ?Executor) ?*anyopaque {
    return if (executor) |chosen| chosen.handle else null;
}

/// The verbs of `SequenceIndex` once more, for a receiver that completes them as `completion`
/// says: blocking answers the result, deferred submits into the caller's `Deferred(Into)`. Each
/// verb is written once, filling its C structure with the result as its output slot.
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
            if (self.diagnostics) |lent| lent.* = held.outcome;
            try held.outcome.check();
            return held.result;
        }

        /// Fails a call the binding refused before it crossed, as `refuseInto` does.
        fn refused(self: Self, status: Status, message: [*:0]const u8) Error {
            return refuseInto(if (completion == .blocking) self.diagnostics else null, status, message);
        }

        /// See `SequenceIndex.diagnosed`; a `Deferred` answers its own `message` instead.
        /// Binding-only.
        pub const diagnosed = if (completion == .blocking) diagnosedBlocking else {};

        /// See `SequenceIndex.contains`; a `Deferred` answers `count` instead.
        /// Calls `usearch_sequence_count`.
        pub const contains = if (completion == .blocking) containsBlocking else {};

        fn diagnosedBlocking(self: Self, outcome: *Outcome) Self {
            var lending = self;
            lending.diagnostics = outcome;
            return lending;
        }

        fn containsBlocking(self: Self, key: Key) Error!bool {
            return (try self.count(key)).entries != 0;
        }

        /// See `SequenceIndex.reserve`. Calls `usearch_sequence_reserve`.
        pub fn reserve(self: Self, entries: usize, payload_bytes: usize) Answer(void) {
            var local: Deferred(void) = undefined;
            const held = self.hold(&local, {});
            var arguments: usearch_sequence_reserve_t = .{
                .index = self.index,
                .executor = self.executor,
                .outcome = &held.outcome,
                .completion = completion,
                .entries = entries,
                .payload_bytes = payload_bytes,
            };
            usearch_sequence_reserve(&arguments);
            return self.end(held);
        }

        /// See `SequenceIndex.clear`. Calls `usearch_sequence_clear`.
        pub fn clear(self: Self) Answer(void) {
            var local: Deferred(void) = undefined;
            const held = self.hold(&local, {});
            var arguments: usearch_sequence_clear_t = .{ .index = self.index, .executor = self.executor, .outcome = &held.outcome, .completion = completion };
            usearch_sequence_clear(&arguments);
            return self.end(held);
        }

        /// See `SequenceIndex.describe`. Calls `usearch_sequence_describe`.
        pub fn describe(self: Self) Answer(SequenceIndexDescription) {
            var local: Deferred(SequenceIndexDescription) = undefined;
            const held = self.hold(&local, .{});
            var arguments: usearch_sequence_describe_t = .{ .index = self.index, .executor = self.executor, .outcome = &held.outcome, .completion = completion, .description = &held.result };
            usearch_sequence_describe(&arguments);
            return self.end(held);
        }

        /// See `SequenceIndex.count`. Calls `usearch_sequence_count`.
        pub fn count(self: Self, key: Key) Answer(SequenceKeyStats) {
            var local: Deferred(SequenceKeyStats) = undefined;
            const held = self.hold(&local, .{});
            var arguments: usearch_sequence_count_t = .{ .index = self.index, .executor = self.executor, .outcome = &held.outcome, .completion = completion, .key = key, .key_stats = &held.result };
            usearch_sequence_count(&arguments);
            return self.end(held);
        }

        /// See `SequenceIndex.get`. Calls `usearch_sequence_get`.
        pub fn get(self: Self, key: Key, arena: *Arena) Answer(TapeEntries(u8)) {
            var local: Deferred(TapeEntries(u8)) = undefined;
            const held = self.hold(&local, .{});
            var arguments: usearch_sequence_get_t = .{
                .index = self.index,
                .executor = self.executor,
                .outcome = &held.outcome,
                .completion = completion,
                .key = key,
                .sequences = &held.result.view,
                .written = &held.result.written,
                .arena = arena.handle,
            };
            usearch_sequence_get(&arguments);
            return self.end(held);
        }

        /// See `SequenceIndex.insert`. Calls `usearch_sequence_insert`.
        pub fn insert(self: Self, keys: []const Key, sequences: TapeView, config: SequenceInsertConfig) Answer(void) {
            // C takes one count for both runs, so only the binding sees the two lengths disagree.
            if (keys.len != sequences.entries) return self.refused(.shape_mismatch, "The key run and the tape disagree in length");
            var local: Deferred(void) = undefined;
            const held = self.hold(&local, {});
            var arguments: usearch_sequence_insert_t = .{
                .index = self.index,
                .executor = self.executor,
                .outcome = &held.outcome,
                .completion = completion,
                .count = sequences.entries,
                .keys = .of(keys),
                .sequences = .of(sequences),
                .expansion_factor = config.expansion_factor,
                .progress = .of(config.progress, config.progress_steps),
                .stats = config.stats,
            };
            usearch_sequence_insert(&arguments);
            return self.end(held);
        }

        /// See `SequenceIndex.refine`. Calls `usearch_sequence_refine`.
        pub fn refine(self: Self, config: SequenceRefineConfig) Answer(DenseRefineReport) {
            var local: Deferred(DenseRefineReport) = undefined;
            const held = self.hold(&local, .{});
            var arguments: usearch_sequence_refine_t = .{
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
            usearch_sequence_refine(&arguments);
            return self.end(held);
        }

        /// See `SequenceIndex.refinePlanFinish`. Calls `usearch_sequence_refine_plan_finish`.
        pub fn refinePlanFinish(self: Self, plan: *SequenceRefinePlan) Answer(DenseRefineReport) {
            var local: Deferred(DenseRefineReport) = undefined;
            const held = self.hold(&local, .{});
            var arguments: usearch_sequence_refine_plan_finish_t = .{
                .index = self.index,
                .executor = self.executor,
                .outcome = &held.outcome,
                .completion = completion,
                .plan = plan.handle,
                .report = &held.result,
            };
            plan.* = .{};
            usearch_sequence_refine_plan_finish(&arguments);
            return self.end(held);
        }

        /// See `SequenceIndex.health`. Calls `usearch_sequence_health`.
        pub fn health(self: Self) Answer(DenseReachability) {
            var local: Deferred(DenseReachability) = undefined;
            const held = self.hold(&local, .{});
            var arguments: usearch_sequence_health_t = .{ .index = self.index, .executor = self.executor, .outcome = &held.outcome, .completion = completion, .reachability = &held.result };
            usearch_sequence_health(&arguments);
            return self.end(held);
        }

        /// See `SequenceIndex.search`. Calls `usearch_sequence_search`.
        pub fn search(self: Self, queries: TapeView, wanted: usize, arena: *Arena, config: SequenceSearchConfig) Answer(Matches) {
            var local: Deferred(Matches) = undefined;
            const held = self.hold(&local, .{ .queries = queries.entries });
            var arguments: usearch_sequence_search_t = .{
                .index = self.index,
                .executor = self.executor,
                .outcome = &held.outcome,
                .completion = completion,
                .count = queries.entries,
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
            usearch_sequence_search(&arguments);
            return self.end(held);
        }

        /// See `SequenceIndex.reshard`; the handles come from the two lists, the executor and the
        /// completion from this receiver. Calls `usearch_sequence_reshard`.
        pub fn reshard(self: Self, sources: []const SequenceIndex, targets: []const SequenceIndex, config: SequenceReshardConfig) Answer(DenseRefineReport) {
            var local: Deferred(DenseRefineReport) = undefined;
            const held = self.hold(&local, .{});
            var arguments: usearch_sequence_reshard_t = .{
                .sources_count = sources.len,
                .sources = @ptrCast(sources.ptr),
                .targets_count = targets.len,
                .targets = @ptrCast(targets.ptr),
                .executor = self.executor,
                .outcome = &held.outcome,
                .completion = completion,
                .report = &held.result,
                .refine = .{ .expansion_factor = config.expansion_factor, .memory_limit = config.memory_limit },
                .seed = config.seed,
                .progress = .of(config.progress, config.progress_steps),
                .stats = config.stats,
                .plan = if (config.plan) |plan| &plan.handle else null,
                .phase = config.phase,
            };
            usearch_sequence_reshard(&arguments);
            return self.end(held);
        }

        /// See `SequenceIndex.reshardPlanFinish`. Calls `usearch_sequence_reshard_plan_finish`.
        pub fn reshardPlanFinish(self: Self, plan: *SequenceReshardPlan) Answer(DenseRefineReport) {
            var local: Deferred(DenseRefineReport) = undefined;
            const held = self.hold(&local, .{});
            var arguments: usearch_sequence_reshard_plan_finish_t = .{
                .executor = self.executor,
                .outcome = &held.outcome,
                .completion = completion,
                .plan = plan.handle,
                .report = &held.result,
            };
            plan.* = .{};
            usearch_sequence_reshard_plan_finish(&arguments);
            return self.end(held);
        }

        /// See `SequenceIndex.sample`. Calls `usearch_sequence_sample`.
        pub fn sample(self: Self, wanted: usize, arena: *Arena, config: DenseSampleConfig) Answer(DenseDraws) {
            // C reads one count for both runs, so only the binding sees the two lengths disagree.
            if (config.shown.len != config.shown_coordinates.len) return self.refused(.shape_mismatch, "The shown keys and coordinates disagree in length");
            var local: Deferred(DenseDraws) = undefined;
            const held = self.hold(&local, .{});
            var arguments: usearch_sequence_sample_t = .{
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
            usearch_sequence_sample(&arguments);
            return self.end(held);
        }

        /// See `SequenceIndex.cursorNext`. Calls `usearch_sequence_cursor_next`.
        pub fn cursorNext(self: Self, cursor: SequenceCursor, wanted: usize, links: bool, arena: *Arena) Answer(DenseCursorBatch) {
            var local: Deferred(DenseCursorBatch) = undefined;
            const held = self.hold(&local, .{});
            var arguments: usearch_sequence_cursor_next_t = .{
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
            usearch_sequence_cursor_next(&arguments);
            return self.end(held);
        }

        /// See `SequenceIndex.save`. Calls `usearch_sequence_save`.
        pub fn save(self: Self, path: []const u8, config: DenseSaveConfig) Answer(void) {
            var local: Deferred(void) = undefined;
            const held = self.hold(&local, {});
            var arguments: usearch_sequence_save_t = .{
                .index = self.index,
                .executor = self.executor,
                .outcome = &held.outcome,
                .completion = completion,
                .path = .of(path),
                .progress = .of(config.progress, config.progress_steps),
            };
            usearch_sequence_save(&arguments);
            return self.end(held);
        }

        /// See `SequenceIndex.saveBuffer`. Calls `usearch_sequence_save_buffer`.
        pub fn saveBuffer(self: Self, arena: *Arena, config: DenseSaveBufferConfig) Answer(Monolith) {
            var local: Deferred(Monolith) = undefined;
            const held = self.hold(&local, .{});
            var arguments: usearch_sequence_save_buffer_t = .{
                .index = self.index,
                .executor = self.executor,
                .outcome = &held.outcome,
                .completion = completion,
                .monolith = &held.result.view,
                .arena = arena.handle,
                .progress = .of(config.progress, config.progress_steps),
            };
            usearch_sequence_save_buffer(&arguments);
            return self.end(held);
        }
    };
}

/// A sequential walk over the index contents - export, migration, and visualization feed off it.
/// Every pull names the index again, through `SequenceIndex.cursorNext`, so free the cursor before
/// the index. Mirrors `usearch_sequence_cursor_t`.
pub const SequenceCursor = struct {
    /// The opaque `usearch_sequence_cursor_t` the ABI hands back.
    handle: ?*anyopaque,

    /// Opens a walk over the current contents; each shard's publication watermark is read as the
    /// walk enters it, so an entry once passed is never revisited. Without `starts` it yields every
    /// entry as stored; with them, the start keys' own entries first in order, then the entries
    /// those link to breadth-first, each once. Calls `usearch_sequence_cursor_init`.
    pub fn init(index: SequenceIndex, starts: []const Key) Error!SequenceCursor {
        var handle: ?*anyopaque = null;
        var outcome: Outcome = .{};
        var arguments: usearch_sequence_cursor_init_t = .{
            .index = index.handle,
            .outcome = &outcome,
            .starts_count = starts.len,
            .starts = starts.ptr,
            .cursor = &handle,
        };
        usearch_sequence_cursor_init(&arguments);
        errdefer usearch_sequence_cursor_free(handle);
        try outcome.check();
        return .{ .handle = handle };
    }

    /// Releases the cursor, waiting for every deferred pull naming it.
    /// Calls `usearch_sequence_cursor_free`.
    pub fn deinit(self: *SequenceCursor) void {
        usearch_sequence_cursor_free(self.handle);
        self.* = undefined;
    }
};

test "metrics keep the values the linked library spells them by" {
    inline for (.{ "levenshtein", "levenshtein-utf8", "needleman-wunsch", "overlap" }, 1..) |spelling, value| {
        const metric: SequenceMetric = @enumFromInt(value);
        try std.testing.expectEqualStrings(spelling, metric.name());
        try std.testing.expectEqual(@as(?SequenceMetric, metric), SequenceMetric.fromName(spelling));
    }
    try std.testing.expectEqual(@as(?SequenceMetric, null), SequenceMetric.fromName("hamming"));
}

test "a tape reads its entries from the offsets, and refuses offsets past the elements" {
    const offsets = [_]u32{ 0, 6, 13 };
    const tape = try TapeView.of(u8, u32, "kittensitting", &offsets);
    try std.testing.expectEqual(@as(usize, 2), tape.entries);
    try std.testing.expectEqual(basic.TapeLayout.u32, tape.layout);
    try std.testing.expectEqual(types.Scalar.u8, tape.scalar);
    try std.testing.expectError(Error.ShapeMismatch, TapeView.of(u8, u32, "kitten", &offsets));
    try std.testing.expectError(Error.ShapeMismatch, TapeView.of(u8, u64, "kitten", &.{}));
}

test { // analyze every declaration, the methods on each handle included
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(SequenceIndex);
    std.testing.refAllDecls(SequenceIndex.On);
    inline for (.{
        void,              usize,   SequenceIndexDescription, SequenceKeyStats, DenseReachability,
        DenseRefineReport, Matches, TapeEntries(u8),          DenseCursorBatch, DenseDraws,
        Monolith,
    }) |Result| std.testing.refAllDecls(Deferred(Result));
    std.testing.refAllDecls(SequenceCursor);
    std.testing.refAllDecls(TapeEntries(u8));
}

test "sequences cross as one tape, and each search finds its own entry first" {
    var executor = try Executor.forRam(.{ .threads = 2 });
    defer executor.deinit();
    var index = try SequenceIndex.initRam(.{ .metric = .levenshtein_utf8 }, &.{});
    defer index.deinit();
    try index.on(executor).reserve(3, 0);
    const description = try index.describe();
    try std.testing.expectEqual(SequenceMetric.levenshtein_utf8, description.metadata.metric);
    try std.testing.expectEqual(@as(usize, 0), description.size);

    const keys = [_]Key{ .fromInteger(1), .fromInteger(2), .fromInteger(3) };
    const words = "kittensittingmitten";
    const offsets = [_]u64{ 0, 6, 13, 19 };
    try index.on(executor).insert(&keys, try .of(u8, u64, words, &offsets), .{});
    try std.testing.expectEqual(@as(usize, 3), (try index.describe()).size);

    // A lent outcome carries the binding's own refusal, as the dense verbs lend it.
    var outcome: Outcome = .{};
    try std.testing.expectError(Error.ShapeMismatch, index.diagnosed(&outcome).insert(keys[0..2], try .of(u8, u64, words, &offsets), .{}));

    var arena: Arena = try .init(executor);
    defer arena.deinit();
    const matches = try index.on(executor).search(try .of(u8, u64, words, &offsets), 1, &arena, .{});
    for (keys, 0..) |key, query| {
        try std.testing.expectEqual(key, matches.query(query).keys[0]);
        try std.testing.expectEqual(@as(f64, 0), matches.query(query).scores[0]);
    }
    const nearly = [_]u64{ 0, 6 };
    const near = try index.search(try .of(u8, u64, "sittin", &nearly), 1, &arena, .{});
    try std.testing.expectEqual(keys[1], near.query(0).keys[0]);
    try std.testing.expectEqual(@as(f64, 1), near.query(0).scores[0]);

    const stats = try index.on(executor).count(keys[1]);
    try std.testing.expectEqual(SequenceKeyStats{ .entries = 1, .total_bytes = 7, .largest_bytes = 7 }, stats);
    try std.testing.expectEqual(true, try index.contains(keys[1]));
    try std.testing.expectEqual(false, try index.on(executor).contains(.fromInteger(99)));
    const stored = try index.on(executor).get(keys[1], &arena);
    try std.testing.expectEqual(@as(usize, 1), stored.written);
    try std.testing.expectEqualStrings("sitting", stored.entry(0));
    const absent = try index.get(.fromInteger(99), &arena);
    try std.testing.expectEqual(@as(usize, 0), absent.written);
    try std.testing.expectEqual(@as(usize, 0), absent.elements().len);

    // A cursor walks every key once, `wanted` at a time.
    var walk = try SequenceCursor.init(index, &.{});
    defer walk.deinit();
    try std.testing.expectEqualSlices(Key, keys[0..2], (try index.on(executor).cursorNext(walk, 2, false, &arena)).keys());
    try std.testing.expectEqualSlices(Key, keys[2..3], (try index.cursorNext(walk, 2, false, &arena)).keys());
    try std.testing.expectEqual(@as(usize, 0), (try index.cursorNext(walk, 2, false, &arena)).written);

    // A monolith round-trips through memory, and its metadata reads without loading it.
    const monolith = (try index.on(executor).saveBuffer(&arena, .{})).bytes();
    const frozen = (try index.describe()).metadata;
    try std.testing.expectEqual(frozen, try SequenceIndex.metadataReadBuffer(monolith, .{ .executor = executor }));
    var copy = try SequenceIndex.loadRamBuffer(monolith, .{ .executor = executor });
    defer copy.deinit();
    try std.testing.expectEqual(@as(usize, 3), (try copy.describe()).size);
    try std.testing.expectEqualStrings("mitten", (try copy.get(keys[2], &arena)).entry(0));

    // So does a file, which a mapping serves read-only.
    const checkpoint = ".zig-cache/usearch-zig-sequence.safegraphs";
    try index.on(executor).save(checkpoint, .{});
    defer std.Io.Dir.cwd().deleteFile(std.testing.io, checkpoint) catch {};
    try std.testing.expectEqual(frozen, try SequenceIndex.metadataRead(checkpoint, .{}));
    var loaded = try SequenceIndex.loadRam(checkpoint, .{ .executor = executor });
    defer loaded.deinit();
    try std.testing.expectEqual(@as(usize, 1), (try loaded.count(keys[0])).entries);
    if (SequenceIndex.viewRam(checkpoint, null)) |opened| {
        var mapped = opened;
        defer mapped.deinit();
        try std.testing.expectError(Error.ReadOnly, mapped.insert(keys[0..1], try .of(u8, u64, "kitten", offsets[0..2]), .{}));
    } else |failure| try std.testing.expectEqual(Error.Unsupported, failure);

    const reachability = try index.on(executor).health();
    try std.testing.expectEqual([2]usize{ 3, 3 }, [2]usize{ reachability.audited, reachability.reachable });
    const draws = try index.on(executor).sample(2, &arena, .{ .seed = 42 });
    try std.testing.expect(draws.written <= 2);
    try std.testing.expectEqual(draws.written, draws.coordinates().len);
}

test "a sequence refine keeps a stopped round in a plan, and a reshard refuses what it cannot move" {
    var executor = try Executor.forRam(.{ .threads = 2 });
    defer executor.deinit();
    const words = "kittensittingmittenbittenfittingknitting";
    const offsets = [_]u64{ 0, 6, 13, 19, 25, 32, 40 };
    var keys: [6]Key = undefined;
    for (&keys, 0..) |*key, position| key.* = .fromInteger(position);
    var first = try SequenceIndex.initRam(.{}, &.{});
    defer first.deinit();
    try first.on(executor).insert(&keys, try .of(u8, u64, words, &offsets), .{});

    const report = try first.on(executor).refine(.{});
    try std.testing.expect(report.changed <= report.reselected);
    try std.testing.expectEqual(@as(usize, 6), report.reachability.audited);

    const Stop = struct {
        fn always(_: void, _: usize, _: usize) types.Continuation {
            return .cancel;
        }
    };
    var plan: SequenceRefinePlan = .{};
    var phase: PhaseProgress = .{};
    try std.testing.expectError(Error.Cancelled, first.on(executor).refine(.{ .progress = .of({}, Stop.always), .plan = &plan, .phase = &phase }));
    try std.testing.expect(plan.handle != null and phase.phase != .done);
    _ = try first.on(executor).refine(.{ .plan = &plan, .phase = &phase });
    try std.testing.expectEqual(types.Phase.done, phase.phase);
    try std.testing.expectEqual(@as(usize, 6), (try first.on(executor).refinePlanFinish(&plan)).reachability.audited);
    try std.testing.expectEqual(SequenceRefinePlan{}, plan);
    // A finished plan holds nothing left to close.
    try std.testing.expectError(Error.InvalidArgument, first.on(executor).refinePlanFinish(&plan));

    // Malformed lists are invalid, foreign entries in a target mismatch, and a lone handle has no
    // shards to recut; none of them moves an entry.
    var second = try SequenceIndex.initRam(.{}, &.{});
    defer second.deinit();
    try second.on(executor).insert(keys[0..1], try .of(u8, u64, words, offsets[0..2]), .{});
    try std.testing.expectError(Error.InvalidArgument, first.on(executor).reshard(&.{}, &.{first}, .{}));
    try std.testing.expectError(Error.InvalidArgument, first.on(executor).reshard(&.{ first, first }, &.{first}, .{}));
    try std.testing.expectError(Error.ConfigMismatch, first.on(executor).reshard(&.{first}, &.{second}, .{}));
    try std.testing.expectError(Error.Unsupported, first.on(executor).reshard(&.{first}, &.{first}, .{}));
    try std.testing.expectEqual([2]usize{ 6, 1 }, [2]usize{ (try first.describe()).size, (try second.describe()).size });

    // One source into several targets cuts it between them, leaving the source empty but live,
    // and a deferred receiver closes the plan a stop left.
    var targets = [_]SequenceIndex{ try SequenceIndex.initRam(.{}, &.{}), try SequenceIndex.initRam(.{}, &.{}) };
    defer for (&targets) |*target| target.deinit();
    var cutting: SequenceReshardPlan = .{};
    try std.testing.expectError(Error.Cancelled, first.on(executor).reshard(&.{first}, &targets, .{ .seed = 7, .progress = .of({}, Stop.always), .plan = &cutting }));
    try std.testing.expectEqual(@as(usize, 0), (try first.describe()).size);
    try std.testing.expectEqual(@as(usize, 6), (try targets[0].describe()).size + (try targets[1].describe()).size);
    var closing: Deferred(DenseRefineReport) = undefined;
    try first.deferred(executor, &closing).reshardPlanFinish(&cutting);
    defer closing.deinit();
    _ = try closing.wait();
    try std.testing.expectEqual(SequenceReshardPlan{}, cutting);

    // Several sources into one target gather every entry there.
    _ = try targets[0].on(executor).reshard(&.{ targets[0], targets[1] }, &.{targets[0]}, .{});
    try std.testing.expectEqual(@as(usize, 6), (try targets[0].describe()).size);
    try std.testing.expectEqual(@as(usize, 0), (try targets[1].describe()).size);
}
