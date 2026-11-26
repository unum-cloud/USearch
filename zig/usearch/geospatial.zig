//! The geospatial modality - Geospatial Search over observations on and above the Earth, like
//! photos, video frames, drone footage and sensor fixes.
//!
//! `GeospatialIndex` is the handle; a slice of `GeospatialObservation`, the library's own eleven
//! doubles, is the batch currency every verb reads, so it crosses with no copy. Every output whose
//! length scales with the call lands in an `Arena` the verb requires and comes back as a view over
//! it, and `GeospatialCursor` walks the contents.
//!
//! The receiver chooses where a verb runs and how it completes, as for the dense modality: the
//! index runs it on the calling thread, `index.on(executor)` blocks on an executor, and
//! `index.deferred(executor, &call)` submits it there into a `Deferred` the caller holds.
//!
//! File: zig/usearch/geospatial.zig
//! Author: Ash Vardanian
//! Date: November 16, 2026

const std = @import("std");

const types = @import("types.zig");
const dense = @import("dense.zig");
const Arena = types.Arena;
const BytesView = types.BytesView;
const Completion = types.Completion;
const Distance = types.Distance;
const Error = types.Error;
const Executor = types.Executor;
const Key = types.Key;
const Matches = types.Matches;
const Outcome = types.Outcome;
const Predicate = types.Predicate;
const PhaseProgress = types.PhaseProgress;
const ProgressCallback = types.ProgressCallback;
const QueryMatches = types.QueryMatches;
const Status = types.Status;
const Traversal = types.Traversal;
const outputSlice = types.outputSlice;
const ArenaWrite = types.ArenaWrite;
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
const usearch_matches_tape_view_t = types.abi.matches_tape_view_t;
const usearch_matches_view_t = types.abi.matches_view_t;
const usearch_nvme_options_t = types.abi.nvme_options_t;
const usearch_progress_t = types.abi.progress_t;
const usearch_ram_options_t = types.abi.ram_options_t;
const usearch_rows_view_t = types.abi.rows_view_t;
const usearch_vram_options_t = types.abi.vram_options_t;

/// One observation as devices record it: a position in WGS-84 degrees and metres above the
/// ellipsoid, its accuracy, the device's view cone, how far it saw, and the Unix seconds it
/// covers. Mirrors `usearch_geospatial_observation_t`.
///
/// A field no device recorded is NaN, never zero, and only widens the tests that read it, so the
/// default leaves every field unrecorded.
pub const GeospatialObservation = extern struct {
    /// Degrees north of the equator, south negative.
    latitude_degrees: f64 = std.math.nan(f64),
    /// Degrees east of the prime meridian, west negative.
    longitude_degrees: f64 = std.math.nan(f64),
    /// Metres above the WGS-84 ellipsoid; NaN places the observation on the handle's terrain, or on
    /// the ellipsoid without one.
    altitude_metres: f64 = std.math.nan(f64),
    /// One standard deviation of the horizontal position, in metres.
    accuracy_metres: f64 = std.math.nan(f64),
    /// The view cone's axis, degrees clockwise from true north.
    heading_degrees: f64 = std.math.nan(f64),
    /// The view cone's axis, degrees above the horizon.
    pitch_degrees: f64 = std.math.nan(f64),
    /// One standard deviation of the heading, in degrees.
    heading_accuracy_degrees: f64 = std.math.nan(f64),
    /// The view cone's full opening angle, in degrees.
    field_of_view_degrees: f64 = std.math.nan(f64),
    /// How far the device saw, in metres along the slant.
    range_metres: f64 = std.math.nan(f64),
    /// The first instant the observation covers, in Unix seconds.
    begin_seconds: f64 = std.math.nan(f64),
    /// The last instant the observation covers, in Unix seconds.
    end_seconds: f64 = std.math.nan(f64),

    /// A position alone, every other field unrecorded. Binding-only.
    pub fn at(latitude_degrees: f64, longitude_degrees: f64) GeospatialObservation {
        return .{ .latitude_degrees = latitude_degrees, .longitude_degrees = longitude_degrees };
    }
};

/// Which entries count as matches of a query beside closeness; the radius and time window narrow
/// every relation alike. Mirrors `usearch_geospatial_relation_t`, which C spells by no name.
///
/// Non-exhaustive, so a relation a newer engine knows arrives intact rather than coerced.
pub const GeospatialRelation = enum(c_int) {
    /// No relation named - an unset field, refused by every query.
    unknown = 0,
    /// Within the radius, widened by three standard deviations of both positions combined.
    near = 1,
    /// Near, with a clear line of sight over the handle's terrain; refused by a handle without one.
    visible = 2,
    /// Near, and the entry's view cone holds the query: which cameras see this place.
    sees = 3,
    /// Near, and the query's view cone holds the entry: what this camera sees.
    seen = 4,
    _,
};

/// Mirrors `usearch_geospatial_terrain_callback_t`.
pub const TerrainFunction = *const fn (usize, usize, usize, usize, [*]f32, usize, ?*anyopaque) callconv(.c) void;

/// A terrain fill handler as the C interface holds it: a `usearch_geospatial_terrain_callback_t`
/// and the state forwarded into it, laid out as the two fields of `usearch_geospatial_terrain_t`.
/// Build one with `GeospatialTerrainFill.of`; the default fills nothing.
pub const GeospatialTerrainFill = extern struct {
    callback: ?TerrainFunction = null,
    state: ?*anyopaque = null,

    /// Wraps `fills` in a trampoline generated per call site, handing it the window's first row
    /// and column, its rows and columns, and the heights to fill as a slice whose rows lie
    /// `stride` heights apart. Pass `{}` for a fill needing no context, and a single-item pointer
    /// otherwise.
    ///
    /// Heights are metres above the WGS-84 ellipsoid, and a cell left NaN never blocks a sight
    /// line. It runs concurrently from the executor's threads, and VRAM handles refuse it.
    pub fn of(
        context: anytype,
        comptime fills: fn (@TypeOf(context), usize, usize, usize, usize, []f32, usize) void,
    ) GeospatialTerrainFill {
        const Context = @TypeOf(context);
        types.checkContext(Context);
        return .{
            .callback = struct {
                fn fill(first_row: usize, first_column: usize, rows: usize, columns: usize, heights: [*]f32, heights_stride: usize, state: ?*anyopaque) callconv(.c) void {
                    const stride = heights_stride / @sizeOf(f32);
                    const length = if (rows == 0) 0 else (rows - 1) * stride + columns;
                    const window = heights[0..length];
                    if (Context == void) return fills({}, first_row, first_column, rows, columns, window, stride);
                    fills(@ptrCast(@alignCast(state.?)), first_row, first_column, rows, columns, window, stride);
                }
            }.fill,
            .state = types.contextPointer(Context, context),
        };
    }
};

/// The elevation every line of sight is tested over: a north-up geographic grid of heights in
/// metres above the WGS-84 ellipsoid, given by exactly one of viewed `heights`, a `fill`, or a
/// dataset `path`, which the handle reads for as long as it lives. The default holds none.
/// Mirrors `usearch_geospatial_terrain_t`.
///
/// A checkpoint never holds it, so every load and view names it again.
pub const GeospatialTerrain = extern struct {
    /// The grid's row count; rows run south from the north edge.
    rows: usize = 0,
    /// The grid's column count; columns run east from the west edge.
    columns: usize = 0,
    /// The latitude of the centre of cell `[0, 0]`, in degrees.
    north_latitude_degrees: f64 = 0,
    /// The longitude of the centre of cell `[0, 0]`, in degrees.
    west_longitude_degrees: f64 = 0,
    /// The step between rows, in degrees, positive.
    row_step_degrees: f64 = 0,
    /// The step between columns, in degrees, positive.
    column_step_degrees: f64 = 0,
    /// The whole grid, row-major, viewed for the handle's lifetime.
    heights: ?[*]const f32 = null,
    /// The bytes between consecutive rows of `heights`; zero means packed.
    heights_stride: usize = 0,
    /// Or a handler filling windows of the grid on demand; VRAM handles refuse it.
    fill: GeospatialTerrainFill = .{},
    /// Or a dataset on disk, which RAM and VRAM handles refuse with `Error.Unsupported`.
    path: BytesView = .{ .data = null, .length = 0 },
};

/// The terrain and graph an empty index freezes - what `GeospatialIndex.initRam` and `initVram`
/// read beside their placement. Binding-only.
pub const GeospatialOptions = struct {
    /// The elevation sight lines clear; the default holds none, which refuses `.visible`.
    terrain: GeospatialTerrain = .{},
    /// The optional connections-per-node bound; zero derives.
    connectivity: usize = 0,
    /// Upper bound on observations sharing one key; zero is unbounded.
    values_per_key: usize = 0,
};

/// The graph a checkpoint froze - everything a reader learns without opening it.
/// Mirrors `usearch_geospatial_metadata_t`.
pub const GeospatialMetadata = extern struct {
    /// Connections per node the graph was built with.
    connectivity: usize = 0,
};

/// A live index's state, reported by `GeospatialIndex.describe`. Each counter is read once, so
/// under concurrent insertions they may come from slightly different moments.
/// Mirrors `usearch_geospatial_index_description_t`.
pub const GeospatialIndexDescription = extern struct {
    /// The frozen connectivity.
    metadata: GeospatialMetadata = .{},
    /// Stored entries.
    size: usize = 0,
    /// Entries hosted without growing again.
    capacity: usize = 0,
    /// Bytes the handle holds, observations and the terrain included; traversal scratch is the
    /// executor's.
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
    pub fn cpuClassName(self: GeospatialIndexDescription) [:0]const u8 {
        return std.mem.span(types.abi.cpu_class_name(self.cpu_class));
    }

    /// The GPU class's spelling, "unknown" for a host index. Calls `usearch_gpu_class_name`.
    pub fn gpuClassName(self: GeospatialIndexDescription) [:0]const u8 {
        return std.mem.span(types.abi.gpu_class_name(self.gpu_class));
    }

    /// The load path's spelling: "none", "host-copy" or "cufile". Calls `usearch_load_path_name`.
    pub fn loadPathName(self: GeospatialIndexDescription) [:0]const u8 {
        return std.mem.span(types.abi.load_path_name(self.load_path));
    }
};

/// What one key holds, enough to size a buffer before retrieving any of it.
/// Mirrors `usearch_geospatial_key_stats_t`.
pub const GeospatialKeyStats = extern struct {
    /// Observations stored under the key; zero means the index holds none.
    entries: usize = 0,
};

/// A refine a cancel left live, which the next `refine` naming it continues and
/// `refinePlanFinish` closes. Mirrors `usearch_geospatial_refine_plan_t`; the default holds none.
pub const GeospatialRefinePlan = extern struct {
    handle: ?*anyopaque = null,
};

/// A reshard a cancel left live, which the next `reshard` naming it continues and
/// `reshardPlanFinish` closes. Mirrors `usearch_geospatial_reshard_plan_t`; the default holds none.
pub const GeospatialReshardPlan = extern struct {
    handle: ?*anyopaque = null,
};

/// Optional settings of `GeospatialIndex.loadRam`.
pub const GeospatialLoadRamConfig = struct {
    /// The executor the read fans out over; null reads on the calling thread.
    executor: ?Executor = null,
    /// The memory domains to place the parts on; empty derives every node the process may use.
    memory_domains: []const i32 = &.{},
    /// Upper bound on entries sharing one key, which no checkpoint records; zero is unbounded.
    values_per_key: usize = 0,
    /// The elevation sight lines clear, which no checkpoint holds; the default holds none.
    terrain: GeospatialTerrain = .{},
    /// Called with shards placed; a stopped load creates no index.
    progress: ProgressCallback = .{},
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    progress_steps: usize = 0,
    /// Where the call's final outcome, its static message included, is copied; null copies nothing.
    diagnostics: ?*Outcome = null,
};

/// Optional settings of `GeospatialIndex.loadVram`.
pub const GeospatialLoadVramConfig = struct {
    /// The executor the upload runs on; null runs on the calling thread and the default stream.
    executor: ?Executor = null,
    /// The devices to place the parts on; empty derives one free device.
    devices: []const i32 = &.{},
    /// Upper bound on entries sharing one key, which no checkpoint records; zero is unbounded.
    values_per_key: usize = 0,
    /// The elevation sight lines clear, which no checkpoint holds; the default holds none.
    terrain: GeospatialTerrain = .{},
    /// Called with shards placed; a stopped load creates no index.
    progress: ProgressCallback = .{},
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    progress_steps: usize = 0,
    /// Where the call's final outcome, its static message included, is copied; null copies nothing.
    diagnostics: ?*Outcome = null,
};

/// Optional settings of `GeospatialIndex.loadRamBuffer`.
pub const GeospatialLoadRamBufferConfig = struct {
    /// The executor the copy fans out over; null copies on the calling thread.
    executor: ?Executor = null,
    /// The memory domains to place shards on; the monolith lands whole on the first.
    memory_domains: []const i32 = &.{},
    /// Upper bound on entries sharing one key, which no checkpoint records; zero is unbounded.
    values_per_key: usize = 0,
    /// The elevation sight lines clear, which no checkpoint holds; the default holds none.
    terrain: GeospatialTerrain = .{},
    /// Called with bytes placed; a stopped load creates no index.
    progress: ProgressCallback = .{},
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    progress_steps: usize = 0,
    /// Where the call's final outcome, its static message included, is copied; null copies nothing.
    diagnostics: ?*Outcome = null,
};

/// Optional settings of `GeospatialIndex.loadVramBuffer`.
pub const GeospatialLoadVramBufferConfig = struct {
    /// The executor the upload runs on; null runs on the calling thread and the default stream.
    executor: ?Executor = null,
    /// The devices to place shards on; the monolith lands whole on the first.
    devices: []const i32 = &.{},
    /// Upper bound on entries sharing one key, which no checkpoint records; zero is unbounded.
    values_per_key: usize = 0,
    /// The elevation sight lines clear, which no checkpoint holds; the default holds none.
    terrain: GeospatialTerrain = .{},
    /// Called with bytes placed; a stopped load creates no index.
    progress: ProgressCallback = .{},
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    progress_steps: usize = 0,
    /// Where the call's final outcome, its static message included, is copied; null copies nothing.
    diagnostics: ?*Outcome = null,
};

/// Optional settings of `GeospatialIndex.viewNvme`.
pub const GeospatialViewNvmeConfig = struct {
    /// The NVMe executor whose rings read the checkpoint's headers; null reads synchronously.
    executor: ?Executor = null,
    /// The devices the parts are served from; empty serves each where its manifest points.
    paths: []const BytesView = &.{},
    /// The elevation sight lines clear, which no checkpoint holds; the default holds none.
    terrain: GeospatialTerrain = .{},
    /// Where the call's final outcome, its static message included, is copied; null copies nothing.
    diagnostics: ?*Outcome = null,
};

/// Optional settings of `GeospatialIndex.insert`.
pub const GeospatialInsertConfig = struct {
    /// The breadth of the candidates the build walk explores; zero derives.
    expansion_factor: usize = 0,
    /// Called with entries admitted; it reports but cannot cancel, as every entry is claimed first.
    progress: ProgressCallback = .{},
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    progress_steps: usize = 0,
    stats: ?*DenseTraversalStats = null,
};

/// Optional settings of `GeospatialIndex.refine`.
pub const GeospatialRefineConfig = struct {
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
    plan: ?*GeospatialRefinePlan = null,
    /// Where the call stopped, `.done` once its round ended; null skips it.
    phase: ?*PhaseProgress = null,
};

/// Optional settings of `GeospatialIndex.search`.
pub const GeospatialSearchConfig = struct {
    /// What each match must be to the query beside close.
    relation: GeospatialRelation = .near,
    /// The radius in metres; zero ranks by distance alone.
    metres: f64 = 0,
    /// Seconds widening both ends of the query's interval; a query with no time matches all.
    seconds: f64 = 0,
    /// The breadth of the candidates the walk explores; zero derives. A selective relation, time
    /// window or predicate wants it wider.
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

/// Optional settings of `GeospatialIndex.within`.
pub const GeospatialWithinConfig = struct {
    /// What each match must be to the query beside close.
    relation: GeospatialRelation = .near,
    /// Seconds widening both ends of the query's interval; a query with no time matches all.
    seconds: f64 = 0,
    /// A walk, or an exact scan of every entry on RAM and NVMe; VRAM answers `Error.Unsupported`.
    traversal: Traversal = .walk,
    /// Admits only the keys it approves; the rest still route the walk. Refused on VRAM.
    predicate: Predicate = .{},
    /// Called with queries resolved; a stopped call answers `Error.Cancelled` and no matches.
    progress: ProgressCallback = .{},
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    progress_steps: usize = 0,
    stats: ?*DenseTraversalStats = null,
};

/// Optional settings of `GeospatialIndex.reshard`.
pub const GeospatialReshardConfig = struct {
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
    plan: ?*GeospatialReshardPlan = null,
    /// Where the call stopped, `.done` once its rounds ended; null skips it.
    phase: ?*PhaseProgress = null,
};

/// The radius and time slack of one query. Mirrors `usearch_geospatial_tolerance_t`.
const usearch_geospatial_tolerance_t = extern struct {
    metres: f64,
    seconds: f64,
};

const usearch_geospatial_init_ram_t = extern struct {
    index: *?*anyopaque,
    outcome: *Outcome,
    terrain: GeospatialTerrain,
    connectivity: usize,
    values_per_key: usize,
    ram: usearch_ram_options_t,
};

const usearch_geospatial_init_vram_t = extern struct {
    index: *?*anyopaque,
    outcome: *Outcome,
    terrain: GeospatialTerrain,
    connectivity: usize,
    values_per_key: usize,
    vram: usearch_vram_options_t,
};

const usearch_geospatial_load_ram_t = extern struct {
    index: *?*anyopaque,
    executor: ?*anyopaque,
    outcome: *Outcome,
    completion: Completion,
    path: BytesView,
    ram: usearch_ram_options_t,
    values_per_key: usize,
    terrain: GeospatialTerrain,
    progress: usearch_progress_t,
};

const usearch_geospatial_load_ram_buffer_t = extern struct {
    index: *?*anyopaque,
    executor: ?*anyopaque,
    outcome: *Outcome,
    completion: Completion,
    buffer: BytesView,
    ram: usearch_ram_options_t,
    values_per_key: usize,
    terrain: GeospatialTerrain,
    progress: usearch_progress_t,
};

const usearch_geospatial_load_vram_t = extern struct {
    index: *?*anyopaque,
    executor: ?*anyopaque,
    outcome: *Outcome,
    completion: Completion,
    path: BytesView,
    vram: usearch_vram_options_t,
    values_per_key: usize,
    terrain: GeospatialTerrain,
    progress: usearch_progress_t,
};

const usearch_geospatial_load_vram_buffer_t = extern struct {
    index: *?*anyopaque,
    executor: ?*anyopaque,
    outcome: *Outcome,
    completion: Completion,
    buffer: BytesView,
    vram: usearch_vram_options_t,
    values_per_key: usize,
    terrain: GeospatialTerrain,
    progress: usearch_progress_t,
};

const usearch_geospatial_view_ram_t = extern struct {
    index: *?*anyopaque,
    outcome: *Outcome,
    path: BytesView,
    terrain: GeospatialTerrain,
    ram: usearch_ram_options_t = .{},
};

const usearch_geospatial_view_nvme_t = extern struct {
    index: *?*anyopaque,
    executor: ?*anyopaque,
    outcome: *Outcome,
    completion: Completion,
    path: BytesView,
    nvme: usearch_nvme_options_t,
    terrain: GeospatialTerrain,
};

const usearch_geospatial_reserve_t = extern struct {
    index: ?*anyopaque,
    executor: ?*anyopaque,
    outcome: *Outcome,
    completion: Completion,
    entries: usize,
};

const usearch_geospatial_clear_t = extern struct {
    index: ?*anyopaque,
    executor: ?*anyopaque,
    outcome: *Outcome,
    completion: Completion,
};

const usearch_geospatial_insert_t = extern struct {
    index: ?*anyopaque,
    executor: ?*anyopaque,
    outcome: *Outcome,
    completion: Completion,
    count: usize,
    keys: usearch_keys_view_t,
    observations: ?[*]const GeospatialObservation,
    expansion_factor: usize,
    progress: usearch_progress_t,
    stats: ?*DenseTraversalStats,
};

const usearch_refine_options_t = extern struct {
    expansion_factor: usize,
    memory_limit: usize,
};

const usearch_geospatial_refine_t = extern struct {
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

const usearch_geospatial_refine_plan_finish_t = extern struct {
    index: ?*anyopaque,
    executor: ?*anyopaque,
    outcome: *Outcome,
    completion: Completion,
    plan: ?*anyopaque,
    report: ?*DenseRefineReport,
};

const usearch_geospatial_health_t = extern struct {
    index: ?*anyopaque,
    executor: ?*anyopaque,
    outcome: *Outcome,
    completion: Completion,
    reachability: *DenseReachability,
};

const usearch_geospatial_search_t = extern struct {
    index: ?*anyopaque,
    executor: ?*anyopaque,
    outcome: *Outcome,
    completion: Completion,
    count: usize,
    queries: ?[*]const GeospatialObservation,
    relation: GeospatialRelation,
    tolerance: usearch_geospatial_tolerance_t,
    matches: *usearch_matches_view_t,
    arena: ?*anyopaque,
    wanted: usize,
    expansion_factor: usize,
    traversal: Traversal,
    predicate: Predicate,
    progress: usearch_progress_t,
    stats: ?*DenseTraversalStats,
};

const usearch_geospatial_within_t = extern struct {
    index: ?*anyopaque,
    executor: ?*anyopaque,
    outcome: *Outcome,
    completion: Completion,
    count: usize,
    queries: ?[*]const GeospatialObservation,
    relation: GeospatialRelation,
    tolerance: usearch_geospatial_tolerance_t,
    matches: *usearch_matches_tape_view_t,
    written: ?*usize,
    arena: ?*anyopaque,
    traversal: Traversal,
    predicate: Predicate,
    progress: usearch_progress_t,
    stats: ?*DenseTraversalStats,
};

const usearch_geospatial_count_t = extern struct {
    index: ?*anyopaque,
    executor: ?*anyopaque,
    outcome: *Outcome,
    completion: Completion,
    key: Key,
    key_stats: *GeospatialKeyStats,
};

const usearch_geospatial_get_t = extern struct {
    index: ?*anyopaque,
    executor: ?*anyopaque,
    outcome: *Outcome,
    completion: Completion,
    key: Key,
    observations: *?[*]const GeospatialObservation,
    written: ?*usize,
    arena: ?*anyopaque,
};

const usearch_geospatial_reshard_t = extern struct {
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

const usearch_geospatial_reshard_plan_finish_t = extern struct {
    executor: ?*anyopaque,
    outcome: *Outcome,
    completion: Completion,
    plan: ?*anyopaque,
    report: ?*DenseRefineReport,
};

const usearch_geospatial_cursor_init_t = extern struct {
    index: ?*anyopaque,
    executor: ?*anyopaque = null,
    outcome: *Outcome,
    starts_count: usize,
    starts: ?[*]const Key,
    cursor: *?*anyopaque,
};

const usearch_geospatial_cursor_next_t = extern struct {
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

const usearch_geospatial_sample_t = extern struct {
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

const usearch_geospatial_save_t = extern struct {
    index: ?*anyopaque,
    executor: ?*anyopaque,
    outcome: *Outcome,
    completion: Completion,
    path: BytesView,
    progress: usearch_progress_t,
};

const usearch_geospatial_save_buffer_t = extern struct {
    index: ?*anyopaque,
    executor: ?*anyopaque,
    outcome: *Outcome,
    completion: Completion,
    monolith: *BytesView,
    arena: ?*anyopaque,
    progress: usearch_progress_t,
};

const usearch_geospatial_metadata_read_t = extern struct {
    executor: ?*anyopaque,
    outcome: *Outcome,
    completion: Completion,
    path: BytesView,
    metadata: *GeospatialMetadata,
};

const usearch_geospatial_metadata_read_buffer_t = extern struct {
    executor: ?*anyopaque,
    outcome: *Outcome,
    completion: Completion,
    buffer: BytesView,
    metadata: *GeospatialMetadata,
};

const usearch_geospatial_describe_t = extern struct {
    index: ?*anyopaque,
    executor: ?*anyopaque,
    outcome: *Outcome,
    completion: Completion,
    description: *GeospatialIndexDescription,
};

extern fn usearch_geospatial_init_ram(arguments: *usearch_geospatial_init_ram_t) void;
extern fn usearch_geospatial_init_vram(arguments: *usearch_geospatial_init_vram_t) void;
extern fn usearch_geospatial_load_ram(arguments: *usearch_geospatial_load_ram_t) void;
extern fn usearch_geospatial_load_ram_buffer(arguments: *usearch_geospatial_load_ram_buffer_t) void;
extern fn usearch_geospatial_load_vram(arguments: *usearch_geospatial_load_vram_t) void;
extern fn usearch_geospatial_load_vram_buffer(arguments: *usearch_geospatial_load_vram_buffer_t) void;
extern fn usearch_geospatial_view_ram(arguments: *usearch_geospatial_view_ram_t) void;
extern fn usearch_geospatial_view_nvme(arguments: *usearch_geospatial_view_nvme_t) void;
extern fn usearch_geospatial_free(index: ?*anyopaque) void;
extern fn usearch_geospatial_reserve(arguments: *usearch_geospatial_reserve_t) void;
extern fn usearch_geospatial_clear(arguments: *usearch_geospatial_clear_t) void;
extern fn usearch_geospatial_insert(arguments: *usearch_geospatial_insert_t) void;
extern fn usearch_geospatial_refine(arguments: *usearch_geospatial_refine_t) void;
extern fn usearch_geospatial_refine_plan_finish(arguments: *usearch_geospatial_refine_plan_finish_t) void;
extern fn usearch_geospatial_health(arguments: *usearch_geospatial_health_t) void;
extern fn usearch_geospatial_search(arguments: *usearch_geospatial_search_t) void;
extern fn usearch_geospatial_within(arguments: *usearch_geospatial_within_t) void;
extern fn usearch_geospatial_count(arguments: *usearch_geospatial_count_t) void;
extern fn usearch_geospatial_get(arguments: *usearch_geospatial_get_t) void;
extern fn usearch_geospatial_reshard(arguments: *usearch_geospatial_reshard_t) void;
extern fn usearch_geospatial_reshard_plan_finish(arguments: *usearch_geospatial_reshard_plan_finish_t) void;
extern fn usearch_geospatial_sample(arguments: *usearch_geospatial_sample_t) void;
extern fn usearch_geospatial_cursor_init(arguments: *usearch_geospatial_cursor_init_t) void;
extern fn usearch_geospatial_cursor_next(arguments: *usearch_geospatial_cursor_next_t) void;
extern fn usearch_geospatial_cursor_free(cursor: ?*anyopaque) void;
extern fn usearch_geospatial_save(arguments: *usearch_geospatial_save_t) void;
extern fn usearch_geospatial_save_buffer(arguments: *usearch_geospatial_save_buffer_t) void;
extern fn usearch_geospatial_metadata_read(arguments: *usearch_geospatial_metadata_read_t) void;
extern fn usearch_geospatial_metadata_read_buffer(arguments: *usearch_geospatial_metadata_read_buffer_t) void;
extern fn usearch_geospatial_describe(arguments: *usearch_geospatial_describe_t) void;

/// Geospatial Search Index over observations on and above the Earth.
///
/// A handle is shareable across threads: searches never write to the graph, and insertions
/// synchronize through per-node locks and publish in order. It is the `Executor` that orders,
/// never the index.
///
/// ```zig
/// const usearch = @import("usearch");
///
/// var index = try usearch.GeospatialIndex.initRam(.{}, &.{});
/// defer index.deinit();
///
/// const keys = [_]usearch.Key{ .fromInteger(1), .fromInteger(2) };
/// const stops = [_]usearch.GeospatialObservation{ .at(51.5308, -0.1238), .at(51.5033, -0.1196) };
/// try index.insert(&keys, &stops, .{});
///
/// var arena = try usearch.Arena.init(null);
/// defer arena.deinit();
/// const nearby = try index.within(stops[0..1], 500, &arena, .{}); // every stop within 500 metres
/// const first = (try nearby.query(0)).keys;
/// ```
pub const GeospatialIndex = extern struct {
    /// The opaque `usearch_geospatial_index_t` the ABI hands back.
    handle: ?*anyopaque,

    /// Every verb of the index, blocking on an executor, refused while another call holds it.
    pub const On = Receiver(.blocking, void);

    /// Every verb of the index, submitted to an executor into a `Deferred(Result)` the caller
    /// holds; only the verbs answering `Result` compile.
    pub fn DeferredOn(comptime Result: type) type {
        return Receiver(.deferred, Result);
    }

    /// Runs every verb on `executor`, whose kind must match the index's residency. Binding-only.
    pub fn on(self: GeospatialIndex, executor: Executor) On {
        return .{ .index = self.handle, .executor = executor.handle, .into = {}, .diagnostics = null };
    }

    /// Submits a verb to `executor`, holding it in `call`, a `*Deferred(Result)` the caller places
    /// anywhere, which stays put until its `deinit`. Binding-only.
    pub fn deferred(self: GeospatialIndex, executor: Executor, call: anytype) DeferredOn(@FieldType(@TypeOf(call.*), "result")) {
        return .{ .index = self.handle, .executor = executor.handle, .into = call, .diagnostics = {} };
    }

    /// Runs every verb on the calling thread, copying each call's final outcome into `outcome`.
    /// Binding-only.
    pub fn diagnosed(self: GeospatialIndex, outcome: *Outcome) On {
        return self.here().diagnosed(outcome);
    }

    /// The receiver the index's own verbs run through: no executor, the calling thread.
    fn here(self: GeospatialIndex) On {
        return .{ .index = self.handle, .executor = null, .into = {}, .diagnostics = null };
    }

    /// Creates an empty, mutable index in host memory, one shard per memory domain; empty derives
    /// every node the process may allocate on. Calls `usearch_geospatial_init_ram`.
    pub fn initRam(options: GeospatialOptions, memory_domains: []const i32) Error!GeospatialIndex {
        var handle: ?*anyopaque = null;
        var outcome: Outcome = .{};
        var arguments: usearch_geospatial_init_ram_t = .{
            .index = &handle,
            .outcome = &outcome,
            .terrain = options.terrain,
            .connectivity = options.connectivity,
            .values_per_key = options.values_per_key,
            .ram = .of(memory_domains),
        };
        usearch_geospatial_init_ram(&arguments);
        return adopt(handle, outcome, null);
    }

    /// Creates an empty, mutable index in device memory, one shard per device and never a replica;
    /// empty derives one free device. Calls `usearch_geospatial_init_vram`.
    pub fn initVram(options: GeospatialOptions, devices: []const i32) Error!GeospatialIndex {
        var handle: ?*anyopaque = null;
        var outcome: Outcome = .{};
        var arguments: usearch_geospatial_init_vram_t = .{
            .index = &handle,
            .outcome = &outcome,
            .terrain = options.terrain,
            .connectivity = options.connectivity,
            .values_per_key = options.values_per_key,
            .vram = .of(devices),
        };
        usearch_geospatial_init_vram(&arguments);
        return adopt(handle, outcome, null);
    }

    /// Copies a checkpoint into a new, mutable index in host memory, shard by shard.
    ///
    /// Point `path` at the `.safegraphs.json` manifest, never at one of the parts it enumerates.
    /// The graph comes from the checkpoint's own headers, and the terrain from the config.
    /// Calls `usearch_geospatial_load_ram`.
    pub fn loadRam(path: []const u8, config: GeospatialLoadRamConfig) Error!GeospatialIndex {
        var handle: ?*anyopaque = null;
        var outcome: Outcome = .{};
        var arguments: usearch_geospatial_load_ram_t = .{
            .index = &handle,
            .executor = handleOf(config.executor),
            .outcome = &outcome,
            .completion = .blocking,
            .path = .of(path),
            .ram = .of(config.memory_domains),
            .values_per_key = config.values_per_key,
            .terrain = config.terrain,
            .progress = .of(config.progress, config.progress_steps),
        };
        usearch_geospatial_load_ram(&arguments);
        return adopt(handle, outcome, config.diagnostics);
    }

    /// Copies a checkpoint into a new, mutable index in device memory, with the placement rules of
    /// `loadRam`. Calls `usearch_geospatial_load_vram`.
    pub fn loadVram(path: []const u8, config: GeospatialLoadVramConfig) Error!GeospatialIndex {
        var handle: ?*anyopaque = null;
        var outcome: Outcome = .{};
        var arguments: usearch_geospatial_load_vram_t = .{
            .index = &handle,
            .executor = handleOf(config.executor),
            .outcome = &outcome,
            .completion = .blocking,
            .path = .of(path),
            .vram = .of(config.devices),
            .values_per_key = config.values_per_key,
            .terrain = config.terrain,
            .progress = .of(config.progress, config.progress_steps),
        };
        usearch_geospatial_load_vram(&arguments);
        return adopt(handle, outcome, config.diagnostics);
    }

    /// Copies a monolith from a memory buffer into a new index in host memory, as `loadRam` does a
    /// path. Calls `usearch_geospatial_load_ram_buffer`.
    pub fn loadRamBuffer(buffer: []const u8, config: GeospatialLoadRamBufferConfig) Error!GeospatialIndex {
        var handle: ?*anyopaque = null;
        var outcome: Outcome = .{};
        var arguments: usearch_geospatial_load_ram_buffer_t = .{
            .index = &handle,
            .executor = handleOf(config.executor),
            .outcome = &outcome,
            .completion = .blocking,
            .buffer = .of(buffer),
            .ram = .of(config.memory_domains),
            .values_per_key = config.values_per_key,
            .terrain = config.terrain,
            .progress = .of(config.progress, config.progress_steps),
        };
        usearch_geospatial_load_ram_buffer(&arguments);
        return adopt(handle, outcome, config.diagnostics);
    }

    /// Copies a monolith from a host or device buffer into a new index in device memory.
    /// Calls `usearch_geospatial_load_vram_buffer`.
    pub fn loadVramBuffer(buffer: []const u8, config: GeospatialLoadVramBufferConfig) Error!GeospatialIndex {
        var handle: ?*anyopaque = null;
        var outcome: Outcome = .{};
        var arguments: usearch_geospatial_load_vram_buffer_t = .{
            .index = &handle,
            .executor = handleOf(config.executor),
            .outcome = &outcome,
            .completion = .blocking,
            .buffer = .of(buffer),
            .vram = .of(config.devices),
            .values_per_key = config.values_per_key,
            .terrain = config.terrain,
            .progress = .of(config.progress, config.progress_steps),
        };
        usearch_geospatial_load_vram_buffer(&arguments);
        return adopt(handle, outcome, config.diagnostics);
    }

    /// Serves a checkpoint in place through a memory mapping, as a read-only index whose mutating
    /// verbs report `Error.ReadOnly`; the terrain is named again, as no checkpoint holds it.
    /// Calls `usearch_geospatial_view_ram`.
    pub fn viewRam(path: []const u8, terrain: GeospatialTerrain) Error!GeospatialIndex {
        var handle: ?*anyopaque = null;
        var outcome: Outcome = .{};
        var arguments: usearch_geospatial_view_ram_t = .{ .index = &handle, .outcome = &outcome, .path = .of(path), .terrain = terrain };
        usearch_geospatial_view_ram(&arguments);
        return adopt(handle, outcome, null);
    }

    /// Serves a checkpoint in place from NVMe drives, read-only and sized past RAM.
    /// Calls `usearch_geospatial_view_nvme`.
    pub fn viewNvme(path: []const u8, config: GeospatialViewNvmeConfig) Error!GeospatialIndex {
        var handle: ?*anyopaque = null;
        var outcome: Outcome = .{};
        var arguments: usearch_geospatial_view_nvme_t = .{
            .index = &handle,
            .executor = handleOf(config.executor),
            .outcome = &outcome,
            .completion = .blocking,
            .path = .of(path),
            .nvme = .of(config.paths),
            .terrain = config.terrain,
        };
        usearch_geospatial_view_nvme(&arguments);
        return adopt(handle, outcome, config.diagnostics);
    }

    /// Reads a checkpoint's graph, without loading it. Calls `usearch_geospatial_metadata_read`.
    pub fn metadataRead(path: []const u8, config: DenseMetadataReadConfig) Error!GeospatialMetadata {
        var found: GeospatialMetadata = .{};
        var outcome: Outcome = .{};
        var arguments: usearch_geospatial_metadata_read_t = .{
            .executor = handleOf(config.executor),
            .outcome = &outcome,
            .completion = .blocking,
            .path = .of(path),
            .metadata = &found,
        };
        usearch_geospatial_metadata_read(&arguments);
        try outcome.checkInto(config.diagnostics);
        return found;
    }

    /// Reads the graph out of a serialized monolith buffer, without loading it.
    /// Calls `usearch_geospatial_metadata_read_buffer`.
    pub fn metadataReadBuffer(buffer: []const u8, config: DenseMetadataReadBufferConfig) Error!GeospatialMetadata {
        var found: GeospatialMetadata = .{};
        var outcome: Outcome = .{};
        var arguments: usearch_geospatial_metadata_read_buffer_t = .{
            .executor = handleOf(config.executor),
            .outcome = &outcome,
            .completion = .blocking,
            .buffer = .of(buffer),
            .metadata = &found,
        };
        usearch_geospatial_metadata_read_buffer(&arguments);
        try outcome.checkInto(config.diagnostics);
        return found;
    }

    fn adopt(handle: ?*anyopaque, outcome: Outcome, diagnostics: ?*Outcome) Error!GeospatialIndex {
        errdefer usearch_geospatial_free(handle);
        try outcome.checkInto(diagnostics);
        return .{ .handle = handle };
    }

    /// Frees the resources associated with the index, waiting for every deferred call naming it;
    /// the handle is invalid afterwards. Calls `usearch_geospatial_free`.
    pub fn deinit(self: *GeospatialIndex) void {
        usearch_geospatial_free(self.handle);
        self.* = undefined;
    }

    /// Reserves room for a total number of entries, the current size included.
    /// Calls `usearch_geospatial_reserve`.
    pub fn reserve(self: GeospatialIndex, entries: usize) Error!void {
        return self.here().reserve(entries);
    }

    /// Erases all entries, keeping the reserved capacity and the terrain. Never runs concurrently
    /// with any other verb on the same handle. Calls `usearch_geospatial_clear`.
    pub fn clear(self: GeospatialIndex) Error!void {
        return self.here().clear();
    }

    /// Reports the frozen metadata and the current counters in one call.
    /// Calls `usearch_geospatial_describe`.
    pub fn describe(self: GeospatialIndex) Error!GeospatialIndexDescription {
        return self.here().describe();
    }

    /// What `key` holds: its observations. Calls `usearch_geospatial_count`.
    pub fn count(self: GeospatialIndex, key: Key) Error!GeospatialKeyStats {
        return self.here().count(key);
    }

    /// Whether the index holds an entry under `key`. Binding-only;
    /// calls `usearch_geospatial_count`.
    pub fn contains(self: GeospatialIndex, key: Key) Error!bool {
        return self.here().contains(key);
    }

    /// Every observation stored under `key` in insertion order, within the rounding the index
    /// stores them at, written in `arena`; an absent key yields none.
    /// Calls `usearch_geospatial_get`.
    pub fn get(self: GeospatialIndex, key: Key, arena: *Arena) Error!GeospatialObservations {
        return self.here().get(key, arena);
    }

    /// Adds observations under their keys - the native batch operation, one key per observation.
    ///
    /// The batch is all-or-nothing: every fallible step runs before the first entry identity is
    /// claimed, so a handler reports progress but cannot stop it. An observation without an
    /// altitude is placed on the handle's terrain, or on the ellipsoid without one.
    /// Calls `usearch_geospatial_insert`.
    pub fn insert(self: GeospatialIndex, keys: []const Key, observations: []const GeospatialObservation, config: GeospatialInsertConfig) Error!void {
        return self.here().insert(keys, observations, config);
    }

    /// Runs one round over the finished graph: a health check, a heal of its islands, and a gather
    /// and restitch of every neighborhood, closed by a second health check, reporting what changed.
    /// A stop at a barrier answers `Error.Cancelled`, and a `plan` in the config keeps the round to
    /// continue later. Calls `usearch_geospatial_refine`.
    pub fn refine(self: GeospatialIndex, config: GeospatialRefineConfig) Error!DenseRefineReport {
        return self.here().refine(config);
    }

    /// Closes a live refine plan, reporting what the whole plan did; the plan is freed and emptied
    /// whatever the call answers. Calls `usearch_geospatial_refine_plan_finish`.
    pub fn refinePlanFinish(self: GeospatialIndex, plan: *GeospatialRefinePlan) Error!DenseRefineReport {
        return self.here().refinePlanFinish(plan);
    }

    /// Measures how much of the graph a search reaches from the root; the entries it misses stay
    /// unfound until a `refine` re-links them. Calls `usearch_geospatial_health`.
    pub fn health(self: GeospatialIndex) Error!DenseReachability {
        return self.here().health();
    }

    /// Ranks up to `wanted` of the nearest entries each query's relation admits, in chord metres
    /// with altitude included, written in `arena` - the native batch operation.
    ///
    /// The relation, the time window and the predicate narrow the walk's candidates, so a query
    /// may hold fewer than `wanted` matches. Calls `usearch_geospatial_search`.
    pub fn search(self: GeospatialIndex, queries: []const GeospatialObservation, wanted: usize, arena: *Arena, config: GeospatialSearchConfig) Error!Matches {
        return self.here().search(queries, wanted, arena, config);
    }

    /// Reports every entry within `metres` of each query that the relation admits, however many,
    /// as a ragged output with Apache Arrow offsets written in `arena`.
    ///
    /// Complete on the converged graph, since the walk sizes its own candidates, and refused with
    /// `Error.InvalidArgument` for a zero radius. Calls `usearch_geospatial_within`.
    pub fn within(self: GeospatialIndex, queries: []const GeospatialObservation, metres: f64, arena: *Arena, config: GeospatialWithinConfig) Error!GeospatialMatches {
        return self.here().within(queries, metres, arena, config);
    }

    /// Draws up to `wanted` keys around `config.focus`, or across the whole index, each hanging
    /// from a shown or earlier draw and placed in three dimensions beside the shown ones, written
    /// in `arena`. Calls `usearch_geospatial_sample`.
    pub fn sample(self: GeospatialIndex, wanted: usize, arena: *Arena, config: DenseSampleConfig) Error!DenseDraws {
        return self.here().sample(wanted, arena, config);
    }

    /// Copies up to `wanted` more entries out of `cursor` into `arena` - their keys, and the keys
    /// each links to when `links` asks - answering an empty batch once the walk is exhausted. The
    /// observations themselves come from `get`. Calls `usearch_geospatial_cursor_next`.
    pub fn cursorNext(self: GeospatialIndex, cursor: GeospatialCursor, wanted: usize, links: bool, arena: *Arena) Error!DenseCursorBatch {
        return self.here().cursorNext(cursor, wanted, links, arena);
    }

    /// Saves the index under `path` in the layout its extension picks - a `.safegraphs` monolith,
    /// or a `.safegraphs.json` manifest beside its shards; the terrain is never saved.
    ///
    /// Demands a mutation-free window; concurrent readers keep searching throughout. A failed
    /// save leaves a file without its header, which every reader rejects, and that leftover is
    /// the caller's to remove. Calls `usearch_geospatial_save`.
    pub fn save(self: GeospatialIndex, path: []const u8, config: DenseSaveConfig) Error!void {
        return self.here().save(path, config);
    }

    /// Saves the index as one monolith - the same bytes `save` writes to a `.safegraphs` path -
    /// written in `arena`. Calls `usearch_geospatial_save_buffer`.
    pub fn saveBuffer(self: GeospatialIndex, arena: *Arena, config: DenseSaveBufferConfig) Error!Monolith {
        return self.here().saveBuffer(arena, config);
    }

    /// Moves the entries of `sources` into `targets` on the calling thread, keeping their links
    /// rather than re-inserting the observations, and runs a round on each target that took any.
    ///
    /// Several sources into one target, itself among them, gather into it; one source into several
    /// targets cuts it into graph-local parts. A source the targets do not name ends empty and
    /// live, and every handle stays the caller's. Calls `usearch_geospatial_reshard`.
    pub fn reshard(sources: []const GeospatialIndex, targets: []const GeospatialIndex, config: GeospatialReshardConfig) Error!DenseRefineReport {
        return (GeospatialIndex{ .handle = null }).here().reshard(sources, targets, config);
    }

    /// Closes a live reshard plan on the calling thread, reporting what the whole plan did; the
    /// plan is freed and emptied whatever the call answers.
    /// Calls `usearch_geospatial_reshard_plan_finish`.
    pub fn reshardPlanFinish(plan: *GeospatialReshardPlan) Error!DenseRefineReport {
        return (GeospatialIndex{ .handle = null }).here().reshardPlanFinish(plan);
    }
};

fn handleOf(executor: ?Executor) ?*anyopaque {
    return if (executor) |chosen| chosen.handle else null;
}

/// The verbs of `GeospatialIndex` once more, for a receiver that completes them as `completion`
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
            if (self.diagnostics) |passed| passed.* = held.outcome;
            try held.outcome.check();
            return held.result;
        }

        /// Fails a call the binding refused before it crossed, as `refuseInto` does.
        fn refused(self: Self, status: Status, message: [*:0]const u8) Error {
            return refuseInto(if (completion == .blocking) self.diagnostics else null, status, message);
        }

        /// See `GeospatialIndex.diagnosed`; a `Deferred` answers its own `message` instead.
        /// Binding-only.
        pub const diagnosed = if (completion == .blocking) diagnosedBlocking else {};

        /// See `GeospatialIndex.contains`; a `Deferred` answers `count` instead.
        /// Calls `usearch_geospatial_count`.
        pub const contains = if (completion == .blocking) containsBlocking else {};

        fn diagnosedBlocking(self: Self, outcome: *Outcome) Self {
            var passing = self;
            passing.diagnostics = outcome;
            return passing;
        }

        fn containsBlocking(self: Self, key: Key) Error!bool {
            return (try self.count(key)).entries != 0;
        }

        /// See `GeospatialIndex.reserve`. Calls `usearch_geospatial_reserve`.
        pub fn reserve(self: Self, entries: usize) Answer(void) {
            var local: Deferred(void) = undefined;
            const held = self.hold(&local, {});
            var arguments: usearch_geospatial_reserve_t = .{ .index = self.index, .executor = self.executor, .outcome = &held.outcome, .completion = completion, .entries = entries };
            usearch_geospatial_reserve(&arguments);
            return self.end(held);
        }

        /// See `GeospatialIndex.clear`. Calls `usearch_geospatial_clear`.
        pub fn clear(self: Self) Answer(void) {
            var local: Deferred(void) = undefined;
            const held = self.hold(&local, {});
            var arguments: usearch_geospatial_clear_t = .{ .index = self.index, .executor = self.executor, .outcome = &held.outcome, .completion = completion };
            usearch_geospatial_clear(&arguments);
            return self.end(held);
        }

        /// See `GeospatialIndex.describe`. Calls `usearch_geospatial_describe`.
        pub fn describe(self: Self) Answer(GeospatialIndexDescription) {
            var local: Deferred(GeospatialIndexDescription) = undefined;
            const held = self.hold(&local, .{});
            var arguments: usearch_geospatial_describe_t = .{ .index = self.index, .executor = self.executor, .outcome = &held.outcome, .completion = completion, .description = &held.result };
            usearch_geospatial_describe(&arguments);
            return self.end(held);
        }

        /// See `GeospatialIndex.count`. Calls `usearch_geospatial_count`.
        pub fn count(self: Self, key: Key) Answer(GeospatialKeyStats) {
            var local: Deferred(GeospatialKeyStats) = undefined;
            const held = self.hold(&local, .{});
            var arguments: usearch_geospatial_count_t = .{ .index = self.index, .executor = self.executor, .outcome = &held.outcome, .completion = completion, .key = key, .key_stats = &held.result };
            usearch_geospatial_count(&arguments);
            return self.end(held);
        }

        /// See `GeospatialIndex.get`. Calls `usearch_geospatial_get`.
        pub fn get(self: Self, key: Key, arena: *Arena) Answer(GeospatialObservations) {
            var local: Deferred(GeospatialObservations) = undefined;
            const held = self.hold(&local, .{});
            held.result.write = .stamp(arena.handle);
            var arguments: usearch_geospatial_get_t = .{
                .index = self.index,
                .executor = self.executor,
                .outcome = &held.outcome,
                .completion = completion,
                .key = key,
                .observations = &held.result.view,
                .written = &held.result.written,
                .arena = arena.handle,
            };
            usearch_geospatial_get(&arguments);
            return self.end(held);
        }

        /// See `GeospatialIndex.insert`. Calls `usearch_geospatial_insert`.
        pub fn insert(self: Self, keys: []const Key, observations: []const GeospatialObservation, config: GeospatialInsertConfig) Answer(void) {
            // C takes one count for both runs, so only the binding sees the two lengths disagree.
            if (keys.len != observations.len) return self.refused(.shape_mismatch, "The key run and the observations disagree in length");
            var local: Deferred(void) = undefined;
            const held = self.hold(&local, {});
            var arguments: usearch_geospatial_insert_t = .{
                .index = self.index,
                .executor = self.executor,
                .outcome = &held.outcome,
                .completion = completion,
                .count = observations.len,
                .keys = .of(keys),
                .observations = observations.ptr,
                .expansion_factor = config.expansion_factor,
                .progress = .of(config.progress, config.progress_steps),
                .stats = config.stats,
            };
            usearch_geospatial_insert(&arguments);
            return self.end(held);
        }

        /// See `GeospatialIndex.refine`. Calls `usearch_geospatial_refine`.
        pub fn refine(self: Self, config: GeospatialRefineConfig) Answer(DenseRefineReport) {
            var local: Deferred(DenseRefineReport) = undefined;
            const held = self.hold(&local, .{});
            var arguments: usearch_geospatial_refine_t = .{
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
            usearch_geospatial_refine(&arguments);
            return self.end(held);
        }

        /// See `GeospatialIndex.refinePlanFinish`. Calls `usearch_geospatial_refine_plan_finish`.
        pub fn refinePlanFinish(self: Self, plan: *GeospatialRefinePlan) Answer(DenseRefineReport) {
            var local: Deferred(DenseRefineReport) = undefined;
            const held = self.hold(&local, .{});
            var arguments: usearch_geospatial_refine_plan_finish_t = .{
                .index = self.index,
                .executor = self.executor,
                .outcome = &held.outcome,
                .completion = completion,
                .plan = plan.handle,
                .report = &held.result,
            };
            plan.* = .{};
            usearch_geospatial_refine_plan_finish(&arguments);
            return self.end(held);
        }

        /// See `GeospatialIndex.health`. Calls `usearch_geospatial_health`.
        pub fn health(self: Self) Answer(DenseReachability) {
            var local: Deferred(DenseReachability) = undefined;
            const held = self.hold(&local, .{});
            var arguments: usearch_geospatial_health_t = .{ .index = self.index, .executor = self.executor, .outcome = &held.outcome, .completion = completion, .reachability = &held.result };
            usearch_geospatial_health(&arguments);
            return self.end(held);
        }

        /// See `GeospatialIndex.search`. Calls `usearch_geospatial_search`.
        pub fn search(self: Self, queries: []const GeospatialObservation, wanted: usize, arena: *Arena, config: GeospatialSearchConfig) Answer(Matches) {
            var local: Deferred(Matches) = undefined;
            const held = self.hold(&local, .{ .queries = queries.len });
            held.result.write = .stamp(arena.handle);
            var arguments: usearch_geospatial_search_t = .{
                .index = self.index,
                .executor = self.executor,
                .outcome = &held.outcome,
                .completion = completion,
                .count = queries.len,
                .queries = queries.ptr,
                .relation = config.relation,
                .tolerance = .{ .metres = config.metres, .seconds = config.seconds },
                .matches = &held.result.view,
                .arena = arena.handle,
                .wanted = wanted,
                .expansion_factor = config.expansion_factor,
                .traversal = config.traversal,
                .predicate = config.predicate,
                .progress = .of(config.progress, config.progress_steps),
                .stats = config.stats,
            };
            usearch_geospatial_search(&arguments);
            return self.end(held);
        }

        /// See `GeospatialIndex.within`. Calls `usearch_geospatial_within`.
        pub fn within(self: Self, queries: []const GeospatialObservation, metres: f64, arena: *Arena, config: GeospatialWithinConfig) Answer(GeospatialMatches) {
            var local: Deferred(GeospatialMatches) = undefined;
            const held = self.hold(&local, .{ .queries = queries.len });
            held.result.write = .stamp(arena.handle);
            var arguments: usearch_geospatial_within_t = .{
                .index = self.index,
                .executor = self.executor,
                .outcome = &held.outcome,
                .completion = completion,
                .count = queries.len,
                .queries = queries.ptr,
                .relation = config.relation,
                .tolerance = .{ .metres = metres, .seconds = config.seconds },
                .matches = &held.result.view,
                .written = &held.result.written,
                .arena = arena.handle,
                .traversal = config.traversal,
                .predicate = config.predicate,
                .progress = .of(config.progress, config.progress_steps),
                .stats = config.stats,
            };
            usearch_geospatial_within(&arguments);
            return self.end(held);
        }

        /// See `GeospatialIndex.reshard`; the handles come from the two lists, the executor and the
        /// completion from this receiver. Calls `usearch_geospatial_reshard`.
        pub fn reshard(self: Self, sources: []const GeospatialIndex, targets: []const GeospatialIndex, config: GeospatialReshardConfig) Answer(DenseRefineReport) {
            var local: Deferred(DenseRefineReport) = undefined;
            const held = self.hold(&local, .{});
            var arguments: usearch_geospatial_reshard_t = .{
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
            usearch_geospatial_reshard(&arguments);
            return self.end(held);
        }

        /// See `GeospatialIndex.reshardPlanFinish`. Calls `usearch_geospatial_reshard_plan_finish`.
        pub fn reshardPlanFinish(self: Self, plan: *GeospatialReshardPlan) Answer(DenseRefineReport) {
            var local: Deferred(DenseRefineReport) = undefined;
            const held = self.hold(&local, .{});
            var arguments: usearch_geospatial_reshard_plan_finish_t = .{
                .executor = self.executor,
                .outcome = &held.outcome,
                .completion = completion,
                .plan = plan.handle,
                .report = &held.result,
            };
            plan.* = .{};
            usearch_geospatial_reshard_plan_finish(&arguments);
            return self.end(held);
        }

        /// See `GeospatialIndex.sample`. Calls `usearch_geospatial_sample`.
        pub fn sample(self: Self, wanted: usize, arena: *Arena, config: DenseSampleConfig) Answer(DenseDraws) {
            // C reads one count for both runs, so only the binding sees the two lengths disagree.
            if (config.shown.len != config.shown_coordinates.len) return self.refused(.shape_mismatch, "The shown keys and coordinates disagree in length");
            var local: Deferred(DenseDraws) = undefined;
            const held = self.hold(&local, .{});
            held.result.write = .stamp(arena.handle);
            var arguments: usearch_geospatial_sample_t = .{
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
            usearch_geospatial_sample(&arguments);
            return self.end(held);
        }

        /// See `GeospatialIndex.cursorNext`. Calls `usearch_geospatial_cursor_next`.
        pub fn cursorNext(self: Self, cursor: GeospatialCursor, wanted: usize, links: bool, arena: *Arena) Answer(DenseCursorBatch) {
            var local: Deferred(DenseCursorBatch) = undefined;
            const held = self.hold(&local, .{});
            held.result.write = .stamp(arena.handle);
            var arguments: usearch_geospatial_cursor_next_t = .{
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
            usearch_geospatial_cursor_next(&arguments);
            return self.end(held);
        }

        /// See `GeospatialIndex.save`. Calls `usearch_geospatial_save`.
        pub fn save(self: Self, path: []const u8, config: DenseSaveConfig) Answer(void) {
            var local: Deferred(void) = undefined;
            const held = self.hold(&local, {});
            var arguments: usearch_geospatial_save_t = .{
                .index = self.index,
                .executor = self.executor,
                .outcome = &held.outcome,
                .completion = completion,
                .path = .of(path),
                .progress = .of(config.progress, config.progress_steps),
            };
            usearch_geospatial_save(&arguments);
            return self.end(held);
        }

        /// See `GeospatialIndex.saveBuffer`. Calls `usearch_geospatial_save_buffer`.
        pub fn saveBuffer(self: Self, arena: *Arena, config: DenseSaveBufferConfig) Answer(Monolith) {
            var local: Deferred(Monolith) = undefined;
            const held = self.hold(&local, .{});
            held.result.write = .stamp(arena.handle);
            var arguments: usearch_geospatial_save_buffer_t = .{
                .index = self.index,
                .executor = self.executor,
                .outcome = &held.outcome,
                .completion = completion,
                .monolith = &held.result.view,
                .arena = arena.handle,
                .progress = .of(config.progress, config.progress_steps),
            };
            usearch_geospatial_save_buffer(&arguments);
            return self.end(held);
        }
    };
}

/// The observations a `get` wrote into an arena, in insertion order, readable until the arena's
/// next call.
pub const GeospatialObservations = struct {
    write: ArenaWrite = .{},
    view: ?[*]const GeospatialObservation = null,
    /// Observations written; zero for an absent key.
    written: usize = 0,

    /// The observations, within the rounding the index stores them at.
    pub fn observations(self: GeospatialObservations) Error![]const GeospatialObservation {
        try self.write.open();
        defer self.write.close();
        return outputSlice(GeospatialObservation, self.view, 0, self.written);
    }
};

/// The ragged output of `GeospatialIndex.within` in an arena, laid out as Apache Arrow does: query
/// `i` spans matches `offsets[i]` up to `offsets[i + 1]` of `keys` and `scores`, readable until the
/// arena's next call. Mirrors `usearch_matches_tape_view_t`.
pub const GeospatialMatches = struct {
    write: ArenaWrite = .{},
    view: usearch_matches_tape_view_t = .{},
    /// Queries the call resolved.
    queries: usize,
    /// Matches written across every query.
    written: usize = 0,

    /// The matches of one query in chord metres, in no promised order, `position` below `queries`.
    pub fn query(self: GeospatialMatches, position: usize) Error!QueryMatches {
        try self.write.open();
        defer self.write.close();
        const tape = try self.offsets();
        const begin = tape[position];
        const length = tape[position + 1] - begin;
        return .{ .keys = outputSlice(Key, self.view.keys, begin, length), .scores = outputSlice(Distance, self.view.scores, begin, length) };
    }

    /// The `queries + 1` offsets.
    pub fn offsets(self: GeospatialMatches) Error![]const usize {
        try self.write.open();
        defer self.write.close();
        return outputSlice(usize, self.view.offsets, 0, self.queries + 1);
    }

    /// Every match's key, end to end.
    pub fn keys(self: GeospatialMatches) Error![]const Key {
        try self.write.open();
        defer self.write.close();
        return outputSlice(Key, self.view.keys, 0, self.written);
    }

    /// Every match's chord metres, positionally paired with `keys`.
    pub fn scores(self: GeospatialMatches) Error![]const Distance {
        try self.write.open();
        defer self.write.close();
        return outputSlice(Distance, self.view.scores, 0, self.written);
    }
};

/// A sequential walk over the index contents - export, migration, and visualization feed off it.
/// Every pull names the index again, through `GeospatialIndex.cursorNext`, so free the cursor
/// before the index. Mirrors `usearch_geospatial_cursor_t`.
pub const GeospatialCursor = struct {
    /// The opaque `usearch_geospatial_cursor_t` the ABI hands back.
    handle: ?*anyopaque,

    /// Opens a walk over the current contents; each shard's publication watermark is read as the
    /// walk enters it, so an entry once passed is never revisited. Without `starts` it yields every
    /// entry as stored; with them, the start keys' own entries first in order, then the entries
    /// those link to breadth-first, each once. Calls `usearch_geospatial_cursor_init`.
    pub fn init(index: GeospatialIndex, starts: []const Key) Error!GeospatialCursor {
        var handle: ?*anyopaque = null;
        var outcome: Outcome = .{};
        var arguments: usearch_geospatial_cursor_init_t = .{
            .index = index.handle,
            .outcome = &outcome,
            .starts_count = starts.len,
            .starts = starts.ptr,
            .cursor = &handle,
        };
        usearch_geospatial_cursor_init(&arguments);
        errdefer usearch_geospatial_cursor_free(handle);
        try outcome.check();
        return .{ .handle = handle };
    }

    /// Releases the cursor, waiting for every deferred pull naming it.
    /// Calls `usearch_geospatial_cursor_free`.
    pub fn deinit(self: *GeospatialCursor) void {
        usearch_geospatial_cursor_free(self.handle);
        self.* = undefined;
    }
};

test "an observation leaves every unrecorded field NaN" {
    const position = GeospatialObservation.at(48.8566, 2.3522);
    try std.testing.expectEqual(@as(f64, 48.8566), position.latitude_degrees);
    try std.testing.expect(std.math.isNan(position.altitude_metres) and std.math.isNan(position.end_seconds));
    try std.testing.expectEqual(GeospatialRelation.near, (GeospatialSearchConfig{}).relation);
}

test "a terrain fill hands its handler the window as rows of the stride" {
    const Grid = struct {
        level: f32,
        fn fill(self: *const @This(), first_row: usize, _: usize, rows: usize, columns: usize, heights: []f32, stride: usize) void {
            for (0..rows) |row| for (0..columns) |column| {
                heights[row * stride + column] = self.level + @as(f32, @floatFromInt(first_row + row));
            };
        }
    };
    const grid: Grid = .{ .level = 100 };
    const filler = GeospatialTerrainFill.of(&grid, Grid.fill);
    var window: [2 * 4]f32 = @splat(0);
    filler.callback.?(3, 0, 2, 3, &window, 4 * @sizeOf(f32), filler.state);
    try std.testing.expectEqualSlices(f32, &.{ 103, 103, 103, 0, 104, 104, 104, 0 }, &window);
}

test { // analyze every declaration, the methods on each handle included
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(GeospatialIndex);
    std.testing.refAllDecls(GeospatialIndex.On);
    inline for (.{
        void,              usize,    GeospatialIndexDescription, GeospatialKeyStats,     DenseReachability,
        DenseRefineReport, Matches,  GeospatialMatches,          GeospatialObservations, DenseCursorBatch,
        DenseDraws,        Monolith,
    }) |Result| std.testing.refAllDecls(Deferred(Result));
    std.testing.refAllDecls(GeospatialCursor);
    std.testing.refAllDecls(GeospatialMatches);
    std.testing.refAllDecls(GeospatialObservations);
}

test "observations cross as the library's own structs, and a search finds each at zero metres" {
    var executor = try Executor.forRam(.{ .threads = 2 });
    defer executor.deinit();
    var index = try GeospatialIndex.initRam(.{}, &.{});
    defer index.deinit();
    try index.on(executor).reserve(3);

    const keys = [_]Key{ .fromInteger(1), .fromInteger(2), .fromInteger(3) };
    const cities = [_]GeospatialObservation{ .at(48.8566, 2.3522), .at(52.52, 13.405), .at(40.7128, -74.006) };
    try index.on(executor).insert(&keys, &cities, .{});
    const description = try index.describe();
    try std.testing.expectEqual(@as(usize, 3), description.size);

    var outcome: Outcome = .{};
    try std.testing.expectError(Error.ShapeMismatch, index.diagnosed(&outcome).insert(keys[0..2], &cities, .{}));

    var arena: Arena = try .init(executor);
    defer arena.deinit();
    const matches = try index.on(executor).search(&cities, 1, &arena, .{ .relation = .near });
    for (keys, 0..) |key, query| {
        try std.testing.expectEqual(key, (try matches.query(query)).keys[0]);
        try std.testing.expectEqual(@as(f64, 0), (try matches.query(query)).scores[0]);
    }
    // A handle without terrain refuses a line of sight.
    try std.testing.expectError(Error.Unsupported, index.search(cities[0..1], 1, &arena, .{ .relation = .visible }));

    try std.testing.expectEqual(GeospatialKeyStats{ .entries = 1 }, try index.on(executor).count(keys[1]));
    try std.testing.expectEqual(true, try index.contains(keys[1]));
    try std.testing.expectEqual(false, try index.on(executor).contains(.fromInteger(99)));
    const stored = try (try index.on(executor).get(keys[1], &arena)).observations();
    try std.testing.expectEqual(@as(usize, 1), stored.len);
    try std.testing.expectApproxEqAbs(@as(f64, 52.52), stored[0].latitude_degrees, 1e-6);
    try std.testing.expect(std.math.isNan(stored[0].heading_degrees));
    try std.testing.expectEqual(@as(usize, 0), (try index.get(.fromInteger(99), &arena)).written);

    var walk = try GeospatialCursor.init(index, &.{});
    defer walk.deinit();
    try std.testing.expectEqualSlices(Key, &keys, try (try index.on(executor).cursorNext(walk, 4, false, &arena)).keys());
    try std.testing.expectEqual(@as(usize, 0), (try index.cursorNext(walk, 4, false, &arena)).written);

    const monolith = try (try index.on(executor).saveBuffer(&arena, .{})).bytes();
    try std.testing.expectEqual(description.metadata, try GeospatialIndex.metadataReadBuffer(monolith, .{ .executor = executor }));
    var copy = try GeospatialIndex.loadRamBuffer(monolith, .{ .executor = executor });
    defer copy.deinit();
    try std.testing.expectEqual(@as(usize, 3), (try copy.describe()).size);

    const checkpoint = ".zig-cache/usearch-zig-geospatial.safegraphs";
    try index.on(executor).save(checkpoint, .{});
    defer std.Io.Dir.cwd().deleteFile(std.testing.io, checkpoint) catch {};
    try std.testing.expectEqual(description.metadata, try GeospatialIndex.metadataRead(checkpoint, .{}));
    var loaded = try GeospatialIndex.loadRam(checkpoint, .{ .executor = executor });
    defer loaded.deinit();
    try std.testing.expectEqual(@as(usize, 1), (try loaded.count(keys[0])).entries);
    if (GeospatialIndex.viewRam(checkpoint, .{})) |opened| {
        var mapped = opened;
        defer mapped.deinit();
        try std.testing.expectError(Error.ReadOnly, mapped.clear());
    } else |failure| try std.testing.expectEqual(Error.Unsupported, failure);

    const reachability = try index.on(executor).health();
    try std.testing.expectEqual([2]usize{ 3, 3 }, [2]usize{ reachability.audited, reachability.reachable });
    const draws = try index.on(executor).sample(2, &arena, .{ .seed = 42 });
    try std.testing.expectEqual(draws.written, (try draws.coordinates()).len);
}

test "within reports every entry inside the radius, and refuses a zero one" {
    var index = try GeospatialIndex.initRam(.{}, &.{});
    defer index.deinit();
    const keys = [_]Key{ .fromInteger(1), .fromInteger(2), .fromInteger(3) };
    const places = [_]GeospatialObservation{ .at(48.8566, 2.3522), .at(48.8656, 2.3522), .at(52.52, 13.405) };
    try index.insert(&keys, &places, .{});

    var arena: Arena = try .init(null);
    defer arena.deinit();
    const nearby = try index.within(places[0..1], 2000, &arena, .{});
    var found: [2]u64 = undefined;
    try std.testing.expectEqual(@as(usize, 2), (try nearby.query(0)).keys.len);
    for ((try nearby.query(0)).keys, &found) |key, *integer| integer.* = key.asInteger().?;
    std.mem.sort(u64, &found, {}, std.sort.asc(u64));
    try std.testing.expectEqual([2]u64{ 1, 2 }, found);
    try std.testing.expectEqual(nearby.written, (try nearby.offsets())[1]);
    try std.testing.expectError(Error.InvalidArgument, index.within(places[0..1], 0, &arena, .{}));

    // A deferred call is the same verb, final once waited on.
    var executor = try Executor.forRam(.{});
    defer executor.deinit();
    var other: Arena = try .init(executor);
    defer other.deinit();
    var call: Deferred(GeospatialMatches) = undefined;
    try index.deferred(executor, &call).within(places[2..3], 2000, &other, .{});
    defer call.deinit();
    try std.testing.expectEqualSlices(Key, keys[2..3], (try (try call.wait()).query(0)).keys);
}

test "a geospatial refine keeps a stopped round in a plan, and a reshard refuses what it cannot move" {
    var executor = try Executor.forRam(.{ .threads = 2 });
    defer executor.deinit();
    var places: [6]GeospatialObservation = undefined;
    var keys: [6]Key = undefined;
    for (&places, &keys, 0..) |*place, *key, position| {
        const step: f64 = @floatFromInt(position);
        place.* = .at(10 + step * 7, -120 + step * 41);
        key.* = .fromInteger(position);
    }
    var first = try GeospatialIndex.initRam(.{}, &.{});
    defer first.deinit();
    try first.on(executor).insert(&keys, &places, .{});

    const report = try first.on(executor).refine(.{});
    try std.testing.expect(report.changed <= report.reselected);
    try std.testing.expectEqual(@as(usize, 6), report.reachability.audited);

    const Stop = struct {
        fn always(_: void, _: usize, _: usize) types.Continuation {
            return .cancel;
        }
    };
    var plan: GeospatialRefinePlan = .{};
    var phase: PhaseProgress = .{};
    try std.testing.expectError(Error.Cancelled, first.on(executor).refine(.{ .progress = .of({}, Stop.always), .plan = &plan, .phase = &phase }));
    try std.testing.expect(plan.handle != null and phase.phase != .done);
    _ = try first.on(executor).refine(.{ .plan = &plan, .phase = &phase });
    try std.testing.expectEqual(types.Phase.done, phase.phase);
    try std.testing.expectEqual(@as(usize, 6), (try first.on(executor).refinePlanFinish(&plan)).reachability.audited);
    try std.testing.expectError(Error.InvalidArgument, first.on(executor).refinePlanFinish(&plan));

    var second = try GeospatialIndex.initRam(.{}, &.{});
    defer second.deinit();
    try second.on(executor).insert(keys[0..1], places[0..1], .{});
    try std.testing.expectError(Error.InvalidArgument, first.on(executor).reshard(&.{}, &.{first}, .{}));
    try std.testing.expectError(Error.InvalidArgument, first.on(executor).reshard(&.{ first, first }, &.{first}, .{}));
    try std.testing.expectError(Error.ConfigMismatch, first.on(executor).reshard(&.{first}, &.{second}, .{}));
    try std.testing.expectError(Error.Unsupported, first.on(executor).reshard(&.{first}, &.{first}, .{}));
    try std.testing.expectEqual([2]usize{ 6, 1 }, [2]usize{ (try first.describe()).size, (try second.describe()).size });

    var targets = [_]GeospatialIndex{ try GeospatialIndex.initRam(.{}, &.{}), try GeospatialIndex.initRam(.{}, &.{}) };
    defer for (&targets) |*target| target.deinit();
    var cutting: GeospatialReshardPlan = .{};
    try std.testing.expectError(Error.Cancelled, first.on(executor).reshard(&.{first}, &targets, .{ .seed = 7, .progress = .of({}, Stop.always), .plan = &cutting }));
    try std.testing.expectEqual(@as(usize, 0), (try first.describe()).size);
    try std.testing.expectEqual(@as(usize, 6), (try targets[0].describe()).size + (try targets[1].describe()).size);
    var closing: Deferred(DenseRefineReport) = undefined;
    try first.deferred(executor, &closing).reshardPlanFinish(&cutting);
    defer closing.deinit();
    _ = try closing.wait();
    try std.testing.expectEqual(GeospatialReshardPlan{}, cutting);

    _ = try targets[0].on(executor).reshard(&.{ targets[0], targets[1] }, &.{targets[0]}, .{});
    try std.testing.expectEqual(@as(usize, 6), (try targets[0].describe()).size);
}
