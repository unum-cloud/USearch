"""Similarity search over observations on and above the Earth: the index, its receivers, and its cursor.

Photos, video frames, drone footage and sensor fixes arrive as `GeospatialObservation`s, the
library's own eleven doubles, so a span of them crosses with no copy. Scores are chord metres
between two positions, altitude included, and a relation narrows which entries count: `NEAR` widens
the radius by both positions' accuracies, `SEES` and `SEEN` test the view cones, and `VISIBLE` tests
a line of sight over the `GeospatialTerrain` the handle was built with.

The verbs mirror the dense ones, and so do the receivers: verbs on the index run on the calling
thread, `index.on(executor)` blocks on that executor, and `index.deferred(executor)` submits there
and answers a `Deferred` at once. The reports, the telemetry, the phases and the insert knobs are
the dense module's own, since the C interface shares them across every modality.

File: mojo/usearch/geospatial.mojo
Author: Ash Vardanian
Date: November 16, 2026
"""

from std.ffi import c_int, c_size_t
from std.math import nan
from std.memory import stack_allocation
from std.sys.info import size_of

from usearch.dense import (
    DenseBatch,
    DenseCursorInitArguments,
    DenseCursorNextArguments,
    DenseCursorNextOutputs,
    DenseClearArguments,
    DenseDraws,
    DenseHealthArguments,
    DenseInsertConfig,
    DenseReachability,
    DenseRefinePlanFinishArguments,
    DenseRefineReport,
    DenseReshardPlanFinishArguments,
    DenseSampleArguments,
    DenseSampleConfig,
    DenseSampleOutputs,
    DenseSaveArguments,
    DenseSaveBufferArguments,
    DenseSaveBufferOutputs,
    DenseTraversalStats,
    PhaseProgress,
    RefineOptionsArguments,
    _slot,
)
from usearch.library import Library
from usearch.types import (
    Address,
    Arena,
    BytesView,
    Call,
    CallHead,
    Completion,
    CPUClass,
    Deferred,
    Distance,
    Error,
    Executor,
    GPUClass,
    Key,
    KeyBytes,
    KeysView,
    LoadPath,
    Matches,
    MatchesTapeViewArguments,
    MatchesViewArguments,
    Monolith,
    NULL_ADDRESS,
    NvmeOptions,
    Outcome,
    PredicateArguments,
    ProgressArguments,
    RamOptions,
    RowsViewArguments,
    ScalarKind,
    Traversal,
    Unwritten,
    Verb,
    VramOptions,
    _anchor,
    _block,
    _cross,
    _in_arena,
)

# region Configuration


@fieldwise_init
struct GeospatialObservation(ImplicitlyCopyable):
    """One observation as devices record it - `usearch_geospatial_observation_t`: every field in its
    natural unit, and NaN wherever nothing was recorded.

    An unknown accuracy counts as exact, an unknown view as looking everywhere without limit, and an
    unknown time as always, so a missing field only ever widens the tests that read it.
    """

    var latitude_degrees: Float64
    """Degrees north on WGS-84, south negative."""
    var longitude_degrees: Float64
    """Degrees east on WGS-84, west negative."""
    var altitude_metres: Float64
    """Metres above the WGS-84 ellipsoid; NaN means on the ground."""
    var accuracy_metres: Float64
    """One standard deviation of the horizontal position, in metres."""
    var heading_degrees: Float64
    """Where the device looked, degrees clockwise from true north."""
    var pitch_degrees: Float64
    """Where the device looked, degrees above the horizon; NaN tests the heading alone."""
    var heading_accuracy_degrees: Float64
    """One standard deviation of the heading, in degrees."""
    var field_of_view_degrees: Float64
    """The full field of view, in degrees."""
    var range_metres: Float64
    """How far the device saw, in metres along the slant."""
    var begin_seconds: Float64
    """The first instant the observation covers, in Unix seconds; NaN makes it an instant at the end."""
    var end_seconds: Float64
    """The last instant the observation covers, in Unix seconds; NaN makes it an instant at the start."""

    def __init__(out self, latitude_degrees: Float64, longitude_degrees: Float64):
        """A position alone, every other field unrecorded. Binding-only.

        Args:
            latitude_degrees: Degrees north on WGS-84, south negative.
            longitude_degrees: Degrees east on WGS-84, west negative.
        """
        self.latitude_degrees = latitude_degrees
        self.longitude_degrees = longitude_degrees
        self.altitude_metres = nan[DType.float64]()
        self.accuracy_metres = nan[DType.float64]()
        self.heading_degrees = nan[DType.float64]()
        self.pitch_degrees = nan[DType.float64]()
        self.heading_accuracy_degrees = nan[DType.float64]()
        self.field_of_view_degrees = nan[DType.float64]()
        self.range_metres = nan[DType.float64]()
        self.begin_seconds = nan[DType.float64]()
        self.end_seconds = nan[DType.float64]()


@fieldwise_init
struct GeospatialRelation(Equatable, ImplicitlyCopyable, TrivialRegisterPassable):
    """Which entries count as matches of a query - `usearch_geospatial_relation_t`; the radius and the
    time window narrow every relation alike."""

    var code: c_int
    """The raw `usearch_geospatial_relation_t` value the ABI exchanges."""
    comptime UNKNOWN = Self(0)
    """No relation named - an unset field, refused by every query."""
    comptime NEAR = Self(1)
    """Within the radius, widened by three standard deviations of both positions combined."""
    comptime VISIBLE = Self(2)
    """Near, with a clear line of sight over the handle's terrain; refused by a handle without one."""
    comptime SEES = Self(3)
    """Near, and the entry's view cone holds the query: which cameras see this place."""
    comptime SEEN = Self(4)
    """Near, and the query's view cone holds the entry: what this camera sees."""


comptime TerrainCallback = def(c_size_t, c_size_t, c_size_t, c_size_t, Address, c_size_t, Address) thin abi("C") -> None
"""`usearch_geospatial_terrain_callback_t`: the first row and column, the rows and columns, the
heights and their row stride in bytes, and the state."""


def _terrain_trampoline[
    State: AnyType, fill: def(Int, Int, Int, Int, Span[Float32, MutUntrackedOrigin], Int, State) thin -> None
](
    first_row: c_size_t,
    first_column: c_size_t,
    rows: c_size_t,
    columns: c_size_t,
    heights: Address,
    heights_stride: c_size_t,
    state: Address,
) abi("C"):
    """The C handler one Mojo fill compiles into, its window spanned again and its stride in heights."""
    var stride = Int(heights_stride) // size_of[Float32]()
    var length = 0 if rows == 0 else (Int(rows) - 1) * stride + Int(columns)
    fill(
        Int(first_row),
        Int(first_column),
        Int(rows),
        Int(columns),
        Span[Float32, MutUntrackedOrigin](
            unsafe_ptr=UnsafePointer[Float32, MutUntrackedOrigin](unsafe_from_address=heights), length=length
        ),
        stride,
        Pointer[State, MutAnyOrigin](unsafe_from_address=state)[],
    )


@fieldwise_init
struct GeospatialTerrainArguments(ImplicitlyCopyable):
    """`usearch_geospatial_terrain_t`, zeroed for none; the dataset `path`, which no residency reads
    yet, is always left empty."""

    var rows: c_size_t
    var columns: c_size_t
    var north_latitude_degrees: Float64
    var west_longitude_degrees: Float64
    var row_step_degrees: Float64
    var column_step_degrees: Float64
    var heights: Address
    var heights_stride: c_size_t
    var callback: Address
    var state: Address
    var path: Address
    var path_length: c_size_t

    def __init__(out self):
        """No terrain."""
        self.rows = c_size_t(0)
        self.columns = c_size_t(0)
        self.north_latitude_degrees = 0
        self.west_longitude_degrees = 0
        self.row_step_degrees = 0
        self.column_step_degrees = 0
        self.heights = NULL_ADDRESS
        self.heights_stride = c_size_t(0)
        self.callback = NULL_ADDRESS
        self.state = NULL_ADDRESS
        self.path = NULL_ADDRESS
        self.path_length = c_size_t(0)


@fieldwise_init
struct GeospatialTerrain[origin: ImmOrigin = ImmUntrackedOrigin](ImplicitlyCopyable):
    """The elevation every line of sight is tested over: a north-up grid of heights in metres above
    the WGS-84 ellipsoid, borrowed whole or filled on demand, which the handle reads for as long as it
    lives. A checkpoint never carries it, so every load and view names it again.

    Parameters:
        origin: The borrowed heights or fill state, named so they outlive the handle built over them.
    """

    var arguments: GeospatialTerrainArguments
    """The grid, its heights or its fill, as the engine reads them."""

    def __init__(out self):
        """No terrain, which refuses `VISIBLE`."""
        self.arguments = GeospatialTerrainArguments()

    @staticmethod
    def of[
        heights_origin: ImmOrigin, //
    ](
        rows: Int,
        columns: Int,
        north_latitude_degrees: Float64,
        west_longitude_degrees: Float64,
        row_step_degrees: Float64,
        column_step_degrees: Float64,
        heights: Span[Float32, heights_origin],
    ) -> GeospatialTerrain[heights_origin]:
        """Borrows a whole grid, row-major and packed. Binding-only.

        Parameters:
            heights_origin: The heights' origin, inferred from the argument.

        Args:
            rows: The grid's row count; rows run south from the north edge.
            columns: The grid's column count; columns run east from the west edge.
            north_latitude_degrees: The latitude of the centre of cell `[0, 0]`.
            west_longitude_degrees: The longitude of the centre of cell `[0, 0]`.
            row_step_degrees: The step between rows, positive.
            column_step_degrees: The step between columns, positive.
            heights: The `rows * columns` heights, in metres above the ellipsoid.

        Returns:
            The terrain every init, load and view takes.
        """
        var lowered = GeospatialTerrainArguments()
        lowered.rows = c_size_t(rows)
        lowered.columns = c_size_t(columns)
        lowered.north_latitude_degrees = north_latitude_degrees
        lowered.west_longitude_degrees = west_longitude_degrees
        lowered.row_step_degrees = row_step_degrees
        lowered.column_step_degrees = column_step_degrees
        lowered.heights = Int(heights.unsafe_ptr())
        return {lowered}

    @staticmethod
    def filled[
        State: AnyType,
        //,
        fill: def(Int, Int, Int, Int, Span[Float32, MutUntrackedOrigin], Int, State) thin -> None,
    ](
        rows: Int,
        columns: Int,
        north_latitude_degrees: Float64,
        west_longitude_degrees: Float64,
        row_step_degrees: Float64,
        column_step_degrees: Float64,
        state: Pointer[State, MutAnyOrigin],
    ) -> GeospatialTerrain[ImmUntrackedOrigin]:
        """Fills windows of the grid on demand, through a Mojo handler called concurrently from the
        executor's threads; VRAM handles refuse it. Binding-only.

        Parameters:
            State: The handler's state, inferred from the argument.
            fill: Writes the heights of the window at its first row and column, `rows` by `columns`,
                rows a stride of heights apart; a cell left NaN never blocks a sight line.

        Args:
            rows: The grid's row count; rows run south from the north edge.
            columns: The grid's column count; columns run east from the west edge.
            north_latitude_degrees: The latitude of the centre of cell `[0, 0]`.
            west_longitude_degrees: The longitude of the centre of cell `[0, 0]`.
            row_step_degrees: The step between rows, positive.
            column_step_degrees: The step between columns, positive.
            state: What the handler reads; it must outlive every handle built over this terrain.

        Returns:
            The terrain every init, load and view takes.
        """
        var callback: TerrainCallback = _terrain_trampoline[State, fill]
        var lowered = GeospatialTerrainArguments()
        lowered.rows = c_size_t(rows)
        lowered.columns = c_size_t(columns)
        lowered.north_latitude_degrees = north_latitude_degrees
        lowered.west_longitude_degrees = west_longitude_degrees
        lowered.row_step_degrees = row_step_degrees
        lowered.column_step_degrees = column_step_degrees
        lowered.callback = Pointer(to=callback).unsafe_bitcast[Address]()[]
        lowered.state = Int(state)
        return {lowered}


@fieldwise_init
struct GeospatialOptions[terrain_origin: ImmOrigin = ImmUntrackedOrigin](ImplicitlyCopyable):
    """The terrain and graph an empty index freezes - what `usearch_geospatial_init_ram_t` and
    `usearch_geospatial_init_vram_t` share, without the placement each names its own way.

    Parameters:
        terrain_origin: The borrowed terrain, inferred where one is named.
    """

    var terrain: GeospatialTerrain[Self.terrain_origin]
    """The elevation sight lines clear; none refuses `VISIBLE`."""
    var connectivity: c_size_t
    """The connections-per-node bound; zero derives to twelve."""
    var values_per_key: c_size_t
    """Upper bound on observations sharing one key; zero is unbounded."""

    def __init__(
        out self,
        terrain: GeospatialTerrain[Self.terrain_origin] = GeospatialTerrain[Self.terrain_origin](),
        connectivity: Int = 0,
        values_per_key: Int = 0,
    ):
        """Describes an index by its terrain, none unless told otherwise.

        Args:
            terrain: The elevation sight lines clear; none refuses `VISIBLE`.
            connectivity: The connections-per-node bound; zero derives to twelve.
            values_per_key: Upper bound on observations sharing one key; zero is unbounded.
        """
        self.terrain = terrain
        self.connectivity = c_size_t(connectivity)
        self.values_per_key = c_size_t(values_per_key)


@fieldwise_init
struct GeospatialMetadata(Defaultable, ImplicitlyCopyable):
    """The graph a checkpoint froze - `usearch_geospatial_metadata_t`, a pure output."""

    var connectivity: c_size_t
    """Connections per node the graph was built with."""

    def __init__(out self):
        """Names no graph, so an unwritten slot reads as an unanswered probe."""
        self.connectivity = c_size_t(0)


@fieldwise_init
struct GeospatialIndexDescription(Defaultable, ImplicitlyCopyable):
    """`usearch_geospatial_index_description_t`: a live index's frozen metadata and current counters."""

    var metadata: GeospatialMetadata
    """The frozen connectivity, as a checkpoint reports it."""
    var size: c_size_t
    """Stored entries."""
    var capacity: c_size_t
    """Entries hosted without growing again."""
    var memory_usage: c_size_t
    """Bytes the handle holds; traversal scratch is the executor's."""
    var cpu_class: CPUClass
    """The CPU class dispatch picked on this machine; `UNKNOWN` for a device index."""
    var gpu_class: GPUClass
    """The GPU class dispatch picked on the index's device; `UNKNOWN` for a host index."""
    var shards_count: c_size_t
    """Live shards."""
    var load_path: LoadPath
    """How the last load from a path read the parts onto the devices."""

    def __init__(out self):
        """Zeroes every field, so an unwritten slot reads as an empty index."""
        self.metadata = GeospatialMetadata()
        self.size = c_size_t(0)
        self.capacity = c_size_t(0)
        self.memory_usage = c_size_t(0)
        self.cpu_class = CPUClass.UNKNOWN
        self.gpu_class = GPUClass.UNKNOWN
        self.shards_count = c_size_t(0)
        self.load_path = LoadPath.NONE


@fieldwise_init
struct GeospatialKeyStats(Defaultable, ImplicitlyCopyable):
    """What `count` reports for one key - `usearch_geospatial_key_stats_t`."""

    var entries: c_size_t
    """Observations stored under the key; zero means the index holds none."""

    def __init__(out self):
        """Zero, which is what a failed call leaves."""
        self.entries = c_size_t(0)


@fieldwise_init
struct GeospatialObservations[origin: MutOrigin](Movable, Sized):
    """The observations `get` wrote into its arena, read in place.

    Parameters:
        origin: The arena the observations live in, which this result borrows.
    """

    var address: Address
    """The first observation, inside the arena."""
    var written: Int
    """Observations written; zero when the key is absent."""

    def __len__(self) -> Int:
        """The observations written.

        Returns:
            Zero when the key is absent.
        """
        return self.written

    def view(ref self) -> Span[GeospatialObservation, origin_of(self)]:
        """Every observation, within the rounding the index stores them at.

        Returns:
            The observations, borrowed from the arena this result reads.
        """
        return _in_arena[GeospatialObservation, origin_of(self)](self.address, self.written)


@fieldwise_init
struct GeospatialMatches[origin: MutOrigin](Movable):
    """The ragged output of `within` - `usearch_matches_tape_view_t` - read in place from its arena:
    query `i` spans matches `offsets()[i]` up to `offsets()[i + 1]` of `keys()` and `distances()`.

    Parameters:
        origin: The arena the matches live in, which this result borrows.
    """

    var view: MatchesTapeViewArguments
    """The keys, distances and offsets, as the engine wrote them."""
    var queries: Int
    """Queries the call resolved."""
    var written: Int
    """Matches written across every query."""

    def offsets(ref self) -> Span[c_size_t, origin_of(self)]:
        """One more offset than queries, as Apache Arrow lays a ragged batch out.

        Returns:
            The offsets into `keys()` and `distances()`, borrowed from the arena this result reads.
        """
        return _in_arena[c_size_t, origin_of(self)](self.view.offsets, self.queries + 1)

    def keys(ref self) -> Span[Key, origin_of(self)]:
        """Each match's entry, borrowed from the arena this result reads.

        Returns:
            One key per match.
        """
        return _in_arena[Key, origin_of(self)](self.view.keys, self.written)

    def distances(ref self) -> Span[Distance, origin_of(self)]:
        """Each match's chord metres, positionally paired with `keys()`.

        Returns:
            One distance per match.
        """
        return _in_arena[Distance, origin_of(self)](self.view.scores, self.written)

    def query_keys(ref self, query: Int) -> Span[Key, origin_of(self)]:
        """One query's keys, trimmed to the matches it found. Binding-only.

        Args:
            query: The query's offset in the batch, below `queries`.

        Returns:
            The keys, borrowed from the arena this result reads.
        """
        var begin = self._offset(query)
        return _in_arena[Key, origin_of(self)](self.view.keys + begin * size_of[Key](), self._offset(query + 1) - begin)

    def query_distances(ref self, query: Int) -> Span[Distance, origin_of(self)]:
        """One query's chord metres, paired positionally with `query_keys`. Binding-only.

        Args:
            query: The query's offset in the batch, below `queries`.

        Returns:
            The distances, borrowed from the arena this result reads.
        """
        var begin = self._offset(query)
        return _in_arena[Distance, origin_of(self)](
            self.view.scores + begin * size_of[Distance](), self._offset(query + 1) - begin
        )

    def _offset(self, position: Int) -> Int:
        """One offset of the report; a cancelled call left the offsets null."""
        if self.view.offsets == NULL_ADDRESS:
            return 0
        return Int(Pointer[c_size_t, ImmUntrackedOrigin](unsafe_from_address=self.view.offsets)[unsafe_offset=position])


struct GeospatialRefinePlan(Movable):
    """One refine in flight - `usearch_geospatial_refine_plan_t`. Binding-only.

    Lent through `GeospatialRefineConfig.plan`, it opens on the first refine, stays live past a cancel
    or the end of a round, continues on every later refine, and closes with `refine_plan_finish`.
    While it lives, the index admits no insertion and no save.
    """

    var address: Address
    """The opaque handle, `NULL_ADDRESS` until a refine opens it."""

    def __init__(out self):
        """A plan no refine opened yet."""
        self.address = NULL_ADDRESS


struct GeospatialReshardPlan(Movable):
    """One reshard in flight - `usearch_geospatial_reshard_plan_t`, with its own lists of sources and
    targets. Binding-only.

    Lent through `GeospatialReshardConfig.plan`, it opens on the first reshard, stays live past a
    cancel, moves what is still pending and continues the rounds on every later one, and closes with
    `geospatial_reshard_plan_finish`. While it lives, every index it names admits no insertion and no
    save.
    """

    var address: Address
    """The opaque handle, `NULL_ADDRESS` until a reshard opens it."""

    def __init__(out self):
        """A plan no reshard opened yet."""
        self.address = NULL_ADDRESS


# endregion Configuration

# region Configs


struct GeospatialRefineConfig[
    stats_origin: MutOrigin = MutUntrackedOrigin,
    plan_origin: MutOrigin = MutUntrackedOrigin,
    phase_origin: MutOrigin = MutUntrackedOrigin,
](ImplicitlyCopyable):
    """The knobs of `usearch_geospatial_refine_t`; zeroed or absent ones derive what the C interface derives.

    Parameters:
        stats_origin: The lent telemetry slot's origin, inferred where one is lent.
        plan_origin: The lent plan's origin, inferred where one is lent.
        phase_origin: The lent phase slot's origin, inferred where one is lent.
    """

    var expansion_factor: Int
    """The breadth of any walk beside the round; zero derives."""
    var memory_limit: Int
    """The bytes a loaded NVMe index holds for the one part it rewrites; zero derives."""
    var progress: ProgressArguments
    """Hears every phase's units at its barriers; a cancel stops the round at that barrier."""
    var stats: Optional[Pointer[DenseTraversalStats, Self.stats_origin]]
    """Receives this call's traversal telemetry, fully overwritten; absent skips reporting."""
    var plan: Optional[Pointer[GeospatialRefinePlan, Self.plan_origin]]
    """The plan this call opens or continues; absent runs one round to its end."""
    var phase: Optional[Pointer[PhaseProgress, Self.phase_origin]]
    """Receives where this call stopped; absent skips it."""

    def __init__(
        out self,
        *,
        expansion_factor: Int = 0,
        memory_limit: Int = 0,
        progress: ProgressArguments = ProgressArguments(),
        stats: Optional[Pointer[DenseTraversalStats, Self.stats_origin]] = Optional[
            Pointer[DenseTraversalStats, Self.stats_origin]
        ](),
        plan: Optional[Pointer[GeospatialRefinePlan, Self.plan_origin]] = Optional[
            Pointer[GeospatialRefinePlan, Self.plan_origin]
        ](),
        phase: Optional[Pointer[PhaseProgress, Self.phase_origin]] = Optional[
            Pointer[PhaseProgress, Self.phase_origin]
        ](),
    ):
        """Names only the knobs that differ from the derived ones.

        Args:
            expansion_factor: The breadth of any walk beside the round; zero derives.
            memory_limit: The bytes a loaded NVMe index holds for the one part it rewrites; zero derives.
            progress: Hears every phase's units at its barriers; a cancel stops the round at that barrier.
            stats: Receives this call's traversal telemetry; absent skips reporting.
            plan: The plan this call opens or continues; absent runs one round to its end.
            phase: Receives where this call stopped; absent skips it.
        """
        self.expansion_factor = expansion_factor
        self.memory_limit = memory_limit
        self.progress = progress
        self.stats = stats
        self.plan = plan
        self.phase = phase


struct GeospatialReshardConfig[
    stats_origin: MutOrigin = MutUntrackedOrigin,
    plan_origin: MutOrigin = MutUntrackedOrigin,
    phase_origin: MutOrigin = MutUntrackedOrigin,
](ImplicitlyCopyable):
    """The knobs of `usearch_geospatial_reshard_t`; zeroed or absent ones derive what the C interface derives.

    Parameters:
        stats_origin: The lent telemetry slot's origin, inferred where one is lent.
        plan_origin: The lent plan's origin, inferred where one is lent.
        phase_origin: The lent phase slot's origin, inferred where one is lent.
    """

    var expansion_factor: Int
    """The breadth of any walk beside the closing rounds; zero derives."""
    var memory_limit: Int
    """The bytes a loaded NVMe index holds for the one part it rewrites; zero derives."""
    var seed: UInt64
    """Picks the part roots of a source cut among several targets; equal seeds cut equally."""
    var progress: ProgressArguments
    """Hears the closing rounds' phases at their barriers; a cancel stops them, never a move."""
    var stats: Optional[Pointer[DenseTraversalStats, Self.stats_origin]]
    """Receives this call's traversal telemetry, fully overwritten; absent skips reporting."""
    var plan: Optional[Pointer[GeospatialReshardPlan, Self.plan_origin]]
    """The plan this call opens or continues; absent moves and runs to the end."""
    var phase: Optional[Pointer[PhaseProgress, Self.phase_origin]]
    """Receives where this call stopped; absent skips it."""

    def __init__(
        out self,
        *,
        expansion_factor: Int = 0,
        memory_limit: Int = 0,
        seed: UInt64 = 0,
        progress: ProgressArguments = ProgressArguments(),
        stats: Optional[Pointer[DenseTraversalStats, Self.stats_origin]] = Optional[
            Pointer[DenseTraversalStats, Self.stats_origin]
        ](),
        plan: Optional[Pointer[GeospatialReshardPlan, Self.plan_origin]] = Optional[
            Pointer[GeospatialReshardPlan, Self.plan_origin]
        ](),
        phase: Optional[Pointer[PhaseProgress, Self.phase_origin]] = Optional[
            Pointer[PhaseProgress, Self.phase_origin]
        ](),
    ):
        """Names only the knobs that differ from the derived ones.

        Args:
            expansion_factor: The breadth of any walk beside the closing rounds; zero derives.
            memory_limit: The bytes a loaded NVMe index holds for the one part it rewrites; zero derives.
            seed: Picks the part roots of a source cut among several targets.
            progress: Hears the closing rounds' phases at their barriers; a cancel stops them, never a move.
            stats: Receives this call's traversal telemetry; absent skips reporting.
            plan: The plan this call opens or continues; absent moves and runs to the end.
            phase: Receives where this call stopped; absent skips it.
        """
        self.expansion_factor = expansion_factor
        self.memory_limit = memory_limit
        self.seed = seed
        self.progress = progress
        self.stats = stats
        self.plan = plan
        self.phase = phase


struct GeospatialSearchConfig[stats_origin: MutOrigin = MutUntrackedOrigin](ImplicitlyCopyable):
    """The knobs of `usearch_geospatial_search_t`; zeroed or absent ones derive what the C interface
    derives.

    The relation, the time window and the `predicate` narrow the walk's candidates, so a selective one
    wants a wider `expansion_factor`, and a query may hold fewer than `wanted` matches.

    Parameters:
        stats_origin: The lent telemetry slot's origin, inferred where one is lent.
    """

    var relation: GeospatialRelation
    """Which entries count as matches of a query."""
    var metres: Float64
    """The radius in metres; zero ranks by distance alone."""
    var seconds: Float64
    """Slack added to both ends of the query's time interval."""
    var expansion_factor: Int
    """The breadth of the candidates the walk explores; zero derives."""
    var traversal: Traversal
    """A walk, or an exact scan of every entry on RAM and NVMe handles; VRAM answers `UNSUPPORTED`."""
    var predicate: PredicateArguments
    """The handler and state judging which candidates are reported; the default admits all."""
    var progress: ProgressArguments
    """Hears the queries resolved so far; a cancel stops the search with no matches."""
    var stats: Optional[Pointer[DenseTraversalStats, Self.stats_origin]]
    """Receives this call's traversal telemetry, fully overwritten; absent skips reporting."""

    def __init__(
        out self,
        *,
        relation: GeospatialRelation = GeospatialRelation.NEAR,
        metres: Float64 = 0,
        seconds: Float64 = 0,
        expansion_factor: Int = 0,
        traversal: Traversal = Traversal.WALK,
        predicate: PredicateArguments = PredicateArguments(),
        progress: ProgressArguments = ProgressArguments(),
        stats: Optional[Pointer[DenseTraversalStats, Self.stats_origin]] = Optional[
            Pointer[DenseTraversalStats, Self.stats_origin]
        ](),
    ):
        """Names only the knobs that differ from the derived ones.

        Args:
            relation: Which entries count as matches of a query.
            metres: The radius in metres; zero ranks by distance alone.
            seconds: Slack added to both ends of the query's time interval.
            expansion_factor: The breadth of the candidates the walk explores; zero derives.
            traversal: A walk, or an exact scan of every entry on RAM and NVMe handles.
            predicate: The handler and state judging which candidates are reported; the default admits all.
            progress: Hears the queries resolved so far; a cancel stops the search with no matches.
            stats: Receives this call's traversal telemetry; absent skips reporting.
        """
        self.relation = relation
        self.metres = metres
        self.seconds = seconds
        self.expansion_factor = expansion_factor
        self.traversal = traversal
        self.predicate = predicate
        self.progress = progress
        self.stats = stats


struct GeospatialWithinConfig[stats_origin: MutOrigin = MutUntrackedOrigin](ImplicitlyCopyable):
    """The knobs of `usearch_geospatial_within_t` beside its radius; zeroed or absent ones derive what
    the C interface derives.

    Containment sizes its own candidates from the cone graph's spanner bound, so it takes no breadth a
    caller could make the report incomplete with.

    Parameters:
        stats_origin: The lent telemetry slot's origin, inferred where one is lent.
    """

    var relation: GeospatialRelation
    """Which entries count as matches of a query."""
    var seconds: Float64
    """Slack added to both ends of the query's time interval."""
    var traversal: Traversal
    """A complete walk, or an exact scan of every entry on RAM and NVMe handles."""
    var predicate: PredicateArguments
    """The handler and state judging which entries are reported; the default admits all."""
    var progress: ProgressArguments
    """Hears the queries resolved so far; a cancel stops the call with no matches."""
    var stats: Optional[Pointer[DenseTraversalStats, Self.stats_origin]]
    """Receives this call's traversal telemetry, fully overwritten; absent skips reporting."""

    def __init__(
        out self,
        *,
        relation: GeospatialRelation = GeospatialRelation.NEAR,
        seconds: Float64 = 0,
        traversal: Traversal = Traversal.WALK,
        predicate: PredicateArguments = PredicateArguments(),
        progress: ProgressArguments = ProgressArguments(),
        stats: Optional[Pointer[DenseTraversalStats, Self.stats_origin]] = Optional[
            Pointer[DenseTraversalStats, Self.stats_origin]
        ](),
    ):
        """Names only the knobs that differ from the derived ones.

        Args:
            relation: Which entries count as matches of a query.
            seconds: Slack added to both ends of the query's time interval.
            traversal: A complete walk, or an exact scan of every entry on RAM and NVMe handles.
            predicate: The handler and state judging which entries are reported; the default admits all.
            progress: Hears the queries resolved so far; a cancel stops the call with no matches.
            stats: Receives this call's traversal telemetry; absent skips reporting.
        """
        self.relation = relation
        self.seconds = seconds
        self.traversal = traversal
        self.predicate = predicate
        self.progress = progress
        self.stats = stats


# endregion Configs

# region Arguments


@fieldwise_init
struct GeospatialInitRamArguments(ImplicitlyCopyable):
    """`usearch_geospatial_init_ram_t`."""

    var index: Pointer[Address, MutUntrackedOrigin]
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var terrain: GeospatialTerrainArguments
    var connectivity: c_size_t
    var values_per_key: c_size_t
    var ram: RamOptions[ImmUntrackedOrigin]


@fieldwise_init
struct GeospatialInitVramArguments(ImplicitlyCopyable):
    """`usearch_geospatial_init_vram_t`."""

    var index: Pointer[Address, MutUntrackedOrigin]
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var terrain: GeospatialTerrainArguments
    var connectivity: c_size_t
    var values_per_key: c_size_t
    var vram: VramOptions[ImmUntrackedOrigin]


@fieldwise_init
struct GeospatialLoadRamArguments(ImplicitlyCopyable):
    """`usearch_geospatial_load_ram_t`, and `usearch_geospatial_load_ram_buffer_t` with `source` a buffer."""

    var index: Pointer[Address, MutUntrackedOrigin]
    var executor: Address
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var completion: Completion
    var source: BytesView[ImmUntrackedOrigin]
    var ram: RamOptions[ImmUntrackedOrigin]
    var values_per_key: c_size_t
    var terrain: GeospatialTerrainArguments
    var progress: ProgressArguments


@fieldwise_init
struct GeospatialLoadVramArguments(ImplicitlyCopyable):
    """`usearch_geospatial_load_vram_t`, and `usearch_geospatial_load_vram_buffer_t` with `source` a buffer."""

    var index: Pointer[Address, MutUntrackedOrigin]
    var executor: Address
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var completion: Completion
    var source: BytesView[ImmUntrackedOrigin]
    var vram: VramOptions[ImmUntrackedOrigin]
    var values_per_key: c_size_t
    var terrain: GeospatialTerrainArguments
    var progress: ProgressArguments


@fieldwise_init
struct GeospatialViewRamArguments(ImplicitlyCopyable):
    """`usearch_geospatial_view_ram_t`."""

    var index: Pointer[Address, MutUntrackedOrigin]
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var path: BytesView[ImmUntrackedOrigin]
    var terrain: GeospatialTerrainArguments
    var ram: RamOptions[ImmUntrackedOrigin]


@fieldwise_init
struct GeospatialViewNvmeArguments(ImplicitlyCopyable):
    """`usearch_geospatial_view_nvme_t`."""

    var index: Pointer[Address, MutUntrackedOrigin]
    var executor: Address
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var completion: Completion
    var path: BytesView[ImmUntrackedOrigin]
    var nvme: NvmeOptions[ImmUntrackedOrigin]
    var terrain: GeospatialTerrainArguments


@fieldwise_init
struct GeospatialReserveArguments(ImplicitlyCopyable):
    """`usearch_geospatial_reserve_t`."""

    var index: Address
    var executor: Address
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var completion: Completion
    var entries: c_size_t


@fieldwise_init
struct GeospatialDescribeArguments(ImplicitlyCopyable):
    """`usearch_geospatial_describe_t`."""

    var index: Address
    var executor: Address
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var completion: Completion
    var description: Pointer[GeospatialIndexDescription, MutUntrackedOrigin]


@fieldwise_init
struct GeospatialInsertArguments(ImplicitlyCopyable):
    """`usearch_geospatial_insert_t`."""

    var index: Address
    var executor: Address
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var completion: Completion
    var count: c_size_t
    var keys: KeysView[ImmUntrackedOrigin]
    var observations: Address
    var expansion_factor: c_size_t
    var progress: ProgressArguments
    var stats: Optional[Pointer[DenseTraversalStats, MutUntrackedOrigin]]


@fieldwise_init
struct GeospatialRefineArguments(ImplicitlyCopyable):
    """`usearch_geospatial_refine_t`."""

    var index: Address
    var executor: Address
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var completion: Completion
    var report: Pointer[DenseRefineReport, MutUntrackedOrigin]
    var refine: RefineOptionsArguments
    var progress: ProgressArguments
    var stats: Optional[Pointer[DenseTraversalStats, MutUntrackedOrigin]]
    var plan: Optional[Pointer[GeospatialRefinePlan, MutUntrackedOrigin]]
    var phase: Optional[Pointer[PhaseProgress, MutUntrackedOrigin]]


@fieldwise_init
struct GeospatialToleranceArguments(ImplicitlyCopyable, TrivialRegisterPassable):
    """`usearch_geospatial_tolerance_t`."""

    var metres: Float64
    var seconds: Float64


@fieldwise_init
struct GeospatialSearchArguments(ImplicitlyCopyable):
    """`usearch_geospatial_search_t`."""

    var index: Address
    var executor: Address
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var completion: Completion
    var count: c_size_t
    var queries: Address
    var relation: GeospatialRelation
    var tolerance: GeospatialToleranceArguments
    var matches: Pointer[MatchesViewArguments, MutUntrackedOrigin]
    var arena: Address
    var wanted: c_size_t
    var expansion_factor: c_size_t
    var traversal: Traversal
    var predicate: PredicateArguments
    var progress: ProgressArguments
    var stats: Optional[Pointer[DenseTraversalStats, MutUntrackedOrigin]]


@fieldwise_init
struct GeospatialWithinArguments(ImplicitlyCopyable):
    """`usearch_geospatial_within_t`."""

    var index: Address
    var executor: Address
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var completion: Completion
    var count: c_size_t
    var queries: Address
    var relation: GeospatialRelation
    var tolerance: GeospatialToleranceArguments
    var matches: Pointer[MatchesTapeViewArguments, MutUntrackedOrigin]
    var written: Pointer[c_size_t, MutUntrackedOrigin]
    var arena: Address
    var traversal: Traversal
    var predicate: PredicateArguments
    var progress: ProgressArguments
    var stats: Optional[Pointer[DenseTraversalStats, MutUntrackedOrigin]]


@fieldwise_init
struct GeospatialCountArguments(ImplicitlyCopyable):
    """`usearch_geospatial_count_t`."""

    var index: Address
    var executor: Address
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var completion: Completion
    var key: KeyBytes
    var key_stats: Pointer[GeospatialKeyStats, MutUntrackedOrigin]


@fieldwise_init
struct GeospatialGetArguments(ImplicitlyCopyable):
    """`usearch_geospatial_get_t`."""

    var index: Address
    var executor: Address
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var completion: Completion
    var key: KeyBytes
    var observations: Pointer[Address, MutUntrackedOrigin]
    var written: Pointer[c_size_t, MutUntrackedOrigin]
    var arena: Address


@fieldwise_init
struct GeospatialReshardArguments(ImplicitlyCopyable):
    """`usearch_geospatial_reshard_t`."""

    var sources_count: c_size_t
    var sources: Pointer[Address, ImmUntrackedOrigin]
    var targets_count: c_size_t
    var targets: Pointer[Address, ImmUntrackedOrigin]
    var executor: Address
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var completion: Completion
    var report: Pointer[DenseRefineReport, MutUntrackedOrigin]
    var refine: RefineOptionsArguments
    var seed: UInt64
    var progress: ProgressArguments
    var stats: Optional[Pointer[DenseTraversalStats, MutUntrackedOrigin]]
    var plan: Optional[Pointer[GeospatialReshardPlan, MutUntrackedOrigin]]
    var phase: Optional[Pointer[PhaseProgress, MutUntrackedOrigin]]


@fieldwise_init
struct GeospatialMetadataReadArguments(ImplicitlyCopyable):
    """`usearch_geospatial_metadata_read_t`, and `usearch_geospatial_metadata_read_buffer_t` with `source`
    a buffer."""

    var executor: Address
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var completion: Completion
    var source: BytesView[ImmUntrackedOrigin]
    var metadata: Pointer[GeospatialMetadata, MutUntrackedOrigin]


comptime GeospatialInitRam = Verb[GeospatialInitRamArguments]
comptime GeospatialInitVram = Verb[GeospatialInitVramArguments]
comptime GeospatialLoadRam = Verb[GeospatialLoadRamArguments]
comptime GeospatialLoadVram = Verb[GeospatialLoadVramArguments]
comptime GeospatialViewRam = Verb[GeospatialViewRamArguments]
comptime GeospatialViewNvme = Verb[GeospatialViewNvmeArguments]
comptime GeospatialReserve = Verb[GeospatialReserveArguments]
comptime GeospatialClear = Verb[DenseClearArguments]
comptime GeospatialDescribe = Verb[GeospatialDescribeArguments]
comptime GeospatialInsert = Verb[GeospatialInsertArguments]
comptime GeospatialRefine = Verb[GeospatialRefineArguments]
comptime GeospatialRefinePlanFinish = Verb[DenseRefinePlanFinishArguments]
comptime GeospatialHealth = Verb[DenseHealthArguments]
comptime GeospatialReshard = Verb[GeospatialReshardArguments]
comptime GeospatialReshardPlanFinish = Verb[DenseReshardPlanFinishArguments]
comptime GeospatialSample = Verb[DenseSampleArguments]
comptime GeospatialSearch = Verb[GeospatialSearchArguments]
comptime GeospatialWithin = Verb[GeospatialWithinArguments]
comptime GeospatialCount = Verb[GeospatialCountArguments]
comptime GeospatialGet = Verb[GeospatialGetArguments]
comptime GeospatialCursorInit = Verb[DenseCursorInitArguments]
comptime GeospatialCursorNext = Verb[DenseCursorNextArguments]
comptime GeospatialSave = Verb[DenseSaveArguments]
comptime GeospatialSaveBuffer = Verb[DenseSaveBufferArguments]
comptime GeospatialMetadataRead = Verb[GeospatialMetadataReadArguments]


# endregion Arguments

# region Calls


@fieldwise_init
struct _GeospatialReserveCall(Call):
    """The crossing of `usearch_geospatial_reserve`."""

    comptime Arguments = GeospatialReserveArguments
    comptime Outputs = Unwritten
    comptime Result = NoneType
    comptime symbol: StaticString = "usearch_geospatial_reserve"
    var entries: Int

    @staticmethod
    def verb(library: Library) -> Verb[Self.Arguments]:
        return library.symbols().geospatial_reserve

    def arguments(self, head: CallHead, outputs: Pointer[Self.Outputs, MutUntrackedOrigin]) -> Self.Arguments:
        return {
            head.index,
            head.executor,
            head.outcome,
            head.completion,
            c_size_t(self.entries),
        }

    def conclude(deinit self, outputs: Unwritten) -> NoneType:
        return None


@fieldwise_init
struct _GeospatialClearCall(Call):
    """The crossing of `usearch_geospatial_clear`."""

    comptime Arguments = DenseClearArguments
    comptime Outputs = Unwritten
    comptime Result = NoneType
    comptime symbol: StaticString = "usearch_geospatial_clear"

    @staticmethod
    def verb(library: Library) -> Verb[Self.Arguments]:
        return library.symbols().geospatial_clear

    def arguments(self, head: CallHead, outputs: Pointer[Self.Outputs, MutUntrackedOrigin]) -> Self.Arguments:
        return {head.index, head.executor, head.outcome, head.completion}

    def conclude(deinit self, outputs: Unwritten) -> NoneType:
        return None


@fieldwise_init
struct _GeospatialDescribeCall(Call):
    """The crossing of `usearch_geospatial_describe`."""

    comptime Arguments = GeospatialDescribeArguments
    comptime Outputs = GeospatialIndexDescription
    comptime Result = GeospatialIndexDescription
    comptime symbol: StaticString = "usearch_geospatial_describe"

    @staticmethod
    def verb(library: Library) -> Verb[Self.Arguments]:
        return library.symbols().geospatial_describe

    def arguments(self, head: CallHead, outputs: Pointer[Self.Outputs, MutUntrackedOrigin]) -> Self.Arguments:
        return {head.index, head.executor, head.outcome, head.completion, outputs}

    def conclude(deinit self, outputs: GeospatialIndexDescription) -> GeospatialIndexDescription:
        return outputs


@fieldwise_init
struct _GeospatialCountCall(Call):
    """The crossing of `usearch_geospatial_count`."""

    comptime Arguments = GeospatialCountArguments
    comptime Outputs = GeospatialKeyStats
    comptime Result = GeospatialKeyStats
    comptime symbol: StaticString = "usearch_geospatial_count"
    var key: Key

    @staticmethod
    def verb(library: Library) -> Verb[Self.Arguments]:
        return library.symbols().geospatial_count

    def arguments(self, head: CallHead, outputs: Pointer[Self.Outputs, MutUntrackedOrigin]) -> Self.Arguments:
        return {head.index, head.executor, head.outcome, head.completion, KeyBytes.of(self.key), outputs}

    def conclude(deinit self, outputs: GeospatialKeyStats) -> GeospatialKeyStats:
        return outputs


@fieldwise_init
struct GeospatialGetOutputs(Defaultable, ImplicitlyCopyable):
    """What `usearch_geospatial_get` writes through its pointers."""

    var observations: Address
    var written: c_size_t

    def __init__(out self):
        """Nothing exported."""
        self.observations = NULL_ADDRESS
        self.written = c_size_t(0)


@fieldwise_init
struct _GeospatialGetCall[origin: MutOrigin](Call):
    """The crossing of `usearch_geospatial_get`, which borrows the arena its observations land in."""

    comptime Arguments = GeospatialGetArguments
    comptime Outputs = GeospatialGetOutputs
    comptime Result = GeospatialObservations[Self.origin]
    comptime symbol: StaticString = "usearch_geospatial_get"
    var key: Key
    var arena: Address

    @staticmethod
    def verb(library: Library) -> Verb[Self.Arguments]:
        return library.symbols().geospatial_get

    def arguments(self, head: CallHead, outputs: Pointer[Self.Outputs, MutUntrackedOrigin]) -> Self.Arguments:
        return {
            head.index,
            head.executor,
            head.outcome,
            head.completion,
            KeyBytes.of(self.key),
            Pointer(to=outputs[].observations),
            Pointer(to=outputs[].written),
            self.arena,
        }

    def conclude(deinit self, outputs: GeospatialGetOutputs) -> Self.Result:
        return {outputs.observations, Int(outputs.written)}


@fieldwise_init
struct _GeospatialInsertCall(Call):
    """The crossing of `usearch_geospatial_insert`."""

    comptime Arguments = GeospatialInsertArguments
    comptime Outputs = Unwritten
    comptime Result = NoneType
    comptime symbol: StaticString = "usearch_geospatial_insert"
    var keys: KeysView[ImmUntrackedOrigin]
    var keys_count: Int
    var observations: Address
    var count: Int
    var expansion_factor: Int
    var progress: ProgressArguments
    var stats: Optional[Pointer[DenseTraversalStats, MutUntrackedOrigin]]

    @staticmethod
    def verb(library: Library) -> Verb[Self.Arguments]:
        return library.symbols().geospatial_insert

    def arguments(self, head: CallHead, outputs: Pointer[Self.Outputs, MutUntrackedOrigin]) -> Self.Arguments:
        return {
            head.index,
            head.executor,
            head.outcome,
            head.completion,
            c_size_t(self.count),
            self.keys,
            self.observations,
            c_size_t(self.expansion_factor),
            self.progress,
            self.stats,
        }

    def lengths_agree(self) -> Bool:
        return self.keys_count == self.count

    def conclude(deinit self, outputs: Unwritten) -> NoneType:
        return None


@fieldwise_init
struct _GeospatialRefineCall(Call):
    """The crossing of `usearch_geospatial_refine`."""

    comptime Arguments = GeospatialRefineArguments
    comptime Outputs = DenseRefineReport
    comptime Result = DenseRefineReport
    comptime symbol: StaticString = "usearch_geospatial_refine"
    var refine: RefineOptionsArguments
    var progress: ProgressArguments
    var stats: Optional[Pointer[DenseTraversalStats, MutUntrackedOrigin]]
    var plan: Optional[Pointer[GeospatialRefinePlan, MutUntrackedOrigin]]
    var phase: Optional[Pointer[PhaseProgress, MutUntrackedOrigin]]

    @staticmethod
    def verb(library: Library) -> Verb[Self.Arguments]:
        return library.symbols().geospatial_refine

    def arguments(self, head: CallHead, outputs: Pointer[Self.Outputs, MutUntrackedOrigin]) -> Self.Arguments:
        return {
            head.index,
            head.executor,
            head.outcome,
            head.completion,
            outputs,
            self.refine,
            self.progress,
            self.stats,
            self.plan,
            self.phase,
        }

    def conclude(deinit self, outputs: DenseRefineReport) -> DenseRefineReport:
        return outputs


@fieldwise_init
struct _GeospatialRefinePlanFinishCall(Call):
    """The crossing of `usearch_geospatial_refine_plan_finish`, which frees the plan whatever it answers."""

    comptime Arguments = DenseRefinePlanFinishArguments
    comptime Outputs = DenseRefineReport
    comptime Result = DenseRefineReport
    comptime symbol: StaticString = "usearch_geospatial_refine_plan_finish"
    var plan: Address

    @staticmethod
    def verb(library: Library) -> Verb[Self.Arguments]:
        return library.symbols().geospatial_refine_plan_finish

    def arguments(self, head: CallHead, outputs: Pointer[Self.Outputs, MutUntrackedOrigin]) -> Self.Arguments:
        return {head.index, head.executor, head.outcome, head.completion, self.plan, outputs}

    def conclude(deinit self, outputs: DenseRefineReport) -> DenseRefineReport:
        return outputs


@fieldwise_init
struct _GeospatialHealthCall(Call):
    """The crossing of `usearch_geospatial_health`."""

    comptime Arguments = DenseHealthArguments
    comptime Outputs = DenseReachability
    comptime Result = DenseReachability
    comptime symbol: StaticString = "usearch_geospatial_health"

    @staticmethod
    def verb(library: Library) -> Verb[Self.Arguments]:
        return library.symbols().geospatial_health

    def arguments(self, head: CallHead, outputs: Pointer[Self.Outputs, MutUntrackedOrigin]) -> Self.Arguments:
        return {head.index, head.executor, head.outcome, head.completion, outputs}

    def conclude(deinit self, outputs: DenseReachability) -> DenseReachability:
        return outputs


struct _GeospatialReshardCall(Call):
    """The crossing of `usearch_geospatial_reshard`, which keeps the handle lists on the heap until the
    call is final."""

    comptime Arguments = GeospatialReshardArguments
    comptime Outputs = DenseRefineReport
    comptime Result = DenseRefineReport
    comptime symbol: StaticString = "usearch_geospatial_reshard"
    var sources: List[Address]
    var targets: List[Address]
    var refine: RefineOptionsArguments
    var seed: UInt64
    var progress: ProgressArguments
    var stats: Optional[Pointer[DenseTraversalStats, MutUntrackedOrigin]]
    var plan: Optional[Pointer[GeospatialReshardPlan, MutUntrackedOrigin]]
    var phase: Optional[Pointer[PhaseProgress, MutUntrackedOrigin]]

    def __init__[
        sources_origin: ImmOrigin,
        targets_origin: ImmOrigin,
        stats_origin: MutOrigin,
        plan_origin: MutOrigin,
        phase_origin: MutOrigin,
        //,
    ](
        out self,
        sources: Span[GeospatialIndex, sources_origin],
        targets: Span[GeospatialIndex, targets_origin],
        config: GeospatialReshardConfig[stats_origin, plan_origin, phase_origin],
    ):
        self.sources = [sources[position].address for position in range(len(sources))]
        self.targets = [targets[position].address for position in range(len(targets))]
        self.refine = RefineOptionsArguments(c_size_t(config.expansion_factor), c_size_t(config.memory_limit))
        self.seed = config.seed
        self.progress = config.progress
        self.stats = _slot(config.stats)
        self.plan = _slot(config.plan)
        self.phase = _slot(config.phase)

    @staticmethod
    def verb(library: Library) -> Verb[Self.Arguments]:
        return library.symbols().geospatial_reshard

    def arguments(self, head: CallHead, outputs: Pointer[Self.Outputs, MutUntrackedOrigin]) -> Self.Arguments:
        return {
            c_size_t(len(self.sources)),
            Pointer[Address, ImmUntrackedOrigin](unsafe_from_address=Int(self.sources.unsafe_ptr())),
            c_size_t(len(self.targets)),
            Pointer[Address, ImmUntrackedOrigin](unsafe_from_address=Int(self.targets.unsafe_ptr())),
            head.executor,
            head.outcome,
            head.completion,
            outputs,
            self.refine,
            self.seed,
            self.progress,
            self.stats,
            self.plan,
            self.phase,
        }

    def conclude(deinit self, outputs: DenseRefineReport) -> DenseRefineReport:
        return outputs


@fieldwise_init
struct _GeospatialReshardPlanFinishCall(Call):
    """The crossing of `usearch_geospatial_reshard_plan_finish`, which frees the plan whatever it answers."""

    comptime Arguments = DenseReshardPlanFinishArguments
    comptime Outputs = DenseRefineReport
    comptime Result = DenseRefineReport
    comptime symbol: StaticString = "usearch_geospatial_reshard_plan_finish"
    var plan: Address

    @staticmethod
    def verb(library: Library) -> Verb[Self.Arguments]:
        return library.symbols().geospatial_reshard_plan_finish

    def arguments(self, head: CallHead, outputs: Pointer[Self.Outputs, MutUntrackedOrigin]) -> Self.Arguments:
        return {head.executor, head.outcome, head.completion, self.plan, outputs}

    def conclude(deinit self, outputs: DenseRefineReport) -> DenseRefineReport:
        return outputs


@fieldwise_init
struct _GeospatialSampleCall[origin: MutOrigin](Call):
    """The crossing of `usearch_geospatial_sample`, which borrows the arena its draws are written in."""

    comptime Arguments = DenseSampleArguments
    comptime Outputs = DenseSampleOutputs
    comptime Result = DenseDraws[Self.origin]
    comptime symbol: StaticString = "usearch_geospatial_sample"
    var shown: KeysView[ImmUntrackedOrigin]
    var shown_count: Int
    var shown_coordinates: RowsViewArguments
    var shown_coordinates_count: Int
    var focus: KeysView[ImmUntrackedOrigin]
    var focus_count: Int
    var wanted: Int
    var seed: UInt64
    var progress: ProgressArguments
    var arena: Address

    @staticmethod
    def verb(library: Library) -> Verb[Self.Arguments]:
        return library.symbols().geospatial_sample

    def arguments(self, head: CallHead, outputs: Pointer[Self.Outputs, MutUntrackedOrigin]) -> Self.Arguments:
        return {
            head.index,
            head.executor,
            head.outcome,
            head.completion,
            c_size_t(self.shown_count),
            self.shown,
            self.shown_coordinates,
            c_size_t(self.focus_count),
            self.focus,
            Pointer(to=outputs[].keys),
            Pointer(to=outputs[].entry_ordinals),
            Pointer(to=outputs[].parents),
            Pointer(to=outputs[].coordinates),
            Pointer(to=outputs[].written),
            self.arena,
            c_size_t(self.wanted),
            self.seed,
            c_size_t(0),
            self.progress,
            None,
        }

    def lengths_agree(self) -> Bool:
        return self.shown_coordinates_count == 3 * self.shown_count

    def conclude(deinit self, outputs: DenseSampleOutputs) -> Self.Result:
        return {outputs}


@fieldwise_init
struct _GeospatialSearchCall[origin: MutOrigin](Call):
    """The crossing of `usearch_geospatial_search`, which borrows the arena its matches land in."""

    comptime Arguments = GeospatialSearchArguments
    comptime Outputs = MatchesViewArguments
    comptime Result = Matches[Self.origin]
    comptime symbol: StaticString = "usearch_geospatial_search"
    var queries: Address
    var count: Int
    var relation: GeospatialRelation
    var tolerance: GeospatialToleranceArguments
    var wanted: Int
    var arena: Address
    var expansion_factor: Int
    var traversal: Traversal
    var predicate: PredicateArguments
    var progress: ProgressArguments
    var stats: Optional[Pointer[DenseTraversalStats, MutUntrackedOrigin]]

    @staticmethod
    def verb(library: Library) -> Verb[Self.Arguments]:
        return library.symbols().geospatial_search

    def arguments(self, head: CallHead, outputs: Pointer[Self.Outputs, MutUntrackedOrigin]) -> Self.Arguments:
        return {
            head.index,
            head.executor,
            head.outcome,
            head.completion,
            c_size_t(self.count),
            self.queries,
            self.relation,
            self.tolerance,
            outputs,
            self.arena,
            c_size_t(self.wanted),
            c_size_t(self.expansion_factor),
            self.traversal,
            self.predicate,
            self.progress,
            self.stats,
        }

    def conclude(deinit self, outputs: MatchesViewArguments) -> Self.Result:
        return {outputs, self.count}


@fieldwise_init
struct GeospatialWithinOutputs(Defaultable, ImplicitlyCopyable):
    """What `usearch_geospatial_within` writes through its pointers."""

    var matches: MatchesTapeViewArguments
    var written: c_size_t

    def __init__(out self):
        """No matches."""
        self.matches = MatchesTapeViewArguments()
        self.written = c_size_t(0)


@fieldwise_init
struct _GeospatialWithinCall[origin: MutOrigin](Call):
    """The crossing of `usearch_geospatial_within`, which borrows the arena its report lands in."""

    comptime Arguments = GeospatialWithinArguments
    comptime Outputs = GeospatialWithinOutputs
    comptime Result = GeospatialMatches[Self.origin]
    comptime symbol: StaticString = "usearch_geospatial_within"
    var queries: Address
    var count: Int
    var relation: GeospatialRelation
    var tolerance: GeospatialToleranceArguments
    var arena: Address
    var traversal: Traversal
    var predicate: PredicateArguments
    var progress: ProgressArguments
    var stats: Optional[Pointer[DenseTraversalStats, MutUntrackedOrigin]]

    @staticmethod
    def verb(library: Library) -> Verb[Self.Arguments]:
        return library.symbols().geospatial_within

    def arguments(self, head: CallHead, outputs: Pointer[Self.Outputs, MutUntrackedOrigin]) -> Self.Arguments:
        return {
            head.index,
            head.executor,
            head.outcome,
            head.completion,
            c_size_t(self.count),
            self.queries,
            self.relation,
            self.tolerance,
            Pointer(to=outputs[].matches),
            Pointer(to=outputs[].written),
            self.arena,
            self.traversal,
            self.predicate,
            self.progress,
            self.stats,
        }

    def conclude(deinit self, outputs: GeospatialWithinOutputs) -> Self.Result:
        return {outputs.matches, self.count, Int(outputs.written)}


@fieldwise_init
struct _GeospatialSaveCall(Call):
    """The crossing of `usearch_geospatial_save`."""

    comptime Arguments = DenseSaveArguments
    comptime Outputs = Unwritten
    comptime Result = NoneType
    comptime symbol: StaticString = "usearch_geospatial_save"
    var path: BytesView[ImmUntrackedOrigin]

    @staticmethod
    def verb(library: Library) -> Verb[Self.Arguments]:
        return library.symbols().geospatial_save

    def arguments(self, head: CallHead, outputs: Pointer[Self.Outputs, MutUntrackedOrigin]) -> Self.Arguments:
        return {head.index, head.executor, head.outcome, head.completion, self.path, ProgressArguments()}

    def conclude(deinit self, outputs: Unwritten) -> NoneType:
        return None


@fieldwise_init
struct _GeospatialSaveBufferCall[origin: MutOrigin](Call):
    """The crossing of `usearch_geospatial_save_buffer`, which borrows the arena the monolith lands in."""

    comptime Arguments = DenseSaveBufferArguments
    comptime Outputs = DenseSaveBufferOutputs
    comptime Result = Monolith[Self.origin]
    comptime symbol: StaticString = "usearch_geospatial_save_buffer"
    var arena: Address

    @staticmethod
    def verb(library: Library) -> Verb[Self.Arguments]:
        return library.symbols().geospatial_save_buffer

    def arguments(self, head: CallHead, outputs: Pointer[Self.Outputs, MutUntrackedOrigin]) -> Self.Arguments:
        return {head.index, head.executor, head.outcome, head.completion, outputs, self.arena, ProgressArguments()}

    def conclude(deinit self, outputs: DenseSaveBufferOutputs) -> Self.Result:
        return {outputs.data, Int(outputs.length)}


@fieldwise_init
struct _GeospatialCursorNextCall[origin: MutOrigin](Call):
    """The crossing of `usearch_geospatial_cursor_next`, which borrows the arena its batch lands in."""

    comptime Arguments = DenseCursorNextArguments
    comptime Outputs = DenseCursorNextOutputs
    comptime Result = DenseBatch[Self.origin]
    comptime symbol: StaticString = "usearch_geospatial_cursor_next"
    var cursor: Address
    var links: Bool
    var wanted: Int
    var arena: Address

    @staticmethod
    def verb(library: Library) -> Verb[Self.Arguments]:
        return library.symbols().geospatial_cursor_next

    def arguments(self, head: CallHead, outputs: Pointer[Self.Outputs, MutUntrackedOrigin]) -> Self.Arguments:
        return {
            head.index,
            head.executor,
            head.outcome,
            head.completion,
            self.cursor,
            Pointer(to=outputs[].keys),
            Optional(Pointer(to=outputs[].links)) if self.links else None,
            Pointer(to=outputs[].written),
            self.arena,
            c_size_t(self.wanted),
        }

    def conclude(deinit self, outputs: DenseCursorNextOutputs) -> Self.Result:
        return {outputs.keys, self.links, outputs.links, Int(outputs.written)}


# endregion Calls

# region Index


struct GeospatialIndex(GeospatialVerbs, Movable):
    """A geospatial index, released when it goes out of scope.

    Insertions and searches overlap freely on a host handle, as on a dense one. Verbs called on the
    index run on the calling thread; `index.on(executor)` runs the same verbs blocking on an
    executor, and `index.deferred(executor)` submits them there.
    """

    var library: Library
    """The loaded engine, held so its symbol table outlives every call made through the handle."""
    var address: Address
    """The opaque `usearch_geospatial_index_t`."""

    # region Lifetime

    def __init__(out self, *, library: Library, address: Address):
        """Adopts a handle the engine just created.

        Args:
            library: The loaded engine the handle was created through, and is freed through.
            address: A `usearch_geospatial_index_t` whose lifetime this index takes over.
        """
        self.library = library
        self.address = address

    @staticmethod
    def init_ram[
        terrain_origin: ImmOrigin, ram_origin: ImmOrigin, //
    ](
        library: Library,
        options: GeospatialOptions[terrain_origin] = GeospatialOptions(),
        ram: RamOptions[ram_origin] = RamOptions(),
    ) raises Error -> Self:
        """Initializes an empty, mutable index in host memory. Calls `usearch_geospatial_init_ram`.

        Parameters:
            terrain_origin: The options' borrowed terrain, inferred from the argument.
            ram_origin: The placement's borrowed domain array, inferred from the argument.

        Args:
            library: The loaded engine, kept so the shared object stays mapped.
            options: The terrain and the graph's bounds; the terrain stays borrowed by the handle.
            ram: The memory domains to place one shard on apiece.

        Returns:
            The new index.

        Raises:
            `Error` with `INVALID_ARGUMENT` for a terrain both borrowed and filled, or
            `CONFIG_MISMATCH` for one whose steps are not positive.
        """
        var created = stack_allocation[1, Address]()
        created[unsafe_offset=0] = NULL_ADDRESS
        var outcome = stack_allocation[1, Outcome]()
        outcome[unsafe_offset=0] = Outcome()
        _cross(
            GeospatialInitRamArguments(
                created.unsafe_origin_cast[MutUntrackedOrigin](),
                outcome.unsafe_origin_cast[MutUntrackedOrigin](),
                options.terrain.arguments,
                options.connectivity,
                options.values_per_key,
                ram.arguments(),
            ),
            library.symbols().geospatial_init_ram,
        )
        library._check(outcome.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0], "usearch_geospatial_init_ram")
        return Self(library=library, address=created.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0])

    @staticmethod
    def init_vram[
        terrain_origin: ImmOrigin, vram_origin: ImmOrigin, //
    ](
        library: Library,
        options: GeospatialOptions[terrain_origin] = GeospatialOptions(),
        vram: VramOptions[vram_origin] = VramOptions(),
    ) raises Error -> Self:
        """Initializes an empty, mutable index in device memory. Calls `usearch_geospatial_init_vram`.

        Parameters:
            terrain_origin: The options' borrowed terrain, inferred from the argument.
            vram_origin: The placement's borrowed ordinal array, inferred from the argument.

        Args:
            library: The loaded engine, kept so the shared object stays mapped.
            options: The terrain and the graph's bounds; the terrain stays borrowed by the handle.
            vram: The device to place the index on.

        Returns:
            The new index.

        Raises:
            `Error` with `UNSUPPORTED` for a filled terrain or a `values_per_key` bound, or if no
            device can be selected or claimed.
        """
        var created = stack_allocation[1, Address]()
        created[unsafe_offset=0] = NULL_ADDRESS
        var outcome = stack_allocation[1, Outcome]()
        outcome[unsafe_offset=0] = Outcome()
        _cross(
            GeospatialInitVramArguments(
                created.unsafe_origin_cast[MutUntrackedOrigin](),
                outcome.unsafe_origin_cast[MutUntrackedOrigin](),
                options.terrain.arguments,
                options.connectivity,
                options.values_per_key,
                vram.arguments(),
            ),
            library.symbols().geospatial_init_vram,
        )
        library._check(outcome.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0], "usearch_geospatial_init_vram")
        return Self(library=library, address=created.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0])

    @staticmethod
    def load_ram[
        path_origin: ImmOrigin, ram_origin: ImmOrigin, terrain_origin: ImmOrigin, //
    ](
        executor: Executor,
        path: StringSlice[path_origin],
        ram: RamOptions[ram_origin] = RamOptions(),
        values_per_key: Int = 0,
        terrain: GeospatialTerrain[terrain_origin] = GeospatialTerrain(),
    ) raises Error -> Self:
        """Copies a checkpoint into a new, mutable index in host memory. Calls `usearch_geospatial_load_ram`.

        Parameters:
            path_origin: The path bytes' origin, inferred from the argument.
            ram_origin: The placement's borrowed domain array, inferred from the argument.
            terrain_origin: The borrowed terrain, inferred from the argument.

        Args:
            executor: The executor the read fans out over; it also lends its engine.
            path: The manifest of a sharded checkpoint, or a single `.safegraphs` monolith.
            ram: The memory domains to place the parts on.
            values_per_key: Upper bound on observations sharing one key, which no checkpoint records.
            terrain: The elevation sight lines clear, which no checkpoint carries.

        Returns:
            The new index; a failed load creates none.

        Raises:
            `Error` if the checkpoint cannot be read or decoded.
        """
        return Self._load_ram(
            executor,
            executor.library.symbols().geospatial_load_ram,
            BytesView.of(path).arguments(),
            ram.arguments(),
            values_per_key,
            terrain.arguments,
            "usearch_geospatial_load_ram",
        )

    @staticmethod
    def load_ram_buffer[
        buffer_origin: ImmOrigin, ram_origin: ImmOrigin, terrain_origin: ImmOrigin, //
    ](
        executor: Executor,
        buffer: Span[UInt8, buffer_origin],
        ram: RamOptions[ram_origin] = RamOptions(),
        values_per_key: Int = 0,
        terrain: GeospatialTerrain[terrain_origin] = GeospatialTerrain(),
    ) raises Error -> Self:
        """Copies a monolith from a memory buffer into a new, mutable index in host memory.
        Calls `usearch_geospatial_load_ram_buffer`.

        Parameters:
            buffer_origin: The source bytes' origin, inferred from the argument.
            ram_origin: The placement's borrowed domain array, inferred from the argument.
            terrain_origin: The borrowed terrain, inferred from the argument.

        Args:
            executor: The executor the copy fans out over; it also lends its engine.
            buffer: Source holding one serialized monolith.
            ram: The memory domains to place shards on; the monolith lands whole on the first.
            values_per_key: Upper bound on observations sharing one key, which no checkpoint records.
            terrain: The elevation sight lines clear, which no checkpoint carries.

        Returns:
            The new index; a failed load creates none.

        Raises:
            `Error` if the bytes cannot be decoded.
        """
        return Self._load_ram(
            executor,
            executor.library.symbols().geospatial_load_ram_buffer,
            BytesView.of(buffer).arguments(),
            ram.arguments(),
            values_per_key,
            terrain.arguments,
            "usearch_geospatial_load_ram_buffer",
        )

    @staticmethod
    def _load_ram(
        executor: Executor,
        verb: GeospatialLoadRam,
        source: BytesView[ImmUntrackedOrigin],
        ram: RamOptions[ImmUntrackedOrigin],
        values_per_key: Int,
        terrain: GeospatialTerrainArguments,
        detail: StaticString,
    ) raises Error -> Self:
        """The path and buffer forms share one argument layout, differing only in the verb."""
        var created = stack_allocation[1, Address]()
        created[unsafe_offset=0] = NULL_ADDRESS
        var outcome = stack_allocation[1, Outcome]()
        outcome[unsafe_offset=0] = Outcome()
        _cross(
            GeospatialLoadRamArguments(
                created.unsafe_origin_cast[MutUntrackedOrigin](),
                executor.address,
                outcome.unsafe_origin_cast[MutUntrackedOrigin](),
                Completion.BLOCKING,
                source,
                ram,
                c_size_t(values_per_key),
                terrain,
                ProgressArguments(),
            ),
            verb,
        )
        executor.library._check(outcome.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0], detail)
        return Self(library=executor.library, address=created.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0])

    @staticmethod
    def load_vram[
        path_origin: ImmOrigin, vram_origin: ImmOrigin, terrain_origin: ImmOrigin, //
    ](
        executor: Executor,
        path: StringSlice[path_origin],
        vram: VramOptions[vram_origin] = VramOptions(),
        values_per_key: Int = 0,
        terrain: GeospatialTerrain[terrain_origin] = GeospatialTerrain(),
    ) raises Error -> Self:
        """Copies a checkpoint into a new, mutable index in device memory. Calls `usearch_geospatial_load_vram`.

        Parameters:
            path_origin: The path bytes' origin, inferred from the argument.
            vram_origin: The placement's borrowed ordinal array, inferred from the argument.
            terrain_origin: The borrowed terrain, inferred from the argument.

        Args:
            executor: The executor the upload runs on; it also lends its engine.
            path: The manifest of a sharded checkpoint, or a single `.safegraphs` monolith.
            vram: The device to place the index on.
            values_per_key: Upper bound on observations sharing one key, which no checkpoint records.
            terrain: The elevation sight lines clear, which no checkpoint carries.

        Returns:
            The new index; a failed load creates none.

        Raises:
            `Error` if the checkpoint cannot be read, or a device cannot hold it.
        """
        return Self._load_vram(
            executor,
            executor.library.symbols().geospatial_load_vram,
            BytesView.of(path).arguments(),
            vram.arguments(),
            values_per_key,
            terrain.arguments,
            "usearch_geospatial_load_vram",
        )

    @staticmethod
    def load_vram_buffer[
        buffer_origin: ImmOrigin, vram_origin: ImmOrigin, terrain_origin: ImmOrigin, //
    ](
        executor: Executor,
        buffer: Span[UInt8, buffer_origin],
        vram: VramOptions[vram_origin] = VramOptions(),
        values_per_key: Int = 0,
        terrain: GeospatialTerrain[terrain_origin] = GeospatialTerrain(),
    ) raises Error -> Self:
        """Copies a monolith from a memory buffer into a new, mutable index in device memory.
        Calls `usearch_geospatial_load_vram_buffer`.

        Parameters:
            buffer_origin: The source bytes' origin, inferred from the argument.
            vram_origin: The placement's borrowed ordinal array, inferred from the argument.
            terrain_origin: The borrowed terrain, inferred from the argument.

        Args:
            executor: The executor the upload runs on; it also lends its engine.
            buffer: Source holding one serialized monolith.
            vram: The device to place the index on.
            values_per_key: Upper bound on observations sharing one key, which no checkpoint records.
            terrain: The elevation sight lines clear, which no checkpoint carries.

        Returns:
            The new index; a failed load creates none.

        Raises:
            `Error` if the bytes cannot be decoded, or a device cannot hold them.
        """
        return Self._load_vram(
            executor,
            executor.library.symbols().geospatial_load_vram_buffer,
            BytesView.of(buffer).arguments(),
            vram.arguments(),
            values_per_key,
            terrain.arguments,
            "usearch_geospatial_load_vram_buffer",
        )

    @staticmethod
    def _load_vram(
        executor: Executor,
        verb: GeospatialLoadVram,
        source: BytesView[ImmUntrackedOrigin],
        vram: VramOptions[ImmUntrackedOrigin],
        values_per_key: Int,
        terrain: GeospatialTerrainArguments,
        detail: StaticString,
    ) raises Error -> Self:
        """The path and buffer forms share one argument layout, differing only in the verb."""
        var created = stack_allocation[1, Address]()
        created[unsafe_offset=0] = NULL_ADDRESS
        var outcome = stack_allocation[1, Outcome]()
        outcome[unsafe_offset=0] = Outcome()
        _cross(
            GeospatialLoadVramArguments(
                created.unsafe_origin_cast[MutUntrackedOrigin](),
                executor.address,
                outcome.unsafe_origin_cast[MutUntrackedOrigin](),
                Completion.BLOCKING,
                source,
                vram,
                c_size_t(values_per_key),
                terrain,
                ProgressArguments(),
            ),
            verb,
        )
        executor.library._check(outcome.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0], detail)
        return Self(library=executor.library, address=created.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0])

    @staticmethod
    def view_ram[
        path_origin: ImmOrigin, terrain_origin: ImmOrigin, //
    ](
        library: Library,
        path: StringSlice[path_origin],
        terrain: GeospatialTerrain[terrain_origin] = GeospatialTerrain(),
    ) raises Error -> Self:
        """Serves a checkpoint in place through a memory mapping, as a read-only index.
        Calls `usearch_geospatial_view_ram`.

        Parameters:
            path_origin: The path bytes' origin, inferred from the argument.
            terrain_origin: The borrowed terrain, inferred from the argument.

        Args:
            library: The loaded engine, kept so the shared object stays mapped.
            path: The manifest of a sharded checkpoint, or a single `.safegraphs` monolith.
            terrain: The elevation sight lines clear, which no checkpoint carries.

        Returns:
            The read-only index.

        Raises:
            `Error` if the checkpoint cannot be mapped or decoded.
        """
        var created = stack_allocation[1, Address]()
        created[unsafe_offset=0] = NULL_ADDRESS
        var outcome = stack_allocation[1, Outcome]()
        outcome[unsafe_offset=0] = Outcome()
        _cross(
            GeospatialViewRamArguments(
                created.unsafe_origin_cast[MutUntrackedOrigin](),
                outcome.unsafe_origin_cast[MutUntrackedOrigin](),
                BytesView.of(path).arguments(),
                terrain.arguments,
                RamOptions().arguments(),
            ),
            library.symbols().geospatial_view_ram,
        )
        library._check(outcome.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0], "usearch_geospatial_view_ram")
        return Self(library=library, address=created.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0])

    @staticmethod
    def view_nvme[
        path_origin: ImmOrigin, nvme_origin: ImmOrigin, terrain_origin: ImmOrigin, //
    ](
        executor: Executor,
        path: StringSlice[path_origin],
        nvme: NvmeOptions[nvme_origin] = NvmeOptions(),
        terrain: GeospatialTerrain[terrain_origin] = GeospatialTerrain(),
    ) raises Error -> Self:
        """Serves a checkpoint in place from NVMe drives, read through the executor's rings.
        Calls `usearch_geospatial_view_nvme`.

        Parameters:
            path_origin: The path bytes' origin, inferred from the argument.
            nvme_origin: The options' borrowed path array, inferred from the argument.
            terrain_origin: The borrowed terrain, inferred from the argument.

        Args:
            executor: The NVMe executor whose rings read the checkpoint; it also lends its engine.
            path: The manifest of a sharded checkpoint, or a single `.safegraphs` monolith.
            nvme: The devices the parts are served from; naming none serves each where it points.
            terrain: The elevation sight lines clear, which no checkpoint carries.

        Returns:
            The read-only index.

        Raises:
            `Error` with `IO` if the checkpoint cannot be opened.
        """
        var created = stack_allocation[1, Address]()
        created[unsafe_offset=0] = NULL_ADDRESS
        var outcome = stack_allocation[1, Outcome]()
        outcome[unsafe_offset=0] = Outcome()
        _cross(
            GeospatialViewNvmeArguments(
                created.unsafe_origin_cast[MutUntrackedOrigin](),
                executor.address,
                outcome.unsafe_origin_cast[MutUntrackedOrigin](),
                Completion.BLOCKING,
                BytesView.of(path).arguments(),
                nvme.arguments(),
                terrain.arguments,
            ),
            executor.library.symbols().geospatial_view_nvme,
        )
        executor.library._check(
            outcome.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0], "usearch_geospatial_view_nvme"
        )
        return Self(library=executor.library, address=created.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0])

    def __deinit__(deinit self):
        """Releases the index and everything it owns. Calls `usearch_geospatial_free`.

        The loader reference is surrendered only after the call, so the engine stays mapped until
        the release returns.
        """
        self.library.symbols().geospatial_free(self.address)
        _ = self.library^

    # endregion Lifetime

    # region Receivers

    def on(ref self, ref executor: Executor) -> GeospatialIndexOn[ImmOrigin(origin_of(self, executor))]:
        """The same verbs, blocking on `executor`, refused while another call holds it. Binding-only.

        Args:
            executor: The executor every verb runs on; its kind must match the handle's residency.

        Returns:
            A receiver borrowing both, which neither may outlive.
        """
        return GeospatialIndexOn[ImmOrigin(origin_of(self, executor))](
            Pointer[GeospatialIndex, ImmOrigin(origin_of(self, executor))](unsafe_from_address=Int(Pointer(to=self))),
            executor.address,
        )

    def deferred(ref self, ref executor: Executor) -> GeospatialIndexDeferredOn[ImmOrigin(origin_of(self, executor))]:
        """The same verbs, each submitted to `executor` and answering a `Deferred` at once. Binding-only.

        Args:
            executor: The executor every verb is submitted to; its kind must match the handle's residency.

        Returns:
            A receiver borrowing both, which neither it nor any `Deferred` it answers may outlive.
        """
        return GeospatialIndexDeferredOn[ImmOrigin(origin_of(self, executor))](
            Pointer[GeospatialIndex, ImmOrigin(origin_of(self, executor))](unsafe_from_address=Int(Pointer(to=self))),
            executor.address,
        )

    def _route[C: Call](self, var call: C) raises Error -> C.Result:
        """Runs `call` on the calling thread."""
        return _block(self.library, call^, self.address, NULL_ADDRESS)

    def cursor(ref self) raises Error -> GeospatialCursor[ImmOrigin(origin_of(self))]:
        """Opens a walk over every entry as stored, its pulls on the calling thread.
        Calls `usearch_geospatial_cursor_init`.

        Returns:
            The cursor, which borrows the index and cannot outlive it.

        Raises:
            `Error` if the residency has no cursor, as VRAM does not.
        """
        return GeospatialCursor[ImmOrigin(origin_of(self))](self.library, self.address, NULL_ADDRESS, 0, None)

    def cursor[
        starts_origin: ImmOrigin, //
    ](ref self, starts: Span[Key, starts_origin]) raises Error -> GeospatialCursor[ImmOrigin(origin_of(self))]:
        """Opens a walk from `starts`, its pulls on the calling thread. Calls `usearch_geospatial_cursor_init`.

        Parameters:
            starts_origin: The start keys' origin, inferred from the argument.

        Args:
            starts: The keys to walk from, the walk yielding their entries first.

        Returns:
            The cursor, which borrows the index and cannot outlive it.

        Raises:
            `Error` if the residency has no cursor, as VRAM does not.
        """
        return GeospatialCursor[ImmOrigin(origin_of(self))](
            self.library,
            self.address,
            NULL_ADDRESS,
            len(starts),
            starts.unsafe_ptr().unsafe_origin_cast[ImmUntrackedOrigin](),
        )

    # endregion Receivers


trait GeospatialVerbs:
    """The verbs every blocking receiver offers - the index on the calling thread, and
    `index.on(executor)` on an executor - each written once over the receiver's `_route`."""

    def _route[C: Call](self, var call: C) raises Error -> C.Result:
        """Runs `call` to its final status on this receiver's executor and answers its result."""
        ...

    def reserve(self, entries: Int) raises Error:
        """Reserves room for incoming entries ahead of insertion. Calls `usearch_geospatial_reserve`.

        Args:
            entries: The desired total capacity in entries, including the current size.

        Raises:
            `Error` if the reservation cannot be satisfied.
        """
        _ = self._route(_GeospatialReserveCall(entries))

    def clear(self) raises Error:
        """Erases every entry, keeping the reserved capacity and the terrain. Calls `usearch_geospatial_clear`.

        Raises:
            `Error` with `READ_ONLY` if the handle serves its checkpoint in place.
        """
        _ = self._route(_GeospatialClearCall())

    def describe(self) raises Error -> GeospatialIndexDescription:
        """Reports the frozen metadata and the current counters in one call. Calls `usearch_geospatial_describe`.

        Returns:
            The description.

        Raises:
            `Error` if the engine refuses the query.
        """
        return self._route(_GeospatialDescribeCall())

    def insert[
        keys_origin: ImmOrigin, observations_origin: ImmOrigin, stats_origin: MutOrigin, //
    ](
        self,
        keys: Span[Key, keys_origin],
        observations: Span[GeospatialObservation, observations_origin],
        config: DenseInsertConfig[stats_origin] = DenseInsertConfig(),
    ) raises Error:
        """Adds observations under their keys, all or nothing. Calls `usearch_geospatial_insert`.

        Parameters:
            keys_origin: The key run's origin, inferred from the argument.
            observations_origin: The observations' origin, inferred with it.
            stats_origin: The lent telemetry slot's origin, inferred with it.

        Args:
            keys: One key per observation; keys need not be unique, so frames of one clip share it.
            observations: The observations to insert, each with a latitude and a longitude.
            config: The breadth of the candidates and the telemetry slot.

        Raises:
            `Error` with `SHAPE_MISMATCH` when the keys and observations differ in count,
            `INVALID_ARGUMENT` for one without a position, `OVERFLOW` past `values_per_key`, or
            `READ_ONLY` for a view.
        """
        _ = self._route(_insert_call(keys, observations, config))

    def refine[
        stats_origin: MutOrigin, plan_origin: MutOrigin, phase_origin: MutOrigin, //
    ](
        self, config: GeospatialRefineConfig[stats_origin, plan_origin, phase_origin] = GeospatialRefineConfig()
    ) raises Error -> DenseRefineReport:
        """Runs one round over the whole graph, healing islands and re-stitching neighborhoods, or
        continues `config.plan`. Calls `usearch_geospatial_refine`.

        Parameters:
            stats_origin: The lent telemetry slot's origin, inferred from the argument.
            plan_origin: The lent plan's origin, inferred with it.
            phase_origin: The lent phase slot's origin, inferred with it.

        Args:
            config: The round's spending, the handler, the telemetry slot, and the plan and phase slots.

        Returns:
            What the round found and changed, its verdict, and its opening and closing health checks.

        Raises:
            `Error` with `INVALID_ARGUMENT` for a plan on the calling thread, `READ_ONLY` for a view, or
            `CANCELLED` once the handler stops the round.
        """
        return self._route(_refine_call(config))

    def refine_plan_finish(self, var plan: GeospatialRefinePlan) raises Error -> DenseRefineReport:
        """Closes and frees a refine plan whatever the call answers, reopening the index.
        Calls `usearch_geospatial_refine_plan_finish`.

        Args:
            plan: The plan a refine on this index and executor opened, consumed by the call.

        Returns:
            What the whole plan did to the graph.

        Raises:
            `Error` with `INVALID_ARGUMENT` for a plan no refine opened, or one already finished.
        """
        return self._route(_GeospatialRefinePlanFinishCall(plan.address))

    def health(self) raises Error -> DenseReachability:
        """Measures how much of the graph a search reaches from the root. Calls `usearch_geospatial_health`.

        Returns:
            The audit; entries it misses stay unfound until a `refine` re-links them.

        Raises:
            `Error` if the engine refuses the audit.
        """
        return self._route(_GeospatialHealthCall())

    def sample[
        shown_origin: ImmOrigin, coordinates_origin: ImmOrigin, focus_origin: ImmOrigin, //
    ](
        self,
        wanted: Int,
        mut arena: Arena,
        config: DenseSampleConfig[shown_origin, coordinates_origin, focus_origin] = DenseSampleConfig(),
    ) raises Error -> DenseDraws[origin_of(arena)]:
        """Draws up to `wanted` keys around `config.focus`, or across the whole index, each hanging from
        a shown or earlier draw and placed in three dimensions. Calls `usearch_geospatial_sample`.

        Parameters:
            shown_origin: The shown keys' origin, inferred from the config.
            coordinates_origin: Their coordinates' origin, inferred with it.
            focus_origin: The focus keys' origin, inferred with it.

        Args:
            wanted: The upper bound on draws.
            arena: The arena to rewind and refill, which the result borrows; a failed call leaves it valid.
            config: The shown keys and their coordinates, the focus and the seed.

        Returns:
            The draws, read in place from the arena.

        Raises:
            `Error` with `SHAPE_MISMATCH` unless there are three coordinates per shown key.
        """
        return self._route(_sample_call[origin_of(arena)](wanted, arena._address, config))

    def search[
        queries_origin: ImmOrigin, stats_origin: MutOrigin, //
    ](
        self,
        queries: Span[GeospatialObservation, queries_origin],
        wanted: Int,
        mut arena: Arena,
        config: GeospatialSearchConfig[stats_origin] = GeospatialSearchConfig(),
    ) raises Error -> Matches[origin_of(arena)]:
        """Finds the `wanted` nearest entries of every query observation that the relation admits,
        writing the matches in a lent arena. Calls `usearch_geospatial_search`.

        Parameters:
            queries_origin: The query observations' origin, inferred from the argument.
            stats_origin: The lent telemetry slot's origin, inferred with it.

        Args:
            queries: The query observations.
            wanted: Matches per query - the "k"; zero lands only the counts.
            arena: The arena to rewind and refill, which the result borrows; a failed call leaves it valid.
            config: The relation, the radius and time slack, the breadth, the predicate and the telemetry slot.

        Returns:
            Every query's matches in chord metres, nearest first, read in place from the arena.

        Raises:
            `Error` with `UNSUPPORTED` for `VISIBLE` on a handle without terrain, or a predicate passed
            to a VRAM handle.
        """
        return self._route(_search_call[origin_of(arena)](queries, wanted, arena._address, config))

    def within[
        queries_origin: ImmOrigin, stats_origin: MutOrigin, //
    ](
        self,
        queries: Span[GeospatialObservation, queries_origin],
        metres: Float64,
        mut arena: Arena,
        config: GeospatialWithinConfig[stats_origin] = GeospatialWithinConfig(),
    ) raises Error -> GeospatialMatches[origin_of(arena)]:
        """Reports every entry within `metres` of each query observation that the relation admits,
        however many, writing them in a lent arena. Calls `usearch_geospatial_within`.

        Parameters:
            queries_origin: The query observations' origin, inferred from the argument.
            stats_origin: The lent telemetry slot's origin, inferred with it.

        Args:
            queries: The query observations.
            metres: The radius, positive.
            arena: The arena to rewind and refill, which the result borrows; a failed call leaves it valid.
            config: The relation, the time slack, the traversal, the predicate and the telemetry slot.

        Returns:
            Every query's matches in chord metres, read in place from the arena.

        Raises:
            `Error` with `INVALID_ARGUMENT` for a zero radius, or `UNSUPPORTED` for `VISIBLE` on a
            handle without terrain.
        """
        return self._route(_within_call[origin_of(arena)](queries, metres, arena._address, config))

    def count(self, key: Key) raises Error -> GeospatialKeyStats:
        """How many observations one key holds. Calls `usearch_geospatial_count`.

        Args:
            key: The key whose entries are counted.

        Returns:
            The observations under the key.

        Raises:
            `Error` if the engine refuses the query.
        """
        return self._route(_GeospatialCountCall(key))

    def get(self, key: Key, mut arena: Arena) raises Error -> GeospatialObservations[origin_of(arena)]:
        """Exports every observation stored under one key into a lent arena, in insertion order and
        within the rounding the index stores them at. Calls `usearch_geospatial_get`.

        Args:
            key: The key whose entries are exported.
            arena: The arena to rewind and refill, which the result borrows; a failed call leaves it valid.

        Returns:
            The observations, read in place from the arena; none when the key is absent.

        Raises:
            `Error` with `UNSUPPORTED` for a handle living on a GPU, whose index exports nothing yet.
        """
        return self._route(_GeospatialGetCall[origin_of(arena)](key, arena._address))

    def save[path_origin: ImmOrigin, //](self, path: StringSlice[path_origin]) raises Error:
        """Writes a checkpoint, the path extension picking the layout. Calls `usearch_geospatial_save`.

        Parameters:
            path_origin: The path bytes' origin, inferred from the argument.

        Args:
            path: Destination path, whose extension picks the layout.

        Raises:
            `Error` if the destination cannot be written, or `UNSUPPORTED` for an NVMe view.
        """
        _ = self._route(_GeospatialSaveCall(BytesView.of(path).arguments()))

    def save_buffer(self, mut arena: Arena) raises Error -> Monolith[origin_of(arena)]:
        """Serializes the index as one monolith into a lent arena. Calls `usearch_geospatial_save_buffer`.

        Args:
            arena: The arena to rewind and refill, which the result borrows; a failed call leaves it valid.

        Returns:
            The monolith, read in place from the arena.

        Raises:
            `Error` if the engine refuses the write.
        """
        return self._route(_GeospatialSaveBufferCall[origin_of(arena)](arena._address))


struct GeospatialIndexOn[origin: ImmOrigin](GeospatialVerbs, ImplicitlyCopyable):
    """An index bound to the executor its verbs block on, as `index.on(executor)` answers it. Binding-only.

    Parameters:
        origin: The index and the executor, which this receiver borrows.
    """

    var index: Pointer[GeospatialIndex, Self.origin]
    """The index every verb addresses."""
    var executor: Address
    """The `usearch_executor_t` every verb runs on."""

    def __init__(out self, index: Pointer[GeospatialIndex, Self.origin], executor: Address):
        """Binds an index to an executor.

        Args:
            index: The index every verb addresses.
            executor: The `usearch_executor_t` every verb runs on.
        """
        self.index = index
        self.executor = executor

    def _route[C: Call](self, var call: C) raises Error -> C.Result:
        """Runs `call` on the bound executor, refused while another call holds it."""
        return _block(self.index[].library, call^, self.index[].address, self.executor)

    def cursor(self) raises Error -> GeospatialCursor[Self.origin]:
        """Opens a walk over every entry as stored, its pulls on the bound executor.
        Calls `usearch_geospatial_cursor_init`.

        Returns:
            The cursor, which borrows the index and the executor and cannot outlive them.

        Raises:
            `Error` if the residency has no cursor, as VRAM does not.
        """
        return GeospatialCursor[Self.origin](self.index[].library, self.index[].address, self.executor, 0, None)

    def cursor[
        starts_origin: ImmOrigin, //
    ](self, starts: Span[Key, starts_origin]) raises Error -> GeospatialCursor[Self.origin]:
        """Opens a walk from `starts`, its pulls on the bound executor. Calls `usearch_geospatial_cursor_init`.

        Parameters:
            starts_origin: The start keys' origin, inferred from the argument.

        Args:
            starts: The keys to walk from, the walk yielding their entries first.

        Returns:
            The cursor, which borrows the index and the executor and cannot outlive them.

        Raises:
            `Error` if the residency has no cursor, as VRAM does not.
        """
        return GeospatialCursor[Self.origin](
            self.index[].library,
            self.index[].address,
            self.executor,
            len(starts),
            starts.unsafe_ptr().unsafe_origin_cast[ImmUntrackedOrigin](),
        )


def _refine_call[
    stats_origin: MutOrigin, plan_origin: MutOrigin, phase_origin: MutOrigin
](config: GeospatialRefineConfig[stats_origin, plan_origin, phase_origin]) -> _GeospatialRefineCall:
    """Lowers a refinement's lent slots into its crossing."""
    return _GeospatialRefineCall(
        RefineOptionsArguments(c_size_t(config.expansion_factor), c_size_t(config.memory_limit)),
        config.progress,
        _slot(config.stats),
        _slot(config.plan),
        _slot(config.phase),
    )


def _insert_call[
    keys_origin: ImmOrigin, observations_origin: ImmOrigin, stats_origin: MutOrigin
](
    keys: Span[Key, keys_origin],
    observations: Span[GeospatialObservation, observations_origin],
    config: DenseInsertConfig[stats_origin],
) -> _GeospatialInsertCall:
    """Lowers an insertion's borrowed inputs into its crossing."""
    return _GeospatialInsertCall(
        KeysView.of(keys).arguments(),
        len(keys),
        Int(observations.unsafe_ptr()),
        len(observations),
        config.expansion_factor,
        config.progress,
        _slot(config.stats),
    )


def _sample_call[
    origin: MutOrigin, shown_origin: ImmOrigin, coordinates_origin: ImmOrigin, focus_origin: ImmOrigin
](
    wanted: Int,
    arena: Address,
    config: DenseSampleConfig[shown_origin, coordinates_origin, focus_origin],
) -> _GeospatialSampleCall[origin]:
    """Lowers a sample's borrowed inputs into its crossing."""
    return _GeospatialSampleCall[origin](
        KeysView.of(config.shown).arguments(),
        len(config.shown),
        RowsViewArguments(
            Int(config.shown_coordinates.unsafe_ptr()), c_size_t(3 * size_of[Float32]()), c_size_t(3), ScalarKind.F32
        ),
        len(config.shown_coordinates),
        KeysView.of(config.focus).arguments(),
        len(config.focus),
        wanted,
        config.seed,
        config.progress,
        arena,
    )


def _search_call[
    origin: MutOrigin, queries_origin: ImmOrigin, stats_origin: MutOrigin
](
    queries: Span[GeospatialObservation, queries_origin],
    wanted: Int,
    arena: Address,
    config: GeospatialSearchConfig[stats_origin],
) -> _GeospatialSearchCall[origin]:
    """Lowers a search's borrowed inputs into its crossing."""
    return _GeospatialSearchCall[origin](
        Int(queries.unsafe_ptr()),
        len(queries),
        config.relation,
        GeospatialToleranceArguments(config.metres, config.seconds),
        wanted,
        arena,
        config.expansion_factor,
        config.traversal,
        config.predicate,
        config.progress,
        _slot(config.stats),
    )


def _within_call[
    origin: MutOrigin, queries_origin: ImmOrigin, stats_origin: MutOrigin
](
    queries: Span[GeospatialObservation, queries_origin],
    metres: Float64,
    arena: Address,
    config: GeospatialWithinConfig[stats_origin],
) -> _GeospatialWithinCall[origin]:
    """Lowers a containment query's borrowed inputs into its crossing."""
    return _GeospatialWithinCall[origin](
        Int(queries.unsafe_ptr()),
        len(queries),
        config.relation,
        GeospatialToleranceArguments(metres, config.seconds),
        arena,
        config.traversal,
        config.predicate,
        config.progress,
        _slot(config.stats),
    )


struct GeospatialIndexDeferredOn[origin: ImmOrigin](ImplicitlyCopyable):
    """An index bound to the executor its verbs are submitted to, as `index.deferred(executor)` answers
    it: every verb answers a `Deferred` at once. Binding-only.

    Parameters:
        origin: The index and the executor, which this receiver borrows.
    """

    var index: Pointer[GeospatialIndex, Self.origin]
    """The index every verb addresses."""
    var executor: Address
    """The `usearch_executor_t` every verb is submitted to."""

    def __init__(out self, index: Pointer[GeospatialIndex, Self.origin], executor: Address):
        """Binds an index to an executor.

        Args:
            index: The index every verb addresses.
            executor: The `usearch_executor_t` every verb is submitted to.
        """
        self.index = index
        self.executor = executor

    def reserve(self, entries: Int) -> Deferred[_GeospatialReserveCall, Self.origin]:
        """Submits `GeospatialVerbs.reserve`. Calls `usearch_geospatial_reserve`."""
        return {
            self.index[].library,
            _GeospatialReserveCall(entries),
            self.index[].address,
            self.executor,
        }

    def clear(self) -> Deferred[_GeospatialClearCall, Self.origin]:
        """Submits `GeospatialVerbs.clear`. Calls `usearch_geospatial_clear`."""
        return {self.index[].library, _GeospatialClearCall(), self.index[].address, self.executor}

    def describe(self) -> Deferred[_GeospatialDescribeCall, Self.origin]:
        """Submits `GeospatialVerbs.describe`. Calls `usearch_geospatial_describe`."""
        return {self.index[].library, _GeospatialDescribeCall(), self.index[].address, self.executor}

    def insert[
        keys_origin: ImmOrigin, observations_origin: ImmOrigin, stats_origin: MutOrigin, //
    ](
        self,
        keys: Span[Key, keys_origin],
        observations: Span[GeospatialObservation, observations_origin],
        config: DenseInsertConfig[stats_origin] = DenseInsertConfig(),
    ) -> Deferred[
        _GeospatialInsertCall,
        ImmOrigin(
            origin_of(
                _anchor[Self.origin](), _anchor[keys_origin](), _anchor[observations_origin](), _anchor[stats_origin]()
            )
        ),
    ]:
        """Submits `GeospatialVerbs.insert`; the keys, observations and telemetry slot stay borrowed until
        final. Calls `usearch_geospatial_insert`."""
        return {self.index[].library, _insert_call(keys, observations, config), self.index[].address, self.executor}

    def refine[
        stats_origin: MutOrigin, plan_origin: MutOrigin, phase_origin: MutOrigin, //
    ](
        self, config: GeospatialRefineConfig[stats_origin, plan_origin, phase_origin] = GeospatialRefineConfig()
    ) -> Deferred[
        _GeospatialRefineCall,
        ImmOrigin(
            origin_of(_anchor[Self.origin](), _anchor[stats_origin](), _anchor[plan_origin](), _anchor[phase_origin]())
        ),
    ]:
        """Submits `GeospatialVerbs.refine`; the telemetry, plan and phase slots stay borrowed until final.
        Calls `usearch_geospatial_refine`."""
        return {self.index[].library, _refine_call(config), self.index[].address, self.executor}

    def refine_plan_finish(
        self, var plan: GeospatialRefinePlan
    ) -> Deferred[_GeospatialRefinePlanFinishCall, Self.origin]:
        """Submits `GeospatialVerbs.refine_plan_finish`. Calls `usearch_geospatial_refine_plan_finish`."""
        return {
            self.index[].library,
            _GeospatialRefinePlanFinishCall(plan.address),
            self.index[].address,
            self.executor,
        }

    def health(self) -> Deferred[_GeospatialHealthCall, Self.origin]:
        """Submits `GeospatialVerbs.health`. Calls `usearch_geospatial_health`."""
        return {self.index[].library, _GeospatialHealthCall(), self.index[].address, self.executor}

    def sample[
        shown_origin: ImmOrigin, coordinates_origin: ImmOrigin, focus_origin: ImmOrigin, //
    ](
        self,
        wanted: Int,
        mut arena: Arena,
        config: DenseSampleConfig[shown_origin, coordinates_origin, focus_origin] = DenseSampleConfig(),
    ) -> Deferred[
        _GeospatialSampleCall[origin_of(arena)],
        ImmOrigin(
            origin_of(
                _anchor[Self.origin](),
                _anchor[shown_origin](),
                _anchor[coordinates_origin](),
                _anchor[focus_origin](),
            )
        ),
    ]:
        """Submits `GeospatialVerbs.sample`; the shown keys, focus and arena stay borrowed until final.
        Calls `usearch_geospatial_sample`."""
        return {
            self.index[].library,
            _sample_call[origin_of(arena)](wanted, arena._address, config),
            self.index[].address,
            self.executor,
        }

    def search[
        queries_origin: ImmOrigin, stats_origin: MutOrigin, //
    ](
        self,
        queries: Span[GeospatialObservation, queries_origin],
        wanted: Int,
        mut arena: Arena,
        config: GeospatialSearchConfig[stats_origin] = GeospatialSearchConfig(),
    ) -> Deferred[
        _GeospatialSearchCall[origin_of(arena)],
        ImmOrigin(origin_of(_anchor[Self.origin](), _anchor[queries_origin](), _anchor[stats_origin]())),
    ]:
        """Submits `GeospatialVerbs.search`; the queries, telemetry slot and arena stay borrowed until final.
        Calls `usearch_geospatial_search`."""
        return {
            self.index[].library,
            _search_call[origin_of(arena)](queries, wanted, arena._address, config),
            self.index[].address,
            self.executor,
        }

    def within[
        queries_origin: ImmOrigin, stats_origin: MutOrigin, //
    ](
        self,
        queries: Span[GeospatialObservation, queries_origin],
        metres: Float64,
        mut arena: Arena,
        config: GeospatialWithinConfig[stats_origin] = GeospatialWithinConfig(),
    ) -> Deferred[
        _GeospatialWithinCall[origin_of(arena)],
        ImmOrigin(origin_of(_anchor[Self.origin](), _anchor[queries_origin](), _anchor[stats_origin]())),
    ]:
        """Submits `GeospatialVerbs.within`; the queries, telemetry slot and arena stay borrowed until final.
        Calls `usearch_geospatial_within`."""
        return {
            self.index[].library,
            _within_call[origin_of(arena)](queries, metres, arena._address, config),
            self.index[].address,
            self.executor,
        }

    def count(self, key: Key) -> Deferred[_GeospatialCountCall, Self.origin]:
        """Submits `GeospatialVerbs.count`. Calls `usearch_geospatial_count`."""
        return {self.index[].library, _GeospatialCountCall(key), self.index[].address, self.executor}

    def get(self, key: Key, mut arena: Arena) -> Deferred[_GeospatialGetCall[origin_of(arena)], Self.origin]:
        """Submits `GeospatialVerbs.get`; the arena stays borrowed until final. Calls `usearch_geospatial_get`."""
        return {
            self.index[].library,
            _GeospatialGetCall[origin_of(arena)](key, arena._address),
            self.index[].address,
            self.executor,
        }

    def save[
        path_origin: ImmOrigin, //
    ](self, path: StringSlice[path_origin]) -> Deferred[
        _GeospatialSaveCall, ImmOrigin(origin_of(_anchor[Self.origin](), _anchor[path_origin]()))
    ]:
        """Submits `GeospatialVerbs.save`; the path stays borrowed until final. Calls `usearch_geospatial_save`."""
        return {
            self.index[].library,
            _GeospatialSaveCall(BytesView.of(path).arguments()),
            self.index[].address,
            self.executor,
        }

    def save_buffer(self, mut arena: Arena) -> Deferred[_GeospatialSaveBufferCall[origin_of(arena)], Self.origin]:
        """Submits `GeospatialVerbs.save_buffer`; the arena stays borrowed until final.
        Calls `usearch_geospatial_save_buffer`."""
        return {
            self.index[].library,
            _GeospatialSaveBufferCall[origin_of(arena)](arena._address),
            self.index[].address,
            self.executor,
        }


struct GeospatialCursor[origin: ImmOrigin]:
    """A walk over the index contents - `usearch_geospatial_cursor_t`.

    Export, migration, and visualization feed off it. It borrows the index and the executor its
    pulls run on, so neither can be released while it lives.

    Parameters:
        origin: The index and the executor, which the cursor borrows.
    """

    var library: Library
    """The loaded engine, held so the release can still reach `usearch_geospatial_cursor_free`."""
    var index: Address
    """The `usearch_geospatial_index_t` the cursor was opened on, which every pull names again."""
    var executor: Address
    """The `usearch_executor_t` every pull runs on."""
    var address: Address
    """The `usearch_geospatial_cursor_t` this struct owns."""

    def __init__(
        out self,
        library: Library,
        index: Address,
        executor: Address,
        starts_count: Int,
        starts: Optional[Pointer[Key, ImmUntrackedOrigin]],
    ) raises Error:
        """Opens a cursor over `index`. Calls `usearch_geospatial_cursor_init`.

        Args:
            library: The loaded engine the cursor is opened through.
            index: The handle to scan.
            executor: The executor every pull runs on.
            starts_count: The start keys at `starts`; zero walks every entry as stored.
            starts: The first start key, or `None` for none.

        Raises:
            `Error` if the residency has no cursor, as VRAM does not.
        """
        var created = stack_allocation[1, Address]()
        created[unsafe_offset=0] = NULL_ADDRESS
        var outcome = stack_allocation[1, Outcome]()
        outcome[unsafe_offset=0] = Outcome()
        _cross(
            DenseCursorInitArguments(
                index,
                outcome.unsafe_origin_cast[MutUntrackedOrigin](),
                c_size_t(starts_count),
                starts,
                created.unsafe_origin_cast[MutUntrackedOrigin](),
            ),
            library.symbols().geospatial_cursor_init,
        )
        library._check(outcome.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0], "usearch_geospatial_cursor_init")
        self.library = library
        self.index = index
        self.executor = executor
        self.address = created.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0]

    def __deinit__(deinit self):
        """Releases the cursor. Calls `usearch_geospatial_cursor_free`."""
        self.library.symbols().geospatial_cursor_free(self.address)
        _ = self.library^

    def next_keys(self, wanted: Int, mut arena: Arena) raises Error -> DenseBatch[origin_of(arena)]:
        """Copies up to `wanted` next keys into a lent arena, skipping their links.
        Calls `usearch_geospatial_cursor_next`.

        Args:
            wanted: The upper bound on keys this pull writes.
            arena: The arena to rewind and refill, which the batch borrows; a failed call leaves it valid.

        Returns:
            The keys, read in place from the arena; an empty batch once the walk is exhausted.

        Raises:
            `Error` if the engine refuses the copy.
        """
        return self._next(_GeospatialCursorNextCall[origin_of(arena)](self.address, False, wanted, arena._address))

    def next_links(self, wanted: Int, mut arena: Arena) raises Error -> DenseBatch[origin_of(arena)]:
        """Copies up to `wanted` next keys into a lent arena, each with the keys it links to.
        Calls `usearch_geospatial_cursor_next`.

        Args:
            wanted: The upper bound on entries this pull writes.
            arena: The arena to rewind and refill, which the batch borrows; a failed call leaves it valid.

        Returns:
            The keys and their links, read in place from the arena; an empty batch once the walk is exhausted.

        Raises:
            `Error` if the engine refuses the copy.
        """
        return self._next(_GeospatialCursorNextCall[origin_of(arena)](self.address, True, wanted, arena._address))

    def _next[C: Call](self, var call: C) raises Error -> C.Result:
        """One blocking pull on the cursor's executor."""
        return _block(self.library, call^, self.index, self.executor)


# endregion Index

# region Reshard


def geospatial_reshard[
    sources_origin: ImmOrigin,
    targets_origin: ImmOrigin,
    stats_origin: MutOrigin,
    plan_origin: MutOrigin,
    phase_origin: MutOrigin,
    //,
](
    executor: Executor,
    sources: Span[GeospatialIndex, sources_origin],
    targets: Span[GeospatialIndex, targets_origin],
    config: GeospatialReshardConfig[stats_origin, plan_origin, phase_origin] = GeospatialReshardConfig(),
) raises Error -> DenseRefineReport:
    """Moves the entries of `sources` into `targets`, keeping their links, then runs a round on every
    target that took entries. Calls `usearch_geospatial_reshard`.

    Several sources into one target among them gather into it, and one source into several empty
    targets cuts it into one graph-local part apiece; a source the targets do not name ends empty and
    live.

    Parameters:
        sources_origin: The sources' origin, inferred from the argument.
        targets_origin: The targets' origin, inferred from the argument.
        stats_origin: The lent telemetry slot's origin, inferred from the config.
        plan_origin: The lent plan's origin, inferred with it.
        phase_origin: The lent phase slot's origin, inferred with it.

    Args:
        executor: The executor every move and the closing rounds fan out over; it also lends its engine.
        sources: The indexes whose entries move, each named once.
        targets: The indexes the entries move into, each named once, a source itself or empty.
        config: The rounds' spending, the seed of a cut, the handler, and the telemetry, plan and
            phase slots.

    Returns:
        What the closing rounds did to the graphs.

    Raises:
        `Error` with `INVALID_ARGUMENT` for an empty list or an index named twice, `CONFIG_MISMATCH`
        for a target neither a source nor empty, or `UNSUPPORTED` for one index recut in place or
        several sources into several targets.
    """
    return _block(executor.library, _GeospatialReshardCall(sources, targets, config), NULL_ADDRESS, executor.address)


def geospatial_reshard_deferred[
    sources_origin: ImmOrigin,
    targets_origin: ImmOrigin,
    stats_origin: MutOrigin,
    plan_origin: MutOrigin,
    phase_origin: MutOrigin,
    //,
](
    ref executor: Executor,
    sources: Span[GeospatialIndex, sources_origin],
    targets: Span[GeospatialIndex, targets_origin],
    config: GeospatialReshardConfig[stats_origin, plan_origin, phase_origin] = GeospatialReshardConfig(),
) -> Deferred[
    _GeospatialReshardCall,
    ImmOrigin(
        origin_of(
            executor,
            _anchor[sources_origin](),
            _anchor[targets_origin](),
            _anchor[stats_origin](),
            _anchor[plan_origin](),
            _anchor[phase_origin](),
        )
    ),
]:
    """Submits `geospatial_reshard` to `executor`, answering a `Deferred` at once; the indexes and slots
    stay borrowed until final. Calls `usearch_geospatial_reshard`."""
    return {executor.library, _GeospatialReshardCall(sources, targets, config), NULL_ADDRESS, executor.address}


def geospatial_reshard_plan_finish(
    executor: Executor, var plan: GeospatialReshardPlan
) raises Error -> DenseRefineReport:
    """Closes and frees a reshard plan whatever the call answers, reopening every index it names.
    Calls `usearch_geospatial_reshard_plan_finish`.

    Args:
        executor: The executor the plan was opened with; it also lends its engine.
        plan: The plan a reshard opened, consumed by the call.

    Returns:
        What the whole plan did to the graphs; zeros while entries were still to move.

    Raises:
        `Error` with `INVALID_ARGUMENT` for a plan no reshard opened, or one opened with another executor.
    """
    return _block(executor.library, _GeospatialReshardPlanFinishCall(plan.address), NULL_ADDRESS, executor.address)


def geospatial_reshard_plan_finish_deferred(
    ref executor: Executor, var plan: GeospatialReshardPlan
) -> Deferred[_GeospatialReshardPlanFinishCall, ImmOrigin(origin_of(executor))]:
    """Submits `geospatial_reshard_plan_finish` to `executor`, answering a `Deferred` at once.
    Calls `usearch_geospatial_reshard_plan_finish`."""
    return {executor.library, _GeospatialReshardPlanFinishCall(plan.address), NULL_ADDRESS, executor.address}


# endregion Reshard

# region Metadata


def _read_metadata(
    executor: Executor, verb: GeospatialMetadataRead, source: BytesView[ImmUntrackedOrigin], detail: StaticString
) raises Error -> GeospatialMetadata:
    """The path and buffer forms share one argument layout, differing only in the verb."""
    var described = stack_allocation[1, GeospatialMetadata]()
    described[unsafe_offset=0] = GeospatialMetadata()
    var outcome = stack_allocation[1, Outcome]()
    outcome[unsafe_offset=0] = Outcome()
    _cross(
        GeospatialMetadataReadArguments(
            executor.address,
            outcome.unsafe_origin_cast[MutUntrackedOrigin](),
            Completion.BLOCKING,
            source,
            described.unsafe_origin_cast[MutUntrackedOrigin](),
        ),
        verb,
    )
    executor.library._check(outcome.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0], detail)
    return described.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0]


def geospatial_metadata_read[
    path_origin: ImmOrigin, //
](executor: Executor, path: StringSlice[path_origin]) raises Error -> GeospatialMetadata:
    """Reads a checkpoint's connectivity without loading it. Calls `usearch_geospatial_metadata_read`.

    Parameters:
        path_origin: The path bytes' origin, inferred from the argument.

    Args:
        executor: The executor the read runs on; it also lends its engine.
        path: The manifest of a sharded checkpoint, or a single monolith.

    Returns:
        The connectivity the checkpoint froze.

    Raises:
        `Error` if the checkpoint cannot be opened or its header cannot be decoded.
    """
    return _read_metadata(
        executor,
        executor.library.symbols().geospatial_metadata_read,
        BytesView.of(path).arguments(),
        "usearch_geospatial_metadata_read",
    )


def geospatial_metadata_read_buffer[
    buffer_origin: ImmOrigin, //
](executor: Executor, buffer: Span[UInt8, buffer_origin]) raises Error -> GeospatialMetadata:
    """Reads a serialized monolith's connectivity without loading it.
    Calls `usearch_geospatial_metadata_read_buffer`.

    Parameters:
        buffer_origin: The source bytes' origin, inferred from the argument.

    Args:
        executor: The executor the read runs on; it also lends its engine.
        buffer: Source holding one serialized monolith.

    Returns:
        The connectivity the monolith froze.

    Raises:
        `Error` if the bytes cannot be decoded.
    """
    return _read_metadata(
        executor,
        executor.library.symbols().geospatial_metadata_read_buffer,
        BytesView.of(buffer).arguments(),
        "usearch_geospatial_metadata_read_buffer",
    )


# endregion Metadata
