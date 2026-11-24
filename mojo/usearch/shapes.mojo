"""Similarity search over 3D point clouds of any size: the index, its receivers, and its cursor.

Molecules, conformers and protein backbones arrive as a `TapeView` of points, every shape's `x`,
`y` and `z` end to end between Apache Arrow offsets that count points, and score under Chamfer,
Hausdorff, Gromov-Wasserstein or Wasserstein-Procrustes, invariant to rotation, reflection,
translation and point order.

The verbs mirror the dense ones, and so do the receivers: verbs on the index run on the calling
thread, `index.on(executor)` blocks on that executor, and `index.deferred(executor)` submits there
and answers a `Deferred` at once. The reports, the telemetry, the phases and the insert and search
knobs are the dense module's own, since the C interface shares them across every modality.

File: mojo/usearch/shapes.mojo
Author: Ash Vardanian
Date: November 16, 2026
"""

from std.ffi import c_int, c_size_t
from std.memory import stack_allocation
from std.sys.info import size_of

from usearch.basic import TapeEntries, TapeViewArguments, TapeView, _no_tape
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
    DenseSearchConfig,
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
    Error,
    Executor,
    GPUClass,
    Key,
    KeyBytes,
    KeysView,
    LoadPath,
    Matches,
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
)

# region Configuration


@fieldwise_init
struct ShapesMetric(Equatable, ImplicitlyCopyable, TrivialRegisterPassable):
    """How two shapes compare, with no correspondence between their points - `usearch_shapes_metric_t`.
    `Library.shapes_metric_name` spells each one as the engine does."""

    var code: c_int
    """The raw `usearch_shapes_metric_t` value the ABI exchanges."""
    comptime UNKNOWN = Self(0)
    """No kind named - an unset field, refused by every constructor."""
    comptime CHAMFER = Self(1)
    """The mean distance to each point's nearest partner, both directions, halved; the cheapest kind."""
    comptime HAUSDORFF = Self(2)
    """The worst of all nearest-partner distances, both directions, so one stray point decides."""
    comptime GROMOV_WASSERSTEIN = Self(3)
    """Transport over each side's internal geometry: the best ranker, and the costliest."""
    comptime WASSERSTEIN_PROCRUSTES = Self(4)
    """The residual after the best rigid motion and the best matching of the points."""


@fieldwise_init
struct ShapesOptions(ImplicitlyCopyable):
    """The geometry and kind an empty index freezes - what `usearch_shapes_init_ram_t` and
    `usearch_shapes_init_vram_t` share, without the placement each names its own way."""

    var metric: ShapesMetric
    """The kind comparing two shapes."""
    var quantization: ScalarKind
    """The scalar type points are stored in: `F16` or `F32`, any other refused with `UNSUPPORTED`."""
    var points: c_size_t
    """The frozen point count when every entry has one; zero admits any count."""
    var connectivity: c_size_t
    """The connections-per-node bound; zero derives."""
    var values_per_key: c_size_t
    """Upper bound on entries sharing one key; zero is unbounded."""

    def __init__(
        out self,
        metric: ShapesMetric = ShapesMetric.CHAMFER,
        quantization: ScalarKind = ScalarKind.F32,
        points: Int = 0,
        connectivity: Int = 0,
        values_per_key: Int = 0,
    ):
        """Describes an index by its kind and storage, Chamfer over `f32` points unless told otherwise.

        Args:
            metric: The kind the graph is frozen under.
            quantization: The scalar type points are stored in: `F16` or `F32`.
            points: The frozen point count when every entry has one; zero admits any count.
            connectivity: The connections-per-node bound; zero derives.
            values_per_key: Upper bound on entries sharing one key; zero is unbounded.
        """
        self.metric = metric
        self.quantization = quantization
        self.points = c_size_t(points)
        self.connectivity = c_size_t(connectivity)
        self.values_per_key = c_size_t(values_per_key)


@fieldwise_init
struct ShapesMetadata(Defaultable, ImplicitlyCopyable):
    """The geometry and kind a checkpoint froze - `usearch_shapes_metadata_t`, a pure output."""

    var metric: ShapesMetric
    """The kind the checkpoint compares shapes with."""
    var quantization: ScalarKind
    """The scalar type its points are stored in."""
    var points: c_size_t
    """The frozen point count, or zero for any count."""
    var connectivity: c_size_t
    """Connections per node the graph was built with."""

    def __init__(out self):
        """Names no kind, so an unwritten slot reads as an unanswered probe."""
        self.metric = ShapesMetric.UNKNOWN
        self.quantization = ScalarKind.UNKNOWN
        self.points = c_size_t(0)
        self.connectivity = c_size_t(0)


@fieldwise_init
struct ShapesIndexDescription(Defaultable, ImplicitlyCopyable):
    """`usearch_shapes_index_description_t`: a live index's frozen metadata and current counters."""

    var metadata: ShapesMetadata
    """The frozen kind, quantization, points and connectivity, as a checkpoint reports them."""
    var size: c_size_t
    """Stored entries."""
    var capacity: c_size_t
    """Entries hosted without growing again."""
    var total_points: c_size_t
    """Points summed across every stored entry."""
    var memory_usage: c_size_t
    """Bytes the handle holds, point storage included; traversal scratch is the executor's."""
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
        self.metadata = ShapesMetadata()
        self.size = c_size_t(0)
        self.capacity = c_size_t(0)
        self.total_points = c_size_t(0)
        self.memory_usage = c_size_t(0)
        self.cpu_class = CPUClass.UNKNOWN
        self.gpu_class = GPUClass.UNKNOWN
        self.shards_count = c_size_t(0)
        self.load_path = LoadPath.NONE


@fieldwise_init
struct ShapesKeyStats(Defaultable, ImplicitlyCopyable):
    """What `count` reports for one key - `usearch_shapes_key_stats_t`."""

    var entries: c_size_t
    """Entries stored under the key; zero means the index holds none."""
    var total_points: c_size_t
    """Points those entries hold together, which sizes a `get`."""
    var largest_points: c_size_t
    """Points of the largest of them."""

    def __init__(out self):
        """Zeroes every count, which is what a failed call leaves."""
        self.entries = c_size_t(0)
        self.total_points = c_size_t(0)
        self.largest_points = c_size_t(0)


struct ShapesRefinePlan(Movable):
    """One refine in flight - `usearch_shapes_refine_plan_t`. Binding-only.

    Lent through `ShapesRefineConfig.plan`, it opens on the first refine, stays live past a cancel
    or the end of a round, continues on every later refine, and closes with `refine_plan_finish`.
    While it lives, the index admits no insertion and no save.
    """

    var address: Address
    """The opaque handle, `NULL_ADDRESS` until a refine opens it."""

    def __init__(out self):
        """A plan no refine opened yet."""
        self.address = NULL_ADDRESS


struct ShapesReshardPlan(Movable):
    """One reshard in flight - `usearch_shapes_reshard_plan_t`, with its own lists of sources and
    targets. Binding-only.

    Lent through `ShapesReshardConfig.plan`, it opens on the first reshard, stays live past a
    cancel, moves what is still pending and continues the rounds on every later one, and closes with
    `shapes_reshard_plan_finish`. While it lives, every index it names admits no insertion and no
    save.
    """

    var address: Address
    """The opaque handle, `NULL_ADDRESS` until a reshard opens it."""

    def __init__(out self):
        """A plan no reshard opened yet."""
        self.address = NULL_ADDRESS


# endregion Configuration

# region Configs


struct ShapesRefineConfig[
    stats_origin: MutOrigin = MutUntrackedOrigin,
    plan_origin: MutOrigin = MutUntrackedOrigin,
    phase_origin: MutOrigin = MutUntrackedOrigin,
](ImplicitlyCopyable):
    """The knobs of `usearch_shapes_refine_t`; zeroed or absent ones derive what the C interface derives.

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
    var plan: Optional[Pointer[ShapesRefinePlan, Self.plan_origin]]
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
        plan: Optional[Pointer[ShapesRefinePlan, Self.plan_origin]] = Optional[
            Pointer[ShapesRefinePlan, Self.plan_origin]
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


struct ShapesReshardConfig[
    stats_origin: MutOrigin = MutUntrackedOrigin,
    plan_origin: MutOrigin = MutUntrackedOrigin,
    phase_origin: MutOrigin = MutUntrackedOrigin,
](ImplicitlyCopyable):
    """The knobs of `usearch_shapes_reshard_t`; zeroed or absent ones derive what the C interface derives.

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
    var plan: Optional[Pointer[ShapesReshardPlan, Self.plan_origin]]
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
        plan: Optional[Pointer[ShapesReshardPlan, Self.plan_origin]] = Optional[
            Pointer[ShapesReshardPlan, Self.plan_origin]
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


# endregion Configs

# region Arguments


@fieldwise_init
struct ShapesInitRamArguments(ImplicitlyCopyable):
    """`usearch_shapes_init_ram_t`."""

    var index: Pointer[Address, MutUntrackedOrigin]
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var metric: ShapesMetric
    var quantization: ScalarKind
    var points: c_size_t
    var connectivity: c_size_t
    var values_per_key: c_size_t
    var ram: RamOptions[ImmUntrackedOrigin]


@fieldwise_init
struct ShapesInitVramArguments(ImplicitlyCopyable):
    """`usearch_shapes_init_vram_t`."""

    var index: Pointer[Address, MutUntrackedOrigin]
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var metric: ShapesMetric
    var quantization: ScalarKind
    var points: c_size_t
    var connectivity: c_size_t
    var values_per_key: c_size_t
    var vram: VramOptions[ImmUntrackedOrigin]


@fieldwise_init
struct ShapesLoadRamArguments(ImplicitlyCopyable):
    """`usearch_shapes_load_ram_t`, and `usearch_shapes_load_ram_buffer_t` with `source` a buffer."""

    var index: Pointer[Address, MutUntrackedOrigin]
    var executor: Address
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var completion: Completion
    var source: BytesView[ImmUntrackedOrigin]
    var ram: RamOptions[ImmUntrackedOrigin]
    var values_per_key: c_size_t
    var progress: ProgressArguments


@fieldwise_init
struct ShapesLoadVramArguments(ImplicitlyCopyable):
    """`usearch_shapes_load_vram_t`, and `usearch_shapes_load_vram_buffer_t` with `source` a buffer."""

    var index: Pointer[Address, MutUntrackedOrigin]
    var executor: Address
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var completion: Completion
    var source: BytesView[ImmUntrackedOrigin]
    var vram: VramOptions[ImmUntrackedOrigin]
    var values_per_key: c_size_t
    var progress: ProgressArguments


@fieldwise_init
struct ShapesViewRamArguments(ImplicitlyCopyable):
    """`usearch_shapes_view_ram_t`."""

    var index: Pointer[Address, MutUntrackedOrigin]
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var path: BytesView[ImmUntrackedOrigin]
    var ram: RamOptions[ImmUntrackedOrigin]


@fieldwise_init
struct ShapesViewNvmeArguments(ImplicitlyCopyable):
    """`usearch_shapes_view_nvme_t`."""

    var index: Pointer[Address, MutUntrackedOrigin]
    var executor: Address
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var completion: Completion
    var path: BytesView[ImmUntrackedOrigin]
    var nvme: NvmeOptions[ImmUntrackedOrigin]


@fieldwise_init
struct ShapesReserveArguments(ImplicitlyCopyable):
    """`usearch_shapes_reserve_t`."""

    var index: Address
    var executor: Address
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var completion: Completion
    var entries: c_size_t
    var points: c_size_t


@fieldwise_init
struct ShapesDescribeArguments(ImplicitlyCopyable):
    """`usearch_shapes_describe_t`."""

    var index: Address
    var executor: Address
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var completion: Completion
    var description: Pointer[ShapesIndexDescription, MutUntrackedOrigin]


@fieldwise_init
struct ShapesInsertArguments(ImplicitlyCopyable):
    """`usearch_shapes_insert_t`."""

    var index: Address
    var executor: Address
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var completion: Completion
    var count: c_size_t
    var keys: KeysView[ImmUntrackedOrigin]
    var shapes: TapeViewArguments
    var expansion_factor: c_size_t
    var progress: ProgressArguments
    var stats: Optional[Pointer[DenseTraversalStats, MutUntrackedOrigin]]


@fieldwise_init
struct ShapesRefineArguments(ImplicitlyCopyable):
    """`usearch_shapes_refine_t`."""

    var index: Address
    var executor: Address
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var completion: Completion
    var report: Pointer[DenseRefineReport, MutUntrackedOrigin]
    var refine: RefineOptionsArguments
    var progress: ProgressArguments
    var stats: Optional[Pointer[DenseTraversalStats, MutUntrackedOrigin]]
    var plan: Optional[Pointer[ShapesRefinePlan, MutUntrackedOrigin]]
    var phase: Optional[Pointer[PhaseProgress, MutUntrackedOrigin]]


@fieldwise_init
struct ShapesSearchArguments(ImplicitlyCopyable):
    """`usearch_shapes_search_t`."""

    var index: Address
    var executor: Address
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var completion: Completion
    var count: c_size_t
    var queries: TapeViewArguments
    var matches: Pointer[MatchesViewArguments, MutUntrackedOrigin]
    var arena: Address
    var wanted: c_size_t
    var expansion_factor: c_size_t
    var traversal: Traversal
    var predicate: PredicateArguments
    var hints: Optional[Pointer[MatchesViewArguments, ImmUntrackedOrigin]]
    var progress: ProgressArguments
    var stats: Optional[Pointer[DenseTraversalStats, MutUntrackedOrigin]]


@fieldwise_init
struct ShapesCountArguments(ImplicitlyCopyable):
    """`usearch_shapes_count_t`."""

    var index: Address
    var executor: Address
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var completion: Completion
    var key: KeyBytes
    var key_stats: Pointer[ShapesKeyStats, MutUntrackedOrigin]


@fieldwise_init
struct ShapesGetArguments(ImplicitlyCopyable):
    """`usearch_shapes_get_t`."""

    var index: Address
    var executor: Address
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var completion: Completion
    var key: KeyBytes
    var shapes: Pointer[TapeViewArguments, MutUntrackedOrigin]
    var written: Pointer[c_size_t, MutUntrackedOrigin]
    var arena: Address
    var outputs_scalar: ScalarKind


@fieldwise_init
struct ShapesReshardArguments(ImplicitlyCopyable):
    """`usearch_shapes_reshard_t`."""

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
    var plan: Optional[Pointer[ShapesReshardPlan, MutUntrackedOrigin]]
    var phase: Optional[Pointer[PhaseProgress, MutUntrackedOrigin]]


@fieldwise_init
struct ShapesMetadataReadArguments(ImplicitlyCopyable):
    """`usearch_shapes_metadata_read_t`, and `usearch_shapes_metadata_read_buffer_t` with `source` a
    buffer."""

    var executor: Address
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var completion: Completion
    var source: BytesView[ImmUntrackedOrigin]
    var metadata: Pointer[ShapesMetadata, MutUntrackedOrigin]


comptime ShapesInitRam = Verb[ShapesInitRamArguments]
comptime ShapesInitVram = Verb[ShapesInitVramArguments]
comptime ShapesLoadRam = Verb[ShapesLoadRamArguments]
comptime ShapesLoadVram = Verb[ShapesLoadVramArguments]
comptime ShapesViewRam = Verb[ShapesViewRamArguments]
comptime ShapesViewNvme = Verb[ShapesViewNvmeArguments]
comptime ShapesReserve = Verb[ShapesReserveArguments]
comptime ShapesClear = Verb[DenseClearArguments]
comptime ShapesDescribe = Verb[ShapesDescribeArguments]
comptime ShapesInsert = Verb[ShapesInsertArguments]
comptime ShapesRefine = Verb[ShapesRefineArguments]
comptime ShapesRefinePlanFinish = Verb[DenseRefinePlanFinishArguments]
comptime ShapesHealth = Verb[DenseHealthArguments]
comptime ShapesReshard = Verb[ShapesReshardArguments]
comptime ShapesReshardPlanFinish = Verb[DenseReshardPlanFinishArguments]
comptime ShapesSample = Verb[DenseSampleArguments]
comptime ShapesSearch = Verb[ShapesSearchArguments]
comptime ShapesCount = Verb[ShapesCountArguments]
comptime ShapesGet = Verb[ShapesGetArguments]
comptime ShapesCursorInit = Verb[DenseCursorInitArguments]
comptime ShapesCursorNext = Verb[DenseCursorNextArguments]
comptime ShapesSave = Verb[DenseSaveArguments]
comptime ShapesSaveBuffer = Verb[DenseSaveBufferArguments]
comptime ShapesMetadataRead = Verb[ShapesMetadataReadArguments]


# endregion Arguments

# region Calls


@fieldwise_init
struct _ShapesReserveCall(Call):
    """The crossing of `usearch_shapes_reserve`."""

    comptime Arguments = ShapesReserveArguments
    comptime Outputs = Unwritten
    comptime Result = NoneType
    comptime symbol: StaticString = "usearch_shapes_reserve"
    var entries: Int
    var points: Int

    @staticmethod
    def verb(library: Library) -> Verb[Self.Arguments]:
        return library.symbols().shapes_reserve

    def arguments(self, head: CallHead, outputs: Pointer[Self.Outputs, MutUntrackedOrigin]) -> Self.Arguments:
        return {
            head.index,
            head.executor,
            head.outcome,
            head.completion,
            c_size_t(self.entries),
            c_size_t(self.points),
        }

    def conclude(deinit self, outputs: Unwritten) -> NoneType:
        return None


@fieldwise_init
struct _ShapesClearCall(Call):
    """The crossing of `usearch_shapes_clear`."""

    comptime Arguments = DenseClearArguments
    comptime Outputs = Unwritten
    comptime Result = NoneType
    comptime symbol: StaticString = "usearch_shapes_clear"

    @staticmethod
    def verb(library: Library) -> Verb[Self.Arguments]:
        return library.symbols().shapes_clear

    def arguments(self, head: CallHead, outputs: Pointer[Self.Outputs, MutUntrackedOrigin]) -> Self.Arguments:
        return {head.index, head.executor, head.outcome, head.completion}

    def conclude(deinit self, outputs: Unwritten) -> NoneType:
        return None


@fieldwise_init
struct _ShapesDescribeCall(Call):
    """The crossing of `usearch_shapes_describe`."""

    comptime Arguments = ShapesDescribeArguments
    comptime Outputs = ShapesIndexDescription
    comptime Result = ShapesIndexDescription
    comptime symbol: StaticString = "usearch_shapes_describe"

    @staticmethod
    def verb(library: Library) -> Verb[Self.Arguments]:
        return library.symbols().shapes_describe

    def arguments(self, head: CallHead, outputs: Pointer[Self.Outputs, MutUntrackedOrigin]) -> Self.Arguments:
        return {head.index, head.executor, head.outcome, head.completion, outputs}

    def conclude(deinit self, outputs: ShapesIndexDescription) -> ShapesIndexDescription:
        return outputs


@fieldwise_init
struct _ShapesCountCall(Call):
    """The crossing of `usearch_shapes_count`."""

    comptime Arguments = ShapesCountArguments
    comptime Outputs = ShapesKeyStats
    comptime Result = ShapesKeyStats
    comptime symbol: StaticString = "usearch_shapes_count"
    var key: Key

    @staticmethod
    def verb(library: Library) -> Verb[Self.Arguments]:
        return library.symbols().shapes_count

    def arguments(self, head: CallHead, outputs: Pointer[Self.Outputs, MutUntrackedOrigin]) -> Self.Arguments:
        return {head.index, head.executor, head.outcome, head.completion, KeyBytes.of(self.key), outputs}

    def conclude(deinit self, outputs: ShapesKeyStats) -> ShapesKeyStats:
        return outputs


@fieldwise_init
struct ShapesGetOutputs(Defaultable, ImplicitlyCopyable):
    """What `usearch_shapes_get` writes through its pointers."""

    var shapes: TapeViewArguments
    var written: c_size_t

    def __init__(out self):
        """Nothing exported."""
        self.shapes = _no_tape()
        self.written = c_size_t(0)


@fieldwise_init
struct _ShapesGetCall[dtype: DType, kind: ScalarKind, origin: MutOrigin](Call):
    """The crossing of `usearch_shapes_get`, which borrows the arena its tape lands in."""

    comptime Arguments = ShapesGetArguments
    comptime Outputs = ShapesGetOutputs
    comptime Result = TapeEntries[Self.dtype, Self.origin, 3]
    comptime symbol: StaticString = "usearch_shapes_get"
    var key: Key
    var arena: Address

    @staticmethod
    def verb(library: Library) -> Verb[Self.Arguments]:
        return library.symbols().shapes_get

    def arguments(self, head: CallHead, outputs: Pointer[Self.Outputs, MutUntrackedOrigin]) -> Self.Arguments:
        return {
            head.index,
            head.executor,
            head.outcome,
            head.completion,
            KeyBytes.of(self.key),
            Pointer(to=outputs[].shapes),
            Pointer(to=outputs[].written),
            self.arena,
            Self.kind,
        }

    def conclude(deinit self, outputs: ShapesGetOutputs) -> Self.Result:
        return {outputs.shapes, Int(outputs.written)}


@fieldwise_init
struct _ShapesInsertCall(Call):
    """The crossing of `usearch_shapes_insert`."""

    comptime Arguments = ShapesInsertArguments
    comptime Outputs = Unwritten
    comptime Result = NoneType
    comptime symbol: StaticString = "usearch_shapes_insert"
    var keys: KeysView[ImmUntrackedOrigin]
    var keys_count: Int
    var shapes: TapeViewArguments
    var count: Int
    var expansion_factor: Int
    var progress: ProgressArguments
    var stats: Optional[Pointer[DenseTraversalStats, MutUntrackedOrigin]]

    @staticmethod
    def verb(library: Library) -> Verb[Self.Arguments]:
        return library.symbols().shapes_insert

    def arguments(self, head: CallHead, outputs: Pointer[Self.Outputs, MutUntrackedOrigin]) -> Self.Arguments:
        return {
            head.index,
            head.executor,
            head.outcome,
            head.completion,
            c_size_t(self.count),
            self.keys,
            self.shapes,
            c_size_t(self.expansion_factor),
            self.progress,
            self.stats,
        }

    def lengths_agree(self) -> Bool:
        return self.keys_count == self.count

    def conclude(deinit self, outputs: Unwritten) -> NoneType:
        return None


@fieldwise_init
struct _ShapesRefineCall(Call):
    """The crossing of `usearch_shapes_refine`."""

    comptime Arguments = ShapesRefineArguments
    comptime Outputs = DenseRefineReport
    comptime Result = DenseRefineReport
    comptime symbol: StaticString = "usearch_shapes_refine"
    var refine: RefineOptionsArguments
    var progress: ProgressArguments
    var stats: Optional[Pointer[DenseTraversalStats, MutUntrackedOrigin]]
    var plan: Optional[Pointer[ShapesRefinePlan, MutUntrackedOrigin]]
    var phase: Optional[Pointer[PhaseProgress, MutUntrackedOrigin]]

    @staticmethod
    def verb(library: Library) -> Verb[Self.Arguments]:
        return library.symbols().shapes_refine

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
struct _ShapesRefinePlanFinishCall(Call):
    """The crossing of `usearch_shapes_refine_plan_finish`, which frees the plan whatever it answers."""

    comptime Arguments = DenseRefinePlanFinishArguments
    comptime Outputs = DenseRefineReport
    comptime Result = DenseRefineReport
    comptime symbol: StaticString = "usearch_shapes_refine_plan_finish"
    var plan: Address

    @staticmethod
    def verb(library: Library) -> Verb[Self.Arguments]:
        return library.symbols().shapes_refine_plan_finish

    def arguments(self, head: CallHead, outputs: Pointer[Self.Outputs, MutUntrackedOrigin]) -> Self.Arguments:
        return {head.index, head.executor, head.outcome, head.completion, self.plan, outputs}

    def conclude(deinit self, outputs: DenseRefineReport) -> DenseRefineReport:
        return outputs


@fieldwise_init
struct _ShapesHealthCall(Call):
    """The crossing of `usearch_shapes_health`."""

    comptime Arguments = DenseHealthArguments
    comptime Outputs = DenseReachability
    comptime Result = DenseReachability
    comptime symbol: StaticString = "usearch_shapes_health"

    @staticmethod
    def verb(library: Library) -> Verb[Self.Arguments]:
        return library.symbols().shapes_health

    def arguments(self, head: CallHead, outputs: Pointer[Self.Outputs, MutUntrackedOrigin]) -> Self.Arguments:
        return {head.index, head.executor, head.outcome, head.completion, outputs}

    def conclude(deinit self, outputs: DenseReachability) -> DenseReachability:
        return outputs


struct _ShapesReshardCall(Call):
    """The crossing of `usearch_shapes_reshard`, which keeps the handle lists on the heap until the
    call is final."""

    comptime Arguments = ShapesReshardArguments
    comptime Outputs = DenseRefineReport
    comptime Result = DenseRefineReport
    comptime symbol: StaticString = "usearch_shapes_reshard"
    var sources: List[Address]
    var targets: List[Address]
    var refine: RefineOptionsArguments
    var seed: UInt64
    var progress: ProgressArguments
    var stats: Optional[Pointer[DenseTraversalStats, MutUntrackedOrigin]]
    var plan: Optional[Pointer[ShapesReshardPlan, MutUntrackedOrigin]]
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
        sources: Span[ShapesIndex, sources_origin],
        targets: Span[ShapesIndex, targets_origin],
        config: ShapesReshardConfig[stats_origin, plan_origin, phase_origin],
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
        return library.symbols().shapes_reshard

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
struct _ShapesReshardPlanFinishCall(Call):
    """The crossing of `usearch_shapes_reshard_plan_finish`, which frees the plan whatever it answers."""

    comptime Arguments = DenseReshardPlanFinishArguments
    comptime Outputs = DenseRefineReport
    comptime Result = DenseRefineReport
    comptime symbol: StaticString = "usearch_shapes_reshard_plan_finish"
    var plan: Address

    @staticmethod
    def verb(library: Library) -> Verb[Self.Arguments]:
        return library.symbols().shapes_reshard_plan_finish

    def arguments(self, head: CallHead, outputs: Pointer[Self.Outputs, MutUntrackedOrigin]) -> Self.Arguments:
        return {head.executor, head.outcome, head.completion, self.plan, outputs}

    def conclude(deinit self, outputs: DenseRefineReport) -> DenseRefineReport:
        return outputs


@fieldwise_init
struct _ShapesSampleCall[origin: MutOrigin](Call):
    """The crossing of `usearch_shapes_sample`, which borrows the arena its draws are written in."""

    comptime Arguments = DenseSampleArguments
    comptime Outputs = DenseSampleOutputs
    comptime Result = DenseDraws[Self.origin]
    comptime symbol: StaticString = "usearch_shapes_sample"
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
        return library.symbols().shapes_sample

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
struct _ShapesSearchCall[origin: MutOrigin](Call):
    """The crossing of `usearch_shapes_search`, which borrows the arena its matches land in."""

    comptime Arguments = ShapesSearchArguments
    comptime Outputs = MatchesViewArguments
    comptime Result = Matches[Self.origin]
    comptime symbol: StaticString = "usearch_shapes_search"
    var queries: TapeViewArguments
    var count: Int
    var wanted: Int
    var arena: Address
    var expansion_factor: Int
    var traversal: Traversal
    var predicate: PredicateArguments
    var progress: ProgressArguments
    var stats: Optional[Pointer[DenseTraversalStats, MutUntrackedOrigin]]

    @staticmethod
    def verb(library: Library) -> Verb[Self.Arguments]:
        return library.symbols().shapes_search

    def arguments(self, head: CallHead, outputs: Pointer[Self.Outputs, MutUntrackedOrigin]) -> Self.Arguments:
        return {
            head.index,
            head.executor,
            head.outcome,
            head.completion,
            c_size_t(self.count),
            self.queries,
            outputs,
            self.arena,
            c_size_t(self.wanted),
            c_size_t(self.expansion_factor),
            self.traversal,
            self.predicate,
            None,
            self.progress,
            self.stats,
        }

    def conclude(deinit self, outputs: MatchesViewArguments) -> Self.Result:
        return {outputs, self.count}


@fieldwise_init
struct _ShapesSaveCall(Call):
    """The crossing of `usearch_shapes_save`."""

    comptime Arguments = DenseSaveArguments
    comptime Outputs = Unwritten
    comptime Result = NoneType
    comptime symbol: StaticString = "usearch_shapes_save"
    var path: BytesView[ImmUntrackedOrigin]

    @staticmethod
    def verb(library: Library) -> Verb[Self.Arguments]:
        return library.symbols().shapes_save

    def arguments(self, head: CallHead, outputs: Pointer[Self.Outputs, MutUntrackedOrigin]) -> Self.Arguments:
        return {head.index, head.executor, head.outcome, head.completion, self.path, ProgressArguments()}

    def conclude(deinit self, outputs: Unwritten) -> NoneType:
        return None


@fieldwise_init
struct _ShapesSaveBufferCall[origin: MutOrigin](Call):
    """The crossing of `usearch_shapes_save_buffer`, which borrows the arena the monolith lands in."""

    comptime Arguments = DenseSaveBufferArguments
    comptime Outputs = DenseSaveBufferOutputs
    comptime Result = Monolith[Self.origin]
    comptime symbol: StaticString = "usearch_shapes_save_buffer"
    var arena: Address

    @staticmethod
    def verb(library: Library) -> Verb[Self.Arguments]:
        return library.symbols().shapes_save_buffer

    def arguments(self, head: CallHead, outputs: Pointer[Self.Outputs, MutUntrackedOrigin]) -> Self.Arguments:
        return {head.index, head.executor, head.outcome, head.completion, outputs, self.arena, ProgressArguments()}

    def conclude(deinit self, outputs: DenseSaveBufferOutputs) -> Self.Result:
        return {outputs.data, Int(outputs.length)}


@fieldwise_init
struct _ShapesCursorNextCall[origin: MutOrigin](Call):
    """The crossing of `usearch_shapes_cursor_next`, which borrows the arena its batch lands in."""

    comptime Arguments = DenseCursorNextArguments
    comptime Outputs = DenseCursorNextOutputs
    comptime Result = DenseBatch[Self.origin]
    comptime symbol: StaticString = "usearch_shapes_cursor_next"
    var cursor: Address
    var links: Bool
    var wanted: Int
    var arena: Address

    @staticmethod
    def verb(library: Library) -> Verb[Self.Arguments]:
        return library.symbols().shapes_cursor_next

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


struct ShapesIndex(Movable, ShapesVerbs):
    """A shape index, released when it goes out of scope.

    Insertions and searches overlap freely on a host handle, as on a dense one. Verbs called on the
    index run on the calling thread; `index.on(executor)` runs the same verbs blocking on an
    executor, and `index.deferred(executor)` submits them there.
    """

    var library: Library
    """The loaded engine, held so its symbol table outlives every call made through the handle."""
    var address: Address
    """The opaque `usearch_shapes_index_t`."""

    # region Lifetime

    def __init__(out self, *, library: Library, address: Address):
        """Adopts a handle the engine just created.

        Args:
            library: The loaded engine the handle was created through, and is freed through.
            address: A `usearch_shapes_index_t` whose lifetime this index takes over.
        """
        self.library = library
        self.address = address

    @staticmethod
    def init_ram[
        ram_origin: ImmOrigin, //
    ](
        library: Library, options: ShapesOptions = ShapesOptions(), ram: RamOptions[ram_origin] = RamOptions()
    ) raises Error -> Self:
        """Initializes an empty, mutable index in host memory. Calls `usearch_shapes_init_ram`.

        Parameters:
            ram_origin: The placement's borrowed domain array, inferred from the argument.

        Args:
            library: The loaded engine, kept so the shared object stays mapped.
            options: The kind, storage and frozen point count; read only during this call.
            ram: The memory domains to place one shard on apiece.

        Returns:
            The new index.

        Raises:
            `Error` with `UNSUPPORTED` for a quantization other than `F16` or `F32`.
        """
        var created = stack_allocation[1, Address]()
        created[unsafe_offset=0] = NULL_ADDRESS
        var outcome = stack_allocation[1, Outcome]()
        outcome[unsafe_offset=0] = Outcome()
        _cross(
            ShapesInitRamArguments(
                created.unsafe_origin_cast[MutUntrackedOrigin](),
                outcome.unsafe_origin_cast[MutUntrackedOrigin](),
                options.metric,
                options.quantization,
                options.points,
                options.connectivity,
                options.values_per_key,
                ram.arguments(),
            ),
            library.symbols().shapes_init_ram,
        )
        library._check(outcome.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0], "usearch_shapes_init_ram")
        return Self(library=library, address=created.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0])

    @staticmethod
    def init_vram[
        vram_origin: ImmOrigin, //
    ](
        library: Library, options: ShapesOptions = ShapesOptions(), vram: VramOptions[vram_origin] = VramOptions()
    ) raises Error -> Self:
        """Initializes an empty, mutable index in device memory. Calls `usearch_shapes_init_vram`.

        Parameters:
            vram_origin: The placement's borrowed ordinal array, inferred from the argument.

        Args:
            library: The loaded engine, kept so the shared object stays mapped.
            options: The kind, storage and frozen point count; read only during this call.
            vram: The device to place the index on.

        Returns:
            The new index.

        Raises:
            `Error` with `UNSUPPORTED` for a kind other than Chamfer or Hausdorff, or if no device can
            be selected or claimed.
        """
        var created = stack_allocation[1, Address]()
        created[unsafe_offset=0] = NULL_ADDRESS
        var outcome = stack_allocation[1, Outcome]()
        outcome[unsafe_offset=0] = Outcome()
        _cross(
            ShapesInitVramArguments(
                created.unsafe_origin_cast[MutUntrackedOrigin](),
                outcome.unsafe_origin_cast[MutUntrackedOrigin](),
                options.metric,
                options.quantization,
                options.points,
                options.connectivity,
                options.values_per_key,
                vram.arguments(),
            ),
            library.symbols().shapes_init_vram,
        )
        library._check(outcome.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0], "usearch_shapes_init_vram")
        return Self(library=library, address=created.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0])

    @staticmethod
    def load_ram[
        path_origin: ImmOrigin, ram_origin: ImmOrigin, //
    ](
        executor: Executor,
        path: StringSlice[path_origin],
        ram: RamOptions[ram_origin] = RamOptions(),
        values_per_key: Int = 0,
    ) raises Error -> Self:
        """Copies a checkpoint into a new, mutable index in host memory. Calls `usearch_shapes_load_ram`.

        Parameters:
            path_origin: The path bytes' origin, inferred from the argument.
            ram_origin: The placement's borrowed domain array, inferred from the argument.

        Args:
            executor: The executor the read fans out over; it also lends its engine.
            path: The manifest of a sharded checkpoint, or a single `.safegraphs` monolith.
            ram: The memory domains to place the parts on.
            values_per_key: Upper bound on entries sharing one key, which no checkpoint records.

        Returns:
            The new index; a failed load creates none.

        Raises:
            `Error` if the checkpoint cannot be read or decoded.
        """
        return Self._load_ram(
            executor,
            executor.library.symbols().shapes_load_ram,
            BytesView.of(path).arguments(),
            ram.arguments(),
            values_per_key,
            "usearch_shapes_load_ram",
        )

    @staticmethod
    def load_ram_buffer[
        buffer_origin: ImmOrigin, ram_origin: ImmOrigin, //
    ](
        executor: Executor,
        buffer: Span[UInt8, buffer_origin],
        ram: RamOptions[ram_origin] = RamOptions(),
        values_per_key: Int = 0,
    ) raises Error -> Self:
        """Copies a monolith from a memory buffer into a new, mutable index in host memory.
        Calls `usearch_shapes_load_ram_buffer`.

        Parameters:
            buffer_origin: The source bytes' origin, inferred from the argument.
            ram_origin: The placement's borrowed domain array, inferred from the argument.

        Args:
            executor: The executor the copy fans out over; it also lends its engine.
            buffer: Source holding one serialized monolith.
            ram: The memory domains to place shards on; the monolith lands whole on the first.
            values_per_key: Upper bound on entries sharing one key, which no checkpoint records.

        Returns:
            The new index; a failed load creates none.

        Raises:
            `Error` if the bytes cannot be decoded.
        """
        return Self._load_ram(
            executor,
            executor.library.symbols().shapes_load_ram_buffer,
            BytesView.of(buffer).arguments(),
            ram.arguments(),
            values_per_key,
            "usearch_shapes_load_ram_buffer",
        )

    @staticmethod
    def _load_ram(
        executor: Executor,
        verb: ShapesLoadRam,
        source: BytesView[ImmUntrackedOrigin],
        ram: RamOptions[ImmUntrackedOrigin],
        values_per_key: Int,
        detail: StaticString,
    ) raises Error -> Self:
        """The path and buffer forms share one argument layout, differing only in the verb."""
        var created = stack_allocation[1, Address]()
        created[unsafe_offset=0] = NULL_ADDRESS
        var outcome = stack_allocation[1, Outcome]()
        outcome[unsafe_offset=0] = Outcome()
        _cross(
            ShapesLoadRamArguments(
                created.unsafe_origin_cast[MutUntrackedOrigin](),
                executor.address,
                outcome.unsafe_origin_cast[MutUntrackedOrigin](),
                Completion.BLOCKING,
                source,
                ram,
                c_size_t(values_per_key),
                ProgressArguments(),
            ),
            verb,
        )
        executor.library._check(outcome.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0], detail)
        return Self(library=executor.library, address=created.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0])

    @staticmethod
    def load_vram[
        path_origin: ImmOrigin, vram_origin: ImmOrigin, //
    ](
        executor: Executor,
        path: StringSlice[path_origin],
        vram: VramOptions[vram_origin] = VramOptions(),
        values_per_key: Int = 0,
    ) raises Error -> Self:
        """Copies a checkpoint into a new, mutable index in device memory. Calls `usearch_shapes_load_vram`.

        Parameters:
            path_origin: The path bytes' origin, inferred from the argument.
            vram_origin: The placement's borrowed ordinal array, inferred from the argument.

        Args:
            executor: The executor the upload runs on; it also lends its engine.
            path: The manifest of a sharded checkpoint, or a single `.safegraphs` monolith.
            vram: The device to place the index on.
            values_per_key: Upper bound on entries sharing one key, which no checkpoint records.

        Returns:
            The new index; a failed load creates none.

        Raises:
            `Error` if the checkpoint cannot be read, or a device cannot hold it.
        """
        return Self._load_vram(
            executor,
            executor.library.symbols().shapes_load_vram,
            BytesView.of(path).arguments(),
            vram.arguments(),
            values_per_key,
            "usearch_shapes_load_vram",
        )

    @staticmethod
    def load_vram_buffer[
        buffer_origin: ImmOrigin, vram_origin: ImmOrigin, //
    ](
        executor: Executor,
        buffer: Span[UInt8, buffer_origin],
        vram: VramOptions[vram_origin] = VramOptions(),
        values_per_key: Int = 0,
    ) raises Error -> Self:
        """Copies a monolith from a memory buffer into a new, mutable index in device memory.
        Calls `usearch_shapes_load_vram_buffer`.

        Parameters:
            buffer_origin: The source bytes' origin, inferred from the argument.
            vram_origin: The placement's borrowed ordinal array, inferred from the argument.

        Args:
            executor: The executor the upload runs on; it also lends its engine.
            buffer: Source holding one serialized monolith.
            vram: The device to place the index on.
            values_per_key: Upper bound on entries sharing one key, which no checkpoint records.

        Returns:
            The new index; a failed load creates none.

        Raises:
            `Error` if the bytes cannot be decoded, or a device cannot hold them.
        """
        return Self._load_vram(
            executor,
            executor.library.symbols().shapes_load_vram_buffer,
            BytesView.of(buffer).arguments(),
            vram.arguments(),
            values_per_key,
            "usearch_shapes_load_vram_buffer",
        )

    @staticmethod
    def _load_vram(
        executor: Executor,
        verb: ShapesLoadVram,
        source: BytesView[ImmUntrackedOrigin],
        vram: VramOptions[ImmUntrackedOrigin],
        values_per_key: Int,
        detail: StaticString,
    ) raises Error -> Self:
        """The path and buffer forms share one argument layout, differing only in the verb."""
        var created = stack_allocation[1, Address]()
        created[unsafe_offset=0] = NULL_ADDRESS
        var outcome = stack_allocation[1, Outcome]()
        outcome[unsafe_offset=0] = Outcome()
        _cross(
            ShapesLoadVramArguments(
                created.unsafe_origin_cast[MutUntrackedOrigin](),
                executor.address,
                outcome.unsafe_origin_cast[MutUntrackedOrigin](),
                Completion.BLOCKING,
                source,
                vram,
                c_size_t(values_per_key),
                ProgressArguments(),
            ),
            verb,
        )
        executor.library._check(outcome.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0], detail)
        return Self(library=executor.library, address=created.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0])

    @staticmethod
    def view_ram[path_origin: ImmOrigin, //](library: Library, path: StringSlice[path_origin]) raises Error -> Self:
        """Serves a checkpoint in place through a memory mapping, as a read-only index.
        Calls `usearch_shapes_view_ram`.

        Parameters:
            path_origin: The path bytes' origin, inferred from the argument.

        Args:
            library: The loaded engine, kept so the shared object stays mapped.
            path: The manifest of a sharded checkpoint, or a single `.safegraphs` monolith.

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
            ShapesViewRamArguments(
                created.unsafe_origin_cast[MutUntrackedOrigin](),
                outcome.unsafe_origin_cast[MutUntrackedOrigin](),
                BytesView.of(path).arguments(),
                RamOptions().arguments(),
            ),
            library.symbols().shapes_view_ram,
        )
        library._check(outcome.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0], "usearch_shapes_view_ram")
        return Self(library=library, address=created.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0])

    @staticmethod
    def view_nvme[
        path_origin: ImmOrigin, nvme_origin: ImmOrigin, //
    ](
        executor: Executor, path: StringSlice[path_origin], nvme: NvmeOptions[nvme_origin] = NvmeOptions()
    ) raises Error -> Self:
        """Serves a checkpoint in place from NVMe drives, read through the executor's rings.
        Calls `usearch_shapes_view_nvme`.

        Parameters:
            path_origin: The path bytes' origin, inferred from the argument.
            nvme_origin: The options' borrowed path array, inferred from the argument.

        Args:
            executor: The NVMe executor whose rings read the checkpoint; it also lends its engine.
            path: The manifest of a sharded checkpoint, or a single `.safegraphs` monolith.
            nvme: The devices the parts are served from; naming none serves each where it points.

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
            ShapesViewNvmeArguments(
                created.unsafe_origin_cast[MutUntrackedOrigin](),
                executor.address,
                outcome.unsafe_origin_cast[MutUntrackedOrigin](),
                Completion.BLOCKING,
                BytesView.of(path).arguments(),
                nvme.arguments(),
            ),
            executor.library.symbols().shapes_view_nvme,
        )
        executor.library._check(outcome.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0], "usearch_shapes_view_nvme")
        return Self(library=executor.library, address=created.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0])

    def __deinit__(deinit self):
        """Releases the index and everything it owns. Calls `usearch_shapes_free`.

        The loader reference is surrendered only after the call, so the engine stays mapped until
        the release returns.
        """
        self.library.symbols().shapes_free(self.address)
        _ = self.library^

    # endregion Lifetime

    # region Receivers

    def on(ref self, ref executor: Executor) -> ShapesIndexOn[ImmOrigin(origin_of(self, executor))]:
        """The same verbs, blocking on `executor`, refused while another call holds it. Binding-only.

        Args:
            executor: The executor every verb runs on; its kind must match the handle's residency.

        Returns:
            A receiver borrowing both, which neither may outlive.
        """
        return ShapesIndexOn[ImmOrigin(origin_of(self, executor))](
            Pointer[ShapesIndex, ImmOrigin(origin_of(self, executor))](unsafe_from_address=Int(Pointer(to=self))),
            executor.address,
        )

    def deferred(ref self, ref executor: Executor) -> ShapesIndexDeferredOn[ImmOrigin(origin_of(self, executor))]:
        """The same verbs, each submitted to `executor` and answering a `Deferred` at once. Binding-only.

        Args:
            executor: The executor every verb is submitted to; its kind must match the handle's residency.

        Returns:
            A receiver borrowing both, which neither it nor any `Deferred` it answers may outlive.
        """
        return ShapesIndexDeferredOn[ImmOrigin(origin_of(self, executor))](
            Pointer[ShapesIndex, ImmOrigin(origin_of(self, executor))](unsafe_from_address=Int(Pointer(to=self))),
            executor.address,
        )

    def _route[C: Call](self, var call: C) raises Error -> C.Result:
        """Runs `call` on the calling thread."""
        return _block(self.library, call^, self.address, NULL_ADDRESS)

    def cursor(ref self) raises Error -> ShapesCursor[ImmOrigin(origin_of(self))]:
        """Opens a walk over every entry as stored, its pulls on the calling thread.
        Calls `usearch_shapes_cursor_init`.

        Returns:
            The cursor, which borrows the index and cannot outlive it.

        Raises:
            `Error` if the residency has no cursor, as VRAM does not.
        """
        return ShapesCursor[ImmOrigin(origin_of(self))](self.library, self.address, NULL_ADDRESS, 0, None)

    def cursor[
        starts_origin: ImmOrigin, //
    ](ref self, starts: Span[Key, starts_origin]) raises Error -> ShapesCursor[ImmOrigin(origin_of(self))]:
        """Opens a walk from `starts`, its pulls on the calling thread. Calls `usearch_shapes_cursor_init`.

        Parameters:
            starts_origin: The start keys' origin, inferred from the argument.

        Args:
            starts: The keys to walk from, the walk yielding their entries first.

        Returns:
            The cursor, which borrows the index and cannot outlive it.

        Raises:
            `Error` if the residency has no cursor, as VRAM does not.
        """
        return ShapesCursor[ImmOrigin(origin_of(self))](
            self.library,
            self.address,
            NULL_ADDRESS,
            len(starts),
            starts.unsafe_ptr().unsafe_origin_cast[ImmUntrackedOrigin](),
        )

    # endregion Receivers


trait ShapesVerbs:
    """The verbs every blocking receiver offers - the index on the calling thread, and
    `index.on(executor)` on an executor - each written once over the receiver's `_route`."""

    def _route[C: Call](self, var call: C) raises Error -> C.Result:
        """Runs `call` to its final status on this receiver's executor and answers its result."""
        ...

    def reserve(self, entries: Int, points: Int = 0) raises Error:
        """Reserves room for incoming entries ahead of insertion. Calls `usearch_shapes_reserve`.

        Args:
            entries: The desired total capacity in entries, including the current size.
            points: The expected total points across every entry; zero grows lazily.

        Raises:
            `Error` if the reservation cannot be satisfied.
        """
        _ = self._route(_ShapesReserveCall(entries, points))

    def clear(self) raises Error:
        """Erases every entry, keeping the reserved capacity and the frozen kind. Calls `usearch_shapes_clear`.

        Raises:
            `Error` with `READ_ONLY` if the handle serves its checkpoint in place.
        """
        _ = self._route(_ShapesClearCall())

    def describe(self) raises Error -> ShapesIndexDescription:
        """Reports the frozen metadata and the current counters in one call. Calls `usearch_shapes_describe`.

        Returns:
            The description.

        Raises:
            `Error` if the engine refuses the query.
        """
        return self._route(_ShapesDescribeCall())

    def insert[
        keys_origin: ImmOrigin, shapes_origin: ImmOrigin, stats_origin: MutOrigin, //
    ](
        self,
        keys: Span[Key, keys_origin],
        shapes: TapeView[shapes_origin],
        config: DenseInsertConfig[stats_origin] = DenseInsertConfig(),
    ) raises Error:
        """Adds a tape of shapes under their keys, all or nothing. Calls `usearch_shapes_insert`.

        Parameters:
            keys_origin: The key run's origin, inferred from the argument.
            shapes_origin: The tape's origin, inferred with it.
            stats_origin: The lent telemetry slot's origin, inferred with it.

        Args:
            keys: One key per shape; keys need not be unique, so conformers of one molecule share it.
            shapes: The shapes to insert, a tape of points three scalars wide.
            config: The breadth of the candidates and the telemetry slot.

        Raises:
            `Error` with `SHAPE_MISMATCH` when the keys and shapes differ in count, `OVERFLOW` past
            `values_per_key`, or `READ_ONLY` for a view.
        """
        _ = self._route(_insert_call(keys, shapes, config))

    def refine[
        stats_origin: MutOrigin, plan_origin: MutOrigin, phase_origin: MutOrigin, //
    ](
        self, config: ShapesRefineConfig[stats_origin, plan_origin, phase_origin] = ShapesRefineConfig()
    ) raises Error -> DenseRefineReport:
        """Runs one round over the whole graph, healing islands and re-stitching neighborhoods, or
        continues `config.plan`. Calls `usearch_shapes_refine`.

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

    def refine_plan_finish(self, var plan: ShapesRefinePlan) raises Error -> DenseRefineReport:
        """Closes and frees a refine plan whatever the call answers, reopening the index.
        Calls `usearch_shapes_refine_plan_finish`.

        Args:
            plan: The plan a refine on this index and executor opened, consumed by the call.

        Returns:
            What the whole plan did to the graph.

        Raises:
            `Error` with `INVALID_ARGUMENT` for a plan no refine opened, or one already finished.
        """
        return self._route(_ShapesRefinePlanFinishCall(plan.address))

    def health(self) raises Error -> DenseReachability:
        """Measures how much of the graph a search reaches from the root. Calls `usearch_shapes_health`.

        Returns:
            The audit; entries it misses stay unfound until a `refine` re-links them.

        Raises:
            `Error` if the engine refuses the audit.
        """
        return self._route(_ShapesHealthCall())

    def sample[
        shown_origin: ImmOrigin, coordinates_origin: ImmOrigin, focus_origin: ImmOrigin, //
    ](
        self,
        wanted: Int,
        mut arena: Arena,
        config: DenseSampleConfig[shown_origin, coordinates_origin, focus_origin] = DenseSampleConfig(),
    ) raises Error -> DenseDraws[origin_of(arena)]:
        """Draws up to `wanted` keys around `config.focus`, or across the whole index, each hanging from
        a shown or earlier draw and placed in three dimensions. Calls `usearch_shapes_sample`.

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
        queries: TapeView[queries_origin],
        wanted: Int,
        mut arena: Arena,
        config: DenseSearchConfig[stats_origin] = DenseSearchConfig(),
    ) raises Error -> Matches[origin_of(arena)]:
        """Finds the `wanted` nearest entries of every query shape, writing the matches in a lent
        arena. Calls `usearch_shapes_search`.

        Parameters:
            queries_origin: The query tape's origin, inferred from the argument.
            stats_origin: The lent telemetry slot's origin, inferred with it.

        Args:
            queries: The query shapes, a tape of points three scalars wide.
            wanted: Matches per query - the "k"; zero lands only the counts.
            arena: The arena to rewind and refill, which the result borrows; a failed call leaves it valid.
            config: The breadth of the candidates, the predicate, and the telemetry slot.

        Returns:
            Every query's matches, nearest first, read in place from the arena.

        Raises:
            `Error` if the engine refuses the search, or a predicate is passed to a VRAM handle.
        """
        return self._route(_search_call[origin_of(arena)](queries, wanted, arena._address, config))

    def count(self, key: Key) raises Error -> ShapesKeyStats:
        """How many entries one key holds, and their point counts. Calls `usearch_shapes_count`.

        Args:
            key: The key whose entries are counted.

        Returns:
            The entries under the key, their total and their largest point count.

        Raises:
            `Error` if the engine refuses the query.
        """
        return self._route(_ShapesCountCall(key))

    def get[
        dtype: DType, kind: ScalarKind = ScalarKind.of[dtype]()
    ](self, key: Key, mut arena: Arena) raises Error -> TapeEntries[dtype, origin_of(arena), 3]:
        """Exports every shape stored under one key into a lent arena, in insertion order and in the
        pose it was inserted in. Calls `usearch_shapes_get`.

        Parameters:
            dtype: The element type the points are read as.
            kind: The scalar kind to export as, converted from storage where it differs, which `dtype`
                must store whole; the kind `dtype` stores by default.

        Args:
            key: The key whose entries are exported.
            arena: The arena to rewind and refill, which the result borrows; a failed call leaves it valid.

        Returns:
            The shapes, three scalars per point, read in place from the arena; none when the key is absent.

        Raises:
            `Error` with `UNSUPPORTED` for a kind that is not a whole-byte scalar.
        """
        comptime assert kind.stores[dtype](), "the element type must store the kind whole"
        return self._route(_ShapesGetCall[dtype, kind, origin_of(arena)](key, arena._address))

    def save[path_origin: ImmOrigin, //](self, path: StringSlice[path_origin]) raises Error:
        """Writes a checkpoint, the path extension picking the layout. Calls `usearch_shapes_save`.

        Parameters:
            path_origin: The path bytes' origin, inferred from the argument.

        Args:
            path: Destination path, whose extension picks the layout.

        Raises:
            `Error` if the destination cannot be written, or `UNSUPPORTED` for an NVMe view.
        """
        _ = self._route(_ShapesSaveCall(BytesView.of(path).arguments()))

    def save_buffer(self, mut arena: Arena) raises Error -> Monolith[origin_of(arena)]:
        """Serializes the index as one monolith into a lent arena. Calls `usearch_shapes_save_buffer`.

        Args:
            arena: The arena to rewind and refill, which the result borrows; a failed call leaves it valid.

        Returns:
            The monolith, read in place from the arena.

        Raises:
            `Error` if the engine refuses the write.
        """
        return self._route(_ShapesSaveBufferCall[origin_of(arena)](arena._address))


struct ShapesIndexOn[origin: ImmOrigin](ImplicitlyCopyable, ShapesVerbs):
    """An index bound to the executor its verbs block on, as `index.on(executor)` answers it. Binding-only.

    Parameters:
        origin: The index and the executor, which this receiver borrows.
    """

    var index: Pointer[ShapesIndex, Self.origin]
    """The index every verb addresses."""
    var executor: Address
    """The `usearch_executor_t` every verb runs on."""

    def __init__(out self, index: Pointer[ShapesIndex, Self.origin], executor: Address):
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

    def cursor(self) raises Error -> ShapesCursor[Self.origin]:
        """Opens a walk over every entry as stored, its pulls on the bound executor.
        Calls `usearch_shapes_cursor_init`.

        Returns:
            The cursor, which borrows the index and the executor and cannot outlive them.

        Raises:
            `Error` if the residency has no cursor, as VRAM does not.
        """
        return ShapesCursor[Self.origin](self.index[].library, self.index[].address, self.executor, 0, None)

    def cursor[
        starts_origin: ImmOrigin, //
    ](self, starts: Span[Key, starts_origin]) raises Error -> ShapesCursor[Self.origin]:
        """Opens a walk from `starts`, its pulls on the bound executor. Calls `usearch_shapes_cursor_init`.

        Parameters:
            starts_origin: The start keys' origin, inferred from the argument.

        Args:
            starts: The keys to walk from, the walk yielding their entries first.

        Returns:
            The cursor, which borrows the index and the executor and cannot outlive them.

        Raises:
            `Error` if the residency has no cursor, as VRAM does not.
        """
        return ShapesCursor[Self.origin](
            self.index[].library,
            self.index[].address,
            self.executor,
            len(starts),
            starts.unsafe_ptr().unsafe_origin_cast[ImmUntrackedOrigin](),
        )


def _refine_call[
    stats_origin: MutOrigin, plan_origin: MutOrigin, phase_origin: MutOrigin
](config: ShapesRefineConfig[stats_origin, plan_origin, phase_origin]) -> _ShapesRefineCall:
    """Lowers a refinement's lent slots into its crossing."""
    return _ShapesRefineCall(
        RefineOptionsArguments(c_size_t(config.expansion_factor), c_size_t(config.memory_limit)),
        config.progress,
        _slot(config.stats),
        _slot(config.plan),
        _slot(config.phase),
    )


def _insert_call[
    keys_origin: ImmOrigin, shapes_origin: ImmOrigin, stats_origin: MutOrigin
](
    keys: Span[Key, keys_origin], shapes: TapeView[shapes_origin], config: DenseInsertConfig[stats_origin]
) -> _ShapesInsertCall:
    """Lowers an insertion's borrowed inputs into its crossing."""
    return _ShapesInsertCall(
        KeysView.of(keys).arguments(),
        len(keys),
        shapes.view,
        shapes.entries,
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
) -> _ShapesSampleCall[origin]:
    """Lowers a sample's borrowed inputs into its crossing."""
    return _ShapesSampleCall[origin](
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
    queries: TapeView[queries_origin],
    wanted: Int,
    arena: Address,
    config: DenseSearchConfig[stats_origin],
) -> _ShapesSearchCall[origin]:
    """Lowers a search's borrowed inputs into its crossing."""
    return _ShapesSearchCall[origin](
        queries.view,
        queries.entries,
        wanted,
        arena,
        config.expansion_factor,
        config.traversal,
        config.predicate,
        config.progress,
        _slot(config.stats),
    )


struct ShapesIndexDeferredOn[origin: ImmOrigin](ImplicitlyCopyable):
    """An index bound to the executor its verbs are submitted to, as `index.deferred(executor)` answers
    it: every verb answers a `Deferred` at once. Binding-only.

    Parameters:
        origin: The index and the executor, which this receiver borrows.
    """

    var index: Pointer[ShapesIndex, Self.origin]
    """The index every verb addresses."""
    var executor: Address
    """The `usearch_executor_t` every verb is submitted to."""

    def __init__(out self, index: Pointer[ShapesIndex, Self.origin], executor: Address):
        """Binds an index to an executor.

        Args:
            index: The index every verb addresses.
            executor: The `usearch_executor_t` every verb is submitted to.
        """
        self.index = index
        self.executor = executor

    def reserve(self, entries: Int, points: Int = 0) -> Deferred[_ShapesReserveCall, Self.origin]:
        """Submits `ShapesVerbs.reserve`. Calls `usearch_shapes_reserve`."""
        return {
            self.index[].library,
            _ShapesReserveCall(entries, points),
            self.index[].address,
            self.executor,
        }

    def clear(self) -> Deferred[_ShapesClearCall, Self.origin]:
        """Submits `ShapesVerbs.clear`. Calls `usearch_shapes_clear`."""
        return {self.index[].library, _ShapesClearCall(), self.index[].address, self.executor}

    def describe(self) -> Deferred[_ShapesDescribeCall, Self.origin]:
        """Submits `ShapesVerbs.describe`. Calls `usearch_shapes_describe`."""
        return {self.index[].library, _ShapesDescribeCall(), self.index[].address, self.executor}

    def insert[
        keys_origin: ImmOrigin, shapes_origin: ImmOrigin, stats_origin: MutOrigin, //
    ](
        self,
        keys: Span[Key, keys_origin],
        shapes: TapeView[shapes_origin],
        config: DenseInsertConfig[stats_origin] = DenseInsertConfig(),
    ) -> Deferred[
        _ShapesInsertCall,
        ImmOrigin(
            origin_of(_anchor[Self.origin](), _anchor[keys_origin](), _anchor[shapes_origin](), _anchor[stats_origin]())
        ),
    ]:
        """Submits `ShapesVerbs.insert`; the keys, tape and telemetry slot stay borrowed until final.
        Calls `usearch_shapes_insert`."""
        return {self.index[].library, _insert_call(keys, shapes, config), self.index[].address, self.executor}

    def refine[
        stats_origin: MutOrigin, plan_origin: MutOrigin, phase_origin: MutOrigin, //
    ](self, config: ShapesRefineConfig[stats_origin, plan_origin, phase_origin] = ShapesRefineConfig()) -> Deferred[
        _ShapesRefineCall,
        ImmOrigin(
            origin_of(_anchor[Self.origin](), _anchor[stats_origin](), _anchor[plan_origin](), _anchor[phase_origin]())
        ),
    ]:
        """Submits `ShapesVerbs.refine`; the telemetry, plan and phase slots stay borrowed until final.
        Calls `usearch_shapes_refine`."""
        return {self.index[].library, _refine_call(config), self.index[].address, self.executor}

    def refine_plan_finish(self, var plan: ShapesRefinePlan) -> Deferred[_ShapesRefinePlanFinishCall, Self.origin]:
        """Submits `ShapesVerbs.refine_plan_finish`. Calls `usearch_shapes_refine_plan_finish`."""
        return {
            self.index[].library,
            _ShapesRefinePlanFinishCall(plan.address),
            self.index[].address,
            self.executor,
        }

    def health(self) -> Deferred[_ShapesHealthCall, Self.origin]:
        """Submits `ShapesVerbs.health`. Calls `usearch_shapes_health`."""
        return {self.index[].library, _ShapesHealthCall(), self.index[].address, self.executor}

    def sample[
        shown_origin: ImmOrigin, coordinates_origin: ImmOrigin, focus_origin: ImmOrigin, //
    ](
        self,
        wanted: Int,
        mut arena: Arena,
        config: DenseSampleConfig[shown_origin, coordinates_origin, focus_origin] = DenseSampleConfig(),
    ) -> Deferred[
        _ShapesSampleCall[origin_of(arena)],
        ImmOrigin(
            origin_of(
                _anchor[Self.origin](),
                _anchor[shown_origin](),
                _anchor[coordinates_origin](),
                _anchor[focus_origin](),
            )
        ),
    ]:
        """Submits `ShapesVerbs.sample`; the shown keys, focus and arena stay borrowed until final.
        Calls `usearch_shapes_sample`."""
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
        queries: TapeView[queries_origin],
        wanted: Int,
        mut arena: Arena,
        config: DenseSearchConfig[stats_origin] = DenseSearchConfig(),
    ) -> Deferred[
        _ShapesSearchCall[origin_of(arena)],
        ImmOrigin(origin_of(_anchor[Self.origin](), _anchor[queries_origin](), _anchor[stats_origin]())),
    ]:
        """Submits `ShapesVerbs.search`; the queries, telemetry slot and arena stay borrowed until final.
        Calls `usearch_shapes_search`."""
        return {
            self.index[].library,
            _search_call[origin_of(arena)](queries, wanted, arena._address, config),
            self.index[].address,
            self.executor,
        }

    def count(self, key: Key) -> Deferred[_ShapesCountCall, Self.origin]:
        """Submits `ShapesVerbs.count`. Calls `usearch_shapes_count`."""
        return {self.index[].library, _ShapesCountCall(key), self.index[].address, self.executor}

    def get[
        dtype: DType, kind: ScalarKind = ScalarKind.of[dtype]()
    ](self, key: Key, mut arena: Arena) -> Deferred[_ShapesGetCall[dtype, kind, origin_of(arena)], Self.origin]:
        """Submits `ShapesVerbs.get`; the arena stays borrowed until final. Calls `usearch_shapes_get`."""
        comptime assert kind.stores[dtype](), "the element type must store the kind whole"
        return {
            self.index[].library,
            _ShapesGetCall[dtype, kind, origin_of(arena)](key, arena._address),
            self.index[].address,
            self.executor,
        }

    def save[
        path_origin: ImmOrigin, //
    ](self, path: StringSlice[path_origin]) -> Deferred[
        _ShapesSaveCall, ImmOrigin(origin_of(_anchor[Self.origin](), _anchor[path_origin]()))
    ]:
        """Submits `ShapesVerbs.save`; the path stays borrowed until final. Calls `usearch_shapes_save`."""
        return {
            self.index[].library,
            _ShapesSaveCall(BytesView.of(path).arguments()),
            self.index[].address,
            self.executor,
        }

    def save_buffer(self, mut arena: Arena) -> Deferred[_ShapesSaveBufferCall[origin_of(arena)], Self.origin]:
        """Submits `ShapesVerbs.save_buffer`; the arena stays borrowed until final.
        Calls `usearch_shapes_save_buffer`."""
        return {
            self.index[].library,
            _ShapesSaveBufferCall[origin_of(arena)](arena._address),
            self.index[].address,
            self.executor,
        }


struct ShapesCursor[origin: ImmOrigin]:
    """A walk over the index contents - `usearch_shapes_cursor_t`.

    Export, migration, and visualization feed off it. It borrows the index and the executor its
    pulls run on, so neither can be released while it lives.

    Parameters:
        origin: The index and the executor, which the cursor borrows.
    """

    var library: Library
    """The loaded engine, held so the release can still reach `usearch_shapes_cursor_free`."""
    var index: Address
    """The `usearch_shapes_index_t` the cursor was opened on, which every pull names again."""
    var executor: Address
    """The `usearch_executor_t` every pull runs on."""
    var address: Address
    """The `usearch_shapes_cursor_t` this struct owns."""

    def __init__(
        out self,
        library: Library,
        index: Address,
        executor: Address,
        starts_count: Int,
        starts: Optional[Pointer[Key, ImmUntrackedOrigin]],
    ) raises Error:
        """Opens a cursor over `index`. Calls `usearch_shapes_cursor_init`.

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
                NULL_ADDRESS,
                outcome.unsafe_origin_cast[MutUntrackedOrigin](),
                c_size_t(starts_count),
                starts,
                created.unsafe_origin_cast[MutUntrackedOrigin](),
            ),
            library.symbols().shapes_cursor_init,
        )
        library._check(outcome.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0], "usearch_shapes_cursor_init")
        self.library = library
        self.index = index
        self.executor = executor
        self.address = created.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0]

    def __deinit__(deinit self):
        """Releases the cursor. Calls `usearch_shapes_cursor_free`."""
        self.library.symbols().shapes_cursor_free(self.address)
        _ = self.library^

    def next_keys(self, wanted: Int, mut arena: Arena) raises Error -> DenseBatch[origin_of(arena)]:
        """Copies up to `wanted` next keys into a lent arena, skipping their links.
        Calls `usearch_shapes_cursor_next`.

        Args:
            wanted: The upper bound on keys this pull writes.
            arena: The arena to rewind and refill, which the batch borrows; a failed call leaves it valid.

        Returns:
            The keys, read in place from the arena; an empty batch once the walk is exhausted.

        Raises:
            `Error` if the engine refuses the copy.
        """
        return self._next(_ShapesCursorNextCall[origin_of(arena)](self.address, False, wanted, arena._address))

    def next_links(self, wanted: Int, mut arena: Arena) raises Error -> DenseBatch[origin_of(arena)]:
        """Copies up to `wanted` next keys into a lent arena, each with the keys it links to.
        Calls `usearch_shapes_cursor_next`.

        Args:
            wanted: The upper bound on entries this pull writes.
            arena: The arena to rewind and refill, which the batch borrows; a failed call leaves it valid.

        Returns:
            The keys and their links, read in place from the arena; an empty batch once the walk is exhausted.

        Raises:
            `Error` if the engine refuses the copy.
        """
        return self._next(_ShapesCursorNextCall[origin_of(arena)](self.address, True, wanted, arena._address))

    def _next[C: Call](self, var call: C) raises Error -> C.Result:
        """One blocking pull on the cursor's executor."""
        return _block(self.library, call^, self.index, self.executor)


# endregion Index

# region Reshard


def shapes_reshard[
    sources_origin: ImmOrigin,
    targets_origin: ImmOrigin,
    stats_origin: MutOrigin,
    plan_origin: MutOrigin,
    phase_origin: MutOrigin,
    //,
](
    executor: Executor,
    sources: Span[ShapesIndex, sources_origin],
    targets: Span[ShapesIndex, targets_origin],
    config: ShapesReshardConfig[stats_origin, plan_origin, phase_origin] = ShapesReshardConfig(),
) raises Error -> DenseRefineReport:
    """Moves the entries of `sources` into `targets`, keeping their links, then runs a round on every
    target that took entries. Calls `usearch_shapes_reshard`.

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
    return _block(executor.library, _ShapesReshardCall(sources, targets, config), NULL_ADDRESS, executor.address)


def shapes_reshard_deferred[
    sources_origin: ImmOrigin,
    targets_origin: ImmOrigin,
    stats_origin: MutOrigin,
    plan_origin: MutOrigin,
    phase_origin: MutOrigin,
    //,
](
    ref executor: Executor,
    sources: Span[ShapesIndex, sources_origin],
    targets: Span[ShapesIndex, targets_origin],
    config: ShapesReshardConfig[stats_origin, plan_origin, phase_origin] = ShapesReshardConfig(),
) -> Deferred[
    _ShapesReshardCall,
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
    """Submits `shapes_reshard` to `executor`, answering a `Deferred` at once; the indexes and slots
    must stay allocated and unchanged until final. Calls `usearch_shapes_reshard`."""
    return {executor.library, _ShapesReshardCall(sources, targets, config), NULL_ADDRESS, executor.address}


def shapes_reshard_plan_finish(executor: Executor, var plan: ShapesReshardPlan) raises Error -> DenseRefineReport:
    """Closes and frees a reshard plan whatever the call answers, reopening every index it names.
    Calls `usearch_shapes_reshard_plan_finish`.

    Args:
        executor: The executor the plan was opened with; it also lends its engine.
        plan: The plan a reshard opened, consumed by the call.

    Returns:
        What the whole plan did to the graphs; zeros while entries were still to move.

    Raises:
        `Error` with `INVALID_ARGUMENT` for a plan no reshard opened, or one opened with another executor.
    """
    return _block(executor.library, _ShapesReshardPlanFinishCall(plan.address), NULL_ADDRESS, executor.address)


def shapes_reshard_plan_finish_deferred(
    ref executor: Executor, var plan: ShapesReshardPlan
) -> Deferred[_ShapesReshardPlanFinishCall, ImmOrigin(origin_of(executor))]:
    """Submits `shapes_reshard_plan_finish` to `executor`, answering a `Deferred` at once.
    Calls `usearch_shapes_reshard_plan_finish`."""
    return {executor.library, _ShapesReshardPlanFinishCall(plan.address), NULL_ADDRESS, executor.address}


# endregion Reshard

# region Metadata


def _read_metadata(
    executor: Executor, verb: ShapesMetadataRead, source: BytesView[ImmUntrackedOrigin], detail: StaticString
) raises Error -> ShapesMetadata:
    """The path and buffer forms share one argument layout, differing only in the verb."""
    var described = stack_allocation[1, ShapesMetadata]()
    described[unsafe_offset=0] = ShapesMetadata()
    var outcome = stack_allocation[1, Outcome]()
    outcome[unsafe_offset=0] = Outcome()
    _cross(
        ShapesMetadataReadArguments(
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


def shapes_metadata_read[
    path_origin: ImmOrigin, //
](executor: Executor, path: StringSlice[path_origin]) raises Error -> ShapesMetadata:
    """Reads a checkpoint's kind, storage and point count without loading it.
    Calls `usearch_shapes_metadata_read`.

    Parameters:
        path_origin: The path bytes' origin, inferred from the argument.

    Args:
        executor: The executor the read runs on; it also lends its engine.
        path: The manifest of a sharded checkpoint, or a single monolith.

    Returns:
        The kind, storage, point count and connectivity the checkpoint froze.

    Raises:
        `Error` if the checkpoint cannot be opened or its header cannot be decoded.
    """
    return _read_metadata(
        executor,
        executor.library.symbols().shapes_metadata_read,
        BytesView.of(path).arguments(),
        "usearch_shapes_metadata_read",
    )


def shapes_metadata_read_buffer[
    buffer_origin: ImmOrigin, //
](executor: Executor, buffer: Span[UInt8, buffer_origin]) raises Error -> ShapesMetadata:
    """Reads a serialized monolith's kind, storage and point count without loading it.
    Calls `usearch_shapes_metadata_read_buffer`.

    Parameters:
        buffer_origin: The source bytes' origin, inferred from the argument.

    Args:
        executor: The executor the read runs on; it also lends its engine.
        buffer: Source holding one serialized monolith.

    Returns:
        The kind, storage, point count and connectivity the monolith froze.

    Raises:
        `Error` if the bytes cannot be decoded.
    """
    return _read_metadata(
        executor,
        executor.library.symbols().shapes_metadata_read_buffer,
        BytesView.of(buffer).arguments(),
        "usearch_shapes_metadata_read_buffer",
    )


# endregion Metadata
