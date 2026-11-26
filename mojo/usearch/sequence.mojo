"""Similarity search over variable-length sequences: the index, its receivers, and its cursor.

Text, DNA, RNA and proteins arrive as a `TapeView` of bytes, every sequence end to end between
Apache Arrow offsets, and score under Levenshtein over bytes or UTF-8 code points, Needleman-Wunsch
with a substitution matrix and a gap cost, or the overlap of fixed-width windows.

The verbs mirror the dense ones, and so do the receivers: verbs on the index run on the calling
thread, `index.on(executor)` blocks on that executor, and `index.deferred(executor)` submits there
and answers a `Deferred` at once. The reports, the telemetry, the phases and the insert and search
knobs are the dense module's own, since the C interface shares them across every modality.

File: mojo/usearch/sequence.mojo
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
    ArenaRead,
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
struct SequenceMetric(Equatable, ImplicitlyCopyable, TrivialRegisterPassable):
    """The sequence scoring kinds - `usearch_sequence_metric_t`. `Library.sequence_metric_name` spells
    each one as the engine does."""

    var code: c_int
    """The raw `usearch_sequence_metric_t` value the ABI exchanges."""
    comptime UNKNOWN = Self(0)
    """No kind named - an unset field, refused by every constructor."""
    comptime LEVENSHTEIN_BYTES = Self(1)
    """Single-byte insertions, deletions and substitutions, each costing one."""
    comptime LEVENSHTEIN_UTF8 = Self(2)
    """The same edits over decoded code points, so an accented character costs one edit."""
    comptime NEEDLEMAN_WUNSCH = Self(3)
    """Global alignment under a substitution matrix and a linear gap cost."""
    comptime OVERLAP = Self(4)
    """The share of the larger set of fixed-width windows the other side misses."""


@fieldwise_init
struct SequenceWeightsArguments(ImplicitlyCopyable, TrivialRegisterPassable):
    """`usearch_sequence_weights_t`."""

    var substitution_matrix: Address
    var gap: Int8


@fieldwise_init
struct SequenceWeights[origin: ImmOrigin = ImmUntrackedOrigin](ImplicitlyCopyable):
    """The substitution matrix and gap cost Needleman-Wunsch scores with - `usearch_sequence_weights_t`.

    Parameters:
        origin: The viewed matrix, named so it outlives every call these weights cross.
    """

    var substitution_matrix: Span[Int8, Self.origin]
    """Row-major 256x256 signed similarities over the byte alphabet."""
    var gap: Int8
    """Cost of one unaligned symbol."""

    def arguments(self) -> SequenceWeightsArguments:
        """The weights as the engine reads them, their view erased.

        Returns:
            The matrix's first score and the gap cost.
        """
        return SequenceWeightsArguments(Int(self.substitution_matrix.unsafe_ptr()), self.gap)


@fieldwise_init
struct SequenceOptions[weights_origin: ImmOrigin = ImmUntrackedOrigin, widths_origin: ImmOrigin = ImmUntrackedOrigin](
    ImplicitlyCopyable
):
    """The kind an empty index freezes - what `usearch_sequence_init_ram_t` and
    `usearch_sequence_init_vram_t` share, without the placement each names its own way.

    Parameters:
        weights_origin: The viewed substitution matrix, inferred where weights are named.
        widths_origin: The viewed window widths, inferred where any are named.
    """

    var metric: SequenceMetric
    """The scoring kind."""
    var weights: Optional[SequenceWeights[Self.weights_origin]]
    """The weights Needleman-Wunsch needs; absent for every other kind, which refuses any."""
    var window_widths: Span[c_size_t, Self.widths_origin]
    """Window widths in bytes for the overlap kind, which needs one at least; every other kind refuses any."""
    var connectivity: c_size_t
    """The connections-per-node bound; zero derives."""
    var values_per_key: c_size_t
    """Upper bound on entries sharing one key; zero is unbounded."""

    def __init__(
        out self,
        metric: SequenceMetric = SequenceMetric.LEVENSHTEIN_BYTES,
        weights: Optional[SequenceWeights[Self.weights_origin]] = Optional[SequenceWeights[Self.weights_origin]](),
        window_widths: Span[c_size_t, Self.widths_origin] = Span[c_size_t, Self.widths_origin](),
        connectivity: Int = 0,
        values_per_key: Int = 0,
    ):
        """Describes an index by its kind, byte Levenshtein unless told otherwise.

        Args:
            metric: The scoring kind the graph is frozen under.
            weights: The weights Needleman-Wunsch needs; absent for every other kind.
            window_widths: Window widths in bytes for the overlap kind.
            connectivity: The connections-per-node bound; zero derives.
            values_per_key: Upper bound on entries sharing one key; zero is unbounded.
        """
        self.metric = metric
        self.weights = weights
        self.window_widths = window_widths
        self.connectivity = c_size_t(connectivity)
        self.values_per_key = c_size_t(values_per_key)


def _pass_weights[
    origin: ImmOrigin
](weights: Optional[SequenceWeights[origin]], slot: Pointer[SequenceWeightsArguments, MutAnyOrigin]) -> Optional[
    Pointer[SequenceWeightsArguments, ImmUntrackedOrigin]
]:
    """Writes optional weights into `slot`, which outlives the call, and answers the pointer the engine reads."""
    if case .Some(named) = weights:
        slot[] = named.arguments()
        return Pointer[SequenceWeightsArguments, ImmUntrackedOrigin](unsafe_from_address=Int(slot))
    return None


@fieldwise_init
struct SequenceMetadata(Defaultable, ImplicitlyCopyable):
    """The kind a checkpoint froze - `usearch_sequence_metadata_t`, a pure output."""

    var metric: SequenceMetric
    """The scoring kind the checkpoint scores with."""
    var connectivity: c_size_t
    """Connections per node the graph was built with."""

    def __init__(out self):
        """Names no kind, so an unwritten slot reads as an unanswered probe."""
        self.metric = SequenceMetric.UNKNOWN
        self.connectivity = c_size_t(0)


@fieldwise_init
struct SequenceIndexDescription(Defaultable, ImplicitlyCopyable):
    """`usearch_sequence_index_description_t`: a live index's frozen metadata and current counters."""

    var metadata: SequenceMetadata
    """The frozen kind and connectivity, as a checkpoint reports them."""
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
        self.metadata = SequenceMetadata()
        self.size = c_size_t(0)
        self.capacity = c_size_t(0)
        self.memory_usage = c_size_t(0)
        self.cpu_class = CPUClass.UNKNOWN
        self.gpu_class = GPUClass.UNKNOWN
        self.shards_count = c_size_t(0)
        self.load_path = LoadPath.NONE


@fieldwise_init
struct SequenceKeyStats(Defaultable, ImplicitlyCopyable):
    """What `count` reports for one key - `usearch_sequence_key_stats_t`."""

    var entries: c_size_t
    """Entries stored under the key; zero means the index holds none."""
    var total_bytes: c_size_t
    """Bytes those entries hold together, which sizes a `get`."""
    var largest_bytes: c_size_t
    """Bytes of the longest of them."""

    def __init__(out self):
        """Zeroes every count, which is what a failed call leaves."""
        self.entries = c_size_t(0)
        self.total_bytes = c_size_t(0)
        self.largest_bytes = c_size_t(0)


struct SequenceRefinePlan(Movable):
    """One refine in flight - `usearch_sequence_refine_plan_t`. Binding-only.

    Passed through `SequenceRefineConfig.plan`, it opens on the first refine, stays live past a cancel
    or the end of a round, continues on every later refine, and closes with `refine_plan_finish`.
    While it lives, the index admits no insertion and no save.
    """

    var address: Address
    """The opaque handle, `NULL_ADDRESS` until a refine opens it."""

    def __init__(out self):
        """A plan no refine opened yet."""
        self.address = NULL_ADDRESS


struct SequenceReshardPlan(Movable):
    """One reshard in flight - `usearch_sequence_reshard_plan_t`, with its own lists of sources and
    targets. Binding-only.

    Passed through `SequenceReshardConfig.plan`, it opens on the first reshard, stays live past a
    cancel, moves what is still pending and continues the rounds on every later one, and closes with
    `sequence_reshard_plan_finish`. While it lives, every index it names admits no insertion and no
    save.
    """

    var address: Address
    """The opaque handle, `NULL_ADDRESS` until a reshard opens it."""

    def __init__(out self):
        """A plan no reshard opened yet."""
        self.address = NULL_ADDRESS


# endregion Configuration

# region Configs


struct SequenceRefineConfig[
    stats_origin: MutOrigin = MutUntrackedOrigin,
    plan_origin: MutOrigin = MutUntrackedOrigin,
    phase_origin: MutOrigin = MutUntrackedOrigin,
](ImplicitlyCopyable):
    """The knobs of `usearch_sequence_refine_t`; zeroed or absent ones derive what the C interface derives.

    Parameters:
        stats_origin: The passed telemetry slot's origin, inferred where one is passed.
        plan_origin: The passed plan's origin, inferred where one is passed.
        phase_origin: The passed phase slot's origin, inferred where one is passed.
    """

    var expansion_factor: Int
    """The breadth of any walk beside the round; zero derives."""
    var memory_limit: Int
    """The bytes a loaded NVMe index holds for the one part it rewrites; zero derives."""
    var progress: ProgressArguments
    """Hears every phase's units at its barriers; a cancel stops the round at that barrier."""
    var stats: Optional[Pointer[DenseTraversalStats, Self.stats_origin]]
    """Receives this call's traversal telemetry, fully overwritten; absent skips reporting."""
    var plan: Optional[Pointer[SequenceRefinePlan, Self.plan_origin]]
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
        plan: Optional[Pointer[SequenceRefinePlan, Self.plan_origin]] = Optional[
            Pointer[SequenceRefinePlan, Self.plan_origin]
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


struct SequenceReshardConfig[
    stats_origin: MutOrigin = MutUntrackedOrigin,
    plan_origin: MutOrigin = MutUntrackedOrigin,
    phase_origin: MutOrigin = MutUntrackedOrigin,
](ImplicitlyCopyable):
    """The knobs of `usearch_sequence_reshard_t`; zeroed or absent ones derive what the C interface derives.

    Parameters:
        stats_origin: The passed telemetry slot's origin, inferred where one is passed.
        plan_origin: The passed plan's origin, inferred where one is passed.
        phase_origin: The passed phase slot's origin, inferred where one is passed.
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
    var plan: Optional[Pointer[SequenceReshardPlan, Self.plan_origin]]
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
        plan: Optional[Pointer[SequenceReshardPlan, Self.plan_origin]] = Optional[
            Pointer[SequenceReshardPlan, Self.plan_origin]
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
struct SequenceInitRamArguments(ImplicitlyCopyable):
    """`usearch_sequence_init_ram_t`."""

    var index: Pointer[Address, MutUntrackedOrigin]
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var metric: SequenceMetric
    var weights: Optional[Pointer[SequenceWeightsArguments, ImmUntrackedOrigin]]
    var window_widths: Optional[Pointer[c_size_t, ImmUntrackedOrigin]]
    var window_widths_count: c_size_t
    var connectivity: c_size_t
    var values_per_key: c_size_t
    var ram: RamOptions[ImmUntrackedOrigin]


@fieldwise_init
struct SequenceInitVramArguments(ImplicitlyCopyable):
    """`usearch_sequence_init_vram_t`."""

    var index: Pointer[Address, MutUntrackedOrigin]
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var metric: SequenceMetric
    var weights: Optional[Pointer[SequenceWeightsArguments, ImmUntrackedOrigin]]
    var window_widths: Optional[Pointer[c_size_t, ImmUntrackedOrigin]]
    var window_widths_count: c_size_t
    var connectivity: c_size_t
    var values_per_key: c_size_t
    var vram: VramOptions[ImmUntrackedOrigin]


@fieldwise_init
struct SequenceLoadRamArguments(ImplicitlyCopyable):
    """`usearch_sequence_load_ram_t`, and `usearch_sequence_load_ram_buffer_t` with `source` a buffer."""

    var index: Pointer[Address, MutUntrackedOrigin]
    var executor: Address
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var completion: Completion
    var source: BytesView[ImmUntrackedOrigin]
    var ram: RamOptions[ImmUntrackedOrigin]
    var values_per_key: c_size_t
    var weights: Optional[Pointer[SequenceWeightsArguments, ImmUntrackedOrigin]]
    var progress: ProgressArguments


@fieldwise_init
struct SequenceLoadVramArguments(ImplicitlyCopyable):
    """`usearch_sequence_load_vram_t`, and `usearch_sequence_load_vram_buffer_t` with `source` a buffer."""

    var index: Pointer[Address, MutUntrackedOrigin]
    var executor: Address
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var completion: Completion
    var source: BytesView[ImmUntrackedOrigin]
    var vram: VramOptions[ImmUntrackedOrigin]
    var values_per_key: c_size_t
    var weights: Optional[Pointer[SequenceWeightsArguments, ImmUntrackedOrigin]]
    var progress: ProgressArguments


@fieldwise_init
struct SequenceViewRamArguments(ImplicitlyCopyable):
    """`usearch_sequence_view_ram_t`."""

    var index: Pointer[Address, MutUntrackedOrigin]
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var path: BytesView[ImmUntrackedOrigin]
    var weights: Optional[Pointer[SequenceWeightsArguments, ImmUntrackedOrigin]]
    var ram: RamOptions[ImmUntrackedOrigin]


@fieldwise_init
struct SequenceViewNvmeArguments(ImplicitlyCopyable):
    """`usearch_sequence_view_nvme_t`."""

    var index: Pointer[Address, MutUntrackedOrigin]
    var executor: Address
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var completion: Completion
    var path: BytesView[ImmUntrackedOrigin]
    var nvme: NvmeOptions[ImmUntrackedOrigin]
    var weights: Optional[Pointer[SequenceWeightsArguments, ImmUntrackedOrigin]]


@fieldwise_init
struct SequenceReserveArguments(ImplicitlyCopyable):
    """`usearch_sequence_reserve_t`."""

    var index: Address
    var executor: Address
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var completion: Completion
    var entries: c_size_t
    var payload_bytes: c_size_t


@fieldwise_init
struct SequenceDescribeArguments(ImplicitlyCopyable):
    """`usearch_sequence_describe_t`."""

    var index: Address
    var executor: Address
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var completion: Completion
    var description: Pointer[SequenceIndexDescription, MutUntrackedOrigin]


@fieldwise_init
struct SequenceInsertArguments(ImplicitlyCopyable):
    """`usearch_sequence_insert_t`."""

    var index: Address
    var executor: Address
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var completion: Completion
    var count: c_size_t
    var keys: KeysView[ImmUntrackedOrigin]
    var sequences: TapeViewArguments
    var expansion_factor: c_size_t
    var progress: ProgressArguments
    var stats: Optional[Pointer[DenseTraversalStats, MutUntrackedOrigin]]


@fieldwise_init
struct SequenceRefineArguments(ImplicitlyCopyable):
    """`usearch_sequence_refine_t`."""

    var index: Address
    var executor: Address
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var completion: Completion
    var report: Pointer[DenseRefineReport, MutUntrackedOrigin]
    var refine: RefineOptionsArguments
    var progress: ProgressArguments
    var stats: Optional[Pointer[DenseTraversalStats, MutUntrackedOrigin]]
    var plan: Optional[Pointer[SequenceRefinePlan, MutUntrackedOrigin]]
    var phase: Optional[Pointer[PhaseProgress, MutUntrackedOrigin]]


@fieldwise_init
struct SequenceSearchArguments(ImplicitlyCopyable):
    """`usearch_sequence_search_t`."""

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
    var progress: ProgressArguments
    var stats: Optional[Pointer[DenseTraversalStats, MutUntrackedOrigin]]


@fieldwise_init
struct SequenceCountArguments(ImplicitlyCopyable):
    """`usearch_sequence_count_t`."""

    var index: Address
    var executor: Address
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var completion: Completion
    var key: KeyBytes
    var key_stats: Pointer[SequenceKeyStats, MutUntrackedOrigin]


@fieldwise_init
struct SequenceGetArguments(ImplicitlyCopyable):
    """`usearch_sequence_get_t`."""

    var index: Address
    var executor: Address
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var completion: Completion
    var key: KeyBytes
    var sequences: Pointer[TapeViewArguments, MutUntrackedOrigin]
    var written: Pointer[c_size_t, MutUntrackedOrigin]
    var arena: Address


@fieldwise_init
struct SequenceReshardArguments(ImplicitlyCopyable):
    """`usearch_sequence_reshard_t`."""

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
    var plan: Optional[Pointer[SequenceReshardPlan, MutUntrackedOrigin]]
    var phase: Optional[Pointer[PhaseProgress, MutUntrackedOrigin]]


@fieldwise_init
struct SequenceMetadataReadArguments(ImplicitlyCopyable):
    """`usearch_sequence_metadata_read_t`, and `usearch_sequence_metadata_read_buffer_t` with `source` a
    buffer."""

    var executor: Address
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var completion: Completion
    var source: BytesView[ImmUntrackedOrigin]
    var metadata: Pointer[SequenceMetadata, MutUntrackedOrigin]


comptime SequenceInitRam = Verb[SequenceInitRamArguments]
comptime SequenceInitVram = Verb[SequenceInitVramArguments]
comptime SequenceLoadRam = Verb[SequenceLoadRamArguments]
comptime SequenceLoadVram = Verb[SequenceLoadVramArguments]
comptime SequenceViewRam = Verb[SequenceViewRamArguments]
comptime SequenceViewNvme = Verb[SequenceViewNvmeArguments]
comptime SequenceReserve = Verb[SequenceReserveArguments]
comptime SequenceClear = Verb[DenseClearArguments]
comptime SequenceDescribe = Verb[SequenceDescribeArguments]
comptime SequenceInsert = Verb[SequenceInsertArguments]
comptime SequenceRefine = Verb[SequenceRefineArguments]
comptime SequenceRefinePlanFinish = Verb[DenseRefinePlanFinishArguments]
comptime SequenceHealth = Verb[DenseHealthArguments]
comptime SequenceReshard = Verb[SequenceReshardArguments]
comptime SequenceReshardPlanFinish = Verb[DenseReshardPlanFinishArguments]
comptime SequenceSample = Verb[DenseSampleArguments]
comptime SequenceSearch = Verb[SequenceSearchArguments]
comptime SequenceCount = Verb[SequenceCountArguments]
comptime SequenceGet = Verb[SequenceGetArguments]
comptime SequenceCursorInit = Verb[DenseCursorInitArguments]
comptime SequenceCursorNext = Verb[DenseCursorNextArguments]
comptime SequenceSave = Verb[DenseSaveArguments]
comptime SequenceSaveBuffer = Verb[DenseSaveBufferArguments]
comptime SequenceMetadataRead = Verb[SequenceMetadataReadArguments]


# endregion Arguments

# region Calls


@fieldwise_init
struct _SequenceReserveCall(Call):
    """The crossing of `usearch_sequence_reserve`."""

    comptime Arguments = SequenceReserveArguments
    comptime Outputs = Unwritten
    comptime Result = NoneType
    comptime symbol: StaticString = "usearch_sequence_reserve"
    var entries: Int
    var payload_bytes: Int

    @staticmethod
    def verb(library: Library) -> Verb[Self.Arguments]:
        return library.symbols().sequence_reserve

    def arguments(self, head: CallHead, outputs: Pointer[Self.Outputs, MutUntrackedOrigin]) -> Self.Arguments:
        return {
            head.index,
            head.executor,
            head.outcome,
            head.completion,
            c_size_t(self.entries),
            c_size_t(self.payload_bytes),
        }

    def conclude(deinit self, outputs: Unwritten, viewing: ArenaRead) -> NoneType:
        return None


@fieldwise_init
struct _SequenceClearCall(Call):
    """The crossing of `usearch_sequence_clear`."""

    comptime Arguments = DenseClearArguments
    comptime Outputs = Unwritten
    comptime Result = NoneType
    comptime symbol: StaticString = "usearch_sequence_clear"

    @staticmethod
    def verb(library: Library) -> Verb[Self.Arguments]:
        return library.symbols().sequence_clear

    def arguments(self, head: CallHead, outputs: Pointer[Self.Outputs, MutUntrackedOrigin]) -> Self.Arguments:
        return {head.index, head.executor, head.outcome, head.completion}

    def conclude(deinit self, outputs: Unwritten, viewing: ArenaRead) -> NoneType:
        return None


@fieldwise_init
struct _SequenceDescribeCall(Call):
    """The crossing of `usearch_sequence_describe`."""

    comptime Arguments = SequenceDescribeArguments
    comptime Outputs = SequenceIndexDescription
    comptime Result = SequenceIndexDescription
    comptime symbol: StaticString = "usearch_sequence_describe"

    @staticmethod
    def verb(library: Library) -> Verb[Self.Arguments]:
        return library.symbols().sequence_describe

    def arguments(self, head: CallHead, outputs: Pointer[Self.Outputs, MutUntrackedOrigin]) -> Self.Arguments:
        return {head.index, head.executor, head.outcome, head.completion, outputs}

    def conclude(deinit self, outputs: SequenceIndexDescription, viewing: ArenaRead) -> SequenceIndexDescription:
        return outputs


@fieldwise_init
struct _SequenceCountCall(Call):
    """The crossing of `usearch_sequence_count`."""

    comptime Arguments = SequenceCountArguments
    comptime Outputs = SequenceKeyStats
    comptime Result = SequenceKeyStats
    comptime symbol: StaticString = "usearch_sequence_count"
    var key: Key

    @staticmethod
    def verb(library: Library) -> Verb[Self.Arguments]:
        return library.symbols().sequence_count

    def arguments(self, head: CallHead, outputs: Pointer[Self.Outputs, MutUntrackedOrigin]) -> Self.Arguments:
        return {head.index, head.executor, head.outcome, head.completion, KeyBytes.of(self.key), outputs}

    def conclude(deinit self, outputs: SequenceKeyStats, viewing: ArenaRead) -> SequenceKeyStats:
        return outputs


@fieldwise_init
struct SequenceGetOutputs(Defaultable, ImplicitlyCopyable):
    """What `usearch_sequence_get` writes through its pointers."""

    var sequences: TapeViewArguments
    var written: c_size_t

    def __init__(out self):
        """Nothing exported."""
        self.sequences = _no_tape()
        self.written = c_size_t(0)


@fieldwise_init
struct _SequenceGetCall[origin: MutOrigin](Call):
    """The crossing of `usearch_sequence_get`, which uses the arena its tape lands in."""

    comptime Arguments = SequenceGetArguments
    comptime Outputs = SequenceGetOutputs
    comptime Result = TapeEntries[DType.uint8, Self.origin]
    comptime symbol: StaticString = "usearch_sequence_get"
    var key: Key
    var arena: Address

    @staticmethod
    def verb(library: Library) -> Verb[Self.Arguments]:
        return library.symbols().sequence_get

    def arguments(self, head: CallHead, outputs: Pointer[Self.Outputs, MutUntrackedOrigin]) -> Self.Arguments:
        return {
            head.index,
            head.executor,
            head.outcome,
            head.completion,
            KeyBytes.of(self.key),
            Pointer(to=outputs[].sequences),
            Pointer(to=outputs[].written),
            self.arena,
        }

    def arena_handle(self) -> Address:
        return self.arena

    def conclude(deinit self, outputs: SequenceGetOutputs, viewing: ArenaRead) -> Self.Result:
        return {outputs.sequences, Int(outputs.written), viewing}


@fieldwise_init
struct _SequenceInsertCall(Call):
    """The crossing of `usearch_sequence_insert`."""

    comptime Arguments = SequenceInsertArguments
    comptime Outputs = Unwritten
    comptime Result = NoneType
    comptime symbol: StaticString = "usearch_sequence_insert"
    var keys: KeysView[ImmUntrackedOrigin]
    var keys_count: Int
    var sequences: TapeViewArguments
    var count: Int
    var expansion_factor: Int
    var progress: ProgressArguments
    var stats: Optional[Pointer[DenseTraversalStats, MutUntrackedOrigin]]

    @staticmethod
    def verb(library: Library) -> Verb[Self.Arguments]:
        return library.symbols().sequence_insert

    def arguments(self, head: CallHead, outputs: Pointer[Self.Outputs, MutUntrackedOrigin]) -> Self.Arguments:
        return {
            head.index,
            head.executor,
            head.outcome,
            head.completion,
            c_size_t(self.count),
            self.keys,
            self.sequences,
            c_size_t(self.expansion_factor),
            self.progress,
            self.stats,
        }

    def lengths_agree(self) -> Bool:
        return self.keys_count == self.count

    def conclude(deinit self, outputs: Unwritten, viewing: ArenaRead) -> NoneType:
        return None


@fieldwise_init
struct _SequenceRefineCall(Call):
    """The crossing of `usearch_sequence_refine`."""

    comptime Arguments = SequenceRefineArguments
    comptime Outputs = DenseRefineReport
    comptime Result = DenseRefineReport
    comptime symbol: StaticString = "usearch_sequence_refine"
    var refine: RefineOptionsArguments
    var progress: ProgressArguments
    var stats: Optional[Pointer[DenseTraversalStats, MutUntrackedOrigin]]
    var plan: Optional[Pointer[SequenceRefinePlan, MutUntrackedOrigin]]
    var phase: Optional[Pointer[PhaseProgress, MutUntrackedOrigin]]

    @staticmethod
    def verb(library: Library) -> Verb[Self.Arguments]:
        return library.symbols().sequence_refine

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

    def conclude(deinit self, outputs: DenseRefineReport, viewing: ArenaRead) -> DenseRefineReport:
        return outputs


@fieldwise_init
struct _SequenceRefinePlanFinishCall(Call):
    """The crossing of `usearch_sequence_refine_plan_finish`, which frees the plan whatever it answers."""

    comptime Arguments = DenseRefinePlanFinishArguments
    comptime Outputs = DenseRefineReport
    comptime Result = DenseRefineReport
    comptime symbol: StaticString = "usearch_sequence_refine_plan_finish"
    var plan: Address

    @staticmethod
    def verb(library: Library) -> Verb[Self.Arguments]:
        return library.symbols().sequence_refine_plan_finish

    def arguments(self, head: CallHead, outputs: Pointer[Self.Outputs, MutUntrackedOrigin]) -> Self.Arguments:
        return {head.index, head.executor, head.outcome, head.completion, self.plan, outputs}

    def conclude(deinit self, outputs: DenseRefineReport, viewing: ArenaRead) -> DenseRefineReport:
        return outputs


@fieldwise_init
struct _SequenceHealthCall(Call):
    """The crossing of `usearch_sequence_health`."""

    comptime Arguments = DenseHealthArguments
    comptime Outputs = DenseReachability
    comptime Result = DenseReachability
    comptime symbol: StaticString = "usearch_sequence_health"

    @staticmethod
    def verb(library: Library) -> Verb[Self.Arguments]:
        return library.symbols().sequence_health

    def arguments(self, head: CallHead, outputs: Pointer[Self.Outputs, MutUntrackedOrigin]) -> Self.Arguments:
        return {head.index, head.executor, head.outcome, head.completion, outputs}

    def conclude(deinit self, outputs: DenseReachability, viewing: ArenaRead) -> DenseReachability:
        return outputs


struct _SequenceReshardCall(Call):
    """The crossing of `usearch_sequence_reshard`, which keeps the handle lists on the heap until the
    call is final."""

    comptime Arguments = SequenceReshardArguments
    comptime Outputs = DenseRefineReport
    comptime Result = DenseRefineReport
    comptime symbol: StaticString = "usearch_sequence_reshard"
    var sources: List[Address]
    var targets: List[Address]
    var refine: RefineOptionsArguments
    var seed: UInt64
    var progress: ProgressArguments
    var stats: Optional[Pointer[DenseTraversalStats, MutUntrackedOrigin]]
    var plan: Optional[Pointer[SequenceReshardPlan, MutUntrackedOrigin]]
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
        sources: Span[SequenceIndex, sources_origin],
        targets: Span[SequenceIndex, targets_origin],
        config: SequenceReshardConfig[stats_origin, plan_origin, phase_origin],
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
        return library.symbols().sequence_reshard

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

    def conclude(deinit self, outputs: DenseRefineReport, viewing: ArenaRead) -> DenseRefineReport:
        return outputs


@fieldwise_init
struct _SequenceReshardPlanFinishCall(Call):
    """The crossing of `usearch_sequence_reshard_plan_finish`, which frees the plan whatever it answers."""

    comptime Arguments = DenseReshardPlanFinishArguments
    comptime Outputs = DenseRefineReport
    comptime Result = DenseRefineReport
    comptime symbol: StaticString = "usearch_sequence_reshard_plan_finish"
    var plan: Address

    @staticmethod
    def verb(library: Library) -> Verb[Self.Arguments]:
        return library.symbols().sequence_reshard_plan_finish

    def arguments(self, head: CallHead, outputs: Pointer[Self.Outputs, MutUntrackedOrigin]) -> Self.Arguments:
        return {head.executor, head.outcome, head.completion, self.plan, outputs}

    def conclude(deinit self, outputs: DenseRefineReport, viewing: ArenaRead) -> DenseRefineReport:
        return outputs


@fieldwise_init
struct _SequenceSampleCall[origin: MutOrigin](Call):
    """The crossing of `usearch_sequence_sample`, which uses the arena its draws are written in."""

    comptime Arguments = DenseSampleArguments
    comptime Outputs = DenseSampleOutputs
    comptime Result = DenseDraws[Self.origin]
    comptime symbol: StaticString = "usearch_sequence_sample"
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
        return library.symbols().sequence_sample

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

    def arena_handle(self) -> Address:
        return self.arena

    def conclude(deinit self, outputs: DenseSampleOutputs, viewing: ArenaRead) -> Self.Result:
        return {outputs, viewing}


@fieldwise_init
struct _SequenceSearchCall[origin: MutOrigin](Call):
    """The crossing of `usearch_sequence_search`, which uses the arena its matches land in."""

    comptime Arguments = SequenceSearchArguments
    comptime Outputs = MatchesViewArguments
    comptime Result = Matches[Self.origin]
    comptime symbol: StaticString = "usearch_sequence_search"
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
        return library.symbols().sequence_search

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
            self.progress,
            self.stats,
        }

    def arena_handle(self) -> Address:
        return self.arena

    def conclude(deinit self, outputs: MatchesViewArguments, viewing: ArenaRead) -> Self.Result:
        return {outputs, self.count, viewing}


@fieldwise_init
struct _SequenceSaveCall(Call):
    """The crossing of `usearch_sequence_save`."""

    comptime Arguments = DenseSaveArguments
    comptime Outputs = Unwritten
    comptime Result = NoneType
    comptime symbol: StaticString = "usearch_sequence_save"
    var path: BytesView[ImmUntrackedOrigin]

    @staticmethod
    def verb(library: Library) -> Verb[Self.Arguments]:
        return library.symbols().sequence_save

    def arguments(self, head: CallHead, outputs: Pointer[Self.Outputs, MutUntrackedOrigin]) -> Self.Arguments:
        return {head.index, head.executor, head.outcome, head.completion, self.path, ProgressArguments()}

    def conclude(deinit self, outputs: Unwritten, viewing: ArenaRead) -> NoneType:
        return None


@fieldwise_init
struct _SequenceSaveBufferCall[origin: MutOrigin](Call):
    """The crossing of `usearch_sequence_save_buffer`, which uses the arena the monolith lands in."""

    comptime Arguments = DenseSaveBufferArguments
    comptime Outputs = DenseSaveBufferOutputs
    comptime Result = Monolith[Self.origin]
    comptime symbol: StaticString = "usearch_sequence_save_buffer"
    var arena: Address

    @staticmethod
    def verb(library: Library) -> Verb[Self.Arguments]:
        return library.symbols().sequence_save_buffer

    def arguments(self, head: CallHead, outputs: Pointer[Self.Outputs, MutUntrackedOrigin]) -> Self.Arguments:
        return {head.index, head.executor, head.outcome, head.completion, outputs, self.arena, ProgressArguments()}

    def arena_handle(self) -> Address:
        return self.arena

    def conclude(deinit self, outputs: DenseSaveBufferOutputs, viewing: ArenaRead) -> Self.Result:
        return {outputs.data, Int(outputs.length), viewing}


@fieldwise_init
struct _SequenceCursorNextCall[origin: MutOrigin](Call):
    """The crossing of `usearch_sequence_cursor_next`, which uses the arena its batch lands in."""

    comptime Arguments = DenseCursorNextArguments
    comptime Outputs = DenseCursorNextOutputs
    comptime Result = DenseBatch[Self.origin]
    comptime symbol: StaticString = "usearch_sequence_cursor_next"
    var cursor: Address
    var links: Bool
    var wanted: Int
    var arena: Address

    @staticmethod
    def verb(library: Library) -> Verb[Self.Arguments]:
        return library.symbols().sequence_cursor_next

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

    def arena_handle(self) -> Address:
        return self.arena

    def conclude(deinit self, outputs: DenseCursorNextOutputs, viewing: ArenaRead) -> Self.Result:
        return {outputs.keys, self.links, outputs.links, Int(outputs.written), viewing}


# endregion Calls

# region Index


struct SequenceIndex(Movable, SequenceVerbs):
    """A sequence index, released when it goes out of scope.

    Insertions and searches overlap freely on one handle, as on a dense one. Verbs called on the
    index run on the calling thread; `index.on(executor)` runs the same verbs blocking on an
    executor, and `index.deferred(executor)` submits them there.
    """

    var library: Library
    """The loaded engine, held so its symbol table outlives every call made through the handle."""
    var address: Address
    """The opaque `usearch_sequence_index_t`."""

    # region Lifetime

    def __init__(out self, *, library: Library, address: Address):
        """Adopts a handle the engine just created.

        Args:
            library: The loaded engine the handle was created through, and is freed through.
            address: A `usearch_sequence_index_t` whose lifetime this index takes over.
        """
        self.library = library
        self.address = address

    @staticmethod
    def init_ram[
        weights_origin: ImmOrigin, widths_origin: ImmOrigin, ram_origin: ImmOrigin, //
    ](
        library: Library,
        options: SequenceOptions[weights_origin, widths_origin] = SequenceOptions(),
        ram: RamOptions[ram_origin] = RamOptions(),
    ) raises Error -> Self:
        """Initializes an empty, mutable index in host memory. Calls `usearch_sequence_init_ram`.

        Parameters:
            weights_origin: The options' viewed substitution matrix, inferred from the argument.
            widths_origin: The options' viewed window widths, inferred with it.
            ram_origin: The placement's viewed domain array, inferred from the argument.

        Args:
            library: The loaded engine, kept so the shared object stays mapped.
            options: The scoring kind and its parameters; read only during this call.
            ram: The memory domains to place one shard on apiece.

        Returns:
            The new index.

        Raises:
            `Error` with `CONFIG_MISMATCH` for weights or window widths the kind does not take.
        """
        var created = stack_allocation[1, Address]()
        created[unsafe_offset=0] = NULL_ADDRESS
        var outcome = stack_allocation[1, Outcome]()
        outcome[unsafe_offset=0] = Outcome()
        var weights = stack_allocation[1, SequenceWeightsArguments]()
        _cross(
            SequenceInitRamArguments(
                created.unsafe_origin_cast[MutUntrackedOrigin](),
                outcome.unsafe_origin_cast[MutUntrackedOrigin](),
                options.metric,
                _pass_weights(options.weights, weights.unsafe_origin_cast[MutAnyOrigin]()),
                _pass_widths(options.window_widths),
                c_size_t(len(options.window_widths)),
                options.connectivity,
                options.values_per_key,
                ram.arguments(),
            ),
            library.symbols().sequence_init_ram,
        )
        library._check(outcome.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0], "usearch_sequence_init_ram")
        return Self(library=library, address=created.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0])

    @staticmethod
    def init_vram[
        weights_origin: ImmOrigin, widths_origin: ImmOrigin, vram_origin: ImmOrigin, //
    ](
        library: Library,
        options: SequenceOptions[weights_origin, widths_origin] = SequenceOptions(),
        vram: VramOptions[vram_origin] = VramOptions(),
    ) raises Error -> Self:
        """Initializes an empty, mutable index in device memory. Calls `usearch_sequence_init_vram`.

        Parameters:
            weights_origin: The options' viewed substitution matrix, inferred from the argument.
            widths_origin: The options' viewed window widths, inferred with it.
            vram_origin: The placement's viewed ordinal array, inferred from the argument.

        Args:
            library: The loaded engine, kept so the shared object stays mapped.
            options: The scoring kind and its parameters; read only during this call.
            vram: The device to place the index on.

        Returns:
            The new index.

        Raises:
            `Error` if the kind is refused, or no device can be selected or claimed.
        """
        var created = stack_allocation[1, Address]()
        created[unsafe_offset=0] = NULL_ADDRESS
        var outcome = stack_allocation[1, Outcome]()
        outcome[unsafe_offset=0] = Outcome()
        var weights = stack_allocation[1, SequenceWeightsArguments]()
        _cross(
            SequenceInitVramArguments(
                created.unsafe_origin_cast[MutUntrackedOrigin](),
                outcome.unsafe_origin_cast[MutUntrackedOrigin](),
                options.metric,
                _pass_weights(options.weights, weights.unsafe_origin_cast[MutAnyOrigin]()),
                _pass_widths(options.window_widths),
                c_size_t(len(options.window_widths)),
                options.connectivity,
                options.values_per_key,
                vram.arguments(),
            ),
            library.symbols().sequence_init_vram,
        )
        library._check(outcome.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0], "usearch_sequence_init_vram")
        return Self(library=library, address=created.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0])

    @staticmethod
    def load_ram[
        path_origin: ImmOrigin, ram_origin: ImmOrigin, weights_origin: ImmOrigin, //
    ](
        executor: Executor,
        path: StringSlice[path_origin],
        ram: RamOptions[ram_origin] = RamOptions(),
        values_per_key: Int = 0,
        weights: Optional[SequenceWeights[weights_origin]] = Optional[SequenceWeights[ImmUntrackedOrigin]](),
    ) raises Error -> Self:
        """Copies a checkpoint into a new, mutable index in host memory. Calls `usearch_sequence_load_ram`.

        Parameters:
            path_origin: The path bytes' origin, inferred from the argument.
            ram_origin: The placement's viewed domain array, inferred from the argument.
            weights_origin: The viewed substitution matrix, inferred from the argument.

        Args:
            executor: The executor the read fans out over; it also supplies its queues.
            path: The manifest of a sharded checkpoint, or a single `.safegraphs` monolith.
            ram: The memory domains to place the parts on.
            values_per_key: Upper bound on entries sharing one key, which no checkpoint records.
            weights: The weights a Needleman-Wunsch checkpoint scores with, which it does not hold.

        Returns:
            The new index; a failed load creates none.

        Raises:
            `Error` if the checkpoint cannot be read or decoded.
        """
        return Self._load_ram(
            executor,
            executor.library.symbols().sequence_load_ram,
            BytesView.of(path).arguments(),
            ram.arguments(),
            values_per_key,
            weights,
            "usearch_sequence_load_ram",
        )

    @staticmethod
    def load_ram_buffer[
        buffer_origin: ImmOrigin, ram_origin: ImmOrigin, weights_origin: ImmOrigin, //
    ](
        executor: Executor,
        buffer: Span[UInt8, buffer_origin],
        ram: RamOptions[ram_origin] = RamOptions(),
        values_per_key: Int = 0,
        weights: Optional[SequenceWeights[weights_origin]] = Optional[SequenceWeights[ImmUntrackedOrigin]](),
    ) raises Error -> Self:
        """Copies a monolith from a memory buffer into a new, mutable index in host memory.
        Calls `usearch_sequence_load_ram_buffer`.

        Parameters:
            buffer_origin: The source bytes' origin, inferred from the argument.
            ram_origin: The placement's viewed domain array, inferred from the argument.
            weights_origin: The viewed substitution matrix, inferred from the argument.

        Args:
            executor: The executor the copy fans out over; it also supplies its queues.
            buffer: Source holding one serialized monolith.
            ram: The memory domains to place shards on; the monolith lands whole on the first.
            values_per_key: Upper bound on entries sharing one key, which no checkpoint records.
            weights: The weights a Needleman-Wunsch monolith scores with, which it does not hold.

        Returns:
            The new index; a failed load creates none.

        Raises:
            `Error` if the bytes cannot be decoded.
        """
        return Self._load_ram(
            executor,
            executor.library.symbols().sequence_load_ram_buffer,
            BytesView.of(buffer).arguments(),
            ram.arguments(),
            values_per_key,
            weights,
            "usearch_sequence_load_ram_buffer",
        )

    @staticmethod
    def _load_ram[
        weights_origin: ImmOrigin
    ](
        executor: Executor,
        verb: SequenceLoadRam,
        source: BytesView[ImmUntrackedOrigin],
        ram: RamOptions[ImmUntrackedOrigin],
        values_per_key: Int,
        weights: Optional[SequenceWeights[weights_origin]],
        detail: StaticString,
    ) raises Error -> Self:
        """The path and buffer forms share one argument layout, differing only in the verb."""
        var created = stack_allocation[1, Address]()
        created[unsafe_offset=0] = NULL_ADDRESS
        var outcome = stack_allocation[1, Outcome]()
        outcome[unsafe_offset=0] = Outcome()
        var passed = stack_allocation[1, SequenceWeightsArguments]()
        _cross(
            SequenceLoadRamArguments(
                created.unsafe_origin_cast[MutUntrackedOrigin](),
                executor.address,
                outcome.unsafe_origin_cast[MutUntrackedOrigin](),
                Completion.BLOCKING,
                source,
                ram,
                c_size_t(values_per_key),
                _pass_weights(weights, passed.unsafe_origin_cast[MutAnyOrigin]()),
                ProgressArguments(),
            ),
            verb,
        )
        executor.library._check(outcome.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0], detail)
        return Self(library=executor.library, address=created.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0])

    @staticmethod
    def load_vram[
        path_origin: ImmOrigin, vram_origin: ImmOrigin, weights_origin: ImmOrigin, //
    ](
        executor: Executor,
        path: StringSlice[path_origin],
        vram: VramOptions[vram_origin] = VramOptions(),
        values_per_key: Int = 0,
        weights: Optional[SequenceWeights[weights_origin]] = Optional[SequenceWeights[ImmUntrackedOrigin]](),
    ) raises Error -> Self:
        """Copies a checkpoint into a new, mutable index in device memory. Calls `usearch_sequence_load_vram`.

        Parameters:
            path_origin: The path bytes' origin, inferred from the argument.
            vram_origin: The placement's viewed ordinal array, inferred from the argument.
            weights_origin: The viewed substitution matrix, inferred from the argument.

        Args:
            executor: The executor the upload runs on; it also supplies its queues.
            path: The manifest of a sharded checkpoint, or a single `.safegraphs` monolith.
            vram: The device to place the index on.
            values_per_key: Upper bound on entries sharing one key, which no checkpoint records.
            weights: The weights a Needleman-Wunsch checkpoint scores with, which it does not hold.

        Returns:
            The new index; a failed load creates none.

        Raises:
            `Error` if the checkpoint cannot be read, or a device cannot hold it.
        """
        return Self._load_vram(
            executor,
            executor.library.symbols().sequence_load_vram,
            BytesView.of(path).arguments(),
            vram.arguments(),
            values_per_key,
            weights,
            "usearch_sequence_load_vram",
        )

    @staticmethod
    def load_vram_buffer[
        buffer_origin: ImmOrigin, vram_origin: ImmOrigin, weights_origin: ImmOrigin, //
    ](
        executor: Executor,
        buffer: Span[UInt8, buffer_origin],
        vram: VramOptions[vram_origin] = VramOptions(),
        values_per_key: Int = 0,
        weights: Optional[SequenceWeights[weights_origin]] = Optional[SequenceWeights[ImmUntrackedOrigin]](),
    ) raises Error -> Self:
        """Copies a monolith from a memory buffer into a new, mutable index in device memory.
        Calls `usearch_sequence_load_vram_buffer`.

        Parameters:
            buffer_origin: The source bytes' origin, inferred from the argument.
            vram_origin: The placement's viewed ordinal array, inferred from the argument.
            weights_origin: The viewed substitution matrix, inferred from the argument.

        Args:
            executor: The executor the upload runs on; it also supplies its queues.
            buffer: Source holding one serialized monolith.
            vram: The device to place the index on.
            values_per_key: Upper bound on entries sharing one key, which no checkpoint records.
            weights: The weights a Needleman-Wunsch monolith scores with, which it does not hold.

        Returns:
            The new index; a failed load creates none.

        Raises:
            `Error` if the bytes cannot be decoded, or a device cannot hold them.
        """
        return Self._load_vram(
            executor,
            executor.library.symbols().sequence_load_vram_buffer,
            BytesView.of(buffer).arguments(),
            vram.arguments(),
            values_per_key,
            weights,
            "usearch_sequence_load_vram_buffer",
        )

    @staticmethod
    def _load_vram[
        weights_origin: ImmOrigin
    ](
        executor: Executor,
        verb: SequenceLoadVram,
        source: BytesView[ImmUntrackedOrigin],
        vram: VramOptions[ImmUntrackedOrigin],
        values_per_key: Int,
        weights: Optional[SequenceWeights[weights_origin]],
        detail: StaticString,
    ) raises Error -> Self:
        """The path and buffer forms share one argument layout, differing only in the verb."""
        var created = stack_allocation[1, Address]()
        created[unsafe_offset=0] = NULL_ADDRESS
        var outcome = stack_allocation[1, Outcome]()
        outcome[unsafe_offset=0] = Outcome()
        var passed = stack_allocation[1, SequenceWeightsArguments]()
        _cross(
            SequenceLoadVramArguments(
                created.unsafe_origin_cast[MutUntrackedOrigin](),
                executor.address,
                outcome.unsafe_origin_cast[MutUntrackedOrigin](),
                Completion.BLOCKING,
                source,
                vram,
                c_size_t(values_per_key),
                _pass_weights(weights, passed.unsafe_origin_cast[MutAnyOrigin]()),
                ProgressArguments(),
            ),
            verb,
        )
        executor.library._check(outcome.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0], detail)
        return Self(library=executor.library, address=created.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0])

    @staticmethod
    def view_ram[
        path_origin: ImmOrigin, weights_origin: ImmOrigin, //
    ](
        library: Library,
        path: StringSlice[path_origin],
        weights: Optional[SequenceWeights[weights_origin]] = Optional[SequenceWeights[ImmUntrackedOrigin]](),
    ) raises Error -> Self:
        """Serves a checkpoint in place through a memory mapping, as a read-only index.
        Calls `usearch_sequence_view_ram`.

        Parameters:
            path_origin: The path bytes' origin, inferred from the argument.
            weights_origin: The viewed substitution matrix, inferred from the argument.

        Args:
            library: The loaded engine, kept so the shared object stays mapped.
            path: The manifest of a sharded checkpoint, or a single `.safegraphs` monolith.
            weights: The weights a Needleman-Wunsch checkpoint scores with, which it does not hold.

        Returns:
            The read-only index.

        Raises:
            `Error` with `UNSUPPORTED` where the engine serves no mapped sequence index.
        """
        var created = stack_allocation[1, Address]()
        created[unsafe_offset=0] = NULL_ADDRESS
        var outcome = stack_allocation[1, Outcome]()
        outcome[unsafe_offset=0] = Outcome()
        var passed = stack_allocation[1, SequenceWeightsArguments]()
        _cross(
            SequenceViewRamArguments(
                created.unsafe_origin_cast[MutUntrackedOrigin](),
                outcome.unsafe_origin_cast[MutUntrackedOrigin](),
                BytesView.of(path).arguments(),
                _pass_weights(weights, passed.unsafe_origin_cast[MutAnyOrigin]()),
                RamOptions().arguments(),
            ),
            library.symbols().sequence_view_ram,
        )
        library._check(outcome.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0], "usearch_sequence_view_ram")
        return Self(library=library, address=created.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0])

    @staticmethod
    def view_nvme[
        path_origin: ImmOrigin, nvme_origin: ImmOrigin, weights_origin: ImmOrigin, //
    ](
        executor: Executor,
        path: StringSlice[path_origin],
        nvme: NvmeOptions[nvme_origin] = NvmeOptions(),
        weights: Optional[SequenceWeights[weights_origin]] = Optional[SequenceWeights[ImmUntrackedOrigin]](),
    ) raises Error -> Self:
        """Serves a checkpoint in place from NVMe drives, read through the executor's rings.
        Calls `usearch_sequence_view_nvme`.

        Parameters:
            path_origin: The path bytes' origin, inferred from the argument.
            nvme_origin: The options' viewed path array, inferred from the argument.
            weights_origin: The viewed substitution matrix, inferred from the argument.

        Args:
            executor: The NVMe executor whose rings read the checkpoint; it also supplies its queues.
            path: The manifest of a sharded checkpoint, or a single `.safegraphs` monolith.
            nvme: The devices the parts are served from; naming none serves each where it points.
            weights: The weights a Needleman-Wunsch checkpoint scores with, which it does not hold.

        Returns:
            The read-only index.

        Raises:
            `Error` with `IO` if the checkpoint cannot be opened.
        """
        var created = stack_allocation[1, Address]()
        created[unsafe_offset=0] = NULL_ADDRESS
        var outcome = stack_allocation[1, Outcome]()
        outcome[unsafe_offset=0] = Outcome()
        var passed = stack_allocation[1, SequenceWeightsArguments]()
        _cross(
            SequenceViewNvmeArguments(
                created.unsafe_origin_cast[MutUntrackedOrigin](),
                executor.address,
                outcome.unsafe_origin_cast[MutUntrackedOrigin](),
                Completion.BLOCKING,
                BytesView.of(path).arguments(),
                nvme.arguments(),
                _pass_weights(weights, passed.unsafe_origin_cast[MutAnyOrigin]()),
            ),
            executor.library.symbols().sequence_view_nvme,
        )
        executor.library._check(
            outcome.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0], "usearch_sequence_view_nvme"
        )
        return Self(library=executor.library, address=created.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0])

    def __deinit__(deinit self):
        """Releases the index and everything it owns. Calls `usearch_sequence_free`.

        The loader reference is surrendered only after the call, so the engine stays mapped until
        the release returns.
        """
        self.library.symbols().sequence_free(self.address)
        _ = self.library^

    # endregion Lifetime

    # region Receivers

    def on(ref self, ref executor: Executor) -> SequenceIndexOn[ImmOrigin(origin_of(self, executor))]:
        """The same verbs, blocking on `executor`, refused while another call holds it. Binding-only.

        Args:
            executor: The executor every verb runs on; its kind must match the handle's residency.

        Returns:
            A receiver viewing both, which neither may outlive.
        """
        return SequenceIndexOn[ImmOrigin(origin_of(self, executor))](
            Pointer[SequenceIndex, ImmOrigin(origin_of(self, executor))](unsafe_from_address=Int(Pointer(to=self))),
            executor.address,
        )

    def deferred(ref self, ref executor: Executor) -> SequenceIndexDeferredOn[ImmOrigin(origin_of(self, executor))]:
        """The same verbs, each submitted to `executor` and answering a `Deferred` at once. Binding-only.

        Args:
            executor: The executor every verb is submitted to; its kind must match the handle's residency.

        Returns:
            A receiver viewing both, which neither it nor any `Deferred` it answers may outlive.
        """
        return SequenceIndexDeferredOn[ImmOrigin(origin_of(self, executor))](
            Pointer[SequenceIndex, ImmOrigin(origin_of(self, executor))](unsafe_from_address=Int(Pointer(to=self))),
            executor.address,
        )

    def _route[C: Call](self, var call: C) raises Error -> C.Result:
        """Runs `call` on the calling thread."""
        return _block(self.library, call^, self.address, NULL_ADDRESS)

    def cursor(ref self) raises Error -> SequenceCursor[ImmOrigin(origin_of(self))]:
        """Opens a walk over every entry as stored, its pulls on the calling thread.
        Calls `usearch_sequence_cursor_init`.

        Returns:
            The cursor, which uses the index and cannot outlive it.

        Raises:
            `Error` if the residency has no cursor, as VRAM does not.
        """
        return SequenceCursor[ImmOrigin(origin_of(self))](self.library, self.address, 0, None)

    def cursor[
        starts_origin: ImmOrigin, //
    ](ref self, starts: Span[Key, starts_origin]) raises Error -> SequenceCursor[ImmOrigin(origin_of(self))]:
        """Opens a walk from `starts`, its pulls on the calling thread. Calls `usearch_sequence_cursor_init`.

        Parameters:
            starts_origin: The start keys' origin, inferred from the argument.

        Args:
            starts: The keys to walk from, the walk yielding their entries first.

        Returns:
            The cursor, which uses the index and cannot outlive it.

        Raises:
            `Error` if the residency has no cursor, as VRAM does not.
        """
        return SequenceCursor[ImmOrigin(origin_of(self))](
            self.library,
            self.address,
            len(starts),
            starts.unsafe_ptr().unsafe_origin_cast[ImmUntrackedOrigin](),
        )

    # endregion Receivers


def _pass_widths[origin: ImmOrigin](widths: Span[c_size_t, origin]) -> Optional[Pointer[c_size_t, ImmUntrackedOrigin]]:
    """The first window width, or `None` for none, its view erased for the call it crosses."""
    if len(widths) == 0:
        return None
    return widths.unsafe_ptr().unsafe_origin_cast[ImmUntrackedOrigin]()


trait SequenceVerbs:
    """The verbs every blocking receiver offers - the index on the calling thread, and
    `index.on(executor)` on an executor - each written once over the receiver's `_route`."""

    def _route[C: Call](self, var call: C) raises Error -> C.Result:
        """Runs `call` to its final status on this receiver's executor and answers its result."""
        ...

    def reserve(self, entries: Int, payload_bytes: Int = 0) raises Error:
        """Reserves room for incoming entries ahead of insertion. Calls `usearch_sequence_reserve`.

        Args:
            entries: The desired total capacity in entries, including the current size.
            payload_bytes: The desired total bytes of sequences; zero reserves none ahead.

        Raises:
            `Error` if the reservation cannot be satisfied.
        """
        _ = self._route(_SequenceReserveCall(entries, payload_bytes))

    def clear(self) raises Error:
        """Erases every entry, keeping the reserved capacity and the frozen kind. Calls `usearch_sequence_clear`.

        Raises:
            `Error` with `READ_ONLY` if the handle serves its checkpoint in place.
        """
        _ = self._route(_SequenceClearCall())

    def describe(self) raises Error -> SequenceIndexDescription:
        """Reports the frozen metadata and the current counters in one call. Calls `usearch_sequence_describe`.

        Returns:
            The description.

        Raises:
            `Error` if the engine refuses the query.
        """
        return self._route(_SequenceDescribeCall())

    def insert[
        keys_origin: ImmOrigin, sequences_origin: ImmOrigin, stats_origin: MutOrigin, //
    ](
        self,
        keys: Span[Key, keys_origin],
        sequences: TapeView[sequences_origin],
        config: DenseInsertConfig[stats_origin] = DenseInsertConfig(),
    ) raises Error:
        """Adds a tape of sequences under their keys, all or nothing. Calls `usearch_sequence_insert`.

        Parameters:
            keys_origin: The key run's origin, inferred from the argument.
            sequences_origin: The tape's origin, inferred with it.
            stats_origin: The passed telemetry slot's origin, inferred with it.

        Args:
            keys: One key per sequence; keys need not be unique.
            sequences: The sequences to insert, a tape of bytes.
            config: The breadth of the candidates and the telemetry slot.

        Raises:
            `Error` with `SHAPE_MISMATCH` when the keys and sequences differ in count, `OVERFLOW` past
            `values_per_key`, or `READ_ONLY` for a view.
        """
        _ = self._route(_insert_call(keys, sequences, config))

    def refine[
        stats_origin: MutOrigin, plan_origin: MutOrigin, phase_origin: MutOrigin, //
    ](
        self, config: SequenceRefineConfig[stats_origin, plan_origin, phase_origin] = SequenceRefineConfig()
    ) raises Error -> DenseRefineReport:
        """Runs one round over the whole graph, healing islands and re-stitching neighborhoods, or
        continues `config.plan`. Calls `usearch_sequence_refine`.

        Parameters:
            stats_origin: The passed telemetry slot's origin, inferred from the argument.
            plan_origin: The passed plan's origin, inferred with it.
            phase_origin: The passed phase slot's origin, inferred with it.

        Args:
            config: The round's spending, the handler, the telemetry slot, and the plan and phase slots.

        Returns:
            What the round found and changed, its verdict, and its opening and closing health checks.

        Raises:
            `Error` with `INVALID_ARGUMENT` for a plan on the calling thread, `READ_ONLY` for a view, or
            `CANCELLED` once the handler stops the round.
        """
        return self._route(_refine_call(config))

    def refine_plan_finish(self, var plan: SequenceRefinePlan) raises Error -> DenseRefineReport:
        """Closes and frees a refine plan whatever the call answers, reopening the index.
        Calls `usearch_sequence_refine_plan_finish`.

        Args:
            plan: The plan a refine on this index and executor opened, consumed by the call.

        Returns:
            What the whole plan did to the graph.

        Raises:
            `Error` with `INVALID_ARGUMENT` for a plan no refine opened, or one already finished.
        """
        return self._route(_SequenceRefinePlanFinishCall(plan.address))

    def health(self) raises Error -> DenseReachability:
        """Measures how much of the graph a search reaches from the root. Calls `usearch_sequence_health`.

        Returns:
            The audit; entries it misses stay unfound until a `refine` re-links them.

        Raises:
            `Error` if the engine refuses the audit.
        """
        return self._route(_SequenceHealthCall())

    def sample[
        shown_origin: ImmOrigin, coordinates_origin: ImmOrigin, focus_origin: ImmOrigin, //
    ](
        self,
        wanted: Int,
        mut arena: Arena,
        config: DenseSampleConfig[shown_origin, coordinates_origin, focus_origin] = DenseSampleConfig(),
    ) raises Error -> DenseDraws[origin_of(arena)]:
        """Draws up to `wanted` keys around `config.focus`, or across the whole index, each hanging from
        a shown or earlier draw and placed in three dimensions. Calls `usearch_sequence_sample`.

        Parameters:
            shown_origin: The shown keys' origin, inferred from the config.
            coordinates_origin: Their coordinates' origin, inferred with it.
            focus_origin: The focus keys' origin, inferred with it.

        Args:
            wanted: The upper bound on draws.
            arena: The arena to rewind and refill, which the result views; a failed call leaves it valid.
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
        """Finds the `wanted` nearest entries of every query sequence, writing the matches in a passed
        arena. Calls `usearch_sequence_search`.

        Parameters:
            queries_origin: The query tape's origin, inferred from the argument.
            stats_origin: The passed telemetry slot's origin, inferred with it.

        Args:
            queries: The query sequences, a tape of bytes.
            wanted: Matches per query - the "k"; zero lands only the counts.
            arena: The arena to rewind and refill, which the result views; a failed call leaves it valid.
            config: The breadth of the candidates, the predicate, and the telemetry slot.

        Returns:
            Every query's matches, nearest first, read in place from the arena.

        Raises:
            `Error` if the engine refuses the search, or a predicate is passed to a VRAM handle.
        """
        return self._route(_search_call[origin_of(arena)](queries, wanted, arena._address, config))

    def count(self, key: Key) raises Error -> SequenceKeyStats:
        """How many entries one key holds, and their byte lengths. Calls `usearch_sequence_count`.

        Args:
            key: The key whose entries are counted.

        Returns:
            The entries under the key, their total and their largest byte length.

        Raises:
            `Error` if the engine refuses the query.
        """
        return self._route(_SequenceCountCall(key))

    def get(self, key: Key, mut arena: Arena) raises Error -> TapeEntries[DType.uint8, origin_of(arena)]:
        """Exports every sequence stored under one key into a passed arena, in insertion order.
        Calls `usearch_sequence_get`.

        Args:
            key: The key whose entries are exported.
            arena: The arena to rewind and refill, which the result views; a failed call leaves it valid.

        Returns:
            The sequences, read in place from the arena; none when the key is absent.

        Raises:
            `Error` if the engine refuses the export.
        """
        return self._route(_SequenceGetCall[origin_of(arena)](key, arena._address))

    def save[path_origin: ImmOrigin, //](self, path: StringSlice[path_origin]) raises Error:
        """Writes a checkpoint, the path extension picking the layout. Calls `usearch_sequence_save`.

        Parameters:
            path_origin: The path bytes' origin, inferred from the argument.

        Args:
            path: Destination path, whose extension picks the layout.

        Raises:
            `Error` if the destination cannot be written, or `UNSUPPORTED` for an NVMe view.
        """
        _ = self._route(_SequenceSaveCall(BytesView.of(path).arguments()))

    def save_buffer(self, mut arena: Arena) raises Error -> Monolith[origin_of(arena)]:
        """Serializes the index as one monolith into a passed arena. Calls `usearch_sequence_save_buffer`.

        Args:
            arena: The arena to rewind and refill, which the result views; a failed call leaves it valid.

        Returns:
            The monolith, read in place from the arena.

        Raises:
            `Error` if the engine refuses the write.
        """
        return self._route(_SequenceSaveBufferCall[origin_of(arena)](arena._address))


struct SequenceIndexOn[origin: ImmOrigin](ImplicitlyCopyable, SequenceVerbs):
    """An index bound to the executor its verbs block on, as `index.on(executor)` answers it. Binding-only.

    Parameters:
        origin: The index and the executor, which this receiver views.
    """

    var index: Pointer[SequenceIndex, Self.origin]
    """The index every verb addresses."""
    var executor: Address
    """The `usearch_executor_t` every verb runs on."""

    def __init__(out self, index: Pointer[SequenceIndex, Self.origin], executor: Address):
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


def _refine_call[
    stats_origin: MutOrigin, plan_origin: MutOrigin, phase_origin: MutOrigin
](config: SequenceRefineConfig[stats_origin, plan_origin, phase_origin]) -> _SequenceRefineCall:
    """Lowers a refinement's passed slots into its crossing."""
    return _SequenceRefineCall(
        RefineOptionsArguments(c_size_t(config.expansion_factor), c_size_t(config.memory_limit)),
        config.progress,
        _slot(config.stats),
        _slot(config.plan),
        _slot(config.phase),
    )


def _insert_call[
    keys_origin: ImmOrigin, sequences_origin: ImmOrigin, stats_origin: MutOrigin
](
    keys: Span[Key, keys_origin], sequences: TapeView[sequences_origin], config: DenseInsertConfig[stats_origin]
) -> _SequenceInsertCall:
    """Lowers an insertion's viewed inputs into its crossing."""
    return _SequenceInsertCall(
        KeysView.of(keys).arguments(),
        len(keys),
        sequences.view,
        sequences.entries,
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
) -> _SequenceSampleCall[origin]:
    """Lowers a sample's viewed inputs into its crossing."""
    return _SequenceSampleCall[origin](
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
) -> _SequenceSearchCall[origin]:
    """Lowers a search's viewed inputs into its crossing."""
    return _SequenceSearchCall[origin](
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


struct SequenceIndexDeferredOn[origin: ImmOrigin](ImplicitlyCopyable):
    """An index bound to the executor its verbs are submitted to, as `index.deferred(executor)` answers
    it: every verb answers a `Deferred` at once. Binding-only.

    Parameters:
        origin: The index and the executor, which this receiver views.
    """

    var index: Pointer[SequenceIndex, Self.origin]
    """The index every verb addresses."""
    var executor: Address
    """The `usearch_executor_t` every verb is submitted to."""

    def __init__(out self, index: Pointer[SequenceIndex, Self.origin], executor: Address):
        """Binds an index to an executor.

        Args:
            index: The index every verb addresses.
            executor: The `usearch_executor_t` every verb is submitted to.
        """
        self.index = index
        self.executor = executor

    def reserve(self, entries: Int, payload_bytes: Int = 0) -> Deferred[_SequenceReserveCall, Self.origin]:
        """Submits `SequenceVerbs.reserve`. Calls `usearch_sequence_reserve`."""
        return {
            self.index[].library,
            _SequenceReserveCall(entries, payload_bytes),
            self.index[].address,
            self.executor,
        }

    def clear(self) -> Deferred[_SequenceClearCall, Self.origin]:
        """Submits `SequenceVerbs.clear`. Calls `usearch_sequence_clear`."""
        return {self.index[].library, _SequenceClearCall(), self.index[].address, self.executor}

    def describe(self) -> Deferred[_SequenceDescribeCall, Self.origin]:
        """Submits `SequenceVerbs.describe`. Calls `usearch_sequence_describe`."""
        return {self.index[].library, _SequenceDescribeCall(), self.index[].address, self.executor}

    def insert[
        keys_origin: ImmOrigin, sequences_origin: ImmOrigin, stats_origin: MutOrigin, //
    ](
        self,
        keys: Span[Key, keys_origin],
        sequences: TapeView[sequences_origin],
        config: DenseInsertConfig[stats_origin] = DenseInsertConfig(),
    ) -> Deferred[
        _SequenceInsertCall,
        ImmOrigin(
            origin_of(
                _anchor[Self.origin](), _anchor[keys_origin](), _anchor[sequences_origin](), _anchor[stats_origin]()
            )
        ),
    ]:
        """Submits `SequenceVerbs.insert`; the keys, tape and telemetry slot must stay allocated and unchanged until final.
        Calls `usearch_sequence_insert`."""
        return {self.index[].library, _insert_call(keys, sequences, config), self.index[].address, self.executor}

    def refine[
        stats_origin: MutOrigin, plan_origin: MutOrigin, phase_origin: MutOrigin, //
    ](self, config: SequenceRefineConfig[stats_origin, plan_origin, phase_origin] = SequenceRefineConfig()) -> Deferred[
        _SequenceRefineCall,
        ImmOrigin(
            origin_of(_anchor[Self.origin](), _anchor[stats_origin](), _anchor[plan_origin](), _anchor[phase_origin]())
        ),
    ]:
        """Submits `SequenceVerbs.refine`; the telemetry, plan and phase slots must stay allocated and unchanged until final.
        Calls `usearch_sequence_refine`."""
        return {self.index[].library, _refine_call(config), self.index[].address, self.executor}

    def refine_plan_finish(self, var plan: SequenceRefinePlan) -> Deferred[_SequenceRefinePlanFinishCall, Self.origin]:
        """Submits `SequenceVerbs.refine_plan_finish`. Calls `usearch_sequence_refine_plan_finish`."""
        return {
            self.index[].library,
            _SequenceRefinePlanFinishCall(plan.address),
            self.index[].address,
            self.executor,
        }

    def health(self) -> Deferred[_SequenceHealthCall, Self.origin]:
        """Submits `SequenceVerbs.health`. Calls `usearch_sequence_health`."""
        return {self.index[].library, _SequenceHealthCall(), self.index[].address, self.executor}

    def sample[
        shown_origin: ImmOrigin, coordinates_origin: ImmOrigin, focus_origin: ImmOrigin, //
    ](
        self,
        wanted: Int,
        mut arena: Arena,
        config: DenseSampleConfig[shown_origin, coordinates_origin, focus_origin] = DenseSampleConfig(),
    ) -> Deferred[
        _SequenceSampleCall[origin_of(arena)],
        ImmOrigin(
            origin_of(
                _anchor[Self.origin](),
                _anchor[shown_origin](),
                _anchor[coordinates_origin](),
                _anchor[focus_origin](),
            )
        ),
    ]:
        """Submits `SequenceVerbs.sample`; the shown keys, focus and arena must stay allocated and unchanged until final.
        Calls `usearch_sequence_sample`."""
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
        _SequenceSearchCall[origin_of(arena)],
        ImmOrigin(origin_of(_anchor[Self.origin](), _anchor[queries_origin](), _anchor[stats_origin]())),
    ]:
        """Submits `SequenceVerbs.search`; the queries, telemetry slot and arena must stay allocated and unchanged until final.
        Calls `usearch_sequence_search`."""
        return {
            self.index[].library,
            _search_call[origin_of(arena)](queries, wanted, arena._address, config),
            self.index[].address,
            self.executor,
        }

    def count(self, key: Key) -> Deferred[_SequenceCountCall, Self.origin]:
        """Submits `SequenceVerbs.count`. Calls `usearch_sequence_count`."""
        return {self.index[].library, _SequenceCountCall(key), self.index[].address, self.executor}

    def get(self, key: Key, mut arena: Arena) -> Deferred[_SequenceGetCall[origin_of(arena)], Self.origin]:
        """Submits `SequenceVerbs.get`; the arena must stay allocated and unchanged until final. Calls `usearch_sequence_get`.
        """
        return {
            self.index[].library,
            _SequenceGetCall[origin_of(arena)](key, arena._address),
            self.index[].address,
            self.executor,
        }

    def save[
        path_origin: ImmOrigin, //
    ](self, path: StringSlice[path_origin]) -> Deferred[
        _SequenceSaveCall, ImmOrigin(origin_of(_anchor[Self.origin](), _anchor[path_origin]()))
    ]:
        """Submits `SequenceVerbs.save`; the path must stay allocated and unchanged until final. Calls `usearch_sequence_save`.
        """
        return {
            self.index[].library,
            _SequenceSaveCall(BytesView.of(path).arguments()),
            self.index[].address,
            self.executor,
        }

    def save_buffer(self, mut arena: Arena) -> Deferred[_SequenceSaveBufferCall[origin_of(arena)], Self.origin]:
        """Submits `SequenceVerbs.save_buffer`; the arena must stay allocated and unchanged until final.
        Calls `usearch_sequence_save_buffer`."""
        return {
            self.index[].library,
            _SequenceSaveBufferCall[origin_of(arena)](arena._address),
            self.index[].address,
            self.executor,
        }


struct SequenceCursor[origin: ImmOrigin]:
    """A walk over the index contents - `usearch_sequence_cursor_t`.

    Export, migration, and visualization feed off it. It uses the index, so the index cannot be
    released while it lives, and it keeps no executor: its pulls run on the calling thread, or
    block on the executor `on` names for each of them.

    Parameters:
        origin: The index, which the cursor views.
    """

    var library: Library
    """The loaded engine, held so the release can still reach `usearch_sequence_cursor_free`."""
    var index: Address
    """The `usearch_sequence_index_t` the cursor was opened on, which every pull names again."""
    var address: Address
    """The `usearch_sequence_cursor_t` this struct owns."""

    def __init__(
        out self,
        library: Library,
        index: Address,
        starts_count: Int,
        starts: Optional[Pointer[Key, ImmUntrackedOrigin]],
    ) raises Error:
        """Opens a cursor over `index`. Calls `usearch_sequence_cursor_init`.

        Args:
            library: The loaded engine the cursor is opened through.
            index: The handle to scan.
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
            library.symbols().sequence_cursor_init,
        )
        library._check(outcome.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0], "usearch_sequence_cursor_init")
        self.library = library
        self.index = index
        self.address = created.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0]

    def __deinit__(deinit self):
        """Releases the cursor. Calls `usearch_sequence_cursor_free`."""
        self.library.symbols().sequence_cursor_free(self.address)
        _ = self.library^

    def next_keys(self, wanted: Int, mut arena: Arena) raises Error -> DenseBatch[origin_of(arena)]:
        """Copies up to `wanted` next keys into a passed arena, skipping their links.
        Calls `usearch_sequence_cursor_next`.

        Args:
            wanted: The upper bound on keys this pull writes.
            arena: The arena to rewind and refill, which the batch views; a failed call leaves it valid.

        Returns:
            The keys, read in place from the arena; an empty batch once the walk is exhausted.

        Raises:
            `Error` if the engine refuses the copy.
        """
        return self._next(
            _SequenceCursorNextCall[origin_of(arena)](self.address, False, wanted, arena._address), NULL_ADDRESS
        )

    def next_links(self, wanted: Int, mut arena: Arena) raises Error -> DenseBatch[origin_of(arena)]:
        """Copies up to `wanted` next keys into a passed arena, each with the keys it links to.
        Calls `usearch_sequence_cursor_next`.

        Args:
            wanted: The upper bound on entries this pull writes.
            arena: The arena to rewind and refill, which the batch views; a failed call leaves it valid.

        Returns:
            The keys and their links, read in place from the arena; an empty batch once the walk is exhausted.

        Raises:
            `Error` if the engine refuses the copy.
        """
        return self._next(
            _SequenceCursorNextCall[origin_of(arena)](self.address, True, wanted, arena._address), NULL_ADDRESS
        )

    def on(ref self, ref executor: Executor) -> SequenceCursorOn[Self.origin, ImmOrigin(origin_of(self, executor))]:
        """The receiver whose pulls block on `executor`, named again for every pull. Binding-only.

        Args:
            executor: The executor every pull runs on; its kind must match the index's residency.

        Returns:
            A receiver viewing the cursor and the executor, which neither may outlive.
        """
        return SequenceCursorOn[Self.origin, ImmOrigin(origin_of(self, executor))](
            Pointer[SequenceCursor[Self.origin], ImmOrigin(origin_of(self, executor))](
                unsafe_from_address=Int(Pointer(to=self))
            ),
            executor.address,
        )

    def _next[C: Call](self, var call: C, executor: Address) raises Error -> C.Result:
        """One blocking pull on `executor`, or on the calling thread for `NULL_ADDRESS`."""
        return _block(self.library, call^, self.index, executor)


struct SequenceCursorOn[cursor_origin: ImmOrigin, origin: ImmOrigin]:
    """A cursor's pulls blocking on one executor, as `SequenceCursor.on` makes them. Binding-only.

    Parameters:
        cursor_origin: The index the cursor views.
        origin: The cursor and the executor, which this receiver views.
    """

    var cursor: Pointer[SequenceCursor[Self.cursor_origin], Self.origin]
    """The cursor every pull advances."""
    var executor: Address
    """The `usearch_executor_t` every pull runs on."""

    def __init__(out self, cursor: Pointer[SequenceCursor[Self.cursor_origin], Self.origin], executor: Address):
        """Binds a cursor to an executor.

        Args:
            cursor: The cursor every pull advances.
            executor: The `usearch_executor_t` every pull runs on.
        """
        self.cursor = cursor
        self.executor = executor

    def next_keys(self, wanted: Int, mut arena: Arena) raises Error -> DenseBatch[origin_of(arena)]:
        """Copies up to `wanted` next keys into a passed arena on the bound executor, skipping their
        links. Calls `usearch_sequence_cursor_next`.

        Args:
            wanted: The upper bound on keys this pull writes.
            arena: The arena to rewind and refill, which the batch views; a failed call leaves it valid.

        Returns:
            The keys, read in place from the arena; an empty batch once the walk is exhausted.

        Raises:
            `Error` if the engine refuses the copy.
        """
        return self.cursor[]._next(
            _SequenceCursorNextCall[origin_of(arena)](self.cursor[].address, False, wanted, arena._address),
            self.executor,
        )

    def next_links(self, wanted: Int, mut arena: Arena) raises Error -> DenseBatch[origin_of(arena)]:
        """Copies up to `wanted` next keys into a passed arena on the bound executor, each with the
        keys it links to. Calls `usearch_sequence_cursor_next`.

        Args:
            wanted: The upper bound on entries this pull writes.
            arena: The arena to rewind and refill, which the batch views; a failed call leaves it valid.

        Returns:
            The keys and their links, read in place from the arena; an empty batch once the walk is exhausted.

        Raises:
            `Error` if the engine refuses the copy.
        """
        return self.cursor[]._next(
            _SequenceCursorNextCall[origin_of(arena)](self.cursor[].address, True, wanted, arena._address),
            self.executor,
        )


# endregion Index

# region Reshard


def sequence_reshard[
    sources_origin: ImmOrigin,
    targets_origin: ImmOrigin,
    stats_origin: MutOrigin,
    plan_origin: MutOrigin,
    phase_origin: MutOrigin,
    //,
](
    executor: Executor,
    sources: Span[SequenceIndex, sources_origin],
    targets: Span[SequenceIndex, targets_origin],
    config: SequenceReshardConfig[stats_origin, plan_origin, phase_origin] = SequenceReshardConfig(),
) raises Error -> DenseRefineReport:
    """Moves the entries of `sources` into `targets`, keeping their links, then runs a round on every
    target that took entries. Calls `usearch_sequence_reshard`.

    Several sources into one target among them gather into it, and one source into several empty
    targets cuts it into one graph-local part apiece; a source the targets do not name ends empty and
    live.

    Parameters:
        sources_origin: The sources' origin, inferred from the argument.
        targets_origin: The targets' origin, inferred from the argument.
        stats_origin: The passed telemetry slot's origin, inferred from the config.
        plan_origin: The passed plan's origin, inferred with it.
        phase_origin: The passed phase slot's origin, inferred with it.

    Args:
        executor: The executor every move and the closing rounds fan out over; it also supplies its queues.
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
    return _block(executor.library, _SequenceReshardCall(sources, targets, config), NULL_ADDRESS, executor.address)


def sequence_reshard_deferred[
    sources_origin: ImmOrigin,
    targets_origin: ImmOrigin,
    stats_origin: MutOrigin,
    plan_origin: MutOrigin,
    phase_origin: MutOrigin,
    //,
](
    ref executor: Executor,
    sources: Span[SequenceIndex, sources_origin],
    targets: Span[SequenceIndex, targets_origin],
    config: SequenceReshardConfig[stats_origin, plan_origin, phase_origin] = SequenceReshardConfig(),
) -> Deferred[
    _SequenceReshardCall,
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
    """Submits `sequence_reshard` to `executor`, answering a `Deferred` at once; the indexes and slots
    must stay allocated and unchanged until final. Calls `usearch_sequence_reshard`."""
    return {executor.library, _SequenceReshardCall(sources, targets, config), NULL_ADDRESS, executor.address}


def sequence_reshard_plan_finish(executor: Executor, var plan: SequenceReshardPlan) raises Error -> DenseRefineReport:
    """Closes and frees a reshard plan whatever the call answers, reopening every index it names.
    Calls `usearch_sequence_reshard_plan_finish`.

    Args:
        executor: The executor the plan was opened with; it also supplies its queues.
        plan: The plan a reshard opened, consumed by the call.

    Returns:
        What the whole plan did to the graphs; zeros while entries were still to move.

    Raises:
        `Error` with `INVALID_ARGUMENT` for a plan no reshard opened, or one opened with another executor.
    """
    return _block(executor.library, _SequenceReshardPlanFinishCall(plan.address), NULL_ADDRESS, executor.address)


def sequence_reshard_plan_finish_deferred(
    ref executor: Executor, var plan: SequenceReshardPlan
) -> Deferred[_SequenceReshardPlanFinishCall, ImmOrigin(origin_of(executor))]:
    """Submits `sequence_reshard_plan_finish` to `executor`, answering a `Deferred` at once.
    Calls `usearch_sequence_reshard_plan_finish`."""
    return {executor.library, _SequenceReshardPlanFinishCall(plan.address), NULL_ADDRESS, executor.address}


# endregion Reshard

# region Metadata


def _read_metadata(
    executor: Executor, verb: SequenceMetadataRead, source: BytesView[ImmUntrackedOrigin], detail: StaticString
) raises Error -> SequenceMetadata:
    """The path and buffer forms share one argument layout, differing only in the verb."""
    var described = stack_allocation[1, SequenceMetadata]()
    described[unsafe_offset=0] = SequenceMetadata()
    var outcome = stack_allocation[1, Outcome]()
    outcome[unsafe_offset=0] = Outcome()
    _cross(
        SequenceMetadataReadArguments(
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


def sequence_metadata_read[
    path_origin: ImmOrigin, //
](executor: Executor, path: StringSlice[path_origin]) raises Error -> SequenceMetadata:
    """Reads a checkpoint's kind and connectivity without loading it. Calls `usearch_sequence_metadata_read`.

    Parameters:
        path_origin: The path bytes' origin, inferred from the argument.

    Args:
        executor: The executor the read runs on; it also supplies its queues.
        path: The manifest of a sharded checkpoint, or a single monolith.

    Returns:
        The kind and connectivity the checkpoint froze.

    Raises:
        `Error` if the checkpoint cannot be opened or its header cannot be decoded.
    """
    return _read_metadata(
        executor,
        executor.library.symbols().sequence_metadata_read,
        BytesView.of(path).arguments(),
        "usearch_sequence_metadata_read",
    )


def sequence_metadata_read_buffer[
    buffer_origin: ImmOrigin, //
](executor: Executor, buffer: Span[UInt8, buffer_origin]) raises Error -> SequenceMetadata:
    """Reads a serialized monolith's kind and connectivity without loading it.
    Calls `usearch_sequence_metadata_read_buffer`.

    Parameters:
        buffer_origin: The source bytes' origin, inferred from the argument.

    Args:
        executor: The executor the read runs on; it also supplies its queues.
        buffer: Source holding one serialized monolith.

    Returns:
        The kind and connectivity the monolith froze.

    Raises:
        `Error` if the bytes cannot be decoded.
    """
    return _read_metadata(
        executor,
        executor.library.symbols().sequence_metadata_read_buffer,
        BytesView.of(buffer).arguments(),
        "usearch_sequence_metadata_read_buffer",
    )


# endregion Metadata
