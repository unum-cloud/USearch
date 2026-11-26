"""The loaded shared object and the one place a raw symbol address is ever touched.

The engine is reached by `dlopen` rather than by linking, which is what lets a consumer build
without a single linker flag. Mojo's `@extern` cannot serve here: it resolves at link time, so a
symbol from a library that is only opened at run time is an undefined reference under both the JIT
and `lld`, and using it would put a link flag in every consumer's build.

Every entry point is resolved once, when the library loads, into a typed `thin abi("C")` function
pointer. `OwnedDLHandle.get_function` would run `dlsym` on every call and parameterises only on the
return type, leaving arguments unchecked; a search on the hot path cannot afford either.

File: mojo/usearch/library.mojo
Author: Ash Vardanian
Date: September 7, 2026
"""

from std.ffi import OwnedDLHandle, __fn_type_is_cabi, c_int
from std.memory import ArcPointer, stack_allocation
from std.sys.info import CompilationTarget

from usearch.dense import (
    DenseAttend,
    DenseClear,
    DenseCount,
    DenseCursorInit,
    DenseCursorNext,
    DenseDescribe,
    DenseGet,
    DenseHealth,
    DenseInitRam,
    DenseInitVram,
    DenseInsert,
    DenseLoadRam,
    DenseLoadNvme,
    DenseLoadRamBuffer,
    DenseLoadVram,
    DenseLoadVramBuffer,
    DenseMaxsim,
    DenseMetadataRead,
    DenseMetadataReadBuffer,
    DenseMetric,
    DenseMultiply,
    DenseRefine,
    DenseRefinePlanFinish,
    DenseReserve,
    DenseReshard,
    DenseReshardPlanFinish,
    DenseSample,
    DenseSave,
    DenseSaveBuffer,
    DenseSearch,
    DenseSearchStreamInit,
    DenseSearchStreamNext,
    DenseViewNvme,
    DenseViewRam,
    Phase,
    RefineVerdict,
)
from usearch.geospatial import (
    GeospatialClear,
    GeospatialCount,
    GeospatialCursorInit,
    GeospatialCursorNext,
    GeospatialDescribe,
    GeospatialGet,
    GeospatialHealth,
    GeospatialInitRam,
    GeospatialInitVram,
    GeospatialInsert,
    GeospatialLoadRam,
    GeospatialLoadVram,
    GeospatialMetadataRead,
    GeospatialRefine,
    GeospatialRefinePlanFinish,
    GeospatialReserve,
    GeospatialReshard,
    GeospatialReshardPlanFinish,
    GeospatialSample,
    GeospatialSave,
    GeospatialSaveBuffer,
    GeospatialSearch,
    GeospatialViewNvme,
    GeospatialViewRam,
    GeospatialWithin,
)
from usearch.sequence import (
    SequenceClear,
    SequenceCount,
    SequenceCursorInit,
    SequenceCursorNext,
    SequenceDescribe,
    SequenceGet,
    SequenceHealth,
    SequenceInitRam,
    SequenceInitVram,
    SequenceInsert,
    SequenceLoadRam,
    SequenceLoadVram,
    SequenceMetadataRead,
    SequenceMetric,
    SequenceRefine,
    SequenceRefinePlanFinish,
    SequenceReserve,
    SequenceReshard,
    SequenceReshardPlanFinish,
    SequenceSample,
    SequenceSave,
    SequenceSaveBuffer,
    SequenceSearch,
    SequenceViewNvme,
    SequenceViewRam,
)
from usearch.shapes import (
    ShapesClear,
    ShapesCount,
    ShapesCursorInit,
    ShapesCursorNext,
    ShapesDescribe,
    ShapesGet,
    ShapesHealth,
    ShapesInitRam,
    ShapesInitVram,
    ShapesInsert,
    ShapesLoadRam,
    ShapesLoadVram,
    ShapesMetadataRead,
    ShapesMetric,
    ShapesRefine,
    ShapesRefinePlanFinish,
    ShapesReserve,
    ShapesReshard,
    ShapesReshardPlanFinish,
    ShapesSample,
    ShapesSave,
    ShapesSaveBuffer,
    ShapesSearch,
    ShapesViewNvme,
    ShapesViewRam,
)
from usearch.types import (
    ArenaDescribe,
    ArenaGeneration,
    ArenaInit,
    CPUClass,
    GPUClass,
    HostAccess,
    BytesView,
    CString,
    Error,
    ExecutorDrain,
    ExecutorForNvme,
    ExecutorForRam,
    ExecutorForVram,
    FromName,
    Mood,
    NameOf,
    Outcome,
    OutcomeStatus,
    Release,
    Residency,
    ScalarKind,
    Status,
    StatusRelease,
    Traversal,
    text_at,
)

comptime LIBRARY_FILE = StaticString(
    "libusearch_shared.dylib" if CompilationTarget.is_macos() else "libusearch_shared.so"
)
"""The loader already searches the environment's library directory, because `mojo build` bakes it
into the binary's runpath, so an unqualified name is enough for a library installed there."""

comptime BINDING_MAJOR = 3
"""The major version this binding was written against; a mismatch is refused at load."""

# region Signatures

comptime StaticText = def() thin abi("C") -> Optional[CString]
"""`usearch_version_name` and both capability getters: a static C string returned by value."""

comptime VersionOf = def(
    Pointer[c_int, MutAnyOrigin], Pointer[c_int, MutAnyOrigin], Pointer[c_int, MutAnyOrigin]
) thin abi("C") -> None
"""`usearch_version`: the linked library's major, minor and patch, written through three `int` pointers.

Typed pointers rather than `Address`: an `alloca` whose address crosses only as an integer argument
was measured reading back the value stored before the call."""

# endregion Signatures


@inline(.always)
def _resolve[
    Signature: TrivialRegisterPassable
](ref handle: OwnedDLHandle, name: StaticString) raises Error -> Signature:
    """One `dlsym`, checked once, typed from here on.

    `get_symbol` answers `Optional` where `get_function` aborts the process, which is what lets a
    stale library name the symbol it is missing instead of dying without a message.
    """
    comptime assert __fn_type_is_cabi[Signature](), 'the signature must hold abi("C")'
    if case .Some(address) = handle.get_symbol[NoneType](name):
        return Pointer(to=address).unsafe_bitcast[Signature]()[]
    raise Error(Status.SYMBOL_MISSING, name)


struct Symbols:
    """Every entry point the C ABI exports, resolved once when the library loads.

    Answers are never cached here, only addresses. An index re-asks the engine for its description
    on every call, because a concurrent insertion changes its size.
    """

    var version_name: StaticText
    """`usearch_version_name`."""
    var capabilities_compiled: StaticText
    """`usearch_capabilities_compiled`."""
    var capabilities_available: StaticText
    """`usearch_capabilities_available`."""

    var outcome_wait: OutcomeStatus
    """`usearch_outcome_wait`."""
    var outcome_status: OutcomeStatus
    """`usearch_outcome_status`."""
    var status_name: NameOf
    """`usearch_status_name`."""
    var status_from_name: FromName
    """`usearch_status_from_name`."""
    var scalar_name: NameOf
    """`usearch_scalar_name`."""
    var scalar_from_name: FromName
    """`usearch_scalar_from_name`."""
    var residency_name: NameOf
    """`usearch_residency_name`."""
    var residency_from_name: FromName
    """`usearch_residency_from_name`."""
    var host_access_name: NameOf
    """`usearch_host_access_name`."""
    var traversal_name: NameOf
    """`usearch_traversal_name`."""
    var traversal_from_name: FromName
    """`usearch_traversal_from_name`."""
    var mood_name: NameOf
    """`usearch_mood_name`."""
    var mood_from_name: FromName
    """`usearch_mood_from_name`."""
    var phase_name: NameOf
    """`usearch_phase_name`."""
    var phase_from_name: FromName
    """`usearch_phase_from_name`."""
    var refine_verdict_name: NameOf
    """`usearch_refine_verdict_name`."""
    var refine_verdict_from_name: FromName
    """`usearch_refine_verdict_from_name`."""
    var cpu_class_name: NameOf
    """`usearch_cpu_class_name`."""
    var gpu_class_name: NameOf
    """`usearch_gpu_class_name`."""
    var dense_metric_name: NameOf
    """`usearch_dense_metric_name`."""
    var dense_metric_from_name: FromName
    """`usearch_dense_metric_from_name`."""

    var arena_init: ArenaInit
    """`usearch_arena_init`."""
    var arena_free: StatusRelease
    """`usearch_arena_free`."""
    var arena_export_open: StatusRelease
    """`usearch_arena_export_open`."""
    var arena_export_close: Release
    """`usearch_arena_export_close`."""
    var arena_generation: ArenaGeneration
    """`usearch_arena_generation`."""
    var arena_describe: ArenaDescribe
    """`usearch_arena_describe`."""
    var executor_for_ram: ExecutorForRam
    """`usearch_executor_for_ram`."""
    var executor_for_nvme: ExecutorForNvme
    """`usearch_executor_for_nvme`."""
    var executor_for_vram: ExecutorForVram
    """`usearch_executor_for_vram`."""
    var executor_drain: ExecutorDrain
    """`usearch_executor_drain`."""
    var executor_free: StatusRelease
    """`usearch_executor_free`."""

    var dense_init_ram: DenseInitRam
    """`usearch_dense_init_ram`."""
    var dense_init_vram: DenseInitVram
    """`usearch_dense_init_vram`."""
    var dense_load_ram: DenseLoadRam
    """`usearch_dense_load_ram`."""
    var dense_load_vram: DenseLoadVram
    """`usearch_dense_load_vram`."""
    var dense_load_ram_buffer: DenseLoadRamBuffer
    """`usearch_dense_load_ram_buffer`."""
    var dense_load_vram_buffer: DenseLoadVramBuffer
    """`usearch_dense_load_vram_buffer`."""
    var dense_view_ram: DenseViewRam
    """`usearch_dense_view_ram`."""
    var dense_view_nvme: DenseViewNvme
    """`usearch_dense_view_nvme`."""
    var dense_load_nvme: DenseLoadNvme
    """`usearch_dense_load_nvme`."""
    var dense_free: Release
    """`usearch_dense_free`."""

    var dense_reserve: DenseReserve
    """`usearch_dense_reserve`."""
    var dense_clear: DenseClear
    """`usearch_dense_clear`."""
    var dense_describe: DenseDescribe
    """`usearch_dense_describe`."""
    var dense_insert: DenseInsert
    """`usearch_dense_insert`."""
    var dense_refine: DenseRefine
    """`usearch_dense_refine`."""
    var dense_refine_plan_finish: DenseRefinePlanFinish
    """`usearch_dense_refine_plan_finish`."""
    var dense_health: DenseHealth
    """`usearch_dense_health`."""
    var dense_reshard: DenseReshard
    """`usearch_dense_reshard`."""
    var dense_reshard_plan_finish: DenseReshardPlanFinish
    """`usearch_dense_reshard_plan_finish`."""
    var dense_sample: DenseSample
    """`usearch_dense_sample`."""
    var dense_search: DenseSearch
    """`usearch_dense_search`."""
    var dense_count: DenseCount
    """`usearch_dense_count`."""
    var dense_get: DenseGet
    """`usearch_dense_get`."""

    var dense_cursor_init: DenseCursorInit
    """`usearch_dense_cursor_init`."""
    var dense_cursor_next: DenseCursorNext
    """`usearch_dense_cursor_next`."""
    var dense_cursor_free: Release
    """`usearch_dense_cursor_free`."""
    var dense_search_stream_init: DenseSearchStreamInit
    """`usearch_dense_search_stream_init`."""
    var dense_search_stream_next: DenseSearchStreamNext
    """`usearch_dense_search_stream_next`."""
    var dense_search_stream_free: Release
    """`usearch_dense_search_stream_free`."""

    var dense_save: DenseSave
    """`usearch_dense_save`."""
    var dense_save_buffer: DenseSaveBuffer
    """`usearch_dense_save_buffer`."""
    var dense_metadata_read: DenseMetadataRead
    """`usearch_dense_metadata_read`."""
    var dense_metadata_read_buffer: DenseMetadataReadBuffer
    """`usearch_dense_metadata_read_buffer`."""

    var dense_multiply: DenseMultiply
    """`usearch_dense_multiply`."""
    var dense_attend: DenseAttend
    """`usearch_dense_attend`."""
    var dense_maxsim: DenseMaxsim
    """`usearch_dense_maxsim`."""

    var sequence_metric_name: NameOf
    """`usearch_sequence_metric_name`."""
    var sequence_metric_from_name: FromName
    """`usearch_sequence_metric_from_name`."""
    var sequence_init_ram: SequenceInitRam
    """`usearch_sequence_init_ram`."""
    var sequence_init_vram: SequenceInitVram
    """`usearch_sequence_init_vram`."""
    var sequence_load_ram: SequenceLoadRam
    """`usearch_sequence_load_ram`."""
    var sequence_load_ram_buffer: SequenceLoadRam
    """`usearch_sequence_load_ram_buffer`."""
    var sequence_load_vram: SequenceLoadVram
    """`usearch_sequence_load_vram`."""
    var sequence_load_vram_buffer: SequenceLoadVram
    """`usearch_sequence_load_vram_buffer`."""
    var sequence_view_ram: SequenceViewRam
    """`usearch_sequence_view_ram`."""
    var sequence_view_nvme: SequenceViewNvme
    """`usearch_sequence_view_nvme`."""
    var sequence_free: Release
    """`usearch_sequence_free`."""
    var sequence_reserve: SequenceReserve
    """`usearch_sequence_reserve`."""
    var sequence_clear: SequenceClear
    """`usearch_sequence_clear`."""
    var sequence_describe: SequenceDescribe
    """`usearch_sequence_describe`."""
    var sequence_insert: SequenceInsert
    """`usearch_sequence_insert`."""
    var sequence_refine: SequenceRefine
    """`usearch_sequence_refine`."""
    var sequence_refine_plan_finish: SequenceRefinePlanFinish
    """`usearch_sequence_refine_plan_finish`."""
    var sequence_health: SequenceHealth
    """`usearch_sequence_health`."""
    var sequence_reshard: SequenceReshard
    """`usearch_sequence_reshard`."""
    var sequence_reshard_plan_finish: SequenceReshardPlanFinish
    """`usearch_sequence_reshard_plan_finish`."""
    var sequence_sample: SequenceSample
    """`usearch_sequence_sample`."""
    var sequence_search: SequenceSearch
    """`usearch_sequence_search`."""
    var sequence_count: SequenceCount
    """`usearch_sequence_count`."""
    var sequence_get: SequenceGet
    """`usearch_sequence_get`."""
    var sequence_cursor_init: SequenceCursorInit
    """`usearch_sequence_cursor_init`."""
    var sequence_cursor_next: SequenceCursorNext
    """`usearch_sequence_cursor_next`."""
    var sequence_cursor_free: Release
    """`usearch_sequence_cursor_free`."""
    var sequence_save: SequenceSave
    """`usearch_sequence_save`."""
    var sequence_save_buffer: SequenceSaveBuffer
    """`usearch_sequence_save_buffer`."""
    var sequence_metadata_read: SequenceMetadataRead
    """`usearch_sequence_metadata_read`."""
    var sequence_metadata_read_buffer: SequenceMetadataRead
    """`usearch_sequence_metadata_read_buffer`."""

    var shapes_metric_name: NameOf
    """`usearch_shapes_metric_name`."""
    var shapes_metric_from_name: FromName
    """`usearch_shapes_metric_from_name`."""
    var shapes_init_ram: ShapesInitRam
    """`usearch_shapes_init_ram`."""
    var shapes_init_vram: ShapesInitVram
    """`usearch_shapes_init_vram`."""
    var shapes_load_ram: ShapesLoadRam
    """`usearch_shapes_load_ram`."""
    var shapes_load_ram_buffer: ShapesLoadRam
    """`usearch_shapes_load_ram_buffer`."""
    var shapes_load_vram: ShapesLoadVram
    """`usearch_shapes_load_vram`."""
    var shapes_load_vram_buffer: ShapesLoadVram
    """`usearch_shapes_load_vram_buffer`."""
    var shapes_view_ram: ShapesViewRam
    """`usearch_shapes_view_ram`."""
    var shapes_view_nvme: ShapesViewNvme
    """`usearch_shapes_view_nvme`."""
    var shapes_free: Release
    """`usearch_shapes_free`."""
    var shapes_reserve: ShapesReserve
    """`usearch_shapes_reserve`."""
    var shapes_clear: ShapesClear
    """`usearch_shapes_clear`."""
    var shapes_describe: ShapesDescribe
    """`usearch_shapes_describe`."""
    var shapes_insert: ShapesInsert
    """`usearch_shapes_insert`."""
    var shapes_refine: ShapesRefine
    """`usearch_shapes_refine`."""
    var shapes_refine_plan_finish: ShapesRefinePlanFinish
    """`usearch_shapes_refine_plan_finish`."""
    var shapes_health: ShapesHealth
    """`usearch_shapes_health`."""
    var shapes_reshard: ShapesReshard
    """`usearch_shapes_reshard`."""
    var shapes_reshard_plan_finish: ShapesReshardPlanFinish
    """`usearch_shapes_reshard_plan_finish`."""
    var shapes_sample: ShapesSample
    """`usearch_shapes_sample`."""
    var shapes_search: ShapesSearch
    """`usearch_shapes_search`."""
    var shapes_count: ShapesCount
    """`usearch_shapes_count`."""
    var shapes_get: ShapesGet
    """`usearch_shapes_get`."""
    var shapes_cursor_init: ShapesCursorInit
    """`usearch_shapes_cursor_init`."""
    var shapes_cursor_next: ShapesCursorNext
    """`usearch_shapes_cursor_next`."""
    var shapes_cursor_free: Release
    """`usearch_shapes_cursor_free`."""
    var shapes_save: ShapesSave
    """`usearch_shapes_save`."""
    var shapes_save_buffer: ShapesSaveBuffer
    """`usearch_shapes_save_buffer`."""
    var shapes_metadata_read: ShapesMetadataRead
    """`usearch_shapes_metadata_read`."""
    var shapes_metadata_read_buffer: ShapesMetadataRead
    """`usearch_shapes_metadata_read_buffer`."""

    var geospatial_init_ram: GeospatialInitRam
    """`usearch_geospatial_init_ram`."""
    var geospatial_init_vram: GeospatialInitVram
    """`usearch_geospatial_init_vram`."""
    var geospatial_load_ram: GeospatialLoadRam
    """`usearch_geospatial_load_ram`."""
    var geospatial_load_ram_buffer: GeospatialLoadRam
    """`usearch_geospatial_load_ram_buffer`."""
    var geospatial_load_vram: GeospatialLoadVram
    """`usearch_geospatial_load_vram`."""
    var geospatial_load_vram_buffer: GeospatialLoadVram
    """`usearch_geospatial_load_vram_buffer`."""
    var geospatial_view_ram: GeospatialViewRam
    """`usearch_geospatial_view_ram`."""
    var geospatial_view_nvme: GeospatialViewNvme
    """`usearch_geospatial_view_nvme`."""
    var geospatial_free: Release
    """`usearch_geospatial_free`."""
    var geospatial_reserve: GeospatialReserve
    """`usearch_geospatial_reserve`."""
    var geospatial_clear: GeospatialClear
    """`usearch_geospatial_clear`."""
    var geospatial_describe: GeospatialDescribe
    """`usearch_geospatial_describe`."""
    var geospatial_insert: GeospatialInsert
    """`usearch_geospatial_insert`."""
    var geospatial_refine: GeospatialRefine
    """`usearch_geospatial_refine`."""
    var geospatial_refine_plan_finish: GeospatialRefinePlanFinish
    """`usearch_geospatial_refine_plan_finish`."""
    var geospatial_health: GeospatialHealth
    """`usearch_geospatial_health`."""
    var geospatial_reshard: GeospatialReshard
    """`usearch_geospatial_reshard`."""
    var geospatial_reshard_plan_finish: GeospatialReshardPlanFinish
    """`usearch_geospatial_reshard_plan_finish`."""
    var geospatial_sample: GeospatialSample
    """`usearch_geospatial_sample`."""
    var geospatial_search: GeospatialSearch
    """`usearch_geospatial_search`."""
    var geospatial_within: GeospatialWithin
    """`usearch_geospatial_within`."""
    var geospatial_count: GeospatialCount
    """`usearch_geospatial_count`."""
    var geospatial_get: GeospatialGet
    """`usearch_geospatial_get`."""
    var geospatial_cursor_init: GeospatialCursorInit
    """`usearch_geospatial_cursor_init`."""
    var geospatial_cursor_next: GeospatialCursorNext
    """`usearch_geospatial_cursor_next`."""
    var geospatial_cursor_free: Release
    """`usearch_geospatial_cursor_free`."""
    var geospatial_save: GeospatialSave
    """`usearch_geospatial_save`."""
    var geospatial_save_buffer: GeospatialSaveBuffer
    """`usearch_geospatial_save_buffer`."""
    var geospatial_metadata_read: GeospatialMetadataRead
    """`usearch_geospatial_metadata_read`."""
    var geospatial_metadata_read_buffer: GeospatialMetadataRead
    """`usearch_geospatial_metadata_read_buffer`."""

    def __init__(out self, ref handle: OwnedDLHandle) raises Error:
        """Resolves every entry point at once, so no call ever pays for a `dlsym`.

        Args:
            handle: The opened library, viewed only while the addresses are read out.

        Raises:
            With `SYMBOL_MISSING`, on the first symbol the loaded library does not export.
        """
        self.version_name = _resolve[StaticText](handle, "usearch_version_name")
        self.capabilities_compiled = _resolve[StaticText](handle, "usearch_capabilities_compiled")
        self.capabilities_available = _resolve[StaticText](handle, "usearch_capabilities_available")

        self.outcome_wait = _resolve[OutcomeStatus](handle, "usearch_outcome_wait")
        self.outcome_status = _resolve[OutcomeStatus](handle, "usearch_outcome_status")
        self.status_name = _resolve[NameOf](handle, "usearch_status_name")
        self.status_from_name = _resolve[FromName](handle, "usearch_status_from_name")
        self.scalar_name = _resolve[NameOf](handle, "usearch_scalar_name")
        self.scalar_from_name = _resolve[FromName](handle, "usearch_scalar_from_name")
        self.residency_name = _resolve[NameOf](handle, "usearch_residency_name")
        self.residency_from_name = _resolve[FromName](handle, "usearch_residency_from_name")
        self.host_access_name = _resolve[NameOf](handle, "usearch_host_access_name")
        self.traversal_name = _resolve[NameOf](handle, "usearch_traversal_name")
        self.traversal_from_name = _resolve[FromName](handle, "usearch_traversal_from_name")
        self.mood_name = _resolve[NameOf](handle, "usearch_mood_name")
        self.mood_from_name = _resolve[FromName](handle, "usearch_mood_from_name")
        self.phase_name = _resolve[NameOf](handle, "usearch_phase_name")
        self.phase_from_name = _resolve[FromName](handle, "usearch_phase_from_name")
        self.refine_verdict_name = _resolve[NameOf](handle, "usearch_refine_verdict_name")
        self.refine_verdict_from_name = _resolve[FromName](handle, "usearch_refine_verdict_from_name")
        self.cpu_class_name = _resolve[NameOf](handle, "usearch_cpu_class_name")
        self.gpu_class_name = _resolve[NameOf](handle, "usearch_gpu_class_name")
        self.dense_metric_name = _resolve[NameOf](handle, "usearch_dense_metric_name")
        self.dense_metric_from_name = _resolve[FromName](handle, "usearch_dense_metric_from_name")

        self.arena_init = _resolve[ArenaInit](handle, "usearch_arena_init")
        self.arena_free = _resolve[StatusRelease](handle, "usearch_arena_free")
        self.arena_export_open = _resolve[StatusRelease](handle, "usearch_arena_export_open")
        self.arena_export_close = _resolve[Release](handle, "usearch_arena_export_close")
        self.arena_generation = _resolve[ArenaGeneration](handle, "usearch_arena_generation")
        self.arena_describe = _resolve[ArenaDescribe](handle, "usearch_arena_describe")
        self.executor_for_ram = _resolve[ExecutorForRam](handle, "usearch_executor_for_ram")
        self.executor_for_nvme = _resolve[ExecutorForNvme](handle, "usearch_executor_for_nvme")
        self.executor_for_vram = _resolve[ExecutorForVram](handle, "usearch_executor_for_vram")
        self.executor_drain = _resolve[ExecutorDrain](handle, "usearch_executor_drain")
        self.executor_free = _resolve[StatusRelease](handle, "usearch_executor_free")

        self.dense_init_ram = _resolve[DenseInitRam](handle, "usearch_dense_init_ram")
        self.dense_init_vram = _resolve[DenseInitVram](handle, "usearch_dense_init_vram")
        self.dense_load_ram = _resolve[DenseLoadRam](handle, "usearch_dense_load_ram")
        self.dense_load_vram = _resolve[DenseLoadVram](handle, "usearch_dense_load_vram")
        self.dense_load_ram_buffer = _resolve[DenseLoadRamBuffer](handle, "usearch_dense_load_ram_buffer")
        self.dense_load_vram_buffer = _resolve[DenseLoadVramBuffer](handle, "usearch_dense_load_vram_buffer")
        self.dense_view_ram = _resolve[DenseViewRam](handle, "usearch_dense_view_ram")
        self.dense_view_nvme = _resolve[DenseViewNvme](handle, "usearch_dense_view_nvme")
        self.dense_load_nvme = _resolve[DenseLoadNvme](handle, "usearch_dense_load_nvme")
        self.dense_free = _resolve[Release](handle, "usearch_dense_free")

        self.dense_reserve = _resolve[DenseReserve](handle, "usearch_dense_reserve")
        self.dense_clear = _resolve[DenseClear](handle, "usearch_dense_clear")
        self.dense_describe = _resolve[DenseDescribe](handle, "usearch_dense_describe")
        self.dense_insert = _resolve[DenseInsert](handle, "usearch_dense_insert")
        self.dense_refine = _resolve[DenseRefine](handle, "usearch_dense_refine")
        self.dense_refine_plan_finish = _resolve[DenseRefinePlanFinish](handle, "usearch_dense_refine_plan_finish")
        self.dense_health = _resolve[DenseHealth](handle, "usearch_dense_health")
        self.dense_reshard = _resolve[DenseReshard](handle, "usearch_dense_reshard")
        self.dense_reshard_plan_finish = _resolve[DenseReshardPlanFinish](handle, "usearch_dense_reshard_plan_finish")
        self.dense_sample = _resolve[DenseSample](handle, "usearch_dense_sample")
        self.dense_search = _resolve[DenseSearch](handle, "usearch_dense_search")
        self.dense_count = _resolve[DenseCount](handle, "usearch_dense_count")
        self.dense_get = _resolve[DenseGet](handle, "usearch_dense_get")

        self.dense_cursor_init = _resolve[DenseCursorInit](handle, "usearch_dense_cursor_init")
        self.dense_cursor_next = _resolve[DenseCursorNext](handle, "usearch_dense_cursor_next")
        self.dense_cursor_free = _resolve[Release](handle, "usearch_dense_cursor_free")
        self.dense_search_stream_init = _resolve[DenseSearchStreamInit](handle, "usearch_dense_search_stream_init")
        self.dense_search_stream_next = _resolve[DenseSearchStreamNext](handle, "usearch_dense_search_stream_next")
        self.dense_search_stream_free = _resolve[Release](handle, "usearch_dense_search_stream_free")

        self.dense_save = _resolve[DenseSave](handle, "usearch_dense_save")
        self.dense_save_buffer = _resolve[DenseSaveBuffer](handle, "usearch_dense_save_buffer")
        self.dense_metadata_read = _resolve[DenseMetadataRead](handle, "usearch_dense_metadata_read")
        self.dense_metadata_read_buffer = _resolve[DenseMetadataReadBuffer](
            handle, "usearch_dense_metadata_read_buffer"
        )

        self.dense_multiply = _resolve[DenseMultiply](handle, "usearch_dense_multiply")
        self.dense_attend = _resolve[DenseAttend](handle, "usearch_dense_attend")
        self.dense_maxsim = _resolve[DenseMaxsim](handle, "usearch_dense_maxsim")

        self.sequence_metric_name = _resolve[NameOf](handle, "usearch_sequence_metric_name")
        self.sequence_metric_from_name = _resolve[FromName](handle, "usearch_sequence_metric_from_name")
        self.sequence_init_ram = _resolve[SequenceInitRam](handle, "usearch_sequence_init_ram")
        self.sequence_init_vram = _resolve[SequenceInitVram](handle, "usearch_sequence_init_vram")
        self.sequence_load_ram = _resolve[SequenceLoadRam](handle, "usearch_sequence_load_ram")
        self.sequence_load_ram_buffer = _resolve[SequenceLoadRam](handle, "usearch_sequence_load_ram_buffer")
        self.sequence_load_vram = _resolve[SequenceLoadVram](handle, "usearch_sequence_load_vram")
        self.sequence_load_vram_buffer = _resolve[SequenceLoadVram](handle, "usearch_sequence_load_vram_buffer")
        self.sequence_view_ram = _resolve[SequenceViewRam](handle, "usearch_sequence_view_ram")
        self.sequence_view_nvme = _resolve[SequenceViewNvme](handle, "usearch_sequence_view_nvme")
        self.sequence_free = _resolve[Release](handle, "usearch_sequence_free")
        self.sequence_reserve = _resolve[SequenceReserve](handle, "usearch_sequence_reserve")
        self.sequence_clear = _resolve[SequenceClear](handle, "usearch_sequence_clear")
        self.sequence_describe = _resolve[SequenceDescribe](handle, "usearch_sequence_describe")
        self.sequence_insert = _resolve[SequenceInsert](handle, "usearch_sequence_insert")
        self.sequence_refine = _resolve[SequenceRefine](handle, "usearch_sequence_refine")
        self.sequence_refine_plan_finish = _resolve[SequenceRefinePlanFinish](
            handle, "usearch_sequence_refine_plan_finish"
        )
        self.sequence_health = _resolve[SequenceHealth](handle, "usearch_sequence_health")
        self.sequence_reshard = _resolve[SequenceReshard](handle, "usearch_sequence_reshard")
        self.sequence_reshard_plan_finish = _resolve[SequenceReshardPlanFinish](
            handle, "usearch_sequence_reshard_plan_finish"
        )
        self.sequence_sample = _resolve[SequenceSample](handle, "usearch_sequence_sample")
        self.sequence_search = _resolve[SequenceSearch](handle, "usearch_sequence_search")
        self.sequence_count = _resolve[SequenceCount](handle, "usearch_sequence_count")
        self.sequence_get = _resolve[SequenceGet](handle, "usearch_sequence_get")
        self.sequence_cursor_init = _resolve[SequenceCursorInit](handle, "usearch_sequence_cursor_init")
        self.sequence_cursor_next = _resolve[SequenceCursorNext](handle, "usearch_sequence_cursor_next")
        self.sequence_cursor_free = _resolve[Release](handle, "usearch_sequence_cursor_free")
        self.sequence_save = _resolve[SequenceSave](handle, "usearch_sequence_save")
        self.sequence_save_buffer = _resolve[SequenceSaveBuffer](handle, "usearch_sequence_save_buffer")
        self.sequence_metadata_read = _resolve[SequenceMetadataRead](handle, "usearch_sequence_metadata_read")
        self.sequence_metadata_read_buffer = _resolve[SequenceMetadataRead](
            handle, "usearch_sequence_metadata_read_buffer"
        )

        self.shapes_metric_name = _resolve[NameOf](handle, "usearch_shapes_metric_name")
        self.shapes_metric_from_name = _resolve[FromName](handle, "usearch_shapes_metric_from_name")
        self.shapes_init_ram = _resolve[ShapesInitRam](handle, "usearch_shapes_init_ram")
        self.shapes_init_vram = _resolve[ShapesInitVram](handle, "usearch_shapes_init_vram")
        self.shapes_load_ram = _resolve[ShapesLoadRam](handle, "usearch_shapes_load_ram")
        self.shapes_load_ram_buffer = _resolve[ShapesLoadRam](handle, "usearch_shapes_load_ram_buffer")
        self.shapes_load_vram = _resolve[ShapesLoadVram](handle, "usearch_shapes_load_vram")
        self.shapes_load_vram_buffer = _resolve[ShapesLoadVram](handle, "usearch_shapes_load_vram_buffer")
        self.shapes_view_ram = _resolve[ShapesViewRam](handle, "usearch_shapes_view_ram")
        self.shapes_view_nvme = _resolve[ShapesViewNvme](handle, "usearch_shapes_view_nvme")
        self.shapes_free = _resolve[Release](handle, "usearch_shapes_free")
        self.shapes_reserve = _resolve[ShapesReserve](handle, "usearch_shapes_reserve")
        self.shapes_clear = _resolve[ShapesClear](handle, "usearch_shapes_clear")
        self.shapes_describe = _resolve[ShapesDescribe](handle, "usearch_shapes_describe")
        self.shapes_insert = _resolve[ShapesInsert](handle, "usearch_shapes_insert")
        self.shapes_refine = _resolve[ShapesRefine](handle, "usearch_shapes_refine")
        self.shapes_refine_plan_finish = _resolve[ShapesRefinePlanFinish](handle, "usearch_shapes_refine_plan_finish")
        self.shapes_health = _resolve[ShapesHealth](handle, "usearch_shapes_health")
        self.shapes_reshard = _resolve[ShapesReshard](handle, "usearch_shapes_reshard")
        self.shapes_reshard_plan_finish = _resolve[ShapesReshardPlanFinish](
            handle, "usearch_shapes_reshard_plan_finish"
        )
        self.shapes_sample = _resolve[ShapesSample](handle, "usearch_shapes_sample")
        self.shapes_search = _resolve[ShapesSearch](handle, "usearch_shapes_search")
        self.shapes_count = _resolve[ShapesCount](handle, "usearch_shapes_count")
        self.shapes_get = _resolve[ShapesGet](handle, "usearch_shapes_get")
        self.shapes_cursor_init = _resolve[ShapesCursorInit](handle, "usearch_shapes_cursor_init")
        self.shapes_cursor_next = _resolve[ShapesCursorNext](handle, "usearch_shapes_cursor_next")
        self.shapes_cursor_free = _resolve[Release](handle, "usearch_shapes_cursor_free")
        self.shapes_save = _resolve[ShapesSave](handle, "usearch_shapes_save")
        self.shapes_save_buffer = _resolve[ShapesSaveBuffer](handle, "usearch_shapes_save_buffer")
        self.shapes_metadata_read = _resolve[ShapesMetadataRead](handle, "usearch_shapes_metadata_read")
        self.shapes_metadata_read_buffer = _resolve[ShapesMetadataRead](handle, "usearch_shapes_metadata_read_buffer")

        self.geospatial_init_ram = _resolve[GeospatialInitRam](handle, "usearch_geospatial_init_ram")
        self.geospatial_init_vram = _resolve[GeospatialInitVram](handle, "usearch_geospatial_init_vram")
        self.geospatial_load_ram = _resolve[GeospatialLoadRam](handle, "usearch_geospatial_load_ram")
        self.geospatial_load_ram_buffer = _resolve[GeospatialLoadRam](handle, "usearch_geospatial_load_ram_buffer")
        self.geospatial_load_vram = _resolve[GeospatialLoadVram](handle, "usearch_geospatial_load_vram")
        self.geospatial_load_vram_buffer = _resolve[GeospatialLoadVram](handle, "usearch_geospatial_load_vram_buffer")
        self.geospatial_view_ram = _resolve[GeospatialViewRam](handle, "usearch_geospatial_view_ram")
        self.geospatial_view_nvme = _resolve[GeospatialViewNvme](handle, "usearch_geospatial_view_nvme")
        self.geospatial_free = _resolve[Release](handle, "usearch_geospatial_free")
        self.geospatial_reserve = _resolve[GeospatialReserve](handle, "usearch_geospatial_reserve")
        self.geospatial_clear = _resolve[GeospatialClear](handle, "usearch_geospatial_clear")
        self.geospatial_describe = _resolve[GeospatialDescribe](handle, "usearch_geospatial_describe")
        self.geospatial_insert = _resolve[GeospatialInsert](handle, "usearch_geospatial_insert")
        self.geospatial_refine = _resolve[GeospatialRefine](handle, "usearch_geospatial_refine")
        self.geospatial_refine_plan_finish = _resolve[GeospatialRefinePlanFinish](
            handle, "usearch_geospatial_refine_plan_finish"
        )
        self.geospatial_health = _resolve[GeospatialHealth](handle, "usearch_geospatial_health")
        self.geospatial_reshard = _resolve[GeospatialReshard](handle, "usearch_geospatial_reshard")
        self.geospatial_reshard_plan_finish = _resolve[GeospatialReshardPlanFinish](
            handle, "usearch_geospatial_reshard_plan_finish"
        )
        self.geospatial_sample = _resolve[GeospatialSample](handle, "usearch_geospatial_sample")
        self.geospatial_search = _resolve[GeospatialSearch](handle, "usearch_geospatial_search")
        self.geospatial_within = _resolve[GeospatialWithin](handle, "usearch_geospatial_within")
        self.geospatial_count = _resolve[GeospatialCount](handle, "usearch_geospatial_count")
        self.geospatial_get = _resolve[GeospatialGet](handle, "usearch_geospatial_get")
        self.geospatial_cursor_init = _resolve[GeospatialCursorInit](handle, "usearch_geospatial_cursor_init")
        self.geospatial_cursor_next = _resolve[GeospatialCursorNext](handle, "usearch_geospatial_cursor_next")
        self.geospatial_cursor_free = _resolve[Release](handle, "usearch_geospatial_cursor_free")
        self.geospatial_save = _resolve[GeospatialSave](handle, "usearch_geospatial_save")
        self.geospatial_save_buffer = _resolve[GeospatialSaveBuffer](handle, "usearch_geospatial_save_buffer")
        self.geospatial_metadata_read = _resolve[GeospatialMetadataRead](handle, "usearch_geospatial_metadata_read")
        self.geospatial_metadata_read_buffer = _resolve[GeospatialMetadataRead](
            handle, "usearch_geospatial_metadata_read_buffer"
        )


struct _Loaded:
    """The handle and its symbols, kept together so neither outlives the other."""

    var handle: OwnedDLHandle
    var symbols: Symbols

    def __init__(out self) raises Error:
        try:
            self.handle = OwnedDLHandle(String(LIBRARY_FILE))
        except:
            raise Error(Status.LIBRARY_MISSING, LIBRARY_FILE)
        var version = _resolve[VersionOf](self.handle, "usearch_version")
        var parts = stack_allocation[3, c_int]().unsafe_origin_cast[MutAnyOrigin]()
        for part in range(3):
            parts[unsafe_offset=part] = c_int(0)
        version(parts, parts.unsafe_offset(1), parts.unsafe_offset(2))
        if Int(parts[unsafe_offset=0]) != BINDING_MAJOR:
            raise Error(Status.VERSION_MISMATCH, "usearch_version")
        self.symbols = Symbols(self.handle)


struct Library(ImplicitlyCopyable):
    """A reference to the loaded engine, cheap to copy and shared by everything it creates.

    Every handle holds one, which is what keeps the shared object mapped for as long as anything
    can still call into it. It is a loader reference and nothing more: an executor never retains an
    index, and an index never retains an executor.
    """

    var shared: ArcPointer[_Loaded]
    """The handle and its symbols, refcounted so the last holder unmaps the object."""

    def __init__(out self) raises Error:
        """Opens the shared object and resolves it, which every handle then shares.

        Raises:
            With `LIBRARY_MISSING`, when nothing is on the loader path, with `VERSION_MISMATCH`
            when what loaded is another major version, or with `SYMBOL_MISSING` when it does not
            export an entry point the binding needs.
        """
        self.shared = ArcPointer(_Loaded())

    @inline(.always)
    def symbols(self) -> ref[origin_of(self.shared[].symbols)] Symbols:
        """The resolved entry points, viewed rather than copied on every call.

        Returns:
            A reference that stays valid for as long as this `Library` is held.
        """
        return self.shared[].symbols

    def version_name(self) -> String:
        """The loaded engine's own version, which is also the cheapest check that loading worked.
        Calls `usearch_version_name`.

        Returns:
            The version string the shared object reports for itself.
        """
        return text_at(self.symbols().version_name())

    def capabilities_compiled(self) -> String:
        """The hardware capabilities this build of the engine was compiled with.
        Calls `usearch_capabilities_compiled`.

        Returns:
            A comma-separated list of capability names.
        """
        return text_at(self.symbols().capabilities_compiled())

    def capabilities_available(self) -> String:
        """The hardware capabilities this machine offers now, which the build does not imply.
        Calls `usearch_capabilities_available`.

        Returns:
            A comma-separated list of capability names.
        """
        return text_at(self.symbols().capabilities_available())

    def status_name(self, status: Status) -> String:
        """The engine's spelling of a status, "unrecognized" for a code it does not name. Calls `usearch_status_name`.
        """
        return text_at(self.symbols().status_name(c_int(status.code)))

    def status_from_name(self, name: StringSlice) raises Error -> Status:
        """Reads a spelling back, raising `INVALID_ARGUMENT` for one it refuses. Calls `usearch_status_from_name`."""
        return Status.of(self._read_named(self.symbols().status_from_name, name, "usearch_status_from_name"))

    def scalar_name(self, scalar: ScalarKind) -> String:
        """The engine's spelling of a scalar kind, as `F32` spells "f32". Calls `usearch_scalar_name`."""
        return text_at(self.symbols().scalar_name(scalar.code))

    def scalar_from_name(self, name: StringSlice) raises Error -> ScalarKind:
        """Reads a spelling back, raising `INVALID_ARGUMENT` for one it refuses. Calls `usearch_scalar_from_name`."""
        return ScalarKind(self._read_named(self.symbols().scalar_from_name, name, "usearch_scalar_from_name"))

    def residency_name(self, residency: Residency) -> String:
        """The engine's spelling of a residency, as `RAM` spells "ram". Calls `usearch_residency_name`."""
        return text_at(self.symbols().residency_name(residency.code))

    def residency_from_name(self, name: StringSlice) raises Error -> Residency:
        """Reads a spelling back, raising `INVALID_ARGUMENT` for one it refuses. Calls `usearch_residency_from_name`."""
        return Residency(self._read_named(self.symbols().residency_from_name, name, "usearch_residency_from_name"))

    def host_access_name(self, host_access: HostAccess) -> String:
        """The engine's spelling of a host access, as `READABLE` spells "readable". Calls `usearch_host_access_name`."""
        return text_at(self.symbols().host_access_name(host_access.code))

    def traversal_name(self, traversal: Traversal) -> String:
        """The engine's spelling of a traversal, as `SCAN` spells "scan". Calls `usearch_traversal_name`."""
        return text_at(self.symbols().traversal_name(traversal.code))

    def traversal_from_name(self, name: StringSlice) raises Error -> Traversal:
        """Reads a spelling back, raising `INVALID_ARGUMENT` for one it refuses. Calls `usearch_traversal_from_name`."""
        return Traversal(self._read_named(self.symbols().traversal_from_name, name, "usearch_traversal_from_name"))

    def cpu_class_name(self, cpu_class: CPUClass) -> String:
        """The engine's spelling of a CPU class, as dispatch reports it. Calls `usearch_cpu_class_name`."""
        return text_at(self.symbols().cpu_class_name(cpu_class.code))

    def gpu_class_name(self, gpu_class: GPUClass) -> String:
        """The engine's spelling of a GPU class, as dispatch reports it. Calls `usearch_gpu_class_name`."""
        return text_at(self.symbols().gpu_class_name(gpu_class.code))

    def mood_name(self, mood: Mood) -> String:
        """The engine's spelling of a mood, as `PARKED` spells "parked". Calls `usearch_mood_name`."""
        return text_at(self.symbols().mood_name(mood.code))

    def mood_from_name(self, name: StringSlice) raises Error -> Mood:
        """Reads a spelling back, raising `INVALID_ARGUMENT` for one it refuses. Calls `usearch_mood_from_name`."""
        return Mood(self._read_named(self.symbols().mood_from_name, name, "usearch_mood_from_name"))

    def phase_name(self, phase: Phase) -> String:
        """The engine's spelling of a plan's phase, as `RESTITCH` spells "restitch". Calls `usearch_phase_name`."""
        return text_at(self.symbols().phase_name(phase.code))

    def phase_from_name(self, name: StringSlice) raises Error -> Phase:
        """Reads a spelling back, raising `INVALID_ARGUMENT` for one it refuses. Calls `usearch_phase_from_name`."""
        return Phase(self._read_named(self.symbols().phase_from_name, name, "usearch_phase_from_name"))

    def refine_verdict_name(self, verdict: RefineVerdict) -> String:
        """The engine's spelling of a refine verdict, as `STALLED` spells "stalled". Calls
        `usearch_refine_verdict_name`."""
        return text_at(self.symbols().refine_verdict_name(verdict.code))

    def refine_verdict_from_name(self, name: StringSlice) raises Error -> RefineVerdict:
        """Reads a spelling back, raising `INVALID_ARGUMENT` for one it refuses. Calls
        `usearch_refine_verdict_from_name`."""
        return RefineVerdict(
            self._read_named(self.symbols().refine_verdict_from_name, name, "usearch_refine_verdict_from_name")
        )

    def _read_named(self, parse: FromName, name: StringSlice, symbol: StaticString) raises Error -> c_int:
        """The enumerator a status-answering reader writes for `name`, or `INVALID_ARGUMENT` when it refuses."""
        var code = stack_allocation[1, c_int]().unsafe_origin_cast[MutAnyOrigin]()
        code[unsafe_offset=0] = c_int(0)
        if parse(BytesView.of(name).arguments(), code) != c_int(0):
            raise Error(Status.INVALID_ARGUMENT, symbol)
        return code[unsafe_offset=0]

    def dense_metric_name(self, metric: DenseMetric) -> String:
        """The engine's spelling of a metric kind, as `INNER` spells "inner". Calls `usearch_dense_metric_name`."""
        return text_at(self.symbols().dense_metric_name(metric.code))

    def dense_metric_from_name(self, name: StringSlice) raises Error -> DenseMetric:
        """Reads a spelling back, raising `INVALID_ARGUMENT` for one it refuses. Calls
        `usearch_dense_metric_from_name`."""
        return DenseMetric(
            self._read_named(self.symbols().dense_metric_from_name, name, "usearch_dense_metric_from_name")
        )

    def sequence_metric_name(self, metric: SequenceMetric) -> String:
        """The engine's spelling of a sequence metric, as `PROTEIN_ALIGNMENT` spells "protein-alignment". Calls
        `usearch_sequence_metric_name`."""
        return text_at(self.symbols().sequence_metric_name(metric.code))

    def sequence_metric_from_name(self, name: StringSlice) raises Error -> SequenceMetric:
        """Reads a spelling back, raising `INVALID_ARGUMENT` for one it refuses. Calls
        `usearch_sequence_metric_from_name`."""
        return SequenceMetric(
            self._read_named(self.symbols().sequence_metric_from_name, name, "usearch_sequence_metric_from_name")
        )

    def shapes_metric_name(self, metric: ShapesMetric) -> String:
        """The engine's spelling of a shapes kind, as `CHAMFER` spells "chamfer". Calls
        `usearch_shapes_metric_name`."""
        return text_at(self.symbols().shapes_metric_name(metric.code))

    def shapes_metric_from_name(self, name: StringSlice) raises Error -> ShapesMetric:
        """Reads a spelling back, raising `INVALID_ARGUMENT` for one it refuses. Calls
        `usearch_shapes_metric_from_name`."""
        return ShapesMetric(
            self._read_named(self.symbols().shapes_metric_from_name, name, "usearch_shapes_metric_from_name")
        )

    def _error(self, outcome: Outcome, symbol: StaticString) -> Error:
        """The failure a final outcome reports, its status spelled by the engine's own table."""
        var status = Status.of(outcome.status)
        return Error(status, symbol, outcome.message, self.symbols().status_name(c_int(status.code)), self)

    def _check(self, outcome: Outcome, symbol: StaticString) raises Error:
        """Raises the failure a final outcome reports, and returns on success."""
        if outcome.status != c_int(0):
            raise self._error(outcome, symbol)
