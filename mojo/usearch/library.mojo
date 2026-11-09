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
    DenseAbsorb,
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
    DenseLoadRamBuffer,
    DenseLoadVram,
    DenseLoadVramBuffer,
    DenseMaxsim,
    DenseMetadataRead,
    DenseMetadataReadBuffer,
    DenseMetric,
    DenseMultiply,
    DenseRefine,
    DenseReserve,
    DenseSample,
    DenseSave,
    DenseSaveBuffer,
    DenseSearch,
    DenseSearchStreamInit,
    DenseSearchStreamNext,
    DenseSplit,
    DenseViewNvme,
    DenseViewRam,
)
from usearch.types import (
    ArenaDescribe,
    ArenaInit,
    CPUClass,
    GPUClass,
    BytesView,
    CString,
    Error,
    ExecutorDrain,
    ExecutorForNvme,
    ExecutorForRam,
    ExecutorForVram,
    ExecutorOnPool,
    FromName,
    FromNameStatus,
    Mood,
    NameOf,
    Outcome,
    OutcomeStatus,
    Release,
    Residency,
    ScalarKind,
    Status,
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
    comptime assert __fn_type_is_cabi[Signature](), 'the signature must carry abi("C")'
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
    var status_from_name: FromNameStatus
    """`usearch_status_from_name`."""
    var scalar_name: NameOf
    """`usearch_scalar_name`."""
    var scalar_from_name: FromName
    """`usearch_scalar_from_name`."""
    var residency_name: NameOf
    """`usearch_residency_name`."""
    var residency_from_name: FromNameStatus
    """`usearch_residency_from_name`."""
    var traversal_name: NameOf
    """`usearch_traversal_name`."""
    var traversal_from_name: FromNameStatus
    """`usearch_traversal_from_name`."""
    var mood_name: NameOf
    """`usearch_mood_name`."""
    var mood_from_name: FromNameStatus
    """`usearch_mood_from_name`."""
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
    var arena_free: Release
    """`usearch_arena_free`."""
    var arena_describe: ArenaDescribe
    """`usearch_arena_describe`."""
    var executor_for_ram: ExecutorForRam
    """`usearch_executor_for_ram`."""
    var executor_on_pool: ExecutorOnPool
    """`usearch_executor_on_pool`."""
    var executor_for_nvme: ExecutorForNvme
    """`usearch_executor_for_nvme`."""
    var executor_for_vram: ExecutorForVram
    """`usearch_executor_for_vram`."""
    var executor_drain: ExecutorDrain
    """`usearch_executor_drain`."""
    var executor_free: Release
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
    var dense_health: DenseHealth
    """`usearch_dense_health`."""
    var dense_absorb: DenseAbsorb
    """`usearch_dense_absorb`."""
    var dense_split: DenseSplit
    """`usearch_dense_split`."""
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

    def __init__(out self, ref handle: OwnedDLHandle) raises Error:
        """Resolves every entry point at once, so no call ever pays for a `dlsym`.

        Args:
            handle: The opened library, borrowed only while the addresses are read out.

        Raises:
            With `SYMBOL_MISSING`, on the first symbol the loaded library does not export.
        """
        self.version_name = _resolve[StaticText](handle, "usearch_version_name")
        self.capabilities_compiled = _resolve[StaticText](handle, "usearch_capabilities_compiled")
        self.capabilities_available = _resolve[StaticText](handle, "usearch_capabilities_available")

        self.outcome_wait = _resolve[OutcomeStatus](handle, "usearch_outcome_wait")
        self.outcome_status = _resolve[OutcomeStatus](handle, "usearch_outcome_status")
        self.status_name = _resolve[NameOf](handle, "usearch_status_name")
        self.status_from_name = _resolve[FromNameStatus](handle, "usearch_status_from_name")
        self.scalar_name = _resolve[NameOf](handle, "usearch_scalar_name")
        self.scalar_from_name = _resolve[FromName](handle, "usearch_scalar_from_name")
        self.residency_name = _resolve[NameOf](handle, "usearch_residency_name")
        self.residency_from_name = _resolve[FromNameStatus](handle, "usearch_residency_from_name")
        self.traversal_name = _resolve[NameOf](handle, "usearch_traversal_name")
        self.traversal_from_name = _resolve[FromNameStatus](handle, "usearch_traversal_from_name")
        self.mood_name = _resolve[NameOf](handle, "usearch_mood_name")
        self.mood_from_name = _resolve[FromNameStatus](handle, "usearch_mood_from_name")
        self.cpu_class_name = _resolve[NameOf](handle, "usearch_cpu_class_name")
        self.gpu_class_name = _resolve[NameOf](handle, "usearch_gpu_class_name")
        self.dense_metric_name = _resolve[NameOf](handle, "usearch_dense_metric_name")
        self.dense_metric_from_name = _resolve[FromName](handle, "usearch_dense_metric_from_name")

        self.arena_init = _resolve[ArenaInit](handle, "usearch_arena_init")
        self.arena_free = _resolve[Release](handle, "usearch_arena_free")
        self.arena_describe = _resolve[ArenaDescribe](handle, "usearch_arena_describe")
        self.executor_for_ram = _resolve[ExecutorForRam](handle, "usearch_executor_for_ram")
        self.executor_on_pool = _resolve[ExecutorOnPool](handle, "usearch_executor_on_pool")
        self.executor_for_nvme = _resolve[ExecutorForNvme](handle, "usearch_executor_for_nvme")
        self.executor_for_vram = _resolve[ExecutorForVram](handle, "usearch_executor_for_vram")
        self.executor_drain = _resolve[ExecutorDrain](handle, "usearch_executor_drain")
        self.executor_free = _resolve[Release](handle, "usearch_executor_free")

        self.dense_init_ram = _resolve[DenseInitRam](handle, "usearch_dense_init_ram")
        self.dense_init_vram = _resolve[DenseInitVram](handle, "usearch_dense_init_vram")
        self.dense_load_ram = _resolve[DenseLoadRam](handle, "usearch_dense_load_ram")
        self.dense_load_vram = _resolve[DenseLoadVram](handle, "usearch_dense_load_vram")
        self.dense_load_ram_buffer = _resolve[DenseLoadRamBuffer](handle, "usearch_dense_load_ram_buffer")
        self.dense_load_vram_buffer = _resolve[DenseLoadVramBuffer](handle, "usearch_dense_load_vram_buffer")
        self.dense_view_ram = _resolve[DenseViewRam](handle, "usearch_dense_view_ram")
        self.dense_view_nvme = _resolve[DenseViewNvme](handle, "usearch_dense_view_nvme")
        self.dense_free = _resolve[Release](handle, "usearch_dense_free")

        self.dense_reserve = _resolve[DenseReserve](handle, "usearch_dense_reserve")
        self.dense_clear = _resolve[DenseClear](handle, "usearch_dense_clear")
        self.dense_describe = _resolve[DenseDescribe](handle, "usearch_dense_describe")
        self.dense_insert = _resolve[DenseInsert](handle, "usearch_dense_insert")
        self.dense_refine = _resolve[DenseRefine](handle, "usearch_dense_refine")
        self.dense_health = _resolve[DenseHealth](handle, "usearch_dense_health")
        self.dense_absorb = _resolve[DenseAbsorb](handle, "usearch_dense_absorb")
        self.dense_split = _resolve[DenseSplit](handle, "usearch_dense_split")
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

    Every handle carries one, which is what keeps the shared object mapped for as long as anything
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
        """The resolved entry points, borrowed rather than copied on every call.

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

    def scalar_from_name(self, name: StringSlice) -> ScalarKind:
        """Reads a spelling back; `UNKNOWN` when unrecognized. Calls `usearch_scalar_from_name`."""
        return ScalarKind(self.symbols().scalar_from_name(BytesView.of(name).arguments()))

    def residency_name(self, residency: Residency) -> String:
        """The engine's spelling of a residency, as `RAM` spells "ram". Calls `usearch_residency_name`."""
        return text_at(self.symbols().residency_name(residency.code))

    def residency_from_name(self, name: StringSlice) raises Error -> Residency:
        """Reads a spelling back, raising `INVALID_ARGUMENT` for one it refuses. Calls `usearch_residency_from_name`."""
        return Residency(self._read_named(self.symbols().residency_from_name, name, "usearch_residency_from_name"))

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

    def _read_named(self, parse: FromNameStatus, name: StringSlice, symbol: StaticString) raises Error -> c_int:
        """The enumerator a status-answering reader writes for `name`, or `INVALID_ARGUMENT` when it refuses."""
        var code = stack_allocation[1, c_int]().unsafe_origin_cast[MutAnyOrigin]()
        code[unsafe_offset=0] = c_int(0)
        if parse(BytesView.of(name).arguments(), code) != c_int(0):
            raise Error(Status.INVALID_ARGUMENT, symbol)
        return code[unsafe_offset=0]

    def dense_metric_name(self, metric: DenseMetric) -> String:
        """The engine's spelling of a metric kind, as `INNER` spells "inner". Calls `usearch_dense_metric_name`."""
        return text_at(self.symbols().dense_metric_name(metric.code))

    def dense_metric_from_name(self, name: StringSlice) -> DenseMetric:
        """Reads a spelling back; `UNKNOWN` when unrecognized. Calls `usearch_dense_metric_from_name`."""
        return DenseMetric(self.symbols().dense_metric_from_name(BytesView.of(name).arguments()))

    def _error(self, outcome: Outcome, symbol: StaticString) -> Error:
        """The failure a final outcome reports, its status spelled by the engine's own table."""
        var status = Status.of(outcome.status)
        return Error(status, symbol, outcome.message, self.symbols().status_name(c_int(status.code)), self)

    def _check(self, outcome: Outcome, symbol: StaticString) raises Error:
        """Raises the failure a final outcome reports, and returns on success."""
        if outcome.status != c_int(0):
            raise self._error(outcome, symbol)
