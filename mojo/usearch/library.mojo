"""The loaded shared object and the one place a raw symbol address is ever touched.

The engine is reached by `dlopen` rather than by linking, which is what lets a consumer build
without a single linker flag. Mojo's `@extern` cannot serve here: it resolves at link time, so a
symbol from a library that is only opened at run time is an undefined reference under both the JIT
and `lld`, and using it would put a link flag in every consumer's build.

Every entry point is resolved once, when the library loads, into a typed `thin abi("C")` function
pointer. `OwnedDLHandle.get_function` would run `dlsym` on every call and parameterises only on the
return type, leaving arguments unchecked; a search on the hot path cannot afford either.
"""

from std.ffi import OwnedDLHandle, __fn_type_is_cabi, c_int
from std.memory import ArcPointer
from std.sys.info import CompilationTarget

from usearch.dense import (
    DenseAdd,
    DenseAttend,
    DenseCapabilities,
    DenseClear,
    DenseCount,
    DenseCursorFree,
    DenseCursorNext,
    DenseFind,
    DenseFindFree,
    DenseFindNext,
    DenseFree,
    DenseGet,
    DenseInit,
    DenseLoad,
    DenseLoadBuffer,
    DenseMaxsim,
    DenseMeasure,
    DenseMerge,
    DenseMetadata,
    DenseMetadataBuffer,
    DenseMultiply,
    DenseOpenCursor,
    DenseRefine,
    DenseReserve,
    DenseSave,
    DenseSaveBuffer,
    DenseSearch,
    DenseSearchIf,
)
from usearch.types import (
    Address,
    Error,
    ExecutorDrain,
    ExecutorForNvme,
    ExecutorForRam,
    ExecutorForVram,
    ExecutorFree,
    Status,
    text_at,
)

comptime LIBRARY_FILE = StaticString("libusearch_c.dylib" if CompilationTarget.is_macos() else "libusearch_c.so")
"""The loader already searches the environment's library directory, because `mojo build` bakes it
into the binary's runpath, so an unqualified name is enough for a library installed there."""

comptime BINDING_MAJOR = 3
"""The major version this binding was written against; a mismatch is refused at load."""

# region Signatures

comptime StaticText = def() thin abi("C") -> Address
"""`usearch_version` and both capability getters: a static C string returned by value."""

# endregion Signatures


@always_inline
def _resolve[
    Signature: TrivialRegisterPassable
](ref handle: OwnedDLHandle, name: StaticString) raises Error -> Signature:
    """One `dlsym`, checked once, typed from here on.

    `get_symbol` answers `Optional` where `get_function` aborts the process, which is what lets a
    stale library name the symbol it is missing instead of dying without a message.
    """
    comptime assert __fn_type_is_cabi[Signature](), 'the signature must carry abi("C")'
    var address = handle.get_symbol[NoneType](name)
    if not address:
        raise Error(Status.SYMBOL_MISSING, name, 0)
    return Pointer(to=address.value()).unsafe_bitcast[Signature]()[]


struct Symbols:
    """Every entry point the C ABI exports, resolved once when the library loads.

    Answers are never cached here, only addresses. An index re-asks the engine for its size and its
    capabilities on every call, because a concurrent insertion changes the first and a reload can
    change the second.
    """

    var version: StaticText
    """`usearch_version`."""
    var capabilities_compiled: StaticText
    """`usearch_capabilities_compiled`."""
    var capabilities_available: StaticText
    """`usearch_capabilities_available`."""

    var executor_for_ram: ExecutorForRam
    """`usearch_executor_for_ram`."""
    var executor_for_nvme: ExecutorForNvme
    """`usearch_executor_for_nvme`."""
    var executor_for_vram: ExecutorForVram
    """`usearch_executor_for_vram`."""
    var executor_drain: ExecutorDrain
    """`usearch_executor_drain`."""
    var executor_free: ExecutorFree
    """`usearch_executor_free`."""

    var dense_init: DenseInit
    """`usearch_dense_init`."""
    var dense_free: DenseFree
    """`usearch_dense_free`."""
    var dense_reserve: DenseReserve
    """`usearch_dense_reserve`."""
    var dense_clear: DenseClear
    """`usearch_dense_clear`."""
    var dense_size: DenseMeasure
    """`usearch_dense_size`."""
    var dense_capacity: DenseMeasure
    """`usearch_dense_capacity`."""
    var dense_dimensions: DenseMeasure
    """`usearch_dense_dimensions`."""
    var dense_connectivity: DenseMeasure
    """`usearch_dense_connectivity`."""
    var dense_memory_usage: DenseMeasure
    """`usearch_dense_memory_usage`."""
    var dense_capabilities: DenseCapabilities
    """`usearch_dense_capabilities`."""
    var dense_storage_requirements: DenseMeasure
    """`usearch_dense_storage_requirements`."""

    var dense_add: DenseAdd
    """`usearch_dense_add`."""
    var dense_refine: DenseRefine
    """`usearch_dense_refine`."""
    var dense_search: DenseSearch
    """`usearch_dense_search`."""
    var dense_search_if: DenseSearchIf
    """`usearch_dense_search_if`."""
    var dense_count: DenseCount
    """`usearch_dense_count`."""
    var dense_get: DenseGet
    """`usearch_dense_get`."""

    var dense_cursor: DenseOpenCursor
    """`usearch_dense_cursor`."""
    var dense_cursor_next: DenseCursorNext
    """`usearch_dense_cursor_next`."""
    var dense_cursor_free: DenseCursorFree
    """`usearch_dense_cursor_free`."""
    var dense_find: DenseFind
    """`usearch_dense_find`."""
    var dense_find_next: DenseFindNext
    """`usearch_dense_find_next`."""
    var dense_find_free: DenseFindFree
    """`usearch_dense_find_free`."""

    var dense_save: DenseSave
    """`usearch_dense_save`."""
    var dense_load: DenseLoad
    """`usearch_dense_load`."""
    var dense_metadata: DenseMetadata
    """`usearch_dense_metadata`."""
    var dense_save_buffer: DenseSaveBuffer
    """`usearch_dense_save_buffer`."""
    var dense_load_buffer: DenseLoadBuffer
    """`usearch_dense_load_buffer`."""
    var dense_metadata_buffer: DenseMetadataBuffer
    """`usearch_dense_metadata_buffer`."""

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
        self.version = _resolve[StaticText](handle, "usearch_version")
        self.capabilities_compiled = _resolve[StaticText](handle, "usearch_capabilities_compiled")
        self.capabilities_available = _resolve[StaticText](handle, "usearch_capabilities_available")

        self.executor_for_ram = _resolve[ExecutorForRam](handle, "usearch_executor_for_ram")
        self.executor_for_nvme = _resolve[ExecutorForNvme](handle, "usearch_executor_for_nvme")
        self.executor_for_vram = _resolve[ExecutorForVram](handle, "usearch_executor_for_vram")
        self.executor_drain = _resolve[ExecutorDrain](handle, "usearch_executor_drain")
        self.executor_free = _resolve[ExecutorFree](handle, "usearch_executor_free")

        self.dense_init = _resolve[DenseInit](handle, "usearch_dense_init")
        self.dense_free = _resolve[DenseFree](handle, "usearch_dense_free")
        self.dense_reserve = _resolve[DenseReserve](handle, "usearch_dense_reserve")
        self.dense_clear = _resolve[DenseClear](handle, "usearch_dense_clear")
        self.dense_size = _resolve[DenseMeasure](handle, "usearch_dense_size")
        self.dense_capacity = _resolve[DenseMeasure](handle, "usearch_dense_capacity")
        self.dense_dimensions = _resolve[DenseMeasure](handle, "usearch_dense_dimensions")
        self.dense_connectivity = _resolve[DenseMeasure](handle, "usearch_dense_connectivity")
        self.dense_memory_usage = _resolve[DenseMeasure](handle, "usearch_dense_memory_usage")
        self.dense_capabilities = _resolve[DenseCapabilities](handle, "usearch_dense_capabilities")
        self.dense_storage_requirements = _resolve[DenseMeasure](handle, "usearch_dense_storage_requirements")

        self.dense_add = _resolve[DenseAdd](handle, "usearch_dense_add")
        self.dense_refine = _resolve[DenseRefine](handle, "usearch_dense_refine")
        self.dense_search = _resolve[DenseSearch](handle, "usearch_dense_search")
        self.dense_search_if = _resolve[DenseSearchIf](handle, "usearch_dense_search_if")
        self.dense_count = _resolve[DenseCount](handle, "usearch_dense_count")
        self.dense_get = _resolve[DenseGet](handle, "usearch_dense_get")

        self.dense_cursor = _resolve[DenseOpenCursor](handle, "usearch_dense_cursor")
        self.dense_cursor_next = _resolve[DenseCursorNext](handle, "usearch_dense_cursor_next")
        self.dense_cursor_free = _resolve[DenseCursorFree](handle, "usearch_dense_cursor_free")
        self.dense_find = _resolve[DenseFind](handle, "usearch_dense_find")
        self.dense_find_next = _resolve[DenseFindNext](handle, "usearch_dense_find_next")
        self.dense_find_free = _resolve[DenseFindFree](handle, "usearch_dense_find_free")

        self.dense_save = _resolve[DenseSave](handle, "usearch_dense_save")
        self.dense_load = _resolve[DenseLoad](handle, "usearch_dense_load")
        self.dense_metadata = _resolve[DenseMetadata](handle, "usearch_dense_metadata")
        self.dense_save_buffer = _resolve[DenseSaveBuffer](handle, "usearch_dense_save_buffer")
        self.dense_load_buffer = _resolve[DenseLoadBuffer](handle, "usearch_dense_load_buffer")
        self.dense_metadata_buffer = _resolve[DenseMetadataBuffer](handle, "usearch_dense_metadata_buffer")

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
            raise Error(Status.LIBRARY_MISSING, LIBRARY_FILE, 0)
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
            With `LIBRARY_MISSING`, when nothing is on the loader path, or with `SYMBOL_MISSING`
            when what loaded does not export an entry point the binding needs.
        """
        self.shared = ArcPointer(_Loaded())

    @always_inline
    def symbols(self) -> ref[origin_of(self.shared[].symbols)] Symbols:
        """The resolved entry points, borrowed rather than copied on every call.

        Returns:
            A reference that stays valid for as long as this `Library` is held.
        """
        return self.shared[].symbols

    def version(self) -> String:
        """The loaded engine's own version, which is also the cheapest check that loading worked.

        Returns:
            The version string the shared object reports for itself.
        """
        return text_at(self.symbols().version())

    def capabilities_compiled(self) -> String:
        """The hardware capabilities this build of the engine was compiled with.

        Returns:
            A comma-separated list of capability names.
        """
        return text_at(self.symbols().capabilities_compiled())

    def capabilities_available(self) -> String:
        """The hardware capabilities this machine offers now, which the build does not imply.

        Returns:
            A comma-separated list of capability names.
        """
        return text_at(self.symbols().capabilities_available())
