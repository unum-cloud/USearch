"""The vocabulary every modality shares: keys, scalars, rows, residencies, and the one error type.

Nothing dense-specific lives here - the metric kinds, the result shapes only a dense index answers,
and the index itself belong to @c dense.mojo, so a second modality adds a module rather than
widening this one.

`Error` is the only error type the binding raises: Mojo allows at most one per function, so a
second would force every caller raising `Error` to catch and convert. A refusal is a failure, not an absence - a key the index does not hold answers zero, but a
call the engine declines raises.

Four Mojo facts shape the descriptors and the pointer aliases below, and all four were measured
rather than assumed.

`MutAnyOrigin` rather than `MutUntrackedOrigin` is load-bearing on anything C writes through: with
a narrower origin the compiler folds a read of the slot back to whatever the caller last stored
there, and every out-parameter reads back as zero.

A typed C pointer that may be null crosses as `Optional[Pointer[T, ...]]`, whose empty case is the
null the ABI asks for and costs nothing to pass, in a field, as an argument, or as a return value.
Opaque handles and `void *` state name no element type, so they cross as `Address`, their numeric
value. A struct holding an `Optional` is not `TrivialRegisterPassable`, so C reads those by address.

Every descriptor that views caller storage holds that storage's origin as a struct parameter,
because Mojo destroys a value at its last use rather than at the end of its scope, and an address
laundered through `Address` is not a use. A descriptor built from a list and handed straight to a
call was measured destroying that list before the call ran, leaving the engine reading a freed
allocation; naming the origin in the descriptor's type moved the destruction after the call. The
pointer fields hold that origin too, so the compiler tracks the pointer itself, and `arguments`
erases it inside the verb, while the descriptor is still alive, to reach the one instantiation the
ABI signatures name. A `Deferred` holds the union of every origin its call
views the same way, which keeps the index, the executor, the arena and the inputs alive until it
is gone, and an `Arena` holds its executor's origin, so the executor outlives it.

A struct whose address crosses to C is materialized into a `stack_allocation` slot rather than held
in a local whose address is taken. An `alloca` is not a value Mojo owns, so nothing destroys it and
nothing folds its store away, where a local's last tracked use is the address-taking itself. A
deferred call's slots outlive the verb that submitted them, so they live inside its `Deferred`,
which is built in the caller's place and never moves.

File: mojo/usearch/types.mojo
Author: Ash Vardanian
Date: September 7, 2026
"""

from std.bit import byte_swap
from std.ffi import c_int, c_size_t
from std.builtin._coroutine import AnyCoroutine, _suspend_async
from std.memory import stack_allocation
from std.runtime._asyncrt import _async_execute
from std.sys.info import size_of

from usearch.library import Library

# region Pointers

comptime CString = Pointer[Int8, ImmUntrackedOrigin]
"""A viewed null-terminated C string, such as an enumerator's spelling or an outcome's message."""

comptime Address = Int
"""An opaque handle or a `void *` crossing as its numeric value, zero for null.

A typed slot crosses as `Pointer`, or as `Optional[Pointer]` where it may be null; this is for the
slots that name no element type. The integer is pointer-width and shares the pointer's argument
class on both x86-64 and AArch64, so it lands in the same register a pointer would.
"""

comptime NULL_ADDRESS = 0
"""The null pointer, as `Address` spells it."""


def _anchor[origin: Origin]() -> ref[origin] NoneType:
    """A reference of the given origin, read only inside `origin_of` to union parameter origins."""
    return Pointer[NoneType, origin](unsafe_from_address=NULL_ADDRESS)[]


# endregion Pointers

# region Errors


@fieldwise_init
struct Status(Equatable, ImplicitlyCopyable, TrivialRegisterPassable):
    """Why a call into the engine failed - the `usearch_status_t` vocabulary, verbatim.

    An `Error` never holds `SUCCESS` or `DEFERRED`: only a `Deferred` reports the latter, and only
    until its call is final. `Library.status_name` spells each one as the engine does.
    """

    var code: Int32
    """The raw `usearch_status_t` value, kept even when this build does not name it."""
    comptime SUCCESS = Self(0)
    """The call completed."""
    comptime DEFERRED = Self(1)
    """The call was submitted deferred and is not final yet; the one non-final status."""
    comptime OUT_OF_MEMORY = Self(2)
    """An allocation or reservation could not be satisfied."""
    comptime SHAPE_MISMATCH = Self(3)
    """Dimensionality or row counts disagree."""
    comptime UNSUPPORTED_LAYOUT = Self(4)
    """A stride or packing the engine cannot traverse."""
    comptime UNKNOWN_KIND = Self(5)
    """A scalar, metric, or residency tag this build does not know."""
    comptime READ_ONLY = Self(6)
    """A mutating verb on a handle that serves its checkpoint in place."""
    comptime IO = Self(7)
    """A file could not be opened, read, written, or parsed."""
    comptime DEVICE = Self(8)
    """A device could not be selected, or its memory could not be claimed."""
    comptime MALFORMED = Self(9)
    """A caller-supplied representation could not be decoded."""
    comptime INTERNAL = Self(10)
    """The engine reported neither success nor a failure, which names a defect rather than a state."""
    comptime OVERFLOW = Self(11)
    """A hard design ceiling was reached, so a smaller request cannot help."""
    comptime UNINITIALIZED = Self(12)
    """The verb needs a prior successful reservation that never happened."""
    comptime CONFIG_MISMATCH = Self(13)
    """The request contradicts an already-frozen configuration."""
    comptime CONTENDED = Self(14)
    """The call was refused rather than waited for, like a submission to an executor another call holds."""
    comptime CANCELLED = Self(15)
    """A progress handler stopped the operation; it published nothing."""
    comptime UNSUPPORTED = Self(16)
    """The verb is declared for every residency but this one does not implement it."""
    comptime INVALID_ARGUMENT = Self(17)
    """An argument breaks the verb's contract, like a required output left null."""
    comptime LIBRARY_MISSING = Self(-1)
    """Nothing to `dlopen` - the binding's own."""
    comptime SYMBOL_MISSING = Self(-2)
    """The library loaded but does not export a symbol the binding needs - the binding's own."""
    comptime VERSION_MISMATCH = Self(-3)
    """The library loaded but reports a different major version than the binding's - the binding's own."""

    @staticmethod
    def of(code: c_int) -> Self:
        """Lifts a raw `usearch_status_t`, keeping an unnamed one rather than guessing.

        Args:
            code: Whatever the C call reported, named by this build or not.

        Returns:
            The matching status, or one holding the unrecognized code verbatim.
        """
        return Self(Int32(code))


@fieldwise_init
struct Outcome(ImplicitlyCopyable):
    """The `usearch_outcome_t` every verb reports through, laid out as the ABI declares it.

    Never marked as anything richer than this: the engine writes the two words and the binding
    reads them back, so the struct exists to have a stable address rather than an interface.
    """

    var status: c_int
    """The `usearch_status_t` the call reported."""
    var message: Optional[CString]
    """The engine's own static message, or `None` when the call succeeded."""

    def __init__(out self):
        """Zeroes the slot, so a call that writes nothing still reads as a success."""
        self.status = c_int(0)
        self.message = None


@fieldwise_init
struct Error(Copyable, ImplicitlyCopyable, Writable):
    """What went wrong inside the engine, and which entry point it was about."""

    var status: Status
    """Which class of failure the engine reported."""
    var detail: StaticString
    """The entry point it was about, for a reader rather than for a branch."""
    var message: Optional[CString]
    """The engine's own static message, or `None` when it held none.

    Messages are static storage inside the shared object, so keeping the address costs nothing and
    outlives the call. `write_to` walks it; there is nothing to free.
    """
    var name: Optional[CString]
    """The engine's spelling of `status` from `usearch_status_name`, or `None` where no engine named it."""
    var library: Optional[Library]
    """The engine `message` and `name` live in, kept mapped for as long as the error is; Mojo would
    otherwise unload it at the last handle's last use, while the error is still being read."""

    def __init__(out self, status: Status, detail: StaticString):
        """A failure the binding caught before any engine could name it.

        Args:
            status: Which class of failure it is.
            detail: The entry point or binding type it was about.
        """
        self.status = status
        self.detail = detail
        self.message = None
        self.name = None
        self.library = None

    def write_to(self, mut writer: Some[Writer]):
        """Writes the status name, the entry point, and the engine's own message together.

        Args:
            writer: The sink the message is appended to.
        """
        writer.write("USearch: ")
        if case .Some(name) = self.name:
            writer.write(text_at(name))
        else:
            match self.status:
                case == Status.LIBRARY_MISSING:
                    writer.write("library-missing")
                case == Status.SYMBOL_MISSING:
                    writer.write("symbol-missing")
                case == Status.VERSION_MISMATCH:
                    writer.write("version-mismatch")
                case _:
                    writer.write("status ", self.status.code)
        writer.write(" [", self.detail, "]")
        if case .Some(message) = self.message:
            writer.write(": ", text_at(message))


def text_at(text: Optional[CString]) -> String:
    """Decodes a null-terminated C string the engine owns, stopping at its terminator.

    Args:
        text: Where the bytes start; `None` reads as the empty string.

    Returns:
        A copy of the bytes, since the caller may keep it longer than it wants a raw address.
    """
    if case .Some(start) = text:
        var bytes = start.unsafe_bitcast[UInt8]()
        var length = 0
        while bytes[unsafe_offset=length] != 0:
            length += 1
        return String(unsafe_from_utf8=Span[UInt8, ImmUntrackedOrigin](unsafe_ptr=bytes, length=length))
    return String("")


# endregion Errors

# region Keys


@fieldwise_init
struct Key(Equatable, ImplicitlyCopyable, TrivialRegisterPassable, Writable):
    """The sixteen opaque bytes naming one entry - `usearch_key_t`, in two words.

    Mojo cannot put a `UInt8` array in a register-passable struct, and a 16-lane SIMD would take an
    argument class the C ABI never gives a byte array, so the bytes cross as the two integers they
    occupy. Keys need not be unique: several entries may share one, which is what `values_per_key`
    and the late-interaction verb are for.
    """

    var leading: UInt64
    """Bytes 0 through 7, zero for an integer identity."""
    var trailing: UInt64
    """Bytes 8 through 15, holding an integer identity big-endian."""

    def __init__(out self, *, integer: UInt64):
        """Builds the key an integer identity names, with the leading bytes zeroed.

        Args:
            integer: The identity to park in bytes 8 through 15, most significant byte first.
        """
        self.leading = 0
        self.trailing = byte_swap(integer)

    def as_integer(self) -> Optional[UInt64]:
        """The integer identity in bytes 8 through 15, and nothing when the leading ones are set.

        Returns:
            The trailing word read big-endian, or `None` for a key whose leading bytes hold
            identity of their own and so has no 64-bit one.
        """
        if self.leading != 0:
            return None
        return byte_swap(self.trailing)

    def write_to(self, mut writer: Some[Writer]):
        """Writes the key as the lowercase dashed UUID the C header calls canonical. Binding-only.

        Args:
            writer: The sink the key is appended to.
        """
        var bytes = Pointer(to=self).unsafe_bitcast[UInt8]()
        for position in range(16):
            if position == 4 or position == 6 or position == 8 or position == 10:
                writer.write("-")
            var byte = Int(bytes[unsafe_offset=position])
            for nibble in [byte >> 4, byte & 15]:
                writer.write(chr(nibble + 48 if nibble < 10 else nibble + 87))


comptime Distance = Float64
"""A distance or similarity score in the metric of the index that produced it - `usearch_distance_t`."""

# endregion Keys

# region Scalars


@fieldwise_init
struct ScalarKind(Equatable, ImplicitlyCopyable, TrivialRegisterPassable):
    """The scalar types the indexes store and exchange - `usearch_scalar_t`, named by width.

    Ordered by shrinking dynamic range, floats first, then signed and unsigned integers. The `xN`
    suffixes spell how many logical dimensions share one byte. `Library.scalar_name` spells each one
    as the engine does.
    """

    var code: c_int
    """The raw `usearch_scalar_t` value the ABI exchanges."""
    comptime UNKNOWN = Self(0)
    """No scalar named - an unset field, or a custom metric owning its own storage."""
    comptime F64 = Self(1)
    """IEEE-754 binary64."""
    comptime F32 = Self(2)
    """IEEE-754 binary32."""
    comptime BF16 = Self(3)
    """Truncated f32 - f32's range on a 7-bit mantissa."""
    comptime F16 = Self(4)
    """IEEE-754 binary16."""
    comptime E5M2 = Self(5)
    """FP8 with a 5-bit exponent - f16's range on two mantissa bits."""
    comptime E4M3 = Self(6)
    """FP8 with a 4-bit exponent, no infinities."""
    comptime E3M2 = Self(7)
    """FP6 with a 3-bit exponent, byte-aligned with two padding bits."""
    comptime E2M3 = Self(8)
    """FP6 with a 2-bit exponent, byte-aligned with two padding bits."""
    comptime I8 = Self(10)
    """8-bit signed integer."""
    comptime I4X2 = Self(11)
    """Two signed nibbles per byte, even dimensions in the low nibble."""
    comptime I32 = Self(12)
    """32-bit signed integer - the coordinates of geospatial points."""
    comptime U8 = Self(20)
    """8-bit unsigned integer."""
    comptime U4X2 = Self(21)
    """Two unsigned nibbles per byte, even dimensions in the low nibble."""
    comptime U1X8 = Self(22)
    """Eight single-bit dimensions per byte - what Hamming and Jaccard score."""

    @staticmethod
    def of[dtype: DType]() -> Self:
        """The kind a Mojo scalar type stores one-to-one; the packed and FP6 kinds have none. Binding-only.

        Parameters:
            dtype: The element type of the caller's storage.

        Returns:
            The matching kind, or `UNKNOWN`, which every verb refuses.
        """
        comptime match dtype:
            case == DType.float64:
                return Self.F64
            case == DType.float32:
                return Self.F32
            case == DType.bfloat16:
                return Self.BF16
            case == DType.float16:
                return Self.F16
            case == DType.float8_e5m2:
                return Self.E5M2
            case == DType.float8_e4m3fn:
                return Self.E4M3
            case == DType.int32:
                return Self.I32
            case == DType.int8:
                return Self.I8
            case == DType.uint8:
                return Self.U8
            case _:
                return Self.UNKNOWN

    def bits_per_scalar(self) -> Int:
        """How many bits one logical dimension occupies, which sub-byte kinds share.

        Returns:
            The width in bits, or zero for `UNKNOWN`, whose storage the caller owns.
        """
        if self == Self.F64:
            return 64
        if self == Self.F32 or self == Self.I32:
            return 32
        if self == Self.BF16 or self == Self.F16:
            return 16
        if self == Self.E5M2 or self == Self.E4M3 or self == Self.E3M2 or self == Self.E2M3:
            return 8
        if self == Self.I8 or self == Self.U8:
            return 8
        if self == Self.I4X2 or self == Self.U4X2:
            return 4
        if self == Self.U1X8:
            return 1
        return 0

    def bytes_for_columns(self, columns: Int) -> Int:
        """The natural width of one row, rounding a sub-byte packing up to whole bytes.

        Args:
            columns: Logical scalars in the row, as the caller counts them.

        Returns:
            The byte count a contiguous row of that width occupies.
        """
        var bits = self.bits_per_scalar()
        return (columns * bits + 7) // 8

    def stores[dtype: DType](self) -> Bool:
        """Whether rows of this kind read as whole `dtype` elements: equal widths, or bytes for packed kinds.
        Binding-only.

        Parameters:
            dtype: The element type the caller reads the rows as.

        Returns:
            True when one element of `dtype` spans exactly one scalar, or one byte of a packed kind.
        """
        var bits = self.bits_per_scalar()
        if bits == 0:
            return False
        if bits < 8:
            return size_of[dtype]() == 1
        return size_of[dtype]() * 8 == bits


# endregion Scalars

# region Residency & Execution


@fieldwise_init
struct Residency(Equatable, ImplicitlyCopyable, TrivialRegisterPassable):
    """Where an index's payload lives and which engine traverses it - `usearch_residency_t`.

    Each lifecycle verb names its residency, as `DenseIndex.init_vram` does, so this tag only travels
    through the name tables, `Library.residency_name` and `Library.residency_from_name`, and back
    from `Arena.describe`.
    """

    var code: c_int
    """The raw `usearch_residency_t` value the ABI exchanges."""
    comptime RAM = Self(1)
    """Volatile memory traversed by CPU threads - the concurrent baseline."""
    comptime VRAM = Self(2)
    """Accelerator memory traversed by GPU warps on the chosen devices."""
    comptime NVME = Self(3)
    """Persisted storage traversed through pluggable I/O engines, sized past RAM."""


@fieldwise_init
struct Traversal(Equatable, ImplicitlyCopyable, TrivialRegisterPassable):
    """How a search reaches the indexed entries it scores - `usearch_traversal_t`."""

    var code: c_int
    """The raw `usearch_traversal_t` value the ABI exchanges."""
    comptime WALK = Self(1)
    """Walk the graph, scoring the entries the expansion reaches - approximate, the default."""
    comptime SCAN = Self(2)
    """Score every entry published when the query's scan starts - exact, on RAM and NVMe handles."""


@fieldwise_init
struct Mood(Equatable, ImplicitlyCopyable, TrivialRegisterPassable):
    """What an executor does with its threads or streams once it goes idle - `usearch_mood_t`."""

    var code: c_int
    """The raw `usearch_mood_t` value the ABI exchanges."""
    comptime PARKED = Self(0)
    """Idle threads sleep; the next call pays a wake-up. The default."""
    comptime SPINNING = Self(1)
    """Idle threads stay hot, trading power for the next call's latency."""


@fieldwise_init
struct CPUClass(Equatable, ImplicitlyCopyable, TrivialRegisterPassable):
    """The CPU family dispatch picks, or a placement caps - `usearch_cpu_class_t`."""

    var code: c_int
    """The raw `usearch_cpu_class_t` value the ABI exchanges."""
    comptime UNKNOWN = Self(0)
    """Derives the CPU's own class."""


@fieldwise_init
struct GPUClass(Equatable, ImplicitlyCopyable, TrivialRegisterPassable):
    """The GPU family dispatch picks, or a placement caps - `usearch_gpu_class_t`."""

    var code: c_int
    """The raw `usearch_gpu_class_t` value the ABI exchanges."""
    comptime UNKNOWN = Self(0)
    """Derives the devices' own class."""


@fieldwise_init
struct NVMeClass(Equatable, ImplicitlyCopyable, TrivialRegisterPassable):
    """The NVMe I/O engine family an executor drives - `usearch_nvme_class_t`."""

    var code: c_int
    """The raw `usearch_nvme_class_t` value the ABI exchanges."""
    comptime UNKNOWN = Self(0)
    """Derives the devices' own class."""


@fieldwise_init
struct LoadPath(Equatable, ImplicitlyCopyable, TrivialRegisterPassable):
    """How a load from a path read the parts onto the devices - `usearch_load_path_t`."""

    var code: c_int
    """The raw `usearch_load_path_t` value the ABI exchanges."""
    comptime NONE = Self(0)
    """Never loaded from a path, or an index on RAM or NVMe."""
    comptime HOST_COPY = Self(1)
    """Through host memory, since some part's file took no cuFile registration."""
    comptime CUFILE = Self(2)
    """Every part straight onto the device through cuFile."""


@fieldwise_init
struct Completion(Equatable, ImplicitlyCopyable, TrivialRegisterPassable):
    """Whether a verb returns with its final status or once submitted - `usearch_completion_t`.

    The receiver picks it: the index and `index.on(executor)` pass `BLOCKING`, and
    `index.deferred(executor)` passes `DEFERRED`, so no verb takes it as an argument.
    """

    var code: c_int
    """The raw `usearch_completion_t` value the ABI exchanges."""
    comptime BLOCKING = Self(0)
    """Return with the final status stored and every output readable."""
    comptime DEFERRED = Self(1)
    """Return once submitted; the status reads `DEFERRED` until final."""


@fieldwise_init
struct ProgressArguments(Defaultable, ImplicitlyCopyable):
    """`usearch_progress_t`: how a long call reports its advance - a handler, its state, and how
    many reports it hears."""

    var callback: Address
    """The optional handler; `NULL_ADDRESS` skips every report."""
    var state: Address
    """Forwarded into every `callback` call."""
    var steps: c_size_t
    """Reports a call makes at most, the last at completion; zero derives to ten."""

    def __init__(out self):
        """No handler, so the call runs as one slice and reports nothing."""
        self.callback = NULL_ADDRESS
        self.state = NULL_ADDRESS
        self.steps = c_size_t(0)

    @staticmethod
    def of[
        State: AnyType, //, handler: def(Int, Int, Pointer[State, MutAnyOrigin]) thin -> Continuation
    ](state: Pointer[State, MutAnyOrigin], steps: Int = 0) -> Self:
        """Installs a Mojo handler, never called concurrently, that may stop the call. Binding-only.

        Parameters:
            State: The handler's state, inferred from the argument.
            handler: Called with the units completed, the units in total, and the state.

        Args:
            state: What the handler reads and writes; it must outlive the call.
            steps: Reports the call makes at most, the last at completion; zero derives to ten.

        Returns:
            The progress slot every config with a `progress` knob takes.
        """
        var callback: ProgressCallback = _progress_trampoline[State, handler]
        var progress = Self()
        progress.callback = Pointer(to=callback).unsafe_bitcast[Address]()[]
        progress.state = Int(state)
        progress.steps = c_size_t(steps)
        return progress


@fieldwise_init
struct Continuation(Equatable, ImplicitlyCopyable, TrivialRegisterPassable):
    """Whether the call a progress handler watches continues or unwinds - `usearch_continuation_t`."""

    var code: c_int
    """The raw `usearch_continuation_t` value the ABI exchanges."""
    comptime CONTINUE = Self(0)
    """Carry on to the next unit of work."""
    comptime CANCEL = Self(1)
    """Stop with `CANCELLED`; whether finished units stand is each verb's own contract."""


comptime ProgressCallback = def(c_size_t, c_size_t, Address) thin abi("C") -> c_int
"""`usearch_progress_callback_t`: the units completed, the units in total, and the state."""


def _progress_trampoline[
    State: AnyType, handler: def(Int, Int, Pointer[State, MutAnyOrigin]) thin -> Continuation
](completed: c_size_t, total: c_size_t, state: Address) abi("C") -> c_int:
    """The C handler one Mojo handler compiles into, its state typed again."""
    return handler(Int(completed), Int(total), Pointer[State, MutAnyOrigin](unsafe_from_address=state)).code


@fieldwise_init
struct PredicateCost(Equatable, ImplicitlyCopyable, TrivialRegisterPassable):
    """How costly one predicate call is, which decides how many keys each call holds."""

    var code: c_int
    """The raw value the ABI exchanges."""
    comptime CHEAP = Self(0)
    """Asked as candidates arrive, at most four keys a call, with nothing buffered."""
    comptime EXPENSIVE = Self(1)
    """Asked once per traversal round or scan run about every candidate it gathered."""


@fieldwise_init
struct PredicateArguments(ImplicitlyCopyable):
    """`usearch_predicate_t`: which candidate keys a filtered search may report - a handler and the
    state it is called with."""

    var callback: Address
    """The handler, compiled in Mojo by `each` or `batch`, or in C; `NULL_ADDRESS` admits every key."""
    var state: Address
    """Forwarded into every `callback` call."""
    var cost: PredicateCost
    """How many keys each `callback` call holds."""

    def __init__(out self):
        """No handler, which admits every key."""
        self.callback = NULL_ADDRESS
        self.state = NULL_ADDRESS
        self.cost = PredicateCost.CHEAP

    @staticmethod
    def each[
        State: AnyType, //, judge: def(Key, Pointer[State, MutAnyOrigin]) thin -> Bool
    ](state: Pointer[State, MutAnyOrigin]) -> Self:
        """Installs a cheap Mojo handler answering one key at a time, asked as candidates are scored and
        called concurrently from the executor's threads. Binding-only.

        Parameters:
            State: The handler's state, inferred from the argument.
            judge: Whether one candidate key may be reported, reading the state but never writing it.

        Args:
            state: What the handler reads; it must outlive the call.

        Returns:
            The predicate a filtered search takes.
        """
        var callback: PredicateCallback = _each_trampoline[State, judge]
        return Self(Pointer(to=callback).unsafe_bitcast[Address]()[], Int(state), PredicateCost.CHEAP)

    @staticmethod
    def batch[
        State: AnyType,
        //,
        judge: def(
            Span[Key, ImmUntrackedOrigin], Span[Admission, MutUntrackedOrigin], Pointer[State, MutAnyOrigin]
        ) thin -> None,
    ](state: Pointer[State, MutAnyOrigin], cost: PredicateCost = PredicateCost.EXPENSIVE) -> Self:
        """Installs a Mojo handler answering a batch at a time, asked once per round unless told it is
        cheap, and called concurrently from the executor's threads. Binding-only.

        Parameters:
            State: The handler's state, inferred from the argument.
            judge: Writes one admission per candidate key, reading the state but never writing it.

        Args:
            state: What the handler reads; it must outlive the call.
            cost: How many keys each call holds.

        Returns:
            The predicate a filtered search takes.
        """
        var callback: PredicateCallback = _batch_trampoline[State, judge]
        return Self(Pointer(to=callback).unsafe_bitcast[Address]()[], Int(state), cost)


comptime PredicateCallback = def(
    Pointer[Key, ImmUntrackedOrigin], Pointer[Admission, MutUntrackedOrigin], c_size_t, Address
) thin abi("C") -> None
"""`usearch_predicate_callback_t`: the candidate keys, one admission slot apiece, their count, and the state."""


def _each_trampoline[
    State: AnyType, judge: def(Key, Pointer[State, MutAnyOrigin]) thin -> Bool
](
    keys: Pointer[Key, ImmUntrackedOrigin],
    admissions: Pointer[Admission, MutUntrackedOrigin],
    count: c_size_t,
    state: Address,
) abi("C"):
    """The C handler one per-key Mojo predicate compiles into, asking it about every key in turn."""
    var typed = Pointer[State, MutAnyOrigin](unsafe_from_address=state)
    var written = admissions.unsafe_origin_cast[MutAnyOrigin]()
    for position in range(Int(count)):
        var admitted = judge(keys[unsafe_offset=position], typed)
        written[unsafe_offset=position] = Admission.ADMIT if admitted else Admission.REJECT


def _batch_trampoline[
    State: AnyType,
    judge: def(
        Span[Key, ImmUntrackedOrigin], Span[Admission, MutUntrackedOrigin], Pointer[State, MutAnyOrigin]
    ) thin -> None,
](
    keys: Pointer[Key, ImmUntrackedOrigin],
    admissions: Pointer[Admission, MutUntrackedOrigin],
    count: c_size_t,
    state: Address,
) abi("C"):
    """The C handler one batch Mojo predicate compiles into, its keys and admissions spanned again."""
    judge(
        Span[Key, ImmUntrackedOrigin](unsafe_ptr=keys, length=Int(count)),
        Span[Admission, MutUntrackedOrigin](unsafe_ptr=admissions, length=Int(count)),
        Pointer[State, MutAnyOrigin](unsafe_from_address=state),
    )


# endregion Residency & Execution

# region Byte Runs


@fieldwise_init
struct BytesView[origin: ImmOrigin = ImmUntrackedOrigin](ImplicitlyCopyable, TrivialRegisterPassable):
    """A viewed run of bytes and its length - `usearch_bytes_view_t`, never null-terminated.

    Crosses by value, which the two-word argument class handles on both x86-64 and AArch64.

    Parameters:
        origin: The viewed storage, named so it outlives every call this view crosses.
    """

    var data: Pointer[Int8, Self.origin]
    """The first byte, read only during the call."""
    var length: c_size_t
    """How many bytes the run covers, since nothing terminates it."""

    @staticmethod
    def of[text_origin: ImmOrigin, //](text: StringSlice[text_origin]) -> BytesView[text_origin]:
        """Views a Mojo string's bytes, which the caller keeps alive across the call.

        Parameters:
            text_origin: The string's origin, inferred from the argument.

        Args:
            text: The characters to hand over.

        Returns:
            A view of exactly those bytes, with no terminator manufactured.
        """
        return BytesView[text_origin](text.unsafe_ptr().unsafe_bitcast[Int8](), c_size_t(text.byte_length()))

    @staticmethod
    def of[bytes_origin: ImmOrigin, //](bytes: Span[UInt8, bytes_origin]) -> BytesView[bytes_origin]:
        """Views a run of raw bytes, such as the monolith `save_buffer` returns.

        Parameters:
            bytes_origin: The run's origin, inferred from the argument.

        Args:
            bytes: The bytes to hand over.

        Returns:
            A view of exactly those bytes.
        """
        return BytesView[bytes_origin](bytes.unsafe_ptr().unsafe_bitcast[Int8](), c_size_t(len(bytes)))

    def arguments(self) -> BytesView[ImmUntrackedOrigin]:
        """The same run with its view erased - the one instantiation the ABI signatures name.

        Returns:
            A view of the same bytes, whose type no longer keeps their storage alive.
        """
        return BytesView[ImmUntrackedOrigin](self.data.unsafe_origin_cast[ImmUntrackedOrigin](), self.length)


# endregion Byte Runs

# region Key Runs


@fieldwise_init
struct KeysView[origin: ImmOrigin = ImmUntrackedOrigin](ImplicitlyCopyable, TrivialRegisterPassable):
    """A viewed, strided run of keys - `usearch_keys_view_t`, one per row of the call's count.

    Holds no count: the verb's own count is the one batch axis. A zero stride repeats the first
    key for every row.

    Parameters:
        origin: The viewed storage, named so it outlives every call this view crosses.
    """

    var data: Pointer[Key, Self.origin]
    """The first key, read only during the call."""
    var stride: c_size_t
    """Bytes between consecutive keys; zero repeats the first."""

    @staticmethod
    def of[keys_origin: ImmOrigin, //](keys: Span[Key, keys_origin]) -> KeysView[keys_origin]:
        """Views a contiguous run of keys as the view the engine reads.

        Parameters:
            keys_origin: The run's origin, inferred from the argument.

        Args:
            keys: The identities the batch is published under.

        Returns:
            A view of exactly those keys.
        """
        return KeysView[keys_origin](keys.unsafe_ptr(), c_size_t(size_of[Key]()))

    def arguments(self) -> KeysView[ImmUntrackedOrigin]:
        """The same run with its view erased - the one instantiation the ABI layouts name.

        Returns:
            A view of the same keys, whose type no longer keeps their storage alive.
        """
        return KeysView[ImmUntrackedOrigin](self.data.unsafe_origin_cast[ImmUntrackedOrigin](), self.stride)


@fieldwise_init
struct KeyBytes(ImplicitlyCopyable, TrivialRegisterPassable):
    """A `usearch_key_t` field inside an argument structure, at the byte array's 4-byte-friendly
    alignment rather than the 8 that `Key`'s two words would force, so the C offsets hold."""

    var first: UInt32
    var second: UInt32
    var third: UInt32
    var fourth: UInt32

    @staticmethod
    def of(key: Key) -> Self:
        """Splits a key into the four words its sixteen bytes occupy.

        Args:
            key: The identity to hold.

        Returns:
            The same bytes, in the same order.
        """
        var words = Pointer(to=key).unsafe_bitcast[UInt32]()
        return Self(
            words[unsafe_offset=0],
            words[unsafe_offset=1],
            words[unsafe_offset=2],
            words[unsafe_offset=3],
        )


# endregion Key Runs

# region Rows


@fieldwise_init
struct RowsViewArguments(ImplicitlyCopyable, TrivialRegisterPassable):
    """`usearch_rows_view_t`, laid out as the ABI declares it: the call's `count` is its row count."""

    var data: Address
    var stride: c_size_t
    var columns: c_size_t
    var scalar: ScalarKind


@fieldwise_init
struct RowsView[origin: ImmOrigin = ImmUntrackedOrigin](ImplicitlyCopyable):
    """A viewed run of rows of one width - `usearch_rows_view_t`, the batch currency every verb reads.

    One vector is a one-row view. The row count becomes the verb's `count`, and the column count
    crosses with the rows, so the engine checks it against the geometry the index froze. The row
    and column counts are scalars and `stride` alone is in bytes; a zero stride repeats the first
    row for every one of them, as BLAS reads it.

    Parameters:
        origin: The viewed rows, named so they outlive every call this view crosses.
    """

    var data: Pointer[NoneType, Self.origin]
    """Base pointer of the first row, read only during the call."""
    var rows: c_size_t
    """Rows the view holds, which the verb takes as its `count`."""
    var columns: c_size_t
    """Logical scalars per row, as the caller counts them."""
    var stride: c_size_t
    """Bytes between the starts of consecutive rows; zero repeats the first row."""
    var scalar: ScalarKind
    """The scalar type the rows hold, converted on the way in where it differs."""

    @staticmethod
    def of[
        dtype: DType, rows_origin: ImmOrigin, //, kind: ScalarKind = ScalarKind.of[dtype]()
    ](rows: Span[Scalar[dtype], rows_origin], columns: Int) raises Error -> RowsView[rows_origin]:
        """Reads a flat run of scalars as contiguous rows of the given width. Binding-only.

        The row count comes from the bytes the run spans divided by the bytes one row of `kind`
        occupies. A sub-byte kind counts logical dimensions, so eight `U1X8` columns are one `UInt8`.

        Parameters:
            dtype: The scalar type of the run, inferred from the argument.
            rows_origin: The run's origin, inferred with it.
            kind: How the engine reads those scalars, which `dtype` must store whole; the kind
                `dtype` stores by default.

        Args:
            rows: The scalars, laid out row after row with no padding between them.
            columns: Logical scalars per row, which must divide the run into whole rows.

        Returns:
            A view of the whole run, strided at its natural row width.

        Raises:
            `Error` when the run does not divide into whole rows of that width.
        """
        comptime assert kind.stores[dtype](), "the element type must store the kind whole"
        var row_bytes = kind.bytes_for_columns(columns)
        var spanned = len(rows) * size_of[Scalar[dtype]]()
        if columns <= 0 or spanned % row_bytes != 0:
            raise Error(Status.SHAPE_MISMATCH, "RowsView.of")
        return RowsView[rows_origin](
            rows.unsafe_ptr().unsafe_bitcast[NoneType](),
            c_size_t(spanned // row_bytes),
            c_size_t(columns),
            c_size_t(row_bytes),
            kind,
        )

    def arguments(self) -> RowsViewArguments:
        """The batch-axis form the verbs take, its row count moved into the call's `count`.

        Returns:
            The same base, stride, width, and scalar kind.
        """
        return RowsViewArguments(Int(self.data), self.stride, self.columns, self.scalar)


# endregion Rows

# region Matches


@fieldwise_init
struct MatchesViewArguments(Defaultable, ImplicitlyCopyable):
    """`usearch_matches_view_t`, laid out as the ABI declares it: `count x columns` keys and scores
    inside the arena, and one count per query."""

    var keys: Address
    var scores: Address
    var counts: Address
    var columns: c_size_t

    def __init__(out self):
        """Nothing written, which is what a failed call leaves."""
        self.keys = NULL_ADDRESS
        self.scores = NULL_ADDRESS
        self.counts = NULL_ADDRESS
        self.columns = c_size_t(0)


@fieldwise_init
struct MatchesTapeViewArguments(Defaultable, ImplicitlyCopyable):
    """`usearch_matches_tape_view_t`, laid out as the ABI declares it: keys and scores end to end
    inside the arena, and one more offset than queries."""

    var keys: Address
    var scores: Address
    var offsets: Address

    def __init__(out self):
        """Nothing written, which is what a failed call leaves."""
        self.keys = NULL_ADDRESS
        self.scores = NULL_ADDRESS
        self.offsets = NULL_ADDRESS


@fieldwise_init
struct KeysTapeViewArguments(Defaultable, ImplicitlyCopyable):
    """`usearch_keys_tape_view_t`, laid out as the ABI declares it: every entry's keys end to end
    inside the arena, and one more offset than entries."""

    var keys: Address
    var offsets: Address

    def __init__(out self):
        """Nothing written, which is what a failed call leaves."""
        self.keys = NULL_ADDRESS
        self.offsets = NULL_ADDRESS


@fieldwise_init
struct Matches[origin: MutOrigin](Movable, Sized):
    """A ranked run a call wrote into its arena - `usearch_matches_view_t`, read in place.

    Every ranking verb - the search, late interaction, and a stream pull - answers exactly this. Each
    count is the number of matches written for its query, never more than `wanted`.

    Parameters:
        origin: The arena the run lives in, which this result views.
    """

    var view: MatchesViewArguments
    """Where the engine wrote the keys, scores and counts, inside the arena."""
    var queries: Int
    """Queries the call answered."""

    def __len__(self) -> Int:
        """The queries the call answered.

        Returns:
            One per query of that call.
        """
        return self.queries

    def query_keys(ref self, query: Int) -> Span[Key, origin_of(self)]:
        """One query's keys, nearest first, trimmed to the matches that query found.

        Args:
            query: The query's offset in the batch, below `len(self)`.

        Returns:
            The keys, viewed from the arena this result reads.
        """
        var start = self.view.keys + query * Int(self.view.columns) * size_of[Key]()
        return _in_arena[Key, origin_of(self)](start, self._found(query))

    def query_distances(ref self, query: Int) -> Span[Distance, origin_of(self)]:
        """One query's scores, paired positionally with `query_keys`.

        Args:
            query: The query's offset in the batch, below `len(self)`.

        Returns:
            The scores, viewed from the arena this result reads.
        """
        var start = self.view.scores + query * Int(self.view.columns) * size_of[Distance]()
        return _in_arena[Distance, origin_of(self)](start, self._found(query))

    def _found(self, query: Int) -> Int:
        """The matches written for one query; a pull with nothing to report left the counts null."""
        if self.view.counts == NULL_ADDRESS:
            return 0
        return Int(Pointer[c_size_t, ImmUntrackedOrigin](unsafe_from_address=self.view.counts)[unsafe_offset=query])


@fieldwise_init
struct Admission(Equatable, ImplicitlyCopyable, TrivialRegisterPassable):
    """Whether a filtered search reports one candidate key or passes over it - `usearch_admission_t`."""

    var code: c_int
    """The raw `usearch_admission_t` value the ABI exchanges."""
    comptime REJECT = Self(0)
    """Skip the key; the walk still routes through it."""
    comptime ADMIT = Self(1)
    """Report the key among the matches."""


# endregion Matches

# region Signatures

comptime Verb[Arguments: AnyType] = def(Pointer[Arguments, MutAnyOrigin]) thin abi("C") -> None
"""Every struct verb: one argument structure named after it, its outputs written through its pointers."""

comptime Release = def(Address) thin abi("C") -> None
"""The positional `_free` family - executors, arenas, indexes, cursors, streams; `NULL` is safe."""

comptime OutcomeStatus = def(Pointer[Outcome, MutAnyOrigin]) thin abi("C") -> c_int
"""`usearch_outcome_wait` and `usearch_outcome_status`: an outcome's status, the first once it is final."""

comptime NameOf = def(c_int) thin abi("C") -> Optional[CString]
"""The `_name` family: an enumerator's static spelling."""

comptime FromName = def(BytesView[ImmUntrackedOrigin], Pointer[c_int, MutAnyOrigin]) thin abi("C") -> c_int
"""The `_from_name` family: reads a spelling back through its pointer, answering a status."""

comptime ExecutorForRam = Verb[ExecutorForRamArguments]
"""`usearch_executor_for_ram`."""
comptime ExecutorForVram = Verb[ExecutorForVramArguments]
"""`usearch_executor_for_vram`."""
comptime ExecutorForNvme = Verb[ExecutorForNvmeArguments]
"""`usearch_executor_for_nvme`."""
comptime ExecutorDrain = Verb[ExecutorDrainArguments]
"""`usearch_executor_drain`."""
comptime ArenaInit = Verb[ArenaInitArguments]
"""`usearch_arena_init`."""

comptime ArenaDescribe = Verb[ArenaDescribeArguments]
"""`usearch_arena_describe`."""

# endregion Signatures

# region Placement


@fieldwise_init
struct RamOptions[origin: ImmOrigin = ImmUntrackedOrigin](ImplicitlyCopyable):
    """The host memory domains a RAM-resident index places its shards in - `usearch_ram_options_t`.

    One shard per named NUMA node, so a dual-socket box names two and gets two graphs whose
    traversals stay local to the cores that walk them. The engine's payload blocks always claim
    huge pages, falling back to base pages where none are provisioned.

    Parameters:
        origin: The viewed node-id array, named so it outlives the constructor that reads it.
    """

    var memory_domains: Optional[Pointer[Int32, Self.origin]]
    """The node ids to place shards on; `None` derives every node the process may use."""
    var memory_domains_count: c_size_t
    """How many ids the array holds; zero derives."""
    var shard_capacity: c_size_t
    """Entries one shard holds before the index opens another shard generation; zero derives."""
    var cpu_class_ceiling: CPUClass
    """The newest class dispatch may pick, intersected with what the machine runs, so a class the machine
    lacks narrows rather than refuses; `UNKNOWN` means no ceiling."""
    var memory_limit: c_size_t
    """Bytes an NVMe view's key tables may take on each named domain; zero sets no ceiling."""

    def __init__(out self):
        """Derives every memory domain the process may allocate on."""
        self.memory_domains = None
        self.memory_domains_count = c_size_t(0)
        self.shard_capacity = c_size_t(0)
        self.cpu_class_ceiling = CPUClass.UNKNOWN
        self.memory_limit = c_size_t(0)

    @staticmethod
    def on[domains_origin: ImmOrigin, //](memory_domains: Span[Int32, domains_origin]) -> RamOptions[domains_origin]:
        """Names the NUMA nodes explicitly, viewing the caller's array for the call.

        Parameters:
            domains_origin: The run's origin, inferred from the argument.

        Args:
            memory_domains: The node ids to place one shard on apiece.

        Returns:
            Options naming exactly those nodes, one shard apiece.
        """
        return RamOptions[domains_origin](
            memory_domains.unsafe_ptr(), c_size_t(len(memory_domains)), c_size_t(0), CPUClass.UNKNOWN, c_size_t(0)
        )

    def arguments(self) -> RamOptions[ImmUntrackedOrigin]:
        """The same options with the view erased - the one instantiation the ABI signature names.

        Returns:
            Options naming the same array, whose type no longer keeps it alive.
        """
        var memory_domains: Optional[Pointer[Int32, ImmUntrackedOrigin]] = None
        if case .Some(domains) = self.memory_domains:
            memory_domains = domains.unsafe_origin_cast[ImmUntrackedOrigin]()
        return RamOptions[ImmUntrackedOrigin](
            memory_domains, self.memory_domains_count, self.shard_capacity, self.cpu_class_ceiling, self.memory_limit
        )


@fieldwise_init
struct VramOptions[origin: ImmOrigin = ImmUntrackedOrigin](ImplicitlyCopyable):
    """The devices a VRAM-resident index places its shards in - `usearch_vram_options_t`.

    Sharded and never replicated, so eight devices host eight slices of one graph rather than eight
    copies of it. Planes are device-resident: an index outgrowing the named devices fails rather
    than paging to the host, and the answer is to name more devices.

    Parameters:
        origin: The viewed ordinal array, named so it outlives the constructor that reads it.
    """

    var devices: Optional[Pointer[Int32, Self.origin]]
    """The CUDA ordinals to place shards on; `None` derives one free device."""
    var devices_count: c_size_t
    """How many ordinals the array holds; zero derives."""
    var shard_capacity: c_size_t
    """Entries one shard holds before the index opens another shard generation; zero derives."""
    var gpu_class_ceiling: GPUClass
    """The newest class dispatch may pick, capped below the best class every named device loads; a ceiling
    no named device loads is refused as unsupported; `UNKNOWN` means no ceiling."""

    def __init__(out self):
        """Derives one free device."""
        self.devices = None
        self.devices_count = c_size_t(0)
        self.shard_capacity = c_size_t(0)
        self.gpu_class_ceiling = GPUClass.UNKNOWN

    @staticmethod
    def on[devices_origin: ImmOrigin, //](devices: Span[Int32, devices_origin]) -> VramOptions[devices_origin]:
        """Names the device ordinals explicitly, viewing the caller's array for the call.

        Parameters:
            devices_origin: The run's origin, inferred from the argument.

        Args:
            devices: The CUDA ordinals to place one shard on apiece.

        Returns:
            Options naming exactly those devices, one shard apiece.
        """
        return VramOptions[devices_origin](devices.unsafe_ptr(), c_size_t(len(devices)), c_size_t(0), GPUClass.UNKNOWN)

    def arguments(self) -> VramOptions[ImmUntrackedOrigin]:
        """The same options with the view erased - the one instantiation the ABI signature names.

        Returns:
            Options naming the same array, whose type no longer keeps it alive.
        """
        var devices: Optional[Pointer[Int32, ImmUntrackedOrigin]] = None
        if case .Some(ordinals) = self.devices:
            devices = ordinals.unsafe_origin_cast[ImmUntrackedOrigin]()
        return VramOptions[ImmUntrackedOrigin](devices, self.devices_count, self.shard_capacity, self.gpu_class_ceiling)


@fieldwise_init
struct NvmeOptions[origin: ImmOrigin = ImmUntrackedOrigin](ImplicitlyCopyable):
    """The devices an NVMe-resident index places its shards on - `usearch_nvme_options_t`.

    A path is whatever identifies a device to the engine that can reach it, so the string is the
    extension point rather than a field per transport: a filesystem path names a kernel device, a
    transport identifier names a namespace the kernel no longer owns, and a FUSE mount is a path
    too. Naming nothing opens every shard where its manifest already points and moves no bytes.

    Parameters:
        origin: The viewed path array, named so it outlives the constructor that reads it.
    """

    var paths: Optional[Pointer[BytesView[ImmUntrackedOrigin], Self.origin]]
    """The device or directory names; `None` leaves every shard where its manifest points."""
    var paths_count: c_size_t
    """How many paths the array holds; zero derives."""
    var cpu_class_ceiling: CPUClass
    """The newest class dispatch may pick, intersected with what the machine runs, so a class the machine
    lacks narrows rather than refuses; `UNKNOWN` means no ceiling."""

    def __init__(out self):
        """Leaves every shard where its own manifest points."""
        self.paths = None
        self.paths_count = c_size_t(0)
        self.cpu_class_ceiling = CPUClass.UNKNOWN

    @staticmethod
    def on[
        paths_origin: ImmOrigin, //
    ](paths: Span[BytesView[ImmUntrackedOrigin], paths_origin]) -> NvmeOptions[paths_origin]:
        """Names the devices or directories explicitly, viewing the caller's array for the call.

        Parameters:
            paths_origin: The run's origin, inferred from the argument.

        Args:
            paths: The device or directory names, one shard apiece.

        Returns:
            Options naming exactly those paths, one shard apiece.
        """
        return NvmeOptions[paths_origin](paths.unsafe_ptr(), c_size_t(len(paths)), CPUClass.UNKNOWN)

    def arguments(self) -> NvmeOptions[ImmUntrackedOrigin]:
        """The same options with the view erased - the one instantiation the ABI signature names.

        Returns:
            Options naming the same array, whose type no longer keeps it alive.
        """
        var paths: Optional[Pointer[BytesView[ImmUntrackedOrigin], ImmUntrackedOrigin]] = None
        if case .Some(names) = self.paths:
            paths = names.unsafe_origin_cast[ImmUntrackedOrigin]()
        return NvmeOptions[ImmUntrackedOrigin](paths, self.paths_count, self.cpu_class_ceiling)


# endregion Placement

# region Executors


@fieldwise_init
struct ExecutorForRamArguments(ImplicitlyCopyable):
    """`usearch_executor_for_ram_t`, laid out as the ABI declares it."""

    var executor: Pointer[Address, MutUntrackedOrigin]
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var ram: RamOptions[ImmUntrackedOrigin]
    var threads: c_size_t
    var scratch_limit: c_size_t


@fieldwise_init
struct ExecutorForVramArguments(ImplicitlyCopyable):
    """`usearch_executor_for_vram_t`, laid out as the ABI declares it."""

    var executor: Pointer[Address, MutUntrackedOrigin]
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var vram: VramOptions[ImmUntrackedOrigin]
    var streams: Address
    var scratch_limit: c_size_t


@fieldwise_init
struct ExecutorForNvmeArguments(ImplicitlyCopyable):
    """`usearch_executor_for_nvme_t`, laid out as the ABI declares it."""

    var executor: Pointer[Address, MutUntrackedOrigin]
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var nvme: NvmeOptions[ImmUntrackedOrigin]
    var ram: RamOptions[ImmUntrackedOrigin]
    var threads: c_size_t
    var nvme_class: NVMeClass
    var scratch_limit: c_size_t


@fieldwise_init
struct ExecutorDrainArguments(ImplicitlyCopyable):
    """`usearch_executor_drain_t`, laid out as the ABI declares it."""

    var executor: Address
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var mood: Mood


struct Executor:
    """The threads or streams a verb runs on, freed when it goes out of scope, after every arena it made.

    `Executor(library)` owns nothing: every verb handed it runs on the calling thread -
    single-threaded on RAM, on the handle's device and default stream on VRAM. The spawning
    constructors fix a kind that must match the residency of every handle they drive. One call holds
    an executor at a time, and a second finding it held finishes `Status.CONTENDED`, as nothing
    queues, so threads that genuinely overlap take one apiece. A deferred call on a caller-exclusive
    pool runs on the pool's thread 0, so polling alone finishes it, while a single thread or a VRAM
    executor runs it at submission.
    """

    var library: Library
    """The loaded engine, held so its symbol table outlives every call made through the handle."""
    var address: Address
    """The opaque `usearch_executor_t` every verb hands back to the engine; `NULL_ADDRESS` runs inline."""

    def __init__(out self, library: Library):
        """The caller-thread executor, which every verb accepts whatever the handle's residency.

        Args:
            library: The loaded engine, kept so the shared object stays mapped.
        """
        self = Self(library=library, address=NULL_ADDRESS)

    def __init__(out self, *, library: Library, address: Address):
        """Adopts a handle that is already spawned, which is how the factories hand one back.

        Args:
            library: The loaded engine the handle was created through, and is freed through.
            address: A `usearch_executor_t` whose lifetime this executor takes over.
        """
        self.library = library
        self.address = address

    def __deinit__(deinit self):
        """Drains every thread or stream, releases the scratch, and frees the executor.
        Calls `usearch_executor_free`.

        The loader reference is surrendered only after the call: reading the entry point is the
        last use of `library`, so without the trailing move Mojo drops the last reference - and
        `dlclose`s the engine - between the load and the branch to it.
        """
        self.library.symbols().executor_free(self.address)
        _ = self.library^

    @staticmethod
    def for_ram[
        ram_origin: ImmOrigin, //
    ](
        library: Library, ram: RamOptions[ram_origin] = RamOptions(), threads: Int = 0, scratch_limit: Int = 0
    ) raises Error -> Self:
        """Spawns host workers for RAM-resident indexes. Calls `usearch_executor_for_ram`.

        Passing the very options the indexes were built with is what makes compute and memory
        domains agree, leaving no second place to name hardware.

        Parameters:
            ram_origin: The options' viewed node-id array, inferred from the argument.

        Args:
            library: The loaded engine, kept so the shared object stays mapped.
            ram: The same options the indexes were initialized with; copied, not viewed.
            threads: Worker threads; zero takes 7/8 of the cores those domains hold.
            scratch_limit: The ceiling on each thread's scratch bytes; zero takes 1 GiB.

        Returns:
            An executor that may drive RAM-resident handles and no others.

        Raises:
            `Error` if the domains cannot be resolved or the OS refuses the threads.
        """
        var created = stack_allocation[1, Address]()
        created[unsafe_offset=0] = NULL_ADDRESS
        var outcome = stack_allocation[1, Outcome]()
        outcome[unsafe_offset=0] = Outcome()
        var arguments = stack_allocation[1, ExecutorForRamArguments]()
        arguments[unsafe_offset=0] = ExecutorForRamArguments(
            created.unsafe_origin_cast[MutUntrackedOrigin](),
            outcome.unsafe_origin_cast[MutUntrackedOrigin](),
            ram.arguments(),
            c_size_t(threads),
            c_size_t(scratch_limit),
        )
        library.symbols().executor_for_ram(arguments.unsafe_origin_cast[MutAnyOrigin]())
        library._check(outcome.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0], "usearch_executor_for_ram")
        return Self(
            library=library,
            address=created.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0],
        )

    @staticmethod
    def for_nvme[
        nvme_origin: ImmOrigin, //
    ](
        library: Library, nvme: NvmeOptions[nvme_origin] = NvmeOptions(), threads: Int = 0, scratch_limit: Int = 0
    ) raises Error -> Self:
        """Spawns host workers for NVMe-resident indexes, one submission ring apiece.
        Calls `usearch_executor_for_nvme`.

        One ring per thread is what keeps submission lock-free: a thread owns its queue outright,
        so an external-memory walk never coordinates with a sibling to issue a read.

        Parameters:
            nvme_origin: The options' viewed path array, inferred from the argument.

        Args:
            library: The loaded engine, kept so the shared object stays mapped.
            nvme: The same options the indexes were initialized with; copied, not viewed.
            threads: Worker threads; zero derives from the devices' queue capabilities.
            scratch_limit: The ceiling on each thread's scratch bytes; zero takes 1 GiB.

        Returns:
            An executor that may drive NVMe-resident handles and no others.

        Raises:
            `Error` if a path cannot be opened or the OS refuses the threads.
        """
        var created = stack_allocation[1, Address]()
        created[unsafe_offset=0] = NULL_ADDRESS
        var outcome = stack_allocation[1, Outcome]()
        outcome[unsafe_offset=0] = Outcome()
        var arguments = stack_allocation[1, ExecutorForNvmeArguments]()
        arguments[unsafe_offset=0] = ExecutorForNvmeArguments(
            created.unsafe_origin_cast[MutUntrackedOrigin](),
            outcome.unsafe_origin_cast[MutUntrackedOrigin](),
            nvme.arguments(),
            RamOptions(),
            c_size_t(threads),
            NVMeClass.UNKNOWN,
            c_size_t(scratch_limit),
        )
        library.symbols().executor_for_nvme(arguments.unsafe_origin_cast[MutAnyOrigin]())
        library._check(outcome.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0], "usearch_executor_for_nvme")
        return Self(
            library=library,
            address=created.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0],
        )

    @staticmethod
    def for_vram[
        vram_origin: ImmOrigin, //
    ](
        library: Library,
        vram: VramOptions[vram_origin] = VramOptions(),
        streams: Address = NULL_ADDRESS,
        scratch_limit: Int = 0,
    ) raises Error -> Self:
        """Orders work on one stream on every named device, all driven by the calling thread.
        Calls `usearch_executor_for_vram`.

        One stream per device is the whole design: a single host thread keeps every device fed, and
        adding host threads would buy nothing, which is why this constructor takes no thread count.

        Parameters:
            vram_origin: The options' viewed ordinal array, inferred from the argument.

        Args:
            library: The loaded engine, kept so the shared object stays mapped.
            vram: The same options the indexes were initialized with; copied, not viewed.
            streams: The caller's CUDA streams, one per device in `vram` order, which answer
                `Status.UNSUPPORTED` until they are wired; `NULL_ADDRESS` uses the one non-blocking
                stream per device the executor opens and destroys with itself.
            scratch_limit: Ceiling on scratch bytes per device; zero lets scratch grow as geometry asks.

        Returns:
            An executor that may drive VRAM-resident handles and no others.

        Raises:
            `Error` if a device cannot be selected or its memory cannot be claimed.
        """
        var created = stack_allocation[1, Address]()
        created[unsafe_offset=0] = NULL_ADDRESS
        var outcome = stack_allocation[1, Outcome]()
        outcome[unsafe_offset=0] = Outcome()
        var arguments = stack_allocation[1, ExecutorForVramArguments]()
        arguments[unsafe_offset=0] = ExecutorForVramArguments(
            created.unsafe_origin_cast[MutUntrackedOrigin](),
            outcome.unsafe_origin_cast[MutUntrackedOrigin](),
            vram.arguments(),
            streams,
            c_size_t(scratch_limit),
        )
        library.symbols().executor_for_vram(arguments.unsafe_origin_cast[MutAnyOrigin]())
        library._check(outcome.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0], "usearch_executor_for_vram")
        return Self(
            library=library,
            address=created.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0],
        )

    def drain(self, mood: Mood = Mood.PARKED) raises Error:
        """Waits until no call holds the executor, then adopts `mood` until the next call.
        Calls `usearch_executor_drain`.

        It only waits, never runs a call itself.

        Args:
            mood: What the threads or streams do once idle; the last drain wins.

        Raises:
            `Error` for a failure of the executor itself, such as a lost device - never one of a
            drained call, which reports through its own outcome.
        """
        var outcome = stack_allocation[1, Outcome]()
        outcome[unsafe_offset=0] = Outcome()
        var arguments = stack_allocation[1, ExecutorDrainArguments]()
        arguments[unsafe_offset=0] = ExecutorDrainArguments(
            self.address, outcome.unsafe_origin_cast[MutUntrackedOrigin](), mood
        )
        self.library.symbols().executor_drain(arguments.unsafe_origin_cast[MutAnyOrigin]())
        self.library._check(outcome.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0], "usearch_executor_drain")


# endregion Executors

# region Arenas


@fieldwise_init
struct ArenaInitArguments(ImplicitlyCopyable):
    """`usearch_arena_init_t`, laid out as the ABI declares it."""

    var arena: Pointer[Address, MutUntrackedOrigin]
    var executor: Address
    var outcome: Pointer[Outcome, MutUntrackedOrigin]


@fieldwise_init
struct ArenaDescription(ImplicitlyCopyable):
    """`usearch_arena_description_t`: where an arena's outputs live."""

    var residency: Residency
    """`RAM` for host pages, `VRAM` for one device's memory."""
    var device: c_size_t
    """The device ordinal, as `Executor.for_vram` takes it, on `VRAM`; 0 for host pages."""


@fieldwise_init
struct ArenaDescribeArguments(ImplicitlyCopyable):
    """`usearch_arena_describe_t`, laid out as the ABI declares it."""

    var arena: Address
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var description: Pointer[ArenaDescription, MutUntrackedOrigin]


def _arena_init(library: Library, executor: Address) raises Error -> Address:
    """Makes an arena where `executor` computes, or host pages for a null one. Calls `usearch_arena_init`."""
    var created = stack_allocation[1, Address]()
    created[unsafe_offset=0] = NULL_ADDRESS
    var outcome = stack_allocation[1, Outcome]()
    outcome[unsafe_offset=0] = Outcome()
    var arguments = stack_allocation[1, ArenaInitArguments]()
    arguments[unsafe_offset=0] = ArenaInitArguments(
        created.unsafe_origin_cast[MutUntrackedOrigin](),
        executor,
        outcome.unsafe_origin_cast[MutUntrackedOrigin](),
    )
    library.symbols().arena_init(arguments.unsafe_origin_cast[MutAnyOrigin]())
    library._check(outcome.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0], "usearch_arena_init")
    return created.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0]


struct Arena[origin: ImmOrigin]:
    """The memory every output whose length scales with a call lands in - `usearch_arena_t`, freed
    when it goes out of scope.

    Made where an executor computes, host pages or the memory of its GPU, decided once; nothing is
    allocated until a call writes into it. Every call handed it rewinds and refills it, which
    invalidates the previous call's outputs, so a steady loop allocates once. One arena serves one
    call at a time; a second concurrent use is refused by the engine as `CONTENDED`, which leaves
    the holder's results readable.

    Every verb takes it as its last required argument, deferred ones too, and the result views it
    in turn, so the arena outlives both; a failed call leaves it valid.

    A result reads the arena in place until its next call rewinds it, and `describe` says where that
    memory lives, which a caller checks before reading a VRAM arena's results on the host. A VRAM
    arena serves only calls on the executor that made it. An arena never outlives that executor: its
    origin holds it, so Mojo frees the arena first.

    Parameters:
        origin: The executor that made it, untracked for `Arena(library)`, which no executor made.
    """

    var _library: Library
    """The loaded engine, held so the release can still reach `usearch_arena_free`."""
    var _address: Address
    """The `usearch_arena_t`, never null."""

    def __init__(out self: Arena[ImmUntrackedOrigin], library: Library) raises Error:
        """Host pages for calls on the calling thread. Calls `usearch_arena_init`.

        Args:
            library: The loaded engine, kept so the shared object stays mapped.

        Raises:
            `Error` if the engine cannot allocate the handle.
        """
        var address = _arena_init(library, NULL_ADDRESS)
        self._library = library
        self._address = address

    def __init__(out self, ref[Self.origin] executor: Executor) raises Error:
        """Memory where `executor` computes - host pages, or the memory of its GPU - serving the calls
        of its residency. Calls `usearch_arena_init`.

        Args:
            executor: The executor whose calls it serves, held until the arena is freed.

        Raises:
            `Error` if the engine cannot allocate the handle.
        """
        var address = _arena_init(executor.library, executor.address)
        self._library = executor.library
        self._address = address

    def __deinit__(deinit self):
        """Frees the arena and every output in it. Calls `usearch_arena_free`.

        The loader reference is surrendered only after the call, for the reason `Executor` records.
        """
        self._library.symbols().arena_free(self._address)
        _ = self._library^

    def describe(self) raises Error -> ArenaDescription:
        """Where its outputs live, as the executor that made it decided. Calls `usearch_arena_describe`.

        Returns:
            `VRAM` and the device for device memory, which the host cannot read, and `RAM` otherwise.

        Raises:
            `Error` if the engine refuses the query.
        """
        var description = stack_allocation[1, ArenaDescription]()
        description[unsafe_offset=0] = ArenaDescription(Residency.RAM, c_size_t(0))
        var outcome = stack_allocation[1, Outcome]()
        outcome[unsafe_offset=0] = Outcome()
        var arguments = stack_allocation[1, ArenaDescribeArguments]()
        arguments[unsafe_offset=0] = ArenaDescribeArguments(
            self._address,
            outcome.unsafe_origin_cast[MutUntrackedOrigin](),
            description.unsafe_origin_cast[MutUntrackedOrigin](),
        )
        self._library.symbols().arena_describe(arguments.unsafe_origin_cast[MutAnyOrigin]())
        self._library._check(outcome.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0], "usearch_arena_describe")
        return description.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0]


def _in_arena[T: Copyable, origin: Origin](address: Address, length: Int) -> Span[T, origin]:
    """`length` values a call wrote into an arena; a failed or empty call left `address` null."""
    if length == 0:
        return Span[T, origin]()
    return Span[T, origin](unsafe_ptr=Pointer[T, origin](unsafe_from_address=address), length=length)


@fieldwise_init
struct Monolith[origin: MutOrigin](Movable):
    """The serialized index `save_buffer` lands in its arena, read in place.

    Parameters:
        origin: The arena the bytes live in, which this result views.
    """

    var data: Address
    """The first byte, inside the arena."""
    var length: Int
    """Bytes the monolith spans."""

    def bytes(ref self) -> Span[UInt8, origin_of(self)]:
        """The monolith, viewed from the arena this result reads.

        Returns:
            The bytes, valid until the arena's next call.
        """
        return _in_arena[UInt8, origin_of(self)](self.data, self.length)


# endregion Arenas

# region Deferred Calls


struct Unwritten(Defaultable, ImplicitlyCopyable):
    """The outputs of a verb that writes nothing but its outcome."""

    def __init__(out self):
        """Holds nothing."""
        pass


@fieldwise_init
struct CallSlots[Outputs: ImplicitlyCopyable & Defaultable & Deinitable](ImplicitlyCopyable):
    """The outcome and the scalar outputs one call writes through, at one stable address."""

    var outcome: Outcome
    var outputs: Self.Outputs

    def __init__(out self):
        """Zeroes the outcome and defaults every output."""
        self.outcome = Outcome()
        self.outputs = Self.Outputs()


@fieldwise_init
struct CallHead(ImplicitlyCopyable):
    """What every verb's arguments name beside its own inputs: the index, the executor, the outcome
    and the completion. A verb naming no index reads only the last three."""

    var index: Address
    var executor: Address
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var completion: Completion


trait Call(Deinitable, Movable):
    """One verb crossing into the engine: what it passes until final, and how its outputs become a result.

    Each receiver routes the same `Call` - the index and `index.on(executor)` block on it, and
    `index.deferred(executor)` wraps it in a `Deferred` - which is what keeps the verbs symmetric.
    """

    comptime Arguments: ImplicitlyCopyable & Deinitable
    """The argument structure the entry point reads, laid out as the ABI declares it."""
    comptime Outputs: ImplicitlyCopyable & Defaultable & Deinitable
    """The fixed-size outputs the engine writes through, such as a count or a view into the arena."""
    comptime Result: Movable & Deinitable
    """What a successful call answers."""
    comptime symbol: StaticString
    """The C entry point, held into every error."""

    @staticmethod
    def verb(library: Library) -> Verb[Self.Arguments]:
        """The resolved entry point."""
        ...

    def arguments(self, head: CallHead, outputs: Pointer[Self.Outputs, MutUntrackedOrigin]) -> Self.Arguments:
        """The argument structure, its outputs pointing into `outputs`."""
        ...

    def lengths_agree(self) -> Bool:
        """Whether inputs the call reads under one count hold as many rows apiece; the binding refuses
        the call as `SHAPE_MISMATCH` otherwise, as only it sees both lengths."""
        return True

    def conclude(deinit self, outputs: Self.Outputs) -> Self.Result:
        """Adopts what the final call wrote, success or not, and builds the result around its arena."""
        ...


def _cross[Arguments: ImplicitlyCopyable & Deinitable, //](arguments: Arguments, verb: Verb[Arguments]):
    """Materializes one argument structure in a stack slot and hands its address to `verb`."""
    var slot = stack_allocation[1, Arguments]()
    slot[unsafe_offset=0] = arguments
    verb(slot.unsafe_origin_cast[MutAnyOrigin]())


def _submit[C: Call, //](library: Library, call: C, head: CallHead, outputs: Pointer[C.Outputs, MutUntrackedOrigin]):
    """Crosses into the engine, or finishes a call whose lengths disagree without crossing."""
    if not call.lengths_agree():
        head.outcome[] = Outcome(c_int(Status.SHAPE_MISMATCH.code), None)
        return
    _cross(call.arguments(head, outputs), C.verb(library))


def _block[C: Call](library: Library, var call: C, index: Address, executor: Address) raises Error -> C.Result:
    """Runs one call to its final status, its slots on the stack, and answers its result or raises."""
    var slots = stack_allocation[1, CallSlots[C.Outputs]]()
    slots[unsafe_offset=0] = CallSlots[C.Outputs]()
    var mutable_slots = slots.unsafe_origin_cast[MutAnyOrigin]()
    _submit(
        library,
        call,
        CallHead(
            index,
            executor,
            Pointer(to=mutable_slots[unsafe_offset=0].outcome).unsafe_origin_cast[MutUntrackedOrigin](),
            Completion.BLOCKING,
        ),
        Pointer(to=mutable_slots[unsafe_offset=0].outputs).unsafe_origin_cast[MutUntrackedOrigin](),
    )
    var written = mutable_slots[unsafe_offset=0]
    var result = call^.conclude(written.outputs)
    library._check(written.outcome, C.symbol)
    return result^


struct Deferred[C: Call, origin: ImmOrigin]:
    """A call submitted to an executor, which holds its outcome and outputs inline until it is final. Binding-only.

    It never moves: a receiver builds it in the caller's place and submits it there, so the engine
    writes into it directly. `status` reads `DEFERRED` until then, without blocking, and `wait`
    blocks until the call is final and answers its result or raises its failure, as many times as
    it is asked. `await` answers the same, yielding to the runtime while the call is pending. The
    result is the view a blocking call answers, viewing the same arena. Dropping an unfinished one
    waits, so nothing the engine still writes is ever freed under it. The origins keep the index,
    the executor, the arena and every viewed input alive for as long as this is.

    A caller-exclusive pool runs a deferred call on its own thread 0, so polling alone finishes it,
    while an executor without threads of its own runs it at submission; a `wait` - or the last use
    of the `Deferred`, which Mojo destroys at once - only waits.

    Parameters:
        C: The verb's crossing, which names the result.
        origin: The union of everything the call views.
    """

    var library: Library
    """The loaded engine, held so the wait can still reach `usearch_outcome_wait`."""
    var slots: CallSlots[Self.C.Outputs]
    """The outcome and outputs the engine writes."""
    var call: Optional[Self.C]
    """What the call passes the engine, until a wait concludes it."""
    var answer: Optional[Self.C.Result]
    """The result, once a wait found the call succeeded."""
    var failure: Optional[Error]
    """The failure, once a wait found the call failed."""

    def __init__(out self, library: Library, var call: Self.C, index: Address, executor: Address):
        """Submits `call` deferred, on the receiver's executor."""
        self.library = library
        self.slots = CallSlots[Self.C.Outputs]()
        self.call = call^
        self.answer = None
        self.failure = None
        var slots = self._slots()
        _submit(
            self.library,
            self.call.value(),
            CallHead(
                index,
                executor,
                Pointer(to=slots[].outcome).unsafe_origin_cast[MutUntrackedOrigin](),
                Completion.DEFERRED,
            ),
            Pointer(to=slots[].outputs).unsafe_origin_cast[MutUntrackedOrigin](),
        )

    def __deinit__(deinit self):
        """Waits until the call is final, then frees what it passed. Calls `usearch_outcome_wait`."""
        if self.call:
            var slots = Pointer[CallSlots[Self.C.Outputs], MutAnyOrigin](
                unsafe_from_address=Int(Pointer(to=self.slots))
            )
            _ = self.library.symbols().outcome_wait(Pointer(to=slots[].outcome))
            _ = self.call.take().conclude(slots[].outputs)
        _ = self.library^

    def _slots(self) -> Pointer[CallSlots[Self.C.Outputs], MutAnyOrigin]:
        """The slots, through the origin the engine's writes need."""
        return Pointer[CallSlots[Self.C.Outputs], MutAnyOrigin](unsafe_from_address=Int(Pointer(to=self.slots)))

    def status(self) -> Status:
        """`DEFERRED` until the call is final, then its final status; never blocks. Calls `usearch_outcome_status`."""
        return Status(self.library.symbols().outcome_status(Pointer(to=self._slots()[].outcome)))

    def wait(mut self) raises Error -> ref[self] Self.C.Result:
        """Blocks until the call is final, then answers its result or raises its failure, every time it
        is asked. Calls `usearch_outcome_wait`.

        Returns:
            The result, owned by this `Deferred` and viewed for as long as it is.

        Raises:
            `Error` holding the call's failure, whether the engine or the binding refused it.
        """
        if self.call:
            var slots = self._slots()
            _ = self.library.symbols().outcome_wait(Pointer(to=slots[].outcome))
            var written = slots[]
            var result = self.call.take().conclude(written.outputs)
            if written.outcome.status == c_int(0):
                self.answer = result^
            else:
                self.failure = self.library._error(written.outcome, Self.C.symbol)
        if case .Some(failure) = self.failure:
            raise failure
        return Pointer(to=self.answer.value()).unsafe_origin_cast[origin_of(self)]()[]

    @always_inline
    def __await__(mut self) raises Error -> ref[self] Self.C.Result:
        """Yields to the runtime until the call is final, then answers as `wait` does.
        Calls `usearch_outcome_status`.

        Returns:
            The result, owned by this `Deferred` and viewed for as long as it is.

        Raises:
            `Error` carrying the call's failure, whether the engine or the binding refused it.
        """
        while self.status() == Status.DEFERRED:

            @always_inline
            def requeue(handle: AnyCoroutine):
                _async_execute[NoneType](handle, -1)

            _suspend_async(requeue)
        return self.wait()


# endregion Deferred Calls
