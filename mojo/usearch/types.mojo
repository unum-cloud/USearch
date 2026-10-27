"""The vocabulary every modality shares: keys, scalars, rows, residencies, and the one error type.

Nothing dense-specific lives here - the metric kinds, the result shapes only a dense index answers,
and the index itself belong to @c dense.mojo, so a second modality adds a module rather than
widening this one.

`Error` is the only error type the binding raises: Mojo 1.1 allows at most one per function, so a
second would force every caller raising `Error` to catch and convert. A refusal is a failure, not an absence - a key the index does not hold answers zero, but a
call the engine declines raises.

Four Mojo 1.1 facts shape the descriptors and the pointer aliases below, and all four were measured
rather than assumed.

`MutAnyOrigin` rather than `MutUntrackedOrigin` is load-bearing on anything C writes through: with
a narrower origin the compiler folds a read of the slot back to whatever the caller last stored
there, and every out-parameter reads back as zero.

Mojo's `Pointer` is non-nullable by construction, so a C pointer that may be null crosses as
`Address` - its numeric value - rather than as a pointer type that cannot represent the null the
ABI asks for.

Every descriptor that borrows caller storage carries that storage's origin as a struct parameter,
because Mojo destroys a value at its last use rather than at the end of its scope, and an address
laundered through `Address` is not a use. A descriptor built from a list and handed straight to a
call was measured destroying that list before the call ran, leaving the engine reading a freed
allocation; naming the origin in the descriptor's type moved the destruction after the call. The
parameter alone does the work - no field need mention it - so the pointer fields keep the untracked
types the C layout needs, and `untracked` erases the parameter at the boundary to reach the one
instantiation the ABI signatures name. A `Deferred` carries the union of every origin its call
borrows the same way, which keeps the index, the executor, and the inputs alive until it is gone.

A struct whose address crosses to C is materialized into a `stack_allocation` slot rather than held
in a local whose address is taken. An `alloca` is not a value Mojo owns, so nothing destroys it and
nothing folds its store away, where a local's last tracked use is the address-taking itself. A
deferred call's slots outlive the verb that submitted them, so they live on the heap instead.

File: mojo/usearch/types.mojo
Author: Ash Vardanian
Date: September 7, 2026
"""

from std.bit import byte_swap
from std.ffi import c_int, c_size_t
from std.memory import ArcPointer, OwnedPointer, stack_allocation
from std.sys.info import size_of

from usearch.library import Library

# region Pointers

comptime CString = Pointer[Int8, ImmUntrackedOrigin]
"""A borrowed null-terminated C string, which only the introspection getters return."""

comptime Address = Int
"""A C pointer crossing as its numeric value, which is how a null one is expressed.

Mojo's `Pointer` is non-nullable by construction - constructing one from zero is refused at compile
time and aborts at run time - so every ABI slot that accepts `NULL` takes this instead. The
integer is pointer-width and shares the pointer's argument class on both x86-64 and AArch64, so it
lands in the same register a pointer would.
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

    An `Error` never carries `SUCCESS` or `DEFERRED`: only a `Deferred` reports the latter, and only
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
    """The call was refused rather than waited for, like a submission to a full deferred table."""
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
            The matching status, or one carrying the unrecognized code verbatim.
        """
        return Self(Int32(code))


@fieldwise_init
struct Outcome(ImplicitlyCopyable, TrivialRegisterPassable):
    """The `usearch_outcome_t` every verb reports through, laid out as the ABI declares it.

    Never marked as anything richer than this: the engine writes the two words and the binding
    reads them back, so the struct exists to have a stable address rather than an interface.
    """

    var status: c_int
    """The `usearch_status_t` the call reported."""
    var message: Address
    """The engine's own static message, or `NULL_ADDRESS` when the call succeeded."""

    def __init__(out self):
        """Zeroes the slot, so a call that writes nothing still reads as a success."""
        self.status = c_int(0)
        self.message = NULL_ADDRESS


@fieldwise_init
struct Error(Copyable, ImplicitlyCopyable, Writable):
    """What went wrong inside the engine, and which entry point it was about."""

    var status: Status
    """Which class of failure the engine reported."""
    var detail: StaticString
    """The entry point it was about, for a reader rather than for a branch."""
    var message: Address
    """The engine's own static message, or `NULL_ADDRESS` when it carried none.

    Messages are static storage inside the shared object, so keeping the address costs nothing and
    outlives the call. `write_to` walks it; there is nothing to free.
    """
    var name: Address
    """The engine's spelling of `status` from `usearch_status_name`, or `NULL_ADDRESS` where no engine named it."""
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
        self.message = NULL_ADDRESS
        self.name = NULL_ADDRESS
        self.library = None

    def write_to(self, mut writer: Some[Writer]):
        """Writes the status name, the entry point, and the engine's own message together.

        Args:
            writer: The sink the message is appended to.
        """
        writer.write("USearch: ")
        if self.name != NULL_ADDRESS:
            writer.write(text_at(self.name))
        elif self.status == Status.LIBRARY_MISSING:
            writer.write("library-missing")
        elif self.status == Status.SYMBOL_MISSING:
            writer.write("symbol-missing")
        elif self.status == Status.VERSION_MISMATCH:
            writer.write("version-mismatch")
        else:
            writer.write("status ", self.status.code)
        writer.write(" [", self.detail, "]")
        if self.message != NULL_ADDRESS:
            writer.write(": ", text_at(self.message))


def text_at(address: Address) -> String:
    """Decodes a null-terminated C string the engine owns, stopping at its terminator.

    Args:
        address: Where the bytes start; `NULL_ADDRESS` reads as the empty string.

    Returns:
        A copy of the bytes, since the caller may keep it longer than it wants a raw address.
    """
    if address == NULL_ADDRESS:
        return String("")
    var bytes = Pointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=address)
    var length = 0
    while bytes[unsafe_offset=length] != 0:
        length += 1
    return String(StringSlice(unsafe_from_utf8=Span[UInt8, ImmUntrackedOrigin](unsafe_ptr=bytes, length=length)))


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
            The trailing word read big-endian, or `None` for a key whose leading bytes carry
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
        comptime if dtype == DType.float64:
            return Self.F64
        elif dtype == DType.float32:
            return Self.F32
        elif dtype == DType.bfloat16:
            return Self.BF16
        elif dtype == DType.float16:
            return Self.F16
        elif dtype == DType.float8_e5m2:
            return Self.E5M2
        elif dtype == DType.float8_e4m3fn:
            return Self.E4M3
        elif dtype == DType.int32:
            return Self.I32
        elif dtype == DType.int8:
            return Self.I8
        elif dtype == DType.uint8:
            return Self.U8
        else:
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
    through the name tables: `Library.residency_name` and `Library.residency_from_name`.
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
    """How a search reaches the indexed rows it scores - `usearch_traversal_t`."""

    var code: c_int
    """The raw `usearch_traversal_t` value the ABI exchanges."""
    comptime WALK = Self(1)
    """Walk the graph, scoring the rows the expansion reaches - approximate, the default."""
    comptime SCAN = Self(2)
    """Score every row published when the query's scan starts - exact, on RAM and NVMe handles."""


@fieldwise_init
struct Mood(Equatable, ImplicitlyCopyable, TrivialRegisterPassable):
    """What an executor does with its threads or streams once it goes idle - `usearch_mood_t`."""

    var code: c_int
    """The raw `usearch_mood_t` value the ABI exchanges."""
    comptime PARKED = Self(1)
    """Idle threads sleep; the next call pays a wake-up."""
    comptime SPINNING = Self(2)
    """Idle threads stay hot, trading power for the next call's latency."""


@fieldwise_init
struct CpuClass(Equatable, ImplicitlyCopyable, TrivialRegisterPassable):
    """The CPU family dispatch picks, or a placement pins - `usearch_cpu_class_t`."""

    var code: c_int
    """The raw `usearch_cpu_class_t` value the ABI exchanges."""
    comptime UNKNOWN = Self(0)
    """Derives the CPU's own class."""


@fieldwise_init
struct GpuClass(Equatable, ImplicitlyCopyable, TrivialRegisterPassable):
    """The GPU family dispatch picks, or a placement pins - `usearch_gpu_class_t`."""

    var code: c_int
    """The raw `usearch_gpu_class_t` value the ABI exchanges."""
    comptime UNKNOWN = Self(0)
    """Derives the devices' own class."""


@fieldwise_init
struct NvmeClass(Equatable, ImplicitlyCopyable, TrivialRegisterPassable):
    """The NVMe I/O engine family an executor drives - `usearch_nvme_class_t`."""

    var code: c_int
    """The raw `usearch_nvme_class_t` value the ABI exchanges."""
    comptime UNKNOWN = Self(0)
    """Derives the devices' own class."""


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
        """No handler, which is all this binding installs."""
        self.callback = NULL_ADDRESS
        self.state = NULL_ADDRESS
        self.steps = c_size_t(0)


@fieldwise_init
struct PredicateCost(Equatable, ImplicitlyCopyable, TrivialRegisterPassable):
    """How costly one predicate call is, which decides how many keys each call carries."""

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
    """A `usearch_predicate_callback_t` compiled in C; `NULL_ADDRESS` admits every key."""
    var state: Address
    """Forwarded into every `callback` call."""
    var cost: PredicateCost
    """How many keys each `callback` call carries."""


# endregion Residency & Execution

# region Byte Runs


@fieldwise_init
struct BytesView[origin: ImmOrigin = ImmUntrackedOrigin](ImplicitlyCopyable, TrivialRegisterPassable):
    """A borrowed run of bytes and its length - `usearch_bytes_view_t`, never null-terminated.

    Crosses by value, which the two-word argument class handles on both x86-64 and AArch64.

    Parameters:
        origin: The borrowed storage, named so it outlives every call this view crosses.
    """

    var data: CString
    """The first byte, borrowed for the duration of the call."""
    var length: c_size_t
    """How many bytes the run covers, since nothing terminates it."""

    @staticmethod
    def of[text_origin: ImmOrigin, //](text: StringSlice[text_origin]) -> BytesView[text_origin]:
        """Borrows a Mojo string's bytes, which the caller keeps alive across the call.

        Parameters:
            text_origin: The string's origin, inferred from the argument.

        Args:
            text: The characters to hand over.

        Returns:
            A view of exactly those bytes, with no terminator manufactured.
        """
        return BytesView[text_origin](
            text.unsafe_ptr().unsafe_bitcast[Int8]().unsafe_origin_cast[ImmUntrackedOrigin](),
            c_size_t(text.byte_length()),
        )

    @staticmethod
    def of[bytes_origin: ImmOrigin, //](bytes: Span[UInt8, bytes_origin]) -> BytesView[bytes_origin]:
        """Borrows a run of raw bytes, such as the monolith `save_buffer` returns.

        Parameters:
            bytes_origin: The run's origin, inferred from the argument.

        Args:
            bytes: The bytes to hand over.

        Returns:
            A view of exactly those bytes.
        """
        return BytesView[bytes_origin](
            CString(unsafe_from_address=Int(bytes.unsafe_ptr())),
            c_size_t(len(bytes)),
        )

    def untracked(self) -> BytesView[ImmUntrackedOrigin]:
        """The same run with its borrow erased - the one instantiation the ABI signatures name.

        Returns:
            A view of the same bytes, whose type no longer keeps their storage alive.
        """
        return BytesView[ImmUntrackedOrigin](self.data, self.length)


# endregion Byte Runs

# region Key Runs


@fieldwise_init
struct KeysView[origin: ImmOrigin = ImmUntrackedOrigin](ImplicitlyCopyable, TrivialRegisterPassable):
    """A borrowed, strided run of keys - `usearch_keys_view_t`, one per row of the call's count.

    Carries no count: the verb's own count is the one batch axis. A zero stride repeats the first
    key for every row.

    Parameters:
        origin: The borrowed storage, named so it outlives every call this view crosses.
    """

    var data: Pointer[Key, ImmUntrackedOrigin]
    """The first key, borrowed for the duration of the call."""
    var stride: c_size_t
    """Bytes between consecutive keys; zero repeats the first."""

    @staticmethod
    def of[keys_origin: ImmOrigin, //](keys: Span[Key, keys_origin]) -> KeysView[keys_origin]:
        """Borrows a contiguous run of keys as the view the engine reads.

        Parameters:
            keys_origin: The run's origin, inferred from the argument.

        Args:
            keys: The identities the batch is published under.

        Returns:
            A view of exactly those keys.
        """
        return KeysView[keys_origin](
            Pointer[Key, ImmUntrackedOrigin](unsafe_from_address=Int(keys.unsafe_ptr())),
            c_size_t(size_of[Key]()),
        )

    def untracked(self) -> KeysView[ImmUntrackedOrigin]:
        """The same run with its borrow erased - the one instantiation the ABI layouts name.

        Returns:
            A view of the same keys, whose type no longer keeps their storage alive.
        """
        return KeysView[ImmUntrackedOrigin](self.data, self.stride)


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
            key: The identity to carry.

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
struct RowsView[origin: ImmOrigin = ImmUntrackedOrigin](ImplicitlyCopyable, TrivialRegisterPassable):
    """A borrowed run of rows of one width - `usearch_rows_view_t`, the batch currency every verb reads.

    One vector is a one-row view. The row count becomes the verb's `count`, and the column count
    crosses with the rows, so the engine checks it against the geometry the index froze. The row
    and column counts are scalars and `stride` alone is in bytes; a zero stride repeats the first
    row for every one of them, as BLAS reads it.

    Parameters:
        origin: The borrowed rows, named so they outlive every call this view crosses.
    """

    var data: Pointer[NoneType, ImmUntrackedOrigin]
    """Base pointer of the first row; non-null, since a Mojo pointer cannot be otherwise."""
    var rows: c_size_t
    """Rows the view carries, which the verb takes as its `count`."""
    var columns: c_size_t
    """Logical scalars per row, as the caller counts them."""
    var stride: c_size_t
    """Bytes between the starts of consecutive rows; zero repeats the first row."""
    var scalar: ScalarKind
    """The scalar type the rows hold, converted on the way in where it differs."""

    @staticmethod
    def of[
        dtype: DType, rows_origin: ImmOrigin, //
    ](
        rows: Span[Scalar[dtype], rows_origin],
        columns: Int,
        scalar: ScalarKind = ScalarKind.of[dtype](),
    ) raises Error -> RowsView[rows_origin]:
        """Reads a flat run of scalars as contiguous rows of the given width. Binding-only.

        The row count comes from the bytes the run spans divided by the bytes one row of
        `scalar` occupies, so a kind that disagrees with `dtype` is refused here rather than
        read past the end by the engine. A sub-byte kind counts logical dimensions, so eight
        `U1X8` columns are one `UInt8`.

        Parameters:
            dtype: The scalar type of the run, inferred from the argument.
            rows_origin: The run's origin, inferred with it.

        Args:
            rows: The scalars, laid out row after row with no padding between them.
            columns: Logical scalars per row, which must divide the run into whole rows.
            scalar: How the engine should read those scalars; the kind `dtype` stores by default.

        Returns:
            A view of the whole run, strided at its natural row width.

        Raises:
            `Error` when the run does not divide into whole rows of that width and kind.
        """
        var row_bytes = scalar.bytes_for_columns(columns)
        var spanned = len(rows) * size_of[Scalar[dtype]]()
        if columns <= 0 or row_bytes == 0 or spanned % row_bytes != 0:
            raise Error(Status.SHAPE_MISMATCH, "RowsView.of")
        return RowsView[rows_origin](
            Pointer[NoneType, ImmUntrackedOrigin](unsafe_from_address=Int(rows.unsafe_ptr())),
            c_size_t(spanned // row_bytes),
            c_size_t(columns),
            c_size_t(row_bytes),
            scalar,
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
    inside the arena, and one count per row."""

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
    inside the arena, and one more offset than rows."""

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
    """`usearch_keys_tape_view_t`, laid out as the ABI declares it: every row's keys end to end
    inside the arena, and one more offset than rows."""

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
    count is the number of matches written to its row, never more than `wanted`.

    Parameters:
        origin: The arena the run lives in, which this result borrows.
    """

    var arena: Pointer[Arena, Self.origin]
    """The arena holding the run."""
    var generation: Int
    """The arena's call this run came from, so a read after a later call is refused."""
    var view: MatchesViewArguments
    """Where the engine wrote the keys, scores and counts, inside the arena."""
    var queries: Int
    """Query rows the call answered."""

    def __len__(self) -> Int:
        """The query rows the call answered.

        Returns:
            One per query row of that call.
        """
        return self.queries

    def row_keys(ref self, row: Int) raises Error -> Span[Key, origin_of(self)]:
        """One row's keys, nearest first, trimmed to the matches that row found.

        Args:
            row: The query row, below `len(self)`.

        Returns:
            The keys, borrowed from the arena this result reads.

        Raises:
            `Error` with `UNINITIALIZED` once a later call rewound the arena, or `DEVICE` over VRAM.
        """
        _readable(self.arena, self.generation, "Matches.row_keys")
        var start = self.view.keys + row * Int(self.view.columns) * size_of[Key]()
        return _in_arena[Key, origin_of(self)](start, self._found(row))

    def row_distances(ref self, row: Int) raises Error -> Span[Distance, origin_of(self)]:
        """One row's scores, paired positionally with `row_keys`.

        Args:
            row: The query row, below `len(self)`.

        Returns:
            The scores, borrowed from the arena this result reads.

        Raises:
            `Error` with `UNINITIALIZED` once a later call rewound the arena, or `DEVICE` over VRAM.
        """
        _readable(self.arena, self.generation, "Matches.row_distances")
        var start = self.view.scores + row * Int(self.view.columns) * size_of[Distance]()
        return _in_arena[Distance, origin_of(self)](start, self._found(row))

    def _found(self, row: Int) -> Int:
        """The matches written to one row; a pull with nothing to report left the counts null."""
        if self.view.counts == NULL_ADDRESS:
            return 0
        return Int(Pointer[c_size_t, ImmUntrackedOrigin](unsafe_from_address=self.view.counts)[unsafe_offset=row])


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

comptime OutcomeStatus = def(Address) thin abi("C") -> c_int
"""`usearch_outcome_wait` and `usearch_outcome_status`: an outcome's status, the first once it is final."""

comptime NameOf = def(c_int) thin abi("C") -> Address
"""The `_name` family: an enumerator's static spelling."""

comptime FromName = def(BytesView[ImmUntrackedOrigin]) thin abi("C") -> c_int
"""The `_from_name` family: reads a spelling back into its enumerator."""

comptime ExecutorForRam = Verb[ExecutorForRamArguments]
"""`usearch_executor_for_ram`."""
comptime ExecutorOnPool = Verb[ExecutorOnPoolArguments]
"""`usearch_executor_on_pool`."""
comptime ExecutorForVram = Verb[ExecutorForVramArguments]
"""`usearch_executor_for_vram`."""
comptime ExecutorForNvme = Verb[ExecutorForNvmeArguments]
"""`usearch_executor_for_nvme`."""
comptime ExecutorDrain = Verb[ExecutorDrainArguments]
"""`usearch_executor_drain`."""
comptime ArenaInit = Verb[ArenaInitArguments]
"""`usearch_arena_init`."""

# endregion Signatures

# region Placement


@fieldwise_init
struct RamOptions[origin: ImmOrigin = ImmUntrackedOrigin](ImplicitlyCopyable):
    """The host memory domains a RAM-resident index places its shards in - `usearch_ram_options_t`.

    One shard per named NUMA node, so a dual-socket box names two and gets two graphs whose
    traversals stay local to the cores that walk them. The engine's payload blocks always claim
    huge pages, falling back to base pages where none are provisioned.

    Parameters:
        origin: The borrowed node-id array, named so it outlives the constructor that reads it.
    """

    var memory_domains: Address
    """The `int32_t` node ids to place shards on; `NULL_ADDRESS` derives every node the process may use."""
    var memory_domains_count: c_size_t
    """How many ids the array holds; zero derives."""
    var shard_capacity: c_size_t
    """Entries one shard holds before the index opens another shard generation; zero derives."""
    var cpu_class: CpuClass
    """The class to dispatch as; `UNKNOWN` derives the CPU's own."""

    def __init__(out self):
        """Derives every memory domain the process may allocate on."""
        self.memory_domains = NULL_ADDRESS
        self.memory_domains_count = c_size_t(0)
        self.shard_capacity = c_size_t(0)
        self.cpu_class = CpuClass.UNKNOWN

    @staticmethod
    def on[domains_origin: ImmOrigin, //](memory_domains: Span[Int32, domains_origin]) -> RamOptions[domains_origin]:
        """Names the NUMA nodes explicitly, borrowing the caller's array for the call.

        Parameters:
            domains_origin: The run's origin, inferred from the argument.

        Args:
            memory_domains: The node ids to place one shard on apiece.

        Returns:
            Options naming exactly those nodes, one shard apiece.
        """
        return RamOptions[domains_origin](
            Int(memory_domains.unsafe_ptr()), c_size_t(len(memory_domains)), c_size_t(0), CpuClass.UNKNOWN
        )

    def untracked(self) -> RamOptions[ImmUntrackedOrigin]:
        """The same options with the borrow erased - the one instantiation the ABI signature names.

        Returns:
            Options naming the same array, whose type no longer keeps it alive.
        """
        return RamOptions[ImmUntrackedOrigin](
            self.memory_domains, self.memory_domains_count, self.shard_capacity, self.cpu_class
        )


@fieldwise_init
struct VramOptions[origin: ImmOrigin = ImmUntrackedOrigin](ImplicitlyCopyable):
    """The devices a VRAM-resident index places its shards in - `usearch_vram_options_t`.

    Sharded and never replicated, so eight devices host eight slices of one graph rather than eight
    copies of it. Planes are device-resident: an index outgrowing the named devices fails rather
    than paging to the host, and the answer is to name more devices.

    Parameters:
        origin: The borrowed ordinal array, named so it outlives the constructor that reads it.
    """

    var devices: Address
    """The CUDA ordinals to place shards on; `NULL_ADDRESS` derives one free device."""
    var devices_count: c_size_t
    """How many ordinals the array holds; zero derives."""
    var shard_capacity: c_size_t
    """Entries one shard holds before the index opens another shard generation; zero derives."""
    var gpu_class: GpuClass
    """The class to dispatch as; `UNKNOWN` derives the devices' own."""

    def __init__(out self):
        """Derives one free device."""
        self.devices = NULL_ADDRESS
        self.devices_count = c_size_t(0)
        self.shard_capacity = c_size_t(0)
        self.gpu_class = GpuClass.UNKNOWN

    @staticmethod
    def on[devices_origin: ImmOrigin, //](devices: Span[Int32, devices_origin]) -> VramOptions[devices_origin]:
        """Names the device ordinals explicitly, borrowing the caller's array for the call.

        Parameters:
            devices_origin: The run's origin, inferred from the argument.

        Args:
            devices: The CUDA ordinals to place one shard on apiece.

        Returns:
            Options naming exactly those devices, one shard apiece.
        """
        return VramOptions[devices_origin](
            Int(devices.unsafe_ptr()), c_size_t(len(devices)), c_size_t(0), GpuClass.UNKNOWN
        )

    def untracked(self) -> VramOptions[ImmUntrackedOrigin]:
        """The same options with the borrow erased - the one instantiation the ABI signature names.

        Returns:
            Options naming the same array, whose type no longer keeps it alive.
        """
        return VramOptions[ImmUntrackedOrigin](self.devices, self.devices_count, self.shard_capacity, self.gpu_class)


@fieldwise_init
struct NvmeOptions[origin: ImmOrigin = ImmUntrackedOrigin](ImplicitlyCopyable):
    """The devices an NVMe-resident index places its shards on - `usearch_nvme_options_t`.

    A path is whatever identifies a device to the engine that can reach it, so the string is the
    extension point rather than a field per transport: a filesystem path names a kernel device, a
    transport identifier names a namespace the kernel no longer owns, and a FUSE mount is a path
    too. Naming nothing opens every shard where its manifest already points and moves no bytes.

    Parameters:
        origin: The borrowed path array, named so it outlives the constructor that reads it.
    """

    var paths: Address
    """A `usearch_bytes_view_t` array; `NULL_ADDRESS` leaves every shard where its manifest points."""
    var paths_count: c_size_t
    """How many paths the array holds; zero derives."""
    var cpu_class: CpuClass
    """The class to dispatch as; `UNKNOWN` derives the CPU's own."""

    def __init__(out self):
        """Leaves every shard where its own manifest points."""
        self.paths = NULL_ADDRESS
        self.paths_count = c_size_t(0)
        self.cpu_class = CpuClass.UNKNOWN

    @staticmethod
    def on[
        paths_origin: ImmOrigin, //
    ](paths: Span[BytesView[ImmUntrackedOrigin], paths_origin]) -> NvmeOptions[paths_origin]:
        """Names the devices or directories explicitly, borrowing the caller's array for the call.

        Parameters:
            paths_origin: The run's origin, inferred from the argument.

        Args:
            paths: The device or directory names, one shard apiece.

        Returns:
            Options naming exactly those paths, one shard apiece.
        """
        return NvmeOptions[paths_origin](Int(paths.unsafe_ptr()), c_size_t(len(paths)), CpuClass.UNKNOWN)

    def untracked(self) -> NvmeOptions[ImmUntrackedOrigin]:
        """The same options with the borrow erased - the one instantiation the ABI signature names.

        Returns:
            Options naming the same array, whose type no longer keeps it alive.
        """
        return NvmeOptions[ImmUntrackedOrigin](self.paths, self.paths_count, self.cpu_class)


# endregion Placement

# region Executors


@fieldwise_init
struct ExecutorForRamArguments(ImplicitlyCopyable):
    """`usearch_executor_for_ram_t`, laid out as the ABI declares it."""

    var executor: Address
    var outcome: Address
    var ram: RamOptions[ImmUntrackedOrigin]
    var threads: c_size_t
    var deferred_limit: c_size_t


@fieldwise_init
struct ExecutorOnPoolArguments(ImplicitlyCopyable):
    """`usearch_executor_on_pool_t`, laid out as the ABI declares it."""

    var executor: Address
    var outcome: Address
    var ram: RamOptions[ImmUntrackedOrigin]
    var pool: Address
    var deferred_limit: c_size_t


@fieldwise_init
struct ExecutorForVramArguments(ImplicitlyCopyable):
    """`usearch_executor_for_vram_t`, laid out as the ABI declares it."""

    var executor: Address
    var outcome: Address
    var vram: VramOptions[ImmUntrackedOrigin]
    var streams: Address
    var scratch_limit: c_size_t
    var deferred_limit: c_size_t


@fieldwise_init
struct ExecutorForNvmeArguments(ImplicitlyCopyable):
    """`usearch_executor_for_nvme_t`, laid out as the ABI declares it."""

    var executor: Address
    var outcome: Address
    var nvme: NvmeOptions[ImmUntrackedOrigin]
    var threads: c_size_t
    var deferred_limit: c_size_t
    var nvme_class: NvmeClass


@fieldwise_init
struct ExecutorDrainArguments(ImplicitlyCopyable):
    """`usearch_executor_drain_t`, laid out as the ABI declares it."""

    var executor: Address
    var outcome: Address
    var mood: Mood


struct _ExecutorHandle:
    """The `usearch_executor_t` an `Executor` owns, shared with every VRAM arena it made and freed with
    the last of them, since such an arena grows, fills and frees on the executor's first stream."""

    var library: Library
    var address: Address

    def __init__(out self, library: Library, address: Address):
        """Takes over a spawned handle, or the null one the caller-thread executor names."""
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


struct Executor:
    """The threads or streams a verb runs on, freed once it and every VRAM arena it made go out of scope.

    `Executor(library)` owns nothing: every verb handed it runs on the calling thread -
    single-threaded on RAM, on the handle's device and default stream on VRAM. The spawning
    constructors fix a kind that must match the residency of every handle they drive. Calls on one
    executor start in submission order, so threads that genuinely overlap take one apiece.
    """

    var library: Library
    """The loaded engine, held so its symbol table outlives every call made through the handle."""
    var address: Address
    """The opaque `usearch_executor_t` every verb hands back to the engine; `NULL_ADDRESS` runs inline."""
    var residency: Residency
    """Where the arenas it makes live; the caller-thread executor makes host pages."""
    var _handle: ArcPointer[_ExecutorHandle]
    """The handle's owner, which a VRAM arena shares so the executor outlives it."""

    def __init__(out self, library: Library):
        """The caller-thread executor, which every verb accepts whatever the handle's residency.

        Args:
            library: The loaded engine, kept so the shared object stays mapped.
        """
        self = Self(library=library, address=NULL_ADDRESS, residency=Residency.RAM)

    def __init__(out self, *, library: Library, address: Address, residency: Residency):
        """Adopts a handle that is already spawned, which is how the factories hand one back.

        Args:
            library: The loaded engine the handle was created through, and is freed through.
            address: A `usearch_executor_t` whose lifetime this executor takes over.
            residency: The residency of the handles it drives, which places the arenas it makes.
        """
        self.library = library
        self.address = address
        self.residency = residency
        self._handle = ArcPointer(_ExecutorHandle(library, address))

    @staticmethod
    def for_ram[
        ram_origin: ImmOrigin, //
    ](
        library: Library,
        ram: RamOptions[ram_origin] = RamOptions(),
        threads: Int = 0,
        deferred_limit: Int = 0,
    ) raises Error -> Self:
        """Spawns host workers for RAM-resident indexes. Calls `usearch_executor_for_ram`.

        Passing the very options the indexes were built with is what makes compute and memory
        domains agree, leaving no second place to name hardware.

        Parameters:
            ram_origin: The options' borrowed node-id array, inferred from the argument.

        Args:
            library: The loaded engine, kept so the shared object stays mapped.
            ram: The same options the indexes were initialized with; copied, not borrowed.
            threads: Worker threads; zero takes 7/8 of the cores those domains carry.
            deferred_limit: Deferred calls it tracks at once, sizing its deferred table; zero takes a default.

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
            Int(created), Int(outcome), ram.untracked(), c_size_t(threads), c_size_t(deferred_limit)
        )
        library.symbols().executor_for_ram(arguments.unsafe_origin_cast[MutAnyOrigin]())
        library._check(outcome.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0], "usearch_executor_for_ram")
        return Self(
            library=library,
            address=created.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0],
            residency=Residency.RAM,
        )

    @staticmethod
    def on_pool[
        ram_origin: ImmOrigin, //
    ](
        library: Library,
        pool: Address,
        ram: RamOptions[ram_origin] = RamOptions(),
        deferred_limit: Int = 0,
    ) raises Error -> Self:
        """Schedules onto a borrowed ForkUnion pool, so several libraries share one set of threads.
        Calls `usearch_executor_on_pool`.

        Parameters:
            ram_origin: The options' borrowed node-id array, inferred from the argument.

        Args:
            library: The loaded engine, kept so the shared object stays mapped.
            pool: A live ForkUnion `fu_pool_t`, which must outlive the executor.
            ram: The same options the indexes were initialized with; copied, not borrowed.
            deferred_limit: Deferred calls it tracks at once, sizing its deferred table; zero takes a default.

        Returns:
            An executor that may drive RAM-resident handles and no others.

        Raises:
            `Error` if the pool or the domains are refused.
        """
        var created = stack_allocation[1, Address]()
        created[unsafe_offset=0] = NULL_ADDRESS
        var outcome = stack_allocation[1, Outcome]()
        outcome[unsafe_offset=0] = Outcome()
        var arguments = stack_allocation[1, ExecutorOnPoolArguments]()
        arguments[unsafe_offset=0] = ExecutorOnPoolArguments(
            Int(created), Int(outcome), ram.untracked(), pool, c_size_t(deferred_limit)
        )
        library.symbols().executor_on_pool(arguments.unsafe_origin_cast[MutAnyOrigin]())
        library._check(outcome.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0], "usearch_executor_on_pool")
        return Self(
            library=library,
            address=created.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0],
            residency=Residency.RAM,
        )

    @staticmethod
    def for_nvme[
        nvme_origin: ImmOrigin, //
    ](
        library: Library,
        nvme: NvmeOptions[nvme_origin] = NvmeOptions(),
        threads: Int = 0,
        deferred_limit: Int = 0,
    ) raises Error -> Self:
        """Spawns host workers for NVMe-resident indexes, one submission ring apiece.
        Calls `usearch_executor_for_nvme`.

        One ring per thread is what keeps submission lock-free: a thread owns its queue outright,
        so an external-memory walk never coordinates with a sibling to issue a read.

        Parameters:
            nvme_origin: The options' borrowed path array, inferred from the argument.

        Args:
            library: The loaded engine, kept so the shared object stays mapped.
            nvme: The same options the indexes were initialized with; copied, not borrowed.
            threads: Worker threads; zero derives from the devices' queue capabilities.
            deferred_limit: Deferred calls it tracks at once, sizing its deferred table; zero takes a default.

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
            Int(created), Int(outcome), nvme.untracked(), c_size_t(threads), c_size_t(deferred_limit), NvmeClass.UNKNOWN
        )
        library.symbols().executor_for_nvme(arguments.unsafe_origin_cast[MutAnyOrigin]())
        library._check(outcome.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0], "usearch_executor_for_nvme")
        return Self(
            library=library,
            address=created.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0],
            residency=Residency.NVME,
        )

    @staticmethod
    def for_vram[
        vram_origin: ImmOrigin, //
    ](
        library: Library,
        vram: VramOptions[vram_origin] = VramOptions(),
        streams: Address = NULL_ADDRESS,
        scratch_limit: Int = 0,
        deferred_limit: Int = 0,
    ) raises Error -> Self:
        """Opens one submission stream on every named device, all driven by the calling thread.
        Calls `usearch_executor_for_vram`.

        One stream per device is the whole design: a single host thread keeps every device fed, and
        adding host threads would buy nothing, which is why this constructor takes no thread count.

        Parameters:
            vram_origin: The options' borrowed ordinal array, inferred from the argument.

        Args:
            library: The loaded engine, kept so the shared object stays mapped.
            vram: The same options the indexes were initialized with; copied, not borrowed.
            streams: Borrowed CUDA streams, one per device in `vram` order; `NULL_ADDRESS` creates its own.
            scratch_limit: Ceiling on scratch bytes per device; zero lets scratch grow as geometry asks.
            deferred_limit: Deferred calls it tracks at once, sizing its deferred table; zero takes a default.

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
            Int(created),
            Int(outcome),
            vram.untracked(),
            streams,
            c_size_t(scratch_limit),
            c_size_t(deferred_limit),
        )
        library.symbols().executor_for_vram(arguments.unsafe_origin_cast[MutAnyOrigin]())
        library._check(outcome.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0], "usearch_executor_for_vram")
        return Self(
            library=library,
            address=created.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0],
            residency=Residency.VRAM,
        )

    def drain(self, mood: Mood = Mood.PARKED) raises Error:
        """Waits for every call submitted before the drain, then adopts `mood` whenever it next idles.
        Calls `usearch_executor_drain`.

        An executor without threads of its own holds a deferred call until its wait, the next drain
        or blocking call, or the free of a handle it names, so this runs every call it still holds.

        Args:
            mood: What the threads or streams do once idle; the last drain wins.

        Raises:
            `Error` for a failure of the executor itself, such as a lost device - never one of a
            drained call, which reports through its own outcome.
        """
        var outcome = stack_allocation[1, Outcome]()
        outcome[unsafe_offset=0] = Outcome()
        var arguments = stack_allocation[1, ExecutorDrainArguments]()
        arguments[unsafe_offset=0] = ExecutorDrainArguments(self.address, Int(outcome), mood)
        self.library.symbols().executor_drain(arguments.unsafe_origin_cast[MutAnyOrigin]())
        self.library._check(outcome.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0], "usearch_executor_drain")


# endregion Executors

# region Arenas


@fieldwise_init
struct ArenaInitArguments(ImplicitlyCopyable):
    """`usearch_arena_init_t`, laid out as the ABI declares it."""

    var arena: Address
    var executor: Address
    var outcome: Address


struct Arena:
    """The memory every output whose length scales with a call lands in - `usearch_arena_t`, freed
    when it goes out of scope.

    Made where an executor computes, host pages or the memory of its GPU, decided once; nothing is
    allocated until a call writes into it. Every call handed it rewinds and refills it, which
    invalidates the previous call's outputs, so a steady loop allocates once. One arena serves one
    call at a time; a second concurrent use is refused by the engine as `CONTENDED`, which leaves
    the holder's results readable.

    Every verb takes it as its last required argument, deferred ones too, and the result borrows it
    in turn, so the arena outlives both; a failed call leaves it valid.

    A result refuses every read with `UNINITIALIZED` once a later call rewinds its arena, and a
    result over a VRAM arena refuses every host read with `DEVICE`. A VRAM arena serves only calls on
    the executor that made it, and keeps that executor alive until it is freed.
    """

    var _library: Library
    """The loaded engine, held so the release can still reach `usearch_arena_free`."""
    var _address: Address
    """The `usearch_arena_t`, never null."""
    var _residency: Residency
    """Where its memory lives, so a host read of device memory is refused rather than made."""
    var _executor: Optional[ArcPointer[_ExecutorHandle]]
    """The VRAM executor that made it, held so the arena is freed first, on that executor's first stream."""
    var _generation: Int
    """Calls the engine took on it so far, which every result records to know when a later one rewound it."""

    def __init__(out self, library: Library) raises Error:
        """Host pages for calls on the calling thread. Calls `usearch_arena_init`.

        Args:
            library: The loaded engine, kept so the shared object stays mapped.

        Raises:
            `Error` if the engine cannot allocate the handle.
        """
        self = Self(Executor(library))

    def __init__(out self, executor: Executor) raises Error:
        """Memory where `executor` computes - host pages, or the memory of its GPU - serving the calls
        of its residency. Calls `usearch_arena_init`.

        Args:
            executor: The executor whose calls it serves; host pages may outlive it, and a VRAM arena
                keeps it alive.

        Raises:
            `Error` if the engine cannot allocate the handle.
        """
        var created = stack_allocation[1, Address]()
        created[unsafe_offset=0] = NULL_ADDRESS
        var outcome = stack_allocation[1, Outcome]()
        outcome[unsafe_offset=0] = Outcome()
        var arguments = stack_allocation[1, ArenaInitArguments]()
        arguments[unsafe_offset=0] = ArenaInitArguments(Int(created), executor.address, Int(outcome))
        executor.library.symbols().arena_init(arguments.unsafe_origin_cast[MutAnyOrigin]())
        executor.library._check(outcome.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0], "usearch_arena_init")
        self._library = executor.library
        self._address = created.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0]
        self._residency = executor.residency
        self._executor = Optional(executor._handle) if executor.residency == Residency.VRAM else None
        self._generation = 0

    def __deinit__(deinit self):
        """Frees the arena and every output in it, then lets a VRAM arena's executor go.
        Calls `usearch_arena_free`.

        The executor and the loader reference are surrendered only after the call, for the reason
        `_ExecutorHandle` records.
        """
        self._library.symbols().arena_free(self._address)
        _ = self._executor^
        _ = self._library^

    def residency(self) -> Residency:
        """Where its memory lives, decided once by the executor that made it. Binding-only.

        Returns:
            `VRAM` for device memory, which host reads refuse, and the host residency otherwise.
        """
        return self._residency


def _served[origin: MutOrigin](arena: Pointer[Arena, origin], status: Status) -> Int:
    """Retires every earlier result once the engine took a call naming the arena; a `CONTENDED` refusal
    never touched it. Answers the generation the call's result is stamped with."""
    var served = Pointer[Arena, MutAnyOrigin](unsafe_from_address=Int(arena))
    if status != Status.CONTENDED:
        served[]._generation += 1
    return served[]._generation


def _readable[origin: MutOrigin](arena: Pointer[Arena, origin], generation: Int, detail: StaticString) raises Error:
    """Refuses a host read of a result over device memory, or over an arena a later call rewound."""
    if arena[]._residency == Residency.VRAM:
        raise Error(Status.DEVICE, detail)
    if arena[]._generation != generation:
        raise Error(Status.UNINITIALIZED, detail)


def _in_arena[T: Copyable, origin: Origin](address: Address, length: Int) -> Span[T, origin]:
    """`length` values a call wrote into an arena; a failed or empty call left `address` null."""
    if length == 0:
        return Span[T, origin]()
    return Span[T, origin](unsafe_ptr=Pointer[T, origin](unsafe_from_address=address), length=length)


@fieldwise_init
struct Monolith[origin: MutOrigin](Movable):
    """The serialized index `save_buffer` lands in its arena, read in place.

    Parameters:
        origin: The arena the bytes live in, which this result borrows.
    """

    var arena: Pointer[Arena, Self.origin]
    """The arena holding the bytes."""
    var generation: Int
    """The arena's call these bytes came from, so a read after a later call is refused."""
    var data: Address
    """The first byte, inside the arena."""
    var length: Int
    """Bytes the monolith spans."""

    def bytes(ref self) raises Error -> Span[UInt8, origin_of(self)]:
        """The monolith, borrowed from the arena this result reads.

        Returns:
            The bytes, valid until the arena's next call.

        Raises:
            `Error` with `UNINITIALIZED` once a later call rewound the arena, or `DEVICE` over VRAM.
        """
        _readable(self.arena, self.generation, "Monolith.bytes")
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


trait Call(Deinitable, Movable):
    """One verb crossing into the engine: what it lends until final, and how its outputs become a result.

    Each receiver routes the same `Call` - the index and `index.on(executor)` block on it, and
    `index.deferred(executor)` wraps it in a `Deferred` - which is what keeps the verbs symmetric.
    """

    comptime Outputs: ImplicitlyCopyable & Defaultable & Deinitable
    """The fixed-size outputs the engine writes through, such as a count or a view into the arena."""
    comptime Result: Movable & Deinitable
    """What a successful call answers."""
    comptime symbol: StaticString
    """The C entry point, carried into every error."""

    def submit(
        mut self,
        library: Library,
        index: Address,
        executor: Address,
        completion: Completion,
        outcome: Address,
        outputs: Address,
    ):
        """Crosses into the engine, or writes the binding's own refusal into `outcome` without crossing."""
        ...

    def served(self, status: Status) -> Int:
        """Stamps the call's arena once the engine took the call, and answers the stamp; the arena-free
        calls keep none."""
        return 0

    def conclude(deinit self, outputs: Self.Outputs, generation: Int) -> Self.Result:
        """Adopts what the final call wrote - success or not - and builds the result around its arena,
        stamped with the `generation` that `served` answered."""
        ...


def _refuse(outcome: Address, status: Status):
    """Finishes a call the binding refused before it crossed, so every receiver reports it alike."""
    Pointer[Outcome, MutAnyOrigin](unsafe_from_address=outcome)[] = Outcome(c_int(status.code), NULL_ADDRESS)


def _block[C: Call](library: Library, var call: C, index: Address, executor: Address) raises Error -> C.Result:
    """Runs one call to its final status, its slots on the stack, and answers its result or raises."""
    var slots = stack_allocation[1, CallSlots[C.Outputs]]()
    slots[unsafe_offset=0] = CallSlots[C.Outputs]()
    var landed = slots.unsafe_origin_cast[MutAnyOrigin]()
    call.submit(
        library,
        index,
        executor,
        Completion.BLOCKING,
        Int(Pointer(to=landed[unsafe_offset=0].outcome)),
        Int(Pointer(to=landed[unsafe_offset=0].outputs)),
    )
    var written = landed[unsafe_offset=0]
    var generation = call.served(Status.of(written.outcome.status))
    var result = call^.conclude(written.outputs, generation)
    library._check(written.outcome, C.symbol)
    return result^


struct Deferred[C: Call, origin: ImmOrigin](Movable):
    """A call submitted to an executor, which owns its outcome and outputs until it is final. Binding-only.

    `status` reads `DEFERRED` until then, without blocking, and `wait` blocks until the call is
    final and answers its result or raises its failure, as many times as it is asked. The result is
    the view a blocking call answers, borrowing the same arena. Dropping an unfinished one waits the
    same way, so nothing the engine still writes is ever freed under it. The origins keep the index,
    the executor, the arena and every borrowed input alive for as long as this is.

    An executor without threads of its own holds a deferred call until its wait, its executor's
    next drain or blocking call, or the free of a handle it names, so a `wait` - or the last use of
    the `Deferred`, which Mojo destroys at once - runs it there.

    Parameters:
        C: The verb's crossing, which names the result.
        origin: The union of everything the call borrows.
    """

    var library: Library
    """The loaded engine, held so the wait can still reach `usearch_outcome_wait`."""
    var slots: OwnedPointer[CallSlots[Self.C.Outputs]]
    """The outcome and scalar outputs the engine writes, on the heap since they outlive the verb."""
    var call: Optional[Self.C]
    """What the call lends the engine, until a wait concludes it."""
    var answer: Optional[Self.C.Result]
    """The result, once a wait found the call succeeded."""
    var failure: Optional[Error]
    """The failure, once a wait found the call failed."""
    var generation: Int
    """The stamp `served` answered at submission, so a result waited on late still reads as stale."""

    def __init__(out self, library: Library, var call: Self.C, index: Address, executor: Address):
        """Submits `call` deferred, on `executor`, against `index`.

        Args:
            library: The loaded engine the call crosses into.
            call: The verb's crossing, owned until the call is final.
            index: The handle the call names, or `NULL_ADDRESS` for one that names none.
            executor: The executor the call is submitted to.
        """
        self.library = library
        self.slots = OwnedPointer(CallSlots[Self.C.Outputs]())
        self.call = call^
        self.answer = None
        self.failure = None
        self.generation = 0
        var slots = self._slots()
        self.call.value().submit(
            self.library,
            index,
            executor,
            Completion.DEFERRED,
            Int(Pointer(to=slots[].outcome)),
            Int(Pointer(to=slots[].outputs)),
        )
        self.generation = self.call.value().served(self.status())

    def __deinit__(deinit self):
        """Waits until the call is final, then frees what it lent. Calls `usearch_outcome_wait`."""
        if self.call:
            var slots = Pointer[CallSlots[Self.C.Outputs], MutAnyOrigin](
                unsafe_from_address=Int(Pointer(to=self.slots[]))
            )
            _ = self.library.symbols().outcome_wait(Int(Pointer(to=slots[].outcome)))
            _ = self.call.take().conclude(slots[].outputs, self.generation)
        _ = self.library^

    def _slots(self) -> Pointer[CallSlots[Self.C.Outputs], MutAnyOrigin]:
        """The heap slots, through the origin the engine's writes need."""
        return Pointer[CallSlots[Self.C.Outputs], MutAnyOrigin](unsafe_from_address=Int(Pointer(to=self.slots[])))

    def status(self) -> Status:
        """`DEFERRED` until the call is final, then its final status; never blocks. Calls `usearch_outcome_status`.

        Returns:
            The status the outcome holds.
        """
        return Status(self.library.symbols().outcome_status(Int(Pointer(to=self._slots()[].outcome))))

    def wait(mut self) raises Error -> ref[self] Self.C.Result:
        """Blocks until the call is final, then answers its result or raises its failure, every time it is asked.
        Calls `usearch_outcome_wait`.

        Returns:
            The result, owned by this `Deferred` and borrowed for as long as it is.

        Raises:
            `Error` carrying the call's failure, whether the engine or the binding refused it.
        """
        if self.call:
            var slots = self._slots()
            _ = self.library.symbols().outcome_wait(Int(Pointer(to=slots[].outcome)))
            var written = slots[]
            var result = self.call.take().conclude(written.outputs, self.generation)
            if written.outcome.status == c_int(0):
                self.answer = result^
            else:
                self.failure = self.library._error(written.outcome, Self.C.symbol)
        if self.failure:
            raise self.failure.value()
        return Pointer(to=self.answer.value()).unsafe_origin_cast[origin_of(self)]()[]


# endregion Deferred Calls
