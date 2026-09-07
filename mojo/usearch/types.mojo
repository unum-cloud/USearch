"""The vocabulary every modality shares: keys, scalars, residencies, and the one error type.

Mirrors @c include/usearch/types.h. Nothing dense-specific lives here - the matrix descriptors, the
metric kinds, and the index itself belong to @c dense.mojo, so a second modality adds a module
rather than widening this one.

`Error` is the only error type the binding raises: Mojo 1.0 allows at most one per function and
never widens a typed `raises` into a plain one, so a second would force every caller to catch and
convert. A refusal is a failure, not an absence - a key the index does not hold answers zero, but a
call the engine declines raises.

Four Mojo 1.0 facts shape the descriptors and the pointer aliases below, and all four were measured
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
instantiation the ABI signatures name.

A struct whose address crosses to C is materialized into a `stack_allocation` slot rather than held
in a local whose address is taken. An `alloca` is not a value Mojo owns, so nothing destroys it and
nothing folds its store away, where a local's last tracked use is the address-taking itself.
"""

from std.bit import byte_swap
from std.ffi import c_int, c_size_t
from std.memory import stack_allocation

from usearch.library import Library

# region Pointers

comptime Handle = Pointer[NoneType, MutUntrackedOrigin]
"""An opaque `usearch_dense_index_t`, `usearch_executor_t`, `usearch_dense_cursor_t`, or
`usearch_dense_finder_t`."""

comptime Context = Pointer[NoneType, MutUntrackedOrigin]
"""The type-punned callback state the C API carries through a predicate or a metric."""

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

comptime OutSize = Pointer[c_size_t, MutAnyOrigin]
"""A `size_t *` the C API writes through, with the aliasing caveat this module's docstring records."""

comptime OutAddress = Pointer[Int, MutAnyOrigin]
"""A handle-sized slot the C API writes through, with the same caveat."""

comptime OutFloat32 = Pointer[Float32, MutAnyOrigin]
"""A `float *` run the C API fills, with the same caveat."""

# endregion Pointers

# region Errors


@fieldwise_init
struct Status(Equatable, ImplicitlyCopyable, TrivialRegisterPassable, Writable):
    """Why a call into the engine failed - the `usearch_status_t` vocabulary, verbatim."""

    var code: Int32
    """The raw `usearch_status_t` value, kept even when this build does not name it."""
    comptime SUCCESS = Self(0)
    """The call completed."""
    comptime OUT_OF_MEMORY = Self(1)
    """An allocation or reservation could not be satisfied."""
    comptime SHAPE_MISMATCH = Self(2)
    """Dimensionality, row counts, or output capacity disagree."""
    comptime UNSUPPORTED_LAYOUT = Self(3)
    """A stride or packing the engine cannot traverse."""
    comptime UNKNOWN_KIND = Self(4)
    """A scalar, metric, or residency tag this build does not know."""
    comptime READ_ONLY = Self(5)
    """A mutating verb on a handle that opens its checkpoint in place."""
    comptime IO = Self(6)
    """A file could not be opened, read, written, or parsed."""
    comptime DEVICE = Self(7)
    """A device could not be selected, or its memory could not be claimed."""
    comptime MALFORMED_INPUT = Self(8)
    """A caller-supplied representation could not be decoded."""
    comptime INTERNAL = Self(9)
    """The engine reported neither success nor a failure, which names a defect rather than a state."""
    comptime OVERFLOW = Self(10)
    """A hard design ceiling was reached, so a smaller request cannot help."""
    comptime UNINITIALIZED = Self(11)
    """The verb needs a prior successful reservation that never happened."""
    comptime CONFIG_MISMATCH = Self(12)
    """The request contradicts an already-frozen configuration."""
    comptime CONTENDED = Self(13)
    """A concurrent operation violated a synchronization contract; discard the output and retry."""
    comptime CANCELLED = Self(14)
    """A progress handler stopped the operation; it published nothing."""
    comptime LIBRARY_MISSING = Self(-1)
    """Nothing to `dlopen`, or what loaded is a different major version - the binding's own."""
    comptime SYMBOL_MISSING = Self(-2)
    """The library loaded but does not export a symbol the binding needs - the binding's own."""

    def write_to(self, mut writer: Some[Writer]):
        """Spells the failure out in prose, so a code this build does not name still reads.

        Args:
            writer: The sink the reason is appended to.
        """
        if self == Self.LIBRARY_MISSING:
            writer.write("the engine is not on the loader path")
        elif self == Self.SYMBOL_MISSING:
            writer.write("the loaded engine is missing a symbol")
        elif self == Self.OUT_OF_MEMORY:
            writer.write("an allocation could not be satisfied")
        elif self == Self.SHAPE_MISMATCH:
            writer.write("the shapes disagree")
        elif self == Self.UNSUPPORTED_LAYOUT:
            writer.write("the layout cannot be traversed")
        elif self == Self.UNKNOWN_KIND:
            writer.write("a scalar, metric, or residency tag is unknown to this build")
        elif self == Self.READ_ONLY:
            writer.write("the handle opens its checkpoint in place")
        elif self == Self.IO:
            writer.write("a file could not be read or written")
        elif self == Self.DEVICE:
            writer.write("a device could not be selected or claimed")
        elif self == Self.MALFORMED_INPUT:
            writer.write("an input could not be decoded")
        elif self == Self.INTERNAL:
            writer.write("the engine reported neither success nor a failure")
        elif self == Self.OVERFLOW:
            writer.write("a hard design ceiling was reached")
        elif self == Self.UNINITIALIZED:
            writer.write("a prior reservation never happened")
        elif self == Self.CONFIG_MISMATCH:
            writer.write("the request contradicts a frozen configuration")
        elif self == Self.CONTENDED:
            writer.write("a concurrent operation violated a synchronization contract")
        elif self == Self.CANCELLED:
            writer.write("the operation was cancelled")
        else:
            writer.write("the engine reported no reason")

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
struct ErrorSlot(ImplicitlyCopyable, TrivialRegisterPassable):
    """The `usearch_error_t` out-parameter every verb takes, laid out as the ABI declares it.

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

    def write_to(self, mut writer: Some[Writer]):
        """Writes the reason, the entry point, and the engine's own message together.

        Args:
            writer: The sink the message is appended to.
        """
        writer.write("USearch: ", self.status, " [", self.detail, "]")
        if self.message != NULL_ADDRESS:
            writer.write(": ", text_at(self.message))

    @staticmethod
    def raise_if_failed(report: Pointer[ErrorSlot, MutAnyOrigin], detail: StaticString) raises Error:
        """Reads the slot back and raises when the engine reported anything but success.

        Args:
            report: The slot the call just wrote through.
            detail: The entry point that wrote it, carried into the message.

        Raises:
            `Error` carrying the reported status and the engine's own static message.
        """
        var written = report[unsafe_offset=0]
        if written.status != c_int(0):
            raise Self(Status.of(written.status), detail, written.message)


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

    def as_integer(self) -> UInt64:
        """The integer identity in bytes 8 through 15, whatever the leading ones hold.

        Returns:
            The trailing word read big-endian, which is the whole identity for a key built from one.
        """
        return byte_swap(self.trailing)

    def write_to(self, mut writer: Some[Writer]):
        """Writes the identity, which is the trailing word for every key an integer named.

        Args:
            writer: The sink the key is appended to.
        """
        writer.write("key(", self.as_integer(), ")")


comptime Distance = Float64
"""A distance or similarity score in the metric of the index that produced it - `usearch_distance_t`."""

# endregion Keys

# region Scalars


@fieldwise_init
struct ScalarKind(Equatable, ImplicitlyCopyable, TrivialRegisterPassable, Writable):
    """The scalar types the indexes store and exchange - `usearch_scalar_t`, named by width.

    Ordered by shrinking dynamic range, floats first, then signed and unsigned integers. The `xN`
    suffixes spell how many logical dimensions share one byte.
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
    comptime U8 = Self(20)
    """8-bit unsigned integer."""
    comptime U4X2 = Self(21)
    """Two unsigned nibbles per byte, even dimensions in the low nibble."""
    comptime U1X8 = Self(22)
    """Eight single-bit dimensions per byte - what Hamming and Jaccard score."""

    def bits_per_scalar(self) -> Int:
        """How many bits one logical dimension occupies, which sub-byte kinds share.

        Returns:
            The width in bits, or zero for `UNKNOWN`, whose storage the caller owns.
        """
        if self == Self.F64:
            return 64
        if self == Self.F32:
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

    def write_to(self, mut writer: Some[Writer]):
        """Writes the raw code, since a name table would drift from the header.

        Args:
            writer: The sink the code is appended to.
        """
        writer.write("scalar(", Int(self.code), ")")


# endregion Scalars

# region Residency & Execution


@fieldwise_init
struct Residency(Equatable, ImplicitlyCopyable, TrivialRegisterPassable, Writable):
    """Where an index's payload lives and which engine traverses it - `usearch_residency_t`.

    Also decides what loading means: materialize into RAM, upload to a device, or open in place on
    disk. An executor's kind is fixed by the constructor that produced it and must match this.
    """

    var code: c_int
    """The raw `usearch_residency_t` value the ABI exchanges."""
    comptime RAM = Self(0)
    """Volatile memory traversed by CPU threads - the concurrent baseline."""
    comptime VRAM = Self(1)
    """Accelerator memory traversed by GPU warps on the chosen devices."""
    comptime NVME = Self(2)
    """Persisted storage traversed through pluggable I/O engines, sized past RAM."""

    def write_to(self, mut writer: Some[Writer]):
        """Names which of the three this is.

        Args:
            writer: The sink the name is appended to.
        """
        if self == Self.RAM:
            writer.write("ram")
        elif self == Self.VRAM:
            writer.write("vram")
        elif self == Self.NVME:
            writer.write("nvme")
        else:
            writer.write("residency(", Int(self.code), ")")


@fieldwise_init
struct Mood(Equatable, ImplicitlyCopyable, TrivialRegisterPassable):
    """What an executor does with its threads or streams between calls - `usearch_mood_t`."""

    var code: c_int
    """The raw `usearch_mood_t` value the ABI exchanges."""
    comptime CHILL = Self(0)
    """Park them; the next call pays a wake-up."""
    comptime GRIND = Self(1)
    """Keep them spinning, trading power for the next call's latency."""


@fieldwise_init
struct Completion(Equatable, ImplicitlyCopyable, TrivialRegisterPassable):
    """Whether a verb returns when its work is done or when it is queued - `usearch_completion_t`."""

    var code: c_int
    """The raw `usearch_completion_t` value the ABI exchanges."""
    comptime BLOCKING = Self(0)
    """Return once the work is finished and its outputs are readable."""
    comptime DEFERRED = Self(1)
    """Return once the work is queued; outputs stay indeterminate until a drain."""


@fieldwise_init
struct ExecutionConfig(ImplicitlyCopyable):
    """Per-call scheduling of the executor - `usearch_execution_config_t`.

    Declared by the ABI's shared vocabulary, and no dense verb accepts one yet, so the binding
    carries the shape without a call site. Deferring moves both the outputs and the failure out of
    the call: nothing a deferred verb writes may be read until the drain returns.
    """

    var mood: Mood
    """What the threads or streams do between calls."""
    var completion: Completion
    """Whether the verb waits for its own work."""


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

    def untracked(self) -> BytesView[ImmUntrackedOrigin]:
        """The same run with its borrow erased - the one instantiation the ABI signatures name.

        Returns:
            A view of the same bytes, whose type no longer keeps their storage alive.
        """
        return BytesView[ImmUntrackedOrigin](self.data, self.length)


@fieldwise_init
struct BytesSpan[origin: MutOrigin = MutUntrackedOrigin](ImplicitlyCopyable, TrivialRegisterPassable):
    """A caller-owned writable run of bytes - `usearch_bytes_span_t`, the mutable twin of the view.

    Pairing the pointer with its length in one argument is what makes a mismatched capacity
    unrepresentable, where two loose arguments only made it unlikely.

    Parameters:
        origin: The lent storage, named so it outlives every call this span crosses.
    """

    var data: Pointer[Int8, MutUntrackedOrigin]
    """The first byte the engine may write; untracked, because Mojo 1.0 refuses `AnyOrigin` in a field."""
    var length: c_size_t
    """How many bytes it may write, which the engine never exceeds."""

    @staticmethod
    def of[bytes_origin: MutOrigin, //](bytes: Span[UInt8, bytes_origin]) -> BytesSpan[bytes_origin]:
        """Lends a run of bytes as the span the engine fills.

        Parameters:
            bytes_origin: The run's origin, inferred from the argument.

        Args:
            bytes: The storage the engine writes into.

        Returns:
            A span over exactly that storage.
        """
        return BytesSpan[bytes_origin](
            bytes.unsafe_ptr().unsafe_bitcast[Int8]().unsafe_origin_cast[MutUntrackedOrigin](),
            c_size_t(len(bytes)),
        )

    def untracked(self) -> BytesSpan[MutUntrackedOrigin]:
        """The same run with its borrow erased - the one instantiation the ABI signatures name.

        Returns:
            A span over the same storage, whose type no longer keeps it alive.
        """
        return BytesSpan[MutUntrackedOrigin](self.data, self.length)


# endregion Byte Runs

# region Key Runs


@fieldwise_init
struct KeysView[origin: ImmOrigin = ImmUntrackedOrigin](ImplicitlyCopyable, TrivialRegisterPassable):
    """A borrowed run of keys - `usearch_keys_view_t`, how a batch names the identities it carries.

    One key per row of the matrix it accompanies, and the engine checks the two counts against
    each other rather than assuming them equal.

    Parameters:
        origin: The borrowed storage, named so it outlives every call this view crosses.
    """

    var data: Pointer[Key, ImmUntrackedOrigin]
    """The first key, borrowed for the duration of the call."""
    var count: c_size_t
    """How many keys the run covers."""

    @staticmethod
    def of[keys_origin: ImmOrigin, //](keys: Span[Key, keys_origin]) -> KeysView[keys_origin]:
        """Borrows a run of keys as the view the engine reads.

        Parameters:
            keys_origin: The run's origin, inferred from the argument.

        Args:
            keys: The identities the batch is published under.

        Returns:
            A view of exactly those keys.
        """
        return KeysView[keys_origin](
            Pointer[Key, ImmUntrackedOrigin](unsafe_from_address=Int(keys.unsafe_ptr())),
            c_size_t(len(keys)),
        )

    def untracked(self) -> KeysView[ImmUntrackedOrigin]:
        """The same run with its borrow erased - the one instantiation the ABI signatures name.

        Returns:
            A view of the same keys, whose type no longer keeps their storage alive.
        """
        return KeysView[ImmUntrackedOrigin](self.data, self.count)


@fieldwise_init
struct KeysSpan[origin: MutOrigin = MutUntrackedOrigin](ImplicitlyCopyable, TrivialRegisterPassable):
    """A caller-owned writable run of keys - `usearch_keys_span_t`, the mutable twin of the view.

    Its capacity is the only bound a streaming verb has on how much it may write, so the pointer
    and that bound travel as one argument rather than as two a caller could contradict.

    Parameters:
        origin: The lent storage, named so it outlives every call this span crosses.
    """

    var data: Pointer[Key, MutUntrackedOrigin]
    """The first key the engine may write; untracked, because Mojo 1.0 refuses `AnyOrigin` in a field."""
    var capacity: c_size_t
    """How many keys it may write, which is the batch size every such verb writes up to."""

    @staticmethod
    def of[keys_origin: MutOrigin, //](keys: Span[Key, keys_origin]) -> KeysSpan[keys_origin]:
        """Lends a run of keys as the span the engine fills.

        Parameters:
            keys_origin: The run's origin, inferred from the argument.

        Args:
            keys: The storage the engine writes identities into.

        Returns:
            A span over exactly that storage.
        """
        return KeysSpan[keys_origin](
            Pointer[Key, MutUntrackedOrigin](unsafe_from_address=Int(keys.unsafe_ptr())),
            c_size_t(len(keys)),
        )

    def untracked(self) -> KeysSpan[MutUntrackedOrigin]:
        """The same run with its borrow erased - the one instantiation the ABI signatures name.

        Returns:
            A span over the same storage, whose type no longer keeps it alive.
        """
        return KeysSpan[MutUntrackedOrigin](self.data, self.capacity)


# endregion Key Runs

# region Signatures

comptime ExecutorForRam = def(
    Pointer[RamOptions[ImmUntrackedOrigin], MutAnyOrigin], c_size_t, Pointer[ErrorSlot, MutAnyOrigin]
) thin abi("C") -> Address
"""`usearch_executor_for_ram`: spawns host workers, answering the handle's address or zero."""

comptime ExecutorForNvme = def(
    Pointer[NvmeOptions[ImmUntrackedOrigin], MutAnyOrigin], c_size_t, Pointer[ErrorSlot, MutAnyOrigin]
) thin abi("C") -> Address
"""`usearch_executor_for_nvme`: spawns host workers with one submission ring apiece."""

comptime ExecutorForVram = def(
    Pointer[VramOptions[ImmUntrackedOrigin], MutAnyOrigin], Pointer[ErrorSlot, MutAnyOrigin]
) thin abi("C") -> Address
"""`usearch_executor_for_vram`: opens one stream per device, taking no thread count at all."""

comptime ExecutorDrain = def(Handle, Pointer[ErrorSlot, MutAnyOrigin]) thin abi("C") -> None
"""`usearch_executor_drain`: waits for every deferred call and reports the first failure."""

comptime ExecutorFree = def(Handle) thin abi("C") -> None
"""`usearch_executor_free`: drains, releases the scratch arena, and frees; cannot fail."""

# endregion Signatures

# region Placement


@fieldwise_init
struct RamOptions[origin: ImmOrigin = ImmUntrackedOrigin](ImplicitlyCopyable):
    """The host memory domains a RAM-resident index places its shards in - `usearch_ram_options_t`.

    One shard per named NUMA node, so a dual-socket box names two and gets two graphs whose
    traversals stay local to the cores that walk them. Arenas always claim huge pages, falling back
    to base pages where none are provisioned.

    Parameters:
        origin: The borrowed node-id array, named so it outlives the constructor that reads it.
    """

    var memory_domains: Address
    """The `int32_t` node ids to place shards on; `NULL_ADDRESS` derives every node the process may use."""
    var memory_domains_count: c_size_t
    """How many ids the array holds; zero derives."""

    def __init__(out self):
        """Derives every memory domain the process may allocate on."""
        self.memory_domains = NULL_ADDRESS
        self.memory_domains_count = c_size_t(0)

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
        return RamOptions[domains_origin](Int(memory_domains.unsafe_ptr()), c_size_t(len(memory_domains)))

    def untracked(self) -> RamOptions[ImmUntrackedOrigin]:
        """The same options with the borrow erased - the one instantiation the ABI signature names.

        Returns:
            Options naming the same array, whose type no longer keeps it alive.
        """
        return RamOptions[ImmUntrackedOrigin](self.memory_domains, self.memory_domains_count)


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

    def __init__(out self):
        """Derives one free device."""
        self.devices = NULL_ADDRESS
        self.devices_count = c_size_t(0)

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
        return VramOptions[devices_origin](Int(devices.unsafe_ptr()), c_size_t(len(devices)))

    def untracked(self) -> VramOptions[ImmUntrackedOrigin]:
        """The same options with the borrow erased - the one instantiation the ABI signature names.

        Returns:
            Options naming the same array, whose type no longer keeps it alive.
        """
        return VramOptions[ImmUntrackedOrigin](self.devices, self.devices_count)


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

    def __init__(out self):
        """Leaves every shard where its own manifest points."""
        self.paths = NULL_ADDRESS
        self.paths_count = c_size_t(0)

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
        return NvmeOptions[paths_origin](Int(paths.unsafe_ptr()), c_size_t(len(paths)))

    def untracked(self) -> NvmeOptions[ImmUntrackedOrigin]:
        """The same options with the borrow erased - the one instantiation the ABI signature names.

        Returns:
            Options naming the same array, whose type no longer keeps it alive.
        """
        return NvmeOptions[ImmUntrackedOrigin](self.paths, self.paths_count)


# endregion Placement

# region Executors


struct Executor:
    """A spawned executor, freed when it goes out of scope.

    Its kind is fixed by the constructor that produced it, and `residency` reports which, so a
    caller pairing it with a handle can check the two agree before the engine does.
    """

    var library: Library
    """The loaded engine, held so its symbol table outlives every call made through the handle."""
    var handle: Handle
    """The opaque `usearch_executor_t` every verb hands back to the engine."""
    var residency: Residency
    """Which residency this executor may drive, decided by the constructor that made it."""

    def __init__[
        ram_origin: ImmOrigin, //
    ](out self, library: Library, ram: RamOptions[ram_origin], threads: Int = 0) raises Error:
        """Spawns host workers for RAM-resident indexes.

        Passing the very options the indexes were built with is what makes compute and memory
        domains agree, leaving no second place to name hardware.

        Parameters:
            ram_origin: The options' borrowed node-id array, inferred from the argument.

        Args:
            library: The loaded engine, kept so the shared object stays mapped.
            ram: The same options the indexes were initialized with; copied, not borrowed.
            threads: Worker threads; zero takes 7/8 of the cores those domains carry.

        Raises:
            `Error` if the domains cannot be resolved or the OS refuses the threads.
        """
        self.library = library
        self.residency = Residency.RAM
        var report = stack_allocation[1, ErrorSlot]()
        report[unsafe_offset=0] = ErrorSlot()
        var slot = report.unsafe_origin_cast[MutAnyOrigin]()
        var options = stack_allocation[1, RamOptions[ImmUntrackedOrigin]]()
        options[unsafe_offset=0] = ram.untracked()
        var address = library.symbols().executor_for_ram(
            options.unsafe_origin_cast[MutAnyOrigin](),
            c_size_t(threads),
            slot,
        )
        Error.raise_if_failed(slot, "usearch_executor_for_ram")
        if address == 0:
            raise Error(Status.INTERNAL, "usearch_executor_for_ram", 0)
        self.handle = Handle(unsafe_from_address=address)

    @staticmethod
    def for_nvme[
        nvme_origin: ImmOrigin, //
    ](library: Library, nvme: NvmeOptions[nvme_origin], threads: Int = 0) raises Error -> Self:
        """Spawns host workers for NVMe-resident indexes, one submission ring apiece.

        One ring per thread is what keeps submission lock-free: a thread owns its queue outright,
        so an external-memory walk never coordinates with a sibling to issue a read.

        Parameters:
            nvme_origin: The options' borrowed path array, inferred from the argument.

        Args:
            library: The loaded engine, kept so the shared object stays mapped.
            nvme: The same options the indexes were initialized with; copied, not borrowed.
            threads: Worker threads; zero derives from the devices' queue capabilities.

        Returns:
            An executor that may drive NVMe-resident handles and no others.

        Raises:
            `Error` if a path cannot be opened or the OS refuses the threads.
        """
        var report = stack_allocation[1, ErrorSlot]()
        report[unsafe_offset=0] = ErrorSlot()
        var slot = report.unsafe_origin_cast[MutAnyOrigin]()
        var options = stack_allocation[1, NvmeOptions[ImmUntrackedOrigin]]()
        options[unsafe_offset=0] = nvme.untracked()
        var address = library.symbols().executor_for_nvme(
            options.unsafe_origin_cast[MutAnyOrigin](),
            c_size_t(threads),
            slot,
        )
        Error.raise_if_failed(slot, "usearch_executor_for_nvme")
        if address == 0:
            raise Error(Status.INTERNAL, "usearch_executor_for_nvme", 0)
        return Self(library=library, handle=Handle(unsafe_from_address=address), residency=Residency.NVME)

    @staticmethod
    def for_vram[vram_origin: ImmOrigin, //](library: Library, vram: VramOptions[vram_origin]) raises Error -> Self:
        """Opens one submission stream on every named device, all driven by the calling thread.

        One stream per device is the whole design: a single host thread keeps every device fed, and
        adding host threads would buy nothing, which is why this constructor takes no thread count.

        Parameters:
            vram_origin: The options' borrowed ordinal array, inferred from the argument.

        Args:
            library: The loaded engine, kept so the shared object stays mapped.
            vram: The same options the indexes were initialized with; copied, not borrowed.

        Returns:
            An executor that may drive VRAM-resident handles and no others.

        Raises:
            `Error` if a device cannot be selected or its memory cannot be claimed.
        """
        var report = stack_allocation[1, ErrorSlot]()
        report[unsafe_offset=0] = ErrorSlot()
        var slot = report.unsafe_origin_cast[MutAnyOrigin]()
        var options = stack_allocation[1, VramOptions[ImmUntrackedOrigin]]()
        options[unsafe_offset=0] = vram.untracked()
        var address = library.symbols().executor_for_vram(options.unsafe_origin_cast[MutAnyOrigin](), slot)
        Error.raise_if_failed(slot, "usearch_executor_for_vram")
        if address == 0:
            raise Error(Status.INTERNAL, "usearch_executor_for_vram", 0)
        return Self(library=library, handle=Handle(unsafe_from_address=address), residency=Residency.VRAM)

    def __init__(out self, *, library: Library, handle: Handle, residency: Residency):
        """Adopts a handle that is already spawned, which is how the two factories hand one back.

        Args:
            library: The loaded engine the handle was created through, and is freed through.
            handle: A `usearch_executor_t` whose lifetime this executor takes over.
            residency: Which residency the constructor that made the handle fixed it to.
        """
        self.library = library
        self.handle = handle
        self.residency = residency

    def __deinit__(deinit self):
        """Drains every thread or stream, releases the scratch arena, and frees the executor.

        The loader reference is surrendered only after the call: reading the entry point is the
        last use of `library`, so without the trailing move Mojo drops the last reference - and
        `dlclose`s the engine - between the load and the branch to it.
        """
        self.library.symbols().executor_free(self.handle)
        _ = self.library^

    def drain(self) raises Error:
        """Waits for every deferred call on this executor to finish.

        Raises:
            `Error` carrying the first failure a drained call reported.
        """
        var report = stack_allocation[1, ErrorSlot]()
        report[unsafe_offset=0] = ErrorSlot()
        var slot = report.unsafe_origin_cast[MutAnyOrigin]()
        self.library.symbols().executor_drain(self.handle, slot)
        Error.raise_if_failed(slot, "usearch_executor_drain")


# endregion Executors
