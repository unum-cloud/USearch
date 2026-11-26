"""Ragged batches the sequence, shapes and geospatial indexes share: Apache Arrow tapes in and out.

Entries of those modalities vary in size, so a batch cannot ride a run of equal rows. A tape lays
every entry's elements end to end, with one more offset than entries around them, and the verb's
`count` is the entry count. An element is a byte of a sequence or a point of a shape, whose
`width` scalars travel together while the offsets count whole points.

`TapeView` views a caller's run as a call's input, `Tape` owns one a caller builds entry by
entry, and `TapeEntries` reads what `get` wrote into its arena, in place.

File: mojo/usearch/basic.mojo
Author: Ash Vardanian
Date: November 16, 2026
"""

from std.ffi import c_int, c_size_t
from std.sys.info import size_of

from usearch.types import Address, ArenaRead, Error, NULL_ADDRESS, ScalarKind, Status, _anchor, _in_arena


@fieldwise_init
struct TapeLayout(Equatable, ImplicitlyCopyable, TrivialRegisterPassable):
    """How wide a tape's offsets are - `usearch_tape_layout_t`."""

    var code: c_int
    """The raw `usearch_tape_layout_t` value the ABI exchanges."""
    comptime U32 = Self(0)
    """Apache Arrow `Binary` or `List`, with 32-bit offsets."""
    comptime U64 = Self(1)
    """Apache Arrow `LargeBinary` or `LargeList`, the same with 64-bit offsets."""


@fieldwise_init
struct TapeViewArguments(ImplicitlyCopyable, TrivialRegisterPassable):
    """`usearch_tape_view_t`, laid out as the ABI declares it: the call's `count` is its entry count,
    and `offsets` is the union member `layout` names."""

    var data: Address
    var layout: TapeLayout
    var offsets: Address
    var scalar: ScalarKind


def _no_tape() -> TapeViewArguments:
    """A tape nothing was written to, which is what a failed call leaves."""
    return TapeViewArguments(NULL_ADDRESS, TapeLayout.U64, NULL_ADDRESS, ScalarKind.UNKNOWN)


@fieldwise_init
struct TapeView[origin: ImmOrigin = ImmUntrackedOrigin](ImplicitlyCopyable):
    """A viewed ragged batch - `usearch_tape_view_t`, the input of every sequence and shapes verb.

    Entry `i` spans elements `offsets[i]` up to `offsets[i + 1]`, counted from the first element, so
    a single entry is a tape of two offsets. The entry count becomes the verb's `count`.

    Parameters:
        origin: The viewed elements and offsets, named so they outlive every call this view crosses.
    """

    var view: TapeViewArguments
    """The first element, the offsets, their width and the scalar kind, as the engine reads them."""
    var entries: Int
    """Entries the tape holds, one fewer than its offsets, which the verb takes as its `count`."""

    @staticmethod
    def of[
        dtype: DType,
        elements_origin: ImmOrigin,
        offsets_origin: ImmOrigin,
        //,
        width: Int = 1,
        kind: ScalarKind = ScalarKind.of[dtype](),
    ](elements: Span[Scalar[dtype], elements_origin], offsets: Span[UInt64, offsets_origin]) raises Error -> TapeView[
        ImmOrigin(origin_of(_anchor[elements_origin](), _anchor[offsets_origin]()))
    ]:
        """Reads a flat run of scalars as entries between 64-bit offsets. Binding-only.

        Parameters:
            dtype: The scalar type of the run, inferred from the argument.
            elements_origin: The run's origin, inferred with it.
            offsets_origin: The offsets' origin, inferred with them.
            width: Scalars per element: one for a sequence's bytes, three for a shape's points.
            kind: How the engine reads those scalars; the kind `dtype` stores by default.

        Args:
            elements: Every entry's scalars, end to end.
            offsets: One more than the entries, non-decreasing, counting elements rather than scalars.

        Returns:
            A view of the whole run.

        Raises:
            `Error` with `SHAPE_MISMATCH` for no offsets, or a last offset past the elements.
        """
        comptime assert kind.stores[dtype](), "the element type must store the kind whole"
        if len(offsets) == 0 or Int(offsets[len(offsets) - 1]) * width > len(elements):
            raise Error(Status.SHAPE_MISMATCH, "TapeView.of")
        return {
            TapeViewArguments(Int(elements.unsafe_ptr()), TapeLayout.U64, Int(offsets.unsafe_ptr()), kind),
            len(offsets) - 1,
        }


struct Tape[dtype: DType, width: Int = 1](Movable, Sized):
    """A ragged batch the caller builds entry by entry, owning its elements and offsets. Binding-only.

    Parameters:
        dtype: The scalar type every element holds.
        width: Scalars per element: one for a sequence's bytes, three for a shape's points.
    """

    var elements: List[Scalar[Self.dtype]]
    """Every entry's scalars, end to end."""
    var offsets: List[UInt64]
    """One more than the entries, counting elements; starts at a single zero."""

    def __init__(out self):
        """An empty tape: no entries, one offset."""
        self.elements = List[Scalar[Self.dtype]]()
        self.offsets = [UInt64(0)]

    def __len__(self) -> Int:
        """The entries appended.

        Returns:
            One fewer than the offsets.
        """
        return len(self.offsets) - 1

    def append(mut self, entry: Span[Scalar[Self.dtype], _]) raises Error:
        """Appends one entry, `width` scalars per element.

        Args:
            entry: The entry's scalars.

        Raises:
            `Error` with `SHAPE_MISMATCH` when the scalars do not divide into whole elements.
        """
        if len(entry) % Self.width != 0:
            raise Error(Status.SHAPE_MISMATCH, "Tape.append")
        self.elements.extend(entry)
        self.offsets.append(UInt64(len(self.elements) // Self.width))

    def append(mut self, text: StringSlice) raises Error:
        """Appends one sequence, the bytes of `text` as they are encoded.

        Args:
            text: The sequence's characters.

        Raises:
            Never: the signature raises only to match the other `append`.
        """
        comptime assert Self.dtype == DType.uint8 and Self.width == 1, "text appends to a tape of bytes"
        for byte in text.as_bytes():
            self.elements.append(rebind[Scalar[Self.dtype]](byte))
        self.offsets.append(UInt64(len(self.elements)))

    def view(ref self) -> TapeView[ImmOrigin(origin_of(self))]:
        """The tape as a call's input, viewing this one.

        Returns:
            A view of every entry appended.
        """
        return {
            TapeViewArguments(
                Int(self.elements.unsafe_ptr()),
                TapeLayout.U64,
                Int(self.offsets.unsafe_ptr()),
                ScalarKind.of[Self.dtype](),
            ),
            len(self),
        }


@fieldwise_init
struct TapeEntries[dtype: DType, origin: MutOrigin, width: Int = 1](Movable, Sized):
    """The entries `get` wrote into its arena as a tape, read in place.

    Parameters:
        dtype: The element type each entry is read as.
        origin: The arena the entries live in, which this result views.
        width: Scalars per element: one for a sequence's bytes, three for a shape's points.
    """

    var view: TapeViewArguments
    """The first element and the offsets, as the engine wrote them."""
    var written: Int
    """Entries written; zero when the key is absent."""

    var viewing: ArenaRead
    """The arena the result reads and the generation the call that wrote it brought it to."""

    def __len__(self) -> Int:
        """The entries written.

        Returns:
            Zero when the key is absent.
        """
        return self.written

    def __getitem__(ref self, entry: Int) raises Error -> Span[Scalar[Self.dtype], origin_of(self)]:
        """One entry's scalars, `width` per element, viewed from the arena this result reads.

        Args:
            entry: Which entry, below `len(self)`.

        Returns:
            The entry's scalars.
        """
        self.viewing.open("TapeEntries.__getitem__")
        var begin = self._offset(entry)
        var viewed = _in_arena[Scalar[Self.dtype], origin_of(self)](
            self.view.data + begin * Self.width * size_of[Scalar[Self.dtype]](),
            (self._offset(entry + 1) - begin) * Self.width,
        )
        self.viewing.close()
        return viewed

    def _offset(self, position: Int) -> Int:
        """One offset of the tape, as wide as `layout` says."""
        if self.view.layout == TapeLayout.U32:
            return Int(
                Pointer[UInt32, ImmUntrackedOrigin](unsafe_from_address=self.view.offsets)[unsafe_offset=position]
            )
        return Int(Pointer[UInt64, ImmUntrackedOrigin](unsafe_from_address=self.view.offsets)[unsafe_offset=position])
