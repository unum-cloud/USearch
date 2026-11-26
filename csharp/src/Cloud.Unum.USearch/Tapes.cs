/**
 *  @file csharp/src/Cloud.Unum.USearch/Tapes.cs
 *  @author Ash Vardanian
 *  @date November 16, 2026
 *  @brief Apache Arrow tapes: the ragged batches the sequence and shape indexes take and write.
 */
using System;
using System.Buffers;
using System.Runtime.InteropServices;
using System.Text;

namespace Cloud.Unum.USearch;

#region Tapes

/// <summary>
/// A ragged batch as Apache Arrow lays one out: every entry's elements end to end, and one more
/// offset than entries around them - how <see cref="SequenceIndex"/> takes byte strings and
/// <see cref="ShapesIndex"/> takes point clouds. Mirrors <c>usearch_tape_view_t</c> as an input.
/// </summary>
/// <remarks>
/// A view only points at memory it never owns: it holds the memory it was built from, and a deferred call pins it
/// until final. Offsets count in steps of <see cref="Width"/> elements, one byte of a sequence or the
/// three coordinates of a point.
/// </remarks>
/// <typeparam name="TElement">The element type, which names the scalar kind the library reads.</typeparam>
public readonly struct TapeView<TElement> where TElement : unmanaged
{
    internal TapeView(ReadOnlyMemory<TElement> elements, ReadOnlyMemory<ulong> offsets, int width, Scalar scalar)
    {
        this.Elements = elements;
        this.Offsets = offsets;
        this.Width = width;
        this.Scalar = scalar;
    }

    /// <summary>Every entry's elements, end to end.</summary>
    public ReadOnlyMemory<TElement> Elements { get; }

    /// <summary>The entries plus one offsets, in steps of <see cref="Width"/> elements.</summary>
    public ReadOnlyMemory<ulong> Offsets { get; }

    /// <summary>The elements one offset step spans: one for a byte, three for a point.</summary>
    public int Width { get; }

    /// <summary>The scalar kind the elements hold.</summary>
    public Scalar Scalar { get; }

    /// <summary>How many entries the tape holds, the call's one count.</summary>
    public int Count => Math.Max(this.Offsets.Length - 1, 0);

    /// <summary>Pins the elements until the handle is disposed.</summary>
    internal MemoryHandle PinElements() => this.Elements.Pin();

    /// <summary>Pins the offsets until the handle is disposed.</summary>
    internal MemoryHandle PinOffsets() => this.Offsets.Pin();

    /// <summary>Passes the tape as the input descriptor every tape verb reads.</summary>
    /// <param name="elements">The first element, pinned for the call.</param>
    /// <param name="offsets">The first offset, pinned for the call.</param>
    /// <returns>The C struct the library reads.</returns>
    internal unsafe NativeTapeView AsView(void* elements, void* offsets) => new()
    {
        Data = (IntPtr)elements, Layout = TapeLayout.U64, Offsets = (IntPtr)offsets, Scalar = this.Scalar,
    };
}

/// <summary>Builds <see cref="TapeView{TElement}"/> batches from the shapes .NET code already holds.</summary>
public static class Tape
{
    /// <summary>Views elements and offsets the caller laid out, checking once that the last offset
    /// stays inside the elements, since the library reads as far as it says.</summary>
    /// <typeparam name="TElement">The element type, which names the scalar kind.</typeparam>
    /// <param name="elements">Every entry's elements, end to end.</param>
    /// <param name="offsets">The entries plus one non-decreasing offsets.</param>
    /// <param name="width">The elements one offset step spans: one for bytes, three for points.</param>
    /// <returns>The view over both.</returns>
    public static TapeView<TElement> Of<TElement>(
        ReadOnlyMemory<TElement> elements, ReadOnlyMemory<ulong> offsets, int width = 1)
        where TElement : unmanaged
    {
        // The call reads the elements up to the last offset, so only this side sees the two disagree.
        if (width <= 0 || (offsets.Length > 0 && offsets.Span[^1] * (ulong)width > (ulong)elements.Length))
        {
            throw new USearchException(Status.ShapeMismatch, "the last offset runs past the elements it spans");
        }

        return new TapeView<TElement>(elements, offsets, width, ScalarKinds.KindOf<TElement>());
    }

    /// <summary>Encodes texts as one tape of their UTF-8 bytes.</summary>
    /// <param name="texts">The texts, one entry apiece.</param>
    /// <returns>The tape.</returns>
    public static TapeView<byte> Utf8(params string[] texts) =>
        Bytes(Array.ConvertAll(texts, text => Encoding.UTF8.GetBytes(text)));

    /// <summary>Lays byte strings end to end as one tape.</summary>
    /// <param name="strings">The byte strings, one entry apiece.</param>
    /// <returns>The tape.</returns>
    public static TapeView<byte> Bytes(params byte[][] strings) => Concatenated(strings, 1);

    /// <summary>Lays point clouds end to end as one tape, each cloud <c>x</c>, <c>y</c> and <c>z</c>
    /// interleaved.</summary>
    /// <typeparam name="TScalar">The coordinate type, which names the scalar kind.</typeparam>
    /// <param name="shapes">The clouds, one entry apiece, each three coordinates per point.</param>
    /// <returns>The tape, its offsets counting points.</returns>
    public static TapeView<TScalar> Points<TScalar>(params TScalar[][] shapes) where TScalar : unmanaged =>
        Concatenated(shapes, 3);

    private static TapeView<TElement> Concatenated<TElement>(TElement[][] entries, int width)
        where TElement : unmanaged
    {
        ulong[] offsets = new ulong[entries.Length + 1];
        int total = 0;
        for (int entry = 0; entry < entries.Length; entry++)
        {
            if (entries[entry].Length % width != 0)
            {
                throw new USearchException(Status.ShapeMismatch, $"entry {entry} is not a whole number of points");
            }

            total += entries[entry].Length;
            offsets[entry + 1] = (ulong)(total / width);
        }

        TElement[] elements = new TElement[total];
        int written = 0;
        foreach (TElement[] entry in entries)
        {
            entry.CopyTo(elements, written);
            written += entry.Length;
        }

        return Of<TElement>(elements, offsets, width);
    }
}

/// <summary>The entries a retrieval wrote into its <see cref="Arena"/> as one tape - the byte strings
/// <see cref="SequenceIndex.Get"/> found under a key, or the point clouds
/// <see cref="ShapesIndex.Get{TScalar}"/> found. Mirrors <c>usearch_tape_view_t</c> as an output.</summary>
/// <remarks>A view by value, readable until the arena's next call or disposal, as
/// <see cref="Matches"/> is.</remarks>
public readonly unsafe struct TapeEntries
{
    private readonly NativeTapeView _view;
    private readonly ArenaDescription _description;

    internal TapeEntries(NativeTapeView view, int count, int width, ArenaDescription description)
    {
        this._view = view;
        this.Count = view.Data == IntPtr.Zero ? 0 : count;
        this.Width = width;
        this._description = description;
    }

    /// <summary>How many entries the call wrote; zero for an absent key.</summary>
    public int Count { get; }

    /// <summary>The elements one offset step spans: one for a byte, three for a point.</summary>
    public int Width { get; }

    /// <summary>The scalar kind the elements hold.</summary>
    public Scalar Scalar => this._view.Scalar;

    /// <summary>Reads one entry as values of a managed element type.</summary>
    /// <typeparam name="TElement">The element type, <c>byte</c> for sequences.</typeparam>
    /// <param name="entry">The entry, in the order the call wrote them.</param>
    /// <returns>Its elements, valid until the arena's next call or disposal.</returns>
    /// <exception cref="USearchException"><see cref="Status.Device"/> over device memory.</exception>
    public ReadOnlySpan<TElement> Entry<TElement>(int entry) where TElement : unmanaged
    {
        if ((uint)entry >= (uint)this.Count)
        {
            throw new ArgumentOutOfRangeException(nameof(entry));
        }

        this._description.Host();
        nuint* offsets = (nuint*)this._view.Offsets;
        nuint begin = offsets[entry] * (nuint)this.Width;
        int length = (int)((offsets[entry + 1] * (nuint)this.Width) - begin);
        return new ReadOnlySpan<TElement>((TElement*)this._view.Data + begin, length);
    }
}

#endregion Tapes

#region Native Vocabulary

/// <summary>How wide a tape's offsets are, as <c>usearch_tape_layout_t</c> names them.</summary>
internal enum TapeLayout : int
{
    /// <summary>32-bit offsets, Apache Arrow's <c>Binary</c> and <c>List</c>.</summary>
    U32 = 0,
    /// <summary>64-bit offsets, Apache Arrow's <c>LargeBinary</c> and <c>LargeList</c>.</summary>
    U64 = 1,
}

/// <summary>A ragged batch, laid out as <c>usearch_tape_view_t</c>.</summary>
[StructLayout(LayoutKind.Sequential)]
internal struct NativeTapeView
{
    public IntPtr Data;
    public TapeLayout Layout;
    public IntPtr Offsets;
    public Scalar Scalar;
}

#endregion Native Vocabulary
