/**
 *  @file csharp/src/Cloud.Unum.USearch/Types.cs
 *  @author Ash Vardanian
 *  @date September 8, 2026
 *  @brief The vocabulary every modality shares: status codes, keys, executors, arenas and the exception they raise.
 */
using System;
using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.ExceptionServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

using static Cloud.Unum.USearch.NativeMethods;

namespace Cloud.Unum.USearch;

#region Errors

/// <summary>
/// What went wrong, in the vocabulary the C interface defines - one enumerator per
/// <c>usearch_status_*_k</c>, numbered alike. A <see cref="USearchException"/> never carries
/// <see cref="Success"/> or <see cref="Deferred"/>, which only <see cref="Deferred.Status"/> reads.
/// </summary>
public enum Status : int
{
    /// <summary>The call completed and carries no message.</summary>
    Success = 0,
    /// <summary>The call was submitted deferred and is not final yet; the one non-final status.</summary>
    Deferred = 1,
    /// <summary>An allocation or reservation could not be satisfied.</summary>
    OutOfMemory = 2,
    /// <summary>Dimensionality or row counts disagree.</summary>
    ShapeMismatch = 3,
    /// <summary>A stride or packing the engine cannot traverse.</summary>
    UnsupportedLayout = 4,
    /// <summary>A scalar, metric, or residency tag this build does not know.</summary>
    UnknownKind = 5,
    /// <summary>A mutating verb reached a handle that serves its checkpoint in place.</summary>
    ReadOnly = 6,
    /// <summary>A file could not be opened, read, written, or parsed.</summary>
    Io = 7,
    /// <summary>A device could not be selected, or its memory could not be claimed.</summary>
    Device = 8,
    /// <summary>A caller-supplied representation could not be decoded.</summary>
    Malformed = 9,
    /// <summary>The library reported neither success nor a failure.</summary>
    Internal = 10,
    /// <summary>A hard design ceiling was reached, like the 32-bit identity space.</summary>
    Overflow = 11,
    /// <summary>The verb needs a prior successful reservation that never happened.</summary>
    Uninitialized = 12,
    /// <summary>The request contradicts an already-frozen configuration.</summary>
    ConfigMismatch = 13,
    /// <summary>A synchronization contract was violated - an executor or an arena serving another
    /// call, or a blocking call from inside a callback of the executor it would wait on.</summary>
    Contended = 14,
    /// <summary>A progress handler asked the operation to stop; what the finished increments leave
    /// behind is each verb's own contract.</summary>
    Cancelled = 15,
    /// <summary>The verb is declared for every residency but this one does not implement it.</summary>
    Unsupported = 16,
    /// <summary>An argument breaks the verb's contract - a required value left out, or a name no
    /// table spells.</summary>
    InvalidArgument = 17,
}

/// <summary>
/// One failed call: a <see cref="USearch.Status"/> every caller can branch on, and the detail
/// the status cannot carry. Faults this binding catches before the boundary report the status
/// the engine would have reported for the same fault.
/// </summary>
public class USearchException : Exception
{
    /// <summary>Initializes a failure carrying a status and its detail.</summary>
    /// <param name="status">What went wrong.</param>
    /// <param name="message">The detail the status cannot carry.</param>
    public USearchException(Status status, string message) : base($"usearch: {status.Name()}: {message}")
    {
        this.Status = status;
    }

    /// <summary>Initializes a failure carrying a status, its detail, and the exception behind it.</summary>
    /// <param name="status">What went wrong.</param>
    /// <param name="message">The detail the status cannot carry.</param>
    /// <param name="cause">The exception this failure wraps.</param>
    public USearchException(Status status, string message, Exception cause)
        : base($"usearch: {status.Name()}: {message}", cause)
    {
        this.Status = status;
    }

    /// <summary>What went wrong.</summary>
    public Status Status { get; }
}

/// <summary>
/// The <c>usearch_outcome_t</c> one call writes: a status every caller can branch on, and a static
/// message carrying the detail the status cannot. Messages are static library storage, so nothing
/// here is deallocated.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
internal struct NativeOutcome
{
    /// <summary>What went wrong, <see cref="Status.Success"/>, or <see cref="Status.Deferred"/> until final.</summary>
    public Status Status;
    /// <summary>The static message, null exactly when the status is success.</summary>
    public IntPtr Message;

    /// <summary>Turns the outcome of a final call into a thrown failure.</summary>
    internal readonly void ThrowIfFailed()
    {
        if (this.Status == Status.Success)
        {
            return;
        }

        throw new USearchException(
            this.Status, Marshal.PtrToStringUTF8(this.Message) ?? "the library reported no detail");
    }
}

#endregion Errors

#region Names

/// <summary>
/// The canonical lowercase, dash-separated spelling of every closed set, read from the C name tables
/// rather than restated here, and the inverse that reads one back.
/// </summary>
public static partial class Names
{
    /// <summary>Spells a status, as <see cref="Status.ShapeMismatch"/> spells "shape-mismatch". Calls
    /// <c>usearch_status_name</c>.</summary>
    /// <param name="status">The status to spell.</param>
    /// <returns>The canonical spelling, or "unrecognized" for a value no status carries.</returns>
    public static string Name(this Status status) => Spelled(usearch_status_name(status));

    /// <summary>Spells a scalar kind, as <see cref="Scalar.F32"/> spells "f32". Calls
    /// <c>usearch_scalar_name</c>.</summary>
    /// <param name="scalar">The scalar kind to spell.</param>
    /// <returns>The canonical spelling.</returns>
    public static string Name(this Scalar scalar) => Spelled(usearch_scalar_name(scalar));

    /// <summary>Spells a mood, as <see cref="Mood.Parked"/> spells "parked". Calls
    /// <c>usearch_mood_name</c>.</summary>
    /// <param name="mood">The mood to spell.</param>
    /// <returns>The canonical spelling.</returns>
    public static string Name(this Mood mood) => Spelled(usearch_mood_name(mood));

    /// <summary>Spells a residency, as <see cref="Residency.Vram"/> spells "vram". Calls
    /// <c>usearch_residency_name</c>.</summary>
    /// <param name="residency">The residency to spell.</param>
    /// <returns>The canonical spelling.</returns>
    public static string Name(this Residency residency) => Spelled(usearch_residency_name(residency));

    /// <summary>Spells a traversal, as <see cref="Traversal.Scan"/> spells "scan". Calls
    /// <c>usearch_traversal_name</c>.</summary>
    /// <param name="traversal">The traversal to spell.</param>
    /// <returns>The canonical spelling.</returns>
    public static string Name(this Traversal traversal) => Spelled(usearch_traversal_name(traversal));

    /// <summary>Reads a status back from its spelling. Calls <c>usearch_status_from_name</c>.</summary>
    /// <param name="name">The canonical spelling.</param>
    /// <param name="status">The status it spells.</param>
    /// <returns>True when the spelling names a status, false otherwise.</returns>
    public static bool TryParse(string name, out Status status)
    {
        Status parsed = default;
        bool read = Read(name, view => usearch_status_from_name(view, out parsed)) == Status.Success;
        status = parsed;
        return read;
    }

    /// <summary>Reads a scalar kind back from its spelling. Calls <c>usearch_scalar_from_name</c>.</summary>
    /// <param name="name">The canonical spelling.</param>
    /// <param name="scalar">The scalar kind it spells.</param>
    /// <returns>True when the spelling names a scalar kind, false otherwise.</returns>
    public static bool TryParse(string name, out Scalar scalar)
    {
        scalar = Read(name, static view => usearch_scalar_from_name(view));
        return scalar.Name() == name;
    }

    /// <summary>Reads a mood back from its spelling. Calls <c>usearch_mood_from_name</c>.</summary>
    /// <param name="name">The canonical spelling.</param>
    /// <param name="mood">The mood it spells.</param>
    /// <returns>True when the spelling names a mood, false otherwise.</returns>
    public static bool TryParse(string name, out Mood mood)
    {
        Mood parsed = default;
        bool read = Read(name, view => usearch_mood_from_name(view, out parsed)) == Status.Success;
        mood = parsed;
        return read;
    }

    /// <summary>Reads a residency back from its spelling. Calls <c>usearch_residency_from_name</c>.</summary>
    /// <param name="name">The canonical spelling.</param>
    /// <param name="residency">The residency it spells.</param>
    /// <returns>True when the spelling names a residency, false otherwise.</returns>
    public static bool TryParse(string name, out Residency residency)
    {
        Residency parsed = default;
        bool read = Read(name, view => usearch_residency_from_name(view, out parsed)) == Status.Success;
        residency = parsed;
        return read;
    }

    /// <summary>Reads a traversal back from its spelling. Calls <c>usearch_traversal_from_name</c>.</summary>
    /// <param name="name">The canonical spelling.</param>
    /// <param name="traversal">The traversal it spells.</param>
    /// <returns>True when the spelling names a traversal, false otherwise.</returns>
    public static bool TryParse(string name, out Traversal traversal)
    {
        Traversal parsed = default;
        bool read = Read(name, view => usearch_traversal_from_name(view, out parsed)) == Status.Success;
        traversal = parsed;
        return read;
    }

    /// <summary>Copies a static spelling out of the library.</summary>
    private static string Spelled(IntPtr spelling) => Marshal.PtrToStringUTF8(spelling) ?? string.Empty;

    /// <summary>Lends a spelling to a C parser, whose fallbacks the round trip above tells apart from
    /// a genuine match, since the tables answer a default rather than a failure.</summary>
    private static unsafe TValue Read<TValue>(string name, Func<NativeBytesView, TValue> parse)
    {
        byte[] encoded = Encoding.UTF8.GetBytes(name);
        fixed (byte* data = encoded)
        {
            return parse(new NativeBytesView { Data = (IntPtr)data, Length = (nuint)encoded.Length });
        }
    }
}

#endregion Names

#region Keys

/// <summary>
/// A 128-bit entry identity, most significant byte first, rendered as a lowercase dashed UUID.
/// Mirrors <c>usearch_key_t</c>.
/// </summary>
/// <remarks>
/// Keys need not be unique: several entries may share one, and <see cref="DenseIndex.Count"/>
/// reports how many. Integer identities park in the low eight bytes, so the key 42 prints as
/// <c>00000000-0000-0000-0000-00000000002a</c>.
/// <para>
/// The sixteen bytes are held in the order the ABI fixes rather than in a <see cref="Guid"/> field,
/// whose first three groups sit little-endian in memory and would cross the boundary byte-swapped.
/// The UUID vocabulary is reached through <see cref="ToGuid"/> and <see cref="FromGuid"/> instead,
/// and it is what renders and parses a key.
/// </para>
/// </remarks>
[StructLayout(LayoutKind.Sequential)]
public struct Key : IEquatable<Key>
{
    /// <summary>The number of identity bytes a key carries.</summary>
    public const int ByteCount = 16;

    /// <summary>The sixteen identity bytes, most significant first.</summary>
    private KeyBytes _bytes;

    /// <summary>The sixteen identity bytes of one key, laid out as the C struct spells them.</summary>
    [InlineArray(ByteCount)]
    private struct KeyBytes
    {
        /// <summary>The first byte; the inline array supplies the fifteen that follow.</summary>
        private byte _first;
    }

    /// <summary>The all-zero key.</summary>
    public static Key Nil => default;

    /// <summary>Parks an integer identity in the low eight bytes, zeroing the rest.</summary>
    /// <param name="value">The integer identity.</param>
    /// <returns>The key holding that identity.</returns>
    public static Key FromUInt64(ulong value)
    {
        Key key = default;
        for (int position = 0; position < 8; position++)
        {
            key._bytes[15 - position] = (byte)(value >> (8 * position));
        }

        return key;
    }

    /// <summary>Reads the integer identity in the low eight bytes.</summary>
    /// <param name="value">The identity, when the high eight bytes carry nothing.</param>
    /// <returns>True when the high eight bytes are zero, false otherwise.</returns>
    public bool TryGetUInt64(out ulong value)
    {
        value = 0;
        for (int position = 0; position < 8; position++)
        {
            if (this._bytes[position] != 0)
            {
                return false;
            }
        }

        for (int position = 8; position < ByteCount; position++)
        {
            value = (value << 8) | this._bytes[position];
        }

        return true;
    }

    /// <summary>Parks an integer identity in the low eight bytes, zeroing the rest.</summary>
    /// <param name="value">The integer identity.</param>
    public static implicit operator Key(ulong value) => FromUInt64(value);

    /// <summary>Reads a UUID's canonical byte order as an identity.</summary>
    /// <param name="identity">The UUID.</param>
    /// <returns>The key its bytes spell.</returns>
    public static Key FromGuid(Guid identity)
    {
        Key key = default;
        Span<byte> bytes = key._bytes;
        identity.TryWriteBytes(bytes, bigEndian: true, out _);
        return key;
    }

    /// <summary>Reads the identity as a UUID, whose canonical byte order is the ABI's.</summary>
    /// <returns>The UUID the sixteen bytes spell.</returns>
    public Guid ToGuid()
    {
        ReadOnlySpan<byte> bytes = this._bytes;
        return new Guid(bytes, bigEndian: true);
    }

    /// <summary>Reads one identity byte, most significant first.</summary>
    /// <param name="position">The byte offset, from zero to fifteen.</param>
    /// <returns>The byte at that offset.</returns>
    public byte this[int position]
    {
        get
        {
            if ((uint)position >= ByteCount)
            {
                throw new ArgumentOutOfRangeException(nameof(position));
            }

            return this._bytes[position];
        }
    }

    /// <summary>Renders the key as a lowercase dashed UUID.</summary>
    /// <returns>The thirty-six character rendering.</returns>
    public override string ToString() => this.ToGuid().ToString("D");

    /// <summary>Reads the rendering <see cref="ToString"/> produces; dashes may sit anywhere.</summary>
    /// <param name="text">The thirty-two hexadecimal digits, dashed or not.</param>
    /// <returns>The key those digits spell.</returns>
    public static Key Parse(string text)
    {
        if (!TryParse(text, out Key key))
        {
            throw new USearchException(Status.Malformed, "the key is not thirty-two hexadecimal digits");
        }

        return key;
    }

    /// <summary>Reads the rendering <see cref="ToString"/> produces, without throwing.</summary>
    /// <remarks>
    /// The dashes are separators rather than positions, so they may sit anywhere or nowhere. What
    /// they leave behind must be thirty-two hexadecimal digits, of either case.
    /// </remarks>
    /// <param name="text">The thirty-two hexadecimal digits, dashed or not.</param>
    /// <param name="key">The key those digits spell, or the nil key when they do not.</param>
    /// <returns>True when the text parsed, false otherwise.</returns>
    public static bool TryParse(string? text, out Key key)
    {
        key = default;
        if (text is null)
        {
            return false;
        }

        Span<char> digits = stackalloc char[ByteCount * 2];
        int written = 0;
        foreach (char character in text)
        {
            if (character == '-')
            {
                continue;
            }

            if (written == digits.Length)
            {
                return false;
            }

            digits[written] = character;
            written++;
        }

        if (written != digits.Length || !Guid.TryParseExact(digits, "N", out Guid identity))
        {
            return false;
        }

        key = FromGuid(identity);
        return true;
    }

    /// <summary>Compares two keys byte by byte.</summary>
    /// <param name="other">The key to compare against.</param>
    /// <returns>True when every byte agrees.</returns>
    public bool Equals(Key other)
    {
        for (int position = 0; position < ByteCount; position++)
        {
            if (this._bytes[position] != other._bytes[position])
            {
                return false;
            }
        }

        return true;
    }

    /// <summary>Compares this key against any object.</summary>
    /// <param name="other">The object to compare against.</param>
    /// <returns>True when it is a key and every byte agrees.</returns>
    public override bool Equals(object? other) => other is Key key && this.Equals(key);

    /// <summary>Hashes the sixteen identity bytes.</summary>
    /// <returns>A hash agreeing with <see cref="Equals(Key)"/>.</returns>
    public override int GetHashCode()
    {
        HashCode hash = default;
        for (int position = 0; position < ByteCount; position++)
        {
            hash.Add(this._bytes[position]);
        }

        return hash.ToHashCode();
    }

    /// <summary>Compares two keys byte by byte.</summary>
    /// <param name="left">The first key.</param>
    /// <param name="right">The second key.</param>
    /// <returns>True when every byte agrees.</returns>
    public static bool operator ==(Key left, Key right) => left.Equals(right);

    /// <summary>Compares two keys byte by byte.</summary>
    /// <param name="left">The first key.</param>
    /// <param name="right">The second key.</param>
    /// <returns>True when any byte differs.</returns>
    public static bool operator !=(Key left, Key right) => !left.Equals(right);
}

#endregion Keys

#region Scalars

/// <summary>
/// A stored or exchanged scalar type, named by width and encoding. Sub-byte kinds pack several
/// dimensions per byte, as their <c>xN</c> suffixes spell. Mirrors <c>usearch_scalar_t</c>.
/// </summary>
/// <remarks>
/// The enumerators are ordered by shrinking dynamic range, and their values carry the numbering
/// gaps the C enum reserves for kinds not yet implemented.
/// </remarks>
public enum Scalar : int
{
    /// <summary>A type the library could not name.</summary>
    Unknown = 0,
    /// <summary>64-bit double precision.</summary>
    F64 = 1,
    /// <summary>32-bit single precision.</summary>
    F32 = 2,
    /// <summary>The brain floating-point format.</summary>
    BF16 = 3,
    /// <summary>IEEE half precision.</summary>
    F16 = 4,
    /// <summary>8-bit floating point with five exponent bits.</summary>
    E5M2 = 5,
    /// <summary>8-bit floating point with four exponent bits.</summary>
    E4M3 = 6,
    /// <summary>6-bit floating point with three exponent bits, one per byte.</summary>
    E3M2 = 7,
    /// <summary>6-bit floating point with two exponent bits, one per byte.</summary>
    E2M3 = 8,
    /// <summary>An 8-bit signed integer.</summary>
    I8 = 10,
    /// <summary>Two 4-bit signed integers per byte.</summary>
    I4x2 = 11,
    /// <summary>A 32-bit signed integer, the coordinates of geospatial points.</summary>
    I32 = 12,
    /// <summary>An 8-bit unsigned integer.</summary>
    U8 = 20,
    /// <summary>Two 4-bit unsigned integers per byte.</summary>
    U4x2 = 21,
    /// <summary>Eight single-bit dimensions per byte.</summary>
    U1x8 = 22,
}

/// <summary>
/// The width arithmetic every scalar kind implies, and the mapping from a managed element type
/// onto the kind the library knows it by.
/// </summary>
internal static class ScalarKinds
{
    /// <summary>Reports the byte width of one stored value; zero for an unnameable kind.</summary>
    /// <param name="kind">The scalar kind.</param>
    /// <returns>The bytes one value occupies.</returns>
    internal static int ValueBytes(this Scalar kind) => kind switch
    {
        Scalar.F64 => 8,
        Scalar.F32 => 4,
        Scalar.BF16 => 2,
        Scalar.F16 => 2,
        Scalar.E5M2 => 1,
        Scalar.E4M3 => 1,
        Scalar.E3M2 => 1,
        Scalar.E2M3 => 1,
        Scalar.I8 => 1,
        Scalar.I4x2 => 1,
        Scalar.I32 => 4,
        Scalar.U8 => 1,
        Scalar.U4x2 => 1,
        Scalar.U1x8 => 1,
        _ => 0,
    };

    /// <summary>Reports how many logical dimensions one stored value carries.</summary>
    /// <param name="kind">The scalar kind.</param>
    /// <returns>One for the byte-and-wider kinds, more for the packed ones.</returns>
    internal static int DimensionsPerValue(this Scalar kind) => kind switch
    {
        Scalar.I4x2 => 2,
        Scalar.U4x2 => 2,
        Scalar.U1x8 => 8,
        _ => 1,
    };

    /// <summary>Reports the bytes one row of the given width occupies, a half-filled trailing value
    /// counting whole.</summary>
    /// <param name="kind">The scalar kind the row holds.</param>
    /// <param name="dimensions">The row's logical width.</param>
    /// <returns>The bytes between consecutive row starts in a packed matrix.</returns>
    internal static ulong RowBytes(this Scalar kind, ulong dimensions)
    {
        ulong perValue = (ulong)kind.DimensionsPerValue();
        return checked((dimensions + perValue - 1) / perValue * (ulong)kind.ValueBytes());
    }

    /// <summary>Names the scalar kind of a managed element type.</summary>
    /// <remarks>Only the kinds .NET has an element type for are nameable this way; the packed and
    /// minifloat kinds are named explicitly when a view is built.</remarks>
    /// <typeparam name="TScalar">The element type of a vector row.</typeparam>
    /// <returns>The kind the library knows it by.</returns>
    internal static Scalar KindOf<TScalar>() where TScalar : unmanaged
    {
        if (typeof(TScalar) == typeof(double))
        {
            return Scalar.F64;
        }

        if (typeof(TScalar) == typeof(float))
        {
            return Scalar.F32;
        }

        if (typeof(TScalar) == typeof(Half))
        {
            return Scalar.F16;
        }

        if (typeof(TScalar) == typeof(sbyte))
        {
            return Scalar.I8;
        }

        if (typeof(TScalar) == typeof(byte))
        {
            return Scalar.U8;
        }

        throw new USearchException(
            Status.UnknownKind, $"no scalar kind names the element type {typeof(TScalar).Name}");
    }
}

#endregion Scalars

#region Rows

/// <summary>
/// A caller-owned slab viewed as rows of equal width: a base, a row count, a logical width, and a
/// byte pitch between consecutive row starts. It is the shape every batch argument takes, as
/// <see cref="DenseIndex.Insert"/> and <see cref="DenseIndex.Search"/> read it. Mirrors
/// <c>usearch_rows_view_t</c>.
/// </summary>
/// <remarks>
/// A view borrows and never owns: it holds the <see cref="Memory{T}"/> it was built from, so the
/// slab stays reachable for as long as the view is, and a deferred call pins it until final.
/// <para>
/// The scalar kind is carried rather than derived, so a slab of bytes reaches the index as packed
/// <see cref="Scalar.U1x8"/> bit sets or as plain <see cref="Scalar.U8"/> integers, the
/// caller having said which.
/// </para>
/// </remarks>
public readonly struct RowsView
{
    /// <summary>The slab the rows are read from, pinned for the length of a call.</summary>
    private readonly ISlab? _slab;

    private RowsView(ISlab slab, int rows, int dimensions, int strideBytes, Scalar scalar)
    {
        this._slab = slab;
        this.Rows = rows;
        this.Dimensions = dimensions;
        this.StrideBytes = strideBytes;
        this.Scalar = scalar;
    }

    /// <summary>How many rows the view spans.</summary>
    public int Rows { get; }

    /// <summary>The logical width of one row.</summary>
    public int Dimensions { get; }

    /// <summary>The byte pitch between consecutive row starts.</summary>
    public int StrideBytes { get; }

    /// <summary>The scalar type the rows hold.</summary>
    public Scalar Scalar { get; }

    /// <summary>Views a packed slab as rows of the given width, naming the kind its element type implies.</summary>
    /// <typeparam name="TScalar">The element type of the slab.</typeparam>
    /// <param name="data">The slab, which divides evenly into rows.</param>
    /// <param name="dimensions">The logical width of one row.</param>
    /// <returns>The view over that slab.</returns>
    public static RowsView RowsOf<TScalar>(Memory<TScalar> data, int dimensions) where TScalar : unmanaged =>
        RowsOf(data, dimensions, ScalarKinds.KindOf<TScalar>());

    /// <summary>Views a packed slab as rows of the given width, under a kind the caller names.</summary>
    /// <typeparam name="TScalar">The element type of the slab.</typeparam>
    /// <param name="data">The slab, which divides evenly into rows.</param>
    /// <param name="dimensions">The logical width of one row.</param>
    /// <param name="scalar">The scalar type the values hold.</param>
    /// <returns>The view over that slab.</returns>
    public static unsafe RowsView RowsOf<TScalar>(Memory<TScalar> data, int dimensions, Scalar scalar)
        where TScalar : unmanaged
    {
        int rowBytes = CheckedRowBytes<TScalar>(dimensions, scalar);
        long bytes = (long)data.Length * sizeof(TScalar);
        if (bytes % rowBytes != 0)
        {
            throw new USearchException(
                Status.ShapeMismatch, $"{bytes} bytes do not divide into rows of {rowBytes}");
        }

        return new RowsView(new Slab<TScalar>(data), (int)(bytes / rowBytes), dimensions, rowBytes, scalar);
    }

    /// <summary>
    /// Views a slab whose pitch exceeds its row width - the padded layout a memory-mapped dataset
    /// arrives in.
    /// </summary>
    /// <typeparam name="TScalar">The element type of the slab.</typeparam>
    /// <param name="data">The slab the rows are cut from.</param>
    /// <param name="rows">How many rows the view spans.</param>
    /// <param name="dimensions">The logical width of one row.</param>
    /// <param name="strideBytes">The byte pitch between consecutive row starts.</param>
    /// <param name="scalar">The scalar type the values hold.</param>
    /// <returns>The view over that slab.</returns>
    public static unsafe RowsView StridedRowsOf<TScalar>(
        Memory<TScalar> data, int rows, int dimensions, int strideBytes, Scalar scalar)
        where TScalar : unmanaged
    {
        int rowBytes = CheckedRowBytes<TScalar>(dimensions, scalar);
        if (strideBytes < rowBytes)
        {
            throw new USearchException(
                Status.UnsupportedLayout, $"a stride of {strideBytes} bytes cannot hold a row of {rowBytes}");
        }

        if (rows < 0)
        {
            throw new USearchException(Status.ShapeMismatch, "a view cannot span a negative row count");
        }

        long bytes = (long)data.Length * sizeof(TScalar);
        long span = rows == 0 ? 0 : ((long)(rows - 1) * strideBytes) + rowBytes;
        if (span > bytes)
        {
            throw new USearchException(
                Status.ShapeMismatch, $"{rows} rows of stride {strideBytes} overrun a slab of {bytes} bytes");
        }

        return new RowsView(new Slab<TScalar>(data), rows, dimensions, strideBytes, scalar);
    }

    /// <summary>The slab's bytes, for a call to fix in place; a default view has none.</summary>
    internal Span<byte> Bytes => this._slab is null ? default : this._slab.Bytes;

    /// <summary>Pins the slab until the handle is disposed; a default view pins nothing.</summary>
    /// <returns>The pin, whose pointer is the first row.</returns>
    internal MemoryHandle Pin() => this._slab?.Pin() ?? default;

    /// <summary>Lends the slab as the input descriptor every batch verb reads, whose row count
    /// travels as the call's own count.</summary>
    /// <param name="data">The first row, fixed or pinned for the call.</param>
    /// <returns>The C struct the library reads.</returns>
    internal unsafe NativeRowsView AsView(byte* data) => new()
    {
        Data = (IntPtr)data,
        Stride = (nuint)this.StrideBytes,
        Columns = (nuint)this.Dimensions,
        Scalar = this.Scalar,
    };

    private static unsafe int CheckedRowBytes<TScalar>(int dimensions, Scalar scalar) where TScalar : unmanaged
    {
        if (dimensions <= 0)
        {
            throw new USearchException(Status.ShapeMismatch, "a row cannot be zero dimensions wide");
        }

        int valueBytes = scalar.ValueBytes();
        if (valueBytes != sizeof(TScalar))
        {
            throw new USearchException(
                Status.UnsupportedLayout, $"a {scalar.Name()} value is {valueBytes} bytes, not {sizeof(TScalar)}");
        }

        ulong rowBytes = scalar.RowBytes((ulong)dimensions);
        if (rowBytes > int.MaxValue)
        {
            throw new USearchException(Status.Overflow, "one row exceeds what a span can address");
        }

        return (int)rowBytes;
    }

    /// <summary>A slab of any element type, pinned without knowing which.</summary>
    private interface ISlab
    {
        Span<byte> Bytes { get; }

        MemoryHandle Pin();
    }

    private sealed class Slab<TScalar>(Memory<TScalar> memory) : ISlab where TScalar : unmanaged
    {
        public Span<byte> Bytes => MemoryMarshal.AsBytes(memory.Span);

        public MemoryHandle Pin() => memory.Pin();
    }
}

#endregion Rows

#region Results

/// <summary>
/// A ranked run a call wrote into its <see cref="Arena"/> - one list per query of up to
/// <see cref="Columns"/> keys, each paired with its distance. Mirrors <c>usearch_matches_view_t</c>.
/// </summary>
/// <remarks>
/// A view by value over the arena, readable until that arena's next call or disposal, after which
/// every accessor refuses with <see cref="Status.Uninitialized"/>; over device memory they refuse
/// with <see cref="Status.Device"/>. A span an accessor lends escapes that check, so it is read before
/// the arena's next call.
/// </remarks>
public readonly unsafe struct Matches
{
    private readonly Arena _arena;
    private readonly NativeMatchesView _view;
    private readonly int _generation;

    internal Matches(Arena arena, int generation, NativeMatchesView view, int queries)
    {
        this._arena = arena;
        this._generation = generation;
        this._view = view;
        this.Queries = queries;
    }

    /// <summary>How many queries the call resolved.</summary>
    public int Queries { get; }

    /// <summary>The columns per query the call wrote, at most its wanted - the "k" in kANN.</summary>
    public int Columns => (int)this._view.Columns;

    /// <summary>Reads one query's matches from the run.</summary>
    /// <param name="query">The query's offset in the batch.</param>
    /// <returns>That query's matches, as many as the call wrote.</returns>
    public QueryMatches this[int query]
    {
        get
        {
            if ((uint)query >= (uint)this.Queries)
            {
                throw new ArgumentOutOfRangeException(nameof(query));
            }

            nuint first = (nuint)query * this._view.Columns;
            return new QueryMatches(
                this._arena, this._generation, this._view.Keys + first, this._view.Scores + first,
                this._view.Counts + query);
        }
    }
}

/// <summary>One query's matches from a <see cref="Matches"/> run, or one pull of a
/// <see cref="DenseSearchStream"/>, read in place from the arena.</summary>
public readonly unsafe struct QueryMatches
{
    private readonly Arena _arena;
    private readonly int _generation;
    private readonly Key* _keys;
    private readonly double* _distances;
    private readonly nuint* _count;

    internal QueryMatches(Arena arena, int generation, Key* keys, double* distances, nuint* count)
    {
        this._arena = arena;
        this._generation = generation;
        this._keys = keys;
        this._distances = distances;
        this._count = count;
    }

    /// <summary>How many matches the call wrote for the query.</summary>
    public int Count
    {
        get
        {
            Arena.Readable(this._arena, this._generation);
            return (int)*this._count;
        }
    }

    /// <summary>The keys the query matched, nearest first.</summary>
    public ReadOnlySpan<Key> Keys => new(this._keys, this.Count);

    /// <summary>The distances positionally paired with <see cref="Keys"/>.</summary>
    public ReadOnlySpan<double> Distances => new(this._distances, this.Count);
}

#endregion Results

#region Filtering

/// <summary>Whether a filtered search admits one candidate key or passes over it, as the C
/// predicate answers.</summary>
internal enum Admission : int
{
    /// <summary>Skip the key; the walk still routes through it.</summary>
    Reject = 0,
    /// <summary>Report the key among the matches.</summary>
    Admit = 1,
}

/// <summary>How costly one predicate call is, which decides how many keys each call carries.</summary>
internal enum PredicateCost : int
{
    /// <summary>Asked as candidates arrive, at most four keys a call, with nothing buffered.</summary>
    Cheap = 0,
    /// <summary>Asked once per traversal round or scan run about every candidate it gathered.</summary>
    Expensive = 1,
}

#endregion Filtering

#region Executors

/// <summary>Where an executor computes, and so where an <see cref="Arena"/> it makes keeps its memory.
/// Mirrors <c>usearch_residency_t</c>.</summary>
public enum Residency : int
{
    /// <summary>Host memory, traversed by CPU threads.</summary>
    Ram = 1,
    /// <summary>Accelerator memory, traversed by GPU warps on a chosen device.</summary>
    Vram = 2,
    /// <summary>Persisted storage, traversed through pluggable I/O engines and sized past RAM.</summary>
    Nvme = 3,
}

/// <summary>How a search reaches the indexed entries it scores. Mirrors <c>usearch_traversal_t</c>.</summary>
public enum Traversal : int
{
    /// <summary>Walk the graph, scoring the entries the expansion reaches - approximate, the default.</summary>
    Walk = 1,
    /// <summary>Score every entry published when the query's scan starts - exact, on RAM and NVMe
    /// handles.</summary>
    Scan = 2,
}

/// <summary>What an executor does with its threads or streams once it goes idle. Mirrors
/// <c>usearch_mood_t</c>.</summary>
public enum Mood : int
{
    /// <summary>Idle threads sleep; the next call pays a wake-up. The default.</summary>
    Parked = 0,
    /// <summary>Idle threads stay hot, trading power for the next call's latency.</summary>
    Spinning = 1,
}

/// <summary>A CPU family USearch ships tuned variants for, which a RAM or NVMe placement or executor may pin.</summary>
public enum CPUClass : int
{
    /// <summary>The portable floor, with no SIMD assumptions.</summary>
    Serial = 1,
    /// <summary>Haswell-class x86: AVX2 and FMA.</summary>
    IntelHaswell = 2,
    /// <summary>Sierra Forest-class x86: AVX-VNNI without AVX-512.</summary>
    IntelSierra = 3,
    /// <summary>Skylake-class x86: the AVX-512 foundation.</summary>
    IntelSkylake = 4,
    /// <summary>Icelake-class x86: AVX-512 VNNI and VPOPCNT.</summary>
    IntelIcelake = 5,
    /// <summary>Genoa-class x86: native bf16 dot products over AVX-512.</summary>
    AmdGenoa = 6,
    /// <summary>Sapphire Rapids-class x86: AVX-512 FP16 and the AMX tiles.</summary>
    IntelSapphire = 7,
    /// <summary>Granite Rapids-class x86: the AMX-FP16 tiles.</summary>
    IntelGranite = 8,
    /// <summary>Diamond Rapids-class x86: AVX10.2's FP8 conversions.</summary>
    IntelDiamond = 9,
    /// <summary>Graviton2-class Arm: the bare NEON floor.</summary>
    AwsG2 = 10,
    /// <summary>Armv8.6 NEON with dot products and bf16.</summary>
    AppleM2 = 11,
    /// <summary>Apple M4-class Arm: the SME outer-product tiles.</summary>
    AppleM4 = 12,
    /// <summary>Graviton5-class Arm: Armv9.2.</summary>
    AwsG5 = 13,
    /// <summary>Vera-class Arm: FP8 dot products.</summary>
    NvidiaVera = 14,
    /// <summary>Alibaba C930-class RISC-V: RVA23 vectors.</summary>
    AlibabaC930 = 15,
    /// <summary>WebAssembly with fixed-width simd128.</summary>
    Wasm32V128 = 16,
    /// <summary>WebAssembly with relaxed SIMD, threads and 64-bit addressing.</summary>
    Wasm64Extended = 17,
    /// <summary>The one class of a monoversioned build.</summary>
    Monoversioned = 18,
}

/// <summary>A GPU family USearch ships tuned variants for, which a VRAM placement or executor may pin.</summary>
public enum GPUClass : int
{
    /// <summary>Ampere-class NVIDIA, and every NVIDIA GPU without a class of its own.</summary>
    NvidiaAmpere = 1,
    /// <summary>Ada-class NVIDIA: the fp8 conversions.</summary>
    NvidiaAda = 2,
    /// <summary>Hopper-class NVIDIA.</summary>
    NvidiaHopper = 3,
    /// <summary>Blackwell-class NVIDIA.</summary>
    NvidiaBlackwell = 4,
    /// <summary>Consumer Blackwell NVIDIA.</summary>
    NvidiaBlackwellRTX = 5,
    /// <summary>Rubin-class NVIDIA.</summary>
    NvidiaRubin = 6,
    /// <summary>CDNA3 AMD: gfx942.</summary>
    AmdCDNA3 = 7,
    /// <summary>CDNA4 AMD: gfx950.</summary>
    AmdCDNA4 = 8,
    /// <summary>CDNA5 AMD: gfx1250.</summary>
    AmdCDNA5 = 9,
    /// <summary>Apple7-family GPUs, and every Apple GPU without a class of its own.</summary>
    AppleMetal7 = 10,
    /// <summary>Apple9-family GPUs.</summary>
    AppleMetal9 = 11,
    /// <summary>Apple10-family GPUs.</summary>
    AppleMetal10 = 12,
}

/// <summary>
/// The workers every fanned-out verb runs on, shared across indexes.
/// </summary>
/// <remarks>
/// An executor owns host threads or device streams; an index owns entries. Keeping them apart is what
/// lets one pool serve several indexes, and what makes the cost of spawning visible at the line
/// that spawns rather than hidden in a constructor.
/// <para>
/// One factory per C constructor, each taking only what its own hardware has: host workers and
/// memory domains for RAM, a borrowed ForkUnion pool for <see cref="OnPool"/>, one submission ring
/// apiece per device path for NVMe, and device ordinals with optional borrowed streams for VRAM,
/// where a single host thread drives one stream per device and there is no thread count to name.
/// The kind must match the residency of every handle the executor drives, and a mismatch fails
/// the call with <see cref="Status.Device"/>.
/// </para>
/// <para>
/// A verb runs on an executor when its receiver names one, as <c>index.On(pool)</c> and
/// <c>index.Deferred(pool)</c> do, and on the caller's own thread when called on the index itself:
/// single-threaded on RAM, on the handle's device and default stream on VRAM.
/// One call holds an executor at a time, and a second finding it held finishes
/// <see cref="Status.Contended"/>, as nothing queues, while handles stay freely concurrent - it is
/// the executor that serializes, never the index. A deferred call on a caller-exclusive pool runs on
/// the pool's thread 0, so polling alone finishes it, while a single thread or a VRAM executor runs
/// it at submission.
/// </para>
/// <example>
/// <code>
/// using Executor pool = Executor.ForRam(threads: 8);
/// using DenseIndex index = DenseIndex.InitRam(DenseOptions.Default with { Dimensions = 3 });
/// index.On(pool).Insert(keys, RowsView.RowsOf&lt;float&gt;(vector, 3));
/// </code>
/// </example>
/// </remarks>
public sealed unsafe class Executor : IDisposable
{
    private readonly NativeExecutorHandle _handle;

    /// <summary>Adopts a handle one of the factories below spawned.</summary>
    /// <param name="handle">The engine handle the factory claimed.</param>
    private Executor(NativeExecutorHandle handle) => this._handle = handle;

    /// <summary>
    /// Spawns host workers for RAM-resident indexes, interleaved across the compute domains local to
    /// the named memory domains - the very IDs the indexes were initialized with. Calls
    /// <c>usearch_executor_for_ram</c>.
    /// </summary>
    /// <param name="threads">The worker count; zero takes 7/8 of the cores those domains carry,
    /// since full occupancy starves the ordered publication path.</param>
    /// <param name="memoryDomains">NUMA node IDs to place shards on; empty derives every node the
    /// process may allocate on.</param>
    /// <param name="cpuClass">The class whose waiters and atomics the threads use; null derives the
    /// CPU's own.</param>
    /// <returns>The spawned executor.</returns>
    public static Executor ForRam(
        ulong threads = 0, ReadOnlySpan<int> memoryDomains = default, CPUClass? cpuClass = null)
    {
        IntPtr spawned = IntPtr.Zero;
        NativeOutcome outcome = default;
        fixed (int* domains = memoryDomains)
        {
            NativeExecutorForRam arguments = new()
            {
                Executor = &spawned,
                Outcome = &outcome,
                Ram = new NativeRamOptions
                {
                    MemoryDomains = (IntPtr)domains, MemoryDomainsCount = (nuint)memoryDomains.Length,
                    CpuClass = (int)(cpuClass ?? 0),
                },
                Threads = (nuint)threads,
            };
            usearch_executor_for_ram(&arguments);
        }

        return new Executor(EngineHandle.Created<NativeExecutorHandle>(spawned, outcome));
    }

    /// <summary>
    /// Wraps a ForkUnion pool another library already runs, so one process keeps one set of
    /// threads while each library keeps its own scratch. Calls <c>usearch_executor_on_pool</c>.
    /// </summary>
    /// <param name="pool">The borrowed <c>fu_pool_t</c>, which must outlive the executor and keeps
    /// its lender's mood.</param>
    /// <param name="memoryDomains">NUMA node IDs of the indexes it will drive; empty derives.</param>
    /// <param name="cpuClass">The class whose waiters and atomics the threads use; null derives the
    /// CPU's own.</param>
    /// <returns>The executor over that pool.</returns>
    public static Executor OnPool(
        IntPtr pool, ReadOnlySpan<int> memoryDomains = default, CPUClass? cpuClass = null)
    {
        IntPtr spawned = IntPtr.Zero;
        NativeOutcome outcome = default;
        fixed (int* domains = memoryDomains)
        {
            NativeExecutorOnPool arguments = new()
            {
                Executor = &spawned,
                Outcome = &outcome,
                Ram = new NativeRamOptions
                {
                    MemoryDomains = (IntPtr)domains, MemoryDomainsCount = (nuint)memoryDomains.Length,
                    CpuClass = (int)(cpuClass ?? 0),
                },
                Pool = pool,
            };
            usearch_executor_on_pool(&arguments);
        }

        return new Executor(EngineHandle.Created<NativeExecutorHandle>(spawned, outcome));
    }

    /// <summary>
    /// Spawns host workers for NVMe-resident indexes, one submission ring apiece, across the compute
    /// domains local to the devices the paths name. Calls <c>usearch_executor_for_nvme</c>.
    /// </summary>
    /// <param name="threads">The ring count; zero derives from those devices' queue
    /// capabilities.</param>
    /// <param name="paths">Devices or directories to place shards on; empty leaves each shard where
    /// its manifest points.</param>
    /// <param name="cpuClass">The class whose waiters and atomics the threads use; null derives the
    /// CPU's own.</param>
    /// <returns>The spawned executor.</returns>
    public static Executor ForNvme(
        ulong threads = 0, ReadOnlySpan<string> paths = default, CPUClass? cpuClass = null)
    {
        IntPtr spawned = IntPtr.Zero;
        NativeOutcome outcome = default;
        NativeBytesView* views = Paths.Encode(paths);
        try
        {
            NativeExecutorForNvme arguments = new()
            {
                Executor = &spawned,
                Outcome = &outcome,
                Nvme = new NativeNvmeOptions
                {
                    Paths = (IntPtr)views, PathsCount = (nuint)paths.Length, CpuClass = (int)(cpuClass ?? 0),
                },
                Threads = (nuint)threads,
            };
            usearch_executor_for_nvme(&arguments);
        }
        finally
        {
            NativeMemory.Free(views);
        }

        return new Executor(EngineHandle.Created<NativeExecutorHandle>(spawned, outcome));
    }

    /// <summary>
    /// Opens one submission stream on every named device, or borrows the caller's, all of them
    /// driven by the one host thread that calls the verbs - which is why this factory takes no
    /// thread count. Calls <c>usearch_executor_for_vram</c>.
    /// </summary>
    /// <param name="devices">CUDA device ordinals to place shards on; empty derives one free
    /// device.</param>
    /// <param name="streams">Borrowed CUDA streams, one per device in that order; empty opens its
    /// own.</param>
    /// <param name="scratchLimit">The ceiling on scratch bytes per device; zero lets scratch grow as
    /// geometry asks.</param>
    /// <param name="gpuClass">A class the devices must run, refused with
    /// <see cref="Status.Unsupported"/> otherwise; null takes the devices' own.</param>
    /// <returns>The opened executor.</returns>
    public static Executor ForVram(
        ReadOnlySpan<int> devices = default,
        ReadOnlySpan<IntPtr> streams = default,
        ulong scratchLimit = 0,
        GPUClass? gpuClass = null)
    {
        // The library reads one stream per device from the pointer alone, so only this side sees a short list.
        if (!streams.IsEmpty && streams.Length != devices.Length)
        {
            throw new USearchException(Status.ShapeMismatch, "the borrowed streams do not pair one per device");
        }

        IntPtr spawned = IntPtr.Zero;
        NativeOutcome outcome = default;
        fixed (int* ordinals = devices)
        fixed (IntPtr* borrowed = streams)
        {
            NativeExecutorForVram arguments = new()
            {
                Executor = &spawned,
                Outcome = &outcome,
                Vram = new NativeVramOptions
                {
                    Devices = (IntPtr)ordinals, DevicesCount = (nuint)devices.Length, GpuClass = (int)(gpuClass ?? 0),
                },
                Streams = (IntPtr)borrowed,
                ScratchLimit = (nuint)scratchLimit,
            };
            usearch_executor_for_vram(&arguments);
        }

        return new Executor(EngineHandle.Created<NativeExecutorHandle>(spawned, outcome));
    }

    /// <summary>Waits until no call holds this executor, then sets the mood its workers take until
    /// the next call. Calls <c>usearch_executor_drain</c>.</summary>
    /// <param name="mood">What the executor does with its workers once idle; the last drain wins.</param>
    public void Drain(Mood mood = Mood.Parked)
    {
        using Lease executor = Lease.Of(this._handle, "executor");
        NativeOutcome outcome = default;
        NativeExecutorDrain arguments = new() { Executor = executor.Pointer, Outcome = &outcome, Mood = mood };
        usearch_executor_drain(&arguments);
        outcome.ThrowIfFailed();
    }

    /// <summary>
    /// Waits out the call holding the executor, releases the scratch it owns, and frees it. Freeing
    /// cannot fail, and a freed executor must not be used. Calls <c>usearch_executor_free</c>.
    /// </summary>
    public void Dispose() => this._handle.Dispose();

    /// <summary>The engine handle the verbs lease.</summary>
    internal NativeExecutorHandle Handle => this._handle;
}

#endregion Executors

#region Arenas

/// <summary>
/// Memory the library writes every output whose length scales with a call into - the matches of
/// <see cref="DenseIndex.Search"/>, the rows of <see cref="DenseIndex.Get"/>, the pairs of
/// <see cref="DenseIndex.Multiply"/>, the bytes of <see cref="DenseIndex.SaveBuffer"/>, cursor and
/// stream batches - reused across calls so a steady loop allocates once. Mirrors <c>usearch_arena_t</c>.
/// </summary>
/// <remarks>
/// The executor it is made for places it where it computes, host pages or its GPU's memory, decided
/// once, and nothing is allocated until a call writes into it. Every call given it rewinds it, so
/// the views the previous call answered refuse with <see cref="Status.Uninitialized"/> from then on,
/// as they do once it is disposed. An arena serves one call at a time: a blocking call borrows it
/// until it returns, a deferred one until final, and a second call naming it meanwhile finishes
/// with <see cref="Status.Contended"/>. A <see cref="Residency.Vram"/> arena holds device
/// memory, which no view lends to the host, so their accessors refuse with <see cref="Status.Device"/>.
/// It grows, fills and frees on its executor's first stream, so it serves only calls on that executor
/// and holds it open until disposed, which defers freeing an executor disposed first; a host arena
/// may outlive its executor.
/// </remarks>
public sealed unsafe class Arena : IDisposable
{
    private readonly NativeArenaHandle _handle;
    private int _generation;

    /// <summary>Makes an empty arena where <paramref name="executor"/> computes. Calls
    /// <c>usearch_arena_init</c>.</summary>
    /// <param name="executor">The executor whose calls the arena serves; null makes host pages for
    /// calls on the caller's thread.</param>
    public Arena(Executor? executor = null)
    {
        using Lease pool = Lease.Of(executor?.Handle, "executor");
        IntPtr created = IntPtr.Zero;
        NativeOutcome outcome = default;
        NativeArenaInit arguments = new() { Arena = &created, Executor = pool.Pointer, Outcome = &outcome };
        usearch_arena_init(&arguments);
        this._handle = EngineHandle.Created<NativeArenaHandle>(created, outcome);
        if (this.Describe().Residency == Residency.Vram)
        {
            this._handle.Executor = Lease.Of(executor!.Handle, "executor");
        }
    }

    /// <summary>Reports where the outputs live, as the executor the arena was made for decided, so a
    /// caller reads them as host memory or hands them to device code without remembering that
    /// executor. Calls <c>usearch_arena_describe</c>.</summary>
    /// <returns>The residency of the outputs and the device holding them, if any.</returns>
    public ArenaDescription Describe()
    {
        using Lease arena = Lease.Of(this._handle, "arena");
        NativeOutcome outcome = default;
        NativeArenaDescription description = default;
        NativeArenaDescribe arguments = new() { Arena = arena.Pointer, Outcome = &outcome, Description = &description };
        usearch_arena_describe(&arguments);
        outcome.ThrowIfFailed();
        return new(description.Residency, description.Residency == Residency.Vram ? (int)description.Device : null);
    }

    /// <summary>Frees the arena and every output in it. Calls <c>usearch_arena_free</c>.</summary>
    public void Dispose() => this._handle.Dispose();

    /// <summary>Leases the arena for the length of one C call, as every handle is leased.</summary>
    /// <param name="arena">The arena the caller lent; a null one is refused.</param>
    /// <returns>The lease, whose pointer the call names.</returns>
    internal static Lease Lend(Arena? arena) => arena is null
        ? throw new USearchException(Status.InvalidArgument, "the arena must not be null")
        : Lease.Of(arena._handle, "arena");

    /// <summary>Retires the views of every earlier call, unless this one was refused for naming a busy
    /// arena, which leaves the arena as the busy call has it.</summary>
    /// <param name="arena">The arena the call named.</param>
    /// <param name="outcome">The outcome the C verb wrote on return.</param>
    /// <returns>The generation the call's views belong to.</returns>
    internal static int Rewound(Arena arena, NativeOutcome* outcome) =>
        usearch_outcome_status(outcome) == Status.Contended
            ? Volatile.Read(ref arena._generation)
            : Interlocked.Increment(ref arena._generation);

    /// <summary>Refuses a view the arena has since rewound or freed, or one over device memory.</summary>
    /// <param name="arena">The arena the view reads; null for a default view.</param>
    /// <param name="generation">The generation the view recorded.</param>
    internal static void Readable(Arena? arena, int generation)
    {
        if (arena is null || arena._handle.IsClosed || Volatile.Read(ref arena._generation) != generation)
        {
            throw new USearchException(Status.Uninitialized, "the arena was rewound or freed since this view");
        }

        if (arena.Describe().Residency == Residency.Vram)
        {
            throw new USearchException(Status.Device, "the view is device memory");
        }
    }
}

/// <summary>Where an arena's outputs live, as <see cref="Arena.Describe"/> reports it. Mirrors
/// <c>usearch_arena_description_t</c>.</summary>
/// <param name="Residency"><see cref="Residency.Ram"/> for host pages, <see cref="Residency.Vram"/> for
/// one device's memory.</param>
/// <param name="Device">The ordinal of the device holding the outputs; null for host pages.</param>
public readonly record struct ArenaDescription(Residency Residency, int? Device);

#endregion Arenas

#region Receivers

/// <summary>
/// A submitted call, which owns its outcome and the pins on its inputs until the call is final, and
/// borrows its index and arena as a blocking call does; the verbs that produce nothing answer this,
/// the rest a <see cref="Deferred{TResult}"/>.
/// </summary>
/// <remarks>
/// <see cref="Status"/> polls without blocking, <see cref="Wait"/> blocks until the call is final and
/// raises its failure, and <c>await</c> runs that wait on the thread pool. Disposing one nobody waited
/// for waits for it, then discards what it produced and lets its arena serve the next call.
/// <para>
/// Every wait after the first answers the same result or raises the same failure, and
/// <see cref="Status"/> keeps reading the final status. One left unfinished and unreachable is
/// released by the first collection that finds it final, rather than freeing memory the library
/// still reads and writes.
/// </para>
/// </remarks>
public class Deferred : IDisposable
{
    private readonly ICall _call;
    private readonly CallOutputs[] _outputs;

    private bool _settled;
    private ExceptionDispatchInfo? _failure;

    internal Deferred(ICall call, CallOutputs[] outputs)
    {
        this._call = call;
        this._outputs = outputs;
    }

    /// <summary>Releases a deferred call nobody waited for, once it is final.</summary>
    ~Deferred() => this.Dispose(false);

    /// <summary><see cref="Status.Deferred"/> until the call is final, then its final status, read
    /// without blocking. Calls <c>usearch_outcome_status</c>.</summary>
    public unsafe Status Status => usearch_outcome_status(&this.Pinned->Outcome);

    /// <summary>Blocks until the call is final, then raises its failure. Calls <c>usearch_outcome_wait</c>.</summary>
    public void Wait()
    {
        this.Settle();
        this._failure?.Throw();
    }

    /// <summary>Waits on the thread pool, so <c>await</c> suspends the caller rather than blocking it.
    /// Binding-only.</summary>
    /// <returns>The awaiter of that wait.</returns>
    public TaskAwaiter GetAwaiter() => Task.Run(this.Wait).GetAwaiter();

    /// <summary>Waits for an unfinished call and discards what it produced; a call already waited for
    /// is left as it is. Calls <c>usearch_outcome_wait</c>.</summary>
    public void Dispose()
    {
        this.Dispose(true);
        GC.SuppressFinalize(this);
    }

    /// <summary>Waits for an unfinished call and releases what it holds; a finalizer that finds the
    /// call still running looks again after the next collection instead.</summary>
    /// <param name="disposing">False when called from the finalizer, which must not block.</param>
    protected virtual unsafe void Dispose(bool disposing)
    {
        if (disposing)
        {
            this.Settle();
            return;
        }

        if (this.Status == Status.Deferred)
        {
            GC.ReRegisterForFinalize(this);
            return;
        }

        this._call.Release(this.Pinned);
    }

    /// <summary>Waits until the call is final, then releases what it holds and keeps what it built or
    /// raised, the first time only.</summary>
    /// <returns>True for the one settle that did so.</returns>
    private protected unsafe bool Settle()
    {
        lock (this._outputs)
        {
            if (this._settled)
            {
                return false;
            }

            CallOutputs* outputs = this.Pinned;
            usearch_outcome_wait(&outputs->Outcome);
            this._call.Release(outputs);
            GC.SuppressFinalize(this);
            try
            {
                this._call.Finish(outputs);
            }
            // Every exception is caught on purpose, so each later wait raises the same one.
#pragma warning disable CA1031
            catch (Exception failure)
#pragma warning restore CA1031
            {
                this._failure = ExceptionDispatchInfo.Capture(failure);
            }

            this._settled = true;
            return true;
        }
    }

    private unsafe CallOutputs* Pinned =>
        (CallOutputs*)Unsafe.AsPointer(ref MemoryMarshal.GetArrayDataReference(this._outputs));
}

/// <summary>A submitted call that answers a <typeparamref name="TResult"/> once final.</summary>
/// <typeparam name="TResult">What the call answers once final.</typeparam>
public sealed class Deferred<TResult> : Deferred
{
    private readonly ICall<TResult> _call;

    internal Deferred(ICall<TResult> call, CallOutputs[] outputs) : base(call, outputs) => this._call = call;

    /// <summary>Blocks until the call is final, then answers what it produced or raises its failure.
    /// Calls <c>usearch_outcome_wait</c>.</summary>
    /// <returns>What the call produced.</returns>
    public new TResult Wait()
    {
        base.Wait();
        return this._call.Result;
    }

    /// <summary>Waits on the thread pool, so <c>await</c> suspends the caller rather than blocking it.
    /// Binding-only.</summary>
    /// <returns>The awaiter of that wait.</returns>
    public new TaskAwaiter<TResult> GetAwaiter() => Task.Run(this.Wait).GetAwaiter();
}

/// <summary>Whether a call returns with its final status, or once submitted to its executor.</summary>
internal enum Completion : int
{
    /// <summary>Return with the final status stored and every output readable.</summary>
    Blocking = 0,
    /// <summary>Return once submitted, the status deferred until final.</summary>
    Deferred = 1,
}

/// <summary>
/// Where a call's outcome and the outputs it reports by pointer are written: on the stack of a blocking
/// call, or in the pinned element a <see cref="Deferred"/> keeps until final. <see cref="View"/> holds
/// the one view a verb writes into its arena, and <see cref="Report"/> the one report struct it
/// fills - its telemetry, audit, or description.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct CallOutputs
{
    public NativeOutcome Outcome;
    public nuint Written;
    public nuint Required;
    public fixed ulong View[4];
    public fixed ulong Report[12];
}

/// <summary>
/// One verb's call: what it lends the library, and what it builds once final. A verb is written
/// once as such a struct, and the receiver alone decides whether it blocks or defers.
/// </summary>
internal unsafe interface ICall
{
    /// <summary>Fixes what the call lends, pinning it past the C verb only when deferred, and leases its
    /// handles for the C verb alone, then calls it with its outputs written to <paramref name="outputs"/>.
    /// Throws only before the C verb is called.</summary>
    void Submit(CallOutputs* outputs, Completion completion);

    /// <summary>Settles what the call handed over - donors it consumed - then unpins what it lent, once
    /// the call is final.</summary>
    void Release(CallOutputs* outputs);

    /// <summary>Raises the failure of a final call, or builds what it produced.</summary>
    void Finish(CallOutputs* outputs);
}

/// <summary>A call that produces a <typeparamref name="TResult"/>, readable once finished.</summary>
/// <typeparam name="TResult">What the call answers once final.</typeparam>
internal interface ICall<TResult> : ICall
{
    /// <summary>What <see cref="ICall.Finish"/> built.</summary>
    TResult Result { get; }
}

/// <summary>
/// The handles every call names, each leased only while the C verb runs, since freeing a handle waits
/// for the deferred calls naming it.
/// </summary>
internal readonly struct CallHandles(EngineHandle index, Executor? executor)
{
    /// <summary>Leases the index for the length of the C verb.</summary>
    /// <returns>The lease, whose pointer the call names.</returns>
    internal Lease LendIndex() => Lease.Of(index, "index");

    /// <summary>Leases the executor for the length of the C verb.</summary>
    /// <returns>The lease, null when the call runs on the caller's thread.</returns>
    internal Lease LendExecutor() => Lease.Of(executor?.Handle, "executor");
}

/// <summary>Runs one verb's call to completion, or submits it.</summary>
internal static unsafe class Calls
{
    /// <summary>Runs the call with its outputs on the stack, final when the C verb returns.</summary>
    /// <param name="call">The call to run.</param>
    /// <returns>The finished call, holding its result.</returns>
    internal static TCall Block<TCall>(TCall call) where TCall : struct, ICall
    {
        CallOutputs outputs = default;
        try
        {
            call.Submit(&outputs, Completion.Blocking);
        }
        finally
        {
            call.Release(&outputs);
        }

        call.Finish(&outputs);
        return call;
    }

    /// <summary>Submits a call that produces nothing.</summary>
    /// <param name="call">The call, boxed so its pins travel with the handle.</param>
    /// <returns>The handle that owns the call until final.</returns>
    internal static Deferred Defer(ICall call) => new(call, Submitted(call));

    /// <summary>Submits a call that produces a result.</summary>
    /// <param name="call">The call, boxed so its pins travel with the handle.</param>
    /// <returns>The handle that owns the call until final.</returns>
    internal static Deferred<TResult> Defer<TResult>(ICall<TResult> call) => new(call, Submitted(call));

    /// <summary>Lowers a bound on matches per query, refusing a negative one, which no <c>size_t</c> spells.</summary>
    /// <param name="wanted">The bound the caller asked for.</param>
    /// <returns>The bound the library reads.</returns>
    internal static nuint Bound(int wanted) => wanted >= 0
        ? (nuint)wanted
        : throw new USearchException(Status.InvalidArgument, "the wanted count is negative");

    /// <summary>Submits the boxed call with its outputs in a pinned element, which never moves.</summary>
    private static CallOutputs[] Submitted(ICall call)
    {
        CallOutputs[] outputs = GC.AllocateArray<CallOutputs>(1, pinned: true);
        CallOutputs* pinned = (CallOutputs*)Unsafe.AsPointer(ref MemoryMarshal.GetArrayDataReference(outputs));
        try
        {
            call.Submit(pinned, Completion.Deferred);
        }
        catch
        {
            call.Release(pinned);
            throw;
        }

        return outputs;
    }
}

#endregion Receivers

#region Introspection

/// <summary>
/// Library-wide probes, tied to no index instance.
/// </summary>
public static class USearchCapabilities
{
    /// <summary>Reports the version of the linked library. Calls <c>usearch_version_name</c>.</summary>
    /// <returns>The version string.</returns>
    public static string Version() => Marshal.PtrToStringUTF8(usearch_version_name()) ?? "unknown";

    /// <summary>Lists the hardware capabilities the linked binary was compiled with, comma separated.
    /// Calls <c>usearch_capabilities_compiled</c>.</summary>
    /// <returns>The capability names.</returns>
    public static string CapabilitiesCompiled() =>
        Marshal.PtrToStringUTF8(usearch_capabilities_compiled()) ?? "serial";

    /// <summary>Lists the hardware capabilities the current machine offers at runtime, comma separated.
    /// Calls <c>usearch_capabilities_available</c>.</summary>
    /// <returns>The capability names.</returns>
    public static string CapabilitiesAvailable() =>
        Marshal.PtrToStringUTF8(usearch_capabilities_available()) ?? "serial";
}

#endregion Introspection

#region Paths

/// <summary>The UTF-8 runs every path argument crosses as.</summary>
internal static unsafe class Paths
{
    /// <summary>Encodes one path, refusing an empty one.</summary>
    /// <param name="path">The path to encode.</param>
    /// <returns>Its UTF-8 bytes, never null-terminated.</returns>
    internal static byte[] Encode(string path)
    {
        if (string.IsNullOrEmpty(path))
        {
            throw new USearchException(Status.InvalidArgument, "the path is empty");
        }

        return Encoding.UTF8.GetBytes(path);
    }

    /// <summary>Encodes a list of paths into one native block: the views, then the bytes they name.</summary>
    /// <param name="paths">The paths to encode.</param>
    /// <returns>The views, which the caller frees with <see cref="NativeMemory.Free"/>.</returns>
    internal static NativeBytesView* Encode(ReadOnlySpan<string> paths)
    {
        int total = 0;
        foreach (string path in paths)
        {
            total += Encoding.UTF8.GetByteCount(path);
        }

        nuint header = (nuint)(paths.Length * sizeof(NativeBytesView));
        NativeBytesView* views = (NativeBytesView*)NativeMemory.Alloc(header + (nuint)total);
        byte* text = (byte*)views + header;
        for (int position = 0; position < paths.Length; position++)
        {
            int written = Encoding.UTF8.GetBytes(paths[position], new Span<byte>(text, total));
            views[position] = new NativeBytesView { Data = (IntPtr)text, Length = (nuint)written };
            text += written;
            total -= written;
        }

        return views;
    }
}

#endregion Paths

#region Native Vocabulary

/// <summary>
/// A borrowed run of bytes and its length, never null-terminated by contract - the shape every
/// path argument and every read-only buffer arrives in.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
internal struct NativeBytesView
{
    /// <summary>The first byte of the run.</summary>
    public IntPtr Data;
    /// <summary>How many bytes the run spans.</summary>
    public nuint Length;
}

/// <summary>
/// A strided run of keys - how a batch names the identities it carries, one per row of the call's
/// count, or one per entry a cursor pull wrote into its arena.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
internal struct NativeKeysView
{
    /// <summary>The first key of the run.</summary>
    public IntPtr Data;
    /// <summary>The bytes between consecutive keys; zero repeats the first, and an output's is one key's size.</summary>
    public nuint Stride;
}

/// <summary>
/// A run of rows along a call's batch axis, carrying no row count: borrowed from the caller as an
/// input, whose width is checked against the geometry the index froze, or written packed into an
/// arena as an output.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
internal struct NativeRowsView
{
    /// <summary>The base of the first row.</summary>
    public IntPtr Data;
    /// <summary>The bytes between consecutive row starts; zero repeats the first row, and an output's
    /// is one row's bytes.</summary>
    public nuint Stride;
    /// <summary>The logical width of one row, refused on an input when it disagrees with the geometry.</summary>
    public nuint Columns;
    /// <summary>The scalar type the rows hold, converted on the way in where it differs.</summary>
    public Scalar Scalar;
}

/// <summary>A ranked run a call wrote into its arena - one list of packed columns per query of the
/// call's count.</summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeMatchesView
{
    /// <summary>The first query's keys, columns per query.</summary>
    public Key* Keys;
    /// <summary>The first query's scores, positionally paired with the keys.</summary>
    public double* Scores;
    /// <summary>The matches written per query, one count per query.</summary>
    public nuint* Counts;
    /// <summary>The columns per query the call wrote, at most its wanted - the "k" in kANN.</summary>
    public nuint Columns;
}

/// <summary>A ragged run of matches a call wrote into its arena: keys and scores end to end, and one
/// more offset than queries.</summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeMatchesTapeView
{
    /// <summary>Every query's keys, end to end.</summary>
    public Key* Keys;
    /// <summary>Every query's scores, positionally paired with the keys.</summary>
    public double* Scores;
    /// <summary>The call's count plus one non-decreasing offsets into the keys and scores.</summary>
    public nuint* Offsets;
}

/// <summary>A ragged run of keys a call wrote into its arena: every entry's keys end to end, and one
/// more offset than entries.</summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeKeysTapeView
{
    /// <summary>Every entry's keys, end to end.</summary>
    public Key* Keys;
    /// <summary>The call's written count plus one non-decreasing offsets into the keys.</summary>
    public nuint* Offsets;
}

/// <summary>The host memory domains a RAM-resident index places its shards in, one shard apiece.</summary>
[StructLayout(LayoutKind.Sequential)]
internal struct NativeRamOptions
{
    /// <summary>NUMA node IDs to place shards on; null derives every node the process may use.</summary>
    public IntPtr MemoryDomains;
    /// <summary>How many node IDs the list holds; zero derives.</summary>
    public nuint MemoryDomainsCount;
    /// <summary>Entries one shard holds before another generation of shards opens; zero leaves it unbounded.</summary>
    public nuint ShardCapacity;
    /// <summary>The CPU class to dispatch as; zero derives the CPU's own.</summary>
    public int CpuClass;
    /// <summary>Bytes an NVMe view's key tables may take on each named domain; zero sets no ceiling.</summary>
    public nuint MemoryLimit;
}

/// <summary>The devices a VRAM-resident index places its shards in, sharded and never replicated.</summary>
[StructLayout(LayoutKind.Sequential)]
internal struct NativeVramOptions
{
    /// <summary>CUDA device ordinals to place shards on; null derives one free device.</summary>
    public IntPtr Devices;
    /// <summary>How many ordinals the list holds; zero derives.</summary>
    public nuint DevicesCount;
    /// <summary>Entries one shard holds before another generation of shards opens; zero leaves it unbounded.</summary>
    public nuint ShardCapacity;
    /// <summary>The GPU class to dispatch as; zero derives the devices' own.</summary>
    public int GpuClass;
}

/// <summary>The devices an NVMe-resident index serves its shards from, one shard per named path.</summary>
[StructLayout(LayoutKind.Sequential)]
internal struct NativeNvmeOptions
{
    /// <summary>Devices or directories to serve shards from; null leaves each shard where its
    /// manifest points.</summary>
    public IntPtr Paths;
    /// <summary>How many paths the list holds; zero derives.</summary>
    public nuint PathsCount;
    /// <summary>The CPU class to dispatch as; zero derives the CPU's own.</summary>
    public int CpuClass;
}

/// <summary>How a long call reports its advance, laid out as <c>usearch_progress_t</c>.</summary>
[StructLayout(LayoutKind.Sequential)]
internal struct NativeProgress
{
    /// <summary>The optional handler; null skips every report.</summary>
    public IntPtr Callback;
    /// <summary>Opaque pointer forwarded into every handler call.</summary>
    public IntPtr State;
    /// <summary>Reports a call makes at most, the last at completion; zero derives to ten.</summary>
    public nuint Steps;
}

/// <summary>Which candidate keys a filtered search may report, laid out as <c>usearch_predicate_t</c>.</summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativePredicate
{
    /// <summary>The optional handler; null admits every key.</summary>
    public delegate* unmanaged[Cdecl]<Key*, Admission*, nuint, IntPtr, void> Callback;
    /// <summary>Opaque pointer forwarded into every handler call.</summary>
    public IntPtr State;
    /// <summary>How many keys each handler call carries.</summary>
    public PredicateCost Cost;
}

/// <summary>The arguments of <c>usearch_executor_for_ram</c>.</summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeExecutorForRam
{
    public IntPtr* Executor;
    public NativeOutcome* Outcome;
    public NativeRamOptions Ram;
    public nuint Threads;
}

/// <summary>The arguments of <c>usearch_executor_on_pool</c>.</summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeExecutorOnPool
{
    public IntPtr* Executor;
    public NativeOutcome* Outcome;
    public NativeRamOptions Ram;
    public IntPtr Pool;
}

/// <summary>The arguments of <c>usearch_executor_for_vram</c>.</summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeExecutorForVram
{
    public IntPtr* Executor;
    public NativeOutcome* Outcome;
    public NativeVramOptions Vram;
    public IntPtr Streams;
    public nuint ScratchLimit;
}

/// <summary>The arguments of <c>usearch_executor_for_nvme</c>.</summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeExecutorForNvme
{
    public IntPtr* Executor;
    public NativeOutcome* Outcome;
    public NativeNvmeOptions Nvme;
    public nuint Threads;
    /// <summary>The NVMe class every ring submits through; zero derives the fastest the host admits.</summary>
    public int NvmeClass;
}

/// <summary>The arguments of <c>usearch_arena_init</c>.</summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeArenaInit
{
    public IntPtr* Arena;
    public IntPtr Executor;
    public NativeOutcome* Outcome;
}

/// <summary>Where an arena's outputs live, laid out as <c>usearch_arena_description_t</c>.</summary>
[StructLayout(LayoutKind.Sequential)]
internal struct NativeArenaDescription
{
    public Residency Residency;
    public nuint Device;
}

/// <summary>The arguments of <c>usearch_arena_describe</c>.</summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeArenaDescribe
{
    public IntPtr Arena;
    public NativeOutcome* Outcome;
    public NativeArenaDescription* Description;
}

/// <summary>The arguments of <c>usearch_executor_drain</c>.</summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeExecutorDrain
{
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Mood Mood;
}

#endregion Native Vocabulary

#region Entry Points

/// <summary>
/// The entry points every modality shares: the introspection probes, the name tables, the
/// executor every fanned-out verb of every modality runs on, the arena every output whose length
/// scales with a call lands in, and the wait every deferred call ends in.
/// </summary>
/// <remarks>
/// The declarations are source-generated rather than reflected, so the marshalling stubs are
/// compiled in and the assembly stays trimmable and ahead-of-time compilable.
/// </remarks>
internal static unsafe partial class NativeMethods
{
    private const string LibraryName = "libusearch_shared";

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial IntPtr usearch_version_name();

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial IntPtr usearch_capabilities_compiled();

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial IntPtr usearch_capabilities_available();

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial IntPtr usearch_cpu_class_name(int cpuClass);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial int usearch_cpu_class_from_name(NativeBytesView name);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial IntPtr usearch_gpu_class_name(int gpuClass);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial int usearch_gpu_class_from_name(NativeBytesView name);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial IntPtr usearch_nvme_class_name(int nvmeClass);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial int usearch_nvme_class_from_name(NativeBytesView name);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial IntPtr usearch_status_name(Status status);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial Status usearch_status_from_name(NativeBytesView name, out Status status);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial IntPtr usearch_scalar_name(Scalar scalar);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial Scalar usearch_scalar_from_name(NativeBytesView name);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial IntPtr usearch_mood_name(Mood mood);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial Status usearch_mood_from_name(NativeBytesView name, out Mood mood);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial IntPtr usearch_residency_name(Residency residency);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial Status usearch_residency_from_name(NativeBytesView name, out Residency residency);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial IntPtr usearch_traversal_name(Traversal traversal);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial Status usearch_traversal_from_name(NativeBytesView name, out Traversal traversal);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial Status usearch_outcome_wait(NativeOutcome* outcome);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial Status usearch_outcome_status(NativeOutcome* outcome);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_executor_for_ram(NativeExecutorForRam* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_executor_on_pool(NativeExecutorOnPool* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_executor_for_nvme(NativeExecutorForNvme* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_executor_for_vram(NativeExecutorForVram* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_executor_drain(NativeExecutorDrain* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_executor_free(IntPtr executor);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_arena_init(NativeArenaInit* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_arena_free(IntPtr arena);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_arena_describe(NativeArenaDescribe* arguments);
}

#endregion Entry Points

#region Handles

/// <summary>
/// One engine handle, released exactly once and never while a call is using it.
/// </summary>
/// <remarks>
/// Every verb holds a <see cref="Lease"/> on the handles its argument struct names, which is what
/// keeps a finalizer or a concurrent dispose from freeing one out from under a running call.
/// </remarks>
internal abstract class EngineHandle : SafeHandle
{
    private protected EngineHandle() : base(IntPtr.Zero, ownsHandle: true) { }

    /// <summary>Whether the engine handed back nothing, which every factory checks.</summary>
    public override bool IsInvalid => this.handle == IntPtr.Zero;

    /// <summary>Wraps the handle a creating verb wrote, raising its failure.</summary>
    /// <typeparam name="THandle">The wrapper that knows how to free it.</typeparam>
    /// <param name="created">The handle the library wrote, or null.</param>
    /// <param name="outcome">The outcome the verb filled.</param>
    /// <returns>The wrapper owning the handle.</returns>
    internal static THandle Created<THandle>(IntPtr created, NativeOutcome outcome)
        where THandle : EngineHandle, new()
    {
        outcome.ThrowIfFailed();
        THandle handle = new();
        handle.SetHandle(created);
        return handle;
    }
}

/// <summary>
/// One call's hold on a handle, which keeps it open until released; the default lease holds nothing
/// and lends null, which is how an absent executor reaches the library.
/// </summary>
internal struct Lease : IDisposable
{
    private EngineHandle? _owner;

    /// <summary>Holds the handle open, refusing one already disposed; a null handle lends null.</summary>
    /// <param name="owner">The handle to hold, if any.</param>
    /// <param name="subject">What the handle is, for the failure message.</param>
    /// <returns>The lease.</returns>
    internal static Lease Of(EngineHandle? owner, string subject)
    {
        if (owner is null)
        {
            return default;
        }

        if (owner.IsClosed)
        {
            throw new USearchException(Status.Uninitialized, $"the {subject} is disposed");
        }

        bool added = false;
        owner.DangerousAddRef(ref added);
        return new Lease { _owner = owner };
    }

    /// <summary>The raw handle to place in an argument struct.</summary>
    internal readonly IntPtr Pointer => this._owner?.DangerousGetHandle() ?? IntPtr.Zero;

    /// <summary>Whether the lease still holds a handle.</summary>
    internal readonly bool Holds => this._owner is not null;

    /// <summary>Lets the handle be released again; a second release does nothing.</summary>
    public void Dispose()
    {
        this._owner?.DangerousRelease();
        this._owner = null;
    }
}

/// <summary>A <c>usearch_executor_t</c>, freed when the wrapper is disposed or collected.</summary>
internal sealed class NativeExecutorHandle : EngineHandle
{
    /// <inheritdoc/>
    protected override bool ReleaseHandle()
    {
        NativeMethods.usearch_executor_free(this.handle);
        return true;
    }
}

/// <summary>A <c>usearch_arena_t</c>, freed when the owning <see cref="Arena"/> is disposed or
/// collected.</summary>
internal sealed class NativeArenaHandle : EngineHandle
{
    /// <summary>The executor a GPU arena frees on, held open until the arena is freed.</summary>
    internal Lease Executor;

    /// <inheritdoc/>
    protected override bool ReleaseHandle()
    {
        NativeMethods.usearch_arena_free(this.handle);
        this.Executor.Dispose();
        return true;
    }
}

#endregion Handles
