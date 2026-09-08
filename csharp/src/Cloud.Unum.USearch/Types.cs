using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

using static Cloud.Unum.USearch.NativeMethods;

namespace Cloud.Unum.USearch;

#region Errors

/// <summary>
/// What went wrong, in a vocabulary shared by the engine and by this binding.
/// </summary>
public enum Status : int
{
    /// <summary>The call completed and carries no message.</summary>
    Success = 0,
    /// <summary>An allocation or reservation could not be satisfied.</summary>
    OutOfMemory = 1,
    /// <summary>Dimensionality, row counts, or output capacity disagree.</summary>
    ShapeMismatch = 2,
    /// <summary>A stride or packing the engine cannot traverse.</summary>
    UnsupportedLayout = 3,
    /// <summary>A scalar, metric, or residency tag this build does not know.</summary>
    UnknownKind = 4,
    /// <summary>A mutating verb reached a handle that opens its checkpoint in place.</summary>
    ReadOnly = 5,
    /// <summary>A file could not be opened, read, written, or parsed.</summary>
    IOFailure = 6,
    /// <summary>A device could not be selected, or its memory could not be claimed.</summary>
    DeviceFailure = 7,
    /// <summary>A caller-supplied representation could not be decoded.</summary>
    MalformedInput = 8,
    /// <summary>The library reported neither success nor a failure.</summary>
    Internal = 9,
    /// <summary>A hard design ceiling was reached, like the 32-bit identity space.</summary>
    Overflow = 10,
    /// <summary>The verb needs a prior successful reservation that never happened.</summary>
    Uninitialized = 11,
    /// <summary>The request contradicts an already-frozen configuration.</summary>
    ConfigMismatch = 12,
    /// <summary>A synchronization contract was violated by a concurrent operation; discard any
    /// produced output and retry in a quieter phase.</summary>
    Contended = 13,
    /// <summary>A progress handler asked the operation to stop; it unwound and published
    /// nothing, so the handle carries whatever it did before the call.</summary>
    Cancelled = 14,
    /// <summary>A status a newer library reported and this build cannot name. An unknown status
    /// is never success.</summary>
    Unrecognized = -1,
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
    public USearchException(Status status, string message) : base($"usearch: {status}: {message}")
    {
        this.Status = status;
    }

    /// <summary>Initializes a failure carrying a status, its detail, and the exception behind it.</summary>
    /// <param name="status">What went wrong.</param>
    /// <param name="message">The detail the status cannot carry.</param>
    /// <param name="cause">The exception this failure wraps.</param>
    public USearchException(Status status, string message, Exception cause)
        : base($"usearch: {status}: {message}", cause)
    {
        this.Status = status;
    }

    /// <summary>What went wrong.</summary>
    public Status Status { get; }
}

/// <summary>
/// The outcome slot one call writes: a status every caller can branch on, and a static message
/// carrying the detail the status cannot. Messages are static library storage, so nothing here is
/// deallocated.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
internal struct NativeError
{
    /// <summary>What went wrong, or <see cref="Status.Success"/>.</summary>
    public Status Status;
    /// <summary>The static message, null exactly when the status is success.</summary>
    public IntPtr Message;

    /// <summary>Turns a populated error slot into a thrown failure.</summary>
    internal readonly void ThrowIfFailed()
    {
        if (this.Status == Status.Success)
        {
            return;
        }

        Status named = this.Status < Status.Success || this.Status > Status.Cancelled
            ? Status.Unrecognized
            : this.Status;
        throw new USearchException(
            named, Marshal.PtrToStringUTF8(this.Message) ?? "the library reported no detail");
    }
}

#endregion Errors

#region Keys

/// <summary>
/// A 128-bit entry identity, most significant byte first, rendered as a lowercase dashed UUID.
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
public struct USearchKey : IEquatable<USearchKey>
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
    public static USearchKey Nil => default;

    /// <summary>Parks an integer identity in the low eight bytes, zeroing the rest.</summary>
    /// <param name="value">The integer identity.</param>
    /// <returns>The key holding that identity.</returns>
    public static USearchKey FromUInt64(ulong value)
    {
        USearchKey key = default;
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
    public static implicit operator USearchKey(ulong value) => FromUInt64(value);

    /// <summary>Reads a UUID's canonical byte order as an identity.</summary>
    /// <param name="identity">The UUID.</param>
    /// <returns>The key its bytes spell.</returns>
    public static USearchKey FromGuid(Guid identity)
    {
        USearchKey key = default;
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
    public static USearchKey Parse(string text)
    {
        if (!TryParse(text, out USearchKey key))
        {
            throw new USearchException(Status.MalformedInput, "the key is not thirty-two hexadecimal digits");
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
    public static bool TryParse(string? text, out USearchKey key)
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
    public bool Equals(USearchKey other)
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
    public override bool Equals(object? other) => other is USearchKey key && this.Equals(key);

    /// <summary>Hashes the sixteen identity bytes.</summary>
    /// <returns>A hash agreeing with <see cref="Equals(USearchKey)"/>.</returns>
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
    public static bool operator ==(USearchKey left, USearchKey right) => left.Equals(right);

    /// <summary>Compares two keys byte by byte.</summary>
    /// <param name="left">The first key.</param>
    /// <param name="right">The second key.</param>
    /// <returns>True when any byte differs.</returns>
    public static bool operator !=(USearchKey left, USearchKey right) => !left.Equals(right);
}

#endregion Keys

#region Scalars

/// <summary>
/// A stored or exchanged scalar type, named by width and encoding. Sub-byte kinds pack several
/// dimensions per byte, as their <c>xN</c> suffixes spell.
/// </summary>
/// <remarks>
/// The enumerators are ordered by shrinking dynamic range, and their values carry the numbering
/// gaps the C enum reserves for kinds that have not landed yet.
/// </remarks>
public enum ScalarKind : int
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
public static class ScalarKinds
{
    /// <summary>Reports the byte width of one stored value; zero for an unnameable kind.</summary>
    /// <param name="kind">The scalar kind.</param>
    /// <returns>The bytes one value occupies.</returns>
    public static int ValueBytes(this ScalarKind kind) => kind switch
    {
        ScalarKind.F64 => 8,
        ScalarKind.F32 => 4,
        ScalarKind.BF16 => 2,
        ScalarKind.F16 => 2,
        ScalarKind.E5M2 => 1,
        ScalarKind.E4M3 => 1,
        ScalarKind.E3M2 => 1,
        ScalarKind.E2M3 => 1,
        ScalarKind.I8 => 1,
        ScalarKind.I4x2 => 1,
        ScalarKind.U8 => 1,
        ScalarKind.U4x2 => 1,
        ScalarKind.U1x8 => 1,
        _ => 0,
    };

    /// <summary>Reports how many logical dimensions one stored value carries.</summary>
    /// <param name="kind">The scalar kind.</param>
    /// <returns>One for the byte-and-wider kinds, more for the packed ones.</returns>
    public static int DimensionsPerValue(this ScalarKind kind) => kind switch
    {
        ScalarKind.I4x2 => 2,
        ScalarKind.U4x2 => 2,
        ScalarKind.U1x8 => 8,
        _ => 1,
    };

    /// <summary>Reports the bytes one row of the given width occupies.</summary>
    /// <remarks>A width that leaves a value half filled is refused rather than rounded up, since
    /// the trailing dimensions of that value are storage the caller never named.</remarks>
    /// <param name="kind">The scalar kind the row holds.</param>
    /// <param name="dimensions">The row's logical width.</param>
    /// <returns>The bytes between consecutive row starts in a packed matrix.</returns>
    public static ulong RowBytes(this ScalarKind kind, ulong dimensions)
    {
        int valueBytes = kind.ValueBytes();
        if (valueBytes == 0)
        {
            throw new USearchException(Status.UnknownKind, "the row names no scalar kind");
        }

        ulong perValue = (ulong)kind.DimensionsPerValue();
        if (dimensions % perValue != 0)
        {
            throw new USearchException(
                Status.ShapeMismatch, $"{dimensions} dimensions do not fill whole {kind} values");
        }

        ulong values = dimensions / perValue;
        if (values > ulong.MaxValue / (ulong)valueBytes)
        {
            throw new USearchException(Status.Overflow, "the row width overflows a 64-bit byte count");
        }

        return values * (ulong)valueBytes;
    }

    /// <summary>Names the scalar kind of a managed element type.</summary>
    /// <remarks>Only the kinds .NET has an element type for are nameable this way; the packed and
    /// minifloat kinds are named explicitly when a matrix is built.</remarks>
    /// <typeparam name="TScalar">The element type of a vector row.</typeparam>
    /// <returns>The kind the library knows it by.</returns>
    public static ScalarKind KindOf<TScalar>() where TScalar : unmanaged
    {
        if (typeof(TScalar) == typeof(double))
        {
            return ScalarKind.F64;
        }

        if (typeof(TScalar) == typeof(float))
        {
            return ScalarKind.F32;
        }

        if (typeof(TScalar) == typeof(Half))
        {
            return ScalarKind.F16;
        }

        if (typeof(TScalar) == typeof(sbyte))
        {
            return ScalarKind.I8;
        }

        if (typeof(TScalar) == typeof(byte))
        {
            return ScalarKind.U8;
        }

        throw new USearchException(
            Status.UnknownKind, $"no scalar kind names the element type {typeof(TScalar).Name}");
    }
}

#endregion Scalars

#region Matrices

/// <summary>
/// A caller-owned slab viewed as rows of equal width: a base, a row count, a logical width, and a
/// byte pitch between consecutive row starts. It is the shape every batch argument and every batch
/// output takes, so one type serves <see cref="DenseIndex.Add"/> and <see cref="DenseIndex.Get"/>
/// alike.
/// </summary>
/// <remarks>
/// A matrix borrows and never owns, so it lives only as long as the span it was built from - which
/// is what makes it a <c>ref struct</c> rather than something a field could hold. The default value
/// names no storage, and the verbs that accept an absent matrix read it as "skip that output".
/// <para>
/// The scalar kind is carried rather than derived, so a slab of bytes reaches the index as packed
/// <see cref="ScalarKind.U1x8"/> bit sets or as plain <see cref="ScalarKind.U8"/> integers, the
/// caller having said which.
/// </para>
/// </remarks>
public readonly unsafe ref struct Matrix
{
    /// <summary>The slab the rows are read from and written into.</summary>
    internal readonly Span<byte> Bytes;

    /// <summary>How many rows the view spans.</summary>
    public int Rows { get; }

    /// <summary>The logical width of one row.</summary>
    public int Dimensions { get; }

    /// <summary>The byte pitch between consecutive row starts.</summary>
    public int StrideBytes { get; }

    /// <summary>The scalar type the rows hold.</summary>
    public ScalarKind Kind { get; }

    /// <summary>Whether the view names no storage, which every verb reads as an absent output.</summary>
    public bool IsEmpty => this.Bytes.IsEmpty;

    private Matrix(Span<byte> bytes, int rows, int dimensions, int strideBytes, ScalarKind kind)
    {
        this.Bytes = bytes;
        this.Rows = rows;
        this.Dimensions = dimensions;
        this.StrideBytes = strideBytes;
        this.Kind = kind;
    }

    /// <summary>Views a packed slab as rows of the given width, naming the kind its element type implies.</summary>
    /// <typeparam name="TScalar">The element type of the slab.</typeparam>
    /// <param name="data">The slab, which divides evenly into rows.</param>
    /// <param name="dimensions">The logical width of one row.</param>
    /// <returns>The view over that slab.</returns>
    public static Matrix RowsOf<TScalar>(Span<TScalar> data, int dimensions) where TScalar : unmanaged =>
        RowsOf(data, dimensions, ScalarKinds.KindOf<TScalar>());

    /// <summary>Views a packed slab as rows of the given width, under a kind the caller names.</summary>
    /// <typeparam name="TScalar">The element type of the slab.</typeparam>
    /// <param name="data">The slab, which divides evenly into rows.</param>
    /// <param name="dimensions">The logical width of one row.</param>
    /// <param name="kind">The scalar type the values hold.</param>
    /// <returns>The view over that slab.</returns>
    public static Matrix RowsOf<TScalar>(Span<TScalar> data, int dimensions, ScalarKind kind)
        where TScalar : unmanaged
    {
        int rowBytes = CheckedRowBytes<TScalar>(dimensions, kind);
        Span<byte> bytes = MemoryMarshal.AsBytes(data);
        if (bytes.Length % rowBytes != 0)
        {
            throw new USearchException(
                Status.ShapeMismatch, $"{bytes.Length} bytes do not divide into rows of {rowBytes}");
        }

        return new Matrix(bytes, bytes.Length / rowBytes, dimensions, rowBytes, kind);
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
    /// <param name="kind">The scalar type the values hold.</param>
    /// <returns>The view over that slab.</returns>
    public static Matrix StridedRowsOf<TScalar>(
        Span<TScalar> data, int rows, int dimensions, int strideBytes, ScalarKind kind)
        where TScalar : unmanaged
    {
        int rowBytes = CheckedRowBytes<TScalar>(dimensions, kind);
        if (strideBytes < rowBytes)
        {
            throw new USearchException(
                Status.UnsupportedLayout, $"a stride of {strideBytes} bytes cannot hold a row of {rowBytes}");
        }

        if (rows < 0)
        {
            throw new USearchException(Status.ShapeMismatch, "a matrix cannot span a negative row count");
        }

        Span<byte> bytes = MemoryMarshal.AsBytes(data);
        long span = rows == 0 ? 0 : ((long)(rows - 1) * strideBytes) + rowBytes;
        if (span > bytes.Length)
        {
            throw new USearchException(
                Status.ShapeMismatch, $"{rows} rows of stride {strideBytes} overrun a slab of {bytes.Length} bytes");
        }

        return new Matrix(bytes, rows, dimensions, strideBytes, kind);
    }

    /// <summary>Lends the slab as the input descriptor every batch verb reads.</summary>
    /// <param name="data">The pinned base of the slab.</param>
    /// <returns>The C struct the library reads.</returns>
    internal NativeMatrixView AsView(byte* data) => new()
    {
        Data = (IntPtr)data,
        Rows = (nuint)this.Rows,
        Columns = (nuint)this.Dimensions,
        StrideBytes = (nuint)this.StrideBytes,
        ScalarKind = this.Kind,
    };

    /// <summary>Lends the slab as the output descriptor a copying verb writes into.</summary>
    /// <param name="data">The pinned base of the slab.</param>
    /// <returns>The C struct the library fills.</returns>
    internal NativeMatrixSpan AsOutput(byte* data) => new()
    {
        Data = (IntPtr)data,
        Capacity = (nuint)this.Rows,
        Columns = (nuint)this.Dimensions,
        StrideBytes = (nuint)this.StrideBytes,
        ScalarKind = this.Kind,
    };

    /// <summary>Refuses a view whose width disagrees with a frozen geometry.</summary>
    /// <param name="dimensions">The width the index froze.</param>
    internal void CheckWidth(ulong dimensions)
    {
        if (this.IsEmpty)
        {
            throw new USearchException(Status.MalformedInput, "the matrix names no storage");
        }

        if ((ulong)this.Dimensions != dimensions)
        {
            throw new USearchException(
                Status.ShapeMismatch, "the matrix rows are not as wide as the index geometry");
        }
    }

    private static int CheckedRowBytes<TScalar>(int dimensions, ScalarKind kind) where TScalar : unmanaged
    {
        if (dimensions <= 0)
        {
            throw new USearchException(Status.ShapeMismatch, "a matrix row cannot be zero dimensions wide");
        }

        int valueBytes = kind.ValueBytes();
        if (valueBytes != sizeof(TScalar))
        {
            throw new USearchException(
                Status.UnknownKind, $"a {kind} value is {valueBytes} bytes, not {sizeof(TScalar)}");
        }

        ulong rowBytes = kind.RowBytes((ulong)dimensions);
        if (rowBytes > int.MaxValue)
        {
            throw new USearchException(Status.Overflow, "one row exceeds what a span can address");
        }

        return (int)rowBytes;
    }
}

#endregion Matrices

#region Results

/// <summary>
/// Where a ranked run lands - three slabs sized together, holding a queries-by-wanted matrix of
/// keys, the distances positionally paired with them, and one match count per query.
/// </summary>
/// <remarks>
/// The wanted count is the "k" in kANN and the row width of both matrices, so it is fixed when the
/// slabs are and cannot be restated afterwards. Lending slabs the caller already owns is what lets
/// a recall sweep allocate once and reuse them across every point.
/// </remarks>
public sealed class SearchResults
{
    private readonly Memory<USearchKey> _keys;
    private readonly Memory<double> _distances;
    private readonly Memory<nuint> _counts;

    private SearchResults(Memory<USearchKey> keys, Memory<double> distances, Memory<nuint> counts, int wanted)
    {
        this._keys = keys;
        this._distances = distances;
        this._counts = counts;
        this.Wanted = wanted;
    }

    /// <summary>Sizes three fresh slabs for a batch of queries, each reporting up to wanted matches.</summary>
    /// <param name="queries">How many queries the batch resolves.</param>
    /// <param name="wanted">The upper bound on matches per query.</param>
    /// <returns>The outputs those slabs make up.</returns>
    public static SearchResults Packed(int queries, int wanted)
    {
        if (queries < 0)
        {
            throw new USearchException(Status.ShapeMismatch, "a batch cannot hold a negative query count");
        }

        if (wanted <= 0)
        {
            throw new USearchException(Status.ShapeMismatch, "the outputs reserve no room for a single match");
        }

        int matches = Extents.CheckedRowMajor((ulong)queries, (ulong)wanted, USearchKey.ByteCount, "search output");
        return new SearchResults(new USearchKey[matches], new double[matches], new nuint[queries], wanted);
    }

    /// <summary>Adopts slabs the caller already owns - the allocation-free path for a sweep.</summary>
    /// <param name="keys">The key matrix, of at least one row of wanted per query.</param>
    /// <param name="distances">The distance matrix, of the same extent.</param>
    /// <param name="counts">One match count per query, whose length fixes the query capacity.</param>
    /// <param name="wanted">The upper bound on matches per query.</param>
    /// <returns>The outputs those slabs make up.</returns>
    public static SearchResults Of(
        Memory<USearchKey> keys, Memory<double> distances, Memory<nuint> counts, int wanted)
    {
        if (wanted <= 0)
        {
            throw new USearchException(Status.ShapeMismatch, "the outputs reserve no room for a single match");
        }

        long matches = (long)counts.Length * wanted;
        if (keys.Length < matches || distances.Length < matches)
        {
            throw new USearchException(Status.ShapeMismatch, "the matrices are narrower than the wanted count");
        }

        return new SearchResults(keys, distances, counts, wanted);
    }

    /// <summary>The upper bound on matches per query - the "k" in kANN.</summary>
    public int Wanted { get; }

    /// <summary>How many queries the slabs were sized for.</summary>
    public int Queries => this._counts.Length;

    /// <summary>Reports how many matches one query found, clamped to what the slabs hold.</summary>
    /// <param name="query">The query row.</param>
    /// <returns>The match count of that query.</returns>
    public int CountOf(int query)
    {
        if ((uint)query >= (uint)this._counts.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(query));
        }

        // A search never reports more than it was asked for, but the count is the library's
        // word on what it wrote, and entries past it were never written at all.
        nuint counted = this._counts.Span[query];
        return counted > (nuint)this.Wanted ? this.Wanted : (int)counted;
    }

    /// <summary>Borrows one query's keys, trimmed to the count that query reported.</summary>
    /// <param name="query">The query row.</param>
    /// <returns>The keys the query matched, nearest first.</returns>
    public ReadOnlySpan<USearchKey> KeysOf(int query) =>
        this._keys.Span.Slice(query * this.Wanted, this.CountOf(query));

    /// <summary>Borrows one query's distances, trimmed to the count that query reported.</summary>
    /// <param name="query">The query row.</param>
    /// <returns>The distances positionally paired with <see cref="KeysOf"/>.</returns>
    public ReadOnlySpan<double> DistancesOf(int query) =>
        this._distances.Span.Slice(query * this.Wanted, this.CountOf(query));

    /// <summary>The key matrix the library writes into.</summary>
    internal Span<USearchKey> KeysSlab => this._keys.Span;

    /// <summary>The distance matrix the library writes into.</summary>
    internal Span<double> DistancesSlab => this._distances.Span;

    /// <summary>The per-query counts the library writes into.</summary>
    internal Span<nuint> CountsSlab => this._counts.Span;

    /// <summary>Refuses slabs that hold fewer rows than the batch about to fill them.</summary>
    /// <param name="queries">How many queries the batch resolves.</param>
    internal void CheckCapacity(ulong queries)
    {
        if ((ulong)this._counts.Length < queries)
        {
            throw new USearchException(Status.ShapeMismatch, "the outputs hold fewer rows than there are queries");
        }
    }
}

#endregion Results

#region Residency

/// <summary>
/// Where an index's payload lives, which engine traverses it, and what loading a checkpoint means
/// for the handle. It also picks the <see cref="Executor"/> factory, and an executor of one kind
/// may only drive handles of the same one.
/// </summary>
/// <remarks>
/// Placement within a kind - the memory domains, the device ordinals, the device paths - is
/// derived rather than named here, so an index spreads over whatever the machine offers.
/// </remarks>
public enum ResidencyKind : int
{
    /// <summary>A volatile-memory index traversed by CPU threads - the concurrent baseline.</summary>
    Ram = 0,
    /// <summary>An accelerator-memory index traversed by GPU warps on a chosen device.</summary>
    Vram = 1,
    /// <summary>A persisted index traversed through pluggable I/O engines, sized past RAM.
    /// External-memory graphs are never modified in place, so the handle is read-only and every
    /// mutating verb fails on it with <see cref="Status.ReadOnly"/>.</summary>
    Nvme = 2,
}

#endregion Residency

#region Filtering

/// <summary>
/// Whether a filtered search admits one candidate key or passes over it.
/// </summary>
/// <remarks>
/// A rejected key still routes the traversal rather than terminating it, so recall does not
/// collapse at low selectivity.
/// </remarks>
public enum Admission : int
{
    /// <summary>Skip the key; the walk still routes through it.</summary>
    Reject = 0,
    /// <summary>Report the key among the matches.</summary>
    Admit = 1,
}

#endregion Filtering

#region Executors

/// <summary>
/// The workers every fanned-out verb runs on, shared across indexes.
/// </summary>
/// <remarks>
/// An executor owns host threads or device streams; an index owns rows. Keeping them apart is what
/// lets one pool serve several indexes, and what makes the cost of spawning visible at the line
/// that spawns rather than hidden in a constructor.
/// <para>
/// The residency picks the factory, and the factory decides what the thread count means: host
/// workers for RAM, one submission ring apiece for NVMe, and nothing at all for VRAM, where a
/// single host thread drives one stream per device. The kind must match the residency of every
/// handle the executor drives, and a mismatch fails the first call with
/// <see cref="Status.DeviceFailure"/>.
/// </para>
/// <para>
/// An executor admits one driver at a time, reporting <see cref="Status.Contended"/> rather than
/// corrupting a dispatch already in flight. Handles stay freely concurrent - it is the executor
/// that serializes, never the index, so a caller wanting genuine concurrency gives each thread its
/// own.
/// </para>
/// <example>
/// <code>
/// using Executor pool = new(ResidencyKind.Ram, threads: 8);
/// using DenseIndex index = new(DenseOptions.Default with { Dimensions = 3 });
/// index.Add(pool, [42UL], Matrix.RowsOf&lt;float&gt;(vector, 3));
/// </code>
/// </example>
/// </remarks>
public sealed unsafe class Executor : IDisposable
{
    private readonly NativeExecutorHandle _handle;

    /// <summary>
    /// Spawns the workers a residency's verbs run on. Zero threads derives from the hardware - 7/8
    /// of the cores for RAM, since full occupancy starves the ordered publication path. A VRAM
    /// executor opens one stream per device and takes no thread count, so the count is unread
    /// there.
    /// </summary>
    /// <param name="residency">Which factory spawns the workers, and what they drive.</param>
    /// <param name="threads">The worker count; zero derives.</param>
    public Executor(ResidencyKind residency, ulong threads = 0)
    {
        NativeError error;
        switch (residency)
        {
            case ResidencyKind.Ram:
                NativeRamOptions ram = default;
                this._handle = usearch_executor_for_ram(&ram, (nuint)threads, out error);
                break;
            case ResidencyKind.Vram:
                NativeVramOptions vram = default;
                this._handle = usearch_executor_for_vram(&vram, out error);
                break;
            case ResidencyKind.Nvme:
                NativeNvmeOptions nvme = default;
                this._handle = usearch_executor_for_nvme(&nvme, (nuint)threads, out error);
                break;
            default:
                throw new USearchException(Status.UnknownKind, "the residency names neither RAM, VRAM, nor NVMe");
        }

        error.ThrowIfFailed();
        if (this._handle.IsInvalid)
        {
            throw new USearchException(Status.Internal, "the library returned no handle and no error");
        }
    }

    /// <summary>Host workers for RAM-resident indexes, with the thread count derived.</summary>
    public Executor() : this(ResidencyKind.Ram, 0)
    {
    }

    /// <summary>Waits for every deferred call on this executor to finish.</summary>
    public void Drain()
    {
        usearch_executor_drain(this.Handle, out NativeError error);
        error.ThrowIfFailed();
    }

    /// <summary>
    /// Drains every thread or stream, releases the scratch arena, and frees the executor. Freeing
    /// cannot fail, and a freed executor must not be used.
    /// </summary>
    public void Dispose() => this._handle.Dispose();

    /// <summary>The handle a verb hands across, refusing a disposed executor.</summary>
    internal NativeExecutorHandle Handle => !this._handle.IsClosed && !this._handle.IsInvalid
        ? this._handle
        : throw new USearchException(Status.Uninitialized, "the executor is disposed or was never initialized");

    /// <summary>Reports the handle of the executor a verb was given, refusing a missing one.</summary>
    /// <remarks>
    /// A null executor reaches the engine as a residency mismatch rather than as a derived pool, so
    /// it is refused here where the caller can still read which argument was left out.
    /// </remarks>
    internal static NativeExecutorHandle HandleOf(Executor executor) =>
        executor is null
            ? throw new USearchException(Status.MalformedInput, "the executor is null")
            : executor.Handle;
}

#endregion Executors

#region Extents

/// <summary>
/// The arithmetic every matrix allocation is checked with. A wanted count has no ceiling, so the
/// product of rows, columns, and element width is what overflows first.
/// </summary>
internal static class Extents
{
    /// <summary>
    /// Reports how many elements a row-major matrix spans, refusing extents no allocation could
    /// hold. Both the element product and its byte extent are checked, since a wide enough
    /// request overflows the second while the first still fits.
    /// </summary>
    /// <param name="rows">The row count.</param>
    /// <param name="columns">The column count of one row.</param>
    /// <param name="elementBytes">How many bytes one element occupies.</param>
    /// <param name="subject">What is being sized, for the failure message.</param>
    /// <returns>The element count, as a managed array length.</returns>
    internal static int CheckedRowMajor(ulong rows, ulong columns, ulong elementBytes, string subject)
    {
        if (rows != 0 && columns > ulong.MaxValue / rows)
        {
            throw new USearchException(Status.Overflow, $"the {subject} extent overflows a 64-bit count");
        }

        ulong elements = rows * columns;
        if (elementBytes != 0 && elements > ulong.MaxValue / elementBytes)
        {
            throw new USearchException(Status.Overflow, $"the {subject} extent overflows a 64-bit byte count");
        }

        if (elements > int.MaxValue)
        {
            throw new USearchException(Status.Overflow, $"the {subject} extent exceeds what one array can hold");
        }

        return (int)elements;
    }
}

#endregion Extents

#region Introspection

/// <summary>
/// Library-wide probes, tied to no index instance.
/// </summary>
public static class USearchCapabilities
{
    /// <summary>Reports the version of the linked library.</summary>
    /// <returns>The version string.</returns>
    public static string Version() => Marshal.PtrToStringUTF8(usearch_version()) ?? "unknown";

    /// <summary>Lists the hardware capabilities the linked binary was compiled with, comma separated.</summary>
    /// <returns>The capability names.</returns>
    public static string CapabilitiesCompiled() =>
        Marshal.PtrToStringUTF8(usearch_capabilities_compiled()) ?? "serial";

    /// <summary>Lists the hardware capabilities the current machine offers at runtime, comma separated.</summary>
    /// <returns>The capability names.</returns>
    public static string CapabilitiesAvailable() =>
        Marshal.PtrToStringUTF8(usearch_capabilities_available()) ?? "serial";
}

#endregion Introspection

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
/// A caller-owned, writable run of bytes - the mutable twin of <see cref="NativeBytesView"/>,
/// carried by every verb that fills a buffer the caller allocated.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
internal struct NativeBytesSpan
{
    /// <summary>The first byte of the run.</summary>
    public IntPtr Data;
    /// <summary>How many bytes the run spans.</summary>
    public nuint Length;
}

/// <summary>
/// A borrowed run of keys and its length - how a batch names the identities it carries, one key
/// per row of the matrix it accompanies.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
internal struct NativeKeysView
{
    /// <summary>The first key of the run.</summary>
    public IntPtr Data;
    /// <summary>How many keys the run spans.</summary>
    public nuint Count;
}

/// <summary>
/// A caller-owned, writable run of keys - the mutable twin of <see cref="NativeKeysView"/>, whose
/// capacity bounds what a streaming verb writes.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
internal struct NativeKeysSpan
{
    /// <summary>The first key of the run.</summary>
    public IntPtr Data;
    /// <summary>How many keys the run can hold.</summary>
    public nuint Capacity;
}

/// <summary>
/// A borrowed strided matrix - the batch currency every modality's row inputs take. A zero stride
/// means rows sit contiguously at their natural width.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
internal struct NativeMatrixView
{
    /// <summary>The base of the first row.</summary>
    public IntPtr Data;
    /// <summary>How many rows the matrix carries.</summary>
    public nuint Rows;
    /// <summary>The logical scalars per row, as the caller counts them.</summary>
    public nuint Columns;
    /// <summary>The bytes between consecutive row starts; zero means contiguous.</summary>
    public nuint StrideBytes;
    /// <summary>The scalar type the rows hold, converted on the way in where it differs.</summary>
    public ScalarKind ScalarKind;
}

/// <summary>
/// The writable twin of <see cref="NativeMatrixView"/>, carried by every verb that lands rows in a
/// matrix the caller allocated.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
internal struct NativeMatrixSpan
{
    /// <summary>The base of the first row.</summary>
    public IntPtr Data;
    /// <summary>How many rows the matrix can hold.</summary>
    public nuint Capacity;
    /// <summary>The logical scalars per row, as the caller counts them.</summary>
    public nuint Columns;
    /// <summary>The bytes between consecutive row starts; zero means contiguous.</summary>
    public nuint StrideBytes;
    /// <summary>The scalar type to write, converted on the way out where it differs.</summary>
    public ScalarKind ScalarKind;
}

/// <summary>
/// Where a ranked run lands - three caller-owned outputs sharing the queries' count, which the
/// queries the call took already fixed.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
internal struct NativeMatchesSpan
{
    /// <summary>The first row of the key output.</summary>
    public IntPtr Keys;
    /// <summary>The bytes between consecutive key rows; zero packs them.</summary>
    public nuint KeysStrideBytes;
    /// <summary>The first row of the score output, positionally paired with the keys.</summary>
    public IntPtr Scores;
    /// <summary>The bytes between consecutive score rows; zero packs them.</summary>
    public nuint ScoresStrideBytes;
    /// <summary>The matches written per query - one entry per query, never strided.</summary>
    public IntPtr Counts;
    /// <summary>The columns per row, and the upper bound on matches - the "k" in kANN.</summary>
    public nuint Wanted;
}

/// <summary>The host memory domains a RAM-resident index places its shards in, one shard apiece.</summary>
[StructLayout(LayoutKind.Sequential)]
internal struct NativeRamOptions
{
    /// <summary>NUMA node ids to place shards on; null derives every node the process may use.</summary>
    public IntPtr MemoryDomains;
    /// <summary>How many node ids the list holds; zero derives.</summary>
    public nuint MemoryDomainsCount;
}

/// <summary>The devices a VRAM-resident index places its shards in, sharded and never replicated.</summary>
[StructLayout(LayoutKind.Sequential)]
internal struct NativeVramOptions
{
    /// <summary>CUDA device ordinals to place shards on; null derives one free device.</summary>
    public IntPtr Devices;
    /// <summary>How many ordinals the list holds; zero derives.</summary>
    public nuint DevicesCount;
}

/// <summary>The devices an NVMe-resident index places its shards on, one shard per named path.</summary>
[StructLayout(LayoutKind.Sequential)]
internal struct NativeNvmeOptions
{
    /// <summary>Devices or directories to place shards on; null leaves each shard where its
    /// manifest points.</summary>
    public IntPtr Paths;
    /// <summary>How many paths the list holds; zero derives.</summary>
    public nuint PathsCount;
}

/// <summary>
/// The tagged union of residency arms. Every arm sits at offset zero, so the widest one decides
/// the size, exactly as the C union does.
/// </summary>
[StructLayout(LayoutKind.Explicit)]
internal struct NativeResidencyOptions
{
    /// <summary>The arm <see cref="ResidencyKind.Ram"/> selects.</summary>
    [FieldOffset(0)] public NativeRamOptions Ram;
    /// <summary>The arm <see cref="ResidencyKind.Vram"/> selects.</summary>
    [FieldOffset(0)] public NativeVramOptions Vram;
    /// <summary>The arm <see cref="ResidencyKind.Nvme"/> selects.</summary>
    [FieldOffset(0)] public NativeNvmeOptions Nvme;
}

#endregion Native Vocabulary

#region Entry Points

/// <summary>
/// The shared C surface of <c>include/usearch/types.h</c>: the introspection probes, and the
/// executor every fanned-out verb of every modality runs on.
/// </summary>
/// <remarks>
/// The declarations are source-generated rather than reflected, so the marshalling stubs are
/// compiled in and the assembly stays trimmable and ahead-of-time compilable.
/// </remarks>
internal static unsafe partial class NativeMethods
{
    private const string LibraryName = "libusearch_c";

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial IntPtr usearch_version();

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial IntPtr usearch_capabilities_compiled();

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial IntPtr usearch_capabilities_available();

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial NativeExecutorHandle usearch_executor_for_ram(
        NativeRamOptions* ram, nuint threads, out NativeError error);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial NativeExecutorHandle usearch_executor_for_nvme(
        NativeNvmeOptions* nvme, nuint threads, out NativeError error);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial NativeExecutorHandle usearch_executor_for_vram(
        NativeVramOptions* vram, out NativeError error);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_executor_drain(NativeExecutorHandle executor, out NativeError error);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_executor_free(IntPtr executor);
}

#endregion Entry Points

#region Handles

/// <summary>
/// One engine handle, released exactly once and never while a call is using it.
/// </summary>
/// <remarks>
/// The marshaller reference-counts a <see cref="SafeHandle"/> across every call it is passed to,
/// which is what keeps a finalizer from freeing a handle out from under a running verb. A raw
/// <see cref="IntPtr"/> argument gives no such guarantee: the managed wrapper is unreachable the
/// instant its handle is read, and collection may follow before the call returns.
/// </remarks>
internal abstract class EngineHandle : SafeHandle
{
    private protected EngineHandle() : base(IntPtr.Zero, ownsHandle: true) { }

    /// <summary>Whether the engine handed back nothing, which every factory checks.</summary>
    public override bool IsInvalid => this.handle == IntPtr.Zero;
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

#endregion Handles
