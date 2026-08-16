using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;

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
    /// <summary>A status a newer library reported and this build cannot name. An unknown status
    /// is never success.</summary>
    Unrecognized = -1,
}

internal partial struct NativeError
{
    /// <summary>Turns a populated error slot into a thrown failure.</summary>
    internal readonly void ThrowIfFailed()
    {
        if (this.Status == Status.Success)
        {
            return;
        }

        Status named = this.Status < Status.Success || this.Status > Status.Contended
            ? Status.Unrecognized
            : this.Status;
        throw new USearchException(
            named, Marshal.PtrToStringAnsi(this.Message) ?? "the library reported no detail");
    }
}

#endregion Errors

#region Keys

/// <summary>
/// A 128-bit entry identity, most significant byte first, rendered as a lowercase dashed UUID.
/// </summary>
/// <remarks>
/// Keys need not be unique: several entries may share one, and <see cref="USearchIndex.Count"/>
/// reports how many. Integer identities park in the low eight bytes, so the key 42 prints as
/// <c>00000000-0000-0000-0000-00000000002a</c>.
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
    public override string ToString()
    {
        const string Hexadecimal = "0123456789abcdef";
        StringBuilder builder = new(36);
        for (int position = 0; position < ByteCount; position++)
        {
            if (position == 4 || position == 6 || position == 8 || position == 10)
            {
                builder.Append('-');
            }

            byte digit = this._bytes[position];
            builder.Append(Hexadecimal[digit >> 4]).Append(Hexadecimal[digit & 0x0F]);
        }

        return builder.ToString();
    }

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

        Span<byte> digits = stackalloc byte[ByteCount * 2];
        int written = 0;
        foreach (char character in text)
        {
            if (character == '-')
            {
                continue;
            }

            if (written == digits.Length || !TryReadHexadecimal(character, out byte digit))
            {
                return false;
            }

            digits[written] = digit;
            written++;
        }

        if (written != digits.Length)
        {
            return false;
        }

        for (int position = 0; position < ByteCount; position++)
        {
            key._bytes[position] = (byte)((digits[position * 2] << 4) | digits[(position * 2) + 1]);
        }

        return true;
    }

    private static bool TryReadHexadecimal(char character, out byte digit)
    {
        if (character >= '0' && character <= '9')
        {
            digit = (byte)(character - '0');
            return true;
        }

        if (character >= 'a' && character <= 'f')
        {
            digit = (byte)(character - 'a' + 10);
            return true;
        }

        if (character >= 'A' && character <= 'F')
        {
            digit = (byte)(character - 'A' + 10);
            return true;
        }

        digit = 0;
        return false;
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
    /// <summary>6-bit floating point with three exponent bits.</summary>
    E3M2 = 7,
    /// <summary>6-bit floating point with two exponent bits.</summary>
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
/// The managed element types a vector row may arrive in, and the row arithmetic they imply.
/// Sub-byte kinds have no managed element type, so a row's byte width is always its element
/// count times the element size.
/// </summary>
internal static class Scalars
{
    /// <summary>Names the scalar kind of a managed element type.</summary>
    /// <typeparam name="TScalar">The element type of a vector row.</typeparam>
    /// <returns>The kind the library knows it by.</returns>
    internal static ScalarKind KindOf<TScalar>() where TScalar : unmanaged
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

    /// <summary>Reports the byte pitch of one packed row.</summary>
    /// <typeparam name="TScalar">The element type of a vector row.</typeparam>
    /// <param name="dimensions">The row's width in dimensions.</param>
    /// <returns>The bytes between consecutive row starts.</returns>
    internal static unsafe nuint RowBytes<TScalar>(ulong dimensions) where TScalar : unmanaged =>
        (nuint)dimensions * (nuint)sizeof(TScalar);
}

#endregion Scalars

#region Residency

/// <summary>Whether the RAM arenas demand a provisioned huge-page pool.</summary>
public enum HugePages : byte
{
    /// <summary>Ordinary pages, whatever the kernel provides.</summary>
    BasePages = 0,
    /// <summary>Fails loudly when the pool is missing, rather than degrading silently.</summary>
    RequireHugePages = 1,
}

/// <summary>Whether a VRAM index may spill past the device's memory.</summary>
public enum Addressing : byte
{
    /// <summary>Everything stays in device memory.</summary>
    DeviceOnly = 0,
    /// <summary>Cold shards spill into host memory through the pager, trading traversal latency
    /// for datasets past the device.</summary>
    UnifiedAddressing = 1,
}

/// <summary>How an NVMe index reaches the block device.</summary>
public enum IOMode : byte
{
    /// <summary>The O_DIRECT-class page-cache bypass - the default and the point.</summary>
    DirectIO = 0,
    /// <summary>Ordinary buffered reads through the page cache.</summary>
    PageCachedIO = 1,
}

/// <summary>
/// Which fields of a <see cref="Residency"/> are read, and what loading a checkpoint means.
/// </summary>
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

/// <summary>
/// Where an index's payload lives and which engine traverses it. Only the fields
/// <see cref="Kind"/> selects are read; the rest are ignored.
/// </summary>
public struct Residency
{
    /// <summary>Selects the fields below, and the meaning of loading a checkpoint.</summary>
    public ResidencyKind Kind;
    /// <summary>The RAM worker count for construction and batched search; zero takes 7/8 of the
    /// logical cores, since full occupancy starves the ordered publication path.</summary>
    public ulong Threads;
    /// <summary>The RAM backing demand.</summary>
    public HugePages HugePages;
    /// <summary>The VRAM accelerator ordinal; negative picks the first free device.</summary>
    public int Device;
    /// <summary>The VRAM spill policy.</summary>
    public Addressing Addressing;
    /// <summary>The NVMe I/O granularity in bytes; zero probes the device. 512 fits one 256-d
    /// <see cref="ScalarKind.I8"/> row.</summary>
    public uint SectorBytes;
    /// <summary>The NVMe submission ring depth; zero derives from the device.</summary>
    public uint QueueDepth;
    /// <summary>How the NVMe handle reaches the block device.</summary>
    public IOMode IOMode;

    /// <summary>A RAM index with a derived thread count and ordinary pages.</summary>
    public static Residency Default => new() { Kind = ResidencyKind.Ram };

    /// <summary>An index hosted by one accelerator.</summary>
    /// <param name="device">The CUDA ordinal; negative picks the first free device.</param>
    /// <param name="addressing">The spill policy.</param>
    /// <returns>The VRAM residency.</returns>
    public static Residency OnDevice(int device, Addressing addressing = Addressing.DeviceOnly) =>
        new() { Kind = ResidencyKind.Vram, Device = device, Addressing = addressing };
}

#endregion Residency

#region Options

/// <summary>
/// A distance kind supported out of the box, with SIMD kernels dispatched by the machine's
/// capabilities.
/// </summary>
/// <remarks>
/// All of them derive from a single inner-product kernel with pre-computed magnitudes; Hamming
/// and Jaccard are bit-level, over the packed sub-byte kinds. The C layer's custom-callback
/// escape hatch stays unexposed here, as in the Rust and Go bindings: a caller-supplied metric
/// forfeits the SIMD kernel chains and the magnitude cache both.
/// </remarks>
public enum DenseMetric : int
{
    /// <summary>A metric the library could not name.</summary>
    Unknown = 0,
    /// <summary>The inner product.</summary>
    Inner = 1,
    /// <summary>The cosine distance.</summary>
    Angular = 2,
    /// <summary>The squared euclidean distance.</summary>
    Euclidean = 3,
    /// <summary>The count of differing bits.</summary>
    Hamming = 4,
    /// <summary>The bit-level Jaccard distance.</summary>
    Jaccard = 5,
}

/// <summary>Whether stored rows round up to the SIMD step quantum.</summary>
public enum DimensionPadding : byte
{
    /// <summary>Rows keep their natural width.</summary>
    NaturalDimensions = 0,
    /// <summary>Cache-line-aligned rows and tail-free unrolled kernels, at slightly larger
    /// arenas.</summary>
    SimdQuantumDimensions = 1,
}

/// <summary>
/// The construction and loading options of a dense index.
/// </summary>
public struct DenseOptions
{
    /// <summary>The width of the indexed vectors; zero defers the geometry to a later
    /// <see cref="USearchIndex.Load"/>, which reads it off the checkpoint.</summary>
    public ulong Dimensions;
    /// <summary>A property of the stored layout, so it applies whichever residency traverses
    /// it.</summary>
    public DimensionPadding DimensionPadding;
    /// <summary>The distance kind used between vectors.</summary>
    public DenseMetric Metric;
    /// <summary>The scalar type vectors are stored and quantized into.</summary>
    public ScalarKind Quantization;
    /// <summary>The connections-per-node bound of the graph; zero derives.</summary>
    public ulong Connectivity;
    /// <summary>Where the payload lives, and what loading a checkpoint means.</summary>
    public Residency Residency;

    /// <summary>An angular <see cref="ScalarKind.BF16"/> index in RAM, with every optional field
    /// derived.</summary>
    public static DenseOptions Default => new()
    {
        Metric = DenseMetric.Angular,
        Quantization = ScalarKind.BF16,
        Residency = Residency.Default,
    };

    /// <summary>Lowers the options, leaving the custom-metric callback null.</summary>
    /// <returns>The C struct the library reads.</returns>
    internal NativeDenseOptions Lower()
    {
        NativeDenseOptions lowered = default;
        lowered.Metric = this.Metric;
        lowered.Quantization = this.Quantization;
        lowered.Dimensions = (nuint)this.Dimensions;
        lowered.PadDimensions = (byte)(this.DimensionPadding == DimensionPadding.SimdQuantumDimensions ? 1 : 0);
        lowered.Connectivity = (nuint)this.Connectivity;
        lowered.Residency = this.Residency.Kind;
        switch (this.Residency.Kind)
        {
            case ResidencyKind.Ram:
                lowered.ResidencyOptions.Ram = new NativeRamOptions
                {
                    Threads = (nuint)this.Residency.Threads,
                    HugePages = (byte)(this.Residency.HugePages == HugePages.RequireHugePages ? 1 : 0),
                };
                break;
            case ResidencyKind.Vram:
                lowered.ResidencyOptions.Vram = new NativeVramOptions
                {
                    Device = this.Residency.Device,
                    UnifiedAddressing = (byte)(this.Residency.Addressing == Addressing.UnifiedAddressing ? 1 : 0),
                };
                break;
            case ResidencyKind.Nvme:
                lowered.ResidencyOptions.Nvme = new NativeNvmeOptions
                {
                    SectorBytes = this.Residency.SectorBytes,
                    QueueDepth = this.Residency.QueueDepth,
                    DirectIO = (byte)(this.Residency.IOMode == IOMode.DirectIO ? 1 : 0),
                };
                break;
            default:
                throw new USearchException(Status.UnknownKind, "the residency names neither RAM, VRAM, nor NVMe");
        }

        return lowered;
    }

    /// <summary>Lifts the options a checkpoint's headers describe.</summary>
    /// <param name="lowered">The C struct the library wrote.</param>
    /// <returns>The managed options.</returns>
    internal static DenseOptions Lift(in NativeDenseOptions lowered)
    {
        DenseOptions options = default;
        options.Dimensions = lowered.Dimensions;
        options.Metric = lowered.Metric;
        options.Quantization = lowered.Quantization;
        options.Connectivity = lowered.Connectivity;
        options.DimensionPadding = lowered.PadDimensions != 0
            ? DimensionPadding.SimdQuantumDimensions
            : DimensionPadding.NaturalDimensions;
        if (options.Metric < DenseMetric.Inner || options.Metric > DenseMetric.Jaccard)
        {
            throw new USearchException(Status.UnknownKind, "the checkpoint names an unknown metric");
        }

        if (options.Quantization == ScalarKind.Unknown || !Enum.IsDefined(typeof(ScalarKind), options.Quantization))
        {
            throw new USearchException(Status.UnknownKind, "the checkpoint names an unknown scalar kind");
        }

        options.Residency = new Residency { Kind = lowered.Residency };
        switch (lowered.Residency)
        {
            case ResidencyKind.Ram:
                options.Residency.Threads = lowered.ResidencyOptions.Ram.Threads;
                options.Residency.HugePages = lowered.ResidencyOptions.Ram.HugePages != 0
                    ? HugePages.RequireHugePages
                    : HugePages.BasePages;
                break;
            case ResidencyKind.Vram:
                options.Residency.Device = lowered.ResidencyOptions.Vram.Device;
                options.Residency.Addressing = lowered.ResidencyOptions.Vram.UnifiedAddressing != 0
                    ? Addressing.UnifiedAddressing
                    : Addressing.DeviceOnly;
                break;
            case ResidencyKind.Nvme:
                options.Residency.SectorBytes = lowered.ResidencyOptions.Nvme.SectorBytes;
                options.Residency.QueueDepth = lowered.ResidencyOptions.Nvme.QueueDepth;
                options.Residency.IOMode = lowered.ResidencyOptions.Nvme.DirectIO != 0
                    ? IOMode.DirectIO
                    : IOMode.PageCachedIO;
                break;
            default:
                throw new USearchException(Status.UnknownKind, "the checkpoint names an unknown residency");
        }

        return options;
    }

    /// <summary>
    /// Lowers the options and lifts them straight back - the marshalling round trip, and the only
    /// thing that exercises the residency union without a library call. A test cannot name the
    /// native structs, so the round trip is reachable only from here.
    /// </summary>
    /// <returns>The options the round trip reconstructs.</returns>
    internal DenseOptions Reload() => Lift(this.Lower());
}

#endregion Options

#region Telemetry

/// <summary>
/// The traversal telemetry one call reports. Every call overwrites it rather than accumulating,
/// so a sweep over many batches sums them itself with <see cref="Accumulate"/>.
/// </summary>
public sealed class TraversalStats
{
    /// <summary>The graph members the call visited.</summary>
    public ulong VisitedMembers { get; internal set; }

    /// <summary>The distances the call computed, diversity re-selections included.</summary>
    public ulong ComputedDistances { get; internal set; }

    /// <summary>The starvation re-seedings - a graph-connectivity health signal rather than a
    /// cost counter.</summary>
    public ulong Restarts { get; internal set; }

    /// <summary>Folds another call's telemetry into this one.</summary>
    /// <param name="other">The telemetry to add.</param>
    public void Accumulate(TraversalStats other)
    {
        this.VisitedMembers += other.VisitedMembers;
        this.ComputedDistances += other.ComputedDistances;
        this.Restarts += other.Restarts;
    }

    /// <summary>Takes over what one call reported, overwriting rather than accumulating.</summary>
    /// <param name="reported">The telemetry the library wrote.</param>
    internal void Adopt(in NativeTraversalStats reported)
    {
        this.VisitedMembers = reported.VisitedMembers;
        this.ComputedDistances = reported.ComputedDistances;
        this.Restarts = reported.Restarts;
    }
}

#endregion Telemetry

#region Results

/// <summary>
/// Where a batched search writes its results - three slabs sized together, holding a
/// queries-by-wanted matrix of keys, the distances positionally paired with them, and one match
/// count per query.
/// </summary>
public sealed class SearchResults
{
    private readonly USearchKey[] _keys;
    private readonly double[] _distances;
    private readonly nuint[] _counts;

    internal SearchResults(USearchKey[] keys, double[] distances, nuint[] counts, int wanted)
    {
        this._keys = keys;
        this._distances = distances;
        this._counts = counts;
        this.Wanted = wanted;
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
        nuint counted = this._counts[query];
        return counted > (nuint)this.Wanted ? this.Wanted : (int)counted;
    }

    /// <summary>Borrows one query's keys, trimmed to the count that query reported.</summary>
    /// <param name="query">The query row.</param>
    /// <returns>The keys the query matched, nearest first.</returns>
    public ReadOnlySpan<USearchKey> KeysOf(int query) =>
        this._keys.AsSpan(query * this.Wanted, this.CountOf(query));

    /// <summary>Borrows one query's distances, trimmed to the count that query reported.</summary>
    /// <param name="query">The query row.</param>
    /// <returns>The distances positionally paired with <see cref="KeysOf"/>.</returns>
    public ReadOnlySpan<double> DistancesOf(int query) =>
        this._distances.AsSpan(query * this.Wanted, this.CountOf(query));
}

#endregion Results

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
