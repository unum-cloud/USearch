/**
 *  @file csharp/src/Cloud.Unum.USearch/SequenceIndex.cs
 *  @author Ash Vardanian
 *  @date November 16, 2026
 *  @brief The sequence index handle, its scoring kinds, and the verbs that run over it.
 */
using System;
using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;

using static Cloud.Unum.USearch.NativeMethods;

namespace Cloud.Unum.USearch;

#region Options

/// <summary>
/// A sequence scoring kind supported out of the box. Mirrors <c>usearch_sequence_metric_t</c>.
/// </summary>
public enum SequenceMetric : int
{
    /// <summary>A kind the library could not name.</summary>
    Unknown = 0,
    /// <summary>Levenshtein edit distance over raw bytes.</summary>
    LevenshteinBytes = 1,
    /// <summary>Levenshtein edit distance over UTF-8 code points.</summary>
    LevenshteinUtf8 = 2,
    /// <summary>Global alignment with a linear gap cost, scored by a substitution matrix.</summary>
    NeedlemanWunsch = 3,
    /// <summary>The share of fixed-width windows one side misses of the other.</summary>
    Overlap = 4,
}

/// <summary>The sequence spellings of <see cref="Names"/>.</summary>
public static partial class Names
{
    /// <summary>Spells a sequence scoring kind, as <see cref="SequenceMetric.LevenshteinBytes"/> spells
    /// "levenshtein". Calls <c>usearch_sequence_metric_name</c>.</summary>
    /// <param name="metric">The kind to spell.</param>
    /// <returns>The canonical spelling.</returns>
    public static string Name(this SequenceMetric metric) => Spelled(usearch_sequence_metric_name(metric));

    /// <summary>Reads a sequence scoring kind back from its spelling. Calls
    /// <c>usearch_sequence_metric_from_name</c>.</summary>
    /// <param name="name">The canonical spelling.</param>
    /// <param name="metric">The kind it spells.</param>
    /// <returns>True when the spelling names a kind, false otherwise.</returns>
    public static bool TryParse(string name, out SequenceMetric metric)
    {
        SequenceMetric parsed = default;
        bool read = Read(name, view => usearch_sequence_metric_from_name(view, out parsed)) == Status.Success;
        metric = parsed;
        return read;
    }
}

/// <summary>
/// The substitution matrix and gap cost <see cref="SequenceMetric.NeedlemanWunsch"/> scores with; the
/// Levenshtein kinds score with unit costs and take none. Mirrors <c>usearch_sequence_weights_t</c>.
/// </summary>
/// <param name="SubstitutionMatrix">Row-major 256×256 signed scores over byte alphabets.</param>
/// <param name="Gap">The cost of one unaligned symbol.</param>
public readonly record struct SequenceWeights(ReadOnlyMemory<sbyte> SubstitutionMatrix, sbyte Gap)
{
    /// <summary>The scores a whole matrix holds, which the library reads in full.</summary>
    internal const int MatrixScores = 256 * 256;

    /// <summary>The matrix of optional weights, checked once against the size the library reads.</summary>
    /// <param name="weights">The weights, if any.</param>
    /// <returns>The scores, empty without weights.</returns>
    internal static ReadOnlySpan<sbyte> MatrixOf(SequenceWeights? weights)
    {
        if (weights is null)
        {
            return default;
        }

        ReadOnlySpan<sbyte> matrix = weights.Value.SubstitutionMatrix.Span;
        // The library reads the whole matrix, so only this side sees a shorter one.
        if (matrix.Length != MatrixScores)
        {
            throw new USearchException(Status.ShapeMismatch, "a substitution matrix holds 256x256 scores");
        }

        return matrix;
    }

    /// <summary>Lowers optional weights over a matrix the caller fixed.</summary>
    /// <param name="weights">The weights, if any.</param>
    /// <param name="matrix">Their matrix, fixed for the call.</param>
    /// <param name="lowered">Where the C struct is written.</param>
    /// <returns>The struct, or null without weights.</returns>
    internal static unsafe NativeSequenceWeights* Lower(
        SequenceWeights? weights, sbyte* matrix, NativeSequenceWeights* lowered)
    {
        if (weights is null)
        {
            return null;
        }

        *lowered = new NativeSequenceWeights { SubstitutionMatrix = matrix, Gap = weights.Value.Gap };
        return lowered;
    }
}

/// <summary>
/// The construction options of a sequence index - the fields <c>usearch_sequence_init_ram_t</c> and
/// <c>usearch_sequence_init_vram_t</c> share.
/// </summary>
public readonly struct SequenceOptions
{
    /// <summary>The scoring kind.</summary>
    public SequenceMetric Metric { get; init; }

    /// <summary>The weights <see cref="SequenceMetric.NeedlemanWunsch"/> needs; null for every other
    /// kind, which refuses any.</summary>
    public SequenceWeights? Weights { get; init; }

    /// <summary>Window widths in bytes for <see cref="SequenceMetric.Overlap"/>, which needs at least one;
    /// every other kind refuses any.</summary>
    public ReadOnlyMemory<nuint> WindowWidths { get; init; }

    /// <summary>The connections-per-node bound of the graph; zero derives.</summary>
    public ulong Connectivity { get; init; }

    /// <summary>The upper bound on entries sharing one key; zero is unbounded.</summary>
    public ulong ValuesPerKey { get; init; }

    /// <summary>A byte Levenshtein index, with every optional field derived.</summary>
    public static SequenceOptions Default => new() { Metric = SequenceMetric.LevenshteinBytes };
}

/// <summary>The scoring kind and connectivity a checkpoint froze. Mirrors
/// <c>usearch_sequence_metadata_t</c>.</summary>
public readonly struct SequenceMetadata
{
    /// <summary>The scoring kind the checkpoint was built under.</summary>
    public SequenceMetric Metric { get; init; }

    /// <summary>The connections per node the graph was built with.</summary>
    public ulong Connectivity { get; init; }

    internal static SequenceMetadata Lift(in NativeSequenceMetadata lowered) => new()
    {
        Metric = lowered.Metric, Connectivity = lowered.Connectivity,
    };
}

/// <summary>A live sequence index's frozen metadata and current counters, read in one call. Mirrors
/// <c>usearch_sequence_index_description_t</c>.</summary>
public readonly struct SequenceIndexDescription
{
    /// <summary>The frozen scoring kind and connectivity.</summary>
    public SequenceMetadata Metadata { get; init; }

    /// <summary>The stored entries.</summary>
    public ulong Size { get; init; }

    /// <summary>The entries hosted without growing again.</summary>
    public ulong Capacity { get; init; }

    /// <summary>The bytes the handle holds, payload storage included.</summary>
    public ulong MemoryUsage { get; init; }

    /// <summary>The CPU class dispatch picked on this machine; "unknown" on VRAM.</summary>
    public string CpuClass { get; init; }

    /// <summary>The GPU class dispatch picked on the index's devices; "unknown" for a host index.</summary>
    public string GpuClass { get; init; }

    /// <summary>The live shards.</summary>
    public ulong ShardsCount { get; init; }
}

/// <summary>What one key holds, enough to size a buffer before retrieving it. Mirrors
/// <c>usearch_sequence_key_stats_t</c>.</summary>
/// <param name="Entries">The entries stored under the key.</param>
/// <param name="TotalBytes">Their bytes in all.</param>
/// <param name="LargestBytes">The bytes of the longest.</param>
public readonly record struct SequenceKeyStats(ulong Entries, ulong TotalBytes, ulong LargestBytes);

#endregion Options

#region Index

/// <summary>
/// A Similarity Search Index over variable-length sequences - text, DNA, RNA and proteins - under
/// edit distances, alignments and window overlap.
/// </summary>
/// <remarks>
/// Every batch is a <see cref="TapeView{TElement}"/> of bytes, so a single sequence is a tape of one.
/// The receivers are the dense index's: the index runs a verb on the caller's thread, <see cref="On"/>
/// blocks on an executor, and <see cref="Deferred"/> submits there and answers a
/// <see cref="Deferred{TResult}"/>. Every callback runs on the engine's threads and must not throw.
/// </remarks>
public sealed unsafe class SequenceIndex : IDisposable
{
    private readonly NativeSequenceIndexHandle _handle;

    private SequenceIndex(NativeSequenceIndexHandle handle) => this._handle = handle;

    /// <summary>The engine handle the verbs reference.</summary>
    internal NativeSequenceIndexHandle Handle => this._handle;

    /// <summary>Creates an empty, mutable sequence index in host memory. Calls
    /// <c>usearch_sequence_init_ram</c>.</summary>
    /// <param name="options">The scoring kind, its weights or windows, and the graph's bounds.</param>
    /// <param name="memoryDomains">NUMA node IDs to place one shard on apiece; empty derives.</param>
    /// <param name="cpuClassCeiling">The newest class dispatch may pick, intersected with what the
    /// machine runs; null means no ceiling.</param>
    /// <returns>The new index.</returns>
    public static SequenceIndex InitRam(
        SequenceOptions options, ReadOnlySpan<int> memoryDomains = default, CPUClass? cpuClassCeiling = null)
    {
        IntPtr created = IntPtr.Zero;
        NativeOutcome outcome = default;
        NativeSequenceWeights weights = default;
        fixed (sbyte* matrix = SequenceWeights.MatrixOf(options.Weights))
        fixed (nuint* widths = options.WindowWidths.Span)
        fixed (int* domains = memoryDomains)
        {
            NativeSequenceInitRam arguments = new()
            {
                Index = &created, Outcome = &outcome, Metric = options.Metric,
                Weights = SequenceWeights.Lower(options.Weights, matrix, &weights),
                WindowWidths = widths, WindowWidthsCount = (nuint)options.WindowWidths.Length,
                Connectivity = (nuint)options.Connectivity, ValuesPerKey = (nuint)options.ValuesPerKey,
                Ram = new NativeRamOptions
                {
                    MemoryDomains = (IntPtr)domains, MemoryDomainsCount = (nuint)memoryDomains.Length,
                    CpuClassCeiling = (int)(cpuClassCeiling ?? 0),
                },
            };
            usearch_sequence_init_ram(&arguments);
        }

        return new SequenceIndex(EngineHandle.Created<NativeSequenceIndexHandle>(created, outcome));
    }

    /// <summary>Creates an empty, mutable sequence index in device memory. Calls
    /// <c>usearch_sequence_init_vram</c>.</summary>
    /// <param name="options">The scoring kind, its weights or windows, and the graph's bounds.</param>
    /// <param name="devices">CUDA device ordinals to place one shard on apiece; empty derives one.</param>
    /// <param name="gpuClassCeiling">The newest class dispatch may pick; null means no ceiling.</param>
    /// <returns>The new index.</returns>
    public static SequenceIndex InitVram(
        SequenceOptions options, ReadOnlySpan<int> devices = default, GPUClass? gpuClassCeiling = null)
    {
        IntPtr created = IntPtr.Zero;
        NativeOutcome outcome = default;
        NativeSequenceWeights weights = default;
        fixed (sbyte* matrix = SequenceWeights.MatrixOf(options.Weights))
        fixed (nuint* widths = options.WindowWidths.Span)
        fixed (int* ordinals = devices)
        {
            NativeSequenceInitVram arguments = new()
            {
                Index = &created, Outcome = &outcome, Metric = options.Metric,
                Weights = SequenceWeights.Lower(options.Weights, matrix, &weights),
                WindowWidths = widths, WindowWidthsCount = (nuint)options.WindowWidths.Length,
                Connectivity = (nuint)options.Connectivity, ValuesPerKey = (nuint)options.ValuesPerKey,
                Vram = new NativeVramOptions
                {
                    Devices = (IntPtr)ordinals, DevicesCount = (nuint)devices.Length,
                    GpuClassCeiling = (int)(gpuClassCeiling ?? 0),
                },
            };
            usearch_sequence_init_vram(&arguments);
        }

        return new SequenceIndex(EngineHandle.Created<NativeSequenceIndexHandle>(created, outcome));
    }

    /// <summary>Copies a checkpoint into a new, mutable index in host memory. Calls
    /// <c>usearch_sequence_load_ram</c>.</summary>
    /// <param name="path">The checkpoint to read, a monolith or a manifest.</param>
    /// <param name="weights">The weights an alignment checkpoint was built with; null otherwise.</param>
    /// <param name="memoryDomains">Where the parts land; empty derives every node.</param>
    /// <param name="valuesPerKey">The upper bound on entries sharing one key; zero is unbounded.</param>
    /// <param name="executor">The workers the read fans out over; null reads on the caller's thread.</param>
    /// <param name="cpuClassCeiling">The newest class dispatch may pick; null means no ceiling.</param>
    /// <returns>The new index.</returns>
    public static SequenceIndex LoadRam(
        string path, SequenceWeights? weights = null, ReadOnlySpan<int> memoryDomains = default,
        ulong valuesPerKey = 0, Executor? executor = null, CPUClass? cpuClassCeiling = null)
    {
        byte[] encoded = Encoding.UTF8.GetBytes(path);
        using HandleReference pool = HandleReference.Of(executor?.Handle, "executor");
        IntPtr created = IntPtr.Zero;
        NativeOutcome outcome = default;
        NativeSequenceWeights lowered = default;
        fixed (byte* data = encoded)
        fixed (sbyte* matrix = SequenceWeights.MatrixOf(weights))
        fixed (int* domains = memoryDomains)
        {
            NativeSequenceLoadRam arguments = new()
            {
                Index = &created, Executor = pool.Pointer, Outcome = &outcome, Path = View(data, encoded.Length),
                Ram = new NativeRamOptions
                {
                    MemoryDomains = (IntPtr)domains, MemoryDomainsCount = (nuint)memoryDomains.Length,
                    CpuClassCeiling = (int)(cpuClassCeiling ?? 0),
                },
                ValuesPerKey = (nuint)valuesPerKey, Weights = SequenceWeights.Lower(weights, matrix, &lowered),
            };
            usearch_sequence_load_ram(&arguments);
        }

        return new SequenceIndex(EngineHandle.Created<NativeSequenceIndexHandle>(created, outcome));
    }

    /// <summary>Copies a checkpoint into a new, mutable index in device memory. Calls
    /// <c>usearch_sequence_load_vram</c>.</summary>
    /// <param name="path">The checkpoint to read.</param>
    /// <param name="weights">The weights an alignment checkpoint was built with; null otherwise.</param>
    /// <param name="devices">Where the parts land; empty derives one free device.</param>
    /// <param name="valuesPerKey">The upper bound on entries sharing one key; zero is unbounded.</param>
    /// <param name="executor">The workers the upload runs on; null uploads on the caller's thread.</param>
    /// <param name="gpuClassCeiling">The newest class dispatch may pick; null means no ceiling.</param>
    /// <returns>The new index.</returns>
    public static SequenceIndex LoadVram(
        string path, SequenceWeights? weights = null, ReadOnlySpan<int> devices = default, ulong valuesPerKey = 0,
        Executor? executor = null, GPUClass? gpuClassCeiling = null)
    {
        byte[] encoded = Encoding.UTF8.GetBytes(path);
        using HandleReference pool = HandleReference.Of(executor?.Handle, "executor");
        IntPtr created = IntPtr.Zero;
        NativeOutcome outcome = default;
        NativeSequenceWeights lowered = default;
        fixed (byte* data = encoded)
        fixed (sbyte* matrix = SequenceWeights.MatrixOf(weights))
        fixed (int* ordinals = devices)
        {
            NativeSequenceLoadVram arguments = new()
            {
                Index = &created, Executor = pool.Pointer, Outcome = &outcome, Path = View(data, encoded.Length),
                Vram = new NativeVramOptions
                {
                    Devices = (IntPtr)ordinals, DevicesCount = (nuint)devices.Length,
                    GpuClassCeiling = (int)(gpuClassCeiling ?? 0),
                },
                ValuesPerKey = (nuint)valuesPerKey, Weights = SequenceWeights.Lower(weights, matrix, &lowered),
            };
            usearch_sequence_load_vram(&arguments);
        }

        return new SequenceIndex(EngineHandle.Created<NativeSequenceIndexHandle>(created, outcome));
    }

    /// <summary>Copies a checkpoint from a buffer into a new, mutable index in host memory. Calls
    /// <c>usearch_sequence_load_ram_buffer</c>.</summary>
    /// <param name="buffer">The checkpoint's bytes, copied in before the call returns.</param>
    /// <param name="weights">The weights an alignment checkpoint was built with; null otherwise.</param>
    /// <param name="memoryDomains">Where the parts land; empty derives every node.</param>
    /// <param name="valuesPerKey">The upper bound on entries sharing one key; zero is unbounded.</param>
    /// <param name="executor">The workers the copy fans out over; null copies on the caller's thread.</param>
    /// <param name="cpuClassCeiling">The newest class dispatch may pick; null means no ceiling.</param>
    /// <returns>The new index.</returns>
    public static SequenceIndex LoadRamBuffer(
        ReadOnlySpan<byte> buffer, SequenceWeights? weights = null, ReadOnlySpan<int> memoryDomains = default,
        ulong valuesPerKey = 0, Executor? executor = null, CPUClass? cpuClassCeiling = null)
    {
        using HandleReference pool = HandleReference.Of(executor?.Handle, "executor");
        IntPtr created = IntPtr.Zero;
        NativeOutcome outcome = default;
        NativeSequenceWeights lowered = default;
        fixed (byte* source = buffer)
        fixed (sbyte* matrix = SequenceWeights.MatrixOf(weights))
        fixed (int* domains = memoryDomains)
        {
            NativeSequenceLoadRamBuffer arguments = new()
            {
                Index = &created, Executor = pool.Pointer, Outcome = &outcome, Buffer = View(source, buffer.Length),
                Ram = new NativeRamOptions
                {
                    MemoryDomains = (IntPtr)domains, MemoryDomainsCount = (nuint)memoryDomains.Length,
                    CpuClassCeiling = (int)(cpuClassCeiling ?? 0),
                },
                ValuesPerKey = (nuint)valuesPerKey, Weights = SequenceWeights.Lower(weights, matrix, &lowered),
            };
            usearch_sequence_load_ram_buffer(&arguments);
        }

        return new SequenceIndex(EngineHandle.Created<NativeSequenceIndexHandle>(created, outcome));
    }

    /// <summary>Copies a checkpoint from a buffer into a new, mutable index in device memory. Calls
    /// <c>usearch_sequence_load_vram_buffer</c>.</summary>
    /// <param name="buffer">The checkpoint's bytes.</param>
    /// <param name="weights">The weights an alignment checkpoint was built with; null otherwise.</param>
    /// <param name="devices">Where the parts land; empty derives one free device.</param>
    /// <param name="valuesPerKey">The upper bound on entries sharing one key; zero is unbounded.</param>
    /// <param name="executor">The workers the upload runs on; null uploads on the caller's thread.</param>
    /// <param name="gpuClassCeiling">The newest class dispatch may pick; null means no ceiling.</param>
    /// <returns>The new index.</returns>
    public static SequenceIndex LoadVramBuffer(
        ReadOnlySpan<byte> buffer, SequenceWeights? weights = null, ReadOnlySpan<int> devices = default,
        ulong valuesPerKey = 0, Executor? executor = null, GPUClass? gpuClassCeiling = null)
    {
        using HandleReference pool = HandleReference.Of(executor?.Handle, "executor");
        IntPtr created = IntPtr.Zero;
        NativeOutcome outcome = default;
        NativeSequenceWeights lowered = default;
        fixed (byte* source = buffer)
        fixed (sbyte* matrix = SequenceWeights.MatrixOf(weights))
        fixed (int* ordinals = devices)
        {
            NativeSequenceLoadVramBuffer arguments = new()
            {
                Index = &created, Executor = pool.Pointer, Outcome = &outcome, Buffer = View(source, buffer.Length),
                Vram = new NativeVramOptions
                {
                    Devices = (IntPtr)ordinals, DevicesCount = (nuint)devices.Length,
                    GpuClassCeiling = (int)(gpuClassCeiling ?? 0),
                },
                ValuesPerKey = (nuint)valuesPerKey, Weights = SequenceWeights.Lower(weights, matrix, &lowered),
            };
            usearch_sequence_load_vram_buffer(&arguments);
        }

        return new SequenceIndex(EngineHandle.Created<NativeSequenceIndexHandle>(created, outcome));
    }

    /// <summary>Serves a checkpoint in place through a memory mapping, as a read-only index. Calls
    /// <c>usearch_sequence_view_ram</c>.</summary>
    /// <param name="path">The checkpoint to serve.</param>
    /// <param name="weights">The weights an alignment checkpoint was built with; null otherwise.</param>
    /// <param name="memoryDomains">Where the parts are served from; empty derives.</param>
    /// <param name="cpuClassCeiling">The newest class dispatch may pick; null means no ceiling.</param>
    /// <returns>The read-only index.</returns>
    public static SequenceIndex ViewRam(
        string path, SequenceWeights? weights = null, ReadOnlySpan<int> memoryDomains = default,
        CPUClass? cpuClassCeiling = null)
    {
        byte[] encoded = Encoding.UTF8.GetBytes(path);
        IntPtr created = IntPtr.Zero;
        NativeOutcome outcome = default;
        NativeSequenceWeights lowered = default;
        fixed (byte* data = encoded)
        fixed (sbyte* matrix = SequenceWeights.MatrixOf(weights))
        fixed (int* domains = memoryDomains)
        {
            NativeSequenceViewRam arguments = new()
            {
                Index = &created, Outcome = &outcome, Path = View(data, encoded.Length),
                Weights = SequenceWeights.Lower(weights, matrix, &lowered),
                Ram = new NativeRamOptions
                {
                    MemoryDomains = (IntPtr)domains, MemoryDomainsCount = (nuint)memoryDomains.Length,
                    CpuClassCeiling = (int)(cpuClassCeiling ?? 0),
                },
            };
            usearch_sequence_view_ram(&arguments);
        }

        return new SequenceIndex(EngineHandle.Created<NativeSequenceIndexHandle>(created, outcome));
    }

    /// <summary>Serves a checkpoint in place from NVMe drives as a read-only index. Calls
    /// <c>usearch_sequence_view_nvme</c>.</summary>
    /// <param name="path">The checkpoint to serve.</param>
    /// <param name="weights">The weights an alignment checkpoint was built with; null otherwise.</param>
    /// <param name="paths">The devices the parts are served from; empty serves each where its manifest
    /// points.</param>
    /// <param name="executor">The NVMe executor whose rings read the headers; null reads on the caller's
    /// thread.</param>
    /// <param name="cpuClassCeiling">The newest class dispatch may pick; null means no ceiling.</param>
    /// <returns>The read-only index.</returns>
    public static SequenceIndex ViewNvme(
        string path, SequenceWeights? weights = null, ReadOnlySpan<string> paths = default, Executor? executor = null,
        CPUClass? cpuClassCeiling = null)
    {
        byte[] encoded = Encoding.UTF8.GetBytes(path);
        using HandleReference pool = HandleReference.Of(executor?.Handle, "executor");
        IntPtr created = IntPtr.Zero;
        NativeOutcome outcome = default;
        NativeSequenceWeights lowered = default;
        NativeBytesView* views = Paths.Encode(paths);
        try
        {
            fixed (byte* data = encoded)
            fixed (sbyte* matrix = SequenceWeights.MatrixOf(weights))
            {
                NativeSequenceViewNvme arguments = new()
                {
                    Index = &created, Executor = pool.Pointer, Outcome = &outcome, Path = View(data, encoded.Length),
                    Nvme = new NativeNvmeOptions
                    {
                        Paths = (IntPtr)views, PathsCount = (nuint)paths.Length,
                        CpuClassCeiling = (int)(cpuClassCeiling ?? 0),
                    },
                    Weights = SequenceWeights.Lower(weights, matrix, &lowered),
                };
                usearch_sequence_view_nvme(&arguments);
            }
        }
        finally
        {
            NativeMemory.Free(views);
        }

        return new SequenceIndex(EngineHandle.Created<NativeSequenceIndexHandle>(created, outcome));
    }

    /// <summary>Reads a checkpoint's scoring kind and connectivity without loading it. Calls
    /// <c>usearch_sequence_metadata_read</c>.</summary>
    /// <param name="path">The checkpoint path.</param>
    /// <param name="executor">The workers the read runs on; null reads on the caller's thread.</param>
    /// <returns>What the checkpoint froze.</returns>
    public static SequenceMetadata MetadataRead(string path, Executor? executor = null)
    {
        byte[] encoded = Encoding.UTF8.GetBytes(path);
        using HandleReference pool = HandleReference.Of(executor?.Handle, "executor");
        NativeSequenceMetadata lowered = default;
        NativeOutcome outcome = default;
        fixed (byte* data = encoded)
        {
            NativeSequenceMetadataRead arguments = new()
            {
                Executor = pool.Pointer, Outcome = &outcome, Path = View(data, encoded.Length), Metadata = &lowered,
            };
            usearch_sequence_metadata_read(&arguments);
        }

        outcome.ThrowIfFailed();
        return SequenceMetadata.Lift(lowered);
    }

    /// <summary>Reads what a checkpoint in a buffer froze without loading it. Calls
    /// <c>usearch_sequence_metadata_read_buffer</c>.</summary>
    /// <param name="buffer">The checkpoint's bytes.</param>
    /// <param name="executor">The workers the read runs on; null reads on the caller's thread.</param>
    /// <returns>What the checkpoint froze.</returns>
    public static SequenceMetadata MetadataReadBuffer(ReadOnlySpan<byte> buffer, Executor? executor = null)
    {
        using HandleReference pool = HandleReference.Of(executor?.Handle, "executor");
        NativeSequenceMetadata lowered = default;
        NativeOutcome outcome = default;
        fixed (byte* source = buffer)
        {
            NativeSequenceMetadataReadBuffer arguments = new()
            {
                Executor = pool.Pointer, Outcome = &outcome, Buffer = View(source, buffer.Length),
                Metadata = &lowered,
            };
            usearch_sequence_metadata_read_buffer(&arguments);
        }

        outcome.ThrowIfFailed();
        return SequenceMetadata.Lift(lowered);
    }

    /// <summary>The receiver whose verbs block on <paramref name="executor"/>. Binding-only.</summary>
    /// <param name="executor">The executor the verbs run on.</param>
    /// <returns>The receiver, holding nothing but the two references.</returns>
    public SequenceIndexOn On(Executor executor) => new(this, executor);

    /// <summary>The receiver whose verbs submit their calls to <paramref name="executor"/> and answer a
    /// <see cref="Deferred{TResult}"/> at once. Binding-only.</summary>
    /// <param name="executor">The executor the calls are submitted to.</param>
    /// <returns>The receiver, holding nothing but the two references.</returns>
    public SequenceIndexDeferredOn Deferred(Executor executor) => new(this, executor);

    /// <inheritdoc cref="SequenceIndexOn.Reserve"/>
    public void Reserve(ulong entries, ulong payloadBytes = 0) =>
        new SequenceIndexOn(this, null).Reserve(entries, payloadBytes);

    /// <inheritdoc cref="SequenceIndexOn.Clear"/>
    public void Clear() => new SequenceIndexOn(this, null).Clear();

    /// <inheritdoc cref="SequenceIndexOn.Describe"/>
    public SequenceIndexDescription Describe() => new SequenceIndexOn(this, null).Describe();

    /// <inheritdoc cref="SequenceIndexOn.Count"/>
    public SequenceKeyStats Count(Key key) => new SequenceIndexOn(this, null).Count(key);

    /// <inheritdoc cref="SequenceIndexOn.Contains"/>
    public bool Contains(Key key) => new SequenceIndexOn(this, null).Contains(key);

    /// <inheritdoc cref="SequenceIndexOn.Get"/>
    public TapeEntries Get(Key key, Arena arena) => new SequenceIndexOn(this, null).Get(key, arena);

    /// <inheritdoc cref="SequenceIndexOn.Insert"/>
    public void Insert(
        ReadOnlyMemory<Key> keys, TapeView<byte> sequences, ulong expansionFactor = 0,
        DenseTraversalStats? stats = null, Progress? progress = null) =>
        new SequenceIndexOn(this, null).Insert(keys, sequences, expansionFactor, stats, progress);

    /// <inheritdoc cref="SequenceIndexOn.Refine"/>
    public RefineReport Refine(
        ulong expansionFactor = 0, ulong memoryLimit = 0, DenseTraversalStats? stats = null,
        Progress? progress = null, SequenceRefinePlan? plan = null) =>
        new SequenceIndexOn(this, null).Refine(expansionFactor, memoryLimit, stats, progress, plan);

    /// <inheritdoc cref="SequenceIndexOn.RefinePlanFinish"/>
    public RefineReport RefinePlanFinish(SequenceRefinePlan plan) =>
        new SequenceIndexOn(this, null).RefinePlanFinish(plan);

    /// <inheritdoc cref="SequenceIndexOn.Health"/>
    public DenseReachability Health() => new SequenceIndexOn(this, null).Health();

    /// <inheritdoc cref="SequenceIndexOn.Search"/>
    public Matches Search(
        TapeView<byte> queries, int wanted, Arena arena, ulong expansionFactor = 0,
        Func<Key, bool>? predicate = null, DenseTraversalStats? stats = null, Traversal traversal = Traversal.Walk,
        Progress? progress = null) =>
        new SequenceIndexOn(this, null).Search(
            queries, wanted, arena, expansionFactor, predicate, stats, traversal, progress);

    /// <inheritdoc cref="SequenceIndexOn.Reshard"/>
    public RefineReport Reshard(
        SequenceIndex[] sources, SequenceIndex[] targets, ulong seed = 0, ulong expansionFactor = 0,
        ulong memoryLimit = 0, DenseTraversalStats? stats = null, Progress? progress = null,
        SequenceReshardPlan? plan = null) =>
        new SequenceIndexOn(this, null).Reshard(
            sources, targets, seed, expansionFactor, memoryLimit, stats, progress, plan);

    /// <inheritdoc cref="SequenceIndexOn.ReshardPlanFinish"/>
    public RefineReport ReshardPlanFinish(SequenceReshardPlan plan) =>
        new SequenceIndexOn(this, null).ReshardPlanFinish(plan);

    /// <inheritdoc cref="SequenceIndexOn.Sample"/>
    public Sample Sample(int wanted, Arena arena, DenseSampleConfig config = default) =>
        new SequenceIndexOn(this, null).Sample(wanted, arena, config);

    /// <inheritdoc cref="SequenceIndexOn.Save"/>
    public void Save(string path, Progress? progress = null) => new SequenceIndexOn(this, null).Save(path, progress);

    /// <inheritdoc cref="SequenceIndexOn.SaveBuffer"/>
    public Monolith SaveBuffer(Arena arena, Progress? progress = null) =>
        new SequenceIndexOn(this, null).SaveBuffer(arena, progress);

    /// <summary>Opens a snapshot cursor over the current contents, pulled on the caller's thread or
    /// through <see cref="SequenceCursor.On"/>. Calls <c>usearch_sequence_cursor_init</c>.</summary>
    /// <param name="starts">The keys to walk from breadth-first; empty yields every entry as stored.</param>
    /// <returns>The cursor, which keeps the index alive until disposed.</returns>
    public SequenceCursor Cursor(ReadOnlySpan<Key> starts = default) => SequenceCursor.Open(this.Handle, starts);

    /// <summary>Releases the index, once every deferred call naming it is final. Calls
    /// <c>usearch_sequence_free</c>.</summary>
    public void Dispose() => this._handle.Dispose();

    private static NativeBytesView View(byte* data, int length) =>
        new() { Data = (IntPtr)data, Length = (nuint)length };
}

/// <summary>
/// A sequence index bound to the executor its verbs block on, as <c>index.On(executor)</c> makes it,
/// or to none, which runs every call on the caller's thread.
/// </summary>
/// <remarks>Binding-only; it holds two references and allocates nothing. Every callback a verb takes
/// runs on the engine's threads and must not throw.</remarks>
public readonly unsafe struct SequenceIndexOn
{
    internal SequenceIndexOn(SequenceIndex index, Executor? executor)
    {
        this.Index = index;
        this.Executor = executor;
    }

    /// <summary>The index the verbs address.</summary>
    public SequenceIndex Index { get; }

    /// <summary>The executor every verb blocks on; null runs them on the caller's thread.</summary>
    public Executor? Executor { get; }

    private CallHandles Handles => new(this.Index.Handle, this.Executor);

    /// <summary>Claims capacity for incoming entries, a hint since insertions grow on demand. Calls
    /// <c>usearch_sequence_reserve</c>.</summary>
    /// <param name="entries">The desired total entries.</param>
    /// <param name="payloadBytes">The desired total sequence bytes; zero derives.</param>
    public void Reserve(ulong entries, ulong payloadBytes = 0) =>
        Calls.Block(new SequenceReserveCall(this.Handles, entries, payloadBytes));

    /// <summary>Erases all entries, keeping the reserved capacity and the frozen kind. Calls
    /// <c>usearch_sequence_clear</c>.</summary>
    public void Clear() => Calls.Block(new SequenceClearCall(this.Handles));

    /// <summary>Reports the frozen metadata and the current counters in one call. Calls
    /// <c>usearch_sequence_describe</c>.</summary>
    /// <returns>The description.</returns>
    public SequenceIndexDescription Describe() => Calls.Block(new SequenceDescribeCall(this.Handles)).Result;

    /// <summary>Counts the entries stored under a key, with their total and largest byte lengths. Calls
    /// <c>usearch_sequence_count</c>.</summary>
    /// <param name="key">The identity to look for.</param>
    /// <returns>The key's entries and their extent.</returns>
    public SequenceKeyStats Count(Key key) => Calls.Block(new SequenceCountCall(this.Handles, key)).Result;

    /// <summary>Reports whether the index holds at least one entry under a key. Calls
    /// <c>usearch_sequence_count</c>.</summary>
    /// <param name="key">The identity to look for.</param>
    /// <returns>True when the index holds one or more.</returns>
    public bool Contains(Key key) => this.Count(key).Entries != 0;

    /// <summary>Retrieves every sequence stored under a key as one tape, in insertion order. Calls
    /// <c>usearch_sequence_get</c>.</summary>
    /// <param name="key">The identity to look for.</param>
    /// <param name="arena">Where the tape lands, rewound by the call.</param>
    /// <returns>The sequences, none when the key is absent, readable until the arena's next call.</returns>
    public TapeEntries Get(Key key, Arena arena) => Calls.Block(new SequenceGetCall(this.Handles, key, arena)).Result;

    /// <summary>Inserts a tape of sequences under their keys - the native batch operation, all or
    /// nothing per call. Calls <c>usearch_sequence_insert</c>.</summary>
    /// <param name="keys">One identity per sequence.</param>
    /// <param name="sequences">The sequences, bytes end to end around byte offsets.</param>
    /// <param name="expansionFactor">The breadth of the build walk; zero derives it.</param>
    /// <param name="stats">The traversal telemetry to fill, when the caller wants any.</param>
    /// <param name="progress">Hears the call's advance and may cancel it; null runs it unwatched.</param>
    public void Insert(
        ReadOnlyMemory<Key> keys, TapeView<byte> sequences, ulong expansionFactor = 0,
        DenseTraversalStats? stats = null, Progress? progress = null) =>
        Calls.Block(new SequenceInsertCall(this.Handles, keys, sequences, expansionFactor, stats, progress));

    /// <summary>Runs one round over the graph: health, heal, hubs, gather and restitch. Calls
    /// <c>usearch_sequence_refine</c>.</summary>
    /// <param name="expansionFactor">The breadth of the re-selection walk; zero derives it.</param>
    /// <param name="memoryLimit">Bytes a loaded NVMe handle holds for the part it rewrites; zero derives.</param>
    /// <param name="stats">The traversal telemetry to fill, when the caller wants any.</param>
    /// <param name="progress">Hears every phase's units and may stop the round at that barrier.</param>
    /// <param name="plan">A plan a stop leaves live and the next call continues; null runs one round.</param>
    /// <returns>What the round did to the graph.</returns>
    public RefineReport Refine(
        ulong expansionFactor = 0, ulong memoryLimit = 0, DenseTraversalStats? stats = null,
        Progress? progress = null, SequenceRefinePlan? plan = null) =>
        Calls.Block(new SequenceRefineCall(this.Handles, expansionFactor, memoryLimit, stats, progress, plan)).Result;

    /// <summary>Closes a refine plan and frees it whatever the call answers. Calls
    /// <c>usearch_sequence_refine_plan_finish</c>.</summary>
    /// <param name="plan">The plan, opened on this index and executor.</param>
    /// <returns>What the whole plan did to the graph.</returns>
    public RefineReport RefinePlanFinish(SequenceRefinePlan plan) =>
        Calls.Block(new SequenceRefinePlanFinishCall(this.Handles, plan)).Result;

    /// <summary>Measures how much of the graph a search reaches from the root. Calls
    /// <c>usearch_sequence_health</c>.</summary>
    /// <returns>The population audited and how much of it is reachable.</returns>
    public DenseReachability Health() => Calls.Block(new SequenceHealthCall(this.Handles)).Result;

    /// <summary>Resolves a tape of query sequences into the nearest entries. Calls
    /// <c>usearch_sequence_search</c>.</summary>
    /// <param name="queries">The query sequences, one query apiece.</param>
    /// <param name="wanted">The upper bound on matches per query.</param>
    /// <param name="arena">Where the run lands, rewound by the call.</param>
    /// <param name="expansionFactor">The breadth of the walk; zero derives it.</param>
    /// <param name="predicate">The per-key admission predicate; null admits every key.</param>
    /// <param name="stats">The traversal telemetry to fill, when the caller wants any.</param>
    /// <param name="traversal">A walk, the default, or an exact scan of every entry.</param>
    /// <param name="progress">Hears the call's advance and may cancel it; null runs it unwatched.</param>
    /// <returns>The run, one list per query, readable until the arena's next call.</returns>
    public Matches Search(
        TapeView<byte> queries, int wanted, Arena arena, ulong expansionFactor = 0,
        Func<Key, bool>? predicate = null, DenseTraversalStats? stats = null, Traversal traversal = Traversal.Walk,
        Progress? progress = null) =>
        Calls.Block(new SequenceSearchCall(
            this.Handles, queries, wanted, arena, expansionFactor, predicate, stats, traversal, progress)).Result;

    /// <summary>Moves the entries of the sources into the targets, keeping their links, then runs a round
    /// on every target that took entries: several sources into one gather, one into several cut. Calls
    /// <c>usearch_sequence_reshard</c>.</summary>
    /// <param name="sources">The handles whose entries move, each named once.</param>
    /// <param name="targets">The handles the entries move into, each a source itself or empty.</param>
    /// <param name="seed">Picks the part roots of a cut: equal seeds cut an unchanged source equally.</param>
    /// <param name="expansionFactor">The breadth of any walk beside the closing rounds; zero derives it.</param>
    /// <param name="memoryLimit">Bytes a loaded NVMe target holds for the part it rewrites; zero derives.</param>
    /// <param name="stats">The traversal telemetry to fill, when the caller wants any.</param>
    /// <param name="progress">Hears the closing rounds' phases and may stop them, never a move.</param>
    /// <param name="plan">A plan a stop leaves live and the next call continues; null runs to the end.</param>
    /// <returns>What the closing rounds did to the graph.</returns>
    public RefineReport Reshard(
        SequenceIndex[] sources, SequenceIndex[] targets, ulong seed = 0, ulong expansionFactor = 0,
        ulong memoryLimit = 0, DenseTraversalStats? stats = null, Progress? progress = null,
        SequenceReshardPlan? plan = null) =>
        Calls.Block(new SequenceReshardCall(
            this.Handles, sources, targets, seed, expansionFactor, memoryLimit, stats, progress, plan)).Result;

    /// <summary>Closes a reshard plan and frees it whatever the call answers. Calls
    /// <c>usearch_sequence_reshard_plan_finish</c>.</summary>
    /// <param name="plan">The plan, opened with this executor.</param>
    /// <returns>What the whole plan did to the graph.</returns>
    public RefineReport ReshardPlanFinish(SequenceReshardPlan plan) =>
        Calls.Block(new SequenceReshardPlanFinishCall(this.Handles, plan)).Result;

    /// <summary>Draws keys around the focus, or across the whole index, each hanging from a shown or
    /// earlier draw and placed in three dimensions. Calls <c>usearch_sequence_sample</c>.</summary>
    /// <param name="wanted">The upper bound on draws.</param>
    /// <param name="arena">Where the draws are written, rewound by the call.</param>
    /// <param name="config">The keys shown and their coordinates, the focus, the seed and the walk.</param>
    /// <returns>The draws, readable until the arena's next call.</returns>
    public Sample Sample(int wanted, Arena arena, DenseSampleConfig config = default) =>
        Calls.Block(new SequenceSampleCall(this.Handles, wanted, arena, config)).Result;

    /// <summary>Writes the index under a path, one monolith or a manifest beside its parts. Calls
    /// <c>usearch_sequence_save</c>.</summary>
    /// <param name="path">Where the checkpoint is written.</param>
    /// <param name="progress">Hears the call's advance and may cancel it; null runs it unwatched.</param>
    public void Save(string path, Progress? progress = null) =>
        Calls.Block(new SequenceSaveCall(this.Handles, path, progress));

    /// <summary>Writes the index into an arena, the same bytes <see cref="Save"/> writes for one shard.
    /// Calls <c>usearch_sequence_save_buffer</c>.</summary>
    /// <param name="arena">Where the bytes land, rewound by the call.</param>
    /// <param name="progress">Hears the call's advance and may cancel it; null runs it unwatched.</param>
    /// <returns>The checkpoint, readable until the arena's next call.</returns>
    public Monolith SaveBuffer(Arena arena, Progress? progress = null) =>
        Calls.Block(new SequenceSaveBufferCall(this.Handles, arena, progress)).Result;
}

/// <summary>A sequence index bound to the executor its verbs submit to, as <c>index.Deferred(executor)</c>
/// makes it, each verb answering a <see cref="Deferred{TResult}"/> once submitted.</summary>
/// <remarks>Binding-only; it holds two references and allocates nothing.</remarks>
public readonly struct SequenceIndexDeferredOn
{
    internal SequenceIndexDeferredOn(SequenceIndex index, Executor executor)
    {
        this.Index = index;
        this.Executor = executor;
    }

    /// <summary>The index the verbs address.</summary>
    public SequenceIndex Index { get; }

    /// <summary>The executor every verb submits to.</summary>
    public Executor Executor { get; }

    private CallHandles Handles => new(this.Index.Handle, this.Executor);

    /// <inheritdoc cref="SequenceIndexOn.Reserve"/>
    public Deferred Reserve(ulong entries, ulong payloadBytes = 0) =>
        Calls.Defer(new SequenceReserveCall(this.Handles, entries, payloadBytes));

    /// <inheritdoc cref="SequenceIndexOn.Clear"/>
    public Deferred Clear() => Calls.Defer(new SequenceClearCall(this.Handles));

    /// <inheritdoc cref="SequenceIndexOn.Describe"/>
    public Deferred<SequenceIndexDescription> Describe() => Calls.Defer(new SequenceDescribeCall(this.Handles));

    /// <inheritdoc cref="SequenceIndexOn.Count"/>
    public Deferred<SequenceKeyStats> Count(Key key) => Calls.Defer(new SequenceCountCall(this.Handles, key));

    /// <inheritdoc cref="SequenceIndexOn.Get"/>
    public Deferred<TapeEntries> Get(Key key, Arena arena) =>
        Calls.Defer(new SequenceGetCall(this.Handles, key, arena));

    /// <inheritdoc cref="SequenceIndexOn.Insert"/>
    public Deferred Insert(
        ReadOnlyMemory<Key> keys, TapeView<byte> sequences, ulong expansionFactor = 0,
        DenseTraversalStats? stats = null, Progress? progress = null) =>
        Calls.Defer(new SequenceInsertCall(this.Handles, keys, sequences, expansionFactor, stats, progress));

    /// <inheritdoc cref="SequenceIndexOn.Refine"/>
    public Deferred<RefineReport> Refine(
        ulong expansionFactor = 0, ulong memoryLimit = 0, DenseTraversalStats? stats = null,
        Progress? progress = null, SequenceRefinePlan? plan = null) =>
        Calls.Defer(new SequenceRefineCall(this.Handles, expansionFactor, memoryLimit, stats, progress, plan));

    /// <inheritdoc cref="SequenceIndexOn.RefinePlanFinish"/>
    public Deferred<RefineReport> RefinePlanFinish(SequenceRefinePlan plan) =>
        Calls.Defer(new SequenceRefinePlanFinishCall(this.Handles, plan));

    /// <inheritdoc cref="SequenceIndexOn.Health"/>
    public Deferred<DenseReachability> Health() => Calls.Defer(new SequenceHealthCall(this.Handles));

    /// <inheritdoc cref="SequenceIndexOn.Search"/>
    public Deferred<Matches> Search(
        TapeView<byte> queries, int wanted, Arena arena, ulong expansionFactor = 0,
        Func<Key, bool>? predicate = null, DenseTraversalStats? stats = null, Traversal traversal = Traversal.Walk,
        Progress? progress = null) =>
        Calls.Defer(new SequenceSearchCall(
            this.Handles, queries, wanted, arena, expansionFactor, predicate, stats, traversal, progress));

    /// <inheritdoc cref="SequenceIndexOn.Reshard"/>
    public Deferred<RefineReport> Reshard(
        SequenceIndex[] sources, SequenceIndex[] targets, ulong seed = 0, ulong expansionFactor = 0,
        ulong memoryLimit = 0, DenseTraversalStats? stats = null, Progress? progress = null,
        SequenceReshardPlan? plan = null) =>
        Calls.Defer(new SequenceReshardCall(
            this.Handles, sources, targets, seed, expansionFactor, memoryLimit, stats, progress, plan));

    /// <inheritdoc cref="SequenceIndexOn.ReshardPlanFinish"/>
    public Deferred<RefineReport> ReshardPlanFinish(SequenceReshardPlan plan) =>
        Calls.Defer(new SequenceReshardPlanFinishCall(this.Handles, plan));

    /// <inheritdoc cref="SequenceIndexOn.Sample"/>
    public Deferred<Sample> Sample(int wanted, Arena arena, DenseSampleConfig config = default) =>
        Calls.Defer(new SequenceSampleCall(this.Handles, wanted, arena, config));

    /// <inheritdoc cref="SequenceIndexOn.Save"/>
    public Deferred Save(string path, Progress? progress = null) =>
        Calls.Defer(new SequenceSaveCall(this.Handles, path, progress));

    /// <inheritdoc cref="SequenceIndexOn.SaveBuffer"/>
    public Deferred<Monolith> SaveBuffer(Arena arena, Progress? progress = null) =>
        Calls.Defer(new SequenceSaveBufferCall(this.Handles, arena, progress));
}

/// <summary>One sequence refine in flight between calls. A fresh one opens on the first
/// <see cref="SequenceIndex.Refine"/> naming it, a stopped round continues on the next, and
/// <see cref="SequenceIndex.RefinePlanFinish"/> closes it.</summary>
public sealed unsafe class SequenceRefinePlan
{
    private readonly PlanCell[] _cell = GC.AllocateArray<PlanCell>(1, pinned: true);

    /// <summary>Where the last call naming the plan stopped, <see cref="Phase.Done"/> once its round
    /// ended.</summary>
    public PhaseProgress Phase => this.Cell->Phase.Lift();

    internal PlanCell* Cell => (PlanCell*)Unsafe.AsPointer(ref MemoryMarshal.GetArrayDataReference(this._cell));
}

/// <summary>One sequence reshard in flight between calls. A fresh one opens on the first
/// <see cref="SequenceIndex.Reshard"/> naming it, which then reads its own lists rather than the call's,
/// and <see cref="SequenceIndex.ReshardPlanFinish"/> closes it.</summary>
public sealed unsafe class SequenceReshardPlan
{
    private readonly PlanCell[] _cell = GC.AllocateArray<PlanCell>(1, pinned: true);

    /// <summary>Where the last call naming the plan stopped, <see cref="Phase.Done"/> once its rounds
    /// ended.</summary>
    public PhaseProgress Phase => this.Cell->Phase.Lift();

    internal PlanCell* Cell => (PlanCell*)Unsafe.AsPointer(ref MemoryMarshal.GetArrayDataReference(this._cell));
}

/// <summary>A snapshot cursor over a sequence index, yielding each entry's key with the keys it links
/// to; the sequences come from <see cref="SequenceIndex.Get"/>. It keeps the index alive until disposed
/// and no executor: its pulls run on the caller's thread, or block on the executor <see cref="On"/>
/// names for each.</summary>
public sealed unsafe class SequenceCursor : IDisposable
{
    private readonly NativeSequenceCursorHandle _handle;

    private SequenceCursor(NativeSequenceCursorHandle handle) => this._handle = handle;

    internal static SequenceCursor Open(EngineHandle index, ReadOnlySpan<Key> starts)
    {
        HandleReference held = HandleReference.Of(index, "index");

        IntPtr created = IntPtr.Zero;
        NativeOutcome outcome = default;
        fixed (Key* run = starts)
        {
            NativeSequenceCursorInit arguments = new()
            {
                Index = held.Pointer, Outcome = &outcome, StartsCount = (nuint)starts.Length, Starts = run,
                Cursor = &created,
            };
            usearch_sequence_cursor_init(&arguments);
        }

        if (outcome.Status != Status.Success)
        {
            held.Dispose();
            outcome.ThrowIfFailed();
        }

        return new SequenceCursor(new NativeSequenceCursorHandle(created, held));
    }

    /// <summary>Copies the next batch of entries into an arena, the keys always and the keys each links
    /// to when asked for. Calls <c>usearch_sequence_cursor_next</c>.</summary>
    /// <param name="wanted">The upper bound on entries the pull writes.</param>
    /// <param name="arena">Where the batch lands, rewound by the pull.</param>
    /// <param name="links">Whether to export the keys each entry links to.</param>
    /// <returns>The batch, empty once the snapshot is exhausted.</returns>
    public DenseBatch Next(int wanted, Arena arena, bool links = false) => this.Pull(null, wanted, arena, links);

    /// <summary>The receiver whose pulls block on <paramref name="executor"/>, named again for every pull.</summary>
    /// <param name="executor">The executor every pull through the receiver runs on.</param>
    /// <returns>The receiver, which offers the same pulls.</returns>
    public SequenceCursorOn On(Executor executor) => new(this, executor);

    /// <summary>One pull on <paramref name="executor"/>, or the caller's thread when null.</summary>
    internal DenseBatch Pull(Executor? executor, int wanted, Arena arena, bool links)
    {
        nuint bound = Calls.Bound(wanted);
        using HandleReference cursor = HandleReference.Of(this._handle, "cursor");
        using HandleReference passed = Arena.Reference(arena);
        using HandleReference pool = HandleReference.Of(executor?.Handle, "executor");
        nuint written = 0;
        NativeOutcome outcome = default;
        NativeKeysView keys = default;
        NativeKeysTapeView linked = default;
        NativeSequenceCursorNext arguments = new()
        {
            Index = this._handle.Index, Executor = pool.Pointer, Outcome = &outcome,
            Cursor = cursor.Pointer, Keys = &keys, Links = links ? &linked : null, Written = &written,
            Arena = passed.Pointer, Wanted = bound,
        };
        usearch_sequence_cursor_next(&arguments);
        outcome.ThrowIfFailed();
        return new DenseBatch(keys, linked, links, (int)written);
    }

    /// <summary>Releases the cursor, then the index it kept alive. Calls
    /// <c>usearch_sequence_cursor_free</c>.</summary>
    public void Dispose() => this._handle.Dispose();
}

/// <summary>A cursor's pulls blocking on one executor, as <see cref="SequenceCursor.On"/> makes them.</summary>
public readonly struct SequenceCursorOn
{
    private readonly SequenceCursor _cursor;
    private readonly Executor _executor;

    internal SequenceCursorOn(SequenceCursor cursor, Executor executor)
    {
        this._cursor = cursor;
        this._executor = executor;
    }

    /// <inheritdoc cref="SequenceCursor.Next"/>
    public DenseBatch Next(int wanted, Arena arena, bool links = false) =>
        this._cursor.Pull(this._executor, wanted, arena, links);
}

#endregion Index

#region Calls

/// <summary>A reservation; see <see cref="SequenceIndexOn.Reserve"/>.</summary>
internal unsafe struct SequenceReserveCall(CallHandles handles, ulong entries, ulong payloadBytes) : ICall
{
    public readonly void Submit(CallOutputs* outputs, Completion completion)
    {
        using HandleReference index = handles.ReferenceIndex();
        using HandleReference executor = handles.ReferenceExecutor();
        NativeSequenceReserve arguments = new()
        {
            Index = index.Pointer, Executor = executor.Pointer, Outcome = &outputs->Outcome, Completion = completion,
            Entries = (nuint)entries, PayloadBytes = (nuint)payloadBytes,
        };
        usearch_sequence_reserve(&arguments);
    }

    public readonly void Release(CallOutputs* outputs) { }

    public readonly void Finish(CallOutputs* outputs) => outputs->Outcome.ThrowIfFailed();
}

/// <summary>An erasure; see <see cref="SequenceIndexOn.Clear"/>.</summary>
internal unsafe struct SequenceClearCall(CallHandles handles) : ICall
{
    public readonly void Submit(CallOutputs* outputs, Completion completion)
    {
        using HandleReference index = handles.ReferenceIndex();
        using HandleReference executor = handles.ReferenceExecutor();
        NativeSequenceClear arguments = new()
        {
            Index = index.Pointer, Executor = executor.Pointer, Outcome = &outputs->Outcome, Completion = completion,
        };
        usearch_sequence_clear(&arguments);
    }

    public readonly void Release(CallOutputs* outputs) { }

    public readonly void Finish(CallOutputs* outputs) => outputs->Outcome.ThrowIfFailed();
}

/// <summary>A description; see <see cref="SequenceIndexOn.Describe"/>.</summary>
internal unsafe struct SequenceDescribeCall(CallHandles handles) : ICall<SequenceIndexDescription>
{
    public SequenceIndexDescription Result { get; private set; }

    public readonly void Submit(CallOutputs* outputs, Completion completion)
    {
        using HandleReference index = handles.ReferenceIndex();
        using HandleReference executor = handles.ReferenceExecutor();
        NativeSequenceDescribe arguments = new()
        {
            Index = index.Pointer, Executor = executor.Pointer, Outcome = &outputs->Outcome, Completion = completion,
            Description = (NativeSequenceDescription*)outputs->Report,
        };
        usearch_sequence_describe(&arguments);
    }

    public readonly void Release(CallOutputs* outputs) { }

    public void Finish(CallOutputs* outputs)
    {
        outputs->Outcome.ThrowIfFailed();
        NativeSequenceDescription* lowered = (NativeSequenceDescription*)outputs->Report;
        this.Result = new SequenceIndexDescription
        {
            Metadata = SequenceMetadata.Lift(lowered->Metadata), Size = lowered->Size, Capacity = lowered->Capacity,
            MemoryUsage = lowered->MemoryUsage,
            CpuClass = Marshal.PtrToStringUTF8(usearch_cpu_class_name(lowered->CpuClass)) ?? "unknown",
            GpuClass = Marshal.PtrToStringUTF8(usearch_gpu_class_name(lowered->GpuClass)) ?? "unknown",
            ShardsCount = lowered->ShardsCount,
        };
    }
}

/// <summary>A count; see <see cref="SequenceIndexOn.Count"/>.</summary>
internal unsafe struct SequenceCountCall(CallHandles handles, Key key) : ICall<SequenceKeyStats>
{
    public SequenceKeyStats Result { get; private set; }

    public readonly void Submit(CallOutputs* outputs, Completion completion)
    {
        using HandleReference index = handles.ReferenceIndex();
        using HandleReference executor = handles.ReferenceExecutor();
        NativeSequenceCount arguments = new()
        {
            Index = index.Pointer, Executor = executor.Pointer, Outcome = &outputs->Outcome, Completion = completion,
            Key = key, KeyStats = (NativeSequenceKeyStats*)outputs->Report,
        };
        usearch_sequence_count(&arguments);
    }

    public readonly void Release(CallOutputs* outputs) { }

    public void Finish(CallOutputs* outputs)
    {
        outputs->Outcome.ThrowIfFailed();
        NativeSequenceKeyStats* lowered = (NativeSequenceKeyStats*)outputs->Report;
        this.Result = new SequenceKeyStats(lowered->Entries, lowered->TotalBytes, lowered->LargestBytes);
    }
}

/// <summary>A retrieval; see <see cref="SequenceIndexOn.Get"/>.</summary>
internal unsafe struct SequenceGetCall(CallHandles handles, Key key, Arena arena) : ICall<TapeEntries>
{
    public TapeEntries Result { get; private set; }

    public readonly void Submit(CallOutputs* outputs, Completion completion)
    {
        using HandleReference index = handles.ReferenceIndex();
        using HandleReference executor = handles.ReferenceExecutor();
        using HandleReference passed = Arena.Reference(arena);
        NativeSequenceGet arguments = new()
        {
            Index = index.Pointer, Executor = executor.Pointer, Outcome = &outputs->Outcome, Completion = completion,
            Key = key, Sequences = (NativeTapeView*)outputs->View, Written = &outputs->Written, Arena = passed.Pointer,
        };
        usearch_sequence_get(&arguments);
    }

    public readonly void Release(CallOutputs* outputs) { }

    public void Finish(CallOutputs* outputs)
    {
        outputs->Outcome.ThrowIfFailed();
        this.Result = new TapeEntries(*(NativeTapeView*)outputs->View, (int)outputs->Written, 1);
    }
}

/// <summary>An insertion; see <see cref="SequenceIndexOn.Insert"/>.</summary>
internal unsafe struct SequenceInsertCall(
    CallHandles handles, ReadOnlyMemory<Key> keys, TapeView<byte> sequences, ulong expansionFactor,
    DenseTraversalStats? stats, Progress? progress) : ICall
{
    private MemoryHandle _keys;
    private MemoryHandle _elements;
    private MemoryHandle _offsets;
    private Registration _progress;

    public void Submit(CallOutputs* outputs, Completion completion)
    {
        // The call has one count, so only this side sees the keys and the sequences disagree.
        if (keys.Length != sequences.Count)
        {
            throw new USearchException(
                Status.ShapeMismatch, "the key count does not match the sequence count it accompanies");
        }

        this._keys = keys.Pin();
        this._elements = sequences.PinElements();
        this._offsets = sequences.PinOffsets();
        using HandleReference index = handles.ReferenceIndex();
        using HandleReference executor = handles.ReferenceExecutor();
        NativeSequenceInsert arguments = new()
        {
            Index = index.Pointer, Executor = executor.Pointer, Outcome = &outputs->Outcome, Completion = completion,
            Count = (nuint)keys.Length,
            Keys = new NativeKeysView { Data = (IntPtr)this._keys.Pointer, Stride = (nuint)sizeof(Key) },
            Sequences = sequences.AsView(this._elements.Pointer, this._offsets.Pointer),
            ExpansionFactor = (nuint)expansionFactor, Progress = this._progress.Progress(progress),
            Stats = (NativeDenseTraversalStats*)outputs->Report,
        };
        usearch_sequence_insert(&arguments);
    }

    public void Release(CallOutputs* outputs)
    {
        this._keys.Dispose();
        this._elements.Dispose();
        this._offsets.Dispose();
        this._progress.Free();
    }

    public readonly void Finish(CallOutputs* outputs)
    {
        outputs->Outcome.ThrowIfFailed();
        stats?.Adopt(*(NativeDenseTraversalStats*)outputs->Report);
    }
}

/// <summary>A refinement; see <see cref="SequenceIndexOn.Refine"/>.</summary>
internal unsafe struct SequenceRefineCall(
    CallHandles handles, ulong expansionFactor, ulong memoryLimit, DenseTraversalStats? stats, Progress? progress,
    SequenceRefinePlan? plan) : ICall<RefineReport>
{
    private Registration _progress;

    public RefineReport Result { get; private set; }

    public void Submit(CallOutputs* outputs, Completion completion)
    {
        using HandleReference index = handles.ReferenceIndex();
        using HandleReference executor = handles.ReferenceExecutor();
        NativeSequenceRefine arguments = new()
        {
            Index = index.Pointer, Executor = executor.Pointer, Outcome = &outputs->Outcome, Completion = completion,
            Report = RefineReport.In(outputs),
            Refine = new() { ExpansionFactor = (nuint)expansionFactor, MemoryLimit = (nuint)memoryLimit },
            Progress = this._progress.Progress(progress), Stats = (NativeDenseTraversalStats*)outputs->Report,
            Plan = plan is null ? null : &plan.Cell->Handle, Phase = plan is null ? null : &plan.Cell->Phase,
        };
        usearch_sequence_refine(&arguments);
    }

    public void Release(CallOutputs* outputs) => this._progress.Free();

    public void Finish(CallOutputs* outputs)
    {
        outputs->Outcome.ThrowIfFailed();
        stats?.Adopt(*(NativeDenseTraversalStats*)outputs->Report);
        this.Result = RefineReport.Lift(RefineReport.In(outputs));
    }
}

/// <summary>A refine plan's close; see <see cref="SequenceIndexOn.RefinePlanFinish"/>.</summary>
internal unsafe struct SequenceRefinePlanFinishCall(CallHandles handles, SequenceRefinePlan plan)
    : ICall<RefineReport>
{
    public RefineReport Result { get; private set; }

    public readonly void Submit(CallOutputs* outputs, Completion completion)
    {
        using HandleReference index = handles.ReferenceIndex();
        using HandleReference executor = handles.ReferenceExecutor();
        NativeSequenceRefinePlanFinish arguments = new()
        {
            Index = index.Pointer, Executor = executor.Pointer, Outcome = &outputs->Outcome, Completion = completion,
            Plan = plan.Cell->Handle, Report = RefineReport.In(outputs),
        };
        usearch_sequence_refine_plan_finish(&arguments);
        plan.Cell->Handle = IntPtr.Zero;
    }

    public readonly void Release(CallOutputs* outputs) { }

    public void Finish(CallOutputs* outputs)
    {
        outputs->Outcome.ThrowIfFailed();
        this.Result = RefineReport.Lift(RefineReport.In(outputs));
    }
}

/// <summary>A reachability audit; see <see cref="SequenceIndexOn.Health"/>.</summary>
internal unsafe struct SequenceHealthCall(CallHandles handles) : ICall<DenseReachability>
{
    public DenseReachability Result { get; private set; }

    public readonly void Submit(CallOutputs* outputs, Completion completion)
    {
        using HandleReference index = handles.ReferenceIndex();
        using HandleReference executor = handles.ReferenceExecutor();
        NativeSequenceHealth arguments = new()
        {
            Index = index.Pointer, Executor = executor.Pointer, Outcome = &outputs->Outcome, Completion = completion,
            Reachability = (NativeDenseReachability*)outputs->Report,
        };
        usearch_sequence_health(&arguments);
    }

    public readonly void Release(CallOutputs* outputs) { }

    public void Finish(CallOutputs* outputs)
    {
        outputs->Outcome.ThrowIfFailed();
        this.Result = DenseReachability.Lift((NativeDenseReachability*)outputs->Report);
    }
}

/// <summary>A search; see <see cref="SequenceIndexOn.Search"/>.</summary>
internal unsafe struct SequenceSearchCall(
    CallHandles handles, TapeView<byte> queries, int wanted, Arena arena, ulong expansionFactor,
    Func<Key, bool>? predicate, DenseTraversalStats? stats, Traversal traversal, Progress? progress) : ICall<Matches>
{
    private MemoryHandle _elements;
    private MemoryHandle _offsets;
    private Registration _predicate;
    private Registration _progress;

    public Matches Result { get; private set; }

    public void Submit(CallOutputs* outputs, Completion completion)
    {
        nuint bound = Calls.Bound(wanted);
        this._elements = queries.PinElements();
        this._offsets = queries.PinOffsets();
        using HandleReference index = handles.ReferenceIndex();
        using HandleReference executor = handles.ReferenceExecutor();
        using HandleReference passed = Arena.Reference(arena);
        NativeSequenceSearch arguments = new()
        {
            Index = index.Pointer, Executor = executor.Pointer, Outcome = &outputs->Outcome, Completion = completion,
            Count = (nuint)queries.Count, Queries = queries.AsView(this._elements.Pointer, this._offsets.Pointer),
            Matches = (NativeMatchesView*)outputs->View, Arena = passed.Pointer, Wanted = bound,
            ExpansionFactor = (nuint)expansionFactor, Traversal = traversal,
            Predicate = this._predicate.Predicate(predicate), Progress = this._progress.Progress(progress),
            Stats = (NativeDenseTraversalStats*)outputs->Report,
        };
        usearch_sequence_search(&arguments);
    }

    public void Release(CallOutputs* outputs)
    {
        this._elements.Dispose();
        this._offsets.Dispose();
        this._predicate.Free();
        this._progress.Free();
    }

    public void Finish(CallOutputs* outputs)
    {
        outputs->Outcome.ThrowIfFailed();
        stats?.Adopt(*(NativeDenseTraversalStats*)outputs->Report);
        this.Result = new Matches(*(NativeMatchesView*)outputs->View, queries.Count);
    }
}

/// <summary>A reshard; see <see cref="SequenceIndexOn.Reshard"/>.</summary>
internal unsafe struct SequenceReshardCall(
    CallHandles handles, SequenceIndex[] sources, SequenceIndex[] targets, ulong seed, ulong expansionFactor,
    ulong memoryLimit, DenseTraversalStats? stats, Progress? progress, SequenceReshardPlan? plan)
    : ICall<RefineReport>
{
    private readonly HandleReference[] _references = new HandleReference[sources.Length + targets.Length];
    private readonly IntPtr[] _slots = new IntPtr[sources.Length + targets.Length];
    private MemoryHandle _pinnedSlots;
    private Registration _progress;

    public RefineReport Result { get; private set; }

    public void Submit(CallOutputs* outputs, Completion completion)
    {
        for (int position = 0; position < this._slots.Length; position++)
        {
            SequenceIndex named = position < sources.Length ? sources[position] : targets[position - sources.Length];
            this._references[position] = HandleReference.Of(named?.Handle, "index");
            this._slots[position] = this._references[position].Pointer;
        }

        this._pinnedSlots = this._slots.AsMemory().Pin();
        IntPtr* slots = (IntPtr*)this._pinnedSlots.Pointer;
        using HandleReference executor = handles.ReferenceExecutor();
        NativeSequenceReshard arguments = new()
        {
            SourcesCount = (nuint)sources.Length, Sources = slots, TargetsCount = (nuint)targets.Length,
            Targets = slots + sources.Length, Executor = executor.Pointer, Outcome = &outputs->Outcome,
            Completion = completion, Report = RefineReport.In(outputs),
            Refine = new() { ExpansionFactor = (nuint)expansionFactor, MemoryLimit = (nuint)memoryLimit },
            Seed = seed, Progress = this._progress.Progress(progress),
            Stats = (NativeDenseTraversalStats*)outputs->Report,
            Plan = plan is null ? null : &plan.Cell->Handle, Phase = plan is null ? null : &plan.Cell->Phase,
        };
        usearch_sequence_reshard(&arguments);
    }

    public void Release(CallOutputs* outputs)
    {
        for (int position = 0; position < this._references.Length; position++)
        {
            this._references[position].Dispose();
        }

        this._pinnedSlots.Dispose();
        this._progress.Free();
    }

    public void Finish(CallOutputs* outputs)
    {
        outputs->Outcome.ThrowIfFailed();
        stats?.Adopt(*(NativeDenseTraversalStats*)outputs->Report);
        this.Result = RefineReport.Lift(RefineReport.In(outputs));
    }
}

/// <summary>A reshard plan's close; see <see cref="SequenceIndexOn.ReshardPlanFinish"/>.</summary>
internal unsafe struct SequenceReshardPlanFinishCall(CallHandles handles, SequenceReshardPlan plan)
    : ICall<RefineReport>
{
    public RefineReport Result { get; private set; }

    public readonly void Submit(CallOutputs* outputs, Completion completion)
    {
        using HandleReference executor = handles.ReferenceExecutor();
        NativeSequenceReshardPlanFinish arguments = new()
        {
            Executor = executor.Pointer, Outcome = &outputs->Outcome, Completion = completion,
            Plan = plan.Cell->Handle, Report = RefineReport.In(outputs),
        };
        usearch_sequence_reshard_plan_finish(&arguments);
        plan.Cell->Handle = IntPtr.Zero;
    }

    public readonly void Release(CallOutputs* outputs) { }

    public void Finish(CallOutputs* outputs)
    {
        outputs->Outcome.ThrowIfFailed();
        this.Result = RefineReport.Lift(RefineReport.In(outputs));
    }
}

/// <summary>A sample; see <see cref="SequenceIndexOn.Sample"/>.</summary>
internal unsafe struct SequenceSampleCall(CallHandles handles, int wanted, Arena arena, DenseSampleConfig config)
    : ICall<Sample>
{
    private MemoryHandle _shown;
    private MemoryHandle _shownCoordinates;
    private MemoryHandle _focus;
    private Registration _progress;

    public Sample Result { get; private set; }

    public void Submit(CallOutputs* outputs, Completion completion)
    {
        ReadOnlyMemory<Key> shown = config.Shown, focus = config.Focus;
        ReadOnlyMemory<float> shownCoordinates = config.ShownCoordinates;
        // The call reads three coordinates per shown key, so only this side sees the two disagree.
        if (shownCoordinates.Length != shown.Length * 3)
        {
            throw new USearchException(Status.ShapeMismatch, "every shown key needs three coordinates");
        }

        nuint bound = Calls.Bound(wanted);
        SampleReports* reports = (SampleReports*)outputs->Report;
        this._shown = shown.Pin();
        this._shownCoordinates = shownCoordinates.Pin();
        this._focus = focus.Pin();
        using HandleReference index = handles.ReferenceIndex();
        using HandleReference executor = handles.ReferenceExecutor();
        using HandleReference passed = Arena.Reference(arena);
        NativeSequenceSample arguments = new()
        {
            Index = index.Pointer, Executor = executor.Pointer, Outcome = &outputs->Outcome, Completion = completion,
            ShownCount = (nuint)shown.Length,
            Shown = new NativeKeysView { Data = (IntPtr)this._shown.Pointer, Stride = (nuint)sizeof(Key) },
            ShownCoordinates = new NativeRowsView
            {
                Data = (IntPtr)this._shownCoordinates.Pointer, Stride = (nuint)(3 * sizeof(float)), Columns = 3,
                Scalar = Scalar.F32,
            },
            FocusCount = (nuint)focus.Length,
            Focus = new NativeKeysView { Data = (IntPtr)this._focus.Pointer, Stride = (nuint)sizeof(Key) },
            Keys = &reports->Keys, EntryOrdinals = &reports->EntryOrdinals,
            Parents = (NativeMatchesView*)outputs->View, Coordinates = &reports->Coordinates,
            Written = &outputs->Written, Arena = passed.Pointer, Wanted = bound, Seed = config.Seed,
            ExpansionFactor = (nuint)config.ExpansionFactor, Progress = this._progress.Progress(config.Progress),
            Stats = &reports->Stats,
        };
        usearch_sequence_sample(&arguments);
    }

    public void Release(CallOutputs* outputs)
    {
        this._shown.Dispose();
        this._shownCoordinates.Dispose();
        this._focus.Dispose();
        this._progress.Free();
    }

    public void Finish(CallOutputs* outputs)
    {
        outputs->Outcome.ThrowIfFailed();
        SampleReports* reports = (SampleReports*)outputs->Report;
        config.Stats?.Adopt(reports->Stats);
        this.Result = new Sample(
            reports->Keys, reports->EntryOrdinals, *(NativeMatchesView*)outputs->View, reports->Coordinates,
            (int)outputs->Written);
    }
}

/// <summary>A checkpoint write; see <see cref="SequenceIndexOn.Save"/>.</summary>
internal unsafe struct SequenceSaveCall(CallHandles handles, string path, Progress? progress) : ICall
{
    private readonly byte[] _path = Encoding.UTF8.GetBytes(path);
    private MemoryHandle _pinnedPath;
    private Registration _progress;

    public void Submit(CallOutputs* outputs, Completion completion)
    {
        this._pinnedPath = this._path.AsMemory().Pin();
        using HandleReference index = handles.ReferenceIndex();
        using HandleReference executor = handles.ReferenceExecutor();
        NativeSequenceSave arguments = new()
        {
            Index = index.Pointer, Executor = executor.Pointer, Outcome = &outputs->Outcome, Completion = completion,
            Path = new NativeBytesView { Data = (IntPtr)this._pinnedPath.Pointer, Length = (nuint)this._path.Length },
            Progress = this._progress.Progress(progress),
        };
        usearch_sequence_save(&arguments);
    }

    public void Release(CallOutputs* outputs)
    {
        this._pinnedPath.Dispose();
        this._progress.Free();
    }

    public readonly void Finish(CallOutputs* outputs) => outputs->Outcome.ThrowIfFailed();
}

/// <summary>A checkpoint write into an arena; see <see cref="SequenceIndexOn.SaveBuffer"/>.</summary>
internal unsafe struct SequenceSaveBufferCall(CallHandles handles, Arena arena, Progress? progress) : ICall<Monolith>
{
    private Registration _progress;

    public Monolith Result { get; private set; }

    public void Submit(CallOutputs* outputs, Completion completion)
    {
        using HandleReference index = handles.ReferenceIndex();
        using HandleReference executor = handles.ReferenceExecutor();
        using HandleReference passed = Arena.Reference(arena);
        NativeSequenceSaveBuffer arguments = new()
        {
            Index = index.Pointer, Executor = executor.Pointer, Outcome = &outputs->Outcome, Completion = completion,
            Monolith = (NativeBytesView*)outputs->View, Arena = passed.Pointer,
            Progress = this._progress.Progress(progress),
        };
        usearch_sequence_save_buffer(&arguments);
    }

    public void Release(CallOutputs* outputs) => this._progress.Free();

    public void Finish(CallOutputs* outputs)
    {
        outputs->Outcome.ThrowIfFailed();
        this.Result = new Monolith(*(NativeBytesView*)outputs->View);
    }
}

#endregion Calls

#region Native Vocabulary

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeSequenceWeights
{
    public sbyte* SubstitutionMatrix;
    public sbyte Gap;
}

[StructLayout(LayoutKind.Sequential)]
internal struct NativeSequenceMetadata
{
    public SequenceMetric Metric;
    public nuint Connectivity;
}

[StructLayout(LayoutKind.Sequential)]
internal struct NativeSequenceDescription
{
    public NativeSequenceMetadata Metadata;
    public nuint Size;
    public nuint Capacity;
    public nuint MemoryUsage;
    public int CpuClass;
    public int GpuClass;
    public nuint ShardsCount;
    public int LoadPath;
}

[StructLayout(LayoutKind.Sequential)]
internal struct NativeSequenceKeyStats
{
    public nuint Entries;
    public nuint TotalBytes;
    public nuint LargestBytes;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeSequenceInitRam
{
    public IntPtr* Index;
    public NativeOutcome* Outcome;
    public SequenceMetric Metric;
    public NativeSequenceWeights* Weights;
    public nuint* WindowWidths;
    public nuint WindowWidthsCount;
    public nuint Connectivity;
    public nuint ValuesPerKey;
    public NativeRamOptions Ram;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeSequenceInitVram
{
    public IntPtr* Index;
    public NativeOutcome* Outcome;
    public SequenceMetric Metric;
    public NativeSequenceWeights* Weights;
    public nuint* WindowWidths;
    public nuint WindowWidthsCount;
    public nuint Connectivity;
    public nuint ValuesPerKey;
    public NativeVramOptions Vram;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeSequenceLoadRam
{
    public IntPtr* Index;
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
    public NativeBytesView Path;
    public NativeRamOptions Ram;
    public nuint ValuesPerKey;
    public NativeSequenceWeights* Weights;
    public NativeProgress Progress;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeSequenceLoadRamBuffer
{
    public IntPtr* Index;
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
    public NativeBytesView Buffer;
    public NativeRamOptions Ram;
    public nuint ValuesPerKey;
    public NativeSequenceWeights* Weights;
    public NativeProgress Progress;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeSequenceLoadVram
{
    public IntPtr* Index;
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
    public NativeBytesView Path;
    public NativeVramOptions Vram;
    public nuint ValuesPerKey;
    public NativeSequenceWeights* Weights;
    public NativeProgress Progress;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeSequenceLoadVramBuffer
{
    public IntPtr* Index;
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
    public NativeBytesView Buffer;
    public NativeVramOptions Vram;
    public nuint ValuesPerKey;
    public NativeSequenceWeights* Weights;
    public NativeProgress Progress;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeSequenceViewRam
{
    public IntPtr* Index;
    public NativeOutcome* Outcome;
    public NativeBytesView Path;
    public NativeSequenceWeights* Weights;
    public NativeRamOptions Ram;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeSequenceViewNvme
{
    public IntPtr* Index;
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
    public NativeBytesView Path;
    public NativeNvmeOptions Nvme;
    public NativeSequenceWeights* Weights;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeSequenceReserve
{
    public IntPtr Index;
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
    public nuint Entries;
    public nuint PayloadBytes;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeSequenceClear
{
    public IntPtr Index;
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeSequenceInsert
{
    public IntPtr Index;
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
    public nuint Count;
    public NativeKeysView Keys;
    public NativeTapeView Sequences;
    public nuint ExpansionFactor;
    public NativeProgress Progress;
    public NativeDenseTraversalStats* Stats;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeSequenceRefine
{
    public IntPtr Index;
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
    public NativeRefineReport* Report;
    public NativeRefineOptions Refine;
    public NativeProgress Progress;
    public NativeDenseTraversalStats* Stats;
    public IntPtr* Plan;
    public NativePhaseProgress* Phase;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeSequenceRefinePlanFinish
{
    public IntPtr Index;
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
    public IntPtr Plan;
    public NativeRefineReport* Report;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeSequenceHealth
{
    public IntPtr Index;
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
    public NativeDenseReachability* Reachability;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeSequenceSearch
{
    public IntPtr Index;
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
    public nuint Count;
    public NativeTapeView Queries;
    public NativeMatchesView* Matches;
    public IntPtr Arena;
    public nuint Wanted;
    public nuint ExpansionFactor;
    public Traversal Traversal;
    public NativePredicate Predicate;
    public NativeProgress Progress;
    public NativeDenseTraversalStats* Stats;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeSequenceCount
{
    public IntPtr Index;
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
    public Key Key;
    public NativeSequenceKeyStats* KeyStats;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeSequenceGet
{
    public IntPtr Index;
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
    public Key Key;
    public NativeTapeView* Sequences;
    public nuint* Written;
    public IntPtr Arena;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeSequenceReshard
{
    public nuint SourcesCount;
    public IntPtr* Sources;
    public nuint TargetsCount;
    public IntPtr* Targets;
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
    public NativeRefineReport* Report;
    public NativeRefineOptions Refine;
    public ulong Seed;
    public NativeProgress Progress;
    public NativeDenseTraversalStats* Stats;
    public IntPtr* Plan;
    public NativePhaseProgress* Phase;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeSequenceReshardPlanFinish
{
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
    public IntPtr Plan;
    public NativeRefineReport* Report;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeSequenceCursorInit
{
    public IntPtr Index;
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public nuint StartsCount;
    public Key* Starts;
    public IntPtr* Cursor;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeSequenceCursorNext
{
    public IntPtr Index;
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
    public IntPtr Cursor;
    public NativeKeysView* Keys;
    public NativeKeysTapeView* Links;
    public nuint* Written;
    public IntPtr Arena;
    public nuint Wanted;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeSequenceSample
{
    public IntPtr Index;
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
    public nuint ShownCount;
    public NativeKeysView Shown;
    public NativeRowsView ShownCoordinates;
    public nuint FocusCount;
    public NativeKeysView Focus;
    public NativeKeysView* Keys;
    public nuint** EntryOrdinals;
    public NativeMatchesView* Parents;
    public NativeRowsView* Coordinates;
    public nuint* Written;
    public IntPtr Arena;
    public nuint Wanted;
    public ulong Seed;
    public nuint ExpansionFactor;
    public NativeProgress Progress;
    public NativeDenseTraversalStats* Stats;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeSequenceSave
{
    public IntPtr Index;
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
    public NativeBytesView Path;
    public NativeProgress Progress;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeSequenceSaveBuffer
{
    public IntPtr Index;
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
    public NativeBytesView* Monolith;
    public IntPtr Arena;
    public NativeProgress Progress;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeSequenceMetadataRead
{
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
    public NativeBytesView Path;
    public NativeSequenceMetadata* Metadata;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeSequenceMetadataReadBuffer
{
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
    public NativeBytesView Buffer;
    public NativeSequenceMetadata* Metadata;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeSequenceDescribe
{
    public IntPtr Index;
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
    public NativeSequenceDescription* Description;
}

#endregion Native Vocabulary

#region Entry Points

/// <summary>The sequence modality's entry points, each taking one argument struct.</summary>
internal static unsafe partial class NativeMethods
{
    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial IntPtr usearch_sequence_metric_name(SequenceMetric metric);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial Status usearch_sequence_metric_from_name(NativeBytesView name, out SequenceMetric metric);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_sequence_init_ram(NativeSequenceInitRam* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_sequence_init_vram(NativeSequenceInitVram* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_sequence_load_ram(NativeSequenceLoadRam* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_sequence_load_ram_buffer(NativeSequenceLoadRamBuffer* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_sequence_load_vram(NativeSequenceLoadVram* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_sequence_load_vram_buffer(NativeSequenceLoadVramBuffer* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_sequence_view_ram(NativeSequenceViewRam* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_sequence_view_nvme(NativeSequenceViewNvme* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_sequence_free(IntPtr index);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_sequence_reserve(NativeSequenceReserve* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_sequence_clear(NativeSequenceClear* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_sequence_insert(NativeSequenceInsert* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_sequence_refine(NativeSequenceRefine* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_sequence_refine_plan_finish(NativeSequenceRefinePlanFinish* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_sequence_health(NativeSequenceHealth* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_sequence_search(NativeSequenceSearch* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_sequence_count(NativeSequenceCount* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_sequence_get(NativeSequenceGet* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_sequence_reshard(NativeSequenceReshard* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_sequence_reshard_plan_finish(NativeSequenceReshardPlanFinish* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_sequence_cursor_init(NativeSequenceCursorInit* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_sequence_cursor_next(NativeSequenceCursorNext* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_sequence_cursor_free(IntPtr cursor);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_sequence_sample(NativeSequenceSample* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_sequence_save(NativeSequenceSave* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_sequence_save_buffer(NativeSequenceSaveBuffer* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_sequence_metadata_read(NativeSequenceMetadataRead* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_sequence_metadata_read_buffer(NativeSequenceMetadataReadBuffer* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_sequence_describe(NativeSequenceDescribe* arguments);
}

#endregion Entry Points

#region Handles

/// <summary>A <c>usearch_sequence_index_t</c>, freed when the wrapper is disposed or collected.</summary>
internal sealed class NativeSequenceIndexHandle : EngineHandle
{
    /// <inheritdoc/>
    protected override bool ReleaseHandle()
    {
        NativeMethods.usearch_sequence_free(this.handle);
        return true;
    }
}

/// <summary>A <c>usearch_sequence_cursor_t</c> and the reference on the index every pull names, both
/// released when the wrapper is disposed or collected.</summary>
internal sealed class NativeSequenceCursorHandle : EngineHandle
{
    private HandleReference _index;

    internal NativeSequenceCursorHandle(IntPtr cursor, HandleReference index)
    {
        this.SetHandle(cursor);
        this._index = index;
    }

    /// <summary>The index the cursor was opened on.</summary>
    internal IntPtr Index => this._index.Pointer;

    /// <inheritdoc/>
    protected override bool ReleaseHandle()
    {
        NativeMethods.usearch_sequence_cursor_free(this.handle);
        this._index.Dispose();
        return true;
    }
}

#endregion Handles
