/**
 *  @file csharp/src/Cloud.Unum.USearch/ShapesIndex.cs
 *  @author Ash Vardanian
 *  @date November 16, 2026
 *  @brief The shapes index handle, its scoring kinds, and the verbs that run over it.
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
/// How two shapes compare, invariant to rotation, reflection, translation and point order. Mirrors
/// <c>usearch_shapes_metric_t</c>.
/// </summary>
public enum ShapesMetric : int
{
    /// <summary>A kind the library could not name.</summary>
    Unknown = 0,
    /// <summary>The mean distance from each point to its nearest partner, both ways.</summary>
    Chamfer = 1,
    /// <summary>The largest distance from any point to its nearest partner, both ways.</summary>
    Hausdorff = 2,
    /// <summary>How differently each side's points sit relative to one another.</summary>
    GromovWasserstein = 3,
    /// <summary>An exact transport plan pairing the points, alternated with the best superposition.</summary>
    WassersteinProcrustes = 4,
}

/// <summary>The shapes spellings of <see cref="Names"/>.</summary>
public static partial class Names
{
    /// <summary>Spells a shapes kind, as <see cref="ShapesMetric.Chamfer"/> spells "chamfer". Calls
    /// <c>usearch_shapes_metric_name</c>.</summary>
    /// <param name="metric">The kind to spell.</param>
    /// <returns>The canonical spelling.</returns>
    public static string Name(this ShapesMetric metric) => Spelled(usearch_shapes_metric_name(metric));

    /// <summary>Reads a shapes kind back from its spelling. Calls <c>usearch_shapes_metric_from_name</c>.</summary>
    /// <param name="name">The canonical spelling.</param>
    /// <param name="metric">The kind it spells.</param>
    /// <returns>True when the spelling names a kind, false otherwise.</returns>
    public static bool TryParse(string name, out ShapesMetric metric)
    {
        ShapesMetric parsed = default;
        bool read = Read(name, view => usearch_shapes_metric_from_name(view, out parsed)) == Status.Success;
        metric = parsed;
        return read;
    }
}

/// <summary>
/// The construction options of a shapes index - the fields <c>usearch_shapes_init_ram_t</c> and
/// <c>usearch_shapes_init_vram_t</c> share.
/// </summary>
public readonly struct ShapesOptions
{
    /// <summary>The scoring kind.</summary>
    public ShapesMetric Metric { get; init; }

    /// <summary>The scalar type points are stored in, <see cref="Scalar.F16"/> or <see cref="Scalar.F32"/>.</summary>
    public Scalar Quantization { get; init; }

    /// <summary>The points every shape must have; zero admits any count.</summary>
    public ulong Points { get; init; }

    /// <summary>The connections-per-node bound of the graph; zero derives.</summary>
    public ulong Connectivity { get; init; }

    /// <summary>The upper bound on entries sharing one key; zero is unbounded.</summary>
    public ulong ValuesPerKey { get; init; }

    /// <summary>A Chamfer index over <see cref="Scalar.F32"/> points, with every optional field derived.</summary>
    public static ShapesOptions Default => new() { Metric = ShapesMetric.Chamfer, Quantization = Scalar.F32 };
}

/// <summary>The scoring kind and geometry a checkpoint froze. Mirrors <c>usearch_shapes_metadata_t</c>.</summary>
public readonly struct ShapesMetadata
{
    /// <summary>The scoring kind the checkpoint was built under.</summary>
    public ShapesMetric Metric { get; init; }

    /// <summary>The scalar type its points are stored in.</summary>
    public Scalar Quantization { get; init; }

    /// <summary>The points every shape has; zero when the count varies.</summary>
    public ulong Points { get; init; }

    /// <summary>The connections per node the graph was built with.</summary>
    public ulong Connectivity { get; init; }

    internal static ShapesMetadata Lift(in NativeShapesMetadata lowered) => new()
    {
        Metric = lowered.Metric, Quantization = lowered.Quantization, Points = lowered.Points,
        Connectivity = lowered.Connectivity,
    };
}

/// <summary>A live shapes index's frozen metadata and current counters, read in one call. Mirrors
/// <c>usearch_shapes_index_description_t</c>.</summary>
public readonly struct ShapesIndexDescription
{
    /// <summary>The frozen scoring kind and geometry.</summary>
    public ShapesMetadata Metadata { get; init; }

    /// <summary>The stored entries.</summary>
    public ulong Size { get; init; }

    /// <summary>The entries hosted without growing again.</summary>
    public ulong Capacity { get; init; }

    /// <summary>The points of every stored entry together.</summary>
    public ulong TotalPoints { get; init; }

    /// <summary>The bytes the handle holds, point storage included.</summary>
    public ulong MemoryUsage { get; init; }

    /// <summary>The CPU class dispatch picked on this machine; "unknown" on VRAM.</summary>
    public string CpuClass { get; init; }

    /// <summary>The GPU class dispatch picked on the index's devices; "unknown" for a host index.</summary>
    public string GpuClass { get; init; }

    /// <summary>The live shards.</summary>
    public ulong ShardsCount { get; init; }
}

/// <summary>What one key holds, enough to size a buffer before retrieving it. Mirrors
/// <c>usearch_shapes_key_stats_t</c>.</summary>
/// <param name="Entries">The entries stored under the key.</param>
/// <param name="TotalPoints">Their points in all.</param>
/// <param name="LargestPoints">The points of the largest.</param>
public readonly record struct ShapesKeyStats(ulong Entries, ulong TotalPoints, ulong LargestPoints);

#endregion Options

#region Index

/// <summary>
/// A Similarity Search Index over 3D point clouds of any size - molecules, conformers and protein
/// backbones - invariant to rotation, reflection, translation and point order.
/// </summary>
/// <remarks>
/// Every batch is a <see cref="TapeView{TElement}"/> of points, so a single shape is a tape of one.
/// The receivers are the dense index's: the index runs a verb on the caller's thread, <see cref="On"/>
/// blocks on an executor, and <see cref="Deferred"/> submits there and answers a
/// <see cref="Deferred{TResult}"/>. Every callback runs on the engine's threads and must not throw.
/// </remarks>
public sealed unsafe class ShapesIndex : IDisposable
{
    private readonly NativeShapesIndexHandle _handle;

    private ShapesIndex(NativeShapesIndexHandle handle) => this._handle = handle;

    /// <summary>The engine handle the verbs reference.</summary>
    internal NativeShapesIndexHandle Handle => this._handle;

    /// <summary>Creates an empty, mutable shapes index in host memory. Calls
    /// <c>usearch_shapes_init_ram</c>.</summary>
    /// <param name="options">The scoring kind, the point quantization and the graph's bounds.</param>
    /// <param name="memoryDomains">NUMA node IDs to place one shard on apiece; empty derives.</param>
    /// <param name="cpuClassCeiling">The newest class dispatch may pick, intersected with what the
    /// machine runs; null means no ceiling.</param>
    /// <returns>The new index.</returns>
    public static ShapesIndex InitRam(
        ShapesOptions options, ReadOnlySpan<int> memoryDomains = default, CPUClass? cpuClassCeiling = null)
    {
        IntPtr created = IntPtr.Zero;
        NativeOutcome outcome = default;
        fixed (int* domains = memoryDomains)
        {
            NativeShapesInitRam arguments = new()
            {
                Index = &created, Outcome = &outcome, Metric = options.Metric, Quantization = options.Quantization,
                Points = (nuint)options.Points, Connectivity = (nuint)options.Connectivity,
                ValuesPerKey = (nuint)options.ValuesPerKey,
                Ram = new NativeRamOptions
                {
                    MemoryDomains = (IntPtr)domains, MemoryDomainsCount = (nuint)memoryDomains.Length,
                    CpuClassCeiling = (int)(cpuClassCeiling ?? 0),
                },
            };
            usearch_shapes_init_ram(&arguments);
        }

        return new ShapesIndex(EngineHandle.Created<NativeShapesIndexHandle>(created, outcome));
    }

    /// <summary>Creates an empty, mutable shapes index in device memory. Calls
    /// <c>usearch_shapes_init_vram</c>.</summary>
    /// <param name="options">The scoring kind, the point quantization and the graph's bounds.</param>
    /// <param name="devices">CUDA device ordinals to place one shard on apiece; empty derives one.</param>
    /// <param name="gpuClassCeiling">The newest class dispatch may pick; null means no ceiling.</param>
    /// <returns>The new index.</returns>
    public static ShapesIndex InitVram(
        ShapesOptions options, ReadOnlySpan<int> devices = default, GPUClass? gpuClassCeiling = null)
    {
        IntPtr created = IntPtr.Zero;
        NativeOutcome outcome = default;
        fixed (int* ordinals = devices)
        {
            NativeShapesInitVram arguments = new()
            {
                Index = &created, Outcome = &outcome, Metric = options.Metric, Quantization = options.Quantization,
                Points = (nuint)options.Points, Connectivity = (nuint)options.Connectivity,
                ValuesPerKey = (nuint)options.ValuesPerKey,
                Vram = new NativeVramOptions
                {
                    Devices = (IntPtr)ordinals, DevicesCount = (nuint)devices.Length,
                    GpuClassCeiling = (int)(gpuClassCeiling ?? 0),
                },
            };
            usearch_shapes_init_vram(&arguments);
        }

        return new ShapesIndex(EngineHandle.Created<NativeShapesIndexHandle>(created, outcome));
    }

    /// <summary>Copies a checkpoint into a new, mutable index in host memory. Calls
    /// <c>usearch_shapes_load_ram</c>.</summary>
    /// <param name="path">The checkpoint to read, a monolith or a manifest.</param>
    /// <param name="memoryDomains">Where the parts land; empty derives every node.</param>
    /// <param name="valuesPerKey">The upper bound on entries sharing one key; zero is unbounded.</param>
    /// <param name="executor">The workers the read fans out over; null reads on the caller's thread.</param>
    /// <param name="cpuClassCeiling">The newest class dispatch may pick; null means no ceiling.</param>
    /// <returns>The new index.</returns>
    public static ShapesIndex LoadRam(
        string path, ReadOnlySpan<int> memoryDomains = default, ulong valuesPerKey = 0, Executor? executor = null,
        CPUClass? cpuClassCeiling = null)
    {
        byte[] encoded = Encoding.UTF8.GetBytes(path);
        using HandleReference pool = HandleReference.Of(executor?.Handle, "executor");
        IntPtr created = IntPtr.Zero;
        NativeOutcome outcome = default;
        fixed (byte* data = encoded)
        fixed (int* domains = memoryDomains)
        {
            NativeShapesLoadRam arguments = new()
            {
                Index = &created, Executor = pool.Pointer, Outcome = &outcome, Path = View(data, encoded.Length),
                Ram = new NativeRamOptions
                {
                    MemoryDomains = (IntPtr)domains, MemoryDomainsCount = (nuint)memoryDomains.Length,
                    CpuClassCeiling = (int)(cpuClassCeiling ?? 0),
                },
                ValuesPerKey = (nuint)valuesPerKey,
            };
            usearch_shapes_load_ram(&arguments);
        }

        return new ShapesIndex(EngineHandle.Created<NativeShapesIndexHandle>(created, outcome));
    }

    /// <summary>Copies a checkpoint into a new, mutable index in device memory. Calls
    /// <c>usearch_shapes_load_vram</c>.</summary>
    /// <param name="path">The checkpoint to read.</param>
    /// <param name="devices">Where the parts land; empty derives one free device.</param>
    /// <param name="valuesPerKey">The upper bound on entries sharing one key; zero is unbounded.</param>
    /// <param name="executor">The workers the upload runs on; null uploads on the caller's thread.</param>
    /// <param name="gpuClassCeiling">The newest class dispatch may pick; null means no ceiling.</param>
    /// <returns>The new index.</returns>
    public static ShapesIndex LoadVram(
        string path, ReadOnlySpan<int> devices = default, ulong valuesPerKey = 0, Executor? executor = null,
        GPUClass? gpuClassCeiling = null)
    {
        byte[] encoded = Encoding.UTF8.GetBytes(path);
        using HandleReference pool = HandleReference.Of(executor?.Handle, "executor");
        IntPtr created = IntPtr.Zero;
        NativeOutcome outcome = default;
        fixed (byte* data = encoded)
        fixed (int* ordinals = devices)
        {
            NativeShapesLoadVram arguments = new()
            {
                Index = &created, Executor = pool.Pointer, Outcome = &outcome, Path = View(data, encoded.Length),
                Vram = new NativeVramOptions
                {
                    Devices = (IntPtr)ordinals, DevicesCount = (nuint)devices.Length,
                    GpuClassCeiling = (int)(gpuClassCeiling ?? 0),
                },
                ValuesPerKey = (nuint)valuesPerKey,
            };
            usearch_shapes_load_vram(&arguments);
        }

        return new ShapesIndex(EngineHandle.Created<NativeShapesIndexHandle>(created, outcome));
    }

    /// <summary>Copies a checkpoint from a buffer into a new, mutable index in host memory. Calls
    /// <c>usearch_shapes_load_ram_buffer</c>.</summary>
    /// <param name="buffer">The checkpoint's bytes, copied in before the call returns.</param>
    /// <param name="memoryDomains">Where the parts land; empty derives every node.</param>
    /// <param name="valuesPerKey">The upper bound on entries sharing one key; zero is unbounded.</param>
    /// <param name="executor">The workers the copy fans out over; null copies on the caller's thread.</param>
    /// <param name="cpuClassCeiling">The newest class dispatch may pick; null means no ceiling.</param>
    /// <returns>The new index.</returns>
    public static ShapesIndex LoadRamBuffer(
        ReadOnlySpan<byte> buffer, ReadOnlySpan<int> memoryDomains = default, ulong valuesPerKey = 0,
        Executor? executor = null, CPUClass? cpuClassCeiling = null)
    {
        using HandleReference pool = HandleReference.Of(executor?.Handle, "executor");
        IntPtr created = IntPtr.Zero;
        NativeOutcome outcome = default;
        fixed (byte* source = buffer)
        fixed (int* domains = memoryDomains)
        {
            NativeShapesLoadRamBuffer arguments = new()
            {
                Index = &created, Executor = pool.Pointer, Outcome = &outcome, Buffer = View(source, buffer.Length),
                Ram = new NativeRamOptions
                {
                    MemoryDomains = (IntPtr)domains, MemoryDomainsCount = (nuint)memoryDomains.Length,
                    CpuClassCeiling = (int)(cpuClassCeiling ?? 0),
                },
                ValuesPerKey = (nuint)valuesPerKey,
            };
            usearch_shapes_load_ram_buffer(&arguments);
        }

        return new ShapesIndex(EngineHandle.Created<NativeShapesIndexHandle>(created, outcome));
    }

    /// <summary>Copies a checkpoint from a buffer into a new, mutable index in device memory. Calls
    /// <c>usearch_shapes_load_vram_buffer</c>.</summary>
    /// <param name="buffer">The checkpoint's bytes.</param>
    /// <param name="devices">Where the parts land; empty derives one free device.</param>
    /// <param name="valuesPerKey">The upper bound on entries sharing one key; zero is unbounded.</param>
    /// <param name="executor">The workers the upload runs on; null uploads on the caller's thread.</param>
    /// <param name="gpuClassCeiling">The newest class dispatch may pick; null means no ceiling.</param>
    /// <returns>The new index.</returns>
    public static ShapesIndex LoadVramBuffer(
        ReadOnlySpan<byte> buffer, ReadOnlySpan<int> devices = default, ulong valuesPerKey = 0,
        Executor? executor = null, GPUClass? gpuClassCeiling = null)
    {
        using HandleReference pool = HandleReference.Of(executor?.Handle, "executor");
        IntPtr created = IntPtr.Zero;
        NativeOutcome outcome = default;
        fixed (byte* source = buffer)
        fixed (int* ordinals = devices)
        {
            NativeShapesLoadVramBuffer arguments = new()
            {
                Index = &created, Executor = pool.Pointer, Outcome = &outcome, Buffer = View(source, buffer.Length),
                Vram = new NativeVramOptions
                {
                    Devices = (IntPtr)ordinals, DevicesCount = (nuint)devices.Length,
                    GpuClassCeiling = (int)(gpuClassCeiling ?? 0),
                },
                ValuesPerKey = (nuint)valuesPerKey,
            };
            usearch_shapes_load_vram_buffer(&arguments);
        }

        return new ShapesIndex(EngineHandle.Created<NativeShapesIndexHandle>(created, outcome));
    }

    /// <summary>Serves a checkpoint in place through a memory mapping, as a read-only index. Calls
    /// <c>usearch_shapes_view_ram</c>.</summary>
    /// <param name="path">The checkpoint to serve.</param>
    /// <param name="memoryDomains">Where the parts are served from; empty derives.</param>
    /// <param name="cpuClassCeiling">The newest class dispatch may pick; null means no ceiling.</param>
    /// <returns>The read-only index.</returns>
    public static ShapesIndex ViewRam(
        string path, ReadOnlySpan<int> memoryDomains = default, CPUClass? cpuClassCeiling = null)
    {
        byte[] encoded = Encoding.UTF8.GetBytes(path);
        IntPtr created = IntPtr.Zero;
        NativeOutcome outcome = default;
        fixed (byte* data = encoded)
        fixed (int* domains = memoryDomains)
        {
            NativeShapesViewRam arguments = new()
            {
                Index = &created, Outcome = &outcome, Path = View(data, encoded.Length),
                Ram = new NativeRamOptions
                {
                    MemoryDomains = (IntPtr)domains, MemoryDomainsCount = (nuint)memoryDomains.Length,
                    CpuClassCeiling = (int)(cpuClassCeiling ?? 0),
                },
            };
            usearch_shapes_view_ram(&arguments);
        }

        return new ShapesIndex(EngineHandle.Created<NativeShapesIndexHandle>(created, outcome));
    }

    /// <summary>Serves a checkpoint in place from NVMe drives as a read-only index. Calls
    /// <c>usearch_shapes_view_nvme</c>.</summary>
    /// <param name="path">The checkpoint to serve.</param>
    /// <param name="paths">The devices the parts are served from; empty serves each where its manifest
    /// points.</param>
    /// <param name="executor">The NVMe executor whose rings read the headers; null reads on the caller's
    /// thread.</param>
    /// <param name="cpuClassCeiling">The newest class dispatch may pick; null means no ceiling.</param>
    /// <returns>The read-only index.</returns>
    public static ShapesIndex ViewNvme(
        string path, ReadOnlySpan<string> paths = default, Executor? executor = null, CPUClass? cpuClassCeiling = null)
    {
        byte[] encoded = Encoding.UTF8.GetBytes(path);
        using HandleReference pool = HandleReference.Of(executor?.Handle, "executor");
        IntPtr created = IntPtr.Zero;
        NativeOutcome outcome = default;
        NativeBytesView* views = Paths.Encode(paths);
        try
        {
            fixed (byte* data = encoded)
            {
                NativeShapesViewNvme arguments = new()
                {
                    Index = &created, Executor = pool.Pointer, Outcome = &outcome, Path = View(data, encoded.Length),
                    Nvme = new NativeNvmeOptions
                    {
                        Paths = (IntPtr)views, PathsCount = (nuint)paths.Length,
                        CpuClassCeiling = (int)(cpuClassCeiling ?? 0),
                    },
                };
                usearch_shapes_view_nvme(&arguments);
            }
        }
        finally
        {
            NativeMemory.Free(views);
        }

        return new ShapesIndex(EngineHandle.Created<NativeShapesIndexHandle>(created, outcome));
    }

    /// <summary>Reads a checkpoint's scoring kind and connectivity without loading it. Calls
    /// <c>usearch_shapes_metadata_read</c>.</summary>
    /// <param name="path">The checkpoint path.</param>
    /// <param name="executor">The workers the read runs on; null reads on the caller's thread.</param>
    /// <returns>What the checkpoint froze.</returns>
    public static ShapesMetadata MetadataRead(string path, Executor? executor = null)
    {
        byte[] encoded = Encoding.UTF8.GetBytes(path);
        using HandleReference pool = HandleReference.Of(executor?.Handle, "executor");
        NativeShapesMetadata lowered = default;
        NativeOutcome outcome = default;
        fixed (byte* data = encoded)
        {
            NativeShapesMetadataRead arguments = new()
            {
                Executor = pool.Pointer, Outcome = &outcome, Path = View(data, encoded.Length), Metadata = &lowered,
            };
            usearch_shapes_metadata_read(&arguments);
        }

        outcome.ThrowIfFailed();
        return ShapesMetadata.Lift(lowered);
    }

    /// <summary>Reads what a checkpoint in a buffer froze without loading it. Calls
    /// <c>usearch_shapes_metadata_read_buffer</c>.</summary>
    /// <param name="buffer">The checkpoint's bytes.</param>
    /// <param name="executor">The workers the read runs on; null reads on the caller's thread.</param>
    /// <returns>What the checkpoint froze.</returns>
    public static ShapesMetadata MetadataReadBuffer(ReadOnlySpan<byte> buffer, Executor? executor = null)
    {
        using HandleReference pool = HandleReference.Of(executor?.Handle, "executor");
        NativeShapesMetadata lowered = default;
        NativeOutcome outcome = default;
        fixed (byte* source = buffer)
        {
            NativeShapesMetadataReadBuffer arguments = new()
            {
                Executor = pool.Pointer, Outcome = &outcome, Buffer = View(source, buffer.Length),
                Metadata = &lowered,
            };
            usearch_shapes_metadata_read_buffer(&arguments);
        }

        outcome.ThrowIfFailed();
        return ShapesMetadata.Lift(lowered);
    }

    /// <summary>The receiver whose verbs block on <paramref name="executor"/>. Binding-only.</summary>
    /// <param name="executor">The executor the verbs run on.</param>
    /// <returns>The receiver, holding nothing but the two references.</returns>
    public ShapesIndexOn On(Executor executor) => new(this, executor);

    /// <summary>The receiver whose verbs submit their calls to <paramref name="executor"/> and answer a
    /// <see cref="Deferred{TResult}"/> at once. Binding-only.</summary>
    /// <param name="executor">The executor the calls are submitted to.</param>
    /// <returns>The receiver, holding nothing but the two references.</returns>
    public ShapesIndexDeferredOn Deferred(Executor executor) => new(this, executor);

    /// <inheritdoc cref="ShapesIndexOn.Reserve"/>
    public void Reserve(ulong entries, ulong points = 0) =>
        new ShapesIndexOn(this, null).Reserve(entries, points);

    /// <inheritdoc cref="ShapesIndexOn.Clear"/>
    public void Clear() => new ShapesIndexOn(this, null).Clear();

    /// <inheritdoc cref="ShapesIndexOn.Describe"/>
    public ShapesIndexDescription Describe() => new ShapesIndexOn(this, null).Describe();

    /// <inheritdoc cref="ShapesIndexOn.Count"/>
    public ShapesKeyStats Count(Key key) => new ShapesIndexOn(this, null).Count(key);

    /// <inheritdoc cref="ShapesIndexOn.Contains"/>
    public bool Contains(Key key) => new ShapesIndexOn(this, null).Contains(key);

    /// <inheritdoc cref="ShapesIndexOn.Get{TScalar}"/>
    public TapeEntries Get<TScalar>(Key key, Arena arena) where TScalar : IScalar =>
        new ShapesIndexOn(this, null).Get<TScalar>(key, arena);

    /// <inheritdoc cref="ShapesIndexOn.Insert{TScalar}"/>
    public void Insert<TScalar>(
        ReadOnlyMemory<Key> keys, TapeView<TScalar> shapes, ulong expansionFactor = 0,
        DenseTraversalStats? stats = null, Progress? progress = null)
        where TScalar : unmanaged =>
        new ShapesIndexOn(this, null).Insert<TScalar>(keys, shapes, expansionFactor, stats, progress);

    /// <inheritdoc cref="ShapesIndexOn.Refine"/>
    public RefineReport Refine(
        ulong expansionFactor = 0, ulong memoryLimit = 0, DenseTraversalStats? stats = null,
        Progress? progress = null, ShapesRefinePlan? plan = null) =>
        new ShapesIndexOn(this, null).Refine(expansionFactor, memoryLimit, stats, progress, plan);

    /// <inheritdoc cref="ShapesIndexOn.RefinePlanFinish"/>
    public RefineReport RefinePlanFinish(ShapesRefinePlan plan) =>
        new ShapesIndexOn(this, null).RefinePlanFinish(plan);

    /// <inheritdoc cref="ShapesIndexOn.Health"/>
    public DenseReachability Health() => new ShapesIndexOn(this, null).Health();

    /// <inheritdoc cref="ShapesIndexOn.Search{TScalar}"/>
    public Matches Search<TScalar>(
        TapeView<TScalar> queries, int wanted, Arena arena, ulong expansionFactor = 0,
        Func<Key, bool>? predicate = null, DenseTraversalStats? stats = null, Traversal traversal = Traversal.Walk,
        Progress? progress = null)
        where TScalar : unmanaged =>
        new ShapesIndexOn(this, null).Search<TScalar>(
            queries, wanted, arena, expansionFactor, predicate, stats, traversal, progress);

    /// <inheritdoc cref="ShapesIndexOn.Reshard"/>
    public RefineReport Reshard(
        ShapesIndex[] sources, ShapesIndex[] targets, ulong seed = 0, ulong expansionFactor = 0,
        ulong memoryLimit = 0, DenseTraversalStats? stats = null, Progress? progress = null,
        ShapesReshardPlan? plan = null) =>
        new ShapesIndexOn(this, null).Reshard(
            sources, targets, seed, expansionFactor, memoryLimit, stats, progress, plan);

    /// <inheritdoc cref="ShapesIndexOn.ReshardPlanFinish"/>
    public RefineReport ReshardPlanFinish(ShapesReshardPlan plan) =>
        new ShapesIndexOn(this, null).ReshardPlanFinish(plan);

    /// <inheritdoc cref="ShapesIndexOn.Sample"/>
    public Sample Sample(int wanted, Arena arena, DenseSampleConfig config = default) =>
        new ShapesIndexOn(this, null).Sample(wanted, arena, config);

    /// <inheritdoc cref="ShapesIndexOn.Save"/>
    public void Save(string path, Progress? progress = null) => new ShapesIndexOn(this, null).Save(path, progress);

    /// <inheritdoc cref="ShapesIndexOn.SaveBuffer"/>
    public Monolith SaveBuffer(Arena arena, Progress? progress = null) =>
        new ShapesIndexOn(this, null).SaveBuffer(arena, progress);

    /// <summary>Opens a snapshot cursor over the current contents, pulled on the caller's thread or
    /// through <see cref="ShapesCursor.On"/>. Calls <c>usearch_shapes_cursor_init</c>.</summary>
    /// <param name="starts">The keys to walk from breadth-first; empty yields every entry as stored.</param>
    /// <returns>The cursor, which keeps the index alive until disposed.</returns>
    public ShapesCursor Cursor(ReadOnlySpan<Key> starts = default) => ShapesCursor.Open(this.Handle, starts);

    /// <summary>Releases the index, once every deferred call naming it is final. Calls
    /// <c>usearch_shapes_free</c>.</summary>
    public void Dispose() => this._handle.Dispose();

    private static NativeBytesView View(byte* data, int length) =>
        new() { Data = (IntPtr)data, Length = (nuint)length };
}

/// <summary>
/// A shapes index bound to the executor its verbs block on, as <c>index.On(executor)</c> makes it,
/// or to none, which runs every call on the caller's thread.
/// </summary>
/// <remarks>Binding-only; it holds two references and allocates nothing. Every callback a verb takes
/// runs on the engine's threads and must not throw.</remarks>
public readonly unsafe struct ShapesIndexOn
{
    internal ShapesIndexOn(ShapesIndex index, Executor? executor)
    {
        this.Index = index;
        this.Executor = executor;
    }

    /// <summary>The index the verbs address.</summary>
    public ShapesIndex Index { get; }

    /// <summary>The executor every verb blocks on; null runs them on the caller's thread.</summary>
    public Executor? Executor { get; }

    private CallHandles Handles => new(this.Index.Handle, this.Executor);

    /// <summary>Claims capacity for incoming entries, a hint since insertions grow on demand. Calls
    /// <c>usearch_shapes_reserve</c>.</summary>
    /// <param name="entries">The desired total entries.</param>
    /// <param name="points">The desired total points; zero derives.</param>
    public void Reserve(ulong entries, ulong points = 0) =>
        Calls.Block(new ShapesReserveCall(this.Handles, entries, points));

    /// <summary>Erases all entries, keeping the reserved capacity and the frozen kind. Calls
    /// <c>usearch_shapes_clear</c>.</summary>
    public void Clear() => Calls.Block(new ShapesClearCall(this.Handles));

    /// <summary>Reports the frozen metadata and the current counters in one call. Calls
    /// <c>usearch_shapes_describe</c>.</summary>
    /// <returns>The description.</returns>
    public ShapesIndexDescription Describe() => Calls.Block(new ShapesDescribeCall(this.Handles)).Result;

    /// <summary>Counts the entries stored under a key, with their total and largest point counts. Calls
    /// <c>usearch_shapes_count</c>.</summary>
    /// <param name="key">The identity to look for.</param>
    /// <returns>The key's entries and their extent.</returns>
    public ShapesKeyStats Count(Key key) => Calls.Block(new ShapesCountCall(this.Handles, key)).Result;

    /// <summary>Reports whether the index holds at least one entry under a key. Calls
    /// <c>usearch_shapes_count</c>.</summary>
    /// <param name="key">The identity to look for.</param>
    /// <returns>True when the index holds one or more.</returns>
    public bool Contains(Key key) => this.Count(key).Entries != 0;

    /// <summary>Retrieves every shape stored under a key as one tape, in insertion order and pose, its
    /// points converted to the kind asked for. Calls
    /// <c>usearch_shapes_get</c>.</summary>
    /// <typeparam name="TScalar">The scalar type the points land as.</typeparam>
    /// <param name="key">The identity to look for.</param>
    /// <param name="arena">Where the tape lands, rewound by the call.</param>
    /// <returns>The shapes, none when the key is absent, readable until the arena's next call.</returns>
    public TapeEntries Get<TScalar>(Key key, Arena arena) where TScalar : IScalar =>
        Calls.Block(new ShapesGetCall(this.Handles, key, arena, TScalar.Kind)).Result;

    /// <summary>Inserts a tape of shapes under their keys - the native batch operation, all or
    /// nothing per call. Calls <c>usearch_shapes_insert</c>.</summary>
    /// <param name="keys">One identity per shapes.</param>
    /// <typeparam name="TScalar">The coordinate type, converted on the way in.</typeparam>
    /// <param name="shapes">The shapes, points end to end around point offsets.</param>
    /// <param name="expansionFactor">The breadth of the build walk; zero derives it.</param>
    /// <param name="stats">The traversal telemetry to fill, when the caller wants any.</param>
    /// <param name="progress">Hears the call's advance and may cancel it; null runs it unwatched.</param>
    public void Insert<TScalar>(
        ReadOnlyMemory<Key> keys, TapeView<TScalar> shapes, ulong expansionFactor = 0,
        DenseTraversalStats? stats = null, Progress? progress = null)
        where TScalar : unmanaged =>
        Calls.Block(new ShapesInsertCall<TScalar>(this.Handles, keys, shapes, expansionFactor, stats, progress));

    /// <summary>Runs one round over the graph: health, heal, hubs, gather and restitch. Calls
    /// <c>usearch_shapes_refine</c>.</summary>
    /// <param name="expansionFactor">The breadth of the re-selection walk; zero derives it.</param>
    /// <param name="memoryLimit">Bytes a loaded NVMe handle holds for the part it rewrites; zero derives.</param>
    /// <param name="stats">The traversal telemetry to fill, when the caller wants any.</param>
    /// <param name="progress">Hears every phase's units and may stop the round at that barrier.</param>
    /// <param name="plan">A plan a stop leaves live and the next call continues; null runs one round.</param>
    /// <returns>What the round did to the graph.</returns>
    public RefineReport Refine(
        ulong expansionFactor = 0, ulong memoryLimit = 0, DenseTraversalStats? stats = null,
        Progress? progress = null, ShapesRefinePlan? plan = null) =>
        Calls.Block(new ShapesRefineCall(this.Handles, expansionFactor, memoryLimit, stats, progress, plan)).Result;

    /// <summary>Closes a refine plan and frees it whatever the call answers. Calls
    /// <c>usearch_shapes_refine_plan_finish</c>.</summary>
    /// <param name="plan">The plan, opened on this index and executor.</param>
    /// <returns>What the whole plan did to the graph.</returns>
    public RefineReport RefinePlanFinish(ShapesRefinePlan plan) =>
        Calls.Block(new ShapesRefinePlanFinishCall(this.Handles, plan)).Result;

    /// <summary>Measures how much of the graph a search reaches from the root. Calls
    /// <c>usearch_shapes_health</c>.</summary>
    /// <returns>The population audited and how much of it is reachable.</returns>
    public DenseReachability Health() => Calls.Block(new ShapesHealthCall(this.Handles)).Result;

    /// <summary>Resolves a tape of query shapes into the nearest entries. Calls
    /// <c>usearch_shapes_search</c>.</summary>
    /// <typeparam name="TScalar">The coordinate type, converted on the way in.</typeparam>
    /// <param name="queries">The query shapes, one query apiece.</param>
    /// <param name="wanted">The upper bound on matches per query.</param>
    /// <param name="arena">Where the run lands, rewound by the call.</param>
    /// <param name="expansionFactor">The breadth of the walk; zero derives it.</param>
    /// <param name="predicate">The per-key admission predicate; null admits every key.</param>
    /// <param name="stats">The traversal telemetry to fill, when the caller wants any.</param>
    /// <param name="traversal">A walk, the default, or an exact scan of every entry.</param>
    /// <param name="progress">Hears the call's advance and may cancel it; null runs it unwatched.</param>
    /// <returns>The run, one list per query, readable until the arena's next call.</returns>
    public Matches Search<TScalar>(
        TapeView<TScalar> queries, int wanted, Arena arena, ulong expansionFactor = 0,
        Func<Key, bool>? predicate = null, DenseTraversalStats? stats = null, Traversal traversal = Traversal.Walk,
        Progress? progress = null)
        where TScalar : unmanaged =>
        Calls.Block(new ShapesSearchCall<TScalar>(
            this.Handles, queries, wanted, arena, expansionFactor, predicate, stats, traversal, progress)).Result;

    /// <summary>Moves the entries of the sources into the targets, keeping their links, then runs a round
    /// on every target that took entries: several sources into one gather, one into several cut. Calls
    /// <c>usearch_shapes_reshard</c>.</summary>
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
        ShapesIndex[] sources, ShapesIndex[] targets, ulong seed = 0, ulong expansionFactor = 0,
        ulong memoryLimit = 0, DenseTraversalStats? stats = null, Progress? progress = null,
        ShapesReshardPlan? plan = null) =>
        Calls.Block(new ShapesReshardCall(
            this.Handles, sources, targets, seed, expansionFactor, memoryLimit, stats, progress, plan)).Result;

    /// <summary>Closes a reshard plan and frees it whatever the call answers. Calls
    /// <c>usearch_shapes_reshard_plan_finish</c>.</summary>
    /// <param name="plan">The plan, opened with this executor.</param>
    /// <returns>What the whole plan did to the graph.</returns>
    public RefineReport ReshardPlanFinish(ShapesReshardPlan plan) =>
        Calls.Block(new ShapesReshardPlanFinishCall(this.Handles, plan)).Result;

    /// <summary>Draws keys around the focus, or across the whole index, each hanging from a shown or
    /// earlier draw and placed in three dimensions. Calls <c>usearch_shapes_sample</c>.</summary>
    /// <param name="wanted">The upper bound on draws.</param>
    /// <param name="arena">Where the draws are written, rewound by the call.</param>
    /// <param name="config">The keys shown and their coordinates, the focus, the seed and the walk.</param>
    /// <returns>The draws, readable until the arena's next call.</returns>
    public Sample Sample(int wanted, Arena arena, DenseSampleConfig config = default) =>
        Calls.Block(new ShapesSampleCall(this.Handles, wanted, arena, config)).Result;

    /// <summary>Writes the index under a path, one monolith or a manifest beside its parts. Calls
    /// <c>usearch_shapes_save</c>.</summary>
    /// <param name="path">Where the checkpoint is written.</param>
    /// <param name="progress">Hears the call's advance and may cancel it; null runs it unwatched.</param>
    public void Save(string path, Progress? progress = null) =>
        Calls.Block(new ShapesSaveCall(this.Handles, path, progress));

    /// <summary>Writes the index into an arena, the same bytes <see cref="Save"/> writes for one shard.
    /// Calls <c>usearch_shapes_save_buffer</c>.</summary>
    /// <param name="arena">Where the bytes land, rewound by the call.</param>
    /// <param name="progress">Hears the call's advance and may cancel it; null runs it unwatched.</param>
    /// <returns>The checkpoint, readable until the arena's next call.</returns>
    public Monolith SaveBuffer(Arena arena, Progress? progress = null) =>
        Calls.Block(new ShapesSaveBufferCall(this.Handles, arena, progress)).Result;
}

/// <summary>A shapes index bound to the executor its verbs submit to, as <c>index.Deferred(executor)</c>
/// makes it, each verb answering a <see cref="Deferred{TResult}"/> once submitted.</summary>
/// <remarks>Binding-only; it holds two references and allocates nothing.</remarks>
public readonly struct ShapesIndexDeferredOn
{
    internal ShapesIndexDeferredOn(ShapesIndex index, Executor executor)
    {
        this.Index = index;
        this.Executor = executor;
    }

    /// <summary>The index the verbs address.</summary>
    public ShapesIndex Index { get; }

    /// <summary>The executor every verb submits to.</summary>
    public Executor Executor { get; }

    private CallHandles Handles => new(this.Index.Handle, this.Executor);

    /// <inheritdoc cref="ShapesIndexOn.Reserve"/>
    public Deferred Reserve(ulong entries, ulong points = 0) =>
        Calls.Defer(new ShapesReserveCall(this.Handles, entries, points));

    /// <inheritdoc cref="ShapesIndexOn.Clear"/>
    public Deferred Clear() => Calls.Defer(new ShapesClearCall(this.Handles));

    /// <inheritdoc cref="ShapesIndexOn.Describe"/>
    public Deferred<ShapesIndexDescription> Describe() => Calls.Defer(new ShapesDescribeCall(this.Handles));

    /// <inheritdoc cref="ShapesIndexOn.Count"/>
    public Deferred<ShapesKeyStats> Count(Key key) => Calls.Defer(new ShapesCountCall(this.Handles, key));

    /// <inheritdoc cref="ShapesIndexOn.Get{TScalar}"/>
    public Deferred<TapeEntries> Get<TScalar>(Key key, Arena arena) where TScalar : IScalar =>
        Calls.Defer(new ShapesGetCall(this.Handles, key, arena, TScalar.Kind));

    /// <inheritdoc cref="ShapesIndexOn.Insert{TScalar}"/>
    public Deferred Insert<TScalar>(
        ReadOnlyMemory<Key> keys, TapeView<TScalar> shapes, ulong expansionFactor = 0,
        DenseTraversalStats? stats = null, Progress? progress = null)
        where TScalar : unmanaged =>
        Calls.Defer(new ShapesInsertCall<TScalar>(this.Handles, keys, shapes, expansionFactor, stats, progress));

    /// <inheritdoc cref="ShapesIndexOn.Refine"/>
    public Deferred<RefineReport> Refine(
        ulong expansionFactor = 0, ulong memoryLimit = 0, DenseTraversalStats? stats = null,
        Progress? progress = null, ShapesRefinePlan? plan = null) =>
        Calls.Defer(new ShapesRefineCall(this.Handles, expansionFactor, memoryLimit, stats, progress, plan));

    /// <inheritdoc cref="ShapesIndexOn.RefinePlanFinish"/>
    public Deferred<RefineReport> RefinePlanFinish(ShapesRefinePlan plan) =>
        Calls.Defer(new ShapesRefinePlanFinishCall(this.Handles, plan));

    /// <inheritdoc cref="ShapesIndexOn.Health"/>
    public Deferred<DenseReachability> Health() => Calls.Defer(new ShapesHealthCall(this.Handles));

    /// <inheritdoc cref="ShapesIndexOn.Search{TScalar}"/>
    public Deferred<Matches> Search<TScalar>(
        TapeView<TScalar> queries, int wanted, Arena arena, ulong expansionFactor = 0,
        Func<Key, bool>? predicate = null, DenseTraversalStats? stats = null, Traversal traversal = Traversal.Walk,
        Progress? progress = null)
        where TScalar : unmanaged =>
        Calls.Defer(new ShapesSearchCall<TScalar>(
            this.Handles, queries, wanted, arena, expansionFactor, predicate, stats, traversal, progress));

    /// <inheritdoc cref="ShapesIndexOn.Reshard"/>
    public Deferred<RefineReport> Reshard(
        ShapesIndex[] sources, ShapesIndex[] targets, ulong seed = 0, ulong expansionFactor = 0,
        ulong memoryLimit = 0, DenseTraversalStats? stats = null, Progress? progress = null,
        ShapesReshardPlan? plan = null) =>
        Calls.Defer(new ShapesReshardCall(
            this.Handles, sources, targets, seed, expansionFactor, memoryLimit, stats, progress, plan));

    /// <inheritdoc cref="ShapesIndexOn.ReshardPlanFinish"/>
    public Deferred<RefineReport> ReshardPlanFinish(ShapesReshardPlan plan) =>
        Calls.Defer(new ShapesReshardPlanFinishCall(this.Handles, plan));

    /// <inheritdoc cref="ShapesIndexOn.Sample"/>
    public Deferred<Sample> Sample(int wanted, Arena arena, DenseSampleConfig config = default) =>
        Calls.Defer(new ShapesSampleCall(this.Handles, wanted, arena, config));

    /// <inheritdoc cref="ShapesIndexOn.Save"/>
    public Deferred Save(string path, Progress? progress = null) =>
        Calls.Defer(new ShapesSaveCall(this.Handles, path, progress));

    /// <inheritdoc cref="ShapesIndexOn.SaveBuffer"/>
    public Deferred<Monolith> SaveBuffer(Arena arena, Progress? progress = null) =>
        Calls.Defer(new ShapesSaveBufferCall(this.Handles, arena, progress));
}

/// <summary>One shapes refine in flight between calls. A fresh one opens on the first
/// <see cref="ShapesIndex.Refine"/> naming it, a stopped round continues on the next, and
/// <see cref="ShapesIndex.RefinePlanFinish"/> closes it.</summary>
public sealed unsafe class ShapesRefinePlan
{
    private readonly PlanCell[] _cell = GC.AllocateArray<PlanCell>(1, pinned: true);

    /// <summary>Where the last call naming the plan stopped, <see cref="Phase.Done"/> once its round
    /// ended.</summary>
    public PhaseProgress Phase => this.Cell->Phase.Lift();

    internal PlanCell* Cell => (PlanCell*)Unsafe.AsPointer(ref MemoryMarshal.GetArrayDataReference(this._cell));
}

/// <summary>One shapes reshard in flight between calls. A fresh one opens on the first
/// <see cref="ShapesIndex.Reshard"/> naming it, which then reads its own lists rather than the call's,
/// and <see cref="ShapesIndex.ReshardPlanFinish"/> closes it.</summary>
public sealed unsafe class ShapesReshardPlan
{
    private readonly PlanCell[] _cell = GC.AllocateArray<PlanCell>(1, pinned: true);

    /// <summary>Where the last call naming the plan stopped, <see cref="Phase.Done"/> once its rounds
    /// ended.</summary>
    public PhaseProgress Phase => this.Cell->Phase.Lift();

    internal PlanCell* Cell => (PlanCell*)Unsafe.AsPointer(ref MemoryMarshal.GetArrayDataReference(this._cell));
}

/// <summary>A snapshot cursor over a shapes index, yielding each entry's key with the keys it links
/// to; the shapes come from <see cref="ShapesIndex.Get"/>. It keeps the index alive until disposed
/// and no executor: its pulls run on the caller's thread, or block on the executor <see cref="On"/>
/// names for each.</summary>
public sealed unsafe class ShapesCursor : IDisposable
{
    private readonly NativeShapesCursorHandle _handle;

    private ShapesCursor(NativeShapesCursorHandle handle) => this._handle = handle;

    internal static ShapesCursor Open(EngineHandle index, ReadOnlySpan<Key> starts)
    {
        HandleReference held = HandleReference.Of(index, "index");

        IntPtr created = IntPtr.Zero;
        NativeOutcome outcome = default;
        fixed (Key* run = starts)
        {
            NativeShapesCursorInit arguments = new()
            {
                Index = held.Pointer, Outcome = &outcome, StartsCount = (nuint)starts.Length, Starts = run,
                Cursor = &created,
            };
            usearch_shapes_cursor_init(&arguments);
        }

        if (outcome.Status != Status.Success)
        {
            held.Dispose();
            outcome.ThrowIfFailed();
        }

        return new ShapesCursor(new NativeShapesCursorHandle(created, held));
    }

    /// <summary>Copies the next batch of entries into an arena, the keys always and the keys each links
    /// to when asked for. Calls <c>usearch_shapes_cursor_next</c>.</summary>
    /// <param name="wanted">The upper bound on entries the pull writes.</param>
    /// <param name="arena">Where the batch lands, rewound by the pull.</param>
    /// <param name="links">Whether to export the keys each entry links to.</param>
    /// <returns>The batch, empty once the snapshot is exhausted.</returns>
    public DenseBatch Next(int wanted, Arena arena, bool links = false) => this.Pull(null, wanted, arena, links);

    /// <summary>The receiver whose pulls block on <paramref name="executor"/>, named again for every pull.</summary>
    /// <param name="executor">The executor every pull through the receiver runs on.</param>
    /// <returns>The receiver, which offers the same pulls.</returns>
    public ShapesCursorOn On(Executor executor) => new(this, executor);

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
        NativeShapesCursorNext arguments = new()
        {
            Index = this._handle.Index, Executor = pool.Pointer, Outcome = &outcome,
            Cursor = cursor.Pointer, Keys = &keys, Links = links ? &linked : null, Written = &written,
            Arena = passed.Pointer, Wanted = bound,
        };
        usearch_shapes_cursor_next(&arguments);
        outcome.ThrowIfFailed();
        return new DenseBatch(keys, linked, links, (int)written, arena.Description);
    }

    /// <summary>Releases the cursor, then the index it kept alive. Calls
    /// <c>usearch_shapes_cursor_free</c>.</summary>
    public void Dispose() => this._handle.Dispose();
}

/// <summary>A cursor's pulls blocking on one executor, as <see cref="ShapesCursor.On"/> makes them.</summary>
public readonly struct ShapesCursorOn
{
    private readonly ShapesCursor _cursor;
    private readonly Executor _executor;

    internal ShapesCursorOn(ShapesCursor cursor, Executor executor)
    {
        this._cursor = cursor;
        this._executor = executor;
    }

    /// <inheritdoc cref="ShapesCursor.Next"/>
    public DenseBatch Next(int wanted, Arena arena, bool links = false) =>
        this._cursor.Pull(this._executor, wanted, arena, links);
}

#endregion Index

#region Calls

/// <summary>A reservation; see <see cref="ShapesIndexOn.Reserve"/>.</summary>
internal unsafe struct ShapesReserveCall(CallHandles handles, ulong entries, ulong points) : ICall
{
    public readonly void Submit(CallOutputs* outputs, Completion completion)
    {
        using HandleReference index = handles.ReferenceIndex();
        using HandleReference executor = handles.ReferenceExecutor();
        NativeShapesReserve arguments = new()
        {
            Index = index.Pointer, Executor = executor.Pointer, Outcome = &outputs->Outcome, Completion = completion,
            Entries = (nuint)entries, Points = (nuint)points,
        };
        usearch_shapes_reserve(&arguments);
    }

    public readonly void Release(CallOutputs* outputs) { }

    public readonly void Finish(CallOutputs* outputs) => outputs->Outcome.ThrowIfFailed();
}

/// <summary>An erasure; see <see cref="ShapesIndexOn.Clear"/>.</summary>
internal unsafe struct ShapesClearCall(CallHandles handles) : ICall
{
    public readonly void Submit(CallOutputs* outputs, Completion completion)
    {
        using HandleReference index = handles.ReferenceIndex();
        using HandleReference executor = handles.ReferenceExecutor();
        NativeShapesClear arguments = new()
        {
            Index = index.Pointer, Executor = executor.Pointer, Outcome = &outputs->Outcome, Completion = completion,
        };
        usearch_shapes_clear(&arguments);
    }

    public readonly void Release(CallOutputs* outputs) { }

    public readonly void Finish(CallOutputs* outputs) => outputs->Outcome.ThrowIfFailed();
}

/// <summary>A description; see <see cref="ShapesIndexOn.Describe"/>.</summary>
internal unsafe struct ShapesDescribeCall(CallHandles handles) : ICall<ShapesIndexDescription>
{
    public ShapesIndexDescription Result { get; private set; }

    public readonly void Submit(CallOutputs* outputs, Completion completion)
    {
        using HandleReference index = handles.ReferenceIndex();
        using HandleReference executor = handles.ReferenceExecutor();
        NativeShapesDescribe arguments = new()
        {
            Index = index.Pointer, Executor = executor.Pointer, Outcome = &outputs->Outcome, Completion = completion,
            Description = (NativeShapesDescription*)outputs->Report,
        };
        usearch_shapes_describe(&arguments);
    }

    public readonly void Release(CallOutputs* outputs) { }

    public void Finish(CallOutputs* outputs)
    {
        outputs->Outcome.ThrowIfFailed();
        NativeShapesDescription* lowered = (NativeShapesDescription*)outputs->Report;
        this.Result = new ShapesIndexDescription
        {
            Metadata = ShapesMetadata.Lift(lowered->Metadata), Size = lowered->Size, Capacity = lowered->Capacity,
            TotalPoints = lowered->TotalPoints, MemoryUsage = lowered->MemoryUsage,
            CpuClass = Marshal.PtrToStringUTF8(usearch_cpu_class_name(lowered->CpuClass)) ?? "unknown",
            GpuClass = Marshal.PtrToStringUTF8(usearch_gpu_class_name(lowered->GpuClass)) ?? "unknown",
            ShardsCount = lowered->ShardsCount,
        };
    }
}

/// <summary>A count; see <see cref="ShapesIndexOn.Count"/>.</summary>
internal unsafe struct ShapesCountCall(CallHandles handles, Key key) : ICall<ShapesKeyStats>
{
    public ShapesKeyStats Result { get; private set; }

    public readonly void Submit(CallOutputs* outputs, Completion completion)
    {
        using HandleReference index = handles.ReferenceIndex();
        using HandleReference executor = handles.ReferenceExecutor();
        NativeShapesCount arguments = new()
        {
            Index = index.Pointer, Executor = executor.Pointer, Outcome = &outputs->Outcome, Completion = completion,
            Key = key, KeyStats = (NativeShapesKeyStats*)outputs->Report,
        };
        usearch_shapes_count(&arguments);
    }

    public readonly void Release(CallOutputs* outputs) { }

    public void Finish(CallOutputs* outputs)
    {
        outputs->Outcome.ThrowIfFailed();
        NativeShapesKeyStats* lowered = (NativeShapesKeyStats*)outputs->Report;
        this.Result = new ShapesKeyStats(lowered->Entries, lowered->TotalPoints, lowered->LargestPoints);
    }
}

/// <summary>A retrieval; see <see cref="ShapesIndexOn.Get{TScalar}"/>.</summary>
internal unsafe struct ShapesGetCall(CallHandles handles, Key key, Arena arena, Scalar outputsScalar)
    : ICall<TapeEntries>
{
    public TapeEntries Result { get; private set; }

    public readonly void Submit(CallOutputs* outputs, Completion completion)
    {
        using HandleReference index = handles.ReferenceIndex();
        using HandleReference executor = handles.ReferenceExecutor();
        using HandleReference passed = Arena.Reference(arena);
        NativeShapesGet arguments = new()
        {
            Index = index.Pointer, Executor = executor.Pointer, Outcome = &outputs->Outcome, Completion = completion,
            Key = key, Shapes = (NativeTapeView*)outputs->View, Written = &outputs->Written, Arena = passed.Pointer,
            OutputsScalar = outputsScalar,
        };
        usearch_shapes_get(&arguments);
    }

    public readonly void Release(CallOutputs* outputs) { }

    public void Finish(CallOutputs* outputs)
    {
        outputs->Outcome.ThrowIfFailed();
        this.Result = new TapeEntries(*(NativeTapeView*)outputs->View, (int)outputs->Written, 3, arena.Description);
    }
}

/// <summary>An insertion; see <see cref="ShapesIndexOn.Insert{TScalar}"/>.</summary>
internal unsafe struct ShapesInsertCall<TScalar>(
    CallHandles handles, ReadOnlyMemory<Key> keys, TapeView<TScalar> shapes, ulong expansionFactor,
    DenseTraversalStats? stats, Progress? progress) : ICall
    where TScalar : unmanaged
{
    private MemoryHandle _keys;
    private MemoryHandle _elements;
    private MemoryHandle _offsets;
    private Registration _progress;

    public void Submit(CallOutputs* outputs, Completion completion)
    {
        // The call has one count, so only this side sees the keys and the shapes disagree.
        if (keys.Length != shapes.Count)
        {
            throw new USearchException(
                Status.ShapeMismatch, "the key count does not match the shape count it accompanies");
        }

        this._keys = keys.Pin();
        this._elements = shapes.PinElements();
        this._offsets = shapes.PinOffsets();
        using HandleReference index = handles.ReferenceIndex();
        using HandleReference executor = handles.ReferenceExecutor();
        NativeShapesInsert arguments = new()
        {
            Index = index.Pointer, Executor = executor.Pointer, Outcome = &outputs->Outcome, Completion = completion,
            Count = (nuint)keys.Length,
            Keys = new NativeKeysView { Data = (IntPtr)this._keys.Pointer, Stride = (nuint)sizeof(Key) },
            Shapes = shapes.AsView(this._elements.Pointer, this._offsets.Pointer),
            ExpansionFactor = (nuint)expansionFactor, Progress = this._progress.Progress(progress),
            Stats = (NativeDenseTraversalStats*)outputs->Report,
        };
        usearch_shapes_insert(&arguments);
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

/// <summary>A refinement; see <see cref="ShapesIndexOn.Refine"/>.</summary>
internal unsafe struct ShapesRefineCall(
    CallHandles handles, ulong expansionFactor, ulong memoryLimit, DenseTraversalStats? stats, Progress? progress,
    ShapesRefinePlan? plan) : ICall<RefineReport>
{
    private Registration _progress;

    public RefineReport Result { get; private set; }

    public void Submit(CallOutputs* outputs, Completion completion)
    {
        using HandleReference index = handles.ReferenceIndex();
        using HandleReference executor = handles.ReferenceExecutor();
        NativeShapesRefine arguments = new()
        {
            Index = index.Pointer, Executor = executor.Pointer, Outcome = &outputs->Outcome, Completion = completion,
            Report = RefineReport.In(outputs),
            Refine = new() { ExpansionFactor = (nuint)expansionFactor, MemoryLimit = (nuint)memoryLimit },
            Progress = this._progress.Progress(progress), Stats = (NativeDenseTraversalStats*)outputs->Report,
            Plan = plan is null ? null : &plan.Cell->Handle, Phase = plan is null ? null : &plan.Cell->Phase,
        };
        usearch_shapes_refine(&arguments);
    }

    public void Release(CallOutputs* outputs) => this._progress.Free();

    public void Finish(CallOutputs* outputs)
    {
        outputs->Outcome.ThrowIfFailed();
        stats?.Adopt(*(NativeDenseTraversalStats*)outputs->Report);
        this.Result = RefineReport.Lift(RefineReport.In(outputs));
    }
}

/// <summary>A refine plan's close; see <see cref="ShapesIndexOn.RefinePlanFinish"/>.</summary>
internal unsafe struct ShapesRefinePlanFinishCall(CallHandles handles, ShapesRefinePlan plan)
    : ICall<RefineReport>
{
    public RefineReport Result { get; private set; }

    public readonly void Submit(CallOutputs* outputs, Completion completion)
    {
        using HandleReference index = handles.ReferenceIndex();
        using HandleReference executor = handles.ReferenceExecutor();
        NativeShapesRefinePlanFinish arguments = new()
        {
            Index = index.Pointer, Executor = executor.Pointer, Outcome = &outputs->Outcome, Completion = completion,
            Plan = plan.Cell->Handle, Report = RefineReport.In(outputs),
        };
        usearch_shapes_refine_plan_finish(&arguments);
        plan.Cell->Handle = IntPtr.Zero;
    }

    public readonly void Release(CallOutputs* outputs) { }

    public void Finish(CallOutputs* outputs)
    {
        outputs->Outcome.ThrowIfFailed();
        this.Result = RefineReport.Lift(RefineReport.In(outputs));
    }
}

/// <summary>A reachability audit; see <see cref="ShapesIndexOn.Health"/>.</summary>
internal unsafe struct ShapesHealthCall(CallHandles handles) : ICall<DenseReachability>
{
    public DenseReachability Result { get; private set; }

    public readonly void Submit(CallOutputs* outputs, Completion completion)
    {
        using HandleReference index = handles.ReferenceIndex();
        using HandleReference executor = handles.ReferenceExecutor();
        NativeShapesHealth arguments = new()
        {
            Index = index.Pointer, Executor = executor.Pointer, Outcome = &outputs->Outcome, Completion = completion,
            Reachability = (NativeDenseReachability*)outputs->Report,
        };
        usearch_shapes_health(&arguments);
    }

    public readonly void Release(CallOutputs* outputs) { }

    public void Finish(CallOutputs* outputs)
    {
        outputs->Outcome.ThrowIfFailed();
        this.Result = DenseReachability.Lift((NativeDenseReachability*)outputs->Report);
    }
}

/// <summary>A search; see <see cref="ShapesIndexOn.Search{TScalar}"/>.</summary>
internal unsafe struct ShapesSearchCall<TScalar>(
    CallHandles handles, TapeView<TScalar> queries, int wanted, Arena arena, ulong expansionFactor,
    Func<Key, bool>? predicate, DenseTraversalStats? stats, Traversal traversal, Progress? progress) : ICall<Matches>
    where TScalar : unmanaged
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
        NativeShapesSearch arguments = new()
        {
            Index = index.Pointer, Executor = executor.Pointer, Outcome = &outputs->Outcome, Completion = completion,
            Count = (nuint)queries.Count, Queries = queries.AsView(this._elements.Pointer, this._offsets.Pointer),
            Matches = (NativeMatchesView*)outputs->View, Arena = passed.Pointer, Wanted = bound,
            ExpansionFactor = (nuint)expansionFactor, Traversal = traversal,
            Predicate = this._predicate.Predicate(predicate), Progress = this._progress.Progress(progress),
            Stats = (NativeDenseTraversalStats*)outputs->Report,
        };
        usearch_shapes_search(&arguments);
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
        this.Result = new Matches(*(NativeMatchesView*)outputs->View, queries.Count, arena.Description);
    }
}

/// <summary>A reshard; see <see cref="ShapesIndexOn.Reshard"/>.</summary>
internal unsafe struct ShapesReshardCall(
    CallHandles handles, ShapesIndex[] sources, ShapesIndex[] targets, ulong seed, ulong expansionFactor,
    ulong memoryLimit, DenseTraversalStats? stats, Progress? progress, ShapesReshardPlan? plan)
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
            ShapesIndex named = position < sources.Length ? sources[position] : targets[position - sources.Length];
            this._references[position] = HandleReference.Of(named?.Handle, "index");
            this._slots[position] = this._references[position].Pointer;
        }

        this._pinnedSlots = this._slots.AsMemory().Pin();
        IntPtr* slots = (IntPtr*)this._pinnedSlots.Pointer;
        using HandleReference executor = handles.ReferenceExecutor();
        NativeShapesReshard arguments = new()
        {
            SourcesCount = (nuint)sources.Length, Sources = slots, TargetsCount = (nuint)targets.Length,
            Targets = slots + sources.Length, Executor = executor.Pointer, Outcome = &outputs->Outcome,
            Completion = completion, Report = RefineReport.In(outputs),
            Refine = new() { ExpansionFactor = (nuint)expansionFactor, MemoryLimit = (nuint)memoryLimit },
            Seed = seed, Progress = this._progress.Progress(progress),
            Stats = (NativeDenseTraversalStats*)outputs->Report,
            Plan = plan is null ? null : &plan.Cell->Handle, Phase = plan is null ? null : &plan.Cell->Phase,
        };
        usearch_shapes_reshard(&arguments);
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

/// <summary>A reshard plan's close; see <see cref="ShapesIndexOn.ReshardPlanFinish"/>.</summary>
internal unsafe struct ShapesReshardPlanFinishCall(CallHandles handles, ShapesReshardPlan plan)
    : ICall<RefineReport>
{
    public RefineReport Result { get; private set; }

    public readonly void Submit(CallOutputs* outputs, Completion completion)
    {
        using HandleReference executor = handles.ReferenceExecutor();
        NativeShapesReshardPlanFinish arguments = new()
        {
            Executor = executor.Pointer, Outcome = &outputs->Outcome, Completion = completion,
            Plan = plan.Cell->Handle, Report = RefineReport.In(outputs),
        };
        usearch_shapes_reshard_plan_finish(&arguments);
        plan.Cell->Handle = IntPtr.Zero;
    }

    public readonly void Release(CallOutputs* outputs) { }

    public void Finish(CallOutputs* outputs)
    {
        outputs->Outcome.ThrowIfFailed();
        this.Result = RefineReport.Lift(RefineReport.In(outputs));
    }
}

/// <summary>A sample; see <see cref="ShapesIndexOn.Sample"/>.</summary>
internal unsafe struct ShapesSampleCall(CallHandles handles, int wanted, Arena arena, DenseSampleConfig config)
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
        NativeShapesSample arguments = new()
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
        usearch_shapes_sample(&arguments);
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
            (int)outputs->Written, arena.Description);
    }
}

/// <summary>A checkpoint write; see <see cref="ShapesIndexOn.Save"/>.</summary>
internal unsafe struct ShapesSaveCall(CallHandles handles, string path, Progress? progress) : ICall
{
    private readonly byte[] _path = Encoding.UTF8.GetBytes(path);
    private MemoryHandle _pinnedPath;
    private Registration _progress;

    public void Submit(CallOutputs* outputs, Completion completion)
    {
        this._pinnedPath = this._path.AsMemory().Pin();
        using HandleReference index = handles.ReferenceIndex();
        using HandleReference executor = handles.ReferenceExecutor();
        NativeShapesSave arguments = new()
        {
            Index = index.Pointer, Executor = executor.Pointer, Outcome = &outputs->Outcome, Completion = completion,
            Path = new NativeBytesView { Data = (IntPtr)this._pinnedPath.Pointer, Length = (nuint)this._path.Length },
            Progress = this._progress.Progress(progress),
        };
        usearch_shapes_save(&arguments);
    }

    public void Release(CallOutputs* outputs)
    {
        this._pinnedPath.Dispose();
        this._progress.Free();
    }

    public readonly void Finish(CallOutputs* outputs) => outputs->Outcome.ThrowIfFailed();
}

/// <summary>A checkpoint write into an arena; see <see cref="ShapesIndexOn.SaveBuffer"/>.</summary>
internal unsafe struct ShapesSaveBufferCall(CallHandles handles, Arena arena, Progress? progress) : ICall<Monolith>
{
    private Registration _progress;

    public Monolith Result { get; private set; }

    public void Submit(CallOutputs* outputs, Completion completion)
    {
        using HandleReference index = handles.ReferenceIndex();
        using HandleReference executor = handles.ReferenceExecutor();
        using HandleReference passed = Arena.Reference(arena);
        NativeShapesSaveBuffer arguments = new()
        {
            Index = index.Pointer, Executor = executor.Pointer, Outcome = &outputs->Outcome, Completion = completion,
            Monolith = (NativeBytesView*)outputs->View, Arena = passed.Pointer,
            Progress = this._progress.Progress(progress),
        };
        usearch_shapes_save_buffer(&arguments);
    }

    public void Release(CallOutputs* outputs) => this._progress.Free();

    public void Finish(CallOutputs* outputs)
    {
        outputs->Outcome.ThrowIfFailed();
        this.Result = new Monolith(*(NativeBytesView*)outputs->View, arena.Description);
    }
}

#endregion Calls

#region Native Vocabulary

[StructLayout(LayoutKind.Sequential)]
internal struct NativeShapesMetadata
{
    public ShapesMetric Metric;
    public Scalar Quantization;
    public nuint Points;
    public nuint Connectivity;
}

[StructLayout(LayoutKind.Sequential)]
internal struct NativeShapesDescription
{
    public NativeShapesMetadata Metadata;
    public nuint Size;
    public nuint Capacity;
    public nuint TotalPoints;
    public nuint MemoryUsage;
    public int CpuClass;
    public int GpuClass;
    public nuint ShardsCount;
    public int LoadPath;
}

[StructLayout(LayoutKind.Sequential)]
internal struct NativeShapesKeyStats
{
    public nuint Entries;
    public nuint TotalPoints;
    public nuint LargestPoints;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeShapesInitRam
{
    public IntPtr* Index;
    public NativeOutcome* Outcome;
    public ShapesMetric Metric;
    public Scalar Quantization;
    public nuint Points;
    public nuint Connectivity;
    public nuint ValuesPerKey;
    public NativeRamOptions Ram;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeShapesInitVram
{
    public IntPtr* Index;
    public NativeOutcome* Outcome;
    public ShapesMetric Metric;
    public Scalar Quantization;
    public nuint Points;
    public nuint Connectivity;
    public nuint ValuesPerKey;
    public NativeVramOptions Vram;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeShapesLoadRam
{
    public IntPtr* Index;
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
    public NativeBytesView Path;
    public NativeRamOptions Ram;
    public nuint Connectivity;
    public nuint ValuesPerKey;
    public NativeProgress Progress;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeShapesLoadRamBuffer
{
    public IntPtr* Index;
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
    public NativeBytesView Buffer;
    public NativeRamOptions Ram;
    public nuint Connectivity;
    public nuint ValuesPerKey;
    public NativeProgress Progress;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeShapesLoadVram
{
    public IntPtr* Index;
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
    public NativeBytesView Path;
    public NativeVramOptions Vram;
    public nuint Connectivity;
    public nuint ValuesPerKey;
    public NativeProgress Progress;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeShapesLoadVramBuffer
{
    public IntPtr* Index;
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
    public NativeBytesView Buffer;
    public NativeVramOptions Vram;
    public nuint Connectivity;
    public nuint ValuesPerKey;
    public NativeProgress Progress;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeShapesViewRam
{
    public IntPtr* Index;
    public NativeOutcome* Outcome;
    public NativeBytesView Path;
    public NativeRamOptions Ram;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeShapesViewNvme
{
    public IntPtr* Index;
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
    public NativeBytesView Path;
    public NativeNvmeOptions Nvme;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeShapesReserve
{
    public IntPtr Index;
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
    public nuint Entries;
    public nuint Points;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeShapesClear
{
    public IntPtr Index;
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeShapesInsert
{
    public IntPtr Index;
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
    public nuint Count;
    public NativeKeysView Keys;
    public NativeTapeView Shapes;
    public nuint ExpansionFactor;
    public NativeProgress Progress;
    public NativeDenseTraversalStats* Stats;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeShapesRefine
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
internal unsafe struct NativeShapesRefinePlanFinish
{
    public IntPtr Index;
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
    public IntPtr Plan;
    public NativeRefineReport* Report;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeShapesHealth
{
    public IntPtr Index;
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
    public NativeDenseReachability* Reachability;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeShapesSearch
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
    public NativeMatchesView* Hints;
    public NativeProgress Progress;
    public NativeDenseTraversalStats* Stats;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeShapesCount
{
    public IntPtr Index;
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
    public Key Key;
    public NativeShapesKeyStats* KeyStats;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeShapesGet
{
    public IntPtr Index;
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
    public Key Key;
    public NativeTapeView* Shapes;
    public nuint* Written;
    public IntPtr Arena;
    public Scalar OutputsScalar;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeShapesReshard
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
internal unsafe struct NativeShapesReshardPlanFinish
{
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
    public IntPtr Plan;
    public NativeRefineReport* Report;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeShapesCursorInit
{
    public IntPtr Index;
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public nuint StartsCount;
    public Key* Starts;
    public IntPtr* Cursor;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeShapesCursorNext
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
internal unsafe struct NativeShapesSample
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
internal unsafe struct NativeShapesSave
{
    public IntPtr Index;
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
    public NativeBytesView Path;
    public NativeProgress Progress;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeShapesSaveBuffer
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
internal unsafe struct NativeShapesMetadataRead
{
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
    public NativeBytesView Path;
    public NativeShapesMetadata* Metadata;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeShapesMetadataReadBuffer
{
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
    public NativeBytesView Buffer;
    public NativeShapesMetadata* Metadata;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeShapesDescribe
{
    public IntPtr Index;
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
    public NativeShapesDescription* Description;
}

#endregion Native Vocabulary

#region Entry Points

/// <summary>The shapes modality's entry points, each taking one argument struct.</summary>
internal static unsafe partial class NativeMethods
{
    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial IntPtr usearch_shapes_metric_name(ShapesMetric metric);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial Status usearch_shapes_metric_from_name(NativeBytesView name, out ShapesMetric metric);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_shapes_init_ram(NativeShapesInitRam* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_shapes_init_vram(NativeShapesInitVram* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_shapes_load_ram(NativeShapesLoadRam* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_shapes_load_ram_buffer(NativeShapesLoadRamBuffer* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_shapes_load_vram(NativeShapesLoadVram* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_shapes_load_vram_buffer(NativeShapesLoadVramBuffer* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_shapes_view_ram(NativeShapesViewRam* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_shapes_view_nvme(NativeShapesViewNvme* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_shapes_free(IntPtr index);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_shapes_reserve(NativeShapesReserve* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_shapes_clear(NativeShapesClear* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_shapes_insert(NativeShapesInsert* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_shapes_refine(NativeShapesRefine* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_shapes_refine_plan_finish(NativeShapesRefinePlanFinish* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_shapes_health(NativeShapesHealth* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_shapes_search(NativeShapesSearch* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_shapes_count(NativeShapesCount* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_shapes_get(NativeShapesGet* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_shapes_reshard(NativeShapesReshard* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_shapes_reshard_plan_finish(NativeShapesReshardPlanFinish* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_shapes_cursor_init(NativeShapesCursorInit* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_shapes_cursor_next(NativeShapesCursorNext* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_shapes_cursor_free(IntPtr cursor);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_shapes_sample(NativeShapesSample* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_shapes_save(NativeShapesSave* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_shapes_save_buffer(NativeShapesSaveBuffer* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_shapes_metadata_read(NativeShapesMetadataRead* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_shapes_metadata_read_buffer(NativeShapesMetadataReadBuffer* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_shapes_describe(NativeShapesDescribe* arguments);
}

#endregion Entry Points

#region Handles

/// <summary>A <c>usearch_shapes_index_t</c>, freed when the wrapper is disposed or collected.</summary>
internal sealed class NativeShapesIndexHandle : EngineHandle
{
    /// <inheritdoc/>
    protected override bool ReleaseHandle()
    {
        NativeMethods.usearch_shapes_free(this.handle);
        return true;
    }
}

/// <summary>A <c>usearch_shapes_cursor_t</c> and the reference on the index every pull names, both
/// released when the wrapper is disposed or collected.</summary>
internal sealed class NativeShapesCursorHandle : EngineHandle
{
    private HandleReference _index;

    internal NativeShapesCursorHandle(IntPtr cursor, HandleReference index)
    {
        this.SetHandle(cursor);
        this._index = index;
    }

    /// <summary>The index the cursor was opened on.</summary>
    internal IntPtr Index => this._index.Pointer;

    /// <inheritdoc/>
    protected override bool ReleaseHandle()
    {
        NativeMethods.usearch_shapes_cursor_free(this.handle);
        this._index.Dispose();
        return true;
    }
}

#endregion Handles
