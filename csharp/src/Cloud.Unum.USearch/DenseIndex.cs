/**
 *  @file csharp/src/Cloud.Unum.USearch/DenseIndex.cs
 *  @author Ash Vardanian
 *  @date September 8, 2026
 *  @brief The dense index handle, its metric and scalar kinds, and the verbs that run over it.
 */
using System;
using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.ExceptionServices;
using System.Runtime.InteropServices;
using System.Threading;

using static Cloud.Unum.USearch.NativeMethods;

namespace Cloud.Unum.USearch;

#region Options

/// <summary>
/// A distance kind supported out of the box, with SIMD kernels dispatched by the machine's
/// capabilities. Mirrors <c>usearch_dense_metric_t</c>.
/// </summary>
/// <remarks>
/// All of them derive from a single inner-product kernel with pre-computed magnitudes; Hamming
/// and Jaccard are bit-level, over the packed sub-byte kinds. The C layer's custom-callback
/// escape hatch stays unexposed here: a caller-supplied metric forfeits the SIMD kernel chains
/// and the magnitude cache both.
/// </remarks>
public enum DenseMetric : int
{
    /// <summary>A metric the library could not name.</summary>
    Unknown = 0,
    /// <summary>The inner product - the only kind the linear-algebra verbs accept.</summary>
    Inner = 1,
    /// <summary>The cosine distance.</summary>
    Angular = 2,
    /// <summary>The L2 distance.</summary>
    Euclidean = 3,
    /// <summary>The count of differing bits.</summary>
    Hamming = 4,
    /// <summary>The bit-level Jaccard distance.</summary>
    Jaccard = 5,
}

/// <summary>The dense spellings of <see cref="Names"/>.</summary>
public static partial class Names
{
    /// <summary>Spells a metric kind, as <see cref="DenseMetric.Inner"/> spells "inner". Calls
    /// <c>usearch_dense_metric_name</c>.</summary>
    /// <param name="metric">The metric kind to spell.</param>
    /// <returns>The canonical spelling.</returns>
    public static string Name(this DenseMetric metric) => Spelled(usearch_dense_metric_name(metric));

    /// <summary>Reads a metric kind back from its spelling. Calls <c>usearch_dense_metric_from_name</c>.</summary>
    /// <param name="name">The canonical spelling.</param>
    /// <param name="metric">The metric kind it spells.</param>
    /// <returns>True when the spelling names a metric kind, false otherwise.</returns>
    public static bool TryParse(string name, out DenseMetric metric)
    {
        metric = Read(name, static view => usearch_dense_metric_from_name(view));
        return metric.Name() == name;
    }
}

/// <summary>
/// The construction options of a dense index, fixed once and read by the handle that is built from
/// them - the fields <c>usearch_dense_init_ram_t</c> and <c>usearch_dense_init_vram_t</c> share.
/// </summary>
/// <remarks>
/// Stored vectors always round up to the SIMD step quantum, so they stay cache-line aligned and every
/// kernel runs tail-free; the index is slightly larger for it, and that is not a knob.
/// <para>
/// Every field is init-only, so a variant comes from <c>DenseOptions.Default with { ... }</c>
/// rather than from mutating a value another handle may already have been built from.
/// </para>
/// </remarks>
public readonly struct DenseOptions
{
    /// <summary>The width of the indexed vectors, which init freezes; zero is refused with
    /// <see cref="Status.ConfigMismatch"/>.</summary>
    public ulong Dimensions { get; init; }

    /// <summary>The distance kind used between vectors.</summary>
    public DenseMetric Metric { get; init; }

    /// <summary>The scalar type vectors are stored and quantized into.</summary>
    public Scalar Quantization { get; init; }

    /// <summary>The connections-per-node bound of the graph; zero derives.</summary>
    public ulong Connectivity { get; init; }

    /// <summary>The upper bound on entries sharing one key; zero is unbounded.</summary>
    public ulong ValuesPerKey { get; init; }

    /// <summary>An angular <see cref="Scalar.BF16"/> index, with every optional field derived.</summary>
    public static DenseOptions Default => new()
    {
        Metric = DenseMetric.Angular,
        Quantization = Scalar.BF16,
    };
}

/// <summary>
/// The geometry and metric a checkpoint froze - everything a reader learns without opening it, and
/// nothing a reader must choose for itself. Mirrors <c>usearch_dense_metadata_t</c>.
/// </summary>
/// <remarks>
/// A checkpoint records neither the residency nor the upper bound on entries sharing one key: the
/// first says where it was written rather than where a reader must keep it, and the second bounds a
/// live handle's mutable state rather than the serialized form. Both stay the caller's to choose in
/// the call that builds a handle from it.
/// </remarks>
public readonly struct DenseMetadata
{
    /// <summary>The scalars per stored vector.</summary>
    public ulong Dimensions { get; init; }

    /// <summary>The distance kind the checkpoint scores with.</summary>
    public DenseMetric Metric { get; init; }

    /// <summary>The scalar type its vectors are stored and quantized into.</summary>
    public Scalar Quantization { get; init; }

    /// <summary>The connections-per-node bound the graph was built with.</summary>
    public ulong Connectivity { get; init; }

    /// <summary>Lifts the geometry and metric a checkpoint's headers describe.</summary>
    /// <param name="lowered">The C struct the library wrote.</param>
    /// <returns>The managed metadata.</returns>
    internal static DenseMetadata Lift(in NativeDenseMetadata lowered) => new()
    {
        Dimensions = lowered.Dimensions,
        Metric = lowered.Metric,
        Quantization = lowered.Quantization,
        Connectivity = lowered.Connectivity,
    };
}

/// <summary>
/// A live index's frozen metadata and current counters, read in one call. Each counter is read
/// once, so under concurrent insertions they may come from slightly different moments. Mirrors
/// <c>usearch_dense_index_description_t</c>.
/// </summary>
public readonly struct DenseIndexDescription
{
    /// <summary>The frozen metric, quantization, dimensions and connectivity.</summary>
    public DenseMetadata Metadata { get; init; }

    /// <summary>The stored entries.</summary>
    public ulong Size { get; init; }

    /// <summary>The entries hosted without growing again.</summary>
    public ulong Capacity { get; init; }

    /// <summary>The bytes the handle holds; traversal scratch is the executor's.</summary>
    public ulong MemoryUsage { get; init; }

    /// <summary>The CPU class dispatch picked on this machine, as <c>usearch_cpu_class_name</c> spells it.</summary>
    public string CpuClass { get; init; }

    /// <summary>The GPU class dispatch picked on the index's devices; "unknown" for a host index.</summary>
    public string GpuClass { get; init; }

    /// <summary>The live shards.</summary>
    public ulong ShardsCount { get; init; }
}

/// <summary>
/// One reachability audit: the population it judged, and how much of it a search reaches. Mirrors
/// <c>usearch_reachability_t</c>.
/// </summary>
public readonly struct DenseReachability
{
    /// <summary>The entries published when the audit started; compare <see cref="Reachable"/> against
    /// this, never a later size.</summary>
    public ulong Audited { get; init; }

    /// <summary>Of those, the entries a search reaches; the rest no search finds until a refinement
    /// re-links them.</summary>
    public ulong Reachable { get; init; }

    /// <summary>The levels the sweep took to converge; the deepest shard's on a
    /// sharded handle.</summary>
    public ulong Sweeps { get; init; }

    /// <summary>The levels the sweep back toward the entry point took, where an audit sweeps both
    /// ways.</summary>
    public ulong BackSweeps { get; init; }
}

/// <summary>
/// What one refine-family call did to the graph: <see cref="DenseIndex.Refine"/>,
/// <see cref="DenseIndex.Absorb"/> and <see cref="DenseIndex.Split"/> land it. Mirrors
/// <c>usearch_refine_report_t</c>.
/// </summary>
public readonly struct RefineReport
{
    /// <summary>The entries whose neighbors were selected again, the islands of every heal round
    /// included.</summary>
    public ulong Reselected { get; init; }

    /// <summary>Of those, the entries whose neighbor set gained a member it lacked; zero means the
    /// graph has settled.</summary>
    public ulong Changed { get; init; }

    /// <summary>The entries the first audit found not reached both ways.</summary>
    public ulong IslandsFound { get; init; }

    /// <summary>The links written between an island and an entry reached both ways.</summary>
    public ulong IslandsReconnected { get; init; }

    /// <summary>The audit closing the heal, as a fresh <see cref="DenseIndex.Health"/> right after
    /// would take it.</summary>
    public DenseReachability Reachability { get; init; }

    internal static unsafe RefineReport Lift(NativeRefineReport* report) => new()
    {
        Reselected = report->Reselected, Changed = report->Changed, IslandsFound = report->IslandsFound,
        IslandsReconnected = report->IslandsReconnected,
        Reachability = new DenseReachability
        {
            Audited = report->Reachability.Audited, Reachable = report->Reachability.Reachable,
            Sweeps = report->Reachability.Sweeps, BackSweeps = report->Reachability.BackSweeps,
        },
    };

    /// <summary>Where a refine-family call writes its report, past the traversal telemetry.</summary>
    internal static unsafe NativeRefineReport* In(CallOutputs* outputs) =>
        (NativeRefineReport*)((byte*)outputs->Report + sizeof(NativeDenseTraversalStats));
}

/// <summary>
/// The online-softmax carry of one query row - Flash Attention's running maximum and partition,
/// which folds one attention call into another over the same queries. Mirrors
/// <c>usearch_dense_softmax_state_t</c>.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct DenseSoftmaxState
{
    /// <summary>The largest logit seen so far - the recurrence's <c>m</c>.</summary>
    public float MaximumScore;

    /// <summary>The sum of exponentials taken against that maximum - the recurrence's <c>l</c>.</summary>
    public float Partition;
}

#endregion Options

#region Telemetry

/// <summary>
/// The traversal telemetry one call reports. Every call overwrites it rather than accumulating,
/// so a sweep over many batches sums them itself with <see cref="Accumulate"/>. Mirrors
/// <c>usearch_traversal_stats_t</c>.
/// </summary>
public sealed class DenseTraversalStats
{
    /// <summary>The graph members the call visited.</summary>
    public ulong VisitedMembers { get; internal set; }

    /// <summary>The distances the call computed, diversity re-selections included.</summary>
    public ulong ComputedDistances { get; internal set; }

    /// <summary>The starvation re-seedings - a graph-connectivity health signal rather than a
    /// cost counter.</summary>
    public ulong Restarts { get; internal set; }

    /// <summary>Folds another call's telemetry into this one. Binding-only.</summary>
    /// <param name="other">The telemetry to add.</param>
    public void Accumulate(DenseTraversalStats other)
    {
        this.VisitedMembers += other.VisitedMembers;
        this.ComputedDistances += other.ComputedDistances;
        this.Restarts += other.Restarts;
    }

    /// <summary>Takes over what one call reported, overwriting rather than accumulating.</summary>
    /// <param name="reported">The telemetry the library wrote.</param>
    internal void Adopt(in NativeDenseTraversalStats reported)
    {
        this.VisitedMembers = reported.VisitedMembers;
        this.ComputedDistances = reported.ComputedDistances;
        this.Restarts = reported.Restarts;
    }
}

#endregion Telemetry

#region Index

/// <summary>
/// A Vector Search Index over equidimensional dense representations.
/// </summary>
/// <remarks>
/// Every verb is a batch over a <see cref="RowsView"/>, so a single vector is a one-row matrix. The
/// breadth of the candidates rides each call rather than the handle, which lets a recall sweep run
/// against one immutable index with no mutation between points.
/// <para>
/// The receiver chooses where a verb runs, and the verb stays the same: called on the index it runs
/// on the caller's own thread, on <see cref="On"/> it blocks on that executor, and on
/// <see cref="Deferred"/> it submits there and answers a <see cref="Deferred{TResult}"/>. The
/// executor's kind must match the residency of the handle.
/// </para>
/// <para>
/// A handle may be shared across threads, and no method here takes a lock: every output whose length
/// scales with the call lands in an <see cref="Arena"/> the caller lends. On a RAM handle,
/// insertions and searches overlap freely: searches read only entries published behind the
/// visibility watermark and never write to the graph, while insertions synchronize through
/// per-node locks and publish in order. Serialization is the one exception, demanding a
/// mutation-free window, though concurrent readers keep searching throughout.
/// </para>
/// </remarks>
public sealed unsafe class DenseIndex : IDisposable
{
    private readonly NativeDenseIndexHandle _handle;

    private DenseIndex(NativeDenseIndexHandle handle) => this._handle = handle;

    /// <summary>The engine handle the verbs lease.</summary>
    internal NativeDenseIndexHandle Handle => this._handle;

    /// <summary>Creates an empty, mutable index in host memory, one shard per named NUMA node. Calls
    /// <c>usearch_dense_init_ram</c>.</summary>
    /// <param name="options">The geometry and metric the index freezes.</param>
    /// <param name="memoryDomains">NUMA node IDs to place one shard on apiece; empty derives every
    /// node the process may allocate on.</param>
    /// <param name="cpuClass">The class the index dispatches as; null derives the CPU's own.</param>
    /// <returns>The new index.</returns>
    public static DenseIndex InitRam(
        DenseOptions options, ReadOnlySpan<int> memoryDomains = default, CPUClass? cpuClass = null)
    {
        IntPtr created = IntPtr.Zero;
        NativeOutcome outcome = default;
        fixed (int* domains = memoryDomains)
        {
            NativeDenseInitRam arguments = new()
            {
                Index = &created,
                Outcome = &outcome,
                Metric = options.Metric,
                Quantization = options.Quantization,
                Dimensions = (nuint)options.Dimensions,
                Connectivity = (nuint)options.Connectivity,
                ValuesPerKey = (nuint)options.ValuesPerKey,
                Ram = new NativeRamOptions
                {
                    MemoryDomains = (IntPtr)domains, MemoryDomainsCount = (nuint)memoryDomains.Length,
                    CpuClass = (int)(cpuClass ?? 0),
                },
            };
            usearch_dense_init_ram(&arguments);
        }

        return new DenseIndex(EngineHandle.Created<NativeDenseIndexHandle>(created, outcome));
    }

    /// <summary>Creates an empty, mutable index in device memory, one shard per named device, sharded
    /// and never replicated. Calls <c>usearch_dense_init_vram</c>.</summary>
    /// <param name="options">The geometry and metric the index freezes.</param>
    /// <param name="devices">CUDA device ordinals to place one shard on apiece; empty derives one free
    /// device.</param>
    /// <param name="gpuClass">The class the index dispatches as; null derives the devices' own.</param>
    /// <returns>The new index.</returns>
    public static DenseIndex InitVram(
        DenseOptions options, ReadOnlySpan<int> devices = default, GPUClass? gpuClass = null)
    {
        IntPtr created = IntPtr.Zero;
        NativeOutcome outcome = default;
        fixed (int* ordinals = devices)
        {
            NativeDenseInitVram arguments = new()
            {
                Index = &created,
                Outcome = &outcome,
                Metric = options.Metric,
                Quantization = options.Quantization,
                Dimensions = (nuint)options.Dimensions,
                Connectivity = (nuint)options.Connectivity,
                ValuesPerKey = (nuint)options.ValuesPerKey,
                Vram = new NativeVramOptions
                {
                    Devices = (IntPtr)ordinals, DevicesCount = (nuint)devices.Length, GpuClass = (int)(gpuClass ?? 0),
                },
            };
            usearch_dense_init_vram(&arguments);
        }

        return new DenseIndex(EngineHandle.Created<NativeDenseIndexHandle>(created, outcome));
    }

    /// <summary>
    /// Copies a checkpoint into a new, mutable index in host memory. Point the path at a
    /// <c>.safegraphs.json</c> manifest or a single <c>.safegraphs</c> monolith; the geometry and
    /// metric come from the checkpoint's own headers. Calls <c>usearch_dense_load_ram</c>.
    /// </summary>
    /// <param name="path">The checkpoint to read.</param>
    /// <param name="memoryDomains">Where the parts land, part <c>p</c> on shard <c>p</c> modulo the
    /// shard count; empty derives every node.</param>
    /// <param name="valuesPerKey">The upper bound on entries sharing one key, which no checkpoint
    /// records; zero is unbounded.</param>
    /// <param name="executor">The workers the read fans out over; null reads on the caller's thread.</param>
    /// <param name="cpuClass">The class the index dispatches as; null derives the CPU's own.</param>
    /// <returns>The new index.</returns>
    public static DenseIndex LoadRam(
        string path, ReadOnlySpan<int> memoryDomains = default, ulong valuesPerKey = 0, Executor? executor = null,
        CPUClass? cpuClass = null)
    {
        byte[] encoded = Paths.Encode(path);
        using Lease pool = Lease.Of(executor?.Handle, "executor");
        IntPtr created = IntPtr.Zero;
        NativeOutcome outcome = default;
        fixed (byte* data = encoded)
        fixed (int* domains = memoryDomains)
        {
            NativeDenseLoadRam arguments = new()
            {
                Index = &created,
                Executor = pool.Pointer,
                Outcome = &outcome,
                Path = Borrow(data, encoded.Length),
                Ram = new NativeRamOptions
                {
                    MemoryDomains = (IntPtr)domains, MemoryDomainsCount = (nuint)memoryDomains.Length,
                    CpuClass = (int)(cpuClass ?? 0),
                },
                ValuesPerKey = (nuint)valuesPerKey,
            };
            usearch_dense_load_ram(&arguments);
        }

        return new DenseIndex(EngineHandle.Created<NativeDenseIndexHandle>(created, outcome));
    }

    /// <summary>Copies a checkpoint into a new, mutable index in device memory, placing its parts as
    /// <see cref="LoadRam"/> does. Calls <c>usearch_dense_load_vram</c>.</summary>
    /// <param name="path">The checkpoint to read.</param>
    /// <param name="devices">Where the parts land, part <c>p</c> on device <c>p</c> modulo their
    /// count; empty derives one free device.</param>
    /// <param name="valuesPerKey">The upper bound on entries sharing one key; zero is unbounded.</param>
    /// <param name="executor">The workers the upload runs on; null uploads on the caller's thread.</param>
    /// <param name="gpuClass">The class the index dispatches as; null derives the devices' own.</param>
    /// <returns>The new index.</returns>
    public static DenseIndex LoadVram(
        string path, ReadOnlySpan<int> devices = default, ulong valuesPerKey = 0, Executor? executor = null,
        GPUClass? gpuClass = null)
    {
        byte[] encoded = Paths.Encode(path);
        using Lease pool = Lease.Of(executor?.Handle, "executor");
        IntPtr created = IntPtr.Zero;
        NativeOutcome outcome = default;
        fixed (byte* data = encoded)
        fixed (int* ordinals = devices)
        {
            NativeDenseLoadVram arguments = new()
            {
                Index = &created,
                Executor = pool.Pointer,
                Outcome = &outcome,
                Path = Borrow(data, encoded.Length),
                Vram = new NativeVramOptions
                {
                    Devices = (IntPtr)ordinals, DevicesCount = (nuint)devices.Length, GpuClass = (int)(gpuClass ?? 0),
                },
                ValuesPerKey = (nuint)valuesPerKey,
            };
            usearch_dense_load_vram(&arguments);
        }

        return new DenseIndex(EngineHandle.Created<NativeDenseIndexHandle>(created, outcome));
    }

    /// <summary>Copies a monolith from a buffer into a new, mutable index in host memory, as
    /// <see cref="LoadRam"/> does a path. Calls <c>usearch_dense_load_ram_buffer</c>.</summary>
    /// <param name="buffer">The monolith's bytes, copied in before the call returns.</param>
    /// <param name="memoryDomains">The NUMA nodes to place shards on; the monolith lands whole on the first.</param>
    /// <param name="valuesPerKey">The upper bound on entries sharing one key; zero is unbounded.</param>
    /// <param name="executor">The workers the copy fans out over; null copies on the caller's thread.</param>
    /// <param name="cpuClass">The class the index dispatches as; null derives the CPU's own.</param>
    /// <returns>The new index.</returns>
    public static DenseIndex LoadRamBuffer(
        ReadOnlySpan<byte> buffer, ReadOnlySpan<int> memoryDomains = default, ulong valuesPerKey = 0,
        Executor? executor = null, CPUClass? cpuClass = null)
    {
        using Lease pool = Lease.Of(executor?.Handle, "executor");
        IntPtr created = IntPtr.Zero;
        NativeOutcome outcome = default;
        fixed (byte* source = buffer)
        fixed (int* domains = memoryDomains)
        {
            NativeDenseLoadRamBuffer arguments = new()
            {
                Index = &created,
                Executor = pool.Pointer,
                Outcome = &outcome,
                Buffer = Borrow(source, buffer.Length),
                Ram = new NativeRamOptions
                {
                    MemoryDomains = (IntPtr)domains, MemoryDomainsCount = (nuint)memoryDomains.Length,
                    CpuClass = (int)(cpuClass ?? 0),
                },
                ValuesPerKey = (nuint)valuesPerKey,
            };
            usearch_dense_load_ram_buffer(&arguments);
        }

        return new DenseIndex(EngineHandle.Created<NativeDenseIndexHandle>(created, outcome));
    }

    /// <summary>Copies a monolith from a buffer into a new, mutable index in device memory. Calls
    /// <c>usearch_dense_load_vram_buffer</c>.</summary>
    /// <param name="buffer">The monolith's bytes, in host or device memory.</param>
    /// <param name="devices">The devices to place shards on; the monolith lands whole on the first.</param>
    /// <param name="valuesPerKey">The upper bound on entries sharing one key; zero is unbounded.</param>
    /// <param name="executor">The workers the upload runs on; null uploads on the caller's thread.</param>
    /// <param name="gpuClass">The class the index dispatches as; null derives the devices' own.</param>
    /// <returns>The new index.</returns>
    public static DenseIndex LoadVramBuffer(
        ReadOnlySpan<byte> buffer, ReadOnlySpan<int> devices = default, ulong valuesPerKey = 0,
        Executor? executor = null, GPUClass? gpuClass = null)
    {
        using Lease pool = Lease.Of(executor?.Handle, "executor");
        IntPtr created = IntPtr.Zero;
        NativeOutcome outcome = default;
        fixed (byte* source = buffer)
        fixed (int* ordinals = devices)
        {
            NativeDenseLoadVramBuffer arguments = new()
            {
                Index = &created,
                Executor = pool.Pointer,
                Outcome = &outcome,
                Buffer = Borrow(source, buffer.Length),
                Vram = new NativeVramOptions
                {
                    Devices = (IntPtr)ordinals, DevicesCount = (nuint)devices.Length, GpuClass = (int)(gpuClass ?? 0),
                },
                ValuesPerKey = (nuint)valuesPerKey,
            };
            usearch_dense_load_vram_buffer(&arguments);
        }

        return new DenseIndex(EngineHandle.Created<NativeDenseIndexHandle>(created, outcome));
    }

    /// <summary>
    /// Serves a checkpoint in place through a memory mapping, as a read-only index whose mutating
    /// verbs report <see cref="Status.ReadOnly"/>; the files must stay unmodified while it lives.
    /// Calls <c>usearch_dense_view_ram</c>.
    /// </summary>
    /// <param name="path">The checkpoint to serve.</param>
    /// <returns>The read-only index.</returns>
    public static DenseIndex ViewRam(string path)
    {
        byte[] encoded = Paths.Encode(path);
        IntPtr created = IntPtr.Zero;
        NativeOutcome outcome = default;
        fixed (byte* data = encoded)
        {
            NativeDenseViewRam arguments = new()
            {
                Index = &created, Outcome = &outcome, Path = Borrow(data, encoded.Length),
            };
            usearch_dense_view_ram(&arguments);
        }

        return new DenseIndex(EngineHandle.Created<NativeDenseIndexHandle>(created, outcome));
    }

    /// <summary>
    /// Serves a checkpoint in place from NVMe drives, read through the executor's submission rings,
    /// as a read-only index. External-memory graphs are never modified in place, so every mutating
    /// verb fails on it with <see cref="Status.ReadOnly"/>. Calls <c>usearch_dense_view_nvme</c>.
    /// </summary>
    /// <remarks>Until NVMe checkpoints carry keys, the engine answers <see cref="Status.Unsupported"/>.</remarks>
    /// <param name="path">The checkpoint to serve.</param>
    /// <param name="paths">The devices the parts are served from; empty serves each where its
    /// manifest points.</param>
    /// <param name="executor">The NVMe executor whose rings read the headers; null reads on the
    /// caller's thread.</param>
    /// <param name="cpuClass">The class the index dispatches as; null derives the CPU's own.</param>
    /// <returns>The read-only index.</returns>
    public static DenseIndex ViewNvme(
        string path, ReadOnlySpan<string> paths = default, Executor? executor = null, CPUClass? cpuClass = null)
    {
        byte[] encoded = Paths.Encode(path);
        using Lease pool = Lease.Of(executor?.Handle, "executor");
        IntPtr created = IntPtr.Zero;
        NativeOutcome outcome = default;
        NativeBytesView* views = Paths.Encode(paths);
        try
        {
            fixed (byte* data = encoded)
            {
                NativeDenseViewNvme arguments = new()
                {
                    Index = &created,
                    Executor = pool.Pointer,
                    Outcome = &outcome,
                    Path = Borrow(data, encoded.Length),
                    Nvme = new NativeNvmeOptions
                    {
                        Paths = (IntPtr)views, PathsCount = (nuint)paths.Length, CpuClass = (int)(cpuClass ?? 0),
                    },
                };
                usearch_dense_view_nvme(&arguments);
            }
        }
        finally
        {
            NativeMemory.Free(views);
        }

        return new DenseIndex(EngineHandle.Created<NativeDenseIndexHandle>(created, outcome));
    }

    /// <summary>
    /// Reads a checkpoint's geometry and metric metadata without loading it - from the manifest's
    /// summary when the path names one, so no shard is opened. Calls <c>usearch_dense_metadata_read</c>.
    /// </summary>
    /// <param name="path">The checkpoint path.</param>
    /// <param name="executor">The workers the read runs on; null reads on the caller's thread.</param>
    /// <returns>The geometry and metric the checkpoint's headers describe.</returns>
    public static DenseMetadata MetadataRead(string path, Executor? executor = null)
    {
        byte[] encoded = Paths.Encode(path);
        using Lease pool = Lease.Of(executor?.Handle, "executor");
        NativeDenseMetadata lowered = default;
        NativeOutcome outcome = default;
        fixed (byte* data = encoded)
        {
            NativeDenseMetadataRead arguments = new()
            {
                Executor = pool.Pointer,
                Outcome = &outcome, Path = Borrow(data, encoded.Length), Metadata = &lowered,
            };
            usearch_dense_metadata_read(&arguments);
        }

        outcome.ThrowIfFailed();
        return DenseMetadata.Lift(lowered);
    }

    /// <summary>Reads geometry and metric metadata from a serialized monolith buffer. Calls
    /// <c>usearch_dense_metadata_read_buffer</c>.</summary>
    /// <param name="buffer">The monolith's bytes.</param>
    /// <param name="executor">The workers the read runs on; null reads on the caller's thread.</param>
    /// <returns>The geometry and metric the monolith's header describes.</returns>
    public static DenseMetadata MetadataReadBuffer(ReadOnlySpan<byte> buffer, Executor? executor = null)
    {
        using Lease pool = Lease.Of(executor?.Handle, "executor");
        NativeDenseMetadata lowered = default;
        NativeOutcome outcome = default;
        fixed (byte* source = buffer)
        {
            NativeDenseMetadataReadBuffer arguments = new()
            {
                Executor = pool.Pointer,
                Outcome = &outcome, Buffer = Borrow(source, buffer.Length), Metadata = &lowered,
            };
            usearch_dense_metadata_read_buffer(&arguments);
        }

        outcome.ThrowIfFailed();
        return DenseMetadata.Lift(lowered);
    }

    /// <summary>The receiver whose verbs block on <paramref name="executor"/>, after every call
    /// submitted there before them. Binding-only.</summary>
    /// <param name="executor">The executor the verbs run on.</param>
    /// <returns>The receiver, holding nothing but the two references.</returns>
    public DenseIndexOn On(Executor executor) => new(this, executor);

    /// <summary>The receiver whose verbs submit their calls to <paramref name="executor"/> and answer a
    /// <see cref="Deferred{TResult}"/> at once. Binding-only.</summary>
    /// <param name="executor">The executor the calls are submitted to.</param>
    /// <returns>The receiver, holding nothing but the two references.</returns>
    public DenseIndexDeferredOn Deferred(Executor executor) => new(this, executor);

    /// <summary>Claims capacity for incoming vectors, counting the entries already stored - a hint,
    /// since insertions grow the index on demand. Calls <c>usearch_dense_reserve</c>.</summary>
    /// <param name="capacity">The desired total capacity.</param>
    public void Reserve(ulong capacity) => Calls.Block(new DenseReserveCall(new(this._handle, null), capacity));

    /// <summary>Erases all entries, keeping the reserved capacity and the frozen geometry; no other
    /// call on the handle may overlap it. Calls <c>usearch_dense_clear</c>.</summary>
    public void Clear() => Calls.Block(new DenseClearCall(new(this._handle, null)));

    /// <summary>Reports the frozen metadata and the current counters in one call. Calls
    /// <c>usearch_dense_describe</c>.</summary>
    /// <returns>The description.</returns>
    public DenseIndexDescription Describe() => Calls.Block(new DenseDescribeCall(new(this._handle, null))).Result;

    /// <summary>Counts the entries stored under a key; zero means the index holds none. Calls
    /// <c>usearch_dense_count</c>.</summary>
    /// <param name="key">The identity to look for.</param>
    /// <returns>How many entries share that key.</returns>
    public ulong Count(Key key) => Calls.Block(new DenseCountCall(new(this._handle, null), key)).Result;

    /// <summary>Reports whether the index holds at least one entry under a key. Calls
    /// <c>usearch_dense_count</c>.</summary>
    /// <param name="key">The identity to look for.</param>
    /// <returns>True when the index holds one or more.</returns>
    public bool Contains(Key key) => this.Count(key) != 0;

    /// <summary>
    /// Retrieves every vector stored under a key, one row per entry sharing the key, in insertion
    /// order, converted to the kind asked for and as wide as the library reports. Calls
    /// <c>usearch_dense_get</c>.
    /// </summary>
    /// <param name="key">The identity to look for.</param>
    /// <param name="arena">Where the rows land, rewound by the call.</param>
    /// <param name="outputsScalar">The scalar type to export as.</param>
    /// <returns>The rows, none when the key is absent, readable until the arena's next call.</returns>
    public DenseVectors Get(Key key, Arena arena, Scalar outputsScalar = Scalar.F32) =>
        Calls.Block(new DenseGetCall(new(this._handle, null), key, arena, outputsScalar)).Result;

    /// <summary>
    /// Inserts a matrix of vectors under their keys - the native batch operation, which overlaps
    /// safely with concurrent searches on the same handle. Calls <c>usearch_dense_insert</c>.
    /// </summary>
    /// <remarks>
    /// The input scalar type is independent of the stored quantization; the library converts on
    /// ingest. A batch is all-or-nothing: every fallible step runs before the first entry identity
    /// is claimed, so a rejected batch publishes nothing.
    /// </remarks>
    /// <param name="keys">One identity per row.</param>
    /// <param name="vectors">The rows to admit.</param>
    /// <param name="expansionFactor">The breadth of the candidates the build walk explores; zero derives
    /// it from the geometry.</param>
    /// <param name="stats">The traversal telemetry to fill, when the caller wants any.</param>
    public void Insert(
        ReadOnlyMemory<Key> keys, RowsView vectors, ulong expansionFactor = 0, DenseTraversalStats? stats = null) =>
        Calls.Block(new DenseInsertCall(new(this._handle, null), keys, vectors, expansionFactor, stats));

    /// <summary>
    /// Re-selects the neighbors of every stored entry against the finished graph, improving edges
    /// that were chosen when the graph around them was still incomplete. Calls
    /// <c>usearch_dense_refine</c>.
    /// </summary>
    /// <remarks>
    /// This is a build step rather than a repair: a cheap insertion pass followed by one refinement
    /// reaches a better graph, in less total time, than a single expensive insertion pass - so a
    /// wide expansion factor belongs here rather than in <see cref="Insert"/>. Searches may run
    /// throughout, since edge re-selection publishes one neighbor list at a time and never removes a node.
    /// </remarks>
    /// <param name="expansionFactor">The breadth of the candidates the re-selection walk explores; zero
    /// derives it from the geometry.</param>
    /// <param name="stats">The traversal telemetry to fill, when the caller wants any.</param>
    /// <returns>The nodes re-selected and changed, the islands found and reconnected, and the audit
    /// closing the heal; <see cref="RefineReport.Changed"/> at zero means the graph has settled, so a
    /// caller loops until it falls below whatever rate it considers settled.</returns>
    public RefineReport Refine(ulong expansionFactor = 0, DenseTraversalStats? stats = null) =>
        Calls.Block(new DenseRefineCall(new(this._handle, null), expansionFactor, stats)).Result;

    /// <summary>
    /// Audits how much of the graph a search can reach: forward reachability from as many landmarks
    /// as one search draws, swept level by level. Read-only, so concurrent insertions merely leave
    /// fresh entries unaudited. Calls <c>usearch_dense_health</c>.
    /// </summary>
    /// <returns>The population audited and how much of it is reachable.</returns>
    public DenseReachability Health() => Calls.Block(new DenseHealthCall(new(this._handle, null))).Result;

    /// <summary>
    /// Resolves a matrix of queries into the nearest entries - the native batch operation,
    /// parallelized internally, which never writes to the graph. Calls <c>usearch_dense_search</c>.
    /// </summary>
    /// <remarks>
    /// A predicate narrows the matches to the keys it admits; rejected entries still route the
    /// traversal, so recall does not collapse at low selectivity. It runs on the executor's worker
    /// threads, so it must be safe for concurrent use, and it is called once per scored candidate
    /// rather than once per reported match. An exception it throws cannot cross the boundary: the
    /// first one is captured, every later candidate is rejected, and the exception is rethrown once
    /// the call is final.
    /// </remarks>
    /// <param name="queries">The query rows, one query apiece.</param>
    /// <param name="wanted">The upper bound on matches per query - the "k" in kANN.</param>
    /// <param name="arena">Where the run lands, rewound by the call.</param>
    /// <param name="expansionFactor">The breadth of the candidates the walk explores; zero derives it
    /// from the geometry.</param>
    /// <param name="predicate">The per-key admission predicate; null admits every key. A host
    /// predicate cannot narrow a device walk, so a VRAM handle refuses one as
    /// <see cref="Status.Device"/>.</param>
    /// <param name="stats">The traversal telemetry to fill, when the caller wants any.</param>
    /// <param name="traversal">A walk, the default, or an exact scan of every entry on RAM and NVMe
    /// handles; a VRAM handle refuses a scan as <see cref="Status.Unsupported"/>.</param>
    /// <returns>The run, one list per query, readable until the arena's next call.</returns>
    public Matches Search(
        RowsView queries, int wanted, Arena arena, ulong expansionFactor = 0, Func<Key, bool>? predicate = null,
        DenseTraversalStats? stats = null, Traversal traversal = Traversal.Walk) =>
        Calls.Block(new DenseSearchCall(
            new(this._handle, null), queries, wanted, arena, expansionFactor, predicate, stats, traversal)).Result;

    /// <summary>
    /// Scores multi-row queries against multi-row entries by late interaction: every query row
    /// takes its best match within a key, and those maxima sum to that key's score. Calls
    /// <c>usearch_dense_maxsim</c>.
    /// </summary>
    /// <remarks>
    /// A document is one key holding its token rows, so this is what a values-per-key bound above
    /// one exists for. It runs on the inner-product kinds alone, since a maximum over logits is
    /// what late interaction sums, and a graph frozen under a distance kind refuses with
    /// <see cref="Status.ConfigMismatch"/>.
    /// </remarks>
    /// <param name="queries">The token rows of every query, laid end to end.</param>
    /// <param name="rowsPerQuery">The rows in every query; uniform, so the row count must be a
    /// whole multiple of it and the quotient is the query count.</param>
    /// <param name="wanted">The upper bound on documents per query - the "k" over documents rather
    /// than over rows; zero keeps every key the query scored.</param>
    /// <param name="arena">Where the scored keys and their offsets land, rewound by the call.</param>
    /// <param name="expansionFactor">The breadth of the candidates the walk explores; zero derives it
    /// from the geometry.</param>
    /// <param name="stats">The traversal telemetry to fill, when the caller wants any.</param>
    /// <returns>The keys of every query, ragged, each key's summed maxima largest first.</returns>
    public DenseProducts Maxsim(
        RowsView queries, int rowsPerQuery, int wanted, Arena arena, ulong expansionFactor = 0,
        DenseTraversalStats? stats = null) =>
        Calls.Block(new DenseMaxsimCall(
            new(this._handle, null), queries, rowsPerQuery, wanted, arena, expansionFactor, stats)).Result;

    /// <summary>
    /// Multiplies a run of rows against the indexed vectors, reporting every pair the traversal scored
    /// rather than only the nearest - the approximate sparse product a graph walk reads off for free.
    /// Calls <c>usearch_dense_multiply</c>.
    /// </summary>
    /// <remarks>
    /// Pairs arrive in traversal order, without duplicates, and the pairs the walk never reaches are
    /// the low and negative products. Runs on the inner-product kinds alone; a graph frozen under a
    /// distance kind refuses with <see cref="Status.ConfigMismatch"/>.
    /// </remarks>
    /// <param name="left">The left-hand rows, one walk apiece.</param>
    /// <param name="arena">Where the pairs and their offsets land, rewound by the call.</param>
    /// <param name="wanted">The bound on pairs kept per row, the first ones its walk scored; zero
    /// keeps every pair.</param>
    /// <param name="expansionFactor">The breadth of the candidates, which widens the emitted pattern;
    /// zero derives it from the geometry.</param>
    /// <param name="stats">The traversal telemetry to fill, when the caller wants any.</param>
    /// <returns>The pairs of every left row, readable until the arena's next call.</returns>
    public DenseProducts Multiply(
        RowsView left, Arena arena, ulong wanted = 0, ulong expansionFactor = 0, DenseTraversalStats? stats = null) =>
        Calls.Block(new DenseMultiplyCall(new(this._handle, null), left, arena, wanted, expansionFactor, stats)).Result;

    /// <summary>
    /// Softmax attention of a run of query heads over every indexed key their graph walks score,
    /// folded as Flash Attention folds key tiles, with no top-k cut. Calls <c>usearch_dense_attend</c>.
    /// </summary>
    /// <remarks>
    /// The index stores no values, so every scored key resolves to a row of <paramref name="values"/>
    /// through <paramref name="rowOf"/>, which runs on the executor's threads together. A key resolving
    /// outside it, or a values matrix of another scalar kind or width than the index's own, refuses
    /// the call with <see cref="Status.ShapeMismatch"/>, since a missing row is a wrong softmax rather
    /// than a smaller one. Runs on the inner-product kinds alone; a graph frozen under a distance kind
    /// refuses with <see cref="Status.ConfigMismatch"/>. <paramref name="rowOf"/> is a host function,
    /// so it serves RAM handles alone: a VRAM walk resolves rows through a device function, which this
    /// binding does not build, so an attention into a <see cref="Residency.Vram"/> arena refuses with
    /// <see cref="Status.Device"/>.
    /// </remarks>
    /// <param name="queries">The query rows; row <c>t * headCount + h</c> is head <c>h</c> of token <c>t</c>.</param>
    /// <param name="values">The value rows the scored keys resolve into, pinned for the call.</param>
    /// <param name="rowOf">Resolves a key to its row of <paramref name="values"/>.</param>
    /// <param name="arena">Where the attended rows land, already divided by the partition, rewound by
    /// the call.</param>
    /// <param name="headCount">The query heads sharing one walk per token; it must divide the query
    /// count, and zero means one.</param>
    /// <param name="carry">The online-softmax state, one per query row, which merges this branch with
    /// another over the same queries; empty skips it.</param>
    /// <param name="outputsScalar">The scalar type the attended rows land as; <see cref="Scalar.Unknown"/>
    /// takes the queries' own.</param>
    /// <param name="expansionFactor">The breadth of the candidates; zero derives it from the geometry.</param>
    /// <param name="scoreScale">The logit scale; zero derives the inverse square root of the
    /// dimensionality.</param>
    /// <param name="stats">The traversal telemetry to fill, when the caller wants any.</param>
    /// <returns>The attended rows, one per query row, readable until the arena's next call.</returns>
    public DenseVectors Attend(
        RowsView queries, RowsView values, Func<Key, int> rowOf, Arena arena, ulong headCount = 0,
        Memory<DenseSoftmaxState> carry = default, Scalar outputsScalar = Scalar.Unknown, ulong expansionFactor = 0,
        float scoreScale = 0, DenseTraversalStats? stats = null) =>
        Calls.Block(new DenseAttendCall(
            new(this._handle, null), queries, values, rowOf, arena, headCount, carry, outputsScalar, expansionFactor,
            scoreScale, stats)).Result;

    /// <summary>
    /// Absorbs every donor into this index, one after another, keeping their graphs rather than
    /// re-inserting their vectors and linking each in across the appended part. Calls
    /// <c>usearch_dense_absorb</c>.
    /// </summary>
    /// <remarks>
    /// An absorbed donor is consumed and refuses every later verb with <see cref="Status.Uninitialized"/>;
    /// one the call did not absorb, such as a donor of another residency, scalar, metric or geometry,
    /// which refuses with <see cref="Status.ConfigMismatch"/>, stays a live index the caller still
    /// owns, though its entries may already have moved here.
    /// </remarks>
    /// <param name="donors">The indexes to absorb.</param>
    /// <returns>What the closing refinement did to the graph.</returns>
    public RefineReport Absorb(params DenseIndex[] donors) =>
        Calls.Block(new DenseAbsorbCall(new(this._handle, null), donors)).Result;

    /// <summary>
    /// Splits this index into the empty receivers, the inverse of <see cref="Absorb"/>: the entries
    /// are partitioned along the graph, each part lands in one receiver with its edges kept, and a
    /// closing refinement re-links every receiver, leaving this index empty and live. Calls
    /// <c>usearch_dense_split</c>.
    /// </summary>
    /// <param name="receivers">At least two empty indexes of this one's residency, scalar, metric,
    /// geometry and shard count.</param>
    /// <param name="seed">Picks the part roots: equal seeds over an unchanged index split equally.</param>
    /// <param name="expansionFactor">The breadth of the closing walks; zero derives it.</param>
    /// <param name="stats">The traversal telemetry to fill, when the caller wants any.</param>
    /// <returns>What the closing refinements did, across the receivers.</returns>
    public RefineReport Split(
        DenseIndex[] receivers, ulong seed = 0, ulong expansionFactor = 0, DenseTraversalStats? stats = null) =>
        Calls.Block(new DenseSplitCall(new(this._handle, null), receivers, seed, expansionFactor, stats)).Result;

    /// <summary>
    /// Draws keys around the focus, or across the whole index, each hanging from a shown or earlier
    /// draw and placed in three dimensions beside the shown ones - so a view grows one connected
    /// sample as it zooms. A larger <paramref name="wanted"/> under the same seed draws a superset;
    /// only a RAM index samples yet. Calls <c>usearch_dense_sample</c>.
    /// </summary>
    /// <param name="wanted">The upper bound on draws.</param>
    /// <param name="arena">Where the draws are written, rewound by the call.</param>
    /// <param name="config">The keys shown and their coordinates, the focus, the seed and the walk.</param>
    /// <returns>The draws, readable until the arena's next call.</returns>
    public Sample Sample(int wanted, Arena arena, DenseSampleConfig config = default) =>
        Calls.Block(new DenseSampleCall(new(this._handle, null), wanted, arena, config)).Result;

    /// <summary>
    /// Writes the index under a path, in the layout its extension picks - one <c>.safegraphs</c>
    /// monolith, or a <c>.safegraphs.json</c> manifest beside its shards. Demands a mutation-free
    /// window; concurrent readers keep searching throughout. Calls <c>usearch_dense_save</c>.
    /// </summary>
    /// <param name="path">Where the checkpoint is written.</param>
    public void Save(string path) => Calls.Block(new DenseSaveCall(new(this._handle, null), path));

    /// <summary>Writes the index as one monolith, the same bytes <see cref="Save"/> writes to a
    /// <c>.safegraphs</c> path. Calls <c>usearch_dense_save_buffer</c>.</summary>
    /// <param name="arena">Where the bytes land, rewound by the call.</param>
    /// <returns>The monolith, readable until the arena's next call.</returns>
    public Monolith SaveBuffer(Arena arena) =>
        Calls.Block(new DenseSaveBufferCall(new(this._handle, null), arena)).Result;

    /// <summary>Opens a walk over the current contents, pulled on the caller's thread; each
    /// shard's publication watermark is read as the walk enters it, so an entry once passed is never
    /// revisited. Without starts it yields every entry as stored; with them, the start keys' own entries
    /// first in order, then the entries those link to breadth-first, each once. Calls
    /// <c>usearch_dense_cursor_init</c>.</summary>
    /// <param name="starts">The keys to walk from; empty walks every row.</param>
    /// <returns>The cursor, which keeps the index alive until disposed.</returns>
    public DenseCursor Cursor(ReadOnlySpan<Key> starts = default) => DenseCursor.Open(this, null, starts);

    /// <summary>
    /// Opens a resumable nearest-neighbor stream for one query, pulled on the caller's thread. The
    /// stream keeps walk state rather than a thread. Calls <c>usearch_dense_search_stream_init</c>.
    /// </summary>
    /// <param name="query">The single query row, whose row count must be one.</param>
    /// <param name="expansionFactor">The breadth of the candidates the walk explores; zero derives it
    /// from the geometry.</param>
    /// <returns>The stream, which keeps the index alive until disposed.</returns>
    public DenseSearchStream SearchStream(RowsView query, ulong expansionFactor = 0) =>
        DenseSearchStream.Open(this, null, query, expansionFactor);

    /// <summary>Releases the index, once every deferred call naming it is final. Freeing cannot fail,
    /// and a freed handle must not be used. Calls <c>usearch_dense_free</c>.</summary>
    public void Dispose() => this._handle.Dispose();

    private static NativeBytesView Borrow(byte* data, int length) =>
        new() { Data = (IntPtr)data, Length = (nuint)length };
}

/// <summary>
/// A dense index bound to the executor its verbs block on, as <c>index.On(executor)</c> makes it.
/// The verbs are the index's own, called after every call submitted to that executor before them.
/// </summary>
/// <remarks>Binding-only; it holds two references and allocates nothing.</remarks>
public readonly struct DenseIndexOn
{
    internal DenseIndexOn(DenseIndex index, Executor executor)
    {
        this.Index = index;
        this.Executor = executor;
    }

    /// <summary>The index the verbs address.</summary>
    public DenseIndex Index { get; }

    /// <summary>The executor every verb blocks on.</summary>
    public Executor Executor { get; }
}

/// <summary>
/// A dense index bound to the executor its verbs submit to, as <c>index.Deferred(executor)</c> makes
/// it. The verbs are the index's own, each answering a <see cref="Deferred{TResult}"/> once submitted.
/// </summary>
/// <remarks>Binding-only; it holds two references and allocates nothing.</remarks>
public readonly struct DenseIndexDeferredOn
{
    internal DenseIndexDeferredOn(DenseIndex index, Executor executor)
    {
        this.Index = index;
        this.Executor = executor;
    }

    /// <summary>The index the verbs address.</summary>
    public DenseIndex Index { get; }

    /// <summary>The executor every verb submits to.</summary>
    public Executor Executor { get; }
}

/// <summary>
/// The verbs of <see cref="DenseIndex"/> on its receivers: <c>index.On(executor)</c> blocks on the
/// executor, and <c>index.Deferred(executor)</c> submits there. Each is the index's own verb, called
/// with the receiver's executor and completion.
/// </summary>
public static class DenseVerbs
{
    /// <inheritdoc cref="DenseIndex.Reserve"/>
    public static void Reserve(this DenseIndexOn on, ulong capacity) =>
        Calls.Block(new DenseReserveCall(Of(on), capacity));

    /// <inheritdoc cref="DenseIndex.Reserve"/>
    public static Deferred Reserve(this DenseIndexDeferredOn on, ulong capacity) =>
        Calls.Defer(new DenseReserveCall(Of(on), capacity));

    /// <inheritdoc cref="DenseIndex.Clear"/>
    public static void Clear(this DenseIndexOn on) => Calls.Block(new DenseClearCall(Of(on)));

    /// <inheritdoc cref="DenseIndex.Clear"/>
    public static Deferred Clear(this DenseIndexDeferredOn on) => Calls.Defer(new DenseClearCall(Of(on)));

    /// <inheritdoc cref="DenseIndex.Describe"/>
    public static DenseIndexDescription Describe(this DenseIndexOn on) =>
        Calls.Block(new DenseDescribeCall(Of(on))).Result;

    /// <inheritdoc cref="DenseIndex.Describe"/>
    public static Deferred<DenseIndexDescription> Describe(this DenseIndexDeferredOn on) =>
        Calls.Defer(new DenseDescribeCall(Of(on)));

    /// <inheritdoc cref="DenseIndex.Count"/>
    public static ulong Count(this DenseIndexOn on, Key key) => Calls.Block(new DenseCountCall(Of(on), key)).Result;

    /// <inheritdoc cref="DenseIndex.Count"/>
    public static Deferred<ulong> Count(this DenseIndexDeferredOn on, Key key) =>
        Calls.Defer(new DenseCountCall(Of(on), key));

    /// <inheritdoc cref="DenseIndex.Contains"/>
    public static bool Contains(this DenseIndexOn on, Key key) => on.Count(key) != 0;

    /// <inheritdoc cref="DenseIndex.Get"/>
    public static DenseVectors Get(this DenseIndexOn on, Key key, Arena arena, Scalar outputsScalar = Scalar.F32) =>
        Calls.Block(new DenseGetCall(Of(on), key, arena, outputsScalar)).Result;

    /// <inheritdoc cref="DenseIndex.Get"/>
    public static Deferred<DenseVectors> Get(
        this DenseIndexDeferredOn on, Key key, Arena arena, Scalar outputsScalar = Scalar.F32) =>
        Calls.Defer(new DenseGetCall(Of(on), key, arena, outputsScalar));

    /// <inheritdoc cref="DenseIndex.Insert"/>
    public static void Insert(
        this DenseIndexOn on, ReadOnlyMemory<Key> keys, RowsView vectors, ulong expansionFactor = 0,
        DenseTraversalStats? stats = null) =>
        Calls.Block(new DenseInsertCall(Of(on), keys, vectors, expansionFactor, stats));

    /// <inheritdoc cref="DenseIndex.Insert"/>
    public static Deferred Insert(
        this DenseIndexDeferredOn on, ReadOnlyMemory<Key> keys, RowsView vectors, ulong expansionFactor = 0,
        DenseTraversalStats? stats = null) =>
        Calls.Defer(new DenseInsertCall(Of(on), keys, vectors, expansionFactor, stats));

    /// <inheritdoc cref="DenseIndex.Refine"/>
    public static RefineReport Refine(
        this DenseIndexOn on, ulong expansionFactor = 0, DenseTraversalStats? stats = null) =>
        Calls.Block(new DenseRefineCall(Of(on), expansionFactor, stats)).Result;

    /// <inheritdoc cref="DenseIndex.Refine"/>
    public static Deferred<RefineReport> Refine(
        this DenseIndexDeferredOn on, ulong expansionFactor = 0, DenseTraversalStats? stats = null) =>
        Calls.Defer(new DenseRefineCall(Of(on), expansionFactor, stats));

    /// <inheritdoc cref="DenseIndex.Health"/>
    public static DenseReachability Health(this DenseIndexOn on) => Calls.Block(new DenseHealthCall(Of(on))).Result;

    /// <inheritdoc cref="DenseIndex.Health"/>
    public static Deferred<DenseReachability> Health(this DenseIndexDeferredOn on) =>
        Calls.Defer(new DenseHealthCall(Of(on)));

    /// <inheritdoc cref="DenseIndex.Search"/>
    public static Matches Search(
        this DenseIndexOn on, RowsView queries, int wanted, Arena arena, ulong expansionFactor = 0,
        Func<Key, bool>? predicate = null, DenseTraversalStats? stats = null, Traversal traversal = Traversal.Walk) =>
        Calls.Block(new DenseSearchCall(
            Of(on), queries, wanted, arena, expansionFactor, predicate, stats, traversal)).Result;

    /// <inheritdoc cref="DenseIndex.Search"/>
    public static Deferred<Matches> Search(
        this DenseIndexDeferredOn on, RowsView queries, int wanted, Arena arena, ulong expansionFactor = 0,
        Func<Key, bool>? predicate = null, DenseTraversalStats? stats = null, Traversal traversal = Traversal.Walk) =>
        Calls.Defer(new DenseSearchCall(Of(on), queries, wanted, arena, expansionFactor, predicate, stats, traversal));

    /// <inheritdoc cref="DenseIndex.Maxsim"/>
    public static DenseProducts Maxsim(
        this DenseIndexOn on, RowsView queries, int rowsPerQuery, int wanted, Arena arena, ulong expansionFactor = 0,
        DenseTraversalStats? stats = null) =>
        Calls.Block(new DenseMaxsimCall(Of(on), queries, rowsPerQuery, wanted, arena, expansionFactor, stats)).Result;

    /// <inheritdoc cref="DenseIndex.Maxsim"/>
    public static Deferred<DenseProducts> Maxsim(
        this DenseIndexDeferredOn on, RowsView queries, int rowsPerQuery, int wanted, Arena arena,
        ulong expansionFactor = 0, DenseTraversalStats? stats = null) =>
        Calls.Defer(new DenseMaxsimCall(Of(on), queries, rowsPerQuery, wanted, arena, expansionFactor, stats));

    /// <inheritdoc cref="DenseIndex.Multiply"/>
    public static DenseProducts Multiply(
        this DenseIndexOn on, RowsView left, Arena arena, ulong wanted = 0, ulong expansionFactor = 0,
        DenseTraversalStats? stats = null) =>
        Calls.Block(new DenseMultiplyCall(Of(on), left, arena, wanted, expansionFactor, stats)).Result;

    /// <inheritdoc cref="DenseIndex.Multiply"/>
    public static Deferred<DenseProducts> Multiply(
        this DenseIndexDeferredOn on, RowsView left, Arena arena, ulong wanted = 0, ulong expansionFactor = 0,
        DenseTraversalStats? stats = null) =>
        Calls.Defer(new DenseMultiplyCall(Of(on), left, arena, wanted, expansionFactor, stats));

    /// <inheritdoc cref="DenseIndex.Attend"/>
    public static DenseVectors Attend(
        this DenseIndexOn on, RowsView queries, RowsView values, Func<Key, int> rowOf, Arena arena,
        ulong headCount = 0, Memory<DenseSoftmaxState> carry = default, Scalar outputsScalar = Scalar.Unknown,
        ulong expansionFactor = 0, float scoreScale = 0, DenseTraversalStats? stats = null) =>
        Calls.Block(new DenseAttendCall(
            Of(on), queries, values, rowOf, arena, headCount, carry, outputsScalar, expansionFactor, scoreScale,
            stats)).Result;

    /// <inheritdoc cref="DenseIndex.Attend"/>
    public static Deferred<DenseVectors> Attend(
        this DenseIndexDeferredOn on, RowsView queries, RowsView values, Func<Key, int> rowOf, Arena arena,
        ulong headCount = 0, Memory<DenseSoftmaxState> carry = default, Scalar outputsScalar = Scalar.Unknown,
        ulong expansionFactor = 0, float scoreScale = 0, DenseTraversalStats? stats = null) =>
        Calls.Defer(new DenseAttendCall(
            Of(on), queries, values, rowOf, arena, headCount, carry, outputsScalar, expansionFactor, scoreScale,
            stats));

    /// <inheritdoc cref="DenseIndex.Absorb"/>
    public static RefineReport Absorb(this DenseIndexOn on, params DenseIndex[] donors) =>
        Calls.Block(new DenseAbsorbCall(Of(on), donors)).Result;

    /// <inheritdoc cref="DenseIndex.Absorb"/>
    public static Deferred<RefineReport> Absorb(this DenseIndexDeferredOn on, params DenseIndex[] donors) =>
        Calls.Defer(new DenseAbsorbCall(Of(on), donors));

    /// <inheritdoc cref="DenseIndex.Split"/>
    public static RefineReport Split(
        this DenseIndexOn on, DenseIndex[] receivers, ulong seed = 0, ulong expansionFactor = 0,
        DenseTraversalStats? stats = null) =>
        Calls.Block(new DenseSplitCall(Of(on), receivers, seed, expansionFactor, stats)).Result;

    /// <inheritdoc cref="DenseIndex.Split"/>
    public static Deferred<RefineReport> Split(
        this DenseIndexDeferredOn on, DenseIndex[] receivers, ulong seed = 0, ulong expansionFactor = 0,
        DenseTraversalStats? stats = null) =>
        Calls.Defer(new DenseSplitCall(Of(on), receivers, seed, expansionFactor, stats));

    /// <inheritdoc cref="DenseIndex.Sample"/>
    public static Sample Sample(this DenseIndexOn on, int wanted, Arena arena, DenseSampleConfig config = default) =>
        Calls.Block(new DenseSampleCall(Of(on), wanted, arena, config)).Result;

    /// <inheritdoc cref="DenseIndex.Sample"/>
    public static Deferred<Sample> Sample(
        this DenseIndexDeferredOn on, int wanted, Arena arena, DenseSampleConfig config = default) =>
        Calls.Defer(new DenseSampleCall(Of(on), wanted, arena, config));

    /// <inheritdoc cref="DenseIndex.Save"/>
    public static void Save(this DenseIndexOn on, string path) => Calls.Block(new DenseSaveCall(Of(on), path));

    /// <inheritdoc cref="DenseIndex.Save"/>
    public static Deferred Save(this DenseIndexDeferredOn on, string path) =>
        Calls.Defer(new DenseSaveCall(Of(on), path));

    /// <inheritdoc cref="DenseIndex.SaveBuffer"/>
    public static Monolith SaveBuffer(this DenseIndexOn on, Arena arena) =>
        Calls.Block(new DenseSaveBufferCall(Of(on), arena)).Result;

    /// <inheritdoc cref="DenseIndex.SaveBuffer"/>
    public static Deferred<Monolith> SaveBuffer(this DenseIndexDeferredOn on, Arena arena) =>
        Calls.Defer(new DenseSaveBufferCall(Of(on), arena));

    /// <summary>Opens a walk over the current contents, pulled blocking on the receiver's
    /// executor. Calls <c>usearch_dense_cursor_init</c>.</summary>
    /// <param name="on">The receiver whose executor every pull runs on.</param>
    /// <param name="starts">The keys to walk from; empty walks every row.</param>
    /// <returns>The cursor, which keeps the index alive until disposed.</returns>
    public static DenseCursor Cursor(this DenseIndexOn on, ReadOnlySpan<Key> starts = default) =>
        DenseCursor.Open(on.Index, on.Executor, starts);

    /// <summary>Opens a resumable nearest-neighbor stream for one query, pulled blocking on the
    /// receiver's executor. Calls <c>usearch_dense_search_stream_init</c>.</summary>
    /// <param name="on">The receiver whose executor every pull runs on.</param>
    /// <param name="query">The single query row, whose row count must be one.</param>
    /// <param name="expansionFactor">The breadth of the candidates the walk explores; zero derives it
    /// from the geometry.</param>
    /// <returns>The stream, which keeps the index alive until disposed.</returns>
    public static DenseSearchStream SearchStream(this DenseIndexOn on, RowsView query, ulong expansionFactor = 0) =>
        DenseSearchStream.Open(on.Index, on.Executor, query, expansionFactor);

    private static CallHandles Of(DenseIndexOn on) => new(on.Index.Handle, on.Executor);

    private static CallHandles Of(DenseIndexDeferredOn on) => new(on.Index.Handle, on.Executor);
}

#endregion Index

#region Calls

/// <summary>A reservation; see <see cref="DenseIndex.Reserve"/>.</summary>
internal unsafe struct DenseReserveCall(CallHandles handles, ulong capacity) : ICall
{

    public void Submit(CallOutputs* outputs, Completion completion)
    {
        using Lease index = handles.LendIndex();
        using Lease executor = handles.LendExecutor();
        NativeDenseReserve arguments = new()
        {
            Index = index.Pointer, Executor = executor.Pointer, Outcome = &outputs->Outcome,
            Completion = completion, Capacity = (nuint)capacity,
        };
        usearch_dense_reserve(&arguments);
    }

    public readonly void Release(CallOutputs* outputs) { }

    public readonly void Finish(CallOutputs* outputs) => outputs->Outcome.ThrowIfFailed();
}

/// <summary>An erasure; see <see cref="DenseIndex.Clear"/>.</summary>
internal unsafe struct DenseClearCall(CallHandles handles) : ICall
{

    public void Submit(CallOutputs* outputs, Completion completion)
    {
        using Lease index = handles.LendIndex();
        using Lease executor = handles.LendExecutor();
        NativeDenseClear arguments = new()
        {
            Index = index.Pointer, Executor = executor.Pointer, Outcome = &outputs->Outcome,
            Completion = completion,
        };
        usearch_dense_clear(&arguments);
    }

    public readonly void Release(CallOutputs* outputs) { }

    public readonly void Finish(CallOutputs* outputs) => outputs->Outcome.ThrowIfFailed();
}

/// <summary>A description; see <see cref="DenseIndex.Describe"/>.</summary>
internal unsafe struct DenseDescribeCall(CallHandles handles) : ICall<DenseIndexDescription>
{

    public DenseIndexDescription Result { get; private set; }

    public void Submit(CallOutputs* outputs, Completion completion)
    {
        using Lease index = handles.LendIndex();
        using Lease executor = handles.LendExecutor();
        NativeDenseDescribe arguments = new()
        {
            Index = index.Pointer, Executor = executor.Pointer, Outcome = &outputs->Outcome,
            Completion = completion, Description = (NativeDenseDescription*)outputs->Report,
        };
        usearch_dense_describe(&arguments);
    }

    public readonly void Release(CallOutputs* outputs) { }

    public void Finish(CallOutputs* outputs)
    {
        outputs->Outcome.ThrowIfFailed();
        NativeDenseDescription* lowered = (NativeDenseDescription*)outputs->Report;
        this.Result = new DenseIndexDescription
        {
            Metadata = DenseMetadata.Lift(lowered->Metadata),
            Size = lowered->Size,
            Capacity = lowered->Capacity,
            MemoryUsage = lowered->MemoryUsage,
            CpuClass = Marshal.PtrToStringUTF8(NativeMethods.usearch_cpu_class_name(lowered->CpuClass)) ?? "unknown",
            GpuClass = Marshal.PtrToStringUTF8(NativeMethods.usearch_gpu_class_name(lowered->GpuClass)) ?? "unknown",
            ShardsCount = lowered->ShardsCount,
        };
    }
}

/// <summary>A count; see <see cref="DenseIndex.Count"/>.</summary>
internal unsafe struct DenseCountCall(CallHandles handles, Key key) : ICall<ulong>
{

    public ulong Result { get; private set; }

    public void Submit(CallOutputs* outputs, Completion completion)
    {
        using Lease index = handles.LendIndex();
        using Lease executor = handles.LendExecutor();
        NativeDenseCount arguments = new()
        {
            Index = index.Pointer, Executor = executor.Pointer, Outcome = &outputs->Outcome,
            Completion = completion, Key = key, Entries = &outputs->Written,
        };
        usearch_dense_count(&arguments);
    }

    public readonly void Release(CallOutputs* outputs) { }

    public void Finish(CallOutputs* outputs)
    {
        outputs->Outcome.ThrowIfFailed();
        this.Result = outputs->Written;
    }
}

/// <summary>A retrieval; see <see cref="DenseIndex.Get"/>.</summary>
internal unsafe struct DenseGetCall(CallHandles handles, Key key, Arena arena, Scalar outputsScalar)
    : ICall<DenseVectors>
{
    private int _generation;

    public DenseVectors Result { get; private set; }

    public void Submit(CallOutputs* outputs, Completion completion)
    {
        using Lease index = handles.LendIndex();
        using Lease executor = handles.LendExecutor();
        using Lease lent = Arena.Lend(arena);
        NativeDenseGet arguments = new()
        {
            Index = index.Pointer, Executor = executor.Pointer, Outcome = &outputs->Outcome, Completion = completion,
            Key = key, Vectors = (NativeRowsView*)outputs->View, Written = &outputs->Written, Arena = lent.Pointer,
            OutputsScalar = outputsScalar,
        };
        usearch_dense_get(&arguments);
        this._generation = Arena.Rewound(arena, &outputs->Outcome);
    }

    public readonly void Release(CallOutputs* outputs) { }

    public void Finish(CallOutputs* outputs)
    {
        outputs->Outcome.ThrowIfFailed();
        this.Result = new DenseVectors(arena, this._generation, *(NativeRowsView*)outputs->View, (int)outputs->Written);
    }
}

/// <summary>An insertion; see <see cref="DenseIndex.Insert"/>.</summary>
internal unsafe struct DenseInsertCall(
    CallHandles handles, ReadOnlyMemory<Key> keys, RowsView vectors, ulong expansionFactor, DenseTraversalStats? stats)
    : ICall
{
    private MemoryHandle _keys;
    private MemoryHandle _vectors;

    public void Submit(CallOutputs* outputs, Completion completion)
    {
        // The library reads as many keys as rows, so only this side sees the two disagree.
        if (keys.Length != vectors.Rows)
        {
            throw new USearchException(
                Status.ShapeMismatch, "the key count does not match the row count it accompanies");
        }

        fixed (Key* keyRun = keys.Span)
        fixed (byte* rows = vectors.Bytes)
        {
            if (completion == Completion.Deferred)
            {
                this._keys = keys.Pin();
                this._vectors = vectors.Pin();
            }

            using Lease index = handles.LendIndex();
            using Lease executor = handles.LendExecutor();
            NativeDenseInsert arguments = new()
            {
                Index = index.Pointer, Executor = executor.Pointer, Outcome = &outputs->Outcome,
                Completion = completion, Count = (nuint)keys.Length,
                Keys = new NativeKeysView { Data = (IntPtr)keyRun, Stride = (nuint)sizeof(Key) },
                Vectors = vectors.AsView(rows), ExpansionFactor = (nuint)expansionFactor,
                Stats = (NativeDenseTraversalStats*)outputs->Report,
            };
            usearch_dense_insert(&arguments);
        }
    }

    public void Release(CallOutputs* outputs)
    {
        this._keys.Dispose();
        this._vectors.Dispose();
    }

    public readonly void Finish(CallOutputs* outputs)
    {
        outputs->Outcome.ThrowIfFailed();
        stats?.Adopt(*(NativeDenseTraversalStats*)outputs->Report);
    }
}

/// <summary>A refinement; see <see cref="DenseIndex.Refine"/>.</summary>
internal unsafe struct DenseRefineCall(CallHandles handles, ulong expansionFactor, DenseTraversalStats? stats)
    : ICall<RefineReport>
{

    public RefineReport Result { get; private set; }

    public void Submit(CallOutputs* outputs, Completion completion)
    {
        using Lease index = handles.LendIndex();
        using Lease executor = handles.LendExecutor();
        NativeDenseRefine arguments = new()
        {
            Index = index.Pointer, Executor = executor.Pointer, Outcome = &outputs->Outcome,
            Completion = completion, Report = RefineReport.In(outputs),
            Refine = new() { ExpansionFactor = (nuint)expansionFactor },
            Stats = (NativeDenseTraversalStats*)outputs->Report,
        };
        usearch_dense_refine(&arguments);
    }

    public readonly void Release(CallOutputs* outputs) { }

    public void Finish(CallOutputs* outputs)
    {
        outputs->Outcome.ThrowIfFailed();
        stats?.Adopt(*(NativeDenseTraversalStats*)outputs->Report);
        this.Result = RefineReport.Lift(RefineReport.In(outputs));
    }
}

/// <summary>A reachability audit; see <see cref="DenseIndex.Health"/>.</summary>
internal unsafe struct DenseHealthCall(CallHandles handles) : ICall<DenseReachability>
{

    public DenseReachability Result { get; private set; }

    public void Submit(CallOutputs* outputs, Completion completion)
    {
        using Lease index = handles.LendIndex();
        using Lease executor = handles.LendExecutor();
        NativeDenseHealth arguments = new()
        {
            Index = index.Pointer, Executor = executor.Pointer, Outcome = &outputs->Outcome,
            Completion = completion, Reachability = (NativeDenseReachability*)outputs->Report,
        };
        usearch_dense_health(&arguments);
    }

    public readonly void Release(CallOutputs* outputs) { }

    public void Finish(CallOutputs* outputs)
    {
        outputs->Outcome.ThrowIfFailed();
        NativeDenseReachability* audit = (NativeDenseReachability*)outputs->Report;
        this.Result = new DenseReachability
        {
            Audited = audit->Audited, Reachable = audit->Reachable, Sweeps = audit->Sweeps,
            BackSweeps = audit->BackSweeps,
        };
    }
}

/// <summary>A search; see <see cref="DenseIndex.Search"/>.</summary>
internal unsafe struct DenseSearchCall(
    CallHandles handles, RowsView queries, int wanted, Arena arena, ulong expansionFactor, Func<Key, bool>? predicate,
    DenseTraversalStats? stats, Traversal traversal) : ICall<Matches>
{
    private readonly Admitting? _admitting = predicate is null ? null : new Admitting(predicate);
    private GCHandle _registration;
    private MemoryHandle _queries;
    private int _generation;

    public Matches Result { get; private set; }

    public void Submit(CallOutputs* outputs, Completion completion)
    {
        nuint bound = Calls.Bound(wanted);
        fixed (byte* rows = queries.Bytes)
        {
            if (completion == Completion.Deferred)
            {
                this._queries = queries.Pin();
            }

            if (this._admitting is not null)
            {
                this._registration = GCHandle.Alloc(this._admitting);
            }

            using Lease index = handles.LendIndex();
            using Lease executor = handles.LendExecutor();
            using Lease lent = Arena.Lend(arena);
            NativeDenseSearch arguments = new()
            {
                Index = index.Pointer, Executor = executor.Pointer, Outcome = &outputs->Outcome, Completion = completion,
                Count = (nuint)queries.Rows, Queries = queries.AsView(rows),
                Matches = (NativeMatchesView*)outputs->View, Arena = lent.Pointer, Wanted = bound,
                ExpansionFactor = (nuint)expansionFactor, Traversal = traversal,
                Predicate = this._admitting is null
                    ? default
                    : new NativePredicate { Callback = &Admitting.Admits, State = GCHandle.ToIntPtr(this._registration) },
                Stats = (NativeDenseTraversalStats*)outputs->Report,
            };
            usearch_dense_search(&arguments);
            this._generation = Arena.Rewound(arena, &outputs->Outcome);
        }
    }

    public void Release(CallOutputs* outputs)
    {
        if (this._registration.IsAllocated)
        {
            this._registration.Free();
        }

        this._queries.Dispose();
    }

    public void Finish(CallOutputs* outputs)
    {
        this._admitting?.Rethrow();
        outputs->Outcome.ThrowIfFailed();
        stats?.Adopt(*(NativeDenseTraversalStats*)outputs->Report);
        this.Result = new Matches(arena, this._generation, *(NativeMatchesView*)outputs->View, queries.Rows);
    }
}

/// <summary>A late-interaction scoring; see <see cref="DenseIndex.Maxsim"/>.</summary>
internal unsafe struct DenseMaxsimCall(
    CallHandles handles, RowsView queries, int rowsPerQuery, int wanted, Arena arena, ulong expansionFactor,
    DenseTraversalStats? stats) : ICall<DenseProducts>
{
    private MemoryHandle _queries;
    private int _generation;

    public DenseProducts Result { get; private set; }

    public void Submit(CallOutputs* outputs, Completion completion)
    {
        // The library reads count times rows-per-query rows, so only this side sees them overrun the slab.
        if (rowsPerQuery <= 0 || queries.Rows % rowsPerQuery != 0)
        {
            throw new USearchException(
                Status.ShapeMismatch, "the token rows are not a whole multiple of the rows per query");
        }

        nuint bound = Calls.Bound(wanted);
        fixed (byte* rows = queries.Bytes)
        {
            if (completion == Completion.Deferred)
            {
                this._queries = queries.Pin();
            }

            using Lease index = handles.LendIndex();
            using Lease executor = handles.LendExecutor();
            using Lease lent = Arena.Lend(arena);
            NativeDenseMaxsim arguments = new()
            {
                Index = index.Pointer, Executor = executor.Pointer, Outcome = &outputs->Outcome, Completion = completion,
                Count = (nuint)(queries.Rows / rowsPerQuery), Queries = queries.AsView(rows),
                RowsPerQuery = (nuint)rowsPerQuery, Matches = (NativeMatchesTapeView*)outputs->View,
                Written = &outputs->Written, Required = &outputs->Required, Arena = lent.Pointer, Wanted = bound,
                ExpansionFactor = (nuint)expansionFactor,
                Stats = (NativeDenseTraversalStats*)outputs->Report,
            };
            usearch_dense_maxsim(&arguments);
            this._generation = Arena.Rewound(arena, &outputs->Outcome);
        }
    }

    public void Release(CallOutputs* outputs) => this._queries.Dispose();

    public void Finish(CallOutputs* outputs)
    {
        outputs->Outcome.ThrowIfFailed();
        stats?.Adopt(*(NativeDenseTraversalStats*)outputs->Report);
        this.Result = new DenseProducts(
            arena, this._generation, *(NativeMatchesTapeView*)outputs->View, queries.Rows / rowsPerQuery,
            outputs->Required);
    }
}

/// <summary>A sparse product; see <see cref="DenseIndex.Multiply"/>.</summary>
internal unsafe struct DenseMultiplyCall(
    CallHandles handles, RowsView left, Arena arena, ulong wanted, ulong expansionFactor, DenseTraversalStats? stats)
    : ICall<DenseProducts>
{
    private MemoryHandle _left;
    private int _generation;

    public DenseProducts Result { get; private set; }

    public void Submit(CallOutputs* outputs, Completion completion)
    {
        fixed (byte* rows = left.Bytes)
        {
            if (completion == Completion.Deferred)
            {
                this._left = left.Pin();
            }

            using Lease index = handles.LendIndex();
            using Lease executor = handles.LendExecutor();
            using Lease lent = Arena.Lend(arena);
            NativeDenseMultiply arguments = new()
            {
                Index = index.Pointer, Executor = executor.Pointer, Outcome = &outputs->Outcome, Completion = completion,
                Count = (nuint)left.Rows, Left = left.AsView(rows), Products = (NativeMatchesTapeView*)outputs->View,
                Written = &outputs->Written, Required = &outputs->Required, Arena = lent.Pointer,
                Wanted = (nuint)wanted, ExpansionFactor = (nuint)expansionFactor,
                Stats = (NativeDenseTraversalStats*)outputs->Report,
            };
            usearch_dense_multiply(&arguments);
            this._generation = Arena.Rewound(arena, &outputs->Outcome);
        }
    }

    public void Release(CallOutputs* outputs) => this._left.Dispose();

    public void Finish(CallOutputs* outputs)
    {
        outputs->Outcome.ThrowIfFailed();
        stats?.Adopt(*(NativeDenseTraversalStats*)outputs->Report);
        this.Result = new DenseProducts(
            arena, this._generation, *(NativeMatchesTapeView*)outputs->View, left.Rows, outputs->Required);
    }
}

/// <summary>A graph-walk attention; see <see cref="DenseIndex.Attend"/>.</summary>
internal unsafe struct DenseAttendCall(
    CallHandles handles, RowsView queries, RowsView values, Func<Key, int> rowOf, Arena arena, ulong headCount,
    Memory<DenseSoftmaxState> carry, Scalar outputsScalar, ulong expansionFactor, float scoreScale,
    DenseTraversalStats? stats) : ICall<DenseVectors>
{
    private readonly RowsView _values = values;
    private readonly ValueRows _lookup = new(rowOf, values);
    private GCHandle _registration;
    private MemoryHandle _queries;
    private MemoryHandle _pinnedValues;
    private MemoryHandle _carry;
    private int _generation;

    public DenseVectors Result { get; private set; }

    public void Submit(CallOutputs* outputs, Completion completion)
    {
        // The carry pointer has no length, so only this side can see it fall short.
        if (!carry.IsEmpty && carry.Length < queries.Rows)
        {
            throw new USearchException(
                Status.ShapeMismatch, "the softmax carry holds fewer entries than there are queries");
        }

        if (arena is null)
        {
            throw new USearchException(Status.InvalidArgument, "the arena must not be null");
        }

        // A VRAM arena serves only a VRAM call, whose walk would run this host lookup on the device.
        if (arena.Describe().Residency == Residency.Vram)
        {
            throw new USearchException(Status.Device, "a VRAM walk needs a device-callable value lookup");
        }

        fixed (byte* queryRows = queries.Bytes)
        fixed (byte* valueRows = this._values.Bytes)
        fixed (DenseSoftmaxState* carried = carry.Span)
        {
            if (completion == Completion.Deferred)
            {
                this._queries = queries.Pin();
                this._pinnedValues = this._values.Pin();
                this._carry = carry.Pin();
            }

            this._lookup.Base = valueRows;
            this._registration = GCHandle.Alloc(this._lookup);
            using Lease index = handles.LendIndex();
            using Lease executor = handles.LendExecutor();
            using Lease lent = Arena.Lend(arena);
            NativeDenseAttend arguments = new()
            {
                Index = index.Pointer, Executor = executor.Pointer, Outcome = &outputs->Outcome, Completion = completion,
                Count = (nuint)queries.Rows, Queries = queries.AsView(queryRows), HeadCount = (nuint)headCount,
                ValueLookup = &ValueRows.Resolve, ValueLookupState = GCHandle.ToIntPtr(this._registration),
                Carry = carried, Outputs = (NativeRowsView*)outputs->View, Arena = lent.Pointer,
                OutputsScalar = outputsScalar, ExpansionFactor = (nuint)expansionFactor, ScoreScale = scoreScale,
                Stats = (NativeDenseTraversalStats*)outputs->Report,
            };
            usearch_dense_attend(&arguments);
            this._generation = Arena.Rewound(arena, &outputs->Outcome);
        }
    }

    public void Release(CallOutputs* outputs)
    {
        if (this._registration.IsAllocated)
        {
            this._registration.Free();
        }

        this._queries.Dispose();
        this._pinnedValues.Dispose();
        this._carry.Dispose();
    }

    public void Finish(CallOutputs* outputs)
    {
        this._lookup.Rethrow();
        outputs->Outcome.ThrowIfFailed();
        stats?.Adopt(*(NativeDenseTraversalStats*)outputs->Report);
        this.Result = new DenseVectors(arena, this._generation, *(NativeRowsView*)outputs->View, queries.Rows);
    }
}

/// <summary>An absorb; see <see cref="DenseIndex.Absorb"/>.</summary>
internal unsafe struct DenseAbsorbCall(CallHandles handles, DenseIndex[] donors) : ICall<RefineReport>
{
    private readonly DenseIndex[] _donors = donors;
    private readonly IntPtr[] _slots = new IntPtr[donors.Length];
    private readonly Lease[] _leases = new Lease[donors.Length];
    private MemoryHandle _pinnedSlots;

    public RefineReport Result { get; private set; }

    public void Submit(CallOutputs* outputs, Completion completion)
    {
        for (int position = 0; position < this._donors.Length; position++)
        {
            this._leases[position] = Lease.Of(this._donors[position].Handle, "donor");
            this._slots[position] = this._leases[position].Pointer;
        }

        fixed (IntPtr* slots = this._slots)
        {
            if (completion == Completion.Deferred)
            {
                this._pinnedSlots = this._slots.AsMemory().Pin();
            }

            using Lease index = handles.LendIndex();
            using Lease executor = handles.LendExecutor();
            NativeDenseAbsorb arguments = new()
            {
                Index = index.Pointer, Executor = executor.Pointer, Outcome = &outputs->Outcome,
                Completion = completion, Count = (nuint)this._donors.Length, Donors = slots,
                Report = RefineReport.In(outputs),
            };
            usearch_dense_absorb(&arguments);
        }
    }

    public void Release(CallOutputs* outputs)
    {
        for (int position = 0; position < this._donors.Length; position++)
        {
            // A nulled slot was consumed, so its wrapper must never free it - marked before the lease ends.
            if (this._leases[position].Holds && this._slots[position] == IntPtr.Zero)
            {
                this._donors[position].Handle.SetHandleAsInvalid();
            }

            this._leases[position].Dispose();
        }

        this._pinnedSlots.Dispose();
    }

    public void Finish(CallOutputs* outputs)
    {
        outputs->Outcome.ThrowIfFailed();
        this.Result = RefineReport.Lift(RefineReport.In(outputs));
    }
}

/// <summary>A split; see <see cref="DenseIndex.Split"/>.</summary>
internal unsafe struct DenseSplitCall(
    CallHandles handles, DenseIndex[] receivers, ulong seed, ulong expansionFactor, DenseTraversalStats? stats)
    : ICall<RefineReport>
{
    private readonly IntPtr[] _slots = new IntPtr[receivers.Length];
    private readonly Lease[] _leases = new Lease[receivers.Length];
    private MemoryHandle _pinnedSlots;

    public RefineReport Result { get; private set; }

    public void Submit(CallOutputs* outputs, Completion completion)
    {
        for (int position = 0; position < receivers.Length; position++)
        {
            this._leases[position] = Lease.Of(receivers[position].Handle, "receiver");
            this._slots[position] = this._leases[position].Pointer;
        }

        fixed (IntPtr* slots = this._slots)
        {
            if (completion == Completion.Deferred)
            {
                this._pinnedSlots = this._slots.AsMemory().Pin();
            }

            using Lease index = handles.LendIndex();
            using Lease executor = handles.LendExecutor();
            NativeDenseSplit arguments = new()
            {
                Index = index.Pointer, Executor = executor.Pointer, Outcome = &outputs->Outcome,
                Completion = completion, Count = (nuint)receivers.Length, Receivers = slots,
                Report = RefineReport.In(outputs),
                Refine = new() { ExpansionFactor = (nuint)expansionFactor }, Seed = seed,
                Stats = (NativeDenseTraversalStats*)outputs->Report,
            };
            usearch_dense_split(&arguments);
        }
    }

    public void Release(CallOutputs* outputs)
    {
        for (int position = 0; position < this._leases.Length; position++)
        {
            this._leases[position].Dispose();
        }

        this._pinnedSlots.Dispose();
    }

    public void Finish(CallOutputs* outputs)
    {
        outputs->Outcome.ThrowIfFailed();
        stats?.Adopt(*(NativeDenseTraversalStats*)outputs->Report);
        this.Result = RefineReport.Lift(RefineReport.In(outputs));
    }
}

/// <summary>A sample; see <see cref="DenseIndex.Sample"/>.</summary>
internal unsafe struct DenseSampleCall(CallHandles handles, int wanted, Arena arena, DenseSampleConfig config)
    : ICall<Sample>
{
    private MemoryHandle _shown;
    private MemoryHandle _shownCoordinates;
    private MemoryHandle _focus;
    private int _generation;

    public Sample Result { get; private set; }

    public void Submit(CallOutputs* outputs, Completion completion)
    {
        ReadOnlyMemory<Key> shown = config.Shown, focus = config.Focus;
        ReadOnlyMemory<float> shownCoordinates = config.ShownCoordinates;
        // The library reads three coordinates per shown key, so only this side sees the two disagree.
        if (shownCoordinates.Length != shown.Length * 3)
        {
            throw new USearchException(Status.ShapeMismatch, "every shown key needs three coordinates");
        }

        nuint bound = Calls.Bound(wanted);
        SampleReports* reports = (SampleReports*)outputs->Report;
        fixed (Key* shownRun = shown.Span)
        fixed (float* coordinates = shownCoordinates.Span)
        fixed (Key* focusRun = focus.Span)
        {
            if (completion == Completion.Deferred)
            {
                this._shown = shown.Pin();
                this._shownCoordinates = shownCoordinates.Pin();
                this._focus = focus.Pin();
            }

            using Lease index = handles.LendIndex();
            using Lease executor = handles.LendExecutor();
            using Lease lent = Arena.Lend(arena);
            NativeDenseSample arguments = new()
            {
                Index = index.Pointer, Executor = executor.Pointer, Outcome = &outputs->Outcome, Completion = completion,
                ShownCount = (nuint)shown.Length,
                Shown = new NativeKeysView { Data = (IntPtr)shownRun, Stride = (nuint)sizeof(Key) },
                ShownCoordinates = new NativeRowsView
                {
                    Data = (IntPtr)coordinates, Stride = (nuint)(3 * sizeof(float)), Columns = 3, Scalar = Scalar.F32,
                },
                FocusCount = (nuint)focus.Length,
                Focus = new NativeKeysView { Data = (IntPtr)focusRun, Stride = (nuint)sizeof(Key) },
                Keys = &reports->Keys, EntryOrdinals = &reports->EntryOrdinals,
                Parents = (NativeMatchesView*)outputs->View, Coordinates = &reports->Coordinates,
                Written = &outputs->Written, Arena = lent.Pointer, Wanted = bound, Seed = config.Seed,
                ExpansionFactor = (nuint)config.ExpansionFactor, Stats = &reports->Stats,
            };
            usearch_dense_sample(&arguments);
            this._generation = Arena.Rewound(arena, &outputs->Outcome);
        }
    }

    public void Release(CallOutputs* outputs)
    {
        this._shown.Dispose();
        this._shownCoordinates.Dispose();
        this._focus.Dispose();
    }

    public void Finish(CallOutputs* outputs)
    {
        outputs->Outcome.ThrowIfFailed();
        SampleReports* reports = (SampleReports*)outputs->Report;
        config.Stats?.Adopt(reports->Stats);
        this.Result = new Sample(
            arena, this._generation, reports->Keys, reports->EntryOrdinals, *(NativeMatchesView*)outputs->View,
            reports->Coordinates, (int)outputs->Written);
    }
}

/// <summary>What a sample writes into <see cref="CallOutputs.Report"/>: its telemetry, then the views of
/// its keys, entry ordinals and coordinates.</summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct SampleReports
{
    public NativeDenseTraversalStats Stats;
    public NativeKeysView Keys;
    public nuint* EntryOrdinals;
    public NativeRowsView Coordinates;
}

/// <summary>A checkpoint write; see <see cref="DenseIndex.Save"/>.</summary>
internal unsafe struct DenseSaveCall(CallHandles handles, string path) : ICall
{
    private readonly byte[] _path = Paths.Encode(path);
    private MemoryHandle _pinnedPath;

    public void Submit(CallOutputs* outputs, Completion completion)
    {
        fixed (byte* path = this._path)
        {
            if (completion == Completion.Deferred)
            {
                this._pinnedPath = this._path.AsMemory().Pin();
            }

            using Lease index = handles.LendIndex();
            using Lease executor = handles.LendExecutor();
            NativeDenseSave arguments = new()
            {
                Index = index.Pointer, Executor = executor.Pointer, Outcome = &outputs->Outcome,
                Completion = completion,
                Path = new NativeBytesView { Data = (IntPtr)path, Length = (nuint)this._path.Length },
            };
            usearch_dense_save(&arguments);
        }
    }

    public void Release(CallOutputs* outputs) => this._pinnedPath.Dispose();

    public readonly void Finish(CallOutputs* outputs) => outputs->Outcome.ThrowIfFailed();
}

/// <summary>A monolith write; see <see cref="DenseIndex.SaveBuffer"/>.</summary>
internal unsafe struct DenseSaveBufferCall(CallHandles handles, Arena arena) : ICall<Monolith>
{
    private int _generation;

    public Monolith Result { get; private set; }

    public void Submit(CallOutputs* outputs, Completion completion)
    {
        using Lease index = handles.LendIndex();
        using Lease executor = handles.LendExecutor();
        using Lease lent = Arena.Lend(arena);
        NativeDenseSaveBuffer arguments = new()
        {
            Index = index.Pointer, Executor = executor.Pointer, Outcome = &outputs->Outcome, Completion = completion,
            Monolith = (NativeBytesView*)outputs->View, Arena = lent.Pointer,
        };
        usearch_dense_save_buffer(&arguments);
        this._generation = Arena.Rewound(arena, &outputs->Outcome);
    }

    public readonly void Release(CallOutputs* outputs) { }

    public void Finish(CallOutputs* outputs)
    {
        outputs->Outcome.ThrowIfFailed();
        this.Result = new Monolith(arena, this._generation, *(NativeBytesView*)outputs->View);
    }
}

/// <summary>
/// Holds a managed predicate the engine calls from its own worker threads, and the first exception
/// it threw. A managed exception must not unwind through native frames, so it is captured here and
/// rethrown once the call is final.
/// </summary>
internal sealed class Admitting(Func<Key, bool> admits)
{
    private ExceptionDispatchInfo? _escaped;

    /// <summary>
    /// The admission callback the engine calls from its own worker threads with a batch of
    /// candidates, reaching the managed predicate through the state pointer the search registered
    /// alongside it.
    /// </summary>
    /// <param name="keys">The candidate entries' identities.</param>
    /// <param name="admissions">Receives whether each key joins the matches.</param>
    /// <param name="count">How many keys the batch holds.</param>
    /// <param name="state">The registered trampoline, forwarded verbatim.</param>
    [UnmanagedCallersOnly(CallConvs = [typeof(CallConvCdecl)])]
    internal static unsafe void Admits(Key* keys, Admission* admissions, nuint count, IntPtr state)
    {
        Admitting? trampoline = GCHandle.FromIntPtr(state).Target as Admitting;
        for (nuint index = 0; index != count; ++index)
        {
            admissions[index] = trampoline is not null && trampoline.Admit(keys[index])
                ? Admission.Admit
                : Admission.Reject;
        }
    }

    /// <summary>Rethrows the first exception the predicate threw, with its own type and stack.</summary>
    internal void Rethrow() => this._escaped?.Throw();

    private bool Admit(Key key)
    {
        if (Volatile.Read(ref this._escaped) is not null)
        {
            return false;
        }

        try
        {
            return admits(key);
        }
        // Every exception is caught on purpose: this frame is called from a native worker
        // thread, and unwinding through native frames terminates the process.
#pragma warning disable CA1031
        catch (Exception failure)
#pragma warning restore CA1031
        {
            Interlocked.CompareExchange(ref this._escaped, ExceptionDispatchInfo.Capture(failure), null);
            return false;
        }
    }
}

/// <summary>
/// Resolves the keys an attention scored to rows of the caller's value matrix, checking the scalar
/// kind and width the library says it reads against the matrix's own, and capturing the first
/// exception the resolver threw as <see cref="Admitting"/> does.
/// </summary>
internal sealed unsafe class ValueRows(Func<Key, int> rowOf, RowsView values)
{
    private ExceptionDispatchInfo? _escaped;

    /// <summary>The first value row, pinned for the length of the call.</summary>
    internal byte* Base { get; set; }

    /// <summary>The <c>usearch_value_lookup_t</c> the engine calls once per scored batch.</summary>
    [UnmanagedCallersOnly(CallConvs = [typeof(CallConvCdecl)])]
    internal static void Resolve(IntPtr state, Key* keys, nuint count, Scalar scalar, nuint columns, void** rows) =>
        ((ValueRows)GCHandle.FromIntPtr(state).Target!).Fill(keys, count, scalar, columns, rows);

    /// <summary>Rethrows the first exception the resolver threw, with its own type and stack.</summary>
    internal void Rethrow() => this._escaped?.Throw();

    private void Fill(Key* keys, nuint count, Scalar scalar, nuint columns, void** rows)
    {
        // Rows of another shape than the library reads are no rows at all, which refuses the call.
        bool readable = scalar == values.Scalar && columns == (nuint)values.Dimensions;
        for (nuint position = 0; position < count; position++)
        {
            int row = readable ? this.RowOf(keys[position]) : -1;
            rows[position] = (uint)row < (uint)values.Rows ? this.Base + ((long)row * values.StrideBytes) : null;
        }
    }

    private int RowOf(Key key)
    {
        if (Volatile.Read(ref this._escaped) is not null)
        {
            return -1;
        }

        try
        {
            return rowOf(key);
        }
        // Every exception is caught on purpose, as in the admission predicate.
#pragma warning disable CA1031
        catch (Exception failure)
#pragma warning restore CA1031
        {
            Interlocked.CompareExchange(ref this._escaped, ExceptionDispatchInfo.Capture(failure), null);
            return -1;
        }
    }
}

#endregion Calls

#region Arena Views

/// <summary>Rows a call wrote packed into its <see cref="Arena"/> - the entries
/// <see cref="DenseIndex.Get"/> found under one key, or the rows <see cref="DenseIndex.Attend"/>
/// produced. Mirrors <c>usearch_rows_view_t</c> as an output.</summary>
/// <remarks>A view by value, readable until the arena's next call or disposal, as
/// <see cref="Matches"/> is.</remarks>
public readonly unsafe struct DenseVectors
{
    private readonly Arena _arena;
    private readonly int _generation;
    private readonly byte* _data;

    internal DenseVectors(Arena arena, int generation, NativeRowsView view, int rows)
    {
        this._arena = arena;
        this._generation = generation;
        this._data = (byte*)view.Data;
        this.Rows = rows;
        this.StrideBytes = (int)view.Stride;
        this.Columns = (int)view.Columns;
        this.Scalar = view.Scalar;
    }

    /// <summary>How many rows the call wrote; zero for an absent key.</summary>
    public int Rows { get; }

    /// <summary>The bytes between consecutive rows, sub-byte kinds included.</summary>
    public int StrideBytes { get; }

    /// <summary>The values in each row, as the library reported them.</summary>
    public int Columns { get; }

    /// <summary>The scalar type the rows hold.</summary>
    public Scalar Scalar { get; }

    /// <summary>Reads one row as values of a managed element type.</summary>
    /// <typeparam name="TScalar">The element type, <c>byte</c> for the packed kinds.</typeparam>
    /// <param name="row">The row, in the order the call wrote them.</param>
    /// <returns>The row, valid until the arena's next call or disposal.</returns>
    public ReadOnlySpan<TScalar> Row<TScalar>(int row) where TScalar : unmanaged
    {
        if ((uint)row >= (uint)this.Rows)
        {
            throw new ArgumentOutOfRangeException(nameof(row));
        }

        Arena.Readable(this._arena, this._generation);
        int values = (int)this.Scalar.RowBytes((ulong)this.Columns) / sizeof(TScalar);
        return new ReadOnlySpan<TScalar>(this._data + ((long)row * this.StrideBytes), values);
    }
}

/// <summary>The pairs <see cref="DenseIndex.Multiply"/> scored, ragged by left row, or the keys
/// <see cref="DenseIndex.Maxsim"/> kept, ragged by query, in its <see cref="Arena"/>. Mirrors
/// <c>usearch_matches_tape_view_t</c>.</summary>
/// <remarks>A view by value, readable until the arena's next call or disposal, as
/// <see cref="Matches"/> is.</remarks>
public readonly unsafe struct DenseProducts
{
    private readonly Arena _arena;
    private readonly int _generation;
    private readonly NativeMatchesTapeView _view;

    internal DenseProducts(Arena arena, int generation, NativeMatchesTapeView view, int queries, nuint required)
    {
        this._arena = arena;
        this._generation = generation;
        this._view = view;
        this.Queries = queries;
        this.Required = (ulong)required;
    }

    /// <summary>How many left rows were multiplied, or queries scored.</summary>
    public int Queries { get; }

    /// <summary>The pairs the walks scored across every left row, or the keys scored across every query,
    /// before the wanted bound cut any.</summary>
    public ulong Required { get; }

    /// <summary>The keys naming each indexed entry one left row reached, in traversal order, or the
    /// keys one query kept, largest score first.</summary>
    /// <param name="query">The left row or query.</param>
    /// <returns>The keys, valid until the arena's next call or disposal.</returns>
    public ReadOnlySpan<Key> Keys(int query) => new(this._view.Keys + this.Begin(query), this.Length(query));

    /// <summary>The products positionally paired with <see cref="Keys"/>.</summary>
    /// <param name="query">The left row or query.</param>
    /// <returns>The products, valid until the arena's next call or disposal.</returns>
    public ReadOnlySpan<double> Scores(int query) => new(this._view.Scores + this.Begin(query), this.Length(query));

    private nuint Begin(int query)
    {
        if ((uint)query >= (uint)this.Queries)
        {
            throw new ArgumentOutOfRangeException(nameof(query));
        }

        Arena.Readable(this._arena, this._generation);
        return this._view.Offsets[query];
    }

    private int Length(int query) => (int)(this._view.Offsets[query + 1] - this._view.Offsets[query]);
}

/// <summary>One serialized monolith <see cref="DenseIndex.SaveBuffer"/> wrote into its
/// <see cref="Arena"/>.</summary>
/// <remarks>A view by value, readable until the arena's next call or disposal, as
/// <see cref="Matches"/> is.</remarks>
public readonly unsafe struct Monolith
{
    private readonly Arena _arena;
    private readonly int _generation;
    private readonly byte* _data;

    internal Monolith(Arena arena, int generation, NativeBytesView view)
    {
        this._arena = arena;
        this._generation = generation;
        this._data = (byte*)view.Data;
        this.Length = (long)view.Length;
    }

    /// <summary>The monolith's length in bytes, which may exceed what one span addresses.</summary>
    public long Length { get; }

    /// <summary>The monolith's bytes, valid until the arena's next call or disposal; a monolith past
    /// 2 GiB refuses with <see cref="Status.Overflow"/>.</summary>
    public ReadOnlySpan<byte> Bytes
    {
        get
        {
            Arena.Readable(this._arena, this._generation);
            return this.Length <= int.MaxValue
                ? new ReadOnlySpan<byte>(this._data, (int)this.Length)
                : throw new USearchException(Status.Overflow, "the monolith exceeds what one span can address");
        }
    }
}

/// <summary>
/// The view a <see cref="DenseIndex.Sample"/> grows and how it draws; the default draws across the
/// whole index with nothing shown yet. Mirrors the inputs and knobs of <c>usearch_dense_sample_t</c>.
/// </summary>
public readonly record struct DenseSampleConfig
{
    /// <summary>The keys already shown: never drawn again, and the paths reach toward them.</summary>
    public ReadOnlyMemory<Key> Shown { get; init; }

    /// <summary>Where each of <see cref="Shown"/> sits, three values apiece, the points later draws
    /// are placed against.</summary>
    public ReadOnlyMemory<float> ShownCoordinates { get; init; }

    /// <summary>The keys to look around; empty looks across the whole index.</summary>
    public ReadOnlyMemory<Key> Focus { get; init; }

    /// <summary>The draw: equal seeds, shown keys and focus over an unchanged index draw equally.</summary>
    public ulong Seed { get; init; }

    /// <summary>The breadth of the walks behind the draws; zero derives it.</summary>
    public ulong ExpansionFactor { get; init; }

    /// <summary>The traversal telemetry to fill, when the caller wants any.</summary>
    public DenseTraversalStats? Stats { get; init; }
}

/// <summary>The draws one <see cref="DenseIndex.Sample"/> wrote into its <see cref="Arena"/>: the keys
/// drawn, which of its key's entries each is, the shown or earlier draw each hangs from, and where
/// each is placed.</summary>
/// <remarks>A view by value, readable until the arena's next call or disposal, as
/// <see cref="Matches"/> is.</remarks>
public readonly unsafe struct Sample
{
    private readonly Arena _arena;
    private readonly int _generation;
    private readonly Key* _keys;
    private readonly nuint* _entryOrdinals;
    private readonly NativeMatchesView _parents;
    private readonly float* _coordinates;

    internal Sample(
        Arena arena, int generation, NativeKeysView keys, nuint* entryOrdinals, NativeMatchesView parents,
        NativeRowsView coordinates, int count)
    {
        this._arena = arena;
        this._generation = generation;
        this._keys = (Key*)keys.Data;
        this._entryOrdinals = entryOrdinals;
        this._parents = parents;
        this._coordinates = (float*)coordinates.Data;
        this.Count = count;
    }

    /// <summary>How many draws the call wrote; fewer than wanted only once the region is exhausted.</summary>
    public int Count { get; }

    /// <summary>The keys drawn.</summary>
    public ReadOnlySpan<Key> Keys
    {
        get
        {
            Arena.Readable(this._arena, this._generation);
            return new ReadOnlySpan<Key>(this._keys, this.Count);
        }
    }

    /// <summary>Which of its key's entries each draw is, in the order <see cref="DenseIndex.Get"/>
    /// writes them.</summary>
    public ReadOnlySpan<nuint> EntryOrdinals
    {
        get
        {
            Arena.Readable(this._arena, this._generation);
            return new ReadOnlySpan<nuint>(this._entryOrdinals, this.Count);
        }
    }

    /// <summary>The one shown or earlier draw each draw hangs from and its distance, one query per
    /// draw.</summary>
    public Matches Parents => new(this._arena, this._generation, this._parents, this.Count);

    /// <summary>Where each draw is placed, three values apiece.</summary>
    public ReadOnlySpan<float> Coordinates
    {
        get
        {
            Arena.Readable(this._arena, this._generation);
            return new ReadOnlySpan<float>(this._coordinates, this.Count * 3);
        }
    }
}

/// <summary>One batch a <see cref="DenseCursor"/> pull wrote into its <see cref="Arena"/>: the keys,
/// and the keys each links to when the pull exported them; the vectors come from
/// <see cref="DenseIndex.Get"/>.</summary>
/// <remarks>A view by value, readable until the arena's next call or disposal, as
/// <see cref="Matches"/> is.</remarks>
public readonly unsafe struct DenseBatch
{
    private readonly Arena _arena;
    private readonly int _generation;
    private readonly Key* _keys;
    private readonly NativeKeysTapeView _links;
    private readonly bool _linked;

    internal DenseBatch(
        Arena arena, int generation, NativeKeysView keys, NativeKeysTapeView links, bool linked, int count)
    {
        this._arena = arena;
        this._generation = generation;
        this._keys = (Key*)keys.Data;
        this._links = links;
        this._linked = linked;
        this.Count = count;
    }

    /// <summary>How many entries the pull wrote; zero once the walk is exhausted.</summary>
    public int Count { get; }

    /// <summary>The identities of the entries, in storage order.</summary>
    public ReadOnlySpan<Key> Keys
    {
        get
        {
            Arena.Readable(this._arena, this._generation);
            return new ReadOnlySpan<Key>(this._keys, this.Count);
        }
    }

    /// <summary>The keys one entry links to, as the graph stored them when the pull ran.</summary>
    /// <param name="entry">The entry's position in <see cref="Keys"/>.</param>
    /// <returns>The keys, valid until the arena's next call or disposal.</returns>
    /// <exception cref="InvalidOperationException">The pull did not export links.</exception>
    public ReadOnlySpan<Key> Links(int entry)
    {
        if (!this._linked)
        {
            throw new InvalidOperationException("the pull exported no links");
        }

        if ((uint)entry >= (uint)this.Count)
        {
            throw new ArgumentOutOfRangeException(nameof(entry));
        }

        Arena.Readable(this._arena, this._generation);
        nuint begin = this._links.Offsets[entry];
        return new ReadOnlySpan<Key>(this._links.Keys + begin, (int)(this._links.Offsets[entry + 1] - begin));
    }
}

#endregion Arena Views

#region Cursors & Streams

/// <summary>
/// A sequential walk over the index contents - export, migration, and visualization feed off
/// it. Each shard's publication watermark is read as the walk enters that shard, so an entry once
/// passed is never revisited, while a later shard may still carry entries published after the walk
/// began. It keeps the index and the executor it pulls on alive until disposed.
/// </summary>
public sealed unsafe class DenseCursor : IDisposable
{
    private readonly NativeCursorHandle _handle;

    private DenseCursor(NativeCursorHandle handle) => this._handle = handle;

    internal static DenseCursor Open(DenseIndex index, Executor? executor, ReadOnlySpan<Key> starts)
    {
        Lease held = Lease.Of(index.Handle, "index");
        Lease pool;
        try
        {
            pool = Lease.Of(executor?.Handle, "executor");
        }
        catch
        {
            held.Dispose();
            throw;
        }

        IntPtr created = IntPtr.Zero;
        NativeOutcome outcome = default;
        fixed (Key* run = starts)
        {
            NativeDenseCursorInit arguments = new()
            {
                Cursor = &created,
                Index = held.Pointer,
                Outcome = &outcome,
                StartsCount = (nuint)starts.Length,
                Starts = run,
            };
            usearch_dense_cursor_init(&arguments);
        }

        if (outcome.Status != Status.Success)
        {
            held.Dispose();
            pool.Dispose();
            outcome.ThrowIfFailed();
        }

        return new DenseCursor(new NativeCursorHandle(created, held, pool));
    }

    /// <summary>Copies the next batch of entries out of the walk into an arena, the keys always and
    /// the keys each links to when asked for. Calls <c>usearch_dense_cursor_next</c>.</summary>
    /// <param name="wanted">The upper bound on entries the pull writes.</param>
    /// <param name="arena">Where the batch lands, rewound by the pull.</param>
    /// <param name="links">Whether to export the keys each entry links to; false exports the keys alone.</param>
    /// <returns>The batch, empty once the walk is exhausted.</returns>
    public DenseBatch Next(int wanted, Arena arena, bool links = false)
    {
        nuint bound = Calls.Bound(wanted);
        using Lease cursor = Lease.Of(this._handle, "cursor");
        using Lease lent = Arena.Lend(arena);
        nuint written = 0;
        NativeOutcome outcome = default;
        NativeKeysView keys = default;
        NativeKeysTapeView linked = default;
        NativeDenseCursorNext arguments = new()
        {
            Index = this._handle.Index,
            Executor = this._handle.Executor,
            Outcome = &outcome,
            Cursor = cursor.Pointer,
            Keys = &keys,
            Links = links ? &linked : null,
            Written = &written,
            Arena = lent.Pointer,
            Wanted = bound,
        };
        usearch_dense_cursor_next(&arguments);
        int generation = Arena.Rewound(arena, &outcome);
        outcome.ThrowIfFailed();
        int count = (int)written;
        return new DenseBatch(arena, generation, keys, linked, links, count);
    }

    /// <summary>Releases the cursor, then the index and executor it kept alive. Calls
    /// <c>usearch_dense_cursor_free</c>.</summary>
    public void Dispose() => this._handle.Dispose();
}

/// <summary>
/// A resumable nearest-neighbor stream: best-first traversal is naturally pausable, so the stream
/// keeps the walk's frontier and visited set between pulls and yields matches lazily. Callers that
/// post-filter aggressively pull until satisfied instead of guessing "k" upfront. It keeps the index
/// and the executor it pulls on alive until disposed.
/// </summary>
public sealed unsafe class DenseSearchStream : IDisposable
{
    private readonly NativeSearchStreamHandle _handle;

    private DenseSearchStream(NativeSearchStreamHandle handle) => this._handle = handle;

    internal static DenseSearchStream Open(DenseIndex index, Executor? executor, RowsView query, ulong expansionFactor)
    {
        if (query.Rows != 1)
        {
            throw new USearchException(Status.ShapeMismatch, "a search stream takes exactly one query row");
        }

        Lease held = Lease.Of(index.Handle, "index");
        Lease pool;
        try
        {
            pool = Lease.Of(executor?.Handle, "executor");
        }
        catch
        {
            held.Dispose();
            throw;
        }

        IntPtr created = IntPtr.Zero;
        NativeOutcome outcome = default;
        fixed (byte* row = query.Bytes)
        {
            NativeDenseSearchStreamInit arguments = new()
            {
                Stream = &created,
                Index = held.Pointer,
                Outcome = &outcome,
                Query = query.AsView(row),
                ExpansionFactor = (nuint)expansionFactor,
            };
            usearch_dense_search_stream_init(&arguments);
        }

        if (outcome.Status != Status.Success)
        {
            held.Dispose();
            pool.Dispose();
            outcome.ThrowIfFailed();
        }

        return new DenseSearchStream(new NativeSearchStreamHandle(created, held, pool));
    }

    /// <summary>
    /// Pulls the next-nearest matches from the stream into an arena. Ordering beyond the expansion
    /// horizon is best-effort nearest, not globally exact. Calls <c>usearch_dense_search_stream_next</c>.
    /// </summary>
    /// <param name="wanted">The upper bound on matches the pull writes.</param>
    /// <param name="arena">Where the pull lands, rewound by it.</param>
    /// <returns>The matches, none once the walk is drained.</returns>
    public QueryMatches Next(int wanted, Arena arena)
    {
        nuint bound = Calls.Bound(wanted);
        using Lease stream = Lease.Of(this._handle, "search stream");
        using Lease lent = Arena.Lend(arena);
        NativeOutcome outcome = default;
        NativeMatchesView matches = default;
        NativeDenseSearchStreamNext arguments = new()
        {
            Index = this._handle.Index,
            Executor = this._handle.Executor,
            Outcome = &outcome,
            Stream = stream.Pointer,
            Matches = &matches,
            Arena = lent.Pointer,
            Wanted = bound,
        };
        usearch_dense_search_stream_next(&arguments);
        int generation = Arena.Rewound(arena, &outcome);
        outcome.ThrowIfFailed();
        return new QueryMatches(arena, generation, matches.Keys, matches.Scores, matches.Counts);
    }

    /// <summary>Releases the stream and the walk state it keeps, then the index and executor it kept
    /// alive. Calls <c>usearch_dense_search_stream_free</c>.</summary>
    public void Dispose() => this._handle.Dispose();
}

#endregion Cursors & Streams

#region Native Vocabulary

/// <summary>The geometry and metric a checkpoint froze, laid out as the C struct spells them.</summary>
[StructLayout(LayoutKind.Sequential)]
internal struct NativeDenseMetadata
{
    /// <summary>The metric kind the checkpoint scores with.</summary>
    public DenseMetric Metric;
    /// <summary>The scalar type its vectors are stored and quantized into.</summary>
    public Scalar Quantization;
    /// <summary>The scalars per stored vector.</summary>
    public nuint Dimensions;
    /// <summary>The connections per node the graph was built with.</summary>
    public nuint Connectivity;
}

/// <summary>A live index's state, laid out as <c>usearch_dense_index_description_t</c>.</summary>
[StructLayout(LayoutKind.Sequential)]
internal struct NativeDenseDescription
{
    public NativeDenseMetadata Metadata;
    public nuint Size;
    public nuint Capacity;
    public nuint MemoryUsage;
    public int CpuClass;
    public int GpuClass;
    public nuint ShardsCount;
}

/// <summary>Traversal telemetry, fully overwritten by the call it is passed to.</summary>
[StructLayout(LayoutKind.Sequential)]
internal struct NativeDenseTraversalStats
{
    /// <summary>Graph members visited while resolving the batch.</summary>
    public nuint VisitedMembers;
    /// <summary>Distances computed while resolving the batch.</summary>
    public nuint ComputedDistances;
    /// <summary>Starvation restarts after the frontier drained prematurely.</summary>
    public nuint Restarts;
    /// <summary>Hint keys no shard held an entry under; zero unless a search was handed hints.</summary>
    public nuint MissingHints;
}

/// <summary>One reachability audit, laid out as <c>usearch_reachability_t</c>.</summary>
[StructLayout(LayoutKind.Sequential)]
internal struct NativeDenseReachability
{
    public nuint Audited;
    public nuint Reachable;
    public nuint Sweeps;
    public nuint BackSweeps;
}

/// <summary>What one refine-family call did to the graph, laid out as <c>usearch_refine_report_t</c>.</summary>
[StructLayout(LayoutKind.Sequential)]
internal struct NativeRefineReport
{
    public nuint Reselected;
    public nuint Changed;
    public nuint IslandsFound;
    public nuint IslandsReconnected;
    public NativeDenseReachability Reachability;
}

/// <summary>How a pass over the graph may spend, laid out as <c>usearch_refine_options_t</c>.</summary>
[StructLayout(LayoutKind.Sequential)]
internal struct NativeRefineOptions
{
    public nuint ExpansionFactor;
    public nuint MemoryLimit;
}

/// <summary>The arguments of <c>usearch_dense_init_ram</c>.</summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeDenseInitRam
{
    public IntPtr* Index;
    public NativeOutcome* Outcome;
    public DenseMetric Metric;
    public Scalar Quantization;
    public nuint Dimensions;
    public nuint Connectivity;
    public nuint ValuesPerKey;
    public NativeRamOptions Ram;
}

/// <summary>The arguments of <c>usearch_dense_init_vram</c>.</summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeDenseInitVram
{
    public IntPtr* Index;
    public NativeOutcome* Outcome;
    public DenseMetric Metric;
    public Scalar Quantization;
    public nuint Dimensions;
    public nuint Connectivity;
    public nuint ValuesPerKey;
    public NativeVramOptions Vram;
}

/// <summary>The arguments of <c>usearch_dense_load_ram</c>.</summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeDenseLoadRam
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

/// <summary>The arguments of <c>usearch_dense_load_vram</c>.</summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeDenseLoadVram
{
    public IntPtr* Index;
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
    public NativeBytesView Path;
    public NativeVramOptions Vram;
    public nuint ValuesPerKey;
    public NativeProgress Progress;
}

/// <summary>The arguments of <c>usearch_dense_load_ram_buffer</c>.</summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeDenseLoadRamBuffer
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

/// <summary>The arguments of <c>usearch_dense_load_vram_buffer</c>.</summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeDenseLoadVramBuffer
{
    public IntPtr* Index;
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
    public NativeBytesView Buffer;
    public NativeVramOptions Vram;
    public nuint ValuesPerKey;
    public NativeProgress Progress;
}

/// <summary>The arguments of <c>usearch_dense_view_ram</c>.</summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeDenseViewRam
{
    public IntPtr* Index;
    public NativeOutcome* Outcome;
    public NativeBytesView Path;
}

/// <summary>The arguments of <c>usearch_dense_view_nvme</c>.</summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeDenseViewNvme
{
    public IntPtr* Index;
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
    public NativeBytesView Path;
    public NativeNvmeOptions Nvme;
    public NativeRamOptions* Ram;
}

/// <summary>The arguments of <c>usearch_dense_reserve</c>.</summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeDenseReserve
{
    public IntPtr Index;
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
    public nuint Capacity;
}

/// <summary>The arguments of <c>usearch_dense_clear</c>.</summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeDenseClear
{
    public IntPtr Index;
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
}

/// <summary>The arguments of <c>usearch_dense_insert</c>.</summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeDenseInsert
{
    public IntPtr Index;
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
    public nuint Count;
    public NativeKeysView Keys;
    public NativeRowsView Vectors;
    public nuint ExpansionFactor;
    public NativeProgress Progress;
    public NativeDenseTraversalStats* Stats;
}

/// <summary>The arguments of <c>usearch_dense_refine</c>.</summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeDenseRefine
{
    public IntPtr Index;
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
    public NativeRefineReport* Report;
    public NativeRefineOptions Refine;
    public NativeProgress Progress;
    public NativeDenseTraversalStats* Stats;
}

/// <summary>The arguments of <c>usearch_dense_health</c>.</summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeDenseHealth
{
    public IntPtr Index;
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
    public NativeDenseReachability* Reachability;
}

/// <summary>The arguments of <c>usearch_dense_search</c>.</summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeDenseSearch
{
    public IntPtr Index;
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
    public nuint Count;
    public NativeRowsView Queries;
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

/// <summary>The arguments of <c>usearch_dense_count</c>.</summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeDenseCount
{
    public IntPtr Index;
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
    public Key Key;
    public nuint* Entries;
}

/// <summary>The arguments of <c>usearch_dense_get</c>.</summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeDenseGet
{
    public IntPtr Index;
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
    public Key Key;
    public NativeRowsView* Vectors;
    public nuint* Written;
    public IntPtr Arena;
    public Scalar OutputsScalar;
}

/// <summary>The arguments of <c>usearch_dense_absorb</c>.</summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeDenseAbsorb
{
    public IntPtr Index;
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
    public nuint Count;
    public IntPtr* Donors;
    public NativeRefineReport* Report;
    public NativeRefineOptions Refine;
    public NativeProgress Progress;
    public NativeDenseTraversalStats* Stats;
}

/// <summary>The arguments of <c>usearch_dense_split</c>.</summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeDenseSplit
{
    public IntPtr Index;
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
    public nuint Count;
    public IntPtr* Receivers;
    public NativeRefineReport* Report;
    public NativeRefineOptions Refine;
    public ulong Seed;
    public NativeProgress Progress;
    public NativeDenseTraversalStats* Stats;
}

/// <summary>The arguments of <c>usearch_dense_sample</c>.</summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeDenseSample
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

/// <summary>The arguments of <c>usearch_dense_cursor_init</c>.</summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeDenseCursorInit
{
    public IntPtr Index;
    public NativeOutcome* Outcome;
    public nuint StartsCount;
    public Key* Starts;
    public IntPtr* Cursor;
}

/// <summary>The arguments of <c>usearch_dense_cursor_next</c>.</summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeDenseCursorNext
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

/// <summary>The arguments of <c>usearch_dense_search_stream_init</c>.</summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeDenseSearchStreamInit
{
    public IntPtr* Stream;
    public IntPtr Index;
    public NativeOutcome* Outcome;
    public NativeRowsView Query;
    public nuint ExpansionFactor;
}

/// <summary>The arguments of <c>usearch_dense_search_stream_next</c>.</summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeDenseSearchStreamNext
{
    public IntPtr Index;
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
    public IntPtr Stream;
    public NativeMatchesView* Matches;
    public IntPtr Arena;
    public nuint Wanted;
}

/// <summary>The arguments of <c>usearch_dense_save</c>.</summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeDenseSave
{
    public IntPtr Index;
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
    public NativeBytesView Path;
    public NativeProgress Progress;
}

/// <summary>The arguments of <c>usearch_dense_save_buffer</c>.</summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeDenseSaveBuffer
{
    public IntPtr Index;
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
    public NativeBytesView* Monolith;
    public IntPtr Arena;
    public NativeProgress Progress;
}

/// <summary>The arguments of <c>usearch_dense_metadata_read</c>.</summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeDenseMetadataRead
{
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
    public NativeBytesView Path;
    public NativeDenseMetadata* Metadata;
}

/// <summary>The arguments of <c>usearch_dense_metadata_read_buffer</c>.</summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeDenseMetadataReadBuffer
{
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
    public NativeBytesView Buffer;
    public NativeDenseMetadata* Metadata;
}

/// <summary>The arguments of <c>usearch_dense_describe</c>.</summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeDenseDescribe
{
    public IntPtr Index;
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
    public NativeDenseDescription* Description;
}

/// <summary>The arguments of <c>usearch_dense_multiply</c>.</summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeDenseMultiply
{
    public IntPtr Index;
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
    public nuint Count;
    public NativeRowsView Left;
    public NativeMatchesTapeView* Products;
    public nuint* Written;
    public nuint* Required;
    public IntPtr Arena;
    public nuint Wanted;
    public nuint ExpansionFactor;
    public Traversal Traversal;
    public NativePredicate Predicate;
    public NativeProgress Progress;
    public NativeDenseTraversalStats* Stats;
}

/// <summary>The arguments of <c>usearch_dense_attend</c>.</summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeDenseAttend
{
    public IntPtr Index;
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
    public nuint Count;
    public NativeRowsView Queries;
    public nuint HeadCount;
    public delegate* unmanaged[Cdecl]<IntPtr, Key*, nuint, Scalar, nuint, void**, void> ValueLookup;
    public IntPtr ValueLookupState;
    public DenseSoftmaxState* Carry;
    public NativeRowsView* Outputs;
    public IntPtr Arena;
    public Scalar OutputsScalar;
    public nuint ExpansionFactor;
    public float ScoreScale;
    public Traversal Traversal;
    public NativePredicate Predicate;
    public NativeProgress Progress;
    public NativeDenseTraversalStats* Stats;
}

/// <summary>The arguments of <c>usearch_dense_maxsim</c>.</summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeDenseMaxsim
{
    public IntPtr Index;
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
    public nuint Count;
    public NativeRowsView Queries;
    public nuint* QueryOffsets;
    public nuint RowsPerQuery;
    public NativeMatchesTapeView* Matches;
    public nuint* Written;
    public nuint* Required;
    public IntPtr Arena;
    public nuint Wanted;
    public nuint ExpansionFactor;
    public Traversal Traversal;
    public NativePredicate Predicate;
    public NativeMatchesView* Hints;
    public NativeProgress Progress;
    public NativeDenseTraversalStats* Stats;
}

#endregion Native Vocabulary

#region Entry Points

/// <summary>
/// The dense modality's entry points. Every verb takes one argument struct and reports through the
/// outcome it names; this binding exposes no progress handler, so the handler crosses as null.
/// </summary>
internal static unsafe partial class NativeMethods
{
    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial IntPtr usearch_dense_metric_name(DenseMetric metric);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial DenseMetric usearch_dense_metric_from_name(NativeBytesView name);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_dense_init_ram(NativeDenseInitRam* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_dense_init_vram(NativeDenseInitVram* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_dense_load_ram(NativeDenseLoadRam* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_dense_load_vram(NativeDenseLoadVram* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_dense_load_ram_buffer(NativeDenseLoadRamBuffer* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_dense_load_vram_buffer(NativeDenseLoadVramBuffer* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_dense_view_ram(NativeDenseViewRam* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_dense_view_nvme(NativeDenseViewNvme* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_dense_free(IntPtr index);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_dense_reserve(NativeDenseReserve* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_dense_clear(NativeDenseClear* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_dense_describe(NativeDenseDescribe* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_dense_insert(NativeDenseInsert* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_dense_refine(NativeDenseRefine* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_dense_health(NativeDenseHealth* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_dense_search(NativeDenseSearch* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_dense_maxsim(NativeDenseMaxsim* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_dense_multiply(NativeDenseMultiply* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_dense_attend(NativeDenseAttend* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_dense_absorb(NativeDenseAbsorb* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_dense_split(NativeDenseSplit* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_dense_sample(NativeDenseSample* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_dense_count(NativeDenseCount* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_dense_get(NativeDenseGet* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_dense_cursor_init(NativeDenseCursorInit* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_dense_cursor_next(NativeDenseCursorNext* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_dense_cursor_free(IntPtr cursor);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_dense_search_stream_init(NativeDenseSearchStreamInit* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_dense_search_stream_next(NativeDenseSearchStreamNext* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_dense_search_stream_free(IntPtr stream);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_dense_save(NativeDenseSave* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_dense_save_buffer(NativeDenseSaveBuffer* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_dense_metadata_read(NativeDenseMetadataRead* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_dense_metadata_read_buffer(NativeDenseMetadataReadBuffer* arguments);
}

#endregion Entry Points

#region Handles

/// <summary>A <c>usearch_dense_index_t</c>, freed when the wrapper is disposed or collected.</summary>
internal sealed class NativeDenseIndexHandle : EngineHandle
{
    /// <inheritdoc/>
    protected override bool ReleaseHandle()
    {
        NativeMethods.usearch_dense_free(this.handle);
        return true;
    }
}

/// <summary>A <c>usearch_dense_cursor_t</c> and the leases on the index and executor every pull
/// names, all released when the wrapper is disposed or collected.</summary>
internal sealed class NativeCursorHandle : EngineHandle
{
    private Lease _index;
    private Lease _executor;

    internal NativeCursorHandle(IntPtr cursor, Lease index, Lease executor)
    {
        this.SetHandle(cursor);
        this._index = index;
        this._executor = executor;
    }

    /// <summary>The index the cursor was opened on.</summary>
    internal IntPtr Index => this._index.Pointer;

    /// <summary>The executor every pull runs on, null for the caller's thread.</summary>
    internal IntPtr Executor => this._executor.Pointer;

    /// <inheritdoc/>
    protected override bool ReleaseHandle()
    {
        NativeMethods.usearch_dense_cursor_free(this.handle);
        this._index.Dispose();
        this._executor.Dispose();
        return true;
    }
}

/// <summary>A <c>usearch_dense_search_stream_t</c> and the leases on the index and executor every
/// pull names, all released when the wrapper is disposed or collected.</summary>
internal sealed class NativeSearchStreamHandle : EngineHandle
{
    private Lease _index;
    private Lease _executor;

    internal NativeSearchStreamHandle(IntPtr stream, Lease index, Lease executor)
    {
        this.SetHandle(stream);
        this._index = index;
        this._executor = executor;
    }

    /// <summary>The index the stream was opened on.</summary>
    internal IntPtr Index => this._index.Pointer;

    /// <summary>The executor every pull runs on, null for the caller's thread.</summary>
    internal IntPtr Executor => this._executor.Pointer;

    /// <inheritdoc/>
    protected override bool ReleaseHandle()
    {
        NativeMethods.usearch_dense_search_stream_free(this.handle);
        this._index.Dispose();
        this._executor.Dispose();
        return true;
    }
}

#endregion Handles
