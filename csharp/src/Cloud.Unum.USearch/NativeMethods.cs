using System;
using System.Runtime.InteropServices;

namespace Cloud.Unum.USearch;

/// <summary>
/// The outcome slot one call writes: a status every caller can branch on, and a static message
/// carrying the detail the status cannot. Messages are static library storage, so nothing here
/// is deallocated.
/// </summary>
/// <remarks>Raising the failure it holds lives beside <see cref="Status"/>.</remarks>
[StructLayout(LayoutKind.Sequential)]
internal partial struct NativeError
{
    /// <summary>What went wrong, or <see cref="Status.Success"/>.</summary>
    public Status Status;
    /// <summary>The static message, null exactly when the status is success.</summary>
    public IntPtr Message;
}

/// <summary>
/// A borrowed run of bytes and its length, never null-terminated by contract - the shape every
/// path argument arrives in.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
internal struct NativeString
{
    /// <summary>The first byte of the run.</summary>
    public IntPtr Data;
    /// <summary>How many bytes the run spans.</summary>
    public nuint Length;
}

/// <summary>RAM-residency behavior: CPU threading and memory backing of the volatile index.</summary>
[StructLayout(LayoutKind.Sequential)]
internal struct NativeRamOptions
{
    /// <summary>Worker threads for construction and batched search; zero derives.</summary>
    public nuint Threads;
    /// <summary>Non-zero demands a provisioned huge-page pool.</summary>
    public byte HugePages;
}

/// <summary>VRAM-residency behavior: device selection for the GPU-traversed index.</summary>
[StructLayout(LayoutKind.Sequential)]
internal struct NativeVramOptions
{
    /// <summary>CUDA device ordinal; negative picks the first free device.</summary>
    public int Device;
    /// <summary>Non-zero allows spilling cold shards into host memory.</summary>
    public byte UnifiedAddressing;
}

/// <summary>NVMe-residency behavior: the I/O engine the handle traverses checkpoints with.</summary>
[StructLayout(LayoutKind.Sequential)]
internal struct NativeNvmeOptions
{
    /// <summary>I/O granularity in bytes; zero probes the device.</summary>
    public uint SectorBytes;
    /// <summary>Submission ring depth; zero derives from the device.</summary>
    public uint QueueDepth;
    /// <summary>Non-zero bypasses the page cache.</summary>
    public byte DirectIO;
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

/// <summary>
/// The construction and loading options a dense handle is built from, laid out as the C struct
/// spells them.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
internal struct NativeDenseOptions
{
    /// <summary>The metric kind used for distance calculations between vectors.</summary>
    public DenseMetric Metric;
    /// <summary>The custom metric callback, which this binding leaves null.</summary>
    public IntPtr MetricCallback;
    /// <summary>The opaque state forwarded into every callback, which stays null too.</summary>
    public IntPtr MetricCallbackState;
    /// <summary>The scalar type vectors are stored and quantized into.</summary>
    public ScalarKind Quantization;
    /// <summary>The width of the indexed vectors; zero defers the geometry to a later load.</summary>
    public nuint Dimensions;
    /// <summary>Non-zero rounds stored rows up to the SIMD step quantum.</summary>
    public byte PadDimensions;
    /// <summary>The connections-per-node bound; zero derives.</summary>
    public nuint Connectivity;
    /// <summary>Selects the active arm of <see cref="ResidencyOptions"/>.</summary>
    public ResidencyKind Residency;
    /// <summary>The arm <see cref="Residency"/> selects.</summary>
    public NativeResidencyOptions ResidencyOptions;
}

/// <summary>Optional per-call settings, shared by insertion and both search forms.</summary>
[StructLayout(LayoutKind.Sequential)]
internal struct NativeExpansionConfig
{
    /// <summary>The candidate-pool breadth; zero derives.</summary>
    public nuint ExpansionFactor;
}

/// <summary>Traversal telemetry, fully overwritten by the call it is passed to.</summary>
[StructLayout(LayoutKind.Sequential)]
internal struct NativeTraversalStats
{
    /// <summary>Graph members visited while resolving the batch.</summary>
    public nuint VisitedMembers;
    /// <summary>Distances computed while resolving the batch.</summary>
    public nuint ComputedDistances;
    /// <summary>Starvation restarts after the frontier drained prematurely.</summary>
    public nuint Restarts;
}

/// <summary>Per-key admission callback; returns non-zero to admit the key.</summary>
/// <param name="key">The candidate entry's identity.</param>
/// <param name="state">The opaque pointer registered alongside the callback.</param>
/// <returns>Non-zero to admit, zero to reject.</returns>
[UnmanagedFunctionPointer(CallingConvention.Cdecl)]
internal delegate int NativePredicate(USearchKey key, IntPtr state);

/// <summary>
/// The dense-modality C surface of <c>include/usearch/dense_index.h</c>, plus the three probes
/// of <c>include/usearch/types.h</c>.
/// </summary>
internal static unsafe class NativeMethods
{
    private const string LibraryName = "libusearch_c";

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern IntPtr usearch_version();

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern IntPtr usearch_capabilities_compiled();

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern IntPtr usearch_capabilities_available();

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern IntPtr usearch_dense_init(NativeDenseOptions* options, out NativeError error);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern void usearch_dense_free(IntPtr index);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern void usearch_dense_reserve(IntPtr index, nuint capacity, out NativeError error);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern void usearch_dense_clear(IntPtr index, out NativeError error);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern nuint usearch_dense_size(IntPtr index, out NativeError error);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern nuint usearch_dense_capacity(IntPtr index, out NativeError error);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern nuint usearch_dense_dimensions(IntPtr index, out NativeError error);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern nuint usearch_dense_connectivity(IntPtr index, out NativeError error);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern nuint usearch_dense_memory_usage(IntPtr index, out NativeError error);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern IntPtr usearch_dense_capabilities(IntPtr index, out NativeError error);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern nuint usearch_dense_add(
        IntPtr index,
        USearchKey* keys,
        void* vectors,
        nuint count,
        nuint stride,
        ScalarKind scalarKind,
        NativeExpansionConfig* config,
        NativeTraversalStats* stats,
        out NativeError error);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern void usearch_dense_search(
        IntPtr index,
        void* queries,
        nuint queriesCount,
        nuint queriesStride,
        ScalarKind scalarKind,
        nuint wanted,
        NativeExpansionConfig* config,
        NativeTraversalStats* stats,
        USearchKey* keys,
        nuint keysStride,
        double* distances,
        nuint distancesStride,
        nuint* counts,
        out NativeError error);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern void usearch_dense_search_if(
        IntPtr index,
        void* queries,
        nuint queriesCount,
        nuint queriesStride,
        ScalarKind scalarKind,
        nuint wanted,
        NativeExpansionConfig* config,
        NativeTraversalStats* stats,
        NativePredicate predicate,
        IntPtr predicateState,
        USearchKey* keys,
        nuint keysStride,
        double* distances,
        nuint distancesStride,
        nuint* counts,
        out NativeError error);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern nuint usearch_dense_count(IntPtr index, USearchKey key, out NativeError error);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern nuint usearch_dense_get(
        IntPtr index,
        USearchKey key,
        void* vectors,
        nuint wanted,
        nuint stride,
        ScalarKind scalarKind,
        out NativeError error);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern void usearch_dense_merge(IntPtr index, IntPtr donor, out NativeError error);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern IntPtr usearch_dense_cursor(IntPtr index, out NativeError error);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern nuint usearch_dense_cursor_next(
        IntPtr cursor,
        USearchKey* keys,
        void* vectors,
        nuint wanted,
        nuint stride,
        ScalarKind scalarKind,
        out NativeError error);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern void usearch_dense_cursor_free(IntPtr cursor);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern IntPtr usearch_dense_find(
        IntPtr index, void* query, ScalarKind scalarKind, out NativeError error);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern nuint usearch_dense_find_next(
        IntPtr finder, USearchKey* keys, double* distances, nuint wanted, out NativeError error);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern void usearch_dense_find_free(IntPtr finder);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern nuint usearch_dense_serialized_length(IntPtr index, out NativeError error);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern void usearch_dense_save(IntPtr index, NativeString path, out NativeError error);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern void usearch_dense_load(IntPtr index, NativeString path, out NativeError error);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern void usearch_dense_metadata(
        NativeString path, NativeDenseOptions* options, out NativeError error);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern void usearch_dense_save_buffer(
        IntPtr index, void* buffer, nuint length, out NativeError error);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern void usearch_dense_load_buffer(
        IntPtr index, void* buffer, nuint length, out NativeError error);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    internal static extern void usearch_dense_metadata_buffer(
        void* buffer, nuint length, NativeDenseOptions* options, out NativeError error);
}
