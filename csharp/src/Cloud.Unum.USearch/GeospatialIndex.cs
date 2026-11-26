/**
 *  @file csharp/src/Cloud.Unum.USearch/GeospatialIndex.cs
 *  @author Ash Vardanian
 *  @date November 16, 2026
 *  @brief The geospatial index handle, its observations, relations and terrain, and the verbs that run over it.
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
/// Which entries a geospatial query admits, every one scored by chord metres. Mirrors
/// <c>usearch_geospatial_relation_t</c>.
/// </summary>
public enum GeospatialRelation : int
{
    /// <summary>A relation the library could not name.</summary>
    Unknown = 0,
    /// <summary>Within the radius, widened by three standard deviations of both accuracies.</summary>
    Near = 1,
    /// <summary>Near, with a clear line of sight over the handle's terrain.</summary>
    Visible = 2,
    /// <summary>The entry's view cone holds the query: which cameras see this place.</summary>
    Sees = 3,
    /// <summary>The query's view cone holds the entry: what this camera sees.</summary>
    Seen = 4,
}

/// <summary>
/// One observation on or above the Earth, in natural units; a field no device recorded stays NaN,
/// never zero, and only widens the tests that read it. Mirrors <c>usearch_geospatial_observation_t</c>.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct GeospatialObservation
{
    /// <summary>WGS-84 latitude in degrees.</summary>
    public double LatitudeDegrees;
    /// <summary>WGS-84 longitude in degrees.</summary>
    public double LongitudeDegrees;
    /// <summary>Metres above the ellipsoid.</summary>
    public double AltitudeMetres;
    /// <summary>The horizontal accuracy, one standard deviation in metres.</summary>
    public double AccuracyMetres;
    /// <summary>The view cone's heading, clockwise from north.</summary>
    public double HeadingDegrees;
    /// <summary>The view cone's pitch above the horizon.</summary>
    public double PitchDegrees;
    /// <summary>The heading's accuracy.</summary>
    public double HeadingAccuracyDegrees;
    /// <summary>The view cone's full angle.</summary>
    public double FieldOfViewDegrees;
    /// <summary>How far the device saw.</summary>
    public double RangeMetres;
    /// <summary>The first Unix second the observation covers.</summary>
    public double BeginSeconds;
    /// <summary>The last Unix second the observation covers.</summary>
    public double EndSeconds;

    /// <summary>An observation of a position alone, every other field unrecorded.</summary>
    /// <param name="latitudeDegrees">WGS-84 latitude in degrees.</param>
    /// <param name="longitudeDegrees">WGS-84 longitude in degrees.</param>
    /// <returns>The observation.</returns>
    public static GeospatialObservation At(double latitudeDegrees, double longitudeDegrees) => new()
    {
        LatitudeDegrees = latitudeDegrees, LongitudeDegrees = longitudeDegrees, AltitudeMetres = double.NaN,
        AccuracyMetres = double.NaN, HeadingDegrees = double.NaN, PitchDegrees = double.NaN,
        HeadingAccuracyDegrees = double.NaN, FieldOfViewDegrees = double.NaN, RangeMetres = double.NaN,
        BeginSeconds = double.NaN, EndSeconds = double.NaN,
    };
}

/// <summary>How far a relation reaches: the radius in metres, and the seconds widening the query's
/// interval on both ends. Mirrors <c>usearch_geospatial_tolerance_t</c>.</summary>
/// <param name="Metres">The radius; zero ranks by distance alone in a search.</param>
/// <param name="Seconds">The widening of the query's time interval.</param>
[StructLayout(LayoutKind.Sequential)]
public readonly record struct GeospatialTolerance(double Metres, double Seconds = 0);

/// <summary>
/// The elevation lines of sight are tested over: a north-up grid of heights in metres above the
/// WGS-84 ellipsoid, viewed by every handle built over it, which keeps it alive. Mirrors
/// <c>usearch_geospatial_terrain_t</c> with viewed heights or a dataset path.
/// </summary>
/// <remarks>The heights live in pinned memory, since the library reads them for the handle's whole
/// lifetime; a checkpoint never holds them, so every load and view names the terrain again.</remarks>
public sealed unsafe class GeospatialTerrain
{
    private readonly float[] _heights;
    private readonly byte[] _path;

    /// <summary>Allocates a grid of NaN heights for the caller to fill through <see cref="Heights"/>.</summary>
    /// <param name="rows">The rows, running south from the north edge.</param>
    /// <param name="columns">The columns, running east from the west edge.</param>
    /// <param name="northLatitudeDegrees">The latitude of the centre of cell [0,0].</param>
    /// <param name="westLongitudeDegrees">The longitude of the centre of cell [0,0].</param>
    /// <param name="rowStepDegrees">The step between rows, positive.</param>
    /// <param name="columnStepDegrees">The step between columns, positive.</param>
    public GeospatialTerrain(
        int rows, int columns, double northLatitudeDegrees, double westLongitudeDegrees, double rowStepDegrees,
        double columnStepDegrees)
    {
        this.Rows = rows;
        this.Columns = columns;
        this.NorthLatitudeDegrees = northLatitudeDegrees;
        this.WestLongitudeDegrees = westLongitudeDegrees;
        this.RowStepDegrees = rowStepDegrees;
        this.ColumnStepDegrees = columnStepDegrees;
        this._heights = GC.AllocateArray<float>(checked(rows * columns), pinned: true);
        this._heights.AsSpan().Fill(float.NaN);
        this._path = [];
    }

    private GeospatialTerrain(string path)
    {
        this._heights = [];
        this._path = GC.AllocateArray<byte>(Encoding.UTF8.GetByteCount(path), pinned: true);
        Encoding.UTF8.GetBytes(path, this._path);
    }

    /// <summary>Names a terrain dataset on disk, which only an NVMe residency will read; RAM and VRAM
    /// handles refuse it with <see cref="Status.Unsupported"/>.</summary>
    /// <param name="path">The dataset.</param>
    /// <returns>The terrain.</returns>
    public static GeospatialTerrain FromPath(string path) => new(path);

    /// <summary>The grid's rows.</summary>
    public int Rows { get; }

    /// <summary>The grid's columns.</summary>
    public int Columns { get; }

    /// <summary>The latitude of the centre of cell [0,0], in degrees.</summary>
    public double NorthLatitudeDegrees { get; }

    /// <summary>The longitude of the centre of cell [0,0], in degrees.</summary>
    public double WestLongitudeDegrees { get; }

    /// <summary>The step between rows, in degrees.</summary>
    public double RowStepDegrees { get; }

    /// <summary>The step between columns, in degrees.</summary>
    public double ColumnStepDegrees { get; }

    /// <summary>The whole grid, row-major, for the caller to fill before building a handle over it.</summary>
    public Span<float> Heights => this._heights;

    /// <summary>Lowers an optional terrain; null lowers the zeroed one that means none.</summary>
    /// <param name="terrain">The terrain, if any.</param>
    /// <returns>The C struct the library reads.</returns>
    internal static NativeGeospatialTerrain Lower(GeospatialTerrain? terrain) => terrain is null
        ? default
        : new NativeGeospatialTerrain
        {
            Rows = (nuint)terrain.Rows, Columns = (nuint)terrain.Columns,
            NorthLatitudeDegrees = terrain.NorthLatitudeDegrees, WestLongitudeDegrees = terrain.WestLongitudeDegrees,
            RowStepDegrees = terrain.RowStepDegrees, ColumnStepDegrees = terrain.ColumnStepDegrees,
            Heights = terrain._heights.Length == 0
                ? null
                : (float*)Unsafe.AsPointer(ref MemoryMarshal.GetArrayDataReference(terrain._heights)),
            Path = new NativeBytesView
            {
                Data = terrain._path.Length == 0
                    ? IntPtr.Zero
                    : (IntPtr)Unsafe.AsPointer(ref MemoryMarshal.GetArrayDataReference(terrain._path)),
                Length = (nuint)terrain._path.Length,
            },
        };
}

/// <summary>
/// The construction options of a geospatial index - the fields <c>usearch_geospatial_init_ram_t</c>
/// and <c>usearch_geospatial_init_vram_t</c> share.
/// </summary>
public readonly struct GeospatialOptions
{
    /// <summary>The terrain lines of sight are tested over; null tests none.</summary>
    public GeospatialTerrain? Terrain { get; init; }

    /// <summary>The connections-per-node bound of the graph; zero derives.</summary>
    public ulong Connectivity { get; init; }

    /// <summary>The upper bound on observations sharing one key; zero is unbounded.</summary>
    public ulong ValuesPerKey { get; init; }
}

/// <summary>The connectivity a checkpoint froze. Mirrors <c>usearch_geospatial_metadata_t</c>.</summary>
public readonly struct GeospatialMetadata
{
    /// <summary>The connections per node the graph was built with.</summary>
    public ulong Connectivity { get; init; }

    internal static GeospatialMetadata Lift(in NativeGeospatialMetadata lowered) =>
        new() { Connectivity = lowered.Connectivity };
}

/// <summary>A live geospatial index's frozen metadata and current counters, read in one call. Mirrors
/// <c>usearch_geospatial_index_description_t</c>.</summary>
public readonly struct GeospatialIndexDescription
{
    /// <summary>The frozen connectivity.</summary>
    public GeospatialMetadata Metadata { get; init; }

    /// <summary>The stored entries.</summary>
    public ulong Size { get; init; }

    /// <summary>The entries hosted without growing again.</summary>
    public ulong Capacity { get; init; }

    /// <summary>The bytes the handle holds, observations and the terrain pack included.</summary>
    public ulong MemoryUsage { get; init; }

    /// <summary>The CPU class dispatch picked on this machine; "unknown" on VRAM.</summary>
    public string CpuClass { get; init; }

    /// <summary>The GPU class dispatch picked on the index's devices; "unknown" for a host index.</summary>
    public string GpuClass { get; init; }

    /// <summary>The live shards.</summary>
    public ulong ShardsCount { get; init; }
}

/// <summary>The observations a <see cref="GeospatialIndex.Get"/> wrote into its <see cref="Arena"/>,
/// within the stored rounding.</summary>
/// <remarks>A view by value, readable until the arena's next call or disposal, as
/// <see cref="Matches"/> is.</remarks>
public readonly unsafe struct GeospatialObservations
{
    private readonly GeospatialObservation* _observations;
    private readonly ArenaDescription _description;

    internal GeospatialObservations(GeospatialObservation* observations, int count, ArenaDescription description)
    {
        this._observations = observations;
        this.Count = observations is null ? 0 : count;
        this._description = description;
    }

    /// <summary>How many observations the call wrote; zero for an absent key.</summary>
    public int Count { get; }

    /// <summary>The observations, in insertion order.</summary>
    public ReadOnlySpan<GeospatialObservation> Observations
    {
        get
        {
            this._description.Host();
            return new(this._observations, this.Count);
        }
    }
}

#endregion Options

#region Index

/// <summary>
/// A Similarity Search Index over observations on and above the Earth - photos, video frames, drone
/// footage and sensor fixes - scored by chord metres under a relation and a tolerance.
/// </summary>
/// <remarks>
/// Every batch is a run of <see cref="GeospatialObservation"/>, so a single observation is a run of one.
/// The receivers are the dense index's: the index runs a verb on the caller's thread, <see cref="On"/>
/// blocks on an executor, and <see cref="Deferred"/> submits there and answers a
/// <see cref="Deferred{TResult}"/>. Every callback runs on the engine's threads and must not throw.
/// </remarks>
public sealed unsafe class GeospatialIndex : IDisposable
{
    private readonly NativeGeospatialIndexHandle _handle;

    private GeospatialIndex(NativeGeospatialIndexHandle handle, GeospatialTerrain? terrain)
    {
        this._handle = handle;
        this.Terrain = terrain;
    }

    /// <summary>The terrain the handle tests lines of sight over, kept alive while the handle views it.</summary>
    public GeospatialTerrain? Terrain { get; }

    /// <summary>The engine handle the verbs reference.</summary>
    internal NativeGeospatialIndexHandle Handle => this._handle;

    /// <summary>Creates an empty, mutable geospatial index in host memory. Calls
    /// <c>usearch_geospatial_init_ram</c>.</summary>
    /// <param name="options">The terrain and the graph's bounds.</param>
    /// <param name="memoryDomains">NUMA node IDs to place one shard on apiece; empty derives.</param>
    /// <param name="cpuClassCeiling">The newest class dispatch may pick, intersected with what the
    /// machine runs; null means no ceiling.</param>
    /// <returns>The new index.</returns>
    public static GeospatialIndex InitRam(
        GeospatialOptions options, ReadOnlySpan<int> memoryDomains = default, CPUClass? cpuClassCeiling = null)
    {
        IntPtr created = IntPtr.Zero;
        NativeOutcome outcome = default;
        fixed (int* domains = memoryDomains)
        {
            NativeGeospatialInitRam arguments = new()
            {
                Index = &created, Outcome = &outcome, Terrain = GeospatialTerrain.Lower(options.Terrain),
                Connectivity = (nuint)options.Connectivity, ValuesPerKey = (nuint)options.ValuesPerKey,
                Ram = new NativeRamOptions
                {
                    MemoryDomains = (IntPtr)domains, MemoryDomainsCount = (nuint)memoryDomains.Length,
                    CpuClassCeiling = (int)(cpuClassCeiling ?? 0),
                },
            };
            usearch_geospatial_init_ram(&arguments);
        }

        return new GeospatialIndex(
            EngineHandle.Created<NativeGeospatialIndexHandle>(created, outcome), options.Terrain);
    }

    /// <summary>Creates an empty, mutable geospatial index in device memory. Calls
    /// <c>usearch_geospatial_init_vram</c>.</summary>
    /// <param name="options">The terrain and the graph's bounds.</param>
    /// <param name="devices">CUDA device ordinals to place one shard on apiece; empty derives one.</param>
    /// <param name="gpuClassCeiling">The newest class dispatch may pick; null means no ceiling.</param>
    /// <returns>The new index.</returns>
    public static GeospatialIndex InitVram(
        GeospatialOptions options, ReadOnlySpan<int> devices = default, GPUClass? gpuClassCeiling = null)
    {
        IntPtr created = IntPtr.Zero;
        NativeOutcome outcome = default;
        fixed (int* ordinals = devices)
        {
            NativeGeospatialInitVram arguments = new()
            {
                Index = &created, Outcome = &outcome, Terrain = GeospatialTerrain.Lower(options.Terrain),
                Connectivity = (nuint)options.Connectivity, ValuesPerKey = (nuint)options.ValuesPerKey,
                Vram = new NativeVramOptions
                {
                    Devices = (IntPtr)ordinals, DevicesCount = (nuint)devices.Length,
                    GpuClassCeiling = (int)(gpuClassCeiling ?? 0),
                },
            };
            usearch_geospatial_init_vram(&arguments);
        }

        return new GeospatialIndex(
            EngineHandle.Created<NativeGeospatialIndexHandle>(created, outcome), options.Terrain);
    }

    /// <summary>Copies a checkpoint into a new, mutable index in host memory. Calls
    /// <c>usearch_geospatial_load_ram</c>.</summary>
    /// <param name="path">The checkpoint to read, a monolith or a manifest.</param>
    /// <param name="terrain">The terrain lines of sight are tested over, which no checkpoint holds.</param>
    /// <param name="memoryDomains">Where the parts land; empty derives every node.</param>
    /// <param name="valuesPerKey">The upper bound on entries sharing one key; zero is unbounded.</param>
    /// <param name="executor">The workers the read fans out over; null reads on the caller's thread.</param>
    /// <param name="cpuClassCeiling">The newest class dispatch may pick; null means no ceiling.</param>
    /// <returns>The new index.</returns>
    public static GeospatialIndex LoadRam(
        string path, GeospatialTerrain? terrain = null, ReadOnlySpan<int> memoryDomains = default,
        ulong valuesPerKey = 0, Executor? executor = null, CPUClass? cpuClassCeiling = null)
    {
        byte[] encoded = Encoding.UTF8.GetBytes(path);
        using HandleReference pool = HandleReference.Of(executor?.Handle, "executor");
        IntPtr created = IntPtr.Zero;
        NativeOutcome outcome = default;
        fixed (byte* data = encoded)
        fixed (int* domains = memoryDomains)
        {
            NativeGeospatialLoadRam arguments = new()
            {
                Index = &created, Executor = pool.Pointer, Outcome = &outcome, Path = View(data, encoded.Length),
                Ram = new NativeRamOptions
                {
                    MemoryDomains = (IntPtr)domains, MemoryDomainsCount = (nuint)memoryDomains.Length,
                    CpuClassCeiling = (int)(cpuClassCeiling ?? 0),
                },
                ValuesPerKey = (nuint)valuesPerKey, Terrain = GeospatialTerrain.Lower(terrain),
            };
            usearch_geospatial_load_ram(&arguments);
        }

        return new GeospatialIndex(EngineHandle.Created<NativeGeospatialIndexHandle>(created, outcome), terrain);
    }

    /// <summary>Copies a checkpoint into a new, mutable index in device memory. Calls
    /// <c>usearch_geospatial_load_vram</c>.</summary>
    /// <param name="path">The checkpoint to read.</param>
    /// <param name="terrain">The terrain lines of sight are tested over, which no checkpoint holds.</param>
    /// <param name="devices">Where the parts land; empty derives one free device.</param>
    /// <param name="valuesPerKey">The upper bound on entries sharing one key; zero is unbounded.</param>
    /// <param name="executor">The workers the upload runs on; null uploads on the caller's thread.</param>
    /// <param name="gpuClassCeiling">The newest class dispatch may pick; null means no ceiling.</param>
    /// <returns>The new index.</returns>
    public static GeospatialIndex LoadVram(
        string path, GeospatialTerrain? terrain = null, ReadOnlySpan<int> devices = default,
        ulong valuesPerKey = 0, Executor? executor = null, GPUClass? gpuClassCeiling = null)
    {
        byte[] encoded = Encoding.UTF8.GetBytes(path);
        using HandleReference pool = HandleReference.Of(executor?.Handle, "executor");
        IntPtr created = IntPtr.Zero;
        NativeOutcome outcome = default;
        fixed (byte* data = encoded)
        fixed (int* ordinals = devices)
        {
            NativeGeospatialLoadVram arguments = new()
            {
                Index = &created, Executor = pool.Pointer, Outcome = &outcome, Path = View(data, encoded.Length),
                Vram = new NativeVramOptions
                {
                    Devices = (IntPtr)ordinals, DevicesCount = (nuint)devices.Length,
                    GpuClassCeiling = (int)(gpuClassCeiling ?? 0),
                },
                ValuesPerKey = (nuint)valuesPerKey, Terrain = GeospatialTerrain.Lower(terrain),
            };
            usearch_geospatial_load_vram(&arguments);
        }

        return new GeospatialIndex(EngineHandle.Created<NativeGeospatialIndexHandle>(created, outcome), terrain);
    }

    /// <summary>Copies a checkpoint from a buffer into a new, mutable index in host memory. Calls
    /// <c>usearch_geospatial_load_ram_buffer</c>.</summary>
    /// <param name="buffer">The checkpoint's bytes, copied in before the call returns.</param>
    /// <param name="terrain">The terrain lines of sight are tested over, which no checkpoint holds.</param>
    /// <param name="memoryDomains">Where the parts land; empty derives every node.</param>
    /// <param name="valuesPerKey">The upper bound on entries sharing one key; zero is unbounded.</param>
    /// <param name="executor">The workers the copy fans out over; null copies on the caller's thread.</param>
    /// <param name="cpuClassCeiling">The newest class dispatch may pick; null means no ceiling.</param>
    /// <returns>The new index.</returns>
    public static GeospatialIndex LoadRamBuffer(
        ReadOnlySpan<byte> buffer, GeospatialTerrain? terrain = null, ReadOnlySpan<int> memoryDomains = default,
        ulong valuesPerKey = 0, Executor? executor = null, CPUClass? cpuClassCeiling = null)
    {
        using HandleReference pool = HandleReference.Of(executor?.Handle, "executor");
        IntPtr created = IntPtr.Zero;
        NativeOutcome outcome = default;
        fixed (byte* source = buffer)
        fixed (int* domains = memoryDomains)
        {
            NativeGeospatialLoadRamBuffer arguments = new()
            {
                Index = &created, Executor = pool.Pointer, Outcome = &outcome, Buffer = View(source, buffer.Length),
                Ram = new NativeRamOptions
                {
                    MemoryDomains = (IntPtr)domains, MemoryDomainsCount = (nuint)memoryDomains.Length,
                    CpuClassCeiling = (int)(cpuClassCeiling ?? 0),
                },
                ValuesPerKey = (nuint)valuesPerKey, Terrain = GeospatialTerrain.Lower(terrain),
            };
            usearch_geospatial_load_ram_buffer(&arguments);
        }

        return new GeospatialIndex(EngineHandle.Created<NativeGeospatialIndexHandle>(created, outcome), terrain);
    }

    /// <summary>Copies a checkpoint from a buffer into a new, mutable index in device memory. Calls
    /// <c>usearch_geospatial_load_vram_buffer</c>.</summary>
    /// <param name="buffer">The checkpoint's bytes.</param>
    /// <param name="terrain">The terrain lines of sight are tested over, which no checkpoint holds.</param>
    /// <param name="devices">Where the parts land; empty derives one free device.</param>
    /// <param name="valuesPerKey">The upper bound on entries sharing one key; zero is unbounded.</param>
    /// <param name="executor">The workers the upload runs on; null uploads on the caller's thread.</param>
    /// <param name="gpuClassCeiling">The newest class dispatch may pick; null means no ceiling.</param>
    /// <returns>The new index.</returns>
    public static GeospatialIndex LoadVramBuffer(
        ReadOnlySpan<byte> buffer, GeospatialTerrain? terrain = null, ReadOnlySpan<int> devices = default,
        ulong valuesPerKey = 0, Executor? executor = null, GPUClass? gpuClassCeiling = null)
    {
        using HandleReference pool = HandleReference.Of(executor?.Handle, "executor");
        IntPtr created = IntPtr.Zero;
        NativeOutcome outcome = default;
        fixed (byte* source = buffer)
        fixed (int* ordinals = devices)
        {
            NativeGeospatialLoadVramBuffer arguments = new()
            {
                Index = &created, Executor = pool.Pointer, Outcome = &outcome, Buffer = View(source, buffer.Length),
                Vram = new NativeVramOptions
                {
                    Devices = (IntPtr)ordinals, DevicesCount = (nuint)devices.Length,
                    GpuClassCeiling = (int)(gpuClassCeiling ?? 0),
                },
                ValuesPerKey = (nuint)valuesPerKey, Terrain = GeospatialTerrain.Lower(terrain),
            };
            usearch_geospatial_load_vram_buffer(&arguments);
        }

        return new GeospatialIndex(EngineHandle.Created<NativeGeospatialIndexHandle>(created, outcome), terrain);
    }

    /// <summary>Serves a checkpoint in place through a memory mapping, as a read-only index. Calls
    /// <c>usearch_geospatial_view_ram</c>.</summary>
    /// <param name="path">The checkpoint to serve.</param>
    /// <param name="terrain">The terrain lines of sight are tested over, which no checkpoint holds.</param>
    /// <param name="memoryDomains">Where the parts are served from; empty derives.</param>
    /// <param name="cpuClassCeiling">The newest class dispatch may pick; null means no ceiling.</param>
    /// <returns>The read-only index.</returns>
    public static GeospatialIndex ViewRam(
        string path, GeospatialTerrain? terrain = null, ReadOnlySpan<int> memoryDomains = default,
        CPUClass? cpuClassCeiling = null)
    {
        byte[] encoded = Encoding.UTF8.GetBytes(path);
        IntPtr created = IntPtr.Zero;
        NativeOutcome outcome = default;
        fixed (byte* data = encoded)
        fixed (int* domains = memoryDomains)
        {
            NativeGeospatialViewRam arguments = new()
            {
                Index = &created, Outcome = &outcome, Path = View(data, encoded.Length),
                Terrain = GeospatialTerrain.Lower(terrain),
                Ram = new NativeRamOptions
                {
                    MemoryDomains = (IntPtr)domains, MemoryDomainsCount = (nuint)memoryDomains.Length,
                    CpuClassCeiling = (int)(cpuClassCeiling ?? 0),
                },
            };
            usearch_geospatial_view_ram(&arguments);
        }

        return new GeospatialIndex(EngineHandle.Created<NativeGeospatialIndexHandle>(created, outcome), terrain);
    }

    /// <summary>Serves a checkpoint in place from NVMe drives as a read-only index. Calls
    /// <c>usearch_geospatial_view_nvme</c>.</summary>
    /// <param name="path">The checkpoint to serve.</param>
    /// <param name="terrain">The terrain lines of sight are tested over, which no checkpoint holds.</param>
    /// <param name="paths">The devices the parts are served from; empty serves each where its manifest
    /// points.</param>
    /// <param name="executor">The NVMe executor whose rings read the headers; null reads on the caller's
    /// thread.</param>
    /// <param name="cpuClassCeiling">The newest class dispatch may pick; null means no ceiling.</param>
    /// <returns>The read-only index.</returns>
    public static GeospatialIndex ViewNvme(
        string path, GeospatialTerrain? terrain = null, ReadOnlySpan<string> paths = default,
        Executor? executor = null, CPUClass? cpuClassCeiling = null)
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
                NativeGeospatialViewNvme arguments = new()
                {
                    Index = &created, Executor = pool.Pointer, Outcome = &outcome, Path = View(data, encoded.Length),
                    Nvme = new NativeNvmeOptions
                    {
                        Paths = (IntPtr)views, PathsCount = (nuint)paths.Length,
                        CpuClassCeiling = (int)(cpuClassCeiling ?? 0),
                    },
                    Terrain = GeospatialTerrain.Lower(terrain),
                };
                usearch_geospatial_view_nvme(&arguments);
            }
        }
        finally
        {
            NativeMemory.Free(views);
        }

        return new GeospatialIndex(EngineHandle.Created<NativeGeospatialIndexHandle>(created, outcome), terrain);
    }

    /// <summary>Reads a checkpoint's scoring kind and connectivity without loading it. Calls
    /// <c>usearch_geospatial_metadata_read</c>.</summary>
    /// <param name="path">The checkpoint path.</param>
    /// <param name="executor">The workers the read runs on; null reads on the caller's thread.</param>
    /// <returns>What the checkpoint froze.</returns>
    public static GeospatialMetadata MetadataRead(string path, Executor? executor = null)
    {
        byte[] encoded = Encoding.UTF8.GetBytes(path);
        using HandleReference pool = HandleReference.Of(executor?.Handle, "executor");
        NativeGeospatialMetadata lowered = default;
        NativeOutcome outcome = default;
        fixed (byte* data = encoded)
        {
            NativeGeospatialMetadataRead arguments = new()
            {
                Executor = pool.Pointer, Outcome = &outcome, Path = View(data, encoded.Length), Metadata = &lowered,
            };
            usearch_geospatial_metadata_read(&arguments);
        }

        outcome.ThrowIfFailed();
        return GeospatialMetadata.Lift(lowered);
    }

    /// <summary>Reads what a checkpoint in a buffer froze without loading it. Calls
    /// <c>usearch_geospatial_metadata_read_buffer</c>.</summary>
    /// <param name="buffer">The checkpoint's bytes.</param>
    /// <param name="executor">The workers the read runs on; null reads on the caller's thread.</param>
    /// <returns>What the checkpoint froze.</returns>
    public static GeospatialMetadata MetadataReadBuffer(ReadOnlySpan<byte> buffer, Executor? executor = null)
    {
        using HandleReference pool = HandleReference.Of(executor?.Handle, "executor");
        NativeGeospatialMetadata lowered = default;
        NativeOutcome outcome = default;
        fixed (byte* source = buffer)
        {
            NativeGeospatialMetadataReadBuffer arguments = new()
            {
                Executor = pool.Pointer, Outcome = &outcome, Buffer = View(source, buffer.Length),
                Metadata = &lowered,
            };
            usearch_geospatial_metadata_read_buffer(&arguments);
        }

        outcome.ThrowIfFailed();
        return GeospatialMetadata.Lift(lowered);
    }

    /// <summary>The receiver whose verbs block on <paramref name="executor"/>. Binding-only.</summary>
    /// <param name="executor">The executor the verbs run on.</param>
    /// <returns>The receiver, holding nothing but the two references.</returns>
    public GeospatialIndexOn On(Executor executor) => new(this, executor);

    /// <summary>The receiver whose verbs submit their calls to <paramref name="executor"/> and answer a
    /// <see cref="Deferred{TResult}"/> at once. Binding-only.</summary>
    /// <param name="executor">The executor the calls are submitted to.</param>
    /// <returns>The receiver, holding nothing but the two references.</returns>
    public GeospatialIndexDeferredOn Deferred(Executor executor) => new(this, executor);

    /// <inheritdoc cref="GeospatialIndexOn.Reserve"/>
    public void Reserve(ulong entries) =>
        new GeospatialIndexOn(this, null).Reserve(entries);

    /// <inheritdoc cref="GeospatialIndexOn.Clear"/>
    public void Clear() => new GeospatialIndexOn(this, null).Clear();

    /// <inheritdoc cref="GeospatialIndexOn.Describe"/>
    public GeospatialIndexDescription Describe() => new GeospatialIndexOn(this, null).Describe();

    /// <inheritdoc cref="GeospatialIndexOn.Count"/>
    public ulong Count(Key key) => new GeospatialIndexOn(this, null).Count(key);

    /// <inheritdoc cref="GeospatialIndexOn.Contains"/>
    public bool Contains(Key key) => new GeospatialIndexOn(this, null).Contains(key);

    /// <inheritdoc cref="GeospatialIndexOn.Get"/>
    public GeospatialObservations Get(Key key, Arena arena) => new GeospatialIndexOn(this, null).Get(key, arena);

    /// <inheritdoc cref="GeospatialIndexOn.Insert"/>
    public void Insert(
        ReadOnlyMemory<Key> keys, ReadOnlyMemory<GeospatialObservation> observations, ulong expansionFactor = 0,
        DenseTraversalStats? stats = null, Progress? progress = null) =>
        new GeospatialIndexOn(this, null).Insert(keys, observations, expansionFactor, stats, progress);

    /// <inheritdoc cref="GeospatialIndexOn.Refine"/>
    public RefineReport Refine(
        ulong expansionFactor = 0, ulong memoryLimit = 0, DenseTraversalStats? stats = null,
        Progress? progress = null, GeospatialRefinePlan? plan = null) =>
        new GeospatialIndexOn(this, null).Refine(expansionFactor, memoryLimit, stats, progress, plan);

    /// <inheritdoc cref="GeospatialIndexOn.RefinePlanFinish"/>
    public RefineReport RefinePlanFinish(GeospatialRefinePlan plan) =>
        new GeospatialIndexOn(this, null).RefinePlanFinish(plan);

    /// <inheritdoc cref="GeospatialIndexOn.Health"/>
    public DenseReachability Health() => new GeospatialIndexOn(this, null).Health();

    /// <inheritdoc cref="GeospatialIndexOn.Search"/>
    public Matches Search(
        ReadOnlyMemory<GeospatialObservation> queries, int wanted, Arena arena,
        GeospatialRelation relation = GeospatialRelation.Near, GeospatialTolerance tolerance = default,
        ulong expansionFactor = 0, Func<Key, bool>? predicate = null, DenseTraversalStats? stats = null,
        Traversal traversal = Traversal.Walk, Progress? progress = null) =>
        new GeospatialIndexOn(this, null).Search(
            queries, wanted, arena, relation, tolerance, expansionFactor, predicate, stats, traversal, progress);

    /// <inheritdoc cref="GeospatialIndexOn.Within"/>
    public DenseProducts Within(
        ReadOnlyMemory<GeospatialObservation> queries, Arena arena, GeospatialTolerance tolerance,
        GeospatialRelation relation = GeospatialRelation.Near, Func<Key, bool>? predicate = null,
        DenseTraversalStats? stats = null, Traversal traversal = Traversal.Walk, Progress? progress = null) =>
        new GeospatialIndexOn(this, null).Within(
            queries, arena, tolerance, relation, predicate, stats, traversal, progress);

    /// <inheritdoc cref="GeospatialIndexOn.Reshard"/>
    public RefineReport Reshard(
        GeospatialIndex[] sources, GeospatialIndex[] targets, ulong seed = 0, ulong expansionFactor = 0,
        ulong memoryLimit = 0, DenseTraversalStats? stats = null, Progress? progress = null,
        GeospatialReshardPlan? plan = null) =>
        new GeospatialIndexOn(this, null).Reshard(
            sources, targets, seed, expansionFactor, memoryLimit, stats, progress, plan);

    /// <inheritdoc cref="GeospatialIndexOn.ReshardPlanFinish"/>
    public RefineReport ReshardPlanFinish(GeospatialReshardPlan plan) =>
        new GeospatialIndexOn(this, null).ReshardPlanFinish(plan);

    /// <inheritdoc cref="GeospatialIndexOn.Sample"/>
    public Sample Sample(int wanted, Arena arena, DenseSampleConfig config = default) =>
        new GeospatialIndexOn(this, null).Sample(wanted, arena, config);

    /// <inheritdoc cref="GeospatialIndexOn.Save"/>
    public void Save(string path, Progress? progress = null) => new GeospatialIndexOn(this, null).Save(path, progress);

    /// <inheritdoc cref="GeospatialIndexOn.SaveBuffer"/>
    public Monolith SaveBuffer(Arena arena, Progress? progress = null) =>
        new GeospatialIndexOn(this, null).SaveBuffer(arena, progress);

    /// <summary>Opens a snapshot cursor over the current contents, pulled on the caller's thread or
    /// through <see cref="GeospatialCursor.On"/>. Calls
    /// <c>usearch_geospatial_cursor_init</c>.</summary>
    /// <param name="starts">The keys to walk from breadth-first; empty yields every entry as stored.</param>
    /// <returns>The cursor, which keeps the index alive until disposed.</returns>
    public GeospatialCursor Cursor(ReadOnlySpan<Key> starts = default) => GeospatialCursor.Open(this.Handle, starts);

    /// <summary>Releases the index, once every deferred call naming it is final. Calls
    /// <c>usearch_geospatial_free</c>.</summary>
    public void Dispose() => this._handle.Dispose();

    private static NativeBytesView View(byte* data, int length) =>
        new() { Data = (IntPtr)data, Length = (nuint)length };
}

/// <summary>
/// A geospatial index bound to the executor its verbs block on, as <c>index.On(executor)</c> makes it,
/// or to none, which runs every call on the caller's thread.
/// </summary>
/// <remarks>Binding-only; it holds two references and allocates nothing. Every callback a verb takes
/// runs on the engine's threads and must not throw.</remarks>
public readonly unsafe struct GeospatialIndexOn
{
    internal GeospatialIndexOn(GeospatialIndex index, Executor? executor)
    {
        this.Index = index;
        this.Executor = executor;
    }

    /// <summary>The index the verbs address.</summary>
    public GeospatialIndex Index { get; }

    /// <summary>The executor every verb blocks on; null runs them on the caller's thread.</summary>
    public Executor? Executor { get; }

    private CallHandles Handles => new(this.Index.Handle, this.Executor);

    /// <summary>Claims capacity for incoming entries, a hint since insertions grow on demand. Calls
    /// <c>usearch_geospatial_reserve</c>.</summary>
    /// <param name="entries">The desired total entries.</param>
    public void Reserve(ulong entries) =>
        Calls.Block(new GeospatialReserveCall(this.Handles, entries));

    /// <summary>Erases all entries, keeping the reserved capacity and the frozen kind. Calls
    /// <c>usearch_geospatial_clear</c>.</summary>
    public void Clear() => Calls.Block(new GeospatialClearCall(this.Handles));

    /// <summary>Reports the frozen metadata and the current counters in one call. Calls
    /// <c>usearch_geospatial_describe</c>.</summary>
    /// <returns>The description.</returns>
    public GeospatialIndexDescription Describe() => Calls.Block(new GeospatialDescribeCall(this.Handles)).Result;

    /// <summary>Counts the observations stored under a key; zero means the index holds none. Calls
    /// <c>usearch_geospatial_count</c>.</summary>
    /// <param name="key">The identity to look for.</param>
    /// <returns>How many observations share that key.</returns>
    public ulong Count(Key key) => Calls.Block(new GeospatialCountCall(this.Handles, key)).Result;

    /// <summary>Reports whether the index holds at least one entry under a key. Calls
    /// <c>usearch_geospatial_count</c>.</summary>
    /// <param name="key">The identity to look for.</param>
    /// <returns>True when the index holds one or more.</returns>
    public bool Contains(Key key) => this.Count(key) != 0;

    /// <summary>Retrieves every observation stored under a key, in insertion order, within the stored
    /// rounding. Calls <c>usearch_geospatial_get</c>.</summary>
    /// <param name="key">The identity to look for.</param>
    /// <param name="arena">Where the observations land, rewound by the call.</param>
    /// <returns>The observations, none when the key is absent, readable until the arena's next call.</returns>
    public GeospatialObservations Get(Key key, Arena arena) =>
        Calls.Block(new GeospatialGetCall(this.Handles, key, arena)).Result;

    /// <summary>Inserts observations under their keys - the native batch operation, all or nothing per
    /// call. Calls <c>usearch_geospatial_insert</c>.</summary>
    /// <param name="keys">One identity per observation, not necessarily unique.</param>
    /// <param name="observations">The observations.</param>
    /// <param name="expansionFactor">The breadth of the build walk; zero derives it.</param>
    /// <param name="stats">The traversal telemetry to fill, when the caller wants any.</param>
    /// <param name="progress">Hears the call's advance and may cancel it; null runs it unwatched.</param>
    public void Insert(
        ReadOnlyMemory<Key> keys, ReadOnlyMemory<GeospatialObservation> observations, ulong expansionFactor = 0,
        DenseTraversalStats? stats = null, Progress? progress = null) =>
        Calls.Block(new GeospatialInsertCall(this.Handles, keys, observations, expansionFactor, stats, progress));

    /// <summary>Runs one round over the graph: health, heal, hubs, gather and restitch. Calls
    /// <c>usearch_geospatial_refine</c>.</summary>
    /// <param name="expansionFactor">The breadth of the re-selection walk; zero derives it.</param>
    /// <param name="memoryLimit">Bytes a loaded NVMe handle holds for the part it rewrites; zero derives.</param>
    /// <param name="stats">The traversal telemetry to fill, when the caller wants any.</param>
    /// <param name="progress">Hears every phase's units and may stop the round at that barrier.</param>
    /// <param name="plan">A plan a stop leaves live and the next call continues; null runs one round.</param>
    /// <returns>What the round did to the graph.</returns>
    public RefineReport Refine(
        ulong expansionFactor = 0, ulong memoryLimit = 0, DenseTraversalStats? stats = null,
        Progress? progress = null, GeospatialRefinePlan? plan = null) =>
        Calls.Block(new GeospatialRefineCall(this.Handles, expansionFactor, memoryLimit, stats, progress, plan)).Result;

    /// <summary>Closes a refine plan and frees it whatever the call answers. Calls
    /// <c>usearch_geospatial_refine_plan_finish</c>.</summary>
    /// <param name="plan">The plan, opened on this index and executor.</param>
    /// <returns>What the whole plan did to the graph.</returns>
    public RefineReport RefinePlanFinish(GeospatialRefinePlan plan) =>
        Calls.Block(new GeospatialRefinePlanFinishCall(this.Handles, plan)).Result;

    /// <summary>Measures how much of the graph a search reaches from the root. Calls
    /// <c>usearch_geospatial_health</c>.</summary>
    /// <returns>The population audited and how much of it is reachable.</returns>
    public DenseReachability Health() => Calls.Block(new GeospatialHealthCall(this.Handles)).Result;

    /// <summary>Finds the nearest entries of each query observation the relation admits, scored by chord
    /// metres. Calls <c>usearch_geospatial_search</c>.</summary>
    /// <param name="queries">The query observations, one query apiece.</param>
    /// <param name="wanted">The upper bound on matches per query.</param>
    /// <param name="arena">Where the run lands, rewound by the call.</param>
    /// <param name="relation">Which entries a query admits.</param>
    /// <param name="tolerance">The radius and time widening; a zero radius ranks by distance alone.</param>
    /// <param name="expansionFactor">The breadth of the walk; zero derives it.</param>
    /// <param name="predicate">The per-key admission predicate; null admits every key.</param>
    /// <param name="stats">The traversal telemetry to fill, when the caller wants any.</param>
    /// <param name="traversal">A walk, the default, or an exact scan of every entry.</param>
    /// <param name="progress">Hears the call's advance and may cancel it; null runs it unwatched.</param>
    /// <returns>The run, one list per query, readable until the arena's next call.</returns>
    public Matches Search(
        ReadOnlyMemory<GeospatialObservation> queries, int wanted, Arena arena,
        GeospatialRelation relation = GeospatialRelation.Near, GeospatialTolerance tolerance = default,
        ulong expansionFactor = 0, Func<Key, bool>? predicate = null, DenseTraversalStats? stats = null,
        Traversal traversal = Traversal.Walk, Progress? progress = null) =>
        Calls.Block(new GeospatialSearchCall(
            this.Handles, queries, wanted, arena, relation, tolerance, expansionFactor, predicate, stats, traversal,
            progress)).Result;

    /// <summary>Reports every entry within the radius of each query observation that the relation
    /// admits, however many there are, as one ragged run per query. Calls
    /// <c>usearch_geospatial_within</c>.</summary>
    /// <param name="queries">The query observations, one query apiece.</param>
    /// <param name="arena">Where the run lands, rewound by the call.</param>
    /// <param name="tolerance">The radius, which must be positive, and the time widening.</param>
    /// <param name="relation">Which entries a query admits.</param>
    /// <param name="predicate">The per-key admission predicate; null admits every key.</param>
    /// <param name="stats">The traversal telemetry to fill, when the caller wants any.</param>
    /// <param name="traversal">A complete walk, the default, or an exact scan of every entry.</param>
    /// <param name="progress">Hears the call's advance and may cancel it; null runs it unwatched.</param>
    /// <returns>The keys and chord metres of every query, readable until the arena's next call.</returns>
    public DenseProducts Within(
        ReadOnlyMemory<GeospatialObservation> queries, Arena arena, GeospatialTolerance tolerance,
        GeospatialRelation relation = GeospatialRelation.Near, Func<Key, bool>? predicate = null,
        DenseTraversalStats? stats = null, Traversal traversal = Traversal.Walk, Progress? progress = null) =>
        Calls.Block(new GeospatialWithinCall(
            this.Handles, queries, arena, relation, tolerance, predicate, stats, traversal, progress)).Result;

    /// <summary>Moves the entries of the sources into the targets, keeping their links, then runs a round
    /// on every target that took entries: several sources into one gather, one into several cut. Calls
    /// <c>usearch_geospatial_reshard</c>.</summary>
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
        GeospatialIndex[] sources, GeospatialIndex[] targets, ulong seed = 0, ulong expansionFactor = 0,
        ulong memoryLimit = 0, DenseTraversalStats? stats = null, Progress? progress = null,
        GeospatialReshardPlan? plan = null) =>
        Calls.Block(new GeospatialReshardCall(
            this.Handles, sources, targets, seed, expansionFactor, memoryLimit, stats, progress, plan)).Result;

    /// <summary>Closes a reshard plan and frees it whatever the call answers. Calls
    /// <c>usearch_geospatial_reshard_plan_finish</c>.</summary>
    /// <param name="plan">The plan, opened with this executor.</param>
    /// <returns>What the whole plan did to the graph.</returns>
    public RefineReport ReshardPlanFinish(GeospatialReshardPlan plan) =>
        Calls.Block(new GeospatialReshardPlanFinishCall(this.Handles, plan)).Result;

    /// <summary>Draws keys around the focus, or across the whole index, each hanging from a shown or
    /// earlier draw and placed in three dimensions. Calls <c>usearch_geospatial_sample</c>.</summary>
    /// <param name="wanted">The upper bound on draws.</param>
    /// <param name="arena">Where the draws are written, rewound by the call.</param>
    /// <param name="config">The keys shown and their coordinates, the focus, the seed and the walk.</param>
    /// <returns>The draws, readable until the arena's next call.</returns>
    public Sample Sample(int wanted, Arena arena, DenseSampleConfig config = default) =>
        Calls.Block(new GeospatialSampleCall(this.Handles, wanted, arena, config)).Result;

    /// <summary>Writes the index under a path, one monolith or a manifest beside its parts. Calls
    /// <c>usearch_geospatial_save</c>.</summary>
    /// <param name="path">Where the checkpoint is written.</param>
    /// <param name="progress">Hears the call's advance and may cancel it; null runs it unwatched.</param>
    public void Save(string path, Progress? progress = null) =>
        Calls.Block(new GeospatialSaveCall(this.Handles, path, progress));

    /// <summary>Writes the index into an arena, the same bytes <see cref="Save"/> writes for one shard.
    /// Calls <c>usearch_geospatial_save_buffer</c>.</summary>
    /// <param name="arena">Where the bytes land, rewound by the call.</param>
    /// <param name="progress">Hears the call's advance and may cancel it; null runs it unwatched.</param>
    /// <returns>The checkpoint, readable until the arena's next call.</returns>
    public Monolith SaveBuffer(Arena arena, Progress? progress = null) =>
        Calls.Block(new GeospatialSaveBufferCall(this.Handles, arena, progress)).Result;
}

/// <summary>A geospatial index bound to the executor its verbs submit to, as <c>index.Deferred(executor)</c>
/// makes it, each verb answering a <see cref="Deferred{TResult}"/> once submitted.</summary>
/// <remarks>Binding-only; it holds two references and allocates nothing.</remarks>
public readonly struct GeospatialIndexDeferredOn
{
    internal GeospatialIndexDeferredOn(GeospatialIndex index, Executor executor)
    {
        this.Index = index;
        this.Executor = executor;
    }

    /// <summary>The index the verbs address.</summary>
    public GeospatialIndex Index { get; }

    /// <summary>The executor every verb submits to.</summary>
    public Executor Executor { get; }

    private CallHandles Handles => new(this.Index.Handle, this.Executor);

    /// <inheritdoc cref="GeospatialIndexOn.Reserve"/>
    public Deferred Reserve(ulong entries) =>
        Calls.Defer(new GeospatialReserveCall(this.Handles, entries));

    /// <inheritdoc cref="GeospatialIndexOn.Clear"/>
    public Deferred Clear() => Calls.Defer(new GeospatialClearCall(this.Handles));

    /// <inheritdoc cref="GeospatialIndexOn.Describe"/>
    public Deferred<GeospatialIndexDescription> Describe() => Calls.Defer(new GeospatialDescribeCall(this.Handles));

    /// <inheritdoc cref="GeospatialIndexOn.Count"/>
    public Deferred<ulong> Count(Key key) => Calls.Defer(new GeospatialCountCall(this.Handles, key));

    /// <inheritdoc cref="GeospatialIndexOn.Get"/>
    public Deferred<GeospatialObservations> Get(Key key, Arena arena) =>
        Calls.Defer(new GeospatialGetCall(this.Handles, key, arena));

    /// <inheritdoc cref="GeospatialIndexOn.Insert"/>
    public Deferred Insert(
        ReadOnlyMemory<Key> keys, ReadOnlyMemory<GeospatialObservation> observations, ulong expansionFactor = 0,
        DenseTraversalStats? stats = null, Progress? progress = null) =>
        Calls.Defer(new GeospatialInsertCall(this.Handles, keys, observations, expansionFactor, stats, progress));

    /// <inheritdoc cref="GeospatialIndexOn.Refine"/>
    public Deferred<RefineReport> Refine(
        ulong expansionFactor = 0, ulong memoryLimit = 0, DenseTraversalStats? stats = null,
        Progress? progress = null, GeospatialRefinePlan? plan = null) =>
        Calls.Defer(new GeospatialRefineCall(this.Handles, expansionFactor, memoryLimit, stats, progress, plan));

    /// <inheritdoc cref="GeospatialIndexOn.RefinePlanFinish"/>
    public Deferred<RefineReport> RefinePlanFinish(GeospatialRefinePlan plan) =>
        Calls.Defer(new GeospatialRefinePlanFinishCall(this.Handles, plan));

    /// <inheritdoc cref="GeospatialIndexOn.Health"/>
    public Deferred<DenseReachability> Health() => Calls.Defer(new GeospatialHealthCall(this.Handles));

    /// <inheritdoc cref="GeospatialIndexOn.Search"/>
    public Deferred<Matches> Search(
        ReadOnlyMemory<GeospatialObservation> queries, int wanted, Arena arena,
        GeospatialRelation relation = GeospatialRelation.Near, GeospatialTolerance tolerance = default,
        ulong expansionFactor = 0, Func<Key, bool>? predicate = null, DenseTraversalStats? stats = null,
        Traversal traversal = Traversal.Walk, Progress? progress = null) =>
        Calls.Defer(new GeospatialSearchCall(
            this.Handles, queries, wanted, arena, relation, tolerance, expansionFactor, predicate, stats, traversal,
            progress));

    /// <inheritdoc cref="GeospatialIndexOn.Within"/>
    public Deferred<DenseProducts> Within(
        ReadOnlyMemory<GeospatialObservation> queries, Arena arena, GeospatialTolerance tolerance,
        GeospatialRelation relation = GeospatialRelation.Near, Func<Key, bool>? predicate = null,
        DenseTraversalStats? stats = null, Traversal traversal = Traversal.Walk, Progress? progress = null) =>
        Calls.Defer(new GeospatialWithinCall(
            this.Handles, queries, arena, relation, tolerance, predicate, stats, traversal, progress));

    /// <inheritdoc cref="GeospatialIndexOn.Reshard"/>
    public Deferred<RefineReport> Reshard(
        GeospatialIndex[] sources, GeospatialIndex[] targets, ulong seed = 0, ulong expansionFactor = 0,
        ulong memoryLimit = 0, DenseTraversalStats? stats = null, Progress? progress = null,
        GeospatialReshardPlan? plan = null) =>
        Calls.Defer(new GeospatialReshardCall(
            this.Handles, sources, targets, seed, expansionFactor, memoryLimit, stats, progress, plan));

    /// <inheritdoc cref="GeospatialIndexOn.ReshardPlanFinish"/>
    public Deferred<RefineReport> ReshardPlanFinish(GeospatialReshardPlan plan) =>
        Calls.Defer(new GeospatialReshardPlanFinishCall(this.Handles, plan));

    /// <inheritdoc cref="GeospatialIndexOn.Sample"/>
    public Deferred<Sample> Sample(int wanted, Arena arena, DenseSampleConfig config = default) =>
        Calls.Defer(new GeospatialSampleCall(this.Handles, wanted, arena, config));

    /// <inheritdoc cref="GeospatialIndexOn.Save"/>
    public Deferred Save(string path, Progress? progress = null) =>
        Calls.Defer(new GeospatialSaveCall(this.Handles, path, progress));

    /// <inheritdoc cref="GeospatialIndexOn.SaveBuffer"/>
    public Deferred<Monolith> SaveBuffer(Arena arena, Progress? progress = null) =>
        Calls.Defer(new GeospatialSaveBufferCall(this.Handles, arena, progress));
}

/// <summary>One geospatial refine in flight between calls. A fresh one opens on the first
/// <see cref="GeospatialIndex.Refine"/> naming it, a stopped round continues on the next, and
/// <see cref="GeospatialIndex.RefinePlanFinish"/> closes it.</summary>
public sealed unsafe class GeospatialRefinePlan
{
    private readonly PlanCell[] _cell = GC.AllocateArray<PlanCell>(1, pinned: true);

    /// <summary>Where the last call naming the plan stopped, <see cref="Phase.Done"/> once its round
    /// ended.</summary>
    public PhaseProgress Phase => this.Cell->Phase.Lift();

    internal PlanCell* Cell => (PlanCell*)Unsafe.AsPointer(ref MemoryMarshal.GetArrayDataReference(this._cell));
}

/// <summary>One geospatial reshard in flight between calls. A fresh one opens on the first
/// <see cref="GeospatialIndex.Reshard"/> naming it, which then reads its own lists rather than the call's,
/// and <see cref="GeospatialIndex.ReshardPlanFinish"/> closes it.</summary>
public sealed unsafe class GeospatialReshardPlan
{
    private readonly PlanCell[] _cell = GC.AllocateArray<PlanCell>(1, pinned: true);

    /// <summary>Where the last call naming the plan stopped, <see cref="Phase.Done"/> once its rounds
    /// ended.</summary>
    public PhaseProgress Phase => this.Cell->Phase.Lift();

    internal PlanCell* Cell => (PlanCell*)Unsafe.AsPointer(ref MemoryMarshal.GetArrayDataReference(this._cell));
}

/// <summary>A snapshot cursor over a geospatial index, yielding each entry's key with the keys it links
/// to; the geospatial come from <see cref="GeospatialIndex.Get"/>. It keeps the index alive until disposed
/// and no executor: its pulls run on the caller's thread, or block on the executor <see cref="On"/>
/// names for each.</summary>
public sealed unsafe class GeospatialCursor : IDisposable
{
    private readonly NativeGeospatialCursorHandle _handle;

    private GeospatialCursor(NativeGeospatialCursorHandle handle) => this._handle = handle;

    internal static GeospatialCursor Open(EngineHandle index, ReadOnlySpan<Key> starts)
    {
        HandleReference held = HandleReference.Of(index, "index");

        IntPtr created = IntPtr.Zero;
        NativeOutcome outcome = default;
        fixed (Key* run = starts)
        {
            NativeGeospatialCursorInit arguments = new()
            {
                Index = held.Pointer, Outcome = &outcome, StartsCount = (nuint)starts.Length, Starts = run,
                Cursor = &created,
            };
            usearch_geospatial_cursor_init(&arguments);
        }

        if (outcome.Status != Status.Success)
        {
            held.Dispose();
            outcome.ThrowIfFailed();
        }

        return new GeospatialCursor(new NativeGeospatialCursorHandle(created, held));
    }

    /// <summary>Copies the next batch of entries into an arena, the keys always and the keys each links
    /// to when asked for. Calls <c>usearch_geospatial_cursor_next</c>.</summary>
    /// <param name="wanted">The upper bound on entries the pull writes.</param>
    /// <param name="arena">Where the batch lands, rewound by the pull.</param>
    /// <param name="links">Whether to export the keys each entry links to.</param>
    /// <returns>The batch, empty once the snapshot is exhausted.</returns>
    public DenseBatch Next(int wanted, Arena arena, bool links = false) => this.Pull(null, wanted, arena, links);

    /// <summary>The receiver whose pulls block on <paramref name="executor"/>, named again for every pull.</summary>
    /// <param name="executor">The executor every pull through the receiver runs on.</param>
    /// <returns>The receiver, which offers the same pulls.</returns>
    public GeospatialCursorOn On(Executor executor) => new(this, executor);

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
        NativeGeospatialCursorNext arguments = new()
        {
            Index = this._handle.Index, Executor = pool.Pointer, Outcome = &outcome,
            Cursor = cursor.Pointer, Keys = &keys, Links = links ? &linked : null, Written = &written,
            Arena = passed.Pointer, Wanted = bound,
        };
        usearch_geospatial_cursor_next(&arguments);
        outcome.ThrowIfFailed();
        return new DenseBatch(keys, linked, links, (int)written, arena.Description);
    }

    /// <summary>Releases the cursor, then the index it kept alive. Calls
    /// <c>usearch_geospatial_cursor_free</c>.</summary>
    public void Dispose() => this._handle.Dispose();
}

/// <summary>A cursor's pulls blocking on one executor, as <see cref="GeospatialCursor.On"/> makes them.</summary>
public readonly struct GeospatialCursorOn
{
    private readonly GeospatialCursor _cursor;
    private readonly Executor _executor;

    internal GeospatialCursorOn(GeospatialCursor cursor, Executor executor)
    {
        this._cursor = cursor;
        this._executor = executor;
    }

    /// <inheritdoc cref="GeospatialCursor.Next"/>
    public DenseBatch Next(int wanted, Arena arena, bool links = false) =>
        this._cursor.Pull(this._executor, wanted, arena, links);
}

#endregion Index

#region Calls

/// <summary>A reservation; see <see cref="GeospatialIndexOn.Reserve"/>.</summary>
internal unsafe struct GeospatialReserveCall(CallHandles handles, ulong entries) : ICall
{
    public readonly void Submit(CallOutputs* outputs, Completion completion)
    {
        using HandleReference index = handles.ReferenceIndex();
        using HandleReference executor = handles.ReferenceExecutor();
        NativeGeospatialReserve arguments = new()
        {
            Index = index.Pointer, Executor = executor.Pointer, Outcome = &outputs->Outcome, Completion = completion,
            Entries = (nuint)entries,
        };
        usearch_geospatial_reserve(&arguments);
    }

    public readonly void Release(CallOutputs* outputs) { }

    public readonly void Finish(CallOutputs* outputs) => outputs->Outcome.ThrowIfFailed();
}

/// <summary>An erasure; see <see cref="GeospatialIndexOn.Clear"/>.</summary>
internal unsafe struct GeospatialClearCall(CallHandles handles) : ICall
{
    public readonly void Submit(CallOutputs* outputs, Completion completion)
    {
        using HandleReference index = handles.ReferenceIndex();
        using HandleReference executor = handles.ReferenceExecutor();
        NativeGeospatialClear arguments = new()
        {
            Index = index.Pointer, Executor = executor.Pointer, Outcome = &outputs->Outcome, Completion = completion,
        };
        usearch_geospatial_clear(&arguments);
    }

    public readonly void Release(CallOutputs* outputs) { }

    public readonly void Finish(CallOutputs* outputs) => outputs->Outcome.ThrowIfFailed();
}

/// <summary>A description; see <see cref="GeospatialIndexOn.Describe"/>.</summary>
internal unsafe struct GeospatialDescribeCall(CallHandles handles) : ICall<GeospatialIndexDescription>
{
    public GeospatialIndexDescription Result { get; private set; }

    public readonly void Submit(CallOutputs* outputs, Completion completion)
    {
        using HandleReference index = handles.ReferenceIndex();
        using HandleReference executor = handles.ReferenceExecutor();
        NativeGeospatialDescribe arguments = new()
        {
            Index = index.Pointer, Executor = executor.Pointer, Outcome = &outputs->Outcome, Completion = completion,
            Description = (NativeGeospatialDescription*)outputs->Report,
        };
        usearch_geospatial_describe(&arguments);
    }

    public readonly void Release(CallOutputs* outputs) { }

    public void Finish(CallOutputs* outputs)
    {
        outputs->Outcome.ThrowIfFailed();
        NativeGeospatialDescription* lowered = (NativeGeospatialDescription*)outputs->Report;
        this.Result = new GeospatialIndexDescription
        {
            Metadata = GeospatialMetadata.Lift(lowered->Metadata), Size = lowered->Size, Capacity = lowered->Capacity,
            MemoryUsage = lowered->MemoryUsage,
            CpuClass = Marshal.PtrToStringUTF8(usearch_cpu_class_name(lowered->CpuClass)) ?? "unknown",
            GpuClass = Marshal.PtrToStringUTF8(usearch_gpu_class_name(lowered->GpuClass)) ?? "unknown",
            ShardsCount = lowered->ShardsCount,
        };
    }
}

/// <summary>A count; see <see cref="GeospatialIndexOn.Count"/>.</summary>
internal unsafe struct GeospatialCountCall(CallHandles handles, Key key) : ICall<ulong>
{
    public ulong Result { get; private set; }

    public readonly void Submit(CallOutputs* outputs, Completion completion)
    {
        using HandleReference index = handles.ReferenceIndex();
        using HandleReference executor = handles.ReferenceExecutor();
        NativeGeospatialCount arguments = new()
        {
            Index = index.Pointer, Executor = executor.Pointer, Outcome = &outputs->Outcome, Completion = completion,
            Key = key, KeyStats = (NativeGeospatialKeyStats*)outputs->Report,
        };
        usearch_geospatial_count(&arguments);
    }

    public readonly void Release(CallOutputs* outputs) { }

    public void Finish(CallOutputs* outputs)
    {
        outputs->Outcome.ThrowIfFailed();
        this.Result = ((NativeGeospatialKeyStats*)outputs->Report)->Entries;
    }
}

/// <summary>A retrieval; see <see cref="GeospatialIndexOn.Get"/>.</summary>
internal unsafe struct GeospatialGetCall(CallHandles handles, Key key, Arena arena) : ICall<GeospatialObservations>
{
    public GeospatialObservations Result { get; private set; }

    public readonly void Submit(CallOutputs* outputs, Completion completion)
    {
        using HandleReference index = handles.ReferenceIndex();
        using HandleReference executor = handles.ReferenceExecutor();
        using HandleReference passed = Arena.Reference(arena);
        NativeGeospatialGet arguments = new()
        {
            Index = index.Pointer, Executor = executor.Pointer, Outcome = &outputs->Outcome, Completion = completion,
            Key = key, Observations = (GeospatialObservation**)outputs->View, Written = &outputs->Written,
            Arena = passed.Pointer,
        };
        usearch_geospatial_get(&arguments);
    }

    public readonly void Release(CallOutputs* outputs) { }

    public void Finish(CallOutputs* outputs)
    {
        outputs->Outcome.ThrowIfFailed();
        this.Result = new GeospatialObservations(
            *(GeospatialObservation**)outputs->View, (int)outputs->Written, arena.Description);
    }
}

/// <summary>An insertion; see <see cref="GeospatialIndexOn.Insert"/>.</summary>
internal unsafe struct GeospatialInsertCall(
    CallHandles handles, ReadOnlyMemory<Key> keys, ReadOnlyMemory<GeospatialObservation> observations,
    ulong expansionFactor, DenseTraversalStats? stats, Progress? progress) : ICall
{
    private MemoryHandle _keys;
    private MemoryHandle _observations;
    private Registration _progress;

    public void Submit(CallOutputs* outputs, Completion completion)
    {
        // The call has one count, so only this side sees the keys and the observations disagree.
        if (keys.Length != observations.Length)
        {
            throw new USearchException(
                Status.ShapeMismatch, "the key count does not match the observation count it accompanies");
        }

        this._keys = keys.Pin();
        this._observations = observations.Pin();
        using HandleReference index = handles.ReferenceIndex();
        using HandleReference executor = handles.ReferenceExecutor();
        NativeGeospatialInsert arguments = new()
        {
            Index = index.Pointer, Executor = executor.Pointer, Outcome = &outputs->Outcome, Completion = completion,
            Count = (nuint)keys.Length,
            Keys = new NativeKeysView { Data = (IntPtr)this._keys.Pointer, Stride = (nuint)sizeof(Key) },
            Observations = (GeospatialObservation*)this._observations.Pointer,
            ExpansionFactor = (nuint)expansionFactor, Progress = this._progress.Progress(progress),
            Stats = (NativeDenseTraversalStats*)outputs->Report,
        };
        usearch_geospatial_insert(&arguments);
    }

    public void Release(CallOutputs* outputs)
    {
        this._keys.Dispose();
        this._observations.Dispose();
        this._progress.Free();
    }

    public readonly void Finish(CallOutputs* outputs)
    {
        outputs->Outcome.ThrowIfFailed();
        stats?.Adopt(*(NativeDenseTraversalStats*)outputs->Report);
    }
}

/// <summary>A ranked search; see <see cref="GeospatialIndexOn.Search"/>.</summary>
internal unsafe struct GeospatialSearchCall(
    CallHandles handles, ReadOnlyMemory<GeospatialObservation> queries, int wanted, Arena arena,
    GeospatialRelation relation, GeospatialTolerance tolerance, ulong expansionFactor, Func<Key, bool>? predicate,
    DenseTraversalStats? stats, Traversal traversal, Progress? progress) : ICall<Matches>
{
    private MemoryHandle _queries;
    private Registration _predicate;
    private Registration _progress;

    public Matches Result { get; private set; }

    public void Submit(CallOutputs* outputs, Completion completion)
    {
        nuint bound = Calls.Bound(wanted);
        this._queries = queries.Pin();
        using HandleReference index = handles.ReferenceIndex();
        using HandleReference executor = handles.ReferenceExecutor();
        using HandleReference passed = Arena.Reference(arena);
        NativeGeospatialSearch arguments = new()
        {
            Index = index.Pointer, Executor = executor.Pointer, Outcome = &outputs->Outcome, Completion = completion,
            Count = (nuint)queries.Length, Queries = (GeospatialObservation*)this._queries.Pointer,
            Relation = relation, Tolerance = tolerance, Matches = (NativeMatchesView*)outputs->View,
            Arena = passed.Pointer, Wanted = bound, ExpansionFactor = (nuint)expansionFactor, Traversal = traversal,
            Predicate = this._predicate.Predicate(predicate), Progress = this._progress.Progress(progress),
            Stats = (NativeDenseTraversalStats*)outputs->Report,
        };
        usearch_geospatial_search(&arguments);
    }

    public void Release(CallOutputs* outputs)
    {
        this._queries.Dispose();
        this._predicate.Free();
        this._progress.Free();
    }

    public void Finish(CallOutputs* outputs)
    {
        outputs->Outcome.ThrowIfFailed();
        stats?.Adopt(*(NativeDenseTraversalStats*)outputs->Report);
        this.Result = new Matches(*(NativeMatchesView*)outputs->View, queries.Length, arena.Description);
    }
}

/// <summary>A containment report; see <see cref="GeospatialIndexOn.Within"/>.</summary>
internal unsafe struct GeospatialWithinCall(
    CallHandles handles, ReadOnlyMemory<GeospatialObservation> queries, Arena arena, GeospatialRelation relation,
    GeospatialTolerance tolerance, Func<Key, bool>? predicate, DenseTraversalStats? stats, Traversal traversal,
    Progress? progress) : ICall<DenseProducts>
{
    private MemoryHandle _queries;
    private Registration _predicate;
    private Registration _progress;

    public DenseProducts Result { get; private set; }

    public void Submit(CallOutputs* outputs, Completion completion)
    {
        this._queries = queries.Pin();
        using HandleReference index = handles.ReferenceIndex();
        using HandleReference executor = handles.ReferenceExecutor();
        using HandleReference passed = Arena.Reference(arena);
        NativeGeospatialWithin arguments = new()
        {
            Index = index.Pointer, Executor = executor.Pointer, Outcome = &outputs->Outcome, Completion = completion,
            Count = (nuint)queries.Length, Queries = (GeospatialObservation*)this._queries.Pointer,
            Relation = relation, Tolerance = tolerance, Matches = (NativeMatchesTapeView*)outputs->View,
            Written = &outputs->Written, Arena = passed.Pointer, Traversal = traversal,
            Predicate = this._predicate.Predicate(predicate), Progress = this._progress.Progress(progress),
            Stats = (NativeDenseTraversalStats*)outputs->Report,
        };
        usearch_geospatial_within(&arguments);
    }

    public void Release(CallOutputs* outputs)
    {
        this._queries.Dispose();
        this._predicate.Free();
        this._progress.Free();
    }

    public void Finish(CallOutputs* outputs)
    {
        outputs->Outcome.ThrowIfFailed();
        stats?.Adopt(*(NativeDenseTraversalStats*)outputs->Report);
        this.Result = new DenseProducts(
            *(NativeMatchesTapeView*)outputs->View, queries.Length, outputs->Written, arena.Description);
    }
}

/// <summary>A refinement; see <see cref="GeospatialIndexOn.Refine"/>.</summary>
internal unsafe struct GeospatialRefineCall(
    CallHandles handles, ulong expansionFactor, ulong memoryLimit, DenseTraversalStats? stats, Progress? progress,
    GeospatialRefinePlan? plan) : ICall<RefineReport>
{
    private Registration _progress;

    public RefineReport Result { get; private set; }

    public void Submit(CallOutputs* outputs, Completion completion)
    {
        using HandleReference index = handles.ReferenceIndex();
        using HandleReference executor = handles.ReferenceExecutor();
        NativeGeospatialRefine arguments = new()
        {
            Index = index.Pointer, Executor = executor.Pointer, Outcome = &outputs->Outcome, Completion = completion,
            Report = RefineReport.In(outputs),
            Refine = new() { ExpansionFactor = (nuint)expansionFactor, MemoryLimit = (nuint)memoryLimit },
            Progress = this._progress.Progress(progress), Stats = (NativeDenseTraversalStats*)outputs->Report,
            Plan = plan is null ? null : &plan.Cell->Handle, Phase = plan is null ? null : &plan.Cell->Phase,
        };
        usearch_geospatial_refine(&arguments);
    }

    public void Release(CallOutputs* outputs) => this._progress.Free();

    public void Finish(CallOutputs* outputs)
    {
        outputs->Outcome.ThrowIfFailed();
        stats?.Adopt(*(NativeDenseTraversalStats*)outputs->Report);
        this.Result = RefineReport.Lift(RefineReport.In(outputs));
    }
}

/// <summary>A refine plan's close; see <see cref="GeospatialIndexOn.RefinePlanFinish"/>.</summary>
internal unsafe struct GeospatialRefinePlanFinishCall(CallHandles handles, GeospatialRefinePlan plan)
    : ICall<RefineReport>
{
    public RefineReport Result { get; private set; }

    public readonly void Submit(CallOutputs* outputs, Completion completion)
    {
        using HandleReference index = handles.ReferenceIndex();
        using HandleReference executor = handles.ReferenceExecutor();
        NativeGeospatialRefinePlanFinish arguments = new()
        {
            Index = index.Pointer, Executor = executor.Pointer, Outcome = &outputs->Outcome, Completion = completion,
            Plan = plan.Cell->Handle, Report = RefineReport.In(outputs),
        };
        usearch_geospatial_refine_plan_finish(&arguments);
        plan.Cell->Handle = IntPtr.Zero;
    }

    public readonly void Release(CallOutputs* outputs) { }

    public void Finish(CallOutputs* outputs)
    {
        outputs->Outcome.ThrowIfFailed();
        this.Result = RefineReport.Lift(RefineReport.In(outputs));
    }
}

/// <summary>A reachability audit; see <see cref="GeospatialIndexOn.Health"/>.</summary>
internal unsafe struct GeospatialHealthCall(CallHandles handles) : ICall<DenseReachability>
{
    public DenseReachability Result { get; private set; }

    public readonly void Submit(CallOutputs* outputs, Completion completion)
    {
        using HandleReference index = handles.ReferenceIndex();
        using HandleReference executor = handles.ReferenceExecutor();
        NativeGeospatialHealth arguments = new()
        {
            Index = index.Pointer, Executor = executor.Pointer, Outcome = &outputs->Outcome, Completion = completion,
            Reachability = (NativeDenseReachability*)outputs->Report,
        };
        usearch_geospatial_health(&arguments);
    }

    public readonly void Release(CallOutputs* outputs) { }

    public void Finish(CallOutputs* outputs)
    {
        outputs->Outcome.ThrowIfFailed();
        this.Result = DenseReachability.Lift((NativeDenseReachability*)outputs->Report);
    }
}

/// <summary>A reshard; see <see cref="GeospatialIndexOn.Reshard"/>.</summary>
internal unsafe struct GeospatialReshardCall(
    CallHandles handles, GeospatialIndex[] sources, GeospatialIndex[] targets, ulong seed, ulong expansionFactor,
    ulong memoryLimit, DenseTraversalStats? stats, Progress? progress, GeospatialReshardPlan? plan)
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
            GeospatialIndex named = position < sources.Length ? sources[position] : targets[position - sources.Length];
            this._references[position] = HandleReference.Of(named?.Handle, "index");
            this._slots[position] = this._references[position].Pointer;
        }

        this._pinnedSlots = this._slots.AsMemory().Pin();
        IntPtr* slots = (IntPtr*)this._pinnedSlots.Pointer;
        using HandleReference executor = handles.ReferenceExecutor();
        NativeGeospatialReshard arguments = new()
        {
            SourcesCount = (nuint)sources.Length, Sources = slots, TargetsCount = (nuint)targets.Length,
            Targets = slots + sources.Length, Executor = executor.Pointer, Outcome = &outputs->Outcome,
            Completion = completion, Report = RefineReport.In(outputs),
            Refine = new() { ExpansionFactor = (nuint)expansionFactor, MemoryLimit = (nuint)memoryLimit },
            Seed = seed, Progress = this._progress.Progress(progress),
            Stats = (NativeDenseTraversalStats*)outputs->Report,
            Plan = plan is null ? null : &plan.Cell->Handle, Phase = plan is null ? null : &plan.Cell->Phase,
        };
        usearch_geospatial_reshard(&arguments);
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

/// <summary>A reshard plan's close; see <see cref="GeospatialIndexOn.ReshardPlanFinish"/>.</summary>
internal unsafe struct GeospatialReshardPlanFinishCall(CallHandles handles, GeospatialReshardPlan plan)
    : ICall<RefineReport>
{
    public RefineReport Result { get; private set; }

    public readonly void Submit(CallOutputs* outputs, Completion completion)
    {
        using HandleReference executor = handles.ReferenceExecutor();
        NativeGeospatialReshardPlanFinish arguments = new()
        {
            Executor = executor.Pointer, Outcome = &outputs->Outcome, Completion = completion,
            Plan = plan.Cell->Handle, Report = RefineReport.In(outputs),
        };
        usearch_geospatial_reshard_plan_finish(&arguments);
        plan.Cell->Handle = IntPtr.Zero;
    }

    public readonly void Release(CallOutputs* outputs) { }

    public void Finish(CallOutputs* outputs)
    {
        outputs->Outcome.ThrowIfFailed();
        this.Result = RefineReport.Lift(RefineReport.In(outputs));
    }
}

/// <summary>A sample; see <see cref="GeospatialIndexOn.Sample"/>.</summary>
internal unsafe struct GeospatialSampleCall(CallHandles handles, int wanted, Arena arena, DenseSampleConfig config)
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
        NativeGeospatialSample arguments = new()
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
        usearch_geospatial_sample(&arguments);
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

/// <summary>A checkpoint write; see <see cref="GeospatialIndexOn.Save"/>.</summary>
internal unsafe struct GeospatialSaveCall(CallHandles handles, string path, Progress? progress) : ICall
{
    private readonly byte[] _path = Encoding.UTF8.GetBytes(path);
    private MemoryHandle _pinnedPath;
    private Registration _progress;

    public void Submit(CallOutputs* outputs, Completion completion)
    {
        this._pinnedPath = this._path.AsMemory().Pin();
        using HandleReference index = handles.ReferenceIndex();
        using HandleReference executor = handles.ReferenceExecutor();
        NativeGeospatialSave arguments = new()
        {
            Index = index.Pointer, Executor = executor.Pointer, Outcome = &outputs->Outcome, Completion = completion,
            Path = new NativeBytesView { Data = (IntPtr)this._pinnedPath.Pointer, Length = (nuint)this._path.Length },
            Progress = this._progress.Progress(progress),
        };
        usearch_geospatial_save(&arguments);
    }

    public void Release(CallOutputs* outputs)
    {
        this._pinnedPath.Dispose();
        this._progress.Free();
    }

    public readonly void Finish(CallOutputs* outputs) => outputs->Outcome.ThrowIfFailed();
}

/// <summary>A checkpoint write into an arena; see <see cref="GeospatialIndexOn.SaveBuffer"/>.</summary>
internal unsafe struct GeospatialSaveBufferCall(CallHandles handles, Arena arena, Progress? progress) : ICall<Monolith>
{
    private Registration _progress;

    public Monolith Result { get; private set; }

    public void Submit(CallOutputs* outputs, Completion completion)
    {
        using HandleReference index = handles.ReferenceIndex();
        using HandleReference executor = handles.ReferenceExecutor();
        using HandleReference passed = Arena.Reference(arena);
        NativeGeospatialSaveBuffer arguments = new()
        {
            Index = index.Pointer, Executor = executor.Pointer, Outcome = &outputs->Outcome, Completion = completion,
            Monolith = (NativeBytesView*)outputs->View, Arena = passed.Pointer,
            Progress = this._progress.Progress(progress),
        };
        usearch_geospatial_save_buffer(&arguments);
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
internal unsafe struct NativeGeospatialTerrain
{
    public nuint Rows;
    public nuint Columns;
    public double NorthLatitudeDegrees;
    public double WestLongitudeDegrees;
    public double RowStepDegrees;
    public double ColumnStepDegrees;
    public float* Heights;
    public nuint HeightsStride;
    public IntPtr Callback;
    public IntPtr State;
    public NativeBytesView Path;
}

[StructLayout(LayoutKind.Sequential)]
internal struct NativeGeospatialMetadata
{
    public nuint Connectivity;
}

[StructLayout(LayoutKind.Sequential)]
internal struct NativeGeospatialDescription
{
    public NativeGeospatialMetadata Metadata;
    public nuint Size;
    public nuint Capacity;
    public nuint MemoryUsage;
    public int CpuClass;
    public int GpuClass;
    public nuint ShardsCount;
    public int LoadPath;
}

[StructLayout(LayoutKind.Sequential)]
internal struct NativeGeospatialKeyStats
{
    public nuint Entries;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeGeospatialInitRam
{
    public IntPtr* Index;
    public NativeOutcome* Outcome;
    public NativeGeospatialTerrain Terrain;
    public nuint Connectivity;
    public nuint ValuesPerKey;
    public NativeRamOptions Ram;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeGeospatialInitVram
{
    public IntPtr* Index;
    public NativeOutcome* Outcome;
    public NativeGeospatialTerrain Terrain;
    public nuint Connectivity;
    public nuint ValuesPerKey;
    public NativeVramOptions Vram;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeGeospatialLoadRam
{
    public IntPtr* Index;
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
    public NativeBytesView Path;
    public NativeRamOptions Ram;
    public nuint ValuesPerKey;
    public NativeGeospatialTerrain Terrain;
    public NativeProgress Progress;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeGeospatialLoadRamBuffer
{
    public IntPtr* Index;
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
    public NativeBytesView Buffer;
    public NativeRamOptions Ram;
    public nuint ValuesPerKey;
    public NativeGeospatialTerrain Terrain;
    public NativeProgress Progress;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeGeospatialLoadVram
{
    public IntPtr* Index;
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
    public NativeBytesView Path;
    public NativeVramOptions Vram;
    public nuint ValuesPerKey;
    public NativeGeospatialTerrain Terrain;
    public NativeProgress Progress;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeGeospatialLoadVramBuffer
{
    public IntPtr* Index;
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
    public NativeBytesView Buffer;
    public NativeVramOptions Vram;
    public nuint ValuesPerKey;
    public NativeGeospatialTerrain Terrain;
    public NativeProgress Progress;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeGeospatialViewRam
{
    public IntPtr* Index;
    public NativeOutcome* Outcome;
    public NativeBytesView Path;
    public NativeGeospatialTerrain Terrain;
    public NativeRamOptions Ram;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeGeospatialViewNvme
{
    public IntPtr* Index;
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
    public NativeBytesView Path;
    public NativeNvmeOptions Nvme;
    public NativeGeospatialTerrain Terrain;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeGeospatialReserve
{
    public IntPtr Index;
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
    public nuint Entries;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeGeospatialClear
{
    public IntPtr Index;
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeGeospatialRefine
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
internal unsafe struct NativeGeospatialRefinePlanFinish
{
    public IntPtr Index;
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
    public IntPtr Plan;
    public NativeRefineReport* Report;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeGeospatialHealth
{
    public IntPtr Index;
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
    public NativeDenseReachability* Reachability;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeGeospatialInsert
{
    public IntPtr Index;
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
    public nuint Count;
    public NativeKeysView Keys;
    public GeospatialObservation* Observations;
    public nuint ExpansionFactor;
    public NativeProgress Progress;
    public NativeDenseTraversalStats* Stats;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeGeospatialSearch
{
    public IntPtr Index;
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
    public nuint Count;
    public GeospatialObservation* Queries;
    public GeospatialRelation Relation;
    public GeospatialTolerance Tolerance;
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
internal unsafe struct NativeGeospatialWithin
{
    public IntPtr Index;
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
    public nuint Count;
    public GeospatialObservation* Queries;
    public GeospatialRelation Relation;
    public GeospatialTolerance Tolerance;
    public NativeMatchesTapeView* Matches;
    public nuint* Written;
    public IntPtr Arena;
    public Traversal Traversal;
    public NativePredicate Predicate;
    public NativeProgress Progress;
    public NativeDenseTraversalStats* Stats;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeGeospatialCount
{
    public IntPtr Index;
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
    public Key Key;
    public NativeGeospatialKeyStats* KeyStats;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeGeospatialGet
{
    public IntPtr Index;
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
    public Key Key;
    public GeospatialObservation** Observations;
    public nuint* Written;
    public IntPtr Arena;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeGeospatialReshard
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
internal unsafe struct NativeGeospatialReshardPlanFinish
{
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
    public IntPtr Plan;
    public NativeRefineReport* Report;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeGeospatialCursorInit
{
    public IntPtr Index;
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public nuint StartsCount;
    public Key* Starts;
    public IntPtr* Cursor;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeGeospatialCursorNext
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
internal unsafe struct NativeGeospatialSample
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
internal unsafe struct NativeGeospatialSave
{
    public IntPtr Index;
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
    public NativeBytesView Path;
    public NativeProgress Progress;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeGeospatialSaveBuffer
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
internal unsafe struct NativeGeospatialMetadataRead
{
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
    public NativeBytesView Path;
    public NativeGeospatialMetadata* Metadata;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeGeospatialMetadataReadBuffer
{
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
    public NativeBytesView Buffer;
    public NativeGeospatialMetadata* Metadata;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeGeospatialDescribe
{
    public IntPtr Index;
    public IntPtr Executor;
    public NativeOutcome* Outcome;
    public Completion Completion;
    public NativeGeospatialDescription* Description;
}

#endregion Native Vocabulary

#region Entry Points

/// <summary>The geospatial modality's entry points, each taking one argument struct.</summary>
internal static unsafe partial class NativeMethods
{
    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_geospatial_init_ram(NativeGeospatialInitRam* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_geospatial_init_vram(NativeGeospatialInitVram* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_geospatial_load_ram(NativeGeospatialLoadRam* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_geospatial_load_ram_buffer(NativeGeospatialLoadRamBuffer* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_geospatial_load_vram(NativeGeospatialLoadVram* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_geospatial_load_vram_buffer(NativeGeospatialLoadVramBuffer* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_geospatial_view_ram(NativeGeospatialViewRam* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_geospatial_view_nvme(NativeGeospatialViewNvme* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_geospatial_free(IntPtr index);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_geospatial_reserve(NativeGeospatialReserve* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_geospatial_clear(NativeGeospatialClear* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_geospatial_insert(NativeGeospatialInsert* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_geospatial_refine(NativeGeospatialRefine* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_geospatial_refine_plan_finish(NativeGeospatialRefinePlanFinish* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_geospatial_health(NativeGeospatialHealth* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_geospatial_search(NativeGeospatialSearch* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_geospatial_within(NativeGeospatialWithin* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_geospatial_count(NativeGeospatialCount* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_geospatial_get(NativeGeospatialGet* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_geospatial_reshard(NativeGeospatialReshard* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_geospatial_reshard_plan_finish(NativeGeospatialReshardPlanFinish* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_geospatial_cursor_init(NativeGeospatialCursorInit* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_geospatial_cursor_next(NativeGeospatialCursorNext* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_geospatial_cursor_free(IntPtr cursor);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_geospatial_sample(NativeGeospatialSample* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_geospatial_save(NativeGeospatialSave* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_geospatial_save_buffer(NativeGeospatialSaveBuffer* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_geospatial_metadata_read(NativeGeospatialMetadataRead* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_geospatial_metadata_read_buffer(NativeGeospatialMetadataReadBuffer* arguments);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_geospatial_describe(NativeGeospatialDescribe* arguments);
}

#endregion Entry Points

#region Handles

/// <summary>A <c>usearch_geospatial_index_t</c>, freed when the wrapper is disposed or collected.</summary>
internal sealed class NativeGeospatialIndexHandle : EngineHandle
{
    /// <inheritdoc/>
    protected override bool ReleaseHandle()
    {
        NativeMethods.usearch_geospatial_free(this.handle);
        return true;
    }
}

/// <summary>A <c>usearch_geospatial_cursor_t</c> and the reference on the index every pull names, both
/// released when the wrapper is disposed or collected.</summary>
internal sealed class NativeGeospatialCursorHandle : EngineHandle
{
    private HandleReference _index;

    internal NativeGeospatialCursorHandle(IntPtr cursor, HandleReference index)
    {
        this.SetHandle(cursor);
        this._index = index;
    }

    /// <summary>The index the cursor was opened on.</summary>
    internal IntPtr Index => this._index.Pointer;

    /// <inheritdoc/>
    protected override bool ReleaseHandle()
    {
        NativeMethods.usearch_geospatial_cursor_free(this.handle);
        this._index.Dispose();
        return true;
    }
}

#endregion Handles
