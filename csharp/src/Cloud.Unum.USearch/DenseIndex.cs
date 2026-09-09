using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;

using static Cloud.Unum.USearch.NativeMethods;

namespace Cloud.Unum.USearch;

#region Options

/// <summary>
/// A distance kind supported out of the box, with SIMD kernels dispatched by the machine's
/// capabilities.
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
    /// <summary>The inner product - the only kind the late-interaction verb accepts.</summary>
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

/// <summary>
/// The construction and loading options of a dense index, fixed once and read by the handle that
/// is built from them.
/// </summary>
/// <remarks>
/// Stored rows always round up to the SIMD step quantum, so rows stay cache-line aligned and every
/// kernel runs tail-free; the arena is slightly larger for it, and that is not a knob.
/// <para>
/// Every field is init-only, so a variant comes from <c>DenseOptions.Default with { ... }</c>
/// rather than from mutating a value another handle may already have been built from.
/// </para>
/// </remarks>
public readonly struct DenseOptions
{
    /// <summary>The width of the indexed vectors; zero defers the geometry to a later
    /// <see cref="DenseIndex.Load"/>, which reads it off the checkpoint.</summary>
    public ulong Dimensions { get; init; }

    /// <summary>The distance kind used between vectors.</summary>
    public DenseMetric Metric { get; init; }

    /// <summary>The scalar type vectors are stored and quantized into.</summary>
    public ScalarKind Quantization { get; init; }

    /// <summary>The connections-per-node bound of the graph; zero derives.</summary>
    public ulong Connectivity { get; init; }

    /// <summary>The upper bound on entries sharing one key; zero derives to one - unique keys.</summary>
    public ulong ValuesPerKey { get; init; }

    /// <summary>Where the payload lives, and what loading a checkpoint means.</summary>
    public ResidencyKind Residency { get; init; }

    /// <summary>An angular <see cref="ScalarKind.BF16"/> index in RAM, with every optional field
    /// derived.</summary>
    public static DenseOptions Default => new()
    {
        Metric = DenseMetric.Angular,
        Quantization = ScalarKind.BF16,
        Residency = ResidencyKind.Ram,
    };

    /// <summary>
    /// Lowers the options, leaving the custom-metric callback null and every placement list empty.
    /// </summary>
    /// <remarks>
    /// Placement within a residency is derived, so the memory domains, device ordinals, and device
    /// paths cross as null runs and the engine spreads over whatever the machine offers.
    /// </remarks>
    /// <returns>The C struct the library reads.</returns>
    internal NativeDenseOptions Lower()
    {
        if (this.Residency is not (ResidencyKind.Ram or ResidencyKind.Vram or ResidencyKind.Nvme))
        {
            throw new USearchException(Status.UnknownKind, "the residency names neither RAM, VRAM, nor NVMe");
        }

        return new NativeDenseOptions
        {
            Metric = this.Metric,
            Quantization = this.Quantization,
            Dimensions = (nuint)this.Dimensions,
            Connectivity = (nuint)this.Connectivity,
            ValuesPerKey = (nuint)this.ValuesPerKey,
            Residency = this.Residency,
        };
    }

    /// <summary>Lifts a lowered struct back into managed options.</summary>
    /// <remarks>
    /// The residency arm carries placement lists the lift cannot reconstruct, so the lifted options
    /// name RAM and the caller overwrites that before constructing a handle.
    /// </remarks>
    /// <param name="lowered">The C struct the library reads.</param>
    /// <returns>The managed options.</returns>
    internal static DenseOptions Lift(in NativeDenseOptions lowered)
    {
        if (lowered.Metric < DenseMetric.Inner || lowered.Metric > DenseMetric.Jaccard)
        {
            throw new USearchException(Status.UnknownKind, "the options name an unknown metric");
        }

        if (lowered.Quantization == ScalarKind.Unknown || !Enum.IsDefined(typeof(ScalarKind), lowered.Quantization))
        {
            throw new USearchException(Status.UnknownKind, "the options name an unknown scalar kind");
        }

        return new DenseOptions
        {
            Dimensions = lowered.Dimensions,
            Metric = lowered.Metric,
            Quantization = lowered.Quantization,
            Connectivity = lowered.Connectivity,
            ValuesPerKey = lowered.ValuesPerKey,
            Residency = ResidencyKind.Ram,
        };
    }

    /// <summary>
    /// Lowers the options and lifts them straight back - the marshalling round trip, and the only
    /// thing that exercises the residency union without a library call. A test cannot name the
    /// native structs, so the round trip is reachable only from here.
    /// </summary>
    /// <returns>The options the round trip reconstructs, placement excepted.</returns>
    internal DenseOptions Reload() => Lift(this.Lower());
}

/// <summary>
/// The geometry and metric a checkpoint froze - everything a reader learns without opening it, and
/// nothing a reader must choose for itself.
/// </summary>
/// <remarks>
/// A checkpoint records neither the residency nor the upper bound on entries sharing one key: the
/// first says where it was written rather than where a reader must keep it, and the second bounds a
/// live handle's mutable state rather than the serialized form. Both stay the caller's to choose in
/// the <see cref="DenseOptions"/> a handle is built from.
/// </remarks>
public readonly struct DenseMetadata
{
    /// <summary>The scalars per stored vector.</summary>
    public ulong Dimensions { get; init; }

    /// <summary>The distance kind the checkpoint scores with.</summary>
    public DenseMetric Metric { get; init; }

    /// <summary>The scalar type its rows are stored and quantized into.</summary>
    public ScalarKind Quantization { get; init; }

    /// <summary>The connections-per-node bound the graph was built with.</summary>
    public ulong Connectivity { get; init; }

    /// <summary>Lifts the geometry and metric a checkpoint's headers describe.</summary>
    /// <param name="lowered">The C struct the library wrote.</param>
    /// <returns>The managed metadata.</returns>
    internal static DenseMetadata Lift(in NativeDenseMetadata lowered)
    {
        if (lowered.Metric < DenseMetric.Inner || lowered.Metric > DenseMetric.Jaccard)
        {
            throw new USearchException(Status.UnknownKind, "the checkpoint names an unknown metric");
        }

        if (lowered.Quantization == ScalarKind.Unknown || !Enum.IsDefined(typeof(ScalarKind), lowered.Quantization))
        {
            throw new USearchException(Status.UnknownKind, "the checkpoint names an unknown scalar kind");
        }

        return new DenseMetadata
        {
            Dimensions = lowered.Dimensions,
            Metric = lowered.Metric,
            Quantization = lowered.Quantization,
            Connectivity = lowered.Connectivity,
        };
    }
}

#endregion Options

#region Per-Call Settings & Telemetry

/// <summary>
/// The optional per-call settings of <see cref="DenseIndex.Add"/> and
/// <see cref="DenseIndex.Refine"/>. A default value derives every setting.
/// </summary>
public readonly struct DenseInsertionOptions
{
    /// <summary>The candidate-pool breadth the build walk explores; zero derives it from the
    /// geometry.</summary>
    public ulong ExpansionFactor { get; init; }

    /// <summary>Renders the settings as the struct the library reads.</summary>
    /// <returns>The C struct the library reads.</returns>
    internal NativeInsertionOptions Lower() => new() { ExpansionFactor = (nuint)this.ExpansionFactor };
}

/// <summary>
/// The optional per-call settings of the read-only walks - <see cref="DenseIndex.Search"/>,
/// <see cref="DenseIndex.SearchIf"/> and <see cref="DenseIndex.MaxSim"/>. A default value derives
/// every setting.
/// </summary>
/// <remarks>
/// The type stays distinct from <see cref="DenseInsertionOptions"/> even where the two agree field
/// for field, so a knob either side gains reaches that side alone.
/// </remarks>
public readonly struct DenseSearchOptions
{
    /// <summary>The candidate-pool breadth the walk explores; zero derives it from the
    /// geometry.</summary>
    public ulong ExpansionFactor { get; init; }

    /// <summary>Renders the settings as the struct the library reads.</summary>
    /// <returns>The C struct the library reads.</returns>
    internal NativeSearchOptions Lower() => new() { ExpansionFactor = (nuint)this.ExpansionFactor };
}

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

#endregion Per-Call Settings & Telemetry

#region Index

/// <summary>
/// A Vector Search Index over equidimensional dense representations - the managed twin of
/// <c>include/usearch/dense_index.h</c>.
/// </summary>
/// <remarks>
/// Every verb is a batch over a <see cref="Matrix"/>, so a single vector is a one-row matrix. The
/// candidate-pool breadth rides each call rather than the handle, which lets a recall sweep run
/// against one immutable index with no mutation between points.
/// <para>
/// Every fanned-out verb takes the <see cref="Executor"/> it runs on, whose kind must match the
/// residency the handle was built for. There is no derived pool and no overload that leaves the
/// executor out: the engine reads a missing one as a residency mismatch, so the absence is a
/// compile-time one instead.
/// </para>
/// <para>
/// A handle may be shared across threads, so no method here takes a lock. On a RAM handle,
/// insertions and searches overlap freely: searches read only entries published behind the
/// visibility watermark and never write to the graph, while insertions synchronize through
/// per-node locks and publish in order. Serialization is the one exception, demanding a
/// mutation-free window, though concurrent readers keep searching throughout.
/// </para>
/// </remarks>
public sealed unsafe class DenseIndex : IDisposable
{
    private readonly NativeIndexHandle _handle;

    /// <summary>Initializes an index for the residency its options select.</summary>
    /// <param name="options">The construction and loading options.</param>
    public DenseIndex(DenseOptions options)
    {
        NativeDenseOptions lowered = options.Lower();
        this._handle = usearch_dense_init(&lowered, out NativeError error);
        error.ThrowIfFailed();
        if (this._handle.IsInvalid)
        {
            throw new USearchException(Status.Internal, "the library returned no handle and no error");
        }
    }

    /// <summary>
    /// Reads a checkpoint's geometry and metric metadata without loading it - from the manifest's
    /// summary when the path names one, so no shard is opened.
    /// </summary>
    /// <param name="path">The checkpoint path.</param>
    /// <returns>The geometry and metric the checkpoint's headers describe.</returns>
    public static DenseMetadata Metadata(string path)
    {
        byte[] encoded = EncodePath(path);
        NativeDenseMetadata lowered = default;
        fixed (byte* data = encoded)
        {
            usearch_dense_metadata(Borrow(data, encoded.Length), &lowered, out NativeError error);
            error.ThrowIfFailed();
        }

        return DenseMetadata.Lift(lowered);
    }

    /// <summary>Reads geometry and metric metadata from a serialized monolith buffer.</summary>
    /// <param name="buffer">The monolith's bytes.</param>
    /// <returns>The geometry and metric the monolith's header describes.</returns>
    public static DenseMetadata MetadataFromBuffer(ReadOnlySpan<byte> buffer)
    {
        if (buffer.IsEmpty)
        {
            throw new USearchException(Status.MalformedInput, "the buffer is empty");
        }

        NativeDenseMetadata lowered = default;
        fixed (byte* source = buffer)
        {
            usearch_dense_metadata_buffer(Borrow(source, buffer.Length), &lowered, out NativeError error);
            error.ThrowIfFailed();
        }

        return DenseMetadata.Lift(lowered);
    }

    /// <summary>Claims capacity for incoming vectors, counting the entries already stored.</summary>
    /// <param name="capacity">The desired total capacity.</param>
    public void Reserve(ulong capacity)
    {
        usearch_dense_reserve(this.Handle, (nuint)capacity, out NativeError error);
        error.ThrowIfFailed();
    }

    /// <summary>Erases all entries, keeping the reserved capacity and the frozen geometry.</summary>
    public void Clear()
    {
        usearch_dense_clear(this.Handle, out NativeError error);
        error.ThrowIfFailed();
    }

    /// <summary>Reports the number of stored entries.</summary>
    /// <returns>The entry count.</returns>
    public ulong Size()
    {
        nuint reading = usearch_dense_size(this.Handle, out NativeError error);
        error.ThrowIfFailed();
        return reading;
    }

    /// <summary>Reports the reserved capacity in entries.</summary>
    /// <returns>The capacity.</returns>
    public ulong Capacity()
    {
        nuint reading = usearch_dense_capacity(this.Handle, out NativeError error);
        error.ThrowIfFailed();
        return reading;
    }

    /// <summary>Reports the frozen width of the stored vectors; zero until a load freezes it.</summary>
    /// <returns>The dimensionality.</returns>
    public ulong Dimensions()
    {
        nuint reading = usearch_dense_dimensions(this.Handle, out NativeError error);
        error.ThrowIfFailed();
        return reading;
    }

    /// <summary>Reports the frozen connections-per-node bound of the graph.</summary>
    /// <returns>The connectivity.</returns>
    public ulong Connectivity()
    {
        nuint reading = usearch_dense_connectivity(this.Handle, out NativeError error);
        error.ThrowIfFailed();
        return reading;
    }

    /// <summary>Reports the memory the index occupies, in bytes.</summary>
    /// <returns>The byte count.</returns>
    public ulong MemoryUsage()
    {
        nuint reading = usearch_dense_memory_usage(this.Handle, out NativeError error);
        error.ThrowIfFailed();
        return reading;
    }

    /// <summary>
    /// Reports the bytes one monolith of the index takes under the default layout - what
    /// <see cref="SaveToBuffer"/> needs. A manifest checkpoint's parts sum to about the same, plus
    /// the manifest.
    /// </summary>
    /// <returns>The byte count.</returns>
    public ulong StorageRequirements()
    {
        nuint reading = usearch_dense_storage_requirements(this.Handle, out NativeError error);
        error.ThrowIfFailed();
        return reading;
    }

    /// <summary>Reports the capability codename the index dispatched to on this machine.</summary>
    /// <returns>The codename.</returns>
    public string Capabilities()
    {
        IntPtr reading = usearch_dense_capabilities(this.Handle, out NativeError error);
        error.ThrowIfFailed();
        return Marshal.PtrToStringUTF8(reading)
            ?? throw new USearchException(Status.Internal, "the library reported no capabilities and no error");
    }

    /// <summary>
    /// Inserts a matrix of vectors under their keys - the native batch operation, which overlaps
    /// safely with concurrent searches on the same handle.
    /// </summary>
    /// <remarks>
    /// The input scalar type is independent of the stored quantization; the library converts on
    /// ingest.
    /// </remarks>
    /// <param name="executor">The workers the insertion runs on.</param>
    /// <param name="keys">One identity per row.</param>
    /// <param name="vectors">The rows to admit.</param>
    /// <param name="options">The per-call settings; a default value derives every one.</param>
    /// <param name="stats">The traversal telemetry to fill, when the caller wants any.</param>
    /// <returns>The row count on success and zero on failure: every fallible step runs before the
    /// first entry identity is claimed, so a rejected batch publishes nothing.</returns>
    public ulong Add(
        Executor executor,
        ReadOnlySpan<USearchKey> keys,
        Matrix vectors,
        DenseInsertionOptions options = default,
        TraversalStats? stats = null)
    {
        NativeExecutorHandle pool = Executor.HandleOf(executor);
        vectors.CheckWidth(this.FrozenDimensions());
        if (keys.Length != vectors.Rows)
        {
            throw new USearchException(
                Status.ShapeMismatch, "the key count does not match the row count it accompanies");
        }

        if (keys.IsEmpty)
        {
            return 0;
        }

        NativeInsertionOptions lowered = options.Lower();
        NativeTraversalStats reported = default;
        nuint added;
        fixed (USearchKey* keysPointer = keys)
        fixed (byte* vectorsPointer = vectors.Bytes)
        {
            NativeKeysView run = new() { Data = (IntPtr)keysPointer, Count = (nuint)keys.Length };
            NativeMatrixView view = vectors.AsView(vectorsPointer);
            added = usearch_dense_add(
                this.Handle, pool, run, &view, &lowered,
                IntPtr.Zero, IntPtr.Zero, &reported, out NativeError error);
            error.ThrowIfFailed();
        }

        stats?.Adopt(reported);
        return added;
    }

    /// <summary>
    /// Re-selects the neighbors of every stored entry against the finished graph, improving edges
    /// that were chosen when the graph around them was still incomplete.
    /// </summary>
    /// <remarks>
    /// This is a build step rather than a repair: a cheap insertion pass followed by one refinement
    /// reaches a better graph, in less total time, than a single expensive insertion pass - so a
    /// wide expansion factor belongs here rather than in <see cref="Add"/>. Searches may run
    /// throughout, since edge re-selection publishes one row at a time and never removes a node.
    /// </remarks>
    /// <param name="executor">The workers the sweep fans out over.</param>
    /// <param name="options">The per-call settings; a default value derives every one.</param>
    /// <param name="stats">The traversal telemetry to fill, when the caller wants any.</param>
    /// <returns>The number of nodes whose edges changed; zero means the graph has settled, so a
    /// caller under a time budget loops until the return falls below whatever rate it considers
    /// settled.</returns>
    public ulong Refine(Executor executor, DenseInsertionOptions options = default, TraversalStats? stats = null)
    {
        NativeExecutorHandle pool = Executor.HandleOf(executor);
        NativeInsertionOptions lowered = options.Lower();
        NativeTraversalStats reported = default;
        nuint changed = usearch_dense_refine(
            this.Handle, pool, &lowered, IntPtr.Zero, IntPtr.Zero, &reported, out NativeError error);
        error.ThrowIfFailed();
        stats?.Adopt(reported);
        return changed;
    }

    /// <summary>
    /// Resolves a matrix of queries into the nearest entries - the native batch operation,
    /// parallelized internally, which never writes to the graph.
    /// </summary>
    /// <param name="executor">The workers the batch runs on.</param>
    /// <param name="queries">The query rows, one query apiece.</param>
    /// <param name="matches">Where the run lands, sized for as many rows as there are queries.</param>
    /// <param name="options">The per-call settings; a default value derives every one.</param>
    /// <param name="stats">The traversal telemetry to fill, when the caller wants any.</param>
    public void Search(
        Executor executor,
        Matrix queries,
        MatchesSpan matches,
        DenseSearchOptions options = default,
        TraversalStats? stats = null) =>
        this.SearchDispatch(executor, queries, matches, options, stats, null);

    /// <summary>
    /// The filtered sibling of <see cref="Search"/>, admitting only the keys the predicate
    /// approves. Non-matching entries still route the traversal, so recall does not collapse at
    /// low selectivity.
    /// </summary>
    /// <remarks>
    /// The predicate runs on the executor's worker threads, so it must be safe for concurrent use,
    /// and it is called once per scored candidate rather than once per reported match. An exception
    /// it throws cannot cross the boundary: the first one is captured, every later candidate is
    /// rejected, and the failure is raised here once the call returns.
    /// </remarks>
    /// <param name="executor">The workers the batch runs on.</param>
    /// <param name="queries">The query rows, one query apiece.</param>
    /// <param name="matches">Where the run lands, sized for as many rows as there are queries.</param>
    /// <param name="admits">The per-key admission predicate.</param>
    /// <param name="options">The per-call settings; a default value derives every one.</param>
    /// <param name="stats">The traversal telemetry to fill, when the caller wants any.</param>
    public void SearchIf(
        Executor executor,
        Matrix queries,
        MatchesSpan matches,
        Func<USearchKey, Admission> admits,
        DenseSearchOptions options = default,
        TraversalStats? stats = null)
    {
        if (admits is null)
        {
            throw new USearchException(Status.MalformedInput, "the admission predicate is null");
        }

        this.SearchDispatch(executor, queries, matches, options, stats, admits);
    }

    /// <summary>
    /// Scores multi-row queries against multi-row entries by late interaction: every query row
    /// takes its best match within a key, and those maxima sum to that key's score.
    /// </summary>
    /// <remarks>
    /// A document is one key holding its token rows, so this is what a values-per-key bound above
    /// one exists for. It runs on the inner-product kinds alone, since a maximum over logits is
    /// what late interaction sums, and a graph frozen under a distance kind refuses with
    /// <see cref="Status.ConfigMismatch"/>.
    /// </remarks>
    /// <param name="executor">The workers the batch runs on.</param>
    /// <param name="queries">The token rows of every query, laid end to end.</param>
    /// <param name="rowsPerQuery">The rows in every query; uniform, so the row count must be a
    /// whole multiple of it and the quotient is the query count.</param>
    /// <param name="matches">Where the scored keys land - the wanted count is the "k" over
    /// documents rather than over rows.</param>
    /// <param name="options">The per-call settings; a default value derives every one.</param>
    /// <param name="stats">The traversal telemetry to fill, when the caller wants any.</param>
    public void MaxSim(
        Executor executor,
        Matrix queries,
        int rowsPerQuery,
        MatchesSpan matches,
        DenseSearchOptions options = default,
        TraversalStats? stats = null)
    {
        NativeExecutorHandle pool = Executor.HandleOf(executor);
        CheckOutputs(matches);
        queries.CheckWidth(this.FrozenDimensions());
        if (rowsPerQuery <= 0 || queries.Rows % rowsPerQuery != 0)
        {
            throw new USearchException(
                Status.ShapeMismatch, "the token rows are not a whole multiple of the rows per query");
        }

        int queryCount = queries.Rows / rowsPerQuery;
        matches.CheckCapacity((ulong)queryCount);
        if (queryCount == 0)
        {
            return;
        }

        NativeSearchOptions lowered = options.Lower();
        NativeTraversalStats reported = default;
        Span<USearchKey> keysSlab = matches.KeysSlab;
        Span<double> distancesSlab = matches.DistancesSlab;
        Span<nuint> countsSlab = matches.CountsSlab;
        fixed (byte* queriesPointer = queries.Bytes)
        fixed (USearchKey* keysPointer = keysSlab)
        fixed (double* distancesPointer = distancesSlab)
        fixed (nuint* countsPointer = countsSlab)
        {
            NativeMatrixView view = queries.AsView(queriesPointer);
            NativeMatchesSpan run = Land(keysPointer, distancesPointer, countsPointer, matches.Wanted);
            usearch_dense_maxsim(
                this.Handle, pool, &view, (nuint)rowsPerQuery, &run, &lowered,
                IntPtr.Zero, IntPtr.Zero, &reported, out NativeError error);
            error.ThrowIfFailed();
        }

        stats?.Adopt(reported);
    }

    /// <summary>Counts the entries stored under a key; zero means the index holds none.</summary>
    /// <param name="key">The identity to look for.</param>
    /// <returns>How many entries share that key.</returns>
    public ulong Count(USearchKey key)
    {
        nuint counted = usearch_dense_count(this.Handle, key, out NativeError error);
        error.ThrowIfFailed();
        return counted;
    }

    /// <summary>Reports whether the index holds at least one entry under a key.</summary>
    /// <param name="key">The identity to look for.</param>
    /// <returns>True when the index holds one or more.</returns>
    public bool Contains(USearchKey key) => this.Count(key) != 0;

    /// <summary>
    /// Retrieves the vectors stored under a key into a matrix, one row per entry sharing the key,
    /// in insertion order.
    /// </summary>
    /// <param name="key">The identity to look for.</param>
    /// <param name="vectors">The rows to fill; a default value probes the count without copying.</param>
    /// <returns>The true number of entries, never the number written: a count above the matrix's
    /// row capacity means the copy was truncated, so size for that many and call again.</returns>
    public ulong Get(USearchKey key, Matrix vectors = default)
    {
        if (vectors.IsEmpty)
        {
            nuint probed = usearch_dense_get(this.Handle, key, null, out NativeError probing);
            probing.ThrowIfFailed();
            return probed;
        }

        vectors.CheckWidth(this.FrozenDimensions());
        nuint entries;
        fixed (byte* vectorsPointer = vectors.Bytes)
        {
            NativeMatrixSpan span = vectors.AsOutput(vectorsPointer);
            entries = usearch_dense_get(this.Handle, key, &span, out NativeError error);
            error.ThrowIfFailed();
        }

        return entries;
    }

    /// <summary>Opens a consistent snapshot cursor over the current contents.</summary>
    /// <returns>The cursor, which the caller disposes.</returns>
    public USearchCursor Cursor()
    {
        NativeCursorHandle cursor = usearch_dense_cursor(this.Handle, out NativeError error);
        error.ThrowIfFailed();
        if (cursor.IsInvalid)
        {
            throw new USearchException(Status.Internal, "the library returned no cursor and no error");
        }

        return new USearchCursor(cursor);
    }

    /// <summary>
    /// Opens a resumable nearest-neighbor stream for one query. A live finder holds one of the
    /// executor's threads until it is disposed, so an executor with few threads and many open
    /// finders starves.
    /// </summary>
    /// <param name="executor">The workers the stream draws one thread from.</param>
    /// <param name="query">The single query row, whose row count must be one.</param>
    /// <returns>The finder, which the caller disposes.</returns>
    public USearchFinder Find(Executor executor, Matrix query)
    {
        NativeExecutorHandle pool = Executor.HandleOf(executor);
        query.CheckWidth(this.FrozenDimensions());
        if (query.Rows != 1)
        {
            throw new USearchException(Status.ShapeMismatch, "a finder takes exactly one query row");
        }

        NativeFinderHandle finder;
        fixed (byte* queryPointer = query.Bytes)
        {
            NativeMatrixView view = query.AsView(queryPointer);
            finder = usearch_dense_find(this.Handle, pool, &view, out NativeError error);
            error.ThrowIfFailed();
        }

        if (finder.IsInvalid)
        {
            throw new USearchException(Status.Internal, "the library returned no finder and no error");
        }

        return new USearchFinder(finder);
    }

    /// <summary>
    /// Writes the index under a path, in the layout its extension picks - one <c>.safegraphs</c>
    /// monolith, or a <c>.safegraphs.json</c> manifest beside its shards. Demands a mutation-free
    /// window; concurrent readers keep searching throughout.
    /// </summary>
    /// <param name="executor">The workers the write fans out over, one shard apiece.</param>
    /// <param name="path">Where the checkpoint is written.</param>
    public void Save(Executor executor, string path)
    {
        NativeExecutorHandle pool = Executor.HandleOf(executor);
        byte[] encoded = EncodePath(path);
        fixed (byte* data = encoded)
        {
            usearch_dense_save(
                this.Handle, pool, Borrow(data, encoded.Length), IntPtr.Zero, IntPtr.Zero, out NativeError error);
            error.ThrowIfFailed();
        }
    }

    /// <summary>
    /// Reads a checkpoint into this handle, shard by shard, with the handle's residency deciding
    /// the behavior: materialize into RAM, upload to the devices, or open in place on disk. A
    /// handle whose geometry was left zeroed freezes it here, from the checkpoint's own headers.
    /// </summary>
    /// <param name="executor">The workers the read fans out over, one shard apiece.</param>
    /// <param name="path">Where the checkpoint is read from.</param>
    public void Load(Executor executor, string path)
    {
        NativeExecutorHandle pool = Executor.HandleOf(executor);
        byte[] encoded = EncodePath(path);
        fixed (byte* data = encoded)
        {
            usearch_dense_load(
                this.Handle, pool, Borrow(data, encoded.Length), IntPtr.Zero, IntPtr.Zero, out NativeError error);
            error.ThrowIfFailed();
        }
    }

    /// <summary>
    /// Writes the index into a caller-owned buffer as one monolith. Size the buffer with
    /// <see cref="StorageRequirements"/>.
    /// </summary>
    /// <param name="executor">The workers the write fans out over.</param>
    /// <param name="buffer">The bytes to write into.</param>
    public void SaveToBuffer(Executor executor, Span<byte> buffer)
    {
        NativeExecutorHandle pool = Executor.HandleOf(executor);
        if (buffer.IsEmpty)
        {
            throw new USearchException(Status.ShapeMismatch, "the buffer is empty");
        }

        fixed (byte* target = buffer)
        {
            usearch_dense_save_buffer(
                this.Handle, pool, Lend(target, buffer.Length), IntPtr.Zero, IntPtr.Zero, out NativeError error);
            error.ThrowIfFailed();
        }
    }

    /// <summary>
    /// Reads a monolith from a buffer into this handle, with the residency deciding the behavior
    /// like <see cref="Load"/>. A residency that opens in place demands the buffer outlive the
    /// handle.
    /// </summary>
    /// <param name="executor">The workers the read fans out over.</param>
    /// <param name="buffer">The monolith's bytes.</param>
    public void LoadFromBuffer(Executor executor, ReadOnlySpan<byte> buffer)
    {
        NativeExecutorHandle pool = Executor.HandleOf(executor);
        if (buffer.IsEmpty)
        {
            throw new USearchException(Status.MalformedInput, "the buffer is empty");
        }

        fixed (byte* source = buffer)
        {
            usearch_dense_load_buffer(
                this.Handle, pool, Borrow(source, buffer.Length), IntPtr.Zero, IntPtr.Zero, out NativeError error);
            error.ThrowIfFailed();
        }
    }

    /// <summary>Releases the index. Freeing cannot fail, and a freed handle must not be used.</summary>
    public void Dispose() => this._handle.Dispose();

    private NativeIndexHandle Handle => !this._handle.IsClosed && !this._handle.IsInvalid
        ? this._handle
        : throw new USearchException(Status.Uninitialized, "the index is disposed or was never initialized");

    private void SearchDispatch(
        Executor executor,
        Matrix queries,
        MatchesSpan matches,
        DenseSearchOptions options,
        TraversalStats? stats,
        Func<USearchKey, Admission>? admits)
    {
        NativeExecutorHandle pool = Executor.HandleOf(executor);
        CheckOutputs(matches);
        queries.CheckWidth(this.FrozenDimensions());
        matches.CheckCapacity((ulong)queries.Rows);
        if (queries.Rows == 0)
        {
            return;
        }

        NativeSearchOptions lowered = options.Lower();
        NativeTraversalStats reported = default;
        Span<USearchKey> keysSlab = matches.KeysSlab;
        Span<double> distancesSlab = matches.DistancesSlab;
        Span<nuint> countsSlab = matches.CountsSlab;
        AdmissionTrampoline? trampoline = admits is null ? null : new AdmissionTrampoline(admits);
        GCHandle registration = trampoline is null ? default : GCHandle.Alloc(trampoline);
        try
        {
            fixed (byte* queriesPointer = queries.Bytes)
            fixed (USearchKey* keysPointer = keysSlab)
            fixed (double* distancesPointer = distancesSlab)
            fixed (nuint* countsPointer = countsSlab)
            {
                NativeMatrixView view = queries.AsView(queriesPointer);
                NativeMatchesSpan run = Land(keysPointer, distancesPointer, countsPointer, matches.Wanted);
                NativeError error;
                if (trampoline is null)
                {
                    usearch_dense_search(
                        this.Handle, pool, &view, &run, &lowered,
                        IntPtr.Zero, IntPtr.Zero, &reported, out error);
                }
                else
                {
                    usearch_dense_search_if(
                        this.Handle, pool, &view, &run, &lowered,
                        &Admits, GCHandle.ToIntPtr(registration),
                        IntPtr.Zero, IntPtr.Zero, &reported, out error);
                }

                trampoline?.Rethrow();
                error.ThrowIfFailed();
            }
        }
        finally
        {
            if (registration.IsAllocated)
            {
                registration.Free();
            }
        }

        stats?.Adopt(reported);
    }

    /// <summary>
    /// The per-key admission callback the engine calls from its own worker threads, reaching the
    /// managed predicate through the state pointer the search registered alongside it.
    /// </summary>
    /// <param name="key">The candidate entry's identity.</param>
    /// <param name="state">The pinned trampoline, forwarded verbatim.</param>
    /// <returns>Whether the key joins the matches.</returns>
    [UnmanagedCallersOnly(CallConvs = [typeof(CallConvCdecl)])]
    private static Admission Admits(USearchKey key, IntPtr state) =>
        GCHandle.FromIntPtr(state).Target is AdmissionTrampoline trampoline
            ? trampoline.Admit(key)
            : Admission.Reject;

    /// <summary>
    /// Reports the frozen width, refusing the verbs that cannot be shaped without one. A zeroed
    /// geometry is legal until a checkpoint freezes it, and every extent derived from it is zero.
    /// </summary>
    private ulong FrozenDimensions()
    {
        ulong dimensions = this.Dimensions();
        if (dimensions == 0)
        {
            throw new USearchException(
                Status.Uninitialized,
                "the geometry is not frozen: construct with dimensions, or load a checkpoint that carries them");
        }

        return dimensions;
    }

    private static void CheckOutputs(MatchesSpan matches)
    {
        if (matches is null)
        {
            throw new USearchException(Status.MalformedInput, "the outputs are null");
        }
    }

    private static byte[] EncodePath(string path)
    {
        if (string.IsNullOrEmpty(path))
        {
            throw new USearchException(Status.MalformedInput, "the path is empty");
        }

        return Encoding.UTF8.GetBytes(path);
    }

    private static NativeBytesView Borrow(byte* data, int length) =>
        new() { Data = (IntPtr)data, Length = (nuint)length };

    private static NativeBytesSpan Lend(byte* data, int length) =>
        new() { Data = (IntPtr)data, Length = (nuint)length };

    private static NativeMatchesSpan Land(USearchKey* keys, double* scores, nuint* counts, int wanted) => new()
    {
        Keys = (IntPtr)keys,
        Scores = (IntPtr)scores,
        Counts = (IntPtr)counts,
        Wanted = (nuint)wanted,
    };

    /// <summary>
    /// Holds a managed predicate the engine calls from its own worker threads, and the first
    /// exception it threw. A managed exception must not unwind through native frames, so it is
    /// captured here and raised once the call has returned.
    /// </summary>
    private sealed class AdmissionTrampoline
    {
        private readonly Func<USearchKey, Admission> _admits;
        private Exception? _escaped;

        internal AdmissionTrampoline(Func<USearchKey, Admission> admits) => this._admits = admits;

        internal Admission Admit(USearchKey key)
        {
            if (Volatile.Read(ref this._escaped) is not null)
            {
                return Admission.Reject;
            }

            try
            {
                return this._admits(key);
            }
            // Every exception is caught on purpose: this frame is called from a native worker
            // thread, and unwinding through native frames terminates the process.
#pragma warning disable CA1031
            catch (Exception failure)
#pragma warning restore CA1031
            {
                Interlocked.CompareExchange(ref this._escaped, failure, null);
                return Admission.Reject;
            }
        }

        internal void Rethrow()
        {
            if (this._escaped is Exception failure)
            {
                throw new USearchException(Status.Internal, "the admission predicate threw", failure);
            }
        }
    }
}

#endregion Index

#region Cursors & Finders

/// <summary>
/// A consistent snapshot cursor over the index contents. It captures the publication watermark at
/// creation, so concurrent inserts never appear mid-iteration, and the walk is a sequential row
/// scan - export, migration, and visualization feed off it.
/// </summary>
public sealed unsafe class USearchCursor : IDisposable
{
    private readonly NativeCursorHandle _handle;

    internal USearchCursor(NativeCursorHandle handle) => this._handle = handle;

    /// <summary>Copies the next batch of entries out of the snapshot.</summary>
    /// <remarks>
    /// The keys are what every batch carries, so their capacity is the batch size and a vector
    /// export must have room for as many.
    /// </remarks>
    /// <param name="keys">The keys to fill; their capacity bounds the batch.</param>
    /// <param name="vectors">The rows to fill, one per key; a default value skips vector export.</param>
    /// <returns>How many rows were written; zero when the snapshot is exhausted.</returns>
    public int Next(Span<USearchKey> keys, Matrix vectors = default)
    {
        if (keys.IsEmpty)
        {
            throw new USearchException(Status.ShapeMismatch, "a batch must hold at least one key");
        }

        if (!vectors.IsEmpty && vectors.Rows < keys.Length)
        {
            throw new USearchException(Status.ShapeMismatch, "the vector rows do not cover every requested key");
        }

        nuint written;
        fixed (USearchKey* keysPointer = keys)
        fixed (byte* vectorsPointer = vectors.Bytes)
        {
            NativeKeysSpan run = new() { Data = (IntPtr)keysPointer, Capacity = (nuint)keys.Length };
            NativeMatrixSpan span = vectors.AsOutput(vectorsPointer);
            NativeMatrixSpan* rows = vectors.IsEmpty ? null : &span;
            written = usearch_dense_cursor_next(this.Handle, run, rows, out NativeError error);
            error.ThrowIfFailed();
        }

        return (int)written;
    }

    /// <summary>Releases the cursor. Freeing cannot fail, and a freed cursor must not be used.</summary>
    public void Dispose() => this._handle.Dispose();

    private NativeCursorHandle Handle => !this._handle.IsClosed && !this._handle.IsInvalid
        ? this._handle
        : throw new USearchException(Status.Uninitialized, "the cursor is disposed");
}

/// <summary>
/// A resumable nearest-neighbor stream: best-first traversal is naturally pausable, so the finder
/// owns a live frontier and visited set and yields matches lazily. Callers that post-filter
/// aggressively pull until satisfied instead of guessing "k" upfront, and each open finder pins one
/// of the executor's threads - dispose it promptly.
/// </summary>
public sealed unsafe class USearchFinder : IDisposable
{
    private readonly NativeFinderHandle _handle;

    internal USearchFinder(NativeFinderHandle handle) => this._handle = handle;

    /// <summary>
    /// Pulls the next matches from the stream. Ordering beyond the expansion horizon is
    /// best-effort nearest, not globally exact.
    /// </summary>
    /// <param name="keys">The keys to fill; their capacity bounds the pull.</param>
    /// <param name="distances">The distances to fill, positionally paired with the keys.</param>
    /// <returns>How many matches were written; zero when the traversal is drained.</returns>
    public int Next(Span<USearchKey> keys, Span<double> distances)
    {
        int wanted = Math.Min(keys.Length, distances.Length);
        if (wanted == 0)
        {
            throw new USearchException(Status.ShapeMismatch, "a pull must hold at least one match");
        }

        nuint written;
        fixed (USearchKey* keysPointer = keys)
        fixed (double* distancesPointer = distances)
        {
            NativeKeysSpan run = new() { Data = (IntPtr)keysPointer, Capacity = (nuint)wanted };
            written = usearch_dense_find_next(this.Handle, run, distancesPointer, out NativeError error);
            error.ThrowIfFailed();
        }

        return (int)written;
    }

    /// <summary>Releases the finder and the traversal context it pinned.</summary>
    public void Dispose() => this._handle.Dispose();

    private NativeFinderHandle Handle => !this._handle.IsClosed && !this._handle.IsInvalid
        ? this._handle
        : throw new USearchException(Status.Uninitialized, "the finder is disposed");
}

#endregion Cursors & Finders

#region Native Vocabulary

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
    /// <summary>The connections-per-node bound; zero derives.</summary>
    public nuint Connectivity;
    /// <summary>The upper bound on entries sharing one key; zero derives to unique keys.</summary>
    public nuint ValuesPerKey;
    /// <summary>Selects the active arm of <see cref="ResidencyOptions"/>.</summary>
    public ResidencyKind Residency;
    /// <summary>The arm <see cref="Residency"/> selects.</summary>
    public NativeResidencyOptions ResidencyOptions;
}

/// <summary>
/// The geometry and metric a checkpoint froze, laid out as the C struct spells them.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
internal struct NativeDenseMetadata
{
    /// <summary>The metric kind the checkpoint scores with.</summary>
    public DenseMetric Metric;
    /// <summary>The scalar type its rows are stored and quantized into.</summary>
    public ScalarKind Quantization;
    /// <summary>The scalars per stored vector.</summary>
    public nuint Dimensions;
    /// <summary>The connections per node the graph was built with.</summary>
    public nuint Connectivity;
}

/// <summary>Optional per-call settings of insertion and refinement; a zero field derives.</summary>
[StructLayout(LayoutKind.Sequential)]
internal struct NativeInsertionOptions
{
    /// <summary>The candidate-pool breadth the build walk explores; zero derives.</summary>
    public nuint ExpansionFactor;
}

/// <summary>Optional per-call settings of the read-only walks; a zero field derives.</summary>
[StructLayout(LayoutKind.Sequential)]
internal struct NativeSearchOptions
{
    /// <summary>The candidate-pool breadth the walk explores; zero derives.</summary>
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

#endregion Native Vocabulary

#region Entry Points

/// <summary>
/// The dense-modality C surface of <c>include/usearch/dense_index.h</c>.
/// </summary>
/// <remarks>
/// Every fanned-out verb takes the executor it runs on as its second argument, and a progress
/// handler with its state before the telemetry tail. This binding exposes no handler, so both of
/// those cross as null.
/// </remarks>
internal static unsafe partial class NativeMethods
{
    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial NativeIndexHandle usearch_dense_init(
        NativeDenseOptions* options, out NativeError error);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_dense_free(IntPtr index);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_dense_reserve(
        NativeIndexHandle index, nuint capacity, out NativeError error);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_dense_clear(NativeIndexHandle index, out NativeError error);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial nuint usearch_dense_size(NativeIndexHandle index, out NativeError error);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial nuint usearch_dense_capacity(NativeIndexHandle index, out NativeError error);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial nuint usearch_dense_dimensions(NativeIndexHandle index, out NativeError error);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial nuint usearch_dense_connectivity(NativeIndexHandle index, out NativeError error);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial nuint usearch_dense_memory_usage(NativeIndexHandle index, out NativeError error);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial IntPtr usearch_dense_capabilities(NativeIndexHandle index, out NativeError error);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial nuint usearch_dense_add(
        NativeIndexHandle index,
        NativeExecutorHandle executor,
        NativeKeysView keys,
        NativeMatrixView* vectors,
        NativeInsertionOptions* options,
        IntPtr progress,
        IntPtr progressState,
        NativeTraversalStats* stats,
        out NativeError error);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial nuint usearch_dense_refine(
        NativeIndexHandle index,
        NativeExecutorHandle executor,
        NativeInsertionOptions* options,
        IntPtr progress,
        IntPtr progressState,
        NativeTraversalStats* stats,
        out NativeError error);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_dense_search(
        NativeIndexHandle index,
        NativeExecutorHandle executor,
        NativeMatrixView* queries,
        NativeMatchesSpan* matches,
        NativeSearchOptions* options,
        IntPtr progress,
        IntPtr progressState,
        NativeTraversalStats* stats,
        out NativeError error);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_dense_search_if(
        NativeIndexHandle index,
        NativeExecutorHandle executor,
        NativeMatrixView* queries,
        NativeMatchesSpan* matches,
        NativeSearchOptions* options,
        delegate* unmanaged[Cdecl]<USearchKey, IntPtr, Admission> predicate,
        IntPtr predicateState,
        IntPtr progress,
        IntPtr progressState,
        NativeTraversalStats* stats,
        out NativeError error);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_dense_maxsim(
        NativeIndexHandle index,
        NativeExecutorHandle executor,
        NativeMatrixView* queries,
        nuint rowsPerQuery,
        NativeMatchesSpan* matches,
        NativeSearchOptions* options,
        IntPtr progress,
        IntPtr progressState,
        NativeTraversalStats* stats,
        out NativeError error);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial nuint usearch_dense_count(
        NativeIndexHandle index, USearchKey key, out NativeError error);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial nuint usearch_dense_get(
        NativeIndexHandle index, USearchKey key, NativeMatrixSpan* vectors, out NativeError error);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial NativeCursorHandle usearch_dense_cursor(
        NativeIndexHandle index, out NativeError error);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial nuint usearch_dense_cursor_next(
        NativeCursorHandle cursor, NativeKeysSpan keys, NativeMatrixSpan* vectors, out NativeError error);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_dense_cursor_free(IntPtr cursor);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial NativeFinderHandle usearch_dense_find(
        NativeIndexHandle index, NativeExecutorHandle executor, NativeMatrixView* query, out NativeError error);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial nuint usearch_dense_find_next(
        NativeFinderHandle finder, NativeKeysSpan keys, double* distances, out NativeError error);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_dense_find_free(IntPtr finder);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial nuint usearch_dense_storage_requirements(
        NativeIndexHandle index, out NativeError error);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_dense_save(
        NativeIndexHandle index,
        NativeExecutorHandle executor,
        NativeBytesView path,
        IntPtr progress,
        IntPtr progressState,
        out NativeError error);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_dense_load(
        NativeIndexHandle index,
        NativeExecutorHandle executor,
        NativeBytesView path,
        IntPtr progress,
        IntPtr progressState,
        out NativeError error);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_dense_metadata(
        NativeBytesView path, NativeDenseMetadata* metadata, out NativeError error);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_dense_save_buffer(
        NativeIndexHandle index,
        NativeExecutorHandle executor,
        NativeBytesSpan buffer,
        IntPtr progress,
        IntPtr progressState,
        out NativeError error);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_dense_load_buffer(
        NativeIndexHandle index,
        NativeExecutorHandle executor,
        NativeBytesView buffer,
        IntPtr progress,
        IntPtr progressState,
        out NativeError error);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void usearch_dense_metadata_buffer(
        NativeBytesView buffer, NativeDenseMetadata* metadata, out NativeError error);
}

#endregion Entry Points

#region Handles

/// <summary>A <c>usearch_dense_index_t</c>, freed when the wrapper is disposed or collected.</summary>
internal sealed class NativeIndexHandle : EngineHandle
{
    /// <inheritdoc/>
    protected override bool ReleaseHandle()
    {
        NativeMethods.usearch_dense_free(this.handle);
        return true;
    }
}

/// <summary>A <c>usearch_dense_cursor_t</c>, freed when the wrapper is disposed or collected.</summary>
internal sealed class NativeCursorHandle : EngineHandle
{
    /// <inheritdoc/>
    protected override bool ReleaseHandle()
    {
        NativeMethods.usearch_dense_cursor_free(this.handle);
        return true;
    }
}

/// <summary>A <c>usearch_dense_finder_t</c>, freed when the wrapper is disposed or collected.</summary>
internal sealed class NativeFinderHandle : EngineHandle
{
    /// <inheritdoc/>
    protected override bool ReleaseHandle()
    {
        NativeMethods.usearch_dense_find_free(this.handle);
        return true;
    }
}

#endregion Handles
