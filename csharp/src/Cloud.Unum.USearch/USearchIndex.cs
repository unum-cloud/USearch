using System;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;

using static Cloud.Unum.USearch.NativeMethods;

namespace Cloud.Unum.USearch;

#region Index

/// <summary>
/// A Vector Search Index over equidimensional dense representations - the managed twin of
/// <c>include/usearch/dense_index.h</c>.
/// </summary>
/// <remarks>
/// Every verb is a batch over row-major matrices, so a single vector is a one-row matrix. The
/// candidate-pool breadth rides each call rather than the handle, which lets a recall sweep run
/// against one immutable index with no mutation between points.
/// <para>
/// A handle may be shared across threads, so no method here takes a lock. On a RAM handle,
/// insertions and searches overlap freely: searches read only entries published behind the
/// visibility watermark and never write to the graph, while insertions synchronize through
/// per-node locks and publish in order. Serialization is the one exception, demanding a
/// mutation-free window, though concurrent readers keep searching throughout.
/// </para>
/// </remarks>
public sealed unsafe class USearchIndex : IDisposable
{
    private IntPtr _handle;

    /// <summary>Initializes an index for the residency its options select.</summary>
    /// <param name="options">The construction and loading options.</param>
    public USearchIndex(DenseOptions options)
    {
        NativeDenseOptions lowered = options.Lower();
        this._handle = usearch_dense_init(&lowered, out NativeError error);
        error.ThrowIfFailed();
        if (this._handle == IntPtr.Zero)
        {
            throw new USearchException(Status.Internal, "the library returned no handle and no error");
        }
    }

    /// <summary>
    /// Reads a checkpoint's geometry and metric metadata without loading it - from the manifest's
    /// summary when the path names one, so no shard is opened.
    /// </summary>
    /// <param name="path">The checkpoint path.</param>
    /// <returns>The options the checkpoint's headers describe.</returns>
    public static DenseOptions Metadata(string path)
    {
        byte[] encoded = EncodePath(path);
        NativeDenseOptions lowered = default;
        fixed (byte* data = encoded)
        {
            usearch_dense_metadata(Borrow(data, encoded.Length), &lowered, out NativeError error);
            error.ThrowIfFailed();
        }

        return DenseOptions.Lift(lowered);
    }

    /// <summary>Reads geometry and metric metadata from a serialized monolith buffer.</summary>
    /// <param name="buffer">The monolith's bytes.</param>
    /// <returns>The options the monolith's header describes.</returns>
    public static DenseOptions MetadataFromBuffer(ReadOnlySpan<byte> buffer)
    {
        if (buffer.IsEmpty)
        {
            throw new USearchException(Status.MalformedInput, "the buffer is empty");
        }

        NativeDenseOptions lowered = default;
        fixed (byte* source = buffer)
        {
            usearch_dense_metadata_buffer(source, (nuint)buffer.Length, &lowered, out NativeError error);
            error.ThrowIfFailed();
        }

        return DenseOptions.Lift(lowered);
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
    /// Reports the expected serialized size in bytes: every shard file, plus the manifest when the
    /// index spans more than one.
    /// </summary>
    /// <returns>The byte count.</returns>
    public ulong SerializedLength()
    {
        nuint reading = usearch_dense_serialized_length(this.Handle, out NativeError error);
        error.ThrowIfFailed();
        return reading;
    }

    /// <summary>Reports the capability codename the index dispatched to on this machine.</summary>
    /// <returns>The codename.</returns>
    public string Capabilities()
    {
        IntPtr reading = usearch_dense_capabilities(this.Handle, out NativeError error);
        error.ThrowIfFailed();
        return Marshal.PtrToStringAnsi(reading)
            ?? throw new USearchException(Status.Internal, "the library reported no capabilities and no error");
    }

    /// <summary>Inserts one vector under its key - the one-row case of the batch below.</summary>
    /// <typeparam name="TScalar">The element type of the vector.</typeparam>
    /// <param name="key">The identity to store the vector under.</param>
    /// <param name="vector">The vector's scalars, one row wide.</param>
    /// <param name="expansionFactor">The candidate-pool breadth; zero derives.</param>
    /// <param name="stats">The traversal telemetry to fill, when the caller wants any.</param>
    /// <returns>One on success; a batch is all-or-nothing.</returns>
    public ulong Add<TScalar>(
        USearchKey key, ReadOnlySpan<TScalar> vector, ulong expansionFactor = 0, TraversalStats? stats = null)
        where TScalar : unmanaged
    {
        return this.Add(new ReadOnlySpan<USearchKey>(in key), vector, expansionFactor, stats);
    }

    /// <summary>
    /// Inserts a row-major matrix of vectors under their keys - the native batch operation, which
    /// overlaps safely with concurrent searches on the same handle.
    /// </summary>
    /// <remarks>
    /// The input scalar type is independent of the stored quantization; the library converts on
    /// ingest. Rows are packed, so the byte pitch is the row's own width.
    /// </remarks>
    /// <typeparam name="TScalar">The element type of the vector rows.</typeparam>
    /// <param name="keys">One identity per row.</param>
    /// <param name="vectors">The rows, packed end to end.</param>
    /// <param name="expansionFactor">The candidate-pool breadth; zero derives.</param>
    /// <param name="stats">The traversal telemetry to fill, when the caller wants any.</param>
    /// <returns>The row count on success and zero on failure: every fallible step runs before the
    /// first entry identity is claimed, so a rejected batch publishes nothing.</returns>
    public ulong Add<TScalar>(
        ReadOnlySpan<USearchKey> keys,
        ReadOnlySpan<TScalar> vectors,
        ulong expansionFactor = 0,
        TraversalStats? stats = null)
        where TScalar : unmanaged
    {
        ulong dimensions = this.FrozenDimensions();
        int expected = Extents.CheckedRowMajor((ulong)keys.Length, dimensions, (ulong)sizeof(TScalar), "insertion");
        if (vectors.Length != expected)
        {
            throw new USearchException(
                Status.ShapeMismatch, "the vector rows are not as wide as the index geometry, or the counts disagree");
        }

        if (keys.IsEmpty)
        {
            return 0;
        }

        NativeExpansionConfig config = new() { ExpansionFactor = (nuint)expansionFactor };
        NativeTraversalStats reported = default;
        nuint added;
        fixed (USearchKey* keysPointer = keys)
        fixed (TScalar* vectorsPointer = vectors)
        {
            added = usearch_dense_add(
                this.Handle, keysPointer, vectorsPointer, (nuint)keys.Length,
                Scalars.RowBytes<TScalar>(dimensions), Scalars.KindOf<TScalar>(),
                &config, &reported, out NativeError error);
            error.ThrowIfFailed();
        }

        stats?.Adopt(reported);
        return added;
    }

    /// <summary>
    /// Resolves a row-major matrix of queries into the nearest entries - the native batch
    /// operation, parallelized internally, which never writes to the graph.
    /// </summary>
    /// <typeparam name="TScalar">The element type of the query rows.</typeparam>
    /// <param name="queries">The query rows, packed end to end.</param>
    /// <param name="wanted">The upper bound on matches per query - the "k" in kANN.</param>
    /// <param name="expansionFactor">The candidate-pool breadth; zero derives.</param>
    /// <param name="stats">The traversal telemetry to fill, when the caller wants any.</param>
    /// <returns>The keys, distances, and per-query counts the batch produced.</returns>
    public SearchResults Search<TScalar>(
        ReadOnlySpan<TScalar> queries, int wanted, ulong expansionFactor = 0, TraversalStats? stats = null)
        where TScalar : unmanaged =>
        this.SearchDispatch(queries, wanted, expansionFactor, stats, null);

    /// <summary>
    /// The filtered sibling of <see cref="Search"/>, admitting only the keys the predicate
    /// approves. Non-matching entries still route the traversal, so recall does not collapse at
    /// low selectivity.
    /// </summary>
    /// <remarks>
    /// The predicate runs on the library's worker threads, so it must be safe for concurrent use.
    /// An exception it throws cannot cross the boundary: the first one is captured, every later
    /// candidate is rejected, and the failure is raised here once the call returns.
    /// </remarks>
    /// <typeparam name="TScalar">The element type of the query rows.</typeparam>
    /// <param name="queries">The query rows, packed end to end.</param>
    /// <param name="wanted">The upper bound on matches per query.</param>
    /// <param name="admits">The per-key admission predicate.</param>
    /// <param name="expansionFactor">The candidate-pool breadth; zero derives.</param>
    /// <param name="stats">The traversal telemetry to fill, when the caller wants any.</param>
    /// <returns>The keys, distances, and per-query counts the batch produced.</returns>
    public SearchResults SearchIf<TScalar>(
        ReadOnlySpan<TScalar> queries,
        int wanted,
        Func<USearchKey, bool> admits,
        ulong expansionFactor = 0,
        TraversalStats? stats = null)
        where TScalar : unmanaged
    {
        if (admits is null)
        {
            throw new USearchException(Status.MalformedInput, "the admission predicate is null");
        }

        return this.SearchDispatch(queries, wanted, expansionFactor, stats, admits);
    }

    private SearchResults SearchDispatch<TScalar>(
        ReadOnlySpan<TScalar> queries,
        int wanted,
        ulong expansionFactor,
        TraversalStats? stats,
        Func<USearchKey, bool>? admits)
        where TScalar : unmanaged
    {
        if (wanted <= 0)
        {
            throw new USearchException(Status.ShapeMismatch, "the outputs reserve no room for a single match");
        }

        ulong dimensions = this.FrozenDimensions();
        if ((ulong)queries.Length % dimensions != 0)
        {
            throw new USearchException(
                Status.ShapeMismatch, "the query rows are not as wide as the index geometry");
        }

        ulong queriesCount = (ulong)queries.Length / dimensions;
        int matches = Extents.CheckedRowMajor(queriesCount, (ulong)wanted, USearchKey.ByteCount, "search output");
        USearchKey[] keys = new USearchKey[matches];
        double[] distances = new double[matches];
        nuint[] counts = new nuint[queriesCount];
        SearchResults results = new(keys, distances, counts, wanted);
        if (queriesCount == 0)
        {
            return results;
        }

        NativeExpansionConfig config = new() { ExpansionFactor = (nuint)expansionFactor };
        NativeTraversalStats reported = default;
        nuint keysStride = (nuint)wanted * USearchKey.ByteCount;
        nuint distancesStride = (nuint)wanted * sizeof(double);
        AdmissionTrampoline? trampoline = admits is null ? null : new AdmissionTrampoline(admits);
        NativePredicate? predicate = trampoline is null ? null : trampoline.Admit;
        fixed (TScalar* queriesPointer = queries)
        fixed (USearchKey* keysPointer = keys)
        fixed (double* distancesPointer = distances)
        fixed (nuint* countsPointer = counts)
        {
            NativeError error;
            if (predicate is null)
            {
                usearch_dense_search(
                    this.Handle, queriesPointer, (nuint)queriesCount, Scalars.RowBytes<TScalar>(dimensions),
                    Scalars.KindOf<TScalar>(), (nuint)wanted, &config, &reported,
                    keysPointer, keysStride, distancesPointer, distancesStride, countsPointer, out error);
            }
            else
            {
                usearch_dense_search_if(
                    this.Handle, queriesPointer, (nuint)queriesCount, Scalars.RowBytes<TScalar>(dimensions),
                    Scalars.KindOf<TScalar>(), (nuint)wanted, &config, &reported, predicate, IntPtr.Zero,
                    keysPointer, keysStride, distancesPointer, distancesStride, countsPointer, out error);
                GC.KeepAlive(predicate);
            }

            trampoline?.Rethrow();
            error.ThrowIfFailed();
        }

        stats?.Adopt(reported);
        return results;
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
    /// Retrieves the vectors stored under a key into a row-major matrix, one row per entry sharing
    /// the key, in insertion order.
    /// </summary>
    /// <typeparam name="TScalar">The element type to export as.</typeparam>
    /// <param name="key">The identity to look for.</param>
    /// <param name="vectors">The rows to fill, packed end to end.</param>
    /// <returns>The true number of entries, never the number written: a count above the matrix's
    /// row capacity means the copy was truncated, so size for that many and call again.</returns>
    public ulong Get<TScalar>(USearchKey key, Span<TScalar> vectors) where TScalar : unmanaged
    {
        ulong dimensions = this.FrozenDimensions();
        if ((ulong)vectors.Length % dimensions != 0)
        {
            throw new USearchException(
                Status.ShapeMismatch, "the output rows are not as wide as the index geometry");
        }

        nuint entries;
        fixed (TScalar* vectorsPointer = vectors)
        {
            entries = usearch_dense_get(
                this.Handle, key, vectorsPointer, (nuint)((ulong)vectors.Length / dimensions),
                Scalars.RowBytes<TScalar>(dimensions), Scalars.KindOf<TScalar>(), out NativeError error);
            error.ThrowIfFailed();
        }

        return entries;
    }

    /// <summary>
    /// Absorbs a donor index - cheaper than re-inserting its vectors one by one, wiring
    /// cross-links via landmark stitching. Readers of this index may keep searching throughout,
    /// since unpublished rows hide behind the watermark.
    /// </summary>
    /// <remarks>
    /// The donor is consumed whether the call succeeds or fails: the C side owns it afterwards, so
    /// the donor handle is emptied here and its own <see cref="Dispose"/> becomes a no-op. The
    /// donor demands exclusive ownership for the duration - any concurrent verb on it, from any
    /// thread, is a use-after-free.
    /// </remarks>
    /// <param name="donor">The index to absorb.</param>
    public void Merge(USearchIndex donor)
    {
        if (donor is null || donor._handle == IntPtr.Zero)
        {
            throw new USearchException(Status.Uninitialized, "the donor index is disposed or was never initialized");
        }

        IntPtr consumed = Interlocked.Exchange(ref donor._handle, IntPtr.Zero);
        usearch_dense_merge(this.Handle, consumed, out NativeError error);
        error.ThrowIfFailed();
    }

    /// <summary>Opens a snapshot cursor over the current contents.</summary>
    /// <returns>The cursor, which the caller disposes.</returns>
    public USearchCursor Snapshot()
    {
        IntPtr cursor = usearch_dense_cursor(this.Handle, out NativeError error);
        error.ThrowIfFailed();
        if (cursor == IntPtr.Zero)
        {
            throw new USearchException(Status.Internal, "the library returned no cursor and no error");
        }

        return new USearchCursor(cursor, this.Dimensions());
    }

    /// <summary>Opens a resumable nearest-neighbor stream for one query.</summary>
    /// <typeparam name="TScalar">The element type of the query row.</typeparam>
    /// <param name="query">The query's scalars, one row wide.</param>
    /// <returns>The finder, which the caller disposes.</returns>
    public USearchFinder Find<TScalar>(ReadOnlySpan<TScalar> query) where TScalar : unmanaged
    {
        ulong dimensions = this.FrozenDimensions();
        if ((ulong)query.Length != dimensions)
        {
            throw new USearchException(Status.ShapeMismatch, "the query row is not as wide as the index geometry");
        }

        IntPtr finder;
        fixed (TScalar* queryPointer = query)
        {
            finder = usearch_dense_find(this.Handle, queryPointer, Scalars.KindOf<TScalar>(), out NativeError error);
            error.ThrowIfFailed();
        }

        if (finder == IntPtr.Zero)
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
    /// <param name="path">Where the checkpoint is written.</param>
    public void Save(string path)
    {
        byte[] encoded = EncodePath(path);
        fixed (byte* data = encoded)
        {
            usearch_dense_save(this.Handle, Borrow(data, encoded.Length), out NativeError error);
            error.ThrowIfFailed();
        }
    }

    /// <summary>
    /// Reads a checkpoint into this handle, shard by shard, with the handle's residency deciding
    /// the behavior: materialize into RAM, upload to the device, or open in place on disk. A
    /// handle whose geometry was left zeroed freezes it here, from the checkpoint's own headers.
    /// </summary>
    /// <param name="path">Where the checkpoint is read from.</param>
    public void Load(string path)
    {
        byte[] encoded = EncodePath(path);
        fixed (byte* data = encoded)
        {
            usearch_dense_load(this.Handle, Borrow(data, encoded.Length), out NativeError error);
            error.ThrowIfFailed();
        }
    }

    /// <summary>
    /// Writes the index into a caller-owned buffer as one monolith. Size the buffer with
    /// <see cref="SerializedLength"/>.
    /// </summary>
    /// <param name="buffer">The bytes to write into.</param>
    public void SaveToBuffer(Span<byte> buffer)
    {
        if (buffer.IsEmpty)
        {
            throw new USearchException(Status.ShapeMismatch, "the buffer is empty");
        }

        fixed (byte* target = buffer)
        {
            usearch_dense_save_buffer(this.Handle, target, (nuint)buffer.Length, out NativeError error);
            error.ThrowIfFailed();
        }
    }

    /// <summary>
    /// Reads a monolith from a buffer into this handle, with the residency deciding the behavior
    /// like <see cref="Load"/>. A residency that opens in place demands the buffer outlive the
    /// handle.
    /// </summary>
    /// <param name="buffer">The monolith's bytes.</param>
    public void LoadFromBuffer(ReadOnlySpan<byte> buffer)
    {
        if (buffer.IsEmpty)
        {
            throw new USearchException(Status.MalformedInput, "the buffer is empty");
        }

        fixed (byte* source = buffer)
        {
            usearch_dense_load_buffer(this.Handle, source, (nuint)buffer.Length, out NativeError error);
            error.ThrowIfFailed();
        }
    }

    /// <summary>Releases the index. Freeing cannot fail, and a freed handle must not be used.</summary>
    public void Dispose()
    {
        IntPtr released = Interlocked.Exchange(ref this._handle, IntPtr.Zero);
        if (released != IntPtr.Zero)
        {
            usearch_dense_free(released);
        }

        GC.SuppressFinalize(this);
    }

    /// <summary>Releases the index when the caller forgot to.</summary>
    ~USearchIndex() => this.Dispose();

    private IntPtr Handle => this._handle != IntPtr.Zero
        ? this._handle
        : throw new USearchException(Status.Uninitialized, "the index is disposed or was consumed by a merge");

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

    private static byte[] EncodePath(string path)
    {
        if (string.IsNullOrEmpty(path))
        {
            throw new USearchException(Status.MalformedInput, "the path is empty");
        }

        return Encoding.UTF8.GetBytes(path);
    }

    private static NativeString Borrow(byte* data, int length) =>
        new() { Data = (IntPtr)data, Length = (nuint)length };

    /// <summary>
    /// Holds a managed predicate the engine calls from its own worker threads, and the first
    /// exception it threw. A managed exception must not unwind through native frames, so it is
    /// captured here and raised once the call has returned.
    /// </summary>
    private sealed class AdmissionTrampoline
    {
        private readonly Func<USearchKey, bool> _admits;
        private Exception? _escaped;

        internal AdmissionTrampoline(Func<USearchKey, bool> admits) => this._admits = admits;

        internal int Admit(USearchKey key, IntPtr state)
        {
            if (Volatile.Read(ref this._escaped) is not null)
            {
                return 0;
            }

            try
            {
                return this._admits(key) ? 1 : 0;
            }
            // Every exception is caught on purpose: this frame is called from a native worker
            // thread, and unwinding through native frames terminates the process.
#pragma warning disable CA1031
            catch (Exception failure)
#pragma warning restore CA1031
            {
                Interlocked.CompareExchange(ref this._escaped, failure, null);
                return 0;
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

#region Cursor & Finder

/// <summary>
/// A consistent snapshot cursor over the index contents. It captures the publication watermark at
/// creation, so concurrent inserts never appear mid-iteration, and the walk is a sequential row
/// scan - export, migration, and visualization feed off it.
/// </summary>
public sealed unsafe class USearchCursor : IDisposable
{
    private IntPtr _handle;
    private readonly ulong _dimensions;

    internal USearchCursor(IntPtr handle, ulong dimensions)
    {
        this._handle = handle;
        this._dimensions = dimensions;
    }

    /// <summary>Copies the next batch of keys out of the snapshot, skipping the vectors.</summary>
    /// <param name="keys">The keys to fill.</param>
    /// <returns>How many rows were written; zero when the snapshot is exhausted.</returns>
    public int Next(Span<USearchKey> keys)
    {
        nuint written;
        fixed (USearchKey* keysPointer = keys)
        {
            written = NativeMethods.usearch_dense_cursor_next(
                this.Handle, keysPointer, null, (nuint)keys.Length, 0, ScalarKind.Unknown, out NativeError error);
            error.ThrowIfFailed();
        }

        return (int)written;
    }

    /// <summary>Copies the next batch of keys and vectors out of the snapshot.</summary>
    /// <typeparam name="TScalar">The element type to export vectors as.</typeparam>
    /// <param name="keys">The keys to fill.</param>
    /// <param name="vectors">The rows to fill, packed end to end, one per key.</param>
    /// <returns>How many rows were written; zero when the snapshot is exhausted.</returns>
    public int Next<TScalar>(Span<USearchKey> keys, Span<TScalar> vectors) where TScalar : unmanaged
    {
        if (this._dimensions == 0)
        {
            throw new USearchException(Status.Uninitialized, "the geometry is not frozen, so a row has no width");
        }

        int rows = Extents.CheckedRowMajor(
            (ulong)keys.Length, this._dimensions, (ulong)sizeof(TScalar), "cursor output");
        if (vectors.Length < rows)
        {
            throw new USearchException(Status.ShapeMismatch, "the vector rows do not cover every requested key");
        }

        nuint written;
        fixed (USearchKey* keysPointer = keys)
        fixed (TScalar* vectorsPointer = vectors)
        {
            written = NativeMethods.usearch_dense_cursor_next(
                this.Handle, keysPointer, vectorsPointer, (nuint)keys.Length,
                Scalars.RowBytes<TScalar>(this._dimensions), Scalars.KindOf<TScalar>(),
                out NativeError error);
            error.ThrowIfFailed();
        }

        return (int)written;
    }

    /// <summary>Releases the cursor. Freeing cannot fail, and a freed cursor must not be used.</summary>
    public void Dispose()
    {
        IntPtr released = Interlocked.Exchange(ref this._handle, IntPtr.Zero);
        if (released != IntPtr.Zero)
        {
            NativeMethods.usearch_dense_cursor_free(released);
        }

        GC.SuppressFinalize(this);
    }

    /// <summary>Releases the cursor when the caller forgot to.</summary>
    ~USearchCursor() => this.Dispose();

    private IntPtr Handle => this._handle != IntPtr.Zero
        ? this._handle
        : throw new USearchException(Status.Uninitialized, "the cursor is disposed");
}

/// <summary>
/// A resumable nearest-neighbor stream: best-first traversal is naturally pausable, so the finder
/// owns a live frontier and visited set and yields matches lazily. Callers that post-filter
/// aggressively pull until satisfied instead of guessing "k" upfront, and each open finder pins
/// one traversal context's memory - dispose it promptly.
/// </summary>
public sealed unsafe class USearchFinder : IDisposable
{
    private IntPtr _handle;

    internal USearchFinder(IntPtr handle) => this._handle = handle;

    /// <summary>
    /// Pulls the next matches from the stream. Ordering beyond the expansion horizon is
    /// best-effort nearest, not globally exact.
    /// </summary>
    /// <param name="keys">The keys to fill.</param>
    /// <param name="distances">The distances to fill, positionally paired with the keys.</param>
    /// <returns>How many matches were written; zero when the traversal is drained.</returns>
    public int Next(Span<USearchKey> keys, Span<double> distances)
    {
        int wanted = Math.Min(keys.Length, distances.Length);
        nuint written;
        fixed (USearchKey* keysPointer = keys)
        fixed (double* distancesPointer = distances)
        {
            written = NativeMethods.usearch_dense_find_next(
                this.Handle, keysPointer, distancesPointer, (nuint)wanted, out NativeError error);
            error.ThrowIfFailed();
        }

        return (int)written;
    }

    /// <summary>Releases the finder and the traversal context it pinned.</summary>
    public void Dispose()
    {
        IntPtr released = Interlocked.Exchange(ref this._handle, IntPtr.Zero);
        if (released != IntPtr.Zero)
        {
            NativeMethods.usearch_dense_find_free(released);
        }

        GC.SuppressFinalize(this);
    }

    /// <summary>Releases the finder when the caller forgot to.</summary>
    ~USearchFinder() => this.Dispose();

    private IntPtr Handle => this._handle != IntPtr.Zero
        ? this._handle
        : throw new USearchException(Status.Uninitialized, "the finder is disposed");
}

#endregion Cursor & Finder

#region Introspection

/// <summary>
/// Library-wide probes, tied to no index instance.
/// </summary>
public static class USearchCapabilities
{
    /// <summary>Reports the version of the linked library.</summary>
    /// <returns>The version string.</returns>
    public static string Version() => Marshal.PtrToStringAnsi(NativeMethods.usearch_version()) ?? "unknown";

    /// <summary>
    /// Lists the hardware capabilities the linked binary was compiled with, comma separated.
    /// </summary>
    /// <returns>The capability names.</returns>
    public static string CapabilitiesCompiled() =>
        Marshal.PtrToStringAnsi(NativeMethods.usearch_capabilities_compiled()) ?? "serial";

    /// <summary>
    /// Lists the hardware capabilities the current machine offers at runtime, comma separated.
    /// </summary>
    /// <returns>The capability names.</returns>
    public static string CapabilitiesAvailable() =>
        Marshal.PtrToStringAnsi(NativeMethods.usearch_capabilities_available()) ?? "serial";
}

#endregion Introspection
