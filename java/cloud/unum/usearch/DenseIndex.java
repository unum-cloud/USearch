package cloud.unum.usearch;

import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;

/**
 * A Vector Search Index over equidimensional dense representations - the Java twin of
 * {@code include/usearch/dense_index.h}.
 *
 * <p>
 * Every verb is a batch over strided matrices, so a memory-mapped dataset reaches the index with no
 * reshape and no copy, and a single vector is a one-row matrix. The candidate-pool breadth rides
 * each call rather than the handle, which lets a recall sweep run against one immutable index with
 * no mutation between points.
 *
 * <p>
 * A handle may be shared across threads, so no method here takes a lock. On a RAM handle,
 * insertions and searches overlap freely: searches read only entries published behind the
 * visibility watermark and never write to the graph, while insertions synchronize through per-node
 * locks and publish in order. Serialization is the one exception, demanding a mutation-free window,
 * though concurrent readers keep searching throughout.
 *
 * <p>
 * The filtered search of the C interface stays unbound. Its predicate is called from the engine's
 * own worker threads, which are not attached to this JVM, so every candidate would pay an attach,
 * an upcall, and an exception check - and a predicate that threw would unwind through native frames
 * that cannot handle it. Filter the results of {@link #search} instead, or narrow the index.
 *
 * <pre>{@code
 * try (DenseIndex index = new DenseOptions().dimensions(2).quantization(ScalarKind.F32).open()) {
 *     index.reserve(2);
 *     index.add(new Key[] {Key.of(1), Key.of(2)}, DenseMatrix.of(new float[] {0, 1, 1, 0}, 2));
 *     DenseMatches matches = index.search(DenseMatrix.of(new float[] {0, 1}, 2), 1);
 *     System.out.println(matches.key(0, 0));
 * }
 * }</pre>
 */
public final class DenseIndex implements AutoCloseable {

    static {
        NativeLibrary.require();
    }

    /** The scalar quantities a handle can be asked for, each one probe of the C interface. */
    private enum Measure {
        SIZE(0),
        CAPACITY(1),
        DIMENSIONS(2),
        CONNECTIVITY(3),
        MEMORY_USAGE(4),
        SERIALIZED_LENGTH(5);

        private final int code;

        Measure(int code) {
            this.code = code;
        }

        int code() {
            return code;
        }
    }

    private volatile long handle;

    /** Initializes an index for the residency the options select. */
    public DenseIndex(DenseOptions options) {
        if (options == null) {
            throw new IndexException(Status.MALFORMED_INPUT, "the options are null");
        }
        handle = nativeInitialize(options.toWords());
        if (handle == 0) {
            throw new IndexException(Status.INTERNAL, "the library returned no handle and no error");
        }
    }

    /**
     * Reads a checkpoint's geometry and metric metadata without loading it - from the manifest's
     * summary when the path names one, so no shard is opened.
     */
    public static DenseOptions metadata(String path) {
        long[] words = new long[OptionsWords.COUNT];
        nativeMetadata(encode(path), words);
        return DenseOptions.fromWords(words);
    }

    /** Reads geometry and metric metadata from a serialized monolith slab. */
    public static DenseOptions metadata(byte[] buffer) {
        return metadataOf(buffer, 0, lengthOf(buffer));
    }

    /** Reads geometry and metric metadata from a serialized monolith buffer. */
    public static DenseOptions metadata(ByteBuffer buffer) {
        return metadataOf(buffer, offsetOf(buffer), lengthOf(buffer));
    }

    private static DenseOptions metadataOf(Object buffer, long byteOffset, long length) {
        long[] words = new long[OptionsWords.COUNT];
        nativeMetadataBuffer(buffer, byteOffset, length, words);
        return DenseOptions.fromWords(words);
    }

    /** Releases the index. It is idempotent, and a freed handle must not be used again. */
    @Override
    public void close() {
        long released = handle;
        handle = 0;
        if (released != 0) {
            nativeFree(released);
        }
    }

    /**
     * Claims capacity for a number of incoming vectors ahead of insertion. The capacity is the
     * desired total, including the current size.
     */
    public void reserve(long capacity) {
        nativeReserve(live(), capacity);
    }

    /** Erases all entries, keeping the reserved capacity and the frozen geometry. */
    public void clear() {
        nativeClear(live());
    }

    /** The number of stored entries. */
    public long size() {
        return nativeProbe(live(), Measure.SIZE.code());
    }

    /** The reserved capacity in entries. */
    public long capacity() {
        return nativeProbe(live(), Measure.CAPACITY.code());
    }

    /**
     * The frozen width of the stored vectors, or zero while the geometry is still unfrozen - an
     * index built with zero dimensions freezes it at its first {@link #load(String)}.
     */
    public long dimensions() {
        return nativeProbe(live(), Measure.DIMENSIONS.code());
    }

    /** The frozen connections-per-node bound of the graph. */
    public long connectivity() {
        return nativeProbe(live(), Measure.CONNECTIVITY.code());
    }

    /** The memory the index occupies, in bytes. */
    public long memoryUsage() {
        return nativeProbe(live(), Measure.MEMORY_USAGE.code());
    }

    /**
     * The expected serialized size in bytes: every shard file, plus the manifest when the index
     * spans more than one.
     */
    public long serializedLength() {
        return nativeProbe(live(), Measure.SERIALIZED_LENGTH.code());
    }

    /** The capability codename the index dispatched to on this machine. */
    public String capabilities() {
        return nativeCapabilities(live());
    }

    /** Inserts one vector under one key. */
    public void add(Key key, DenseMatrix vector) {
        if (vector.rows() != 1) {
            throw new IndexException(Status.SHAPE_MISMATCH, "one key admits exactly one row");
        }
        add(new Key[] {key}, vector, 0, null);
    }

    /** Inserts a matrix of vectors under their keys, with every optional setting derived. */
    public long add(Key[] keys, DenseMatrix vectors) {
        return add(keys, vectors, 0, null);
    }

    /**
     * Inserts a matrix of vectors under their keys - the native batch operation, where a single
     * vector is a one-row matrix. Overlaps safely with concurrent searches on the same handle.
     *
     * <p>
     * The input scalar type is independent of the stored quantization; the library converts on
     * ingest. An expansion factor of zero derives the candidate-pool breadth, and a null stats skips
     * telemetry.
     *
     * @return the row count on success - a batch is all-or-nothing, since every fallible step runs
     *     before the first entry identity is claimed. An accelerator fault severe enough to poison
     *     the device context is the sole exception, reporting DEVICE_FAILURE and latching the index
     *     unusable rather than partially applied.
     */
    public long add(Key[] keys, DenseMatrix vectors, long expansionFactor, DenseStats stats) {
        long index = live();
        vectors.validate(dimensions());
        if (keys == null) {
            throw new IndexException(Status.MALFORMED_INPUT, "the keys are null");
        }
        if (keys.length != vectors.rows()) {
            throw new IndexException(Status.SHAPE_MISMATCH, "the key count does not match the vector row count");
        }
        if (keys.length == 0) {
            return 0;
        }
        long[] telemetry = stats == null ? null : new long[OptionsWords.STATS_COUNT];
        long added = nativeAdd(index, flatten(keys), vectors.storage(), vectors.byteOffset(), vectors.rows(),
                vectors.stride(), vectors.kind().code(), expansionFactor, telemetry);
        if (stats != null) {
            stats.adopt(telemetry);
        }
        return added;
    }

    /** Resolves a matrix of queries into freshly sized outputs. */
    public DenseMatches search(DenseMatrix queries, long wanted) {
        DenseMatches matches = new DenseMatches(queries.rows(), wanted);
        searchInto(queries, matches, 0, null);
        return matches;
    }

    /**
     * Resolves a matrix of queries into caller-owned outputs - the allocation-free batch path,
     * parallelized internally. Never writes to the graph, so any number of these may overlap with
     * concurrent insertions.
     */
    public void searchInto(DenseMatrix queries, DenseMatches matches, long expansionFactor, DenseStats stats) {
        long index = live();
        queries.validate(dimensions());
        if (queries.rows() > matches.queries()) {
            throw new IndexException(Status.SHAPE_MISMATCH, "the outputs hold fewer rows than there are queries");
        }
        if (queries.rows() == 0) {
            return;
        }
        long[] telemetry = stats == null ? null : new long[OptionsWords.STATS_COUNT];
        nativeSearch(index, queries.storage(), queries.byteOffset(), queries.rows(), queries.stride(),
                queries.kind().code(), matches.wanted(), expansionFactor, telemetry,
                matches.keySlab(), matches.distanceSlab(), matches.countSlab());
        if (stats != null) {
            stats.adopt(telemetry);
        }
    }

    /** How many entries are stored under a key; zero means the index holds none. */
    public long count(Key key) {
        return nativeCount(live(), key.toBytes());
    }

    /**
     * Whether the index holds at least one entry under a key. The C interface has no membership
     * verb of its own, since a count already answers it.
     */
    public boolean contains(Key key) {
        return count(key) != 0;
    }

    /**
     * Retrieves the vectors stored under a key into the matrix, one row per entry sharing the key,
     * in insertion order.
     *
     * @return the true number of entries, never the number written: a count above the matrix's row
     *     capacity means the copy was truncated, so size for that many and call again.
     *     {@link #count(Key)} is the same number without the copy.
     */
    public long get(Key key, DenseMatrix vectors) {
        long index = live();
        vectors.validate(dimensions());
        return nativeGet(index, key.toBytes(), vectors.storage(), vectors.byteOffset(), vectors.rows(),
                vectors.stride(), vectors.kind().code());
    }

    /**
     * Absorbs a donor index - cheaper than re-inserting its vectors one by one, wiring cross-links
     * via landmark stitching. Readers of this index may keep searching throughout, since unpublished
     * rows hide behind the watermark.
     *
     * <p>
     * The donor is consumed whether the call succeeds or fails: the native side owns it afterwards,
     * so the donor's handle is emptied here and its own {@link #close()} becomes a no-op. The donor
     * demands exclusive ownership for the whole call - any concurrent verb on it, from any thread,
     * is a use-after-free this binding cannot detect.
     */
    public void merge(DenseIndex donor) {
        long index = live();
        if (donor == null) {
            throw new IndexException(Status.UNINITIALIZED, "the donor index is not initialized");
        }
        if (donor == this) {
            throw new IndexException(Status.MALFORMED_INPUT, "an index cannot absorb itself");
        }
        long consumed = donor.live();
        donor.handle = 0;
        nativeMerge(index, consumed);
    }

    /**
     * Writes the index under a path, in the layout its extension picks - one {@code .safegraphs}
     * monolith, or a {@code .safegraphs.json} manifest beside its shards. Demands a mutation-free
     * window; concurrent readers keep searching throughout.
     */
    public void save(String path) {
        nativeSave(live(), encode(path));
    }

    /**
     * Reads a checkpoint into this handle, shard by shard, with the handle's residency deciding the
     * behavior: materialize into RAM, upload to the device, or open in place on disk. A handle whose
     * geometry was left zeroed freezes it here, from the checkpoint's own headers.
     */
    public void load(String path) {
        nativeLoad(live(), encode(path));
    }

    /** Writes the index into a caller-owned slab as one monolith; size it with
     * {@link #serializedLength()}. */
    public void saveBuffer(byte[] buffer) {
        nativeSaveBuffer(live(), buffer, 0, lengthOf(buffer));
    }

    /** Writes the index into a caller-owned direct buffer as one monolith. */
    public void saveBuffer(ByteBuffer buffer) {
        nativeSaveBuffer(live(), buffer, offsetOf(buffer), lengthOf(buffer));
    }

    /** Reads a monolith from a slab into this handle, with the residency deciding the behavior. */
    public void loadBuffer(byte[] buffer) {
        nativeLoadBuffer(live(), buffer, 0, lengthOf(buffer));
    }

    /**
     * Reads a monolith from a direct buffer into this handle, with the residency deciding the
     * behavior like {@link #load(String)}. A residency that opens in place demands the buffer
     * outlive the handle.
     */
    public void loadBuffer(ByteBuffer buffer) {
        nativeLoadBuffer(live(), buffer, offsetOf(buffer), lengthOf(buffer));
    }

    /** Reports the handle, refusing a closed index. */
    private long live() {
        long current = handle;
        if (current == 0) {
            throw new IndexException(Status.UNINITIALIZED, "the index is freed or was never initialized");
        }
        return current;
    }

    /** Packs the keys into the contiguous byte matrix the C interface reads. */
    private static byte[] flatten(Key[] keys) {
        byte[] slab = new byte[Extents.toArrayLength(
                Extents.product(keys.length, Key.BYTES, "the key matrix"), "the key matrix")];
        for (int row = 0; row < keys.length; ++row) {
            keys[row].writeInto(slab, row * Key.BYTES);
        }
        return slab;
    }

    /** Borrows a path as the pointer-and-length pair every path argument takes, never terminated. */
    private static byte[] encode(String path) {
        if (path == null || path.isEmpty()) {
            throw new IndexException(Status.MALFORMED_INPUT, "the path is empty");
        }
        return path.getBytes(StandardCharsets.UTF_8);
    }

    private static long lengthOf(byte[] buffer) {
        if (buffer == null || buffer.length == 0) {
            throw new IndexException(Status.MALFORMED_INPUT, "the buffer is empty");
        }
        return buffer.length;
    }

    private static long lengthOf(ByteBuffer buffer) {
        if (buffer == null || !buffer.isDirect()) {
            throw new IndexException(
                    Status.UNSUPPORTED_LAYOUT, "a heap buffer has no stable address; allocate a direct one");
        }
        if (buffer.remaining() == 0) {
            throw new IndexException(Status.MALFORMED_INPUT, "the buffer is empty");
        }
        return buffer.remaining();
    }

    private static long offsetOf(ByteBuffer buffer) {
        return buffer.position();
    }

    private static native long nativeInitialize(long[] options);

    private static native void nativeFree(long handle);

    private static native long nativeProbe(long handle, int measure);

    private static native String nativeCapabilities(long handle);

    private static native void nativeReserve(long handle, long capacity);

    private static native void nativeClear(long handle);

    private static native long nativeAdd(long handle, byte[] keys, Object vectors, long byteOffset, long rows,
            long stride, int scalarKind, long expansionFactor, long[] stats);

    private static native void nativeSearch(long handle, Object queries, long byteOffset, long rows, long stride,
            int scalarKind, long wanted, long expansionFactor, long[] stats, byte[] keys, double[] distances,
            long[] counts);

    private static native long nativeCount(long handle, byte[] key);

    private static native long nativeGet(long handle, byte[] key, Object vectors, long byteOffset, long wanted,
            long stride, int scalarKind);

    private static native void nativeMerge(long handle, long donor);

    private static native void nativeSave(long handle, byte[] path);

    private static native void nativeLoad(long handle, byte[] path);

    private static native void nativeSaveBuffer(long handle, Object buffer, long byteOffset, long length);

    private static native void nativeLoadBuffer(long handle, Object buffer, long byteOffset, long length);

    private static native void nativeMetadata(byte[] path, long[] options);

    private static native void nativeMetadataBuffer(Object buffer, long byteOffset, long length, long[] options);

    /** Lowers a flat options layout into the C struct and lifts it straight back. */
    static native void nativeReloadOptions(long[] input, long[] output);
}
