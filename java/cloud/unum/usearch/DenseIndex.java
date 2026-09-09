package cloud.unum.usearch;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;

import cloud.unum.usearch.Types.Executor;
import cloud.unum.usearch.Types.Extents;
import cloud.unum.usearch.Types.Key;
import cloud.unum.usearch.Types.Matches;
import cloud.unum.usearch.Types.Matrix;
import cloud.unum.usearch.Types.Residency;
import cloud.unum.usearch.Types.ScalarKind;
import cloud.unum.usearch.Types.Status;
import cloud.unum.usearch.Types.USearchException;

/**
 * A Vector Search Index over equidimensional dense representations - the Java twin of
 * {@code include/usearch/dense_index.h}, with the modality's own options, metric, telemetry and
 * per-call settings as nested types.
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
 * The filtered search of the C interface stays unbound: its predicate is called from engine threads
 * that are not attached to this JVM. Filter the results of {@link #search} instead, or narrow the
 * index.
 *
 * <p>
 * Every method here reports failure as an unchecked {@link USearchException} carrying the
 * {@link Status} the engine named, so the per-method tags below list only the statuses that method
 * distinctly produces. Two apply to all of them and are not repeated: UNINITIALIZED on a handle
 * that was closed, and CONTENDED when a second thread is already driving the same executor.
 *
 * <pre>{@code
 * try (Executor pool = Executor.forRam(0);
 *         DenseIndex index = new DenseOptions().dimensions(2).quantization(ScalarKind.F32).open()) {
 *     index.reserve(2);
 *     index.add(new Key[] {Key.of(1), Key.of(2)}, Matrix.of(new float[] {0, 1, 1, 0}, 2), pool);
 *     Matches matches = Matches.packed(1, 1);
 *     index.search(Matrix.of(new float[] {0, 1}, 2), matches, pool, SearchOptions.DERIVED);
 *     System.out.println(matches.key(0, 0));
 * }
 * }</pre>
 */
public final class DenseIndex implements AutoCloseable {

    static {
        Library.require();
    }

    /**
     * A distance kind supported out of the box, with SIMD kernels dispatched by the machine's
     * capabilities. All of them derive from a single inner-product kernel with pre-computed
     * magnitudes; Hamming and Jaccard are bit-level, over the packed sub-byte kinds. The C layer's
     * custom-callback escape hatch stays unexposed, since a caller-supplied metric would be scored
     * from engine threads that are not attached to this JVM.
     */
    public enum DenseMetric {

        /** A metric the library could not name. */
        UNKNOWN(0, "unknown"),
        /** The inner product. */
        INNER(1, "inner"),
        /** The cosine distance. */
        ANGULAR(2, "angular"),
        /** The euclidean distance. */
        EUCLIDEAN(3, "euclidean"),
        /** The count of differing bits. */
        HAMMING(4, "hamming"),
        /** The bit-level Jaccard distance. */
        JACCARD(5, "jaccard");

        private final int code;
        private final String stem;

        DenseMetric(int code, String stem) {
            this.code = code;
            this.stem = stem;
        }

        /** The C enumerator's numeric value. */
        public int code() {
            return code;
        }

        /** Names the metric a code carries; a code this build cannot name becomes UNKNOWN. */
        public static DenseMetric of(int code) {
            for (DenseMetric candidate : values()) {
                if (candidate.code == code) {
                    return candidate;
                }
            }
            return UNKNOWN;
        }

        /** Renders the metric as its C enumerator's stem. */
        @Override
        public String toString() {
            return stem;
        }
    }

    /**
     * The construction and loading options of a dense index.
     *
     * <p>
     * Every setter returns this instance, so an index is configured in one expression:
     *
     * <pre>{@code
     * try (DenseIndex index = new DenseOptions()
     *         .dimensions(768)
     *         .metric(DenseMetric.ANGULAR)
     *         .quantization(ScalarKind.I8)
     *         .open()) {
     *     index.reserve(1_000_000);
     * }
     * }</pre>
     */
    public static final class DenseOptions {

        private DenseMetric metric = DenseMetric.ANGULAR;
        private ScalarKind quantization = ScalarKind.BF16;
        private long dimensions = 0;
        private long connectivity = 0;
        private long valuesPerKey = 0;
        private Residency residency = Residency.RAM;

        /** An angular BF16 index in RAM, with every optional field derived. */
        public DenseOptions() {}

        /** The distance kind used between vectors. */
        public DenseOptions metric(DenseMetric metric) {
            this.metric = metric;
            return this;
        }

        /** The scalar type vectors are stored and quantized into. */
        public DenseOptions quantization(ScalarKind quantization) {
            this.quantization = quantization;
            return this;
        }

        /**
         * The width of the indexed vectors; zero defers the geometry to a later
         * {@link DenseIndex#load(String, Executor)}, which reads it off the checkpoint.
         *
         * @throws USearchException SHAPE_MISMATCH on a negative width.
         */
        public DenseOptions dimensions(long dimensions) {
            if (dimensions < 0) {
                throw new USearchException(Status.SHAPE_MISMATCH, "the dimensionality is negative");
            }
            this.dimensions = dimensions;
            return this;
        }

        /**
         * The connections-per-node bound of the graph; zero derives.
         *
         * @throws USearchException SHAPE_MISMATCH on a negative bound.
         */
        public DenseOptions connectivity(long connectivity) {
            if (connectivity < 0) {
                throw new USearchException(Status.SHAPE_MISMATCH, "the connectivity is negative");
            }
            this.connectivity = connectivity;
            return this;
        }

        /**
         * The upper bound on entries sharing one key; zero derives to one - unique keys.
         *
         * @throws USearchException SHAPE_MISMATCH on a negative bound.
         */
        public DenseOptions valuesPerKey(long valuesPerKey) {
            if (valuesPerKey < 0) {
                throw new USearchException(Status.SHAPE_MISMATCH, "the entries per key is negative");
            }
            this.valuesPerKey = valuesPerKey;
            return this;
        }

        /** Where the payload lives, and what loading a checkpoint means. */
        public DenseOptions residency(Residency residency) {
            this.residency = residency;
            return this;
        }

        /** The distance kind used between vectors. */
        public DenseMetric metric() {
            return metric;
        }

        /** The scalar type vectors are stored and quantized into. */
        public ScalarKind quantization() {
            return quantization;
        }

        /** The width of the indexed vectors; zero means the geometry is still unfrozen. */
        public long dimensions() {
            return dimensions;
        }

        /** The connections-per-node bound of the graph; zero means it derives. */
        public long connectivity() {
            return connectivity;
        }

        /** The upper bound on entries sharing one key; zero means unique keys. */
        public long valuesPerKey() {
            return valuesPerKey;
        }

        /** Where the payload lives, and what loading a checkpoint means. */
        public Residency residency() {
            return residency;
        }

        /** Initializes an index for the residency these options select. */
        public DenseIndex open() {
            return new DenseIndex(this);
        }

        /** Lowers the options into the flat layout the native layer reads. */
        long[] toWords() {
            long[] words = new long[OptionsWords.COUNT];
            words[OptionsWords.METRIC] = metric.code();
            words[OptionsWords.QUANTIZATION] = quantization.code();
            words[OptionsWords.DIMENSIONS] = dimensions;
            words[OptionsWords.CONNECTIVITY] = connectivity;
            words[OptionsWords.VALUES_PER_KEY] = valuesPerKey;
            words[OptionsWords.RESIDENCY] = residency.code();
            return words;
        }

        /** Lifts a flat layout the native layer wrote back into options. */
        static DenseOptions fromWords(long[] words) {
            DenseOptions options = new DenseOptions();
            options.metric = DenseMetric.of(Extents.toIndex(words[OptionsWords.METRIC], "the metric code"));
            options.quantization = ScalarKind.of(Extents.toIndex(words[OptionsWords.QUANTIZATION], "the scalar code"));
            options.dimensions = words[OptionsWords.DIMENSIONS];
            options.connectivity = words[OptionsWords.CONNECTIVITY];
            options.valuesPerKey = words[OptionsWords.VALUES_PER_KEY];
            options.residency = Residency.of(Extents.toIndex(words[OptionsWords.RESIDENCY], "the residency code"));
            if (options.metric == DenseMetric.UNKNOWN) {
                throw new USearchException(Status.UNKNOWN_KIND, "the options name an unknown metric");
            }
            if (options.quantization == ScalarKind.UNKNOWN) {
                throw new USearchException(Status.UNKNOWN_KIND, "the options name an unknown scalar kind");
            }
            return options;
        }

        /**
         * Lowers the options and lifts them straight back - the marshalling round trip through the C
         * struct and its residency tag, which nothing else on this side can reach.
         */
        DenseOptions reload() {
            long[] words = new long[OptionsWords.COUNT];
            nativeReloadOptions(toWords(), words);
            return fromWords(words);
        }

        /** Renders the options as the fields a reader would set. */
        @Override
        public String toString() {
            return "DenseOptions(dimensions=" + dimensions
                    + ", metric=" + metric
                    + ", quantization=" + quantization
                    + ", connectivity=" + connectivity
                    + ", valuesPerKey=" + valuesPerKey
                    + ", residency=" + residency + ")";
        }
    }

    /**
     * The flat layout a {@link DenseOptions} travels in, one word per field.
     *
     * <p>
     * The C options struct ends in a residency-tagged union naming where shards are placed, and JNI has
     * no way to describe a union to Java. Only the tag crosses, so placement derives on the other side.
     * The native layer is the only place that reads or writes the struct itself.
     */
    static final class OptionsWords {

        private OptionsWords() {}

        /** The metric kind's code. */
        static final int METRIC = 0;
        /** The stored quantization's code. */
        static final int QUANTIZATION = 1;
        /** The frozen dimensionality; zero defers the geometry to a later load. */
        static final int DIMENSIONS = 2;
        /** The connections-per-node bound; zero derives. */
        static final int CONNECTIVITY = 3;
        /** The upper bound on entries sharing one key; zero derives to one. */
        static final int VALUES_PER_KEY = 4;
        /** The residency's code, selecting where the payload lives. */
        static final int RESIDENCY = 5;
        /** How many words one options structure occupies. */
        static final int COUNT = 6;

        /** The three traversal counters a call reports, in the order the native layer writes them. */
        static final int STATS_COUNT = 3;
    }

    /**
     * The geometry and metric a checkpoint froze - everything a reader learns without opening it, and
     * nothing a reader must choose for itself. It travels in its own flat layout, one word per
     * component.
     *
     * <p>
     * A checkpoint records neither the residency nor the bound on entries sharing one key: the first
     * says where it was written rather than where a reader must keep it, and the second bounds a live
     * handle's mutable state rather than the serialized form. Both stay the caller's to choose in the
     * {@link DenseOptions} a handle is built from.
     *
     * @param dimensions the scalars per stored vector.
     * @param metric the distance kind the checkpoint scores with.
     * @param quantization the scalar type its rows are stored and quantized into.
     * @param connectivity the connections-per-node bound the graph was built with.
     */
    public record DenseMetadata(long dimensions, DenseMetric metric, ScalarKind quantization, long connectivity) {

        /** The metric kind's code. */
        static final int METRIC = 0;
        /** The stored quantization's code. */
        static final int QUANTIZATION = 1;
        /** The frozen dimensionality. */
        static final int DIMENSIONS = 2;
        /** The connections-per-node bound the graph was built with. */
        static final int CONNECTIVITY = 3;
        /** How many words one metadata structure occupies. */
        static final int COUNT = 4;

        /**
         * Lifts the geometry and metric a checkpoint's headers describe.
         *
         * @throws USearchException UNKNOWN_KIND when the checkpoint names a metric or scalar this
         *     build cannot name.
         */
        static DenseMetadata fromWords(long[] words) {
            DenseMetric metric = DenseMetric.of(Extents.toIndex(words[METRIC], "the metric code"));
            ScalarKind quantization = ScalarKind.of(Extents.toIndex(words[QUANTIZATION], "the scalar code"));
            if (metric == DenseMetric.UNKNOWN) {
                throw new USearchException(Status.UNKNOWN_KIND, "the checkpoint names an unknown metric");
            }
            if (quantization == ScalarKind.UNKNOWN) {
                throw new USearchException(Status.UNKNOWN_KIND, "the checkpoint names an unknown scalar kind");
            }
            return new DenseMetadata(words[DIMENSIONS], metric, quantization, words[CONNECTIVITY]);
        }
    }

    /**
     * The per-call settings of {@link DenseIndex#add} and {@link DenseIndex#refine} - a pure input,
     * never written.
     *
     * @param expansionFactor the candidate-pool breadth the build walk explores; zero derives.
     */
    public record InsertionOptions(long expansionFactor) {

        /** Every setting derived from the frozen geometry. */
        public static final InsertionOptions DERIVED = new InsertionOptions(0);

        /** @throws USearchException SHAPE_MISMATCH on a negative breadth, which no pool may have. */
        public InsertionOptions {
            if (expansionFactor < 0) {
                throw new USearchException(Status.SHAPE_MISMATCH, "the expansion factor is negative");
            }
        }
    }

    /**
     * The per-call settings of the read-only walks - a pure input, never written. The late-interaction
     * verb reads the same breadth as emitted-pair density rather than as recall.
     *
     * @param expansionFactor the candidate-pool breadth the walk explores; zero derives.
     */
    public record SearchOptions(long expansionFactor) {

        /** Every setting derived from the frozen geometry. */
        public static final SearchOptions DERIVED = new SearchOptions(0);

        /** @throws USearchException SHAPE_MISMATCH on a negative breadth, which no pool may have. */
        public SearchOptions {
            if (expansionFactor < 0) {
                throw new USearchException(Status.SHAPE_MISMATCH, "the expansion factor is negative");
            }
        }
    }

    /**
     * The traversal telemetry one call reports. Every verb that walks the graph returns a fresh set
     * rather than accumulating into a caller's, so a sweep over many batches sums them itself with
     * {@link #plus(DenseStats)}.
     *
     * @param visitedMembers graph members the call visited.
     * @param computedDistances distances the call computed, diversity re-selections included.
     * @param restarts starvation re-seedings - a graph-connectivity health signal rather than a cost
     *     counter.
     */
    public record DenseStats(long visitedMembers, long computedDistances, long restarts) {

        /** The counters a call that walked nothing reports. */
        public static final DenseStats NONE = new DenseStats(0, 0, 0);

        /** Sums this telemetry with another call's. */
        public DenseStats plus(DenseStats other) {
            return new DenseStats(
                    visitedMembers + other.visitedMembers,
                    computedDistances + other.computedDistances,
                    restarts + other.restarts);
        }

        /** Lifts the three counters the native layer wrote. */
        static DenseStats fromWords(long[] words) {
            return new DenseStats(words[0], words[1], words[2]);
        }
    }

    /**
     * What one refinement pass did.
     *
     * @param changedNodes nodes whose edges the pass re-selected; zero means the graph has settled.
     * @param stats the traversal telemetry the sweep reported.
     */
    public record Refinement(long changedNodes, DenseStats stats) {}

    /** The scalar quantities a handle can be asked for, each one probe of the C interface. */
    private enum Measure {

        /** The number of stored entries. */
        SIZE(0),
        /** The reserved capacity in entries. */
        CAPACITY(1),
        /** The frozen width of the stored vectors. */
        DIMENSIONS(2),
        /** The frozen connections-per-node bound of the graph. */
        CONNECTIVITY(3),
        /** The memory the index occupies, in bytes. */
        MEMORY_USAGE(4),
        /** The bytes one monolith of the index takes under the default layout. */
        STORAGE_REQUIREMENTS(5);

        private final int code;

        Measure(int code) {
            this.code = code;
        }

        /** The code the native layer switches on. */
        int code() {
            return code;
        }
    }

    private volatile long handle;

    /**
     * Initializes an index for the residency the options select.
     *
     * @throws USearchException MALFORMED_INPUT on null options, UNKNOWN_KIND on a metric or scalar
     *     this build cannot name, and DEVICE_FAILURE when a device residency finds no device.
     */
    public DenseIndex(DenseOptions options) {
        if (options == null) {
            throw new USearchException(Status.MALFORMED_INPUT, "the options are null");
        }
        handle = nativeInitialize(options.toWords());
        if (handle == 0) {
            throw new USearchException(Status.INTERNAL, "the library returned no handle and no error");
        }
    }

    /**
     * Reads a checkpoint's geometry and metric metadata without loading it - from the manifest's
     * summary when the path names one, so no shard is opened.
     *
     * @throws USearchException MALFORMED_INPUT on an empty path or a checkpoint that ends early,
     *     IO_FAILURE when the file cannot be opened, and UNKNOWN_KIND when it names a metric or
     *     scalar this build cannot name.
     */
    public static DenseMetadata metadata(String path) {
        long[] words = new long[DenseMetadata.COUNT];
        nativeMetadata(encode(path), words);
        return DenseMetadata.fromWords(words);
    }

    /**
     * Reads geometry and metric metadata from a serialized monolith slab.
     *
     * @throws USearchException MALFORMED_INPUT on an empty slab or one that ends early, and
     *     UNKNOWN_KIND when it names a metric or scalar this build cannot name.
     */
    public static DenseMetadata metadata(byte[] buffer) {
        long[] words = new long[DenseMetadata.COUNT];
        nativeMetadataBuffer(buffer, Matrix.Storage.BYTES.code(), 0, lengthOf(buffer), words);
        return DenseMetadata.fromWords(words);
    }

    /**
     * Reads geometry and metric metadata from a serialized monolith buffer.
     *
     * @throws USearchException UNSUPPORTED_LAYOUT for a heap buffer, MALFORMED_INPUT on an empty
     *     buffer or one that ends early, and UNKNOWN_KIND when it names a metric or scalar this
     *     build cannot name.
     */
    public static DenseMetadata metadata(ByteBuffer buffer) {
        long[] words = new long[DenseMetadata.COUNT];
        nativeMetadataBuffer(buffer, Matrix.Storage.DIRECT.code(), offsetOf(buffer), lengthOf(buffer), words);
        return DenseMetadata.fromWords(words);
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
     *
     * @throws USearchException OUT_OF_MEMORY when the arena cannot be claimed, READ_ONLY on a handle
     *     that opened its checkpoint in place, and OVERFLOW past the identity space's ceiling.
     */
    public void reserve(long capacity) {
        nativeReserve(live(), capacity);
    }

    /**
     * Erases all entries, keeping the reserved capacity and the frozen geometry.
     *
     * @throws USearchException READ_ONLY on a handle that opened its checkpoint in place.
     */
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
     * index built with zero dimensions freezes it at its first {@link #load(String, Executor)}.
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
     * The bytes one monolith of the index takes under the default layout - what
     * {@link #saveBuffer(byte[], Executor)} needs; a manifest checkpoint's parts sum to about the
     * same, plus the manifest.
     */
    public long storageRequirements() {
        return nativeProbe(live(), Measure.STORAGE_REQUIREMENTS.code());
    }

    /** The capability codename the index dispatched to on this machine. */
    public String capabilities() {
        return nativeCapabilities(live());
    }

    /**
     * Inserts one vector under one key, fanned out over an executor.
     *
     * @throws USearchException SHAPE_MISMATCH when the matrix is not exactly one row wide.
     */
    public DenseStats add(Key key, Matrix vector, Executor executor) {
        if (vector.rows() != 1) {
            throw new USearchException(Status.SHAPE_MISMATCH, "one key admits exactly one row");
        }
        return add(new Key[] {key}, vector, executor, InsertionOptions.DERIVED);
    }

    /**
     * Inserts a matrix of vectors under their keys, fanned out over an executor.
     *
     * @throws USearchException the statuses {@link #add(Key[], Matrix, Executor, InsertionOptions)}
     *     names, with the candidate-pool breadth derived.
     */
    public DenseStats add(Key[] keys, Matrix vectors, Executor executor) {
        return add(keys, vectors, executor, InsertionOptions.DERIVED);
    }

    /**
     * Inserts a matrix of vectors under their keys - the native batch operation, where a single
     * vector is a one-row matrix. Overlaps safely with concurrent searches on the same handle.
     *
     * <p>
     * The input scalar type is independent of the stored quantization; the library converts on
     * ingest. A batch is all-or-nothing: a returning call published every row, a failing one
     * published none. An accelerator fault severe enough to poison the device context is the sole
     * exception, reporting DEVICE_FAILURE and latching the index unusable rather than partially
     * applied.
     *
     * @return the traversal telemetry the build walk reported.
     * @throws USearchException SHAPE_MISMATCH when the key count and the row count disagree or the
     *     rows are not as wide as the frozen geometry, UNKNOWN_KIND on a scalar the engine cannot
     *     convert, READ_ONLY on a handle that opened its checkpoint in place, and OUT_OF_MEMORY
     *     when the batch outgrows the reservation.
     */
    public DenseStats add(Key[] keys, Matrix vectors, Executor executor, InsertionOptions options) {
        long index = live();
        vectors.validate(dimensions());
        if (keys == null) {
            throw new USearchException(Status.MALFORMED_INPUT, "the keys are null");
        }
        if (keys.length != vectors.rows()) {
            throw new USearchException(Status.SHAPE_MISMATCH, "the key count does not match the vector row count");
        }
        if (keys.length == 0) {
            return DenseStats.NONE;
        }
        long[] telemetry = new long[OptionsWords.STATS_COUNT];
        long published = nativeAdd(
                index,
                Executor.handleOf(executor),
                Key.flatten(keys),
                vectors.slab(),
                vectors.storage().code(),
                vectors.byteOffset(),
                vectors.rows(),
                vectors.columns(),
                vectors.strideBytes(),
                vectors.kind().code(),
                options.expansionFactor(),
                telemetry);
        if (published != keys.length) {
            throw new USearchException(
                    Status.INTERNAL, "the batch published " + published + " of " + keys.length + " rows");
        }
        return DenseStats.fromWords(telemetry);
    }

    /**
     * Re-selects the neighbors of every stored entry, fanned out over an executor.
     *
     * @throws USearchException the statuses {@link #refine(Executor, InsertionOptions)} names, with
     *     the candidate-pool breadth derived.
     */
    public Refinement refine(Executor executor) {
        return refine(executor, InsertionOptions.DERIVED);
    }

    /**
     * Re-selects the neighbors of every stored entry against the finished graph, improving edges that
     * were chosen when the graph around them was still incomplete.
     *
     * <p>
     * Searches may run throughout, since edge re-selection publishes one row at a time and never
     * removes a node; insertions may not. The sweep resumes where a previous one stopped, so a
     * caller loops until the changed-node count falls below whatever rate it considers settled.
     *
     * @throws USearchException READ_ONLY on a handle that opened its checkpoint in place. A stopped
     *     sweep is not a failure: it leaves the graph correct and merely less improved, so nothing
     *     is reported for it.
     */
    public Refinement refine(Executor executor, InsertionOptions options) {
        long[] telemetry = new long[OptionsWords.STATS_COUNT];
        long changed = nativeRefine(live(), Executor.handleOf(executor), options.expansionFactor(), telemetry);
        return new Refinement(changed, DenseStats.fromWords(telemetry));
    }

    /**
     * Resolves a matrix of queries into caller-owned outputs - the allocation-free batch path,
     * parallelized internally. Never writes to the graph, so any number of these may overlap with
     * concurrent insertions.
     *
     * @return the traversal telemetry the walk reported.
     * @throws USearchException SHAPE_MISMATCH when the rows are not as wide as the frozen geometry or
     *     the outputs hold fewer rows than there are queries, and UNKNOWN_KIND on a scalar the
     *     engine cannot convert.
     */
    public DenseStats search(Matrix queries, Matches matches, Executor executor, SearchOptions options) {
        long index = live();
        queries.validate(dimensions());
        if (queries.rows() > matches.queries()) {
            throw new USearchException(Status.SHAPE_MISMATCH, "the outputs hold fewer rows than there are queries");
        }
        if (queries.rows() == 0) {
            return DenseStats.NONE;
        }
        long[] telemetry = new long[OptionsWords.STATS_COUNT];
        nativeSearch(
                index,
                Executor.handleOf(executor),
                queries.slab(),
                queries.storage().code(),
                queries.byteOffset(),
                queries.rows(),
                queries.columns(),
                queries.strideBytes(),
                queries.kind().code(),
                matches.keySlab(),
                matches.scoreSlab(),
                matches.countSlab(),
                matches.wanted(),
                options.expansionFactor(),
                telemetry);
        return DenseStats.fromWords(telemetry);
    }

    /**
     * Late-interaction scoring of multi-row queries against multi-row entries: every query row takes
     * its best match within a key, and those maxima sum to that key's score. A document is one key
     * holding its token rows, so the wanted count is the "k" over keys rather than over rows.
     *
     * <p>
     * Every query carries the same number of rows, so a padded token batch applies as is, and the
     * query count is the row count divided by that. Approximate in the same way a search is: a token
     * whose neighborhood the walk never entered contributes no maximum, so a key can score below its
     * exact value but never above it. Runs on the inner-product kinds alone; a graph frozen under a
     * distance kind reports CONFIG_MISMATCH.
     *
     * @return the traversal telemetry the walk reported.
     * @throws USearchException CONFIG_MISMATCH on a graph frozen under a distance kind, and
     *     SHAPE_MISMATCH when the token rows do not divide into whole queries, the rows are not as
     *     wide as the frozen geometry, or the outputs hold fewer rows than there are queries.
     */
    public DenseStats maxsim(
            Matrix queries, long rowsPerQuery, Matches matches, Executor executor, SearchOptions options) {
        long index = live();
        queries.validate(dimensions());
        long resolved = queryCountOf(queries, rowsPerQuery);
        if (resolved > matches.queries()) {
            throw new USearchException(Status.SHAPE_MISMATCH, "the outputs hold fewer rows than there are queries");
        }
        if (resolved == 0) {
            return DenseStats.NONE;
        }
        long[] telemetry = new long[OptionsWords.STATS_COUNT];
        nativeMaxsim(
                index,
                Executor.handleOf(executor),
                queries.slab(),
                queries.storage().code(),
                queries.byteOffset(),
                queries.rows(),
                queries.columns(),
                queries.strideBytes(),
                queries.kind().code(),
                rowsPerQuery,
                matches.keySlab(),
                matches.scoreSlab(),
                matches.countSlab(),
                matches.wanted(),
                options.expansionFactor(),
                telemetry);
        return DenseStats.fromWords(telemetry);
    }

    /** How many entries are stored under a key; zero means the index holds none. */
    public long count(Key key) {
        return nativeCount(live(), key.toBytes());
    }

    /** Whether the index holds at least one entry under a key. */
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
     * @throws USearchException SHAPE_MISMATCH when the rows are not as wide as the frozen geometry,
     *     and UNKNOWN_KIND on a scalar the engine cannot convert.
     */
    public long get(Key key, Matrix vectors) {
        long index = live();
        vectors.validate(dimensions());
        return nativeGet(
                index,
                key.toBytes(),
                vectors.slab(),
                vectors.storage().code(),
                vectors.byteOffset(),
                vectors.rows(),
                vectors.columns(),
                vectors.strideBytes(),
                vectors.kind().code());
    }

    /**
     * Writes the index under a path, in the layout its extension picks - one {@code .safegraphs}
     * monolith, or a {@code .safegraphs.json} manifest beside its shards - fanned out over an
     * executor, one shard per thread or stream. Demands a mutation-free window; concurrent readers
     * keep searching throughout.
     *
     * @throws USearchException MALFORMED_INPUT on an empty path, and IO_FAILURE when the destination
     *     cannot be written - which leaves a file without its header, for the caller to remove.
     */
    public void save(String path, Executor executor) {
        nativeSave(live(), Executor.handleOf(executor), encode(path));
    }

    /**
     * Reads a checkpoint into this handle, shard by shard, with the handle's residency deciding the
     * behavior: materialize into RAM, upload to the device, or open in place on disk. A handle whose
     * geometry was left zeroed freezes it here, from the checkpoint's own headers.
     *
     * @throws USearchException MALFORMED_INPUT on an empty path or a checkpoint that ends early,
     *     IO_FAILURE when the file cannot be read, and CONFIG_MISMATCH when its geometry
     *     contradicts one this handle already froze. A failed load publishes nothing, so the handle
     *     is empty afterwards and loads cleanly again.
     */
    public void load(String path, Executor executor) {
        nativeLoad(live(), Executor.handleOf(executor), encode(path));
    }

    /**
     * Writes the index into a caller-owned slab as one monolith, fanned out over an executor.
     *
     * @throws USearchException MALFORMED_INPUT on an empty slab, and SHAPE_MISMATCH when it is
     *     shorter than {@link #storageRequirements()}.
     */
    public void saveBuffer(byte[] buffer, Executor executor) {
        nativeSaveBuffer(live(), Executor.handleOf(executor), buffer, Matrix.Storage.BYTES.code(), 0, lengthOf(buffer));
    }

    /**
     * Writes the index into a caller-owned direct buffer, fanned out over an executor.
     *
     * @throws USearchException UNSUPPORTED_LAYOUT for a heap buffer, MALFORMED_INPUT on an empty one,
     *     and SHAPE_MISMATCH when it is shorter than {@link #storageRequirements()}.
     */
    public void saveBuffer(ByteBuffer buffer, Executor executor) {
        nativeSaveBuffer(
                live(),
                Executor.handleOf(executor),
                buffer,
                Matrix.Storage.DIRECT.code(),
                offsetOf(buffer),
                lengthOf(buffer));
    }

    /**
     * Reads a monolith from a slab into this handle, fanned out over an executor.
     *
     * @throws USearchException MALFORMED_INPUT on an empty slab or a monolith that ends early, and
     *     CONFIG_MISMATCH when its geometry contradicts one this handle already froze.
     */
    public void loadBuffer(byte[] buffer, Executor executor) {
        nativeLoadBuffer(live(), Executor.handleOf(executor), buffer, Matrix.Storage.BYTES.code(), 0, lengthOf(buffer));
    }

    /**
     * Reads a monolith from a direct buffer into this handle, with the residency deciding the
     * behavior like {@link #load(String, Executor)}. A residency that opens in place demands the
     * buffer outlive the handle.
     *
     * @throws USearchException UNSUPPORTED_LAYOUT for a heap buffer, MALFORMED_INPUT on an empty one
     *     or a monolith that ends early, and CONFIG_MISMATCH when its geometry contradicts one this
     *     handle already froze.
     */
    public void loadBuffer(ByteBuffer buffer, Executor executor) {
        nativeLoadBuffer(
                live(),
                Executor.handleOf(executor),
                buffer,
                Matrix.Storage.DIRECT.code(),
                offsetOf(buffer),
                lengthOf(buffer));
    }

    /** Reports the handle, refusing a closed index. */
    private long live() {
        long current = handle;
        if (current == 0) {
            throw new USearchException(Status.UNINITIALIZED, "the index is freed or was never initialized");
        }
        return current;
    }

    /** How many queries a token batch carries, refusing rows that do not divide into whole ones. */
    private static long queryCountOf(Matrix queries, long rowsPerQuery) {
        if (rowsPerQuery <= 0) {
            throw new USearchException(Status.SHAPE_MISMATCH, "a query cannot carry zero rows");
        }
        if (queries.rows() % rowsPerQuery != 0) {
            throw new USearchException(Status.SHAPE_MISMATCH, "the token rows do not divide into whole queries");
        }
        return queries.rows() / rowsPerQuery;
    }

    /** Borrows a path as the pointer-and-length pair every path argument takes, never terminated. */
    private static byte[] encode(String path) {
        if (path == null || path.isEmpty()) {
            throw new USearchException(Status.MALFORMED_INPUT, "the path is empty");
        }
        return path.getBytes(StandardCharsets.UTF_8);
    }

    /** The bytes a slab carries, refusing an empty one no checkpoint could fill. */
    private static long lengthOf(byte[] buffer) {
        if (buffer == null || buffer.length == 0) {
            throw new USearchException(Status.MALFORMED_INPUT, "the buffer is empty");
        }
        return buffer.length;
    }

    /** The bytes a direct buffer carries, refusing a heap one that has no stable address. */
    private static long lengthOf(ByteBuffer buffer) {
        if (buffer == null || !buffer.isDirect()) {
            throw new USearchException(
                    Status.UNSUPPORTED_LAYOUT, "a heap buffer has no stable address; allocate a direct one");
        }
        if (buffer.remaining() == 0) {
            throw new USearchException(Status.MALFORMED_INPUT, "the buffer is empty");
        }
        return buffer.remaining();
    }

    /** The byte offset a positioned buffer starts at. */
    private static long offsetOf(ByteBuffer buffer) {
        return buffer.position();
    }

    private static final MethodHandle INITIALIZE = Library.downcall(
            "usearch_dense_init", FunctionDescriptor.of(Library.POINTER, Library.POINTER, Library.POINTER));

    private static final MethodHandle RELEASE =
            Library.downcall("usearch_dense_free", FunctionDescriptor.ofVoid(Library.POINTER));

    private static final MethodHandle RESERVE = Library.downcall(
            "usearch_dense_reserve", FunctionDescriptor.ofVoid(Library.POINTER, Library.SIZE, Library.POINTER));

    private static final MethodHandle CLEAR =
            Library.downcall("usearch_dense_clear", FunctionDescriptor.ofVoid(Library.POINTER, Library.POINTER));

    private static final MethodHandle CAPABILITIES = Library.downcall(
            "usearch_dense_capabilities", FunctionDescriptor.of(Library.POINTER, Library.POINTER, Library.POINTER));

    /** One handle per measure, indexed by the {@link Measure} code, so the probe is a lookup. */
    private static final MethodHandle[] MEASURES = {
        Library.downcall("usearch_dense_size", FunctionDescriptor.of(Library.SIZE, Library.POINTER, Library.POINTER)),
        Library.downcall(
                "usearch_dense_capacity", FunctionDescriptor.of(Library.SIZE, Library.POINTER, Library.POINTER)),
        Library.downcall(
                "usearch_dense_dimensions", FunctionDescriptor.of(Library.SIZE, Library.POINTER, Library.POINTER)),
        Library.downcall(
                "usearch_dense_connectivity", FunctionDescriptor.of(Library.SIZE, Library.POINTER, Library.POINTER)),
        Library.downcall(
                "usearch_dense_memory_usage", FunctionDescriptor.of(Library.SIZE, Library.POINTER, Library.POINTER)),
        Library.downcall(
                "usearch_dense_storage_requirements",
                FunctionDescriptor.of(Library.SIZE, Library.POINTER, Library.POINTER)),
    };

    private static final MethodHandle ADD = Library.downcall(
            "usearch_dense_add",
            FunctionDescriptor.of(
                    Library.SIZE,
                    Library.POINTER,
                    Library.POINTER,
                    Library.KEYS_VIEW,
                    Library.POINTER,
                    Library.POINTER,
                    Library.POINTER,
                    Library.POINTER,
                    Library.POINTER,
                    Library.POINTER));

    private static final MethodHandle REFINE = Library.downcall(
            "usearch_dense_refine",
            FunctionDescriptor.of(
                    Library.SIZE,
                    Library.POINTER,
                    Library.POINTER,
                    Library.POINTER,
                    Library.POINTER,
                    Library.POINTER,
                    Library.POINTER,
                    Library.POINTER));

    private static final MethodHandle SEARCH = Library.downcall(
            "usearch_dense_search",
            FunctionDescriptor.ofVoid(
                    Library.POINTER,
                    Library.POINTER,
                    Library.POINTER,
                    Library.POINTER,
                    Library.POINTER,
                    Library.POINTER,
                    Library.POINTER,
                    Library.POINTER,
                    Library.POINTER));

    private static final MethodHandle MAXSIM = Library.downcall(
            "usearch_dense_maxsim",
            FunctionDescriptor.ofVoid(
                    Library.POINTER,
                    Library.POINTER,
                    Library.POINTER,
                    Library.SIZE,
                    Library.POINTER,
                    Library.POINTER,
                    Library.POINTER,
                    Library.POINTER,
                    Library.POINTER,
                    Library.POINTER));

    private static final MethodHandle COUNT = Library.downcall(
            "usearch_dense_count", FunctionDescriptor.of(Library.SIZE, Library.POINTER, Library.KEY, Library.POINTER));

    private static final MethodHandle GET = Library.downcall(
            "usearch_dense_get",
            FunctionDescriptor.of(Library.SIZE, Library.POINTER, Library.KEY, Library.POINTER, Library.POINTER));

    private static final MethodHandle SAVE = Library.downcall(
            "usearch_dense_save",
            FunctionDescriptor.ofVoid(
                    Library.POINTER,
                    Library.POINTER,
                    Library.BYTES_VIEW,
                    Library.POINTER,
                    Library.POINTER,
                    Library.POINTER));

    private static final MethodHandle LOAD = Library.downcall(
            "usearch_dense_load",
            FunctionDescriptor.ofVoid(
                    Library.POINTER,
                    Library.POINTER,
                    Library.BYTES_VIEW,
                    Library.POINTER,
                    Library.POINTER,
                    Library.POINTER));

    private static final MethodHandle SAVE_BUFFER = Library.downcall(
            "usearch_dense_save_buffer",
            FunctionDescriptor.ofVoid(
                    Library.POINTER,
                    Library.POINTER,
                    Library.BYTES_VIEW,
                    Library.POINTER,
                    Library.POINTER,
                    Library.POINTER));

    private static final MethodHandle LOAD_BUFFER = Library.downcall(
            "usearch_dense_load_buffer",
            FunctionDescriptor.ofVoid(
                    Library.POINTER,
                    Library.POINTER,
                    Library.BYTES_VIEW,
                    Library.POINTER,
                    Library.POINTER,
                    Library.POINTER));

    private static final MethodHandle METADATA = Library.downcall(
            "usearch_dense_metadata", FunctionDescriptor.ofVoid(Library.BYTES_VIEW, Library.POINTER, Library.POINTER));

    private static final MethodHandle METADATA_BUFFER = Library.downcall(
            "usearch_dense_metadata_buffer",
            FunctionDescriptor.ofVoid(Library.BYTES_VIEW, Library.POINTER, Library.POINTER));

    /**
     * Stages one Java slab into native memory. A direct {@link ByteBuffer} is addressed where it
     * already lives; a heap array is copied, because no Java array has a stable address.
     */
    private static MemorySegment lend(Arena arena, Object slab, int storage, long byteOffset, long lengthBytes) {
        if (slab == null) {
            return MemorySegment.NULL;
        }
        if (storage == DIRECT_STORAGE) {
            return MemorySegment.ofBuffer((ByteBuffer) slab).asSlice(byteOffset, lengthBytes);
        }
        MemorySegment staged = arena.allocate(lengthBytes);
        MemorySegment.copy(heapOf(slab), byteOffset, staged, 0, lengthBytes);
        return staged;
    }

    /** Copies a staged slab's contents back, which every output that is not direct needs. */
    private static void restore(MemorySegment staged, Object slab, int storage, long byteOffset, long lengthBytes) {
        if (slab == null || storage == DIRECT_STORAGE) {
            return;
        }
        MemorySegment.copy(staged, 0, heapOf(slab), byteOffset, lengthBytes);
    }

    /** The heap slab as a segment, whichever of the four element widths it holds. */
    private static MemorySegment heapOf(Object slab) {
        return switch (slab) {
            case double[] doubles -> MemorySegment.ofArray(doubles);
            case float[] floats -> MemorySegment.ofArray(floats);
            case short[] shorts -> MemorySegment.ofArray(shorts);
            case byte[] bytes -> MemorySegment.ofArray(bytes);
            default -> throw new USearchException(Status.UNSUPPORTED_LAYOUT, "the slab is of no known element width");
        };
    }

    /** The {@code Storage.DIRECT} code, which addresses a buffer in place rather than copying. */
    private static final int DIRECT_STORAGE = 4;

    /** Fills a @c usearch_matrix_view_t over a staged slab. */
    private static MemorySegment matrixView(
            Arena arena, MemorySegment rowsData, long rows, long columns, long strideBytes, int scalarKind) {
        MemorySegment view = arena.allocate(Library.MATRIX_VIEW);
        view.set(Library.POINTER, 0, rowsData);
        view.set(Library.SIZE, 8, rows);
        view.set(Library.SIZE, 16, columns);
        view.set(Library.SIZE, 24, strideBytes);
        view.set(Library.ENUM, 32, scalarKind);
        return view;
    }

    /** Fills a @c usearch_bytes_view_t over a staged byte run. */
    private static MemorySegment bytesView(Arena arena, MemorySegment data, long length) {
        MemorySegment view = arena.allocate(Library.BYTES_VIEW);
        view.set(Library.POINTER, 0, data);
        view.set(Library.SIZE, 8, length);
        return view;
    }

    /** The per-call knob both option structs carry, or null when nothing was asked for. */
    private static MemorySegment expansion(Arena arena, long expansionFactor) {
        if (expansionFactor == 0) {
            return MemorySegment.NULL;
        }
        MemorySegment options = arena.allocate(Library.SIZE);
        options.set(Library.SIZE, 0, expansionFactor);
        return options;
    }

    /** Lifts the three telemetry counters back into the caller's array, when one was asked for. */
    private static void reportStats(MemorySegment staged, long[] stats) {
        if (stats == null) {
            return;
        }
        for (int index = 0; index != Math.min(stats.length, 3); ++index) {
            stats[index] = staged.get(Library.SIZE, index * 8L);
        }
    }

    /** A zeroed telemetry block, or null when the caller wants no reporting. */
    private static MemorySegment statsBlock(Arena arena, long[] stats) {
        return stats == null ? MemorySegment.NULL : arena.allocate(Library.TRAVERSAL_STATS);
    }

    private static long nativeInitialize(long[] options) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment error = Library.error(arena);
            MemorySegment lowered = arena.allocate(Library.DENSE_OPTIONS);
            lowered.set(Library.ENUM, 0, (int) options[0]);
            lowered.set(Library.ENUM, 24, (int) options[1]);
            lowered.set(Library.SIZE, 32, options[2]);
            lowered.set(Library.SIZE, 40, options[3]);
            lowered.set(Library.SIZE, 48, options[4]);
            lowered.set(Library.ENUM, 56, (int) options[5]);
            MemorySegment opened = (MemorySegment) INITIALIZE.invokeExact(lowered, error);
            Library.raise(error);
            return opened.address();
        } catch (Throwable failure) {
            throw Types.rethrow(failure);
        }
    }

    private static void nativeFree(long handle) {
        try {
            RELEASE.invokeExact(MemorySegment.ofAddress(handle));
        } catch (Throwable failure) {
            throw Types.rethrow(failure);
        }
    }

    private static long nativeProbe(long handle, int measure) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment error = Library.error(arena);
            long answered = (long) MEASURES[measure].invokeExact(MemorySegment.ofAddress(handle), error);
            Library.raise(error);
            return answered;
        } catch (Throwable failure) {
            throw Types.rethrow(failure);
        }
    }

    @SuppressWarnings("restricted")
    private static String nativeCapabilities(long handle) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment error = Library.error(arena);
            MemorySegment named = (MemorySegment) CAPABILITIES.invokeExact(MemorySegment.ofAddress(handle), error);
            Library.raise(error);
            return named.equals(MemorySegment.NULL)
                    ? ""
                    : named.reinterpret(Long.MAX_VALUE).getString(0);
        } catch (Throwable failure) {
            throw Types.rethrow(failure);
        }
    }

    private static void nativeReserve(long handle, long capacity) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment error = Library.error(arena);
            RESERVE.invokeExact(MemorySegment.ofAddress(handle), capacity, error);
            Library.raise(error);
        } catch (Throwable failure) {
            throw Types.rethrow(failure);
        }
    }

    private static void nativeClear(long handle) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment error = Library.error(arena);
            CLEAR.invokeExact(MemorySegment.ofAddress(handle), error);
            Library.raise(error);
        } catch (Throwable failure) {
            throw Types.rethrow(failure);
        }
    }

    private static long nativeAdd(
            long handle,
            long executor,
            byte[] keys,
            Object vectors,
            int storage,
            long byteOffset,
            long rows,
            long columns,
            long strideBytes,
            int scalarKind,
            long expansionFactor,
            long[] stats) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment error = Library.error(arena);
            MemorySegment keyRun = arena.allocate(Library.KEYS_VIEW);
            keyRun.set(Library.POINTER, 0, arena.allocateFrom(ValueLayout.JAVA_BYTE, keys));
            keyRun.set(Library.SIZE, 8, rows);
            MemorySegment staged = lend(arena, vectors, storage, byteOffset, rows * strideBytes);
            MemorySegment view = matrixView(arena, staged, rows, columns, strideBytes, scalarKind);
            MemorySegment telemetry = statsBlock(arena, stats);
            long published = (long) ADD.invokeExact(
                    MemorySegment.ofAddress(handle),
                    MemorySegment.ofAddress(executor),
                    keyRun,
                    view,
                    expansion(arena, expansionFactor),
                    MemorySegment.NULL,
                    MemorySegment.NULL,
                    telemetry,
                    error);
            reportStats(telemetry, stats);
            Library.raise(error);
            return published;
        } catch (Throwable failure) {
            throw Types.rethrow(failure);
        }
    }

    private static long nativeRefine(long handle, long executor, long expansionFactor, long[] stats) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment error = Library.error(arena);
            MemorySegment telemetry = statsBlock(arena, stats);
            long settled = (long) REFINE.invokeExact(
                    MemorySegment.ofAddress(handle),
                    MemorySegment.ofAddress(executor),
                    expansion(arena, expansionFactor),
                    MemorySegment.NULL,
                    MemorySegment.NULL,
                    telemetry,
                    error);
            reportStats(telemetry, stats);
            Library.raise(error);
            return settled;
        } catch (Throwable failure) {
            throw Types.rethrow(failure);
        }
    }

    /** Stages the three ranked-output buffers and the span that names them. */
    private static MemorySegment matchesSpan(
            Arena arena, MemorySegment keys, MemorySegment scores, MemorySegment counts, long wanted) {
        MemorySegment span = arena.allocate(Library.MATCHES_SPAN);
        span.set(Library.POINTER, 0, keys);
        span.set(Library.SIZE, 8, wanted * Key.BYTES);
        span.set(Library.POINTER, 16, scores);
        span.set(Library.SIZE, 24, wanted * Double.BYTES);
        span.set(Library.POINTER, 32, counts);
        span.set(Library.SIZE, 40, wanted);
        return span;
    }

    private static void nativeSearch(
            long handle,
            long executor,
            Object queries,
            int storage,
            long byteOffset,
            long rows,
            long columns,
            long strideBytes,
            int scalarKind,
            byte[] keys,
            double[] scores,
            long[] counts,
            long wanted,
            long expansionFactor,
            long[] stats) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment error = Library.error(arena);
            MemorySegment view = matrixView(
                    arena,
                    lend(arena, queries, storage, byteOffset, rows * strideBytes),
                    rows,
                    columns,
                    strideBytes,
                    scalarKind);
            MemorySegment foundKeys = arena.allocate(keys.length);
            MemorySegment foundScores = arena.allocate((long) scores.length * Double.BYTES);
            MemorySegment foundCounts = arena.allocate((long) counts.length * Long.BYTES);
            MemorySegment telemetry = statsBlock(arena, stats);
            SEARCH.invokeExact(
                    MemorySegment.ofAddress(handle),
                    MemorySegment.ofAddress(executor),
                    view,
                    matchesSpan(arena, foundKeys, foundScores, foundCounts, wanted),
                    expansion(arena, expansionFactor),
                    MemorySegment.NULL,
                    MemorySegment.NULL,
                    telemetry,
                    error);
            reportStats(telemetry, stats);
            Library.raise(error);
            MemorySegment.copy(foundKeys, ValueLayout.JAVA_BYTE, 0, keys, 0, keys.length);
            MemorySegment.copy(foundScores, ValueLayout.JAVA_DOUBLE, 0, scores, 0, scores.length);
            MemorySegment.copy(foundCounts, ValueLayout.JAVA_LONG, 0, counts, 0, counts.length);
        } catch (Throwable failure) {
            throw Types.rethrow(failure);
        }
    }

    private static void nativeMaxsim(
            long handle,
            long executor,
            Object queries,
            int storage,
            long byteOffset,
            long rows,
            long columns,
            long strideBytes,
            int scalarKind,
            long rowsPerQuery,
            byte[] keys,
            double[] scores,
            long[] counts,
            long wanted,
            long expansionFactor,
            long[] stats) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment error = Library.error(arena);
            MemorySegment view = matrixView(
                    arena,
                    lend(arena, queries, storage, byteOffset, rows * strideBytes),
                    rows,
                    columns,
                    strideBytes,
                    scalarKind);
            MemorySegment foundKeys = arena.allocate(keys.length);
            MemorySegment foundScores = arena.allocate((long) scores.length * Double.BYTES);
            MemorySegment foundCounts = arena.allocate((long) counts.length * Long.BYTES);
            MemorySegment telemetry = statsBlock(arena, stats);
            MAXSIM.invokeExact(
                    MemorySegment.ofAddress(handle),
                    MemorySegment.ofAddress(executor),
                    view,
                    rowsPerQuery,
                    matchesSpan(arena, foundKeys, foundScores, foundCounts, wanted),
                    expansion(arena, expansionFactor),
                    MemorySegment.NULL,
                    MemorySegment.NULL,
                    telemetry,
                    error);
            reportStats(telemetry, stats);
            Library.raise(error);
            MemorySegment.copy(foundKeys, ValueLayout.JAVA_BYTE, 0, keys, 0, keys.length);
            MemorySegment.copy(foundScores, ValueLayout.JAVA_DOUBLE, 0, scores, 0, scores.length);
            MemorySegment.copy(foundCounts, ValueLayout.JAVA_LONG, 0, counts, 0, counts.length);
        } catch (Throwable failure) {
            throw Types.rethrow(failure);
        }
    }

    private static long nativeCount(long handle, byte[] key) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment error = Library.error(arena);
            MemorySegment identity = arena.allocateFrom(ValueLayout.JAVA_BYTE, key);
            long held = (long) COUNT.invokeExact(MemorySegment.ofAddress(handle), identity, error);
            Library.raise(error);
            return held;
        } catch (Throwable failure) {
            throw Types.rethrow(failure);
        }
    }

    private static long nativeGet(
            long handle,
            byte[] key,
            Object vectors,
            int storage,
            long byteOffset,
            long rows,
            long columns,
            long strideBytes,
            int scalarKind) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment error = Library.error(arena);
            MemorySegment identity = arena.allocateFrom(ValueLayout.JAVA_BYTE, key);
            MemorySegment span = MemorySegment.NULL;
            MemorySegment staged = MemorySegment.NULL;
            if (vectors != null) {
                staged = lend(arena, vectors, storage, byteOffset, rows * strideBytes);
                span = arena.allocate(Library.MATRIX_SPAN);
                span.set(Library.POINTER, 0, staged);
                span.set(Library.SIZE, 8, rows);
                span.set(Library.SIZE, 16, columns);
                span.set(Library.SIZE, 24, strideBytes);
                span.set(Library.ENUM, 32, scalarKind);
            }
            long held = (long) GET.invokeExact(MemorySegment.ofAddress(handle), identity, span, error);
            Library.raise(error);
            if (vectors != null) {
                restore(staged, vectors, storage, byteOffset, rows * strideBytes);
            }
            return held;
        } catch (Throwable failure) {
            throw Types.rethrow(failure);
        }
    }

    private static void nativeSave(long handle, long executor, byte[] path) {
        checkpoint(SAVE, handle, executor, path);
    }

    private static void nativeLoad(long handle, long executor, byte[] path) {
        checkpoint(LOAD, handle, executor, path);
    }

    /** The shared body of the two path-addressed checkpoint verbs, whose shapes are identical. */
    private static void checkpoint(MethodHandle verb, long handle, long executor, byte[] path) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment error = Library.error(arena);
            MemorySegment named = bytesView(arena, arena.allocateFrom(ValueLayout.JAVA_BYTE, path), path.length);
            verb.invokeExact(
                    MemorySegment.ofAddress(handle),
                    MemorySegment.ofAddress(executor),
                    named,
                    MemorySegment.NULL,
                    MemorySegment.NULL,
                    error);
            Library.raise(error);
        } catch (Throwable failure) {
            throw Types.rethrow(failure);
        }
    }

    private static void nativeSaveBuffer(
            long handle, long executor, Object buffer, int storage, long byteOffset, long length) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment error = Library.error(arena);
            MemorySegment staged = lend(arena, buffer, storage, byteOffset, length);
            SAVE_BUFFER.invokeExact(
                    MemorySegment.ofAddress(handle),
                    MemorySegment.ofAddress(executor),
                    bytesView(arena, staged, length),
                    MemorySegment.NULL,
                    MemorySegment.NULL,
                    error);
            Library.raise(error);
            restore(staged, buffer, storage, byteOffset, length);
        } catch (Throwable failure) {
            throw Types.rethrow(failure);
        }
    }

    private static void nativeLoadBuffer(
            long handle, long executor, Object buffer, int storage, long byteOffset, long length) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment error = Library.error(arena);
            LOAD_BUFFER.invokeExact(
                    MemorySegment.ofAddress(handle),
                    MemorySegment.ofAddress(executor),
                    bytesView(arena, lend(arena, buffer, storage, byteOffset, length), length),
                    MemorySegment.NULL,
                    MemorySegment.NULL,
                    error);
            Library.raise(error);
        } catch (Throwable failure) {
            throw Types.rethrow(failure);
        }
    }

    private static void nativeMetadata(byte[] path, long[] metadata) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment error = Library.error(arena);
            MemorySegment frozen = arena.allocate(Library.DENSE_METADATA);
            METADATA.invokeExact(
                    bytesView(arena, arena.allocateFrom(ValueLayout.JAVA_BYTE, path), path.length), frozen, error);
            Library.raise(error);
            liftMetadata(frozen, metadata);
        } catch (Throwable failure) {
            throw Types.rethrow(failure);
        }
    }

    private static void nativeMetadataBuffer(
            Object buffer, int storage, long byteOffset, long length, long[] metadata) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment error = Library.error(arena);
            MemorySegment frozen = arena.allocate(Library.DENSE_METADATA);
            METADATA_BUFFER.invokeExact(
                    bytesView(arena, lend(arena, buffer, storage, byteOffset, length), length), frozen, error);
            Library.raise(error);
            liftMetadata(frozen, metadata);
        } catch (Throwable failure) {
            throw Types.rethrow(failure);
        }
    }

    /** Lifts the four frozen fields into the flat array the caller reads them from. */
    private static void liftMetadata(MemorySegment frozen, long[] metadata) {
        metadata[0] = frozen.get(Library.ENUM, 0);
        metadata[1] = frozen.get(Library.ENUM, 4);
        metadata[2] = frozen.get(Library.SIZE, 8);
        metadata[3] = frozen.get(Library.SIZE, 16);
    }

    /**
     * Lowers the flat words into the C struct and lifts them straight back, which is the marshalling
     * round trip the options test exercises. It reaches no engine verb: the struct is the whole
     * subject, so writing and reading it back is the entire check.
     */
    private static void nativeReloadOptions(long[] input, long[] output) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment lowered = arena.allocate(Library.DENSE_OPTIONS);
            lowered.set(Library.ENUM, 0, (int) input[OptionsWords.METRIC]);
            lowered.set(Library.ENUM, 24, (int) input[OptionsWords.QUANTIZATION]);
            lowered.set(Library.SIZE, 32, input[OptionsWords.DIMENSIONS]);
            lowered.set(Library.SIZE, 40, input[OptionsWords.CONNECTIVITY]);
            lowered.set(Library.SIZE, 48, input[OptionsWords.VALUES_PER_KEY]);
            lowered.set(Library.ENUM, 56, (int) input[OptionsWords.RESIDENCY]);
            output[OptionsWords.METRIC] = lowered.get(Library.ENUM, 0);
            output[OptionsWords.QUANTIZATION] = lowered.get(Library.ENUM, 24);
            output[OptionsWords.DIMENSIONS] = lowered.get(Library.SIZE, 32);
            output[OptionsWords.CONNECTIVITY] = lowered.get(Library.SIZE, 40);
            output[OptionsWords.VALUES_PER_KEY] = lowered.get(Library.SIZE, 48);
            output[OptionsWords.RESIDENCY] = lowered.get(Library.ENUM, 56);
        }
    }
}
