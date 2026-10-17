package cloud.unum.usearch;

import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.StructLayout;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.util.function.Consumer;
import java.util.function.ToLongFunction;

import org.jspecify.annotations.Nullable;

import cloud.unum.usearch.DenseVerbs.Request;
import cloud.unum.usearch.Types.Arena;
import cloud.unum.usearch.Types.Call;
import cloud.unum.usearch.Types.Deferred;
import cloud.unum.usearch.Types.Executor;
import cloud.unum.usearch.Types.Handle;
import cloud.unum.usearch.Types.Key;
import cloud.unum.usearch.Types.Matches;
import cloud.unum.usearch.Types.Monolith;
import cloud.unum.usearch.Types.RowsView;
import cloud.unum.usearch.Types.Scalar;
import cloud.unum.usearch.Types.Status;
import cloud.unum.usearch.Types.Traversal;
import cloud.unum.usearch.Types.USearchException;
import cloud.unum.usearch.Types.Verb;

/**
 * A Vector Search Index over equidimensional dense representations, with the modality's own
 * options, metric, telemetry and per-call settings as nested types.
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
 * The receiver chooses how a call runs, and the verb stays the same: the index runs every verb on
 * the caller's thread, the {@link On} that {@link #on} makes blocks on an executor, and the
 * {@link DeferredOn} that {@link #deferred} makes submits there and answers a {@link Deferred}. The
 * filtered search and the progress handlers of the C interface stay unbound: they run on engine
 * threads, where every candidate would pay an upcall and a throwing handler could not unwind. Filter
 * the results of {@link #search} instead, or narrow the index.
 *
 * <p>
 * Every method here reports failure as an unchecked {@link USearchException} carrying the
 * {@link Status} the engine named, so the per-method tags below list only the statuses that method
 * distinctly produces. An {@link IllegalStateException} on a handle that was closed applies to all
 * of them and is not repeated.
 *
 * <pre>{@code
 * DenseOptions options = new DenseOptions().dimensions(2);
 * try (Executor pool = Executor.forRam(new int[0], 0, 0);
 *         Arena arena = new Arena(pool);
 *         DenseIndex index = DenseIndex.initRam(options, new int[0])) {
 *     index.reserve(2);
 *     index.insert(new Key[] {Key.of(1), Key.of(2)}, RowsView.of(new float[] {0, 1, 1, 0}, 2));
 *     Matches matches = index.on(pool).search(RowsView.of(new float[] {0, 1}, 2), 1, arena);
 *     System.out.println(matches.key(0, 0));
 * }
 * }</pre>
 */
public final class DenseIndex extends DenseVerbs implements AutoCloseable {

    /**
     * A distance kind supported out of the box, with SIMD kernels dispatched by the machine's
     * capabilities. All of them derive from a single inner-product kernel with pre-computed
     * magnitudes; Hamming and Jaccard are bit-level, over the packed sub-byte kinds. The C layer's
     * custom-callback escape hatch stays unexposed, since a caller-supplied metric would be scored
     * from engine threads that are not attached to this JVM.
     */
    public enum DenseMetric {

        /** A metric the library could not name. */
        UNKNOWN(0),
        /** The inner product. */
        INNER(1),
        /** The cosine distance. */
        ANGULAR(2),
        /** The euclidean distance. */
        EUCLIDEAN(3),
        /** The count of differing bits. */
        HAMMING(4),
        /** The bit-level Jaccard distance. */
        JACCARD(5);

        private final int code;

        DenseMetric(int code) {
            this.code = code;
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

        /** Renders the metric as its C enumerator's stem. Calls {@code usearch_dense_metric_name}. */
        @Override
        public String toString() {
            return Library.text(METRIC_NAME, code);
        }
    }

    /**
     * The geometry and metric an empty index freezes - the fields {@code usearch_dense_init_ram_t}
     * and {@code usearch_dense_init_vram_t} share.
     *
     * <p>
     * Every setter returns this instance, so an index is configured in one expression and created by
     * the lifecycle verb that takes it:
     *
     * <pre>{@code
     * try (DenseIndex index = DenseIndex.initRam(
     *         new DenseOptions().dimensions(768).metric(DenseMetric.ANGULAR).quantization(Scalar.I8),
     *         new int[0])) {
     *     index.reserve(1_000_000);
     * }
     * }</pre>
     */
    public static final class DenseOptions {

        private DenseMetric metric = DenseMetric.ANGULAR;
        private Scalar quantization = Scalar.BF16;
        private long dimensions = 0;
        private long connectivity = 0;
        private long valuesPerKey = 0;

        /** An angular BF16 index, with every optional field derived. */
        public DenseOptions() {}

        /** The distance kind used between vectors. */
        public DenseOptions metric(DenseMetric metric) {
            this.metric = metric;
            return this;
        }

        /** The scalar type vectors are stored and quantized into. */
        public DenseOptions quantization(Scalar quantization) {
            this.quantization = quantization;
            return this;
        }

        /**
         * The width of the indexed vectors.
         *
         * @throws USearchException INVALID_ARGUMENT on a negative width.
         */
        public DenseOptions dimensions(long dimensions) {
            this.dimensions = nonNegative(dimensions, "the dimensionality is negative");
            return this;
        }

        /**
         * The connections-per-node bound of the graph; zero derives.
         *
         * @throws USearchException INVALID_ARGUMENT on a negative bound.
         */
        public DenseOptions connectivity(long connectivity) {
            this.connectivity = nonNegative(connectivity, "the connectivity is negative");
            return this;
        }

        /**
         * The upper bound on entries sharing one key; zero derives to one - unique keys. No
         * checkpoint records it, so every load takes it as an argument of its own.
         *
         * @throws USearchException INVALID_ARGUMENT on a negative bound.
         */
        public DenseOptions valuesPerKey(long valuesPerKey) {
            this.valuesPerKey = nonNegative(valuesPerKey, "the entries per key is negative");
            return this;
        }

        /** The distance kind used between vectors. */
        public DenseMetric metric() {
            return metric;
        }

        /** The scalar type vectors are stored and quantized into. */
        public Scalar quantization() {
            return quantization;
        }

        /** The width of the indexed vectors. */
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

        /** Renders the options as the fields a reader would set. */
        @Override
        public String toString() {
            return "DenseOptions(dimensions=" + dimensions
                    + ", metric=" + metric
                    + ", quantization=" + quantization
                    + ", connectivity=" + connectivity
                    + ", valuesPerKey=" + valuesPerKey + ")";
        }

        /** Fills the fields both init structures share. */
        void fill(Call call) {
            call.enumeration("metric", metric.code());
            call.enumeration("quantization", quantization.code());
            call.size("dimensions", dimensions);
            call.size("connectivity", connectivity);
            call.size("values_per_key", valuesPerKey);
        }
    }

    /**
     * The geometry and metric a checkpoint froze - everything a reader learns without opening it, and
     * nothing a reader must choose for itself.
     *
     * <p>
     * A checkpoint records neither the placement nor the bound on entries sharing one key: the first
     * says where it was written rather than where a reader must keep it, and the second bounds a live
     * handle's mutable state rather than the serialized form. Both stay the caller's to choose in the
     * lifecycle verb a handle is loaded with.
     *
     * <p>
     * Mirrors {@code usearch_dense_metadata_t}.
     *
     * @param dimensions the scalars per stored vector.
     * @param metric the distance kind the checkpoint scores with.
     * @param quantization the scalar type its rows are stored and quantized into.
     * @param connectivity the connections-per-node bound the graph was built with.
     */
    public record DenseMetadata(long dimensions, DenseMetric metric, Scalar quantization, long connectivity) {

        /** Lifts a {@code usearch_dense_metadata_t}. */
        static DenseMetadata lift(MemorySegment metadata) {
            return new DenseMetadata(
                    metadata.get(Library.SIZE, Library.offset(METADATA, "dimensions")),
                    DenseMetric.of(metadata.get(Library.ENUM, Library.offset(METADATA, "metric"))),
                    Scalar.of(metadata.get(Library.ENUM, Library.offset(METADATA, "quantization"))),
                    metadata.get(Library.SIZE, Library.offset(METADATA, "connectivity")));
        }
    }

    /**
     * A live index's state. Each counter is read once, so under concurrent insertions they may come
     * from slightly different moments.
     *
     * <p>
     * Mirrors {@code usearch_dense_index_description_t}.
     *
     * @param metadata the frozen metric, quantization, dimensions and connectivity.
     * @param size the stored entries.
     * @param capacity the entries hosted without growing again.
     * @param memoryUsage the bytes the handle holds; traversal scratch is the executor's.
     * @param capabilities the codename of the kernels the index dispatched to on this machine.
     */
    public record DenseIndexDescription(
            DenseMetadata metadata, long size, long capacity, long memoryUsage, String capabilities) {

        /** Lifts a {@code usearch_dense_index_description_t}. */
        static DenseIndexDescription lift(MemorySegment description) {
            return new DenseIndexDescription(
                    DenseMetadata.lift(
                            description.asSlice(Library.offset(DESCRIPTION, "metadata"), METADATA.byteSize())),
                    description.get(Library.SIZE, Library.offset(DESCRIPTION, "size")),
                    description.get(Library.SIZE, Library.offset(DESCRIPTION, "capacity")),
                    description.get(Library.SIZE, Library.offset(DESCRIPTION, "memory_usage")),
                    Library.string(description.get(Library.POINTER, Library.offset(DESCRIPTION, "capabilities"))));
        }
    }

    /**
     * One reachability audit: the population it judged, and how much of it a search reaches.
     *
     * <p>
     * Mirrors {@code usearch_reachability_t}.
     *
     * @param audited the entries published when the audit started; compare {@code reachable} against
     *     this, never a later size.
     * @param reachable the entries a search reaches; the rest no search finds until a refinement
     *     re-links them.
     * @param sweeps the levels the sweep took to converge, a diameter proxy.
     */
    public record DenseReachability(long audited, long reachable, long sweeps) {

        /** Lifts a {@code usearch_reachability_t}. */
        static DenseReachability lift(MemorySegment reachability) {
            long[] fields = reachability.toArray(ValueLayout.JAVA_LONG);
            return new DenseReachability(fields[0], fields[1], fields[2]);
        }
    }

    /**
     * The online-softmax carry of one query row - Flash Attention's running maximum and partition,
     * which is what lets one attention call's output fold into another's over the same queries.
     *
     * <p>
     * Mirrors {@code usearch_dense_softmax_state_t}.
     *
     * @param maximumScore the largest logit seen so far - the recurrence's {@code m}.
     * @param partition the sum of exponentials taken against that maximum - the recurrence's
     *     {@code l}.
     */
    public record DenseSoftmaxState(float maximumScore, float partition) {

        /** Lifts a {@code usearch_dense_softmax_state_t}. */
        static DenseSoftmaxState lift(MemorySegment state) {
            return new DenseSoftmaxState(state.get(ValueLayout.JAVA_FLOAT, 0), state.get(ValueLayout.JAVA_FLOAT, 4));
        }
    }

    /**
     * The traversal telemetry one call reports, fully overwritten by every call it is passed to, so
     * one reading always belongs to one call and a sweep over many batches sums them itself.
     *
     * <p>
     * Mirrors {@code usearch_traversal_stats_t}.
     */
    public static final class DenseTraversalStats {

        private long visitedMembers;
        private long computedDistances;
        private long restarts;

        /** A reading no call has written yet. */
        public DenseTraversalStats() {}

        /** Graph members the call visited. */
        public long visitedMembers() {
            return visitedMembers;
        }

        /** Distances the call computed, diversity re-selections included. */
        public long computedDistances() {
            return computedDistances;
        }

        /** Starvation re-seedings - a graph-connectivity health signal rather than a cost counter. */
        public long restarts() {
            return restarts;
        }

        /** Overwrites the reading with what a call reported. */
        void read(MemorySegment reported) {
            long[] fields = reported.toArray(ValueLayout.JAVA_LONG);
            visitedMembers = fields[0];
            computedDistances = fields[1];
            restarts = fields[2];
        }

        /** Renders the three counters. */
        @Override
        public String toString() {
            return "DenseTraversalStats(visitedMembers=" + visitedMembers
                    + ", computedDistances=" + computedDistances
                    + ", restarts=" + restarts + ")";
        }
    }

    /**
     * Resolves the keys a walk scores to their value rows, which the index does not store: each key
     * names one row of {@code values}, as {@code row} answers it, and a row outside the matrix, a
     * thrown exception, or a matrix of another scalar or width than the call states leaves the key
     * without one, which refuses the call. The function runs on the executor's threads together.
     *
     * <p>
     * The values are borrowed once, here, and so is the upcall every call hands across: a native slab
     * is read in place and must stay open while the lookup is used, and a heap one is copied once.
     *
     * <p>
     * Mirrors {@code usearch_value_lookup_t}, whose scalar and columns are checked against the matrix.
     * It is a host function, so it serves RAM handles alone: a VRAM walk resolves rows through a
     * device function, which this binding does not build, so an attention into a VRAM arena refuses
     * it with DEVICE.
     */
    public static final class ValueLookup {

        private final RowsView values;
        private final ToLongFunction<Key> row;

        /** The value rows the upcall addresses: the caller's native slab, or the one copy of a heap one. */
        private final MemorySegment base;

        /** The upcall stub, freed once the lookup and every call lending it are unreachable. */
        private final MemorySegment upcall;

        /**
         * Borrows the values and builds the upcall once.
         *
         * @param values the value rows, in the index's quantization and as wide as its keys.
         * @param row the row of {@code values} a key's value lives in.
         */
        @SuppressWarnings("restricted")
        public ValueLookup(RowsView values, ToLongFunction<Key> row) {
            this.values = values;
            this.row = row;
            MemorySegment slab = values.segment().asSlice(0, values.spanBytes());
            this.base = slab.isNative()
                    ? slab
                    : java.lang.foreign.Arena.ofAuto()
                            .allocate(slab.byteSize(), 8)
                            .copyFrom(slab);
            // The stub roots its target until freed, so the target holds plain numbers rather than this lookup.
            Resolver resolver = new Resolver(
                    base.address(),
                    values.rows(),
                    values.strideBytes(),
                    values.scalar().code(),
                    values.columns(),
                    row);
            this.upcall = Linker.nativeLinker()
                    .upcallStub(RESOLVE.bindTo(resolver), LOOKUP, java.lang.foreign.Arena.ofAuto());
        }

        /** The value rows, in the index's quantization and as wide as its keys. */
        public RowsView values() {
            return values;
        }

        /** The row of {@link #values()} a key's value lives in. */
        public ToLongFunction<Key> row() {
            return row;
        }

        /** Hands the upcall across under the call's outcome, which latches what the function raises, until final. */
        void lend(Call call) {
            call.pointer("value_lookup", upcall);
            call.pointer("value_lookup_state", call.outcome);
            call.lend(this);
        }

        /** The geometry and function every call's upcall resolves keys with. */
        private record Resolver(
                long base, long rows, long strideBytes, int scalar, long columns, ToLongFunction<Key> row) {

            /** Writes one row pointer per key, null where the key resolves to none. */
            @SuppressWarnings({"restricted", "unused"})
            void resolve(
                    MemorySegment state,
                    MemorySegment keys,
                    long count,
                    int asked,
                    long width,
                    MemorySegment resolved) {
                MemorySegment named = keys.reinterpret(count * Key.BYTES);
                MemorySegment found = resolved.reinterpret(count * Library.POINTER.byteSize());
                boolean fits = asked == scalar && width == columns;
                for (long entry = 0; entry != count; ++entry) {
                    MemorySegment located = MemorySegment.NULL;
                    try {
                        long position = fits ? row.applyAsLong(Key.read(named, entry * Key.BYTES)) : -1;
                        if (position >= 0 && position < rows) {
                            located = MemorySegment.ofAddress(base + position * strideBytes);
                        }
                    } catch (Throwable failure) {
                        Call.latch(state, failure);
                    }
                    found.setAtIndex(Library.POINTER, entry, located);
                }
            }
        }
    }

    /**
     * The knobs of an insertion; see {@link DenseVerbs#insert(Key[], RowsView, DenseInsertConfig)}.
     * Mirrors {@code usearch_dense_insert_t}.
     */
    public static final class DenseInsertConfig {

        long expansionFactor;

        @Nullable
        DenseTraversalStats stats;

        /** Every knob derived. */
        public DenseInsertConfig() {}

        /**
         * The candidate-pool breadth the build walk explores; zero derives.
         *
         * @throws USearchException INVALID_ARGUMENT on a negative breadth, which no pool may have.
         */
        public DenseInsertConfig expansionFactor(long expansionFactor) {
            this.expansionFactor = DenseVerbs.nonNegative(expansionFactor);
            return this;
        }

        /** Where the traversal telemetry lands, overwritten once the call succeeded. */
        public DenseInsertConfig stats(DenseTraversalStats stats) {
            this.stats = stats;
            return this;
        }
    }

    /**
     * The knobs of a refinement; see {@link DenseVerbs#refine(DenseRefineConfig)}. Mirrors
     * {@code usearch_dense_refine_t}.
     */
    public static final class DenseRefineConfig {

        long expansionFactor;

        @Nullable
        DenseTraversalStats stats;

        /** Every knob derived. */
        public DenseRefineConfig() {}

        /**
         * The candidate-pool breadth the re-selection walk explores; zero derives.
         *
         * @throws USearchException INVALID_ARGUMENT on a negative breadth, which no pool may have.
         */
        public DenseRefineConfig expansionFactor(long expansionFactor) {
            this.expansionFactor = DenseVerbs.nonNegative(expansionFactor);
            return this;
        }

        /** Where the traversal telemetry lands, overwritten once the call succeeded. */
        public DenseRefineConfig stats(DenseTraversalStats stats) {
            this.stats = stats;
            return this;
        }
    }

    /**
     * How many matches a search keeps and the knobs of its walk; see
     * {@link DenseVerbs#search(RowsView, DenseSearchConfig, Arena)}. Mirrors {@code usearch_dense_search_t}.
     */
    public static final class DenseSearchConfig {

        final long wanted;
        long expansionFactor;
        Traversal traversal = Traversal.WALK;

        @Nullable
        DenseTraversalStats stats;

        /**
         * Up to {@code wanted} matches per query - the "k" in kANN.
         *
         * @throws USearchException INVALID_ARGUMENT on a negative count.
         */
        public DenseSearchConfig(long wanted) {
            this.wanted = nonNegative(wanted, "the wanted count is negative");
        }

        /**
         * The candidate-pool breadth the walk explores; zero derives.
         *
         * @throws USearchException INVALID_ARGUMENT on a negative breadth, which no pool may have.
         */
        public DenseSearchConfig expansionFactor(long expansionFactor) {
            this.expansionFactor = DenseVerbs.nonNegative(expansionFactor);
            return this;
        }

        /**
         * A walk, the default, or an exact scan of every row on RAM and NVMe handles; a VRAM handle
         * refuses a scan as UNSUPPORTED.
         */
        public DenseSearchConfig traversal(Traversal traversal) {
            this.traversal = traversal;
            return this;
        }

        /** Where the traversal telemetry lands, overwritten once the call succeeded. */
        public DenseSearchConfig stats(DenseTraversalStats stats) {
            this.stats = stats;
            return this;
        }
    }

    /**
     * How many documents a late-interaction search keeps and the knobs of its walk; see
     * {@link DenseVerbs#maxsim(RowsView, long, DenseMaxsimConfig, Arena)}. Mirrors {@code usearch_dense_maxsim_t}.
     */
    public static final class DenseMaxsimConfig {

        final long wanted;
        long expansionFactor;

        @Nullable
        DenseTraversalStats stats;

        /**
         * Up to {@code wanted} documents per query - the "k" over keys, not rows.
         *
         * @throws USearchException INVALID_ARGUMENT on a negative count.
         */
        public DenseMaxsimConfig(long wanted) {
            this.wanted = nonNegative(wanted, "the wanted count is negative");
        }

        /**
         * The candidate-pool breadth the walk explores; zero derives.
         *
         * @throws USearchException INVALID_ARGUMENT on a negative breadth, which no pool may have.
         */
        public DenseMaxsimConfig expansionFactor(long expansionFactor) {
            this.expansionFactor = DenseVerbs.nonNegative(expansionFactor);
            return this;
        }

        /** Where the traversal telemetry lands, overwritten once the call succeeded. */
        public DenseMaxsimConfig stats(DenseTraversalStats stats) {
            this.stats = stats;
            return this;
        }
    }

    /**
     * How many pairs a sparse product keeps and the knobs of its walk; see
     * {@link DenseVerbs#multiply(RowsView, DenseMultiplyConfig, Arena)}. Mirrors {@code usearch_dense_multiply_t}.
     */
    public static final class DenseMultiplyConfig {

        long wanted;
        long expansionFactor;

        @Nullable
        DenseTraversalStats stats;

        /** Every pair kept, with every knob derived. */
        public DenseMultiplyConfig() {}

        /**
         * The bound on pairs kept per row, the first ones its walk scored; zero keeps all.
         *
         * @throws USearchException INVALID_ARGUMENT on a negative count.
         */
        public DenseMultiplyConfig wanted(long wanted) {
            this.wanted = nonNegative(wanted, "the wanted count is negative");
            return this;
        }

        /**
         * The candidate-pool breadth, which widens the emitted pattern; zero derives.
         *
         * @throws USearchException INVALID_ARGUMENT on a negative breadth, which no pool may have.
         */
        public DenseMultiplyConfig expansionFactor(long expansionFactor) {
            this.expansionFactor = DenseVerbs.nonNegative(expansionFactor);
            return this;
        }

        /** Where the traversal telemetry lands, overwritten once the call succeeded. */
        public DenseMultiplyConfig stats(DenseTraversalStats stats) {
            this.stats = stats;
            return this;
        }
    }

    /**
     * The knobs of an attention; see {@link DenseVerbs#attend(RowsView, ValueLookup, DenseAttendConfig, Arena)}.
     * Mirrors {@code usearch_dense_attend_t}.
     */
    public static final class DenseAttendConfig {

        long headCount;
        Scalar outputsScalar = Scalar.UNKNOWN;
        DenseSoftmaxState @Nullable [] carry;
        long expansionFactor;
        float scoreScale;

        @Nullable
        DenseTraversalStats stats;

        /** Every knob derived, the outputs in the queries' own scalar. */
        public DenseAttendConfig() {}

        /**
         * The query heads sharing the index's key-value head, and with it one walk per token: row
         * {@code t * headCount + h} is head {@code h} of token {@code t}, so it must divide the rows;
         * zero means one.
         *
         * @throws USearchException INVALID_ARGUMENT on a negative count.
         */
        public DenseAttendConfig headCount(long headCount) {
            this.headCount = nonNegative(headCount, "the head count is negative");
            return this;
        }

        /** The scalar type the attended rows land as; UNKNOWN, the default, takes the queries' own. */
        public DenseAttendConfig outputsScalar(Scalar outputsScalar) {
            this.outputsScalar = outputsScalar;
            return this;
        }

        /** Where the online-softmax state lands, one per query row, once the call succeeded. */
        public DenseAttendConfig carry(DenseSoftmaxState[] carry) {
            this.carry = carry;
            return this;
        }

        /**
         * The candidate-pool breadth; zero derives. A wider pool walks further, folding more of the tail.
         *
         * @throws USearchException INVALID_ARGUMENT on a negative breadth, which no pool may have.
         */
        public DenseAttendConfig expansionFactor(long expansionFactor) {
            this.expansionFactor = DenseVerbs.nonNegative(expansionFactor);
            return this;
        }

        /** The logit scale, conventionally the inverse square root of the head dimensionality; zero derives it. */
        public DenseAttendConfig scoreScale(float scoreScale) {
            this.scoreScale = scoreScale;
            return this;
        }

        /** Where the traversal telemetry lands, overwritten once the call succeeded. */
        public DenseAttendConfig stats(DenseTraversalStats stats) {
            this.stats = stats;
            return this;
        }
    }

    /**
     * Rows a call wrote into its arena - the entries {@link DenseVerbs#get} found under a key, the
     * outputs of {@link DenseVerbs#attend}, or the rows of a {@link DenseBatch} - at the width, pitch
     * and scalar the library reported. Valid until the arena's next call or its close.
     */
    public static final class DenseVectors {

        private final Arena arena;
        private final MemorySegment data;
        private final long rows;
        private final long columns;
        private final long strideBytes;
        private final Scalar scalar;

        private DenseVectors(
                Arena arena, MemorySegment data, long rows, long columns, long strideBytes, Scalar scalar) {
            this.arena = arena;
            this.data = data;
            this.rows = rows;
            this.columns = columns;
            this.strideBytes = strideBytes;
            this.scalar = scalar;
        }

        /** Lifts the {@code usearch_rows_view_t} of {@code rows} rows a call wrote into its arena. */
        static DenseVectors lift(Call call, MemorySegment view, long rows) {
            long stride = view.get(Library.SIZE, Library.offset(Library.ROWS_VIEW, "stride"));
            return new DenseVectors(
                    call.arena(),
                    call.reach(view.get(Library.POINTER, 0), Types.Extents.product(rows, stride, "the rows")),
                    rows,
                    view.get(Library.SIZE, Library.offset(Library.ROWS_VIEW, "columns")),
                    stride,
                    Scalar.of(view.get(Library.ENUM, Library.offset(Library.ROWS_VIEW, "scalar"))));
        }

        /** How many rows the call wrote; zero when a key holds none. */
        public long rows() {
            return rows;
        }

        /** The values in each row, as the library reported them. */
        public long columns() {
            return columns;
        }

        /** The byte pitch between consecutive rows, as the library packed them. */
        public long strideBytes() {
            return strideBytes;
        }

        /** The scalar type the rows were exported as. */
        public Scalar scalar() {
            return scalar;
        }

        /**
         * The rows in place, in native byte order.
         *
         * @throws USearchException DEVICE when the arena holds device memory.
         */
        public MemorySegment segment() {
            arena.host();
            return data;
        }
    }

    /**
     * The ragged sparse product {@link DenseVerbs#multiply} wrote into its arena: per left row, every
     * pair its walk scored, in traversal order. Valid until the arena's next call or its close.
     */
    public static final class DenseProducts {

        private final Arena arena;
        private final MemorySegment offsets;
        private final int queries;
        private final MemorySegment keys;
        private final MemorySegment scores;
        private final long required;

        private DenseProducts(
                Arena arena,
                MemorySegment offsets,
                int queries,
                MemorySegment keys,
                MemorySegment scores,
                long required) {
            this.arena = arena;
            this.offsets = offsets;
            this.queries = queries;
            this.keys = keys;
            this.scores = scores;
            this.required = required;
        }

        /** Lifts the {@code usearch_matches_tape_view_t} a call over {@code queries} rows wrote into its arena. */
        static DenseProducts lift(Call call, MemorySegment view, int queries, long pairs, long required) {
            StructLayout tape = Library.MATCHES_TAPE_VIEW;
            return new DenseProducts(
                    call.arena(),
                    call.reach(view.get(Library.POINTER, Library.offset(tape, "offsets")), (queries + 1L) * Long.BYTES),
                    queries,
                    call.reach(view.get(Library.POINTER, Library.offset(tape, "keys")), pairs * Key.BYTES),
                    call.reach(view.get(Library.POINTER, Library.offset(tape, "scores")), pairs * Double.BYTES),
                    required);
        }

        /** Left rows the call multiplied. */
        public int queries() {
            return queries;
        }

        /** Pairs the walks scored across every row, before the wanted bound cut any. */
        public long required() {
            return required;
        }

        /**
         * How many pairs one left row holds.
         *
         * @throws USearchException SHAPE_MISMATCH when the row is outside the batch, and DEVICE when
         *     the arena holds device memory.
         */
        public long count(int row) {
            arena.host();
            if (row < 0 || row >= queries) {
                throw new USearchException(Status.SHAPE_MISMATCH, "the row is outside the batch");
            }
            return offsets.getAtIndex(ValueLayout.JAVA_LONG, row + 1) - offsets.getAtIndex(ValueLayout.JAVA_LONG, row);
        }

        /**
         * The key naming the indexed row of one pair.
         *
         * @throws USearchException SHAPE_MISMATCH when the row is outside the batch, or the rank past
         *     its pairs.
         */
        public Key key(int row, long rank) {
            return Key.read(keys, pairOf(row, rank) * Key.BYTES);
        }

        /**
         * The product of one pair.
         *
         * @throws USearchException SHAPE_MISMATCH when the row is outside the batch, or the rank past
         *     its pairs.
         */
        public double score(int row, long rank) {
            return scores.getAtIndex(ValueLayout.JAVA_DOUBLE, pairOf(row, rank));
        }

        private long pairOf(int row, long rank) {
            long pairs = count(row);
            if (rank < 0 || rank >= pairs) {
                throw new USearchException(Status.SHAPE_MISMATCH, "the row holds " + pairs + " pairs");
            }
            return offsets.getAtIndex(ValueLayout.JAVA_LONG, row) + rank;
        }
    }

    /**
     * One pull of a {@link DenseCursor}, written into its arena: the keys of the entries passed, and
     * the keys each links to when asked for. Valid until the arena's next call or its close.
     */
    public static final class DenseBatch {

        private final Arena arena;
        private final MemorySegment keys;
        private final long keysStride;
        private final long written;
        private final @Nullable MemorySegment linkOffsets;
        private final MemorySegment linkKeys;

        private DenseBatch(
                Arena arena,
                MemorySegment keys,
                long keysStride,
                long written,
                @Nullable MemorySegment linkOffsets,
                MemorySegment linkKeys) {
            this.arena = arena;
            this.keys = keys;
            this.keysStride = keysStride;
            this.written = written;
            this.linkOffsets = linkOffsets;
            this.linkKeys = linkKeys;
        }

        /** How many entries the pull wrote; zero once the walk is exhausted. */
        public long written() {
            return written;
        }

        /**
         * The key of one entry, in scan order.
         *
         * @throws USearchException SHAPE_MISMATCH when the entry is past the batch, and DEVICE when the
         *     arena holds device memory.
         */
        public Key key(long entry) {
            arena.host();
            if (entry < 0 || entry >= written) {
                throw new USearchException(Status.SHAPE_MISMATCH, "the batch holds " + written + " entries");
            }
            return Key.read(keys, entry * keysStride);
        }

        /**
         * How many keys one entry links to.
         *
         * @throws USearchException INVALID_ARGUMENT when the pull exported no links, SHAPE_MISMATCH when
         *     the entry is past the batch, and DEVICE when the arena holds device memory.
         */
        public long linkCount(long entry) {
            MemorySegment offsets = linked(entry);
            return offsets.getAtIndex(ValueLayout.JAVA_LONG, entry + 1) - offsets.getAtIndex(ValueLayout.JAVA_LONG, entry);
        }

        /**
         * One key an entry links to, as the graph stored it when the pull ran.
         *
         * @throws USearchException the statuses {@link #linkCount(long)} names, and SHAPE_MISMATCH when
         *     the rank is past the entry's links.
         */
        public Key link(long entry, long rank) {
            long begin = linked(entry).getAtIndex(ValueLayout.JAVA_LONG, entry);
            if (rank < 0 || rank >= linkCount(entry)) {
                throw new USearchException(Status.SHAPE_MISMATCH, "the entry links to " + linkCount(entry) + " keys");
            }
            return Key.read(linkKeys, (begin + rank) * Key.BYTES);
        }

        private MemorySegment linked(long entry) {
            arena.host();
            if (linkOffsets == null) {
                throw new USearchException(Status.INVALID_ARGUMENT, "the pull exported no links");
            }
            if (entry < 0 || entry >= written) {
                throw new USearchException(Status.SHAPE_MISMATCH, "the batch holds " + written + " entries");
            }
            return linkOffsets;
        }
    }

    /**
     * A sequential walk over the index contents, pulled blocking on the executor of the receiver
     * that opened it - export, migration, and visualization feed off it. It serves one thread at a
     * time, and an overlapping pull reports CONTENDED.
     */
    public static final class DenseCursor implements AutoCloseable {

        private final Handle handle;
        private final DenseIndex index;
        private final @Nullable Executor executor;

        DenseCursor(MemorySegment opened, DenseIndex index, @Nullable Executor executor) {
            this.handle = new Handle(opened, CURSOR_FREE);
            this.index = index;
            this.executor = executor;
        }

        /**
         * Copies the keys of up to {@code wanted} next entries into {@code arena}. Calls
         * {@code usearch_dense_cursor_next}.
         *
         * @throws USearchException INVALID_ARGUMENT on a negative count, and CONTENDED when the arena
         *     serves another call.
         */
        public DenseBatch next(long wanted, Arena arena) {
            return pull(wanted, false, arena);
        }

        /**
         * Copies the keys of up to {@code wanted} next entries into {@code arena}, each with the keys it
         * links to. Calls {@code usearch_dense_cursor_next}.
         *
         * @throws USearchException INVALID_ARGUMENT on a negative count, and CONTENDED when the arena
         *     serves another call.
         */
        public DenseBatch nextLinks(long wanted, Arena arena) {
            return pull(wanted, true, arena);
        }

        private DenseBatch pull(long wanted, boolean linked, Arena arena) {
            long bound = nonNegative(wanted, "the wanted count is negative");
            return Call.block(CURSOR_NEXT, index.handle, executor, call -> {
                call.lease("cursor", handle);
                MemorySegment keys = call.output("keys", Library.KEYS_VIEW);
                MemorySegment tape = linked ? call.output("links", Library.KEYS_TAPE_VIEW) : MemorySegment.NULL;
                MemorySegment written = call.output("written", Library.SIZE);
                call.arena("arena", arena);
                call.size("wanted", bound);
                return () -> {
                    long count = written.get(Library.SIZE, 0);
                    long stride = keys.get(Library.SIZE, Library.POINTER.byteSize());
                    MemorySegment offsets = null;
                    MemorySegment linkKeys = MemorySegment.NULL;
                    if (linked) {
                        offsets = call.reach(
                                tape.get(Library.POINTER, Library.offset(Library.KEYS_TAPE_VIEW, "offsets")),
                                (count + 1) * Long.BYTES);
                        long total = count == 0 ? 0 : offsets.getAtIndex(ValueLayout.JAVA_LONG, count);
                        linkKeys = call.reach(
                                tape.get(Library.POINTER, Library.offset(Library.KEYS_TAPE_VIEW, "keys")),
                                Types.Extents.product(total, Key.BYTES, "the links"));
                    }
                    return new DenseBatch(
                            call.arena(),
                            call.reach(keys.get(Library.POINTER, 0), Types.Extents.product(count, stride, "the keys")),
                            stride,
                            count,
                            offsets,
                            linkKeys);
                };
            });
        }

        /**
         * Releases the cursor. It is idempotent. Calls {@code usearch_dense_cursor_free}.
         *
         * @throws IllegalStateException while another thread's pull still runs on it.
         */
        @Override
        public void close() {
            handle.close();
        }
    }

    /**
     * A resumable search over one query, pulled blocking on the executor of the receiver that opened
     * it: the stream keeps the walk's frontier and visited set between pulls and yields matches
     * lazily, holding state rather than a thread.
     */
    public static final class DenseSearchStream implements AutoCloseable {

        private final Handle handle;
        private final DenseIndex index;
        private final @Nullable Executor executor;

        DenseSearchStream(MemorySegment opened, DenseIndex index, @Nullable Executor executor) {
            this.handle = new Handle(opened, SEARCH_STREAM_FREE);
            this.index = index;
            this.executor = executor;
        }

        /**
         * Pulls up to {@code wanted} next-nearest matches into {@code arena}, as one query row of
         * {@link Matches}; a row of none means the walk is drained. Calls
         * {@code usearch_dense_search_stream_next}.
         *
         * @throws USearchException INVALID_ARGUMENT on a negative count, and CONTENDED when the arena
         *     serves another call.
         */
        public Matches next(long wanted, Arena arena) {
            long bound = nonNegative(wanted, "the wanted count is negative");
            return Call.block(SEARCH_STREAM_NEXT, index.handle, executor, call -> {
                call.lease("stream", handle);
                MemorySegment matches = call.output("matches", Library.MATCHES_VIEW);
                call.arena("arena", arena);
                call.size("wanted", bound);
                return () -> Matches.lift(call, matches, 1);
            });
        }

        /**
         * Releases the stream and the walk state it keeps. It is idempotent. Calls
         * {@code usearch_dense_search_stream_free}.
         *
         * @throws IllegalStateException while another thread's pull still runs on it.
         */
        @Override
        public void close() {
            handle.close();
        }
    }

    /** The receiver whose verbs block on an executor, after the calls submitted there before them. */
    public static final class On extends DenseVerbs {

        private final DenseIndex index;
        private final Executor executor;

        On(DenseIndex index, Executor executor) {
            this.index = index;
            this.executor = executor;
        }

        @Override
        DenseIndex index() {
            return index;
        }

        @Override
        Executor executor() {
            return executor;
        }
    }

    /**
     * The receiver whose verbs submit their calls to an executor, each answering a {@link Deferred}
     * that owns what the call reads until final, and whose join answers the view the blocking verb
     * would, over the same borrowed arena. Its verbs are those of {@link DenseVerbs}, which each
     * documents, less the scans that pull one batch at a time.
     */
    public static final class DeferredOn {

        private final DenseIndex index;
        private final Executor executor;

        DeferredOn(DenseIndex index, Executor executor) {
            this.index = index;
            this.executor = executor;
        }

        private <Result> Deferred<Result> submit(Request<Result> request) {
            return Call.submit(request.verb(), index.handle, executor, request.fill());
        }

        /** Submits {@link DenseVerbs#reserve}. Calls {@code usearch_dense_reserve}. */
        public Deferred<Void> reserve(long capacity) {
            return submit(DenseVerbs.reserving(capacity));
        }

        /** Submits {@link DenseVerbs#clear}. Calls {@code usearch_dense_clear}. */
        public Deferred<Void> clear() {
            return submit(DenseVerbs.clearing());
        }

        /** Submits {@link DenseVerbs#describe}. Calls {@code usearch_dense_describe}. */
        public Deferred<DenseIndexDescription> describe() {
            return submit(DenseVerbs.describing());
        }

        /** Submits {@link DenseVerbs#count}. Calls {@code usearch_dense_count}. */
        public Deferred<Long> count(Key key) {
            return submit(DenseVerbs.counting(key));
        }

        /** Submits {@link DenseVerbs#get}. Calls {@code usearch_dense_get}. */
        public Deferred<DenseVectors> get(Key key, Scalar outputsScalar, Arena arena) {
            return submit(DenseVerbs.getting(key, outputsScalar, arena));
        }

        /** Submits {@link DenseVerbs#insert(Key[], RowsView)}. Calls {@code usearch_dense_insert}. */
        public Deferred<Void> insert(Key[] keys, RowsView vectors) {
            return submit(DenseVerbs.inserting(keys, vectors, new DenseInsertConfig()));
        }

        /** Submits {@link DenseVerbs#insert(Key[], RowsView, DenseInsertConfig)}. Calls {@code usearch_dense_insert}. */
        public Deferred<Void> insert(Key[] keys, RowsView vectors, DenseInsertConfig config) {
            return submit(DenseVerbs.inserting(keys, vectors, config));
        }

        /** Submits {@link DenseVerbs#refine()}. Calls {@code usearch_dense_refine}. */
        public Deferred<Long> refine() {
            return submit(DenseVerbs.refining(new DenseRefineConfig()));
        }

        /** Submits {@link DenseVerbs#refine(DenseRefineConfig)}. Calls {@code usearch_dense_refine}. */
        public Deferred<Long> refine(DenseRefineConfig config) {
            return submit(DenseVerbs.refining(config));
        }

        /** Submits {@link DenseVerbs#health}. Calls {@code usearch_dense_health}. */
        public Deferred<DenseReachability> health() {
            return submit(DenseVerbs.auditing());
        }

        /** Submits {@link DenseVerbs#search(RowsView, long, Arena)}. Calls {@code usearch_dense_search}. */
        public Deferred<Matches> search(RowsView queries, long wanted, Arena arena) {
            return submit(DenseVerbs.searching(queries, new DenseSearchConfig(wanted), arena));
        }

        /**
         * Submits {@link DenseVerbs#search(RowsView, DenseSearchConfig, Arena)}. Calls
         * {@code usearch_dense_search}.
         */
        public Deferred<Matches> search(RowsView queries, DenseSearchConfig config, Arena arena) {
            return submit(DenseVerbs.searching(queries, config, arena));
        }

        /** Submits {@link DenseVerbs#maxsim(RowsView, long, long, Arena)}. Calls {@code usearch_dense_maxsim}. */
        public Deferred<Matches> maxsim(RowsView queries, long rowsPerQuery, long wanted, Arena arena) {
            return submit(DenseVerbs.scoring(queries, rowsPerQuery, new DenseMaxsimConfig(wanted), arena));
        }

        /**
         * Submits {@link DenseVerbs#maxsim(RowsView, long, DenseMaxsimConfig, Arena)}. Calls
         * {@code usearch_dense_maxsim}.
         */
        public Deferred<Matches> maxsim(RowsView queries, long rowsPerQuery, DenseMaxsimConfig config, Arena arena) {
            return submit(DenseVerbs.scoring(queries, rowsPerQuery, config, arena));
        }

        /** Submits {@link DenseVerbs#multiply(RowsView, Arena)}. Calls {@code usearch_dense_multiply}. */
        public Deferred<DenseProducts> multiply(RowsView left, Arena arena) {
            return submit(DenseVerbs.multiplying(left, new DenseMultiplyConfig(), arena));
        }

        /**
         * Submits {@link DenseVerbs#multiply(RowsView, DenseMultiplyConfig, Arena)}. Calls
         * {@code usearch_dense_multiply}.
         */
        public Deferred<DenseProducts> multiply(RowsView left, DenseMultiplyConfig config, Arena arena) {
            return submit(DenseVerbs.multiplying(left, config, arena));
        }

        /** Submits {@link DenseVerbs#attend(RowsView, ValueLookup, Arena)}. Calls {@code usearch_dense_attend}. */
        public Deferred<DenseVectors> attend(RowsView queries, ValueLookup valueLookup, Arena arena) {
            return submit(DenseVerbs.attending(queries, valueLookup, new DenseAttendConfig(), arena));
        }

        /**
         * Submits {@link DenseVerbs#attend(RowsView, ValueLookup, DenseAttendConfig, Arena)}. Calls
         * {@code usearch_dense_attend}.
         */
        public Deferred<DenseVectors> attend(
                RowsView queries, ValueLookup valueLookup, DenseAttendConfig config, Arena arena) {
            return submit(DenseVerbs.attending(queries, valueLookup, config, arena));
        }

        /** Submits {@link DenseVerbs#absorb}, surrendering the donors at once. Calls {@code usearch_dense_absorb}. */
        public Deferred<Void> absorb(DenseIndex... donors) {
            return submit(DenseVerbs.absorbing(donors));
        }

        /** Submits {@link DenseVerbs#save}. Calls {@code usearch_dense_save}. */
        public Deferred<Void> save(String path) {
            return submit(DenseVerbs.saving(path));
        }

        /** Submits {@link DenseVerbs#saveBuffer}. Calls {@code usearch_dense_save_buffer}. */
        public Deferred<Monolith> saveBuffer(Arena arena) {
            return submit(DenseVerbs.serializing(arena));
        }
    }

    /** Replaced once final by an absorb that surrendered this index as a donor and left it live. */
    volatile Handle handle;

    private DenseIndex(MemorySegment created) {
        handle = new Handle(created, FREE);
    }

    @Override
    DenseIndex index() {
        return this;
    }

    @Override
    @Nullable
    Executor executor() {
        return null;
    }

    /**
     * The receiver whose verbs block on {@code executor}, after the calls submitted there before
     * them. Binding-only.
     */
    public On on(Executor executor) {
        return new On(this, executor);
    }

    /** The receiver whose verbs submit to {@code executor}, each answering a {@link Deferred}. Binding-only. */
    public DeferredOn deferred(Executor executor) {
        return new DeferredOn(this, executor);
    }

    /**
     * Creates an empty, mutable index in host memory, one shard per NUMA node named. Calls
     * {@code usearch_dense_init_ram}.
     *
     * @param memoryDomains NUMA node ids to place shards on; an empty list derives every node.
     * @throws USearchException UNKNOWN_KIND on a metric this build cannot name.
     */
    public static DenseIndex initRam(DenseOptions options, int[] memoryDomains) {
        return create(INIT_RAM, null, call -> {
            options.fill(call);
            call.ordinals("ram", memoryDomains);
        });
    }

    /**
     * Creates an empty, mutable index in device memory, one shard per device named. Calls
     * {@code usearch_dense_init_vram}.
     *
     * @param devices CUDA device ordinals to place shards on; an empty list derives one free device.
     * @throws USearchException UNKNOWN_KIND on a metric this build cannot name, and DEVICE when no
     *     device answers.
     */
    public static DenseIndex initVram(DenseOptions options, int[] devices) {
        return create(INIT_VRAM, null, call -> {
            options.fill(call);
            call.ordinals("vram", devices);
        });
    }

    /**
     * Copies a checkpoint into a new index in host memory on the caller's thread, over every memory
     * domain, with unique keys.
     *
     * @throws USearchException the statuses {@link #loadRam(String, Executor, int[], long)} names.
     */
    public static DenseIndex loadRam(String path) {
        return loadRam(path, null, new int[0], 0);
    }

    /**
     * Copies a checkpoint into a new, mutable index in host memory, shard by shard. The geometry and
     * metric come from the checkpoint's own headers, and part {@code p} lands whole on shard
     * {@code p} modulo the shard count. Calls {@code usearch_dense_load_ram}.
     *
     * @param executor the executor the read fans out over, or {@code null} for the caller's thread.
     * @param memoryDomains NUMA node ids to place the parts on; an empty list derives every node.
     * @param valuesPerKey the upper bound on entries sharing one key, which no checkpoint records;
     *     zero derives to one.
     * @throws USearchException MALFORMED on a checkpoint that ends early, and IO when the file cannot
     *     be read.
     */
    public static DenseIndex loadRam(String path, @Nullable Executor executor, int[] memoryDomains, long valuesPerKey) {
        return create(LOAD_RAM, executor, call -> {
            call.path("path", path);
            call.ordinals("ram", memoryDomains);
            call.size("values_per_key", nonNegative(valuesPerKey, "the entries per key is negative"));
        });
    }

    /**
     * Uploads a checkpoint into a new index in device memory on the default stream of one free
     * device, with unique keys.
     *
     * @throws USearchException the statuses {@link #loadVram(String, Executor, int[], long)} names.
     */
    public static DenseIndex loadVram(String path) {
        return loadVram(path, null, new int[0], 0);
    }

    /**
     * Copies a checkpoint into a new, mutable index in device memory, with the placement rules of
     * {@link #loadRam(String, Executor, int[], long)}. Calls {@code usearch_dense_load_vram}.
     *
     * @param executor the executor the upload runs on, or {@code null} for each device's default
     *     stream.
     * @param devices CUDA device ordinals to place the parts on; an empty list derives one free device.
     * @param valuesPerKey the upper bound on entries sharing one key; zero derives to one.
     * @throws USearchException DEVICE when no device answers, and the statuses a RAM load names.
     */
    public static DenseIndex loadVram(String path, @Nullable Executor executor, int[] devices, long valuesPerKey) {
        return create(LOAD_VRAM, executor, call -> {
            call.path("path", path);
            call.ordinals("vram", devices);
            call.size("values_per_key", nonNegative(valuesPerKey, "the entries per key is negative"));
        });
    }

    /**
     * Copies a monolith from a buffer into a new index in host memory on the caller's thread, with
     * unique keys.
     *
     * @throws USearchException the statuses {@link #loadRamBuffer(MemorySegment, Executor, int[], long)}
     *     names.
     */
    public static DenseIndex loadRamBuffer(MemorySegment buffer) {
        return loadRamBuffer(buffer, null, new int[0], 0);
    }

    /**
     * Copies a monolith from a buffer into a new, mutable index in host memory, as
     * {@link #loadRam(String, Executor, int[], long)} reads a path; the monolith lands whole on the
     * first shard. A native buffer is read in place, a heap one is copied first. Calls
     * {@code usearch_dense_load_ram_buffer}.
     *
     * @throws USearchException MALFORMED on a monolith that ends early.
     */
    public static DenseIndex loadRamBuffer(
            MemorySegment buffer, @Nullable Executor executor, int[] memoryDomains, long valuesPerKey) {
        return create(LOAD_RAM_BUFFER, executor, call -> {
            call.bytes("buffer", buffer);
            call.ordinals("ram", memoryDomains);
            call.size("values_per_key", nonNegative(valuesPerKey, "the entries per key is negative"));
        });
    }

    /**
     * Uploads a monolith from a buffer into a new index in device memory on the default stream of one
     * free device, with unique keys.
     *
     * @throws USearchException the statuses
     *     {@link #loadVramBuffer(MemorySegment, Executor, int[], long)} names.
     */
    public static DenseIndex loadVramBuffer(MemorySegment buffer) {
        return loadVramBuffer(buffer, null, new int[0], 0);
    }

    /**
     * Copies a monolith from a buffer into a new, mutable index in device memory, as
     * {@link #loadRamBuffer(MemorySegment, Executor, int[], long)} does in host memory. Calls
     * {@code usearch_dense_load_vram_buffer}.
     *
     * @throws USearchException DEVICE when no device answers, and MALFORMED on a monolith that ends
     *     early.
     */
    public static DenseIndex loadVramBuffer(
            MemorySegment buffer, @Nullable Executor executor, int[] devices, long valuesPerKey) {
        return create(LOAD_VRAM_BUFFER, executor, call -> {
            call.bytes("buffer", buffer);
            call.ordinals("vram", devices);
            call.size("values_per_key", nonNegative(valuesPerKey, "the entries per key is negative"));
        });
    }

    /**
     * Serves a checkpoint in place through a memory mapping, as a read-only index whose mutating
     * verbs report READ_ONLY. Nothing is copied, and the files must stay unmodified until the index
     * is closed. Calls {@code usearch_dense_view_ram}.
     *
     * @throws USearchException MALFORMED on a checkpoint that ends early, and IO when the file cannot
     *     be mapped.
     */
    public static DenseIndex viewRam(String path) {
        return create(VIEW_RAM, null, call -> call.path("path", path));
    }

    /**
     * Serves a checkpoint in place from NVMe drives on the caller's thread, each part where its
     * manifest points.
     *
     * @throws USearchException the statuses {@link #viewNvme(String, Executor, String[])} names.
     */
    public static DenseIndex viewNvme(String path) {
        return viewNvme(path, null, new String[0]);
    }

    /**
     * Serves a checkpoint in place from NVMe drives, read through the executor's submission rings, as
     * a read-only index. Calls {@code usearch_dense_view_nvme}.
     *
     * @param executor the NVMe executor whose rings read the headers, or {@code null} for the caller's
     *     thread.
     * @param paths devices or directories to serve the parts from; an empty list serves each where
     *     its manifest points.
     * @throws USearchException UNSUPPORTED while NVMe checkpoints carry no keys and no shard wrapper.
     */
    public static DenseIndex viewNvme(String path, @Nullable Executor executor, String[] paths) {
        return create(VIEW_NVME, executor, call -> {
            call.path("path", path);
            call.paths("nvme", paths);
        });
    }

    /**
     * Reads a checkpoint's geometry and metric on the caller's thread without loading it.
     *
     * @throws USearchException the statuses {@link #metadataRead(String, Executor)} names.
     */
    public static DenseMetadata metadataRead(String path) {
        return metadataRead(path, null);
    }

    /**
     * Reads a checkpoint's geometry and metric without loading it - from the manifest's summary when
     * the path names one, so no shard is opened. Calls {@code usearch_dense_metadata_read}.
     *
     * @throws USearchException MALFORMED on a checkpoint that ends early, IO when the file cannot be
     *     opened, and UNKNOWN_KIND when it names a metric this build cannot name.
     */
    public static DenseMetadata metadataRead(String path, @Nullable Executor executor) {
        return metadata(METADATA_READ, executor, call -> call.path("path", path));
    }

    /**
     * Reads geometry and metric metadata from a serialized monolith on the caller's thread.
     *
     * @throws USearchException the statuses {@link #metadataReadBuffer(MemorySegment, Executor)} names.
     */
    public static DenseMetadata metadataReadBuffer(MemorySegment buffer) {
        return metadataReadBuffer(buffer, null);
    }

    /**
     * Reads geometry and metric metadata from a serialized monolith buffer. Calls
     * {@code usearch_dense_metadata_read_buffer}.
     *
     * @throws USearchException MALFORMED on one that ends early, and UNKNOWN_KIND when it names a
     *     metric this build cannot name.
     */
    public static DenseMetadata metadataReadBuffer(MemorySegment buffer, @Nullable Executor executor) {
        return metadata(METADATA_READ_BUFFER, executor, call -> call.bytes("buffer", buffer));
    }

    /**
     * Releases the index, after every deferred call naming it is final. It is idempotent, and a freed
     * handle must not be used again; one the program drops without closing is released once it is
     * unreachable. Calls {@code usearch_dense_free}.
     *
     * @throws IllegalStateException while another thread's blocking call still runs on it.
     */
    @Override
    public void close() {
        handle.close();
    }

    /** Renders the index as the handle it wraps. */
    @Override
    public String toString() {
        return "DenseIndex(" + handle + ")";
    }

    /** Runs one lifecycle verb and adopts the handle it created. */
    private static DenseIndex create(Verb verb, @Nullable Executor executor, Consumer<Call> fill) {
        return Call.block(verb, null, executor, call -> {
            MemorySegment created = call.output("index", Library.POINTER);
            fill.accept(call);
            return () -> new DenseIndex(created.get(Library.POINTER, 0));
        });
    }

    private static DenseMetadata metadata(Verb verb, @Nullable Executor executor, Consumer<Call> fill) {
        return Call.block(verb, null, executor, call -> {
            fill.accept(call);
            MemorySegment metadata = call.output("metadata", METADATA);
            return () -> DenseMetadata.lift(metadata);
        });
    }

    private static long nonNegative(long value, String message) {
        if (value < 0) {
            throw new USearchException(Status.INVALID_ARGUMENT, message);
        }
        return value;
    }

    private static final MemoryLayout PREDICATE = MemoryLayout.structLayout(
                    Library.POINTER.withName("callback"), Library.POINTER.withName("state"))
            .withName("predicate");
    private static final MemoryLayout PROGRESS = MemoryLayout.structLayout(
                    Library.POINTER.withName("callback"),
                    Library.POINTER.withName("state"),
                    Library.SIZE.withName("steps"))
            .withName("progress");
    private static final MemoryLayout STATS = Library.POINTER.withName("stats");

    /** {@code usearch_dense_metadata_t}. */
    static final StructLayout METADATA = MemoryLayout.structLayout(
            Library.ENUM.withName("metric"),
            Library.ENUM.withName("quantization"),
            Library.SIZE.withName("dimensions"),
            Library.SIZE.withName("connectivity"));

    /** {@code usearch_dense_index_description_t}. */
    static final StructLayout DESCRIPTION = MemoryLayout.structLayout(
            METADATA.withName("metadata"),
            Library.SIZE.withName("size"),
            Library.SIZE.withName("capacity"),
            Library.SIZE.withName("memory_usage"),
            Library.POINTER.withName("capabilities"));

    /** {@code usearch_reachability_t}. */
    static final StructLayout REACHABILITY = MemoryLayout.structLayout(
            Library.SIZE.withName("audited"), Library.SIZE.withName("reachable"), Library.SIZE.withName("sweeps"));

    /** {@code usearch_dense_softmax_state_t}. */
    static final StructLayout SOFTMAX_STATE = MemoryLayout.structLayout(
            ValueLayout.JAVA_FLOAT.withName("maximum_score"), ValueLayout.JAVA_FLOAT.withName("partition"));

    /** {@code usearch_dense_init_ram_t} or its VRAM twin, whose placement field is named {@code placement}. */
    private static StructLayout initializing(String placement) {
        return MemoryLayout.structLayout(
                Library.POINTER.withName("index"),
                Library.POINTER.withName("outcome"),
                Library.ENUM.withName("metric"),
                Library.ENUM.withName("quantization"),
                Library.SIZE.withName("dimensions"),
                Library.SIZE.withName("connectivity"),
                Library.SIZE.withName("values_per_key"),
                Library.PLACEMENT.withName(placement));
    }

    /** {@code usearch_dense_load_*_t}, reading {@code source} into the placement named {@code placement}. */
    private static StructLayout loading(String source, String placement) {
        return Library.opening(
                Library.padding(),
                Library.BYTES_VIEW.withName(source),
                Library.PLACEMENT.withName(placement),
                Library.SIZE.withName("values_per_key"),
                PROGRESS);
    }

    /** {@code usearch_dense_metadata_read_t} or its buffer twin, reading {@code source}. */
    private static StructLayout reading(String source) {
        return MemoryLayout.structLayout(
                Library.POINTER.withName("executor"),
                Library.POINTER.withName("outcome"),
                Library.ENUM.withName("completion"),
                Library.padding(),
                Library.BYTES_VIEW.withName(source),
                Library.POINTER.withName("metadata"));
    }

    static final Verb INIT_RAM = new Verb("usearch_dense_init_ram", initializing("ram"));
    static final Verb INIT_VRAM = new Verb("usearch_dense_init_vram", initializing("vram"));
    static final Verb LOAD_RAM = new Verb("usearch_dense_load_ram", loading("path", "ram"));
    static final Verb LOAD_VRAM = new Verb("usearch_dense_load_vram", loading("path", "vram"));
    static final Verb LOAD_RAM_BUFFER = new Verb("usearch_dense_load_ram_buffer", loading("buffer", "ram"));
    static final Verb LOAD_VRAM_BUFFER = new Verb("usearch_dense_load_vram_buffer", loading("buffer", "vram"));

    static final Verb VIEW_RAM = new Verb(
            "usearch_dense_view_ram",
            MemoryLayout.structLayout(
                    Library.POINTER.withName("index"),
                    Library.POINTER.withName("outcome"),
                    Library.BYTES_VIEW.withName("path")));

    static final Verb VIEW_NVME = new Verb(
            "usearch_dense_view_nvme",
            Library.opening(
                    Library.padding(), Library.BYTES_VIEW.withName("path"), Library.PLACEMENT.withName("nvme")));

    static final Verb RESERVE =
            new Verb("usearch_dense_reserve", Library.opening(Library.padding(), Library.SIZE.withName("capacity")));

    static final Verb CLEAR = new Verb("usearch_dense_clear", Library.opening(Library.padding()));

    static final Verb DESCRIBE = new Verb(
            "usearch_dense_describe", Library.opening(Library.padding(), Library.POINTER.withName("description")));

    static final Verb INSERT = new Verb(
            "usearch_dense_insert",
            Library.opening(
                    Library.padding(),
                    Library.SIZE.withName("count"),
                    Library.KEYS_VIEW.withName("keys"),
                    Library.ROWS_VIEW.withName("vectors"),
                    Library.SIZE.withName("expansion_factor"),
                    PROGRESS,
                    STATS));

    static final Verb REFINE = new Verb(
            "usearch_dense_refine",
            Library.opening(
                    Library.padding(),
                    Library.POINTER.withName("changed"),
                    Library.POINTER.withName("reachability"),
                    Library.SIZE.withName("expansion_factor"),
                    PROGRESS,
                    STATS));

    static final Verb HEALTH = new Verb(
            "usearch_dense_health", Library.opening(Library.padding(), Library.POINTER.withName("reachability")));

    static final Verb SEARCH = new Verb(
            "usearch_dense_search",
            Library.opening(
                    Library.padding(),
                    Library.SIZE.withName("count"),
                    Library.ROWS_VIEW.withName("queries"),
                    Library.POINTER.withName("matches"),
                    Library.POINTER.withName("arena"),
                    Library.SIZE.withName("wanted"),
                    Library.SIZE.withName("expansion_factor"),
                    Library.ENUM.withName("traversal"),
                    Library.padding(),
                    PREDICATE,
                    PROGRESS,
                    STATS));

    // The key follows the completion unpadded, since sixteen bytes align to one.
    static final Verb COUNT = new Verb(
            "usearch_dense_count",
            Library.opening(Library.KEY.withName("key"), Library.padding(), Library.POINTER.withName("entries")));

    static final Verb GET = new Verb(
            "usearch_dense_get",
            Library.opening(
                    Library.KEY.withName("key"),
                    Library.padding(),
                    Library.POINTER.withName("vectors"),
                    Library.POINTER.withName("written"),
                    Library.POINTER.withName("arena"),
                    Library.ENUM.withName("outputs_scalar"),
                    Library.padding()));

    static final Verb ABSORB = new Verb(
            "usearch_dense_absorb",
            Library.opening(
                    Library.padding(),
                    Library.SIZE.withName("count"),
                    Library.POINTER.withName("donors"),
                    Library.POINTER.withName("changed"),
                    Library.POINTER.withName("reachability"),
                    Library.SIZE.withName("expansion_factor"),
                    PROGRESS,
                    STATS));

    static final Verb CURSOR_INIT = new Verb(
            "usearch_dense_cursor_init",
            MemoryLayout.structLayout(
                    Library.POINTER.withName("cursor"),
                    Library.POINTER.withName("index"),
                    Library.POINTER.withName("outcome"),
                    Library.SIZE.withName("starts_count"),
                    Library.POINTER.withName("starts")));

    static final Verb CURSOR_NEXT = new Verb(
            "usearch_dense_cursor_next",
            Library.opening(
                    Library.padding(),
                    Library.POINTER.withName("cursor"),
                    Library.POINTER.withName("keys"),
                    Library.POINTER.withName("links"),
                    Library.POINTER.withName("written"),
                    Library.POINTER.withName("arena"),
                    Library.SIZE.withName("wanted")));

    static final Verb SEARCH_STREAM_INIT = new Verb(
            "usearch_dense_search_stream_init",
            MemoryLayout.structLayout(
                    Library.POINTER.withName("stream"),
                    Library.POINTER.withName("index"),
                    Library.POINTER.withName("outcome"),
                    Library.ROWS_VIEW.withName("query"),
                    Library.SIZE.withName("expansion_factor")));

    static final Verb SEARCH_STREAM_NEXT = new Verb(
            "usearch_dense_search_stream_next",
            Library.opening(
                    Library.padding(),
                    Library.POINTER.withName("stream"),
                    Library.POINTER.withName("matches"),
                    Library.POINTER.withName("arena"),
                    Library.SIZE.withName("wanted")));

    static final Verb SAVE = new Verb(
            "usearch_dense_save", Library.opening(Library.padding(), Library.BYTES_VIEW.withName("path"), PROGRESS));

    static final Verb SAVE_BUFFER = new Verb(
            "usearch_dense_save_buffer",
            Library.opening(
                    Library.padding(),
                    Library.POINTER.withName("monolith"),
                    Library.POINTER.withName("arena"),
                    PROGRESS));

    static final Verb METADATA_READ = new Verb("usearch_dense_metadata_read", reading("path"));

    static final Verb METADATA_READ_BUFFER = new Verb("usearch_dense_metadata_read_buffer", reading("buffer"));

    static final Verb MULTIPLY = new Verb(
            "usearch_dense_multiply",
            Library.opening(
                    Library.padding(),
                    Library.SIZE.withName("count"),
                    Library.ROWS_VIEW.withName("left"),
                    Library.POINTER.withName("products"),
                    Library.POINTER.withName("written"),
                    Library.POINTER.withName("required"),
                    Library.POINTER.withName("arena"),
                    Library.SIZE.withName("wanted"),
                    Library.SIZE.withName("expansion_factor"),
                    PROGRESS,
                    STATS));

    static final Verb ATTEND = new Verb(
            "usearch_dense_attend",
            Library.opening(
                    Library.padding(),
                    Library.SIZE.withName("count"),
                    Library.ROWS_VIEW.withName("queries"),
                    Library.SIZE.withName("head_count"),
                    Library.POINTER.withName("value_lookup"),
                    Library.POINTER.withName("value_lookup_state"),
                    Library.POINTER.withName("carry"),
                    Library.POINTER.withName("outputs"),
                    Library.POINTER.withName("arena"),
                    Library.ENUM.withName("outputs_scalar"),
                    Library.padding(),
                    Library.SIZE.withName("expansion_factor"),
                    ValueLayout.JAVA_FLOAT.withName("score_scale"),
                    Library.padding(),
                    PROGRESS,
                    STATS));

    static final Verb MAXSIM = new Verb(
            "usearch_dense_maxsim",
            Library.opening(
                    Library.padding(),
                    Library.SIZE.withName("count"),
                    Library.ROWS_VIEW.withName("queries"),
                    Library.SIZE.withName("rows_per_query"),
                    Library.POINTER.withName("matches"),
                    Library.POINTER.withName("arena"),
                    Library.SIZE.withName("wanted"),
                    Library.SIZE.withName("expansion_factor"),
                    PROGRESS,
                    STATS));

    /** {@code usearch_value_lookup_t}. */
    private static final FunctionDescriptor LOOKUP = FunctionDescriptor.ofVoid(
            Library.POINTER, Library.POINTER, Library.SIZE, Library.ENUM, Library.SIZE, Library.POINTER);

    private static final MethodHandle RESOLVE;

    static {
        try {
            RESOLVE = MethodHandles.lookup().findVirtual(ValueLookup.Resolver.class, "resolve", LOOKUP.toMethodType());
        } catch (ReflectiveOperationException missing) {
            throw new ExceptionInInitializerError(missing);
        }
    }

    static final MethodHandle FREE = Library.downcall("usearch_dense_free", FunctionDescriptor.ofVoid(Library.POINTER));

    private static final MethodHandle CURSOR_FREE =
            Library.downcall("usearch_dense_cursor_free", FunctionDescriptor.ofVoid(Library.POINTER));

    private static final MethodHandle SEARCH_STREAM_FREE =
            Library.downcall("usearch_dense_search_stream_free", FunctionDescriptor.ofVoid(Library.POINTER));

    private static final MethodHandle METRIC_NAME = Library.namer("usearch_dense_metric_name");
}
