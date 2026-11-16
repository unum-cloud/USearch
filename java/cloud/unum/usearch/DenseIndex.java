/**
 *  @file java/cloud/unum/usearch/DenseIndex.java
 *  @author Ash Vardanian
 *  @date August 16, 2026
 *  @brief The dense index handle, its options, and the receivers that run its verbs on a chosen executor.
 */
package cloud.unum.usearch;

import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.StructLayout;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.util.function.Consumer;
import java.util.function.Predicate;
import java.util.function.ToLongFunction;

import org.jspecify.annotations.Nullable;

import cloud.unum.usearch.DenseCalls.Request;
import cloud.unum.usearch.Types.Arena;
import cloud.unum.usearch.Types.CPUClass;
import cloud.unum.usearch.Types.Call;
import cloud.unum.usearch.Types.Deferred;
import cloud.unum.usearch.Types.Executor;
import cloud.unum.usearch.Types.GPUClass;
import cloud.unum.usearch.Types.Handle;
import cloud.unum.usearch.Types.Key;
import cloud.unum.usearch.Types.Matches;
import cloud.unum.usearch.Types.Monolith;
import cloud.unum.usearch.Types.PhaseProgress;
import cloud.unum.usearch.Types.Progress;
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
 * reshape and no copy, and a single vector is a one-row matrix. The breadth of the candidates rides
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
 * {@link DeferredOn} that {@link #deferred} makes submits there and answers a {@link Deferred}.
 * Predicates, progress handlers and value lookups run on the executor's threads and must not throw;
 * an exception escaping one terminates the process.
 *
 * <p>
 * Every method here reports failure as an unchecked {@link USearchException} carrying the
 * {@link Status} the engine named, so the per-method tags below list only the statuses that method
 * distinctly produces. An {@link IllegalStateException} on a handle that was closed applies to all
 * of them and is not repeated.
 *
 * <pre>{@code
 * DenseOptions options = new DenseOptions().dimensions(2);
 * try (Executor pool = Executor.forRam(new int[0], null, 0);
 *         Arena arena = new Arena(pool);
 *         DenseIndex index = DenseIndex.initRam(options, new int[0])) {
 *     index.reserve(2);
 *     index.insert(new Key[] {Key.of(1), Key.of(2)}, RowsView.of(new float[] {0, 1, 1, 0}, 2));
 *     Matches matches = index.on(pool).search(RowsView.of(new float[] {0, 1}, 2), 1, arena);
 *     System.out.println(matches.key(0, 0));
 * }
 * }</pre>
 */
public final class DenseIndex implements AutoCloseable {

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
        private @Nullable CPUClass cpuClassCeiling = null;
        private @Nullable GPUClass gpuClassCeiling = null;

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

        /** The width of the indexed vectors, which init freezes, refusing zero with CONFIG_MISMATCH. */
        public DenseOptions dimensions(long dimensions) {
            this.dimensions = dimensions;
            return this;
        }

        /** The connections-per-node bound of the graph; zero derives. */
        public DenseOptions connectivity(long connectivity) {
            this.connectivity = connectivity;
            return this;
        }

        /**
         * The upper bound on entries sharing one key; zero is unbounded. No checkpoint records it,
         * so every load takes it as an argument of its own.
         */
        public DenseOptions valuesPerKey(long valuesPerKey) {
            this.valuesPerKey = valuesPerKey;
            return this;
        }

        /**
         * The newest class dispatch may pick, intersected with what the machine runs, so a class
         * the machine lacks narrows rather than refuses; {@code null} means no ceiling.
         */
        public DenseOptions cpuClassCeiling(@Nullable CPUClass cpuClassCeiling) {
            this.cpuClassCeiling = cpuClassCeiling;
            return this;
        }

        /**
         * The newest class dispatch may pick, capped below the best class every named device loads;
         * a ceiling no named device loads is refused as unsupported; {@code null} means no ceiling.
         */
        public DenseOptions gpuClassCeiling(@Nullable GPUClass gpuClassCeiling) {
            this.gpuClassCeiling = gpuClassCeiling;
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

        /** The upper bound on entries sharing one key; zero is unbounded. */
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

        /** Fills the fields both init structures share, at the same offsets in each. */
        void fill(Call call) {
            call.enumeration(INIT_METRIC, metric.code());
            call.enumeration(INIT_QUANTIZATION, quantization.code());
            call.size(INIT_DIMENSIONS, dimensions);
            call.size(INIT_CONNECTIVITY, connectivity);
            call.size(INIT_VALUES_PER_KEY, valuesPerKey);
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
     * @param quantization the scalar type its vectors are stored and quantized into.
     * @param connectivity the connections-per-node bound the graph was built with.
     */
    public record DenseMetadata(long dimensions, DenseMetric metric, Scalar quantization, long connectivity) {

        /** Lifts the {@code usearch_dense_metadata_t} at {@code at} of {@code struct}. */
        static DenseMetadata lift(MemorySegment struct, long at) {
            return new DenseMetadata(
                    (long) METADATA_DIMENSIONS.get(struct, at),
                    DenseMetric.of((int) METADATA_METRIC.get(struct, at)),
                    Scalar.of((int) METADATA_QUANTIZATION.get(struct, at)),
                    (long) METADATA_CONNECTIVITY.get(struct, at));
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
     * @param cpuClass the CPU class dispatch picked on this machine, as {@code usearch_cpu_class_name}
     *     spells it.
     * @param gpuClass the GPU class dispatch picked on the index's devices; "unknown" for a host index.
     * @param shardsCount the live shards.
     */
    public record DenseIndexDescription(
            DenseMetadata metadata,
            long size,
            long capacity,
            long memoryUsage,
            String cpuClass,
            String gpuClass,
            long shardsCount) {

        /** Lifts a {@code usearch_dense_index_description_t}. */
        static DenseIndexDescription lift(MemorySegment description) {
            return new DenseIndexDescription(
                    DenseMetadata.lift(description, DESCRIPTION_METADATA),
                    (long) DESCRIPTION_SIZE.get(description, 0L),
                    (long) DESCRIPTION_CAPACITY.get(description, 0L),
                    (long) DESCRIPTION_MEMORY_USAGE.get(description, 0L),
                    Library.text(CPU_CLASS_NAME, (int) DESCRIPTION_CPU_CLASS.get(description, 0L)),
                    Library.text(GPU_CLASS_NAME, (int) DESCRIPTION_GPU_CLASS.get(description, 0L)),
                    (long) DESCRIPTION_SHARDS_COUNT.get(description, 0L));
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
     * @param depthFromRoot the depths the walk from the root took to converge.
     * @param depthToRoot the depths the walk back toward the root took, where a health check
     *     walks both ways.
     * @param unlinked the entries no reached entry links to.
     * @param underlinked the entries linked to by fewer entries than they link to.
     * @param beyondMostPopulatedDepth the entries deeper than the depth holding the most entries.
     * @param unreachedIslands the islands no entry outside them links into.
     * @param strandedIslands the islands with no link out.
     * @param components the components when direction is ignored.
     */
    public record DenseReachability(
            long audited,
            long reachable,
            long depthFromRoot,
            long depthToRoot,
            long unlinked,
            long underlinked,
            long beyondMostPopulatedDepth,
            long unreachedIslands,
            long strandedIslands,
            long components) {

        /** Lifts a {@code usearch_reachability_t}. */
        static DenseReachability lift(MemorySegment reachability) {
            return new DenseReachability(
                    (long) REACHABILITY_AUDITED.get(reachability, 0L),
                    (long) REACHABILITY_REACHABLE.get(reachability, 0L),
                    (long) REACHABILITY_DEPTH_FROM_ROOT.get(reachability, 0L),
                    (long) REACHABILITY_DEPTH_TO_ROOT.get(reachability, 0L),
                    (long) REACHABILITY_UNLINKED.get(reachability, 0L),
                    (long) REACHABILITY_UNDERLINKED.get(reachability, 0L),
                    (long) REACHABILITY_BEYOND_MOST_POPULATED_DEPTH.get(reachability, 0L),
                    (long) REACHABILITY_UNREACHED_ISLANDS.get(reachability, 0L),
                    (long) REACHABILITY_STRANDED_ISLANDS.get(reachability, 0L),
                    (long) REACHABILITY_COMPONENTS.get(reachability, 0L));
        }
    }

    /** What a round concluded, so no caller picks a change rate. */
    public enum RefineVerdict {

        /** No round concluded: a report no round filled yet. */
        NONE,
        /** The round changed some entry's link set, so another round can help. */
        IMPROVED,
        /** The round changed no entry's link set. */
        CONVERGED,
        /** The round lowered nothing while something stands, as full entries leave no slot. */
        STALLED,
        /** The caller stopped the round; the phase names where. */
        STOPPED;

        /** Renders the verdict as the library spells it. Calls {@code usearch_refine_verdict_name}. */
        @Override
        public String toString() {
            return Library.text(VERDICT_NAME, ordinal());
        }
    }

    /**
     * What one refining call did to the graph: {@link DenseIndex#refine(DenseRefineConfig)} and
     * {@link DenseIndex#reshard} land it, so far while their plan stays live, and each plan's
     * {@code finish} lands the whole plan's.
     *
     * <p>
     * Mirrors {@code usearch_refine_report_t}.
     *
     * @param reselected the entries whose neighbors were selected again, the islands of every heal
     *     round included.
     * @param changed of those, the entries whose neighbor set gained a member it lacked; zero means
     *     the graph has settled.
     * @param islandsFound the entries the first audit found not reached both ways.
     * @param islandsReconnected the links written between an island and an entry reached both ways.
     * @param neighborhoods the neighborhoods the round re-stitched.
     * @param widestNeighborhood the entries of the widest neighborhood.
     * @param addedLinks the links the round added.
     * @param droppedLinks the links the round dropped.
     * @param verdict what the round concluded.
     * @param opening the health check opening the round.
     * @param reachability the health check closing the heal, as a fresh {@link DenseIndex#health()} right
     *     after would take it.
     */
    public record RefineReport(
            long reselected,
            long changed,
            long islandsFound,
            long islandsReconnected,
            long neighborhoods,
            long widestNeighborhood,
            long addedLinks,
            long droppedLinks,
            RefineVerdict verdict,
            DenseReachability opening,
            DenseReachability reachability) {

        /** Lifts a {@code usearch_refine_report_t}. */
        static RefineReport lift(MemorySegment report) {
            return new RefineReport(
                    (long) REPORT_RESELECTED.get(report, 0L),
                    (long) REPORT_CHANGED.get(report, 0L),
                    (long) REPORT_ISLANDS_FOUND.get(report, 0L),
                    (long) REPORT_ISLANDS_RECONNECTED.get(report, 0L),
                    (long) REPORT_NEIGHBORHOODS.get(report, 0L),
                    (long) REPORT_WIDEST_NEIGHBORHOOD.get(report, 0L),
                    (long) REPORT_ADDED_LINKS.get(report, 0L),
                    (long) REPORT_DROPPED_LINKS.get(report, 0L),
                    RefineVerdict.values()[(int) REPORT_VERDICT.get(report, 0L)],
                    DenseReachability.lift(report.asSlice(REPORT_OPENING, REACHABILITY.byteSize())),
                    DenseReachability.lift(report.asSlice(REPORT_REACHABILITY, REACHABILITY.byteSize())));
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
     * names one row of {@code values}, as {@code row} answers it, and a row outside the matrix or a
     * matrix of another scalar or width than the call states leaves the key without one, which
     * refuses the call. The function runs on the executor's threads together and must not throw; an
     * exception escaping it terminates the process.
     *
     * <p>
     * The values are borrowed once, here, and so is the upcall every call hands across: a native slab
     * is read in place and must stay open while the lookup is used, and a heap one is copied once.
     *
     * <p>
     * Mirrors {@code usearch_value_lookup_t}, whose scalar and columns are checked against the matrix.
     * It is a host function, so it serves RAM handles alone: a VRAM walk resolves rows through a
     * device function, which this binding does not build.
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

        /** Hands the upcall across, keeping it alive until the call is final. */
        void lend(Call call) {
            call.pointer(ATTEND_VALUE_LOOKUP, upcall);
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
                    long position = fits ? row.applyAsLong(Key.read(named, entry * Key.BYTES)) : -1;
                    found.setAtIndex(
                            Library.POINTER,
                            entry,
                            position >= 0 && position < rows
                                    ? MemorySegment.ofAddress(base + position * strideBytes)
                                    : MemorySegment.NULL);
                }
            }
        }
    }

    /**
     * The knobs of an insertion; see {@link DenseIndex#insert(Key[], RowsView, DenseInsertConfig)}.
     * Mirrors {@code usearch_dense_insert_t}.
     */
    public static final class DenseInsertConfig {

        long expansionFactor;

        @Nullable
        Progress progress;

        @Nullable
        DenseTraversalStats stats;

        /** Every knob derived. */
        public DenseInsertConfig() {}

        /** The breadth of the candidates the build walk explores; zero derives. */
        public DenseInsertConfig expansionFactor(long expansionFactor) {
            this.expansionFactor = expansionFactor;
            return this;
        }

        /** Hears how far the insertion has advanced; an insertion cannot be cancelled, so its answer is ignored. */
        public DenseInsertConfig progress(Progress progress) {
            this.progress = progress;
            return this;
        }

        /** Where the traversal telemetry lands, overwritten once the call succeeded. */
        public DenseInsertConfig stats(DenseTraversalStats stats) {
            this.stats = stats;
            return this;
        }
    }

    /**
     * The knobs of a refinement and the plan it continues; see
     * {@link DenseIndex#refine(DenseRefineConfig)}. Mirrors {@code usearch_dense_refine_t}.
     */
    public static final class DenseRefineConfig {

        long expansionFactor;
        long memoryLimit;

        @Nullable
        DenseRefinePlan plan;

        @Nullable
        Progress progress;

        @Nullable
        DenseTraversalStats stats;

        /** Every knob derived, one round run to its end. */
        public DenseRefineConfig() {}

        /** The breadth of the candidates any walk beside the rounds explores; zero derives. */
        public DenseRefineConfig expansionFactor(long expansionFactor) {
            this.expansionFactor = expansionFactor;
            return this;
        }

        /** The bytes a loaded NVMe index holds for the one part it rewrites; zero derives, and less streams it. */
        public DenseRefineConfig memoryLimit(long memoryLimit) {
            this.memoryLimit = memoryLimit;
            return this;
        }

        /** The plan the call opens or continues, so a cancel leaves its round resumable. */
        public DenseRefineConfig plan(DenseRefinePlan plan) {
            this.plan = plan;
            return this;
        }

        /** Hears every barrier of every phase, and may cancel the round there. */
        public DenseRefineConfig progress(Progress progress) {
            this.progress = progress;
            return this;
        }

        /** Where the traversal telemetry lands, overwritten once the call succeeded. */
        public DenseRefineConfig stats(DenseTraversalStats stats) {
            this.stats = stats;
            return this;
        }
    }

    /**
     * The placement a reshard recuts toward, the plan it continues, and the knobs of its closing
     * rounds, by default keeping every handle's own placement; see
     * {@link DenseIndex#reshard(DenseIndex[], DenseIndex[], DenseReshardConfig)}. Mirrors
     * {@code usearch_dense_reshard_t}.
     */
    public static final class DenseReshardConfig {

        int @Nullable [] memoryDomains;
        int @Nullable [] devices;
        String @Nullable [] paths;
        long shardCapacity;
        long expansionFactor;
        long memoryLimit;
        long seed;

        @Nullable
        DenseReshardPlan plan;

        @Nullable
        Progress progress;

        @Nullable
        DenseTraversalStats stats;

        /** Every placement kept and every knob derived, under seed zero. */
        public DenseReshardConfig() {}

        /** NUMA node IDs of a RAM placement, one shard per node at least; an empty list derives them. */
        public DenseReshardConfig memoryDomains(int[] memoryDomains) {
            this.memoryDomains = memoryDomains;
            return this;
        }

        /** CUDA ordinals of a VRAM placement, one shard per device at least. */
        public DenseReshardConfig devices(int[] devices) {
            this.devices = devices;
            return this;
        }

        /** Drives of an NVMe placement, one shard per drive at least. */
        public DenseReshardConfig paths(String[] paths) {
            this.paths = paths;
            return this;
        }

        /** Entries per shard of the RAM or VRAM placement; zero keeps the handle's own bound. */
        public DenseReshardConfig shardCapacity(long shardCapacity) {
            this.shardCapacity = shardCapacity;
            return this;
        }

        /** The breadth of the candidates any walk beside the closing rounds explores; zero derives. */
        public DenseReshardConfig expansionFactor(long expansionFactor) {
            this.expansionFactor = expansionFactor;
            return this;
        }

        /** The bytes a loaded NVMe target holds for the one part it rewrites; zero derives, and less streams it. */
        public DenseReshardConfig memoryLimit(long memoryLimit) {
            this.memoryLimit = memoryLimit;
            return this;
        }

        /** Picks the part roots of a source cut among several targets: equal seeds over an unchanged source cut equally. */
        public DenseReshardConfig seed(long seed) {
            this.seed = seed;
            return this;
        }

        /** The plan the call opens or continues, so a cancel leaves its closing rounds resumable. */
        public DenseReshardConfig plan(DenseReshardPlan plan) {
            this.plan = plan;
            return this;
        }

        /** Hears every barrier of the closing rounds, and may cancel them there; a move is never cancelled. */
        public DenseReshardConfig progress(Progress progress) {
            this.progress = progress;
            return this;
        }

        /** Where the traversal telemetry lands, overwritten once the call succeeded. */
        public DenseReshardConfig stats(DenseTraversalStats stats) {
            this.stats = stats;
            return this;
        }
    }

    /**
     * The view a sample grows and how it draws, by default across the whole index with nothing
     * shown yet; see {@link DenseIndex#sample(long, Arena, DenseSampleConfig)}. Mirrors
     * {@code usearch_dense_sample_t}.
     */
    public static final class DenseSampleConfig {

        Key[] shown = new Key[0];
        float[] shownCoordinates = new float[0];
        Key[] focus = new Key[0];
        long seed;
        long expansionFactor;

        @Nullable
        Progress progress;

        @Nullable
        DenseTraversalStats stats;

        /**
         * The keys already shown, never drawn again, and where each sits, three values apiece - the
         * points later draws are placed against.
         *
         * @throws USearchException SHAPE_MISMATCH unless the coordinates hold three per key.
         */
        public DenseSampleConfig shown(Key[] keys, float[] coordinates) {
            if ((long) keys.length * 3 != coordinates.length) {
                throw new USearchException(Status.SHAPE_MISMATCH, "every shown key needs three coordinates");
            }
            this.shown = keys;
            this.shownCoordinates = coordinates;
            return this;
        }

        /** The keys to look around, the roots the draws grow from; none looks across the whole index. */
        public DenseSampleConfig focus(Key... keys) {
            this.focus = keys;
            return this;
        }

        /** The draw: equal seeds, shown keys and focus over an unchanged index draw equally. */
        public DenseSampleConfig seed(long seed) {
            this.seed = seed;
            return this;
        }

        /** The breadth of the candidates the walks behind the draws explore; zero derives. */
        public DenseSampleConfig expansionFactor(long expansionFactor) {
            this.expansionFactor = expansionFactor;
            return this;
        }

        /** Hears how far the call has advanced, and may cancel it. */
        public DenseSampleConfig progress(Progress progress) {
            this.progress = progress;
            return this;
        }

        /** Where the traversal telemetry lands, overwritten once the call succeeded. */
        public DenseSampleConfig stats(DenseTraversalStats stats) {
            this.stats = stats;
            return this;
        }
    }

    /**
     * How many matches a search keeps and the knobs of its walk; see
     * {@link DenseIndex#search(RowsView, DenseSearchConfig, Arena)}. Mirrors {@code usearch_dense_search_t}.
     */
    public static final class DenseSearchConfig {

        final long wanted;
        long expansionFactor;
        Traversal traversal = Traversal.WALK;

        @Nullable
        Predicate<Key> predicate;

        @Nullable
        Progress progress;

        @Nullable
        DenseTraversalStats stats;

        /** Up to {@code wanted} matches per query - the "k" in kANN. */
        public DenseSearchConfig(long wanted) {
            this.wanted = wanted;
        }

        /** The breadth of the candidates the walk explores; zero derives. */
        public DenseSearchConfig expansionFactor(long expansionFactor) {
            this.expansionFactor = expansionFactor;
            return this;
        }

        /**
         * A walk, the default, or an exact scan of every entry on RAM and NVMe handles; a VRAM handle
         * refuses a scan as UNSUPPORTED.
         */
        public DenseSearchConfig traversal(Traversal traversal) {
            this.traversal = traversal;
            return this;
        }

        /**
         * Which candidate keys the walk may report, asked from the executor's threads together; it
         * must not throw, as an exception escaping it terminates the process.
         */
        public DenseSearchConfig predicate(Predicate<Key> predicate) {
            this.predicate = predicate;
            return this;
        }

        /** Hears how far the call has advanced, and may cancel it. */
        public DenseSearchConfig progress(Progress progress) {
            this.progress = progress;
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
     * {@link DenseIndex#maxsim(RowsView, long, DenseMaxsimConfig, Arena)}. Mirrors {@code usearch_dense_maxsim_t}.
     */
    public static final class DenseMaxsimConfig {

        final long wanted;
        long expansionFactor;

        @Nullable
        Predicate<Key> predicate;

        @Nullable
        Progress progress;

        @Nullable
        DenseTraversalStats stats;

        /**
         * Up to {@code wanted} documents per query - the "k" over keys, not rows; zero keeps every key
         * the query scored.
         */
        public DenseMaxsimConfig(long wanted) {
            this.wanted = wanted;
        }

        /** The breadth of the candidates the walk explores; zero derives. */
        public DenseMaxsimConfig expansionFactor(long expansionFactor) {
            this.expansionFactor = expansionFactor;
            return this;
        }

        /**
         * Which candidate keys the walk may report, asked from the executor's threads together; it
         * must not throw, as an exception escaping it terminates the process.
         */
        public DenseMaxsimConfig predicate(Predicate<Key> predicate) {
            this.predicate = predicate;
            return this;
        }

        /** Hears how far the call has advanced, and may cancel it. */
        public DenseMaxsimConfig progress(Progress progress) {
            this.progress = progress;
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
     * {@link DenseIndex#multiply(RowsView, DenseMultiplyConfig, Arena)}. Mirrors {@code usearch_dense_multiply_t}.
     */
    public static final class DenseMultiplyConfig {

        long wanted;
        long expansionFactor;

        @Nullable
        Predicate<Key> predicate;

        @Nullable
        Progress progress;

        @Nullable
        DenseTraversalStats stats;

        /** Every pair kept, with every knob derived. */
        public DenseMultiplyConfig() {}

        /** The bound on pairs kept per left row, the first ones its walk scored; zero keeps all. */
        public DenseMultiplyConfig wanted(long wanted) {
            this.wanted = wanted;
            return this;
        }

        /** The breadth of the candidates, which widens the emitted pattern; zero derives. */
        public DenseMultiplyConfig expansionFactor(long expansionFactor) {
            this.expansionFactor = expansionFactor;
            return this;
        }

        /**
         * Which candidate keys the walk may report, asked from the executor's threads together; it
         * must not throw, as an exception escaping it terminates the process.
         */
        public DenseMultiplyConfig predicate(Predicate<Key> predicate) {
            this.predicate = predicate;
            return this;
        }

        /** Hears how far the call has advanced, and may cancel it. */
        public DenseMultiplyConfig progress(Progress progress) {
            this.progress = progress;
            return this;
        }

        /** Where the traversal telemetry lands, overwritten once the call succeeded. */
        public DenseMultiplyConfig stats(DenseTraversalStats stats) {
            this.stats = stats;
            return this;
        }
    }

    /**
     * The knobs of an attention; see {@link DenseIndex#attend(RowsView, ValueLookup, DenseAttendConfig, Arena)}.
     * Mirrors {@code usearch_dense_attend_t}.
     */
    public static final class DenseAttendConfig {

        long headCount;
        Scalar outputsScalar = Scalar.UNKNOWN;
        DenseSoftmaxState @Nullable [] carry;
        long expansionFactor;
        float scoreScale;

        @Nullable
        Predicate<Key> predicate;

        @Nullable
        Progress progress;

        @Nullable
        DenseTraversalStats stats;

        /** Every knob derived, the outputs in the queries' own scalar. */
        public DenseAttendConfig() {}

        /**
         * The query heads sharing the index's key-value head, and with it one walk per token: row
         * {@code t * headCount + h} is head {@code h} of token {@code t}, so it must divide the rows;
         * zero means one.
         */
        public DenseAttendConfig headCount(long headCount) {
            this.headCount = headCount;
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

        /** The breadth of the candidates; zero derives. Wider candidates walk further, folding more of the tail. */
        public DenseAttendConfig expansionFactor(long expansionFactor) {
            this.expansionFactor = expansionFactor;
            return this;
        }

        /** The logit scale, conventionally the inverse square root of the head dimensionality; zero derives it. */
        public DenseAttendConfig scoreScale(float scoreScale) {
            this.scoreScale = scoreScale;
            return this;
        }

        /**
         * Which candidate keys the walk may report, asked from the executor's threads together; it
         * must not throw, as an exception escaping it terminates the process.
         */
        public DenseAttendConfig predicate(Predicate<Key> predicate) {
            this.predicate = predicate;
            return this;
        }

        /** Hears how far the call has advanced, and may cancel it. */
        public DenseAttendConfig progress(Progress progress) {
            this.progress = progress;
            return this;
        }

        /** Where the traversal telemetry lands, overwritten once the call succeeded. */
        public DenseAttendConfig stats(DenseTraversalStats stats) {
            this.stats = stats;
            return this;
        }
    }

    /**
     * A refinement in flight, which {@link DenseIndex#refine(DenseRefineConfig)} opens on its first
     * call and continues on every later one, so a cancel keeps the round in hand resumable. The index
     * admits no insertion and no save while it is open, and every call naming it names the index and
     * executor it was opened with. {@link DenseIndex#refinePlanFinish} closes it on the receiver that
     * opened it; a plan dropped open is never finished, and keeps the index closed to insertions.
     */
    public static final class DenseRefinePlan {

        private final MemorySegment plan = java.lang.foreign.Arena.ofAuto().allocate(Library.POINTER);
        private final MemorySegment phase = java.lang.foreign.Arena.ofAuto().allocate(Library.PHASE_PROGRESS);

        /** A plan no call has opened yet. */
        public DenseRefinePlan() {}

        /** Where the last call naming the plan stopped. */
        public PhaseProgress phase() {
            return PhaseProgress.lift(phase);
        }

        void lend(Call call) {
            call.pointer(REFINE_PLAN, plan);
            call.pointer(REFINE_PHASE, phase);
            call.lend(this);
        }

        /** The open plan a finish frees, leaving this one to open afresh on its next call. */
        MemorySegment take() {
            MemorySegment open = plan.get(Library.POINTER, 0);
            plan.set(Library.POINTER, 0, MemorySegment.NULL);
            return open;
        }
    }

    /**
     * A reshard in flight, which {@link DenseIndex#reshard(DenseIndex[], DenseIndex[], DenseReshardConfig)}
     * opens on its first call and continues on every later one, moving what is still pending and
     * resuming the closing rounds a cancel stopped. Every index it names admits no insertion and no
     * save while it is open, and every call naming it names the executor it was opened with.
     * {@link DenseIndex#reshardPlanFinish} closes it on a receiver of that executor; a plan dropped
     * open is never finished, and keeps every index it names closed to insertions.
     */
    public static final class DenseReshardPlan {

        private final MemorySegment plan = java.lang.foreign.Arena.ofAuto().allocate(Library.POINTER);
        private final MemorySegment phase = java.lang.foreign.Arena.ofAuto().allocate(Library.PHASE_PROGRESS);

        /** A plan no call has opened yet. */
        public DenseReshardPlan() {}

        /** Where the last call naming the plan stopped. */
        public PhaseProgress phase() {
            return PhaseProgress.lift(phase);
        }

        void lend(Call call) {
            call.pointer(RESHARD_PLAN, plan);
            call.pointer(RESHARD_PHASE, phase);
            call.lend(this);
        }

        /** The open plan a finish frees, leaving this one to open afresh on its next call. */
        MemorySegment take() {
            MemorySegment open = plan.get(Library.POINTER, 0);
            plan.set(Library.POINTER, 0, MemorySegment.NULL);
            return open;
        }
    }

    /**
     * Rows a call wrote into its arena - the entries {@link DenseIndex#get} found under a key, the
     * outputs of {@link DenseIndex#attend}, or the vectors of a {@link DenseBatch} - at the width, pitch
     * and scalar the library reported. Valid until the arena's next call or its close.
     */
    public static final class DenseVectors {

        private final MemorySegment data;
        private final long rows;
        private final long columns;
        private final long strideBytes;
        private final Scalar scalar;

        private DenseVectors(MemorySegment data, long rows, long columns, long strideBytes, Scalar scalar) {
            this.data = data;
            this.rows = rows;
            this.columns = columns;
            this.strideBytes = strideBytes;
            this.scalar = scalar;
        }

        /** Lifts the {@code usearch_rows_view_t} of {@code rows} rows a call wrote into its arena. */
        static DenseVectors lift(Call call, MemorySegment view, long rows) {
            long stride = (long) Library.ROWS_STRIDE.get(view, 0L);
            return new DenseVectors(
                    call.reach(
                            (MemorySegment) Library.ROWS_DATA.get(view, 0L),
                            Types.Extents.product(rows, stride, "the rows")),
                    rows,
                    (long) Library.ROWS_COLUMNS.get(view, 0L),
                    stride,
                    Scalar.of((int) Library.ROWS_SCALAR.get(view, 0L)));
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

        /** The rows in place, in native byte order. */
        public MemorySegment segment() {
            return data;
        }
    }

    /**
     * The ragged sparse product {@link DenseIndex#multiply} wrote into its arena: per left row, every
     * pair its walk scored, in traversal order; or the keys {@link DenseIndex#maxsim} kept per query,
     * largest score first. Valid until the arena's next call or its close.
     */
    public static final class DenseProducts {

        private final MemorySegment offsets;
        private final int queries;
        private final MemorySegment keys;
        private final MemorySegment scores;
        private final long required;

        private DenseProducts(
                MemorySegment offsets, int queries, MemorySegment keys, MemorySegment scores, long required) {
            this.offsets = offsets;
            this.queries = queries;
            this.keys = keys;
            this.scores = scores;
            this.required = required;
        }

        /** Lifts the {@code usearch_matches_tape_view_t} a call over {@code queries} wrote into its arena. */
        static DenseProducts lift(Call call, MemorySegment view, int queries, long pairs, long required) {
            return new DenseProducts(
                    call.reach((MemorySegment) Library.TAPE_OFFSETS.get(view, 0L), (queries + 1L) * Long.BYTES),
                    queries,
                    call.reach((MemorySegment) Library.TAPE_KEYS.get(view, 0L), pairs * Key.BYTES),
                    call.reach((MemorySegment) Library.TAPE_SCORES.get(view, 0L), pairs * Double.BYTES),
                    required);
        }

        /** Left rows the call multiplied, or queries it scored. */
        public int queries() {
            return queries;
        }

        /**
         * Pairs the walks scored across every left row, or keys scored across every query, before the
         * wanted bound cut any.
         */
        public long required() {
            return required;
        }

        /**
         * How many pairs one left row, or keys one query, holds.
         *
         * @throws USearchException SHAPE_MISMATCH when the query is outside the batch.
         */
        public long count(int query) {
            if (query < 0 || query >= queries) {
                throw new USearchException(Status.SHAPE_MISMATCH, "the query is outside the batch");
            }
            return offsets.getAtIndex(ValueLayout.JAVA_LONG, query + 1)
                    - offsets.getAtIndex(ValueLayout.JAVA_LONG, query);
        }

        /**
         * The key naming the indexed entry of one pair.
         *
         * @throws USearchException SHAPE_MISMATCH when the query is outside the batch, or the rank past
         *     its pairs.
         */
        public Key key(int query, long rank) {
            return Key.read(keys, pairOf(query, rank) * Key.BYTES);
        }

        /**
         * The product of one pair.
         *
         * @throws USearchException SHAPE_MISMATCH when the query is outside the batch, or the rank past
         *     its pairs.
         */
        public double score(int query, long rank) {
            return scores.getAtIndex(ValueLayout.JAVA_DOUBLE, pairOf(query, rank));
        }

        private long pairOf(int query, long rank) {
            long pairs = count(query);
            if (rank < 0 || rank >= pairs) {
                throw new USearchException(Status.SHAPE_MISMATCH, "the query holds " + pairs + " pairs");
            }
            return offsets.getAtIndex(ValueLayout.JAVA_LONG, query) + rank;
        }
    }

    /**
     * The draws of one {@link DenseIndex#sample(long, Arena, DenseSampleConfig)}, written into its arena:
     * the keys drawn, which of its key's entries each is, the shown or earlier draw each hangs from,
     * and where each is placed. Valid until the arena's next call or its close.
     */
    public static final class Sample {

        private final MemorySegment keys;
        private final MemorySegment entryOrdinals;
        private final Matches parents;
        private final MemorySegment coordinates;
        private final long written;

        private Sample(
                MemorySegment keys,
                MemorySegment entryOrdinals,
                Matches parents,
                MemorySegment coordinates,
                long written) {
            this.keys = keys;
            this.entryOrdinals = entryOrdinals;
            this.parents = parents;
            this.coordinates = coordinates;
            this.written = written;
        }

        /** Lifts the outputs of a {@code usearch_dense_sample} that wrote {@code written} draws. */
        static Sample lift(
                Call call,
                MemorySegment keys,
                MemorySegment entryOrdinals,
                MemorySegment parents,
                MemorySegment coordinates,
                long written) {
            return new Sample(
                    call.reach(
                            (MemorySegment) Library.KEYS_DATA.get(keys, 0L),
                            Types.Extents.product(written, Key.BYTES, "the keys")),
                    call.reach(
                            entryOrdinals.get(Library.POINTER, 0),
                            Types.Extents.product(written, Long.BYTES, "the entry ordinals")),
                    Matches.lift(call, parents, written),
                    call.reach(
                            (MemorySegment) Library.ROWS_DATA.get(coordinates, 0L),
                            Types.Extents.product(written, 3 * Float.BYTES, "the coordinates")),
                    written);
        }

        /** How many draws the call wrote; fewer than wanted only once the region is exhausted. */
        public long written() {
            return written;
        }

        /**
         * The key of one draw.
         *
         * @throws USearchException SHAPE_MISMATCH when the draw is past the sample.
         */
        public Key key(long draw) {
            return Key.read(keys, checked(draw) * Key.BYTES);
        }

        /**
         * Which of its key's entries one draw is, in the order {@link DenseIndex#get} writes them.
         *
         * @throws USearchException the statuses {@link #key(long)} names.
         */
        public long entryOrdinal(long draw) {
            return entryOrdinals.getAtIndex(ValueLayout.JAVA_LONG, checked(draw));
        }

        /** The one shown or earlier draw each draw hangs from and its distance, one query per draw. */
        public Matches parents() {
            return parents;
        }

        /**
         * One coordinate of where a draw is placed, {@code axis} below three.
         *
         * @throws USearchException the statuses {@link #key(long)} names, and SHAPE_MISMATCH on an axis
         *     past the third.
         */
        public float coordinate(long draw, int axis) {
            if (axis < 0 || axis >= 3) {
                throw new USearchException(Status.SHAPE_MISMATCH, "a draw is placed in three dimensions");
            }
            return coordinates.getAtIndex(ValueLayout.JAVA_FLOAT, checked(draw) * 3 + axis);
        }

        private long checked(long draw) {
            if (draw < 0 || draw >= written) {
                throw new USearchException(Status.SHAPE_MISMATCH, "the sample holds " + written + " draws");
            }
            return draw;
        }
    }

    /**
     * One pull of a {@link DenseCursor}, written into its arena: the keys of the entries passed, and
     * the keys each links to when asked for. Valid until the arena's next call or its close.
     */
    public static final class DenseBatch {

        private final MemorySegment keys;
        private final long keysStride;
        private final long written;
        private final @Nullable MemorySegment linkOffsets;
        private final MemorySegment linkKeys;

        private DenseBatch(
                MemorySegment keys,
                long keysStride,
                long written,
                @Nullable MemorySegment linkOffsets,
                MemorySegment linkKeys) {
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
         * @throws USearchException SHAPE_MISMATCH when the entry is past the batch.
         */
        public Key key(long entry) {
            if (entry < 0 || entry >= written) {
                throw new USearchException(Status.SHAPE_MISMATCH, "the batch holds " + written + " entries");
            }
            return Key.read(keys, entry * keysStride);
        }

        /**
         * How many keys one entry links to.
         *
         * @throws USearchException INVALID_ARGUMENT when the pull exported no links, and SHAPE_MISMATCH
         *     when the entry is past the batch.
         */
        public long linkCount(long entry) {
            MemorySegment offsets = linked(entry);
            return offsets.getAtIndex(ValueLayout.JAVA_LONG, entry + 1)
                    - offsets.getAtIndex(ValueLayout.JAVA_LONG, entry);
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
         * @throws USearchException CONTENDED when the arena serves another call.
         */
        public DenseBatch next(long wanted, Arena arena) {
            return pull(wanted, false, arena);
        }

        /**
         * Copies the keys of up to {@code wanted} next entries into {@code arena}, each with the keys it
         * links to. Calls {@code usearch_dense_cursor_next}.
         *
         * @throws USearchException CONTENDED when the arena serves another call.
         */
        public DenseBatch nextLinks(long wanted, Arena arena) {
            return pull(wanted, true, arena);
        }

        private DenseBatch pull(long wanted, boolean linked, Arena arena) {
            return Call.block(CURSOR_NEXT, index.handle, executor, call -> {
                call.lease(CURSOR_NEXT_CURSOR, handle);
                MemorySegment keys = call.output(CURSOR_NEXT_KEYS, Library.KEYS_VIEW);
                MemorySegment tape =
                        linked ? call.output(CURSOR_NEXT_LINKS, Library.KEYS_TAPE_VIEW) : MemorySegment.NULL;
                MemorySegment written = call.output(CURSOR_NEXT_WRITTEN, Library.SIZE);
                call.arena(CURSOR_NEXT_ARENA, arena);
                call.size(CURSOR_NEXT_WANTED, wanted);
                return () -> {
                    long count = written.get(Library.SIZE, 0);
                    long stride = (long) Library.KEYS_STRIDE.get(keys, 0L);
                    MemorySegment offsets = null;
                    MemorySegment linkKeys = MemorySegment.NULL;
                    if (linked) {
                        offsets = call.reach(
                                (MemorySegment) Library.KEYS_TAPE_OFFSETS.get(tape, 0L), (count + 1) * Long.BYTES);
                        long total = count == 0 ? 0 : offsets.getAtIndex(ValueLayout.JAVA_LONG, count);
                        linkKeys = call.reach(
                                (MemorySegment) Library.KEYS_TAPE_KEYS.get(tape, 0L),
                                Types.Extents.product(total, Key.BYTES, "the links"));
                    }
                    return new DenseBatch(
                            call.reach(
                                    (MemorySegment) Library.KEYS_DATA.get(keys, 0L),
                                    Types.Extents.product(count, stride, "the keys")),
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
         * @throws USearchException CONTENDED when the arena serves another call.
         */
        public Matches next(long wanted, Arena arena) {
            return Call.block(SEARCH_STREAM_NEXT, index.handle, executor, call -> {
                call.lease(SEARCH_STREAM_NEXT_STREAM, handle);
                MemorySegment matches = call.output(SEARCH_STREAM_NEXT_MATCHES, Library.MATCHES_VIEW);
                call.arena(SEARCH_STREAM_NEXT_ARENA, arena);
                call.size(SEARCH_STREAM_NEXT_WANTED, wanted);
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

    /** The receiver whose verbs block on an executor, contended while another call holds it. */
    public static final class On {

        private final DenseIndex index;
        private final Executor executor;

        On(DenseIndex index, Executor executor) {
            this.index = index;
            this.executor = executor;
        }

        private <Result> Result block(Request<Result> request) {
            return Call.block(request.verb(), index.handle, executor, request.fill());
        }

        /** Runs {@link DenseIndex#reserve(long)} on this receiver's executor. */
        public void reserve(long capacity) {
            block(DenseCalls.reserving(capacity));
        }

        /** Runs {@link DenseIndex#clear()} on this receiver's executor. */
        public void clear() {
            block(DenseCalls.clearing());
        }

        /** Runs {@link DenseIndex#describe()} on this receiver's executor. */
        public DenseIndexDescription describe() {
            return block(DenseCalls.describing());
        }

        /** Runs {@link DenseIndex#count(Key)} on this receiver's executor. */
        public long count(Key key) {
            return block(DenseCalls.counting(key));
        }

        /** Runs {@link DenseIndex#contains(Key)} on this receiver's executor. */
        public boolean contains(Key key) {
            return count(key) != 0;
        }

        /** Runs {@link DenseIndex#get(Key, Scalar, Arena)} on this receiver's executor. */
        public DenseVectors get(Key key, Scalar outputsScalar, Arena arena) {
            return block(DenseCalls.getting(key, outputsScalar, arena));
        }

        /** Runs {@link DenseIndex#insert(Key[], RowsView)} on this receiver's executor. */
        public void insert(Key[] keys, RowsView vectors) {
            block(DenseCalls.inserting(keys, vectors, new DenseInsertConfig()));
        }

        /** Runs {@link DenseIndex#insert(Key[], RowsView, DenseInsertConfig)} on this receiver's executor. */
        public void insert(Key[] keys, RowsView vectors, DenseInsertConfig config) {
            block(DenseCalls.inserting(keys, vectors, config));
        }

        /** Runs {@link DenseIndex#refine()} on this receiver's executor. */
        public RefineReport refine() {
            return block(DenseCalls.refining(new DenseRefineConfig()));
        }

        /** Runs {@link DenseIndex#refine(DenseRefineConfig)} on this receiver's executor. */
        public RefineReport refine(DenseRefineConfig config) {
            return block(DenseCalls.refining(config));
        }

        /** Runs {@link DenseIndex#refinePlanFinish(DenseRefinePlan)} on this receiver's executor. */
        public RefineReport refinePlanFinish(DenseRefinePlan plan) {
            return block(DenseCalls.refinePlanFinishing(plan));
        }

        /** Runs {@link DenseIndex#health()} on this receiver's executor. */
        public DenseReachability health() {
            return block(DenseCalls.auditing());
        }

        /** Runs {@link DenseIndex#search(RowsView, long, Arena)} on this receiver's executor. */
        public Matches search(RowsView queries, long wanted, Arena arena) {
            return block(DenseCalls.searching(queries, new DenseSearchConfig(wanted), arena));
        }

        /** Runs {@link DenseIndex#search(RowsView, DenseSearchConfig, Arena)} on this receiver's executor. */
        public Matches search(RowsView queries, DenseSearchConfig config, Arena arena) {
            return block(DenseCalls.searching(queries, config, arena));
        }

        /** Runs {@link DenseIndex#maxsim(RowsView, long, long, Arena)} on this receiver's executor. */
        public DenseProducts maxsim(RowsView queries, long rowsPerQuery, long wanted, Arena arena) {
            return block(DenseCalls.scoring(queries, rowsPerQuery, new DenseMaxsimConfig(wanted), arena));
        }

        /** Runs {@link DenseIndex#maxsim(RowsView, long, DenseMaxsimConfig, Arena)} on this receiver's executor. */
        public DenseProducts maxsim(RowsView queries, long rowsPerQuery, DenseMaxsimConfig config, Arena arena) {
            return block(DenseCalls.scoring(queries, rowsPerQuery, config, arena));
        }

        /** Runs {@link DenseIndex#multiply(RowsView, Arena)} on this receiver's executor. */
        public DenseProducts multiply(RowsView left, Arena arena) {
            return block(DenseCalls.multiplying(left, new DenseMultiplyConfig(), arena));
        }

        /** Runs {@link DenseIndex#multiply(RowsView, DenseMultiplyConfig, Arena)} on this receiver's executor. */
        public DenseProducts multiply(RowsView left, DenseMultiplyConfig config, Arena arena) {
            return block(DenseCalls.multiplying(left, config, arena));
        }

        /** Runs {@link DenseIndex#attend(RowsView, ValueLookup, Arena)} on this receiver's executor. */
        public DenseVectors attend(RowsView queries, ValueLookup valueLookup, Arena arena) {
            return block(DenseCalls.attending(queries, valueLookup, new DenseAttendConfig(), arena));
        }

        /**
         * Runs {@link DenseIndex#attend(RowsView, ValueLookup, DenseAttendConfig, Arena)}
         * on this receiver's executor.
         */
        public DenseVectors attend(RowsView queries, ValueLookup valueLookup, DenseAttendConfig config, Arena arena) {
            return block(DenseCalls.attending(queries, valueLookup, config, arena));
        }

        /**
         * Runs {@link DenseIndex#reshard(DenseIndex[], DenseIndex[], DenseReshardConfig)}
         * on this receiver's executor.
         */
        public RefineReport reshard(DenseIndex[] sources, DenseIndex[] targets, DenseReshardConfig config) {
            return block(DenseCalls.resharding(sources, targets, config));
        }

        /** Runs {@link DenseIndex#reshardPlanFinish(DenseReshardPlan)} on this receiver's executor. */
        public RefineReport reshardPlanFinish(DenseReshardPlan plan) {
            return block(DenseCalls.reshardPlanFinishing(plan));
        }

        /** Runs {@link DenseIndex#sample(long, Arena, DenseSampleConfig)} on this receiver's executor. */
        public Sample sample(long wanted, Arena arena, DenseSampleConfig config) {
            return block(DenseCalls.sampling(wanted, arena, config));
        }

        /** Runs {@link DenseIndex#save(String)} on this receiver's executor. */
        public void save(String path) {
            block(DenseCalls.saving(path, null));
        }

        /** Runs {@link DenseIndex#save(String, Progress)} on this receiver's executor. */
        public void save(String path, Progress progress) {
            block(DenseCalls.saving(path, progress));
        }

        /** Runs {@link DenseIndex#saveBuffer(Arena)} on this receiver's executor. */
        public Monolith saveBuffer(Arena arena) {
            return block(DenseCalls.serializing(arena, null));
        }

        /** Runs {@link DenseIndex#saveBuffer(Arena, Progress)} on this receiver's executor. */
        public Monolith saveBuffer(Arena arena, Progress progress) {
            return block(DenseCalls.serializing(arena, progress));
        }

        /** Runs {@link DenseIndex#cursor(Key...)} on this receiver's executor. */
        public DenseCursor cursor(Key... starts) {
            return block(DenseCalls.cursorOpening(index, executor, starts));
        }

        /** Runs {@link DenseIndex#searchStream(RowsView)} on this receiver's executor. */
        public DenseSearchStream searchStream(RowsView query) {
            return searchStream(query, 0);
        }

        /** Runs {@link DenseIndex#searchStream(RowsView, long)} on this receiver's executor. */
        public DenseSearchStream searchStream(RowsView query, long expansionFactor) {
            return block(DenseCalls.searchStreamOpening(index, executor, query, expansionFactor));
        }
    }

    /**
     * The receiver whose verbs submit their calls to an executor, each answering a {@link Deferred}
     * that owns what the call reads until final, and whose join answers the view the blocking verb
     * would, over the same borrowed arena. Its verbs are those of {@link DenseIndex}, which each
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

        /** Submits {@link DenseIndex#reserve}. Calls {@code usearch_dense_reserve}. */
        public Deferred<Void> reserve(long capacity) {
            return submit(DenseCalls.reserving(capacity));
        }

        /** Submits {@link DenseIndex#clear}. Calls {@code usearch_dense_clear}. */
        public Deferred<Void> clear() {
            return submit(DenseCalls.clearing());
        }

        /** Submits {@link DenseIndex#describe}. Calls {@code usearch_dense_describe}. */
        public Deferred<DenseIndexDescription> describe() {
            return submit(DenseCalls.describing());
        }

        /** Submits {@link DenseIndex#count}. Calls {@code usearch_dense_count}. */
        public Deferred<Long> count(Key key) {
            return submit(DenseCalls.counting(key));
        }

        /** Submits {@link DenseIndex#get}. Calls {@code usearch_dense_get}. */
        public Deferred<DenseVectors> get(Key key, Scalar outputsScalar, Arena arena) {
            return submit(DenseCalls.getting(key, outputsScalar, arena));
        }

        /** Submits {@link DenseIndex#insert(Key[], RowsView)}. Calls {@code usearch_dense_insert}. */
        public Deferred<Void> insert(Key[] keys, RowsView vectors) {
            return submit(DenseCalls.inserting(keys, vectors, new DenseInsertConfig()));
        }

        /** Submits {@link DenseIndex#insert(Key[], RowsView, DenseInsertConfig)}. Calls {@code usearch_dense_insert}. */
        public Deferred<Void> insert(Key[] keys, RowsView vectors, DenseInsertConfig config) {
            return submit(DenseCalls.inserting(keys, vectors, config));
        }

        /** Submits {@link DenseIndex#refine()}. Calls {@code usearch_dense_refine}. */
        public Deferred<RefineReport> refine() {
            return submit(DenseCalls.refining(new DenseRefineConfig()));
        }

        /** Submits {@link DenseIndex#refine(DenseRefineConfig)}. Calls {@code usearch_dense_refine}. */
        public Deferred<RefineReport> refine(DenseRefineConfig config) {
            return submit(DenseCalls.refining(config));
        }

        /** Submits {@link DenseIndex#refinePlanFinish}. Calls {@code usearch_dense_refine_plan_finish}. */
        public Deferred<RefineReport> refinePlanFinish(DenseRefinePlan plan) {
            return submit(DenseCalls.refinePlanFinishing(plan));
        }

        /** Submits {@link DenseIndex#health}. Calls {@code usearch_dense_health}. */
        public Deferred<DenseReachability> health() {
            return submit(DenseCalls.auditing());
        }

        /** Submits {@link DenseIndex#search(RowsView, long, Arena)}. Calls {@code usearch_dense_search}. */
        public Deferred<Matches> search(RowsView queries, long wanted, Arena arena) {
            return submit(DenseCalls.searching(queries, new DenseSearchConfig(wanted), arena));
        }

        /**
         * Submits {@link DenseIndex#search(RowsView, DenseSearchConfig, Arena)}. Calls
         * {@code usearch_dense_search}.
         */
        public Deferred<Matches> search(RowsView queries, DenseSearchConfig config, Arena arena) {
            return submit(DenseCalls.searching(queries, config, arena));
        }

        /** Submits {@link DenseIndex#maxsim(RowsView, long, long, Arena)}. Calls {@code usearch_dense_maxsim}. */
        public Deferred<DenseProducts> maxsim(RowsView queries, long rowsPerQuery, long wanted, Arena arena) {
            return submit(DenseCalls.scoring(queries, rowsPerQuery, new DenseMaxsimConfig(wanted), arena));
        }

        /**
         * Submits {@link DenseIndex#maxsim(RowsView, long, DenseMaxsimConfig, Arena)}. Calls
         * {@code usearch_dense_maxsim}.
         */
        public Deferred<DenseProducts> maxsim(
                RowsView queries, long rowsPerQuery, DenseMaxsimConfig config, Arena arena) {
            return submit(DenseCalls.scoring(queries, rowsPerQuery, config, arena));
        }

        /** Submits {@link DenseIndex#multiply(RowsView, Arena)}. Calls {@code usearch_dense_multiply}. */
        public Deferred<DenseProducts> multiply(RowsView left, Arena arena) {
            return submit(DenseCalls.multiplying(left, new DenseMultiplyConfig(), arena));
        }

        /**
         * Submits {@link DenseIndex#multiply(RowsView, DenseMultiplyConfig, Arena)}. Calls
         * {@code usearch_dense_multiply}.
         */
        public Deferred<DenseProducts> multiply(RowsView left, DenseMultiplyConfig config, Arena arena) {
            return submit(DenseCalls.multiplying(left, config, arena));
        }

        /** Submits {@link DenseIndex#attend(RowsView, ValueLookup, Arena)}. Calls {@code usearch_dense_attend}. */
        public Deferred<DenseVectors> attend(RowsView queries, ValueLookup valueLookup, Arena arena) {
            return submit(DenseCalls.attending(queries, valueLookup, new DenseAttendConfig(), arena));
        }

        /**
         * Submits {@link DenseIndex#attend(RowsView, ValueLookup, DenseAttendConfig, Arena)}. Calls
         * {@code usearch_dense_attend}.
         */
        public Deferred<DenseVectors> attend(
                RowsView queries, ValueLookup valueLookup, DenseAttendConfig config, Arena arena) {
            return submit(DenseCalls.attending(queries, valueLookup, config, arena));
        }

        /**
         * Submits {@link DenseIndex#reshard(DenseIndex[], DenseIndex[], DenseReshardConfig)}. Calls
         * {@code usearch_dense_reshard}.
         */
        public Deferred<RefineReport> reshard(DenseIndex[] sources, DenseIndex[] targets, DenseReshardConfig config) {
            return submit(DenseCalls.resharding(sources, targets, config));
        }

        /** Submits {@link DenseIndex#reshardPlanFinish}. Calls {@code usearch_dense_reshard_plan_finish}. */
        public Deferred<RefineReport> reshardPlanFinish(DenseReshardPlan plan) {
            return submit(DenseCalls.reshardPlanFinishing(plan));
        }

        /** Submits {@link DenseIndex#sample(long, Arena, DenseSampleConfig)}. Calls {@code usearch_dense_sample}. */
        public Deferred<Sample> sample(long wanted, Arena arena, DenseSampleConfig config) {
            return submit(DenseCalls.sampling(wanted, arena, config));
        }

        /** Submits {@link DenseIndex#save(String)}. Calls {@code usearch_dense_save}. */
        public Deferred<Void> save(String path) {
            return submit(DenseCalls.saving(path, null));
        }

        /** Submits {@link DenseIndex#save(String, Progress)}. Calls {@code usearch_dense_save}. */
        public Deferred<Void> save(String path, Progress progress) {
            return submit(DenseCalls.saving(path, progress));
        }

        /** Submits {@link DenseIndex#saveBuffer(Arena)}. Calls {@code usearch_dense_save_buffer}. */
        public Deferred<Monolith> saveBuffer(Arena arena) {
            return submit(DenseCalls.serializing(arena, null));
        }

        /** Submits {@link DenseIndex#saveBuffer(Arena, Progress)}. Calls {@code usearch_dense_save_buffer}. */
        public Deferred<Monolith> saveBuffer(Arena arena, Progress progress) {
            return submit(DenseCalls.serializing(arena, progress));
        }
    }

    final Handle handle;

    private DenseIndex(MemorySegment created) {
        handle = new Handle(created, FREE);
    }

    /**
     * The receiver whose verbs block on {@code executor}, contended while another call holds it.
     * Binding-only.
     */
    public On on(Executor executor) {
        return new On(this, executor);
    }

    /** The receiver whose verbs submit to {@code executor}, each answering a {@link Deferred}. Binding-only. */
    public DeferredOn deferred(Executor executor) {
        return new DeferredOn(this, executor);
    }

    private <Result> Result block(Request<Result> request) {
        return Call.block(request.verb(), handle, null, request.fill());
    }

    /**
     * Claims capacity for a number of incoming vectors ahead of insertion, a hint, since insertions
     * grow the index on demand. The capacity is the desired total, including the current size.
     * Calls {@code usearch_dense_reserve}.
     *
     * @param capacity the desired total number of entries, including the current size.
     * @throws USearchException OUT_OF_MEMORY when the capacity cannot be claimed, READ_ONLY on a
     *     view, and OVERFLOW past the identity space's ceiling.
     */
    public void reserve(long capacity) {
        block(DenseCalls.reserving(capacity));
    }

    /**
     * Erases all entries, keeping the reserved capacity and the frozen geometry. No other call may
     * overlap it. Calls {@code usearch_dense_clear}.
     *
     * @throws USearchException READ_ONLY on a view.
     */
    public void clear() {
        block(DenseCalls.clearing());
    }

    /** Reports the frozen metadata and the current counters in one call. Calls {@code usearch_dense_describe}. */
    public DenseIndexDescription describe() {
        return block(DenseCalls.describing());
    }

    /**
     * How many entries are stored under a key; zero means the index holds none. Calls {@code usearch_dense_count}.
     *
     * @param key the key to look up.
     */
    public long count(Key key) {
        return block(DenseCalls.counting(key));
    }

    /**
     * Whether the index holds at least one entry under a key. Binding-only sugar over {@link #count}.
     *
     * @param key the key to look up.
     */
    public boolean contains(Key key) {
        return count(key) != 0;
    }

    /**
     * Retrieves every vector stored under a key into {@code arena}, one row per entry sharing it, in
     * insertion order, packed at the natural width of the kind asked for - one call, however many
     * there turn out to be. Calls {@code usearch_dense_get}.
     *
     * @param key the key to look up.
     * @param outputsScalar the scalar kind the rows are converted into.
     * @param arena the arena the output lands in, borrowed until the call returns.
     * @return the rows read in place, none when the key holds nothing.
     * @throws USearchException UNKNOWN_KIND on a scalar the engine cannot convert into, and
     *     CONTENDED when the arena serves another call.
     */
    public DenseVectors get(Key key, Scalar outputsScalar, Arena arena) {
        return block(DenseCalls.getting(key, outputsScalar, arena));
    }

    /**
     * Inserts a matrix of vectors under their keys, with every knob derived.
     *
     * @param keys one key per vector row.
     * @param vectors the rows to insert, of any scalar kind the engine converts.
     * @throws USearchException the statuses {@link #insert(Key[], RowsView, DenseInsertConfig)} names.
     */
    public void insert(Key[] keys, RowsView vectors) {
        block(DenseCalls.inserting(keys, vectors, new DenseInsertConfig()));
    }

    /**
     * Inserts a matrix of vectors under their keys - the native batch operation, where a single
     * vector is a one-row matrix. Overlaps safely with concurrent searches on the same handle.
     *
     * <p>
     * The input scalar type is independent of the stored quantization; the library converts on
     * ingest. A batch is all-or-nothing: a returning call published every entry, a failing one
     * published none. An accelerator fault severe enough to poison the device context is the sole
     * exception, reporting DEVICE and latching the index unusable rather than partially applied.
     * Calls {@code usearch_dense_insert}.
     *
     * @param keys one key per vector row.
     * @param vectors the rows to insert, of any scalar kind the engine converts.
     * @param config the insertion knobs.
     * @throws USearchException SHAPE_MISMATCH when the key count and the row count disagree or the
     *     rows are not as wide as the frozen geometry, UNKNOWN_KIND on a scalar the engine cannot convert, READ_ONLY on a view, and
     *     OUT_OF_MEMORY when the batch cannot be accommodated.
     */
    public void insert(Key[] keys, RowsView vectors, DenseInsertConfig config) {
        block(DenseCalls.inserting(keys, vectors, config));
    }

    /**
     * Re-selects the neighbors of every stored entry, with every knob derived.
     *
     * @throws USearchException the statuses {@link #refine(DenseRefineConfig)} names.
     */
    public RefineReport refine() {
        return block(DenseCalls.refining(new DenseRefineConfig()));
    }

    /**
     * Re-selects the neighbors of every stored entry against the finished graph, improving neighbors
     * chosen when the graph around them was still incomplete.
     *
     * <p>
     * Searches may run throughout on a RAM handle, since re-selection publishes one node at a time
     * and never removes one; insertions may not. Re-selecting a settled node changes nothing, so a
     * caller loops until the report's {@code changed} falls below whatever rate it considers settled.
     * Calls {@code usearch_dense_refine}.
     *
     * @param config the refinement knobs.
     * @return the nodes re-selected and changed, the islands found and reconnected, and the audit
     *     closing the heal; {@code changed} at zero means the graph has settled.
     * @throws USearchException READ_ONLY on a view.
     */
    public RefineReport refine(DenseRefineConfig config) {
        return block(DenseCalls.refining(config));
    }

    /**
     * Closes the round a cancel left in hand, as a stop at its barrier would, and frees the plan
     * whatever the call answers, reopening the index; the plan opens afresh on its next call. Calls
     * {@code usearch_dense_refine_plan_finish}.
     *
     * @param plan the plan, opened on this index and executor.
     * @return what the whole plan did to the graph.
     * @throws USearchException INVALID_ARGUMENT on a plan no call opened, or one opened on another
     *     index or executor.
     */
    public RefineReport refinePlanFinish(DenseRefinePlan plan) {
        return block(DenseCalls.refinePlanFinishing(plan));
    }

    /**
     * Audits how much of the graph a search reaches: forward reachability from as many landmarks as
     * one search draws, swept level by level. Read-only, so concurrent insertions merely leave fresh
     * entries unaudited. Calls {@code usearch_dense_health}.
     */
    public DenseReachability health() {
        return block(DenseCalls.auditing());
    }

    /**
     * Resolves a matrix of queries into up to {@code wanted} nearest matches apiece, with every other
     * knob derived.
     *
     * @param queries the query rows, as wide as the frozen geometry.
     * @param wanted the most matches kept per query.
     * @param arena the arena the output lands in, borrowed until the call returns.
     * @throws USearchException the statuses {@link #search(RowsView, DenseSearchConfig, Arena)} names.
     */
    public Matches search(RowsView queries, long wanted, Arena arena) {
        return block(DenseCalls.searching(queries, new DenseSearchConfig(wanted), arena));
    }

    /**
     * Resolves a matrix of queries into {@code arena} - the native batch operation, parallelized
     * internally. Never writes to the graph, so any number of these may overlap with concurrent
     * insertions. Calls {@code usearch_dense_search}.
     *
     * @param queries the query rows, as wide as the frozen geometry.
     * @param config the wanted count and the other search knobs.
     * @param arena the arena the output lands in, borrowed until the call returns.
     * @throws USearchException SHAPE_MISMATCH when the rows are not as wide as the frozen geometry,
     *     UNKNOWN_KIND on a scalar the engine cannot convert, and CONTENDED when the arena serves
     *     another call.
     */
    public Matches search(RowsView queries, DenseSearchConfig config, Arena arena) {
        return block(DenseCalls.searching(queries, config, arena));
    }

    /**
     * Late-interaction scoring keeping up to {@code wanted} keys per query, zero keeping every key a
     * query scored, with every other knob derived.
     *
     * @param queries the query rows, as wide as the frozen geometry.
     * @param rowsPerQuery the token rows every query carries.
     * @param wanted the most keys kept per query, zero keeping every key it scored.
     * @param arena the arena the output lands in, borrowed until the call returns.
     * @throws USearchException the statuses {@link #maxsim(RowsView, long, DenseMaxsimConfig, Arena)}
     *     names.
     */
    public DenseProducts maxsim(RowsView queries, long rowsPerQuery, long wanted, Arena arena) {
        return block(DenseCalls.scoring(queries, rowsPerQuery, new DenseMaxsimConfig(wanted), arena));
    }

    /**
     * Late-interaction scoring of multi-row queries against multi-vector documents: every query
     * row takes its best match within a key, and those maxima sum to that key's score. A document
     * is one key holding its token rows, so the wanted count is the "k" over keys rather than over
     * rows.
     *
     * <p>
     * Every query carries the same number of rows, so a padded token batch applies as is, and the
     * query count is the row count divided by that. Approximate in the same way a search is: a key
     * can score below its exact value but never above it. Runs on the inner-product kinds alone.
     * Each query's keys land in {@code arena} largest score first, as one ragged output like
     * {@link #multiply}'s. Calls {@code usearch_dense_maxsim}.
     *
     * @param queries the query rows, as wide as the frozen geometry.
     * @param rowsPerQuery the token rows every query carries.
     * @param config the wanted count and the other scoring knobs.
     * @param arena the arena the output lands in, borrowed until the call returns.
     * @throws USearchException CONFIG_MISMATCH on a graph frozen under a distance kind,
     *     SHAPE_MISMATCH when the token rows do not divide into whole queries or the rows are not as
     *     wide as the frozen geometry, and CONTENDED when the arena serves another call.
     */
    public DenseProducts maxsim(RowsView queries, long rowsPerQuery, DenseMaxsimConfig config, Arena arena) {
        return block(DenseCalls.scoring(queries, rowsPerQuery, config, arena));
    }

    /**
     * Approximate sparse product keeping every pair, with every other knob derived.
     *
     * @param left the rows multiplied against the indexed vectors.
     * @param arena the arena the output lands in, borrowed until the call returns.
     * @throws USearchException the statuses {@link #multiply(RowsView, DenseMultiplyConfig, Arena)}
     *     names.
     */
    public DenseProducts multiply(RowsView left, Arena arena) {
        return block(DenseCalls.multiplying(left, new DenseMultiplyConfig(), arena));
    }

    /**
     * Approximate sparse product of a matrix against the indexed vectors - the linear-algebra sibling
     * of {@link #search}, reporting every pair the traversal scored rather than only the nearest.
     * Pairs arrive in traversal order, and the pairs the walk never reaches are the low and negative
     * products. Runs on the inner-product kinds alone.
     *
     * <p>
     * Only the call knows how many pairs it scores, so the keys, the products and their offsets land
     * in {@code arena} as one ragged output. Calls {@code usearch_dense_multiply}.
     *
     * @param left the rows multiplied against the indexed vectors.
     * @param config the product knobs.
     * @param arena the arena the output lands in, borrowed until the call returns.
     * @throws USearchException CONFIG_MISMATCH on a graph frozen under a distance kind,
     *     SHAPE_MISMATCH when the rows are not as wide as the frozen geometry, and CONTENDED when the
     *     arena serves another call.
     */
    public DenseProducts multiply(RowsView left, DenseMultiplyConfig config, Arena arena) {
        return block(DenseCalls.multiplying(left, config, arena));
    }

    /**
     * Softmax attention with every knob derived.
     *
     * @param queries the query rows, as wide as the frozen geometry.
     * @param valueLookup resolves every scored key to its value row.
     * @param arena the arena the output lands in, borrowed until the call returns.
     * @throws USearchException the statuses
     *     {@link #attend(RowsView, ValueLookup, DenseAttendConfig, Arena)} names.
     */
    public DenseVectors attend(RowsView queries, ValueLookup valueLookup, Arena arena) {
        return block(DenseCalls.attending(queries, valueLookup, new DenseAttendConfig(), arena));
    }

    /**
     * Softmax attention of every query row over every indexed key its walk scores, folded as Flash
     * Attention folds key tiles, with no top-k cut, writing one row per query into {@code arena}
     * already divided by the partition. Runs on the inner-product kinds alone, since a softmax needs
     * logits.
     *
     * <p>
     * The index holds one layer's keys for one key-value head and no values, so the lookup resolves
     * every scored key to its value row. Calls {@code usearch_dense_attend}.
     *
     * @param queries the query rows, as wide as the frozen geometry.
     * @param valueLookup resolves every scored key to its value row.
     * @param config the attention knobs.
     * @param arena the arena the output lands in, borrowed until the call returns.
     * @throws USearchException SHAPE_MISMATCH when a scored key resolves to no row or the head count
     *     does not divide the rows, CONFIG_MISMATCH on a graph frozen under a distance kind, and
     *     CONTENDED when the arena serves another call.
     */
    public DenseVectors attend(RowsView queries, ValueLookup valueLookup, DenseAttendConfig config, Arena arena) {
        return block(DenseCalls.attending(queries, valueLookup, config, arena));
    }

    /**
     * Moves the entries of {@code sources} into {@code targets} on this receiver's executor, keeping
     * their links rather than re-inserting their vectors, then runs a round on every target that took
     * entries. Several sources into one target among them gather into it; one source into several
     * targets cuts it into one graph-local part per target; one index as its only source and target
     * recuts its own shards toward the configured placement. Every index stays the caller's: a source
     * the targets do not name ends empty and live. This receiver's own index takes part only where it
     * is named. Calls {@code usearch_dense_reshard}.
     *
     * @param sources the indexes whose entries move, each named once.
     * @param targets the indexes the entries move into, each a source itself or empty.
     * @param config the placement, the plan, and the knobs of the closing rounds.
     * @return what the closing rounds did to the graph.
     * @throws USearchException READ_ONLY on an NVMe view, INVALID_ARGUMENT on no sources or targets or
     *     an index named twice, CONFIG_MISMATCH on an index of another residency, scalar, metric or
     *     geometry, or a target neither a source nor empty, and UNSUPPORTED on several sources into
     *     several targets.
     */
    public RefineReport reshard(DenseIndex[] sources, DenseIndex[] targets, DenseReshardConfig config) {
        return block(DenseCalls.resharding(sources, targets, config));
    }

    /**
     * Closes the rounds a cancel stopped and frees the plan whatever the call answers, reopening
     * every index it names; a plan whose entries have not all moved runs no round. Calls
     * {@code usearch_dense_reshard_plan_finish}.
     *
     * @param plan the plan, opened with this receiver's executor.
     * @return what the whole plan did to the graph.
     * @throws USearchException INVALID_ARGUMENT on a plan no call opened, or one opened with another
     *     executor.
     */
    public RefineReport reshardPlanFinish(DenseReshardPlan plan) {
        return block(DenseCalls.reshardPlanFinishing(plan));
    }

    /**
     * Draws keys around the focus, or across the whole index, each hanging from a shown or earlier
     * draw and placed in three dimensions beside the shown ones - so a view grows one connected
     * sample as it zooms. Writes up to {@code wanted} draws, and a larger count under the same seed
     * draws a superset; only a RAM index samples yet. Calls {@code usearch_dense_sample}.
     *
     * @param wanted the most draws written.
     * @param arena the arena the output lands in, borrowed until the call returns.
     * @param config the focus, seed and other sampling knobs.
     * @throws USearchException UNSUPPORTED on a GPU or NVMe handle, and CONTENDED when the arena
     *     serves another call.
     */
    public Sample sample(long wanted, Arena arena, DenseSampleConfig config) {
        return block(DenseCalls.sampling(wanted, arena, config));
    }

    /**
     * Writes the index under a path, in the layout its extension picks - one {@code .safegraphs}
     * monolith, or a {@code .safegraphs.json} manifest beside its shards. Demands a mutation-free
     * window; concurrent readers keep searching throughout. Calls {@code usearch_dense_save}.
     *
     * @param path the destination; its extension picks the layout.
     * @throws USearchException IO when the destination cannot be written - which leaves a file
     *     without its header, for the caller to remove.
     */
    public void save(String path) {
        block(DenseCalls.saving(path, null));
    }

    /**
     * Writes the index under a path, as {@link #save(String)} does, reporting each shard written.
     *
     * @param path the destination; its extension picks the layout.
     * @param progress hears each shard written, and may cancel the write.
     * @throws USearchException the statuses {@link #save(String)} names, and CANCELLED once the
     *     progress handler stopped it.
     */
    public void save(String path, Progress progress) {
        block(DenseCalls.saving(path, progress));
    }

    /**
     * Writes the index as one monolith - every shard, entries in placement order - the same bytes
     * {@link #save} writes to a {@code .safegraphs} path, into {@code arena}. Calls
     * {@code usearch_dense_save_buffer}.
     *
     * @param arena the arena the output lands in, borrowed until the call returns.
     * @throws USearchException CONTENDED when the arena serves another call.
     */
    public Monolith saveBuffer(Arena arena) {
        return block(DenseCalls.serializing(arena, null));
    }

    /**
     * Writes the index as one monolith into {@code arena}, as {@link #saveBuffer(Arena)} does,
     * reporting each shard written.
     *
     * @param arena the arena the output lands in, borrowed until the call returns.
     * @param progress hears each shard written, and may cancel the write.
     * @throws USearchException the statuses {@link #saveBuffer(Arena)} names, and CANCELLED once the
     *     progress handler stopped it.
     */
    public Monolith saveBuffer(Arena arena, Progress progress) {
        return block(DenseCalls.serializing(arena, progress));
    }

    /**
     * Opens a walk over the current contents, pulled on this receiver's executor. Each shard's
     * publication watermark is read as the walk enters it, so an entry once passed is never revisited.
     * Without {@code starts} it yields every entry as stored; with them, the start keys' own entries
     * first in order, then the entries those link to breadth-first, each once. Calls
     * {@code usearch_dense_cursor_init}.
     *
     * @param starts the keys whose entries the walk starts from, or none to yield every entry.
     */
    public DenseCursor cursor(Key... starts) {
        return block(DenseCalls.cursorOpening(this, null, starts));
    }

    /**
     * Opens a resumable search for one query row, with the breadth of the candidates derived.
     *
     * @param query the query row; only the first row of the view is read.
     * @throws USearchException the statuses {@link #searchStream(RowsView, long)} names.
     */
    public DenseSearchStream searchStream(RowsView query) {
        return searchStream(query, 0);
    }

    /**
     * Opens a resumable search for one query row, pulled on this receiver's executor: best-first
     * traversal is naturally pausable, so callers that post-filter aggressively pull until satisfied
     * instead of guessing the wanted count upfront. Calls {@code usearch_dense_search_stream_init}.
     *
     * @param query the query row; only the first row of the view is read.
     * @param expansionFactor the breadth of the candidates, zero to derive it.
     * @throws USearchException UNSUPPORTED where no residency implements the stream yet.
     */
    public DenseSearchStream searchStream(RowsView query, long expansionFactor) {
        return block(DenseCalls.searchStreamOpening(this, null, query, expansionFactor));
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
            call.ram(call.arguments, INIT_RAM_RAM, memoryDomains, options.cpuClassCeiling);
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
            call.vram(call.arguments, INIT_VRAM_VRAM, devices, options.gpuClassCeiling);
        });
    }

    /**
     * Copies a checkpoint into a new index in host memory on the caller's thread, over every memory
     * domain, with no bound on entries per key.
     *
     * @throws USearchException the statuses {@link #loadRam(String, Executor, int[], CPUClass, long, Progress)}
     *     names.
     */
    public static DenseIndex loadRam(String path) {
        return loadRam(path, null, new int[0], null, 0, null);
    }

    /**
     * Copies a checkpoint into a new, mutable index in host memory, shard by shard. The geometry and
     * metric come from the checkpoint's own headers, and part {@code p} lands whole on shard
     * {@code p} modulo the shard count. Calls {@code usearch_dense_load_ram}.
     *
     * @param executor the executor the read fans out over, or {@code null} for the caller's thread.
     * @param memoryDomains NUMA node ids to place the parts on; an empty list derives every node.
     * @param cpuClassCeiling the newest class dispatch may pick, intersected with what the machine
     *     runs, so a class the machine lacks narrows rather than refuses; {@code null} means no
     *     ceiling.
     * @param valuesPerKey the upper bound on entries sharing one key, which no checkpoint records;
     *     zero is unbounded.
     * @param progress hears each part read and may cancel the load, or {@code null} for none.
     * @throws USearchException MALFORMED on a checkpoint that ends early, and IO when the file cannot
     *     be read.
     */
    public static DenseIndex loadRam(
            String path,
            @Nullable Executor executor,
            int[] memoryDomains,
            @Nullable CPUClass cpuClassCeiling,
            long valuesPerKey,
            @Nullable Progress progress) {
        return create(LOAD_RAM, executor, call -> {
            call.path(LOAD_RAM_SOURCE, path);
            call.ram(call.arguments, LOAD_RAM_RAM, memoryDomains, cpuClassCeiling);
            call.size(LOAD_RAM_VALUES_PER_KEY, valuesPerKey);
            call.progress(LOAD_RAM_PROGRESS, progress);
        });
    }

    /**
     * Uploads a checkpoint into a new index in device memory on the default stream of one free
     * device, with no bound on entries per key.
     *
     * @throws USearchException the statuses {@link #loadVram(String, Executor, int[], GPUClass, long, Progress)}
     *     names.
     */
    public static DenseIndex loadVram(String path) {
        return loadVram(path, null, new int[0], null, 0, null);
    }

    /**
     * Copies a checkpoint into a new, mutable index in device memory, with the placement rules of
     * {@link #loadRam(String, Executor, int[], CPUClass, long, Progress)}. Calls {@code usearch_dense_load_vram}.
     *
     * @param executor the executor the upload runs on, or {@code null} for each device's default
     *     stream.
     * @param devices CUDA device ordinals to place the parts on; an empty list derives one free device.
     * @param gpuClassCeiling the newest class dispatch may pick, capped below the best class every
     *     named device loads; a ceiling no named device loads is refused as UNSUPPORTED; {@code
     *     null} means no ceiling.
     * @param valuesPerKey the upper bound on entries sharing one key; zero is unbounded.
     * @param progress hears each part uploaded and may cancel the load, or {@code null} for none.
     * @throws USearchException DEVICE when no device answers, and the statuses a RAM load names.
     */
    public static DenseIndex loadVram(
            String path,
            @Nullable Executor executor,
            int[] devices,
            @Nullable GPUClass gpuClassCeiling,
            long valuesPerKey,
            @Nullable Progress progress) {
        return create(LOAD_VRAM, executor, call -> {
            call.path(LOAD_VRAM_SOURCE, path);
            call.vram(call.arguments, LOAD_VRAM_VRAM, devices, gpuClassCeiling);
            call.size(LOAD_VRAM_VALUES_PER_KEY, valuesPerKey);
            call.progress(LOAD_VRAM_PROGRESS, progress);
        });
    }

    /**
     * Copies a monolith from a buffer into a new index in host memory on the caller's thread, with
     * no bound on entries per key.
     *
     * @throws USearchException the statuses
     *     {@link #loadRamBuffer(MemorySegment, Executor, int[], CPUClass, long, Progress)} names.
     */
    public static DenseIndex loadRamBuffer(MemorySegment buffer) {
        return loadRamBuffer(buffer, null, new int[0], null, 0, null);
    }

    /**
     * Copies a monolith from a buffer into a new, mutable index in host memory, as
     * {@link #loadRam(String, Executor, int[], CPUClass, long, Progress)} reads a path; the monolith lands
     * whole on the first shard. A native buffer is read in place, a heap one is copied first. Calls
     * {@code usearch_dense_load_ram_buffer}.
     *
     * @throws USearchException MALFORMED on a monolith that ends early.
     */
    public static DenseIndex loadRamBuffer(
            MemorySegment buffer,
            @Nullable Executor executor,
            int[] memoryDomains,
            @Nullable CPUClass cpuClassCeiling,
            long valuesPerKey,
            @Nullable Progress progress) {
        return create(LOAD_RAM_BUFFER, executor, call -> {
            call.bytes(LOAD_RAM_SOURCE, buffer);
            call.ram(call.arguments, LOAD_RAM_RAM, memoryDomains, cpuClassCeiling);
            call.size(LOAD_RAM_VALUES_PER_KEY, valuesPerKey);
            call.progress(LOAD_RAM_PROGRESS, progress);
        });
    }

    /**
     * Uploads a monolith from a buffer into a new index in device memory on the default stream of one
     * free device, with no bound on entries per key.
     *
     * @throws USearchException the statuses
     *     {@link #loadVramBuffer(MemorySegment, Executor, int[], GPUClass, long, Progress)} names.
     */
    public static DenseIndex loadVramBuffer(MemorySegment buffer) {
        return loadVramBuffer(buffer, null, new int[0], null, 0, null);
    }

    /**
     * Copies a monolith from a buffer into a new, mutable index in device memory, as
     * {@link #loadRamBuffer(MemorySegment, Executor, int[], CPUClass, long, Progress)} does in host memory.
     * Calls {@code usearch_dense_load_vram_buffer}.
     *
     * @throws USearchException DEVICE when no device answers, and MALFORMED on a monolith that ends
     *     early.
     */
    public static DenseIndex loadVramBuffer(
            MemorySegment buffer,
            @Nullable Executor executor,
            int[] devices,
            @Nullable GPUClass gpuClassCeiling,
            long valuesPerKey,
            @Nullable Progress progress) {
        return create(LOAD_VRAM_BUFFER, executor, call -> {
            call.bytes(LOAD_VRAM_SOURCE, buffer);
            call.vram(call.arguments, LOAD_VRAM_VRAM, devices, gpuClassCeiling);
            call.size(LOAD_VRAM_VALUES_PER_KEY, valuesPerKey);
            call.progress(LOAD_VRAM_PROGRESS, progress);
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
        return create(VIEW_RAM, null, call -> call.path(VIEW_RAM_PATH, path));
    }

    /**
     * Serves a checkpoint in place from NVMe drives on the caller's thread, each part where its
     * manifest points.
     *
     * @throws USearchException the statuses {@link #viewNvme(String, Executor, String[], CPUClass, int[])} names.
     */
    public static DenseIndex viewNvme(String path) {
        return viewNvme(path, null, new String[0], null, null);
    }

    /**
     * Serves a checkpoint in place from NVMe drives, read through the executor's submission rings, as
     * a read-only index. Calls {@code usearch_dense_view_nvme}.
     *
     * @param executor the NVMe executor whose rings read the headers, or {@code null} for the caller's
     *     thread.
     * @param paths devices or directories to serve the parts from; an empty list serves each where
     *     its manifest points.
     * @param cpuClassCeiling the newest class dispatch may pick, intersected with what the machine
     *     runs, so a class the machine lacks narrows rather than refuses; {@code null} means no
     *     ceiling.
     * @param keyTableDomains NUMA node IDs of a key table held in RAM, which counts, reads, samples,
     *     MaxSim and hinted searches need; an empty list derives them, and {@code null} holds none.
     * @throws USearchException IO when the checkpoint cannot be opened.
     */
    public static DenseIndex viewNvme(
            String path,
            @Nullable Executor executor,
            String[] paths,
            @Nullable CPUClass cpuClassCeiling,
            int @Nullable [] keyTableDomains) {
        return create(VIEW_NVME, executor, call -> serving(call, path, paths, cpuClassCeiling, keyTableDomains));
    }

    /**
     * Opens a checkpoint's manifest from NVMe drives as a writable index: it answers every read a view
     * answers, and {@link DenseIndex#refine} rewrites its parts one at a time. Takes what
     * {@link #viewNvme(String, Executor, String[], CPUClass, int[])} takes, with {@code paths} naming
     * none. Calls {@code usearch_dense_load_nvme}.
     *
     * @throws USearchException UNSUPPORTED on a lone monolith, which has no manifest to replace, or on
     *     paths naming any; CONTENDED when another writable handle holds the manifest.
     */
    public static DenseIndex loadNvme(
            String path,
            @Nullable Executor executor,
            String[] paths,
            @Nullable CPUClass cpuClassCeiling,
            int @Nullable [] keyTableDomains) {
        return create(LOAD_NVME, executor, call -> serving(call, path, paths, cpuClassCeiling, keyTableDomains));
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
        return metadata(METADATA_READ, executor, call -> call.path(READ_SOURCE, path));
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
        return metadata(METADATA_READ_BUFFER, executor, call -> call.bytes(READ_SOURCE, buffer));
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
            MemorySegment created = call.output(java.util.Objects.requireNonNull(verb.index()), Library.POINTER);
            fill.accept(call);
            return () -> new DenseIndex(created.get(Library.POINTER, 0));
        });
    }

    /** Fills a view or a load of an NVMe checkpoint, with a key table in RAM unless its domains are {@code null}. */
    private static void serving(
            Call call,
            String path,
            String[] paths,
            @Nullable CPUClass cpuClassCeiling,
            int @Nullable [] keyTableDomains) {
        call.path(SERVE_NVME_PATH, path);
        call.nvme(call.arguments, SERVE_NVME_NVME, paths, cpuClassCeiling);
        if (keyTableDomains != null) {
            call.ram(call.output(SERVE_NVME_RAM, Library.RAM_OPTIONS), 0, keyTableDomains, cpuClassCeiling);
        }
    }

    private static DenseMetadata metadata(Verb verb, @Nullable Executor executor, Consumer<Call> fill) {
        return Call.block(verb, null, executor, call -> {
            fill.accept(call);
            MemorySegment metadata = call.output(READ_METADATA, METADATA);
            return () -> DenseMetadata.lift(metadata, 0);
        });
    }

    private static final MemoryLayout REFINE_OPTIONS = MemoryLayout.structLayout(
                    Library.SIZE.withName("expansion_factor"), Library.SIZE.withName("memory_limit"))
            .withName("refine");
    private static final MemoryLayout STATS = Library.POINTER.withName("stats");

    /** {@code usearch_dense_metadata_t}. */
    static final StructLayout METADATA = MemoryLayout.structLayout(
            Library.ENUM.withName("metric"),
            Library.ENUM.withName("quantization"),
            Library.SIZE.withName("dimensions"),
            Library.SIZE.withName("connectivity"));

    private static final VarHandle METADATA_METRIC = Library.field(METADATA, "metric");
    private static final VarHandle METADATA_QUANTIZATION = Library.field(METADATA, "quantization");
    private static final VarHandle METADATA_DIMENSIONS = Library.field(METADATA, "dimensions");
    private static final VarHandle METADATA_CONNECTIVITY = Library.field(METADATA, "connectivity");

    /** {@code usearch_dense_index_description_t}. */
    static final StructLayout DESCRIPTION = MemoryLayout.structLayout(
            METADATA.withName("metadata"),
            Library.SIZE.withName("size"),
            Library.SIZE.withName("capacity"),
            Library.SIZE.withName("memory_usage"),
            Library.ENUM.withName("cpu_class"),
            Library.ENUM.withName("gpu_class"),
            Library.SIZE.withName("shards_count"),
            Library.ENUM.withName("load_path"),
            Library.padding());

    private static final long DESCRIPTION_METADATA = Library.offset(DESCRIPTION, "metadata");
    private static final VarHandle DESCRIPTION_SIZE = Library.field(DESCRIPTION, "size");
    private static final VarHandle DESCRIPTION_CAPACITY = Library.field(DESCRIPTION, "capacity");
    private static final VarHandle DESCRIPTION_MEMORY_USAGE = Library.field(DESCRIPTION, "memory_usage");
    private static final VarHandle DESCRIPTION_CPU_CLASS = Library.field(DESCRIPTION, "cpu_class");
    private static final VarHandle DESCRIPTION_GPU_CLASS = Library.field(DESCRIPTION, "gpu_class");
    private static final VarHandle DESCRIPTION_SHARDS_COUNT = Library.field(DESCRIPTION, "shards_count");

    /** Spells a {@code usearch_cpu_class_t}, a {@code usearch_gpu_class_t} and a {@code usearch_refine_verdict_t}. */
    private static final MethodHandle CPU_CLASS_NAME = Library.namer("usearch_cpu_class_name");

    private static final MethodHandle GPU_CLASS_NAME = Library.namer("usearch_gpu_class_name");
    private static final MethodHandle VERDICT_NAME = Library.namer("usearch_refine_verdict_name");

    /** {@code usearch_reachability_t}. */
    static final StructLayout REACHABILITY = MemoryLayout.structLayout(
            Library.SIZE.withName("audited"),
            Library.SIZE.withName("reachable"),
            Library.SIZE.withName("depth_from_root"),
            Library.SIZE.withName("depth_to_root"),
            Library.SIZE.withName("unlinked"),
            Library.SIZE.withName("underlinked"),
            Library.SIZE.withName("beyond_most_populated_depth"),
            Library.SIZE.withName("unreached_islands"),
            Library.SIZE.withName("stranded_islands"),
            Library.SIZE.withName("components"));

    /** {@code usearch_refine_report_t}. */
    static final StructLayout REPORT = MemoryLayout.structLayout(
            Library.SIZE.withName("reselected"),
            Library.SIZE.withName("changed"),
            Library.SIZE.withName("islands_found"),
            Library.SIZE.withName("islands_reconnected"),
            Library.SIZE.withName("neighborhoods"),
            Library.SIZE.withName("widest_neighborhood"),
            Library.SIZE.withName("added_links"),
            Library.SIZE.withName("dropped_links"),
            Library.ENUM.withName("verdict"),
            Library.padding(),
            REACHABILITY.withName("opening"),
            REACHABILITY.withName("reachability"));

    private static final VarHandle REACHABILITY_AUDITED = Library.field(REACHABILITY, "audited");
    private static final VarHandle REACHABILITY_REACHABLE = Library.field(REACHABILITY, "reachable");
    private static final VarHandle REACHABILITY_DEPTH_FROM_ROOT = Library.field(REACHABILITY, "depth_from_root");
    private static final VarHandle REACHABILITY_DEPTH_TO_ROOT = Library.field(REACHABILITY, "depth_to_root");
    private static final VarHandle REACHABILITY_UNLINKED = Library.field(REACHABILITY, "unlinked");
    private static final VarHandle REACHABILITY_UNDERLINKED = Library.field(REACHABILITY, "underlinked");
    private static final VarHandle REACHABILITY_BEYOND_MOST_POPULATED_DEPTH =
            Library.field(REACHABILITY, "beyond_most_populated_depth");
    private static final VarHandle REACHABILITY_UNREACHED_ISLANDS = Library.field(REACHABILITY, "unreached_islands");
    private static final VarHandle REACHABILITY_STRANDED_ISLANDS = Library.field(REACHABILITY, "stranded_islands");
    private static final VarHandle REACHABILITY_COMPONENTS = Library.field(REACHABILITY, "components");

    private static final VarHandle REPORT_RESELECTED = Library.field(REPORT, "reselected");
    private static final VarHandle REPORT_CHANGED = Library.field(REPORT, "changed");
    private static final VarHandle REPORT_ISLANDS_FOUND = Library.field(REPORT, "islands_found");
    private static final VarHandle REPORT_ISLANDS_RECONNECTED = Library.field(REPORT, "islands_reconnected");
    private static final VarHandle REPORT_NEIGHBORHOODS = Library.field(REPORT, "neighborhoods");
    private static final VarHandle REPORT_WIDEST_NEIGHBORHOOD = Library.field(REPORT, "widest_neighborhood");
    private static final VarHandle REPORT_ADDED_LINKS = Library.field(REPORT, "added_links");
    private static final VarHandle REPORT_DROPPED_LINKS = Library.field(REPORT, "dropped_links");
    private static final VarHandle REPORT_VERDICT = Library.field(REPORT, "verdict");
    private static final long REPORT_OPENING = Library.offset(REPORT, "opening");
    private static final long REPORT_REACHABILITY = Library.offset(REPORT, "reachability");

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
                Library.placement(placement));
    }

    static final Verb INIT_RAM = new Verb("usearch_dense_init_ram", initializing("ram"));
    static final Verb INIT_VRAM = new Verb("usearch_dense_init_vram", initializing("vram"));
    static final VarHandle INIT_METRIC = INIT_RAM.field("metric");
    static final VarHandle INIT_QUANTIZATION = INIT_RAM.field("quantization");
    static final VarHandle INIT_DIMENSIONS = INIT_RAM.field("dimensions");
    static final VarHandle INIT_CONNECTIVITY = INIT_RAM.field("connectivity");
    static final VarHandle INIT_VALUES_PER_KEY = INIT_RAM.field("values_per_key");
    static final long INIT_RAM_RAM = INIT_RAM.at("ram");
    static final long INIT_VRAM_VRAM = INIT_VRAM.at("vram");

    /** {@code usearch_dense_load_ram_t} and its buffer twin, reading {@code source}. */
    private static final StructLayout LOADING_RAM = Library.opening(
            Library.padding(),
            Library.BYTES_VIEW.withName("source"),
            Library.placement("ram"),
            Library.SIZE.withName("connectivity"),
            Library.SIZE.withName("values_per_key"),
            Library.PROGRESS);

    static final Verb LOAD_RAM = new Verb("usearch_dense_load_ram", LOADING_RAM);
    static final Verb LOAD_RAM_BUFFER = new Verb("usearch_dense_load_ram_buffer", LOADING_RAM);
    static final long LOAD_RAM_SOURCE = LOAD_RAM.at("source");
    static final long LOAD_RAM_RAM = LOAD_RAM.at("ram");
    static final VarHandle LOAD_RAM_VALUES_PER_KEY = LOAD_RAM.field("values_per_key");
    static final long LOAD_RAM_PROGRESS = LOAD_RAM.at("progress");

    /** {@code usearch_dense_load_vram_t} and its buffer twin, reading {@code source}. */
    private static final StructLayout LOADING_VRAM = Library.opening(
            Library.padding(),
            Library.BYTES_VIEW.withName("source"),
            Library.placement("vram"),
            Library.SIZE.withName("values_per_key"),
            Library.PROGRESS);

    static final Verb LOAD_VRAM = new Verb("usearch_dense_load_vram", LOADING_VRAM);
    static final Verb LOAD_VRAM_BUFFER = new Verb("usearch_dense_load_vram_buffer", LOADING_VRAM);
    static final long LOAD_VRAM_SOURCE = LOAD_VRAM.at("source");
    static final long LOAD_VRAM_VRAM = LOAD_VRAM.at("vram");
    static final VarHandle LOAD_VRAM_VALUES_PER_KEY = LOAD_VRAM.field("values_per_key");
    static final long LOAD_VRAM_PROGRESS = LOAD_VRAM.at("progress");

    static final Verb VIEW_RAM = new Verb(
            "usearch_dense_view_ram",
            MemoryLayout.structLayout(
                    Library.POINTER.withName("index"),
                    Library.POINTER.withName("outcome"),
                    Library.BYTES_VIEW.withName("path")));

    static final long VIEW_RAM_PATH = VIEW_RAM.at("path");

    /** {@code usearch_dense_view_nvme_t} and {@code usearch_dense_load_nvme_t}. */
    private static final StructLayout SERVING_NVME = Library.opening(
            Library.padding(),
            Library.BYTES_VIEW.withName("path"),
            Library.placement("nvme"),
            Library.POINTER.withName("ram"));

    static final Verb VIEW_NVME = new Verb("usearch_dense_view_nvme", SERVING_NVME);
    static final Verb LOAD_NVME = new Verb("usearch_dense_load_nvme", SERVING_NVME);
    static final long SERVE_NVME_PATH = VIEW_NVME.at("path");
    static final long SERVE_NVME_NVME = VIEW_NVME.at("nvme");
    static final VarHandle SERVE_NVME_RAM = VIEW_NVME.field("ram");

    static final Verb RESERVE =
            new Verb("usearch_dense_reserve", Library.opening(Library.padding(), Library.SIZE.withName("capacity")));

    static final VarHandle RESERVE_CAPACITY = RESERVE.field("capacity");

    static final Verb CLEAR = new Verb("usearch_dense_clear", Library.opening(Library.padding()));

    static final Verb DESCRIBE = new Verb(
            "usearch_dense_describe", Library.opening(Library.padding(), Library.POINTER.withName("description")));

    static final VarHandle DESCRIBE_DESCRIPTION = DESCRIBE.field("description");

    static final Verb INSERT = new Verb(
            "usearch_dense_insert",
            Library.opening(
                    Library.padding(),
                    Library.SIZE.withName("count"),
                    Library.KEYS_VIEW.withName("keys"),
                    Library.ROWS_VIEW.withName("vectors"),
                    Library.SIZE.withName("expansion_factor"),
                    Library.PROGRESS,
                    STATS));

    static final VarHandle INSERT_COUNT = INSERT.field("count");
    static final long INSERT_KEYS = INSERT.at("keys");
    static final long INSERT_VECTORS = INSERT.at("vectors");
    static final VarHandle INSERT_EXPANSION_FACTOR = INSERT.field("expansion_factor");
    static final long INSERT_PROGRESS = INSERT.at("progress");
    static final VarHandle INSERT_STATS = INSERT.field("stats");

    static final Verb REFINE = new Verb(
            "usearch_dense_refine",
            Library.opening(
                    Library.padding(),
                    Library.POINTER.withName("report"),
                    REFINE_OPTIONS,
                    Library.PROGRESS,
                    STATS,
                    Library.POINTER.withName("plan"),
                    Library.POINTER.withName("phase")));

    static final VarHandle REFINE_REPORT = REFINE.field("report");
    static final VarHandle REFINE_EXPANSION_FACTOR = REFINE.field("refine", "expansion_factor");
    static final VarHandle REFINE_MEMORY_LIMIT = REFINE.field("refine", "memory_limit");
    static final long REFINE_PROGRESS = REFINE.at("progress");
    static final VarHandle REFINE_STATS = REFINE.field("stats");
    static final VarHandle REFINE_PLAN = REFINE.field("plan");
    static final VarHandle REFINE_PHASE = REFINE.field("phase");

    static final Verb REFINE_PLAN_FINISH = new Verb(
            "usearch_dense_refine_plan_finish",
            Library.opening(Library.padding(), Library.POINTER.withName("plan"), Library.POINTER.withName("report")));

    static final VarHandle REFINE_PLAN_FINISH_PLAN = REFINE_PLAN_FINISH.field("plan");
    static final VarHandle REFINE_PLAN_FINISH_REPORT = REFINE_PLAN_FINISH.field("report");

    static final Verb HEALTH = new Verb(
            "usearch_dense_health", Library.opening(Library.padding(), Library.POINTER.withName("reachability")));

    static final VarHandle HEALTH_REACHABILITY = HEALTH.field("reachability");

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
                    Library.PREDICATE,
                    Library.POINTER.withName("hints"),
                    Library.PROGRESS,
                    STATS));

    static final VarHandle SEARCH_COUNT = SEARCH.field("count");
    static final long SEARCH_QUERIES = SEARCH.at("queries");
    static final VarHandle SEARCH_MATCHES = SEARCH.field("matches");
    static final VarHandle SEARCH_ARENA = SEARCH.field("arena");
    static final VarHandle SEARCH_WANTED = SEARCH.field("wanted");
    static final VarHandle SEARCH_EXPANSION_FACTOR = SEARCH.field("expansion_factor");
    static final VarHandle SEARCH_TRAVERSAL = SEARCH.field("traversal");
    static final long SEARCH_PREDICATE = SEARCH.at("predicate");
    static final long SEARCH_PROGRESS = SEARCH.at("progress");
    static final VarHandle SEARCH_STATS = SEARCH.field("stats");

    // The key follows the completion unpadded, since sixteen bytes align to one.
    static final Verb COUNT = new Verb(
            "usearch_dense_count",
            Library.opening(Library.KEY.withName("key"), Library.padding(), Library.POINTER.withName("entries")));

    static final long COUNT_KEY = COUNT.at("key");
    static final VarHandle COUNT_ENTRIES = COUNT.field("entries");

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

    static final long GET_KEY = GET.at("key");
    static final VarHandle GET_VECTORS = GET.field("vectors");
    static final VarHandle GET_WRITTEN = GET.field("written");
    static final VarHandle GET_ARENA = GET.field("arena");
    static final VarHandle GET_OUTPUTS_SCALAR = GET.field("outputs_scalar");

    static final Verb RESHARD = new Verb(
            "usearch_dense_reshard",
            MemoryLayout.structLayout(
                    Library.SIZE.withName("sources_count"),
                    Library.POINTER.withName("sources"),
                    Library.SIZE.withName("targets_count"),
                    Library.POINTER.withName("targets"),
                    Library.POINTER.withName("executor"),
                    Library.POINTER.withName("outcome"),
                    Library.ENUM.withName("completion"),
                    Library.padding(),
                    Library.POINTER.withName("ram"),
                    Library.POINTER.withName("vram"),
                    Library.POINTER.withName("nvme"),
                    Library.POINTER.withName("report"),
                    REFINE_OPTIONS,
                    ValueLayout.JAVA_LONG.withName("seed"),
                    Library.PROGRESS,
                    STATS,
                    Library.POINTER.withName("plan"),
                    Library.POINTER.withName("phase")));

    static final VarHandle RESHARD_SOURCES_COUNT = RESHARD.field("sources_count");
    static final VarHandle RESHARD_SOURCES = RESHARD.field("sources");
    static final VarHandle RESHARD_TARGETS_COUNT = RESHARD.field("targets_count");
    static final VarHandle RESHARD_TARGETS = RESHARD.field("targets");
    static final VarHandle RESHARD_RAM = RESHARD.field("ram");
    static final VarHandle RESHARD_VRAM = RESHARD.field("vram");
    static final VarHandle RESHARD_NVME = RESHARD.field("nvme");
    static final VarHandle RESHARD_REPORT = RESHARD.field("report");
    static final VarHandle RESHARD_EXPANSION_FACTOR = RESHARD.field("refine", "expansion_factor");
    static final VarHandle RESHARD_MEMORY_LIMIT = RESHARD.field("refine", "memory_limit");
    static final VarHandle RESHARD_SEED = RESHARD.field("seed");
    static final long RESHARD_PROGRESS = RESHARD.at("progress");
    static final VarHandle RESHARD_STATS = RESHARD.field("stats");
    static final VarHandle RESHARD_PLAN = RESHARD.field("plan");
    static final VarHandle RESHARD_PHASE = RESHARD.field("phase");

    static final Verb RESHARD_PLAN_FINISH = new Verb(
            "usearch_dense_reshard_plan_finish",
            MemoryLayout.structLayout(
                    Library.POINTER.withName("executor"),
                    Library.POINTER.withName("outcome"),
                    Library.ENUM.withName("completion"),
                    Library.padding(),
                    Library.POINTER.withName("plan"),
                    Library.POINTER.withName("report")));

    static final VarHandle RESHARD_PLAN_FINISH_PLAN = RESHARD_PLAN_FINISH.field("plan");
    static final VarHandle RESHARD_PLAN_FINISH_REPORT = RESHARD_PLAN_FINISH.field("report");

    static final Verb SAMPLE = new Verb(
            "usearch_dense_sample",
            Library.opening(
                    Library.padding(),
                    Library.SIZE.withName("shown_count"),
                    Library.KEYS_VIEW.withName("shown"),
                    Library.ROWS_VIEW.withName("shown_coordinates"),
                    Library.SIZE.withName("focus_count"),
                    Library.KEYS_VIEW.withName("focus"),
                    Library.POINTER.withName("keys"),
                    Library.POINTER.withName("entry_ordinals"),
                    Library.POINTER.withName("parents"),
                    Library.POINTER.withName("coordinates"),
                    Library.POINTER.withName("written"),
                    Library.POINTER.withName("arena"),
                    Library.SIZE.withName("wanted"),
                    ValueLayout.JAVA_LONG.withName("seed"),
                    Library.SIZE.withName("expansion_factor"),
                    Library.PROGRESS,
                    STATS));

    static final VarHandle SAMPLE_SHOWN_COUNT = SAMPLE.field("shown_count");
    static final long SAMPLE_SHOWN = SAMPLE.at("shown");
    static final long SAMPLE_SHOWN_COORDINATES = SAMPLE.at("shown_coordinates");
    static final VarHandle SAMPLE_FOCUS_COUNT = SAMPLE.field("focus_count");
    static final long SAMPLE_FOCUS = SAMPLE.at("focus");
    static final VarHandle SAMPLE_KEYS = SAMPLE.field("keys");
    static final VarHandle SAMPLE_ENTRY_ORDINALS = SAMPLE.field("entry_ordinals");
    static final VarHandle SAMPLE_PARENTS = SAMPLE.field("parents");
    static final VarHandle SAMPLE_COORDINATES = SAMPLE.field("coordinates");
    static final VarHandle SAMPLE_WRITTEN = SAMPLE.field("written");
    static final VarHandle SAMPLE_ARENA = SAMPLE.field("arena");
    static final VarHandle SAMPLE_WANTED = SAMPLE.field("wanted");
    static final VarHandle SAMPLE_SEED = SAMPLE.field("seed");
    static final VarHandle SAMPLE_EXPANSION_FACTOR = SAMPLE.field("expansion_factor");
    static final long SAMPLE_PROGRESS = SAMPLE.at("progress");
    static final VarHandle SAMPLE_STATS = SAMPLE.field("stats");

    static final Verb CURSOR_INIT = new Verb(
            "usearch_dense_cursor_init",
            MemoryLayout.structLayout(
                    Library.POINTER.withName("index"),
                    Library.POINTER.withName("outcome"),
                    Library.SIZE.withName("starts_count"),
                    Library.POINTER.withName("starts"),
                    Library.POINTER.withName("cursor")));

    static final VarHandle CURSOR_INIT_STARTS_COUNT = CURSOR_INIT.field("starts_count");
    static final VarHandle CURSOR_INIT_STARTS = CURSOR_INIT.field("starts");
    static final VarHandle CURSOR_INIT_CURSOR = CURSOR_INIT.field("cursor");

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

    private static final VarHandle CURSOR_NEXT_CURSOR = CURSOR_NEXT.field("cursor");
    private static final VarHandle CURSOR_NEXT_KEYS = CURSOR_NEXT.field("keys");
    private static final VarHandle CURSOR_NEXT_LINKS = CURSOR_NEXT.field("links");
    private static final VarHandle CURSOR_NEXT_WRITTEN = CURSOR_NEXT.field("written");
    private static final VarHandle CURSOR_NEXT_ARENA = CURSOR_NEXT.field("arena");
    private static final VarHandle CURSOR_NEXT_WANTED = CURSOR_NEXT.field("wanted");

    static final Verb SEARCH_STREAM_INIT = new Verb(
            "usearch_dense_search_stream_init",
            MemoryLayout.structLayout(
                    Library.POINTER.withName("stream"),
                    Library.POINTER.withName("index"),
                    Library.POINTER.withName("outcome"),
                    Library.ROWS_VIEW.withName("query"),
                    Library.SIZE.withName("expansion_factor")));

    static final VarHandle SEARCH_STREAM_INIT_STREAM = SEARCH_STREAM_INIT.field("stream");
    static final long SEARCH_STREAM_INIT_QUERY = SEARCH_STREAM_INIT.at("query");
    static final VarHandle SEARCH_STREAM_INIT_EXPANSION_FACTOR = SEARCH_STREAM_INIT.field("expansion_factor");

    static final Verb SEARCH_STREAM_NEXT = new Verb(
            "usearch_dense_search_stream_next",
            Library.opening(
                    Library.padding(),
                    Library.POINTER.withName("stream"),
                    Library.POINTER.withName("matches"),
                    Library.POINTER.withName("arena"),
                    Library.SIZE.withName("wanted")));

    private static final VarHandle SEARCH_STREAM_NEXT_STREAM = SEARCH_STREAM_NEXT.field("stream");
    private static final VarHandle SEARCH_STREAM_NEXT_MATCHES = SEARCH_STREAM_NEXT.field("matches");
    private static final VarHandle SEARCH_STREAM_NEXT_ARENA = SEARCH_STREAM_NEXT.field("arena");
    private static final VarHandle SEARCH_STREAM_NEXT_WANTED = SEARCH_STREAM_NEXT.field("wanted");

    static final Verb SAVE = new Verb(
            "usearch_dense_save",
            Library.opening(Library.padding(), Library.BYTES_VIEW.withName("path"), Library.PROGRESS));

    static final long SAVE_PATH = SAVE.at("path");
    static final long SAVE_PROGRESS = SAVE.at("progress");

    static final Verb SAVE_BUFFER = new Verb(
            "usearch_dense_save_buffer",
            Library.opening(
                    Library.padding(),
                    Library.POINTER.withName("monolith"),
                    Library.POINTER.withName("arena"),
                    Library.PROGRESS));

    static final VarHandle SAVE_BUFFER_MONOLITH = SAVE_BUFFER.field("monolith");
    static final VarHandle SAVE_BUFFER_ARENA = SAVE_BUFFER.field("arena");
    static final long SAVE_BUFFER_PROGRESS = SAVE_BUFFER.at("progress");

    /** {@code usearch_dense_metadata_read_t} and its buffer twin, reading {@code source}. */
    private static final StructLayout READING = MemoryLayout.structLayout(
            Library.POINTER.withName("executor"),
            Library.POINTER.withName("outcome"),
            Library.ENUM.withName("completion"),
            Library.padding(),
            Library.BYTES_VIEW.withName("source"),
            Library.POINTER.withName("metadata"));

    static final Verb METADATA_READ = new Verb("usearch_dense_metadata_read", READING);
    static final Verb METADATA_READ_BUFFER = new Verb("usearch_dense_metadata_read_buffer", READING);
    private static final long READ_SOURCE = METADATA_READ.at("source");
    private static final VarHandle READ_METADATA = METADATA_READ.field("metadata");

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
                    Library.ENUM.withName("traversal"),
                    Library.padding(),
                    Library.PREDICATE,
                    Library.PROGRESS,
                    STATS));

    static final VarHandle MULTIPLY_COUNT = MULTIPLY.field("count");
    static final long MULTIPLY_LEFT = MULTIPLY.at("left");
    static final VarHandle MULTIPLY_PRODUCTS = MULTIPLY.field("products");
    static final VarHandle MULTIPLY_WRITTEN = MULTIPLY.field("written");
    static final VarHandle MULTIPLY_REQUIRED = MULTIPLY.field("required");
    static final VarHandle MULTIPLY_ARENA = MULTIPLY.field("arena");
    static final VarHandle MULTIPLY_WANTED = MULTIPLY.field("wanted");
    static final VarHandle MULTIPLY_EXPANSION_FACTOR = MULTIPLY.field("expansion_factor");
    static final long MULTIPLY_PREDICATE = MULTIPLY.at("predicate");
    static final long MULTIPLY_PROGRESS = MULTIPLY.at("progress");
    static final VarHandle MULTIPLY_STATS = MULTIPLY.field("stats");

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
                    Library.ENUM.withName("traversal"),
                    Library.PREDICATE,
                    Library.PROGRESS,
                    STATS));

    static final VarHandle ATTEND_COUNT = ATTEND.field("count");
    static final long ATTEND_QUERIES = ATTEND.at("queries");
    static final VarHandle ATTEND_HEAD_COUNT = ATTEND.field("head_count");
    static final VarHandle ATTEND_VALUE_LOOKUP = ATTEND.field("value_lookup");
    static final VarHandle ATTEND_CARRY = ATTEND.field("carry");
    static final VarHandle ATTEND_OUTPUTS = ATTEND.field("outputs");
    static final VarHandle ATTEND_ARENA = ATTEND.field("arena");
    static final VarHandle ATTEND_OUTPUTS_SCALAR = ATTEND.field("outputs_scalar");
    static final VarHandle ATTEND_EXPANSION_FACTOR = ATTEND.field("expansion_factor");
    static final VarHandle ATTEND_SCORE_SCALE = ATTEND.field("score_scale");
    static final long ATTEND_PREDICATE = ATTEND.at("predicate");
    static final long ATTEND_PROGRESS = ATTEND.at("progress");
    static final VarHandle ATTEND_STATS = ATTEND.field("stats");

    static final Verb MAXSIM = new Verb(
            "usearch_dense_maxsim",
            Library.opening(
                    Library.padding(),
                    Library.SIZE.withName("count"),
                    Library.ROWS_VIEW.withName("queries"),
                    Library.POINTER.withName("query_offsets"),
                    Library.SIZE.withName("rows_per_query"),
                    Library.POINTER.withName("matches"),
                    Library.POINTER.withName("written"),
                    Library.POINTER.withName("required"),
                    Library.POINTER.withName("arena"),
                    Library.SIZE.withName("wanted"),
                    Library.SIZE.withName("expansion_factor"),
                    Library.ENUM.withName("traversal"),
                    Library.padding(),
                    Library.PREDICATE,
                    Library.POINTER.withName("hints"),
                    Library.PROGRESS,
                    STATS));

    static final VarHandle MAXSIM_COUNT = MAXSIM.field("count");
    static final long MAXSIM_QUERIES = MAXSIM.at("queries");
    static final VarHandle MAXSIM_ROWS_PER_QUERY = MAXSIM.field("rows_per_query");
    static final VarHandle MAXSIM_MATCHES = MAXSIM.field("matches");
    static final VarHandle MAXSIM_WRITTEN = MAXSIM.field("written");
    static final VarHandle MAXSIM_REQUIRED = MAXSIM.field("required");
    static final VarHandle MAXSIM_ARENA = MAXSIM.field("arena");
    static final VarHandle MAXSIM_WANTED = MAXSIM.field("wanted");
    static final VarHandle MAXSIM_EXPANSION_FACTOR = MAXSIM.field("expansion_factor");
    static final long MAXSIM_PREDICATE = MAXSIM.at("predicate");
    static final long MAXSIM_PROGRESS = MAXSIM.at("progress");
    static final VarHandle MAXSIM_STATS = MAXSIM.field("stats");

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
