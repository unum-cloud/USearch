/**
 *  @file java/cloud/unum/usearch/SequenceIndex.java
 *  @author Ash Vardanian
 *  @date November 16, 2026
 *  @brief The sequence index handle, its options, and the receivers that run its verbs on a chosen executor.
 */
package cloud.unum.usearch;

import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.StructLayout;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.VarHandle;
import java.util.function.Consumer;

import org.jspecify.annotations.Nullable;

import cloud.unum.usearch.BasicCalls.Family;
import cloud.unum.usearch.BasicCalls.PlanCell;
import cloud.unum.usearch.BasicCalls.Rounds;
import cloud.unum.usearch.DenseCalls.Request;
import cloud.unum.usearch.DenseIndex.DenseBatch;
import cloud.unum.usearch.DenseIndex.DenseInsertConfig;
import cloud.unum.usearch.DenseIndex.DenseReachability;
import cloud.unum.usearch.DenseIndex.DenseSampleConfig;
import cloud.unum.usearch.DenseIndex.DenseSearchConfig;
import cloud.unum.usearch.DenseIndex.DenseTraversalStats;
import cloud.unum.usearch.DenseIndex.RefineReport;
import cloud.unum.usearch.DenseIndex.Sample;
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
import cloud.unum.usearch.Types.Status;
import cloud.unum.usearch.Types.USearchException;
import cloud.unum.usearch.Types.Verb;

/**
 * A Similarity Search Index over variable-length sequences - proteins, DNA, RNA and UTF-8 text -
 * under alignment, overlap and containment metrics, with the modality's own options, metric
 * and per-call settings as nested types.
 *
 * <p>
 * Every batch is a {@link Tape}: the bytes of every sequence end to end, and one more offset than
 * sequences around them. The verbs, receivers, plans and reports are the {@link DenseIndex} ones:
 * the index runs every verb on the caller's thread, {@link #on} blocks on an executor, and
 * {@link #deferred} submits there and answers a {@link Deferred}. Predicates and progress handlers
 * run on the executor's threads and must not throw; an exception escaping one terminates the process.
 *
 * <pre>{@code
 * try (Executor pool = Executor.forRam(new int[0], null, 0);
 *         Arena arena = new Arena(pool);
 *         SequenceIndex index = SequenceIndex.initRam(new SequenceOptions(), new int[0])) {
 *     index.on(pool).insert(new Key[] {Key.of(1), Key.of(2)}, Tape.ofStrings("MKTAYIAKQR", "MKSAYIAKQR"));
 *     Matches matches = index.on(pool).search(Tape.ofStrings("MKSAYIAKQR"), 1, arena);
 *     System.out.println(matches.key(0, 0));
 * }
 * }</pre>
 */
public final class SequenceIndex implements AutoCloseable {

    /**
     * The sequence metrics supported out of the box: each names what the sequences are, the form
     * they are stored in, and the score ranking them. Every metric stores a sequence folded, and a
     * get returns the stored form, never the bytes an insertion passed: packed class codes for a
     * protein alignment, one class code a byte for a DNA overlap, and case-folded UTF-8 for a
     * containment.
     */
    public enum SequenceMetric {

        /** A metric the library could not name. */
        UNKNOWN(0),
        /**
         * Global alignment of proteins with linear gap costs: BLOSUM62 with a gap of 7 over the 20
         * amino acids unless the index names its own {@link SequenceCosts}. Both sequences are aligned
         * end to end over the classes the costs fold bytes into, and every unaligned symbol is charged
         * the gap cost, so a score is quadratic in time. A stored sequence reads back as its class
         * codes, five bits apiece, little-endian behind one byte counting the slots the last bytes
         * leave unused.
         */
        PROTEIN_ALIGNMENT(1),
        /**
         * Shifted overlap of DNA or RNA: the cost of aligning the two sequences where one slides over
         * the other, at the shift most windows vote for, on either strand. Bytes fold into the four
         * classes of A, C, G and T in either case, U read as T, under costs that charge a transition
         * half a transversion and a gap of 2, unless the index names its own. The query also scores
         * as its reverse complement, and the lower distance counts. A stored sequence reads back as
         * its class codes, one byte apiece.
         */
        DNA_OVERLAP(2),
        /**
         * Window containment of UTF-8 text: the share of the query's distinct windows the stored side
         * lacks, whatever the case. A text is case-folded as Unicode folds it before it is stored or
         * scored, which may lengthen it up to three times. Asymmetric, and exact at every radius. A
         * stored text reads back folded. Takes no costs.
         */
        UTF8_CONTAINMENT(3);

        private final int code;

        SequenceMetric(int code) {
            this.code = code;
        }

        /** The C enumerator's numeric value. */
        public int code() {
            return code;
        }

        /** Names the kind a code holds; a code this build cannot name becomes UNKNOWN. */
        public static SequenceMetric of(int code) {
            for (SequenceMetric candidate : values()) {
                if (candidate.code == code) {
                    return candidate;
                }
            }
            return UNKNOWN;
        }

        /** Renders the kind as the checkpoint spells it. Calls {@code usearch_sequence_metric_name}. */
        @Override
        public String toString() {
            return Library.text(METRIC_NAME, code);
        }

        /**
         * Reads a kind back from the spelling {@link #toString} gives it. Calls
         * {@code usearch_sequence_metric_from_name}.
         *
         * @throws USearchException INVALID_ARGUMENT on a name the library does not spell.
         */
        public static SequenceMetric fromName(String name) {
            return of(BasicCalls.fromName(METRIC_FROM_NAME, name));
        }
    }

    /**
     * How alike two bytes are and the cost of leaving one unaligned, where an index scores by other
     * costs than its metric's own. A checkpoint records the classes and costs, so loads and views
     * take none. A DNA overlap needs costs folding bytes into exactly the four classes of A, C, G and
     * T. Mirrors {@code usearch_sequence_costs_t}.
     *
     * @param similarities row-major 256x256 signed similarities between raw bytes.
     * @param gap the cost of one unaligned symbol.
     */
    public record SequenceCosts(byte[] similarities, byte gap) {

        /**
         * Checks the table holds every byte, which C reads whole.
         *
         * @throws USearchException SHAPE_MISMATCH unless the similarities hold 256x256 scores.
         */
        public SequenceCosts {
            if (similarities.length != 256 * 256) {
                throw new USearchException(Status.SHAPE_MISMATCH, "similarities hold 256x256 scores");
            }
        }

        /** Copies the costs into the call's memory, answering the pointer C reads, or null for the metric's own. */
        static MemorySegment pass(Call call, @Nullable SequenceCosts costs) {
            if (costs == null) {
                return MemorySegment.NULL;
            }
            MemorySegment struct = call.memory.allocate(COSTS);
            COSTS_SIMILARITIES.set(struct, 0L, call.input(MemorySegment.ofArray(costs.similarities)));
            COSTS_GAP.set(struct, 0L, costs.gap);
            return struct;
        }
    }

    /**
     * The metric an empty index freezes, with its costs - the fields
     * {@code usearch_sequence_init_ram_t} and {@code usearch_sequence_init_vram_t} share. Every setter
     * returns this instance.
     */
    public static final class SequenceOptions {

        private SequenceMetric metric = SequenceMetric.PROTEIN_ALIGNMENT;
        private @Nullable SequenceCosts costs = null;
        private long connectivity = 0;
        private long sketchBytes = 0;
        private long valuesPerKey = 0;
        private @Nullable CPUClass cpuClassCeiling = null;
        private @Nullable GPUClass gpuClassCeiling = null;

        /** A protein alignment index, with every optional field derived. */
        public SequenceOptions() {}

        /** What the sequences are and the score ranking them. */
        public SequenceOptions metric(SequenceMetric metric) {
            this.metric = metric;
            return this;
        }

        /** The costs to score by; {@code null} for the metric's own, refused for a containment. */
        public SequenceOptions costs(@Nullable SequenceCosts costs) {
            this.costs = costs;
            return this;
        }

        /** The connections-per-node bound of the graph; zero derives. */
        public SequenceOptions connectivity(long connectivity) {
            this.connectivity = connectivity;
            return this;
        }

        /**
         * The bytes of the sketch in front of each stored sequence, a multiple of four; zero takes the metric's
         * default, and a DNA overlap keeps none and refuses any.
         */
        public SequenceOptions sketchBytes(long sketchBytes) {
            this.sketchBytes = sketchBytes;
            return this;
        }

        /** The upper bound on entries sharing one key; zero is unbounded. */
        public SequenceOptions valuesPerKey(long valuesPerKey) {
            this.valuesPerKey = valuesPerKey;
            return this;
        }

        /** The newest CPU class dispatch may pick, intersected with what the machine runs; {@code null} means none. */
        public SequenceOptions cpuClassCeiling(@Nullable CPUClass cpuClassCeiling) {
            this.cpuClassCeiling = cpuClassCeiling;
            return this;
        }

        /** The newest GPU class dispatch may pick; {@code null} means no ceiling. */
        public SequenceOptions gpuClassCeiling(@Nullable GPUClass gpuClassCeiling) {
            this.gpuClassCeiling = gpuClassCeiling;
            return this;
        }

        /** Fills the fields both init structures share, at the same offsets in each. */
        void fill(Call call) {
            call.enumeration(INIT_METRIC, metric.code());
            call.pointer(INIT_COSTS, SequenceCosts.pass(call, costs));
            call.size(INIT_CONNECTIVITY, connectivity);
            call.size(INIT_SKETCH_BYTES, sketchBytes);
            call.size(INIT_VALUES_PER_KEY, valuesPerKey);
        }
    }

    /**
     * The metric and geometry a checkpoint froze. Mirrors {@code usearch_sequence_metadata_t}.
     *
     * @param metric the metric the checkpoint was built under.
     * @param connectivity the connections per node the graph was built with.
     * @param classes the classes the recorded costs folded bytes into; zero for a containment.
     * @param sketchBytes the bytes of the sketch in front of each stored sequence; zero when the metric keeps none.
     */
    public record SequenceMetadata(SequenceMetric metric, long connectivity, long classes, long sketchBytes) {

        static SequenceMetadata lift(MemorySegment struct, long at) {
            return new SequenceMetadata(
                    SequenceMetric.of((int) METADATA_METRIC.get(struct, at)),
                    (long) METADATA_CONNECTIVITY.get(struct, at),
                    (long) METADATA_CLASSES.get(struct, at),
                    (long) METADATA_SKETCH_BYTES.get(struct, at));
        }
    }

    /**
     * A live index's state, each counter read once. Mirrors {@code usearch_sequence_index_description_t}.
     *
     * @param metadata the frozen kind and geometry.
     * @param size the stored entries.
     * @param capacity the entries hosted without growing again.
     * @param memoryUsage the bytes the handle holds, payload storage included.
     * @param cpuClass the CPU class dispatch picked, as {@code usearch_cpu_class_name} spells it.
     * @param gpuClass the GPU class dispatch picked; "unknown" for a host index.
     * @param shardsCount the live shards.
     */
    public record SequenceIndexDescription(
            SequenceMetadata metadata,
            long size,
            long capacity,
            long memoryUsage,
            String cpuClass,
            String gpuClass,
            long shardsCount) {

        static SequenceIndexDescription lift(MemorySegment description) {
            return new SequenceIndexDescription(
                    SequenceMetadata.lift(description, DESCRIPTION_METADATA),
                    (long) DESCRIPTION_SIZE.get(description, 0L),
                    (long) DESCRIPTION_CAPACITY.get(description, 0L),
                    (long) DESCRIPTION_MEMORY_USAGE.get(description, 0L),
                    Library.text(CPU_CLASS_NAME, (int) DESCRIPTION_CPU_CLASS.get(description, 0L)),
                    Library.text(GPU_CLASS_NAME, (int) DESCRIPTION_GPU_CLASS.get(description, 0L)),
                    (long) DESCRIPTION_SHARDS_COUNT.get(description, 0L));
        }
    }

    /**
     * The entries one key holds and the extent of their bytes. Mirrors {@code usearch_sequence_key_stats_t}.
     *
     * @param entries the entries stored under the key; zero when it holds none.
     * @param totalBytes their bytes summed.
     * @param largestBytes the bytes of the longest.
     */
    public record SequenceKeyStats(long entries, long totalBytes, long largestBytes) {

        static SequenceKeyStats lift(MemorySegment stats) {
            long[] fields = stats.toArray(ValueLayout.JAVA_LONG);
            return new SequenceKeyStats(fields[0], fields[1], fields[2]);
        }
    }

    /**
     * The knobs of a refinement and the plan it continues; see {@link #refine(SequenceRefineConfig)}.
     * Mirrors {@code usearch_sequence_refine_t}.
     */
    public static final class SequenceRefineConfig {

        private long expansionFactor;
        private long memoryLimit;
        private @Nullable PlanCell plan;
        private @Nullable Progress progress;
        private @Nullable DenseTraversalStats stats;

        /** Every knob derived, one round run to its end. */
        public SequenceRefineConfig() {}

        /** The breadth of the candidates any walk beside the rounds explores; zero derives. */
        public SequenceRefineConfig expansionFactor(long expansionFactor) {
            this.expansionFactor = expansionFactor;
            return this;
        }

        /** The bytes a loaded NVMe index holds for the one part it rewrites; zero derives. */
        public SequenceRefineConfig memoryLimit(long memoryLimit) {
            this.memoryLimit = memoryLimit;
            return this;
        }

        /** The plan the call opens or continues, so a cancel leaves its round resumable. */
        public SequenceRefineConfig plan(SequenceRefinePlan plan) {
            this.plan = plan.cell;
            return this;
        }

        /** Hears every barrier of every phase, and may cancel the round there. */
        public SequenceRefineConfig progress(Progress progress) {
            this.progress = progress;
            return this;
        }

        /** Where the traversal telemetry lands, overwritten once the call succeeded. */
        public SequenceRefineConfig stats(DenseTraversalStats stats) {
            this.stats = stats;
            return this;
        }

        Rounds rounds() {
            return new Rounds(expansionFactor, memoryLimit, plan, progress, stats);
        }
    }

    /**
     * The plan a reshard continues, the seed of its cut and the knobs of its closing rounds; see
     * {@link #reshard(SequenceIndex[], SequenceIndex[], SequenceReshardConfig)}. Mirrors
     * {@code usearch_sequence_reshard_t}.
     */
    public static final class SequenceReshardConfig {

        private long expansionFactor;
        private long memoryLimit;
        private long seed;
        private @Nullable PlanCell plan;
        private @Nullable Progress progress;
        private @Nullable DenseTraversalStats stats;

        /** Every knob derived, under seed zero. */
        public SequenceReshardConfig() {}

        /** The breadth of the candidates any walk beside the closing rounds explores; zero derives. */
        public SequenceReshardConfig expansionFactor(long expansionFactor) {
            this.expansionFactor = expansionFactor;
            return this;
        }

        /** The bytes a loaded NVMe target holds for the one part it rewrites; zero derives. */
        public SequenceReshardConfig memoryLimit(long memoryLimit) {
            this.memoryLimit = memoryLimit;
            return this;
        }

        /** Picks the part roots of a cut: equal seeds over an unchanged source cut equally. */
        public SequenceReshardConfig seed(long seed) {
            this.seed = seed;
            return this;
        }

        /** The plan the call opens or continues, so a cancel leaves its closing rounds resumable. */
        public SequenceReshardConfig plan(SequenceReshardPlan plan) {
            this.plan = plan.cell;
            return this;
        }

        /** Hears every barrier of the closing rounds, and may cancel them there; a move is never cancelled. */
        public SequenceReshardConfig progress(Progress progress) {
            this.progress = progress;
            return this;
        }

        /** Where the traversal telemetry lands, overwritten once the call succeeded. */
        public SequenceReshardConfig stats(DenseTraversalStats stats) {
            this.stats = stats;
            return this;
        }

        Rounds rounds() {
            return new Rounds(expansionFactor, memoryLimit, plan, progress, stats);
        }
    }

    /**
     * A refinement in flight, which {@link #refine(SequenceRefineConfig)} opens on its first call and
     * continues on every later one. The index admits no insertion and no save while it is open, and
     * every call naming it names the index and executor it was opened with;
     * {@link #refinePlanFinish} closes it.
     */
    public static final class SequenceRefinePlan {

        final PlanCell cell = new PlanCell();

        /** A plan no call has opened yet. */
        public SequenceRefinePlan() {}

        /** Where the last call naming the plan stopped. */
        public PhaseProgress phase() {
            return cell.phase();
        }
    }

    /**
     * A reshard in flight, which {@link #reshard(SequenceIndex[], SequenceIndex[], SequenceReshardConfig)}
     * opens on its first call and continues on every later one. Every index it names admits no
     * insertion and no save while it is open; {@link #reshardPlanFinish} closes it.
     */
    public static final class SequenceReshardPlan {

        final PlanCell cell = new PlanCell();

        /** A plan no call has opened yet. */
        public SequenceReshardPlan() {}

        /** Where the last call naming the plan stopped. */
        public PhaseProgress phase() {
            return cell.phase();
        }
    }

    /**
     * A sequential walk over the index contents; the sequences themselves come from
     * {@link SequenceIndex#get}. It keeps no executor: its pulls run on the calling thread, or block on
     * the executor {@link #on(Executor)} names for each of them. An overlapping pull reports
     * CONTENDED.
     */
    public static final class SequenceCursor implements AutoCloseable {

        private final Handle handle;
        private final SequenceIndex index;

        SequenceCursor(MemorySegment opened, SequenceIndex index) {
            this.handle = new Handle(opened, FAMILY.cursorFree());
            this.index = index;
        }

        /** The receiver whose pulls block on {@code executor}, named again for every pull. Binding-only. */
        public On on(Executor executor) {
            return new On(this, executor);
        }

        /** Copies the keys of up to {@code wanted} next entries into {@code arena}. Calls {@code usearch_sequence_cursor_next}. */
        public DenseBatch next(long wanted, Arena arena) {
            return pull(null, wanted, false, arena);
        }

        /** Copies up to {@code wanted} next entries, each with the keys it links to. Calls {@code usearch_sequence_cursor_next}. */
        public DenseBatch nextLinks(long wanted, Arena arena) {
            return pull(null, wanted, true, arena);
        }

        private DenseBatch pull(@Nullable Executor executor, long wanted, boolean linked, Arena arena) {
            return DenseIndex.pulling(FAMILY.cursorNext(), index.handle, executor, handle, wanted, linked, arena);
        }

        /** A cursor's pulls blocking on one executor, contended while another call holds it. */
        public static final class On {

            private final SequenceCursor cursor;
            private final Executor executor;

            On(SequenceCursor cursor, Executor executor) {
                this.cursor = cursor;
                this.executor = executor;
            }

            /** Runs {@link SequenceCursor#next(long, Arena)} on this receiver's executor. */
            public DenseBatch next(long wanted, Arena arena) {
                return cursor.pull(executor, wanted, false, arena);
            }

            /** Runs {@link SequenceCursor#nextLinks(long, Arena)} on this receiver's executor. */
            public DenseBatch nextLinks(long wanted, Arena arena) {
                return cursor.pull(executor, wanted, true, arena);
            }
        }

        /** Releases the cursor; idempotent. Calls {@code usearch_sequence_cursor_free}. */
        @Override
        public void close() {
            handle.close();
        }
    }

    /** The receiver whose verbs block on an executor, contended while another call holds it. */
    public static final class On {

        private final SequenceIndex index;
        private final Executor executor;

        On(SequenceIndex index, Executor executor) {
            this.index = index;
            this.executor = executor;
        }

        private <Result> Result block(Request<Result> request) {
            return Call.block(request.verb(), index.handle, executor, request.fill());
        }

        /** Runs {@link SequenceIndex#reserve} on this receiver's executor. */
        public void reserve(long entries, long payloadBytes) {
            block(reserving(entries, payloadBytes));
        }

        /** Runs {@link SequenceIndex#clear} on this receiver's executor. */
        public void clear() {
            block(DenseCalls.clearing(FAMILY.clear()));
        }

        /** Runs {@link SequenceIndex#describe} on this receiver's executor. */
        public SequenceIndexDescription describe() {
            return block(describing());
        }

        /** Runs {@link SequenceIndex#count} on this receiver's executor. */
        public SequenceKeyStats count(Key key) {
            return block(counting(key));
        }

        /** Runs {@link SequenceIndex#contains} on this receiver's executor. */
        public boolean contains(Key key) {
            return count(key).entries() != 0;
        }

        /** Runs {@link SequenceIndex#get} on this receiver's executor. */
        public Tape get(Key key, Arena arena) {
            return block(getting(key, arena));
        }

        /** Runs {@link SequenceIndex#insert(Key[], Tape)} on this receiver's executor. */
        public void insert(Key[] keys, Tape sequences) {
            block(inserting(keys, sequences, new DenseInsertConfig()));
        }

        /** Runs {@link SequenceIndex#insert(Key[], Tape, DenseInsertConfig)} on this receiver's executor. */
        public void insert(Key[] keys, Tape sequences, DenseInsertConfig config) {
            block(inserting(keys, sequences, config));
        }

        /** Runs {@link SequenceIndex#refine()} on this receiver's executor. */
        public RefineReport refine() {
            return block(BasicCalls.refining(FAMILY, new SequenceRefineConfig().rounds()));
        }

        /** Runs {@link SequenceIndex#refine(SequenceRefineConfig)} on this receiver's executor. */
        public RefineReport refine(SequenceRefineConfig config) {
            return block(BasicCalls.refining(FAMILY, config.rounds()));
        }

        /** Runs {@link SequenceIndex#refinePlanFinish} on this receiver's executor. */
        public RefineReport refinePlanFinish(SequenceRefinePlan plan) {
            return block(BasicCalls.refinePlanFinishing(FAMILY, plan.cell));
        }

        /** Runs {@link SequenceIndex#health} on this receiver's executor. */
        public DenseReachability health() {
            return block(DenseCalls.auditing(FAMILY.health()));
        }

        /** Runs {@link SequenceIndex#search(Tape, long, Arena)} on this receiver's executor. */
        public Matches search(Tape queries, long wanted, Arena arena) {
            return block(searching(queries, new DenseSearchConfig(wanted), arena));
        }

        /** Runs {@link SequenceIndex#search(Tape, DenseSearchConfig, Arena)} on this receiver's executor. */
        public Matches search(Tape queries, DenseSearchConfig config, Arena arena) {
            return block(searching(queries, config, arena));
        }

        /** Runs {@link SequenceIndex#reshard} on this receiver's executor. */
        public RefineReport reshard(SequenceIndex[] sources, SequenceIndex[] targets, SequenceReshardConfig config) {
            return block(resharding(sources, targets, config));
        }

        /** Runs {@link SequenceIndex#reshardPlanFinish} on this receiver's executor. */
        public RefineReport reshardPlanFinish(SequenceReshardPlan plan) {
            return block(BasicCalls.reshardPlanFinishing(FAMILY, plan.cell));
        }

        /** Runs {@link SequenceIndex#sample} on this receiver's executor. */
        public Sample sample(long wanted, Arena arena, DenseSampleConfig config) {
            return block(DenseCalls.sampling(FAMILY.sample(), wanted, arena, config));
        }

        /** Runs {@link SequenceIndex#save} on this receiver's executor. */
        public void save(String path) {
            block(DenseCalls.saving(FAMILY.save(), path, null));
        }

        /** Runs {@link SequenceIndex#saveBuffer} on this receiver's executor. */
        public Monolith saveBuffer(Arena arena) {
            return block(DenseCalls.serializing(FAMILY.saveBuffer(), arena, null));
        }
    }

    /**
     * The receiver whose verbs submit their calls to an executor, each answering a {@link Deferred}
     * that owns what the call reads until final. Its verbs are those of {@link SequenceIndex}, less
     * the cursor that pulls one batch at a time.
     */
    public static final class DeferredOn {

        private final SequenceIndex index;
        private final Executor executor;

        DeferredOn(SequenceIndex index, Executor executor) {
            this.index = index;
            this.executor = executor;
        }

        private <Result> Deferred<Result> submit(Request<Result> request) {
            return Call.submit(request.verb(), index.handle, executor, request.fill());
        }

        /** Submits {@link SequenceIndex#reserve}. Calls {@code usearch_sequence_reserve}. */
        public Deferred<Void> reserve(long entries, long payloadBytes) {
            return submit(reserving(entries, payloadBytes));
        }

        /** Submits {@link SequenceIndex#clear}. Calls {@code usearch_sequence_clear}. */
        public Deferred<Void> clear() {
            return submit(DenseCalls.clearing(FAMILY.clear()));
        }

        /** Submits {@link SequenceIndex#describe}. Calls {@code usearch_sequence_describe}. */
        public Deferred<SequenceIndexDescription> describe() {
            return submit(describing());
        }

        /** Submits {@link SequenceIndex#count}. Calls {@code usearch_sequence_count}. */
        public Deferred<SequenceKeyStats> count(Key key) {
            return submit(counting(key));
        }

        /** Submits {@link SequenceIndex#get}. Calls {@code usearch_sequence_get}. */
        public Deferred<Tape> get(Key key, Arena arena) {
            return submit(getting(key, arena));
        }

        /** Submits {@link SequenceIndex#insert(Key[], Tape, DenseInsertConfig)}. Calls {@code usearch_sequence_insert}. */
        public Deferred<Void> insert(Key[] keys, Tape sequences, DenseInsertConfig config) {
            return submit(inserting(keys, sequences, config));
        }

        /** Submits {@link SequenceIndex#refine(SequenceRefineConfig)}. Calls {@code usearch_sequence_refine}. */
        public Deferred<RefineReport> refine(SequenceRefineConfig config) {
            return submit(BasicCalls.refining(FAMILY, config.rounds()));
        }

        /** Submits {@link SequenceIndex#refinePlanFinish}. Calls {@code usearch_sequence_refine_plan_finish}. */
        public Deferred<RefineReport> refinePlanFinish(SequenceRefinePlan plan) {
            return submit(BasicCalls.refinePlanFinishing(FAMILY, plan.cell));
        }

        /** Submits {@link SequenceIndex#health}. Calls {@code usearch_sequence_health}. */
        public Deferred<DenseReachability> health() {
            return submit(DenseCalls.auditing(FAMILY.health()));
        }

        /** Submits {@link SequenceIndex#search(Tape, DenseSearchConfig, Arena)}. Calls {@code usearch_sequence_search}. */
        public Deferred<Matches> search(Tape queries, DenseSearchConfig config, Arena arena) {
            return submit(searching(queries, config, arena));
        }

        /** Submits {@link SequenceIndex#reshard}. Calls {@code usearch_sequence_reshard}. */
        public Deferred<RefineReport> reshard(
                SequenceIndex[] sources, SequenceIndex[] targets, SequenceReshardConfig config) {
            return submit(resharding(sources, targets, config));
        }

        /** Submits {@link SequenceIndex#reshardPlanFinish}. Calls {@code usearch_sequence_reshard_plan_finish}. */
        public Deferred<RefineReport> reshardPlanFinish(SequenceReshardPlan plan) {
            return submit(BasicCalls.reshardPlanFinishing(FAMILY, plan.cell));
        }

        /** Submits {@link SequenceIndex#sample}. Calls {@code usearch_sequence_sample}. */
        public Deferred<Sample> sample(long wanted, Arena arena, DenseSampleConfig config) {
            return submit(DenseCalls.sampling(FAMILY.sample(), wanted, arena, config));
        }

        /** Submits {@link SequenceIndex#save}. Calls {@code usearch_sequence_save}. */
        public Deferred<Void> save(String path) {
            return submit(DenseCalls.saving(FAMILY.save(), path, null));
        }

        /** Submits {@link SequenceIndex#saveBuffer}. Calls {@code usearch_sequence_save_buffer}. */
        public Deferred<Monolith> saveBuffer(Arena arena) {
            return submit(DenseCalls.serializing(FAMILY.saveBuffer(), arena, null));
        }
    }

    final Handle handle;

    private SequenceIndex(MemorySegment created) {
        handle = new Handle(created, FAMILY.free());
    }

    /** The receiver whose verbs block on {@code executor}. Binding-only. */
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
     * Claims room for entries and their bytes ahead of insertion, a hint, since insertions grow the
     * index on demand. Calls {@code usearch_sequence_reserve}.
     *
     * @param entries the desired total number of entries, including the current size.
     * @param payloadBytes the desired total sequence bytes.
     * @throws USearchException READ_ONLY on a view.
     */
    public void reserve(long entries, long payloadBytes) {
        block(reserving(entries, payloadBytes));
    }

    /** Erases all entries, keeping the frozen kind. Calls {@code usearch_sequence_clear}. */
    public void clear() {
        block(DenseCalls.clearing(FAMILY.clear()));
    }

    /** Reports the frozen metadata and the current counters in one call. Calls {@code usearch_sequence_describe}. */
    public SequenceIndexDescription describe() {
        return block(describing());
    }

    /**
     * How many entries a key holds, with their total and largest byte lengths. Calls
     * {@code usearch_sequence_count}.
     */
    public SequenceKeyStats count(Key key) {
        return block(counting(key));
    }

    /** Whether the index holds at least one entry under a key. Binding-only sugar over {@link #count}. */
    public boolean contains(Key key) {
        return count(key).entries() != 0;
    }

    /**
     * Retrieves every sequence stored under a key into {@code arena}, in insertion order. Calls
     * {@code usearch_sequence_get}.
     *
     * @return the tape read in place, of no entries when the key holds nothing.
     * @throws USearchException CONTENDED when the arena serves another call.
     */
    public Tape get(Key key, Arena arena) {
        return block(getting(key, arena));
    }

    /**
     * Inserts sequences under their keys, with every knob derived.
     *
     * @throws USearchException the statuses {@link #insert(Key[], Tape, DenseInsertConfig)} names.
     */
    public void insert(Key[] keys, Tape sequences) {
        block(inserting(keys, sequences, new DenseInsertConfig()));
    }

    /**
     * Inserts a tape of sequences under their keys - all-or-nothing per call. Calls
     * {@code usearch_sequence_insert}.
     *
     * @throws USearchException SHAPE_MISMATCH when the key count and the tape's entries disagree,
     *     READ_ONLY on a view, and OVERFLOW when a key already holds {@code valuesPerKey} entries.
     */
    public void insert(Key[] keys, Tape sequences, DenseInsertConfig config) {
        block(inserting(keys, sequences, config));
    }

    /**
     * Runs one round over every stored entry, with every knob derived.
     *
     * @throws USearchException the statuses {@link #refine(SequenceRefineConfig)} names.
     */
    public RefineReport refine() {
        return block(BasicCalls.refining(FAMILY, new SequenceRefineConfig().rounds()));
    }

    /**
     * Re-selects the neighbors of every stored entry against the finished graph, as
     * {@link DenseIndex#refine(DenseIndex.DenseRefineConfig)} does. Calls {@code usearch_sequence_refine}.
     *
     * @throws USearchException READ_ONLY on a view, and INVALID_ARGUMENT on a plan without an executor.
     */
    public RefineReport refine(SequenceRefineConfig config) {
        return block(BasicCalls.refining(FAMILY, config.rounds()));
    }

    /**
     * Closes the round a cancel left in hand and frees the plan whatever the call answers. Calls
     * {@code usearch_sequence_refine_plan_finish}.
     *
     * @throws USearchException INVALID_ARGUMENT on a plan no call opened, or one already finished.
     */
    public RefineReport refinePlanFinish(SequenceRefinePlan plan) {
        return block(BasicCalls.refinePlanFinishing(FAMILY, plan.cell));
    }

    /** Audits how much of the graph a search reaches. Calls {@code usearch_sequence_health}. */
    public DenseReachability health() {
        return block(DenseCalls.auditing(FAMILY.health()));
    }

    /**
     * Resolves a tape of queries into up to {@code wanted} nearest matches apiece, with every other
     * knob derived.
     *
     * @throws USearchException the statuses {@link #search(Tape, DenseSearchConfig, Arena)} names.
     */
    public Matches search(Tape queries, long wanted, Arena arena) {
        return block(searching(queries, new DenseSearchConfig(wanted), arena));
    }

    /**
     * Performs k-Nearest Neighbors Search for a tape of queries, by a walk or an exact scan. Calls
     * {@code usearch_sequence_search}.
     *
     * @throws USearchException CONTENDED when the arena serves another call.
     */
    public Matches search(Tape queries, DenseSearchConfig config, Arena arena) {
        return block(searching(queries, config, arena));
    }

    /**
     * Moves the entries of {@code sources} into {@code targets}, then runs a round on every target that
     * took entries, as {@link DenseIndex#reshard} does. Calls {@code usearch_sequence_reshard}.
     *
     * @throws USearchException INVALID_ARGUMENT on no sources or targets or an index named twice,
     *     CONFIG_MISMATCH on another kind or a target neither a source nor empty, and UNSUPPORTED on
     *     one index as its own only source and target.
     */
    public RefineReport reshard(SequenceIndex[] sources, SequenceIndex[] targets, SequenceReshardConfig config) {
        return block(resharding(sources, targets, config));
    }

    /**
     * Closes the rounds a cancel stopped and frees the plan whatever the call answers. Calls
     * {@code usearch_sequence_reshard_plan_finish}.
     *
     * @throws USearchException INVALID_ARGUMENT on a plan no call opened.
     */
    public RefineReport reshardPlanFinish(SequenceReshardPlan plan) {
        return block(BasicCalls.reshardPlanFinishing(FAMILY, plan.cell));
    }

    /**
     * Draws keys around the focus, or across the whole index, placed in three dimensions, as
     * {@link DenseIndex#sample} does. Calls {@code usearch_sequence_sample}.
     */
    public Sample sample(long wanted, Arena arena, DenseSampleConfig config) {
        return block(DenseCalls.sampling(FAMILY.sample(), wanted, arena, config));
    }

    /**
     * Writes the index under a path: one monolith for a lone shard, or a manifest beside one part per
     * shard. Calls {@code usearch_sequence_save}.
     *
     * @throws USearchException IO when the destination cannot be written.
     */
    public void save(String path) {
        block(DenseCalls.saving(FAMILY.save(), path, null));
    }

    /** Writes the index into {@code arena}, the bytes {@link #save} writes. Calls {@code usearch_sequence_save_buffer}. */
    public Monolith saveBuffer(Arena arena) {
        return block(DenseCalls.serializing(FAMILY.saveBuffer(), arena, null));
    }

    /**
     * Opens a snapshot cursor over the contents, every entry as stored or breadth-first from
     * {@code starts}. Calls {@code usearch_sequence_cursor_init}.
     */
    public SequenceCursor cursor(Key... starts) {
        return block(BasicCalls.cursorOpening(FAMILY, starts, opened -> new SequenceCursor(opened, this)));
    }

    /**
     * Creates an empty, mutable index in host memory. Calls {@code usearch_sequence_init_ram}.
     *
     * @param memoryDomains NUMA node ids to place shards on; an empty list derives every node.
     * @throws USearchException UNKNOWN_KIND on an unknown metric, and CONFIG_MISMATCH on costs
     *     the library refuses.
     */
    public static SequenceIndex initRam(SequenceOptions options, int[] memoryDomains) {
        return create(INIT_RAM, null, call -> {
            options.fill(call);
            call.ram(call.arguments, INIT_RAM.at("ram"), memoryDomains, options.cpuClassCeiling);
        });
    }

    /**
     * Creates an empty, mutable index in device memory. Calls {@code usearch_sequence_init_vram}.
     *
     * @throws USearchException DEVICE when no device answers or the build has no GPU kernels.
     */
    public static SequenceIndex initVram(SequenceOptions options, int[] devices) {
        return create(INIT_VRAM, null, call -> {
            options.fill(call);
            call.vram(call.arguments, INIT_VRAM.at("vram"), devices, options.gpuClassCeiling);
        });
    }

    /**
     * Copies a checkpoint into a new, mutable index in host memory. Calls {@code usearch_sequence_load_ram}.
     *
     * @throws USearchException MALFORMED on a checkpoint that ends early, and IO when it cannot be read.
     */
    public static SequenceIndex loadRam(
            String path,
            @Nullable Executor executor,
            int[] memoryDomains,
            long valuesPerKey,
            @Nullable Progress progress) {
        return create(
                LOAD_RAM,
                executor,
                call -> loading(call, true, memoryDomains, valuesPerKey, progress)
                        .path(LOAD_SOURCE, path));
    }

    /** Copies a monolith from a buffer into a new index in host memory. Calls {@code usearch_sequence_load_ram_buffer}. */
    public static SequenceIndex loadRamBuffer(
            MemorySegment buffer,
            @Nullable Executor executor,
            int[] memoryDomains,
            long valuesPerKey,
            @Nullable Progress progress) {
        return create(
                LOAD_RAM_BUFFER,
                executor,
                call -> loading(call, true, memoryDomains, valuesPerKey, progress)
                        .bytes(LOAD_SOURCE, buffer));
    }

    /** Copies a checkpoint into a new index in device memory. Calls {@code usearch_sequence_load_vram}. */
    public static SequenceIndex loadVram(
            String path, @Nullable Executor executor, int[] devices, long valuesPerKey, @Nullable Progress progress) {
        return create(
                LOAD_VRAM,
                executor,
                call -> loading(call, false, devices, valuesPerKey, progress).path(LOAD_SOURCE, path));
    }

    /** Copies a monolith from a buffer into a new index in device memory. Calls {@code usearch_sequence_load_vram_buffer}. */
    public static SequenceIndex loadVramBuffer(
            MemorySegment buffer,
            @Nullable Executor executor,
            int[] devices,
            long valuesPerKey,
            @Nullable Progress progress) {
        return create(
                LOAD_VRAM_BUFFER,
                executor,
                call -> loading(call, false, devices, valuesPerKey, progress).bytes(LOAD_SOURCE, buffer));
    }

    /**
     * Serves a checkpoint in place through a memory mapping, as a read-only index. Calls
     * {@code usearch_sequence_view_ram}.
     *
     * @throws USearchException UNSUPPORTED where the engine maps no sequence checkpoint.
     */
    public static SequenceIndex viewRam(String path) {
        return create(VIEW_RAM, null, call -> call.path(VIEW_RAM.at("path"), path));
    }

    /**
     * Serves a checkpoint in place from NVMe drives as a read-only index. Calls
     * {@code usearch_sequence_view_nvme}.
     *
     * @param paths devices or directories to serve the parts from; an empty list serves each where its
     *     manifest points.
     */
    public static SequenceIndex viewNvme(
            String path, @Nullable Executor executor, String[] paths, @Nullable CPUClass cpuClassCeiling) {
        return create(VIEW_NVME, executor, call -> {
            call.path(VIEW_NVME.at("path"), path);
            call.nvme(call.arguments, VIEW_NVME.at("nvme"), paths, cpuClassCeiling);
        });
    }

    /** Reads a checkpoint's kind and geometry without loading it. Calls {@code usearch_sequence_metadata_read}. */
    public static SequenceMetadata metadataRead(String path, @Nullable Executor executor) {
        return BasicCalls.metadata(
                FAMILY.metadataRead(),
                executor,
                call -> call.path(BasicCalls.READ_SOURCE, path),
                METADATA,
                metadata -> SequenceMetadata.lift(metadata, 0));
    }

    /** Reads a monolith's kind and geometry without loading it. Calls {@code usearch_sequence_metadata_read_buffer}. */
    public static SequenceMetadata metadataReadBuffer(MemorySegment buffer, @Nullable Executor executor) {
        return BasicCalls.metadata(
                FAMILY.metadataReadBuffer(),
                executor,
                call -> call.bytes(BasicCalls.READ_SOURCE, buffer),
                METADATA,
                metadata -> SequenceMetadata.lift(metadata, 0));
    }

    /**
     * Releases the index, after every deferred call naming it is final. It is idempotent. Calls
     * {@code usearch_sequence_free}.
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
        return "SequenceIndex(" + handle + ")";
    }

    private static Request<Void> reserving(long entries, long payloadBytes) {
        return new Request<>(RESERVE, call -> {
            call.size(RESERVE.field("entries"), entries);
            call.size(RESERVE.field("payload_bytes"), payloadBytes);
            return () -> null;
        });
    }

    private static Request<SequenceIndexDescription> describing() {
        return BasicCalls.describing(FAMILY, DESCRIPTION, SequenceIndexDescription::lift);
    }

    private static Request<SequenceKeyStats> counting(Key key) {
        return BasicCalls.counting(FAMILY, key, KEY_STATS, SequenceKeyStats::lift);
    }

    private static Request<Tape> getting(Key key, Arena arena) {
        return new Request<>(GET, call -> {
            call.key(GET.at("key"), key);
            MemorySegment sequences = call.output(GET_SEQUENCES, BasicCalls.TAPE_VIEW);
            MemorySegment written = call.output(GET_WRITTEN, Library.SIZE);
            call.arena(GET_ARENA, arena);
            return () -> Tape.lift(call, sequences, written.get(Library.SIZE, 0), 1);
        });
    }

    private static Request<Void> inserting(Key[] keys, Tape sequences, DenseInsertConfig config) {
        // The call has one count, so only this side sees the keys and the tape disagree.
        if (keys.length != sequences.count()) {
            throw new USearchException(Status.SHAPE_MISMATCH, "the key count does not match the tape's entries");
        }
        return new Request<>(INSERT, call -> {
            call.size(INSERT_COUNT, keys.length);
            call.keys(INSERT.at("keys"), keys);
            sequences.fill(call, INSERT.at("sequences"));
            call.size(INSERT_EXPANSION_FACTOR, config.expansionFactor);
            call.progress(INSERT.at("progress"), config.progress);
            call.stats(INSERT_STATS, config.stats);
            return () -> null;
        });
    }

    private static Request<Matches> searching(Tape queries, DenseSearchConfig config, Arena arena) {
        return new Request<>(SEARCH, call -> {
            call.size(SEARCH_COUNT, queries.count());
            queries.fill(call, SEARCH.at("queries"));
            MemorySegment matches = call.output(SEARCH_MATCHES, Library.MATCHES_VIEW);
            call.arena(SEARCH_ARENA, arena);
            call.size(SEARCH_WANTED, config.wanted);
            call.size(SEARCH_EXPANSION_FACTOR, config.expansionFactor);
            call.enumeration(SEARCH_TRAVERSAL, config.traversal.code());
            call.predicate(SEARCH.at("predicate"), config.predicate);
            call.progress(SEARCH.at("progress"), config.progress);
            call.stats(SEARCH_STATS, config.stats);
            return () -> Matches.lift(call, matches, queries.count());
        });
    }

    private static Request<RefineReport> resharding(
            SequenceIndex[] sources, SequenceIndex[] targets, SequenceReshardConfig config) {
        Handle[] from = new Handle[sources.length];
        for (int order = 0; order != sources.length; ++order) {
            from[order] = sources[order].handle;
        }
        Handle[] into = new Handle[targets.length];
        for (int order = 0; order != targets.length; ++order) {
            into[order] = targets[order].handle;
        }
        return BasicCalls.resharding(FAMILY, from, into, config.seed, config.rounds());
    }

    /** Fills the placement, bound and handler every load shares, answering the call for its source. */
    private static Call loading(
            Call call, boolean ram, int[] placement, long valuesPerKey, @Nullable Progress progress) {
        if (ram) {
            call.ram(call.arguments, LOAD_RAM.at("ram"), placement, null);
        } else {
            call.vram(call.arguments, LOAD_VRAM.at("vram"), placement, null);
        }
        call.size(ram ? LOAD_RAM_VALUES_PER_KEY : LOAD_VRAM_VALUES_PER_KEY, valuesPerKey);
        call.progress(ram ? LOAD_RAM.at("progress") : LOAD_VRAM.at("progress"), progress);
        return call;
    }

    /** Runs one lifecycle verb and adopts the handle it created. */
    private static SequenceIndex create(Verb verb, @Nullable Executor executor, Consumer<Call> fill) {
        return Call.block(verb, null, executor, call -> {
            MemorySegment created = call.output(java.util.Objects.requireNonNull(verb.index()), Library.POINTER);
            fill.accept(call);
            return () -> new SequenceIndex(created.get(Library.POINTER, 0));
        });
    }

    static final Family FAMILY = Family.of("sequence");

    /** {@code usearch_sequence_costs_t}. */
    private static final StructLayout COSTS = MemoryLayout.structLayout(
            Library.POINTER.withName("similarities"),
            ValueLayout.JAVA_BYTE.withName("gap"),
            MemoryLayout.paddingLayout(7));

    private static final VarHandle COSTS_SIMILARITIES = Library.field(COSTS, "similarities");
    private static final VarHandle COSTS_GAP = Library.field(COSTS, "gap");

    /** {@code usearch_sequence_metadata_t}. */
    private static final StructLayout METADATA = MemoryLayout.structLayout(
            Library.ENUM.withName("metric"),
            Library.padding(),
            Library.SIZE.withName("connectivity"),
            Library.SIZE.withName("classes"),
            Library.SIZE.withName("sketch_bytes"));

    private static final VarHandle METADATA_METRIC = Library.field(METADATA, "metric");
    private static final VarHandle METADATA_CONNECTIVITY = Library.field(METADATA, "connectivity");
    private static final VarHandle METADATA_CLASSES = Library.field(METADATA, "classes");
    private static final VarHandle METADATA_SKETCH_BYTES = Library.field(METADATA, "sketch_bytes");

    /** {@code usearch_sequence_index_description_t}. */
    private static final StructLayout DESCRIPTION = MemoryLayout.structLayout(
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

    /** {@code usearch_sequence_key_stats_t}. */
    private static final StructLayout KEY_STATS = MemoryLayout.structLayout(
            Library.SIZE.withName("entries"),
            Library.SIZE.withName("total_bytes"),
            Library.SIZE.withName("largest_bytes"));

    /**
     * The fields {@code usearch_sequence_init_ram_t} and its VRAM twin share, at equal offsets. The
     * handles address it, as a RAM structure's handle refuses the shorter VRAM one.
     */
    private static final StructLayout INIT_SHARED = MemoryLayout.structLayout(
            Library.POINTER.withName("index"),
            Library.POINTER.withName("outcome"),
            Library.ENUM.withName("metric"),
            Library.padding(),
            Library.POINTER.withName("costs"),
            Library.SIZE.withName("connectivity"),
            Library.SIZE.withName("sketch_bytes"),
            Library.SIZE.withName("values_per_key"));

    /** {@code usearch_sequence_init_ram_t} or its VRAM twin, whose placement field is named {@code placement}. */
    private static StructLayout initializing(String placement) {
        return BasicCalls.followedBy(INIT_SHARED, Library.placement(placement));
    }

    private static final Verb INIT_RAM = new Verb("usearch_sequence_init_ram", initializing("ram"));
    private static final Verb INIT_VRAM = new Verb("usearch_sequence_init_vram", initializing("vram"));
    private static final VarHandle INIT_METRIC = Library.field(INIT_SHARED, "metric");
    private static final VarHandle INIT_COSTS = Library.field(INIT_SHARED, "costs");
    private static final VarHandle INIT_CONNECTIVITY = Library.field(INIT_SHARED, "connectivity");
    private static final VarHandle INIT_SKETCH_BYTES = Library.field(INIT_SHARED, "sketch_bytes");
    private static final VarHandle INIT_VALUES_PER_KEY = Library.field(INIT_SHARED, "values_per_key");

    /** {@code usearch_sequence_load_ram_t}, its VRAM twin, and their buffer twins, reading {@code source}. */
    private static StructLayout loadingLayout(String placement) {
        return Library.opening(
                Library.padding(),
                Library.BYTES_VIEW.withName("source"),
                Library.placement(placement),
                Library.SIZE.withName("connectivity"),
                Library.SIZE.withName("sketch_bytes"),
                Library.SIZE.withName("values_per_key"),
                Library.PROGRESS);
    }

    private static final Verb LOAD_RAM = new Verb("usearch_sequence_load_ram", loadingLayout("ram"));
    private static final Verb LOAD_RAM_BUFFER = new Verb("usearch_sequence_load_ram_buffer", loadingLayout("ram"));
    private static final Verb LOAD_VRAM = new Verb("usearch_sequence_load_vram", loadingLayout("vram"));
    private static final Verb LOAD_VRAM_BUFFER = new Verb("usearch_sequence_load_vram_buffer", loadingLayout("vram"));
    private static final long LOAD_SOURCE = LOAD_RAM.at("source");
    private static final VarHandle LOAD_RAM_VALUES_PER_KEY = LOAD_RAM.field("values_per_key");
    private static final VarHandle LOAD_VRAM_VALUES_PER_KEY = LOAD_VRAM.field("values_per_key");

    private static final Verb VIEW_RAM = new Verb(
            "usearch_sequence_view_ram",
            MemoryLayout.structLayout(
                    Library.POINTER.withName("index"),
                    Library.POINTER.withName("outcome"),
                    Library.BYTES_VIEW.withName("path"),
                    Library.placement("ram")));

    private static final Verb VIEW_NVME = new Verb(
            "usearch_sequence_view_nvme",
            Library.opening(
                    Library.padding(),
                    Library.BYTES_VIEW.withName("path"),
                    Library.placement("nvme"),
                    Library.SIZE.withName("sketch_bytes")));

    private static final Verb RESERVE = new Verb(
            "usearch_sequence_reserve",
            Library.opening(
                    Library.padding(), Library.SIZE.withName("entries"), Library.SIZE.withName("payload_bytes")));

    private static final Verb INSERT = new Verb(
            "usearch_sequence_insert",
            Library.opening(
                    Library.padding(),
                    Library.SIZE.withName("count"),
                    Library.KEYS_VIEW.withName("keys"),
                    BasicCalls.TAPE_VIEW.withName("sequences"),
                    Library.SIZE.withName("expansion_factor"),
                    Library.PROGRESS,
                    Library.POINTER.withName("stats")));

    private static final VarHandle INSERT_COUNT = INSERT.field("count");
    private static final VarHandle INSERT_EXPANSION_FACTOR = INSERT.field("expansion_factor");
    private static final VarHandle INSERT_STATS = INSERT.field("stats");

    private static final Verb SEARCH = new Verb(
            "usearch_sequence_search",
            Library.opening(
                    Library.padding(),
                    Library.SIZE.withName("count"),
                    BasicCalls.TAPE_VIEW.withName("queries"),
                    Library.POINTER.withName("matches"),
                    Library.POINTER.withName("arena"),
                    Library.SIZE.withName("wanted"),
                    Library.SIZE.withName("expansion_factor"),
                    Library.ENUM.withName("traversal"),
                    Library.padding(),
                    Library.PREDICATE,
                    Library.POINTER.withName("hints"),
                    Library.PROGRESS,
                    Library.POINTER.withName("stats")));

    private static final VarHandle SEARCH_COUNT = SEARCH.field("count");
    private static final VarHandle SEARCH_MATCHES = SEARCH.field("matches");
    private static final VarHandle SEARCH_ARENA = SEARCH.field("arena");
    private static final VarHandle SEARCH_WANTED = SEARCH.field("wanted");
    private static final VarHandle SEARCH_EXPANSION_FACTOR = SEARCH.field("expansion_factor");
    private static final VarHandle SEARCH_TRAVERSAL = SEARCH.field("traversal");
    private static final VarHandle SEARCH_STATS = SEARCH.field("stats");

    // The key follows the completion unpadded, since sixteen bytes align to one.
    private static final Verb GET = new Verb(
            "usearch_sequence_get",
            Library.opening(
                    Library.KEY.withName("key"),
                    Library.padding(),
                    Library.POINTER.withName("sequences"),
                    Library.POINTER.withName("written"),
                    Library.POINTER.withName("arena")));

    private static final VarHandle GET_SEQUENCES = GET.field("sequences");
    private static final VarHandle GET_WRITTEN = GET.field("written");
    private static final VarHandle GET_ARENA = GET.field("arena");

    private static final MethodHandle METRIC_NAME = Library.namer("usearch_sequence_metric_name");
    private static final MethodHandle METRIC_FROM_NAME = BasicCalls.nameReader("usearch_sequence_metric_from_name");
    private static final MethodHandle CPU_CLASS_NAME = Library.namer("usearch_cpu_class_name");
    private static final MethodHandle GPU_CLASS_NAME = Library.namer("usearch_gpu_class_name");
}
