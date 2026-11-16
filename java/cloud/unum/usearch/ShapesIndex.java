/**
 *  @file java/cloud/unum/usearch/ShapesIndex.java
 *  @author Ash Vardanian
 *  @date November 16, 2026
 *  @brief The shapes index handle, its options, and the receivers that run its verbs on a chosen executor.
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
import cloud.unum.usearch.Types.Scalar;
import cloud.unum.usearch.Types.Status;
import cloud.unum.usearch.Types.USearchException;
import cloud.unum.usearch.Types.Verb;

/**
 * A Similarity Search Index over 3D point clouds of any size - molecules, conformers and protein
 * backbones - under Chamfer, Hausdorff, Gromov-Wasserstein and Wasserstein-Procrustes, with the
 * modality's own options, metric and per-call settings as nested types.
 *
 * <p>
 * Every batch is a {@link Tape}: the points of every shape end to end, {@code x}, {@code y} and
 * {@code z} interleaved, and one more offset than shapes around them, counting points. The verbs, receivers, plans and reports are the {@link DenseIndex} ones:
 * the index runs every verb on the caller's thread, {@link #on} blocks on an executor, and
 * {@link #deferred} submits there and answers a {@link Deferred}. Predicates and progress handlers
 * run on the executor's threads and must not throw; an exception escaping one terminates the process.
 *
 * <pre>{@code
 * try (Executor pool = Executor.forRam(new int[0], null, 0);
 *         Arena arena = new Arena(pool);
 *         ShapesIndex index = ShapesIndex.initRam(new ShapesOptions(), new int[0])) {
 *     Tape shapes = Tape.ofPoints(new float[] {0, 0, 0, 1, 0, 0}, new float[] {0, 0, 0, 0, 2, 0, 0, 0, 2});
 *     index.on(pool).insert(new Key[] {Key.of(1), Key.of(2)}, shapes);
 *     Matches matches = index.on(pool).search(shapes, 1, arena);
 *     System.out.println(matches.key(0, 0));
 * }
 * }</pre>
 */
public final class ShapesIndex implements AutoCloseable {

    /** How two shapes compare, each invariant to rotation, reflection, translation and point order. */
    public enum ShapesMetric {

        /** A kind the library could not name. */
        UNKNOWN(0),
        /** Every point paired with its nearest partner, averaged both ways, after aligning principal axes. */
        CHAMFER(1),
        /** The farthest any point lies from its nearest partner, after aligning principal axes. */
        HAUSDORFF(2),
        /** How each side's internal distances compare, needing no superposition at all. */
        GROMOV_WASSERSTEIN(3),
        /** Points paired by an exact transport plan, alternated with the best rotation or reflection. */
        WASSERSTEIN_PROCRUSTES(4);

        private final int code;

        ShapesMetric(int code) {
            this.code = code;
        }

        /** The C enumerator's numeric value. */
        public int code() {
            return code;
        }

        /** Names the kind a code carries; a code this build cannot name becomes UNKNOWN. */
        public static ShapesMetric of(int code) {
            for (ShapesMetric candidate : values()) {
                if (candidate.code == code) {
                    return candidate;
                }
            }
            return UNKNOWN;
        }

        /** Renders the kind as the checkpoint spells it. Calls {@code usearch_shapes_metric_name}. */
        @Override
        public String toString() {
            return Library.text(METRIC_NAME, code);
        }

        /**
         * Reads a kind back from the spelling {@link #toString} gives it. Calls
         * {@code usearch_shapes_metric_from_name}.
         *
         * @throws USearchException INVALID_ARGUMENT on a name the library does not spell.
         */
        public static ShapesMetric fromName(String name) {
            return of(BasicCalls.fromName(METRIC_FROM_NAME, name));
        }
    }

    /**
     * The kind and point storage an empty index freezes - the fields {@code usearch_shapes_init_ram_t}
     * and {@code usearch_shapes_init_vram_t} share. Every setter returns this instance.
     */
    public static final class ShapesOptions {

        private ShapesMetric metric = ShapesMetric.CHAMFER;
        private Scalar quantization = Scalar.F32;
        private long points = 0;
        private long connectivity = 0;
        private long valuesPerKey = 0;
        private @Nullable CPUClass cpuClassCeiling = null;
        private @Nullable GPUClass gpuClassCeiling = null;

        /** A Chamfer index of F32 points, with every optional field derived. */
        public ShapesOptions() {}

        /** The comparison kind. */
        public ShapesOptions metric(ShapesMetric metric) {
            this.metric = metric;
            return this;
        }

        /** The scalar type points are stored in: F16 or F32, any other refused as UNSUPPORTED. */
        public ShapesOptions quantization(Scalar quantization) {
            this.quantization = quantization;
            return this;
        }

        /** The point count every entry must have, refusing others with SHAPE_MISMATCH; zero lets them vary. */
        public ShapesOptions points(long points) {
            this.points = points;
            return this;
        }

        /** The connections-per-node bound of the graph; zero derives. */
        public ShapesOptions connectivity(long connectivity) {
            this.connectivity = connectivity;
            return this;
        }

        /** The upper bound on entries sharing one key; zero is unbounded. */
        public ShapesOptions valuesPerKey(long valuesPerKey) {
            this.valuesPerKey = valuesPerKey;
            return this;
        }

        /** The newest CPU class dispatch may pick, intersected with what the machine runs; {@code null} means none. */
        public ShapesOptions cpuClassCeiling(@Nullable CPUClass cpuClassCeiling) {
            this.cpuClassCeiling = cpuClassCeiling;
            return this;
        }

        /** The newest GPU class dispatch may pick; {@code null} means no ceiling. */
        public ShapesOptions gpuClassCeiling(@Nullable GPUClass gpuClassCeiling) {
            this.gpuClassCeiling = gpuClassCeiling;
            return this;
        }

        /** Fills the fields both init structures share, at the same offsets in each. */
        void fill(Call call) {
            call.enumeration(INIT_METRIC, metric.code());
            call.enumeration(INIT_QUANTIZATION, quantization.code());
            call.size(INIT_POINTS, points);
            call.size(INIT_CONNECTIVITY, connectivity);
            call.size(INIT_VALUES_PER_KEY, valuesPerKey);
        }
    }

    /**
     * The kind and point storage a checkpoint froze. Mirrors {@code usearch_shapes_metadata_t}.
     *
     * @param metric the comparison kind the checkpoint was built under.
     * @param quantization the scalar type its points are stored in.
     * @param points the point count every entry has, or zero where they vary.
     * @param connectivity the connections per node the graph was built with.
     */
    public record ShapesMetadata(ShapesMetric metric, Scalar quantization, long points, long connectivity) {

        static ShapesMetadata lift(MemorySegment struct, long at) {
            return new ShapesMetadata(
                    ShapesMetric.of((int) METADATA_METRIC.get(struct, at)),
                    Scalar.of((int) METADATA_QUANTIZATION.get(struct, at)),
                    (long) METADATA_POINTS.get(struct, at),
                    (long) METADATA_CONNECTIVITY.get(struct, at));
        }
    }

    /**
     * A live index's state, each counter read once. Mirrors {@code usearch_shapes_index_description_t}.
     *
     * @param metadata the frozen kind, point storage and connectivity.
     * @param size the stored entries.
     * @param capacity the entries hosted without growing again.
     * @param totalPoints the points of every stored entry summed.
     * @param memoryUsage the bytes the handle holds, payload storage included.
     * @param cpuClass the CPU class dispatch picked, as {@code usearch_cpu_class_name} spells it.
     * @param gpuClass the GPU class dispatch picked; "unknown" for a host index.
     * @param shardsCount the live shards.
     */
    public record ShapesIndexDescription(
            ShapesMetadata metadata,
            long size,
            long capacity,
            long totalPoints,
            long memoryUsage,
            String cpuClass,
            String gpuClass,
            long shardsCount) {

        static ShapesIndexDescription lift(MemorySegment description) {
            return new ShapesIndexDescription(
                    ShapesMetadata.lift(description, DESCRIPTION_METADATA),
                    (long) DESCRIPTION_SIZE.get(description, 0L),
                    (long) DESCRIPTION_CAPACITY.get(description, 0L),
                    (long) DESCRIPTION_TOTAL_POINTS.get(description, 0L),
                    (long) DESCRIPTION_MEMORY_USAGE.get(description, 0L),
                    Library.text(CPU_CLASS_NAME, (int) DESCRIPTION_CPU_CLASS.get(description, 0L)),
                    Library.text(GPU_CLASS_NAME, (int) DESCRIPTION_GPU_CLASS.get(description, 0L)),
                    (long) DESCRIPTION_SHARDS_COUNT.get(description, 0L));
        }
    }

    /**
     * The entries one key holds and the extent of their points. Mirrors {@code usearch_shapes_key_stats_t}.
     *
     * @param entries the entries stored under the key; zero when it holds none.
     * @param totalPoints their points summed.
     * @param largestPoints the points of the largest.
     */
    public record ShapesKeyStats(long entries, long totalPoints, long largestPoints) {

        static ShapesKeyStats lift(MemorySegment stats) {
            long[] fields = stats.toArray(ValueLayout.JAVA_LONG);
            return new ShapesKeyStats(fields[0], fields[1], fields[2]);
        }
    }

    /**
     * The knobs of a refinement and the plan it continues; see {@link #refine(ShapesRefineConfig)}.
     * Mirrors {@code usearch_shapes_refine_t}.
     */
    public static final class ShapesRefineConfig {

        private long expansionFactor;
        private long memoryLimit;
        private @Nullable PlanCell plan;
        private @Nullable Progress progress;
        private @Nullable DenseTraversalStats stats;

        /** Every knob derived, one round run to its end. */
        public ShapesRefineConfig() {}

        /** The breadth of the candidates any walk beside the rounds explores; zero derives. */
        public ShapesRefineConfig expansionFactor(long expansionFactor) {
            this.expansionFactor = expansionFactor;
            return this;
        }

        /** The bytes a loaded NVMe index holds for the one part it rewrites; zero derives. */
        public ShapesRefineConfig memoryLimit(long memoryLimit) {
            this.memoryLimit = memoryLimit;
            return this;
        }

        /** The plan the call opens or continues, so a cancel leaves its round resumable. */
        public ShapesRefineConfig plan(ShapesRefinePlan plan) {
            this.plan = plan.cell;
            return this;
        }

        /** Hears every barrier of every phase, and may cancel the round there. */
        public ShapesRefineConfig progress(Progress progress) {
            this.progress = progress;
            return this;
        }

        /** Where the traversal telemetry lands, overwritten once the call succeeded. */
        public ShapesRefineConfig stats(DenseTraversalStats stats) {
            this.stats = stats;
            return this;
        }

        Rounds rounds() {
            return new Rounds(expansionFactor, memoryLimit, plan, progress, stats);
        }
    }

    /**
     * The plan a reshard continues, the seed of its cut and the knobs of its closing rounds; see
     * {@link #reshard(ShapesIndex[], ShapesIndex[], ShapesReshardConfig)}. Mirrors
     * {@code usearch_shapes_reshard_t}.
     */
    public static final class ShapesReshardConfig {

        private long expansionFactor;
        private long memoryLimit;
        private long seed;
        private @Nullable PlanCell plan;
        private @Nullable Progress progress;
        private @Nullable DenseTraversalStats stats;

        /** Every knob derived, under seed zero. */
        public ShapesReshardConfig() {}

        /** The breadth of the candidates any walk beside the closing rounds explores; zero derives. */
        public ShapesReshardConfig expansionFactor(long expansionFactor) {
            this.expansionFactor = expansionFactor;
            return this;
        }

        /** The bytes a loaded NVMe target holds for the one part it rewrites; zero derives. */
        public ShapesReshardConfig memoryLimit(long memoryLimit) {
            this.memoryLimit = memoryLimit;
            return this;
        }

        /** Picks the part roots of a cut: equal seeds over an unchanged source cut equally. */
        public ShapesReshardConfig seed(long seed) {
            this.seed = seed;
            return this;
        }

        /** The plan the call opens or continues, so a cancel leaves its closing rounds resumable. */
        public ShapesReshardConfig plan(ShapesReshardPlan plan) {
            this.plan = plan.cell;
            return this;
        }

        /** Hears every barrier of the closing rounds, and may cancel them there; a move is never cancelled. */
        public ShapesReshardConfig progress(Progress progress) {
            this.progress = progress;
            return this;
        }

        /** Where the traversal telemetry lands, overwritten once the call succeeded. */
        public ShapesReshardConfig stats(DenseTraversalStats stats) {
            this.stats = stats;
            return this;
        }

        Rounds rounds() {
            return new Rounds(expansionFactor, memoryLimit, plan, progress, stats);
        }
    }

    /**
     * A refinement in flight, which {@link #refine(ShapesRefineConfig)} opens on its first call and
     * continues on every later one. The index admits no insertion and no save while it is open, and
     * every call naming it names the index and executor it was opened with;
     * {@link #refinePlanFinish} closes it.
     */
    public static final class ShapesRefinePlan {

        final PlanCell cell = new PlanCell();

        /** A plan no call has opened yet. */
        public ShapesRefinePlan() {}

        /** Where the last call naming the plan stopped. */
        public PhaseProgress phase() {
            return cell.phase();
        }
    }

    /**
     * A reshard in flight, which {@link #reshard(ShapesIndex[], ShapesIndex[], ShapesReshardConfig)}
     * opens on its first call and continues on every later one. Every index it names admits no
     * insertion and no save while it is open; {@link #reshardPlanFinish} closes it.
     */
    public static final class ShapesReshardPlan {

        final PlanCell cell = new PlanCell();

        /** A plan no call has opened yet. */
        public ShapesReshardPlan() {}

        /** Where the last call naming the plan stopped. */
        public PhaseProgress phase() {
            return cell.phase();
        }
    }

    /**
     * A sequential walk over the index contents, pulled blocking on the executor of the receiver that
     * opened it; the shapes themselves come from {@link ShapesIndex#get}. An overlapping pull
     * reports CONTENDED.
     */
    public static final class ShapesCursor implements AutoCloseable {

        private final Handle handle;
        private final ShapesIndex index;
        private final @Nullable Executor executor;

        ShapesCursor(MemorySegment opened, ShapesIndex index, @Nullable Executor executor) {
            this.handle = new Handle(opened, FAMILY.cursorFree());
            this.index = index;
            this.executor = executor;
        }

        /** Copies the keys of up to {@code wanted} next entries into {@code arena}. Calls {@code usearch_shapes_cursor_next}. */
        public DenseBatch next(long wanted, Arena arena) {
            return DenseIndex.pulling(FAMILY.cursorNext(), index.handle, executor, handle, wanted, false, arena);
        }

        /** Copies up to {@code wanted} next entries, each with the keys it links to. Calls {@code usearch_shapes_cursor_next}. */
        public DenseBatch nextLinks(long wanted, Arena arena) {
            return DenseIndex.pulling(FAMILY.cursorNext(), index.handle, executor, handle, wanted, true, arena);
        }

        /** Releases the cursor; idempotent. Calls {@code usearch_shapes_cursor_free}. */
        @Override
        public void close() {
            handle.close();
        }
    }

    /** The receiver whose verbs block on an executor, contended while another call holds it. */
    public static final class On {

        private final ShapesIndex index;
        private final Executor executor;

        On(ShapesIndex index, Executor executor) {
            this.index = index;
            this.executor = executor;
        }

        private <Result> Result block(Request<Result> request) {
            return Call.block(request.verb(), index.handle, executor, request.fill());
        }

        /** Runs {@link ShapesIndex#reserve} on this receiver's executor. */
        public void reserve(long entries, long points) {
            block(reserving(entries, points));
        }

        /** Runs {@link ShapesIndex#clear} on this receiver's executor. */
        public void clear() {
            block(DenseCalls.clearing(FAMILY.clear()));
        }

        /** Runs {@link ShapesIndex#describe} on this receiver's executor. */
        public ShapesIndexDescription describe() {
            return block(describing());
        }

        /** Runs {@link ShapesIndex#count} on this receiver's executor. */
        public ShapesKeyStats count(Key key) {
            return block(counting(key));
        }

        /** Runs {@link ShapesIndex#contains} on this receiver's executor. */
        public boolean contains(Key key) {
            return count(key).entries() != 0;
        }

        /** Runs {@link ShapesIndex#get} on this receiver's executor. */
        public Tape get(Key key, Scalar outputsScalar, Arena arena) {
            return block(getting(key, outputsScalar, arena));
        }

        /** Runs {@link ShapesIndex#insert(Key[], Tape)} on this receiver's executor. */
        public void insert(Key[] keys, Tape shapes) {
            block(inserting(keys, shapes, new DenseInsertConfig()));
        }

        /** Runs {@link ShapesIndex#insert(Key[], Tape, DenseInsertConfig)} on this receiver's executor. */
        public void insert(Key[] keys, Tape shapes, DenseInsertConfig config) {
            block(inserting(keys, shapes, config));
        }

        /** Runs {@link ShapesIndex#refine()} on this receiver's executor. */
        public RefineReport refine() {
            return block(BasicCalls.refining(FAMILY, new ShapesRefineConfig().rounds()));
        }

        /** Runs {@link ShapesIndex#refine(ShapesRefineConfig)} on this receiver's executor. */
        public RefineReport refine(ShapesRefineConfig config) {
            return block(BasicCalls.refining(FAMILY, config.rounds()));
        }

        /** Runs {@link ShapesIndex#refinePlanFinish} on this receiver's executor. */
        public RefineReport refinePlanFinish(ShapesRefinePlan plan) {
            return block(BasicCalls.refinePlanFinishing(FAMILY, plan.cell));
        }

        /** Runs {@link ShapesIndex#health} on this receiver's executor. */
        public DenseReachability health() {
            return block(DenseCalls.auditing(FAMILY.health()));
        }

        /** Runs {@link ShapesIndex#search(Tape, long, Arena)} on this receiver's executor. */
        public Matches search(Tape queries, long wanted, Arena arena) {
            return block(searching(queries, new DenseSearchConfig(wanted), arena));
        }

        /** Runs {@link ShapesIndex#search(Tape, DenseSearchConfig, Arena)} on this receiver's executor. */
        public Matches search(Tape queries, DenseSearchConfig config, Arena arena) {
            return block(searching(queries, config, arena));
        }

        /** Runs {@link ShapesIndex#reshard} on this receiver's executor. */
        public RefineReport reshard(ShapesIndex[] sources, ShapesIndex[] targets, ShapesReshardConfig config) {
            return block(resharding(sources, targets, config));
        }

        /** Runs {@link ShapesIndex#reshardPlanFinish} on this receiver's executor. */
        public RefineReport reshardPlanFinish(ShapesReshardPlan plan) {
            return block(BasicCalls.reshardPlanFinishing(FAMILY, plan.cell));
        }

        /** Runs {@link ShapesIndex#sample} on this receiver's executor. */
        public Sample sample(long wanted, Arena arena, DenseSampleConfig config) {
            return block(DenseCalls.sampling(FAMILY.sample(), wanted, arena, config));
        }

        /** Runs {@link ShapesIndex#save} on this receiver's executor. */
        public void save(String path) {
            block(DenseCalls.saving(FAMILY.save(), path, null));
        }

        /** Runs {@link ShapesIndex#saveBuffer} on this receiver's executor. */
        public Monolith saveBuffer(Arena arena) {
            return block(DenseCalls.serializing(FAMILY.saveBuffer(), arena, null));
        }

        /** Runs {@link ShapesIndex#cursor} on this receiver's executor. */
        public ShapesCursor cursor(Key... starts) {
            return block(BasicCalls.cursorOpening(FAMILY, starts, opened -> new ShapesCursor(opened, index, executor)));
        }
    }

    /**
     * The receiver whose verbs submit their calls to an executor, each answering a {@link Deferred}
     * that owns what the call reads until final. Its verbs are those of {@link ShapesIndex}, less
     * the cursor that pulls one batch at a time.
     */
    public static final class DeferredOn {

        private final ShapesIndex index;
        private final Executor executor;

        DeferredOn(ShapesIndex index, Executor executor) {
            this.index = index;
            this.executor = executor;
        }

        private <Result> Deferred<Result> submit(Request<Result> request) {
            return Call.submit(request.verb(), index.handle, executor, request.fill());
        }

        /** Submits {@link ShapesIndex#reserve}. Calls {@code usearch_shapes_reserve}. */
        public Deferred<Void> reserve(long entries, long points) {
            return submit(reserving(entries, points));
        }

        /** Submits {@link ShapesIndex#clear}. Calls {@code usearch_shapes_clear}. */
        public Deferred<Void> clear() {
            return submit(DenseCalls.clearing(FAMILY.clear()));
        }

        /** Submits {@link ShapesIndex#describe}. Calls {@code usearch_shapes_describe}. */
        public Deferred<ShapesIndexDescription> describe() {
            return submit(describing());
        }

        /** Submits {@link ShapesIndex#count}. Calls {@code usearch_shapes_count}. */
        public Deferred<ShapesKeyStats> count(Key key) {
            return submit(counting(key));
        }

        /** Submits {@link ShapesIndex#get}. Calls {@code usearch_shapes_get}. */
        public Deferred<Tape> get(Key key, Scalar outputsScalar, Arena arena) {
            return submit(getting(key, outputsScalar, arena));
        }

        /** Submits {@link ShapesIndex#insert(Key[], Tape, DenseInsertConfig)}. Calls {@code usearch_shapes_insert}. */
        public Deferred<Void> insert(Key[] keys, Tape shapes, DenseInsertConfig config) {
            return submit(inserting(keys, shapes, config));
        }

        /** Submits {@link ShapesIndex#refine(ShapesRefineConfig)}. Calls {@code usearch_shapes_refine}. */
        public Deferred<RefineReport> refine(ShapesRefineConfig config) {
            return submit(BasicCalls.refining(FAMILY, config.rounds()));
        }

        /** Submits {@link ShapesIndex#refinePlanFinish}. Calls {@code usearch_shapes_refine_plan_finish}. */
        public Deferred<RefineReport> refinePlanFinish(ShapesRefinePlan plan) {
            return submit(BasicCalls.refinePlanFinishing(FAMILY, plan.cell));
        }

        /** Submits {@link ShapesIndex#health}. Calls {@code usearch_shapes_health}. */
        public Deferred<DenseReachability> health() {
            return submit(DenseCalls.auditing(FAMILY.health()));
        }

        /** Submits {@link ShapesIndex#search(Tape, DenseSearchConfig, Arena)}. Calls {@code usearch_shapes_search}. */
        public Deferred<Matches> search(Tape queries, DenseSearchConfig config, Arena arena) {
            return submit(searching(queries, config, arena));
        }

        /** Submits {@link ShapesIndex#reshard}. Calls {@code usearch_shapes_reshard}. */
        public Deferred<RefineReport> reshard(
                ShapesIndex[] sources, ShapesIndex[] targets, ShapesReshardConfig config) {
            return submit(resharding(sources, targets, config));
        }

        /** Submits {@link ShapesIndex#reshardPlanFinish}. Calls {@code usearch_shapes_reshard_plan_finish}. */
        public Deferred<RefineReport> reshardPlanFinish(ShapesReshardPlan plan) {
            return submit(BasicCalls.reshardPlanFinishing(FAMILY, plan.cell));
        }

        /** Submits {@link ShapesIndex#sample}. Calls {@code usearch_shapes_sample}. */
        public Deferred<Sample> sample(long wanted, Arena arena, DenseSampleConfig config) {
            return submit(DenseCalls.sampling(FAMILY.sample(), wanted, arena, config));
        }

        /** Submits {@link ShapesIndex#save}. Calls {@code usearch_shapes_save}. */
        public Deferred<Void> save(String path) {
            return submit(DenseCalls.saving(FAMILY.save(), path, null));
        }

        /** Submits {@link ShapesIndex#saveBuffer}. Calls {@code usearch_shapes_save_buffer}. */
        public Deferred<Monolith> saveBuffer(Arena arena) {
            return submit(DenseCalls.serializing(FAMILY.saveBuffer(), arena, null));
        }
    }

    final Handle handle;

    private ShapesIndex(MemorySegment created) {
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
     * Claims room for entries and their points ahead of insertion, a hint, since insertions grow the
     * index on demand. Calls {@code usearch_shapes_reserve}.
     *
     * @param entries the desired total number of entries, including the current size.
     * @param points the desired total points.
     * @throws USearchException READ_ONLY on a view.
     */
    public void reserve(long entries, long points) {
        block(reserving(entries, points));
    }

    /** Erases all entries, keeping the frozen kind. Calls {@code usearch_shapes_clear}. */
    public void clear() {
        block(DenseCalls.clearing(FAMILY.clear()));
    }

    /** Reports the frozen metadata and the current counters in one call. Calls {@code usearch_shapes_describe}. */
    public ShapesIndexDescription describe() {
        return block(describing());
    }

    /**
     * How many entries a key holds, with their total and largest byte lengths. Calls
     * {@code usearch_shapes_count}.
     */
    public ShapesKeyStats count(Key key) {
        return block(counting(key));
    }

    /** Whether the index holds at least one entry under a key. Binding-only sugar over {@link #count}. */
    public boolean contains(Key key) {
        return count(key).entries() != 0;
    }

    /**
     * Retrieves every shape stored under a key into {@code arena}, in insertion order and in its
     * inserted pose, converted to {@code outputsScalar}. Calls {@code usearch_shapes_get}.
     *
     * @return the tape read in place, of no entries when the key holds nothing.
     * @throws USearchException CONTENDED when the arena serves another call.
     */
    public Tape get(Key key, Scalar outputsScalar, Arena arena) {
        return block(getting(key, outputsScalar, arena));
    }

    /**
     * Inserts shapes under their keys, with every knob derived.
     *
     * @throws USearchException the statuses {@link #insert(Key[], Tape, DenseInsertConfig)} names.
     */
    public void insert(Key[] keys, Tape shapes) {
        block(inserting(keys, shapes, new DenseInsertConfig()));
    }

    /**
     * Inserts a tape of shapes under their keys, point counts free to differ - all-or-nothing per call. Calls
     * {@code usearch_shapes_insert}.
     *
     * @throws USearchException SHAPE_MISMATCH when the key count and the tape's entries disagree,
     *     or an entry's point count differs from a frozen one, READ_ONLY on a view, and OVERFLOW when a key already holds {@code valuesPerKey} entries.
     */
    public void insert(Key[] keys, Tape shapes, DenseInsertConfig config) {
        block(inserting(keys, shapes, config));
    }

    /**
     * Runs one round over every stored entry, with every knob derived.
     *
     * @throws USearchException the statuses {@link #refine(ShapesRefineConfig)} names.
     */
    public RefineReport refine() {
        return block(BasicCalls.refining(FAMILY, new ShapesRefineConfig().rounds()));
    }

    /**
     * Re-selects the neighbors of every stored entry against the finished graph, as
     * {@link DenseIndex#refine(DenseIndex.DenseRefineConfig)} does. Calls {@code usearch_shapes_refine}.
     *
     * @throws USearchException READ_ONLY on a view, and INVALID_ARGUMENT on a plan without an executor.
     */
    public RefineReport refine(ShapesRefineConfig config) {
        return block(BasicCalls.refining(FAMILY, config.rounds()));
    }

    /**
     * Closes the round a cancel left in hand and frees the plan whatever the call answers. Calls
     * {@code usearch_shapes_refine_plan_finish}.
     *
     * @throws USearchException INVALID_ARGUMENT on a plan no call opened, or one already finished.
     */
    public RefineReport refinePlanFinish(ShapesRefinePlan plan) {
        return block(BasicCalls.refinePlanFinishing(FAMILY, plan.cell));
    }

    /** Audits how much of the graph a search reaches. Calls {@code usearch_shapes_health}. */
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
     * {@code usearch_shapes_search}.
     *
     * @throws USearchException CONTENDED when the arena serves another call.
     */
    public Matches search(Tape queries, DenseSearchConfig config, Arena arena) {
        return block(searching(queries, config, arena));
    }

    /**
     * Moves the entries of {@code sources} into {@code targets}, then runs a round on every target that
     * took entries, as {@link DenseIndex#reshard} does. Calls {@code usearch_shapes_reshard}.
     *
     * @throws USearchException INVALID_ARGUMENT on no sources or targets or an index named twice,
     *     CONFIG_MISMATCH on another kind or a target neither a source nor empty, and UNSUPPORTED on
     *     one index as its own only source and target.
     */
    public RefineReport reshard(ShapesIndex[] sources, ShapesIndex[] targets, ShapesReshardConfig config) {
        return block(resharding(sources, targets, config));
    }

    /**
     * Closes the rounds a cancel stopped and frees the plan whatever the call answers. Calls
     * {@code usearch_shapes_reshard_plan_finish}.
     *
     * @throws USearchException INVALID_ARGUMENT on a plan no call opened.
     */
    public RefineReport reshardPlanFinish(ShapesReshardPlan plan) {
        return block(BasicCalls.reshardPlanFinishing(FAMILY, plan.cell));
    }

    /**
     * Draws keys around the focus, or across the whole index, placed in three dimensions, as
     * {@link DenseIndex#sample} does. Calls {@code usearch_shapes_sample}.
     */
    public Sample sample(long wanted, Arena arena, DenseSampleConfig config) {
        return block(DenseCalls.sampling(FAMILY.sample(), wanted, arena, config));
    }

    /**
     * Writes the index under a path: one monolith for a lone shard, or a manifest beside one part per
     * shard. Calls {@code usearch_shapes_save}.
     *
     * @throws USearchException IO when the destination cannot be written.
     */
    public void save(String path) {
        block(DenseCalls.saving(FAMILY.save(), path, null));
    }

    /** Writes the index into {@code arena}, the bytes {@link #save} writes. Calls {@code usearch_shapes_save_buffer}. */
    public Monolith saveBuffer(Arena arena) {
        return block(DenseCalls.serializing(FAMILY.saveBuffer(), arena, null));
    }

    /**
     * Opens a snapshot cursor over the contents, every entry as stored or breadth-first from
     * {@code starts}. Calls {@code usearch_shapes_cursor_init}.
     */
    public ShapesCursor cursor(Key... starts) {
        return block(BasicCalls.cursorOpening(FAMILY, starts, opened -> new ShapesCursor(opened, this, null)));
    }

    /**
     * Creates an empty, mutable index in host memory. Calls {@code usearch_shapes_init_ram}.
     *
     * @param memoryDomains NUMA node ids to place shards on; an empty list derives every node.
     * @throws USearchException UNKNOWN_KIND on an unknown kind, and UNSUPPORTED on points stored
     *     other than F16 or F32.
     */
    public static ShapesIndex initRam(ShapesOptions options, int[] memoryDomains) {
        return create(INIT_RAM, null, call -> {
            options.fill(call);
            call.ram(call.arguments, INIT_RAM.at("ram"), memoryDomains, options.cpuClassCeiling);
        });
    }

    /**
     * Creates an empty, mutable index in device memory. Calls {@code usearch_shapes_init_vram}.
     *
     * @throws USearchException DEVICE when no device answers or the build has no GPU kernels.
     */
    public static ShapesIndex initVram(ShapesOptions options, int[] devices) {
        return create(INIT_VRAM, null, call -> {
            options.fill(call);
            call.vram(call.arguments, INIT_VRAM.at("vram"), devices, options.gpuClassCeiling);
        });
    }

    /**
     * Copies a checkpoint into a new, mutable index in host memory. Calls {@code usearch_shapes_load_ram}.
     *
     * @throws USearchException MALFORMED on a checkpoint that ends early, and IO when it cannot be read.
     */
    public static ShapesIndex loadRam(
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

    /** Copies a monolith from a buffer into a new index in host memory. Calls {@code usearch_shapes_load_ram_buffer}. */
    public static ShapesIndex loadRamBuffer(
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

    /** Copies a checkpoint into a new index in device memory. Calls {@code usearch_shapes_load_vram}. */
    public static ShapesIndex loadVram(
            String path, @Nullable Executor executor, int[] devices, long valuesPerKey, @Nullable Progress progress) {
        return create(
                LOAD_VRAM,
                executor,
                call -> loading(call, false, devices, valuesPerKey, progress).path(LOAD_SOURCE, path));
    }

    /** Copies a monolith from a buffer into a new index in device memory. Calls {@code usearch_shapes_load_vram_buffer}. */
    public static ShapesIndex loadVramBuffer(
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
     * {@code usearch_shapes_view_ram}.
     *
     * @throws USearchException UNSUPPORTED where the engine maps no shapes checkpoint.
     */
    public static ShapesIndex viewRam(String path) {
        return create(VIEW_RAM, null, call -> call.path(VIEW_RAM.at("path"), path));
    }

    /**
     * Serves a checkpoint in place from NVMe drives as a read-only index. Calls
     * {@code usearch_shapes_view_nvme}.
     *
     * @param paths devices or directories to serve the parts from; an empty list serves each where its
     *     manifest points.
     */
    public static ShapesIndex viewNvme(
            String path, @Nullable Executor executor, String[] paths, @Nullable CPUClass cpuClassCeiling) {
        return create(VIEW_NVME, executor, call -> {
            call.path(VIEW_NVME.at("path"), path);
            call.nvme(call.arguments, VIEW_NVME.at("nvme"), paths, cpuClassCeiling);
        });
    }

    /** Reads a checkpoint's kind and connectivity without loading it. Calls {@code usearch_shapes_metadata_read}. */
    public static ShapesMetadata metadataRead(String path, @Nullable Executor executor) {
        return BasicCalls.metadata(
                FAMILY.metadataRead(),
                executor,
                call -> call.path(BasicCalls.READ_SOURCE, path),
                METADATA,
                metadata -> ShapesMetadata.lift(metadata, 0));
    }

    /** Reads a monolith's kind and connectivity without loading it. Calls {@code usearch_shapes_metadata_read_buffer}. */
    public static ShapesMetadata metadataReadBuffer(MemorySegment buffer, @Nullable Executor executor) {
        return BasicCalls.metadata(
                FAMILY.metadataReadBuffer(),
                executor,
                call -> call.bytes(BasicCalls.READ_SOURCE, buffer),
                METADATA,
                metadata -> ShapesMetadata.lift(metadata, 0));
    }

    /**
     * Releases the index, after every deferred call naming it is final. It is idempotent. Calls
     * {@code usearch_shapes_free}.
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
        return "ShapesIndex(" + handle + ")";
    }

    private static Request<Void> reserving(long entries, long points) {
        return new Request<>(RESERVE, call -> {
            call.size(RESERVE.field("entries"), entries);
            call.size(RESERVE.field("points"), points);
            return () -> null;
        });
    }

    private static Request<ShapesIndexDescription> describing() {
        return BasicCalls.describing(FAMILY, DESCRIPTION, ShapesIndexDescription::lift);
    }

    private static Request<ShapesKeyStats> counting(Key key) {
        return BasicCalls.counting(FAMILY, key, KEY_STATS, ShapesKeyStats::lift);
    }

    private static Request<Tape> getting(Key key, Scalar outputsScalar, Arena arena) {
        return new Request<>(GET, call -> {
            call.key(GET.at("key"), key);
            MemorySegment shapes = call.output(GET_SHAPES, BasicCalls.TAPE_VIEW);
            MemorySegment written = call.output(GET_WRITTEN, Library.SIZE);
            call.arena(GET_ARENA, arena);
            call.enumeration(GET_OUTPUTS_SCALAR, outputsScalar.code());
            return () -> Tape.lift(call, shapes, written.get(Library.SIZE, 0), 3);
        });
    }

    private static Request<Void> inserting(Key[] keys, Tape shapes, DenseInsertConfig config) {
        // The call has one count, so only this side sees the keys and the tape disagree.
        if (keys.length != shapes.count()) {
            throw new USearchException(Status.SHAPE_MISMATCH, "the key count does not match the tape's entries");
        }
        return new Request<>(INSERT, call -> {
            call.size(INSERT_COUNT, keys.length);
            call.keys(INSERT.at("keys"), keys);
            shapes.fill(call, INSERT.at("shapes"));
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
            ShapesIndex[] sources, ShapesIndex[] targets, ShapesReshardConfig config) {
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
    private static ShapesIndex create(Verb verb, @Nullable Executor executor, Consumer<Call> fill) {
        return Call.block(verb, null, executor, call -> {
            MemorySegment created = call.output(java.util.Objects.requireNonNull(verb.index()), Library.POINTER);
            fill.accept(call);
            return () -> new ShapesIndex(created.get(Library.POINTER, 0));
        });
    }

    static final Family FAMILY = Family.of("shapes");

    /** {@code usearch_shapes_metadata_t}. */
    private static final StructLayout METADATA = MemoryLayout.structLayout(
            Library.ENUM.withName("metric"),
            Library.ENUM.withName("quantization"),
            Library.SIZE.withName("points"),
            Library.SIZE.withName("connectivity"));

    private static final VarHandle METADATA_METRIC = Library.field(METADATA, "metric");
    private static final VarHandle METADATA_QUANTIZATION = Library.field(METADATA, "quantization");
    private static final VarHandle METADATA_POINTS = Library.field(METADATA, "points");
    private static final VarHandle METADATA_CONNECTIVITY = Library.field(METADATA, "connectivity");

    /** {@code usearch_shapes_index_description_t}. */
    private static final StructLayout DESCRIPTION = MemoryLayout.structLayout(
            METADATA.withName("metadata"),
            Library.SIZE.withName("size"),
            Library.SIZE.withName("capacity"),
            Library.SIZE.withName("total_points"),
            Library.SIZE.withName("memory_usage"),
            Library.ENUM.withName("cpu_class"),
            Library.ENUM.withName("gpu_class"),
            Library.SIZE.withName("shards_count"),
            Library.ENUM.withName("load_path"),
            Library.padding());

    private static final long DESCRIPTION_METADATA = Library.offset(DESCRIPTION, "metadata");
    private static final VarHandle DESCRIPTION_SIZE = Library.field(DESCRIPTION, "size");
    private static final VarHandle DESCRIPTION_CAPACITY = Library.field(DESCRIPTION, "capacity");
    private static final VarHandle DESCRIPTION_TOTAL_POINTS = Library.field(DESCRIPTION, "total_points");
    private static final VarHandle DESCRIPTION_MEMORY_USAGE = Library.field(DESCRIPTION, "memory_usage");
    private static final VarHandle DESCRIPTION_CPU_CLASS = Library.field(DESCRIPTION, "cpu_class");
    private static final VarHandle DESCRIPTION_GPU_CLASS = Library.field(DESCRIPTION, "gpu_class");
    private static final VarHandle DESCRIPTION_SHARDS_COUNT = Library.field(DESCRIPTION, "shards_count");

    /** {@code usearch_shapes_key_stats_t}. */
    private static final StructLayout KEY_STATS = MemoryLayout.structLayout(
            Library.SIZE.withName("entries"),
            Library.SIZE.withName("total_points"),
            Library.SIZE.withName("largest_points"));

    /**
     * The fields {@code usearch_shapes_init_ram_t} and its VRAM twin share, at equal offsets. The
     * handles address it, as a RAM structure's handle refuses the shorter VRAM one.
     */
    private static final StructLayout INIT_SHARED = MemoryLayout.structLayout(
            Library.POINTER.withName("index"),
            Library.POINTER.withName("outcome"),
            Library.ENUM.withName("metric"),
            Library.ENUM.withName("quantization"),
            Library.SIZE.withName("points"),
            Library.SIZE.withName("connectivity"),
            Library.SIZE.withName("values_per_key"));

    /** {@code usearch_shapes_init_ram_t} or its VRAM twin, whose placement field is named {@code placement}. */
    private static StructLayout initializing(String placement) {
        return BasicCalls.followedBy(INIT_SHARED, Library.placement(placement));
    }

    private static final Verb INIT_RAM = new Verb("usearch_shapes_init_ram", initializing("ram"));
    private static final Verb INIT_VRAM = new Verb("usearch_shapes_init_vram", initializing("vram"));
    private static final VarHandle INIT_METRIC = Library.field(INIT_SHARED, "metric");
    private static final VarHandle INIT_QUANTIZATION = Library.field(INIT_SHARED, "quantization");
    private static final VarHandle INIT_POINTS = Library.field(INIT_SHARED, "points");
    private static final VarHandle INIT_CONNECTIVITY = Library.field(INIT_SHARED, "connectivity");
    private static final VarHandle INIT_VALUES_PER_KEY = Library.field(INIT_SHARED, "values_per_key");

    /** {@code usearch_shapes_load_ram_t}, its VRAM twin, and their buffer twins, reading {@code source}. */
    private static StructLayout loadingLayout(String placement) {
        return Library.opening(
                Library.padding(),
                Library.BYTES_VIEW.withName("source"),
                Library.placement(placement),
                Library.SIZE.withName("values_per_key"),
                Library.PROGRESS);
    }

    private static final Verb LOAD_RAM = new Verb("usearch_shapes_load_ram", loadingLayout("ram"));
    private static final Verb LOAD_RAM_BUFFER = new Verb("usearch_shapes_load_ram_buffer", loadingLayout("ram"));
    private static final Verb LOAD_VRAM = new Verb("usearch_shapes_load_vram", loadingLayout("vram"));
    private static final Verb LOAD_VRAM_BUFFER = new Verb("usearch_shapes_load_vram_buffer", loadingLayout("vram"));
    private static final long LOAD_SOURCE = LOAD_RAM.at("source");
    private static final VarHandle LOAD_RAM_VALUES_PER_KEY = LOAD_RAM.field("values_per_key");
    private static final VarHandle LOAD_VRAM_VALUES_PER_KEY = LOAD_VRAM.field("values_per_key");

    private static final Verb VIEW_RAM = new Verb(
            "usearch_shapes_view_ram",
            MemoryLayout.structLayout(
                    Library.POINTER.withName("index"),
                    Library.POINTER.withName("outcome"),
                    Library.BYTES_VIEW.withName("path"),
                    Library.placement("ram")));

    private static final Verb VIEW_NVME = new Verb(
            "usearch_shapes_view_nvme",
            Library.opening(Library.padding(), Library.BYTES_VIEW.withName("path"), Library.placement("nvme")));

    private static final Verb RESERVE = new Verb(
            "usearch_shapes_reserve",
            Library.opening(Library.padding(), Library.SIZE.withName("entries"), Library.SIZE.withName("points")));

    private static final Verb INSERT = new Verb(
            "usearch_shapes_insert",
            Library.opening(
                    Library.padding(),
                    Library.SIZE.withName("count"),
                    Library.KEYS_VIEW.withName("keys"),
                    BasicCalls.TAPE_VIEW.withName("shapes"),
                    Library.SIZE.withName("expansion_factor"),
                    Library.PROGRESS,
                    Library.POINTER.withName("stats")));

    private static final VarHandle INSERT_COUNT = INSERT.field("count");
    private static final VarHandle INSERT_EXPANSION_FACTOR = INSERT.field("expansion_factor");
    private static final VarHandle INSERT_STATS = INSERT.field("stats");

    private static final Verb SEARCH = new Verb(
            "usearch_shapes_search",
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
            "usearch_shapes_get",
            Library.opening(
                    Library.KEY.withName("key"),
                    Library.padding(),
                    Library.POINTER.withName("shapes"),
                    Library.POINTER.withName("written"),
                    Library.POINTER.withName("arena"),
                    Library.ENUM.withName("outputs_scalar"),
                    Library.padding()));

    private static final VarHandle GET_SHAPES = GET.field("shapes");
    private static final VarHandle GET_OUTPUTS_SCALAR = GET.field("outputs_scalar");
    private static final VarHandle GET_WRITTEN = GET.field("written");
    private static final VarHandle GET_ARENA = GET.field("arena");

    private static final MethodHandle METRIC_NAME = Library.namer("usearch_shapes_metric_name");
    private static final MethodHandle METRIC_FROM_NAME = BasicCalls.nameReader("usearch_shapes_metric_from_name");
    private static final MethodHandle CPU_CLASS_NAME = Library.namer("usearch_cpu_class_name");
    private static final MethodHandle GPU_CLASS_NAME = Library.namer("usearch_gpu_class_name");
}
