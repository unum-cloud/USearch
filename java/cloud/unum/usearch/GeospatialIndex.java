/**
 *  @file java/cloud/unum/usearch/GeospatialIndex.java
 *  @author Ash Vardanian
 *  @date November 16, 2026
 *  @brief The geospatial index handle, its options, and the receivers that run its verbs on a chosen executor.
 */
package cloud.unum.usearch;

import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.StructLayout;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.VarHandle;
import java.util.function.Consumer;
import java.util.function.Predicate;

import org.jspecify.annotations.Nullable;

import cloud.unum.usearch.BasicCalls.Family;
import cloud.unum.usearch.BasicCalls.PlanCell;
import cloud.unum.usearch.BasicCalls.Rounds;
import cloud.unum.usearch.DenseCalls.Request;
import cloud.unum.usearch.DenseIndex.DenseBatch;
import cloud.unum.usearch.DenseIndex.DenseInsertConfig;
import cloud.unum.usearch.DenseIndex.DenseProducts;
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
import cloud.unum.usearch.Types.Traversal;
import cloud.unum.usearch.Types.USearchException;
import cloud.unum.usearch.Types.Verb;

/**
 * A Similarity Search Index over observations on and above the Earth - photos, video frames, drone
 * footage and sensor fixes - each where it was, which way it looked, how far it saw and when, with
 * the modality's own options, relations and per-call settings as nested types.
 *
 * <p>
 * Every batch is an array of {@link Observation}s, whose unrecorded fields stay NaN, never zero.
 * The verbs, receivers, plans and reports are the {@link DenseIndex} ones: the index runs every verb
 * on the caller's thread, {@link #on} blocks on an executor, and {@link #deferred} submits there and
 * answers a {@link Deferred}. Predicates and progress handlers run on the executor's threads and
 * must not throw; an exception escaping one terminates the process.
 *
 * <pre>{@code
 * try (Executor pool = Executor.forRam(new int[0], null, 0);
 *         Arena arena = new Arena(pool);
 *         GeospatialIndex index = GeospatialIndex.initRam(new GeospatialOptions(), new int[0])) {
 *     Observation[] stops = {Observation.at(51.5308, -0.1238), Observation.at(51.5033, -0.1196)};
 *     index.on(pool).insert(new Key[] {Key.of(1), Key.of(2)}, stops);
 *     DenseProducts nearby = index.on(pool).within(stops, GeospatialRelation.NEAR, new Tolerance(500, 0), arena);
 *     System.out.println(nearby.count(0));
 * }
 * }</pre>
 */
public final class GeospatialIndex implements AutoCloseable {

    /**
     * One observation, eleven doubles in natural units. A field no device recorded stays NaN, which
     * only widens the tests that read it. Mirrors {@code usearch_geospatial_observation_t}.
     *
     * @param latitudeDegrees the WGS-84 latitude.
     * @param longitudeDegrees the WGS-84 longitude.
     * @param altitudeMetres the height above the ellipsoid.
     * @param accuracyMetres the horizontal accuracy, one standard deviation.
     * @param headingDegrees the compass bearing the device looked along.
     * @param pitchDegrees the angle above the horizon it looked at.
     * @param headingAccuracyDegrees the uncertainty of the heading.
     * @param fieldOfViewDegrees the horizontal angle its view cone spans.
     * @param rangeMetres how far it saw.
     * @param beginSeconds the first Unix second it covers.
     * @param endSeconds the last Unix second it covers.
     */
    public record Observation(
            double latitudeDegrees,
            double longitudeDegrees,
            double altitudeMetres,
            double accuracyMetres,
            double headingDegrees,
            double pitchDegrees,
            double headingAccuracyDegrees,
            double fieldOfViewDegrees,
            double rangeMetres,
            double beginSeconds,
            double endSeconds) {

        /** A position alone, every other field unrecorded. */
        public static Observation at(double latitudeDegrees, double longitudeDegrees) {
            double nan = Double.NaN;
            return new Observation(latitudeDegrees, longitudeDegrees, nan, nan, nan, nan, nan, nan, nan, nan, nan);
        }

        /** Packs observations into the call's memory as the array C reads. */
        static MemorySegment staged(Call call, Observation[] observations) {
            MemorySegment staged = call.memory.allocate(OBSERVATION, Math.max(observations.length, 1));
            for (int row = 0; row != observations.length; ++row) {
                Observation observation = observations[row];
                double[] fields = {
                    observation.latitudeDegrees, observation.longitudeDegrees, observation.altitudeMetres,
                    observation.accuracyMetres, observation.headingDegrees, observation.pitchDegrees,
                    observation.headingAccuracyDegrees, observation.fieldOfViewDegrees, observation.rangeMetres,
                    observation.beginSeconds, observation.endSeconds
                };
                MemorySegment.copy(fields, 0, staged, ValueLayout.JAVA_DOUBLE, row * OBSERVATION.byteSize(), 11);
            }
            return staged;
        }

        static Observation lift(MemorySegment observations, long row) {
            double[] fields = observations
                    .asSlice(row * OBSERVATION.byteSize(), OBSERVATION.byteSize())
                    .toArray(ValueLayout.JAVA_DOUBLE);
            return new Observation(
                    fields[0],
                    fields[1],
                    fields[2],
                    fields[3],
                    fields[4],
                    fields[5],
                    fields[6],
                    fields[7],
                    fields[8],
                    fields[9],
                    fields[10]);
        }
    }

    /**
     * Which entries count as matches. The C interface spells no names for it, so it renders as the
     * Java constant. Mirrors {@code usearch_geospatial_relation_t}.
     */
    public enum GeospatialRelation {

        /** A relation the library could not name. */
        UNKNOWN,
        /** Within the radius, widened by three standard deviations of both positions' accuracies. */
        NEAR,
        /** Near, with a clear line of sight over the handle's terrain, which a handle without one refuses. */
        VISIBLE,
        /** Near, and the entry's view cone holds the query: which cameras see this place. */
        SEES,
        /** Near, and the query's view cone holds the entry: what this camera sees. */
        SEEN
    }

    /**
     * The radius and time slack of a search. Mirrors {@code usearch_geospatial_tolerance_t}.
     *
     * @param metres the radius, positive for {@link GeospatialIndex#within}; zero ranks a search by
     *     distance alone.
     * @param seconds how far the query's interval widens on both ends.
     */
    public record Tolerance(double metres, double seconds) {}

    /**
     * The elevation every line of sight is tested over: a north-up grid of heights above the WGS-84
     * ellipsoid, viewed for the life of every index it is given to, or a dataset path. No
     * checkpoint holds it, so every load and view names it again. Mirrors
     * {@code usearch_geospatial_terrain_t}, less the fill callback.
     */
    public static final class GeospatialTerrain {

        private final long rows;
        private final long columns;
        private final double northLatitudeDegrees;
        private final double westLongitudeDegrees;
        private final double rowStepDegrees;
        private final double columnStepDegrees;
        private final MemorySegment heights;
        private final long heightsStrideBytes;
        private final @Nullable String path;

        private GeospatialTerrain(
                long rows,
                long columns,
                double northLatitudeDegrees,
                double westLongitudeDegrees,
                double rowStepDegrees,
                double columnStepDegrees,
                MemorySegment heights,
                long heightsStrideBytes,
                @Nullable String path) {
            this.rows = rows;
            this.columns = columns;
            this.northLatitudeDegrees = northLatitudeDegrees;
            this.westLongitudeDegrees = westLongitudeDegrees;
            this.rowStepDegrees = rowStepDegrees;
            this.columnStepDegrees = columnStepDegrees;
            this.heights = heights;
            this.heightsStrideBytes = heightsStrideBytes;
            this.path = path;
        }

        /**
         * A packed row-major grid of heights in metres, copied once into native memory the terrain
         * keeps; cell {@code [0,0]} is centred at the north-west corner given.
         *
         * @throws USearchException SHAPE_MISMATCH unless the heights hold {@code rows * columns} cells.
         */
        public static GeospatialTerrain ofHeights(
                long rows,
                long columns,
                double northLatitudeDegrees,
                double westLongitudeDegrees,
                double rowStepDegrees,
                double columnStepDegrees,
                float[] heights) {
            if (Types.Extents.product(rows, columns, "the grid") != heights.length) {
                throw new USearchException(Status.SHAPE_MISMATCH, "the grid holds rows times columns heights");
            }
            MemorySegment copy = java.lang.foreign.Arena.ofAuto()
                    .allocate((long) heights.length * Float.BYTES, Float.BYTES)
                    .copyFrom(MemorySegment.ofArray(heights));
            return new GeospatialTerrain(
                    rows,
                    columns,
                    northLatitudeDegrees,
                    westLongitudeDegrees,
                    rowStepDegrees,
                    columnStepDegrees,
                    copy,
                    0,
                    null);
        }

        /**
         * A grid of heights read in place from native memory, {@code heightsStrideBytes} apart, zero
         * meaning packed; the segment must stay open while any index given the terrain lives.
         *
         * @throws USearchException SHAPE_MISMATCH when the grid overruns the segment.
         */
        public static GeospatialTerrain ofHeights(
                long rows,
                long columns,
                double northLatitudeDegrees,
                double westLongitudeDegrees,
                double rowStepDegrees,
                double columnStepDegrees,
                MemorySegment heights,
                long heightsStrideBytes) {
            long pitch = heightsStrideBytes == 0
                    ? Types.Extents.product(columns, Float.BYTES, "the row")
                    : heightsStrideBytes;
            if (rows != 0
                    && Types.Extents.product(rows - 1, pitch, "the grid") + columns * Float.BYTES
                            > heights.byteSize()) {
                throw new USearchException(Status.SHAPE_MISMATCH, "the grid overruns the heights");
            }
            if (!heights.isNative()) {
                throw new USearchException(Status.INVALID_ARGUMENT, "viewed heights live in native memory");
            }
            return new GeospatialTerrain(
                    rows,
                    columns,
                    northLatitudeDegrees,
                    westLongitudeDegrees,
                    rowStepDegrees,
                    columnStepDegrees,
                    heights,
                    heightsStrideBytes,
                    null);
        }

        /** A dataset on disk, which only an NVMe residency will read; RAM and VRAM handles refuse it. */
        public static GeospatialTerrain ofPath(String path) {
            return new GeospatialTerrain(0, 0, 0, 0, 0, 0, MemorySegment.NULL, 0, path);
        }

        /** Fills the {@code usearch_geospatial_terrain_t} embedded at {@code at}, or leaves it zeroed for none. */
        static void fill(Call call, long at, @Nullable GeospatialTerrain terrain) {
            if (terrain == null) {
                return;
            }
            MemorySegment struct = call.arguments.asSlice(at, TERRAIN.byteSize());
            TERRAIN_ROWS.set(struct, 0L, terrain.rows);
            TERRAIN_COLUMNS.set(struct, 0L, terrain.columns);
            TERRAIN_NORTH.set(struct, 0L, terrain.northLatitudeDegrees);
            TERRAIN_WEST.set(struct, 0L, terrain.westLongitudeDegrees);
            TERRAIN_ROW_STEP.set(struct, 0L, terrain.rowStepDegrees);
            TERRAIN_COLUMN_STEP.set(struct, 0L, terrain.columnStepDegrees);
            TERRAIN_HEIGHTS.set(struct, 0L, terrain.heights);
            TERRAIN_HEIGHTS_STRIDE.set(struct, 0L, terrain.heightsStrideBytes);
            if (terrain.path != null) {
                call.path(at + TERRAIN_PATH, terrain.path);
            }
            call.pass(terrain);
        }
    }

    /**
     * The terrain an empty index views and the bounds it freezes - the fields
     * {@code usearch_geospatial_init_ram_t} and {@code usearch_geospatial_init_vram_t} share. Every
     * setter returns this instance.
     */
    public static final class GeospatialOptions {

        private @Nullable GeospatialTerrain terrain = null;
        private long connectivity = 0;
        private long valuesPerKey = 0;
        private @Nullable CPUClass cpuClassCeiling = null;
        private @Nullable GPUClass gpuClassCeiling = null;

        /** An index without terrain, with every optional field derived. */
        public GeospatialOptions() {}

        /** The terrain sight lines are tested over; none refuses {@link GeospatialRelation#VISIBLE}. */
        public GeospatialOptions terrain(GeospatialTerrain terrain) {
            this.terrain = terrain;
            return this;
        }

        /** The connections-per-node bound of the graph; zero derives. */
        public GeospatialOptions connectivity(long connectivity) {
            this.connectivity = connectivity;
            return this;
        }

        /** The upper bound on entries sharing one key, like a camera's frames; zero is unbounded. */
        public GeospatialOptions valuesPerKey(long valuesPerKey) {
            this.valuesPerKey = valuesPerKey;
            return this;
        }

        /** The newest CPU class dispatch may pick, intersected with what the machine runs; {@code null} means none. */
        public GeospatialOptions cpuClassCeiling(@Nullable CPUClass cpuClassCeiling) {
            this.cpuClassCeiling = cpuClassCeiling;
            return this;
        }

        /** The newest GPU class dispatch may pick; {@code null} means no ceiling. */
        public GeospatialOptions gpuClassCeiling(@Nullable GPUClass gpuClassCeiling) {
            this.gpuClassCeiling = gpuClassCeiling;
            return this;
        }

        /** Fills the fields both init structures share, at the same offsets in each. */
        void fill(Call call) {
            GeospatialTerrain.fill(call, INIT_TERRAIN, terrain);
            call.size(INIT_CONNECTIVITY, connectivity);
            call.size(INIT_VALUES_PER_KEY, valuesPerKey);
        }
    }

    /**
     * What a checkpoint froze. Mirrors {@code usearch_geospatial_metadata_t}.
     *
     * @param connectivity the connections per node the graph was built with.
     */
    public record GeospatialMetadata(long connectivity) {

        static GeospatialMetadata lift(MemorySegment struct, long at) {
            return new GeospatialMetadata((long) METADATA_CONNECTIVITY.get(struct, at));
        }
    }

    /**
     * A live index's state, each counter read once. Mirrors {@code usearch_geospatial_index_description_t}.
     *
     * @param metadata the frozen connectivity.
     * @param size the stored entries.
     * @param capacity the entries hosted without growing again.
     * @param memoryUsage the bytes the handle holds, payload storage included.
     * @param cpuClass the CPU class dispatch picked, as {@code usearch_cpu_class_name} spells it.
     * @param gpuClass the GPU class dispatch picked; "unknown" for a host index.
     * @param shardsCount the live shards.
     */
    public record GeospatialIndexDescription(
            GeospatialMetadata metadata,
            long size,
            long capacity,
            long memoryUsage,
            String cpuClass,
            String gpuClass,
            long shardsCount) {

        static GeospatialIndexDescription lift(MemorySegment description) {
            return new GeospatialIndexDescription(
                    GeospatialMetadata.lift(description, DESCRIPTION_METADATA),
                    (long) DESCRIPTION_SIZE.get(description, 0L),
                    (long) DESCRIPTION_CAPACITY.get(description, 0L),
                    (long) DESCRIPTION_MEMORY_USAGE.get(description, 0L),
                    Library.text(CPU_CLASS_NAME, (int) DESCRIPTION_CPU_CLASS.get(description, 0L)),
                    Library.text(GPU_CLASS_NAME, (int) DESCRIPTION_GPU_CLASS.get(description, 0L)),
                    (long) DESCRIPTION_SHARDS_COUNT.get(description, 0L));
        }
    }

    /**
     * The knobs of {@link GeospatialIndex#within}, which keeps every match its relation admits.
     * Mirrors {@code usearch_geospatial_within_t}.
     */
    public static final class GeospatialWithinConfig {

        private Traversal traversal = Traversal.WALK;
        private @Nullable Predicate<Key> predicate;
        private @Nullable Progress progress;
        private @Nullable DenseTraversalStats stats;

        /** A walk with every knob derived. */
        public GeospatialWithinConfig() {}

        /** A walk, the default, or an exact scan of every entry. */
        public GeospatialWithinConfig traversal(Traversal traversal) {
            this.traversal = traversal;
            return this;
        }

        /** Which candidate keys the walk may report; it must not throw. */
        public GeospatialWithinConfig predicate(Predicate<Key> predicate) {
            this.predicate = predicate;
            return this;
        }

        /** Hears how far the call has advanced, and may cancel it. */
        public GeospatialWithinConfig progress(Progress progress) {
            this.progress = progress;
            return this;
        }

        /** Where the traversal telemetry lands, overwritten once the call succeeded. */
        public GeospatialWithinConfig stats(DenseTraversalStats stats) {
            this.stats = stats;
            return this;
        }
    }

    /**
     * The knobs of a refinement and the plan it continues; see {@link #refine(GeospatialRefineConfig)}.
     * Mirrors {@code usearch_geospatial_refine_t}.
     */
    public static final class GeospatialRefineConfig {

        private long expansionFactor;
        private long memoryLimit;
        private @Nullable PlanCell plan;
        private @Nullable Progress progress;
        private @Nullable DenseTraversalStats stats;

        /** Every knob derived, one round run to its end. */
        public GeospatialRefineConfig() {}

        /** The breadth of the candidates any walk beside the rounds explores; zero derives. */
        public GeospatialRefineConfig expansionFactor(long expansionFactor) {
            this.expansionFactor = expansionFactor;
            return this;
        }

        /** The bytes a loaded NVMe index holds for the one part it rewrites; zero derives. */
        public GeospatialRefineConfig memoryLimit(long memoryLimit) {
            this.memoryLimit = memoryLimit;
            return this;
        }

        /** The plan the call opens or continues, so a cancel leaves its round resumable. */
        public GeospatialRefineConfig plan(GeospatialRefinePlan plan) {
            this.plan = plan.cell;
            return this;
        }

        /** Hears every barrier of every phase, and may cancel the round there. */
        public GeospatialRefineConfig progress(Progress progress) {
            this.progress = progress;
            return this;
        }

        /** Where the traversal telemetry lands, overwritten once the call succeeded. */
        public GeospatialRefineConfig stats(DenseTraversalStats stats) {
            this.stats = stats;
            return this;
        }

        Rounds rounds() {
            return new Rounds(expansionFactor, memoryLimit, plan, progress, stats);
        }
    }

    /**
     * The plan a reshard continues, the seed of its cut and the knobs of its closing rounds; see
     * {@link #reshard(GeospatialIndex[], GeospatialIndex[], GeospatialReshardConfig)}. Mirrors
     * {@code usearch_geospatial_reshard_t}.
     */
    public static final class GeospatialReshardConfig {

        private long expansionFactor;
        private long memoryLimit;
        private long seed;
        private @Nullable PlanCell plan;
        private @Nullable Progress progress;
        private @Nullable DenseTraversalStats stats;

        /** Every knob derived, under seed zero. */
        public GeospatialReshardConfig() {}

        /** The breadth of the candidates any walk beside the closing rounds explores; zero derives. */
        public GeospatialReshardConfig expansionFactor(long expansionFactor) {
            this.expansionFactor = expansionFactor;
            return this;
        }

        /** The bytes a loaded NVMe target holds for the one part it rewrites; zero derives. */
        public GeospatialReshardConfig memoryLimit(long memoryLimit) {
            this.memoryLimit = memoryLimit;
            return this;
        }

        /** Picks the part roots of a cut: equal seeds over an unchanged source cut equally. */
        public GeospatialReshardConfig seed(long seed) {
            this.seed = seed;
            return this;
        }

        /** The plan the call opens or continues, so a cancel leaves its closing rounds resumable. */
        public GeospatialReshardConfig plan(GeospatialReshardPlan plan) {
            this.plan = plan.cell;
            return this;
        }

        /** Hears every barrier of the closing rounds, and may cancel them there; a move is never cancelled. */
        public GeospatialReshardConfig progress(Progress progress) {
            this.progress = progress;
            return this;
        }

        /** Where the traversal telemetry lands, overwritten once the call succeeded. */
        public GeospatialReshardConfig stats(DenseTraversalStats stats) {
            this.stats = stats;
            return this;
        }

        Rounds rounds() {
            return new Rounds(expansionFactor, memoryLimit, plan, progress, stats);
        }
    }

    /**
     * A refinement in flight, which {@link #refine(GeospatialRefineConfig)} opens on its first call and
     * continues on every later one. The index admits no insertion and no save while it is open, and
     * every call naming it names the index and executor it was opened with;
     * {@link #refinePlanFinish} closes it.
     */
    public static final class GeospatialRefinePlan {

        final PlanCell cell = new PlanCell();

        /** A plan no call has opened yet. */
        public GeospatialRefinePlan() {}

        /** Where the last call naming the plan stopped. */
        public PhaseProgress phase() {
            return cell.phase();
        }
    }

    /**
     * A reshard in flight, which {@link #reshard(GeospatialIndex[], GeospatialIndex[], GeospatialReshardConfig)}
     * opens on its first call and continues on every later one. Every index it names admits no
     * insertion and no save while it is open; {@link #reshardPlanFinish} closes it.
     */
    public static final class GeospatialReshardPlan {

        final PlanCell cell = new PlanCell();

        /** A plan no call has opened yet. */
        public GeospatialReshardPlan() {}

        /** Where the last call naming the plan stopped. */
        public PhaseProgress phase() {
            return cell.phase();
        }
    }

    /**
     * A sequential walk over the index contents; the observations themselves come from
     * {@link GeospatialIndex#get}. It keeps no executor: its pulls run on the calling thread, or block on
     * the executor {@link #on(Executor)} names for each of them. An overlapping pull reports
     * CONTENDED.
     */
    public static final class GeospatialCursor implements AutoCloseable {

        private final Handle handle;
        private final GeospatialIndex index;

        GeospatialCursor(MemorySegment opened, GeospatialIndex index) {
            this.handle = new Handle(opened, FAMILY.cursorFree());
            this.index = index;
        }

        /** The receiver whose pulls block on {@code executor}, named again for every pull. Binding-only. */
        public On on(Executor executor) {
            return new On(this, executor);
        }

        /** Copies the keys of up to {@code wanted} next entries into {@code arena}. Calls {@code usearch_geospatial_cursor_next}. */
        public DenseBatch next(long wanted, Arena arena) {
            return pull(null, wanted, false, arena);
        }

        /** Copies up to {@code wanted} next entries, each with the keys it links to. Calls {@code usearch_geospatial_cursor_next}. */
        public DenseBatch nextLinks(long wanted, Arena arena) {
            return pull(null, wanted, true, arena);
        }

        private DenseBatch pull(@Nullable Executor executor, long wanted, boolean linked, Arena arena) {
            return DenseIndex.pulling(FAMILY.cursorNext(), index.handle, executor, handle, wanted, linked, arena);
        }

        /** A cursor's pulls blocking on one executor, contended while another call holds it. */
        public static final class On {

            private final GeospatialCursor cursor;
            private final Executor executor;

            On(GeospatialCursor cursor, Executor executor) {
                this.cursor = cursor;
                this.executor = executor;
            }

            /** Runs {@link GeospatialCursor#next(long, Arena)} on this receiver's executor. */
            public DenseBatch next(long wanted, Arena arena) {
                return cursor.pull(executor, wanted, false, arena);
            }

            /** Runs {@link GeospatialCursor#nextLinks(long, Arena)} on this receiver's executor. */
            public DenseBatch nextLinks(long wanted, Arena arena) {
                return cursor.pull(executor, wanted, true, arena);
            }
        }

        /** Releases the cursor; idempotent. Calls {@code usearch_geospatial_cursor_free}. */
        @Override
        public void close() {
            handle.close();
        }
    }

    /** The receiver whose verbs block on an executor, contended while another call holds it. */
    public static final class On {

        private final GeospatialIndex index;
        private final Executor executor;

        On(GeospatialIndex index, Executor executor) {
            this.index = index;
            this.executor = executor;
        }

        private <Result> Result block(Request<Result> request) {
            return Call.block(request.verb(), index.handle, executor, request.fill());
        }

        /** Runs {@link GeospatialIndex#reserve} on this receiver's executor. */
        public void reserve(long entries) {
            block(reserving(entries));
        }

        /** Runs {@link GeospatialIndex#clear} on this receiver's executor. */
        public void clear() {
            block(DenseCalls.clearing(FAMILY.clear()));
        }

        /** Runs {@link GeospatialIndex#describe} on this receiver's executor. */
        public GeospatialIndexDescription describe() {
            return block(describing());
        }

        /** Runs {@link GeospatialIndex#count} on this receiver's executor. */
        public long count(Key key) {
            return block(counting(key));
        }

        /** Runs {@link GeospatialIndex#contains} on this receiver's executor. */
        public boolean contains(Key key) {
            return count(key) != 0;
        }

        /** Runs {@link GeospatialIndex#get} on this receiver's executor. */
        public Observation[] get(Key key, Arena arena) {
            return block(getting(key, arena));
        }

        /** Runs {@link GeospatialIndex#insert(Key[], Observation[])} on this receiver's executor. */
        public void insert(Key[] keys, Observation[] observations) {
            block(inserting(keys, observations, new DenseInsertConfig()));
        }

        /** Runs {@link GeospatialIndex#insert(Key[], Observation[], DenseInsertConfig)} on this receiver's executor. */
        public void insert(Key[] keys, Observation[] observations, DenseInsertConfig config) {
            block(inserting(keys, observations, config));
        }

        /** Runs {@link GeospatialIndex#refine()} on this receiver's executor. */
        public RefineReport refine() {
            return block(BasicCalls.refining(FAMILY, new GeospatialRefineConfig().rounds()));
        }

        /** Runs {@link GeospatialIndex#refine(GeospatialRefineConfig)} on this receiver's executor. */
        public RefineReport refine(GeospatialRefineConfig config) {
            return block(BasicCalls.refining(FAMILY, config.rounds()));
        }

        /** Runs {@link GeospatialIndex#refinePlanFinish} on this receiver's executor. */
        public RefineReport refinePlanFinish(GeospatialRefinePlan plan) {
            return block(BasicCalls.refinePlanFinishing(FAMILY, plan.cell));
        }

        /** Runs {@link GeospatialIndex#health} on this receiver's executor. */
        public DenseReachability health() {
            return block(DenseCalls.auditing(FAMILY.health()));
        }

        /** Runs {@link GeospatialIndex#search(Observation[], GeospatialRelation, Tolerance, long, Arena)} on this receiver's executor. */
        public Matches search(
                Observation[] queries, GeospatialRelation relation, Tolerance tolerance, long wanted, Arena arena) {
            return block(searching(queries, relation, tolerance, new DenseSearchConfig(wanted), arena));
        }

        /** Runs {@link GeospatialIndex#search(Observation[], GeospatialRelation, Tolerance, DenseSearchConfig, Arena)} on this receiver's executor. */
        public Matches search(
                Observation[] queries,
                GeospatialRelation relation,
                Tolerance tolerance,
                DenseSearchConfig config,
                Arena arena) {
            return block(searching(queries, relation, tolerance, config, arena));
        }

        /** Runs {@link GeospatialIndex#within(Observation[], GeospatialRelation, Tolerance, Arena)} on this receiver's executor. */
        public DenseProducts within(
                Observation[] queries, GeospatialRelation relation, Tolerance tolerance, Arena arena) {
            return block(withinRequest(queries, relation, tolerance, new GeospatialWithinConfig(), arena));
        }

        /** Runs {@link GeospatialIndex#within(Observation[], GeospatialRelation, Tolerance, GeospatialWithinConfig, Arena)} on this receiver's executor. */
        public DenseProducts within(
                Observation[] queries,
                GeospatialRelation relation,
                Tolerance tolerance,
                GeospatialWithinConfig config,
                Arena arena) {
            return block(withinRequest(queries, relation, tolerance, config, arena));
        }

        /** Runs {@link GeospatialIndex#reshard} on this receiver's executor. */
        public RefineReport reshard(
                GeospatialIndex[] sources, GeospatialIndex[] targets, GeospatialReshardConfig config) {
            return block(resharding(sources, targets, config));
        }

        /** Runs {@link GeospatialIndex#reshardPlanFinish} on this receiver's executor. */
        public RefineReport reshardPlanFinish(GeospatialReshardPlan plan) {
            return block(BasicCalls.reshardPlanFinishing(FAMILY, plan.cell));
        }

        /** Runs {@link GeospatialIndex#sample} on this receiver's executor. */
        public Sample sample(long wanted, Arena arena, DenseSampleConfig config) {
            return block(DenseCalls.sampling(FAMILY.sample(), wanted, arena, config));
        }

        /** Runs {@link GeospatialIndex#save} on this receiver's executor. */
        public void save(String path) {
            block(DenseCalls.saving(FAMILY.save(), path, null));
        }

        /** Runs {@link GeospatialIndex#saveBuffer} on this receiver's executor. */
        public Monolith saveBuffer(Arena arena) {
            return block(DenseCalls.serializing(FAMILY.saveBuffer(), arena, null));
        }
    }

    /**
     * The receiver whose verbs submit their calls to an executor, each answering a {@link Deferred}
     * that owns what the call reads until final. Its verbs are those of {@link GeospatialIndex}, less
     * the cursor that pulls one batch at a time.
     */
    public static final class DeferredOn {

        private final GeospatialIndex index;
        private final Executor executor;

        DeferredOn(GeospatialIndex index, Executor executor) {
            this.index = index;
            this.executor = executor;
        }

        private <Result> Deferred<Result> submit(Request<Result> request) {
            return Call.submit(request.verb(), index.handle, executor, request.fill());
        }

        /** Submits {@link GeospatialIndex#reserve}. Calls {@code usearch_geospatial_reserve}. */
        public Deferred<Void> reserve(long entries) {
            return submit(reserving(entries));
        }

        /** Submits {@link GeospatialIndex#clear}. Calls {@code usearch_geospatial_clear}. */
        public Deferred<Void> clear() {
            return submit(DenseCalls.clearing(FAMILY.clear()));
        }

        /** Submits {@link GeospatialIndex#describe}. Calls {@code usearch_geospatial_describe}. */
        public Deferred<GeospatialIndexDescription> describe() {
            return submit(describing());
        }

        /** Submits {@link GeospatialIndex#count}. Calls {@code usearch_geospatial_count}. */
        public Deferred<Long> count(Key key) {
            return submit(counting(key));
        }

        /** Submits {@link GeospatialIndex#get}. Calls {@code usearch_geospatial_get}. */
        public Deferred<Observation[]> get(Key key, Arena arena) {
            return submit(getting(key, arena));
        }

        /** Submits {@link GeospatialIndex#insert(Key[], Observation[], DenseInsertConfig)}. Calls {@code usearch_geospatial_insert}. */
        public Deferred<Void> insert(Key[] keys, Observation[] observations, DenseInsertConfig config) {
            return submit(inserting(keys, observations, config));
        }

        /** Submits {@link GeospatialIndex#refine(GeospatialRefineConfig)}. Calls {@code usearch_geospatial_refine}. */
        public Deferred<RefineReport> refine(GeospatialRefineConfig config) {
            return submit(BasicCalls.refining(FAMILY, config.rounds()));
        }

        /** Submits {@link GeospatialIndex#refinePlanFinish}. Calls {@code usearch_geospatial_refine_plan_finish}. */
        public Deferred<RefineReport> refinePlanFinish(GeospatialRefinePlan plan) {
            return submit(BasicCalls.refinePlanFinishing(FAMILY, plan.cell));
        }

        /** Submits {@link GeospatialIndex#health}. Calls {@code usearch_geospatial_health}. */
        public Deferred<DenseReachability> health() {
            return submit(DenseCalls.auditing(FAMILY.health()));
        }

        /** Submits {@link GeospatialIndex#search(Observation[], GeospatialRelation, Tolerance, DenseSearchConfig, Arena)}. Calls {@code usearch_geospatial_search}. */
        public Deferred<Matches> search(
                Observation[] queries,
                GeospatialRelation relation,
                Tolerance tolerance,
                DenseSearchConfig config,
                Arena arena) {
            return submit(searching(queries, relation, tolerance, config, arena));
        }

        /** Submits {@link GeospatialIndex#within(Observation[], GeospatialRelation, Tolerance, GeospatialWithinConfig, Arena)}. Calls {@code usearch_geospatial_within}. */
        public Deferred<DenseProducts> within(
                Observation[] queries,
                GeospatialRelation relation,
                Tolerance tolerance,
                GeospatialWithinConfig config,
                Arena arena) {
            return submit(withinRequest(queries, relation, tolerance, config, arena));
        }

        /** Submits {@link GeospatialIndex#reshard}. Calls {@code usearch_geospatial_reshard}. */
        public Deferred<RefineReport> reshard(
                GeospatialIndex[] sources, GeospatialIndex[] targets, GeospatialReshardConfig config) {
            return submit(resharding(sources, targets, config));
        }

        /** Submits {@link GeospatialIndex#reshardPlanFinish}. Calls {@code usearch_geospatial_reshard_plan_finish}. */
        public Deferred<RefineReport> reshardPlanFinish(GeospatialReshardPlan plan) {
            return submit(BasicCalls.reshardPlanFinishing(FAMILY, plan.cell));
        }

        /** Submits {@link GeospatialIndex#sample}. Calls {@code usearch_geospatial_sample}. */
        public Deferred<Sample> sample(long wanted, Arena arena, DenseSampleConfig config) {
            return submit(DenseCalls.sampling(FAMILY.sample(), wanted, arena, config));
        }

        /** Submits {@link GeospatialIndex#save}. Calls {@code usearch_geospatial_save}. */
        public Deferred<Void> save(String path) {
            return submit(DenseCalls.saving(FAMILY.save(), path, null));
        }

        /** Submits {@link GeospatialIndex#saveBuffer}. Calls {@code usearch_geospatial_save_buffer}. */
        public Deferred<Monolith> saveBuffer(Arena arena) {
            return submit(DenseCalls.serializing(FAMILY.saveBuffer(), arena, null));
        }
    }

    final Handle handle;

    /** Adopts a created handle, keeping the terrain it views reachable until the handle is freed. */
    private GeospatialIndex(MemorySegment created, @Nullable GeospatialTerrain terrain) {
        handle = new Handle(created, FAMILY.free(), terrain);
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
     * Claims room for entries ahead of insertion, a hint, since insertions grow the
     * index on demand. Calls {@code usearch_geospatial_reserve}.
     *
     * @param entries the desired total number of entries, including the current size.
     * @throws USearchException READ_ONLY on a view.
     */
    public void reserve(long entries) {
        block(reserving(entries));
    }

    /** Erases all entries, keeping the frozen kind. Calls {@code usearch_geospatial_clear}. */
    public void clear() {
        block(DenseCalls.clearing(FAMILY.clear()));
    }

    /** Reports the frozen metadata and the current counters in one call. Calls {@code usearch_geospatial_describe}. */
    public GeospatialIndexDescription describe() {
        return block(describing());
    }

    /**
     * How many observations a key holds; zero means the index holds none. Calls
     * {@code usearch_geospatial_count}.
     */
    public long count(Key key) {
        return block(counting(key));
    }

    /** Whether the index holds at least one entry under a key. Binding-only sugar over {@link #count}. */
    public boolean contains(Key key) {
        return count(key) != 0;
    }

    /**
     * Retrieves every observation stored under a key, in insertion order and within the stored
     * rounding, through {@code arena}. Calls {@code usearch_geospatial_get}.
     *
     * @return the observations copied out, none when the key holds nothing.
     * @throws USearchException CONTENDED when the arena serves another call.
     */
    public Observation[] get(Key key, Arena arena) {
        return block(getting(key, arena));
    }

    /**
     * Inserts observations under their keys, with every knob derived.
     *
     * @throws USearchException the statuses {@link #insert(Key[], Observation[], DenseInsertConfig)} names.
     */
    public void insert(Key[] keys, Observation[] observations) {
        block(inserting(keys, observations, new DenseInsertConfig()));
    }

    /**
     * Inserts observations under their keys - all-or-nothing per call. Calls
     * {@code usearch_geospatial_insert}.
     *
     * @throws USearchException SHAPE_MISMATCH when the key count and the observations disagree,
     *     READ_ONLY on a view, and OVERFLOW when a key already holds {@code valuesPerKey} entries.
     */
    public void insert(Key[] keys, Observation[] observations, DenseInsertConfig config) {
        block(inserting(keys, observations, config));
    }

    /**
     * Runs one round over every stored entry, with every knob derived.
     *
     * @throws USearchException the statuses {@link #refine(GeospatialRefineConfig)} names.
     */
    public RefineReport refine() {
        return block(BasicCalls.refining(FAMILY, new GeospatialRefineConfig().rounds()));
    }

    /**
     * Re-selects the neighbors of every stored entry against the finished graph, as
     * {@link DenseIndex#refine(DenseIndex.DenseRefineConfig)} does. Calls {@code usearch_geospatial_refine}.
     *
     * @throws USearchException READ_ONLY on a view, and INVALID_ARGUMENT on a plan without an executor.
     */
    public RefineReport refine(GeospatialRefineConfig config) {
        return block(BasicCalls.refining(FAMILY, config.rounds()));
    }

    /**
     * Closes the round a cancel left in hand and frees the plan whatever the call answers. Calls
     * {@code usearch_geospatial_refine_plan_finish}.
     *
     * @throws USearchException INVALID_ARGUMENT on a plan no call opened, or one already finished.
     */
    public RefineReport refinePlanFinish(GeospatialRefinePlan plan) {
        return block(BasicCalls.refinePlanFinishing(FAMILY, plan.cell));
    }

    /** Audits how much of the graph a search reaches. Calls {@code usearch_geospatial_health}. */
    public DenseReachability health() {
        return block(DenseCalls.auditing(FAMILY.health()));
    }

    /**
     * Ranks up to {@code wanted} nearest entries per query that {@code relation} admits, with every
     * other knob derived.
     *
     * @throws USearchException the statuses
     *     {@link #search(Observation[], GeospatialRelation, Tolerance, DenseSearchConfig, Arena)} names.
     */
    public Matches search(
            Observation[] queries, GeospatialRelation relation, Tolerance tolerance, long wanted, Arena arena) {
        return block(searching(queries, relation, tolerance, new DenseSearchConfig(wanted), arena));
    }

    /**
     * Ranks the nearest entries of each query that {@code relation} admits inside {@code tolerance},
     * a zero radius ranking by distance alone; every score is the chord between the two positions in
     * metres. Calls {@code usearch_geospatial_search}.
     *
     * @throws USearchException UNSUPPORTED on {@link GeospatialRelation#VISIBLE} over a handle without
     *     terrain, and CONTENDED when the arena serves another call.
     */
    public Matches search(
            Observation[] queries,
            GeospatialRelation relation,
            Tolerance tolerance,
            DenseSearchConfig config,
            Arena arena) {
        return block(searching(queries, relation, tolerance, config, arena));
    }

    /**
     * Reports every entry {@code relation} admits inside {@code tolerance}, with every knob derived.
     *
     * @throws USearchException the statuses
     *     {@link #within(Observation[], GeospatialRelation, Tolerance, GeospatialWithinConfig, Arena)} names.
     */
    public DenseProducts within(Observation[] queries, GeospatialRelation relation, Tolerance tolerance, Arena arena) {
        return block(withinRequest(queries, relation, tolerance, new GeospatialWithinConfig(), arena));
    }

    /**
     * Reports every entry {@code relation} admits inside {@code tolerance}, however many there turn
     * out to be, as one run per query in {@code arena}, scored by the chord in metres. Calls
     * {@code usearch_geospatial_within}.
     *
     * @throws USearchException INVALID_ARGUMENT on a radius that is not positive, and CONTENDED when
     *     the arena serves another call.
     */
    public DenseProducts within(
            Observation[] queries,
            GeospatialRelation relation,
            Tolerance tolerance,
            GeospatialWithinConfig config,
            Arena arena) {
        return block(withinRequest(queries, relation, tolerance, config, arena));
    }

    /**
     * Moves the entries of {@code sources} into {@code targets}, then runs a round on every target that
     * took entries, as {@link DenseIndex#reshard} does. Calls {@code usearch_geospatial_reshard}.
     *
     * @throws USearchException INVALID_ARGUMENT on no sources or targets or an index named twice,
     *     CONFIG_MISMATCH on another kind or a target neither a source nor empty, and UNSUPPORTED on
     *     one index as its own only source and target.
     */
    public RefineReport reshard(GeospatialIndex[] sources, GeospatialIndex[] targets, GeospatialReshardConfig config) {
        return block(resharding(sources, targets, config));
    }

    /**
     * Closes the rounds a cancel stopped and frees the plan whatever the call answers. Calls
     * {@code usearch_geospatial_reshard_plan_finish}.
     *
     * @throws USearchException INVALID_ARGUMENT on a plan no call opened.
     */
    public RefineReport reshardPlanFinish(GeospatialReshardPlan plan) {
        return block(BasicCalls.reshardPlanFinishing(FAMILY, plan.cell));
    }

    /**
     * Draws keys around the focus, or across the whole index, placed in three dimensions, as
     * {@link DenseIndex#sample} does. Calls {@code usearch_geospatial_sample}.
     */
    public Sample sample(long wanted, Arena arena, DenseSampleConfig config) {
        return block(DenseCalls.sampling(FAMILY.sample(), wanted, arena, config));
    }

    /**
     * Writes the index under a path: one monolith for a lone shard, or a manifest beside one part per
     * shard. Calls {@code usearch_geospatial_save}.
     *
     * @throws USearchException IO when the destination cannot be written.
     */
    public void save(String path) {
        block(DenseCalls.saving(FAMILY.save(), path, null));
    }

    /** Writes the index into {@code arena}, the bytes {@link #save} writes. Calls {@code usearch_geospatial_save_buffer}. */
    public Monolith saveBuffer(Arena arena) {
        return block(DenseCalls.serializing(FAMILY.saveBuffer(), arena, null));
    }

    /**
     * Opens a snapshot cursor over the contents, every entry as stored or breadth-first from
     * {@code starts}. Calls {@code usearch_geospatial_cursor_init}.
     */
    public GeospatialCursor cursor(Key... starts) {
        return block(BasicCalls.cursorOpening(FAMILY, starts, opened -> new GeospatialCursor(opened, this)));
    }

    /**
     * Creates an empty, mutable index in host memory. Calls {@code usearch_geospatial_init_ram}.
     *
     * @param memoryDomains NUMA node ids to place shards on; an empty list derives every node.
     * @throws USearchException UNSUPPORTED on a terrain path, which no residency reads yet.
     */
    public static GeospatialIndex initRam(GeospatialOptions options, int[] memoryDomains) {
        return create(INIT_RAM, null, options.terrain, call -> {
            options.fill(call);
            call.ram(call.arguments, INIT_RAM.at("ram"), memoryDomains, options.cpuClassCeiling);
        });
    }

    /**
     * Creates an empty, mutable index in device memory. Calls {@code usearch_geospatial_init_vram}.
     *
     * @throws USearchException DEVICE when no device answers or the build has no GPU kernels.
     */
    public static GeospatialIndex initVram(GeospatialOptions options, int[] devices) {
        return create(INIT_VRAM, null, options.terrain, call -> {
            options.fill(call);
            call.vram(call.arguments, INIT_VRAM.at("vram"), devices, options.gpuClassCeiling);
        });
    }

    /**
     * Copies a checkpoint into a new, mutable index in host memory. Calls {@code usearch_geospatial_load_ram}.
     *
     * @throws USearchException MALFORMED on a checkpoint that ends early, and IO when it cannot be read.
     */
    public static GeospatialIndex loadRam(
            String path,
            @Nullable Executor executor,
            int[] memoryDomains,
            long valuesPerKey,
            @Nullable GeospatialTerrain terrain,
            @Nullable Progress progress) {
        return create(
                LOAD_RAM,
                executor,
                terrain,
                call -> loading(call, true, memoryDomains, valuesPerKey, terrain, progress)
                        .path(LOAD_SOURCE, path));
    }

    /** Copies a monolith from a buffer into a new index in host memory. Calls {@code usearch_geospatial_load_ram_buffer}. */
    public static GeospatialIndex loadRamBuffer(
            MemorySegment buffer,
            @Nullable Executor executor,
            int[] memoryDomains,
            long valuesPerKey,
            @Nullable GeospatialTerrain terrain,
            @Nullable Progress progress) {
        return create(
                LOAD_RAM_BUFFER,
                executor,
                terrain,
                call -> loading(call, true, memoryDomains, valuesPerKey, terrain, progress)
                        .bytes(LOAD_SOURCE, buffer));
    }

    /** Copies a checkpoint into a new index in device memory. Calls {@code usearch_geospatial_load_vram}. */
    public static GeospatialIndex loadVram(
            String path,
            @Nullable Executor executor,
            int[] devices,
            long valuesPerKey,
            @Nullable GeospatialTerrain terrain,
            @Nullable Progress progress) {
        return create(
                LOAD_VRAM,
                executor,
                terrain,
                call -> loading(call, false, devices, valuesPerKey, terrain, progress)
                        .path(LOAD_SOURCE, path));
    }

    /** Copies a monolith from a buffer into a new index in device memory. Calls {@code usearch_geospatial_load_vram_buffer}. */
    public static GeospatialIndex loadVramBuffer(
            MemorySegment buffer,
            @Nullable Executor executor,
            int[] devices,
            long valuesPerKey,
            @Nullable GeospatialTerrain terrain,
            @Nullable Progress progress) {
        return create(
                LOAD_VRAM_BUFFER,
                executor,
                terrain,
                call -> loading(call, false, devices, valuesPerKey, terrain, progress)
                        .bytes(LOAD_SOURCE, buffer));
    }

    /**
     * Serves a checkpoint in place through a memory mapping, as a read-only index. Calls
     * {@code usearch_geospatial_view_ram}.
     *
     * @throws USearchException UNSUPPORTED where the engine maps no geospatial checkpoint.
     */
    public static GeospatialIndex viewRam(String path, @Nullable GeospatialTerrain terrain) {
        return create(VIEW_RAM, null, terrain, call -> {
            call.path(VIEW_RAM.at("path"), path);
            GeospatialTerrain.fill(call, VIEW_RAM.at("terrain"), terrain);
        });
    }

    /**
     * Serves a checkpoint in place from NVMe drives as a read-only index. Calls
     * {@code usearch_geospatial_view_nvme}.
     *
     * @param paths devices or directories to serve the parts from; an empty list serves each where its
     *     manifest points.
     */
    public static GeospatialIndex viewNvme(
            String path,
            @Nullable Executor executor,
            String[] paths,
            @Nullable CPUClass cpuClassCeiling,
            @Nullable GeospatialTerrain terrain) {
        return create(VIEW_NVME, executor, terrain, call -> {
            call.path(VIEW_NVME.at("path"), path);
            call.nvme(call.arguments, VIEW_NVME.at("nvme"), paths, cpuClassCeiling);
            GeospatialTerrain.fill(call, VIEW_NVME.at("terrain"), terrain);
        });
    }

    /** Reads a checkpoint's kind and connectivity without loading it. Calls {@code usearch_geospatial_metadata_read}. */
    public static GeospatialMetadata metadataRead(String path, @Nullable Executor executor) {
        return BasicCalls.metadata(
                FAMILY.metadataRead(),
                executor,
                call -> call.path(BasicCalls.READ_SOURCE, path),
                METADATA,
                metadata -> GeospatialMetadata.lift(metadata, 0));
    }

    /** Reads a monolith's kind and connectivity without loading it. Calls {@code usearch_geospatial_metadata_read_buffer}. */
    public static GeospatialMetadata metadataReadBuffer(MemorySegment buffer, @Nullable Executor executor) {
        return BasicCalls.metadata(
                FAMILY.metadataReadBuffer(),
                executor,
                call -> call.bytes(BasicCalls.READ_SOURCE, buffer),
                METADATA,
                metadata -> GeospatialMetadata.lift(metadata, 0));
    }

    /**
     * Releases the index, after every deferred call naming it is final. It is idempotent. Calls
     * {@code usearch_geospatial_free}.
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
        return "GeospatialIndex(" + handle + ")";
    }

    private static Request<Void> reserving(long entries) {
        return new Request<>(RESERVE, call -> {
            call.size(RESERVE.field("entries"), entries);
            return () -> null;
        });
    }

    private static Request<GeospatialIndexDescription> describing() {
        return BasicCalls.describing(FAMILY, DESCRIPTION, GeospatialIndexDescription::lift);
    }

    private static Request<Long> counting(Key key) {
        return BasicCalls.counting(FAMILY, key, Library.SIZE, stats -> stats.get(Library.SIZE, 0));
    }

    private static Request<Observation[]> getting(Key key, Arena arena) {
        return new Request<>(GET, call -> {
            call.key(GET.at("key"), key);
            MemorySegment observations = call.output(GET_OBSERVATIONS, Library.POINTER);
            MemorySegment written = call.output(GET_WRITTEN, Library.SIZE);
            call.arena(GET_ARENA, arena);
            return () -> {
                long count = written.get(Library.SIZE, 0);
                MemorySegment rows = call.reach(
                        observations.get(Library.POINTER, 0),
                        Types.Extents.product(count, OBSERVATION.byteSize(), "the observations"));
                Observation[] lifted = new Observation[Types.Extents.toArrayLength(count, "the observations")];
                for (int row = 0; row != lifted.length; ++row) {
                    lifted[row] = Observation.lift(rows, row);
                }
                return lifted;
            };
        });
    }

    private static Request<Void> inserting(Key[] keys, Observation[] observations, DenseInsertConfig config) {
        // The call has one count, so only this side sees the keys and the tape disagree.
        if (keys.length != observations.length) {
            throw new USearchException(Status.SHAPE_MISMATCH, "the key count does not match the observations");
        }
        return new Request<>(INSERT, call -> {
            call.size(INSERT_COUNT, keys.length);
            call.keys(INSERT.at("keys"), keys);
            call.pointer(INSERT_OBSERVATIONS, Observation.staged(call, observations));
            call.size(INSERT_EXPANSION_FACTOR, config.expansionFactor);
            call.progress(INSERT.at("progress"), config.progress);
            call.stats(INSERT_STATS, config.stats);
            return () -> null;
        });
    }

    private static Request<Matches> searching(
            Observation[] queries,
            GeospatialRelation relation,
            Tolerance tolerance,
            DenseSearchConfig config,
            Arena arena) {
        return new Request<>(SEARCH, call -> {
            call.size(SEARCH_COUNT, queries.length);
            call.pointer(SEARCH_QUERIES, Observation.staged(call, queries));
            call.enumeration(SEARCH_RELATION, relation.ordinal());
            SEARCH_METRES.set(call.arguments, 0L, tolerance.metres());
            SEARCH_SECONDS.set(call.arguments, 0L, tolerance.seconds());
            MemorySegment matches = call.output(SEARCH_MATCHES, Library.MATCHES_VIEW);
            call.arena(SEARCH_ARENA, arena);
            call.size(SEARCH_WANTED, config.wanted);
            call.size(SEARCH_EXPANSION_FACTOR, config.expansionFactor);
            call.enumeration(SEARCH_TRAVERSAL, config.traversal.code());
            call.predicate(SEARCH.at("predicate"), config.predicate);
            call.progress(SEARCH.at("progress"), config.progress);
            call.stats(SEARCH_STATS, config.stats);
            return () -> Matches.lift(call, matches, queries.length);
        });
    }

    private static Request<DenseProducts> withinRequest(
            Observation[] queries,
            GeospatialRelation relation,
            Tolerance tolerance,
            GeospatialWithinConfig config,
            Arena arena) {
        return new Request<>(WITHIN, call -> {
            call.size(WITHIN_COUNT, queries.length);
            call.pointer(WITHIN_QUERIES, Observation.staged(call, queries));
            call.enumeration(WITHIN_RELATION, relation.ordinal());
            WITHIN_METRES.set(call.arguments, 0L, tolerance.metres());
            WITHIN_SECONDS.set(call.arguments, 0L, tolerance.seconds());
            MemorySegment matches = call.output(WITHIN_MATCHES, Library.MATCHES_TAPE_VIEW);
            MemorySegment written = call.output(WITHIN_WRITTEN, Library.SIZE);
            call.arena(WITHIN_ARENA, arena);
            call.enumeration(WITHIN_TRAVERSAL, config.traversal.code());
            call.predicate(WITHIN.at("predicate"), config.predicate);
            call.progress(WITHIN.at("progress"), config.progress);
            call.stats(WITHIN_STATS, config.stats);
            return () -> {
                long pairs = written.get(Library.SIZE, 0);
                return DenseProducts.lift(call, matches, queries.length, pairs, pairs);
            };
        });
    }

    private static Request<RefineReport> resharding(
            GeospatialIndex[] sources, GeospatialIndex[] targets, GeospatialReshardConfig config) {
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

    /** Fills the placement, bound, terrain and handler every load shares, answering the call for its source. */
    private static Call loading(
            Call call,
            boolean ram,
            int[] placement,
            long valuesPerKey,
            @Nullable GeospatialTerrain terrain,
            @Nullable Progress progress) {
        if (ram) {
            call.ram(call.arguments, LOAD_RAM.at("ram"), placement, null);
        } else {
            call.vram(call.arguments, LOAD_VRAM.at("vram"), placement, null);
        }
        call.size(ram ? LOAD_RAM_VALUES_PER_KEY : LOAD_VRAM_VALUES_PER_KEY, valuesPerKey);
        GeospatialTerrain.fill(call, ram ? LOAD_RAM.at("terrain") : LOAD_VRAM.at("terrain"), terrain);
        call.progress(ram ? LOAD_RAM.at("progress") : LOAD_VRAM.at("progress"), progress);
        return call;
    }

    /** Runs one lifecycle verb and adopts the handle it created. */
    private static GeospatialIndex create(
            Verb verb, @Nullable Executor executor, @Nullable GeospatialTerrain terrain, Consumer<Call> fill) {
        return Call.block(verb, null, executor, call -> {
            MemorySegment created = call.output(java.util.Objects.requireNonNull(verb.index()), Library.POINTER);
            fill.accept(call);
            return () -> new GeospatialIndex(created.get(Library.POINTER, 0), terrain);
        });
    }

    static final Family FAMILY = Family.of("geospatial");

    /** {@code usearch_geospatial_observation_t}, eleven doubles. */
    private static final MemoryLayout OBSERVATION = MemoryLayout.sequenceLayout(11, ValueLayout.JAVA_DOUBLE);

    /** {@code usearch_geospatial_terrain_t}. */
    private static final StructLayout TERRAIN = MemoryLayout.structLayout(
            Library.SIZE.withName("rows"),
            Library.SIZE.withName("columns"),
            ValueLayout.JAVA_DOUBLE.withName("north_latitude_degrees"),
            ValueLayout.JAVA_DOUBLE.withName("west_longitude_degrees"),
            ValueLayout.JAVA_DOUBLE.withName("row_step_degrees"),
            ValueLayout.JAVA_DOUBLE.withName("column_step_degrees"),
            Library.POINTER.withName("heights"),
            Library.SIZE.withName("heights_stride"),
            Library.POINTER.withName("callback"),
            Library.POINTER.withName("state"),
            Library.BYTES_VIEW.withName("path"));

    private static final VarHandle TERRAIN_ROWS = Library.field(TERRAIN, "rows");
    private static final VarHandle TERRAIN_COLUMNS = Library.field(TERRAIN, "columns");
    private static final VarHandle TERRAIN_NORTH = Library.field(TERRAIN, "north_latitude_degrees");
    private static final VarHandle TERRAIN_WEST = Library.field(TERRAIN, "west_longitude_degrees");
    private static final VarHandle TERRAIN_ROW_STEP = Library.field(TERRAIN, "row_step_degrees");
    private static final VarHandle TERRAIN_COLUMN_STEP = Library.field(TERRAIN, "column_step_degrees");
    private static final VarHandle TERRAIN_HEIGHTS = Library.field(TERRAIN, "heights");
    private static final VarHandle TERRAIN_HEIGHTS_STRIDE = Library.field(TERRAIN, "heights_stride");
    private static final long TERRAIN_PATH = Library.offset(TERRAIN, "path");

    /** {@code usearch_geospatial_tolerance_t}. */
    private static final StructLayout TOLERANCE = MemoryLayout.structLayout(
            ValueLayout.JAVA_DOUBLE.withName("metres"), ValueLayout.JAVA_DOUBLE.withName("seconds"));

    /** {@code usearch_geospatial_metadata_t}. */
    private static final StructLayout METADATA = MemoryLayout.structLayout(Library.SIZE.withName("connectivity"));

    private static final VarHandle METADATA_CONNECTIVITY = Library.field(METADATA, "connectivity");

    /** {@code usearch_geospatial_index_description_t}. */
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

    /**
     * The fields {@code usearch_geospatial_init_ram_t} and its VRAM twin share, at equal offsets.
     * The handles address it, as a RAM structure's handle refuses the shorter VRAM one.
     */
    private static final StructLayout INIT_SHARED = MemoryLayout.structLayout(
            Library.POINTER.withName("index"),
            Library.POINTER.withName("outcome"),
            TERRAIN.withName("terrain"),
            Library.SIZE.withName("connectivity"),
            Library.SIZE.withName("values_per_key"));

    /** {@code usearch_geospatial_init_ram_t} or its VRAM twin, whose placement field is named {@code placement}. */
    private static StructLayout initializing(String placement) {
        return BasicCalls.followedBy(INIT_SHARED, Library.placement(placement));
    }

    private static final Verb INIT_RAM = new Verb("usearch_geospatial_init_ram", initializing("ram"));
    private static final Verb INIT_VRAM = new Verb("usearch_geospatial_init_vram", initializing("vram"));
    private static final long INIT_TERRAIN = INIT_RAM.at("terrain");
    private static final VarHandle INIT_CONNECTIVITY = Library.field(INIT_SHARED, "connectivity");
    private static final VarHandle INIT_VALUES_PER_KEY = Library.field(INIT_SHARED, "values_per_key");

    /** {@code usearch_geospatial_load_ram_t}, its VRAM twin, and their buffer twins, reading {@code source}. */
    private static StructLayout loadingLayout(String placement) {
        return Library.opening(
                Library.padding(),
                Library.BYTES_VIEW.withName("source"),
                Library.placement(placement),
                Library.SIZE.withName("connectivity"),
                Library.SIZE.withName("values_per_key"),
                TERRAIN.withName("terrain"),
                Library.PROGRESS);
    }

    private static final Verb LOAD_RAM = new Verb("usearch_geospatial_load_ram", loadingLayout("ram"));
    private static final Verb LOAD_RAM_BUFFER = new Verb("usearch_geospatial_load_ram_buffer", loadingLayout("ram"));
    private static final Verb LOAD_VRAM = new Verb("usearch_geospatial_load_vram", loadingLayout("vram"));
    private static final Verb LOAD_VRAM_BUFFER = new Verb("usearch_geospatial_load_vram_buffer", loadingLayout("vram"));
    private static final long LOAD_SOURCE = LOAD_RAM.at("source");
    private static final VarHandle LOAD_RAM_VALUES_PER_KEY = LOAD_RAM.field("values_per_key");
    private static final VarHandle LOAD_VRAM_VALUES_PER_KEY = LOAD_VRAM.field("values_per_key");

    private static final Verb VIEW_RAM = new Verb(
            "usearch_geospatial_view_ram",
            MemoryLayout.structLayout(
                    Library.POINTER.withName("index"),
                    Library.POINTER.withName("outcome"),
                    Library.BYTES_VIEW.withName("path"),
                    TERRAIN.withName("terrain"),
                    Library.placement("ram")));

    private static final Verb VIEW_NVME = new Verb(
            "usearch_geospatial_view_nvme",
            Library.opening(
                    Library.padding(),
                    Library.BYTES_VIEW.withName("path"),
                    Library.placement("nvme"),
                    TERRAIN.withName("terrain")));

    private static final Verb RESERVE = new Verb(
            "usearch_geospatial_reserve", Library.opening(Library.padding(), Library.SIZE.withName("entries")));

    private static final Verb INSERT = new Verb(
            "usearch_geospatial_insert",
            Library.opening(
                    Library.padding(),
                    Library.SIZE.withName("count"),
                    Library.KEYS_VIEW.withName("keys"),
                    Library.POINTER.withName("observations"),
                    Library.SIZE.withName("expansion_factor"),
                    Library.PROGRESS,
                    Library.POINTER.withName("stats")));

    private static final VarHandle INSERT_COUNT = INSERT.field("count");
    private static final VarHandle INSERT_OBSERVATIONS = INSERT.field("observations");
    private static final VarHandle INSERT_EXPANSION_FACTOR = INSERT.field("expansion_factor");
    private static final VarHandle INSERT_STATS = INSERT.field("stats");

    private static final Verb SEARCH = new Verb(
            "usearch_geospatial_search",
            Library.opening(
                    Library.padding(),
                    Library.SIZE.withName("count"),
                    Library.POINTER.withName("queries"),
                    Library.ENUM.withName("relation"),
                    Library.padding(),
                    TOLERANCE.withName("tolerance"),
                    Library.POINTER.withName("matches"),
                    Library.POINTER.withName("arena"),
                    Library.SIZE.withName("wanted"),
                    Library.SIZE.withName("expansion_factor"),
                    Library.ENUM.withName("traversal"),
                    Library.padding(),
                    Library.PREDICATE,
                    Library.PROGRESS,
                    Library.POINTER.withName("stats")));

    private static final VarHandle SEARCH_COUNT = SEARCH.field("count");
    private static final VarHandle SEARCH_QUERIES = SEARCH.field("queries");
    private static final VarHandle SEARCH_RELATION = SEARCH.field("relation");
    private static final VarHandle SEARCH_METRES = SEARCH.field("tolerance", "metres");
    private static final VarHandle SEARCH_SECONDS = SEARCH.field("tolerance", "seconds");
    private static final VarHandle SEARCH_MATCHES = SEARCH.field("matches");
    private static final VarHandle SEARCH_ARENA = SEARCH.field("arena");
    private static final VarHandle SEARCH_WANTED = SEARCH.field("wanted");
    private static final VarHandle SEARCH_EXPANSION_FACTOR = SEARCH.field("expansion_factor");
    private static final VarHandle SEARCH_TRAVERSAL = SEARCH.field("traversal");
    private static final VarHandle SEARCH_STATS = SEARCH.field("stats");

    private static final Verb WITHIN = new Verb(
            "usearch_geospatial_within",
            Library.opening(
                    Library.padding(),
                    Library.SIZE.withName("count"),
                    Library.POINTER.withName("queries"),
                    Library.ENUM.withName("relation"),
                    Library.padding(),
                    TOLERANCE.withName("tolerance"),
                    Library.POINTER.withName("matches"),
                    Library.POINTER.withName("written"),
                    Library.POINTER.withName("arena"),
                    Library.ENUM.withName("traversal"),
                    Library.padding(),
                    Library.PREDICATE,
                    Library.PROGRESS,
                    Library.POINTER.withName("stats")));

    private static final VarHandle WITHIN_COUNT = WITHIN.field("count");
    private static final VarHandle WITHIN_QUERIES = WITHIN.field("queries");
    private static final VarHandle WITHIN_RELATION = WITHIN.field("relation");
    private static final VarHandle WITHIN_METRES = WITHIN.field("tolerance", "metres");
    private static final VarHandle WITHIN_SECONDS = WITHIN.field("tolerance", "seconds");
    private static final VarHandle WITHIN_MATCHES = WITHIN.field("matches");
    private static final VarHandle WITHIN_WRITTEN = WITHIN.field("written");
    private static final VarHandle WITHIN_ARENA = WITHIN.field("arena");
    private static final VarHandle WITHIN_TRAVERSAL = WITHIN.field("traversal");
    private static final VarHandle WITHIN_STATS = WITHIN.field("stats");

    // The key follows the completion unpadded, since sixteen bytes align to one.
    private static final Verb GET = new Verb(
            "usearch_geospatial_get",
            Library.opening(
                    Library.KEY.withName("key"),
                    Library.padding(),
                    Library.POINTER.withName("observations"),
                    Library.POINTER.withName("written"),
                    Library.POINTER.withName("arena")));

    private static final VarHandle GET_OBSERVATIONS = GET.field("observations");
    private static final VarHandle GET_WRITTEN = GET.field("written");
    private static final VarHandle GET_ARENA = GET.field("arena");

    private static final MethodHandle CPU_CLASS_NAME = Library.namer("usearch_cpu_class_name");
    private static final MethodHandle GPU_CLASS_NAME = Library.namer("usearch_gpu_class_name");
}
