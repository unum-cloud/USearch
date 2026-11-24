package cloud.unum.usearch;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.File;
import java.util.Arrays;
import java.util.Random;
import java.util.UUID;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.function.Executable;

import cloud.unum.usearch.DenseIndex.DenseBatch;
import cloud.unum.usearch.DenseIndex.DenseProducts;
import cloud.unum.usearch.DenseIndex.RefineReport;
import cloud.unum.usearch.GeospatialIndex.GeospatialIndexDescription;
import cloud.unum.usearch.GeospatialIndex.GeospatialOptions;
import cloud.unum.usearch.GeospatialIndex.GeospatialRefineConfig;
import cloud.unum.usearch.GeospatialIndex.GeospatialRefinePlan;
import cloud.unum.usearch.GeospatialIndex.GeospatialRelation;
import cloud.unum.usearch.GeospatialIndex.GeospatialReshardConfig;
import cloud.unum.usearch.GeospatialIndex.GeospatialReshardPlan;
import cloud.unum.usearch.GeospatialIndex.Observation;
import cloud.unum.usearch.GeospatialIndex.Tolerance;
import cloud.unum.usearch.SequenceIndex.SequenceCursor;
import cloud.unum.usearch.SequenceIndex.SequenceIndexDescription;
import cloud.unum.usearch.SequenceIndex.SequenceKeyStats;
import cloud.unum.usearch.SequenceIndex.SequenceMetric;
import cloud.unum.usearch.SequenceIndex.SequenceOptions;
import cloud.unum.usearch.SequenceIndex.SequenceRefineConfig;
import cloud.unum.usearch.SequenceIndex.SequenceRefinePlan;
import cloud.unum.usearch.SequenceIndex.SequenceReshardConfig;
import cloud.unum.usearch.SequenceIndex.SequenceReshardPlan;
import cloud.unum.usearch.ShapesIndex.ShapesIndexDescription;
import cloud.unum.usearch.ShapesIndex.ShapesKeyStats;
import cloud.unum.usearch.ShapesIndex.ShapesMetric;
import cloud.unum.usearch.ShapesIndex.ShapesOptions;
import cloud.unum.usearch.ShapesIndex.ShapesRefineConfig;
import cloud.unum.usearch.ShapesIndex.ShapesRefinePlan;
import cloud.unum.usearch.ShapesIndex.ShapesReshardConfig;
import cloud.unum.usearch.ShapesIndex.ShapesReshardPlan;
import cloud.unum.usearch.Types.Arena;
import cloud.unum.usearch.Types.Deferred;
import cloud.unum.usearch.Types.Executor;
import cloud.unum.usearch.Types.Key;
import cloud.unum.usearch.Types.Matches;
import cloud.unum.usearch.Types.Monolith;
import cloud.unum.usearch.Types.Phase;
import cloud.unum.usearch.Types.PhaseProgress;
import cloud.unum.usearch.Types.Scalar;
import cloud.unum.usearch.Types.Status;
import cloud.unum.usearch.Types.USearchException;

/**
 * Sequence, shapes and geospatial indexes, the modalities beside the dense one, as
 * {@code test/modalities.py} exercises them in Python. Each verb forwards to the C library and
 * raises whatever its outcome says, so nothing here asserts recall beyond an entry finding itself.
 *
 * <p>
 * The {@code readme*} tests mirror the snippets under {@code java/README.md}'s Sequence Indexes,
 * Shape Indexes and Geospatial Indexes sections line by line; whoever edits a snippet edits its
 * mirror. They differ only in running on the suite's shared {@code POOL} instead of opening one.
 */
public class ModalitiesTest {

    private static final Executor POOL = Executor.forRam(new int[0], null, 0);

    @AfterAll
    public static void releasePool() {
        POOL.close();
    }

    /** Asserts {@code action} refuses with {@code status}. */
    private static void refuses(Status status, Executable action) {
        assertEquals(status, assertThrows(USearchException.class, action).status());
    }

    /** Keys {@code first} up to {@code first + count}. */
    private static Key[] keys(int first, int count) {
        Key[] keys = new Key[count];
        for (int order = 0; order != count; ++order) {
            keys[order] = Key.of(first + order);
        }
        return keys;
    }

    private static String temporaryPath(String extension) {
        File file = new File(System.getProperty("java.io.tmpdir"), "usearch-" + UUID.randomUUID() + extension);
        file.deleteOnExit();
        return file.getPath();
    }

    /** Lowercase words of 8 to 16 letters. */
    private static Tape sequenceEntries(int count, long seed) {
        Random random = new Random(seed);
        byte[][] words = new byte[count][];
        for (int entry = 0; entry != count; ++entry) {
            words[entry] = new byte[8 + random.nextInt(9)];
            for (int letter = 0; letter != words[entry].length; ++letter) {
                words[entry][letter] = (byte) ('a' + random.nextInt(26));
            }
        }
        return Tape.ofBytes(words);
    }

    /** Clouds of 4 to 12 points in the unit cube. */
    private static Tape shapesEntries(int count, long seed) {
        Random random = new Random(seed);
        float[][] clouds = new float[count][];
        for (int entry = 0; entry != count; ++entry) {
            clouds[entry] = new float[3 * (4 + random.nextInt(9))];
            for (int coordinate = 0; coordinate != clouds[entry].length; ++coordinate) {
                clouds[entry][coordinate] = random.nextFloat();
            }
        }
        return Tape.ofPoints(clouds);
    }

    /** Positions between 60 degrees south and north, every other field unrecorded. */
    private static Observation[] geospatialEntries(int count, long seed) {
        Random random = new Random(seed);
        Observation[] observations = new Observation[count];
        for (int entry = 0; entry != count; ++entry) {
            observations[entry] = Observation.at(random.nextDouble() * 120 - 60, random.nextDouble() * 360 - 180);
        }
        return observations;
    }

    private static SequenceIndex sequenceBuilt(int count, int first, long seed) {
        SequenceIndex index = SequenceIndex.initRam(new SequenceOptions(), new int[0]);
        if (count != 0) {
            index.on(POOL).insert(keys(first, count), sequenceEntries(count, seed));
        }
        return index;
    }

    private static ShapesIndex shapesBuilt(int count, int first, long seed) {
        ShapesIndex index = ShapesIndex.initRam(new ShapesOptions(), new int[0]);
        if (count != 0) {
            index.on(POOL).insert(keys(first, count), shapesEntries(count, seed));
        }
        return index;
    }

    private static GeospatialIndex geospatialBuilt(int count, int first, long seed) {
        GeospatialIndex index = GeospatialIndex.initRam(new GeospatialOptions(), new int[0]);
        if (count != 0) {
            index.on(POOL).insert(keys(first, count), geospatialEntries(count, seed));
        }
        return index;
    }

    private static Observation[] places(double... coordinates) {
        Observation[] observations = new Observation[coordinates.length / 2];
        for (int entry = 0; entry != observations.length; ++entry) {
            observations[entry] = Observation.at(coordinates[2 * entry], coordinates[2 * entry + 1]);
        }
        return observations;
    }

    // region Sequence

    @Test
    public void sequenceInitVramSurfacesTheLibraryStatus() {
        try (SequenceIndex index = SequenceIndex.initVram(new SequenceOptions(), new int[0])) {
            assertNotEquals("unknown", index.describe().gpuClass());
        } catch (USearchException refused) {
            assertNotEquals(Status.SUCCESS, refused.status());
            assertNotEquals(Status.DEFERRED, refused.status());
        }
    }

    @Test
    public void sequenceRefineReportsARoundOverEveryEntry() {
        try (SequenceIndex index = sequenceBuilt(64, 0, 0)) {
            RefineReport report = index.on(POOL).refine();
            assertTrue(report.changed() <= report.reselected());
            assertEquals(64, report.reachability().audited());
            assertEquals(64, index.describe().size());
            assertTrue(index.contains(Key.of(17)));
        }
    }

    @Test
    public void sequenceRefinePlanResumesAndFinishes() {
        try (SequenceIndex index = sequenceBuilt(64, 0, 0)) {
            SequenceRefinePlan plan = new SequenceRefinePlan();
            try {
                index.on(POOL).refine(new SequenceRefineConfig().plan(plan).progress((completed, total) -> false));
            } catch (USearchException cancelled) {
                assertEquals(Status.CANCELLED, cancelled.status());
            }
            PhaseProgress stopped = plan.phase();
            assertNotEquals(Phase.DONE, stopped.phase());
            assertTrue(stopped.completed() <= stopped.total());
            index.on(POOL).refine(new SequenceRefineConfig().plan(plan));
            assertEquals(Phase.DONE, plan.phase().phase());
            assertEquals(
                    64, index.on(POOL).refinePlanFinish(plan).reachability().audited());
            refuses(Status.INVALID_ARGUMENT, () -> index.on(POOL).refinePlanFinish(plan));
        }
    }

    @Test
    public void sequenceRefinePlanNeedsAnExecutor() {
        try (SequenceIndex index = sequenceBuilt(0, 0, 0)) {
            refuses(
                    Status.INVALID_ARGUMENT,
                    () -> index.refine(new SequenceRefineConfig().plan(new SequenceRefinePlan())));
        }
    }

    @Test
    public void sequenceReshardGathersEverySourceIntoTheTarget() {
        try (SequenceIndex first = sequenceBuilt(32, 0, 0);
                SequenceIndex second = sequenceBuilt(32, 32, 1);
                SequenceIndex third = sequenceBuilt(32, 64, 2)) {
            RefineReport report = first.on(POOL)
                    .reshard(
                            new SequenceIndex[] {first, second, third},
                            new SequenceIndex[] {first},
                            new SequenceReshardConfig());
            assertTrue(report.changed() <= report.reselected());
            assertEquals(96, first.describe().size());
            assertTrue(first.contains(Key.of(40)) && first.contains(Key.of(80)));
            assertEquals(0, second.describe().size() + third.describe().size());
        }
    }

    @Test
    public void sequenceReshardCutsOneSourceIntoSeveralTargets() {
        try (SequenceIndex source = sequenceBuilt(64, 0, 0);
                SequenceIndex left = sequenceBuilt(0, 0, 0);
                SequenceIndex right = sequenceBuilt(0, 0, 0)) {
            source.on(POOL)
                    .reshard(
                            new SequenceIndex[] {source},
                            new SequenceIndex[] {left, right},
                            new SequenceReshardConfig().seed(7));
            assertEquals(64, left.describe().size() + right.describe().size());
            assertTrue(left.describe().size() != 0 && right.describe().size() != 0);
            assertEquals(0, source.describe().size());
        }
    }

    @Test
    public void sequenceReshardCutRunsThroughAPlanAndADeferredFinish() {
        try (SequenceIndex source = sequenceBuilt(64, 0, 0);
                SequenceIndex left = sequenceBuilt(0, 0, 0);
                SequenceIndex right = sequenceBuilt(0, 0, 0)) {
            SequenceIndex[] sources = {source};
            SequenceIndex[] targets = {left, right};
            SequenceReshardPlan plan = new SequenceReshardPlan();
            try {
                source.on(POOL)
                        .reshard(
                                sources,
                                targets,
                                new SequenceReshardConfig().plan(plan).progress((completed, total) -> false));
            } catch (USearchException cancelled) {
                assertEquals(Status.CANCELLED, cancelled.status());
            }
            assertEquals(64, left.describe().size() + right.describe().size());
            assertNotEquals(Phase.DONE, plan.phase().phase());
            source.on(POOL).reshard(sources, targets, new SequenceReshardConfig().plan(plan));
            assertEquals(Phase.DONE, plan.phase().phase());
            try (Deferred<RefineReport> finished = source.deferred(POOL).reshardPlanFinish(plan)) {
                finished.join();
            }
        }
    }

    @Test
    public void sequenceReshardRecutOfOneHandleIsUnsupported() {
        try (SequenceIndex index = sequenceBuilt(16, 0, 0)) {
            refuses(
                    Status.UNSUPPORTED,
                    () -> index.on(POOL)
                            .reshard(
                                    new SequenceIndex[] {index},
                                    new SequenceIndex[] {index},
                                    new SequenceReshardConfig()));
            assertEquals(16, index.describe().size());
        }
    }

    @Test
    public void sequenceReshardRefusesMalformedListsAndMovesNothing() {
        try (SequenceIndex first = sequenceBuilt(8, 0, 0);
                SequenceIndex second = sequenceBuilt(8, 8, 1)) {
            SequenceReshardConfig config = new SequenceReshardConfig();
            SequenceIndex[] into = {first};
            refuses(Status.INVALID_ARGUMENT, () -> first.on(POOL).reshard(new SequenceIndex[0], into, config));
            refuses(
                    Status.INVALID_ARGUMENT,
                    () -> first.on(POOL).reshard(new SequenceIndex[] {first, first}, into, config));
            refuses(
                    Status.CONFIG_MISMATCH,
                    () -> first.on(POOL).reshard(new SequenceIndex[] {first}, new SequenceIndex[] {second}, config));
            assertEquals(16, first.describe().size() + second.describe().size());
        }
    }

    @Test
    public void sequenceSavesAndLoadsThroughAPathAndABuffer() {
        String path = temporaryPath(".usearch");
        try (SequenceIndex index = sequenceBuilt(16, 0, 0);
                Arena arena = new Arena(POOL)) {
            index.on(POOL).save(path);
            Monolith monolith = index.on(POOL).saveBuffer(arena);
            try (SequenceIndex fromPath = SequenceIndex.loadRam(path, POOL, new int[0], 0, null, null);
                    SequenceIndex fromBuffer =
                            SequenceIndex.loadRamBuffer(monolith.segment(), POOL, new int[0], 0, null, null)) {
                assertEquals(16, fromPath.describe().size());
                assertEquals(16, fromBuffer.describe().size());
                assertTrue(fromBuffer.contains(Key.of(9)));
            }
            assertEquals(
                    SequenceIndex.metadataRead(path, POOL), SequenceIndex.metadataReadBuffer(monolith.segment(), POOL));
        }
    }

    @Test
    public void sequenceCursorWalksEveryEntryWithItsLinks() {
        try (SequenceIndex index = sequenceBuilt(16, 0, 0);
                SequenceCursor cursor = index.cursor();
                Arena arena = new Arena(POOL)) {
            DenseBatch batch = cursor.on(POOL).nextLinks(64, arena);
            assertEquals(16, batch.written());
            assertTrue(batch.linkCount(0) > 0);
        }
    }

    @Test
    public void sequenceDeferredReceiverAnswersWhatTheBlockingOneDoes() {
        try (SequenceIndex index = sequenceBuilt(8, 0, 0);
                Deferred<SequenceIndexDescription> described =
                        index.deferred(POOL).describe()) {
            assertEquals(index.describe().size(), described.join().size());
        }
    }

    // endregion Sequence

    // region Shapes

    @Test
    public void shapesInitVramSurfacesTheLibraryStatus() {
        try (ShapesIndex index = ShapesIndex.initVram(new ShapesOptions(), new int[0])) {
            assertNotEquals("unknown", index.describe().gpuClass());
        } catch (USearchException refused) {
            assertNotEquals(Status.SUCCESS, refused.status());
            assertNotEquals(Status.DEFERRED, refused.status());
        }
    }

    @Test
    public void shapesRefineReportsARoundOverEveryEntry() {
        try (ShapesIndex index = shapesBuilt(64, 0, 0)) {
            RefineReport report = index.on(POOL).refine();
            assertTrue(report.changed() <= report.reselected());
            assertEquals(64, report.reachability().audited());
            assertEquals(64, index.describe().size());
            assertTrue(index.contains(Key.of(17)));
        }
    }

    @Test
    public void shapesRefinePlanResumesAndFinishes() {
        try (ShapesIndex index = shapesBuilt(64, 0, 0)) {
            ShapesRefinePlan plan = new ShapesRefinePlan();
            try {
                index.on(POOL).refine(new ShapesRefineConfig().plan(plan).progress((completed, total) -> false));
            } catch (USearchException cancelled) {
                assertEquals(Status.CANCELLED, cancelled.status());
            }
            PhaseProgress stopped = plan.phase();
            assertNotEquals(Phase.DONE, stopped.phase());
            assertTrue(stopped.completed() <= stopped.total());
            index.on(POOL).refine(new ShapesRefineConfig().plan(plan));
            assertEquals(Phase.DONE, plan.phase().phase());
            assertEquals(
                    64, index.on(POOL).refinePlanFinish(plan).reachability().audited());
            refuses(Status.INVALID_ARGUMENT, () -> index.on(POOL).refinePlanFinish(plan));
        }
    }

    @Test
    public void shapesRefinePlanNeedsAnExecutor() {
        try (ShapesIndex index = shapesBuilt(0, 0, 0)) {
            refuses(Status.INVALID_ARGUMENT, () -> index.refine(new ShapesRefineConfig().plan(new ShapesRefinePlan())));
        }
    }

    @Test
    public void shapesReshardGathersEverySourceIntoTheTarget() {
        try (ShapesIndex first = shapesBuilt(32, 0, 0);
                ShapesIndex second = shapesBuilt(32, 32, 1);
                ShapesIndex third = shapesBuilt(32, 64, 2)) {
            RefineReport report = first.on(POOL)
                    .reshard(
                            new ShapesIndex[] {first, second, third},
                            new ShapesIndex[] {first},
                            new ShapesReshardConfig());
            assertTrue(report.changed() <= report.reselected());
            assertEquals(96, first.describe().size());
            assertTrue(first.contains(Key.of(40)) && first.contains(Key.of(80)));
            assertEquals(0, second.describe().size() + third.describe().size());
        }
    }

    @Test
    public void shapesReshardCutsOneSourceIntoSeveralTargets() {
        try (ShapesIndex source = shapesBuilt(64, 0, 0);
                ShapesIndex left = shapesBuilt(0, 0, 0);
                ShapesIndex right = shapesBuilt(0, 0, 0)) {
            source.on(POOL)
                    .reshard(
                            new ShapesIndex[] {source},
                            new ShapesIndex[] {left, right},
                            new ShapesReshardConfig().seed(7));
            assertEquals(64, left.describe().size() + right.describe().size());
            assertTrue(left.describe().size() != 0 && right.describe().size() != 0);
            assertEquals(0, source.describe().size());
        }
    }

    @Test
    public void shapesReshardCutRunsThroughAPlanAndADeferredFinish() {
        try (ShapesIndex source = shapesBuilt(64, 0, 0);
                ShapesIndex left = shapesBuilt(0, 0, 0);
                ShapesIndex right = shapesBuilt(0, 0, 0)) {
            ShapesIndex[] sources = {source};
            ShapesIndex[] targets = {left, right};
            ShapesReshardPlan plan = new ShapesReshardPlan();
            try {
                source.on(POOL)
                        .reshard(
                                sources,
                                targets,
                                new ShapesReshardConfig().plan(plan).progress((completed, total) -> false));
            } catch (USearchException cancelled) {
                assertEquals(Status.CANCELLED, cancelled.status());
            }
            assertEquals(64, left.describe().size() + right.describe().size());
            assertNotEquals(Phase.DONE, plan.phase().phase());
            source.on(POOL).reshard(sources, targets, new ShapesReshardConfig().plan(plan));
            assertEquals(Phase.DONE, plan.phase().phase());
            try (Deferred<RefineReport> finished = source.deferred(POOL).reshardPlanFinish(plan)) {
                finished.join();
            }
        }
    }

    @Test
    public void shapesReshardRecutOfOneHandleIsUnsupported() {
        try (ShapesIndex index = shapesBuilt(16, 0, 0)) {
            refuses(
                    Status.UNSUPPORTED,
                    () -> index.on(POOL)
                            .reshard(new ShapesIndex[] {index}, new ShapesIndex[] {index}, new ShapesReshardConfig()));
            assertEquals(16, index.describe().size());
        }
    }

    @Test
    public void shapesReshardRefusesMalformedListsAndMovesNothing() {
        try (ShapesIndex first = shapesBuilt(8, 0, 0);
                ShapesIndex second = shapesBuilt(8, 8, 1)) {
            ShapesReshardConfig config = new ShapesReshardConfig();
            ShapesIndex[] into = {first};
            refuses(Status.INVALID_ARGUMENT, () -> first.on(POOL).reshard(new ShapesIndex[0], into, config));
            refuses(
                    Status.INVALID_ARGUMENT,
                    () -> first.on(POOL).reshard(new ShapesIndex[] {first, first}, into, config));
            refuses(
                    Status.CONFIG_MISMATCH,
                    () -> first.on(POOL).reshard(new ShapesIndex[] {first}, new ShapesIndex[] {second}, config));
            assertEquals(16, first.describe().size() + second.describe().size());
        }
    }

    @Test
    public void shapesSavesAndLoadsThroughAPathAndABuffer() {
        String path = temporaryPath(".usearch");
        try (ShapesIndex index = shapesBuilt(16, 0, 0);
                Arena arena = new Arena(POOL)) {
            index.on(POOL).save(path);
            Monolith monolith = index.on(POOL).saveBuffer(arena);
            try (ShapesIndex fromPath = ShapesIndex.loadRam(path, POOL, new int[0], 0, null);
                    ShapesIndex fromBuffer = ShapesIndex.loadRamBuffer(monolith.segment(), POOL, new int[0], 0, null)) {
                assertEquals(16, fromPath.describe().size());
                assertEquals(16, fromBuffer.describe().size());
                assertTrue(fromBuffer.contains(Key.of(9)));
            }
            assertEquals(
                    ShapesIndex.metadataRead(path, POOL), ShapesIndex.metadataReadBuffer(monolith.segment(), POOL));
        }
    }

    @Test
    public void shapesCursorWalksEveryEntryWithItsLinks() {
        try (ShapesIndex index = shapesBuilt(16, 0, 0);
                ShapesIndex.ShapesCursor cursor = index.cursor();
                Arena arena = new Arena(POOL)) {
            DenseBatch batch = cursor.on(POOL).nextLinks(64, arena);
            assertEquals(16, batch.written());
            assertTrue(batch.linkCount(0) > 0);
        }
    }

    @Test
    public void shapesDeferredReceiverAnswersWhatTheBlockingOneDoes() {
        try (ShapesIndex index = shapesBuilt(8, 0, 0);
                Deferred<ShapesIndexDescription> described =
                        index.deferred(POOL).describe()) {
            assertEquals(index.describe().size(), described.join().size());
        }
    }

    // endregion Shapes

    // region Geospatial

    @Test
    public void geospatialInitVramSurfacesTheLibraryStatus() {
        try (GeospatialIndex index = GeospatialIndex.initVram(new GeospatialOptions(), new int[0])) {
            assertNotEquals("unknown", index.describe().gpuClass());
        } catch (USearchException refused) {
            assertNotEquals(Status.SUCCESS, refused.status());
            assertNotEquals(Status.DEFERRED, refused.status());
        }
    }

    @Test
    public void geospatialRefineReportsARoundOverEveryEntry() {
        try (GeospatialIndex index = geospatialBuilt(64, 0, 0)) {
            RefineReport report = index.on(POOL).refine();
            assertTrue(report.changed() <= report.reselected());
            assertEquals(64, report.reachability().audited());
            assertEquals(64, index.describe().size());
            assertTrue(index.contains(Key.of(17)));
        }
    }

    @Test
    public void geospatialRefinePlanResumesAndFinishes() {
        try (GeospatialIndex index = geospatialBuilt(64, 0, 0)) {
            GeospatialRefinePlan plan = new GeospatialRefinePlan();
            try {
                index.on(POOL).refine(new GeospatialRefineConfig().plan(plan).progress((completed, total) -> false));
            } catch (USearchException cancelled) {
                assertEquals(Status.CANCELLED, cancelled.status());
            }
            PhaseProgress stopped = plan.phase();
            assertNotEquals(Phase.DONE, stopped.phase());
            assertTrue(stopped.completed() <= stopped.total());
            index.on(POOL).refine(new GeospatialRefineConfig().plan(plan));
            assertEquals(Phase.DONE, plan.phase().phase());
            assertEquals(
                    64, index.on(POOL).refinePlanFinish(plan).reachability().audited());
            refuses(Status.INVALID_ARGUMENT, () -> index.on(POOL).refinePlanFinish(plan));
        }
    }

    @Test
    public void geospatialRefinePlanNeedsAnExecutor() {
        try (GeospatialIndex index = geospatialBuilt(0, 0, 0)) {
            refuses(
                    Status.INVALID_ARGUMENT,
                    () -> index.refine(new GeospatialRefineConfig().plan(new GeospatialRefinePlan())));
        }
    }

    @Test
    public void geospatialReshardGathersEverySourceIntoTheTarget() {
        try (GeospatialIndex first = geospatialBuilt(32, 0, 0);
                GeospatialIndex second = geospatialBuilt(32, 32, 1);
                GeospatialIndex third = geospatialBuilt(32, 64, 2)) {
            RefineReport report = first.on(POOL)
                    .reshard(
                            new GeospatialIndex[] {first, second, third},
                            new GeospatialIndex[] {first},
                            new GeospatialReshardConfig());
            assertTrue(report.changed() <= report.reselected());
            assertEquals(96, first.describe().size());
            assertTrue(first.contains(Key.of(40)) && first.contains(Key.of(80)));
            assertEquals(0, second.describe().size() + third.describe().size());
        }
    }

    @Test
    public void geospatialReshardCutsOneSourceIntoSeveralTargets() {
        try (GeospatialIndex source = geospatialBuilt(64, 0, 0);
                GeospatialIndex left = geospatialBuilt(0, 0, 0);
                GeospatialIndex right = geospatialBuilt(0, 0, 0)) {
            source.on(POOL)
                    .reshard(
                            new GeospatialIndex[] {source},
                            new GeospatialIndex[] {left, right},
                            new GeospatialReshardConfig().seed(7));
            assertEquals(64, left.describe().size() + right.describe().size());
            assertTrue(left.describe().size() != 0 && right.describe().size() != 0);
            assertEquals(0, source.describe().size());
        }
    }

    @Test
    public void geospatialReshardCutRunsThroughAPlanAndADeferredFinish() {
        try (GeospatialIndex source = geospatialBuilt(64, 0, 0);
                GeospatialIndex left = geospatialBuilt(0, 0, 0);
                GeospatialIndex right = geospatialBuilt(0, 0, 0)) {
            GeospatialIndex[] sources = {source};
            GeospatialIndex[] targets = {left, right};
            GeospatialReshardPlan plan = new GeospatialReshardPlan();
            try {
                source.on(POOL)
                        .reshard(
                                sources,
                                targets,
                                new GeospatialReshardConfig().plan(plan).progress((completed, total) -> false));
            } catch (USearchException cancelled) {
                assertEquals(Status.CANCELLED, cancelled.status());
            }
            assertEquals(64, left.describe().size() + right.describe().size());
            assertNotEquals(Phase.DONE, plan.phase().phase());
            source.on(POOL).reshard(sources, targets, new GeospatialReshardConfig().plan(plan));
            assertEquals(Phase.DONE, plan.phase().phase());
            try (Deferred<RefineReport> finished = source.deferred(POOL).reshardPlanFinish(plan)) {
                finished.join();
            }
        }
    }

    @Test
    public void geospatialReshardRecutOfOneHandleIsUnsupported() {
        try (GeospatialIndex index = geospatialBuilt(16, 0, 0)) {
            refuses(
                    Status.UNSUPPORTED,
                    () -> index.on(POOL)
                            .reshard(
                                    new GeospatialIndex[] {index},
                                    new GeospatialIndex[] {index},
                                    new GeospatialReshardConfig()));
            assertEquals(16, index.describe().size());
        }
    }

    @Test
    public void geospatialReshardRefusesMalformedListsAndMovesNothing() {
        try (GeospatialIndex first = geospatialBuilt(8, 0, 0);
                GeospatialIndex second = geospatialBuilt(8, 8, 1)) {
            GeospatialReshardConfig config = new GeospatialReshardConfig();
            GeospatialIndex[] into = {first};
            refuses(Status.INVALID_ARGUMENT, () -> first.on(POOL).reshard(new GeospatialIndex[0], into, config));
            refuses(
                    Status.INVALID_ARGUMENT,
                    () -> first.on(POOL).reshard(new GeospatialIndex[] {first, first}, into, config));
            refuses(
                    Status.CONFIG_MISMATCH,
                    () -> first.on(POOL)
                            .reshard(new GeospatialIndex[] {first}, new GeospatialIndex[] {second}, config));
            assertEquals(16, first.describe().size() + second.describe().size());
        }
    }

    @Test
    public void geospatialSavesAndLoadsThroughAPathAndABuffer() {
        String path = temporaryPath(".usearch");
        try (GeospatialIndex index = geospatialBuilt(16, 0, 0);
                Arena arena = new Arena(POOL)) {
            index.on(POOL).save(path);
            Monolith monolith = index.on(POOL).saveBuffer(arena);
            try (GeospatialIndex fromPath = GeospatialIndex.loadRam(path, POOL, new int[0], 0, null, null);
                    GeospatialIndex fromBuffer =
                            GeospatialIndex.loadRamBuffer(monolith.segment(), POOL, new int[0], 0, null, null)) {
                assertEquals(16, fromPath.describe().size());
                assertEquals(16, fromBuffer.describe().size());
                assertTrue(fromBuffer.contains(Key.of(9)));
            }
            assertEquals(
                    GeospatialIndex.metadataRead(path, POOL),
                    GeospatialIndex.metadataReadBuffer(monolith.segment(), POOL));
        }
    }

    @Test
    public void geospatialCursorWalksEveryEntryWithItsLinks() {
        try (GeospatialIndex index = geospatialBuilt(16, 0, 0);
                GeospatialIndex.GeospatialCursor cursor = index.cursor();
                Arena arena = new Arena(POOL)) {
            DenseBatch batch = cursor.on(POOL).nextLinks(64, arena);
            assertEquals(16, batch.written());
            assertTrue(batch.linkCount(0) > 0);
        }
    }

    @Test
    public void geospatialDeferredReceiverAnswersWhatTheBlockingOneDoes() {
        try (GeospatialIndex index = geospatialBuilt(8, 0, 0);
                Deferred<GeospatialIndexDescription> described =
                        index.deferred(POOL).describe()) {
            assertEquals(index.describe().size(), described.join().size());
        }
    }

    // endregion Geospatial

    // region Per-modality

    /** Mirrors the snippet under Sequence Indexes in {@code java/README.md}. */
    @Test
    public void readmeSequenceSnippet() {
        Executor pool = POOL;
        try (SequenceIndex words = SequenceIndex.initRam(
                        new SequenceOptions().metric(SequenceMetric.LEVENSHTEIN_UTF8), new int[0]);
                Arena arena = new Arena(pool)) {
            Key[] keys = {Key.of(1), Key.of(2), Key.of(3)};
            words.on(pool).insert(keys, Tape.ofStrings("kitten", "sitting", "mitten"));
            Matches matches = words.on(pool).search(Tape.ofStrings("sitting"), 1, arena);
            SequenceKeyStats stats = words.count(Key.of(2));
            System.out.println(matches.key(0, 0) + " at " + matches.score(0, 0) + ", " + stats.totalBytes() + " bytes");
            assertEquals(Key.of(2), matches.key(0, 0));
            assertEquals(0, matches.score(0, 0));
            assertEquals(new SequenceKeyStats(1, 7, 7), stats);
        }
    }

    /** Mirrors the snippet under Shape Indexes in {@code java/README.md}. */
    @Test
    public void readmeShapesSnippet() {
        Executor pool = POOL;
        try (ShapesIndex molecules =
                        ShapesIndex.initRam(new ShapesOptions().metric(ShapesMetric.HAUSDORFF), new int[0]);
                Arena arena = new Arena(pool)) {
            Tape atoms =
                    Tape.ofPoints(new float[] {0, 0, 0, 1.1f, 0, 0}, new float[] {0, 0, 0, 1.5f, 0, 0, 0, 1.5f, 0});
            molecules.on(pool).insert(new Key[] {Key.of(1), Key.of(2)}, atoms);
            Tape stored = molecules.on(pool).get(Key.of(2), Scalar.F32, arena);
            System.out.println(stored.count() + " entry of " + stored.points(0).length / 3 + " points");
            assertEquals(1, stored.count());
            assertArrayEquals(new float[] {0, 0, 0, 1.5f, 0, 0, 0, 1.5f, 0}, stored.points(0), 1e-3f);
        }
    }

    /** Mirrors the snippet under Geospatial Indexes in {@code java/README.md}. */
    @Test
    public void readmeGeospatialSnippet() {
        Executor pool = POOL;
        try (GeospatialIndex places = GeospatialIndex.initRam(new GeospatialOptions(), new int[0]);
                Arena arena = new Arena(pool)) {
            Observation[] stops = {
                Observation.at(51.5308, -0.1238), Observation.at(51.5320, -0.1233), Observation.at(51.5033, -0.1196)
            };
            places.on(pool).insert(new Key[] {Key.of(1), Key.of(2), Key.of(3)}, stops);
            Observation[] station = {Observation.at(51.5308, -0.1238)};
            DenseProducts nearby =
                    places.on(pool).within(station, GeospatialRelation.NEAR, new Tolerance(500, 0), arena);
            System.out.println(nearby.count(0) + " stops within 500 m");
            assertEquals(2, nearby.count(0));
        }
    }

    @Test
    public void sequenceInitRamStartsEmptyAndKeepsTheMetric() {
        try (SequenceIndex index =
                SequenceIndex.initRam(new SequenceOptions().metric(SequenceMetric.LEVENSHTEIN_UTF8), new int[0])) {
            SequenceIndexDescription described = index.describe();
            assertEquals(0, described.size());
            assertEquals(SequenceMetric.LEVENSHTEIN_UTF8, described.metadata().metric());
            assertNotEquals("unknown", described.cpuClass());
            assertEquals("unknown", described.gpuClass());
        }
    }

    @Test
    public void sequenceSearchGetAndCountReadTheInsertedEntries() {
        try (SequenceIndex index = SequenceIndex.initRam(new SequenceOptions(), new int[0]);
                Arena arena = new Arena(POOL)) {
            index.on(POOL).insert(keys(1, 3), Tape.ofStrings("kitten", "sitting", "mitten"));
            assertEquals(3, index.describe().size());
            Matches matches = index.on(POOL).search(Tape.ofStrings("sitting"), 1, arena);
            assertEquals(Key.of(2), matches.key(0, 0));
            assertEquals(0, matches.score(0, 0));
            Tape stored = index.on(POOL).get(Key.of(2), arena);
            assertEquals(1, stored.count());
            assertEquals("sitting", stored.string(0));
            assertEquals(0, index.count(Key.of(9)).entries());
            assertEquals(0, index.on(POOL).get(Key.of(9), arena).count());
        }
    }

    @Test
    public void sequenceKindsRefuseWeightsAndWidthsTheyDoNotTake() {
        refuses(
                Status.CONFIG_MISMATCH,
                () -> SequenceIndex.initRam(new SequenceOptions().metric(SequenceMetric.NEEDLEMAN_WUNSCH), new int[0]));
        refuses(Status.CONFIG_MISMATCH, () -> SequenceIndex.initRam(new SequenceOptions().windowWidths(3), new int[0]));
        refuses(Status.SHAPE_MISMATCH, () -> new SequenceIndex.SequenceWeights(new byte[16], (byte) 1));
        try (SequenceIndex index = SequenceIndex.initRam(
                new SequenceOptions().metric(SequenceMetric.OVERLAP).windowWidths(2, 3), new int[0])) {
            assertEquals(SequenceMetric.OVERLAP, index.describe().metadata().metric());
            assertEquals("overlap", SequenceMetric.OVERLAP.toString());
        }
    }

    @Test
    public void sequenceMetricsReadBackFromTheirNames() {
        for (SequenceMetric metric : SequenceMetric.values()) {
            if (metric != SequenceMetric.UNKNOWN) {
                assertEquals(metric, SequenceMetric.fromName(metric.toString()));
            }
        }
        refuses(Status.INVALID_ARGUMENT, () -> SequenceMetric.fromName("hamming-ish"));
        refuses(Status.INVALID_ARGUMENT, () -> SequenceMetric.fromName(""));
    }

    @Test
    public void sequenceInsertRefusesKeysTheTapeDoesNotMatch() {
        try (SequenceIndex index = SequenceIndex.initRam(new SequenceOptions(), new int[0])) {
            refuses(Status.SHAPE_MISMATCH, () -> index.insert(keys(0, 2), Tape.ofStrings("one")));
            assertEquals(0, index.describe().size());
        }
    }

    @Test
    public void shapesOfDifferentPointCountsShareOneBatch() {
        Random random = new Random(42);
        float[][] clouds = new float[3][];
        int[] points = {5, 9, 13};
        for (int entry = 0; entry != 3; ++entry) {
            clouds[entry] = new float[3 * points[entry]];
            for (int coordinate = 0; coordinate != clouds[entry].length; ++coordinate) {
                clouds[entry][coordinate] = random.nextFloat();
            }
        }
        try (ShapesIndex index = ShapesIndex.initRam(new ShapesOptions().metric(ShapesMetric.HAUSDORFF), new int[0]);
                Arena arena = new Arena(POOL)) {
            index.on(POOL).insert(keys(1, 3), Tape.ofPoints(clouds));
            ShapesIndexDescription described = index.describe();
            assertEquals(3, described.size());
            assertEquals(27, described.totalPoints());
            assertEquals(ShapesMetric.HAUSDORFF, described.metadata().metric());
            assertEquals(new ShapesKeyStats(1, 13, 13), index.count(Key.of(3)));
            Matches matches = index.on(POOL).search(Tape.ofPoints(clouds[2]), 1, arena);
            assertEquals(Key.of(3), matches.key(0, 0));
            assertArrayEquals(
                    clouds[0], index.on(POOL).get(Key.of(1), Scalar.F32, arena).points(0), 1e-4f);
        }
    }

    @Test
    public void shapesMetricsReadBackFromTheirNames() {
        for (ShapesMetric metric : ShapesMetric.values()) {
            if (metric != ShapesMetric.UNKNOWN) {
                assertEquals(metric, ShapesMetric.fromName(metric.toString()));
            }
        }
        assertEquals("gromov-wasserstein", ShapesMetric.GROMOV_WASSERSTEIN.toString());
        assertEquals("wasserstein-procrustes", ShapesMetric.WASSERSTEIN_PROCRUSTES.toString());
        refuses(Status.INVALID_ARGUMENT, () -> ShapesMetric.fromName("gromov_wasserstein"));
        refuses(Status.INVALID_ARGUMENT, () -> ShapesMetric.fromName("chamfer-ish"));
        refuses(Status.INVALID_ARGUMENT, () -> ShapesMetric.fromName(""));
    }

    @Test
    public void shapesRefuseCloudsOfPartialPoints() {
        refuses(Status.SHAPE_MISMATCH, () -> Tape.ofPoints(new float[] {0, 1}));
    }

    @Test
    public void geospatialSearchFindsTheInsertedEntry() {
        Observation[] batch = places(48.8566, 2.3522, 52.52, 13.405, 40.7128, -74.006);
        try (GeospatialIndex index = GeospatialIndex.initRam(new GeospatialOptions(), new int[0]);
                Arena arena = new Arena(POOL)) {
            index.on(POOL).insert(keys(1, 3), batch);
            GeospatialIndexDescription described = index.describe();
            assertEquals(3, described.size());
            Matches matches = index.on(POOL)
                    .search(new Observation[] {batch[1]}, GeospatialRelation.NEAR, new Tolerance(0, 0), 1, arena);
            assertEquals(Key.of(2), matches.key(0, 0));
            assertEquals(0, matches.score(0, 0));
            assertEquals(1, index.count(Key.of(2)));
            Observation stored = index.on(POOL).get(Key.of(2), arena)[0];
            assertEquals(52.52, stored.latitudeDegrees(), 1e-6);
            assertTrue(Double.isNaN(stored.headingDegrees()));
        }
    }

    @Test
    public void geospatialWithinReportsOnlyTheEntriesInsideTheRadius() {
        Observation[] batch = places(48.8566, 2.3522, 48.8656, 2.3522, 52.52, 13.405);
        try (GeospatialIndex index = GeospatialIndex.initRam(new GeospatialOptions(), new int[0]);
                Arena arena = new Arena(POOL)) {
            index.on(POOL).insert(keys(1, 3), batch);
            DenseProducts report = index.on(POOL)
                    .within(new Observation[] {batch[0]}, GeospatialRelation.NEAR, new Tolerance(2000, 0), arena);
            long[] found = new long[(int) report.count(0)];
            for (int rank = 0; rank != found.length; ++rank) {
                found[rank] = report.key(0, rank).toUUID().getLeastSignificantBits();
            }
            Arrays.sort(found);
            assertArrayEquals(new long[] {1, 2}, found);
        }
    }

    @Test
    public void geospatialVisibleNeedsATerrain() {
        try (GeospatialIndex index = geospatialBuilt(4, 0, 0);
                Arena arena = new Arena(POOL)) {
            Observation[] query = places(0, 0);
            assertThrows(
                    USearchException.class,
                    () -> index.on(POOL).search(query, GeospatialRelation.VISIBLE, new Tolerance(1000, 0), 1, arena));
        }
    }

    // endregion Per-modality
}
