package cloud.unum.usearch;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import cloud.unum.usearch.DenseIndex.DenseMetric;
import cloud.unum.usearch.DenseIndex.DenseOptions;
import cloud.unum.usearch.DenseIndex.DenseStats;
import cloud.unum.usearch.DenseIndex.InsertionOptions;
import cloud.unum.usearch.DenseIndex.Refinement;
import cloud.unum.usearch.DenseIndex.SearchOptions;
import cloud.unum.usearch.Types.Executor;
import cloud.unum.usearch.Types.Extents;
import cloud.unum.usearch.Types.Key;
import cloud.unum.usearch.Types.Matches;
import cloud.unum.usearch.Types.Matrix;
import cloud.unum.usearch.Types.Residency;
import cloud.unum.usearch.Types.ScalarKind;
import cloud.unum.usearch.Types.Status;
import cloud.unum.usearch.Types.USearchException;
import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.UUID;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Test;

/**
 * The boundary this SDK owns: how a value is marshalled, how an extent is checked, and how a handle
 * dies. That vector search returns near neighbors is the engine's contract and is tested where the
 * engine lives, so nothing here asserts recall.
 *
 * <p>
 * The tests live in the binding's own package so they can reach the marshalling round trip and the
 * output slabs, which no caller ever sees and which nothing else can exercise.
 */
public class DenseIndexTest {

    /** One pool every fanned-out verb here runs on; the engine derives none. */
    private static final Executor POOL = new Executor();

    @AfterAll
    public static void releasePool() {
        POOL.close();
    }

    private static DenseOptions twoDimensional() {
        return new DenseOptions().dimensions(2).metric(DenseMetric.EUCLIDEAN).quantization(ScalarKind.F32);
    }

    private static DenseIndex populated() {
        DenseIndex index = twoDimensional().open();
        index.reserve(4);
        index.add(new Key[] {Key.of(10), Key.of(20)}, Matrix.of(new float[] {0, 1, 1, 0}, 2), POOL);
        return index;
    }

    @Test
    public void keyRendersAsDashedUuid() {
        assertEquals("00000000-0000-0000-0000-00000000002a", Key.of(42).toString());
        assertEquals(Key.NIL, Key.of(0));
        assertEquals(Key.of(42), Key.parse("00000000-0000-0000-0000-00000000002a"));
        assertEquals(Key.of(42), Key.parse("0000000000000000000000000000002A"));
        assertEquals(Key.of(42), Key.fromUUID(new UUID(0, 42)));
        assertEquals(new UUID(0, 42), Key.of(42).toUUID());
        assertNotEquals(Key.of(42), Key.of(43));

        // Keys order by their bytes unsigned, where UUID reads the same two longs as signed.
        assertTrue(Key.of(42).compareTo(Key.of(43)) < 0);
        assertTrue(Key.parse("ffffffff-0000-0000-0000-000000000000").compareTo(Key.of(43)) > 0);
    }

    @Test
    public void keyBytesAreMostSignificantFirst() {
        byte[] bytes = Key.of(0x0102030405060708L).toBytes();
        assertArrayEquals(new byte[] {0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8}, bytes);
        assertEquals(0x0102030405060708L, Key.ofBytes(bytes).toLong());

        Key wide = Key.parse("ffffffff-0000-0000-0000-000000000001");
        assertFalse(wide.isIntegral());
        assertEquals(
                Status.MALFORMED_INPUT,
                assertThrows(USearchException.class, wide::toLong).status());
    }

    @Test
    public void keyRejectsMalformedText() {
        for (String malformed :
                new String[] {"", "00000000-0000-0000-0000-0000000000zz", "00000000-0000-0000-0000-0000000000000"}) {
            assertEquals(
                    Status.MALFORMED_INPUT,
                    assertThrows(USearchException.class, () -> Key.parse(malformed))
                            .status());
        }
    }

    @Test
    public void optionsRoundTripThroughEveryResidency() {
        DenseOptions ram = new DenseOptions()
                .dimensions(96)
                .metric(DenseMetric.JACCARD)
                .quantization(ScalarKind.U1X8)
                .connectivity(24)
                .valuesPerKey(3)
                .residency(Residency.RAM);
        DenseOptions reloaded = ram.reload();
        assertEquals(96, reloaded.dimensions());
        assertEquals(DenseMetric.JACCARD, reloaded.metric());
        assertEquals(ScalarKind.U1X8, reloaded.quantization());
        assertEquals(24, reloaded.connectivity());
        assertEquals(3, reloaded.valuesPerKey());
        assertEquals(Residency.RAM, reloaded.residency());

        assertEquals(
                Residency.VRAM,
                twoDimensional().residency(Residency.VRAM).reload().residency());
        assertEquals(
                Residency.NVME,
                twoDimensional().residency(Residency.NVME).reload().residency());
    }

    @Test
    public void optionsRejectATagThisBuildCannotName() {
        assertEquals(
                Status.UNKNOWN_KIND,
                assertThrows(USearchException.class, () -> Residency.of(99)).status());

        long[] metric = twoDimensional().toWords();
        metric[DenseIndex.OptionsWords.METRIC] = 77;
        assertEquals(
                Status.UNKNOWN_KIND,
                assertThrows(USearchException.class, () -> DenseOptions.fromWords(metric))
                        .status());

        long[] scalar = twoDimensional().toWords();
        scalar[DenseIndex.OptionsWords.QUANTIZATION] = 77;
        assertEquals(
                Status.UNKNOWN_KIND,
                assertThrows(USearchException.class, () -> DenseOptions.fromWords(scalar))
                        .status());
    }

    @Test
    public void extentsRefuseToWrap() {
        assertEquals(
                Status.OVERFLOW,
                assertThrows(USearchException.class, () -> Extents.product(Long.MAX_VALUE / 3, 4, "the extent"))
                        .status());
        assertEquals(
                Status.OVERFLOW,
                assertThrows(USearchException.class, () -> Extents.toArrayLength(1L << 40, "the extent"))
                        .status());
        assertEquals(
                Status.SHAPE_MISMATCH,
                assertThrows(USearchException.class, () -> Extents.product(-1, 4, "the extent"))
                        .status());

        // The queries-by-wanted product feeds three slab lengths, so it is checked in the constructor.
        assertEquals(
                Status.OVERFLOW,
                assertThrows(USearchException.class, () -> new Matches(1_000_000, 1_000_000))
                        .status());
    }

    @Test
    public void perCallOptionsRefuseANegativeBreadth() {
        assertEquals(
                Status.SHAPE_MISMATCH,
                assertThrows(USearchException.class, () -> new InsertionOptions(-1))
                        .status());
        assertEquals(
                Status.SHAPE_MISMATCH,
                assertThrows(USearchException.class, () -> new SearchOptions(-1))
                        .status());
        assertEquals(0, InsertionOptions.DERIVED.expansionFactor());
        assertEquals(0, SearchOptions.DERIVED.expansionFactor());
    }

    @Test
    public void matricesRefuseLayoutsTheEngineCannotTraverse() {
        assertEquals(
                Status.SHAPE_MISMATCH,
                assertThrows(USearchException.class, () -> Matrix.of(new float[] {1, 2, 3}, 2))
                        .status());
        assertEquals(
                Status.UNSUPPORTED_LAYOUT,
                assertThrows(
                                USearchException.class,
                                () -> Matrix.strided(new float[] {1, 2, 3, 4}, 2, 2, 4, ScalarKind.F32))
                        .status());
        assertEquals(
                Status.SHAPE_MISMATCH,
                assertThrows(USearchException.class, () -> Matrix.strided(new float[] {1, 2}, 4, 2, 8, ScalarKind.F32))
                        .status());
        assertEquals(
                Status.UNSUPPORTED_LAYOUT,
                assertThrows(USearchException.class, () -> Matrix.of(new short[] {1, 2}, 2, ScalarKind.F64))
                        .status());
        assertEquals(
                Status.UNSUPPORTED_LAYOUT,
                assertThrows(
                                USearchException.class,
                                () -> Matrix.strided(ByteBuffer.allocate(64), 2, 2, 8, ScalarKind.F32))
                        .status());
        assertEquals(
                Status.UNSUPPORTED_LAYOUT,
                assertThrows(USearchException.class, () -> Matrix.strided("not a slab", 1, 2, 8, ScalarKind.F32))
                        .status());

        // Sub-byte kinds count logical dimensions, so sixteen bits fill two bytes of one row.
        Matrix packed = Matrix.of(new byte[] {1, 2, 3, 4}, 16, ScalarKind.U1X8);
        assertEquals(2, packed.rows());
        assertEquals(2, packed.strideBytes());
        assertEquals(16, packed.columns());
    }

    @Test
    public void strideArithmeticSkipsThePadding() {
        try (DenseIndex index = twoDimensional().open()) {
            index.reserve(2);
            // Four floats per row, of which the trailing two are padding no dimension names.
            float[] padded = {0, 1, 9, 9, 1, 0, 9, 9};
            index.add(new Key[] {Key.of(1), Key.of(2)}, Matrix.strided(padded, 2, 2, 16, ScalarKind.F32), POOL);

            Matrix retrieved = Matrix.allocate(1, 2, ScalarKind.F32);
            assertEquals(1, index.get(Key.of(2), retrieved));
            assertArrayEquals(new float[] {1, 0}, retrieved.floats(), 0f);
        }
    }

    @Test
    public void searchWritesOnlyWhatTheCountsClaim() {
        try (DenseIndex index = populated()) {
            Matches matches = index.search(Matrix.of(new float[] {0, 1}, 2), 4, POOL);
            assertEquals(1, matches.queries());
            assertEquals(4, matches.wanted());
            assertEquals(2, matches.count(0));
            assertFalse(matches.truncated(0));
            assertEquals(Key.of(10), matches.key(0, 0));

            // Slots past the count hold memory the engine never wrote, so reading one is refused.
            assertEquals(
                    Status.SHAPE_MISMATCH,
                    assertThrows(USearchException.class, () -> matches.key(0, 2))
                            .status());
            assertEquals(
                    Status.SHAPE_MISMATCH,
                    assertThrows(USearchException.class, () -> matches.score(0, 3))
                            .status());
            assertEquals(
                    Status.SHAPE_MISMATCH,
                    assertThrows(USearchException.class, () -> matches.count(1)).status());
        }
    }

    @Test
    public void searchClampsACountThatOverrunsItsRoom() {
        // The C interface reports the true extent, which may exceed the room reserved for it.
        Matches matches = new Matches(2, 2);
        matches.countSlab()[0] = 9;
        matches.countSlab()[1] = 1;
        assertEquals(2, matches.count(0));
        assertEquals(9, matches.reportedCount(0));
        assertTrue(matches.truncated(0));
        assertFalse(matches.truncated(1));
        assertEquals(
                Status.SHAPE_MISMATCH,
                assertThrows(USearchException.class, () -> matches.key(1, 1)).status());
    }

    @Test
    public void everyWalkReportsTraversalTelemetry() {
        try (DenseIndex index = populated()) {
            Matches matches = new Matches(1, 2);
            DenseStats searched =
                    index.searchInto(Matrix.of(new float[] {0, 1}, 2), matches, POOL, new SearchOptions(64));
            assertTrue(searched.visitedMembers() > 0);
            assertTrue(searched.computedDistances() > 0);
            assertEquals(searched, DenseStats.NONE.plus(searched));
            assertEquals(2 * searched.restarts(), searched.plus(searched).restarts());

            Refinement pass = index.refine(POOL, new InsertionOptions(32));
            assertTrue(pass.stats().visitedMembers() > 0);
        }
    }

    @Test
    public void lateInteractionDividesTokenRowsIntoQueries() {
        try (DenseIndex index = new DenseOptions()
                .dimensions(2)
                .metric(DenseMetric.INNER)
                .quantization(ScalarKind.F32)
                .valuesPerKey(2)
                .open()) {
            index.reserve(4);
            // Two documents of two token rows apiece, so a key names a document and not a row.
            index.add(
                    new Key[] {Key.of(10), Key.of(10), Key.of(20), Key.of(20)},
                    Matrix.of(new float[] {1, 0, 0, 1, 2, 0, 0, 2}, 2),
                    POOL);

            Matches matches = index.maxsim(Matrix.of(new float[] {1, 0, 0, 1}, 2), 2, 2, POOL);
            assertEquals(1, matches.queries());
            assertEquals(2, matches.count(0));

            // The rows of a batch must divide into whole queries, and the check runs before the call.
            assertEquals(
                    Status.SHAPE_MISMATCH,
                    assertThrows(
                                    USearchException.class,
                                    () -> index.maxsim(Matrix.of(new float[] {1, 0, 0, 1}, 2), 3, 1, POOL))
                            .status());
        }
    }

    @Test
    public void lateInteractionRefusesADistanceKind() {
        // No Java-side check produces CONFIG_MISMATCH, so this is the one failure that can only have
        // come from the boundary's exception factory, and its detail from the engine's own string.
        try (DenseIndex index = populated()) {
            USearchException refused = assertThrows(
                    USearchException.class, () -> index.maxsim(Matrix.of(new float[] {0, 1}, 2), 1, 1, POOL));
            assertEquals(Status.CONFIG_MISMATCH, refused.status());
            assertTrue(refused.getMessage().startsWith("usearch: config-mismatch: "));
            assertFalse(refused.getMessage().endsWith(": "));
        }
    }

    @Test
    public void unfrozenGeometryDefersEveryDerivedExtent() {
        // A zero dimensionality is legal: the geometry arrives with a later checkpoint.
        try (DenseIndex index = new DenseOptions().quantization(ScalarKind.F32).open()) {
            assertEquals(0, index.dimensions());
            assertEquals(
                    Status.UNINITIALIZED,
                    assertThrows(
                                    USearchException.class,
                                    () -> index.add(new Key[] {Key.of(1)}, Matrix.of(new float[] {0, 1}, 2), POOL))
                            .status());
        }
    }

    @Test
    public void geometryMismatchIsCaughtBeforeTheBoundary() {
        try (DenseIndex index = twoDimensional().open()) {
            index.reserve(1);
            assertEquals(
                    Status.SHAPE_MISMATCH,
                    assertThrows(
                                    USearchException.class,
                                    () -> index.add(new Key[] {Key.of(1)}, Matrix.of(new float[] {0, 1, 2}, 3), POOL))
                            .status());
            assertEquals(
                    Status.SHAPE_MISMATCH,
                    assertThrows(
                                    USearchException.class,
                                    () -> index.add(
                                            new Key[] {Key.of(1), Key.of(2)}, Matrix.of(new float[] {0, 1}, 2), POOL))
                            .status());
        }
    }

    @Test
    public void aScalarKindTheLibraryRejectsSurfacesAsUnknownKind() {
        try (DenseIndex index = twoDimensional().open()) {
            index.reserve(1);
            assertEquals(
                    Status.UNKNOWN_KIND,
                    assertThrows(
                                    USearchException.class,
                                    () -> index.add(
                                            new Key[] {Key.of(1)},
                                            Matrix.of(new byte[] {1, 2}, 2, ScalarKind.E5M2),
                                            POOL))
                            .status());
        }
    }

    @Test
    public void getReportsTheTrueExtentEvenWhenItTruncates() {
        try (DenseIndex index = twoDimensional().open()) {
            index.reserve(3);
            // Keys need not be unique, so three entries share one and the count is the extent.
            index.add(new Key[] {Key.of(5), Key.of(5), Key.of(5)}, Matrix.of(new float[] {0, 1, 1, 0, 1, 1}, 2), POOL);
            assertEquals(3, index.count(Key.of(5)));
            assertTrue(index.contains(Key.of(5)));
            assertFalse(index.contains(Key.of(6)));

            Matrix narrow = Matrix.allocate(1, 2, ScalarKind.F32);
            assertEquals(3, index.get(Key.of(5), narrow));
            assertArrayEquals(new float[] {0, 1}, narrow.floats(), 0f);
        }
    }

    @Test
    public void aClosedHandleRefusesEveryVerb() {
        DenseIndex index = populated();
        index.close();
        index.close();
        assertEquals(
                Status.UNINITIALIZED,
                assertThrows(USearchException.class, index::size).status());
        assertEquals(
                Status.UNINITIALIZED,
                assertThrows(USearchException.class, () -> index.reserve(1)).status());
    }

    @Test
    public void checkpointsRoundTripThroughAPathAndABuffer() throws Exception {
        File checkpoint = File.createTempFile("usearch", ".safegraphs");
        checkpoint.deleteOnExit();
        byte[] slab;
        try (DenseIndex index = populated()) {
            index.save(checkpoint.getAbsolutePath(), POOL);
            slab = new byte[Extents.toArrayLength(index.storageRequirements(), "the checkpoint")];
            index.saveBuffer(slab, POOL);
        }

        DenseOptions described = DenseIndex.metadata(checkpoint.getAbsolutePath());
        assertEquals(2, described.dimensions());
        assertEquals(DenseMetric.EUCLIDEAN, described.metric());
        assertEquals(2, DenseIndex.metadata(slab).dimensions());

        try (DenseIndex reopened = twoDimensional().open()) {
            reopened.load(checkpoint.getAbsolutePath(), POOL);
            assertEquals(2, reopened.size());
        }

        ByteBuffer direct = ByteBuffer.allocateDirect(slab.length).order(ByteOrder.nativeOrder());
        direct.put(slab).flip();
        try (DenseIndex reopened = twoDimensional().open()) {
            reopened.loadBuffer(direct, POOL);
            assertEquals(2, reopened.size());
            assertEquals(2, DenseIndex.metadata(direct).dimensions());
        }
    }

    @Test
    public void directBuffersCrossWithoutACopy() {
        try (DenseIndex index = twoDimensional().open()) {
            index.reserve(2);
            ByteBuffer rows = ByteBuffer.allocateDirect(4 * Float.BYTES).order(ByteOrder.nativeOrder());
            rows.asFloatBuffer().put(new float[] {0, 1, 1, 0});
            index.add(
                    new Key[] {Key.of(1), Key.of(2)},
                    Matrix.strided(rows, 2, 2, 2L * Float.BYTES, ScalarKind.F32),
                    POOL);
            assertEquals(2, index.size());

            ByteBuffer queries = ByteBuffer.allocateDirect(2 * Float.BYTES).order(ByteOrder.nativeOrder());
            queries.asFloatBuffer().put(new float[] {1, 0});
            Matches matches = index.search(Matrix.strided(queries, 1, 2, 2L * Float.BYTES, ScalarKind.F32), 1, POOL);
            assertEquals(Key.of(2), matches.key(0, 0));
        }
    }

    @Test
    public void anExplicitlySizedExecutorDrivesAndDrains() {
        try (Executor pool = new Executor(Residency.RAM, 2);
                DenseIndex index = twoDimensional().open()) {
            index.reserve(2);
            index.add(new Key[] {Key.of(10), Key.of(20)}, Matrix.of(new float[] {0, 1, 1, 0}, 2), pool);
            index.search(Matrix.of(new float[] {0, 1}, 2), 1, pool);
            pool.drain();
            assertEquals(2, index.size());
        }
    }

    @Test
    public void aClosedExecutorRefusesEveryVerb() {
        Executor pool = new Executor();
        pool.close();
        pool.close();
        assertEquals(
                Status.UNINITIALIZED,
                assertThrows(USearchException.class, pool::drain).status());
        assertEquals(
                Status.SHAPE_MISMATCH,
                assertThrows(USearchException.class, () -> new Executor(Residency.RAM, -1))
                        .status());
    }

    @Test
    public void theLinkedLibraryNamesItself() {
        assertFalse(Types.version().isEmpty());
        assertFalse(Types.capabilitiesCompiled().isEmpty());
        assertFalse(Types.capabilitiesAvailable().isEmpty());
        try (DenseIndex index = populated()) {
            assertFalse(index.capabilities().isEmpty());
            assertTrue(index.capacity() >= index.size());
            assertTrue(index.memoryUsage() > 0);
            assertTrue(index.connectivity() > 0);
            index.clear();
            assertEquals(0, index.size());
        }
    }
}
