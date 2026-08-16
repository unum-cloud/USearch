package cloud.unum.usearch;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.junit.Test;

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

    private static DenseOptions twoDimensional() {
        return new DenseOptions().dimensions(2).metric(DenseMetric.EUCLIDEAN).quantization(ScalarKind.F32);
    }

    private static DenseIndex populated() {
        DenseIndex index = twoDimensional().open();
        index.reserve(4);
        index.add(new Key[] {Key.of(10), Key.of(20)}, DenseMatrix.of(new float[] {0, 1, 1, 0}, 2));
        return index;
    }

    @Test
    public void keyRendersAsDashedUuid() {
        assertEquals("00000000-0000-0000-0000-00000000002a", Key.of(42).toString());
        assertEquals("00000000-0000-0000-0000-000000000000", Key.NIL.toString());
        assertEquals(Key.of(42), Key.parse("00000000-0000-0000-0000-00000000002a"));
        assertEquals(Key.of(42), Key.parse("0000000000000000000000000000002A"));
        assertEquals(Key.of(42).hashCode(), Key.parse(Key.of(42).toString()).hashCode());
        assertNotEquals(Key.of(42), Key.of(43));
        assertTrue(Key.of(42).compareTo(Key.of(43)) < 0);
    }

    @Test
    public void keyBytesAreMostSignificantFirst() {
        byte[] bytes = Key.of(0x0102030405060708L).toBytes();
        assertArrayEquals(new byte[] {0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8}, bytes);
        assertEquals(0x0102030405060708L, Key.ofBytes(bytes).toLong());
        assertTrue(Key.of(7).isIntegral());

        Key wide = Key.parse("ffffffff-0000-0000-0000-000000000001");
        assertFalse(wide.isIntegral());
        assertEquals(Status.MALFORMED_INPUT, assertThrows(IndexException.class, wide::toLong).status());
    }

    @Test
    public void keyRejectsMalformedText() {
        for (String malformed : new String[] {"", "42", "not-a-key", "00000000-0000-0000-0000-0000000000zz",
                                              "00000000-0000-0000-0000-0000000000000"}) {
            assertEquals(Status.MALFORMED_INPUT,
                    assertThrows(IndexException.class, () -> Key.parse(malformed)).status());
        }
    }

    @Test
    public void optionsRoundTripThroughEveryResidencyArm() {
        DenseOptions ram = new DenseOptions()
                .dimensions(96)
                .metric(DenseMetric.JACCARD)
                .quantization(ScalarKind.U1X8)
                .connectivity(24)
                .dimensionPadding(DenseOptions.DimensionPadding.SIMD_QUANTUM)
                .residency(Residency.ram(7, Residency.HugePages.REQUIRED));
        DenseOptions reloaded = ram.reload();
        assertEquals(96, reloaded.dimensions());
        assertEquals(DenseMetric.JACCARD, reloaded.metric());
        assertEquals(ScalarKind.U1X8, reloaded.quantization());
        assertEquals(24, reloaded.connectivity());
        assertEquals(DenseOptions.DimensionPadding.SIMD_QUANTUM, reloaded.dimensionPadding());
        assertEquals(Residency.ram(7, Residency.HugePages.REQUIRED), reloaded.residency());

        Residency vram = Residency.vram(3, Residency.Addressing.UNIFIED);
        assertEquals(vram, twoDimensional().residency(vram).reload().residency());

        Residency nvme = Residency.nvme(512, 128, Residency.IoMode.PAGE_CACHED);
        assertEquals(nvme, twoDimensional().residency(nvme).reload().residency());

        // The arms overlay the same words, so a field of one must never surface through another.
        assertEquals(0, Residency.vram(3, Residency.Addressing.UNIFIED).threads());
        assertEquals(Residency.IoMode.DIRECT, Residency.ram(7, Residency.HugePages.BASE).ioMode());
    }

    @Test
    public void optionsRejectATagThisBuildCannotName() {
        long[] words = twoDimensional().toWords();
        words[OptionsWords.RESIDENCY] = 99;
        assertEquals(Status.UNKNOWN_KIND,
                assertThrows(IndexException.class, () -> Residency.readFrom(words)).status());

        long[] metric = twoDimensional().toWords();
        metric[OptionsWords.METRIC] = 77;
        assertEquals(Status.UNKNOWN_KIND,
                assertThrows(IndexException.class, () -> DenseOptions.fromWords(metric)).status());

        long[] scalar = twoDimensional().toWords();
        scalar[OptionsWords.QUANTIZATION] = 77;
        assertEquals(Status.UNKNOWN_KIND,
                assertThrows(IndexException.class, () -> DenseOptions.fromWords(scalar)).status());
    }

    @Test
    public void extentsRefuseToWrap() {
        assertEquals(Status.OVERFLOW,
                assertThrows(IndexException.class, () -> Extents.product(Long.MAX_VALUE / 3, 4, "the extent"))
                        .status());
        assertEquals(Status.OVERFLOW,
                assertThrows(IndexException.class, () -> Extents.toArrayLength(1L << 40, "the extent")).status());
        assertEquals(Status.SHAPE_MISMATCH,
                assertThrows(IndexException.class, () -> Extents.product(-1, 4, "the extent")).status());

        // A million queries times a million matches wraps a 32-bit length four hundred times over,
        // and the guard is what turns that into a refusal instead of a short slab.
        assertEquals(Status.OVERFLOW,
                assertThrows(IndexException.class, () -> new DenseMatches(1_000_000, 1_000_000)).status());
    }

    @Test
    public void matricesRefuseLayoutsTheEngineCannotTraverse() {
        assertEquals(Status.SHAPE_MISMATCH,
                assertThrows(IndexException.class, () -> DenseMatrix.of(new float[] {1, 2, 3}, 2)).status());
        assertEquals(Status.UNSUPPORTED_LAYOUT,
                assertThrows(IndexException.class,
                        () -> DenseMatrix.strided(new float[] {1, 2, 3, 4}, 2, 2, 4, ScalarKind.F32))
                        .status());
        assertEquals(Status.SHAPE_MISMATCH,
                assertThrows(IndexException.class,
                        () -> DenseMatrix.strided(new float[] {1, 2}, 4, 2, 8, ScalarKind.F32))
                        .status());
        assertEquals(Status.UNSUPPORTED_LAYOUT,
                assertThrows(IndexException.class, () -> DenseMatrix.of(new short[] {1, 2}, 2, ScalarKind.F64))
                        .status());
        assertEquals(Status.UNSUPPORTED_LAYOUT,
                assertThrows(IndexException.class,
                        () -> DenseMatrix.strided(ByteBuffer.allocate(64), 2, 2, 8, ScalarKind.F32))
                        .status());

        // Sub-byte kinds count logical dimensions, so sixteen bits fill two bytes of one row.
        DenseMatrix packed = DenseMatrix.of(new byte[] {1, 2, 3, 4}, 16, ScalarKind.U1X8);
        assertEquals(2, packed.rows());
        assertEquals(2, packed.stride());
    }

    @Test
    public void strideArithmeticSkipsThePadding() {
        try (DenseIndex index = twoDimensional().open()) {
            index.reserve(2);
            // Four floats per row, of which the trailing two are padding no dimension names.
            float[] padded = {0, 1, 9, 9, 1, 0, 9, 9};
            index.add(new Key[] {Key.of(1), Key.of(2)},
                    DenseMatrix.strided(padded, 2, 2, 16, ScalarKind.F32));

            DenseMatrix retrieved = DenseMatrix.allocate(1, 2, ScalarKind.F32);
            assertEquals(1, index.get(Key.of(2), retrieved));
            assertArrayEquals(new float[] {1, 0}, retrieved.floats(), 0f);
        }
    }

    @Test
    public void searchWritesOnlyWhatTheCountsClaim() {
        try (DenseIndex index = populated()) {
            DenseMatches matches = index.search(DenseMatrix.of(new float[] {0, 1}, 2), 4);
            assertEquals(1, matches.queries());
            assertEquals(4, matches.wanted());
            assertEquals(2, matches.count(0));
            assertFalse(matches.truncated(0));
            assertEquals(Key.of(10), matches.key(0, 0));

            // Slots past the count hold memory the engine never wrote, so reading one is refused.
            assertEquals(Status.SHAPE_MISMATCH,
                    assertThrows(IndexException.class, () -> matches.key(0, 2)).status());
            assertEquals(Status.SHAPE_MISMATCH,
                    assertThrows(IndexException.class, () -> matches.distance(0, 3)).status());
            assertEquals(Status.SHAPE_MISMATCH,
                    assertThrows(IndexException.class, () -> matches.count(1)).status());
        }
    }

    @Test
    public void searchClampsACountThatOverrunsItsRoom() {
        // The C interface reports the true extent, which may exceed the room reserved for it -
        // the one place a count and a readable slot count diverge.
        DenseMatches matches = new DenseMatches(2, 2);
        matches.countSlab()[0] = 9;
        matches.countSlab()[1] = 1;
        assertEquals(2, matches.count(0));
        assertEquals(9, matches.reportedCount(0));
        assertTrue(matches.truncated(0));
        assertFalse(matches.truncated(1));
        assertEquals(Status.SHAPE_MISMATCH,
                assertThrows(IndexException.class, () -> matches.key(1, 1)).status());
    }

    @Test
    public void searchReportsTraversalTelemetry() {
        try (DenseIndex index = populated()) {
            DenseStats stats = new DenseStats();
            DenseMatches matches = new DenseMatches(1, 2);
            index.searchInto(DenseMatrix.of(new float[] {0, 1}, 2), matches, 64, stats);
            assertTrue(stats.visitedMembers() > 0);
            assertTrue(stats.computedDistances() > 0);

            DenseStats sum = new DenseStats();
            sum.accumulate(stats);
            sum.accumulate(stats);
            assertEquals(2 * stats.restarts(), sum.restarts());
        }
    }

    @Test
    public void unfrozenGeometryDefersEveryDerivedExtent() {
        // A zero dimensionality is legal: the geometry arrives with a later checkpoint, and until
        // then no row-byte figure derived from it means anything.
        try (DenseIndex index = new DenseOptions().quantization(ScalarKind.F32).open()) {
            assertEquals(0, index.dimensions());
            assertEquals(Status.UNINITIALIZED,
                    assertThrows(IndexException.class,
                            () -> index.add(new Key[] {Key.of(1)}, DenseMatrix.of(new float[] {0, 1}, 2)))
                            .status());
        }
    }

    @Test
    public void geometryMismatchIsCaughtBeforeTheBoundary() {
        try (DenseIndex index = twoDimensional().open()) {
            index.reserve(1);
            assertEquals(Status.SHAPE_MISMATCH,
                    assertThrows(IndexException.class,
                            () -> index.add(new Key[] {Key.of(1)}, DenseMatrix.of(new float[] {0, 1, 2}, 3)))
                            .status());
            assertEquals(Status.SHAPE_MISMATCH,
                    assertThrows(IndexException.class,
                            () -> index.add(new Key[] {Key.of(1), Key.of(2)},
                                    DenseMatrix.of(new float[] {0, 1}, 2)))
                            .status());
        }
    }

    @Test
    public void aScalarKindTheLibraryRejectsSurfacesAsUnknownKind() {
        try (DenseIndex index = twoDimensional().open()) {
            index.reserve(1);
            assertEquals(Status.UNKNOWN_KIND,
                    assertThrows(IndexException.class,
                            () -> index.add(new Key[] {Key.of(1)},
                                    DenseMatrix.of(new byte[] {1, 2}, 2, ScalarKind.E5M2)))
                            .status());
        }
    }

    @Test
    public void getReportsTheTrueExtentEvenWhenItTruncates() {
        try (DenseIndex index = twoDimensional().open()) {
            index.reserve(3);
            // Keys need not be unique, so three entries share one and the count is the extent.
            index.add(new Key[] {Key.of(5), Key.of(5), Key.of(5)},
                    DenseMatrix.of(new float[] {0, 1, 1, 0, 1, 1}, 2));
            assertEquals(3, index.count(Key.of(5)));
            assertTrue(index.contains(Key.of(5)));
            assertFalse(index.contains(Key.of(6)));

            DenseMatrix narrow = DenseMatrix.allocate(1, 2, ScalarKind.F32);
            assertEquals(3, index.get(Key.of(5), narrow));
            assertArrayEquals(new float[] {0, 1}, narrow.floats(), 0f);
        }
    }

    @Test
    public void aClosedHandleRefusesEveryVerb() {
        DenseIndex index = populated();
        index.close();
        index.close();
        assertEquals(Status.UNINITIALIZED, assertThrows(IndexException.class, index::size).status());
        assertEquals(Status.UNINITIALIZED, assertThrows(IndexException.class, () -> index.reserve(1)).status());
        assertEquals(Status.UNINITIALIZED,
                assertThrows(IndexException.class, () -> index.count(Key.of(1))).status());
    }

    @Test
    public void mergeConsumesTheDonor() {
        try (DenseIndex host = populated()) {
            DenseIndex donor = twoDimensional().open();
            donor.reserve(1);
            donor.add(Key.of(30), DenseMatrix.of(new float[] {1, 1}, 2));

            host.merge(donor);
            assertEquals(3, host.size());

            // The native side owns the donor now, so its handle is empty and its close is a no-op.
            assertEquals(Status.UNINITIALIZED, assertThrows(IndexException.class, donor::size).status());
            donor.close();
            assertEquals(Status.MALFORMED_INPUT,
                    assertThrows(IndexException.class, () -> host.merge(host)).status());
        }
    }

    @Test
    public void checkpointsRoundTripThroughAPathAndABuffer() throws Exception {
        File checkpoint = File.createTempFile("usearch", ".safegraphs");
        checkpoint.deleteOnExit();
        byte[] slab;
        try (DenseIndex index = populated()) {
            index.save(checkpoint.getAbsolutePath());
            slab = new byte[Extents.toArrayLength(index.serializedLength(), "the checkpoint")];
            index.saveBuffer(slab);
        }

        DenseOptions described = DenseIndex.metadata(checkpoint.getAbsolutePath());
        assertEquals(2, described.dimensions());
        assertEquals(DenseMetric.EUCLIDEAN, described.metric());
        assertEquals(2, DenseIndex.metadata(slab).dimensions());

        try (DenseIndex reopened = twoDimensional().open()) {
            reopened.load(checkpoint.getAbsolutePath());
            assertEquals(2, reopened.size());
            assertEquals(Key.of(10), reopened.search(DenseMatrix.of(new float[] {0, 1}, 2), 1).key(0, 0));
        }

        ByteBuffer direct = ByteBuffer.allocateDirect(slab.length).order(ByteOrder.nativeOrder());
        direct.put(slab).flip();
        try (DenseIndex reopened = twoDimensional().open()) {
            reopened.loadBuffer(direct);
            assertEquals(2, reopened.size());
        }
    }

    @Test
    public void directBuffersCrossWithoutACopy() {
        try (DenseIndex index = twoDimensional().open()) {
            index.reserve(2);
            ByteBuffer rows = ByteBuffer.allocateDirect(4 * Float.BYTES).order(ByteOrder.nativeOrder());
            rows.asFloatBuffer().put(new float[] {0, 1, 1, 0});
            index.add(new Key[] {Key.of(1), Key.of(2)},
                    DenseMatrix.strided(rows, 2, 2, 2L * Float.BYTES, ScalarKind.F32));
            assertEquals(2, index.size());

            ByteBuffer queries = ByteBuffer.allocateDirect(2 * Float.BYTES).order(ByteOrder.nativeOrder());
            queries.asFloatBuffer().put(new float[] {1, 0});
            DenseMatches matches = index.search(
                    DenseMatrix.strided(queries, 1, 2, 2L * Float.BYTES, ScalarKind.F32), 1);
            assertEquals(Key.of(2), matches.key(0, 0));
        }
    }

    @Test
    public void theLinkedLibraryNamesItself() {
        assertFalse(Usearch.version().isEmpty());
        assertFalse(Usearch.capabilitiesCompiled().isEmpty());
        assertFalse(Usearch.capabilitiesAvailable().isEmpty());
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
