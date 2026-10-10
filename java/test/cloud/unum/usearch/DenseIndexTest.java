package cloud.unum.usearch;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.File;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicReference;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Test;

import cloud.unum.usearch.DenseIndex.DenseAttendConfig;
import cloud.unum.usearch.DenseIndex.DenseCursor;
import cloud.unum.usearch.DenseIndex.DenseIndexDescription;
import cloud.unum.usearch.DenseIndex.DenseInsertConfig;
import cloud.unum.usearch.DenseIndex.DenseMetadata;
import cloud.unum.usearch.DenseIndex.DenseMetric;
import cloud.unum.usearch.DenseIndex.DenseOptions;
import cloud.unum.usearch.DenseIndex.DenseReachability;
import cloud.unum.usearch.DenseIndex.DenseRefineConfig;
import cloud.unum.usearch.DenseIndex.DenseSearchConfig;
import cloud.unum.usearch.DenseIndex.DenseSearchStream;
import cloud.unum.usearch.DenseIndex.DenseSoftmaxState;
import cloud.unum.usearch.DenseIndex.DenseTraversalStats;
import cloud.unum.usearch.DenseIndex.ValueLookup;
import cloud.unum.usearch.Types.Arena;
import cloud.unum.usearch.Types.Deferred;
import cloud.unum.usearch.Types.Executor;
import cloud.unum.usearch.Types.Extents;
import cloud.unum.usearch.Types.Key;
import cloud.unum.usearch.Types.Matches;
import cloud.unum.usearch.Types.Monolith;
import cloud.unum.usearch.Types.Mood;
import cloud.unum.usearch.Types.RowsView;
import cloud.unum.usearch.Types.Scalar;
import cloud.unum.usearch.Types.Status;
import cloud.unum.usearch.Types.USearchException;

/**
 * The boundary this SDK owns: how a value is marshalled, how an extent is checked, how a call
 * completes, and how a handle dies. That vector search returns near neighbors is the engine's
 * contract and is tested where the engine lives, so nothing here asserts recall.
 *
 * <p>
 * The tests live in the binding's own package so they can reach the extent arithmetic, which no
 * caller ever sees and which nothing else can exercise.
 */
public class DenseIndexTest {

    /** One pool most verbs here run on; the ones called on the index itself run on the calling thread. */
    private static final Executor POOL = Executor.forRam(new int[0], 0, 0);

    @AfterAll
    public static void releasePool() {
        POOL.close();
    }

    private static DenseOptions twoDimensional() {
        return new DenseOptions().dimensions(2).metric(DenseMetric.EUCLIDEAN).quantization(Scalar.F32);
    }

    private static DenseIndex empty(DenseOptions options) {
        return DenseIndex.initRam(options, new int[0]);
    }

    private static float[] floatsOf(DenseIndex.Vectors vectors) {
        return vectors.segment().toArray(ValueLayout.JAVA_FLOAT);
    }

    private static DenseIndex populated() {
        DenseIndex index = empty(twoDimensional());
        index.on(POOL).reserve(4);
        index.on(POOL).insert(new Key[] {Key.of(10), Key.of(20)}, RowsView.of(new float[] {0, 1, 1, 0}, 2));
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
                Status.MALFORMED,
                assertThrows(USearchException.class, wide::toLong).status());
    }

    @Test
    public void keyRejectsMalformedText() {
        for (String malformed :
                new String[] {"", "00000000-0000-0000-0000-0000000000zz", "00000000-0000-0000-0000-0000000000000"}) {
            assertEquals(
                    Status.MALFORMED,
                    assertThrows(USearchException.class, () -> Key.parse(malformed))
                            .status());
        }
    }

    @Test
    public void nvmeIsServedInPlaceNeverInitialized() {
        assertEquals(
                Status.UNSUPPORTED,
                assertThrows(USearchException.class, () -> DenseIndex.viewNvme("index.safegraphs"))
                        .status());
    }

    @Test
    public void statusesMirrorTheCEnumerators() {
        assertEquals(Status.DEFERRED, Status.of(1));
        assertEquals(Status.CONTENDED, Status.of(14));
        assertEquals(Status.INVALID_ARGUMENT, Status.of(17));
        // C spells a status it cannot name as invalid-argument, and so does this side.
        assertEquals(Status.INVALID_ARGUMENT, Status.of(99));
        // Every spelling comes from the C name tables.
        assertEquals("invalid-argument", Status.INVALID_ARGUMENT.toString());
        assertEquals("deferred", Status.DEFERRED.toString());
        assertEquals("u1x8", Scalar.U1X8.toString());
        assertEquals("spinning", Mood.SPINNING.toString());
        assertEquals("jaccard", DenseMetric.JACCARD.toString());
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

        // The queries-by-wanted product feeds three slab lengths, so it is checked before they are sized.
        assertEquals(
                Status.OVERFLOW,
                assertThrows(USearchException.class, () -> Matches.allocate(1L << 40, 1L << 40))
                        .status());
    }

    @Test
    public void matchesReadTheirShapeOffTheSlabs() {
        Matches adopted = Matches.of(
                MemorySegment.ofArray(new byte[6 * Key.BYTES]),
                MemorySegment.ofArray(new double[6]),
                MemorySegment.ofArray(new long[2]));
        assertEquals(2, adopted.queries());
        assertEquals(3, adopted.wanted());

        assertEquals(
                Status.SHAPE_MISMATCH,
                assertThrows(
                                USearchException.class,
                                () -> Matches.of(
                                        MemorySegment.ofArray(new byte[6 * Key.BYTES]),
                                        MemorySegment.ofArray(new double[6]),
                                        MemorySegment.ofArray(new long[0])))
                        .status());
        assertEquals(
                Status.SHAPE_MISMATCH,
                assertThrows(
                                USearchException.class,
                                () -> Matches.of(
                                        MemorySegment.ofArray(new byte[6 * Key.BYTES]),
                                        MemorySegment.ofArray(new double[5]),
                                        MemorySegment.ofArray(new long[2])))
                        .status());
        assertEquals(
                Status.SHAPE_MISMATCH,
                assertThrows(
                                USearchException.class,
                                () -> Matches.of(
                                        MemorySegment.ofArray(new byte[5 * Key.BYTES]),
                                        MemorySegment.ofArray(new double[5]),
                                        MemorySegment.ofArray(new long[2])))
                        .status());
    }

    /** Heap slabs are staged for the call and copied back, so a caller's arrays see the matches. */
    @Test
    public void heapMatchesReceiveTheRun() {
        try (DenseIndex index = populated()) {
            long[] counts = new long[1];
            byte[] keys = new byte[2 * Key.BYTES];
            Matches matches = Matches.of(
                    MemorySegment.ofArray(keys), MemorySegment.ofArray(new double[2]), MemorySegment.ofArray(counts));
            assertSame(matches, index.search(RowsView.of(new float[] {0, 1}, 2), new DenseSearchConfig(matches)));
            assertEquals(2, counts[0]);
            assertEquals(Key.of(10), Key.ofBytes(keys));
        }
    }

    @Test
    public void perCallConfigsRefuseANegativeBreadth() {
        assertEquals(
                Status.INVALID_ARGUMENT,
                assertThrows(USearchException.class, () -> new DenseInsertConfig().expansionFactor(-1))
                        .status());
        assertEquals(
                Status.INVALID_ARGUMENT,
                assertThrows(USearchException.class, () -> new DenseSearchConfig(1).expansionFactor(-1))
                        .status());
    }

    @Test
    public void rowsViewsRefuseLayoutsTheEngineCannotTraverse() {
        assertEquals(
                Status.SHAPE_MISMATCH,
                assertThrows(USearchException.class, () -> RowsView.of(new float[] {1, 2, 3}, 2))
                        .status());
        assertEquals(
                Status.UNSUPPORTED_LAYOUT,
                assertThrows(
                                USearchException.class,
                                () -> RowsView.strided(
                                        MemorySegment.ofArray(new float[] {1, 2, 3, 4}), 2, 2, 4, Scalar.F32))
                        .status());
        assertEquals(
                Status.SHAPE_MISMATCH,
                assertThrows(
                                USearchException.class,
                                () -> RowsView.strided(MemorySegment.ofArray(new float[] {1, 2}), 4, 2, 8, Scalar.F32))
                        .status());
        assertEquals(
                Status.UNKNOWN_KIND,
                assertThrows(USearchException.class, () -> RowsView.of(new byte[] {1, 2}, 2, Scalar.UNKNOWN))
                        .status());

        // Sub-byte kinds count logical dimensions, so sixteen bits fill two bytes of one row.
        RowsView packed = RowsView.of(new byte[] {1, 2, 3, 4}, 16, Scalar.U1X8);
        assertEquals(2, packed.rows());
        assertEquals(2, packed.strideBytes());
        assertEquals(16, packed.columns());
    }

    @Test
    public void strideArithmeticSkipsThePadding() {
        try (DenseIndex index = empty(twoDimensional())) {
            index.reserve(2);
            // Four floats per row, of which the trailing two are padding no dimension names.
            float[] padded = {0, 1, 9, 9, 1, 0, 9, 9};
            index.insert(
                    new Key[] {Key.of(1), Key.of(2)},
                    RowsView.strided(MemorySegment.ofArray(padded), 2, 2, 16, Scalar.F32));

            try (DenseIndex.Vectors retrieved = index.get(Key.of(2), Scalar.F32)) {
                assertArrayEquals(new float[] {1, 0}, floatsOf(retrieved), 0f);
                // The width comes from what C reported, never from a describe call.
                assertEquals(2, retrieved.columns());
            }
        }
    }

    @Test
    public void searchWritesOnlyWhatTheCountsClaim() {
        try (DenseIndex index = populated()) {
            Matches matches = index.on(POOL).search(RowsView.of(new float[] {0, 1}, 2), 4);
            assertEquals(1, matches.queries());
            assertEquals(4, matches.wanted());
            assertEquals(2, matches.count(0));
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
    public void everyWalkReportsTraversalTelemetry() {
        try (DenseIndex index = populated()) {
            DenseTraversalStats searched = new DenseTraversalStats();
            index.on(POOL)
                    .search(
                            RowsView.of(new float[] {0, 1}, 2),
                            new DenseSearchConfig(2).expansionFactor(64).stats(searched));
            assertTrue(searched.visitedMembers() > 0);
            assertTrue(searched.computedDistances() > 0);

            DenseTraversalStats refined = new DenseTraversalStats();
            index.on(POOL).refine(new DenseRefineConfig().expansionFactor(32).stats(refined));
            assertTrue(refined.visitedMembers() > 0);
        }
    }

    @Test
    public void healthAuditsTheGraph() {
        try (DenseIndex index = populated()) {
            DenseReachability audit = index.on(POOL).health();
            assertEquals(2, audit.audited());
            assertTrue(audit.reachable() <= audit.audited());
        }
    }

    @Test
    public void lateInteractionDividesTokenRowsIntoQueries() {
        try (DenseIndex index = empty(new DenseOptions()
                .dimensions(2)
                .metric(DenseMetric.INNER)
                .quantization(Scalar.F32)
                .valuesPerKey(2))) {
            index.on(POOL).reserve(4);
            // Two documents of two token rows apiece, so a key names a document and not a row.
            index.on(POOL)
                    .insert(
                            new Key[] {Key.of(10), Key.of(10), Key.of(20), Key.of(20)},
                            RowsView.of(new float[] {1, 0, 0, 1, 2, 0, 0, 2}, 2));

            Matches matches = index.on(POOL).maxsim(RowsView.of(new float[] {1, 0, 0, 1}, 2), 2, 2);
            assertEquals(1, matches.queries());
            assertEquals(2, matches.count(0));

            // The rows of a batch must divide into whole queries, and the check runs before the call.
            assertEquals(
                    Status.SHAPE_MISMATCH,
                    assertThrows(
                                    USearchException.class,
                                    () -> index.on(POOL).maxsim(RowsView.of(new float[] {1, 0, 0, 1}, 2), 3, 1))
                            .status());
        }
    }

    @Test
    public void lateInteractionRefusesADistanceKind() {
        // No Java-side check produces CONFIG_MISMATCH, so this is the one failure that can only have
        // come from the boundary's exception factory, and its detail from the engine's own string.
        try (DenseIndex index = populated()) {
            USearchException refused = assertThrows(
                    USearchException.class, () -> index.on(POOL).maxsim(RowsView.of(new float[] {0, 1}, 2), 1, 1));
            assertEquals(Status.CONFIG_MISMATCH, refused.status());
            assertTrue(refused.getMessage().startsWith("usearch: config-mismatch: "));
            assertFalse(refused.getMessage().endsWith(": "));
        }
    }

    @Test
    public void unfrozenGeometryRefusesRows() {
        try (DenseIndex index = empty(new DenseOptions().quantization(Scalar.F32))) {
            assertEquals(0, index.describe().metadata().dimensions());
            assertEquals(
                    Status.UNINITIALIZED,
                    assertThrows(
                                    USearchException.class,
                                    () -> index.insert(new Key[] {Key.of(1)}, RowsView.of(new float[] {0, 1}, 2)))
                            .status());
        }
    }

    @Test
    public void aColumnsMismatchSurfacesFromTheEngine() {
        try (DenseIndex index = empty(twoDimensional())) {
            index.reserve(1);
            // The view carries its own width, so C refuses it and its message comes through.
            USearchException refused = assertThrows(
                    USearchException.class,
                    () -> index.insert(new Key[] {Key.of(1)}, RowsView.of(new float[] {0, 1, 2}, 3)));
            assertEquals(Status.SHAPE_MISMATCH, refused.status());
            assertFalse(refused.getMessage().endsWith(": "));
            // C counts one axis, so only this side sees two keys over one row.
            assertEquals(
                    Status.SHAPE_MISMATCH,
                    assertThrows(
                                    USearchException.class,
                                    () -> index.insert(
                                            new Key[] {Key.of(1), Key.of(2)}, RowsView.of(new float[] {0, 1}, 2)))
                            .status());
        }
    }

    @Test
    public void getWithoutAnArenaOwnsItsRows() {
        try (DenseIndex index = empty(twoDimensional())) {
            index.on(POOL).reserve(3);
            // Keys need not be unique, so three entries share one and the call alone sizes the export.
            index.on(POOL)
                    .insert(
                            new Key[] {Key.of(5), Key.of(5), Key.of(5)},
                            RowsView.of(new float[] {0, 1, 1, 0, 1, 1}, 2));
            assertEquals(3, index.on(POOL).count(Key.of(5)));
            assertFalse(index.contains(Key.of(6)));

            DenseIndex.Vectors every = index.get(Key.of(5), Scalar.F64);
            DenseIndex.Vectors absent = index.on(POOL).get(Key.of(6), Scalar.F32);
            // Each result owns its arena, so a later call cannot invalidate an earlier one.
            assertEquals(3, every.rows());
            assertEquals(2 * Double.BYTES, every.strideBytes());
            assertArrayEquals(new double[] {0, 1, 1, 0, 1, 1}, every.segment().toArray(ValueLayout.JAVA_DOUBLE), 0);
            assertEquals(0, absent.rows());
            MemorySegment rows = every.segment();
            every.close();
            absent.close();
            assertThrows(IllegalStateException.class, () -> rows.get(ValueLayout.JAVA_DOUBLE, 0));
        }
    }

    @Test
    public void aSharedArenaInvalidatesThePreviousResult() {
        try (DenseIndex index = populated();
                Arena arena = new Arena()) {
            DenseIndex.Vectors first = index.on(POOL).get(Key.of(10), Scalar.F32, arena);
            assertArrayEquals(new float[] {0, 1}, floatsOf(first), 0f);
            DenseIndex.Vectors second = index.on(POOL).get(Key.of(20), Scalar.F32, arena);
            assertArrayEquals(new float[] {1, 0}, floatsOf(second), 0f);
            // The arena was rewound for the second call, so the first view refuses to read freed rows.
            assertThrows(IllegalStateException.class, () -> floatsOf(first));
        }
    }

    @Test
    public void multiplyLandsRaggedProducts() {
        try (DenseIndex index = empty(new DenseOptions()
                        .dimensions(2)
                        .metric(DenseMetric.INNER)
                        .quantization(Scalar.F32));
                Arena arena = new Arena()) {
            index.on(POOL).insert(new Key[] {Key.of(10), Key.of(20)}, RowsView.of(new float[] {1, 0, 0, 2}, 2));
            try (DenseIndex.Products products = index.on(POOL)
                    .multiply(
                            RowsView.of(new float[] {1, 1, 0, 1}, 2),
                            new DenseIndex.DenseMultiplyConfig().arena(arena))) {
                assertEquals(2, products.queries());
                assertEquals(2, products.count(0));
                assertTrue(products.required() >= products.count(0) + products.count(1));
                double total = 0;
                for (int rank = 0; rank != products.count(0); ++rank) {
                    total += products.score(0, rank);
                }
                assertEquals(3, total, 1e-6);
                assertEquals(
                        Status.SHAPE_MISMATCH,
                        assertThrows(USearchException.class, () -> products.key(0, 2))
                                .status());
            }
        }
    }

    /** The value lookup is told the scalar and width it must answer in, and a key it misses refuses the call. */
    @Test
    public void attentionResolvesValuesThroughTheLookup() {
        try (DenseIndex index =
                empty(new DenseOptions().dimensions(2).metric(DenseMetric.INNER).quantization(Scalar.F32))) {
            index.insert(new Key[] {Key.of(0), Key.of(1)}, RowsView.of(new float[] {1, 0, 0, 1}, 2));
            RowsView values = RowsView.of(new float[] {5, 5, 7, 7}, 2);
            ValueLookup byKey = new ValueLookup(values, Key::toLong);
            RowsView outputs = RowsView.allocate(1, 2, Scalar.F32);
            DenseSoftmaxState[] carry = new DenseSoftmaxState[1];
            index.on(POOL)
                    .attend(RowsView.of(new float[] {1, 0}, 2), byKey, new DenseAttendConfig(outputs).carry(carry));
            float[] attended = outputs.segment().toArray(ValueLayout.JAVA_FLOAT);
            // A convex mix of the two value rows, weighted toward the first key.
            assertTrue(attended[0] > 5 && attended[0] < 6);
            assertTrue(carry[0].partition() > 1);

            // Values of another scalar than the index stores resolve to nothing, so the call is refused.
            ValueLookup mistyped = new ValueLookup(RowsView.of(new double[] {5, 5, 7, 7}, 2), Key::toLong);
            assertEquals(
                    Status.SHAPE_MISMATCH,
                    assertThrows(
                                    USearchException.class,
                                    () -> index.attend(RowsView.of(new float[] {1, 0}, 2), mistyped, outputs))
                            .status());

            // An exception the lookup raises outranks the status it turned into.
            ValueLookup throwing = new ValueLookup(values, key -> {
                throw new ArithmeticException("no value for " + key);
            });
            assertThrows(
                    ArithmeticException.class,
                    () -> index.attend(RowsView.of(new float[] {1, 0}, 2), throwing, outputs));
        }
    }

    /** A call leases its index, so closing the index from inside its own call is refused, not a use-after-free. */
    @Test
    public void aCallLeasesItsIndex() {
        DenseIndex index =
                empty(new DenseOptions().dimensions(2).metric(DenseMetric.INNER).quantization(Scalar.F32));
        index.insert(new Key[] {Key.of(0)}, RowsView.of(new float[] {1, 0}, 2));
        AtomicReference<RuntimeException> refused = new AtomicReference<>();
        ValueLookup closing = new ValueLookup(RowsView.of(new float[] {5, 5}, 2), key -> {
            try {
                index.close();
            } catch (IllegalStateException leased) {
                refused.set(leased);
            }
            return 0;
        });
        index.attend(RowsView.of(new float[] {1, 0}, 2), closing, RowsView.allocate(1, 2, Scalar.F32));
        assertInstanceOf(IllegalStateException.class, refused.get());
        assertEquals(1, index.describe().size());
        index.close();
    }

    @Test
    public void aClosedHandleRefusesEveryVerb() {
        DenseIndex index = populated();
        index.close();
        index.close();
        assertThrows(IllegalStateException.class, index::describe);
        assertThrows(IllegalStateException.class, () -> index.on(POOL).reserve(1));
    }

    @Test
    public void checkpointsRoundTripThroughAPathAndABuffer() throws Exception {
        File checkpoint = File.createTempFile("usearch", ".safegraphs");
        checkpoint.deleteOnExit();
        Monolith monolith;
        try (DenseIndex index = populated()) {
            index.on(POOL).save(checkpoint.getAbsolutePath());
            monolith = index.on(POOL).saveBuffer();
        }

        DenseMetadata described = DenseIndex.metadataRead(checkpoint.getAbsolutePath(), POOL);
        assertEquals(2, described.dimensions());
        assertEquals(DenseMetric.EUCLIDEAN, described.metric());
        assertEquals(Scalar.F32, described.quantization());
        byte[] slab = monolith.segment().toArray(ValueLayout.JAVA_BYTE);
        assertEquals(
                2, DenseIndex.metadataReadBuffer(MemorySegment.ofArray(slab)).dimensions());

        try (DenseIndex reopened = DenseIndex.loadRam(checkpoint.getAbsolutePath(), POOL, new int[0], 0)) {
            assertEquals(2, reopened.describe().size());
            assertEquals(2, reopened.describe().metadata().dimensions());
        }

        // The monolith outlives the index that wrote it, since it owns its arena.
        try (monolith;
                DenseIndex reopened = DenseIndex.loadRamBuffer(monolith.segment(), POOL, new int[0], 0)) {
            assertEquals(2, reopened.describe().size());
        }

        try (DenseIndex viewed = DenseIndex.viewRam(checkpoint.getAbsolutePath())) {
            assertEquals(2, viewed.count(Key.of(10)) + viewed.count(Key.of(20)));
            assertEquals(
                    Status.READ_ONLY,
                    assertThrows(
                                    USearchException.class,
                                    () -> viewed.insert(new Key[] {Key.of(1)}, RowsView.of(new float[] {0, 1}, 2)))
                            .status());
        }
    }

    /**
     * A heap array is staged into native memory for the duration of one call, so the engine reads
     * the values the array held when the call began. Mutating the array afterwards cannot reach
     * what was stored - which is what distinguishes it from the native segment below, where the
     * engine addresses the caller's own memory in place.
     */
    @Test
    public void heapArraysAreCopiedRatherThanAliased() {
        try (DenseIndex index = empty(twoDimensional())) {
            index.reserve(1);
            float[] row = new float[] {0, 1};
            index.insert(new Key[] {Key.of(1)}, RowsView.of(row, 2));

            // Overwriting the source after the batch must leave the stored row untouched.
            row[0] = 9;
            row[1] = 9;

            try (DenseIndex.Vectors exported = index.get(Key.of(1), Scalar.F32)) {
                assertArrayEquals(new float[] {0, 1}, floatsOf(exported), 0);
            }
        }
    }

    @Test
    public void directBuffersCrossWithoutACopy() {
        try (DenseIndex index = empty(twoDimensional())) {
            index.reserve(2);
            ByteBuffer rows = ByteBuffer.allocateDirect(4 * Float.BYTES).order(ByteOrder.nativeOrder());
            rows.asFloatBuffer().put(new float[] {0, 1, 1, 0});
            index.on(POOL)
                    .insert(new Key[] {Key.of(1), Key.of(2)}, RowsView.of(MemorySegment.ofBuffer(rows), 2, Scalar.F32));
            assertEquals(2, index.describe().size());

            // A positioned buffer is captured from its position, where the first query starts.
            ByteBuffer queries = ByteBuffer.allocateDirect(4 * Float.BYTES).order(ByteOrder.nativeOrder());
            queries.asFloatBuffer().put(new float[] {0, 1, 1, 0});
            queries.position(2 * Float.BYTES);
            RowsView query = RowsView.strided(MemorySegment.ofBuffer(queries), 1, 2, 2L * Float.BYTES, Scalar.F32);
            // Moving the buffer afterwards cannot widen what the matrix captured.
            queries.position(0);
            assertEquals(Key.of(2), index.on(POOL).search(query, 1).key(0, 0));
        }
    }

    /** The stand-in holds a deferred call until the executor drains, so the status is observable before the join. */
    @Test
    public void deferredCallsCompleteOnTheirExecutor() {
        try (Executor pool = Executor.forRam(new int[0], 2, 4);
                DenseIndex index = empty(twoDimensional())) {
            DenseIndex.DeferredOn deferred = index.deferred(pool);
            try (Deferred<Void> inserted =
                    deferred.insert(new Key[] {Key.of(10), Key.of(20)}, RowsView.of(new float[] {0, 1, 1, 0}, 2))) {
                Deferred<Matches> searched = deferred.search(RowsView.of(new float[] {1, 0}, 2), 1);
                assertEquals(Status.DEFERRED, inserted.status());
                assertEquals(Status.DEFERRED, searched.status());
                pool.drain(Mood.PARKED);
                assertEquals(Status.SUCCESS, searched.status());
                Matches joined = searched.join();
                assertEquals(Key.of(20), joined.key(0, 0));
                // The first join freed the outcome, so later ones repeat what it kept, until closed.
                assertSame(joined, searched.join());
                assertEquals(Status.SUCCESS, searched.status());
                searched.close();
                assertThrows(IllegalStateException.class, searched::join);
                assertThrows(IllegalStateException.class, searched::status);
            }

            // A failure travels to every join, and a call closed unjoined waits rather than leaking.
            Deferred<Long> counted = deferred.count(Key.of(20));
            Deferred<Matches> refused = deferred.maxsim(RowsView.of(new float[] {1, 0}, 2), 1, 1);
            pool.drain(Mood.SPINNING);
            USearchException reported = assertThrows(USearchException.class, refused::join);
            assertEquals(Status.CONFIG_MISMATCH, reported.status());
            assertSame(reported, assertThrows(USearchException.class, refused::join));
            assertEquals(Status.CONFIG_MISMATCH, refused.status());
            counted.close();
            assertEquals(2, index.on(pool).describe().size());
        }
    }

    /** A deferred call holds what it reads, so a fresh arena outlives the submission and lands with the join. */
    @Test
    public void deferredCallsOwnTheirArena() {
        try (DenseIndex index = populated()) {
            Deferred<DenseIndex.Vectors> fetched = index.deferred(POOL).get(Key.of(20), Scalar.F32);
            System.gc();
            POOL.drain(Mood.PARKED);
            try (DenseIndex.Vectors rows = fetched.join()) {
                assertArrayEquals(new float[] {1, 0}, floatsOf(rows), 0f);
            }
        }
    }

    /** A deferred call's rows are read through the scope its own rewind opened, which a later call closes. */
    @Test
    public void aLateJoinCannotReadARewoundArena() {
        try (DenseIndex index = populated();
                Arena arena = new Arena()) {
            Deferred<DenseIndex.Vectors> early = index.deferred(POOL).get(Key.of(10), Scalar.F32, arena);
            POOL.drain(Mood.PARKED);
            DenseIndex.Vectors later = index.on(POOL).get(Key.of(20), Scalar.F32, arena);
            assertThrows(IllegalStateException.class, early::join);
            assertArrayEquals(new float[] {1, 0}, floatsOf(later), 0f);
        }
    }

    @Test
    public void cursorsTakeTheirWidthsFromTheCaller() {
        try (DenseIndex index = populated();
                DenseCursor cursor = index.on(POOL).cursor()) {
            Key[] keys = new Key[4];
            float[] rows = new float[8];
            assertEquals(2, cursor.next(keys, RowsView.of(rows, 2)));
            assertEquals(Key.of(10), keys[0]);
            assertArrayEquals(new float[] {0, 1, 1, 0, 0, 0, 0, 0}, rows, 0f);
            assertEquals(0, cursor.next(keys));
        }
    }

    @Test
    public void searchStreamsPullUntilDrained() {
        try (DenseIndex index = populated();
                DenseSearchStream stream = index.on(POOL).searchStream(RowsView.of(new float[] {0, 1}, 2))) {
            assertEquals(Key.of(10), stream.next(1).key(0, 0));
            assertEquals(Key.of(20), stream.next(1).key(0, 0));
            assertEquals(0, stream.next(1).count(0));
        }
    }

    @Test
    public void mergeConsumesItsDonors() {
        try (DenseIndex index = populated()) {
            DenseIndex donor = populated();
            index.on(POOL).merge(donor);
            assertEquals(4, index.describe().size());
            assertThrows(IllegalStateException.class, donor::describe);
            donor.close();
            assertEquals(
                    Status.INVALID_ARGUMENT,
                    assertThrows(USearchException.class, () -> index.merge(index))
                            .status());
        }
    }

    @Test
    public void theCallingThreadNeedsNoExecutor() {
        try (DenseIndex index = empty(twoDimensional())) {
            index.insert(new Key[] {Key.of(10), Key.of(20)}, RowsView.of(new float[] {0, 1, 1, 0}, 2));
            assertEquals(
                    Key.of(20),
                    index.search(RowsView.of(new float[] {1, 0}, 2), 1).key(0, 0));
        }
    }

    @Test
    public void aClosedExecutorRefusesEveryVerb() {
        Executor pool = Executor.forRam(new int[0], 0, 0);
        pool.close();
        pool.close();
        assertThrows(IllegalStateException.class, () -> pool.drain(Mood.PARKED));
        try (DenseIndex index = populated()) {
            assertThrows(IllegalStateException.class, () -> index.on(pool).describe());
        }
        assertEquals(
                Status.INVALID_ARGUMENT,
                assertThrows(USearchException.class, () -> Executor.forRam(new int[0], -1, 0))
                        .status());
        assertEquals(
                Status.SHAPE_MISMATCH,
                assertThrows(USearchException.class, () -> Executor.forVram(new int[] {0}, new MemorySegment[2], 0, 0))
                        .status());
    }

    @Test
    public void theLinkedLibraryNamesItself() {
        assertFalse(Types.version().isEmpty());
        assertFalse(Types.capabilitiesCompiled().isEmpty());
        assertFalse(Types.capabilitiesAvailable().isEmpty());
        try (DenseIndex index = populated()) {
            DenseIndexDescription description = index.on(POOL).describe();
            assertFalse(description.capabilities().isEmpty());
            assertEquals(2, description.size());
            assertTrue(description.capacity() >= description.size());
            assertTrue(description.memoryUsage() > 0);
            assertTrue(description.metadata().connectivity() > 0);
            index.on(POOL).clear();
            assertEquals(0, index.describe().size());
        }
    }
}
