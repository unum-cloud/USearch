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
import java.util.Arrays;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicReference;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Test;

import cloud.unum.usearch.DenseIndex.DenseAttendConfig;
import cloud.unum.usearch.DenseIndex.DenseBatch;
import cloud.unum.usearch.DenseIndex.DenseCursor;
import cloud.unum.usearch.DenseIndex.DenseIndexDescription;
import cloud.unum.usearch.DenseIndex.DenseInsertConfig;
import cloud.unum.usearch.DenseIndex.DenseMetadata;
import cloud.unum.usearch.DenseIndex.DenseMetric;
import cloud.unum.usearch.DenseIndex.DenseOptions;
import cloud.unum.usearch.DenseIndex.DenseProducts;
import cloud.unum.usearch.DenseIndex.DenseReachability;
import cloud.unum.usearch.DenseIndex.DenseRefineConfig;
import cloud.unum.usearch.DenseIndex.DenseSearchConfig;
import cloud.unum.usearch.DenseIndex.DenseSearchStream;
import cloud.unum.usearch.DenseIndex.DenseSoftmaxState;
import cloud.unum.usearch.DenseIndex.DenseTraversalStats;
import cloud.unum.usearch.DenseIndex.DenseVectors;
import cloud.unum.usearch.DenseIndex.ValueLookup;
import cloud.unum.usearch.Types.Arena;
import cloud.unum.usearch.Types.Deferred;
import cloud.unum.usearch.Types.Executor;
import cloud.unum.usearch.Types.Extents;
import cloud.unum.usearch.Types.Key;
import cloud.unum.usearch.Types.Matches;
import cloud.unum.usearch.Types.Monolith;
import cloud.unum.usearch.Types.Mood;
import cloud.unum.usearch.Types.Residency;
import cloud.unum.usearch.Types.RowsView;
import cloud.unum.usearch.Types.Scalar;
import cloud.unum.usearch.Types.Status;
import cloud.unum.usearch.Types.Traversal;
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
    private static final Executor POOL = Executor.forRam(new int[0], null, 0, 0);

    @AfterAll
    public static void releasePool() {
        POOL.close();
    }

    private static DenseOptions twoDimensional() {
        return new DenseOptions().dimensions(2).metric(DenseMetric.EUCLIDEAN).quantization(Scalar.BF16);
    }

    private static DenseIndex empty(DenseOptions options) {
        return DenseIndex.initRam(options, new int[0]);
    }

    private static float[] floatsOf(DenseVectors vectors) {
        return vectors.segment().toArray(ValueLayout.JAVA_FLOAT);
    }

    /** The bf16 bits of values whose low sixteen f32 bits are zero, as every value here is. */
    private static RowsView bf16RowsOf(long columns, float... values) {
        short[] bits = new short[values.length];
        for (int position = 0; position != values.length; ++position) {
            bits[position] = (short) (Float.floatToRawIntBits(values[position]) >>> 16);
        }
        return RowsView.of(bits, columns, Scalar.BF16);
    }

    private static DenseIndex populated() {
        DenseIndex index = empty(twoDimensional());
        index.on(POOL).reserve(4);
        index.on(POOL).insert(new Key[] {Key.of(10), Key.of(20)}, RowsView.of(new float[] {0, 1, 1, 0}, 2));
        return index;
    }

    @Test
    public void layoutsMatchTheirStructs() {
        long word = ValueLayout.ADDRESS.byteSize();
        assertEquals(2 * word, Library.offset(Library.RAM_OPTIONS, "shard_capacity"));
        assertEquals(3 * word, Library.offset(Library.RAM_OPTIONS, "cpu_class"));
        assertEquals(4 * word, Library.RAM_OPTIONS.byteSize());
        assertEquals(3 * word, Library.offset(Library.VRAM_OPTIONS, "gpu_class"));
        assertEquals(4 * word, Library.VRAM_OPTIONS.byteSize());
        assertEquals(2 * word, Library.offset(Library.NVME_OPTIONS, "cpu_class"));
        assertEquals(3 * word, Library.NVME_OPTIONS.byteSize());
        assertEquals(2 * word, Library.offset(Library.TRAVERSAL_STATS, "restarts"));
        assertEquals(3 * word, Library.TRAVERSAL_STATS.byteSize());
        long cpuClass = Library.offset(DenseIndex.DESCRIPTION, "cpu_class");
        assertEquals(DenseIndex.METADATA.byteSize() + 3 * word, cpuClass);
        assertEquals(cpuClass + 4, Library.offset(DenseIndex.DESCRIPTION, "gpu_class"));
        assertEquals(cpuClass + 8, Library.offset(DenseIndex.DESCRIPTION, "shards_count"));
        assertEquals(cpuClass + 8 + word, DenseIndex.DESCRIPTION.byteSize());
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
        // A build without the NVMe tier refuses the view; one with it fails to open the absent checkpoint.
        Status refused = assertThrows(USearchException.class, () -> DenseIndex.viewNvme("index.safegraphs"))
                .status();
        assertTrue(refused == Status.UNSUPPORTED || refused == Status.IO, refused.toString());
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
        assertEquals("i32", Scalar.I32.toString());
        assertEquals("spinning", Mood.SPINNING.toString());
        assertEquals("jaccard", DenseMetric.JACCARD.toString());
        assertEquals("vram", Residency.VRAM.toString());
        assertEquals("scan", Traversal.SCAN.toString());
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
        // C reads a negative wanted count as a size past any machine's, so this side refuses it.
        assertEquals(
                Status.INVALID_ARGUMENT,
                assertThrows(USearchException.class, () -> new DenseSearchConfig(-1))
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
        try (DenseIndex index = empty(twoDimensional());
                Arena arena = new Arena()) {
            index.reserve(2);
            // Four floats per row, of which the trailing two are padding no dimension names.
            float[] padded = {0, 1, 9, 9, 1, 0, 9, 9};
            index.insert(
                    new Key[] {Key.of(1), Key.of(2)},
                    RowsView.strided(MemorySegment.ofArray(padded), 2, 2, 16, Scalar.F32));

            DenseVectors retrieved = index.get(Key.of(2), Scalar.F32, arena);
            assertArrayEquals(new float[] {1, 0}, floatsOf(retrieved), 0f);
            // The width comes from what C reported, never from a describe call.
            assertEquals(2, retrieved.columns());
        }
    }

    @Test
    public void searchWritesOnlyWhatTheCountsClaim() {
        try (DenseIndex index = populated();
                Arena arena = new Arena(POOL)) {
            Matches matches = index.on(POOL).search(RowsView.of(new float[] {0, 1}, 2), 4, arena);
            assertEquals(1, matches.queries());
            assertEquals(4, matches.columns());
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

            // A zero wanted count lands only the counts, which C accepts, so neither side refuses it.
            try (Arena other = new Arena()) {
                assertEquals(
                        0,
                        index.search(RowsView.of(new float[] {0, 1}, 2), 0, other)
                                .count(0));
            }
        }
    }

    @Test
    public void everyWalkReportsTraversalTelemetry() {
        try (DenseIndex index = populated();
                Arena arena = new Arena()) {
            DenseTraversalStats searched = new DenseTraversalStats();
            index.on(POOL)
                    .search(
                            RowsView.of(new float[] {0, 1}, 2),
                            new DenseSearchConfig(2).expansionFactor(64).stats(searched),
                            arena);
            assertTrue(searched.visitedMembers() > 0);
            assertTrue(searched.computedDistances() > 0);

            DenseTraversalStats refined = new DenseTraversalStats();
            index.on(POOL).refine(new DenseRefineConfig().expansionFactor(32).stats(refined));
            assertTrue(refined.visitedMembers() > 0);
        }
    }

    @Test
    public void scansScoreEveryRowSoEveryQueryFindsItself() {
        try (DenseIndex index = empty(new DenseOptions().dimensions(2));
                Arena arena = new Arena()) {
            index.reserve(3);
            RowsView rows = RowsView.of(new float[] {1, 0, 0, 1, 1, 1}, 2);
            index.insert(new Key[] {Key.of(1), Key.of(2), Key.of(3)}, rows);
            Matches matches = index.search(rows, new DenseSearchConfig(1).traversal(Traversal.SCAN), arena);
            assertEquals(3, matches.queries());
            for (int query = 0; query != 3; ++query) {
                assertEquals(Key.of(query + 1), matches.key(query, 0));
            }
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
                        .valuesPerKey(2));
                Arena arena = new Arena()) {
            index.on(POOL).reserve(4);
            // Two documents of two token rows apiece, so a key names a document and not a row.
            index.on(POOL)
                    .insert(
                            new Key[] {Key.of(10), Key.of(10), Key.of(20), Key.of(20)},
                            RowsView.of(new float[] {1, 0, 0, 1, 2, 0, 0, 2}, 2));

            RowsView tokens = RowsView.of(new float[] {1, 0, 0, 1}, 2);
            DenseProducts matches = index.on(POOL).maxsim(tokens, 2, 2, arena);
            assertEquals(1, matches.queries());
            assertEquals(2, matches.count(0));
            long bounded = matches.count(0);
            assertTrue(index.on(POOL).maxsim(tokens, 2, 0, arena).count(0) >= bounded);

            // The rows of a batch must divide into whole queries, and the check runs before the call.
            assertEquals(
                    Status.SHAPE_MISMATCH,
                    assertThrows(
                                    USearchException.class,
                                    () -> index.on(POOL).maxsim(RowsView.of(new float[] {1, 0, 0, 1}, 2), 3, 1, arena))
                            .status());
        }
    }

    @Test
    public void lateInteractionRefusesADistanceKind() {
        // No Java-side check produces CONFIG_MISMATCH, so this is the one failure that can only have
        // come from the boundary's exception factory, and its detail from the engine's own string.
        try (DenseIndex index = populated();
                Arena arena = new Arena()) {
            USearchException refused = assertThrows(
                    USearchException.class,
                    () -> index.on(POOL).maxsim(RowsView.of(new float[] {0, 1}, 2), 1, 1, arena));
            assertEquals(Status.CONFIG_MISMATCH, refused.status());
            assertTrue(refused.getMessage().startsWith("usearch: config-mismatch: "));
            assertFalse(refused.getMessage().endsWith(": "));
        }
    }

    @Test
    public void zeroDimensionsAreRefusedAtInit() {
        assertEquals(
                Status.CONFIG_MISMATCH,
                assertThrows(USearchException.class, () -> empty(new DenseOptions())).status());
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
    public void getLandsEveryEntryUnderAKey() {
        try (DenseIndex index = empty(twoDimensional())) {
            index.on(POOL).reserve(3);
            // Keys need not be unique, so three entries share one and the call alone sizes the export.
            index.on(POOL)
                    .insert(
                            new Key[] {Key.of(5), Key.of(5), Key.of(5)},
                            RowsView.of(new float[] {0, 1, 1, 0, 1, 1}, 2));
            assertEquals(3, index.on(POOL).count(Key.of(5)));
            assertFalse(index.contains(Key.of(6)));

            Arena arena = new Arena(POOL);
            assertEquals(Residency.RAM, arena.residency());
            DenseVectors every = index.get(Key.of(5), Scalar.F64, arena);
            assertEquals(3, every.rows());
            assertEquals(2 * Double.BYTES, every.strideBytes());
            // A sharded index deals one batch's entries across its shards, so only the multiset is promised.
            double[] values = every.segment().toArray(ValueLayout.JAVA_DOUBLE);
            double[][] landed = new double[3][];
            for (int row = 0; row != landed.length; ++row) {
                landed[row] = Arrays.copyOfRange(values, row * 2, row * 2 + 2);
            }
            Arrays.sort(landed, Arrays::compare);
            assertArrayEquals(new double[][] {{0, 1}, {1, 0}, {1, 1}}, landed);
            MemorySegment rows = every.segment();
            assertEquals(0, index.on(POOL).get(Key.of(6), Scalar.F32, arena).rows());
            // The second call rewound the arena, so the first one's rows refuse to be read.
            assertThrows(IllegalStateException.class, () -> rows.get(ValueLayout.JAVA_DOUBLE, 0));

            // A null arena is refused as C refuses one, and a closed one cannot be named again.
            assertEquals(
                    Status.INVALID_ARGUMENT,
                    assertThrows(USearchException.class, () -> index.get(Key.of(5), Scalar.F32, null))
                            .status());
            arena.close();
            assertThrows(IllegalStateException.class, () -> index.get(Key.of(5), Scalar.F32, arena));
        }
    }

    @Test
    public void aSharedArenaInvalidatesThePreviousResult() {
        try (DenseIndex index = populated();
                Arena arena = new Arena()) {
            DenseVectors first = index.on(POOL).get(Key.of(10), Scalar.F32, arena);
            assertArrayEquals(new float[] {0, 1}, floatsOf(first), 0f);
            Matches second = index.on(POOL).search(RowsView.of(new float[] {1, 0}, 2), 1, arena);
            assertEquals(Key.of(20), second.key(0, 0));
            // Any verb may reuse any arena, and each rewinds it, so the first view refuses to read freed rows.
            assertThrows(IllegalStateException.class, () -> floatsOf(first));
        }
    }

    @Test
    public void multiplyLandsRaggedProducts() {
        try (DenseIndex index = empty(new DenseOptions().dimensions(2).metric(DenseMetric.INNER));
                Arena arena = new Arena()) {
            index.on(POOL).reserve(2);
            index.on(POOL).insert(new Key[] {Key.of(10), Key.of(20)}, RowsView.of(new float[] {1, 0, 0, 2}, 2));
            DenseProducts products = index.on(POOL).multiply(RowsView.of(new float[] {1, 1, 0, 1}, 2), arena);
            assertEquals(2, products.queries());
            assertEquals(2, products.count(0));
            assertEquals(2, products.count(1));
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

    /** The value lookup is told the scalar and width it must answer in, and a key it misses refuses the call. */
    @Test
    public void attentionResolvesValuesThroughTheLookup() {
        try (DenseIndex index = empty(new DenseOptions().dimensions(2).metric(DenseMetric.INNER));
                Arena arena = new Arena()) {
            index.reserve(2);
            index.insert(new Key[] {Key.of(0), Key.of(1)}, RowsView.of(new float[] {1, 0, 0, 1}, 2));
            RowsView values = bf16RowsOf(2, 5, 5, 7, 7);
            ValueLookup byKey = new ValueLookup(values, Key::toLong);
            DenseSoftmaxState[] carry = new DenseSoftmaxState[1];
            DenseVectors outputs = index.on(POOL)
                    .attend(RowsView.of(new float[] {1, 0}, 2), byKey, new DenseAttendConfig().carry(carry), arena);
            assertEquals(1, outputs.rows());
            assertEquals(Scalar.F32, outputs.scalar());
            float[] attended = floatsOf(outputs);
            // A convex mix of the two value rows, weighted toward the first key.
            assertTrue(attended[0] > 5 && attended[0] < 6);
            assertTrue(carry[0].partition() > 1);

            // Values of another scalar than the index stores resolve to nothing, so the call is refused.
            ValueLookup mistyped = new ValueLookup(RowsView.of(new double[] {5, 5, 7, 7}, 2), Key::toLong);
            assertEquals(
                    Status.SHAPE_MISMATCH,
                    assertThrows(
                                    USearchException.class,
                                    () -> index.attend(RowsView.of(new float[] {1, 0}, 2), mistyped, arena))
                            .status());

            // An exception the lookup raises outranks the status it turned into.
            ValueLookup throwing = new ValueLookup(values, key -> {
                throw new ArithmeticException("no value for " + key);
            });
            assertThrows(
                    ArithmeticException.class, () -> index.attend(RowsView.of(new float[] {1, 0}, 2), throwing, arena));
        }
    }

    /** A call leases its index, so closing the index from inside its own call is refused, not a use-after-free. */
    @Test
    public void aCallLeasesItsIndex() {
        DenseIndex index = empty(new DenseOptions().dimensions(2).metric(DenseMetric.INNER));
        index.reserve(1);
        index.insert(new Key[] {Key.of(0)}, RowsView.of(new float[] {1, 0}, 2));
        AtomicReference<RuntimeException> refused = new AtomicReference<>();
        ValueLookup closing = new ValueLookup(bf16RowsOf(2, 5, 5), key -> {
            try {
                index.close();
            } catch (IllegalStateException leased) {
                refused.set(leased);
            }
            return 0;
        });
        try (Arena arena = new Arena()) {
            index.attend(RowsView.of(new float[] {1, 0}, 2), closing, arena);
        }
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
        Arena arena = new Arena(POOL);
        Monolith monolith;
        try (DenseIndex index = populated()) {
            index.on(POOL).save(checkpoint.getAbsolutePath());
            monolith = index.on(POOL).saveBuffer(arena);
        }

        DenseMetadata described = DenseIndex.metadataRead(checkpoint.getAbsolutePath(), POOL);
        assertEquals(2, described.dimensions());
        assertEquals(DenseMetric.EUCLIDEAN, described.metric());
        assertEquals(Scalar.BF16, described.quantization());
        byte[] slab = monolith.segment().toArray(ValueLayout.JAVA_BYTE);
        assertEquals(
                2, DenseIndex.metadataReadBuffer(MemorySegment.ofArray(slab)).dimensions());

        try (DenseIndex reopened = DenseIndex.loadRam(checkpoint.getAbsolutePath(), POOL, new int[0], null, 0)) {
            assertEquals(2, reopened.describe().size());
            assertEquals(2, reopened.describe().metadata().dimensions());
        }

        // The monolith lives in the arena, so it outlives the index that wrote it.
        try (arena;
                DenseIndex reopened = DenseIndex.loadRamBuffer(monolith.segment(), POOL, new int[0], null, 0)) {
            assertEquals(2, reopened.describe().size());
        }

        DenseIndex viewed;
        try {
            viewed = DenseIndex.viewRam(checkpoint.getAbsolutePath());
        } catch (USearchException refused) {
            // RAM serves no mapping in place yet.
            assertEquals(Status.UNSUPPORTED, refused.status());
            return;
        }
        try (viewed) {
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
        try (DenseIndex index = empty(twoDimensional());
                Arena arena = new Arena()) {
            index.reserve(1);
            float[] row = new float[] {0, 1};
            index.insert(new Key[] {Key.of(1)}, RowsView.of(row, 2));

            // Overwriting the source after the batch must leave the stored vector untouched.
            row[0] = 9;
            row[1] = 9;

            assertArrayEquals(new float[] {0, 1}, floatsOf(index.get(Key.of(1), Scalar.F32, arena)), 0);
        }
    }

    @Test
    public void directBuffersCrossWithoutACopy() {
        try (DenseIndex index = empty(twoDimensional());
                Arena arena = new Arena()) {
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
            assertEquals(Key.of(2), index.on(POOL).search(query, 1, arena).key(0, 0));
        }
    }

    /** The stand-in holds a deferred call until its wait, so the status is observable before the join. */
    @Test
    public void deferredCallsCompleteOnTheirExecutor() {
        try (Executor pool = Executor.forRam(new int[0], null, 2, 4);
                Arena arena = new Arena(pool);
                DenseIndex index = empty(twoDimensional())) {
            index.reserve(2);
            DenseIndex.DeferredOn deferred = index.deferred(pool);
            try (Deferred<Void> inserted =
                    deferred.insert(new Key[] {Key.of(10), Key.of(20)}, RowsView.of(new float[] {0, 1, 1, 0}, 2))) {
                Deferred<Matches> searched = deferred.search(RowsView.of(new float[] {1, 0}, 2), 1, arena);
                assertEquals(Status.DEFERRED, inserted.status());
                assertEquals(Status.DEFERRED, searched.status());
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
            Deferred<DenseProducts> refused = deferred.maxsim(RowsView.of(new float[] {1, 0}, 2), 1, 1, arena);
            USearchException reported = assertThrows(USearchException.class, refused::join);
            assertEquals(Status.CONFIG_MISMATCH, reported.status());
            assertSame(reported, assertThrows(USearchException.class, refused::join));
            assertEquals(Status.CONFIG_MISMATCH, refused.status());
            // The failed call released its arena, so the next one may claim it.
            assertEquals(
                    Key.of(20),
                    index.on(pool)
                            .search(RowsView.of(new float[] {1, 0}, 2), 1, arena)
                            .key(0, 0));
            counted.close();
            assertEquals(2, index.on(pool).describe().size());
        }
    }

    /** A deferred call borrows its arena, so closing the arena first waits for the call and voids its view. */
    @Test
    public void closingAnArenaWaitsForTheDeferredCallsNamingIt() {
        try (DenseIndex index = populated()) {
            Arena arena = new Arena(POOL);
            try (Deferred<DenseVectors> fetched = index.deferred(POOL).get(Key.of(20), Scalar.F32, arena)) {
                arena.close();
                assertEquals(Status.SUCCESS, fetched.status());
                assertThrows(IllegalStateException.class, fetched::join);
            }
        }
    }

    /** A call refused because a deferred one holds the arena leaves the held call's rows readable. */
    @Test
    public void aContendedArenaKeepsTheHeldResult() {
        try (DenseIndex index = populated();
                Arena arena = new Arena();
                Deferred<DenseVectors> held = index.deferred(POOL).get(Key.of(10), Scalar.F32, arena)) {
            assertEquals(Status.DEFERRED, held.status());
            assertEquals(
                    Status.CONTENDED,
                    assertThrows(USearchException.class, () -> index.on(POOL).get(Key.of(20), Scalar.F32, arena))
                            .status());
            DenseVectors early = held.join();
            assertArrayEquals(new float[] {0, 1}, floatsOf(early), 0f);

            // Once joined, the arena serves the next call, whose claim retires the earlier rows.
            DenseVectors later = index.on(POOL).get(Key.of(20), Scalar.F32, arena);
            assertArrayEquals(new float[] {1, 0}, floatsOf(later), 0f);
            assertThrows(IllegalStateException.class, () -> floatsOf(early));
        }
    }

    @Test
    public void cursorsWriteBatchesIntoTheirArena() {
        try (DenseIndex index = populated();
                Arena arena = new Arena(POOL);
                DenseCursor cursor = index.on(POOL).cursor()) {
            DenseBatch batch = cursor.nextLinks(4, arena);
            assertEquals(2, batch.written());
            assertEquals(Key.of(10), batch.key(0));
            batch.linkCount(0);
            DenseBatch drained = cursor.next(4, arena);
            assertEquals(0, drained.written());
            assertEquals(
                    Status.INVALID_ARGUMENT,
                    assertThrows(USearchException.class, () -> drained.linkCount(0)).status());
            assertThrows(IllegalStateException.class, () -> batch.key(0));
        }
    }

    @Test
    public void searchStreamsPullUntilDrained() {
        try (DenseIndex index = populated();
                Arena arena = new Arena()) {
            DenseSearchStream opened;
            try {
                opened = index.on(POOL).searchStream(RowsView.of(new float[] {0, 1}, 2));
            } catch (USearchException refused) {
                // A residency that streams no matches yet refuses the stream itself.
                assertEquals(Status.UNSUPPORTED, refused.status());
                return;
            }
            try (DenseSearchStream stream = opened) {
                assertEquals(Key.of(10), stream.next(1, arena).key(0, 0));
                assertEquals(Key.of(20), stream.next(1, arena).key(0, 0));
                assertEquals(0, stream.next(1, arena).count(0));
            }
        }
    }

    @Test
    public void absorbConsumesOnlyTheDonorsItAbsorbed() {
        try (DenseIndex index = populated()) {
            DenseIndex donor = populated();
            // A donor named twice is refused before anything is surrendered, so it stays usable.
            assertEquals(
                    Status.INVALID_ARGUMENT,
                    assertThrows(USearchException.class, () -> index.absorb(donor, donor))
                            .status());
            assertEquals(2, donor.describe().size());
            index.on(POOL).absorb(donor);
            assertEquals(4, index.describe().size());
            assertThrows(IllegalStateException.class, donor::describe);
            donor.close();
            assertEquals(
                    Status.INVALID_ARGUMENT,
                    assertThrows(USearchException.class, () -> index.absorb(index))
                            .status());

            try (DenseIndex other = empty(twoDimensional().quantization(Scalar.I8))) {
                other.reserve(1);
                other.insert(new Key[] {Key.of(30)}, RowsView.of(new float[] {1, 1}, 2));
                try {
                    index.on(POOL).absorb(other);
                } catch (USearchException refused) {
                    // The engine refuses another scalar, which the stand-in absorbs, and hands the donor back live.
                    assertEquals(Status.CONFIG_MISMATCH, refused.status());
                    assertEquals(1, other.describe().size());
                }
            }
        }
    }

    @Test
    public void theCallingThreadNeedsNoExecutor() {
        try (DenseIndex index = empty(twoDimensional());
                Arena arena = new Arena()) {
            index.reserve(2);
            index.insert(new Key[] {Key.of(10), Key.of(20)}, RowsView.of(new float[] {0, 1, 1, 0}, 2));
            assertEquals(
                    Key.of(20),
                    index.search(RowsView.of(new float[] {1, 0}, 2), 1, arena).key(0, 0));
        }
    }

    @Test
    public void aClosedExecutorRefusesEveryVerb() {
        Executor pool = Executor.forRam(new int[0], null, 0, 0);
        pool.close();
        pool.close();
        assertThrows(IllegalStateException.class, () -> pool.drain(Mood.PARKED));
        try (DenseIndex index = populated()) {
            assertThrows(IllegalStateException.class, () -> index.on(pool).describe());
        }
        assertEquals(
                Status.INVALID_ARGUMENT,
                assertThrows(USearchException.class, () -> Executor.forRam(new int[0], null, -1, 0))
                        .status());
        assertEquals(
                Status.SHAPE_MISMATCH,
                assertThrows(USearchException.class, () -> Executor.forVram(new int[] {0}, null, new MemorySegment[2], 0, 0))
                        .status());
    }

    @Test
    public void theLinkedLibraryNamesItself() {
        assertFalse(Types.capabilitiesCompiled().isEmpty());
        assertFalse(Types.capabilitiesAvailable().isEmpty());
        try (DenseIndex index = populated()) {
            DenseIndexDescription description = index.on(POOL).describe();
            assertFalse(description.cpuClass().isEmpty());
            assertEquals("unknown", description.gpuClass());
            assertEquals(2, description.size());
            assertTrue(description.capacity() >= description.size());
            assertTrue(description.memoryUsage() > 0);
            assertTrue(description.metadata().connectivity() > 0);
            index.on(POOL).clear();
            assertEquals(0, index.describe().size());
        }
    }
}
