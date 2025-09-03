
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import cloud.unum.usearch.Index;
import java.io.File;
import java.io.IOException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;
import org.junit.AfterClass;
import org.junit.Test;

public class IndexTest {

    public static void deleteDirectoryFiles(String path) {
        File directory = new File(path);
        if (!directory.isDirectory()) {
            return;
        }

        for (File f : directory.listFiles()) {
            f.delete();
        }
    }

    @Test
    public void test() {
        String path = "./tmp/";
        deleteDirectoryFiles(path);

        try (Index index = new Index.Config().metric("cos").dimensions(2).build()) {
            float vec[] = {10, 20};
            index.reserve(10);
            index.add(42, vec);
            long[] keys = index.search(vec, 5);
        }
    }

    @AfterClass
    public static void tearDown() {
        System.out.println("Java Tests Passed!");
    }

    @Test
    public void testGetSuccess() {
        try (Index index = new Index.Config().metric("cos").dimensions(2).build()) {
            float vec[] = {10, 20};
            index.reserve(10);
            index.add(42, vec);

            assertArrayEquals(vec, index.get(42), 0.01f);
        }
    }

    @Test
    public void testGetFailed() {
        try (Index index = new Index.Config().metric("cos").dimensions(2).build()) {
            float vec[] = {10, 20};
            index.reserve(10);
            index.add(42, vec);

            assertThrows(IllegalArgumentException.class, () -> index.get(41));
        }
    }

    @Test
    public void testUseAfterClose() {
        Index index = new Index.Config().metric("cos").dimensions(2).build();
        float vec[] = {10, 20};
        index.reserve(10);
        index.add(42, vec);
        assertEquals(1, index.size());
        index.close();
        assertThrows(IllegalStateException.class, () -> index.size());
    }

    @Test
    public void testLoadFromPath() throws IOException {
        File indexFile = File.createTempFile("test", "uidx");

        float vec[] = {10, 20};
        try (Index index = new Index.Config().metric("cos").dimensions(2).build()) {
            index.reserve(10);
            index.add(42, vec);
            index.save(indexFile.getAbsolutePath());
        }

        try (Index index = Index.loadFromPath(indexFile.getAbsolutePath())) {
            assertArrayEquals(vec, index.get(42), 0.01f);
        }
    }

    @Test
    public void testLargeVectors() throws IOException {
        File indexFile = File.createTempFile("test", "uidx");

        int dimensions = 256;
        int numVectors = 100;
        try (Index index = new Index.Config().metric("cos").dimensions(dimensions).build()) {
            index.reserve(numVectors);
            for (int v = 0; v < numVectors; v++) {
                index.add(v + 1, randomVector(dimensions));
            }
            index.save(indexFile.getAbsolutePath());
        }

        try (Index index = Index.loadFromPath(indexFile.getAbsolutePath())) {
            for (int i = 0; i < 100; i++) {
                long[] keys = index.search(randomVector(dimensions), 10);
                for (long key : keys) {
                    assertTrue(key >= 1 && key <= numVectors);
                }
            }
        }
    }

    @Test
    public void testShortResults() throws IOException {
        int dimensions = 256;
        int numVectors = 5;
        try (Index index = new Index.Config().metric("cos").dimensions(dimensions).build()) {
            index.reserve(numVectors);
            for (int v = 0; v < numVectors; v++) {
                index.add(v + 1, randomVector(dimensions));
            }

            long[] keys = index.search(randomVector(dimensions), numVectors + 100);
            assertEquals(numVectors, keys.length);
            for (long key : keys) {
                assertNotEquals(0, key);
            }
        }
    }

    private static float[] randomVector(int dimensions) {
        float[] vector = new float[dimensions];
        for (int i = 0; i < dimensions; i++) {
            vector[i] = ThreadLocalRandom.current().nextFloat(2.f);
        }
        return vector;
    }

    @Test
    public void testMemoryUsage() {
        try (Index index = new Index.Config().metric("cos").dimensions(256).build()) {
            // Test empty index
            long initialMemory = index.memoryUsage();
            assertTrue("Initial memory usage should be positive", initialMemory > 0);

            // Add some vectors
            index.reserve(1000);
            long afterReserve = index.memoryUsage();
            assertTrue("Memory should increase after reserve", afterReserve >= initialMemory);

            // Add vectors
            float[] vector = new float[256];
            for (int i = 0; i < 100; i++) {
                for (int j = 0; j < 256; j++) {
                    vector[j] = (float) Math.random();
                }
                index.add(i, vector);
            }

            long afterAdding = index.memoryUsage();
            assertTrue("Memory should increase after adding vectors", afterAdding > afterReserve);

            // Memory should be reasonable (not too small, not too large)
            assertTrue(
                    "Memory usage should be reasonable",
                    afterAdding > 1000 && afterAdding < 1_000_000_000L);
        }
    }

    @Test
    public void testHardwareAccelerationAPIs() {
        try (Index index
                = new Index.Config().metric("cos").quantization("f32").dimensions(10).build()) {
            // Test hardware acceleration API
            String hardwareAcceleration = index.hardwareAcceleration();
            assertNotEquals("Hardware acceleration should not be null", null, hardwareAcceleration);
            assertTrue(
                    "Hardware acceleration should be non-empty", !hardwareAcceleration.isEmpty());

            // Test metric kind API
            String metricKind = index.getMetricKind();
            assertEquals("Metric kind should be cos", "cos", metricKind);

            // Test scalar kind API
            String scalarKind = index.getScalarKind();
            assertEquals("Scalar kind should be f32", "f32", scalarKind);

            System.out.println("Hardware acceleration: " + hardwareAcceleration);
            System.out.println("Metric kind: " + metricKind);
            System.out.println("Scalar kind: " + scalarKind);
        }
    }

    @Test
    public void testDoubleVectorWithFloat64() {
        try (Index index
                = new Index.Config().metric("cos").dimensions(3).quantization("f64").build()) {
            double[] vec = {1.0, 2.0, 3.0};
            index.reserve(10);
            index.add(42, vec);

            double[] retrieved = new double[3];
            index.getInto(42, retrieved);
            assertArrayEquals(vec, retrieved, 0.01);
        }
    }

    @Test
    public void testByteVectorWithInt8() {
        try (Index index
                = new Index.Config().metric("cos").dimensions(4).quantization("i8").build()) {
            byte[] vec = {10, 20, 30, 40};
            index.reserve(10);
            index.add(42, vec);

            byte[] retrieved = new byte[4];
            index.getInto(42, retrieved);
            assertArrayEquals(vec, retrieved);
        }
    }

    @Test
    public void testGetIntoBufferMethods() {
        try (Index index = new Index.Config().metric("cos").dimensions(3).build()) {
            float[] vecF32 = {1.0f, 2.0f, 3.0f};
            index.reserve(10);
            index.add(42, vecF32);

            float[] bufferF32 = new float[3];
            index.getInto(42, bufferF32);
            assertArrayEquals(vecF32, bufferF32, 0.01f);
        }

        try (Index index
                = new Index.Config().metric("cos").dimensions(3).quantization("f64").build()) {
            double[] vecF64 = {1.0, 2.0, 3.0};
            index.reserve(10);
            index.add(43, vecF64);

            double[] bufferF64 = new double[3];
            index.getInto(43, bufferF64);
            assertArrayEquals(vecF64, bufferF64, 0.01);
        }

        try (Index index
                = new Index.Config().metric("cos").dimensions(4).quantization("i8").build()) {
            byte[] vecI8 = {10, 20, 30, 40};
            index.reserve(10);
            index.add(44, vecI8);

            byte[] bufferI8 = new byte[4];
            index.getInto(44, bufferI8);
            assertArrayEquals(vecI8, bufferI8);
        }
    }

    @Test
    public void testConcurrentAdd() throws Exception {
        try (Index index = new Index.Config().metric("cos").dimensions(4).build()) {
            final int threadsCount = 10;
            index.reserve(1000, threadsCount);

            ExecutorService executor = Executors.newFixedThreadPool(threadsCount);
            @SuppressWarnings("unchecked")
            CompletableFuture<Void>[] futures = new CompletableFuture[threadsCount];

            for (int t = 0; t < threadsCount; t++) {
                final int threadId = t;
                futures[t]
                        = CompletableFuture.runAsync(
                                () -> {
                                    for (int i = 0; i < 50; i++) {
                                        long key = threadId * 50L + i;
                                        float[] vector = randomVector(4);
                                        index.add(key, vector);
                                    }
                                },
                                executor);
            }

            CompletableFuture.allOf(futures).get(10, TimeUnit.SECONDS);
            executor.shutdown();

            assertEquals(50L * threadsCount, index.size());
        }
    }

    @Test
    public void testConcurrentSearch() throws Exception {
        try (Index index = new Index.Config().metric("cos").dimensions(4).build()) {
            final int threadsCount = 5;
            index.reserve(100, threadsCount);

            // Add some vectors first
            for (int i = 0; i < 100; i++) {
                index.add(i, randomVector(4));
            }

            ExecutorService executor = Executors.newFixedThreadPool(threadsCount);
            @SuppressWarnings("unchecked")
            CompletableFuture<long[]>[] futures = new CompletableFuture[threadsCount];

            for (int t = 0; t < threadsCount; t++) {
                futures[t]
                        = CompletableFuture.supplyAsync(
                                () -> {
                                    float[] queryVector = randomVector(4);
                                    return index.search(queryVector, 10);
                                },
                                executor);
            }

            for (CompletableFuture<long[]> future : futures) {
                long[] results = future.get(10, TimeUnit.SECONDS);
                assertTrue(results.length > 0);
                assertTrue(results.length <= 10);
            }

            executor.shutdown();
        }
    }

    @Test
    public void testBatchAdd() {
        try (Index index = new Index.Config().metric("cos").dimensions(2).build()) {
            index.reserve(10);

            // Create a batch of 3 vectors concatenated
            float[] batchVectors = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
            index.add(100, batchVectors); // Should add keys 100, 101, 102

            assertEquals(3, index.size());

            // Verify each vector was added correctly
            assertArrayEquals(new float[]{1.0f, 2.0f}, index.get(100), 0.01f);
            assertArrayEquals(new float[]{3.0f, 4.0f}, index.get(101), 0.01f);
            assertArrayEquals(new float[]{5.0f, 6.0f}, index.get(102), 0.01f);
        }
    }

    @Test
    public void testBatchAddDouble() {
        try (Index index
                = new Index.Config().metric("cos").dimensions(3).quantization("f64").build()) {
            index.reserve(10);

            // Create a batch of 2 double vectors concatenated
            double[] batchVectors = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
            index.add(200, batchVectors); // Should add keys 200, 201

            assertEquals(2, index.size());

            double[] buffer = new double[3];
            index.getInto(200, buffer);
            assertArrayEquals(new double[]{1.0, 2.0, 3.0}, buffer, 0.01);

            index.getInto(201, buffer);
            assertArrayEquals(new double[]{4.0, 5.0, 6.0}, buffer, 0.01);
        }
    }

    @Test
    public void testBatchAddByte() {
        try (Index index
                = new Index.Config().metric("cos").dimensions(4).quantization("i8").build()) {
            index.reserve(10);

            // Create a batch of 2 byte vectors concatenated
            byte[] batchVectors = {10, 20, 30, 40, 50, 60, 70, 80};
            index.add(300, batchVectors); // Should add keys 300, 301

            assertEquals(2, index.size());

            byte[] buffer = new byte[4];
            index.getInto(300, buffer);
            assertArrayEquals(new byte[]{10, 20, 30, 40}, buffer);

            index.getInto(301, buffer);
            assertArrayEquals(new byte[]{50, 60, 70, 80}, buffer);
        }
    }

    @Test
    public void testBatchDetection() {
        try (Index index = new Index.Config().metric("cos").dimensions(2).build()) {
            index.reserve(10);

            // Valid batch: 4 elements = 2 vectors of dimension 2
            float[] validBatch = {1.0f, 2.0f, 3.0f, 4.0f};
            index.add(10, validBatch);
            assertEquals(2, index.size());

            // Invalid batch: 3 elements, not divisible by dimensions (2)
            float[] invalidBatch = {1.0f, 2.0f, 3.0f};
            assertThrows(IllegalArgumentException.class, () -> index.add(20, invalidBatch));

            // Still should be 2 vectors (invalid batch should not have been added)
            assertEquals(2, index.size());
        }
    }

    @Test
    public void testByteBufferOperations() {
        try (Index index = new Index.Config().metric("cos").dimensions(256).build()) {
            index.reserve(1000);

            // Test FloatBuffer operations
            java.nio.ByteBuffer byteBuffer
                    = java.nio.ByteBuffer.allocateDirect(256 * Float.BYTES)
                            .order(java.nio.ByteOrder.nativeOrder());
            java.nio.FloatBuffer floatBuffer = byteBuffer.asFloatBuffer();

            // Fill buffer with test data
            for (int i = 0; i < 256; i++) {
                floatBuffer.put(i, (float) Math.sin(i * 0.1));
            }

            // Test add with ByteBuffer
            index.add(1000L, floatBuffer);
            assertEquals(1, index.size());

            // Test search with ByteBuffer
            long[] results = index.search(floatBuffer, 5);
            assertEquals(1, results.length);
            assertEquals(1000L, results[0]);

            // Verify data consistency by comparing with array method
            float[] arrayData = new float[256];
            floatBuffer.rewind();
            floatBuffer.get(arrayData);

            long[] arrayResults = index.search(arrayData, 5);
            assertArrayEquals(results, arrayResults);
        }
    }

    @Test
    public void testByteBufferDoubleOperations() {
        try (Index index
                = new Index.Config().metric("cos").dimensions(128).quantization("f64").build()) {
            index.reserve(100);

            java.nio.ByteBuffer byteBuffer
                    = java.nio.ByteBuffer.allocateDirect(128 * Double.BYTES)
                            .order(java.nio.ByteOrder.nativeOrder());
            java.nio.DoubleBuffer doubleBuffer = byteBuffer.asDoubleBuffer();

            // Fill buffer with test data
            for (int i = 0; i < 128; i++) {
                doubleBuffer.put(i, Math.cos(i * 0.05));
            }

            index.add(2000L, doubleBuffer);
            long[] results = index.search(doubleBuffer, 3);
            assertEquals(1, results.length);
            assertEquals(2000L, results[0]);
        }
    }

    @Test
    public void testByteBufferByteOperations() {
        try (Index index
                = new Index.Config().metric("cos").dimensions(64).quantization("i8").build()) {
            index.reserve(50);

            java.nio.ByteBuffer byteBuffer
                    = java.nio.ByteBuffer.allocateDirect(64).order(java.nio.ByteOrder.nativeOrder());

            // Fill buffer with test data
            for (int i = 0; i < 64; i++) {
                byteBuffer.put(i, (byte) (i % 127));
            }

            byteBuffer.rewind();
            index.add(3000L, byteBuffer);

            byteBuffer.rewind();
            long[] results = index.search(byteBuffer, 2);
            assertEquals(1, results.length);
            assertEquals(3000L, results[0]);
        }
    }

    @Test
    public void testByteBufferPerformanceComparison() {
        int dimensions = 512;
        int numVectors = 1000;
        int numQueries = 100;

        try (Index index = new Index.Config().metric("cos").dimensions(dimensions).build()) {
            index.reserve(numVectors);

            // Prepare test data
            float[][] vectors = new float[numVectors][dimensions];
            java.nio.ByteBuffer[] buffers = new java.nio.ByteBuffer[numVectors];

            for (int i = 0; i < numVectors; i++) {
                vectors[i] = randomVector(dimensions);
                buffers[i]
                        = java.nio.ByteBuffer.allocateDirect(dimensions * Float.BYTES)
                                .order(java.nio.ByteOrder.nativeOrder());
                buffers[i].asFloatBuffer().put(vectors[i]);
            }

            // Test array-based add performance
            long arrayAddStart = System.nanoTime();
            for (int i = 0; i < numVectors; i++) {
                index.add(i, vectors[i]);
            }
            long arrayAddTime = System.nanoTime() - arrayAddStart;

            // Clear index for ByteBuffer test
            try (Index bufferIndex
                    = new Index.Config().metric("cos").dimensions(dimensions).build()) {
                bufferIndex.reserve(numVectors);

                // Test ByteBuffer-based add performance
                long bufferAddStart = System.nanoTime();
                for (int i = 0; i < numVectors; i++) {
                    bufferIndex.add(i, buffers[i].asFloatBuffer());
                }
                long bufferAddTime = System.nanoTime() - bufferAddStart;

                // Test search performance
                float[] queryVector = randomVector(dimensions);
                java.nio.ByteBuffer queryBuffer
                        = java.nio.ByteBuffer.allocateDirect(dimensions * Float.BYTES)
                                .order(java.nio.ByteOrder.nativeOrder());
                queryBuffer.asFloatBuffer().put(queryVector);

                // Array-based search
                long arraySearchStart = System.nanoTime();
                for (int i = 0; i < numQueries; i++) {
                    index.search(queryVector, 10);
                }
                long arraySearchTime = System.nanoTime() - arraySearchStart;

                // ByteBuffer-based search
                long bufferSearchStart = System.nanoTime();
                for (int i = 0; i < numQueries; i++) {
                    bufferIndex.search(queryBuffer.asFloatBuffer(), 10);
                }
                long bufferSearchTime = System.nanoTime() - bufferSearchStart;

                // Print performance results
                System.out.println("Performance Comparison (ns):");
                System.out.printf(
                        "Array Add: %,d | ByteBuffer Add: %,d (%.2fx faster)%n",
                        arrayAddTime, bufferAddTime, (double) arrayAddTime / bufferAddTime);
                System.out.printf(
                        "Array Search: %,d | ByteBuffer Search: %,d (%.2fx faster)%n",
                        arraySearchTime,
                        bufferSearchTime,
                        (double) arraySearchTime / bufferSearchTime);

                // Verify correctness - results should be equivalent
                long[] arrayResults = index.search(queryVector, 10);
                long[] bufferResults = bufferIndex.search(queryBuffer.asFloatBuffer(), 10);
                assertEquals(
                        "Search results should be equivalent",
                        arrayResults.length,
                        bufferResults.length);
            }
        }
    }

    @Test
    public void testSearchIntoZeroAllocation() {
        try (Index index = new Index.Config().metric("cos").dimensions(128).build()) {
            index.reserve(100);

            // Add some test vectors
            java.nio.ByteBuffer vectorBuffer
                    = java.nio.ByteBuffer.allocateDirect(128 * Float.BYTES)
                            .order(java.nio.ByteOrder.nativeOrder());
            java.nio.FloatBuffer floatBuffer = vectorBuffer.asFloatBuffer();

            for (int i = 0; i < 10; i++) {
                for (int j = 0; j < 128; j++) {
                    floatBuffer.put(j, (float) (Math.sin(i + j * 0.1) + i * 0.01));
                }
                floatBuffer.rewind();
                index.add(i, floatBuffer);
            }

            // Prepare query and results buffers
            java.nio.ByteBuffer queryBuffer
                    = java.nio.ByteBuffer.allocateDirect(128 * Float.BYTES)
                            .order(java.nio.ByteOrder.nativeOrder());
            java.nio.FloatBuffer queryFloat = queryBuffer.asFloatBuffer();

            java.nio.ByteBuffer resultsBuffer
                    = java.nio.ByteBuffer.allocateDirect(5 * Long.BYTES)
                            .order(java.nio.ByteOrder.nativeOrder());
            java.nio.LongBuffer resultsLong = resultsBuffer.asLongBuffer();

            // Set up query vector (same as vector 3)
            for (int j = 0; j < 128; j++) {
                queryFloat.put(j, (float) (Math.sin(3 + j * 0.1) + 3 * 0.01));
            }
            queryFloat.rewind();

            // Test searchInto - should find vector 3 first
            int found = index.searchInto(queryFloat, resultsLong, 5);
            assertTrue("Should find at least 1 result", found >= 1);
            assertTrue("Should find at most 5 results", found <= 5);

            // Verify buffer position was advanced
            assertEquals(
                    "Results buffer position should be advanced", found, resultsLong.position());

            // First result should be key 3 (exact match)
            resultsLong.rewind();
            assertEquals("First result should be exact match", 3L, resultsLong.get(0));
        }
    }

    @Test
    public void testSearchIntoDoubleBuffer() {
        try (Index index
                = new Index.Config().metric("cos").dimensions(64).quantization("f64").build()) {
            index.reserve(50);

            java.nio.ByteBuffer vectorBuffer
                    = java.nio.ByteBuffer.allocateDirect(64 * Double.BYTES)
                            .order(java.nio.ByteOrder.nativeOrder());
            java.nio.DoubleBuffer doubleBuffer = vectorBuffer.asDoubleBuffer();

            // Add test vectors
            for (int i = 0; i < 5; i++) {
                for (int j = 0; j < 64; j++) {
                    doubleBuffer.put(j, Math.cos(i + j * 0.05));
                }
                doubleBuffer.rewind();
                index.add(100 + i, doubleBuffer);
            }

            // Query with vector similar to index 2
            for (int j = 0; j < 64; j++) {
                doubleBuffer.put(j, Math.cos(2 + j * 0.05));
            }
            doubleBuffer.rewind();

            java.nio.ByteBuffer resultsBuffer
                    = java.nio.ByteBuffer.allocateDirect(3 * Long.BYTES)
                            .order(java.nio.ByteOrder.nativeOrder());
            java.nio.LongBuffer resultsLong = resultsBuffer.asLongBuffer();

            int found = index.searchInto(doubleBuffer, resultsLong, 3);
            assertTrue("Should find results", found > 0);
            assertEquals("First result should be key 102", 102L, resultsLong.get(0));
        }
    }

    @Test
    public void testSearchIntoByteBuffer() {
        try (Index index
                = new Index.Config().metric("cos").dimensions(32).quantization("i8").build()) {
            index.reserve(20);

            java.nio.ByteBuffer vectorBuffer
                    = java.nio.ByteBuffer.allocateDirect(32).order(java.nio.ByteOrder.nativeOrder());

            // Add test vectors
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 32; j++) {
                    vectorBuffer.put(j, (byte) ((i * 10 + j) % 127));
                }
                vectorBuffer.rewind();
                index.add(200 + i, vectorBuffer);
            }

            // Query with exact match to vector 1
            for (int j = 0; j < 32; j++) {
                vectorBuffer.put(j, (byte) ((1 * 10 + j) % 127));
            }
            vectorBuffer.rewind();

            java.nio.ByteBuffer resultsBuffer
                    = java.nio.ByteBuffer.allocateDirect(2 * Long.BYTES)
                            .order(java.nio.ByteOrder.nativeOrder());
            java.nio.LongBuffer resultsLong = resultsBuffer.asLongBuffer();

            int found = index.searchInto(vectorBuffer, resultsLong, 2);
            assertTrue("Should find results", found > 0);
            assertEquals("First result should be key 201", 201L, resultsLong.get(0));
        }
    }

    @Test
    public void testPlatformCapabilities() {
        // Test runtime hardware capabilities
        String[] available = Index.hardwareAccelerationAvailable();
        assertNotEquals("Available capabilities should not be null", null, available);
        assertTrue("Platform should have at least serial capability", available.length > 0);

        // Test compile-time capabilities
        String[] compiled = Index.hardwareAccelerationCompiled();
        assertNotEquals("Compiled capabilities should not be null", null, compiled);
        assertTrue("Should have at least serial compiled", compiled.length > 0);

        // Should always include serial as baseline in both
        boolean hasAvailableSerial = false;
        boolean hasCompiledSerial = false;

        for (String cap : available) {
            if ("serial".equals(cap)) {
                hasAvailableSerial = true;
                break;
            }
        }

        for (String cap : compiled) {
            if ("serial".equals(cap)) {
                hasCompiledSerial = true;
                break;
            }
        }

        assertTrue("Platform should always support serial capability", hasAvailableSerial);
        assertTrue("Serial should always be compiled", hasCompiledSerial);

        // Test library version
        String version = Index.version();
        assertNotEquals("Library version should not be null", null, version);
        assertTrue("Library version should be non-empty", !version.isEmpty());

        // Test dynamic dispatch detection
        boolean usesDynamicDispatch = Index.usesDynamicDispatch();
        System.out.println("Uses dynamic dispatch: " + usesDynamicDispatch);

        System.out.println("Available capabilities: " + String.join(", ", available));
        System.out.println("Compiled capabilities: " + String.join(", ", compiled));
        System.out.println("Library version: " + version);
    }
}
