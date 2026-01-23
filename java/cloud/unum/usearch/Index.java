/**
 * Java bindings for Unum USearch vector search library.
 *
 * <p>
 * USearch is a high-performance approximate nearest neighbor (ANN) search
 * engine optimized for vector similarity search. This Java binding provides a
 * convenient interface to the underlying C++ implementation.
 *
 * <h2>Key Features:</h2>
 * <ul>
 * <li>Multiple distance metrics (Cosine, Euclidean, Haversine, etc.)</li>
 * <li>Multiple quantization types (Float32, BFloat16, Float16, Int8, Binary)</li>
 * <li>SIMD-accelerated distance calculations</li>
 * <li>Memory-efficient storage with configurable precision</li>
 * <li>Thread-safe operations for concurrent construction or search</li>
 * <li>Persistent storage with save/load capabilities</li>
 * </ul>
 *
 * <h2>Basic Usage:</h2>
 * <pre>{@code
 * // Create an index for 128-dimensional vectors using cosine similarity
 * try (Index index = new Index.Config()
 *         .metric(Index.Metric.COSINE)
 *         .quantization(Index.Quantization.FLOAT32)
 *         .dimensions(128)
 *         .connectivity(16)
 *         .build()) {
 *
 *     // Add vectors
 *     float[] vector1 = {1.0f, 2.0f, 3.0f, ...}; // 128 dimensions
 *     index.add(42L, vector1);
 *
 *     // Search for similar vectors
 *     float[] query = {1.1f, 2.1f, 3.1f, ...};
 *     long[] results = index.search(query, 10); // Find 10 nearest neighbors
 *
 *     // Retrieve a vector by key
 *     float[] retrieved = index.get(42L);
 * }
 * }</pre>
 *
 * <h2>Advanced Configuration:</h2>
 * <pre>{@code
 * Index index = new Index.Config()
 *     .metric(Index.Metric.EUCLIDEAN_SQUARED)    // Distance metric (L2² - no sqrt)
 *     .quantization(Index.Quantization.FLOAT16)  // Storage precision
 *     .dimensions(768)                           // Vector dimensions
 *     .capacity(1000000)                         // Expected number of vectors
 *     .connectivity(32)                          // Graph connectivity (higher = better recall)
 *     .expansion_add(128)                        // Search width during insertion
 *     .expansion_search(64)                      // Search width during queries
 *     .build();
 * }</pre>
 *
 * <h2>Thread Safety:</h2>
 * <p>
 * USearch index operations are thread-safe for many concurrent reads or many
 * concurrent writes. Operations except {@code search()} and {@code add()}
 * shouldn't be called concurrently.</p>
 *
 * <h2>Memory Management:</h2>
 * <p>
 * This class implements {@link AutoCloseable} for automatic resource
 * management. Always use try-with-resources or explicitly call {@link #close()}
 * to free native memory.</p>
 *
 * @see <a href="https://github.com/unum-cloud/usearch">USearch GitHub
 * Repository</a>
 * @see <a href="https://unum-cloud.github.io/usearch/">USearch
 * Documentation</a>
 */
package cloud.unum.usearch;

import java.io.IOException;

public class Index implements AutoCloseable {

    /**
     * Distance metric constants for vector similarity calculations. These
     * constants can be used with {@link Config#metric(String)}.
     */
    public static final class Metric {

        /**
         * Inner product (dot product) similarity
         */
        public static final String INNER_PRODUCT = "ip";

        /**
         * Cosine similarity for normalized vectors
         */
        public static final String COSINE = "cos";

        /**
         * Squared Euclidean distance (L2²) - faster than true Euclidean, no
         * sqrt
         */
        public static final String EUCLIDEAN_SQUARED = "l2sq";

        /**
         * Haversine distance for geographic coordinates
         */
        public static final String HAVERSINE = "haversine";

        /**
         * Hamming distance for binary vectors
         */
        public static final String HAMMING = "hamming";

        /**
         * Jaccard similarity coefficient
         */
        public static final String JACCARD = "jaccard";
    }

    /**
     * Scalar quantization types for vector storage. These constants can be used
     * with {@link Config#quantization(String)}.
     */
    public static final class Quantization {

        /**
         * 64-bit floating point (double precision)
         */
        public static final String FLOAT64 = "f64";

        /**
         * 32-bit floating point (single precision)
         */
        public static final String FLOAT32 = "f32";

        /**
         * Brain Float 16 (half precision with a wider exponent)
         */
        public static final String BFLOAT16 = "bf16";

        /**
         * 16-bit floating point (half precision)
         */
        public static final String FLOAT16 = "f16";

        /**
         * 8-bit integer quantization
         */
        public static final String INT8 = "i8";

        /**
         * Binary quantization (1 bit per dimension, 8 dimensions per word)
         */
        public static final String BINARY = "b1";
    }

    private long c_ptr = 0;

    /**
     * Creates a new instance of Index with specified parameters.
     *
     * @param metric distance metric for vector similarity calculation
     * @param quantization scalar quantization type for vector storage
     * @param dimensions number of vector dimensions
     * @param capacity initial index capacity
     * @param connectivity max connections per node in graph
     * @param expansion_add search width during vector insertion
     * @param expansion_search search width during queries
     */
    public Index(
            String metric,
            String quantization,
            long dimensions,
            long capacity,
            long connectivity,
            long expansion_add,
            long expansion_search) {
        this(
                c_create(
                        metric,
                        quantization,
                        dimensions,
                        capacity,
                        connectivity,
                        expansion_add,
                        expansion_search));
    }

    private Index(long c_ptr) {
        this.c_ptr = c_ptr;
    }

    /**
     * Loads an index from file into memory.
     *
     * @param path file path to load from
     * @return mutable Index instance
     * @throws Error if loading fails
     */
    public static Index loadFromPath(String path) {
        return new Index(c_createFromFile(path, false));
    }

    /**
     * Creates read-only view of index from file.
     *
     * @param path file path to load from
     * @return immutable Index view
     * @throws Error if loading fails
     */
    public static Index viewFromPath(String path) {
        return new Index(c_createFromFile(path, true));
    }

    @Override
    public void close() {
        if (c_ptr == 0) {
            throw new IllegalStateException("Index already closed");
        }
        c_destroy(c_ptr);
        c_ptr = 0;
    }

    /**
     * Returns number of vectors in the index.
     *
     * @return current index size
     */
    public long size() {
        if (c_ptr == 0) {
            throw new IllegalStateException("Index already closed");
        }
        return c_size(c_ptr);
    }

    /**
     * Returns graph connectivity parameter.
     *
     * @return max connections per node
     */
    public long connectivity() {
        if (c_ptr == 0) {
            throw new IllegalStateException("Index already closed");
        }
        return c_connectivity(c_ptr);
    }

    /**
     * Returns vector dimensionality.
     *
     * @return number of dimensions per vector
     */
    public long dimensions() {
        if (c_ptr == 0) {
            throw new IllegalStateException("Index already closed");
        }
        return c_dimensions(c_ptr);
    }

    /**
     * Returns current index capacity.
     *
     * @return total capacity including current size
     */
    public long capacity() {
        if (c_ptr == 0) {
            throw new IllegalStateException("Index already closed");
        }
        return c_capacity(c_ptr);
    }

    /**
     * Reserves memory for incoming vectors.
     *
     * @param capacity desired total capacity
     */
    public void reserve(long capacity) {
        if (c_ptr == 0) {
            throw new IllegalStateException("Index already closed");
        }
        // Pass zeros to use all available contexts on the device
        c_reserve(c_ptr, capacity, 0, 0);
    }

    /**
     * Reserves memory and configures thread contexts.
     * Use this to explicitly control concurrent add/search capacities.
     *
     * @param capacity desired total capacity
     * @param threadsAdd maximum concurrent add contexts
     * @param threadsSearch maximum concurrent search contexts
     */
    public void reserve(long capacity, long threadsAdd, long threadsSearch) {
        if (c_ptr == 0) {
            throw new IllegalStateException("Index already closed");
        }
        c_reserve(c_ptr, capacity, threadsAdd, threadsSearch);
    }

    /**
     * Reserves memory and sets the same number of contexts
     * for both add and search operations.
     *
     * @param capacity desired total capacity
     * @param threads number of contexts for both add and search
     */
    public void reserve(long capacity, long threads) {
        reserve(capacity, threads, threads);
    }

    /**
     * Adds vector to index with specified key.
     *
     * @param key vector identifier
     * @param vector vector data
     */
    public void add(long key, float vector[]) {
        if (c_ptr == 0) {
            throw new IllegalStateException("Index already closed");
        }
        c_add_f32(c_ptr, key, vector);
    }

    /**
     * Adds vector using zero-copy FloatBuffer.
     *
     * @param key vector identifier
     * @param vector vector data as FloatBuffer
     */
    public void add(long key, java.nio.FloatBuffer vector) {
        if (c_ptr == 0) {
            throw new IllegalStateException("Index already closed");
        }
        if (vector.remaining() != dimensions()) {
            throw new IllegalArgumentException(
                    String.format(
                            "Vector dimensions mismatch: expected %d but got %d",
                            dimensions(), vector.remaining()));
        }
        c_add_f32_buffer(c_ptr, key, vector);
    }

    /**
     * Searches for nearest neighbors.
     *
     * @param vector query vector
     * @param count number of neighbors to find
     * @return array of neighbor keys
     */
    public long[] search(float vector[], long count) {
        if (c_ptr == 0) {
            throw new IllegalStateException("Index already closed");
        }
        return c_search_f32(c_ptr, vector, count);
    }

    /**
     * Searches using zero-copy FloatBuffer.
     *
     * @param vector query vector as FloatBuffer
     * @param count number of neighbors to find
     * @return array of neighbor keys
     */
    public long[] search(java.nio.FloatBuffer vector, long count) {
        if (c_ptr == 0) {
            throw new IllegalStateException("Index already closed");
        }
        if (vector.remaining() != dimensions()) {
            throw new IllegalArgumentException(
                    String.format(
                            "Vector dimensions mismatch: expected %d but got %d",
                            dimensions(), vector.remaining()));
        }
        return c_search_f32_buffer(c_ptr, vector, count);
    }

    /**
     * Searches with zero-copy input and zero-allocation output.
     *
     * @param query query vector as FloatBuffer
     * @param results output buffer for result keys
     * @param maxCount maximum results to find
     * @return actual number of results found
     */
    public int searchInto(java.nio.FloatBuffer query, java.nio.LongBuffer results, long maxCount) {
        if (c_ptr == 0) {
            throw new IllegalStateException("Index already closed");
        }
        if (query.remaining() != dimensions()) {
            throw new IllegalArgumentException(
                    String.format(
                            "Query vector dimensions mismatch: expected %d but got %d",
                            dimensions(), query.remaining()));
        }
        if (results.remaining() < maxCount) {
            throw new IllegalArgumentException(
                    String.format(
                            "Results buffer too small: need %d but only %d remaining",
                            maxCount, results.remaining()));
        }
        int found = c_search_into_f32_buffer(c_ptr, query, results, maxCount);
        // Advance position by the actual number of results, but don't exceed the
        // buffer's remaining capacity
        int currentPosition = results.position();
        int newPosition = Math.min(currentPosition + found, results.limit());
        results.position(newPosition);
        return found;
    }

    /**
     * Retrieves vector by key.
     *
     * @param key vector identifier
     * @return vector contents
     * @throws IllegalArgumentException if key not found
     */
    public float[] get(long key) {
        if (c_ptr == 0) {
            throw new IllegalStateException("Index already closed");
        }
        return c_get(c_ptr, key);
    }

    /**
     * Adds double precision vector to index.
     *
     * @param key vector identifier
     * @param vector double precision vector data
     */
    public void add(long key, double vector[]) {
        if (c_ptr == 0) {
            throw new IllegalStateException("Index already closed");
        }
        c_add_f64(c_ptr, key, vector);
    }

    /**
     * Adds double precision vector using zero-copy DoubleBuffer.
     *
     * @param key vector identifier
     * @param vector vector data as DoubleBuffer
     */
    public void add(long key, java.nio.DoubleBuffer vector) {
        if (c_ptr == 0) {
            throw new IllegalStateException("Index already closed");
        }
        if (vector.remaining() != dimensions()) {
            throw new IllegalArgumentException(
                    String.format(
                            "Vector dimensions mismatch: expected %d but got %d",
                            dimensions(), vector.remaining()));
        }
        c_add_f64_buffer(c_ptr, key, vector);
    }

    /**
     * Searches using double precision query vector.
     *
     * @param vector double precision query vector
     * @param count number of neighbors to find
     * @return array of neighbor keys
     */
    public long[] search(double vector[], long count) {
        if (c_ptr == 0) {
            throw new IllegalStateException("Index already closed");
        }
        return c_search_f64(c_ptr, vector, count);
    }

    /**
     * Searches using zero-copy DoubleBuffer.
     *
     * @param vector query vector as DoubleBuffer
     * @param count number of neighbors to find
     * @return array of neighbor keys
     */
    public long[] search(java.nio.DoubleBuffer vector, long count) {
        if (c_ptr == 0) {
            throw new IllegalStateException("Index already closed");
        }
        if (vector.remaining() != dimensions()) {
            throw new IllegalArgumentException(
                    String.format(
                            "Vector dimensions mismatch: expected %d but got %d",
                            dimensions(), vector.remaining()));
        }
        return c_search_f64_buffer(c_ptr, vector, count);
    }

    /**
     * Searches with zero-copy double input and zero-allocation output.
     *
     * @param query query vector as DoubleBuffer
     * @param results output buffer for result keys
     * @param maxCount maximum results to find
     * @return actual number of results found
     */
    public int searchInto(java.nio.DoubleBuffer query, java.nio.LongBuffer results, long maxCount) {
        if (c_ptr == 0) {
            throw new IllegalStateException("Index already closed");
        }
        if (query.remaining() != dimensions()) {
            throw new IllegalArgumentException(
                    String.format(
                            "Query vector dimensions mismatch: expected %d but got %d",
                            dimensions(), query.remaining()));
        }
        if (results.remaining() < maxCount) {
            throw new IllegalArgumentException(
                    String.format(
                            "Results buffer too small: need %d but only %d remaining",
                            maxCount, results.remaining()));
        }
        int found = c_search_into_f64_buffer(c_ptr, query, results, maxCount);
        // Advance position by the actual number of results, but don't exceed the
        // buffer's remaining capacity
        int currentPosition = results.position();
        int newPosition = Math.min(currentPosition + found, results.limit());
        results.position(newPosition);
        return found;
    }

    /**
     * Adds int8 quantized vector to index.
     *
     * @param key vector identifier
     * @param vector int8 quantized vector data
     */
    public void add(long key, byte vector[]) {
        if (c_ptr == 0) {
            throw new IllegalStateException("Index already closed");
        }
        c_add_i8(c_ptr, key, vector);
    }

    /**
     * Adds int8 quantized vector using zero-copy ByteBuffer.
     *
     * @param key vector identifier
     * @param vector vector data as ByteBuffer
     */
    public void add(long key, java.nio.ByteBuffer vector) {
        if (c_ptr == 0) {
            throw new IllegalStateException("Index already closed");
        }
        if (vector.remaining() != dimensions()) {
            throw new IllegalArgumentException(
                    String.format(
                            "Vector dimensions mismatch: expected %d but got %d",
                            dimensions(), vector.remaining()));
        }
        c_add_i8_buffer(c_ptr, key, vector);
    }

    /**
     * Searches using int8 quantized query vector.
     *
     * @param vector int8 quantized query vector
     * @param count number of neighbors to find
     * @return array of neighbor keys
     */
    public long[] search(byte vector[], long count) {
        if (c_ptr == 0) {
            throw new IllegalStateException("Index already closed");
        }
        return c_search_i8(c_ptr, vector, count);
    }

    /**
     * Searches using zero-copy ByteBuffer for int8 vectors.
     *
     * @param vector query vector as ByteBuffer
     * @param count number of neighbors to find
     * @return array of neighbor keys
     */
    public long[] search(java.nio.ByteBuffer vector, long count) {
        if (c_ptr == 0) {
            throw new IllegalStateException("Index already closed");
        }
        if (vector.remaining() != dimensions()) {
            throw new IllegalArgumentException(
                    String.format(
                            "Vector dimensions mismatch: expected %d but got %d",
                            dimensions(), vector.remaining()));
        }
        return c_search_i8_buffer(c_ptr, vector, count);
    }

    /**
     * Searches with zero-copy byte input and zero-allocation output.
     *
     * @param query query vector as ByteBuffer
     * @param results output buffer for result keys
     * @param maxCount maximum results to find
     * @return actual number of results found
     */
    public int searchInto(java.nio.ByteBuffer query, java.nio.LongBuffer results, long maxCount) {
        if (c_ptr == 0) {
            throw new IllegalStateException("Index already closed");
        }
        if (query.remaining() != dimensions()) {
            throw new IllegalArgumentException(
                    String.format(
                            "Query vector dimensions mismatch: expected %d but got %d",
                            dimensions(), query.remaining()));
        }
        if (results.remaining() < maxCount) {
            throw new IllegalArgumentException(
                    String.format(
                            "Results buffer too small: need %d but only %d remaining",
                            maxCount, results.remaining()));
        }
        int found = c_search_into_i8_buffer(c_ptr, query, results, maxCount);
        // Advance position by the actual number of results, but don't exceed the
        // buffer's remaining capacity
        int currentPosition = results.position();
        int newPosition = Math.min(currentPosition + found, results.limit());
        results.position(newPosition);
        return found;
    }

    /**
     * Retrieves vector into provided float buffer.
     *
     * @param key vector identifier
     * @param buffer buffer to populate with vector data
     * @throws IllegalArgumentException if key not found or buffer size
     * incorrect
     */
    public void getInto(long key, float[] buffer) {
        if (c_ptr == 0) {
            throw new IllegalStateException("Index already closed");
        }
        c_get_into_f32(c_ptr, key, buffer);
    }

    /**
     * Retrieves vector into provided double buffer.
     *
     * @param key vector identifier
     * @param buffer buffer to populate with vector data
     * @throws IllegalArgumentException if key not found or buffer size
     * incorrect
     */
    public void getInto(long key, double[] buffer) {
        if (c_ptr == 0) {
            throw new IllegalStateException("Index already closed");
        }
        c_get_into_f64(c_ptr, key, buffer);
    }

    /**
     * Retrieves vector into provided byte buffer.
     *
     * @param key vector identifier
     * @param buffer buffer to populate with vector data
     * @throws IllegalArgumentException if key not found or buffer size
     * incorrect
     */
    public void getInto(long key, byte[] buffer) {
        if (c_ptr == 0) {
            throw new IllegalStateException("Index already closed");
        }
        c_get_into_i8(c_ptr, key, buffer);
    }

    /**
     * Saves index to file.
     *
     * @param path file path to save to
     */
    public void save(String path) {
        if (c_ptr == 0) {
            throw new IllegalStateException("Index already closed");
        }
        c_save(c_ptr, path);
    }

    /**
     * Loads index from file.
     *
     * @param path file path to load from
     */
    public void load(String path) {
        if (c_ptr == 0) {
            throw new IllegalStateException("Index already closed");
        }
        c_load(c_ptr, path);
    }

    /**
     * Creates read-only view from file without copying to memory.
     *
     * @param path file path to view
     */
    public void view(String path) {
        if (c_ptr == 0) {
            throw new IllegalStateException("Index already closed");
        }
        c_view(c_ptr, path);
    }

    /**
     * Removes vector from index.
     *
     * @param key vector identifier to remove
     * @return true if removed successfully, false otherwise
     */
    public boolean remove(long key) {
        if (c_ptr == 0) {
            throw new IllegalStateException("Index already closed");
        }
        return c_remove(c_ptr, key);
    }

    /**
     * Renames vector to map to different key.
     *
     * @param from current vector key
     * @param to new vector key
     * @return true if renamed successfully, false otherwise
     */
    public boolean rename(long from, long to) {
        if (c_ptr == 0) {
            throw new IllegalStateException("Index already closed");
        }
        return c_rename(c_ptr, from, to);
    }

    /**
     * Returns memory usage in bytes (graph structure + vectors).
     *
     * @return total memory usage in bytes
     */
    public long memoryUsage() {
        if (c_ptr == 0) {
            throw new IllegalStateException("Index already closed");
        }
        return c_memory_usage(c_ptr);
    }

    /**
     * Returns hardware acceleration used by this index.
     *
     * @return ISA name ("auto" if none, otherwise ISA like "neon", "sve", etc.)
     */
    public String hardwareAcceleration() {
        if (c_ptr == 0) {
            throw new IllegalStateException("Index already closed");
        }
        return c_hardware_acceleration(c_ptr);
    }

    /**
     * Returns distance metric used by this index.
     *
     * @return metric kind ("cos", "l2sq", "ip", etc.)
     */
    public String getMetricKind() {
        if (c_ptr == 0) {
            throw new IllegalStateException("Index already closed");
        }
        return c_metric_kind(c_ptr);
    }

    /**
     * Returns scalar quantization type used by this index.
     *
     * @return scalar kind ("f32", "f16", "bf16", "i8", etc.)
     */
    public String getScalarKind() {
        if (c_ptr == 0) {
            throw new IllegalStateException("Index already closed");
        }
        return c_scalar_kind(c_ptr);
    }

    /**
     * Returns all SIMD capabilities available on the current platform at runtime.
     *
     * @return array of runtime capability names (e.g., ["serial", "haswell", "skylake", "neon"])
     */
    public static String[] hardwareAccelerationAvailable() {
        return c_hardware_acceleration_available();
    }

    /**
     * Returns all SIMD capabilities compiled into this library build.
     *
     * @return array of compiled capability names based on preprocessor macros
     */
    public static String[] hardwareAccelerationCompiled() {
        return c_hardware_acceleration_compiled();
    }

    /**
     * Returns the USearch library version.
     *
     * @return version string
     */
    public static String version() {
        return c_library_version();
    }

    /**
     * Returns whether this USearch build was compiled with dynamic SIMD dispatch.
     *
     * @return true if dynamic dispatch is enabled, false if using compile-time selection
     */
    public static boolean usesDynamicDispatch() {
        return c_uses_dynamic_dispatch();
    }

    /**
     * Builder for configuring Index instances. Uses builder pattern - call
     * {@link #build()} to create Index.
     */
    public static class Config {

        private String _metric = "ip";
        private String _quantization = "f32";
        private long _dimensions = 0;
        private long _capacity = 0;
        private long _connectivity = 0;
        private long _expansion_add = 0;
        private long _expansion_search = 0;

        /**
         * Creates new Config with default settings.
         */
        public Config() {
        }

        /**
         * Builds Index with current configuration.
         *
         * @return configured Index instance
         */
        public Index build() {
            return new Index(
                    _metric,
                    _quantization,
                    _dimensions,
                    _capacity,
                    _connectivity,
                    _expansion_add,
                    _expansion_search);
        }

        /**
         * Sets distance metric.
         *
         * @param _metric metric type
         * @return this Config instance
         */
        public Config metric(String _metric) {
            this._metric = _metric;
            return this;
        }

        /**
         * Sets scalar quantization type.
         *
         * @param _quantization quantization type
         * @return this Config instance
         */
        public Config quantization(String _quantization) {
            this._quantization = _quantization;
            return this;
        }

        /**
         * Sets vector dimensions.
         *
         * @param _dimensions number of dimensions
         * @return this Config instance
         */
        public Config dimensions(long _dimensions) {
            this._dimensions = _dimensions;
            return this;
        }

        /**
         * Sets initial index capacity.
         *
         * @param _capacity index capacity
         * @return this Config instance
         */
        public Config capacity(long _capacity) {
            this._capacity = _capacity;
            return this;
        }

        /**
         * Sets graph connectivity (max connections per node).
         *
         * @param _connectivity connectivity value
         * @return this Config instance
         */
        public Config connectivity(long _connectivity) {
            this._connectivity = _connectivity;
            return this;
        }

        /**
         * Sets search width for vector insertion.
         *
         * @param _expansion_add expansion factor for adding
         * @return this Config instance
         */
        public Config expansion_add(long _expansion_add) {
            this._expansion_add = _expansion_add;
            return this;
        }

        /**
         * Sets search width for queries.
         *
         * @param _expansion_search expansion factor for search
         * @return this Config instance
         */
        public Config expansion_search(long _expansion_search) {
            this._expansion_search = _expansion_search;
            return this;
        }
    }

    static {
        try {
            System.loadLibrary("usearch"); // used for tests. This library in classpath only
        } catch (UnsatisfiedLinkError e) {
            try {
                loadLibraryFromJar();
            } catch (IOException e1) {
                throw new RuntimeException(
                        "Failed to load USearch native library: " + e1.getMessage(), e1);
            }
        }
    }

    private static void loadLibraryFromJar() throws IOException {
        String osName = System.getProperty("os.name").toLowerCase();
        String osArch = System.getProperty("os.arch").toLowerCase();

        String libName;
        if (osName.contains("mac") || osName.contains("darwin")) {
            libName = "libusearch_jni.dylib";
        } else if (osName.contains("windows")) {
            libName = "libusearch_jni.dll";
        } else {
            libName = "libusearch_jni.so";
        }

        // Try architecture-specific first, then fall back to generic
        String[] searchPaths = {
            "/usearch-native/"
            + getArchSpecificPath()
            + "/"
            + libName, // e.g., /usearch-native/linux-x86_64/libusearch.so
            "/usearch-native/" + libName // fallback to generic path
        };

        IOException lastException = null;
        for (String path : searchPaths) {
            try {
                NativeUtils.loadLibraryFromJar(path);
                return; // Success!
            } catch (IOException e) {
                lastException = e;
                // Continue to next path
            }
        }

        throw new IOException(
                "Could not find native library for "
                + osName
                + " "
                + osArch
                + ". Tried paths: "
                + String.join(", ", searchPaths),
                lastException);
    }

    private static String getArchSpecificPath() {
        String osName = System.getProperty("os.name").toLowerCase();
        String osArch = System.getProperty("os.arch").toLowerCase();

        // Normalize architecture names
        String normalizedArch;
        if (osArch.equals("amd64") || osArch.equals("x86_64")) {
            normalizedArch = "amd64";
        } else if (osArch.equals("aarch64") || osArch.equals("arm64")) {
            normalizedArch = "arm64";
        } else if (osArch.equals("x86") || osArch.equals("i386")) {
            normalizedArch = "x86";
        } else if (osArch.equals("armv7l") || osArch.contains("armv7")) {
            normalizedArch = "arm32";
        } else {
            normalizedArch = osArch;
        }

        // Detect Android vs regular Linux
        boolean isAndroid
                = System.getProperty("java.vendor", "").toLowerCase().contains("android")
                || System.getProperty("java.vm.name", "").toLowerCase().contains("dalvik")
                || System.getProperty("java.specification.vendor", "")
                        .toLowerCase()
                        .contains("android");

        // Create platform-specific path
        if (osName.contains("mac") || osName.contains("darwin")) {
            return "darwin-" + normalizedArch;
        } else if (osName.contains("windows")) {
            return "windows-" + normalizedArch;
        } else if (isAndroid) {
            return "android-" + normalizedArch;
        } else {
            return "linux-" + normalizedArch;
        }
    }

    /**
     * Simple test method for Index functionality.
     *
     * @param args command line arguments (unused)
     */
    public static void main(String[] args) {
        try (Index index = new Index.Config().metric("cos").dimensions(100).build()) {
            index.size();
        }
        System.out.println("Java tests passed!");
    }

    private static native long c_create(
            String metric,
            String quantization,
            long dimensions,
            long capacity,
            long connectivity,
            long expansion_add,
            long expansion_search);

    private static native long c_createFromFile(String path, boolean view);

    private static native void c_destroy(long ptr);

    private static native long c_size(long ptr);

    private static native long c_connectivity(long ptr);

    private static native long c_dimensions(long ptr);

    private static native long c_capacity(long ptr);

    private static native void c_reserve(long ptr, long capacity, long threadsAdd, long threadsSearch);

    private static native void c_save(long ptr, String path);

    private static native void c_load(long ptr, String path);

    private static native void c_view(long ptr, String path);

    private static native boolean c_remove(long ptr, long key);

    private static native boolean c_rename(long ptr, long from, long to);

    private static native long c_memory_usage(long ptr);

    private static native String c_hardware_acceleration(long ptr);

    private static native String c_metric_kind(long ptr);

    private static native String c_scalar_kind(long ptr);

    private static native String[] c_hardware_acceleration_available();

    private static native String[] c_hardware_acceleration_compiled();

    private static native String c_library_version();

    private static native boolean c_uses_dynamic_dispatch();

    private static native float[] c_get(long ptr, long key);

    // Overloaded methods:
    private static native void c_add_f32(long ptr, long key, float vector[]);

    private static native void c_add_f64(long ptr, long key, double vector[]);

    private static native void c_add_i8(long ptr, long key, byte vector[]);

    private static native long[] c_search_f32(long ptr, float vector[], long count);

    private static native long[] c_search_f64(long ptr, double vector[], long count);

    private static native long[] c_search_i8(long ptr, byte vector[], long count);

    private static native void c_get_into_f32(long ptr, long key, float buffer[]);

    private static native void c_get_into_f64(long ptr, long key, double buffer[]);

    private static native void c_get_into_i8(long ptr, long key, byte buffer[]);

    // ByteBuffer overloads for zero-copy operations:
    private static native void c_add_f32_buffer(long ptr, long key, java.nio.FloatBuffer vector);

    private static native void c_add_f64_buffer(long ptr, long key, java.nio.DoubleBuffer vector);

    private static native void c_add_i8_buffer(long ptr, long key, java.nio.ByteBuffer vector);

    private static native long[] c_search_f32_buffer(
            long ptr, java.nio.FloatBuffer vector, long count);

    private static native long[] c_search_f64_buffer(
            long ptr, java.nio.DoubleBuffer vector, long count);

    private static native long[] c_search_i8_buffer(
            long ptr, java.nio.ByteBuffer vector, long count);

    // Zero-allocation searchInto methods:
    private static native int c_search_into_f32_buffer(
            long ptr, java.nio.FloatBuffer query, java.nio.LongBuffer results, long maxCount);

    private static native int c_search_into_f64_buffer(
            long ptr, java.nio.DoubleBuffer query, java.nio.LongBuffer results, long maxCount);

    private static native int c_search_into_i8_buffer(
            long ptr, java.nio.ByteBuffer query, java.nio.LongBuffer results, long maxCount);
}
