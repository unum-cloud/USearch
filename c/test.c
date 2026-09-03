/**
 *  @file       test.c
 *  @author     Ash Vardanian
 *  @brief      Unit tests for the pure-C ABI of USearch (`usearch.h`).
 *  @date       June 25, 2023
 *
 *  Exercises the lifecycle of `usearch_index_t` through the public C surface:
 *  index creation with every supported metric and scalar kind, `add` / `get` /
 *  `find` / `remove`, on-disk `save` / `load` / `view`, and error propagation
 *  via `usearch_error_t`. The harness is intentionally dependency-free so it
 *  can run in the same matrix as the C++ tests and on cross-compilation
 *  targets where only the C runtime is available.
 *
 *  On startup we install a signal handler (see `install_crash_handlers`) that
 *  prints a native back-trace before re-raising, so CI logs pinpoint the
 *  faulting frame instead of stopping at a bare exit code.
 */
#include <errno.h>
#include <math.h>
#include <signal.h> // `signal`, `raise`, `SIGSEGV`
#include <stdio.h>  // `remove`
#include <stdlib.h>
#include <string.h> // `memset`
#include <sys/stat.h>

/* Back-trace support for the C test harness. The `signal` API is standard C;
 * the back-trace itself is taken via an OS-specific facility since C has no
 * standard stack-introspection API. On Windows, `dbghelp.h` references types
 * (e.g. `PSTR`) that are only defined after `windows.h`, so the two headers
 * are separated by a blank line to keep clang-format from re-sorting them
 * into a single alphabetized block. */
#if defined(_WIN32)
#include <windows.h>

#include <dbghelp.h>
#pragma comment(lib, "Dbghelp.lib")
#elif defined(__unix__) || defined(__APPLE__)
#include <execinfo.h>
#include <unistd.h>
#endif

#include "usearch.h"

static void usearch_write_backtrace(int signal_number) {
    fprintf(stderr, "\n[usearch] Fatal signal %d. Back-trace:\n", signal_number);
#if defined(_WIN32)
    enum { backtrace_depth_limit = 64 };
    void* backtrace_frames[backtrace_depth_limit];
    USHORT backtrace_depth = CaptureStackBackTrace(0, backtrace_depth_limit, backtrace_frames, NULL);
    HANDLE current_process = GetCurrentProcess();
    SymInitialize(current_process, NULL, TRUE);

    unsigned char symbol_info_buffer[sizeof(SYMBOL_INFO) + 256 * sizeof(char)];
    SYMBOL_INFO* symbol_info = (SYMBOL_INFO*)symbol_info_buffer;
    symbol_info->MaxNameLen = 255;
    symbol_info->SizeOfStruct = sizeof(SYMBOL_INFO);

    for (USHORT frame_index = 0; frame_index < backtrace_depth; ++frame_index) {
        if (SymFromAddr(current_process, (DWORD64)backtrace_frames[frame_index], 0, symbol_info))
            fprintf(stderr, "  #%2u %s + 0x%llx\n", (unsigned)frame_index, symbol_info->Name,
                    (unsigned long long)((DWORD64)backtrace_frames[frame_index] - symbol_info->Address));
        else
            fprintf(stderr, "  #%2u %p\n", (unsigned)frame_index, backtrace_frames[frame_index]);
    }
#elif defined(__unix__) || defined(__APPLE__)
    enum { backtrace_depth_limit = 64 };
    void* backtrace_frames[backtrace_depth_limit];
    int backtrace_depth = backtrace(backtrace_frames, backtrace_depth_limit);
    backtrace_symbols_fd(backtrace_frames, backtrace_depth, STDERR_FILENO);
#else
    (void)signal_number;
    fprintf(stderr, "  <back-trace unavailable on this platform>\n");
#endif
    fflush(stderr);
}

static void usearch_crash_handler(int signal_number) {
    usearch_write_backtrace(signal_number);
    /* Restore the default disposition and re-raise so the shell / CI sees the true exit status. */
    signal(signal_number, SIG_DFL);
    raise(signal_number);
}

static void install_crash_handlers(void) {
    int const fatal_signals[] = {SIGSEGV, SIGABRT, SIGILL, SIGFPE};
    for (unsigned signal_index = 0; signal_index < sizeof(fatal_signals) / sizeof(fatal_signals[0]); ++signal_index)
        signal(fatal_signals[signal_index], &usearch_crash_handler);
}

void expect(bool must_be_true, char const* message) {
    if (must_be_true)
        return;
    message = message ? message : "C unit test failed";
    printf("Assert: %s\n", message);
    exit(-1);
}

#define expect_eq(a, b, message) expect(a == b, message)

/**
 * @brief Creates and initializes vectors with random float values.
 *
 * @param count The number of vectors.
 * @param dimensions The number of dimensions per vector.
 * @return A pointer to the first element of the vectors, that must be @b free-ed afterwards.
 */
float* create_vectors(size_t const count, size_t const dimensions) {
    float* data = (float*)malloc(count * dimensions * sizeof(float));
    expect(data, "Failed to allocate memory");
    for (size_t index = 0; index < count * dimensions; ++index)
        data[index] = (float)rand() / (float)RAND_MAX;
    return data;
}

usearch_init_options_t create_options(size_t const dimensions) {
    usearch_init_options_t opts;
    opts.connectivity = 3; // 32 in faiss
    opts.dimensions = dimensions;
    opts.expansion_add = 40;    // 40 in faiss
    opts.expansion_search = 16; // 10 in faiss
    opts.metric_kind = usearch_metric_ip_k;
    opts.metric = NULL;
    opts.quantization = usearch_scalar_f32_k;
    opts.multi = false;
    return opts;
}

/**
 *  This test is designed to verify the initialization of the index with specific dimensions and ensures that the
 *  associated properties are set correctly. It initializes the index twice, checking for errors at each step, and
 *  performs a reserve operation to pre-allocate space in the index, verifying the correct settings of size, capacity,
 *  dimensions, and connectivity after each operation.
 */
void test_init(size_t const collection_size, size_t const dimensions) {
    printf("Test: Index Initialization... %zu vectors, %zu dimensions \n", collection_size, dimensions);

    // Init index
    usearch_error_t error = NULL;
    usearch_init_options_t opts = create_options(dimensions);
    usearch_index_t index = usearch_init(&opts, &error);
    expect(!error, error);
    usearch_free(index, &error);
    expect(!error, error);

    // Init second time
    index = usearch_init(&opts, &error);
    expect(!error, error);

    expect_eq(usearch_size(index, &error), 0, error);
    expect_eq(usearch_capacity(index, &error), 0, error);
    expect_eq(usearch_dimensions(index, &error), dimensions, error);
    expect_eq(usearch_connectivity(index, &error), opts.connectivity, error);

    // Reserve
    usearch_reserve(index, collection_size, &error);
    expect(!error, error);
    expect_eq(usearch_size(index, &error), 0, error);
    expect(usearch_capacity(index, &error) >= collection_size, error);
    expect_eq(usearch_dimensions(index, &error), dimensions, error);
    expect_eq(usearch_connectivity(index, &error), opts.connectivity, error);
    expect(usearch_hardware_acceleration(index, &error), error);
    expect(usearch_memory_usage(index, &error), error);

    usearch_free(index, &error);
    expect(!error, error);

    printf("Test: Index Initialization - PASSED\n");
}

/**
 *  This test validates the addition of vectors to the index. It initializes the index and reserves space for vectors.
 *  It then iteratively adds vectors to the index and checks if the index contains the added vectors by verifying the
 *  size, capacity, and presence of each vector in the index.
 */
void test_add_vector(size_t const collection_size, size_t const dimensions) {
    printf("Test: Add Vector... %zu vectors, %zu dimensions \n", collection_size, dimensions);

    usearch_error_t error = NULL;
    usearch_init_options_t opts = create_options(dimensions);
    usearch_index_t index = usearch_init(&opts, &error);
    usearch_reserve(index, collection_size, &error);

    // Add vectors
    float* data = create_vectors(collection_size, dimensions);
    for (size_t i = 0; i < collection_size; ++i) {
        usearch_key_t key = i;
        usearch_add(index, key, data + i * dimensions, usearch_scalar_f32_k, &error);
        expect(!error, error);
    }

    expect_eq(usearch_size(index, &error), collection_size, error);
    expect(usearch_capacity(index, &error) >= collection_size, error);

    // Check vectors in the index
    for (size_t i = 0; i < collection_size; ++i) {
        usearch_key_t key = i;
        expect(usearch_contains(index, key, &error), error);
    }
    expect(!usearch_contains(index, -1, &error), error); // Non existing key

    free(data);
    usearch_free(index, &error);
    printf("Test: Add Vector - PASSED\n");
}

/**
 *  This test ensures that vectors added to the index can be correctly found. It initializes the index, reserves space,
 *  and adds vectors. It then performs a search query for each added vector to ensure that the vectors are correctly
 *  found in the index, validating the count of found vectors.
 */
void test_find_vector(size_t const collection_size, size_t const dimensions) {
    printf("Test: Find Vector... %zu vectors, %zu dimensions \n", collection_size, dimensions);

    usearch_error_t error = NULL;
    usearch_init_options_t opts = create_options(dimensions);
    usearch_index_t index = usearch_init(&opts, &error);
    usearch_reserve(index, collection_size, &error);

    // Create result buffers
    usearch_key_t* keys = (usearch_key_t*)malloc(collection_size * sizeof(usearch_key_t));
    float* distances = (float*)malloc(collection_size * sizeof(float));
    expect(keys && distances, "Failed to allocate memory");

    // Add vectors
    float* data = create_vectors(collection_size, dimensions);
    for (size_t i = 0; i < collection_size; ++i) {
        usearch_key_t key = i;
        usearch_add(index, key, data + i * dimensions, usearch_scalar_f32_k, &error);
        expect(!error, error);
    }

    // Find the vectors
    for (size_t i = 0; i < collection_size; i++) {
        size_t found_count = usearch_search(index, data + i * dimensions, usearch_scalar_f32_k, collection_size, keys,
                                            distances, &error);
        expect(!error, error);
        expect(found_count >= 1 && found_count <= collection_size, "Vector is missing");
    }

    free(data);
    free(keys);
    free(distances);
    usearch_free(index, &error);
    printf("Test: Find Vector - PASSED\n");
}

/**
 *  This test checks the ability of the index to handle multiple vectors associated with the same key. It initializes
 *  the index with the multi-option enabled, reserves space, and adds multiple vectors with the same key. The test then
 *  retrieves vectors associated with the key from the index and checks the count of retrieved vectors.
 */
void test_get_vector(size_t const collection_size, size_t const dimensions) {
    printf("Test: Get Vector... %zu vectors, %zu dimensions \n", collection_size, dimensions);

    usearch_error_t error = NULL;
    usearch_init_options_t opts = create_options(dimensions);
    opts.multi = true;
    usearch_index_t index = usearch_init(&opts, &error);
    usearch_reserve(index, collection_size, &error);

    // Create result buffers
    float* vectors = (float*)malloc(collection_size * dimensions * sizeof(float));
    expect(vectors, "Failed to allocate memory");

    // Add multiple vectors with SAME key
    usearch_key_t const key = 1;
    float* data = create_vectors(collection_size, dimensions);
    for (size_t i = 0; i < collection_size; i++) {
        usearch_add(index, key, data + i * dimensions, usearch_scalar_f32_k, &error);
        expect(!error, error);
    }

    // Retrieve vectors from index
    size_t found_count = usearch_get(index, key, collection_size, vectors, usearch_scalar_f32_k, &error);
    expect_eq(found_count, collection_size, "Vector is missing");

    free(vectors);
    free(data);
    usearch_free(index, &error);

    printf("Test: Get Vector - PASSED\n");
}

/**
 *  This test ensures that vectors can be successfully removed from the index. It initializes the index, reserves space,
 *  and adds vectors. It then iteratively removes each vector from the index and checks for errors. However, note that
 *  the assert in this test expects an error, indicating that the remove functionality is not currently supported.
 */
void test_remove_vector(size_t const collection_size, size_t const dimensions) {
    printf("Test: Remove Vector... %zu vectors, %zu dimensions \n", collection_size, dimensions);

    usearch_error_t error = NULL;
    usearch_init_options_t opts = create_options(dimensions);
    usearch_index_t index = usearch_init(&opts, &error);
    usearch_reserve(index, collection_size, &error);

    // Add vectors
    float* data = create_vectors(collection_size, dimensions);
    for (size_t i = 0; i < collection_size; ++i) {
        usearch_key_t key = i;
        usearch_add(index, key, data + i * dimensions, usearch_scalar_f32_k, &error);
        expect(!error, error);
    }

    // Remove the vectors
    for (size_t i = 0; i < collection_size; i++) {
        usearch_key_t key = i;
        usearch_remove(index, key, &error);
        expect(!error, "Currently, Remove is not supported");
    }

    free(data);
    usearch_free(index, &error);
    printf("Test: Remove Vector - PASSED\n");
}

/**
 *  This test validates the save and load functionality of the index. It initializes the index, reserves space, and adds
 *  vectors. The index is then saved to a file and freed. A new index is initialized, and the previously saved index is
 *  loaded into it. The test then validates the loaded index properties and ensures that it contains all the vectors
 *  from the saved index.
 */
void test_save_load(size_t const collection_size, size_t const dimensions) {
    printf("Test: Save/Load... %zu vectors, %zu dimensions \n", collection_size, dimensions);
    float* data = create_vectors(collection_size, dimensions);

    usearch_error_t error = NULL;
    usearch_init_options_t opts = create_options(dimensions);
    usearch_init_options_t weird_ops = opts;
    weird_ops.connectivity = 11;
    weird_ops.expansion_add = 15;
    weird_ops.expansion_search = 19;
    weird_ops.metric_kind = usearch_metric_pearson_k;
    weird_ops.quantization = usearch_scalar_f64_k;

    {

        usearch_index_t index = usearch_init(&weird_ops, &error);
        usearch_reserve(index, collection_size, &error);

        // Add vectors
        for (size_t i = 0; i < collection_size; ++i) {
            usearch_key_t key = i;
            usearch_add(index, key, data + i * dimensions, usearch_scalar_f32_k, &error);
            expect(!error, error);
        }

        // Save and free the index
        usearch_save(index, "tmp.usearch", &error);
        expect(!error, error);
        usearch_free(index, &error);
        expect(!error, error);
    }

    // Reset the options
    opts.connectivity = 0;
    opts.dimensions = 0;
    opts.expansion_add = 0;
    opts.expansion_search = 0;
    opts.metric = NULL;
    opts.quantization = usearch_scalar_unknown_k;
    opts.metric_kind = usearch_metric_unknown_k;

    // Reinit
    {

        usearch_index_t index = usearch_init(NULL, &error);
        expect(!error, error);
        // expect(usearch_size(index, &error) == 0, error);

        // Load
        usearch_load(index, "tmp.usearch", &error);
        expect(!error, error);
        expect(usearch_size(index, &error) == collection_size, error);
        expect(usearch_capacity(index, &error) == collection_size, error);
        expect(usearch_dimensions(index, &error) == dimensions, error);
        expect(usearch_connectivity(index, &error) == weird_ops.connectivity, error);

        // Check vectors in the index
        for (size_t i = 0; i < collection_size; ++i) {
            usearch_key_t key = i;
            expect(usearch_contains(index, key, &error), error);
        }

        // Create result buffers
        usearch_key_t* keys = (usearch_key_t*)malloc(collection_size * sizeof(usearch_key_t));
        float* distances = (float*)malloc(collection_size * sizeof(float));
        expect(keys && distances, "Failed to allocate memory");

        // Find the vectors
        usearch_change_threads_search(index, 1, &error);
        for (size_t i = 0; i < collection_size; i++) {
            size_t found_count = usearch_search(index, data + i * dimensions, usearch_scalar_f32_k, collection_size,
                                                keys, distances, &error);
            expect(!error, error);
            expect(found_count >= 1 && found_count <= collection_size, "Vector is missing");
        }

        free(keys);
        free(distances);
        usearch_free(index, &error);
    }

    free(data);

    // Remove the file from disk
    remove("tmp.usearch");
    printf("Test: Save/Load - PASSED\n");
}

/**
 *  This test is designed to validate the view functionality of the index. It initializes the index, reserves space, and
 *  adds vectors. The index is then saved to a file and freed. A new index is initialized and a view is created from the
 *  saved index file. The test is mainly focused on ensuring that no errors occur during these operations, but it does
 *  not verify the properties or contents of the viewed index.
 */
void test_view(size_t const collection_size, size_t const dimensions) {
    printf("Test: View... %zu vectors, %zu dimensions \n", collection_size, dimensions);

    usearch_error_t error = NULL;
    usearch_init_options_t opts = create_options(dimensions);
    usearch_index_t index = usearch_init(&opts, &error);
    usearch_reserve(index, collection_size, &error);

    // Add vectors
    float* data = create_vectors(collection_size, dimensions);
    for (size_t i = 0; i < collection_size; ++i) {
        usearch_key_t key = i;
        usearch_add(index, key, data + i * dimensions, usearch_scalar_f32_k, &error);
        expect(!error, error);
    }

    // Save and free the index
    usearch_save(index, "tmp.usearch", &error);
    expect(!error, error);
    usearch_free(index, &error);
    expect(!error, error);

    // Reinit
    index = usearch_init(&opts, &error);
    expect(!error, error);

    // View
    usearch_view(index, "tmp.usearch", &error);
    expect(!error, error);

    free(data);
    usearch_free(index, &error);
    printf("Test: View - PASSED\n");
}

void test_mini_float_quantizations(size_t const collection_size, size_t const dimensions) {
    printf("Test: Mini-float quantizations... %zu vectors, %zu dimensions\n", collection_size, dimensions);
    usearch_scalar_kind_t kinds[] = {
        usearch_scalar_e5m2_k,
        usearch_scalar_e4m3_k,
        usearch_scalar_e3m2_k,
        usearch_scalar_e2m3_k,
    };
    float* data = create_vectors(collection_size, dimensions);
    usearch_key_t* keys = (usearch_key_t*)malloc(collection_size * sizeof(usearch_key_t));
    float* distances = (float*)malloc(collection_size * sizeof(float));
    expect(keys && distances, "Failed to allocate memory");

    for (size_t k = 0; k < sizeof(kinds) / sizeof(kinds[0]); ++k) {
        usearch_error_t error = NULL;
        usearch_init_options_t opts = create_options(dimensions);
        opts.quantization = kinds[k];
        usearch_index_t index = usearch_init(&opts, &error);
        expect(!error, error);
        usearch_reserve(index, collection_size, &error);
        expect(!error, error);
        for (size_t i = 0; i < collection_size; ++i) {
            usearch_add(index, (usearch_key_t)i, data + i * dimensions, usearch_scalar_f32_k, &error);
            expect(!error, error);
        }
        expect_eq(usearch_size(index, &error), collection_size, error);
        for (size_t i = 0; i < collection_size; ++i) {
            size_t found =
                usearch_search(index, data + i * dimensions, usearch_scalar_f32_k, 1, keys, distances, &error);
            expect(!error, error);
            expect(found >= 1, "Vector not found");
        }
        usearch_free(index, &error);
    }
    free(data);
    free(keys);
    free(distances);
    printf("Test: Mini-float quantizations - PASSED\n");
}

static size_t counted_cosine_calls = 0;

usearch_distance_t counted_cosine_f32(void const* first_ptr, void const* second_ptr) {
    float const* first = (float const*)first_ptr;
    float const* second = (float const*)second_ptr;
    float dot = 0, first_norm = 0, second_norm = 0;
    ++counted_cosine_calls;
    for (size_t dimension = 0; dimension != 3; ++dimension) {
        dot += first[dimension] * second[dimension];
        first_norm += first[dimension] * first[dimension];
        second_norm += second[dimension] * second[dimension];
    }
    if (first_norm == 0)
        return second_norm == 0 ? 0 : 1;
    if (second_norm == 0)
        return 1;
    return 1 - dot / (sqrtf(first_norm) * sqrtf(second_norm));
}

int keep_even_keys(usearch_key_t key, void* state) {
    size_t* calls = (size_t*)state;
    ++*calls;
    return (key % 2) == 0;
}

static void expect_distance_for_key(usearch_key_t const* keys, float const* distances, size_t count,
                                    usearch_key_t key, float expected, float tolerance) {
    for (size_t i = 0; i != count; ++i)
        if (keys[i] == key) {
            expect(fabsf(distances[i] - expected) <= tolerance, "Unexpected cosine distance");
            return;
        }
    expect(false, "Expected cosine key is missing");
}

void expect_cosine_results(usearch_index_t index, void const* query, usearch_scalar_kind_t query_kind,
                           size_t expected_count, float tolerance) {
    usearch_error_t error = NULL;
    usearch_key_t keys[8] = {0};
    float distances[8] = {0};
    usearch_key_t const known_keys[] = {10, 21, 30, 41, 50};
    size_t found = usearch_search(index, query, query_kind, expected_count, keys, distances, &error);
    expect(!error, error);
    expect_eq(found, expected_count, "Unexpected cosine result count");
    for (size_t i = 0; i != found; ++i) {
        expect(distances[i] >= 0 && distances[i] <= 2 + tolerance, "Invalid cosine distance range");
        if (i)
            expect(distances[i - 1] <= distances[i] + tolerance, "Cosine results are not ordered");
        bool known = false;
        for (size_t j = 0; j != sizeof(known_keys) / sizeof(known_keys[0]); ++j)
            known = known || keys[i] == known_keys[j];
        expect(known, "Cosine result contains an invalid key");
        for (size_t j = 0; j != i; ++j)
            expect(keys[j] != keys[i], "Cosine result contains a duplicate key");
    }
}

static bool files_equal(char const* first_path, char const* second_path) {
    FILE* first = fopen(first_path, "rb");
    FILE* second = fopen(second_path, "rb");
    if (!first || !second) {
        if (first)
            fclose(first);
        if (second)
            fclose(second);
        return false;
    }
    bool equal = true;
    while (equal) {
        unsigned char first_bytes[256], second_bytes[256];
        size_t first_count = fread(first_bytes, 1, sizeof(first_bytes), first);
        size_t second_count = fread(second_bytes, 1, sizeof(second_bytes), second);
        equal = first_count == second_count && memcmp(first_bytes, second_bytes, first_count) == 0;
        if (!first_count || !second_count)
            break;
    }
    fclose(first);
    fclose(second);
    return equal;
}

static size_t test_cosine_norm_cache_kind(usearch_scalar_kind_t quantization, char const* saved_path,
                                          char const* roundtrip_path, float tolerance) {
    float const vectors[5][3] = {
        {1, 0, 0},
        {0, 1, 0},
        {-1, 0, 0},
        {0, 0, 0},
        {1, 1, 0},
    };
    usearch_key_t const vector_keys[5] = {10, 21, 30, 41, 50};
    float const query_x[3] = {1, 0, 0};
    float const query_zero[3] = {0, 0, 0};
    float const query_z[3] = {0, 0, 1};
    double const query_x_f64[3] = {1, 0, 0};
    float const replacement[3] = {0, 0, 1};
    usearch_key_t keys[8] = {0};
    float distances[8] = {0};
    usearch_error_t error = NULL;

    usearch_init_options_t options = create_options(3);
    options.metric_kind = usearch_metric_cos_k;
    options.quantization = quantization;
    usearch_index_t index = usearch_init(&options, &error);
    expect(!error && index, error);
    usearch_reserve(index, 8, &error);
    expect(!error, error);
    size_t reserved_memory = usearch_memory_usage(index, &error);
    expect(reserved_memory > 8 * sizeof(uint32_t), "Cosine memory accounting is incomplete");

    for (size_t i = 0; i != 5; ++i) {
        usearch_add(index, vector_keys[i], vectors[i], usearch_scalar_f32_k, &error);
        expect(!error, error);
    }

    expect_cosine_results(index, query_x, usearch_scalar_f32_k, 5, tolerance);
    size_t found = usearch_search(index, query_x, usearch_scalar_f32_k, 5, keys, distances, &error);
    expect_distance_for_key(keys, distances, found, 10, 0, tolerance);
    expect_distance_for_key(keys, distances, found, 50, 0.29289323f, tolerance);
    expect_distance_for_key(keys, distances, found, 21, 1, tolerance);
    expect_distance_for_key(keys, distances, found, 41, 1, tolerance);
    expect_distance_for_key(keys, distances, found, 30, 2, tolerance);

    expect_cosine_results(index, query_zero, usearch_scalar_f32_k, 5, tolerance);
    found = usearch_search(index, query_zero, usearch_scalar_f32_k, 5, keys, distances, &error);
    expect_distance_for_key(keys, distances, found, 41, 0, tolerance);
    for (size_t i = 0; i != found; ++i)
        if (keys[i] != 41)
            expect(fabsf(distances[i] - 1) <= tolerance, "One-zero cosine semantics changed");

    found = usearch_search(index, query_x_f64, usearch_scalar_f64_k, 5, keys, distances, &error);
    expect(!error, error);
    expect_eq(found, 5, "f64 cosine query cast lost results");
    expect_distance_for_key(keys, distances, found, 10, 0, tolerance);

    size_t filter_calls = 0;
    found = usearch_filtered_search(index, query_x, usearch_scalar_f32_k, 5, &keep_even_keys, &filter_calls, keys,
                                    distances, &error);
    expect(!error, error);
    expect_eq(found, 3, "Filtered cosine search returned the wrong count");
    expect(filter_calls > 0, "Cosine filter was not invoked");
    for (size_t i = 0; i != found; ++i)
        expect((keys[i] % 2) == 0, "Cosine filter admitted an odd key");

    expect_eq(usearch_remove(index, 10, &error), 1, "Failed to remove cached cosine vector");
    expect(!error, error);
    usearch_add(index, 10, replacement, usearch_scalar_f32_k, &error);
    expect(!error, error);
    found = usearch_search(index, query_z, usearch_scalar_f32_k, 5, keys, distances, &error);
    expect(!error && found == 5, error);
    expect_distance_for_key(keys, distances, found, 10, 0, tolerance);
    found = usearch_search(index, query_x, usearch_scalar_f32_k, 5, keys, distances, &error);
    expect_distance_for_key(keys, distances, found, 10, 1, tolerance);

    size_t serialized_length = usearch_serialized_length(index, &error);
    usearch_save(index, saved_path, &error);
    expect(!error, error);
    struct stat saved_stat;
    expect(stat(saved_path, &saved_stat) == 0, "Failed to stat saved cosine index");
    expect_eq((size_t)saved_stat.st_size, serialized_length, "Serialized cosine length differs from file bytes");

    usearch_index_t loaded = usearch_init(NULL, &error);
    expect(!error && loaded, error);
    usearch_load(loaded, saved_path, &error);
    expect(!error, error);
    found = usearch_search(loaded, query_z, usearch_scalar_f32_k, 5, keys, distances, &error);
    expect(!error && found == 5, error);
    expect_distance_for_key(keys, distances, found, 10, 0, tolerance);
    expect(usearch_memory_usage(loaded, &error) > 0, "Loaded cosine index did not report memory");
    usearch_save(loaded, roundtrip_path, &error);
    expect(!error, error);
    struct stat roundtrip_stat;
    expect(stat(roundtrip_path, &roundtrip_stat) == 0, "Failed to stat round-tripped cosine index");
    expect_eq((size_t)roundtrip_stat.st_size, serialized_length, "Round-trip changed serialized cosine length");
    expect(files_equal(saved_path, roundtrip_path), "Cosine norm cache changed serialized bytes");

    usearch_index_t viewed = usearch_init(NULL, &error);
    expect(!error && viewed, error);
    usearch_view(viewed, saved_path, &error);
    expect(!error, error);
    found = usearch_search(viewed, query_z, usearch_scalar_f32_k, 5, keys, distances, &error);
    expect(!error && found == 5, error);
    expect_distance_for_key(keys, distances, found, 10, 0, tolerance);

    usearch_clear(index, &error);
    expect(!error && usearch_size(index, &error) == 0, error);
    usearch_add(index, 77, query_x, usearch_scalar_f32_k, &error);
    expect(!error, error);
    found = usearch_search(index, query_x, usearch_scalar_f32_k, 1, keys, distances, &error);
    expect(!error && found == 1 && keys[0] == 77 && distances[0] <= tolerance,
           "Cosine cache did not recover after clear");

    usearch_free(viewed, &error);
    usearch_free(loaded, &error);
    usearch_free(index, &error);
    remove(saved_path);
    remove(roundtrip_path);
    return reserved_memory;
}

static void test_nonfinite_cosine_fallback(void) {
    usearch_error_t error = NULL;
    usearch_init_options_t options = create_options(2);
    options.metric_kind = usearch_metric_cos_k;
    options.quantization = usearch_scalar_f32_k;
    usearch_index_t index = usearch_init(&options, &error);
    expect(!error && index, error);
    usearch_reserve(index, 4, &error);
    expect(!error, error);

    uint32_t nan_bits = 0x7fc00001u, inf_bits = 0x7f800000u;
    float nan_value = 0, inf_value = 0;
    memcpy(&nan_value, &nan_bits, sizeof(nan_value));
    memcpy(&inf_value, &inf_bits, sizeof(inf_value));
    float const finite_vector[2] = {1, 0};
    float const zero_vector[2] = {0, 0};
    float const nan_vector[2] = {nan_value, 1};
    float const inf_vector[2] = {inf_value, 1};
    usearch_add(index, 1, finite_vector, usearch_scalar_f32_k, &error);
    usearch_add(index, 2, zero_vector, usearch_scalar_f32_k, &error);
    usearch_add(index, 3, nan_vector, usearch_scalar_f32_k, &error);
    usearch_add(index, 4, inf_vector, usearch_scalar_f32_k, &error);
    expect(!error, error);

    usearch_key_t keys[4] = {0};
    float distances[4] = {0};
    size_t found = usearch_search(index, finite_vector, usearch_scalar_f32_k, 4, keys, distances, &error);
    expect(!error && found > 0 && found <= 4, "Nonfinite member fallback failed");
    found = usearch_search(index, nan_vector, usearch_scalar_f32_k, 4, keys, distances, &error);
    expect(!error && found > 0 && found <= 4, "Nonfinite query fallback failed");
    usearch_free(index, &error);

    options.quantization = usearch_scalar_bf16_k;
    index = usearch_init(&options, &error);
    expect(!error && index, error);
    usearch_reserve(index, 3, &error);
    expect(!error, error);
    uint16_t const bf16_finite[2] = {0x3f80u, 0};
    uint16_t const bf16_nan[2] = {0x7fc1u, 0x3f80u};
    uint16_t const bf16_inf[2] = {0x7f80u, 0x3f80u};
    usearch_add(index, 1, bf16_finite, usearch_scalar_bf16_k, &error);
    usearch_add(index, 2, bf16_nan, usearch_scalar_bf16_k, &error);
    usearch_add(index, 3, bf16_inf, usearch_scalar_bf16_k, &error);
    expect(!error, error);
    found = usearch_search(index, bf16_finite, usearch_scalar_bf16_k, 3, keys, distances, &error);
    expect(!error && found > 0 && found <= 3, "Raw bf16 nonfinite member fallback failed");
    found = usearch_search(index, bf16_nan, usearch_scalar_bf16_k, 3, keys, distances, &error);
    expect(!error && found > 0 && found <= 3, "Raw bf16 nonfinite query fallback failed");
    usearch_free(index, &error);
}

void test_cosine_norm_cache(void) {
    printf("Test: Cosine norm cache...\n");
    size_t f32_cosine_memory = test_cosine_norm_cache_kind(
        usearch_scalar_f32_k, "tmp_cosine_norm_f32.usearch", "tmp_cosine_norm_f32_roundtrip.usearch", 2e-5f);
    size_t bf16_cosine_memory = test_cosine_norm_cache_kind(
        usearch_scalar_bf16_k, "tmp_cosine_norm_bf16.usearch", "tmp_cosine_norm_bf16_roundtrip.usearch", 2e-3f);

    // Without NumKong the library reports exactly "serial" and routes built-in metrics through the auto-vectorized
    // kernels, which are the only ones eligible for the norm sidecar. With NumKong compiled in, the built-in cosine
    // kernels come from NumKong and no sidecar may ever be allocated. Either way the accounting must be exact.
    bool const expect_sidecar = strcmp(usearch_hardware_acceleration_compiled(), "serial") == 0;
    size_t const expected_sidecar_bytes = expect_sidecar ? 8 * sizeof(uint32_t) : 0;

    usearch_error_t error = NULL;
    usearch_init_options_t options = create_options(3);
    options.metric_kind = usearch_metric_l2sq_k;
    options.quantization = usearch_scalar_f32_k;
    usearch_index_t changed_metric = usearch_init(&options, &error);
    expect(!error && changed_metric, error);
    usearch_reserve(changed_metric, 8, &error);
    expect(!error, error);
    size_t l2_reserved_memory = usearch_memory_usage(changed_metric, &error);
    expect_eq(f32_cosine_memory, l2_reserved_memory + expected_sidecar_bytes,
              "f32 cosine sidecar memory accounting is incorrect");
    expect_eq(bf16_cosine_memory, l2_reserved_memory + expected_sidecar_bytes,
              "bf16 cosine sidecar memory accounting is incorrect");

    float const callback_vectors[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 0}};
    for (size_t i = 0; i != 3; ++i)
        usearch_add(changed_metric, i + 1, callback_vectors[i], usearch_scalar_f32_k, &error);
    expect(!error, error);
    size_t memory_before_change = usearch_memory_usage(changed_metric, &error);
    usearch_change_metric_kind(changed_metric, usearch_metric_cos_k, &error);
    expect(!error, error);
    size_t memory_after_change = usearch_memory_usage(changed_metric, &error);
    expect_eq(memory_after_change, memory_before_change + expected_sidecar_bytes,
              "Changing to cosine reported an invalid sidecar size");
    usearch_key_t keys[3] = {0};
    float distances[3] = {0};
    size_t found = usearch_search(changed_metric, callback_vectors[0], usearch_scalar_f32_k, 3, keys, distances, &error);
    expect(!error && found == 3, "Changed-metric cosine search failed");
    // After the change the index must behave exactly like a cosine index built from scratch: the query {1,0,0}
    // is at distance 0 from key 1, orthogonal (distance 1) to key 2, and at distance 1 from the zero vector, key 3.
    expect_distance_for_key(keys, distances, found, 1, 0, 2e-5f);
    expect_distance_for_key(keys, distances, found, 2, 1, 2e-5f);
    expect_distance_for_key(keys, distances, found, 3, 1, 2e-5f);
    usearch_change_metric_kind(changed_metric, usearch_metric_l2sq_k, &error);
    expect(!error, error);
    expect_eq(usearch_memory_usage(changed_metric, &error), memory_before_change,
              "Changing away from cosine did not release the sidecar");
    usearch_free(changed_metric, &error);

    options.metric_kind = usearch_metric_cos_k;
    options.metric = &counted_cosine_f32;
    usearch_index_t callback_index = usearch_init(&options, &error);
    expect(!error && callback_index, error);
    usearch_reserve(callback_index, 8, &error);
    expect(!error, error);
    expect_eq(usearch_memory_usage(callback_index, &error), l2_reserved_memory,
              "Cosine-labelled callback incorrectly allocated a norm sidecar");
    for (size_t i = 0; i != 3; ++i)
        usearch_add(callback_index, i + 1, callback_vectors[i], usearch_scalar_f32_k, &error);
    expect(!error, error);
    counted_cosine_calls = 0;
    found = usearch_search(callback_index, callback_vectors[0], usearch_scalar_f32_k, 3, keys, distances, &error);
    expect(!error && found == 3, "Cosine callback search failed");
    expect(counted_cosine_calls > 0, "Built-in cosine cache captured a custom callback");
    usearch_free(callback_index, &error);

    test_nonfinite_cosine_fallback();
    printf("Test: Cosine norm cache - PASSED\n");
}

int main(int argc, char const* argv[]) {
    install_crash_handlers();
    printf("Running tests...\n");
    printf("USearch version: %s\n", usearch_version());

    size_t collection_sizes[] = {11, 512};
    size_t dimensions[] = {83, 2}; // Not all distance functions make sense for 1 dimensional data
    for (size_t index = 0; index < sizeof(collection_sizes) / sizeof(collection_sizes[0]); ++index) {
        for (size_t jdx = 0; jdx < sizeof(dimensions) / sizeof(dimensions[0]); ++jdx) {
            test_init(collection_sizes[index], dimensions[jdx]);
            test_add_vector(collection_sizes[index], dimensions[jdx]);
            test_find_vector(collection_sizes[index], dimensions[jdx]);
            test_get_vector(collection_sizes[index], dimensions[jdx]);
            test_remove_vector(collection_sizes[index], dimensions[jdx]);
            test_save_load(collection_sizes[index], dimensions[jdx]);
            test_view(collection_sizes[index], dimensions[jdx]);
            test_mini_float_quantizations(collection_sizes[index], dimensions[jdx]);
        }
    }

    test_cosine_norm_cache();

    (void)argc;
    (void)argv;
    return 0;
}
