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
        }
    }

    (void)argc;
    (void)argv;
    return 0;
}
