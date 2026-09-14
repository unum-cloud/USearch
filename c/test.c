/**
 *  @file c/test.c
 *  @author Ash Vardanian
 *  @date June 25, 2023
 *  @brief Unit tests for the pure-C ABI of USearch (`usearch/dense_index.h`).
 *
 *  Exercises the dense surface through the ABI every binding consumes: initialization and
 *  reservation, batch `add` and `search`, retrieval by key, and the error contract of
 *  `usearch_error_t`. Scoped to the verbs `src/` instantiates today - the header runs ahead of
 *  the implementation, and a smoke test that cannot link proves nothing. The harness is
 *  intentionally dependency-free so it can run in the same matrix as the C++ tests and on
 *  cross-compilation targets where only the C runtime is available.
 *
 *  On startup we install a signal handler (see `install_crash_handlers`) that
 *  prints a native back-trace before re-raising, so CI logs pinpoint the
 *  faulting frame instead of stopping at a bare exit code.
 */
#include <math.h>    // `fabsf`
#include <signal.h>  // `signal`, `raise`, `SIGSEGV`
#include <stdbool.h> // `bool`
#include <stdint.h>  // `uint8_t`, `uint64_t`
#include <stdio.h>   // `printf`
#include <stdlib.h>  // `malloc`, `free`
#include <string.h>  // `memset`, `memcmp`

/* `dbghelp.h` needs types `windows.h` defines, so a blank line keeps clang-format from sorting it first. */
#if defined(_WIN32)
#include <windows.h>

#include <dbghelp.h> // `CaptureStackBackTrace`, `SymFromAddr`
#pragma comment(lib, "Dbghelp.lib")
#elif __has_include(<execinfo.h>)
#include <execinfo.h> // `backtrace`, `backtrace_symbols_fd`
#include <unistd.h>   // `STDERR_FILENO`
#endif

#include <usearch/dense_index.h>

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
        else fprintf(stderr, "  #%2u %p\n", (unsigned)frame_index, backtrace_frames[frame_index]);
    }
#elif __has_include(<execinfo.h>)
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
    if (must_be_true) return;
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
    for (size_t index = 0; index < count * dimensions; ++index) data[index] = (float)rand() / (float)RAND_MAX;
    return data;
}

/** Wraps an integer identity into the 128-bit key the ABI takes, low bytes last. */
usearch_key_t key_of(uint64_t value) {
    usearch_key_t key;
    memset(key.bytes, 0, sizeof(key.bytes));
    for (unsigned byte_index = 0; byte_index != 8; ++byte_index)
        key.bytes[15 - byte_index] = (uint8_t)(value >> (byte_index * 8));
    return key;
}

/** An angular RAM index over @p dimensions -wide `f32` rows, every optional field derived. */
usearch_dense_options_t create_options(size_t const dimensions) {
    usearch_dense_options_t options;
    memset(&options, 0, sizeof(options));
    options.metric = usearch_dense_metric_angular_k;
    options.quantization = usearch_scalar_bf16_k;
    options.dimensions = dimensions;
    options.connectivity = 16;
    options.residency = usearch_residency_ram_k;
    return options;
}

/** Initializes an index twice, checking the error slot at each step, and reserves capacity,
 *  verifying that size, capacity, dimensions, and connectivity read back as configured. */
void test_init(size_t const collection_size, size_t const dimensions) {
    printf("Test: Index Initialization... %zu vectors, %zu dimensions\n", collection_size, dimensions);

    usearch_dense_options_t options = create_options(dimensions);
    usearch_error_t error = {0};

    usearch_dense_index_t index = usearch_dense_init(&options, &error);
    expect(!error.message, error.message);
    expect(index != NULL, "Initialization returned no handle");
    usearch_dense_free(index);

    index = usearch_dense_init(&options, &error);
    expect(!error.message, error.message);

    usearch_dense_reserve(index, collection_size, &error);
    expect(!error.message, error.message);

    expect_eq(usearch_dense_size(index, &error), 0, "Fresh index must be empty");
    expect(!error.message, error.message);
    expect(usearch_dense_capacity(index, &error) >= collection_size, "Capacity must cover the reservation");
    expect(!error.message, error.message);
    expect_eq(usearch_dense_dimensions(index, &error), dimensions, "Dimensions must read back as configured");
    expect(!error.message, error.message);
    expect_eq(usearch_dense_connectivity(index, &error), options.connectivity,
              "Connectivity must read back as configured");
    expect(!error.message, error.message);

    usearch_dense_free(index);
    printf("Test: Index Initialization - PASSED\n");
}

/** Inserts a batch and queries every row back, which an exact self-search must answer with the
 *  row's own key. The batch is all-or-nothing, so a short return is a failure even without an
 *  error status. */
void test_add_and_search(size_t const collection_size, size_t const dimensions) {
    printf("Test: Add & Search... %zu vectors, %zu dimensions\n", collection_size, dimensions);

    usearch_dense_options_t options = create_options(dimensions);
    usearch_error_t error = {0};
    usearch_dense_index_t index = usearch_dense_init(&options, &error);
    expect(!error.message, error.message);
    usearch_dense_reserve(index, collection_size, &error);
    expect(!error.message, error.message);
    usearch_executor_t const executor = usearch_executor_for_ram(&options.ram, 0, &error);
    expect(!error.message, error.message);

    float* const data = create_vectors(collection_size, dimensions);
    usearch_key_t* const keys = (usearch_key_t*)malloc(collection_size * sizeof(usearch_key_t));
    expect(keys, "Failed to allocate keys");
    for (size_t row = 0; row != collection_size; ++row) keys[row] = key_of(row);

    size_t const row_bytes = dimensions * sizeof(float);
    usearch_keys_view_t const key_run = {keys, collection_size};
    usearch_matrix_view_t const rows = {data, collection_size, dimensions, row_bytes, usearch_scalar_f32_k};
    size_t const inserted = usearch_dense_add(index, executor, key_run, &rows, NULL, NULL, NULL, NULL, &error);
    expect(!error.message, error.message);
    expect_eq(inserted, collection_size, "A batch is all-or-nothing");
    expect_eq(usearch_dense_size(index, &error), collection_size, "Every row must be published");

    // One match per query, so the output matrices are one column wide.
    usearch_key_t* const found_keys = (usearch_key_t*)malloc(collection_size * sizeof(usearch_key_t));
    usearch_distance_t* const found_distances = (usearch_distance_t*)malloc(collection_size *
                                                                            sizeof(usearch_distance_t));
    size_t* const counts = (size_t*)malloc(collection_size * sizeof(size_t));
    expect(found_keys && found_distances && counts, "Failed to allocate outputs");

    usearch_matches_span_t const matches = {
        found_keys, sizeof(usearch_key_t), found_distances, sizeof(usearch_distance_t), counts, 1};
    usearch_dense_search(index, executor, &rows, &matches, NULL, NULL, NULL, NULL, &error);
    expect(!error.message, error.message);

    size_t recalled = 0;
    for (size_t row = 0; row != collection_size; ++row) {
        expect(counts[row] != 0, "Every query must find something in a non-empty index");
        usearch_key_t const expected = key_of(row);
        recalled += memcmp(found_keys[row].bytes, expected.bytes, sizeof(expected.bytes)) == 0;
    }
    // Approximate search may miss, but a self-query missing itself often means a broken graph.
    expect(recalled * 10 >= collection_size * 9, "Self-recall collapsed below 90%");

    free(counts);
    free(found_distances);
    free(found_keys);
    free(keys);
    free(data);
    usearch_executor_free(executor);
    usearch_dense_free(index);
    printf("Test: Add & Search - PASSED\n");
}

/** Reads entries back by key. `usearch_dense_get` reports the true entry count rather than the
 *  number written, so a probe with a NULL output is how a caller sizes its buffer. */
void test_count_and_get(size_t const dimensions) {
    printf("Test: Count & Get... %zu dimensions\n", dimensions);

    usearch_dense_options_t options = create_options(dimensions);
    usearch_error_t error = {0};
    usearch_dense_index_t index = usearch_dense_init(&options, &error);
    expect(!error.message, error.message);
    usearch_dense_reserve(index, 2, &error);
    expect(!error.message, error.message);
    usearch_executor_t const executor = usearch_executor_for_ram(&options.ram, 0, &error);
    expect(!error.message, error.message);

    float* const data = create_vectors(2, dimensions);
    usearch_key_t const keys[2] = {key_of(7), key_of(8)};
    size_t const row_bytes = dimensions * sizeof(float);
    usearch_keys_view_t const key_run = {keys, 2};
    usearch_matrix_view_t const rows = {data, 2, dimensions, row_bytes, usearch_scalar_f32_k};
    usearch_dense_add(index, executor, key_run, &rows, NULL, NULL, NULL, NULL, &error);
    expect(!error.message, error.message);

    expect_eq(usearch_dense_count(index, key_of(7), &error), 1, "The key was inserted once");
    expect(!error.message, error.message);
    expect_eq(usearch_dense_count(index, key_of(9), &error), 0, "The key was never inserted");
    expect(!error.message, error.message);

    expect_eq(usearch_dense_get(index, key_of(7), NULL, &error), 1, "A NULL output probes the count");
    expect(!error.message, error.message);

    float* const retrieved = (float*)malloc(row_bytes);
    expect(retrieved, "Failed to allocate the retrieval buffer");
    usearch_matrix_span_t const retrieved_rows = {retrieved, 1, dimensions, row_bytes, usearch_scalar_f32_k};
    expect_eq(usearch_dense_get(index, key_of(7), &retrieved_rows, &error), 1,
              "Retrieval must report the true entry count");
    expect(!error.message, error.message);
    for (size_t position = 0; position != dimensions; ++position)
        expect(fabsf(retrieved[position] - data[position]) <= fabsf(data[position]) / 128,
               "A bf16 index must return rows within its 7-bit mantissa");

    free(retrieved);
    free(data);
    usearch_executor_free(executor);
    usearch_dense_free(index);
    printf("Test: Count & Get - PASSED\n");
}

/** A message is @c NULL exactly when the status is success, and the slot a caller reuses is
 *  fully overwritten - the contract every binding branches on. Only the success half is
 *  exercised here; the failing verbs are the ones `src/` has yet to instantiate. */
void test_error_contract(size_t const dimensions) {
    printf("Test: Error Contract\n");

    usearch_dense_options_t options = create_options(dimensions);
    // A slot carrying an earlier failure must come back clean, not merely unchanged.
    usearch_error_t error = {usearch_error_io_k, "a stale message the call must overwrite"};

    usearch_dense_index_t const index = usearch_dense_init(&options, &error);
    expect(index != NULL, "Initialization returned no handle");
    expect_eq(error.status, usearch_success_k, "A successful call must report success");
    expect(error.message == NULL, "A successful call must carry no message");

    usearch_dense_reserve(index, 4, &error);
    expect_eq(error.status, usearch_success_k, "A successful call must report success");
    expect(error.message == NULL, "A successful call must carry no message");

    // A key the index never saw is a state, not a failure.
    expect_eq(usearch_dense_count(index, key_of(4242), &error), 0, "An absent key counts zero");
    expect_eq(error.status, usearch_success_k, "An absent key is not an error");

    usearch_dense_free(index);
    printf("Test: Error Contract - PASSED\n");
}

/** The library-wide probes, which take no handle and cannot fail. */
void test_library_probes(void) {
    printf("Test: Library Probes\n");
    expect(usearch_version() != NULL, "The version must be reported");
    expect(usearch_capabilities_compiled() != NULL, "The compiled capabilities must be reported");
    expect(usearch_capabilities_available() != NULL, "The available capabilities must be reported");
    printf("Test: Library Probes - PASSED\n");
}

int main(int argc, char* argv[]) {
    (void)argc;
    (void)argv;
    install_crash_handlers();

    printf("USearch C ABI tests: %s\n", usearch_version());
    printf("  compiled: %s\n", usearch_capabilities_compiled());
    printf("  available: %s\n", usearch_capabilities_available());

    test_library_probes();
    test_error_contract(64);

    size_t const collection_sizes[] = {11, 512};
    size_t const dimension_counts[] = {83, 256};
    for (size_t size_index = 0; size_index != 2; ++size_index)
        for (size_t dimension_index = 0; dimension_index != 2; ++dimension_index) {
            test_init(collection_sizes[size_index], dimension_counts[dimension_index]);
            test_add_and_search(collection_sizes[size_index], dimension_counts[dimension_index]);
        }
    test_count_and_get(dimension_counts[0]);

    printf("All tests passed!\n");
    return 0;
}
