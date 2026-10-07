/**
 *  @file c/test.c
 *  @author Ash Vardanian
 *  @date June 25, 2023
 *  @brief Unit tests for the pure-C ABI of USearch (`usearch/dense_index.h`).
 *
 *  Exercises the dense surface through the ABI every binding consumes: initialization and
 *  reservation, batch `add`, `search` and `multiply`, retrieval by key into an arena, and the
 *  contract of `usearch_outcome_t`. Every call here blocks, so its outcome is final once it
 *  returns. Scoped to the verbs `src/` instantiates today - the header runs ahead of the
 *  implementation, and a smoke test that cannot link proves nothing. The harness is
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

/** An angular `bf16` RAM index over @p dimensions -wide rows, every other optional field derived. */
usearch_dense_index_t create_index(size_t const dimensions) {
    usearch_outcome_t outcome = {0};
    usearch_dense_index_t index = NULL;
    usearch_dense_init_ram(&(usearch_dense_init_ram_t) {.index = &index,
                                                        .outcome = &outcome,
                                                        .metric = usearch_dense_metric_angular_k,
                                                        .quantization = usearch_scalar_bf16_k,
                                                        .dimensions = dimensions,
                                                        .connectivity = 16});
    expect(!outcome.message, outcome.message);
    expect(index != NULL, "Initialization returned no handle");
    return index;
}

/** A RAM executor with every field derived. */
usearch_executor_t create_executor(void) {
    usearch_outcome_t outcome = {0};
    usearch_executor_t executor = NULL;
    usearch_executor_for_ram(&(usearch_executor_for_ram_t) {.executor = &executor, .outcome = &outcome});
    expect(!outcome.message, outcome.message);
    return executor;
}

/** Admits the keys whose integer identity is even. */
usearch_admission_t admits_even(usearch_key_t key, void* state) {
    (void)state;
    return key.bytes[15] % 2 == 0 ? usearch_admit_k : usearch_reject_k;
}

/** Initializes an index twice, checking the outcome at each step, and reserves capacity,
 *  verifying that size, capacity, dimensions, and connectivity describe back as configured. */
void test_init(size_t const collection_size, size_t const dimensions) {
    printf("Test: Index Initialization... %zu vectors, %zu dimensions\n", collection_size, dimensions);

    usearch_outcome_t outcome = {0};
    usearch_dense_free(create_index(dimensions));
    usearch_dense_index_t const index = create_index(dimensions);
    usearch_executor_t const executor = create_executor();

    usearch_dense_reserve(&(usearch_dense_reserve_t) {.index = index,
                                                      .executor = executor,
                                                      .outcome = &outcome,
                                                      .capacity = collection_size});
    expect(!outcome.message, outcome.message);

    usearch_dense_index_description_t description = {0};
    usearch_dense_describe(&(usearch_dense_describe_t) {.index = index,
                                                        .executor = executor,
                                                        .outcome = &outcome,
                                                        .description = &description});
    expect(!outcome.message, outcome.message);
    expect_eq(description.size, 0, "Fresh index must be empty");
    expect(description.capacity >= collection_size, "Capacity must cover the reservation");
    expect_eq(description.metadata.dimensions, dimensions, "Dimensions must read back as configured");
    expect_eq(description.metadata.connectivity, 16, "Connectivity must read back as configured");

    usearch_executor_free(executor);
    usearch_dense_free(index);
    printf("Test: Index Initialization - PASSED\n");
}

/** Inserts a batch and queries every row back, which an exact self-search must answer with the
 *  row's own key. The batch is all-or-nothing, so a short size is a failure even without an
 *  error status. A predicate then narrows the same queries to the keys it admits. */
void test_add_and_search(size_t const collection_size, size_t const dimensions) {
    printf("Test: Add & Search... %zu vectors, %zu dimensions\n", collection_size, dimensions);

    usearch_outcome_t outcome = {0};
    usearch_dense_index_t const index = create_index(dimensions);
    usearch_executor_t const executor = create_executor();
    usearch_dense_reserve(&(usearch_dense_reserve_t) {.index = index,
                                                      .executor = executor,
                                                      .outcome = &outcome,
                                                      .capacity = collection_size});
    expect(!outcome.message, outcome.message);

    float* const data = create_vectors(collection_size, dimensions);
    usearch_key_t* const keys = (usearch_key_t*)malloc(collection_size * sizeof(usearch_key_t));
    expect(keys, "Failed to allocate keys");
    for (size_t row = 0; row != collection_size; ++row) keys[row] = key_of(row);

    size_t const row_bytes = dimensions * sizeof(float);
    usearch_dense_add(&(usearch_dense_add_t) {.index = index,
                                              .executor = executor,
                                              .outcome = &outcome,
                                              .count = collection_size,
                                              .keys = {keys, sizeof(usearch_key_t)},
                                              .vectors = {data, row_bytes, usearch_scalar_f32_k}});
    expect(!outcome.message, outcome.message);
    usearch_dense_index_description_t description = {0};
    usearch_dense_describe(&(usearch_dense_describe_t) {.index = index,
                                                        .executor = executor,
                                                        .outcome = &outcome,
                                                        .description = &description});
    expect(!outcome.message, outcome.message);
    expect_eq(description.size, collection_size, "A batch is all-or-nothing, so every row must be published");

    // One match per query, so the output matrices are one column wide.
    usearch_key_t* const found_keys = (usearch_key_t*)malloc(collection_size * sizeof(usearch_key_t));
    usearch_distance_t* const found_distances = (usearch_distance_t*)malloc(collection_size *
                                                                            sizeof(usearch_distance_t));
    size_t* const counts = (size_t*)malloc(collection_size * sizeof(size_t));
    expect(found_keys && found_distances && counts, "Failed to allocate outputs");

    usearch_dense_search(
        &(usearch_dense_search_t) {.index = index,
                                   .executor = executor,
                                   .outcome = &outcome,
                                   .count = collection_size,
                                   .queries = {data, row_bytes, usearch_scalar_f32_k},
                                   .matches = {found_keys, sizeof(usearch_key_t), found_distances,
                                               sizeof(usearch_distance_t), counts, collection_size, 1}});
    expect(!outcome.message, outcome.message);

    size_t recalled = 0;
    for (size_t row = 0; row != collection_size; ++row) {
        expect(counts[row] != 0, "Every query must find something in a non-empty index");
        usearch_key_t const expected = key_of(row);
        recalled += memcmp(found_keys[row].bytes, expected.bytes, sizeof(expected.bytes)) == 0;
    }
    // Approximate search may miss, but a self-query missing itself often means a broken graph.
    expect(recalled * 10 >= collection_size * 9, "Self-recall collapsed below 90%");

    usearch_dense_search(&(usearch_dense_search_t) {.index = index,
                                                    .executor = executor,
                                                    .outcome = &outcome,
                                                    .count = collection_size,
                                                    .queries = {data, row_bytes, usearch_scalar_f32_k},
                                                    .matches = {found_keys, sizeof(usearch_key_t), found_distances,
                                                                sizeof(usearch_distance_t), counts, collection_size, 1},
                                                    .predicate = admits_even});
    expect(!outcome.message, outcome.message);
    for (size_t row = 0; row != collection_size; ++row)
        expect(counts[row] == 0 || found_keys[row].bytes[15] % 2 == 0, "A predicate must narrow the matches");

    free(counts);
    free(found_distances);
    free(found_keys);
    free(keys);
    free(data);
    usearch_executor_free(executor);
    usearch_dense_free(index);
    printf("Test: Add & Search - PASSED\n");
}

/** Reads entries back by key. `usearch_dense_get` sizes its own output, landing every row under
 *  the key in an arena and reporting how many there were. */
void test_count_and_get(size_t const dimensions) {
    printf("Test: Count & Get... %zu dimensions\n", dimensions);

    usearch_outcome_t outcome = {0};
    usearch_dense_index_t const index = create_index(dimensions);
    usearch_executor_t const executor = create_executor();
    usearch_dense_reserve(
        &(usearch_dense_reserve_t) {.index = index, .executor = executor, .outcome = &outcome, .capacity = 2});
    expect(!outcome.message, outcome.message);

    float* const data = create_vectors(2, dimensions);
    usearch_key_t const keys[2] = {key_of(7), key_of(8)};
    size_t const row_bytes = dimensions * sizeof(float);
    usearch_dense_add(&(usearch_dense_add_t) {.index = index,
                                              .executor = executor,
                                              .outcome = &outcome,
                                              .count = 2,
                                              .keys = {keys, sizeof(usearch_key_t)},
                                              .vectors = {data, row_bytes, usearch_scalar_f32_k}});
    expect(!outcome.message, outcome.message);

    size_t entries = 0;
    usearch_dense_count(&(usearch_dense_count_t) {.index = index,
                                                  .executor = executor,
                                                  .outcome = &outcome,
                                                  .key = key_of(7),
                                                  .entries = &entries});
    expect(!outcome.message, outcome.message);
    expect_eq(entries, 1, "The key was inserted once");
    usearch_dense_count(&(usearch_dense_count_t) {.index = index,
                                                  .executor = executor,
                                                  .outcome = &outcome,
                                                  .key = key_of(9),
                                                  .entries = &entries});
    expect(!outcome.message, outcome.message);
    expect_eq(entries, 0, "The key was never inserted");

    float* retrieved = NULL;
    size_t written = 0;
    usearch_arena_t arena = NULL;
    usearch_dense_get(&(usearch_dense_get_t) {.index = index,
                                              .executor = executor,
                                              .outcome = &outcome,
                                              .key = key_of(7),
                                              .scalar = usearch_scalar_f32_k,
                                              .vectors = (void**)&retrieved,
                                              .written = &written,
                                              .arena = &arena});
    expect(!outcome.message, outcome.message);
    expect_eq(written, 1, "Retrieval must report the true entry count");
    expect(retrieved != NULL, "Retrieval must land the rows in the arena");
    for (size_t position = 0; position != dimensions; ++position)
        expect(fabsf(retrieved[position] - data[position]) <= fabsf(data[position]) / 128,
               "A bf16 index must return rows within its 7-bit mantissa");

    char const path[] = "usearch-test-c-view.safegraphs";
    usearch_bytes_view_t const path_view = {(uint8_t const*)path, sizeof(path) - 1};
    usearch_dense_save(
        &(usearch_dense_save_t) {.index = index, .executor = executor, .outcome = &outcome, .path = path_view});
    expect(!outcome.message, outcome.message);
    usearch_dense_index_t view = NULL;
    usearch_dense_view_ram(&(usearch_dense_view_ram_t) {.index = &view, .outcome = &outcome, .path = path_view});
    // An implementation that cannot serve a mapping yet says so, and has no view to hold to the contract.
    if (outcome.status != usearch_status_unsupported_k) {
        expect(!outcome.message, outcome.message);
        usearch_dense_add(&(usearch_dense_add_t) {.index = view,
                                                  .executor = executor,
                                                  .outcome = &outcome,
                                                  .count = 1,
                                                  .keys = {keys, sizeof(usearch_key_t)},
                                                  .vectors = {data, row_bytes, usearch_scalar_f32_k}});
        expect_eq(outcome.status, usearch_status_read_only_k, "A view must refuse every mutation");
        usearch_dense_free(view);
    }
    remove(path);

    usearch_arena_free(arena);
    free(data);
    usearch_executor_free(executor);
    usearch_dense_free(index);
    printf("Test: Count & Get - PASSED\n");
}

/** A packed kind's row is as wide as its logical dimensions, so every product spans all of them -
 *  a row read at its storage width is either refused or scored over half its dimensions. */
void test_multiply_packed(void) {
    printf("Test: Multiply Packed Rows\n");

    enum { rows_count = 8, dimensions = 16 };
    usearch_outcome_t outcome = {0};
    usearch_dense_index_t index = NULL;
    usearch_dense_init_ram(&(usearch_dense_init_ram_t) {.index = &index,
                                                        .outcome = &outcome,
                                                        .metric = usearch_dense_metric_inner_k,
                                                        .quantization = usearch_scalar_i4x2_k,
                                                        .dimensions = dimensions,
                                                        .connectivity = 16});
    expect(!outcome.message, outcome.message);
    usearch_executor_t const executor = create_executor();
    usearch_dense_reserve(
        &(usearch_dense_reserve_t) {.index = index, .executor = executor, .outcome = &outcome, .capacity = rows_count});
    expect(!outcome.message, outcome.message);

    // Small integers survive the cast into `i4` exactly, so every product is known up front.
    float data[rows_count * dimensions];
    usearch_key_t keys[rows_count];
    for (size_t row = 0; row != rows_count; ++row) {
        keys[row] = key_of(row);
        for (size_t column = 0; column != dimensions; ++column)
            data[row * dimensions + column] = (float)((row + column) % 5) - 2;
    }
    usearch_dense_add(&(usearch_dense_add_t) {.index = index,
                                              .executor = executor,
                                              .outcome = &outcome,
                                              .count = rows_count,
                                              .keys = {keys, sizeof(usearch_key_t)},
                                              .vectors = {data, dimensions * sizeof(float), usearch_scalar_f32_k}});
    expect(!outcome.message, outcome.message);

    size_t offsets[rows_count + 1];
    usearch_key_t* columns = NULL;
    usearch_distance_t* products = NULL;
    usearch_arena_t arena = NULL;
    usearch_dense_multiply(
        &(usearch_dense_multiply_t) {.index = index,
                                     .executor = executor,
                                     .outcome = &outcome,
                                     .count = rows_count,
                                     .left = {data, dimensions * sizeof(float), usearch_scalar_f32_k},
                                     .offsets = offsets,
                                     .keys = &columns,
                                     .scores = &products,
                                     .arena = &arena});
    expect(!outcome.message, outcome.message);

    for (size_t row = 0; row != rows_count; ++row) {
        expect(offsets[row + 1] > offsets[row], "Every walk scores at least its entry point");
        for (size_t pair = offsets[row]; pair != offsets[row + 1]; ++pair) {
            size_t const named = columns[pair].bytes[15];
            double exact = 0;
            for (size_t column = 0; column != dimensions; ++column)
                exact += data[row * dimensions + column] * data[named * dimensions + column];
            expect(products[pair] == exact, "A product must span every packed dimension");
        }
    }

    // A NULL executor runs the call on this thread, and a row still inside the arena it rewinds is refused.
    usearch_dense_multiply(
        &(usearch_dense_multiply_t) {.index = index,
                                     .outcome = &outcome,
                                     .count = 1,
                                     .left = {data, dimensions * sizeof(float), usearch_scalar_f32_k},
                                     .offsets = offsets,
                                     .keys = &columns,
                                     .scores = &products,
                                     .arena = &arena});
    expect(!outcome.message, outcome.message);
    usearch_dense_multiply(&(usearch_dense_multiply_t) {.index = index,
                                                        .outcome = &outcome,
                                                        .count = 1,
                                                        .left = {columns, sizeof(usearch_key_t), usearch_scalar_u8_k},
                                                        .offsets = offsets,
                                                        .keys = &columns,
                                                        .scores = &products,
                                                        .arena = &arena});
    expect_eq(outcome.status, usearch_status_invalid_argument_k, "An input inside the rewound arena is refused");

    usearch_arena_free(arena);
    usearch_executor_free(executor);
    usearch_dense_free(index);
    printf("Test: Multiply Packed Rows - PASSED\n");
}

/** A message is @c NULL exactly when the status is success, and the outcome a caller reuses is
 *  fully overwritten - the contract every binding branches on. Only the success half is
 *  exercised here; the failing verbs are the ones `src/` has yet to instantiate. */
void test_outcome_contract(size_t const dimensions) {
    printf("Test: Outcome Contract\n");

    // An outcome carrying an earlier failure must come back clean, not merely unchanged.
    usearch_outcome_t outcome = {usearch_status_io_k, "a stale message the call must overwrite"};

    usearch_dense_index_t index = NULL;
    usearch_dense_init_ram(&(usearch_dense_init_ram_t) {.index = &index,
                                                        .outcome = &outcome,
                                                        .metric = usearch_dense_metric_angular_k,
                                                        .quantization = usearch_scalar_bf16_k,
                                                        .dimensions = dimensions});
    expect(index != NULL, "Initialization returned no handle");
    expect_eq(outcome.status, usearch_status_success_k, "A successful call must report success");
    expect(outcome.message == NULL, "A successful call must carry no message");

    usearch_executor_t const executor = create_executor();
    outcome = (usearch_outcome_t) {usearch_status_io_k, "a stale message the call must overwrite"};
    usearch_dense_reserve(
        &(usearch_dense_reserve_t) {.index = index, .executor = executor, .outcome = &outcome, .capacity = 4});
    expect_eq(outcome.status, usearch_status_success_k, "A successful call must report success");
    expect(outcome.message == NULL, "A successful call must carry no message");

    // A key the index never saw is a state, not a failure.
    size_t entries = 1;
    usearch_dense_count(&(usearch_dense_count_t) {.index = index,
                                                  .executor = executor,
                                                  .outcome = &outcome,
                                                  .key = key_of(4242),
                                                  .entries = &entries});
    expect_eq(entries, 0, "An absent key counts zero");
    expect_eq(outcome.status, usearch_status_success_k, "An absent key is not an error");

    usearch_executor_free(executor);
    usearch_dense_free(index);
    printf("Test: Outcome Contract - PASSED\n");
}

/** The library-wide probes, which take no handle and cannot fail. */
void test_library_probes(void) {
    printf("Test: Library Probes\n");
    int major = -1, minor = -1, patch = -1;
    usearch_version(&major, &minor, &patch);
    expect_eq(major, USEARCH_VERSION_MAJOR, "The linked major version must match the header's");
    expect(minor >= 0 && patch >= 0, "The version must be reported");
    expect(usearch_version_name() != NULL, "The version must be reported");
    expect(usearch_capabilities_compiled() != NULL, "The compiled capabilities must be reported");
    expect(usearch_capabilities_available() != NULL, "The available capabilities must be reported");
    usearch_status_t const unnamed = usearch_status_from_name((usearch_bytes_view_t) {(uint8_t const*)"no-such", 7});
    expect_eq(unnamed, usearch_status_invalid_argument_k, "An unrecognized status name is an invalid argument");
    printf("Test: Library Probes - PASSED\n");
}

int main(int argc, char* argv[]) {
    (void)argc;
    (void)argv;
    install_crash_handlers();

    printf("USearch %s\n", usearch_version_name());
    printf("- Compiled for: %s\n", usearch_capabilities_compiled());
    printf("- This machine: %s\n", usearch_capabilities_available());

    test_library_probes();
    test_outcome_contract(64);

    size_t const collection_sizes[] = {11, 512};
    size_t const dimension_counts[] = {83, 256};
    for (size_t size_index = 0; size_index != 2; ++size_index)
        for (size_t dimension_index = 0; dimension_index != 2; ++dimension_index) {
            test_init(collection_sizes[size_index], dimension_counts[dimension_index]);
            test_add_and_search(collection_sizes[size_index], dimension_counts[dimension_index]);
        }
    test_count_and_get(dimension_counts[0]);
    test_multiply_packed();

    printf("All tests passed!\n");
    return 0;
}
