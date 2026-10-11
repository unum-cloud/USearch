/**
 *  @file c/test.c
 *  @author Ash Vardanian
 *  @date June 25, 2023
 *  @brief Unit tests for the pure-C ABI of USearch (`usearch/dense_index.h`).
 *
 *  Exercises the dense surface through the ABI every binding consumes: initialization and
 *  reservation, batch `add`, `search`, `multiply` and `attend`, retrieval by key, the arenas every
 *  output lands in, the contract of `usearch_outcome_t`, and deferred calls on an executor. A blocking call's outcome
 * is final once it returns; a deferred one is waited on or drained. Scoped to the verbs `src/` instantiates today - the
 * header runs ahead of the implementation, and a smoke test that cannot link proves nothing. The harness is
 * intentionally dependency-free so it can run in the same matrix as the C++ tests and on cross-compilation targets
 * where only the C runtime is available.
 *
 *  On startup we install a signal handler (see `install_crash_handlers`) that
 *  prints a native back-trace before re-raising, so CI logs pinpoint the
 *  faulting frame instead of stopping at a bare exit code.
 */
#include <math.h>    // `fabsf`, `fabs`
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

/** A RAM executor tracking @p deferred_limit deferred calls, every other field derived. */
usearch_executor_t create_executor(size_t const deferred_limit) {
    usearch_outcome_t outcome = {0};
    usearch_executor_t executor = NULL;
    usearch_executor_for_ram(
        &(usearch_executor_for_ram_t) {.executor = &executor, .outcome = &outcome, .deferred_limit = deferred_limit});
    expect(!outcome.message, outcome.message);
    return executor;
}

/** An arena placed where @p executor computes, or host pages for calls on this thread when it is NULL. */
usearch_arena_t create_arena(usearch_executor_t const executor) {
    usearch_outcome_t outcome = {0};
    usearch_arena_t arena = NULL;
    usearch_arena_init(&(usearch_arena_init_t) {.arena = &arena, .executor = executor, .outcome = &outcome});
    expect(!outcome.message, outcome.message);
    expect(arena != NULL, "Arena initialization returned no handle");
    return arena;
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
    usearch_executor_t const executor = create_executor(0);

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
    usearch_executor_t const executor = create_executor(0);
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
    usearch_dense_insert(&(usearch_dense_insert_t) {.index = index,
                                                    .executor = executor,
                                                    .outcome = &outcome,
                                                    .count = collection_size,
                                                    .keys = {keys, sizeof(usearch_key_t)},
                                                    .vectors = {data, row_bytes, dimensions, usearch_scalar_f32_k}});
    expect(!outcome.message, outcome.message);
    usearch_dense_index_description_t description = {0};
    usearch_dense_describe(&(usearch_dense_describe_t) {.index = index,
                                                        .executor = executor,
                                                        .outcome = &outcome,
                                                        .description = &description});
    expect(!outcome.message, outcome.message);
    expect_eq(description.size, collection_size, "A batch is all-or-nothing, so every row must be published");

    // One match per query, so every row of the matches is one column wide.
    usearch_arena_t const arena = create_arena(executor);
    usearch_matches_view_t matches = {0};
    usearch_dense_search(&(usearch_dense_search_t) {.index = index,
                                                    .executor = executor,
                                                    .outcome = &outcome,
                                                    .count = collection_size,
                                                    .queries = {data, row_bytes, dimensions, usearch_scalar_f32_k},
                                                    .matches = &matches,
                                                    .arena = arena,
                                                    .wanted = 1});
    expect(!outcome.message, outcome.message);
    expect_eq(matches.columns, 1, "The matches must report the row width the call used");

    size_t recalled = 0;
    for (size_t row = 0; row != collection_size; ++row) {
        expect(matches.counts[row] != 0, "Every query must find something in a non-empty index");
        usearch_key_t const expected = key_of(row);
        recalled += memcmp(matches.keys[row].bytes, expected.bytes, sizeof(expected.bytes)) == 0;
    }
    // Approximate search may miss, but a self-query missing itself often means a broken graph.
    expect(recalled * 10 >= collection_size * 9, "Self-recall collapsed below 90%");

    usearch_dense_search(&(usearch_dense_search_t) {.index = index,
                                                    .executor = executor,
                                                    .outcome = &outcome,
                                                    .count = collection_size,
                                                    .queries = {data, row_bytes, dimensions, usearch_scalar_f32_k},
                                                    .matches = &matches,
                                                    .arena = arena,
                                                    .wanted = 1,
                                                    .predicate = admits_even});
    expect(!outcome.message, outcome.message);
    for (size_t row = 0; row != collection_size; ++row)
        expect(matches.counts[row] == 0 || matches.keys[row].bytes[15] % 2 == 0, "A predicate must narrow the matches");

    // A zero `wanted` lands only the counts.
    usearch_dense_search(&(usearch_dense_search_t) {.index = index,
                                                    .executor = executor,
                                                    .outcome = &outcome,
                                                    .count = collection_size,
                                                    .queries = {data, row_bytes, dimensions, usearch_scalar_f32_k},
                                                    .matches = &matches,
                                                    .arena = arena});
    expect(!outcome.message, outcome.message);
    for (size_t row = 0; row != collection_size; ++row)
        expect_eq(matches.counts[row], 0, "A zero `wanted` lands zero counts");

    usearch_arena_free(arena);
    free(keys);
    free(data);
    usearch_executor_free(executor);
    usearch_dense_free(index);
    printf("Test: Add & Search - PASSED\n");
}

/** What a progress handler heard over one call. */
typedef struct progress_reports_t {
    size_t calls;
    size_t completed;
    size_t total;
    bool ascending;
} progress_reports_t;

/** Records one report, checking the counts only grow and never pass the total. */
static usearch_continuation_t record_progress(size_t completed, size_t total, void* state) {
    progress_reports_t* const reports = (progress_reports_t*)state;
    reports->ascending = reports->ascending && completed > reports->completed && completed <= total;
    reports->calls++;
    reports->completed = completed;
    reports->total = total;
    return usearch_continue_k;
}

/** A batch the handler hears about in several steps still lands whole: every step is reported in
 *  order up to the whole batch, and every row is published. */
void test_add_progress(void) {
    printf("Test: Add Progress\n");

    // Wider than two of the engine's 4096-row steps, so its handler hears more than one report.
    enum { rows_count = 2 * 4096 + 1, dimensions = 16 };
    usearch_outcome_t outcome = {0};
    usearch_dense_index_t const index = create_index(dimensions);
    usearch_executor_t const executor = create_executor(0);
    usearch_dense_reserve(
        &(usearch_dense_reserve_t) {.index = index, .executor = executor, .outcome = &outcome, .capacity = rows_count});
    expect(!outcome.message, outcome.message);

    float* const data = create_vectors(rows_count, dimensions);
    usearch_key_t* const keys = (usearch_key_t*)malloc(rows_count * sizeof(usearch_key_t));
    expect(keys, "Failed to allocate keys");
    for (size_t row = 0; row != rows_count; ++row) keys[row] = key_of(row);

    progress_reports_t reports = {0, 0, 0, true};
    usearch_dense_insert(
        &(usearch_dense_insert_t) {.index = index,
                                   .executor = executor,
                                   .outcome = &outcome,
                                   .count = rows_count,
                                   .keys = {keys, sizeof(usearch_key_t)},
                                   .vectors = {data, dimensions * sizeof(float), dimensions, usearch_scalar_f32_k},
                                   .progress = record_progress,
                                   .progress_state = &reports});
    expect(!outcome.message, outcome.message);
    expect(reports.ascending && reports.calls > 1, "Every step must be reported, in order");
    expect(reports.completed == rows_count && reports.total == rows_count, "The last report must cover the batch");

    usearch_dense_index_description_t description = {0};
    usearch_dense_describe(&(usearch_dense_describe_t) {.index = index,
                                                        .executor = executor,
                                                        .outcome = &outcome,
                                                        .description = &description});
    expect(!outcome.message, outcome.message);
    expect_eq(description.size, rows_count, "A reported batch must publish every row");

    free(keys);
    free(data);
    usearch_executor_free(executor);
    usearch_dense_free(index);
    printf("Test: Add Progress - PASSED\n");
}

/** Reads entries back by key. `usearch_dense_get` sizes its own output, landing every row under
 *  the key in an arena and reporting how many there were. */
void test_count_and_get(size_t const dimensions) {
    printf("Test: Count & Get... %zu dimensions\n", dimensions);

    usearch_outcome_t outcome = {0};
    usearch_dense_index_t const index = create_index(dimensions);
    usearch_executor_t const executor = create_executor(0);
    usearch_dense_reserve(
        &(usearch_dense_reserve_t) {.index = index, .executor = executor, .outcome = &outcome, .capacity = 2});
    expect(!outcome.message, outcome.message);

    float* const data = create_vectors(2, dimensions);
    usearch_key_t const keys[2] = {key_of(7), key_of(8)};
    size_t const row_bytes = dimensions * sizeof(float);
    usearch_dense_insert(&(usearch_dense_insert_t) {.index = index,
                                                    .executor = executor,
                                                    .outcome = &outcome,
                                                    .count = 2,
                                                    .keys = {keys, sizeof(usearch_key_t)},
                                                    .vectors = {data, row_bytes, dimensions, usearch_scalar_f32_k}});
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

    usearch_rows_view_t vectors = {0};
    size_t written = 0;
    usearch_arena_t const arena = create_arena(executor);
    usearch_dense_get(&(usearch_dense_get_t) {.index = index,
                                              .executor = executor,
                                              .outcome = &outcome,
                                              .key = key_of(7),
                                              .vectors = &vectors,
                                              .written = &written,
                                              .arena = arena,
                                              .outputs_scalar = usearch_scalar_f32_k});
    expect(!outcome.message, outcome.message);
    expect_eq(written, 1, "Retrieval must report the true entry count");
    expect_eq(vectors.stride, row_bytes, "Retrieval must report the pitch of the rows it packed");
    expect_eq(vectors.columns, dimensions, "Retrieval must report the width of the rows it packed");
    expect(vectors.data != NULL, "Retrieval must land the rows in the arena");
    float const* const retrieved = (float const*)vectors.data;
    for (size_t position = 0; position != dimensions; ++position)
        expect(fabsf(retrieved[position] - data[position]) <= fabsf(data[position]) / 128,
               "A bf16 index must return rows within its 7-bit mantissa");

    usearch_dense_insert(
        &(usearch_dense_insert_t) {.index = index,
                                   .executor = executor,
                                   .outcome = &outcome,
                                   .count = 1,
                                   .keys = {keys, sizeof(usearch_key_t)},
                                   .vectors = {data, row_bytes, dimensions - 1, usearch_scalar_f32_k}});
    expect_eq(outcome.status, usearch_status_shape_mismatch_k, "Rows of another width than the frozen one are refused");

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
        usearch_dense_insert(
            &(usearch_dense_insert_t) {.index = view,
                                       .executor = executor,
                                       .outcome = &outcome,
                                       .count = 1,
                                       .keys = {keys, sizeof(usearch_key_t)},
                                       .vectors = {data, row_bytes, dimensions, usearch_scalar_f32_k}});
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

/** Arenas come from an executor or from none, and every verb lands its outputs in whichever one it
 *  is given: a NULL arena is refused, one arena serves different verbs in turn, and a failed call
 *  leaves its outputs NULL. Each call rewinds the arena, so the views of the call before it are
 *  dead from then on; that is a contract on the caller, never read back here. */
void test_arenas(size_t const dimensions) {
    printf("Test: Arenas... %zu dimensions\n", dimensions);

    usearch_outcome_t outcome = {0};
    usearch_dense_index_t const index = create_index(dimensions);
    usearch_executor_t const executor = create_executor(0);
    usearch_arena_t const placed = create_arena(executor);
    usearch_arena_t const local = create_arena(NULL);
    float* const data = create_vectors(2, dimensions);
    usearch_key_t const keys[2] = {key_of(7), key_of(8)};
    size_t const row_bytes = dimensions * sizeof(float);
    usearch_dense_reserve(
        &(usearch_dense_reserve_t) {.index = index, .executor = executor, .outcome = &outcome, .capacity = 2});
    expect(!outcome.message, outcome.message);
    usearch_dense_insert(&(usearch_dense_insert_t) {.index = index,
                                                    .executor = executor,
                                                    .outcome = &outcome,
                                                    .count = 2,
                                                    .keys = {keys, sizeof(usearch_key_t)},
                                                    .vectors = {data, row_bytes, dimensions, usearch_scalar_f32_k}});
    expect(!outcome.message, outcome.message);

    usearch_matches_view_t matches = {0};
    usearch_dense_search_t search = {.index = index,
                                     .executor = executor,
                                     .outcome = &outcome,
                                     .count = 1,
                                     .queries = {data, row_bytes, dimensions, usearch_scalar_f32_k},
                                     .matches = &matches,
                                     .wanted = 2};
    usearch_dense_search(&search);
    expect_eq(outcome.status, usearch_status_invalid_argument_k, "A NULL arena is refused, never created");
    expect(matches.keys == NULL && matches.counts == NULL, "A refused call leaves its outputs NULL");

    search.arena = placed;
    usearch_dense_search(&search);
    expect(!outcome.message, outcome.message);
    expect(matches.counts[0] == 2 && memcmp(matches.keys[0].bytes, keys[0].bytes, sizeof(usearch_key_t)) == 0,
           "A row must find itself first");

    // The get rewinds the arena the search wrote into, so `matches` is dead from here on.
    usearch_rows_view_t vectors = {0};
    size_t written = 0;
    usearch_dense_get_t get = {.index = index,
                               .executor = executor,
                               .outcome = &outcome,
                               .key = key_of(8),
                               .vectors = &vectors,
                               .written = &written,
                               .arena = placed,
                               .outputs_scalar = usearch_scalar_f32_k};
    usearch_dense_get(&get);
    expect(!outcome.message, outcome.message);
    expect(written == 1 && vectors.data != NULL, "Another verb must reuse the same arena");

    get.outputs_scalar = usearch_scalar_unknown_k;
    usearch_dense_get(&get);
    expect_eq(outcome.status, usearch_status_unknown_kind_k, "A kind naming no scalar is refused");
    expect(vectors.data == NULL && written == 0, "A failed call leaves its outputs NULL");

    // An arena made without an executor serves calls on this thread.
    search.executor = NULL, search.arena = local;
    usearch_dense_search(&search);
    expect(!outcome.message, outcome.message);
    expect_eq(matches.counts[0], 2, "An arena of host pages serves a call on the calling thread");

    usearch_arena_free(local);
    usearch_arena_free(placed);
    usearch_arena_free(NULL);
    free(data);
    usearch_executor_free(executor);
    usearch_dense_free(index);
    printf("Test: Arenas - PASSED\n");
}

/** A cursor or a stream keeps no hold on its index, so every pull names it, and a pull naming any
 *  other handle is refused before the walk moves. */
void test_walks_name_their_index(size_t const dimensions) {
    printf("Test: Walks Name Their Index\n");

    usearch_outcome_t outcome = {0};
    usearch_dense_index_t const index = create_index(dimensions);
    usearch_dense_index_t const other = create_index(dimensions);
    usearch_executor_t const executor = create_executor(0);
    size_t const row_bytes = dimensions * sizeof(float);
    float* const data = (float*)calloc(2 * dimensions, sizeof(float));
    data[0] = 1, data[dimensions + 1] = 1;
    usearch_key_t const keys[2] = {key_of(3), key_of(5)};
    usearch_dense_reserve(
        &(usearch_dense_reserve_t) {.index = index, .executor = executor, .outcome = &outcome, .capacity = 2});
    expect(!outcome.message, outcome.message);
    usearch_dense_insert(&(usearch_dense_insert_t) {.index = index,
                                                    .executor = executor,
                                                    .outcome = &outcome,
                                                    .count = 2,
                                                    .keys = {keys, sizeof(usearch_key_t)},
                                                    .vectors = {data, row_bytes, dimensions, usearch_scalar_f32_k}});
    expect(!outcome.message, outcome.message);

    usearch_dense_cursor_t cursor = NULL;
    usearch_dense_cursor_init(&(usearch_dense_cursor_init_t) {.cursor = &cursor, .index = index, .outcome = &outcome});
    expect(!outcome.message, outcome.message);
    usearch_arena_t const arena = create_arena(executor);
    usearch_keys_view_t walked = {0};
    size_t written = 0;
    usearch_dense_cursor_next(&(usearch_dense_cursor_next_t) {.index = other,
                                                              .executor = executor,
                                                              .outcome = &outcome,
                                                              .cursor = cursor,
                                                              .keys = &walked,
                                                              .written = &written,
                                                              .arena = arena,
                                                              .wanted = 4});
    expect_eq(outcome.status, usearch_status_invalid_argument_k, "A cursor pull must name the index it was opened on");
    expect(written == 0 && walked.data == NULL, "A refused pull hands out nothing");
    usearch_dense_cursor_next(&(usearch_dense_cursor_next_t) {.index = index,
                                                              .executor = executor,
                                                              .outcome = &outcome,
                                                              .cursor = cursor,
                                                              .keys = &walked,
                                                              .written = &written,
                                                              .arena = arena,
                                                              .wanted = 4});
    expect(!outcome.message, outcome.message);
    expect_eq(written, 2, "A refused pull must leave the walk where it was");
    expect(memcmp(walked.data[1].bytes, keys[1].bytes, sizeof(usearch_key_t)) == 0, "A pull lands keys in order");
    usearch_dense_cursor_free(cursor);

    usearch_dense_search_stream_t stream = NULL;
    usearch_dense_search_stream_init(
        &(usearch_dense_search_stream_init_t) {.stream = &stream,
                                               .index = index,
                                               .outcome = &outcome,
                                               .query = {data, 0, dimensions, usearch_scalar_f32_k}});
    // An implementation that cannot stream yet says so, and has no stream to hold to the contract.
    if (outcome.status != usearch_status_unsupported_k) {
        expect(!outcome.message, outcome.message);
        usearch_matches_view_t pulled = {0};
        usearch_dense_search_stream_next(&(usearch_dense_search_stream_next_t) {.index = other,
                                                                                .executor = executor,
                                                                                .outcome = &outcome,
                                                                                .stream = stream,
                                                                                .matches = &pulled,
                                                                                .arena = arena,
                                                                                .wanted = 2});
        expect_eq(outcome.status, usearch_status_invalid_argument_k,
                  "A stream pull must name the index it was opened on");
        usearch_dense_search_stream_next(&(usearch_dense_search_stream_next_t) {.index = index,
                                                                                .executor = executor,
                                                                                .outcome = &outcome,
                                                                                .stream = stream,
                                                                                .matches = &pulled,
                                                                                .arena = arena,
                                                                                .wanted = 2});
        expect(!outcome.message, outcome.message);
        expect_eq(pulled.counts[0], 2, "A refused pull must leave the stream where it was");
        usearch_dense_search_stream_free(stream);
    }

    usearch_arena_free(arena);
    free(data);
    usearch_executor_free(executor);
    usearch_dense_free(other);
    usearch_dense_free(index);
    printf("Test: Walks Name Their Index - PASSED\n");
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
    usearch_executor_t const executor = create_executor(0);
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
    usearch_dense_insert(
        &(usearch_dense_insert_t) {.index = index,
                                   .executor = executor,
                                   .outcome = &outcome,
                                   .count = rows_count,
                                   .keys = {keys, sizeof(usearch_key_t)},
                                   .vectors = {data, dimensions * sizeof(float), dimensions, usearch_scalar_f32_k}});
    expect(!outcome.message, outcome.message);

    usearch_matches_tape_view_t products = {0};
    usearch_arena_t const arena = create_arena(executor);
    usearch_dense_multiply(
        &(usearch_dense_multiply_t) {.index = index,
                                     .executor = executor,
                                     .outcome = &outcome,
                                     .count = rows_count,
                                     .left = {data, dimensions * sizeof(float), dimensions, usearch_scalar_f32_k},
                                     .products = &products,
                                     .arena = arena});
    expect(!outcome.message, outcome.message);

    for (size_t row = 0; row != rows_count; ++row) {
        expect(products.offsets[row + 1] > products.offsets[row], "Every walk scores at least its entry point");
        for (size_t pair = products.offsets[row]; pair != products.offsets[row + 1]; ++pair) {
            size_t const named = products.keys[pair].bytes[15];
            double exact = 0;
            for (size_t column = 0; column != dimensions; ++column)
                exact += data[row * dimensions + column] * data[named * dimensions + column];
            expect(products.scores[pair] == exact, "A product must span every packed dimension");
        }
    }

    // A NULL executor runs the call on this thread, and a row still inside the arena it rewinds is refused.
    usearch_dense_multiply(
        &(usearch_dense_multiply_t) {.index = index,
                                     .outcome = &outcome,
                                     .count = 1,
                                     .left = {data, dimensions * sizeof(float), dimensions, usearch_scalar_f32_k},
                                     .products = &products,
                                     .arena = arena});
    expect(!outcome.message, outcome.message);
    usearch_dense_multiply(
        &(usearch_dense_multiply_t) {.index = index,
                                     .outcome = &outcome,
                                     .count = 1,
                                     .left = {products.keys, sizeof(usearch_key_t), dimensions, usearch_scalar_u8_k},
                                     .products = &products,
                                     .arena = arena});
    expect_eq(outcome.status, usearch_status_invalid_argument_k, "An input inside the rewound arena is refused");
    expect(products.keys == NULL && products.offsets == NULL, "A refused call leaves its outputs NULL");

    usearch_arena_free(arena);
    usearch_executor_free(executor);
    usearch_dense_free(index);
    printf("Test: Multiply Packed Rows - PASSED\n");
}

/** The `int8_t` value table an attend test resolves keys through, its first @c rows resolvable. */
typedef struct attend_values_t {
    int8_t const* data;
    size_t rows;
    size_t columns;
    usearch_scalar_t scalar;
} attend_values_t;

/** Resolves a key made by `key_of` to its row of the table, NULL past the resolvable rows. */
static void attend_value_rows(void* state, usearch_key_t const* keys, size_t count, usearch_scalar_t scalar,
                              size_t columns, void const** rows) {
    attend_values_t const* const values = (attend_values_t const*)state;
    expect(scalar == values->scalar && columns == values->columns, "The lookup must learn the stored row shape");
    for (size_t key = 0; key != count; ++key) {
        size_t const token = keys[key].bytes[15];
        rows[key] = token < values->rows ? values->data + token * values->columns : NULL;
    }
}

/** A graph this small walks every key, so attention reproduces the exact softmax over all of them, with one
 *  head per walk and with two sharing it; heads that do not divide the rows, a key without a value row, a
 *  device row lookup or no host one, and a distance kind refuse the call. */
void test_attend(void) {
    printf("Test: Attend\n");

    enum { tokens = 4, dimensions = 4, heads = 2 };
    usearch_outcome_t outcome = {0};
    usearch_dense_index_t index = NULL;
    usearch_dense_init_ram(&(usearch_dense_init_ram_t) {.index = &index,
                                                        .outcome = &outcome,
                                                        .metric = usearch_dense_metric_inner_k,
                                                        .quantization = usearch_scalar_i8_k,
                                                        .dimensions = dimensions});
    expect(!outcome.message, outcome.message);
    usearch_executor_t const executor = create_executor(0);
    usearch_dense_reserve(
        &(usearch_dense_reserve_t) {.index = index, .executor = executor, .outcome = &outcome, .capacity = tokens});
    expect(!outcome.message, outcome.message);

    // Token `t` scores a dot of `t` against a query of ones, and its value row is `{t, 1, 0, 0}`.
    float rows[tokens * dimensions] = {0};
    int8_t table[tokens * dimensions] = {0};
    usearch_key_t keys[tokens];
    for (size_t token = 0; token != tokens; ++token) {
        keys[token] = key_of(token);
        rows[token * dimensions + token] = (float)token;
        table[token * dimensions] = (int8_t)token, table[token * dimensions + 1] = 1;
    }
    usearch_dense_insert(
        &(usearch_dense_insert_t) {.index = index,
                                   .executor = executor,
                                   .outcome = &outcome,
                                   .count = tokens,
                                   .keys = {keys, sizeof(usearch_key_t)},
                                   .vectors = {rows, dimensions * sizeof(float), dimensions, usearch_scalar_f32_k}});
    expect(!outcome.message, outcome.message);

    // A logit scale of ln 2 turns the ones' dots into weights 1/8, 1/4, 1/2 and 1, and the negated
    // head's into the same weights in reverse, both summing to 1.875.
    float const queries[heads * dimensions] = {1, 1, 1, 1, -1, -1, -1, -1};
    usearch_rows_view_t outputs = {0};
    usearch_dense_softmax_state_t carry[heads];
    attend_values_t values = {table, tokens, dimensions, usearch_scalar_i8_k};
    usearch_arena_t const arena = create_arena(executor);
    usearch_dense_attend_t attend = {.index = index,
                                     .executor = executor,
                                     .outcome = &outcome,
                                     .count = heads,
                                     .queries = {queries, dimensions * sizeof(float), dimensions, usearch_scalar_f32_k},
                                     .value_lookup = attend_value_rows,
                                     .value_lookup_state = &values,
                                     .outputs = &outputs,
                                     .carry = carry,
                                     .arena = arena,
                                     .score_scale = 0.69314718f};
    for (size_t head_count = 1; head_count <= heads; ++head_count) {
        attend.head_count = head_count;
        memset(carry, 0, sizeof(carry));
        usearch_dense_attend(&attend);
        expect(!outcome.message, outcome.message);
        expect(outputs.scalar == usearch_scalar_f32_k && outputs.columns == dimensions,
               "A zero output scalar lands the queries' own, as wide as the keys");
        float const* const attended = (float const*)outputs.data;
        size_t const pitch = outputs.stride / sizeof(float);
        expect(fabsf(carry[0].maximum_score - 3 * 0.69314718f) < 1e-5f && fabsf(carry[1].maximum_score) < 1e-5f,
               "Each head's carry must hold its own largest logit");
        expect(fabsf(carry[0].partition - 1.875f) < 1e-4f && fabsf(carry[1].partition - 1.875f) < 1e-4f,
               "Each head's carry must hold its partition over every key");
        expect(fabsf(attended[0] - 4.25f / 1.875f) < 1e-4f && fabsf(attended[pitch] - 1.375f / 1.875f) < 1e-4f,
               "Every key must weigh the value row it resolves to");
        expect(fabsf(attended[1] - 1) < 1e-4f && fabsf(attended[pitch + 1] - 1) < 1e-4f && attended[2] == 0 &&
                   attended[pitch + 3] == 0,
               "The attended rows must be normalized and as wide as the keys");
    }

    attend.head_count = 3;
    usearch_dense_attend(&attend);
    expect_eq(outcome.status, usearch_status_shape_mismatch_k, "Heads that do not divide the rows must refuse");
    expect(outputs.data == NULL, "A refused call leaves its outputs NULL");

    attend.head_count = heads;
    values.rows = tokens - 1;
    usearch_dense_attend(&attend);
    expect_eq(outcome.status, usearch_status_shape_mismatch_k, "A key without a value row must refuse the call");

    values.rows = tokens;
    attend.value_lookup = NULL;
    usearch_dense_attend(&attend);
    expect_eq(outcome.status, usearch_status_invalid_argument_k, "A missing value lookup must refuse the call");

    usearch_dense_index_t const angular = create_index(dimensions);
    attend.index = angular;
    attend.value_lookup = attend_value_rows;
    usearch_dense_attend(&attend);
    expect_eq(outcome.status, usearch_status_config_mismatch_k, "A distance kind must refuse attention");

    usearch_arena_free(arena);
    usearch_executor_free(executor);
    usearch_dense_free(angular);
    usearch_dense_free(index);
    printf("Test: Attend - PASSED\n");
}

/** A RAM index saved as one monolith serves the same keys and rows from storage: an NVMe executor
 *  views the file, its search, count, get, health and cursor answer as the RAM index did, and every
 *  mutation is refused. An implementation without an NVMe residency says so, and the test skips. */
void test_nvme_view(size_t const dimensions) {
    printf("Test: NVMe View... %zu dimensions\n", dimensions);

    enum { rows_count = 64 };
    usearch_outcome_t outcome = {0};
    usearch_dense_index_t const index = create_index(dimensions);
    usearch_executor_t const executor = create_executor(0);
    usearch_dense_reserve(
        &(usearch_dense_reserve_t) {.index = index, .executor = executor, .outcome = &outcome, .capacity = rows_count});
    expect(!outcome.message, outcome.message);
    float* const data = create_vectors(rows_count, dimensions);
    usearch_key_t keys[rows_count];
    for (size_t row = 0; row != rows_count; ++row) keys[row] = key_of(row);
    size_t const row_bytes = dimensions * sizeof(float);
    usearch_dense_insert(&(usearch_dense_insert_t) {.index = index,
                                                    .executor = executor,
                                                    .outcome = &outcome,
                                                    .count = rows_count,
                                                    .keys = {keys, sizeof(usearch_key_t)},
                                                    .vectors = {data, row_bytes, dimensions, usearch_scalar_f32_k}});
    expect(!outcome.message, outcome.message);
    char const path[] = "usearch-test-c-nvme.safegraphs";
    usearch_bytes_view_t const path_view = {(uint8_t const*)path, sizeof(path) - 1};
    usearch_dense_save(
        &(usearch_dense_save_t) {.index = index, .executor = executor, .outcome = &outcome, .path = path_view});
    expect(!outcome.message, outcome.message);

    usearch_executor_t storage = NULL;
    usearch_dense_index_t view = NULL;
    usearch_executor_for_nvme(&(usearch_executor_for_nvme_t) {.executor = &storage, .outcome = &outcome, .threads = 2});
    if (outcome.status == usearch_status_success_k)
        usearch_dense_view_nvme(
            &(usearch_dense_view_nvme_t) {.index = &view, .executor = storage, .outcome = &outcome, .path = path_view});
    // An implementation without an NVMe residency says so, and has no view to hold to the contract.
    if (outcome.status == usearch_status_unsupported_k) {
        printf("Test: NVMe View - SKIPPED: %s\n", outcome.message);
        usearch_executor_free(storage);
        remove(path);
        free(data);
        usearch_executor_free(executor);
        usearch_dense_free(index);
        return;
    }
    expect(!outcome.message, outcome.message);

    usearch_dense_index_description_t description = {0};
    usearch_dense_describe(&(usearch_dense_describe_t) {.index = view,
                                                        .executor = storage,
                                                        .outcome = &outcome,
                                                        .description = &description});
    expect(!outcome.message, outcome.message);
    expect_eq(description.size, rows_count, "A view must serve every saved row");
    expect_eq(description.capacity, rows_count, "A view hosts exactly the rows it serves");
    expect_eq(description.metadata.dimensions, dimensions, "A view must serve the saved geometry");

    usearch_dense_reachability_t reachability = {0};
    usearch_dense_health(&(usearch_dense_health_t) {.index = view,
                                                    .executor = storage,
                                                    .outcome = &outcome,
                                                    .reachability = &reachability});
    expect(!outcome.message, outcome.message);
    expect_eq(reachability.audited, rows_count, "A view must audit every saved row");

    usearch_arena_t const arena = create_arena(storage);
    usearch_matches_view_t matches = {0};
    usearch_dense_search(&(usearch_dense_search_t) {.index = view,
                                                    .executor = storage,
                                                    .outcome = &outcome,
                                                    .count = rows_count,
                                                    .queries = {data, row_bytes, dimensions, usearch_scalar_f32_k},
                                                    .matches = &matches,
                                                    .arena = arena,
                                                    .wanted = 1});
    expect(!outcome.message, outcome.message);
    size_t recalled = 0;
    for (size_t row = 0; row != rows_count; ++row)
        recalled += matches.counts[row] && memcmp(matches.keys[row].bytes, keys[row].bytes, sizeof(usearch_key_t)) == 0;
    expect(recalled * 10 >= rows_count * 9, "A viewed graph must find its own rows as the RAM one does");

    size_t entries = 0;
    usearch_dense_count(&(usearch_dense_count_t) {.index = view,
                                                  .executor = storage,
                                                  .outcome = &outcome,
                                                  .key = key_of(5),
                                                  .entries = &entries});
    expect(!outcome.message, outcome.message);
    expect_eq(entries, 1, "A saved key must count once in its view");
    usearch_dense_count(&(usearch_dense_count_t) {.index = view,
                                                  .executor = storage,
                                                  .outcome = &outcome,
                                                  .key = key_of(rows_count + 7),
                                                  .entries = &entries});
    expect(!outcome.message, outcome.message);
    expect_eq(entries, 0, "A key never saved must count zero");

    usearch_rows_view_t vectors = {0};
    size_t written = 0;
    usearch_dense_get(&(usearch_dense_get_t) {.index = view,
                                              .executor = storage,
                                              .outcome = &outcome,
                                              .key = key_of(5),
                                              .vectors = &vectors,
                                              .written = &written,
                                              .arena = arena,
                                              .outputs_scalar = usearch_scalar_f32_k});
    expect(!outcome.message, outcome.message);
    expect(written == 1 && vectors.data != NULL && vectors.columns == dimensions, "A view must read a key's row back");
    float const* const retrieved = (float const*)vectors.data;
    for (size_t position = 0; position != dimensions; ++position)
        expect(fabsf(retrieved[position] - data[5 * dimensions + position]) <=
                   fabsf(data[5 * dimensions + position]) / 128,
               "A bf16 view must return rows within its 7-bit mantissa");

    usearch_dense_cursor_t cursor = NULL;
    usearch_dense_cursor_init(&(usearch_dense_cursor_init_t) {.cursor = &cursor, .index = view, .outcome = &outcome});
    expect(!outcome.message, outcome.message);
    usearch_keys_view_t walked = {0};
    usearch_dense_cursor_next(&(usearch_dense_cursor_next_t) {.index = view,
                                                              .executor = storage,
                                                              .outcome = &outcome,
                                                              .cursor = cursor,
                                                              .keys = &walked,
                                                              .written = &written,
                                                              .arena = arena,
                                                              .wanted = rows_count});
    expect(!outcome.message, outcome.message);
    expect_eq(written, rows_count, "A view's cursor must read every saved row");
    usearch_dense_cursor_free(cursor);

    usearch_dense_insert(&(usearch_dense_insert_t) {.index = view,
                                                    .executor = storage,
                                                    .outcome = &outcome,
                                                    .count = 1,
                                                    .keys = {keys, sizeof(usearch_key_t)},
                                                    .vectors = {data, row_bytes, dimensions, usearch_scalar_f32_k}});
    expect_eq(outcome.status, usearch_status_read_only_k, "A view must refuse every mutation");
    usearch_dense_save(
        &(usearch_dense_save_t) {.index = view, .executor = storage, .outcome = &outcome, .path = path_view});
    expect_eq(outcome.status, usearch_status_unsupported_k, "A view already is its checkpoint");

    usearch_arena_free(arena);
    usearch_dense_free(view);
    usearch_executor_free(storage);
    remove(path);
    free(data);
    usearch_executor_free(executor);
    usearch_dense_free(index);
    printf("Test: NVMe View - PASSED\n");
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

    usearch_executor_t const executor = create_executor(0);
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
    expect(strcmp(usearch_status_name(usearch_status_deferred_k), "deferred") == 0,
           "The non-final status spells deferred");
    expect(strcmp(usearch_status_name((usearch_status_t)999), "unrecognized") == 0,
           "A value no status carries spells unrecognized");
    usearch_mood_t const spinning = usearch_mood_from_name((usearch_bytes_view_t) {(uint8_t const*)"spinning", 8});
    expect_eq(spinning, usearch_mood_spinning_k, "A mood reads back from its name");
    expect(strcmp(usearch_mood_name(usearch_mood_parked_k), "parked") == 0, "A mood spells its name");
    printf("Test: Library Probes - PASSED\n");
}

enum { deferred_rows = 128, deferred_dimensions = 16, deferred_wanted = 4 };

/** One search's outputs, the arena they land in, and its outcome, which a deferred call holds until final. */
typedef struct search_outputs_t {
    usearch_matches_view_t matches;
    usearch_arena_t arena;
    usearch_outcome_t outcome;
} search_outputs_t;

/** Whether two searches of every row landed the same matches. */
bool same_matches(usearch_matches_view_t const* first, usearch_matches_view_t const* second) {
    size_t const cells = deferred_rows * deferred_wanted;
    return memcmp(first->keys, second->keys, cells * sizeof(usearch_key_t)) == 0 &&
           memcmp(first->scores, second->scores, cells * sizeof(usearch_distance_t)) == 0 &&
           memcmp(first->counts, second->counts, deferred_rows * sizeof(size_t)) == 0;
}

/** An index holding @p rows under keys equal to their positions, filled through @p executor. */
usearch_dense_index_t create_filled_index(float const* rows, usearch_executor_t executor) {
    usearch_outcome_t outcome = {0};
    usearch_dense_index_t const index = create_index(deferred_dimensions);
    usearch_key_t keys[deferred_rows];
    for (size_t row = 0; row != deferred_rows; ++row) keys[row] = key_of(row);
    usearch_dense_reserve(&(usearch_dense_reserve_t) {.index = index,
                                                      .executor = executor,
                                                      .outcome = &outcome,
                                                      .capacity = deferred_rows});
    expect(!outcome.message, outcome.message);
    usearch_dense_insert(&(usearch_dense_insert_t) {
        .index = index,
        .executor = executor,
        .outcome = &outcome,
        .count = deferred_rows,
        .keys = {keys, sizeof(usearch_key_t)},
        .vectors = {rows, deferred_dimensions * sizeof(float), deferred_dimensions, usearch_scalar_f32_k}});
    expect(!outcome.message, outcome.message);
    return index;
}

/** Submits a search of every row of @p rows into @p outputs; its descriptor dies when this returns. */
void search_every_row(usearch_dense_index_t index, usearch_executor_t executor, usearch_completion_t completion,
                      float const* rows, search_outputs_t* outputs) {
    usearch_dense_search(&(usearch_dense_search_t) {
        .index = index,
        .executor = executor,
        .outcome = &outputs->outcome,
        .completion = completion,
        .count = deferred_rows,
        .queries = {rows, deferred_dimensions * sizeof(float), deferred_dimensions, usearch_scalar_f32_k},
        .matches = &outputs->matches,
        .arena = outputs->arena,
        .wanted = deferred_wanted});
}

/** Waits for every call submitted to @p executor. */
void drain(usearch_executor_t executor) {
    usearch_outcome_t outcome = {0};
    usearch_executor_drain(&(usearch_executor_drain_t) {.executor = executor, .outcome = &outcome});
    expect(!outcome.message, outcome.message);
}

/** A deferred search stays deferred until waited on, then lands what a blocking one does, though its
 *  descriptor died at submission; without an executor it is final when it returns. */
void test_deferred_search(void) {
    printf("Test: Deferred Search\n");
    float* const rows = create_vectors(deferred_rows, deferred_dimensions);
    usearch_executor_t const executor = create_executor(0);
    usearch_dense_index_t const index = create_filled_index(rows, executor);
    search_outputs_t blocking = {.arena = create_arena(executor)};
    search_outputs_t deferred = {.arena = create_arena(executor)};

    search_every_row(index, executor, usearch_completion_blocking_k, rows, &blocking);
    expect(!blocking.outcome.message, blocking.outcome.message);

    search_every_row(index, executor, usearch_completion_deferred_k, rows, &deferred);
    expect_eq(deferred.outcome.status, usearch_status_deferred_k, "A submitted call is deferred until waited on");
    expect_eq(usearch_outcome_wait(&deferred.outcome), usearch_status_success_k, "A waited search must finish");
    expect(same_matches(&blocking.matches, &deferred.matches),
           "A deferred search must land the matches a blocking one does");

    // An arena made by an executor serves a call without one, since both compute on the host.
    search_every_row(index, NULL, usearch_completion_deferred_k, rows, &deferred);
    expect_eq(deferred.outcome.status, usearch_status_success_k, "A call without an executor is final when it returns");
    expect(same_matches(&blocking.matches, &deferred.matches), "Every receiver must agree");

    usearch_arena_free(deferred.arena);
    usearch_arena_free(blocking.arena);
    usearch_dense_free(index);
    usearch_executor_free(executor);
    free(rows);
    printf("Test: Deferred Search - PASSED\n");
}

/** A submission past the deferred limit is refused at once, and a drain makes room again. */
void test_deferred_limit(void) {
    printf("Test: Deferred Limit\n");
    float* const rows = create_vectors(deferred_rows, deferred_dimensions);
    usearch_executor_t const executor = create_executor(1);
    usearch_dense_index_t const index = create_filled_index(rows, executor);
    search_outputs_t first = {.arena = create_arena(executor)};
    search_outputs_t second = {.arena = create_arena(executor)};

    search_every_row(index, executor, usearch_completion_deferred_k, rows, &first);
    search_every_row(index, executor, usearch_completion_deferred_k, rows, &second);
    expect_eq(second.outcome.status, usearch_status_contended_k, "A full deferred table must refuse at once");
    drain(executor);
    expect_eq(first.outcome.status, usearch_status_success_k, "The earlier call must finish at the drain");
    // The refused submission let go of its arena, so the retry may claim it.
    search_every_row(index, executor, usearch_completion_deferred_k, rows, &second);
    drain(executor);
    expect_eq(second.outcome.status, usearch_status_success_k, "A drained table takes a call again");

    usearch_arena_free(second.arena);
    usearch_arena_free(first.arena);
    usearch_dense_free(index);
    usearch_executor_free(executor);
    free(rows);
    printf("Test: Deferred Limit - PASSED\n");
}

/** Freeing an index finishes the deferred calls naming it before its memory goes. */
void test_free_finishes_deferred(void) {
    printf("Test: Free Finishes Deferred\n");
    float* const rows = create_vectors(deferred_rows, deferred_dimensions);
    usearch_executor_t const executor = create_executor(0);
    usearch_dense_index_t const index = create_filled_index(rows, executor);
    search_outputs_t outputs = {.arena = create_arena(executor)};

    search_every_row(index, executor, usearch_completion_deferred_k, rows, &outputs);
    usearch_dense_free(index);
    expect_eq(outputs.outcome.status, usearch_status_success_k, "A free must finish the calls naming the handle");
    for (size_t row = 0; row != deferred_rows; ++row)
        expect_eq(outputs.matches.counts[row], deferred_wanted, "A call finished by a free lands its matches");

    usearch_arena_free(outputs.arena);
    usearch_executor_free(executor);
    free(rows);
    printf("Test: Free Finishes Deferred - PASSED\n");
}

/** A blocking call on an executor runs every call submitted before it first. */
void test_blocking_runs_after_earlier(void) {
    printf("Test: Blocking Runs After Earlier\n");
    float* const rows = create_vectors(deferred_rows, deferred_dimensions);
    usearch_executor_t const executor = create_executor(0);
    usearch_dense_index_t const index = create_filled_index(rows, executor);
    search_outputs_t earlier = {.arena = create_arena(executor)};
    search_outputs_t later = {.arena = create_arena(executor)};

    search_every_row(index, executor, usearch_completion_deferred_k, rows, &earlier);
    search_every_row(index, executor, usearch_completion_blocking_k, rows, &later);
    expect_eq(earlier.outcome.status, usearch_status_success_k, "An earlier deferred call must finish first");

    usearch_arena_free(later.arena);
    usearch_arena_free(earlier.arena);
    usearch_dense_free(index);
    usearch_executor_free(executor);
    free(rows);
    printf("Test: Blocking Runs After Earlier - PASSED\n");
}

/** A deferred call holds its arena from submission until final, so another call naming it meanwhile is
 *  refused, and the arena serves again once the deferred call is waited on. */
void test_deferred_holds_arena(void) {
    printf("Test: Deferred Holds Arena\n");
    float* const rows = create_vectors(deferred_rows, deferred_dimensions);
    usearch_executor_t const executor = create_executor(0);
    usearch_dense_index_t const index = create_filled_index(rows, executor);
    search_outputs_t outputs = {.arena = create_arena(executor)};

    search_every_row(index, executor, usearch_completion_deferred_k, rows, &outputs);
    expect_eq(outputs.outcome.status, usearch_status_deferred_k, "A submitted call is deferred until waited on");

    // Without an executor the get runs at once, draining nothing, so it meets the deferred call's hold.
    usearch_outcome_t outcome = {0};
    usearch_rows_view_t vectors = {0};
    usearch_dense_get_t get = {.index = index,
                               .outcome = &outcome,
                               .key = key_of(3),
                               .vectors = &vectors,
                               .arena = outputs.arena,
                               .outputs_scalar = usearch_scalar_f32_k};
    usearch_dense_get(&get);
    expect_eq(outcome.status, usearch_status_contended_k, "An arena a deferred call holds refuses another call");
    expect(vectors.data == NULL, "A refused call leaves its outputs NULL");

    expect_eq(usearch_outcome_wait(&outputs.outcome), usearch_status_success_k, "A waited search must finish");
    expect_eq(outputs.matches.counts[0], deferred_wanted, "The deferred call lands its matches in the held arena");
    usearch_dense_get(&get);
    expect(!outcome.message, outcome.message);
    expect(vectors.data != NULL, "A final call lets go of its arena");

    usearch_arena_free(outputs.arena);
    usearch_dense_free(index);
    usearch_executor_free(executor);
    free(rows);
    printf("Test: Deferred Holds Arena - PASSED\n");
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
    test_add_progress();
    test_count_and_get(dimension_counts[0]);
    test_arenas(dimension_counts[0]);
    test_walks_name_their_index(dimension_counts[0]);
    test_multiply_packed();
    test_attend();
    test_nvme_view(dimension_counts[0]);
    test_deferred_search();
    test_deferred_limit();
    test_free_finishes_deferred();
    test_blocking_runs_after_earlier();
    test_deferred_holds_arena();

    printf("All tests passed!\n");
    return 0;
}
