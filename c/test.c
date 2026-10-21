/**
 *  @file c/test.c
 *  @author Ash Vardanian
 *  @date June 25, 2023
 *  @brief Unit tests for the pure-C ABI of USearch.
 *
 *  Exercises the dense surface through the ABI every binding consumes: initialization and
 *  reservation, batch `add`, `search`, `multiply` and `attend`, retrieval by key, the arenas every
 *  output lands in, the contract of `usearch_outcome_t`, and deferred calls on an executor. The
 *  sequence, shapes and geospatial surfaces each get one checkpoint round trip. A blocking call's
 *  outcome is final once it returns; a deferred one is waited on or drained. Scoped to the verbs
 *  `src/` instantiates today - the header runs ahead of the implementation, and a smoke test that
 *  cannot link proves nothing. The harness is intentionally dependency-free so it can run in the
 *  same matrix as the C++ tests and on cross-compilation targets where only the C runtime is
 *  available.
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
#include <stdlib.h>  // `malloc`, `free`, `getenv`, `strtoul`, `srand`
#include <string.h>  // `memset`, `memcmp`, `strstr`
#include <time.h>    // `time`, for a drawn seed

/* `dbghelp.h` needs types `windows.h` defines, so a blank line keeps clang-format from sorting it first. */
#if defined(_WIN32)
#include <windows.h>

#include <dbghelp.h> // `CaptureStackBackTrace`, `SymFromAddr`
#pragma comment(lib, "Dbghelp.lib")
#elif __has_include(<execinfo.h>)
#include <execinfo.h> // `backtrace`, `backtrace_symbols_fd`
#include <unistd.h>   // `STDERR_FILENO`
#endif

#include "usearch/dense_index.h"
#include "usearch/geospatial_index.h"
#include "usearch/sequence_index.h"
#include "usearch/shapes_index.h"

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

/** A substring of the tests to run, from `USEARCH_FILTER`; `NULL` runs every test. */
static char const* test_filter = NULL;

/** Whether the test @p name passes `USEARCH_FILTER`. */
static bool selected(char const* name) { return !test_filter || strstr(name, test_filter) != NULL; }

/** The seed of `rand`: 42, unless `USEARCH_SEED` names another or is "random", which draws one
 *  from the clock. */
static unsigned read_seed(void) {
    char const* const text = getenv("USEARCH_SEED");
    if (!text || !*text) return 42;
    if (strcmp(text, "random") == 0) return (unsigned)time(NULL);
    char* end = NULL;
    unsigned long const seed = strtoul(text, &end, 10);
    if (*end) {
        fprintf(stderr, "USEARCH_SEED=\"%s\" does not parse, expected an unsigned integer or random\n", text);
        exit(1);
    }
    return (unsigned)seed;
}

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

/** Admits the keys whose integer identity is even, recording in @p state the widest batch it was
 *  asked about. */
void admits_even(usearch_key_t const* keys, usearch_admission_t* admissions, size_t count, void* state) {
    size_t* const widest = (size_t*)state;
    if (count > *widest) *widest = count;
    for (size_t index = 0; index != count; ++index)
        admissions[index] = keys[index].bytes[15] % 2 == 0 ? usearch_admit_k : usearch_reject_k;
}

/** Searches the @p count rows of @p data on @p index under an expensive @ref admits_even, walking
 *  and then scanning, and expects every match even: each thread's family of gathers lives through
 *  the call however the engine rewinds its scratch. */
void expect_expensive_narrows(usearch_dense_index_t index, usearch_executor_t executor, usearch_arena_t arena,
                              float* data, size_t count, size_t dimensions) {
    usearch_outcome_t outcome = {0};
    usearch_matches_view_t matches = {0};
    size_t widest_batch = 0;
    usearch_traversal_t const traversals[2] = {usearch_traversal_walk_k, usearch_traversal_scan_k};
    for (size_t pass = 0; pass != 2; ++pass) {
        usearch_dense_search(
            &(usearch_dense_search_t) {.index = index,
                                       .executor = executor,
                                       .outcome = &outcome,
                                       .count = count,
                                       .queries = {data, dimensions * sizeof(float), dimensions, usearch_scalar_f32_k},
                                       .matches = &matches,
                                       .arena = arena,
                                       .wanted = 1,
                                       .traversal = traversals[pass],
                                       .predicate = {admits_even, &widest_batch, usearch_predicate_cost_expensive_k}});
        expect(!outcome.message, outcome.message);
        for (size_t row = 0; row != count; ++row)
            expect(matches.counts[row] == 0 || matches.keys[row].bytes[15] % 2 == 0,
                   "An expensive predicate must narrow the matches");
    }
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

#if USEARCH_DENSE_DTYPES < 11
    // The ladder's next rung is the first storage dtype this build leaves out, so it answers
    // unsupported, never an unknown kind.
    usearch_scalar_t const ladder[] = {usearch_scalar_bf16_k, usearch_scalar_u1x8_k, usearch_scalar_i8_k,
                                       usearch_scalar_e4m3_k, usearch_scalar_f16_k,  usearch_scalar_u8_k,
                                       usearch_scalar_e5m2_k, usearch_scalar_i4x2_k, usearch_scalar_u4x2_k,
                                       usearch_scalar_e3m2_k, usearch_scalar_e2m3_k};
    usearch_dense_index_t refused = NULL;
    usearch_dense_init_ram(&(usearch_dense_init_ram_t) {.index = &refused,
                                                        .outcome = &outcome,
                                                        .metric = usearch_dense_metric_inner_k,
                                                        .quantization = ladder[USEARCH_DENSE_DTYPES],
                                                        .dimensions = dimensions});
    expect_eq(outcome.status, usearch_status_unsupported_k, "A storage dtype the build left out is unsupported");
    expect(refused == NULL, "A refused initialization returns no handle");
#endif

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

    size_t widest_batch = 0;
    usearch_dense_search(&(usearch_dense_search_t) {.index = index,
                                                    .executor = executor,
                                                    .outcome = &outcome,
                                                    .count = collection_size,
                                                    .queries = {data, row_bytes, dimensions, usearch_scalar_f32_k},
                                                    .matches = &matches,
                                                    .arena = arena,
                                                    .wanted = 1,
                                                    .predicate = {admits_even, &widest_batch}});
    expect(!outcome.message, outcome.message);
    for (size_t row = 0; row != collection_size; ++row)
        expect(matches.counts[row] == 0 || matches.keys[row].bytes[15] % 2 == 0, "A predicate must narrow the matches");
    expect(widest_batch != 0 && widest_batch <= 4, "A walk must ask its predicate about one scored quad at a time");
    expect_expensive_narrows(index, executor, arena, data, collection_size, dimensions);

    // Where the walk may miss a few self-queries, an exact scan finds every one.
    usearch_dense_search(&(usearch_dense_search_t) {.index = index,
                                                    .executor = executor,
                                                    .outcome = &outcome,
                                                    .count = collection_size,
                                                    .queries = {data, row_bytes, dimensions, usearch_scalar_f32_k},
                                                    .matches = &matches,
                                                    .arena = arena,
                                                    .wanted = 1,
                                                    .traversal = usearch_traversal_scan_k});
    expect(!outcome.message, outcome.message);
    for (size_t row = 0; row != collection_size; ++row) {
        usearch_key_t const expected = key_of(row);
        expect(matches.counts[row] == 1 && memcmp(matches.keys[row].bytes, expected.bytes, sizeof(expected.bytes)) == 0,
               "An exact scan must find every self-query itself");
    }

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

/** Stops a call at its first report. */
static usearch_continuation_t cancel_progress(size_t completed, size_t total, void* state) {
    (void)completed;
    (void)total;
    (void)state;
    return usearch_cancel_k;
}

/** Records the first report, then stops the call there. */
static usearch_continuation_t record_and_cancel(size_t completed, size_t total, void* state) {
    record_progress(completed, total, state);
    return usearch_cancel_k;
}

/** A batch the handler hears about in several steps still lands whole: every step is reported in
 *  order up to the whole batch, and every row is published. */
void test_add_progress(void) {
    printf("Test: Add Progress\n");

    // A handler asking for four reports hears four, the batch split into nearly equal slices.
    enum { rows_count = 101, dimensions = 16, steps = 4 };
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
                                   .progress = {record_progress, &reports, steps}});
    expect(!outcome.message, outcome.message);
    expect(reports.ascending && reports.calls == steps, "Every step must be reported once, in order");
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

/** The first of four indexes over the quarters of one dataset absorbs the rest: a handler
 *  cancelling the closing refine of a one-shot absorb still leaves both its donors consumed, the
 *  call answering @c usearch_status_cancelled_k, and the last donor lands through a refinement
 *  handle, whose pulls are each cancelled until the pass ends while the index refuses insertions,
 *  every pull continuing the count where the last stopped and reporting running totals that only
 *  grow, with no more nodes changed than visited. Every donor slot is nulled as it is consumed, the
 *  sizes add up, the reported audit matches a health call, and every key resolves to one reachable
 *  row. */
void test_absorb(size_t const dimensions) {
    printf("Test: Absorb... %zu dimensions\n", dimensions);

    enum { indexes_count = 4, rows_per_index = 300, rows_count = indexes_count * rows_per_index };
    usearch_outcome_t outcome = {0};
    // Two threads, as the C++ suites drive: every key's count is a fan-out of its own.
    usearch_executor_t executor = NULL;
    usearch_executor_for_ram(&(usearch_executor_for_ram_t) {.executor = &executor, .outcome = &outcome, .threads = 2});
    expect(!outcome.message, outcome.message);
    float* const data = create_vectors(rows_count, dimensions);
    usearch_key_t* const keys = (usearch_key_t*)malloc(rows_count * sizeof(usearch_key_t));
    expect(keys, "Failed to allocate keys");
    for (size_t row = 0; row != rows_count; ++row) keys[row] = key_of(row);
    usearch_dense_index_t indexes[indexes_count];
    for (size_t which = 0; which != indexes_count; ++which) {
        indexes[which] = create_index(dimensions);
        usearch_dense_reserve(&(usearch_dense_reserve_t) {.index = indexes[which],
                                                          .executor = executor,
                                                          .outcome = &outcome,
                                                          .capacity = rows_per_index});
        expect(!outcome.message, outcome.message);
        usearch_dense_insert(
            &(usearch_dense_insert_t) {.index = indexes[which],
                                       .executor = executor,
                                       .outcome = &outcome,
                                       .count = rows_per_index,
                                       .keys = {keys + which * rows_per_index, sizeof(usearch_key_t)},
                                       .vectors = {data + which * rows_per_index * dimensions,
                                                   dimensions * sizeof(float), dimensions, usearch_scalar_f32_k}});
        expect(!outcome.message, outcome.message);
    }

    usearch_dense_absorb(&(usearch_dense_absorb_t) {.index = indexes[0],
                                                    .executor = executor,
                                                    .outcome = &outcome,
                                                    .count = indexes_count - 2,
                                                    .donors = indexes + 1,
                                                    .progress = {cancel_progress}});
    expect_eq(outcome.status, usearch_status_cancelled_k, "A cancel must stop only the closing refine");
    expect(indexes[1] == NULL && indexes[2] == NULL && indexes[3] != NULL,
           "Every donor of the call must be consumed and nulled, and no other");
    usearch_dense_index_description_t description = {0};
    usearch_dense_describe(&(usearch_dense_describe_t) {.index = indexes[0],
                                                        .executor = executor,
                                                        .outcome = &outcome,
                                                        .description = &description});
    expect(!outcome.message, outcome.message);
    expect_eq(description.size, 3 * rows_per_index, "The donors must stay absorbed past the cancel");

    usearch_dense_refinement_t refinement = NULL;
    usearch_dense_refinement_init(&(usearch_dense_refinement_init_t) {.index = indexes[0],
                                                                      .executor = executor,
                                                                      .outcome = &outcome,
                                                                      .refinement = &refinement,
                                                                      .donors = indexes + 3,
                                                                      .count = 1});
    expect(!outcome.message, outcome.message);
    expect(refinement && indexes[3] != NULL, "A donor lands at the first pull, not at the opening");
    usearch_key_t const late_key = key_of(rows_count);
    usearch_dense_insert(
        &(usearch_dense_insert_t) {.index = indexes[0],
                                   .executor = executor,
                                   .outcome = &outcome,
                                   .count = 1,
                                   .keys = {&late_key, sizeof(usearch_key_t)},
                                   .vectors = {data, dimensions * sizeof(float), dimensions, usearch_scalar_f32_k}});
    expect_eq(outcome.status, usearch_status_contended_k, "A refinement must hold its index closed");

    // Every pull cancels at its first report, so the pass ends over several pulls, each reporting
    // the running totals; the closing one may only heal, so its totals need not grow.
    progress_reports_t reports = {0, 0, 0, true};
    size_t visited = 0, changed = 0, pulls = 0;
    usearch_traversal_stats_t stats = {0};
    do {
        size_t pulled_visited = 0, pulled_changed = 0;
        usearch_traversal_stats_t pulled_stats = {0};
        usearch_dense_refinement_next(&(usearch_dense_refinement_next_t) {.index = indexes[0],
                                                                          .executor = executor,
                                                                          .outcome = &outcome,
                                                                          .refinement = refinement,
                                                                          .progress = {record_and_cancel, &reports},
                                                                          .visited = &pulled_visited,
                                                                          .changed = &pulled_changed,
                                                                          .stats = &pulled_stats});
        bool const cancelled = outcome.status == usearch_status_cancelled_k;
        expect(cancelled || !outcome.message, outcome.message);
        expect(indexes[3] == NULL, "Every consumed donor slot must be nulled");
        expect(pulled_changed <= pulled_visited, "No more nodes may change than were re-selected");
        expect(pulled_visited >= visited + cancelled && pulled_changed >= changed &&
                   pulled_stats.computed_distances >= stats.computed_distances + cancelled,
               "The running totals must grow from pull to pull");
        visited = pulled_visited, changed = pulled_changed, stats = pulled_stats;
        ++pulls;
        if (!cancelled) break;
    } while (pulls != rows_count);
    expect(pulls >= 2 && outcome.status == usearch_status_success_k, "The pass must end over several pulls");
    expect(reports.calls > 0 && reports.total == rows_count, "The handler must hear the walks over every row");
    expect(reports.ascending, "Every pull must count on from where the last stopped");

    size_t finished_changed = 0;
    usearch_reachability_t reported = {0};
    usearch_traversal_stats_t finished_stats = {0};
    usearch_dense_refinement_finish(&(usearch_dense_refinement_finish_t) {.index = indexes[0],
                                                                          .executor = executor,
                                                                          .outcome = &outcome,
                                                                          .refinement = refinement,
                                                                          .changed = &finished_changed,
                                                                          .reachability = &reported,
                                                                          .stats = &finished_stats});
    expect(!outcome.message, outcome.message);
    expect(finished_changed >= changed && finished_changed > 0 &&
               finished_stats.computed_distances >= stats.computed_distances,
           "The refinement must report at least the work its pulls reported");

    usearch_dense_describe(&(usearch_dense_describe_t) {.index = indexes[0],
                                                        .executor = executor,
                                                        .outcome = &outcome,
                                                        .description = &description});
    expect(!outcome.message, outcome.message);
    expect_eq(description.size, rows_count, "The absorbed sizes must add up");
    for (size_t row = 0; row != rows_count; ++row) {
        size_t entries = 0;
        usearch_dense_count(&(usearch_dense_count_t) {.index = indexes[0],
                                                      .executor = executor,
                                                      .outcome = &outcome,
                                                      .key = keys[row],
                                                      .entries = &entries});
        expect(!outcome.message, outcome.message);
        expect_eq(entries, 1, "Every key must resolve to one absorbed row");
    }
    usearch_reachability_t reachability = {0};
    usearch_dense_health(&(usearch_dense_health_t) {.index = indexes[0],
                                                    .executor = executor,
                                                    .outcome = &outcome,
                                                    .reachability = &reachability});
    expect(!outcome.message, outcome.message);
    expect_eq(reachability.reachable, rows_count, "Every absorbed row must stay reachable");
    expect(reported.audited == reachability.audited && reported.reachable == reachability.reachable,
           "The closing audit must match a health call right after");

    free(keys);
    free(data);
    usearch_executor_free(executor);
    usearch_dense_free(indexes[0]);
    printf("Test: Absorb - PASSED\n");
}

/** Draws @p count random reads of 100 to 199 nucleotides, end to end around @p count + 1
 *  @p offsets; the bytes must be @b free-ed afterwards. */
char* create_reads(size_t const count, size_t* offsets) {
    offsets[0] = 0;
    for (size_t read = 0; read != count; ++read) offsets[read + 1] = offsets[read] + 100 + (size_t)rand() % 100;
    char* const bytes = (char*)malloc(offsets[count]);
    expect(bytes, "Failed to allocate the reads");
    for (size_t position = 0; position != offsets[count]; ++position) bytes[position] = "ACGT"[rand() % 4];
    return bytes;
}

/** The first of four sequence indexes over the quarters of one set of reads absorbs the rest, as
 *  @c test_absorb drives the dense ones: a one-shot absorb cancelled in its closing refine still
 *  consumes both its donors and answers @c usearch_status_cancelled_k, and the last donor lands
 *  through a refinement handle, whose pulls each cancel until the pass ends while the index
 *  refuses insertions, the running totals only growing. The finished report covers every read, a
 *  one-shot refine after it reaches every read, and every key resolves to one read. */
void test_sequence_absorb(void) {
    printf("Test: Sequence Absorb\n");
    enum { indexes_count = 4, reads_per_index = 60, reads_count = indexes_count * reads_per_index };
    usearch_outcome_t outcome = {0};
    usearch_executor_t executor = NULL;
    usearch_executor_for_ram(&(usearch_executor_for_ram_t) {.executor = &executor, .outcome = &outcome, .threads = 2});
    expect(!outcome.message, outcome.message);
    size_t offsets[reads_count + 1];
    char* const bytes = create_reads(reads_count, offsets);
    usearch_key_t keys[reads_count];
    for (size_t read = 0; read != reads_count; ++read) keys[read] = key_of(read);
    usearch_sequence_index_t indexes[indexes_count];
    for (size_t which = 0; which != indexes_count; ++which) {
        usearch_sequence_init_ram(&(usearch_sequence_init_ram_t) {.index = &indexes[which],
                                                                  .outcome = &outcome,
                                                                  .metric = usearch_sequence_metric_levenshtein_bytes_k,
                                                                  .connectivity = 8});
        expect(!outcome.message, outcome.message);
        // Eight candidates wire a deliberately weak graph, which the refinement then improves.
        usearch_sequence_insert(
            &(usearch_sequence_insert_t) {.index = indexes[which],
                                          .executor = executor,
                                          .outcome = &outcome,
                                          .count = reads_per_index,
                                          .keys = {keys + which * reads_per_index, sizeof(usearch_key_t)},
                                          .sequences = {bytes, offsets + which * reads_per_index, usearch_scalar_u8_k},
                                          .expansion_factor = 8});
        expect(!outcome.message, outcome.message);
    }

    usearch_sequence_absorb(&(usearch_sequence_absorb_t) {.index = indexes[0],
                                                          .executor = executor,
                                                          .outcome = &outcome,
                                                          .count = indexes_count - 2,
                                                          .donors = indexes + 1,
                                                          .progress = {cancel_progress}});
    expect_eq(outcome.status, usearch_status_cancelled_k, "A cancel must stop only the closing refine");
    expect(indexes[1] == NULL && indexes[2] == NULL && indexes[3] != NULL,
           "Every donor of the call must be consumed and nulled, and no other");
    usearch_sequence_index_description_t description = {0};
    usearch_sequence_describe(&(usearch_sequence_describe_t) {.index = indexes[0],
                                                              .executor = executor,
                                                              .outcome = &outcome,
                                                              .description = &description});
    expect(!outcome.message, outcome.message);
    expect_eq(description.size, 3 * reads_per_index, "The donors must stay absorbed past the cancel");

    usearch_sequence_refinement_t refinement = NULL;
    usearch_sequence_refinement_init(&(usearch_sequence_refinement_init_t) {.index = indexes[0],
                                                                            .executor = executor,
                                                                            .outcome = &outcome,
                                                                            .refinement = &refinement,
                                                                            .donors = indexes + 3,
                                                                            .count = 1});
    expect(!outcome.message, outcome.message);
    expect(refinement && indexes[3] != NULL, "A donor lands at the first pull, not at the opening");
    usearch_key_t const late_key = key_of(reads_count);
    usearch_sequence_insert(&(usearch_sequence_insert_t) {.index = indexes[0],
                                                          .executor = executor,
                                                          .outcome = &outcome,
                                                          .count = 1,
                                                          .keys = {&late_key, sizeof(usearch_key_t)},
                                                          .sequences = {bytes, offsets, usearch_scalar_u8_k}});
    expect_eq(outcome.status, usearch_status_contended_k, "A refinement must hold its index closed");

    progress_reports_t reports = {0, 0, 0, true};
    size_t visited = 0, changed = 0, pulls = 0;
    usearch_traversal_stats_t stats = {0};
    do {
        size_t pulled_visited = 0, pulled_changed = 0;
        usearch_traversal_stats_t pulled_stats = {0};
        usearch_sequence_refinement_next(
            &(usearch_sequence_refinement_next_t) {.index = indexes[0],
                                                   .executor = executor,
                                                   .outcome = &outcome,
                                                   .refinement = refinement,
                                                   .progress = {record_and_cancel, &reports},
                                                   .visited = &pulled_visited,
                                                   .changed = &pulled_changed,
                                                   .stats = &pulled_stats});
        bool const cancelled = outcome.status == usearch_status_cancelled_k;
        expect(cancelled || !outcome.message, outcome.message);
        expect(indexes[3] == NULL, "Every consumed donor slot must be nulled");
        expect(pulled_changed <= pulled_visited, "No more reads may change than were re-selected");
        expect(pulled_visited >= visited + cancelled && pulled_changed >= changed &&
                   pulled_stats.computed_distances >= stats.computed_distances + cancelled,
               "The running totals must grow from pull to pull");
        visited = pulled_visited, changed = pulled_changed, stats = pulled_stats;
        ++pulls;
        if (!cancelled) break;
    } while (pulls != reads_count);
    expect(pulls >= 2 && outcome.status == usearch_status_success_k, "The pass must end over several pulls");
    expect(reports.calls > 0 && reports.total == reads_count, "The handler must hear the walks over every read");
    expect(reports.ascending, "Every pull must count on from where the last stopped");

    size_t finished_changed = 0;
    usearch_reachability_t reported = {0};
    usearch_sequence_refinement_finish(&(usearch_sequence_refinement_finish_t) {.index = indexes[0],
                                                                                .executor = executor,
                                                                                .outcome = &outcome,
                                                                                .refinement = refinement,
                                                                                .changed = &finished_changed,
                                                                                .reachability = &reported});
    expect(!outcome.message, outcome.message);
    expect(finished_changed >= changed && finished_changed > 0, "The refinement must report what its pulls changed");
    expect_eq(reported.audited, reads_count, "The closing audit must judge every read");

    // The index reopened: a one-shot refine runs to its end and leaves every read reachable.
    usearch_reachability_t refined = {0};
    size_t refined_changed = 0;
    usearch_sequence_refine(&(usearch_sequence_refine_t) {.index = indexes[0],
                                                          .executor = executor,
                                                          .outcome = &outcome,
                                                          .changed = &refined_changed,
                                                          .reachability = &refined});
    expect(!outcome.message, outcome.message);
    expect(refined.audited == reads_count && refined.reachable == reads_count, "Every read must stay reachable");
    for (size_t read = 0; read != reads_count; ++read) {
        usearch_sequence_key_stats_t key_stats = {0};
        usearch_sequence_count(&(usearch_sequence_count_t) {.index = indexes[0],
                                                            .executor = executor,
                                                            .outcome = &outcome,
                                                            .key = keys[read],
                                                            .key_stats = &key_stats});
        expect(!outcome.message, outcome.message);
        expect_eq(key_stats.entries, 1, "Every key must resolve to one absorbed read");
    }

    free(bytes);
    usearch_sequence_free(indexes[0]);
    usearch_executor_free(executor);
    printf("Test: Sequence Absorb - PASSED\n");
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

    // Small integers through every nibble, which bf16, `i4x2` and `u1x8` all hold exactly.
    float* const data = (float*)malloc(2 * dimensions * sizeof(float));
    expect(data, "Failed to allocate memory");
    for (size_t index = 0; index != 2 * dimensions; ++index) data[index] = (float)((int)(index % 16) - 8);
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
        expect(retrieved[position] == data[position], "A bf16 index must return small integers exactly");

    // Sub-byte rows come in whole bytes, as NumKong casts them and sub-byte indexes store them.
    usearch_dense_get_t get_packed = {.index = index,
                                      .executor = executor,
                                      .outcome = &outcome,
                                      .key = key_of(7),
                                      .vectors = &vectors,
                                      .written = &written,
                                      .arena = arena,
                                      .outputs_scalar = usearch_scalar_u1x8_k};
    usearch_dense_get(&get_packed);
    if (dimensions % 8)
        expect_eq(outcome.status, usearch_status_shape_mismatch_k, "A bit row ending inside a byte is refused");
    else {
        expect(!outcome.message, outcome.message);
        uint8_t const* const bits = (uint8_t const*)vectors.data;
        for (size_t position = 0; position != dimensions; ++position)
            expect(((bits[position / 8] >> (7 - position % 8)) & 1) == (data[position] != 0),
                   "A bit row must hold whether each value is non-zero");
    }
    get_packed.outputs_scalar = usearch_scalar_i4x2_k;
    usearch_dense_get(&get_packed);
    if (dimensions % 2)
        expect_eq(outcome.status, usearch_status_shape_mismatch_k, "A nibble row ending inside a byte is refused");
    else {
        expect(!outcome.message, outcome.message);
        uint8_t const* const nibbles = (uint8_t const*)vectors.data;
        for (size_t position = 0; position != dimensions; ++position) {
            int const nibble = position % 2 ? nibbles[position / 2] & 0xF : nibbles[position / 2] >> 4;
            expect((nibble ^ 8) - 8 == (int)data[position], "A nibble row must hold each value exactly");
        }
    }

#if USEARCH_DENSE_DTYPES >= 4
    // An angular index is blind to magnitude, so rows far below e4m3's smallest step keep their
    // direction, whether they arrive as f32 or as f64.
    usearch_dense_index_t narrow = NULL;
    usearch_dense_init_ram(&(usearch_dense_init_ram_t) {.index = &narrow,
                                                        .outcome = &outcome,
                                                        .metric = usearch_dense_metric_angular_k,
                                                        .quantization = usearch_scalar_e4m3_k,
                                                        .dimensions = dimensions});
    expect(!outcome.message, outcome.message);
    usearch_dense_reserve(
        &(usearch_dense_reserve_t) {.index = narrow, .executor = executor, .outcome = &outcome, .capacity = 2});
    expect(!outcome.message, outcome.message);
    float* const tiny = (float*)malloc(dimensions * sizeof(float));
    double* const tiny_wide = (double*)malloc(dimensions * sizeof(double));
    expect(tiny && tiny_wide, "Failed to allocate the tiny rows");
    for (size_t position = 0; position != dimensions; ++position)
        tiny[position] = data[position] * 1e-4f, tiny_wide[position] = data[dimensions + position] * 1e-4;
    usearch_dense_insert(&(usearch_dense_insert_t) {.index = narrow,
                                                    .executor = executor,
                                                    .outcome = &outcome,
                                                    .count = 1,
                                                    .keys = {&keys[0], sizeof(usearch_key_t)},
                                                    .vectors = {tiny, row_bytes, dimensions, usearch_scalar_f32_k}});
    expect(!outcome.message, outcome.message);
    usearch_dense_insert(&(usearch_dense_insert_t) {
        .index = narrow,
        .executor = executor,
        .outcome = &outcome,
        .count = 1,
        .keys = {&keys[1], sizeof(usearch_key_t)},
        .vectors = {tiny_wide, dimensions * sizeof(double), dimensions, usearch_scalar_f64_k}});
    expect(!outcome.message, outcome.message);
    for (size_t row = 0; row != 2; ++row) {
        usearch_dense_get(&(usearch_dense_get_t) {.index = narrow,
                                                  .executor = executor,
                                                  .outcome = &outcome,
                                                  .key = keys[row],
                                                  .vectors = &vectors,
                                                  .written = &written,
                                                  .arena = arena,
                                                  .outputs_scalar = usearch_scalar_f32_k});
        expect(!outcome.message && written == 1, outcome.message);
        float const* const direction = (float const*)vectors.data;
        double dot = 0, original_norm = 0, direction_norm = 0;
        for (size_t position = 0; position != dimensions; ++position) {
            double const original = data[row * dimensions + position];
            dot += original * direction[position];
            original_norm += original * original, direction_norm += (double)direction[position] * direction[position];
        }
        expect(dot / sqrt(original_norm * direction_norm) > 0.99, "A tiny row must keep its direction in e4m3");
    }
    free(tiny_wide);
    free(tiny);
    usearch_dense_free(narrow);
#endif

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

enum { neighborhood_rows_k = 200, neighborhood_dimensions_k = 16, sampled_wanted_k = 48, sampled_links_k = 3 };

/** The integer identity a key wraps, as @c key_of wrapped it. */
uint64_t value_of(usearch_key_t key) {
    uint64_t value = 0;
    for (unsigned byte_index = 0; byte_index != 8; ++byte_index)
        value |= (uint64_t)key.bytes[15 - byte_index] << (byte_index * 8);
    return value;
}

/** The Euclidean distance between two of @p rows. */
double row_distance(float const* rows, size_t first, size_t second) {
    double squared = 0;
    for (size_t dimension = 0; dimension != neighborhood_dimensions_k; ++dimension) {
        double const delta = (double)rows[first * neighborhood_dimensions_k + dimension] -
                             (double)rows[second * neighborhood_dimensions_k + dimension];
        squared += delta * delta;
    }
    return sqrt(squared);
}

/** A Euclidean index over @p rows, each under the key of its position but row 1, which shares row
 *  0's key, so one key holds two entries. */
usearch_dense_index_t create_neighborhood(float const* rows, usearch_executor_t executor) {
    usearch_outcome_t outcome = {0};
    usearch_dense_index_t index = NULL;
    usearch_dense_init_ram(&(usearch_dense_init_ram_t) {.index = &index,
                                                        .outcome = &outcome,
                                                        .metric = usearch_dense_metric_euclidean_k,
                                                        .quantization = usearch_scalar_bf16_k,
                                                        .dimensions = neighborhood_dimensions_k,
                                                        .connectivity = 8,
                                                        .values_per_key = 2});
    expect(!outcome.message, outcome.message);
    usearch_key_t keys[neighborhood_rows_k];
    for (size_t row = 0; row != neighborhood_rows_k; ++row) keys[row] = key_of(row == 1 ? 0 : row);
    usearch_dense_reserve(&(usearch_dense_reserve_t) {.index = index,
                                                      .executor = executor,
                                                      .outcome = &outcome,
                                                      .capacity = neighborhood_rows_k});
    expect(!outcome.message, outcome.message);
    usearch_dense_insert(&(usearch_dense_insert_t) {
        .index = index,
        .executor = executor,
        .outcome = &outcome,
        .count = neighborhood_rows_k,
        .keys = {keys, sizeof(usearch_key_t)},
        .vectors = {rows, neighborhood_dimensions_k * sizeof(float), neighborhood_dimensions_k, usearch_scalar_f32_k}});
    expect(!outcome.message, outcome.message);
    return index;
}

/** One run of draws with their links, copied out of the arena every call rewinds. */
typedef struct sampled_t {
    size_t written;
    usearch_key_t keys[sampled_wanted_k];
    size_t offsets[sampled_wanted_k + 1];
    usearch_key_t linked[sampled_wanted_k * sampled_links_k];
    usearch_distance_t distances[sampled_wanted_k * sampled_links_k];
} sampled_t;

/** Draws up to @p wanted more of the sequence @p seed names, past the draws @p sampled holds, and
 *  appends them; answers the outcome, which an implementation that cannot sample fails. */
usearch_outcome_t sample_more(usearch_dense_index_t index, usearch_executor_t executor, usearch_arena_t arena,
                              uint64_t seed, size_t wanted, sampled_t* sampled) {
    usearch_outcome_t outcome = {0};
    usearch_keys_view_t drawn = {0};
    usearch_matches_tape_view_t links = {0};
    size_t written = 0;
    usearch_dense_sample(&(usearch_dense_sample_t) {.index = index,
                                                    .executor = executor,
                                                    .outcome = &outcome,
                                                    .keys = &drawn,
                                                    .links = &links,
                                                    .written = &written,
                                                    .arena = arena,
                                                    .wanted = wanted,
                                                    .offset = sampled->written,
                                                    .links_per_draw = sampled_links_k,
                                                    .seed = seed});
    if (outcome.status != usearch_status_success_k) return outcome;
    expect(written <= wanted, "A sample writes at most the draws wanted");
    size_t const base = sampled->written;
    for (size_t draw = 0; draw != written; ++draw) {
        sampled->keys[base + draw] = drawn.data[draw];
        sampled->offsets[base + draw + 1] = sampled->offsets[base] + links.offsets[draw + 1];
    }
    size_t const linked = links.offsets[written];
    memcpy(sampled->linked + sampled->offsets[base], links.keys, linked * sizeof(usearch_key_t));
    memcpy(sampled->distances + sampled->offsets[base], links.scores, linked * sizeof(usearch_distance_t));
    sampled->written = base + written;
    return outcome;
}

/** Whether two runs drew the same keys and links, their distances within a relative tolerance. */
bool same_samples(sampled_t const* first, sampled_t const* second) {
    size_t const links = first->offsets[first->written];
    if (first->written != second->written ||
        memcmp(first->keys, second->keys, first->written * sizeof(usearch_key_t)) != 0 ||
        memcmp(first->offsets, second->offsets, (first->written + 1) * sizeof(size_t)) != 0 ||
        memcmp(first->linked, second->linked, links * sizeof(usearch_key_t)) != 0)
        return false;
    for (size_t link = 0; link != links; ++link)
        if (fabs(first->distances[link] - second->distances[link]) > 1e-6 * fabs(first->distances[link]) + 1e-12)
            return false;
    return true;
}

/** A sample extends one seeded sequence: two calls joined by @c offset draw what one call does, an
 *  equal seed repeats it, every draw after the first links to earlier ones, and a radius around an
 *  anchor bounds the region. An implementation that cannot sample says so, and the test skips. */
void test_sample(void) {
    printf("Test: Sample\n");
    usearch_outcome_t outcome = {0};
    usearch_executor_t const executor = create_executor(0);
    usearch_arena_t const arena = create_arena(executor);
    float* const rows = create_vectors(neighborhood_rows_k, neighborhood_dimensions_k);
    usearch_dense_index_t const index = create_neighborhood(rows, executor);

    sampled_t once = {0}, joined = {0}, repeated = {0};
    outcome = sample_more(index, executor, arena, 42, sampled_wanted_k, &once);
    if (outcome.status == usearch_status_unsupported_k) {
        printf("Test: Sample - SKIPPED: %s\n", outcome.message);
        usearch_dense_free(index);
        free(rows);
        usearch_arena_free(arena);
        usearch_executor_free(executor);
        return;
    }
    expect(!outcome.message, outcome.message);
    expect_eq(once.written, sampled_wanted_k, "A sample of the whole index draws all it wants");
    outcome = sample_more(index, executor, arena, 42, 20, &joined);
    expect(!outcome.message, outcome.message);
    outcome = sample_more(index, executor, arena, 42, sampled_wanted_k - 20, &joined);
    expect(!outcome.message, outcome.message);
    expect(same_samples(&once, &joined), "Two calls joined by their offset must draw what one call does");
    outcome = sample_more(index, executor, arena, 42, sampled_wanted_k, &repeated);
    expect(!outcome.message, outcome.message);
    expect(same_samples(&once, &repeated), "An equal seed must repeat the sequence");

    for (size_t draw = 0; draw != once.written; ++draw) {
        size_t const links = once.offsets[draw + 1] - once.offsets[draw];
        expect(draw ? links >= 1 && links <= sampled_links_k : links == 0,
               "Every draw after the first, and only those, links to earlier draws");
        for (size_t link = once.offsets[draw]; link != once.offsets[draw + 1]; ++link) {
            bool earlier = false;
            for (size_t previous = 0; previous != draw; ++previous)
                earlier = earlier ||
                          memcmp(once.linked[link].bytes, once.keys[previous].bytes, sizeof(usearch_key_t)) == 0;
            expect(earlier, "A link must name an earlier draw");
        }
    }

    // The mean distance from row 0 splits the index, so its region is neither empty nor whole.
    double radius = 0;
    for (size_t row = 0; row != neighborhood_rows_k; ++row) radius += row_distance(rows, 0, row);
    radius /= neighborhood_rows_k;
    usearch_keys_view_t drawn = {0};
    usearch_matches_tape_view_t links = {0};
    size_t written = 0;
    usearch_dense_sample(&(usearch_dense_sample_t) {
        .index = index,
        .executor = executor,
        .outcome = &outcome,
        .anchors_count = 1,
        .anchors = {rows, neighborhood_dimensions_k * sizeof(float), neighborhood_dimensions_k, usearch_scalar_f32_k},
        .keys = &drawn,
        .links = &links,
        .written = &written,
        .arena = arena,
        .wanted = neighborhood_rows_k,
        .links_per_draw = 1,
        .seed = 7,
        .radius = radius});
    expect(!outcome.message, outcome.message);
    expect(written >= 1 && written < neighborhood_rows_k, "A radius must bound the region drawn from");
    for (size_t draw = 0; draw != written; ++draw)
        expect(row_distance(rows, 0, value_of(drawn.data[draw])) <= radius * (1 + 1e-6),
               "No draw may lie outside the radius");

    usearch_dense_free(index);
    free(rows);
    usearch_arena_free(arena);
    usearch_executor_free(executor);
    printf("Test: Sample - PASSED\n");
}

/** Pulls @p cursor to its end and frees it, counting in @p yielded how often each key's integer
 *  identity came up; answers the entries pulled. */
size_t pull_to_end(usearch_dense_index_t index, usearch_executor_t executor, usearch_arena_t arena,
                   usearch_dense_cursor_t cursor, size_t* yielded) {
    usearch_outcome_t outcome = {0};
    size_t pulled = 0, written = 0;
    do {
        usearch_keys_view_t keys = {0};
        usearch_dense_cursor_next(&(usearch_dense_cursor_next_t) {.index = index,
                                                                  .executor = executor,
                                                                  .outcome = &outcome,
                                                                  .cursor = cursor,
                                                                  .keys = &keys,
                                                                  .written = &written,
                                                                  .arena = arena,
                                                                  .wanted = 64});
        expect(!outcome.message, outcome.message);
        for (size_t entry = 0; entry != written; ++entry) {
            uint64_t const value = value_of(keys.data[entry]);
            expect(value < neighborhood_rows_k, "A cursor yields only keys the index holds");
            ++yielded[value];
        }
        pulled += written;
    } while (written);
    usearch_dense_cursor_free(cursor);
    return pulled;
}

/** A cursor opened at a key yields that key's own entries, then the entries they link to, once
 *  each, and an absent key starts nothing, while a plain scan still yields every row once. An
 *  implementation that cannot step yet says so, and that half skips. */
void test_cursor_steps(void) {
    printf("Test: Cursor Steps\n");
    usearch_outcome_t outcome = {0};
    usearch_executor_t const executor = create_executor(0);
    usearch_arena_t const arena = create_arena(executor);
    float* const rows = create_vectors(neighborhood_rows_k, neighborhood_dimensions_k);
    usearch_dense_index_t const index = create_neighborhood(rows, executor);

    size_t yielded[neighborhood_rows_k] = {0};
    usearch_dense_cursor_t cursor = NULL;
    usearch_dense_cursor_init(&(usearch_dense_cursor_init_t) {.cursor = &cursor, .index = index, .outcome = &outcome});
    expect(!outcome.message, outcome.message);
    expect_eq(pull_to_end(index, executor, arena, cursor, yielded), neighborhood_rows_k,
              "A plain scan must yield every row");
    for (size_t row = 0; row != neighborhood_rows_k; ++row)
        expect(yielded[row] == (row == 0 ? 2u : row == 1 ? 0u : 1u), "A plain scan must yield every row once");

    usearch_key_t const start = key_of(0);
    usearch_dense_cursor_init(&(usearch_dense_cursor_init_t) {.cursor = &cursor,
                                                              .index = index,
                                                              .outcome = &outcome,
                                                              .starts_count = 1,
                                                              .starts = &start});
    if (outcome.status == usearch_status_unsupported_k)
        printf("Test: Cursor Steps - SKIPPED the step: %s\n", outcome.message);
    else {
        expect(!outcome.message, outcome.message);
        memset(yielded, 0, sizeof(yielded));
        expect(pull_to_end(index, executor, arena, cursor, yielded) > 2, "A key's entries must link somewhere");
        expect_eq(yielded[0], 2, "A walk must yield both of the start key's own entries");
        for (size_t row = 1; row != neighborhood_rows_k; ++row)
            expect(yielded[row] <= 1, "A walk must yield each linked entry once");

        usearch_key_t const absent = key_of(neighborhood_rows_k);
        usearch_dense_cursor_init(&(usearch_dense_cursor_init_t) {.cursor = &cursor,
                                                                  .index = index,
                                                                  .outcome = &outcome,
                                                                  .starts_count = 1,
                                                                  .starts = &absent});
        expect(!outcome.message, outcome.message);
        expect_eq(pull_to_end(index, executor, arena, cursor, yielded), 0, "An absent key must step nowhere");
    }

    usearch_dense_free(index);
    free(rows);
    usearch_arena_free(arena);
    usearch_executor_free(executor);
    printf("Test: Cursor Steps - PASSED\n");
}

/** Sequences round-trip through a checkpoint: a stored query finds itself, then its one-edit
 *  neighbor, and `get` hands the bytes back, before the save and after the load alike. The overlap
 *  kind refuses to go without window widths. */
void test_sequence_round_trip(void) {
    printf("Test: Sequence Round Trip\n");
    usearch_outcome_t outcome = {0};
    usearch_executor_t const executor = create_executor(0);
    usearch_arena_t const arena = create_arena(executor);

    usearch_sequence_index_t refused = NULL;
    usearch_sequence_init_ram(&(usearch_sequence_init_ram_t) {.index = &refused,
                                                              .outcome = &outcome,
                                                              .metric = usearch_sequence_metric_overlap_k});
    expect_eq(outcome.status, usearch_status_config_mismatch_k, "The overlap kind needs window widths");
    expect(refused == NULL, "A refused initialization returns no handle");

    char const bytes[] = "kittensittingmittenflask";
    size_t const offsets[] = {0, 6, 13, 19, 24};
    usearch_key_t const keys[] = {key_of(0), key_of(1), key_of(2), key_of(3)};
    usearch_sequence_index_t index = NULL;
    usearch_sequence_init_ram(&(usearch_sequence_init_ram_t) {.index = &index,
                                                              .outcome = &outcome,
                                                              .metric = usearch_sequence_metric_levenshtein_bytes_k});
    expect(!outcome.message, outcome.message);
    usearch_sequence_insert(&(usearch_sequence_insert_t) {.index = index,
                                                          .executor = executor,
                                                          .outcome = &outcome,
                                                          .count = 4,
                                                          .keys = {keys, sizeof(usearch_key_t)},
                                                          .sequences = {bytes, offsets, usearch_scalar_u8_k}});
    expect(!outcome.message, outcome.message);
    size_t const falling_offsets[] = {0, 6, 3};
    usearch_sequence_insert(&(usearch_sequence_insert_t) {.index = index,
                                                          .executor = executor,
                                                          .outcome = &outcome,
                                                          .count = 2,
                                                          .keys = {keys, sizeof(usearch_key_t)},
                                                          .sequences = {bytes, falling_offsets, usearch_scalar_u8_k}});
    expect_eq(outcome.status, usearch_status_invalid_argument_k, "Falling offsets are refused before any insert");

    char const path[] = "usearch-test-c-sequence.safegraphs";
    usearch_bytes_view_t const path_view = {(uint8_t const*)path, sizeof(path) - 1};
    for (int reloaded = 0; reloaded != 2; ++reloaded) {
        usearch_matches_view_t matches = {0};
        usearch_sequence_search(&(usearch_sequence_search_t) {.index = index,
                                                              .executor = executor,
                                                              .outcome = &outcome,
                                                              .count = 1,
                                                              .queries = {bytes, offsets, usearch_scalar_u8_k},
                                                              .matches = &matches,
                                                              .arena = arena,
                                                              .wanted = 2});
        expect(!outcome.message, outcome.message);
        expect(matches.counts[0] == 2 && value_of(matches.keys[0]) == 0 && matches.scores[0] == 0,
               "A stored query finds itself first");
        expect(value_of(matches.keys[1]) == 2 && matches.scores[1] == 1, "One substitution away comes second");

        usearch_tape_view_t got = {0};
        size_t written = 0;
        usearch_sequence_get(&(usearch_sequence_get_t) {.index = index,
                                                        .executor = executor,
                                                        .outcome = &outcome,
                                                        .key = key_of(1),
                                                        .sequences = &got,
                                                        .written = &written,
                                                        .arena = arena});
        expect(!outcome.message, outcome.message);
        expect(written == 1 && got.offsets[1] - got.offsets[0] == 7 &&
                   memcmp((uint8_t const*)got.data + got.offsets[0], "sitting", 7) == 0,
               "Get hands the stored bytes back");
        if (reloaded) break;

        usearch_sequence_save(
            &(usearch_sequence_save_t) {.index = index, .executor = executor, .outcome = &outcome, .path = path_view});
        expect(!outcome.message, outcome.message);
        usearch_sequence_free(index);
        index = NULL;
        usearch_sequence_load_ram(&(usearch_sequence_load_ram_t) {.index = &index,
                                                                  .executor = executor,
                                                                  .outcome = &outcome,
                                                                  .path = path_view});
        expect(!outcome.message, outcome.message);
    }
    usearch_sequence_index_description_t description = {0};
    usearch_sequence_describe(&(usearch_sequence_describe_t) {.index = index,
                                                              .executor = executor,
                                                              .outcome = &outcome,
                                                              .description = &description});
    expect(!outcome.message, outcome.message);
    expect(description.size == 4 && description.metadata.metric == usearch_sequence_metric_levenshtein_bytes_k,
           "A loaded index describes the saved one");

    // The same checkpoint served from storage answers reads as RAM does, refusing writes and saves.
    usearch_executor_t storage = NULL;
    usearch_executor_for_nvme(&(usearch_executor_for_nvme_t) {.executor = &storage, .outcome = &outcome, .threads = 1});
    expect(!outcome.message, outcome.message);
    usearch_sequence_index_t viewed = NULL;
    usearch_sequence_view_nvme(&(
        usearch_sequence_view_nvme_t) {.index = &viewed, .executor = storage, .outcome = &outcome, .path = path_view});
    if (outcome.status == usearch_status_unsupported_k)
        printf("Test: Sequence Round Trip - SKIPPED the NVMe view: %s\n", outcome.message);
    else {
        expect(!outcome.message, outcome.message);
        usearch_matches_view_t viewed_matches = {0};
        usearch_sequence_search(&(usearch_sequence_search_t) {.index = viewed,
                                                              .executor = storage,
                                                              .outcome = &outcome,
                                                              .count = 1,
                                                              .queries = {bytes, offsets, usearch_scalar_u8_k},
                                                              .matches = &viewed_matches,
                                                              .arena = arena,
                                                              .wanted = 1});
        expect(!outcome.message, outcome.message);
        expect(viewed_matches.counts[0] == 1 && value_of(viewed_matches.keys[0]) == 0 && viewed_matches.scores[0] == 0,
               "A view from storage finds a stored query at itself");
        usearch_sequence_key_stats_t stats = {0};
        usearch_sequence_count(&(usearch_sequence_count_t) {.index = viewed,
                                                            .executor = storage,
                                                            .outcome = &outcome,
                                                            .key = key_of(1),
                                                            .key_stats = &stats});
        expect(!outcome.message, outcome.message);
        expect(stats.entries == 1 && stats.total_bytes == 7 && stats.largest_bytes == 7,
               "A view from storage measures a key's entries");
        usearch_tape_view_t viewed_got = {0};
        size_t viewed_written = 0;
        usearch_sequence_get(&(usearch_sequence_get_t) {.index = viewed,
                                                        .executor = storage,
                                                        .outcome = &outcome,
                                                        .key = key_of(1),
                                                        .sequences = &viewed_got,
                                                        .written = &viewed_written,
                                                        .arena = arena});
        expect(!outcome.message, outcome.message);
        expect(viewed_written == 1 && viewed_got.offsets[1] - viewed_got.offsets[0] == 7 &&
                   memcmp((uint8_t const*)viewed_got.data + viewed_got.offsets[0], "sitting", 7) == 0,
               "A view from storage hands the stored bytes back");
        usearch_sequence_refine(
            &(usearch_sequence_refine_t) {.index = viewed, .executor = storage, .outcome = &outcome});
        expect_eq(outcome.status, usearch_status_read_only_k, "A view from storage refines nothing");
        usearch_sequence_cursor_t cursor = NULL;
        usearch_sequence_cursor_init(
            &(usearch_sequence_cursor_init_t) {.cursor = &cursor, .index = viewed, .outcome = &outcome});
        expect(!outcome.message, outcome.message);
        size_t pulled[2] = {0};
        for (size_t pull = 0; pull != 2; ++pull) {
            usearch_keys_view_t cursor_keys = {0};
            usearch_sequence_cursor_next(&(usearch_sequence_cursor_next_t) {.index = viewed,
                                                                            .executor = storage,
                                                                            .outcome = &outcome,
                                                                            .cursor = cursor,
                                                                            .keys = &cursor_keys,
                                                                            .written = &pulled[pull],
                                                                            .arena = arena,
                                                                            .wanted = 3});
            expect(!outcome.message, outcome.message);
            if (pull == 0)
                expect(pulled[0] == 3 && value_of(((usearch_key_t const*)cursor_keys.data)[1]) == 1 &&
                           value_of(((usearch_key_t const*)cursor_keys.data)[2]) == 2,
                       "A cursor over storage reads rows in insertion order");
        }
        expect(pulled[1] == 1, "A cursor over storage stops at the last row");
        usearch_sequence_cursor_free(cursor);
        usearch_sequence_insert(&(usearch_sequence_insert_t) {.index = viewed,
                                                              .executor = storage,
                                                              .outcome = &outcome,
                                                              .count = 1,
                                                              .keys = {keys, sizeof(usearch_key_t)},
                                                              .sequences = {bytes, offsets, usearch_scalar_u8_k}});
        expect_eq(outcome.status, usearch_status_read_only_k, "A view from storage refuses every mutation");
        usearch_sequence_save(
            &(usearch_sequence_save_t) {.index = viewed, .executor = storage, .outcome = &outcome, .path = path_view});
        expect_eq(outcome.status, usearch_status_unsupported_k, "A view from storage already is its checkpoint");
        usearch_sequence_free(viewed);
    }
    usearch_executor_free(storage);
    remove(path);

    usearch_sequence_free(index);
    usearch_arena_free(arena);
    usearch_executor_free(executor);
    printf("Test: Sequence Round Trip - PASSED\n");
}

/** A GPU sequence index takes byte and rune Levenshtein, inserts and describes its rows, and lands
 *  a search's matches in its executor's device arena; alignments and the verbs it lacks refuse.
 *  Skipped without a GPU. */
void test_sequence_vram(void) {
    usearch_outcome_t outcome = {0};
    size_t devices_count = 0;
    usearch_gpu_devices(&(usearch_gpu_devices_t) {.outcome = &outcome, .devices_count = &devices_count});
    if (!devices_count) return;
    printf("Test: Sequence VRAM\n");
    int32_t const devices[1] = {0};
    usearch_vram_options_t const vram = {.devices = devices, .devices_count = 1};

    usearch_sequence_index_t refused = NULL;
    usearch_sequence_init_vram(&(usearch_sequence_init_vram_t) {.index = &refused,
                                                                .outcome = &outcome,
                                                                .metric = usearch_sequence_metric_needleman_wunsch_k,
                                                                .vram = vram});
    expect_eq(outcome.status, usearch_status_unsupported_k, "No alignment serves a GPU yet");
    expect(refused == NULL, "A refused initialization returns no handle");
    usearch_sequence_init_vram(&(usearch_sequence_init_vram_t) {.index = &refused,
                                                                .outcome = &outcome,
                                                                .metric = usearch_sequence_metric_levenshtein_bytes_k,
                                                                .values_per_key = 2,
                                                                .vram = vram});
    expect_eq(outcome.status, usearch_status_unsupported_k, "A GPU sequence index bounds no key's entries");

    usearch_sequence_index_t index = NULL;
    usearch_sequence_init_vram(&(usearch_sequence_init_vram_t) {.index = &index,
                                                                .outcome = &outcome,
                                                                .metric = usearch_sequence_metric_levenshtein_bytes_k,
                                                                .vram = vram});
    if (outcome.status == usearch_status_device_k) {
        printf("Test: Sequence VRAM - SKIPPED: %s\n", outcome.message);
        return;
    }
    expect(!outcome.message, outcome.message);
    usearch_executor_t executor = NULL;
    usearch_executor_for_vram(
        &(usearch_executor_for_vram_t) {.executor = &executor, .outcome = &outcome, .vram = vram});
    expect(!outcome.message, outcome.message);

    char const bytes[] = "kittensittingmittenflask";
    size_t const offsets[] = {0, 6, 13, 19, 24};
    usearch_key_t const keys[] = {key_of(0), key_of(1), key_of(2), key_of(3)};
    usearch_sequence_insert(&(usearch_sequence_insert_t) {.index = index,
                                                          .executor = executor,
                                                          .outcome = &outcome,
                                                          .count = 4,
                                                          .keys = {keys, sizeof(usearch_key_t)},
                                                          .sequences = {bytes, offsets, usearch_scalar_u8_k}});
    expect(!outcome.message, outcome.message);
    usearch_sequence_refine(&(usearch_sequence_refine_t) {.index = index, .executor = executor, .outcome = &outcome});
    expect_eq(outcome.status, usearch_status_unsupported_k, "A GPU index refines nothing yet");
    usearch_sequence_index_description_t description = {0};
    usearch_sequence_describe(&(usearch_sequence_describe_t) {.index = index,
                                                              .executor = executor,
                                                              .outcome = &outcome,
                                                              .description = &description});
    expect(!outcome.message, outcome.message);
    expect(description.size == 4 && description.metadata.metric == usearch_sequence_metric_levenshtein_bytes_k,
           "A GPU index describes the rows it holds");

    // The matches land in device memory, which plain C cannot read; the C++ VRAM suite checks them.
    usearch_arena_t const arena = create_arena(executor);
    usearch_matches_view_t matches = {0};
    usearch_sequence_search(&(usearch_sequence_search_t) {.index = index,
                                                          .executor = executor,
                                                          .outcome = &outcome,
                                                          .count = 4,
                                                          .queries = {bytes, offsets, usearch_scalar_u8_k},
                                                          .matches = &matches,
                                                          .arena = arena,
                                                          .wanted = 2});
    expect(!outcome.message, outcome.message);
    expect(matches.keys && matches.scores && matches.counts, "A GPU search lands its matches in the arena");

    char const path[] = "usearch-test-c-sequence-vram.safegraphs";
    usearch_sequence_save(&(usearch_sequence_save_t) {.index = index,
                                                      .executor = executor,
                                                      .outcome = &outcome,
                                                      .path = {(uint8_t const*)path, sizeof(path) - 1}});
    expect_eq(outcome.status, usearch_status_unsupported_k, "A GPU sequence index saves no checkpoint yet");

    usearch_arena_free(arena);
    usearch_sequence_free(index);
    usearch_executor_free(executor);
    printf("Test: Sequence VRAM - PASSED\n");
}

/** Shapes round-trip the same way: a stored pose finds itself at zero, and `get` exports its points
 *  back, before the save and after the load alike. A frozen point count refuses a shape of any
 *  other. */
void test_shapes_round_trip(void) {
    printf("Test: Shapes Round Trip\n");
    usearch_outcome_t outcome = {0};
    usearch_executor_t const executor = create_executor(0);
    usearch_arena_t const arena = create_arena(executor);

    float const points[] = {
        0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, // a tetrahedron's corners
        0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, // the same, twice the size
        0, 0, 0, 1, 1, 0, 2, 3, 0, 4, 5, 1, // a bent chain
    };
    size_t const offsets[] = {0, 4, 8, 12};
    usearch_key_t const keys[] = {key_of(0), key_of(1), key_of(2)};
    usearch_shapes_index_t index = NULL;
    usearch_shapes_init_ram(&(usearch_shapes_init_ram_t) {.index = &index,
                                                          .outcome = &outcome,
                                                          .metric = usearch_shapes_metric_chamfer_k,
                                                          .quantization = usearch_scalar_f32_k,
                                                          .points = 4});
    expect(!outcome.message, outcome.message);

    size_t const short_offsets[] = {0, 3};
    usearch_shapes_insert(&(usearch_shapes_insert_t) {.index = index,
                                                      .executor = executor,
                                                      .outcome = &outcome,
                                                      .count = 1,
                                                      .keys = {keys, sizeof(usearch_key_t)},
                                                      .shapes = {points, short_offsets, usearch_scalar_f32_k}});
    expect_eq(outcome.status, usearch_status_shape_mismatch_k, "A frozen point count refuses any other");
    size_t const falling_offsets[] = {0, 4, 2};
    usearch_shapes_insert(&(usearch_shapes_insert_t) {.index = index,
                                                      .executor = executor,
                                                      .outcome = &outcome,
                                                      .count = 2,
                                                      .keys = {keys, sizeof(usearch_key_t)},
                                                      .shapes = {points, falling_offsets, usearch_scalar_f32_k}});
    expect_eq(outcome.status, usearch_status_invalid_argument_k, "Falling offsets are refused before any insert");
    usearch_shapes_insert(&(usearch_shapes_insert_t) {.index = index,
                                                      .executor = executor,
                                                      .outcome = &outcome,
                                                      .count = 3,
                                                      .keys = {keys, sizeof(usearch_key_t)},
                                                      .shapes = {points, offsets, usearch_scalar_f32_k}});
    expect(!outcome.message, outcome.message);

    char const path[] = "usearch-test-c-shapes.safegraphs";
    usearch_bytes_view_t const path_view = {(uint8_t const*)path, sizeof(path) - 1};
    for (int reloaded = 0; reloaded != 2; ++reloaded) {
        usearch_matches_view_t matches = {0};
        usearch_shapes_search(&(usearch_shapes_search_t) {.index = index,
                                                          .executor = executor,
                                                          .outcome = &outcome,
                                                          .count = 1,
                                                          .queries = {points, offsets, usearch_scalar_f32_k},
                                                          .matches = &matches,
                                                          .arena = arena,
                                                          .wanted = 1});
        expect(!outcome.message, outcome.message);
        expect(matches.counts[0] == 1 && value_of(matches.keys[0]) == 0 && fabs(matches.scores[0]) < 1e-5,
               "A stored pose finds itself at zero");

        usearch_tape_view_t got = {0};
        size_t written = 0;
        usearch_shapes_get(&(usearch_shapes_get_t) {.index = index,
                                                    .executor = executor,
                                                    .outcome = &outcome,
                                                    .key = key_of(2),
                                                    .shapes = &got,
                                                    .written = &written,
                                                    .arena = arena,
                                                    .outputs_scalar = usearch_scalar_f32_k});
        expect(!outcome.message, outcome.message);
        expect(written == 1 && got.offsets[1] - got.offsets[0] == 4, "Get hands every point back");
        float const* const exported = (float const*)got.data + got.offsets[0] * 3;
        for (size_t coordinate = 0; coordinate != 12; ++coordinate)
            expect(fabsf(exported[coordinate] - points[24 + coordinate]) < 1e-6f, "Get hands the stored pose back");
        if (reloaded) break;

        usearch_shapes_save(
            &(usearch_shapes_save_t) {.index = index, .executor = executor, .outcome = &outcome, .path = path_view});
        expect(!outcome.message, outcome.message);
        usearch_shapes_free(index);
        index = NULL;
        usearch_shapes_load_ram(&(
            usearch_shapes_load_ram_t) {.index = &index, .executor = executor, .outcome = &outcome, .path = path_view});
        expect(!outcome.message, outcome.message);
    }
    usearch_shapes_index_description_t description = {0};
    usearch_shapes_describe(&(usearch_shapes_describe_t) {.index = index,
                                                          .executor = executor,
                                                          .outcome = &outcome,
                                                          .description = &description});
    expect(!outcome.message, outcome.message);
    expect(description.size == 3 && description.total_points == 12 && description.metadata.points == 4,
           "A loaded index describes the saved one");

    // The same checkpoint served from storage answers reads as RAM does, refusing writes and saves.
    usearch_executor_t storage = NULL;
    usearch_executor_for_nvme(&(usearch_executor_for_nvme_t) {.executor = &storage, .outcome = &outcome, .threads = 1});
    expect(!outcome.message, outcome.message);
    usearch_shapes_index_t viewed = NULL;
    usearch_shapes_view_nvme(
        &(usearch_shapes_view_nvme_t) {.index = &viewed, .executor = storage, .outcome = &outcome, .path = path_view});
    if (outcome.status == usearch_status_unsupported_k)
        printf("Test: Shapes Round Trip - SKIPPED the NVMe view: %s\n", outcome.message);
    else {
        expect(!outcome.message, outcome.message);
        usearch_matches_view_t viewed_matches = {0};
        usearch_shapes_search(&(usearch_shapes_search_t) {.index = viewed,
                                                          .executor = storage,
                                                          .outcome = &outcome,
                                                          .count = 1,
                                                          .queries = {points, offsets, usearch_scalar_f32_k},
                                                          .matches = &viewed_matches,
                                                          .arena = arena,
                                                          .wanted = 1});
        expect(!outcome.message, outcome.message);
        expect(viewed_matches.counts[0] == 1 && value_of(viewed_matches.keys[0]) == 0 &&
                   fabs(viewed_matches.scores[0]) < 1e-5,
               "A view from storage finds a stored pose at zero");
        usearch_shapes_key_stats_t stats = {0};
        usearch_shapes_count(&(usearch_shapes_count_t) {.index = viewed,
                                                        .executor = storage,
                                                        .outcome = &outcome,
                                                        .key = key_of(2),
                                                        .key_stats = &stats});
        expect(!outcome.message, outcome.message);
        expect(stats.entries == 1 && stats.total_points == 4 && stats.largest_points == 4,
               "A view from storage measures a key's entries");
        usearch_tape_view_t viewed_got = {0};
        size_t viewed_written = 0;
        usearch_shapes_get(&(usearch_shapes_get_t) {.index = viewed,
                                                    .executor = storage,
                                                    .outcome = &outcome,
                                                    .key = key_of(2),
                                                    .shapes = &viewed_got,
                                                    .written = &viewed_written,
                                                    .arena = arena,
                                                    .outputs_scalar = usearch_scalar_f32_k});
        expect(!outcome.message, outcome.message);
        expect(viewed_written == 1 && viewed_got.offsets[1] - viewed_got.offsets[0] == 4,
               "A view from storage hands every point back");
        float const* const viewed_points = (float const*)viewed_got.data + viewed_got.offsets[0] * 3;
        for (size_t coordinate = 0; coordinate != 12; ++coordinate)
            expect(fabsf(viewed_points[coordinate] - points[24 + coordinate]) < 1e-6f,
                   "A view from storage hands the stored pose back");
        usearch_shapes_cursor_t cursor = NULL;
        usearch_shapes_cursor_init(
            &(usearch_shapes_cursor_init_t) {.cursor = &cursor, .index = viewed, .outcome = &outcome});
        expect(!outcome.message, outcome.message);
        usearch_keys_view_t cursor_keys = {0};
        size_t pulled = 0;
        usearch_shapes_cursor_next(&(usearch_shapes_cursor_next_t) {.index = viewed,
                                                                    .executor = storage,
                                                                    .outcome = &outcome,
                                                                    .cursor = cursor,
                                                                    .keys = &cursor_keys,
                                                                    .written = &pulled,
                                                                    .arena = arena,
                                                                    .wanted = 8});
        expect(!outcome.message, outcome.message);
        expect(pulled == 3 && value_of(((usearch_key_t const*)cursor_keys.data)[1]) == 1 &&
                   value_of(((usearch_key_t const*)cursor_keys.data)[2]) == 2,
               "A cursor over storage reads every pose in insertion order");
        usearch_shapes_cursor_free(cursor);
        usearch_shapes_insert(&(usearch_shapes_insert_t) {.index = viewed,
                                                          .executor = storage,
                                                          .outcome = &outcome,
                                                          .count = 1,
                                                          .keys = {keys, sizeof(usearch_key_t)},
                                                          .shapes = {points, offsets, usearch_scalar_f32_k}});
        expect_eq(outcome.status, usearch_status_read_only_k, "A view from storage refuses every mutation");
        usearch_shapes_save(
            &(usearch_shapes_save_t) {.index = viewed, .executor = storage, .outcome = &outcome, .path = path_view});
        expect_eq(outcome.status, usearch_status_unsupported_k, "A view from storage already is its checkpoint");
        usearch_shapes_free(viewed);
    }
    usearch_executor_free(storage);
    remove(path);

    usearch_shapes_free(index);
    usearch_arena_free(arena);
    usearch_executor_free(executor);
    printf("Test: Shapes Round Trip - PASSED\n");
}

/** A GPU shapes index takes Chamfer and Hausdorff alone, inserts and describes its clouds, and
 *  lands a search's matches in its executor's device arena; the verbs it lacks refuse. Skipped
 *  without a GPU. */
void test_shapes_vram(void) {
    usearch_outcome_t outcome = {0};
    size_t devices_count = 0;
    usearch_gpu_devices(&(usearch_gpu_devices_t) {.outcome = &outcome, .devices_count = &devices_count});
    if (!devices_count) return;
    printf("Test: Shapes VRAM\n");
    int32_t const devices[1] = {0};
    usearch_vram_options_t const vram = {.devices = devices, .devices_count = 1};

    usearch_shapes_index_t refused = NULL;
    usearch_shapes_init_vram(&(usearch_shapes_init_vram_t) {.index = &refused,
                                                            .outcome = &outcome,
                                                            .metric = usearch_shapes_metric_wasserstein_procrustes_k,
                                                            .quantization = usearch_scalar_f16_k,
                                                            .vram = vram});
    expect_eq(outcome.status, usearch_status_unsupported_k, "Only Chamfer and Hausdorff serve a GPU");
    expect(refused == NULL, "A refused initialization returns no handle");

    usearch_shapes_index_t index = NULL;
    usearch_shapes_init_vram(&(usearch_shapes_init_vram_t) {.index = &index,
                                                            .outcome = &outcome,
                                                            .metric = usearch_shapes_metric_hausdorff_k,
                                                            .quantization = usearch_scalar_f16_k,
                                                            .vram = vram});
    if (outcome.status == usearch_status_device_k) {
        printf("Test: Shapes VRAM - SKIPPED: %s\n", outcome.message);
        return;
    }
    expect(!outcome.message, outcome.message);
    usearch_executor_t executor = NULL;
    usearch_executor_for_vram(
        &(usearch_executor_for_vram_t) {.executor = &executor, .outcome = &outcome, .vram = vram});
    expect(!outcome.message, outcome.message);

    // Clouds arrive in f32 and are stored in f16, so every batch casts before it packs.
    float const points[] = {
        0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,          // a tetrahedron's corners
        0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2,          // the same, twice the size
        0, 0, 0, 1, 1, 0, 2, 3, 0, 4, 5, 1, 6, 6, 2, // a bent chain, one point longer
    };
    size_t const offsets[] = {0, 4, 8, 13};
    usearch_key_t const keys[] = {key_of(0), key_of(1), key_of(2)};
    usearch_shapes_insert(&(usearch_shapes_insert_t) {.index = index,
                                                      .executor = executor,
                                                      .outcome = &outcome,
                                                      .count = 3,
                                                      .keys = {keys, sizeof(usearch_key_t)},
                                                      .shapes = {points, offsets, usearch_scalar_f32_k}});
    expect(!outcome.message, outcome.message);
    usearch_shapes_index_description_t description = {0};
    usearch_shapes_describe(&(usearch_shapes_describe_t) {.index = index,
                                                          .executor = executor,
                                                          .outcome = &outcome,
                                                          .description = &description});
    expect(!outcome.message, outcome.message);
    expect(description.size == 3 && description.total_points == 13 &&
               description.metadata.metric == usearch_shapes_metric_hausdorff_k,
           "A GPU index describes the clouds it holds");

    // The matches land in device memory, which plain C cannot read; the C++ VRAM suite checks them.
    usearch_arena_t const arena = create_arena(executor);
    usearch_matches_view_t matches = {0};
    usearch_shapes_search(&(usearch_shapes_search_t) {.index = index,
                                                      .executor = executor,
                                                      .outcome = &outcome,
                                                      .count = 3,
                                                      .queries = {points, offsets, usearch_scalar_f32_k},
                                                      .matches = &matches,
                                                      .arena = arena,
                                                      .wanted = 2});
    expect(!outcome.message, outcome.message);
    expect(matches.keys && matches.scores && matches.counts, "A GPU search lands its matches in the arena");

    char const path[] = "usearch-test-c-shapes-vram.safegraphs";
    usearch_shapes_save(&(usearch_shapes_save_t) {.index = index,
                                                  .executor = executor,
                                                  .outcome = &outcome,
                                                  .path = {(uint8_t const*)path, sizeof(path) - 1}});
    expect_eq(outcome.status, usearch_status_unsupported_k, "A GPU shapes index saves no checkpoint yet");

    usearch_arena_free(arena);
    usearch_shapes_free(index);
    usearch_executor_free(executor);
    printf("Test: Shapes VRAM - PASSED\n");
}

/** An observation of a position alone, every other field unrecorded. */
usearch_geospatial_observation_t observed_at(double latitude_degrees, double longitude_degrees,
                                             double altitude_metres) {
    usearch_geospatial_observation_t const observation = {
        latitude_degrees, longitude_degrees, altitude_metres, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN};
    return observation;
}

/** Geospatial round-trips the same way: an observation finds itself at zero metres and `get` hands
 *  it back, before the save and after the load alike, while a ridge on the handle's terrain hides
 *  what stands behind it. */
void test_geospatial_round_trip(void) {
    printf("Test: Geospatial Round Trip\n");
    usearch_outcome_t outcome = {0};
    usearch_executor_t const executor = create_executor(0);
    usearch_arena_t const arena = create_arena(executor);

    // Six rows about 56 m apart over flat ground, with a 100 m ridge along the third; every entry
    // stands inside the grid, since one on its edge may round off it, and off the grid nothing
    // blocks a sight line.
    enum { terrain_rows = 6, terrain_columns = 3 };
    float heights[terrain_rows * terrain_columns] = {0};
    for (size_t column = 0; column != terrain_columns; ++column) heights[2 * terrain_columns + column] = 100;
    usearch_geospatial_terrain_t const terrain = {.rows = terrain_rows,
                                                  .columns = terrain_columns,
                                                  .north_latitude_degrees = 51.5015,
                                                  .west_longitude_degrees = -0.1005,
                                                  .row_step_degrees = 0.0005,
                                                  .column_step_degrees = 0.0005,
                                                  .heights = heights};

    // A viewer, one entry behind the ridge, one in the open on the terrain, and one in Paris.
    usearch_geospatial_observation_t const observations[] = {observed_at(51.5, -0.1, 2), observed_at(51.501, -0.1, 2),
                                                             observed_at(51.4995, -0.1, NAN),
                                                             observed_at(48.8566, 2.3522, 35)};
    usearch_key_t const keys[] = {key_of(0), key_of(1), key_of(2), key_of(3)};
    usearch_geospatial_index_t index = NULL;
    usearch_geospatial_init_ram(
        &(usearch_geospatial_init_ram_t) {.index = &index, .outcome = &outcome, .terrain = terrain});
    expect(!outcome.message, outcome.message);
    usearch_geospatial_insert(&(usearch_geospatial_insert_t) {.index = index,
                                                              .executor = executor,
                                                              .outcome = &outcome,
                                                              .count = 4,
                                                              .keys = {keys, sizeof(usearch_key_t)},
                                                              .observations = observations});
    expect(!outcome.message, outcome.message);

    char const path[] = "usearch-test-c-geospatial.safegraphs";
    usearch_bytes_view_t const path_view = {(uint8_t const*)path, sizeof(path) - 1};
    for (int reloaded = 0; reloaded != 2; ++reloaded) {
        usearch_matches_view_t matches = {0};
        usearch_geospatial_search(&(usearch_geospatial_search_t) {.index = index,
                                                                  .executor = executor,
                                                                  .outcome = &outcome,
                                                                  .count = 1,
                                                                  .queries = observations,
                                                                  .relation = usearch_geospatial_relation_near_k,
                                                                  .tolerance = {.metres = 500},
                                                                  .matches = &matches,
                                                                  .arena = arena,
                                                                  .wanted = 4});
        expect(!outcome.message, outcome.message);
        expect(matches.counts[0] == 3 && value_of(matches.keys[0]) == 0 && matches.scores[0] < 0.01 &&
                   value_of(matches.keys[1]) == 2 && value_of(matches.keys[2]) == 1,
               "An observation finds itself at zero metres, nearest first, and the radius drops Paris");

        usearch_geospatial_search(&(usearch_geospatial_search_t) {.index = index,
                                                                  .executor = executor,
                                                                  .outcome = &outcome,
                                                                  .count = 1,
                                                                  .queries = observations,
                                                                  .relation = usearch_geospatial_relation_visible_k,
                                                                  .tolerance = {.metres = 500},
                                                                  .matches = &matches,
                                                                  .arena = arena,
                                                                  .wanted = 4});
        expect(!outcome.message, outcome.message);
        expect(matches.counts[0] == 2 && value_of(matches.keys[0]) == 0 && value_of(matches.keys[1]) == 2,
               "The ridge hides the entry behind it");

        usearch_geospatial_observation_t const* got = NULL;
        size_t written = 0;
        usearch_geospatial_get(&(usearch_geospatial_get_t) {.index = index,
                                                            .executor = executor,
                                                            .outcome = &outcome,
                                                            .key = key_of(2),
                                                            .observations = &got,
                                                            .written = &written,
                                                            .arena = arena});
        expect(!outcome.message, outcome.message);
        expect(written == 1 && fabs(got[0].latitude_degrees - 51.4995) < 1e-6 &&
                   fabs(got[0].longitude_degrees + 0.1) < 1e-6 && fabs(got[0].altitude_metres) < 0.02 &&
                   isnan(got[0].heading_degrees),
               "Get hands the observation back, standing on the terrain");
        if (reloaded) break;

        usearch_geospatial_save(&(
            usearch_geospatial_save_t) {.index = index, .executor = executor, .outcome = &outcome, .path = path_view});
        expect(!outcome.message, outcome.message);
        usearch_geospatial_free(index);
        index = NULL;
        usearch_geospatial_load_ram(&(usearch_geospatial_load_ram_t) {.index = &index,
                                                                      .executor = executor,
                                                                      .outcome = &outcome,
                                                                      .path = path_view,
                                                                      .terrain = terrain});
        expect(!outcome.message, outcome.message);
    }
    usearch_geospatial_index_description_t description = {0};
    usearch_geospatial_describe(&(usearch_geospatial_describe_t) {.index = index,
                                                                  .executor = executor,
                                                                  .outcome = &outcome,
                                                                  .description = &description});
    expect(!outcome.message, outcome.message);
    expect(description.size == 4 && description.metadata.connectivity == 12, "A loaded index describes the saved one");

    // The same checkpoint served from storage judges every relation on the rows its walks read and
    // answers reads and containment as RAM does, refusing writes and saves.
    usearch_executor_t storage = NULL;
    usearch_executor_for_nvme(&(usearch_executor_for_nvme_t) {.executor = &storage, .outcome = &outcome, .threads = 1});
    expect(!outcome.message, outcome.message);
    usearch_geospatial_index_t viewed = NULL;
    usearch_geospatial_view_nvme(&(usearch_geospatial_view_nvme_t) {.index = &viewed,
                                                                    .executor = storage,
                                                                    .outcome = &outcome,
                                                                    .path = path_view,
                                                                    .terrain = terrain});
    if (outcome.status == usearch_status_unsupported_k)
        printf("Test: Geospatial Round Trip - SKIPPED the NVMe view: %s\n", outcome.message);
    else {
        expect(!outcome.message, outcome.message);
        usearch_matches_view_t matches = {0};
        usearch_geospatial_search(&(usearch_geospatial_search_t) {.index = viewed,
                                                                  .executor = storage,
                                                                  .outcome = &outcome,
                                                                  .count = 1,
                                                                  .queries = observations,
                                                                  .relation = usearch_geospatial_relation_near_k,
                                                                  .tolerance = {.metres = 500},
                                                                  .matches = &matches,
                                                                  .arena = arena,
                                                                  .wanted = 4});
        expect(!outcome.message, outcome.message);
        expect(matches.counts[0] == 3 && value_of(matches.keys[0]) == 0 && matches.scores[0] < 0.01 &&
                   value_of(matches.keys[1]) == 2 && value_of(matches.keys[2]) == 1,
               "A view from storage finds an observation at zero metres, and the radius drops Paris");
        usearch_geospatial_search(&(usearch_geospatial_search_t) {.index = viewed,
                                                                  .executor = storage,
                                                                  .outcome = &outcome,
                                                                  .count = 1,
                                                                  .queries = observations,
                                                                  .relation = usearch_geospatial_relation_visible_k,
                                                                  .tolerance = {.metres = 500},
                                                                  .matches = &matches,
                                                                  .arena = arena,
                                                                  .wanted = 4});
        expect(!outcome.message, outcome.message);
        expect(matches.counts[0] == 2 && value_of(matches.keys[0]) == 0 && value_of(matches.keys[1]) == 2,
               "A view from storage hides the entry behind the ridge");

        usearch_geospatial_key_stats_t stats = {0};
        usearch_geospatial_count(&(usearch_geospatial_count_t) {.index = viewed,
                                                                .executor = storage,
                                                                .outcome = &outcome,
                                                                .key = key_of(2),
                                                                .key_stats = &stats});
        expect(!outcome.message, outcome.message);
        expect(stats.entries == 1, "A view from storage counts a key's entries");
        usearch_geospatial_observation_t const* got = NULL;
        size_t written = 0;
        usearch_geospatial_get(&(usearch_geospatial_get_t) {.index = viewed,
                                                            .executor = storage,
                                                            .outcome = &outcome,
                                                            .key = key_of(2),
                                                            .observations = &got,
                                                            .written = &written,
                                                            .arena = arena});
        expect(!outcome.message, outcome.message);
        expect(written == 1 && fabs(got[0].latitude_degrees - 51.4995) < 1e-6 && fabs(got[0].altitude_metres) < 0.02,
               "A view from storage hands the observation back");

        usearch_geospatial_cursor_t cursor = NULL;
        usearch_geospatial_cursor_init(
            &(usearch_geospatial_cursor_init_t) {.cursor = &cursor, .index = viewed, .outcome = &outcome});
        expect(!outcome.message, outcome.message);
        usearch_keys_view_t cursor_keys = {0};
        size_t pulled = 0;
        usearch_geospatial_cursor_next(&(usearch_geospatial_cursor_next_t) {.index = viewed,
                                                                            .executor = storage,
                                                                            .outcome = &outcome,
                                                                            .cursor = cursor,
                                                                            .keys = &cursor_keys,
                                                                            .written = &pulled,
                                                                            .arena = arena,
                                                                            .wanted = 8});
        expect(!outcome.message, outcome.message);
        expect(pulled == 4 && value_of(((usearch_key_t const*)cursor_keys.data)[3]) == 3,
               "A cursor over storage reads every observation in insertion order");
        usearch_geospatial_cursor_free(cursor);

        usearch_geospatial_insert(&(usearch_geospatial_insert_t) {.index = viewed,
                                                                  .executor = storage,
                                                                  .outcome = &outcome,
                                                                  .count = 1,
                                                                  .keys = {keys, sizeof(usearch_key_t)},
                                                                  .observations = observations});
        expect_eq(outcome.status, usearch_status_read_only_k, "A view from storage refuses every mutation");

        // Containment over storage reports the entries the walk in RAM reports, in any order.
        usearch_matches_tape_view_t balls = {0};
        size_t ram_reported = 0, nvme_reported = 0;
        unsigned ram_keys = 0, nvme_keys = 0;
        usearch_geospatial_within(&(usearch_geospatial_within_t) {.index = index,
                                                                  .executor = executor,
                                                                  .outcome = &outcome,
                                                                  .count = 1,
                                                                  .queries = observations,
                                                                  .relation = usearch_geospatial_relation_near_k,
                                                                  .tolerance = {.metres = 500},
                                                                  .matches = &balls,
                                                                  .written = &ram_reported,
                                                                  .arena = arena});
        expect(!outcome.message, outcome.message);
        for (size_t match = 0; match != ram_reported; ++match) ram_keys |= 1u << value_of(balls.keys[match]);
        usearch_geospatial_within(&(usearch_geospatial_within_t) {.index = viewed,
                                                                  .executor = storage,
                                                                  .outcome = &outcome,
                                                                  .count = 1,
                                                                  .queries = observations,
                                                                  .relation = usearch_geospatial_relation_near_k,
                                                                  .tolerance = {.metres = 500},
                                                                  .matches = &balls,
                                                                  .written = &nvme_reported,
                                                                  .arena = arena});
        expect(!outcome.message, outcome.message);
        for (size_t match = 0; match != nvme_reported; ++match) nvme_keys |= 1u << value_of(balls.keys[match]);
        expect(ram_reported == 3 && nvme_reported == 3 && ram_keys == 7 && nvme_keys == 7,
               "A view from storage reports the containment RAM reports, Paris left out");
        usearch_geospatial_save(&(
            usearch_geospatial_save_t) {.index = viewed, .executor = storage, .outcome = &outcome, .path = path_view});
        expect_eq(outcome.status, usearch_status_unsupported_k, "A view from storage already is its checkpoint");
        usearch_geospatial_free(viewed);
    }
    usearch_executor_free(storage);
    remove(path);

    usearch_geospatial_free(index);
    usearch_arena_free(arena);
    usearch_executor_free(executor);
    printf("Test: Geospatial Round Trip - PASSED\n");
}

/** Shapes and geospatial absorb and refine alike: one index absorbs another of its geometry and
 *  refuses one of another, and a one-shot refine after leaves every entry reachable. */
void test_shapes_and_geospatial_absorb(void) {
    printf("Test: Shapes & Geospatial Absorb\n");
    enum { entries_per_index = 120, points_per_shape = 8 };
    usearch_outcome_t outcome = {0};
    usearch_executor_t const executor = create_executor(0);
    usearch_key_t keys[2 * entries_per_index];
    for (size_t entry = 0; entry != 2 * entries_per_index; ++entry) keys[entry] = key_of(entry);

    float points[2 * entries_per_index * points_per_shape * 3];
    for (size_t coordinate = 0; coordinate != sizeof(points) / sizeof(points[0]); ++coordinate)
        points[coordinate] = (float)rand() / (float)RAND_MAX;
    size_t offsets[entries_per_index + 1];
    for (size_t shape = 0; shape != entries_per_index + 1; ++shape) offsets[shape] = shape * points_per_shape;
    usearch_shapes_index_t shapes[3];
    for (size_t which = 0; which != 3; ++which) {
        usearch_shapes_init_ram(&(usearch_shapes_init_ram_t) {.index = &shapes[which],
                                                              .outcome = &outcome,
                                                              .metric = usearch_shapes_metric_chamfer_k,
                                                              .quantization = usearch_scalar_f32_k,
                                                              .points = which == 2 ? 0 : points_per_shape});
        expect(!outcome.message, outcome.message);
        if (which == 2) break;
        usearch_shapes_insert(&(usearch_shapes_insert_t) {
            .index = shapes[which],
            .executor = executor,
            .outcome = &outcome,
            .count = entries_per_index,
            .keys = {keys + which * entries_per_index, sizeof(usearch_key_t)},
            .shapes = {points + which * entries_per_index * points_per_shape * 3, offsets, usearch_scalar_f32_k},
            .expansion_factor = 4});
        expect(!outcome.message, outcome.message);
    }
    usearch_shapes_absorb(&(usearch_shapes_absorb_t) {.index = shapes[0],
                                                      .executor = executor,
                                                      .outcome = &outcome,
                                                      .count = 1,
                                                      .donors = shapes + 2});
    expect_eq(outcome.status, usearch_status_config_mismatch_k, "A donor of another point count is refused");
    expect(shapes[2] != NULL, "A refused donor stays the caller's");
    usearch_reachability_t reachability = {0};
    usearch_shapes_absorb(&(usearch_shapes_absorb_t) {.index = shapes[0],
                                                      .executor = executor,
                                                      .outcome = &outcome,
                                                      .count = 1,
                                                      .donors = shapes + 1,
                                                      .reachability = &reachability});
    expect(!outcome.message, outcome.message);
    expect(shapes[1] == NULL && reachability.audited == 2 * entries_per_index, "The donor's shapes must land");
    usearch_shapes_refine(&(usearch_shapes_refine_t) {.index = shapes[0],
                                                      .executor = executor,
                                                      .outcome = &outcome,
                                                      .reachability = &reachability});
    expect(!outcome.message, outcome.message);
    expect(reachability.reachable == 2 * entries_per_index, "Every absorbed shape must stay reachable");
    usearch_shapes_index_description_t shapes_description = {0};
    usearch_shapes_describe(&(usearch_shapes_describe_t) {.index = shapes[0],
                                                          .executor = executor,
                                                          .outcome = &outcome,
                                                          .description = &shapes_description});
    expect(!outcome.message, outcome.message);
    expect_eq(shapes_description.size, 2 * entries_per_index, "The absorbed sizes must add up");
    usearch_shapes_free(shapes[0]);
    usearch_shapes_free(shapes[2]);

    // Observations scattered over a city, a few hundred metres apart.
    usearch_geospatial_observation_t observations[2 * entries_per_index];
    for (size_t entry = 0; entry != 2 * entries_per_index; ++entry)
        observations[entry] = observed_at(51.45 + 0.1 * rand() / RAND_MAX, -0.15 + 0.1 * rand() / RAND_MAX, NAN);
    usearch_geospatial_index_t places[2];
    for (size_t which = 0; which != 2; ++which) {
        usearch_geospatial_init_ram(&(usearch_geospatial_init_ram_t) {.index = &places[which], .outcome = &outcome});
        expect(!outcome.message, outcome.message);
        usearch_geospatial_insert(
            &(usearch_geospatial_insert_t) {.index = places[which],
                                            .executor = executor,
                                            .outcome = &outcome,
                                            .count = entries_per_index,
                                            .keys = {keys + which * entries_per_index, sizeof(usearch_key_t)},
                                            .observations = observations + which * entries_per_index,
                                            .expansion_factor = 16});
        expect(!outcome.message, outcome.message);
    }
    usearch_geospatial_absorb(&(usearch_geospatial_absorb_t) {.index = places[0],
                                                              .executor = executor,
                                                              .outcome = &outcome,
                                                              .count = 1,
                                                              .donors = places + 1});
    expect(!outcome.message, outcome.message);
    expect(places[1] == NULL, "The donor's observations must land");
    usearch_geospatial_refine(&(usearch_geospatial_refine_t) {.index = places[0],
                                                              .executor = executor,
                                                              .outcome = &outcome,
                                                              .reachability = &reachability});
    expect(!outcome.message, outcome.message);
    expect(reachability.audited == 2 * entries_per_index && reachability.reachable == 2 * entries_per_index,
           "Every absorbed observation must stay reachable");
    usearch_geospatial_free(places[0]);
    usearch_executor_free(executor);
    printf("Test: Shapes & Geospatial Absorb - PASSED\n");
}

/** A GPU geospatial index refuses a terrain and lines of sight, inserts and describes its rows, and
 *  finds every observation itself first, at zero metres, on host pages; the verbs it lacks refuse.
 *  Skipped without a GPU. */
void test_geospatial_vram(void) {
    usearch_outcome_t outcome = {0};
    size_t devices_count = 0;
    usearch_gpu_devices(&(usearch_gpu_devices_t) {.outcome = &outcome, .devices_count = &devices_count});
    if (!devices_count) return;
    printf("Test: Geospatial VRAM\n");
    int32_t const devices[1] = {0};
    usearch_vram_options_t const vram = {.devices = devices, .devices_count = 1};

    float const height = 0;
    usearch_geospatial_terrain_t const terrain = {.rows = 1,
                                                  .columns = 1,
                                                  .row_step_degrees = 1,
                                                  .column_step_degrees = 1,
                                                  .heights = &height};
    usearch_geospatial_index_t refused = NULL;
    usearch_geospatial_init_vram(
        &(usearch_geospatial_init_vram_t) {.index = &refused, .outcome = &outcome, .terrain = terrain, .vram = vram});
    expect_eq(outcome.status, usearch_status_unsupported_k, "A GPU geospatial index tests no lines of sight");
    expect(refused == NULL, "A refused initialization returns no handle");

    usearch_geospatial_index_t index = NULL;
    usearch_geospatial_init_vram(
        &(usearch_geospatial_init_vram_t) {.index = &index, .outcome = &outcome, .vram = vram});
    if (outcome.status == usearch_status_device_k) {
        printf("Test: Geospatial VRAM - SKIPPED: %s\n", outcome.message);
        return;
    }
    expect(!outcome.message, outcome.message);
    usearch_executor_t executor = NULL;
    usearch_executor_for_vram(
        &(usearch_executor_for_vram_t) {.executor = &executor, .outcome = &outcome, .vram = vram});
    expect(!outcome.message, outcome.message);

    usearch_geospatial_observation_t const observations[] = {
        observed_at(51.5, -0.1, NAN), observed_at(51.501, -0.1, NAN), observed_at(51.5, -0.101, NAN),
        observed_at(51.501, -0.101, NAN)};
    usearch_key_t const keys[] = {key_of(0), key_of(1), key_of(2), key_of(3)};
    usearch_geospatial_insert(&(usearch_geospatial_insert_t) {.index = index,
                                                              .executor = executor,
                                                              .outcome = &outcome,
                                                              .count = 4,
                                                              .keys = {keys, sizeof(usearch_key_t)},
                                                              .observations = observations});
    expect(!outcome.message, outcome.message);
    usearch_geospatial_index_description_t description = {0};
    usearch_geospatial_describe(&(usearch_geospatial_describe_t) {.index = index,
                                                                  .executor = executor,
                                                                  .outcome = &outcome,
                                                                  .description = &description});
    expect(!outcome.message, outcome.message);
    expect(description.size == 4 && description.metadata.connectivity == 16,
           "A GPU index describes its rows, twelve neighbors rounded to a power of two");

    usearch_arena_t const arena = create_arena(NULL);
    usearch_matches_view_t matches = {0};
    usearch_geospatial_search_t search = {.index = index,
                                          .executor = executor,
                                          .outcome = &outcome,
                                          .count = 4,
                                          .queries = observations,
                                          .relation = usearch_geospatial_relation_near_k,
                                          .matches = &matches,
                                          .arena = arena,
                                          .wanted = 2};
    usearch_geospatial_search(&search);
    expect(!outcome.message, outcome.message);
    for (size_t row = 0; row != 4; ++row) {
        expect_eq(matches.counts[row], 2, "Every observation finds itself and a neighbor");
        expect(memcmp(&matches.keys[row * 2], &keys[row], sizeof(usearch_key_t)) == 0 &&
                   matches.scores[row * 2] < 1e-3 && matches.scores[row * 2 + 1] > 60,
               "An observation finds itself first, at zero metres");
    }
    search.relation = usearch_geospatial_relation_visible_k;
    usearch_geospatial_search(&search);
    expect_eq(outcome.status, usearch_status_unsupported_k, "A GPU geospatial index tests no lines of sight");

    char const path[] = "usearch-test-c-geospatial-vram.safegraphs";
    usearch_geospatial_save(&(usearch_geospatial_save_t) {.index = index,
                                                          .executor = executor,
                                                          .outcome = &outcome,
                                                          .path = {(uint8_t const*)path, sizeof(path) - 1}});
    expect_eq(outcome.status, usearch_status_unsupported_k, "A GPU geospatial index saves no checkpoint yet");

    usearch_arena_free(arena);
    usearch_geospatial_free(index);
    usearch_executor_free(executor);
    printf("Test: Geospatial VRAM - PASSED\n");
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
                                                        .quantization = usearch_scalar_u1x8_k,
                                                        .dimensions = dimensions,
                                                        .connectivity = 16});
    expect(!outcome.message, outcome.message);
    usearch_executor_t const executor = create_executor(0);
    usearch_dense_reserve(
        &(usearch_dense_reserve_t) {.index = index, .executor = executor, .outcome = &outcome, .capacity = rows_count});
    expect(!outcome.message, outcome.message);

    // Zeros and ones survive the cast into `u1x8` exactly, so every product is known up front.
    float data[rows_count * dimensions];
    usearch_key_t keys[rows_count];
    for (size_t row = 0; row != rows_count; ++row) {
        keys[row] = key_of(row);
        for (size_t column = 0; column != dimensions; ++column)
            data[row * dimensions + column] = (float)((row + column) % 3 == 0);
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

    // Narrower outputs land the same rows to their own precision, and a NULL carry keeps no state.
    attend.outputs_scalar = usearch_scalar_bf16_k;
    attend.carry = NULL;
    usearch_dense_attend(&attend);
    expect(!outcome.message, outcome.message);
    expect(outputs.scalar == usearch_scalar_bf16_k && outputs.stride == dimensions * sizeof(uint16_t),
           "Outputs land packed in the scalar asked for");
    uint16_t const* const narrowed = (uint16_t const*)outputs.data;
    uint32_t const widened_bits = (uint32_t)narrowed[0] << 16;
    float widened;
    memcpy(&widened, &widened_bits, sizeof(widened));
    expect(fabsf(widened - 4.25f / 1.875f) < 1e-2f, "A bf16 output must round the f32 one");
    attend.outputs_scalar = 0;
    attend.carry = carry;

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

/** Every landed run of @p matches against @p exact, `[query][document]` totals: as many keys as
 *  @p expected_count per query, largest first, each at its exact total, even keys alone when
 *  @p even_only. */
static void expect_maxsim_tape(usearch_matches_tape_view_t matches, size_t written, size_t queries, size_t documents,
                               double const* exact, size_t const* expected_count, bool even_only) {
    expect(matches.offsets && matches.offsets[0] == 0, "A tape starts at zero");
    expect_eq(written, matches.offsets[queries], "The written count is the tape's last offset");
    for (size_t query = 0; query != queries; ++query) {
        size_t const first = matches.offsets[query], end = matches.offsets[query + 1];
        expect_eq(end - first, expected_count[query], "Every query lands as many keys as it kept");
        for (size_t entry = first; entry != end; ++entry) {
            uint64_t const document = value_of(matches.keys[entry]);
            expect(document < documents, "A landed key names a document");
            expect(!even_only || document % 2 == 0, "A predicate narrows the landed keys");
            expect(fabs(matches.scores[entry] - exact[query * documents + document]) < 1e-3,
                   "A landed key scores its exact summed maxima");
            expect(entry == first || matches.scores[entry] <= matches.scores[entry - 1], "Keys land largest first");
        }
    }
}

/** Documents of 2 to 4 token rows hold small integers, which `bf16` keeps exactly, so every MaxSim
 *  total is exact: queries of 1, 2 and 3 rows land a ragged tape, walks and a scan alike, every
 *  document without a bound, the best one with one while `required` counts them all, and even
 *  documents alone under a predicate. */
void test_maxsim(void) {
    printf("Test: MaxSim\n");

    enum { documents = 6, dimensions = 8, rows_count = 18, query_rows = 6, queries = 3 };
    size_t const document_rows[documents] = {2, 3, 4, 2, 3, 4};
    usearch_outcome_t outcome = {0};
    usearch_dense_index_t index = NULL;
    usearch_dense_init_ram(&(usearch_dense_init_ram_t) {.index = &index,
                                                        .outcome = &outcome,
                                                        .metric = usearch_dense_metric_inner_k,
                                                        .quantization = usearch_scalar_bf16_k,
                                                        .dimensions = dimensions,
                                                        .connectivity = 16});
    expect(!outcome.message, outcome.message);
    usearch_executor_t const executor = create_executor(0);
    usearch_dense_reserve(
        &(usearch_dense_reserve_t) {.index = index, .executor = executor, .outcome = &outcome, .capacity = rows_count});
    expect(!outcome.message, outcome.message);

    float rows[rows_count * dimensions];
    usearch_key_t keys[rows_count];
    for (size_t document = 0, row = 0; document != documents; ++document)
        for (size_t token = 0; token != document_rows[document]; ++token, ++row) {
            keys[row] = key_of(document);
            for (size_t column = 0; column != dimensions; ++column)
                rows[row * dimensions + column] = (float)((row * 7 + column * 3 + document) % 4);
        }
    usearch_dense_insert(
        &(usearch_dense_insert_t) {.index = index,
                                   .executor = executor,
                                   .outcome = &outcome,
                                   .count = rows_count,
                                   .keys = {keys, sizeof(usearch_key_t)},
                                   .vectors = {rows, dimensions * sizeof(float), dimensions, usearch_scalar_f32_k}});
    expect(!outcome.message, outcome.message);

    // Every query row's best dot among a document's tokens, summed over the query's rows.
    float query_vectors[query_rows * dimensions];
    for (size_t position = 0; position != query_rows * dimensions; ++position)
        query_vectors[position] = (float)((position * 5) % 3);
    size_t const query_offsets[queries + 1] = {0, 1, 3, 6};
    double exact[queries * documents] = {0};
    for (size_t query = 0; query != queries; ++query)
        for (size_t query_row = query_offsets[query]; query_row != query_offsets[query + 1]; ++query_row)
            for (size_t document = 0, row = 0; document != documents; row += document_rows[document++]) {
                double best = -INFINITY;
                for (size_t token = 0; token != document_rows[document]; ++token) {
                    double dot = 0;
                    for (size_t column = 0; column != dimensions; ++column)
                        dot += (double)query_vectors[query_row * dimensions + column] *
                               rows[(row + token) * dimensions + column];
                    if (dot > best) best = dot;
                }
                exact[query * documents + document] += best;
            }

    usearch_arena_t const arena = create_arena(executor);
    usearch_matches_tape_view_t matches = {0};
    size_t written = 0, required = 0;
    usearch_dense_maxsim_t maxsim = {
        .index = index,
        .executor = executor,
        .outcome = &outcome,
        .count = queries,
        .queries = {query_vectors, dimensions * sizeof(float), dimensions, usearch_scalar_f32_k},
        .query_offsets = query_offsets,
        .matches = &matches,
        .written = &written,
        .required = &required,
        .arena = arena,
        .expansion_factor = rows_count};
    size_t const every[queries] = {documents, documents, documents};
    size_t const best_only[queries] = {1, 1, 1};
    size_t const even_ones[queries] = {documents / 2, documents / 2, documents / 2};
    usearch_traversal_t const traversals[2] = {usearch_traversal_walk_k, usearch_traversal_scan_k};
    for (size_t form = 0; form != 2; ++form) {
        maxsim.traversal = traversals[form];
        maxsim.wanted = 0;
        maxsim.predicate = (usearch_predicate_t) {0};
        usearch_dense_maxsim(&maxsim);
        expect(!outcome.message, outcome.message);
        expect_maxsim_tape(matches, written, queries, documents, exact, every, false);

        maxsim.wanted = 1;
        usearch_dense_maxsim(&maxsim);
        expect(!outcome.message, outcome.message);
        expect_maxsim_tape(matches, written, queries, documents, exact, best_only, false);
        expect_eq(required, queries * documents, "Required counts every key scored before wanted cut them");
        for (size_t query = 0; query != queries; ++query)
            for (size_t document = 0; document != documents; ++document)
                expect(exact[query * documents + document] <= matches.scores[matches.offsets[query]] + 1e-3,
                       "The one kept key is the best");

        size_t widest_batch = 0;
        maxsim.wanted = 0;
        maxsim.predicate = (usearch_predicate_t) {admits_even, &widest_batch};
        usearch_dense_maxsim(&maxsim);
        expect(!outcome.message, outcome.message);
        expect_maxsim_tape(matches, written, queries, documents, exact, even_ones, true);
        expect_eq(required, queries * documents / 2, "Required counts no key the predicate rejected");
    }

    usearch_arena_free(arena);
    usearch_executor_free(executor);
    usearch_dense_free(index);
    printf("Test: MaxSim - PASSED\n");
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

    usearch_reachability_t reachability = {0};
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

    usearch_dense_search(&(usearch_dense_search_t) {.index = view,
                                                    .executor = storage,
                                                    .outcome = &outcome,
                                                    .count = rows_count,
                                                    .queries = {data, row_bytes, dimensions, usearch_scalar_f32_k},
                                                    .matches = &matches,
                                                    .arena = arena,
                                                    .wanted = 1,
                                                    .traversal = usearch_traversal_scan_k});
    expect(!outcome.message, outcome.message);
    for (size_t row = 0; row != rows_count; ++row)
        expect(matches.counts[row] == 1 && memcmp(matches.keys[row].bytes, keys[row].bytes, sizeof(usearch_key_t)) == 0,
               "An exact scan of a view must find every self-query itself");
    expect_expensive_narrows(view, storage, arena, data, rows_count, dimensions);

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

/** A SafeGraphs monolith shaped like a benchmark's save - `u32` keys, `bf16` rows, 16 neighbors
 *  each - framed here by hand, opens through the NVMe view with every key zero-extended into the
 *  16-byte ones the ABI speaks, so it counts and ranks under the identity it was saved with. */
void test_nvme_view_narrow_keys(void) {
    printf("Test: NVMe View of u32 Keys...\n");

    enum { rows_count = 4, dimensions = 8, connectivity = 16, alignment = 64 };
    enum { keys_bytes = rows_count * 4, values_bytes = rows_count * dimensions * 2 };
    enum { values_first = alignment, neighbors_first = values_first + alignment };
    enum { neighbors_bound = neighbors_first + rows_count * connectivity * 4 };
    char json[512];
    int json_length = snprintf(json, sizeof(json), "{\"__metadata__\":{\"alignment\":%d,\"metric\":\"dot\"},",
                               alignment);
    json_length += snprintf(json + json_length, sizeof(json) - (size_t)json_length,
                            "\"key\":{\"dtype\":\"u32\",\"data\":[0,%d]},", keys_bytes);
    json_length += snprintf(json + json_length, sizeof(json) - (size_t)json_length,
                            "\"value\":{\"dtype\":\"bf16\",\"shape\":%d,\"data\":[%d,%d]},", dimensions, values_first,
                            values_first + values_bytes);
    json_length += snprintf(json + json_length, sizeof(json) - (size_t)json_length,
                            "\"neighbors\":{\"dtype\":\"u32\",\"shape\":%d,\"data\":[%d,%d]}}", connectivity,
                            neighbors_first, neighbors_bound);
    size_t const data_offset = (16 + (size_t)json_length + alignment - 1) / alignment * alignment;
    size_t const file_bytes = data_offset + neighbors_bound;
    uint8_t* const file = (uint8_t*)calloc(file_bytes, 1);
    expect(file, "Failed to allocate memory");
    memcpy(file, "SAFEGRPH", 8);
    for (unsigned byte_index = 0; byte_index != 8; ++byte_index)
        file[8 + byte_index] = (uint8_t)((uint64_t)(data_offset - 16) >> (byte_index * 8));
    memset(file + 16, ' ', data_offset - 16);
    memcpy(file + 16, json, (size_t)json_length);

    // Row `r` holds `r + 1` in every dimension, exact in `bf16`'s upper half of the `f32` bits, is
    // keyed `100 + r`, and links every other row before the all-ones sentinels.
    for (size_t row = 0; row != rows_count; ++row) {
        uint8_t* const key = file + data_offset + row * 4;
        for (unsigned byte_index = 0; byte_index != 4; ++byte_index)
            key[byte_index] = (uint8_t)((100 + row) >> (byte_index * 8));
        float const value = (float)(row + 1);
        uint32_t bits;
        memcpy(&bits, &value, sizeof(bits));
        for (size_t position = 0; position != dimensions; ++position) {
            uint8_t* const scalar = file + data_offset + values_first + (row * dimensions + position) * 2;
            scalar[0] = (uint8_t)(bits >> 16);
            scalar[1] = (uint8_t)(bits >> 24);
        }
        uint8_t* const neighbors = file + data_offset + neighbors_first + row * connectivity * 4;
        memset(neighbors, 0xFF, connectivity * 4);
        for (size_t other = 0, slot = 0; other != rows_count; ++other) {
            if (other == row) continue;
            memset(neighbors + slot * 4, 0, 4);
            neighbors[slot * 4] = (uint8_t)other;
            ++slot;
        }
    }
    char const path[] = "usearch-test-c-nvme-u32.safegraphs";
    FILE* const stream = fopen(path, "wb");
    expect(stream && fwrite(file, 1, file_bytes, stream) == file_bytes && fclose(stream) == 0,
           "Failed to write the hand-framed monolith");
    free(file);

    usearch_outcome_t outcome = {0};
    usearch_executor_t storage = NULL;
    usearch_dense_index_t view = NULL;
    usearch_bytes_view_t const path_view = {(uint8_t const*)path, sizeof(path) - 1};
    usearch_executor_for_nvme(&(usearch_executor_for_nvme_t) {.executor = &storage, .outcome = &outcome, .threads = 2});
    if (outcome.status == usearch_status_success_k)
        usearch_dense_view_nvme(
            &(usearch_dense_view_nvme_t) {.index = &view, .executor = storage, .outcome = &outcome, .path = path_view});
    if (outcome.status == usearch_status_unsupported_k) {
        printf("Test: NVMe View of u32 Keys - SKIPPED: %s\n", outcome.message);
        usearch_executor_free(storage);
        remove(path);
        return;
    }
    expect(!outcome.message, outcome.message);

    size_t entries = 0;
    usearch_dense_count(&(usearch_dense_count_t) {.index = view,
                                                  .executor = storage,
                                                  .outcome = &outcome,
                                                  .key = key_of(102),
                                                  .entries = &entries});
    expect(!outcome.message, outcome.message);
    expect_eq(entries, 1, "A `u32` key must count once under its widened identity");

    // Every row's dot with a positive query grows with its row number, so the last row ranks first.
    float query[dimensions];
    for (size_t position = 0; position != dimensions; ++position) query[position] = 1;
    usearch_arena_t const arena = create_arena(storage);
    usearch_matches_view_t matches = {0};
    usearch_dense_search(
        &(usearch_dense_search_t) {.index = view,
                                   .executor = storage,
                                   .outcome = &outcome,
                                   .count = 1,
                                   .queries = {query, dimensions * sizeof(float), dimensions, usearch_scalar_f32_k},
                                   .matches = &matches,
                                   .arena = arena,
                                   .wanted = 1});
    expect(!outcome.message, outcome.message);
    usearch_key_t const last = key_of(100 + rows_count - 1);
    expect(matches.counts[0] == 1 && memcmp(matches.keys[0].bytes, last.bytes, sizeof(usearch_key_t)) == 0,
           "A walk over `u32` keys must resolve them widened");

    usearch_arena_free(arena);
    usearch_dense_free(view);
    usearch_executor_free(storage);
    remove(path);
    printf("Test: NVMe View of u32 Keys - PASSED\n");
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
    usearch_traversal_t const scan = usearch_traversal_from_name((usearch_bytes_view_t) {(uint8_t const*)"scan", 4});
    expect_eq(scan, usearch_traversal_scan_k, "A traversal reads back from its name");
    expect(strcmp(usearch_traversal_name(usearch_traversal_walk_k), "walk") == 0, "A traversal spells its name");

    // A machine without GPUs lists none, so the call succeeds everywhere and names what it lists.
    usearch_gpu_device_t devices[8];
    size_t devices_count = 0;
    usearch_outcome_t outcome = {0};
    usearch_gpu_devices(&(usearch_gpu_devices_t) {.outcome = &outcome,
                                                  .devices = devices,
                                                  .devices_capacity = 8,
                                                  .devices_count = &devices_count});
    expect_eq(outcome.status, usearch_status_success_k, "Listing the GPUs must succeed");
    for (size_t position = 0; position != devices_count && position != 8; ++position) {
        expect(devices[position].vendor != usearch_gpu_vendor_unknown_k, "A listed GPU names its vendor");
        expect(strcmp(usearch_gpu_class_name(devices[position].gpu_class), "unknown") != 0,
               "A listed GPU names its class");
        printf("- GPU %zu: %s, %s, %u compute units, %llu MiB\n", position, devices[position].name,
               usearch_gpu_class_name(devices[position].gpu_class), devices[position].compute_units,
               (unsigned long long)(devices[position].memory_bytes >> 20));
    }
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
    // Landmarks are drawn per thread, so only one as wide as a call without an executor
    // draws the same.
    usearch_outcome_t spawned = {0};
    usearch_executor_t executor = NULL;
    usearch_executor_for_ram(&(usearch_executor_for_ram_t) {.executor = &executor, .outcome = &spawned, .threads = 1});
    expect(!spawned.message, spawned.message);
    usearch_dense_index_t const index = create_filled_index(rows, executor);
    search_outputs_t blocking = {.arena = create_arena(executor)};
    search_outputs_t deferred = {.arena = create_arena(executor)};

    search_every_row(index, executor, usearch_completion_blocking_k, rows, &blocking);
    expect(!blocking.outcome.message, blocking.outcome.message);

    search_every_row(index, executor, usearch_completion_deferred_k, rows, &deferred);
    expect_eq(usearch_outcome_status(&deferred.outcome), usearch_status_deferred_k,
              "A submitted call is deferred until waited on");
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
    expect_eq(usearch_outcome_status(&outputs.outcome), usearch_status_deferred_k,
              "A submitted call is deferred until waited on");

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

/** A three-part checkpoint loaded onto one GPU keeps a shard per part on it, loads every row, and
 *  reaches exactly the rows the saved graph reached. Skipped without a GPU. */
void test_vram_parts_share_a_device(size_t const collection_size, size_t const dimensions) {
    usearch_outcome_t outcome = {0};
    size_t devices_count = 0;
    usearch_gpu_devices(&(usearch_gpu_devices_t) {.outcome = &outcome, .devices_count = &devices_count});
    if (!devices_count) return;
    printf("Test: VRAM Parts Share a Device... %zu vectors, %zu dimensions\n", collection_size, dimensions);

    // The three parts come from naming the first GPU thrice; the load names it once.
    int32_t const devices[3] = {0, 0, 0};
    usearch_vram_options_t const thrice = {.devices = devices, .devices_count = 3};
    usearch_vram_options_t const once = {.devices = devices, .devices_count = 1};
    usearch_dense_index_t index = NULL;
    usearch_dense_init_vram(&(usearch_dense_init_vram_t) {.index = &index,
                                                          .outcome = &outcome,
                                                          .metric = usearch_dense_metric_angular_k,
                                                          .quantization = usearch_scalar_bf16_k,
                                                          .dimensions = dimensions,
                                                          .vram = thrice});
    expect(!outcome.message, outcome.message);
    usearch_executor_t executor = NULL;
    usearch_executor_for_vram(
        &(usearch_executor_for_vram_t) {.executor = &executor, .outcome = &outcome, .vram = thrice});
    expect(!outcome.message, outcome.message);

    float* const data = create_vectors(collection_size, dimensions);
    usearch_key_t* const keys = (usearch_key_t*)malloc(collection_size * sizeof(usearch_key_t));
    expect(keys, "Failed to allocate keys");
    for (size_t row = 0; row != collection_size; ++row) keys[row] = key_of(row);
    size_t const row_bytes = dimensions * sizeof(float);
    usearch_dense_reserve(&(usearch_dense_reserve_t) {.index = index,
                                                      .executor = executor,
                                                      .outcome = &outcome,
                                                      .capacity = collection_size});
    expect(!outcome.message, outcome.message);
    usearch_dense_insert(&(usearch_dense_insert_t) {.index = index,
                                                    .executor = executor,
                                                    .outcome = &outcome,
                                                    .count = collection_size,
                                                    .keys = {keys, sizeof(usearch_key_t)},
                                                    .vectors = {data, row_bytes, dimensions, usearch_scalar_f32_k}});
    expect(!outcome.message, outcome.message);
    // Insertions may strand rows, so the reload is judged against the saved graph's own audit.
    usearch_reachability_t saved_reachability = {0};
    usearch_dense_health(&(usearch_dense_health_t) {.index = index,
                                                    .executor = executor,
                                                    .outcome = &outcome,
                                                    .reachability = &saved_reachability});
    expect(!outcome.message, outcome.message);
    char const path[] = "usearch-test-c-parts.json";
    usearch_bytes_view_t const path_view = {(uint8_t const*)path, sizeof(path) - 1};
    usearch_dense_save(
        &(usearch_dense_save_t) {.index = index, .executor = executor, .outcome = &outcome, .path = path_view});
    expect(!outcome.message, outcome.message);
    usearch_executor_free(executor);
    usearch_dense_free(index);

    // An executor over the one named device drives every shard there, all on its one stream.
    usearch_executor_for_vram(
        &(usearch_executor_for_vram_t) {.executor = &executor, .outcome = &outcome, .vram = once});
    expect(!outcome.message, outcome.message);
    usearch_dense_load_vram(&(usearch_dense_load_vram_t) {.index = &index,
                                                          .executor = executor,
                                                          .outcome = &outcome,
                                                          .path = path_view,
                                                          .vram = once});
    expect(!outcome.message, outcome.message);

    usearch_dense_index_description_t description = {0};
    usearch_dense_describe(&(usearch_dense_describe_t) {.index = index,
                                                        .executor = executor,
                                                        .outcome = &outcome,
                                                        .description = &description});
    expect(!outcome.message, outcome.message);
    expect_eq(description.size, collection_size, "The parts must load every saved row onto one device");
    usearch_reachability_t reachability = {0};
    usearch_dense_health(&(usearch_dense_health_t) {.index = index,
                                                    .executor = executor,
                                                    .outcome = &outcome,
                                                    .reachability = &reachability});
    expect(!outcome.message, outcome.message);
    expect(reachability.audited == saved_reachability.audited && reachability.reachable == saved_reachability.reachable,
           "The loaded parts must reach exactly the rows the saved ones did");

    // The matches land in device memory, which plain C cannot read;
    // the C++ VRAM suite checks their recall.
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
    expect(matches.keys && matches.counts, "A VRAM search must land its matches in the arena");

    usearch_arena_free(arena);
    usearch_dense_free(index);
    usearch_executor_free(executor);
    remove(path);
    for (int part = 0; part != 4; ++part) {
        char part_path[64];
        snprintf(part_path, sizeof(part_path), "usearch-test-c-parts-%05d-of-00003.safegraphs", part);
        remove(part_path);
    }
    free(keys);
    free(data);
    printf("Test: VRAM Parts Share a Device - PASSED\n");
}

/** An exact scan on the first GPU scores every row once per query and refuses a predicate as a walk
 *  does where the vendor scans, and refuses with `usearch_status_unsupported_k` elsewhere. */
void test_vram_scan(size_t const collection_size, size_t const dimensions) {
    usearch_outcome_t outcome = {0};
    usearch_gpu_device_t device = {0};
    size_t devices_count = 0;
    usearch_gpu_devices(&(usearch_gpu_devices_t) {.outcome = &outcome,
                                                  .devices = &device,
                                                  .devices_capacity = 1,
                                                  .devices_count = &devices_count});
    if (!devices_count) return;
    printf("Test: VRAM Scan... %zu vectors, %zu dimensions\n", collection_size, dimensions);

    int32_t const devices[1] = {0};
    usearch_vram_options_t const vram = {.devices = devices, .devices_count = 1};
    usearch_dense_index_t index = NULL;
    usearch_dense_init_vram(&(usearch_dense_init_vram_t) {.index = &index,
                                                          .outcome = &outcome,
                                                          .metric = usearch_dense_metric_euclidean_k,
                                                          .quantization = usearch_scalar_bf16_k,
                                                          .dimensions = dimensions,
                                                          .vram = vram});
    expect(!outcome.message, outcome.message);
    usearch_executor_t executor = NULL;
    usearch_executor_for_vram(
        &(usearch_executor_for_vram_t) {.executor = &executor, .outcome = &outcome, .vram = vram});
    expect(!outcome.message, outcome.message);

    float* const data = create_vectors(collection_size, dimensions);
    usearch_key_t* const keys = (usearch_key_t*)malloc(collection_size * sizeof(usearch_key_t));
    expect(keys, "Failed to allocate keys");
    for (size_t row = 0; row != collection_size; ++row) keys[row] = key_of(row);
    size_t const row_bytes = dimensions * sizeof(float);
    usearch_dense_reserve(&(usearch_dense_reserve_t) {.index = index,
                                                      .executor = executor,
                                                      .outcome = &outcome,
                                                      .capacity = collection_size});
    expect(!outcome.message, outcome.message);
    usearch_dense_insert(&(usearch_dense_insert_t) {.index = index,
                                                    .executor = executor,
                                                    .outcome = &outcome,
                                                    .count = collection_size,
                                                    .keys = {keys, sizeof(usearch_key_t)},
                                                    .vectors = {data, row_bytes, dimensions, usearch_scalar_f32_k}});
    expect(!outcome.message, outcome.message);

    // The matches land in device memory, which plain C cannot read; the C++ VRAM suite checks them.
    usearch_arena_t const arena = create_arena(executor);
    usearch_matches_view_t matches = {0};
    usearch_traversal_stats_t stats = {0};
    usearch_dense_search_t search = {.index = index,
                                     .executor = executor,
                                     .outcome = &outcome,
                                     .count = collection_size,
                                     .queries = {data, row_bytes, dimensions, usearch_scalar_f32_k},
                                     .matches = &matches,
                                     .arena = arena,
                                     .wanted = 10,
                                     .traversal = usearch_traversal_scan_k,
                                     .stats = &stats};
    usearch_dense_search(&search);
    if (device.vendor == usearch_gpu_vendor_nvidia_k) {
        expect(!outcome.message, outcome.message);
        expect(matches.keys && matches.counts, "A VRAM scan lands its matches in the arena");
        expect_eq(stats.computed_distances, collection_size * collection_size, "A scan scores every row per query");
        size_t widest_batch = 0;
        search.predicate = (usearch_predicate_t) {admits_even, &widest_batch};
        usearch_dense_search(&search);
        expect_eq(outcome.status, usearch_status_unsupported_k, "A host predicate cannot narrow a device scan");
    }
    else expect_eq(outcome.status, usearch_status_unsupported_k, "Only NVIDIA devices scan");

    usearch_arena_free(arena);
    usearch_dense_free(index);
    usearch_executor_free(executor);
    free(keys);
    free(data);
    printf("Test: VRAM Scan - PASSED\n");
}

int main(int argc, char* argv[]) {
    (void)argc;
    (void)argv;
    install_crash_handlers();

    printf("USearch %s\n", usearch_version_name());
    printf("- Compiled for: %s\n", usearch_capabilities_compiled());
    printf("- This machine: %s\n", usearch_capabilities_available());
    unsigned const seed = read_seed();
    char const* const filter = getenv("USEARCH_FILTER");
    if (filter && *filter) test_filter = filter;
    srand(seed);
    printf("- Seed: %u\n", seed);
    printf("- Filter: %s\n", test_filter ? test_filter : "none");

    if (selected("library_probes")) test_library_probes();
    if (selected("outcome_contract")) test_outcome_contract(64);

    size_t const collection_sizes[] = {11, 512};
    size_t const dimension_counts[] = {83, 256};
    for (size_t size_index = 0; size_index != 2; ++size_index)
        for (size_t dimension_index = 0; dimension_index != 2; ++dimension_index) {
            if (selected("init")) test_init(collection_sizes[size_index], dimension_counts[dimension_index]);
            if (selected("add_and_search"))
                test_add_and_search(collection_sizes[size_index], dimension_counts[dimension_index]);
        }
    if (selected("add_progress")) test_add_progress();
    if (selected("absorb")) test_absorb(dimension_counts[0]);
    if (selected("sequence_absorb")) test_sequence_absorb();
    if (selected("count_and_get")) test_count_and_get(dimension_counts[0]);
    if (selected("count_and_get")) test_count_and_get(dimension_counts[1]);
    if (selected("arenas")) test_arenas(dimension_counts[0]);
    if (selected("walks_name_their_index")) test_walks_name_their_index(dimension_counts[0]);
    if (selected("sample")) test_sample();
    if (selected("cursor_steps")) test_cursor_steps();
    if (selected("sequence_round_trip")) test_sequence_round_trip();
    if (selected("sequence_vram")) test_sequence_vram();
    if (selected("shapes_round_trip")) test_shapes_round_trip();
    if (selected("shapes_vram")) test_shapes_vram();
    if (selected("geospatial_round_trip")) test_geospatial_round_trip();
    if (selected("shapes_and_geospatial_absorb")) test_shapes_and_geospatial_absorb();
    if (selected("geospatial_vram")) test_geospatial_vram();
#if USEARCH_DENSE_DTYPES >= 2
    if (selected("multiply_packed")) test_multiply_packed();
#endif
#if USEARCH_DENSE_DTYPES >= 3
    if (selected("attend")) test_attend();
#endif
    if (selected("maxsim")) test_maxsim();
    if (selected("nvme_view")) test_nvme_view(dimension_counts[0]);
    if (selected("nvme_view_narrow_keys")) test_nvme_view_narrow_keys();
    if (selected("vram_parts_share_a_device")) test_vram_parts_share_a_device(collection_sizes[1], dimension_counts[1]);
    if (selected("vram_scan")) test_vram_scan(collection_sizes[1], dimension_counts[1]);
    if (selected("deferred_search")) test_deferred_search();
    if (selected("deferred_limit")) test_deferred_limit();
    if (selected("free_finishes_deferred")) test_free_finishes_deferred();
    if (selected("blocking_runs_after_earlier")) test_blocking_runs_after_earlier();
    if (selected("deferred_holds_arena")) test_deferred_holds_arena();

    printf("All tests passed!\n");
    return 0;
}
