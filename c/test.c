/**
 *  @file c/test.c
 *  @author Ash Vardanian
 *  @date June 25, 2023
 *  @brief Unit tests for the pure-C ABI of USearch.
 *
 *  Exercises the dense surface through the ABI every binding consumes: initialization and
 *  reservation, batch @c add, @c search, @c multiply and @c attend, retrieval by key, the arenas
 *  every output lands in, the contract of @c usearch_outcome_t, and deferred calls on an executor.
 *  The sequence, shapes and geospatial surfaces each get one checkpoint round trip. A blocking
 *  call's outcome is final once it returns; a deferred one is waited on or drained. Scoped to the
 *  verbs @c src/ instantiates today - the header runs ahead of the implementation, and a smoke
 *  test that cannot link proves nothing. The harness is intentionally dependency-free so it can
 *  run in the same matrix as the C++ tests and on cross-compilation targets where only the C
 *  runtime is available.
 *
 *  On startup we install a signal handler (see @c install_crash_handlers) that
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

#include "usearch/arrow.h"
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

/** A substring of the tests to run, from @c USEARCH_FILTER; @c NULL runs every test. */
static char const* test_filter = NULL;

/** Whether the test @p name passes @c USEARCH_FILTER. */
static bool selected(char const* name) { return !test_filter || strstr(name, test_filter) != NULL; }

/** The seed of @c rand: 42, unless @c USEARCH_SEED names another or is "random", which draws one
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

/** An angular @c bf16 RAM index over @p dimensions -wide rows, every other optional field
 *  derived. */
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

/** A RAM executor of @p threads threads, zero taking the default, every other field derived. */
usearch_executor_t create_executor(size_t const threads) {
    usearch_outcome_t outcome = {0};
    usearch_executor_t executor = NULL;
    usearch_executor_for_ram(
        &(usearch_executor_for_ram_t) {.executor = &executor, .outcome = &outcome, .threads = threads});
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

/** A tape over @p offsets as a caller holding @c size_t offsets lends them, in their own width. */
usearch_tape_view_t tape_of(void const* data, size_t const* offsets, usearch_scalar_t scalar) {
    usearch_tape_view_t tape = {.data = data, .scalar = scalar};
    if (sizeof(size_t) == sizeof(uint64_t))
        tape.layout = usearch_tape_u64_k, tape.offsets.u64 = (uint64_t const*)offsets;
    else tape.offsets.u32 = (uint32_t const*)offsets;
    return tape;
}

/** The offsets of an output @p tape, which the library writes as wide as @c size_t. */
size_t const* offsets_of(usearch_tape_view_t tape) {
    return sizeof(size_t) == sizeof(uint64_t) ? (size_t const*)tape.offsets.u64 : (size_t const*)tape.offsets.u32;
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

    // A geometry no graph can take is refused at init, before any shard grows.
    size_t const refused_geometries[][2] = {{0, 16}, {dimensions, 24}};
    for (size_t geometry = 0; geometry != 2; ++geometry) {
        usearch_dense_index_t malformed = NULL;
        usearch_dense_init_ram(&(usearch_dense_init_ram_t) {.index = &malformed,
                                                            .outcome = &outcome,
                                                            .metric = usearch_dense_metric_angular_k,
                                                            .quantization = usearch_scalar_bf16_k,
                                                            .dimensions = refused_geometries[geometry][0],
                                                            .connectivity = refused_geometries[geometry][1]});
        expect_eq(outcome.status, usearch_status_config_mismatch_k, "A malformed geometry is refused at init");
        expect(malformed == NULL, "A refused initialization returns no handle");
    }

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

/** Describes, counts, gets and searches @p index before any insert grew it, each answering nothing
 *  with success; the matches of a device arena land where plain C cannot read them. */
void expect_empty_reads(usearch_dense_index_t const index, usearch_executor_t const executor, size_t const dimensions,
                        bool const host_matches) {
    usearch_outcome_t outcome = {0};
    usearch_dense_index_description_t description = {0};
    usearch_dense_describe(&(usearch_dense_describe_t) {.index = index,
                                                        .executor = executor,
                                                        .outcome = &outcome,
                                                        .description = &description});
    expect(!outcome.message, outcome.message);
    expect_eq(description.size, 0, "An empty index describes no entries");

    size_t entries = 1;
    usearch_dense_count(&(usearch_dense_count_t) {.index = index,
                                                  .executor = executor,
                                                  .outcome = &outcome,
                                                  .key = key_of(0),
                                                  .entries = &entries});
    expect(!outcome.message, outcome.message);
    expect_eq(entries, 0, "An empty index counts no entries");

    usearch_arena_t const arena = create_arena(executor);
    usearch_rows_view_t vectors = {0};
    size_t written = 1;
    usearch_dense_get(&(usearch_dense_get_t) {.index = index,
                                              .executor = executor,
                                              .outcome = &outcome,
                                              .key = key_of(0),
                                              .vectors = &vectors,
                                              .written = &written,
                                              .arena = arena,
                                              .outputs_scalar = usearch_scalar_f32_k});
    expect(!outcome.message, outcome.message);
    expect(written == 0 && !vectors.data, "An empty index gets no rows");

    float* const query = create_vectors(1, dimensions);
    usearch_matches_view_t matches = {0};
    usearch_dense_search(
        &(usearch_dense_search_t) {.index = index,
                                   .executor = executor,
                                   .outcome = &outcome,
                                   .count = 1,
                                   .queries = {query, dimensions * sizeof(float), dimensions, usearch_scalar_f32_k},
                                   .matches = &matches,
                                   .arena = arena,
                                   .wanted = 4});
    expect(!outcome.message, outcome.message);
    expect(matches.counts != NULL, "An empty search still lands its counts");
    if (host_matches) expect_eq(matches.counts[0], 0, "An empty index finds no matches");
    free(query);
    usearch_arena_free(arena);
}

/** An index fresh from init, in RAM and on the first GPU when one is listed, answers every read
 *  with nothing and success. */
void test_empty_reads(size_t const dimensions) {
    printf("Test: Empty Reads... %zu dimensions\n", dimensions);
    usearch_dense_index_t const index = create_index(dimensions);
    usearch_executor_t const executor = create_executor(0);
    expect_empty_reads(index, executor, dimensions, true);
    usearch_executor_free(executor);
    usearch_dense_free(index);

    usearch_outcome_t outcome = {0};
    size_t devices_count = 0;
    usearch_gpu_devices(&(usearch_gpu_devices_t) {.outcome = &outcome, .devices_count = &devices_count});
    int32_t const devices[1] = {0};
    usearch_vram_options_t const vram = {.devices = devices, .devices_count = 1};
    usearch_dense_index_t device_index = NULL;
    if (devices_count)
        usearch_dense_init_vram(&(usearch_dense_init_vram_t) {.index = &device_index,
                                                              .outcome = &outcome,
                                                              .metric = usearch_dense_metric_angular_k,
                                                              .quantization = usearch_scalar_bf16_k,
                                                              .dimensions = dimensions,
                                                              .vram = vram});
    expect(device_index || !devices_count || outcome.status == usearch_status_device_k, outcome.message);
    if (device_index) {
        usearch_executor_t device_executor = NULL;
        usearch_executor_for_vram(
            &(usearch_executor_for_vram_t) {.executor = &device_executor, .outcome = &outcome, .vram = vram});
        expect(!outcome.message, outcome.message);
        expect_empty_reads(device_index, device_executor, dimensions, false);
        usearch_executor_free(device_executor);
        usearch_dense_free(device_index);
    }
    printf("Test: Empty Reads - PASSED\n");
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
 *  order up to the whole batch, and every entry is published. */
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

/** A RAM index bounded at a quarter of its rows holds four shards, and compacts in place under a
 *  bound of half its rows: nothing is consumed, and every key still resolves to one entry. A fresh
 *  index then compacts two halves into itself, which frees and nulls both sources. */
void test_compact(size_t const dimensions) {
    printf("Test: Compact... %zu dimensions\n", dimensions);

    enum { rows_count = 1200, quarter = rows_count / 4 };
    usearch_outcome_t outcome = {0};
    usearch_executor_t executor = NULL;
    usearch_executor_for_ram(&(usearch_executor_for_ram_t) {.executor = &executor, .outcome = &outcome, .threads = 2});
    expect(!outcome.message, outcome.message);
    float* const data = create_vectors(rows_count, dimensions);
    usearch_key_t* const keys = (usearch_key_t*)malloc(rows_count * sizeof(usearch_key_t));
    expect(keys, "Failed to allocate keys");
    for (size_t row = 0; row != rows_count; ++row) keys[row] = key_of(row);

    usearch_dense_index_t indexes[3] = {NULL, NULL, NULL};
    size_t const shard_capacities[3] = {quarter, 0, 0};
    for (size_t which = 0; which != 3; ++which) {
        usearch_dense_init_ram(&(usearch_dense_init_ram_t) {.index = &indexes[which],
                                                            .outcome = &outcome,
                                                            .metric = usearch_dense_metric_angular_k,
                                                            .quantization = usearch_scalar_bf16_k,
                                                            .dimensions = dimensions,
                                                            .connectivity = 16,
                                                            .ram = {.shard_capacity = shard_capacities[which]}});
        expect(!outcome.message, outcome.message);
    }
    // The bounded index takes every entry, one generation of shards per quarter.
    usearch_dense_reserve(&(usearch_dense_reserve_t) {.index = indexes[0],
                                                      .executor = executor,
                                                      .outcome = &outcome,
                                                      .capacity = rows_count});
    expect(!outcome.message, outcome.message);
    usearch_dense_insert(
        &(usearch_dense_insert_t) {.index = indexes[0],
                                   .executor = executor,
                                   .outcome = &outcome,
                                   .count = rows_count,
                                   .keys = {keys, sizeof(usearch_key_t)},
                                   .vectors = {data, dimensions * sizeof(float), dimensions, usearch_scalar_f32_k}});
    expect(!outcome.message, outcome.message);

    usearch_dense_compact(&(usearch_dense_compact_t) {.target = indexes[0],
                                                      .executor = executor,
                                                      .outcome = &outcome,
                                                      .count = 1,
                                                      .sources = indexes,
                                                      .shard_capacity = rows_count / 2});
    expect(!outcome.message, outcome.message);
    expect(indexes[0] != NULL, "Compacting in place consumes nothing");
    for (size_t row = 0; row != rows_count; ++row) {
        size_t entries = 0;
        usearch_dense_count(&(usearch_dense_count_t) {.index = indexes[0],
                                                      .executor = executor,
                                                      .outcome = &outcome,
                                                      .key = keys[row],
                                                      .entries = &entries});
        expect(!outcome.message, outcome.message);
        expect_eq(entries, 1, "Every key must resolve to one row after compacting in place");
    }

    // Out of place, both halves land in the fresh index, and their handles go.
    usearch_dense_index_t const target = indexes[2];
    usearch_dense_reserve(&(
        usearch_dense_reserve_t) {.index = indexes[1], .executor = executor, .outcome = &outcome, .capacity = quarter});
    expect(!outcome.message, outcome.message);
    usearch_dense_insert(
        &(usearch_dense_insert_t) {.index = indexes[1],
                                   .executor = executor,
                                   .outcome = &outcome,
                                   .count = quarter,
                                   .keys = {keys, sizeof(usearch_key_t)},
                                   .vectors = {data, dimensions * sizeof(float), dimensions, usearch_scalar_f32_k}});
    expect(!outcome.message, outcome.message);
    usearch_dense_compact(&(usearch_dense_compact_t) {.target = target,
                                                      .executor = executor,
                                                      .outcome = &outcome,
                                                      .count = 2,
                                                      .sources = indexes});
    expect(!outcome.message, outcome.message);
    expect(indexes[0] == NULL && indexes[1] == NULL, "Every consumed source must be freed and nulled");
    usearch_dense_index_description_t description = {0};
    usearch_dense_describe(&(usearch_dense_describe_t) {.index = target,
                                                        .executor = executor,
                                                        .outcome = &outcome,
                                                        .description = &description});
    expect(!outcome.message, outcome.message);
    expect_eq(description.size, rows_count + quarter, "Every source row must land in the target");

    free(keys);
    free(data);
    usearch_executor_free(executor);
    usearch_dense_free(target);
    printf("Test: Compact - PASSED\n");
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
                                                                      .donors = indexes + 3,
                                                                      .count = 1,
                                                                      .refinement = &refinement});
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
                                                                          .reselected = &pulled_visited,
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
    expect(reports.calls > 0 && reports.total >= rows_per_index,
           "The handler must hear the walks over every donor row");
    expect(reports.ascending, "Every pull must count on from where the last stopped");

    usearch_refine_report_t finished = {0};
    usearch_traversal_stats_t finished_stats = {0};
    usearch_dense_refinement_finish(&(usearch_dense_refinement_finish_t) {.index = indexes[0],
                                                                          .executor = executor,
                                                                          .outcome = &outcome,
                                                                          .refinement = refinement,
                                                                          .report = &finished,
                                                                          .stats = &finished_stats});
    expect(!outcome.message, outcome.message);
    expect(finished.changed >= changed && finished.changed > 0 &&
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
    expect(finished.reachability.audited == reachability.audited &&
               finished.reachability.reachable == reachability.reachable,
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
        usearch_sequence_insert(&(usearch_sequence_insert_t) {
            .index = indexes[which],
            .executor = executor,
            .outcome = &outcome,
            .count = reads_per_index,
            .keys = {keys + which * reads_per_index, sizeof(usearch_key_t)},
            .sequences = tape_of(bytes, offsets + which * reads_per_index, usearch_scalar_u8_k),
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
                                                                            .donors = indexes + 3,
                                                                            .count = 1,
                                                                            .refinement = &refinement});
    expect(!outcome.message, outcome.message);
    expect(refinement && indexes[3] != NULL, "A donor lands at the first pull, not at the opening");
    usearch_key_t const late_key = key_of(reads_count);
    usearch_sequence_insert(&(usearch_sequence_insert_t) {.index = indexes[0],
                                                          .executor = executor,
                                                          .outcome = &outcome,
                                                          .count = 1,
                                                          .keys = {&late_key, sizeof(usearch_key_t)},
                                                          .sequences = tape_of(bytes, offsets, usearch_scalar_u8_k)});
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
                                                   .reselected = &pulled_visited,
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
    expect(reports.calls > 0 && reports.total >= reads_per_index,
           "The handler must hear the walks over every donor read");
    expect(reports.ascending, "Every pull must count on from where the last stopped");

    usearch_refine_report_t finished = {0};
    usearch_sequence_refinement_finish(&(usearch_sequence_refinement_finish_t) {.index = indexes[0],
                                                                                .executor = executor,
                                                                                .outcome = &outcome,
                                                                                .refinement = refinement,
                                                                                .report = &finished});
    expect(!outcome.message, outcome.message);
    expect(finished.changed >= changed && finished.changed > 0, "The refinement must report what its pulls changed");
    expect_eq(finished.reachability.audited, reads_count, "The closing audit must judge every read");

    // The index reopened: a one-shot refine runs to its end and leaves every read reachable.
    usearch_refine_report_t refined = {0};
    usearch_sequence_refine(&(usearch_sequence_refine_t) {.index = indexes[0],
                                                          .executor = executor,
                                                          .outcome = &outcome,
                                                          .report = &refined});
    expect(!outcome.message, outcome.message);
    expect(refined.reachability.audited == reads_count && refined.reachability.reachable == reads_count,
           "Every read must stay reachable");
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

/** Reads entries back by key. @c usearch_dense_get sizes its own output, writing every row under
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

    usearch_arena_description_t description = {usearch_residency_unknown_k, 1};
    usearch_arena_describe(
        &(usearch_arena_describe_t) {.arena = local, .outcome = &outcome, .description = &description});
    expect(!outcome.message, outcome.message);
    expect(description.residency == usearch_residency_ram_k && description.device == 0,
           "An arena made without an executor is host pages");
    usearch_arena_describe(
        &(usearch_arena_describe_t) {.arena = NULL, .outcome = &outcome, .description = &description});
    expect_eq(outcome.status, usearch_status_invalid_argument_k, "Describing a NULL arena is refused");

    size_t devices_count = 0;
    usearch_gpu_devices(&(usearch_gpu_devices_t) {.outcome = &outcome, .devices_count = &devices_count});
    int32_t const devices[1] = {0};
    usearch_executor_t device_executor = NULL;
    if (devices_count)
        usearch_executor_for_vram(&(usearch_executor_for_vram_t) {.executor = &device_executor,
                                                                  .outcome = &outcome,
                                                                  .vram = {.devices = devices, .devices_count = 1}});
    if (device_executor) {
        usearch_arena_t const device_arena = create_arena(device_executor);
        usearch_arena_describe(
            &(usearch_arena_describe_t) {.arena = device_arena, .outcome = &outcome, .description = &description});
        expect(!outcome.message, outcome.message);
        expect(description.residency == usearch_residency_vram_k && description.device == 0,
               "An arena made by a device executor lives on its first device");
        usearch_arena_free(device_arena);
        usearch_executor_free(device_executor);
    }

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
    usearch_dense_cursor_init(&(usearch_dense_cursor_init_t) {.index = index, .outcome = &outcome, .cursor = &cursor});
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

enum { neighborhood_rows_k = 200, neighborhood_dimensions_k = 16, sampled_wanted_k = 48 };

/** The integer identity a key wraps, as @c key_of wrapped it. */
uint64_t value_of(usearch_key_t key) {
    uint64_t value = 0;
    for (unsigned byte_index = 0; byte_index != 8; ++byte_index)
        value |= (uint64_t)key.bytes[15 - byte_index] << (byte_index * 8);
    return value;
}

/** A Euclidean index over @p rows, each under the key of its position but row 1, which shares row
 *  0's key, so one key holds two entries; refined, since a concurrent build reaches every row from
 *  row 0 only after a refine, and the cursor tests step the graph rather than judge the build. */
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
    usearch_dense_refine(&(usearch_dense_refine_t) {.index = index, .executor = executor, .outcome = &outcome});
    expect(!outcome.message, outcome.message);
    return index;
}

/** One sample's draws, copied out of the arena every call rewinds. */
typedef struct sampled_t {
    size_t written;
    usearch_key_t keys[neighborhood_rows_k];
    size_t ordinals[neighborhood_rows_k];
    size_t parented[neighborhood_rows_k];
    usearch_key_t parents[neighborhood_rows_k];
    float coordinates[neighborhood_rows_k * 3];
} sampled_t;

/** Draws up to @p wanted entries under @p seed around the @p focus_count keys at @p focus, the
 *  draws of @p shown, when given, standing as shown keys at their coordinates, and copies them into
 *  @p sampled; answers the outcome, which an implementation that cannot sample fails. */
usearch_outcome_t sample_draws(usearch_dense_index_t index, usearch_executor_t executor, usearch_arena_t arena,
                               sampled_t const* shown, usearch_key_t const* focus, size_t focus_count, size_t wanted,
                               uint64_t seed, sampled_t* sampled) {
    usearch_outcome_t outcome = {0};
    usearch_keys_view_t keys = {0};
    size_t const* ordinals = NULL;
    usearch_matches_view_t parents = {0};
    usearch_rows_view_t coordinates = {0};
    size_t written = 0;
    usearch_dense_sample(&(usearch_dense_sample_t) {
        .index = index,
        .executor = executor,
        .outcome = &outcome,
        .shown_count = shown ? shown->written : 0,
        .shown = {shown ? shown->keys : NULL, sizeof(usearch_key_t)},
        .shown_coordinates = {shown ? shown->coordinates : NULL, 3 * sizeof(float), 3, usearch_scalar_f32_k},
        .focus_count = focus_count,
        .focus = {focus, sizeof(usearch_key_t)},
        .keys = &keys,
        .entry_ordinals = &ordinals,
        .parents = &parents,
        .coordinates = &coordinates,
        .written = &written,
        .arena = arena,
        .wanted = wanted,
        .seed = seed});
    if (outcome.status != usearch_status_success_k) return outcome;
    expect(written <= wanted && written <= neighborhood_rows_k, "A sample writes at most the draws wanted");
    expect(coordinates.columns == 3 && coordinates.scalar == usearch_scalar_f32_k,
           "Draws are placed in three f32 dimensions");
    sampled->written = written;
    for (size_t draw = 0; draw != written; ++draw) {
        sampled->keys[draw] = keys.data[draw];
        sampled->ordinals[draw] = ordinals[draw];
        sampled->parented[draw] = parents.counts[draw];
        if (parents.counts[draw]) sampled->parents[draw] = parents.keys[draw * parents.columns];
        memcpy(sampled->coordinates + draw * 3, (char const*)coordinates.data + draw * coordinates.stride,
               3 * sizeof(float));
    }
    return outcome;
}

/** Whether two samples drew the same entries from the same parents at the same coordinates, bit for
 *  bit. */
bool same_samples(sampled_t const* first, sampled_t const* second) {
    size_t const written = first->written;
    return written == second->written && !memcmp(first->keys, second->keys, written * sizeof(usearch_key_t)) &&
           !memcmp(first->ordinals, second->ordinals, written * sizeof(size_t)) &&
           !memcmp(first->parented, second->parented, written * sizeof(size_t)) &&
           !memcmp(first->parents, second->parents, written * sizeof(usearch_key_t)) &&
           !memcmp(first->coordinates, second->coordinates, written * 3 * sizeof(float));
}

/** Whether @p sampled drew the entry @p ordinal of @p key among its first @p count draws. */
bool drew(sampled_t const* sampled, size_t count, usearch_key_t key, size_t ordinal) {
    for (size_t draw = 0; draw != count; ++draw)
        if (!memcmp(sampled->keys[draw].bytes, key.bytes, sizeof(key.bytes)) && sampled->ordinals[draw] == ordinal)
            return true;
    return false;
}

/** Whether @p sampled drew @p key at all among its first @p count draws. */
bool drew_key(sampled_t const* sampled, size_t count, usearch_key_t key) {
    for (size_t draw = 0; draw != count; ++draw)
        if (!memcmp(sampled->keys[draw].bytes, key.bytes, sizeof(key.bytes))) return true;
    return false;
}

/** How many entries @p key holds, as @c usearch_dense_get lands them, copying the one at
 *  @p ordinal into @p row when it exists. */
size_t get_entry(usearch_dense_index_t index, usearch_executor_t executor, usearch_arena_t arena, usearch_key_t key,
                 size_t ordinal, float* row) {
    usearch_outcome_t outcome = {0};
    usearch_rows_view_t vectors = {0};
    size_t written = 0;
    usearch_dense_get(&(usearch_dense_get_t) {.index = index,
                                              .executor = executor,
                                              .outcome = &outcome,
                                              .key = key,
                                              .vectors = &vectors,
                                              .written = &written,
                                              .arena = arena,
                                              .outputs_scalar = usearch_scalar_f32_k});
    expect(!outcome.message, outcome.message);
    if (ordinal < written)
        memcpy(row, (char const*)vectors.data + ordinal * vectors.stride, neighborhood_dimensions_k * sizeof(float));
    return written;
}

/** Every draw of @p sampled names an entry of its key, as @c get orders them, once; never a key of
 *  @p shown; hangs from a key of @p shown or an earlier draw; and lies at finite coordinates. */
void expect_draws(usearch_dense_index_t index, usearch_executor_t executor, usearch_arena_t arena,
                  sampled_t const* sampled, sampled_t const* shown) {
    float row[neighborhood_dimensions_k];
    size_t const shown_count = shown ? shown->written : 0;
    for (size_t draw = 0; draw != sampled->written; ++draw) {
        usearch_key_t const key = sampled->keys[draw];
        expect(sampled->ordinals[draw] < get_entry(index, executor, arena, key, 0, row),
               "A draw names an entry of its key, as get counts them");
        expect(!drew(sampled, draw, key, sampled->ordinals[draw]), "An entry is drawn once");
        expect(!shown || !drew_key(shown, shown_count, key), "A shown key is never drawn again");
        expect(!sampled->parented[draw] || drew_key(sampled, draw, sampled->parents[draw]) ||
                   (shown && drew_key(shown, shown_count, sampled->parents[draw])),
               "A draw hangs from a shown or an earlier draw");
        for (size_t axis = 0; axis != 3; ++axis)
            expect(isfinite(sampled->coordinates[draw * 3 + axis]), "A draw lies at finite coordinates");
    }
}

/** Draw @p counter of the SplitMix64 stream @p seed starts. */
uint64_t splitmix64_at(uint64_t seed, uint64_t counter) {
    uint64_t word = seed + (counter + 1) * 0x9E3779B97F4A7C15ull;
    word = (word ^ (word >> 30)) * 0xBF58476D1CE4E5B9ull;
    word = (word ^ (word >> 27)) * 0x94D049BB133111EBull;
    return word ^ (word >> 31);
}

/** Where a sample with nothing shown places @p row under @p seed: its product with the Gaussian
 *  matrix whose entry for dimension d and axis c is the Box-Muller cosine of draws 2n and 2n+1,
 *  with n = 3d+c, each mapped into (0, 1], over the square root of the dimensions, accumulated in
 *  f64 and rounded once. */
void project(float const* row, uint64_t seed, float* coordinates) {
    for (size_t axis = 0; axis != 3; ++axis) {
        double sum = 0;
        for (size_t dimension = 0; dimension != neighborhood_dimensions_k; ++dimension) {
            uint64_t const counter = 2 * (3 * dimension + axis);
            double const radial = ((double)(splitmix64_at(seed, counter) >> 11) + 1) * 0x1p-53;
            double const angular = ((double)(splitmix64_at(seed, counter + 1) >> 11) + 1) * 0x1p-53;
            sum += (double)row[dimension] * sqrt(-2 * log(radial)) * cos(6.283185307179586 * angular);
        }
        coordinates[axis] = (float)(sum / sqrt((double)neighborhood_dimensions_k));
    }
}

/** A sample draws entries and places them: an equal seed repeats it bit for bit; a smaller one
 *  draws a subset, so zoom levels nest; every draw names an entry as @c get orders its key's, once,
 *  hangs from a shown or earlier draw, and lies at finite coordinates; shown keys are never drawn
 *  again and keep their coordinates. A golden cell over small integer rows, focused on every key,
 *  draws every entry where the projection places it, bit for bit. An implementation that cannot
 *  sample says so, and the test skips. */
void test_sample(void) {
    printf("Test: Sample\n");
    usearch_executor_t const executor = create_executor(0);
    usearch_arena_t const arena = create_arena(executor);
    float* const rows = create_vectors(neighborhood_rows_k, neighborhood_dimensions_k);
    // Small integers survive bf16 exactly, so the golden projection reads the rows as inserted.
    for (size_t value = 0; value != neighborhood_rows_k * neighborhood_dimensions_k; ++value)
        rows[value] = floorf(rows[value] * 8);
    usearch_dense_index_t const index = create_neighborhood(rows, executor);
    sampled_t* const samples = (sampled_t*)calloc(5, sizeof(sampled_t));
    sampled_t *const whole = samples, *const repeated = samples + 1, *const nested = samples + 2;
    sampled_t *const around = samples + 3, *const golden = samples + 4;

    usearch_outcome_t outcome = sample_draws(index, executor, arena, NULL, NULL, 0, sampled_wanted_k, 42, whole);
    if (outcome.status == usearch_status_unsupported_k) {
        printf("Test: Sample - SKIPPED: %s\n", outcome.message);
        free(samples);
        usearch_dense_free(index);
        free(rows);
        usearch_arena_free(arena);
        usearch_executor_free(executor);
        return;
    }
    expect(!outcome.message, outcome.message);
    expect_eq(whole->written, sampled_wanted_k, "A sample of the whole index draws all it wants");
    expect_draws(index, executor, arena, whole, NULL);
    outcome = sample_draws(index, executor, arena, NULL, NULL, 0, sampled_wanted_k, 42, repeated);
    expect(!outcome.message, outcome.message);
    expect(same_samples(whole, repeated), "An equal seed, shown keys and focus must draw alike, bit for bit");
    outcome = sample_draws(index, executor, arena, NULL, NULL, 0, sampled_wanted_k / 3, 42, nested);
    expect(!outcome.message, outcome.message);
    for (size_t draw = 0; draw != nested->written; ++draw)
        expect(drew(whole, whole->written, nested->keys[draw], nested->ordinals[draw]),
               "A smaller sample must draw a subset of a larger one, so zoom levels nest");

    // Zooming in shows the nested draws and looks around one key.
    float shown_coordinates[sampled_wanted_k * 3];
    memcpy(shown_coordinates, nested->coordinates, sizeof(shown_coordinates));
    usearch_key_t const focus = key_of(5);
    outcome = sample_draws(index, executor, arena, nested, &focus, 1, sampled_wanted_k, 42, around);
    expect(!outcome.message, outcome.message);
    expect(around->written > 0, "A sample around a key draws");
    expect_draws(index, executor, arena, around, nested);
    expect(!memcmp(shown_coordinates, nested->coordinates, sizeof(shown_coordinates)), "Shown points keep theirs");

    // Every key in focus and nothing shown: every entry is drawn, placed by the projection.
    usearch_key_t every_key[neighborhood_rows_k - 1];
    for (size_t key = 0; key != neighborhood_rows_k - 1; ++key) every_key[key] = key_of(key ? key + 1 : 0);
    outcome = sample_draws(index, executor, arena, NULL, every_key, neighborhood_rows_k - 1, neighborhood_rows_k, 7,
                           golden);
    expect(!outcome.message, outcome.message);
    expect_eq(golden->written, neighborhood_rows_k, "A focus on every key draws every entry");
    expect_draws(index, executor, arena, golden, NULL);
    for (size_t draw = 0; draw != golden->written; ++draw) {
        float row[neighborhood_dimensions_k], projected[3];
        get_entry(index, executor, arena, golden->keys[draw], golden->ordinals[draw], row);
        project(row, 7, projected);
        expect(!memcmp(projected, golden->coordinates + draw * 3, sizeof(projected)),
               "A draw with nothing shown lies where the projection of its row places it, bit for bit");
    }

    free(samples);
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
 *  each, and an absent key starts nothing, while a plain scan still yields every entry once. An
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
    usearch_dense_cursor_init(&(usearch_dense_cursor_init_t) {.index = index, .outcome = &outcome, .cursor = &cursor});
    expect(!outcome.message, outcome.message);
    expect_eq(pull_to_end(index, executor, arena, cursor, yielded), neighborhood_rows_k,
              "A plain scan must yield every row");
    for (size_t row = 0; row != neighborhood_rows_k; ++row)
        expect(yielded[row] == (row == 0 ? 2u : row == 1 ? 0u : 1u), "A plain scan must yield every row once");

    usearch_key_t const start = key_of(0);
    usearch_dense_cursor_init(&(usearch_dense_cursor_init_t) {.index = index,
                                                              .outcome = &outcome,
                                                              .starts_count = 1,
                                                              .starts = &start,
                                                              .cursor = &cursor});
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
        usearch_dense_cursor_init(&(usearch_dense_cursor_init_t) {.index = index,
                                                                  .outcome = &outcome,
                                                                  .starts_count = 1,
                                                                  .starts = &absent,
                                                                  .cursor = &cursor});
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
 *  neighbor, and @c get hands the bytes back, before the save and after the load alike. The overlap
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
                                                          .sequences = tape_of(bytes, offsets, usearch_scalar_u8_k)});
    expect(!outcome.message, outcome.message);
    size_t const falling_offsets[] = {0, 6, 3};
    usearch_sequence_insert(
        &(usearch_sequence_insert_t) {.index = index,
                                      .executor = executor,
                                      .outcome = &outcome,
                                      .count = 2,
                                      .keys = {keys, sizeof(usearch_key_t)},
                                      .sequences = tape_of(bytes, falling_offsets, usearch_scalar_u8_k)});
    expect_eq(outcome.status, usearch_status_invalid_argument_k, "Falling offsets are refused before any insert");

    char const path[] = "usearch-test-c-sequence.safegraphs";
    usearch_bytes_view_t const path_view = {(uint8_t const*)path, sizeof(path) - 1};
    for (int reloaded = 0; reloaded != 2; ++reloaded) {
        usearch_matches_view_t matches = {0};
        usearch_sequence_search(&(usearch_sequence_search_t) {.index = index,
                                                              .executor = executor,
                                                              .outcome = &outcome,
                                                              .count = 1,
                                                              .queries = tape_of(bytes, offsets, usearch_scalar_u8_k),
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
        expect(written == 1 && offsets_of(got)[1] - offsets_of(got)[0] == 7 &&
                   memcmp((uint8_t const*)got.data + offsets_of(got)[0], "sitting", 7) == 0,
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
    expect(description.load_path == usearch_load_path_none_k, "A load into RAM names no device load path");

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
                                                              .queries = tape_of(bytes, offsets, usearch_scalar_u8_k),
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
        expect(viewed_written == 1 && offsets_of(viewed_got)[1] - offsets_of(viewed_got)[0] == 7 &&
                   memcmp((uint8_t const*)viewed_got.data + offsets_of(viewed_got)[0], "sitting", 7) == 0,
               "A view from storage hands the stored bytes back");
        usearch_sequence_refine(
            &(usearch_sequence_refine_t) {.index = viewed, .executor = storage, .outcome = &outcome});
        expect_eq(outcome.status, usearch_status_read_only_k, "A view from storage refines nothing");
        usearch_sequence_cursor_t cursor = NULL;
        usearch_sequence_cursor_init(
            &(usearch_sequence_cursor_init_t) {.index = viewed, .outcome = &outcome, .cursor = &cursor});
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
        usearch_sequence_insert(
            &(usearch_sequence_insert_t) {.index = viewed,
                                          .executor = storage,
                                          .outcome = &outcome,
                                          .count = 1,
                                          .keys = {keys, sizeof(usearch_key_t)},
                                          .sequences = tape_of(bytes, offsets, usearch_scalar_u8_k)});
        expect_eq(outcome.status, usearch_status_read_only_k, "A view from storage refuses every mutation");
        usearch_sequence_save(
            &(usearch_sequence_save_t) {.index = viewed, .executor = storage, .outcome = &outcome, .path = path_view});
        expect_eq(outcome.status, usearch_status_unsupported_k, "A view from storage already is its checkpoint");
        usearch_reachability_t reachability = {0};
        usearch_sequence_health(&(usearch_sequence_health_t) {.index = viewed,
                                                              .executor = storage,
                                                              .outcome = &outcome,
                                                              .reachability = &reachability});
        expect(!outcome.message, outcome.message);
        expect(reachability.audited == 4 && reachability.reachable == 4,
               "A view from storage audits and reaches every entry");
        usearch_key_t const start = key_of(0);
        usearch_sequence_cursor_t walk = NULL;
        usearch_sequence_cursor_init(&(usearch_sequence_cursor_init_t) {.index = viewed,
                                                                        .outcome = &outcome,
                                                                        .starts_count = 1,
                                                                        .starts = &start,
                                                                        .cursor = &walk});
        expect(!outcome.message, outcome.message);
        usearch_keys_view_t walked = {0};
        usearch_keys_tape_view_t links = {0};
        size_t walked_count = 0;
        usearch_sequence_cursor_next(&(usearch_sequence_cursor_next_t) {.index = viewed,
                                                                        .executor = storage,
                                                                        .outcome = &outcome,
                                                                        .cursor = walk,
                                                                        .keys = &walked,
                                                                        .links = &links,
                                                                        .written = &walked_count,
                                                                        .arena = arena,
                                                                        .wanted = 4});
        expect(!outcome.message, outcome.message);
        expect(walked_count > 1 && value_of(walked.data[0]) == 0 && links.offsets[walked_count] > links.offsets[0],
               "A cursor over storage starts at its start key and hands every entry's links");
        usearch_sequence_cursor_free(walk);
        usearch_keys_view_t sampled_keys = {0};
        size_t const* sampled_ordinals = NULL;
        usearch_matches_view_t sampled_parents = {0};
        usearch_rows_view_t sampled_coordinates = {0};
        size_t sampled = 0;
        usearch_sequence_sample(&(usearch_sequence_sample_t) {.index = viewed,
                                                              .executor = storage,
                                                              .outcome = &outcome,
                                                              .keys = &sampled_keys,
                                                              .entry_ordinals = &sampled_ordinals,
                                                              .parents = &sampled_parents,
                                                              .coordinates = &sampled_coordinates,
                                                              .written = &sampled,
                                                              .arena = arena,
                                                              .wanted = 4,
                                                              .seed = 42});
        expect(!outcome.message, outcome.message);
        expect_eq(sampled, 4, "A view from storage samples every entry it holds");
        usearch_bytes_view_t monolith = {0};
        usearch_sequence_save_buffer(&(usearch_sequence_save_buffer_t) {.index = viewed,
                                                                        .executor = storage,
                                                                        .outcome = &outcome,
                                                                        .monolith = &monolith,
                                                                        .arena = arena});
        expect_eq(outcome.status, usearch_status_unsupported_k, "A view from storage saves into no buffer");
        expect(outcome.message != NULL, "A view's refusal to save says why");
        usearch_sequence_free(viewed);
    }
    usearch_executor_free(storage);
    remove(path);

    usearch_sequence_free(index);
    usearch_arena_free(arena);
    usearch_executor_free(executor);
    printf("Test: Sequence Round Trip - PASSED\n");
}

/** A sequence RAM index bounded at ten entries a shard takes twenty-five reads in one batch,
 *  opening three generations of shards on its one default domain: every key counts once, every read
 *  finds itself first across the shards, a cursor pulls every key once, and a checkpoint writes a
 *  manifest beside one part per shard that reloads as many shards. */
void test_sequence_shards(void) {
    printf("Test: Sequence Shards\n");
    enum { reads_count = 25, shard_capacity = 10 };
    usearch_outcome_t outcome = {0};
    usearch_executor_t executor = NULL;
    usearch_executor_for_ram(&(usearch_executor_for_ram_t) {.executor = &executor, .outcome = &outcome, .threads = 2});
    expect(!outcome.message, outcome.message);
    usearch_arena_t const arena = create_arena(executor);
    size_t offsets[reads_count + 1];
    char* const bytes = create_reads(reads_count, offsets);
    usearch_key_t keys[reads_count];
    for (size_t read = 0; read != reads_count; ++read) keys[read] = key_of(read);

    usearch_sequence_index_t index = NULL;
    usearch_sequence_init_ram(&(usearch_sequence_init_ram_t) {.index = &index,
                                                              .outcome = &outcome,
                                                              .metric = usearch_sequence_metric_levenshtein_bytes_k,
                                                              .ram = {.shard_capacity = shard_capacity}});
    expect(!outcome.message, outcome.message);
    usearch_sequence_insert(&(usearch_sequence_insert_t) {.index = index,
                                                          .executor = executor,
                                                          .outcome = &outcome,
                                                          .count = reads_count,
                                                          .keys = {keys, sizeof(usearch_key_t)},
                                                          .sequences = tape_of(bytes, offsets, usearch_scalar_u8_k)});
    expect(!outcome.message, outcome.message);
    usearch_sequence_index_description_t description = {0};
    usearch_sequence_describe(&(usearch_sequence_describe_t) {.index = index,
                                                              .executor = executor,
                                                              .outcome = &outcome,
                                                              .description = &description});
    expect(!outcome.message, outcome.message);
    expect_eq(description.size, reads_count, "Every shard's rows count once");

    for (size_t read = 0; read != reads_count; ++read) {
        usearch_sequence_key_stats_t stats = {0};
        usearch_sequence_count(&(usearch_sequence_count_t) {.index = index,
                                                            .executor = executor,
                                                            .outcome = &outcome,
                                                            .key = keys[read],
                                                            .key_stats = &stats});
        expect(!outcome.message, outcome.message);
        expect_eq(stats.entries, 1, "Every key resolves to one read across the shards");
    }

    usearch_matches_view_t matches = {0};
    usearch_sequence_search(&(usearch_sequence_search_t) {.index = index,
                                                          .executor = executor,
                                                          .outcome = &outcome,
                                                          .count = reads_count,
                                                          .queries = tape_of(bytes, offsets, usearch_scalar_u8_k),
                                                          .matches = &matches,
                                                          .arena = arena,
                                                          .wanted = 1});
    expect(!outcome.message, outcome.message);
    for (size_t read = 0; read != reads_count; ++read)
        expect(matches.counts[read] == 1 && value_of(matches.keys[read]) == read && matches.scores[read] == 0,
               "Every read finds itself first, whichever shard holds it");

    usearch_sequence_cursor_t cursor = NULL;
    usearch_sequence_cursor_init(
        &(usearch_sequence_cursor_init_t) {.index = index, .outcome = &outcome, .cursor = &cursor});
    expect(!outcome.message, outcome.message);
    bool pulled[reads_count] = {false};
    size_t pulled_count = 0;
    for (size_t written = 1; written;) {
        usearch_keys_view_t pulled_keys = {0};
        usearch_sequence_cursor_next(&(usearch_sequence_cursor_next_t) {.index = index,
                                                                        .executor = executor,
                                                                        .outcome = &outcome,
                                                                        .cursor = cursor,
                                                                        .keys = &pulled_keys,
                                                                        .written = &written,
                                                                        .arena = arena,
                                                                        .wanted = 7});
        expect(!outcome.message, outcome.message);
        for (size_t entry = 0; entry != written; ++entry) {
            uint64_t const value = value_of(pulled_keys.data[entry]);
            expect(value < reads_count && !pulled[value], "A cursor pulls every key once, shard after shard");
            pulled[value] = true;
            ++pulled_count;
        }
    }
    expect_eq(pulled_count, reads_count, "A cursor pulls every shard's keys");
    usearch_sequence_cursor_free(cursor);

    char const path[] = "usearch-test-c-sequence-shards.safegraphs";
    usearch_bytes_view_t const path_view = {(uint8_t const*)path, sizeof(path) - 1};
    usearch_sequence_save(
        &(usearch_sequence_save_t) {.index = index, .executor = executor, .outcome = &outcome, .path = path_view});
    expect(!outcome.message, outcome.message);
    usearch_sequence_free(index);
    for (int part = 1; part <= 3; ++part) {
        char part_path[64];
        snprintf(part_path, sizeof(part_path), "usearch-test-c-sequence-shards-%05d-of-00003.safegraphs", part);
        FILE* const part_file = fopen(part_path, "rb");
        expect(part_file != NULL, "A save of three shards writes one part per shard beside its manifest");
        if (part_file) fclose(part_file);
    }

    index = NULL;
    usearch_sequence_load_ram(
        &(usearch_sequence_load_ram_t) {.index = &index, .executor = executor, .outcome = &outcome, .path = path_view});
    expect(!outcome.message, outcome.message);
    usearch_sequence_index_description_t reloaded = {0};
    usearch_sequence_describe(&(usearch_sequence_describe_t) {.index = index,
                                                              .executor = executor,
                                                              .outcome = &outcome,
                                                              .description = &reloaded});
    expect(!outcome.message, outcome.message);
    expect_eq(reloaded.size, reads_count, "The manifest reloads every part's rows");
    expect_eq(reloaded.shards_count, 3, "A load opens one shard per part");
    usearch_sequence_search(&(usearch_sequence_search_t) {.index = index,
                                                          .executor = executor,
                                                          .outcome = &outcome,
                                                          .count = reads_count,
                                                          .queries = tape_of(bytes, offsets, usearch_scalar_u8_k),
                                                          .matches = &matches,
                                                          .arena = arena,
                                                          .wanted = 1});
    expect(!outcome.message, outcome.message);
    for (size_t read = 0; read != reads_count; ++read)
        expect(matches.counts[read] == 1 && value_of(matches.keys[read]) == read && matches.scores[read] == 0,
               "Every read finds itself first in the reloaded parts");

    usearch_sequence_free(index);
    remove(path);
    for (int part = 1; part <= 3; ++part) {
        char part_path[64];
        snprintf(part_path, sizeof(part_path), "usearch-test-c-sequence-shards-%05d-of-00003.safegraphs", part);
        remove(part_path);
    }
    free(bytes);
    usearch_arena_free(arena);
    usearch_executor_free(executor);
    printf("Test: Sequence Shards - PASSED\n");
}

/** A GPU sequence index takes byte and rune Levenshtein, inserts and describes its entries, and
 *  lands a search's matches in its executor's device arena; local alignments, a global one without
 *  weights and the verbs it lacks refuse. An overlap index lands its scores in a host arena on the
 *  scale a RAM one does, and a Needleman-Wunsch one its exact costs over packed codes. Skipped
 *  without a GPU. */
void test_sequence_vram(void) {
    usearch_outcome_t outcome = {0};
    size_t devices_count = 0;
    usearch_gpu_devices(&(usearch_gpu_devices_t) {.outcome = &outcome, .devices_count = &devices_count});
    if (!devices_count) return;
    printf("Test: Sequence VRAM\n");
    int32_t const devices[1] = {0};
    usearch_vram_options_t const vram = {.devices = devices, .devices_count = 1};

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

    usearch_sequence_index_t refused = NULL;
    usearch_sequence_init_vram(&(usearch_sequence_init_vram_t) {.index = &refused,
                                                                .outcome = &outcome,
                                                                .metric = usearch_sequence_metric_needleman_wunsch_k,
                                                                .vram = vram});
    expect_eq(outcome.status, usearch_status_config_mismatch_k, "An alignment needs weights");
    usearch_sequence_init_vram(&(usearch_sequence_init_vram_t) {.index = &refused,
                                                                .outcome = &outcome,
                                                                .metric = usearch_sequence_metric_levenshtein_bytes_k,
                                                                .values_per_key = 2,
                                                                .vram = vram});
    expect_eq(outcome.status, usearch_status_unsupported_k, "A GPU sequence index bounds no key's entries");

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
                                                          .sequences = tape_of(bytes, offsets, usearch_scalar_u8_k)});
    expect(!outcome.message, outcome.message);
    usearch_sequence_refine(&(usearch_sequence_refine_t) {.index = index, .executor = executor, .outcome = &outcome});
    expect(!outcome.message, outcome.message);
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
                                                          .queries = tape_of(bytes, offsets, usearch_scalar_u8_k),
                                                          .matches = &matches,
                                                          .arena = arena,
                                                          .wanted = 2});
    expect(!outcome.message, outcome.message);
    expect(matches.keys && matches.scores && matches.counts, "A GPU search lands its matches in the arena");

    // "mitten" holds three of the four windows "kitten" has at three bytes.
    size_t const widths[] = {3};
    usearch_sequence_index_t overlap = NULL;
    usearch_sequence_init_vram(&(usearch_sequence_init_vram_t) {.index = &overlap,
                                                                .outcome = &outcome,
                                                                .metric = usearch_sequence_metric_overlap_k,
                                                                .window_widths = widths,
                                                                .window_widths_count = 1,
                                                                .vram = vram});
    expect(!outcome.message, outcome.message);
    usearch_sequence_insert(&(usearch_sequence_insert_t) {.index = overlap,
                                                          .executor = executor,
                                                          .outcome = &outcome,
                                                          .count = 4,
                                                          .keys = {keys, sizeof(usearch_key_t)},
                                                          .sequences = tape_of(bytes, offsets, usearch_scalar_u8_k)});
    expect(!outcome.message, outcome.message);
    usearch_arena_t const host_arena = create_arena(NULL);
    usearch_matches_view_t overlaps = {0};
    usearch_sequence_search(&(usearch_sequence_search_t) {.index = overlap,
                                                          .executor = executor,
                                                          .outcome = &outcome,
                                                          .count = 1,
                                                          .queries = tape_of(bytes, offsets, usearch_scalar_u8_k),
                                                          .matches = &overlaps,
                                                          .arena = host_arena,
                                                          .wanted = 2});
    expect(!outcome.message, outcome.message);
    expect(overlaps.counts[0] == 2 && value_of(overlaps.keys[0]) == 0 && overlaps.scores[0] == 0,
           "A stored overlap query finds itself first");
    expect(value_of(overlaps.keys[1]) == 2 && overlaps.scores[1] == 0.25, "Missing a quarter of windows scores 0.25");
    usearch_sequence_free(overlap);

    // "ACGT", "ACGA", "ACG" and "TTTT" packed four nucleotides a byte behind their unused slots,
    // scored with transitions at 1, transversions and gaps at 2.
    static int8_t similarities[256 * 256];
    for (size_t first = 0; first != 4; ++first)
        for (size_t second = 0; second != 4; ++second)
            similarities[first * 256 + second] = first == second ? 2 : (first & 1) == (second & 1) ? 1 : 0;
    usearch_sequence_weights_t const weights = {.substitution_matrix = similarities, .gap = 2};
    unsigned char const packed[] = {0x00, 0xE4, 0x00, 0x24, 0x01, 0x24, 0x00, 0xFF};
    size_t const packed_offsets[] = {0, 2, 4, 6, 8};
    usearch_sequence_index_t aligned = NULL;
    usearch_sequence_init_vram(&(usearch_sequence_init_vram_t) {.index = &aligned,
                                                                .outcome = &outcome,
                                                                .metric = usearch_sequence_metric_needleman_wunsch_k,
                                                                .weights = &weights,
                                                                .vram = vram});
    expect(!outcome.message, outcome.message);
    usearch_sequence_insert(
        &(usearch_sequence_insert_t) {.index = aligned,
                                      .executor = executor,
                                      .outcome = &outcome,
                                      .count = 4,
                                      .keys = {keys, sizeof(usearch_key_t)},
                                      .sequences = tape_of(packed, packed_offsets, usearch_scalar_u8_k)});
    expect(!outcome.message, outcome.message);
    usearch_matches_view_t alignments = {0};
    usearch_sequence_search(
        &(usearch_sequence_search_t) {.index = aligned,
                                      .executor = executor,
                                      .outcome = &outcome,
                                      .count = 1,
                                      .queries = tape_of(packed, packed_offsets, usearch_scalar_u8_k),
                                      .matches = &alignments,
                                      .arena = host_arena,
                                      .wanted = 4});
    expect(!outcome.message, outcome.message);
    expect(alignments.counts[0] == 4 && value_of(alignments.keys[0]) == 0 && alignments.scores[0] == 0,
           "A stored alignment query finds itself first");
    expect(alignments.scores[1] == 2 && alignments.scores[2] == 2 && alignments.scores[3] == 5,
           "A transversion and a gap cost 2, and two transversions with a transition 5");
    usearch_arena_free(host_arena);
    usearch_sequence_free(aligned);

    char const path[] = "usearch-test-c-sequence-vram.safegraphs";
    usearch_sequence_save(&(usearch_sequence_save_t) {.index = index,
                                                      .executor = executor,
                                                      .outcome = &outcome,
                                                      .path = {(uint8_t const*)path, sizeof(path) - 1}});
    expect(!outcome.message, outcome.message);
    usearch_sequence_index_t loaded = NULL;
    usearch_sequence_load_ram(&(usearch_sequence_load_ram_t) {.index = &loaded,
                                                              .outcome = &outcome,
                                                              .path = {(uint8_t const*)path, sizeof(path) - 1}});
    expect(!outcome.message, outcome.message);
    usearch_sequence_index_description_t loaded_description = {0};
    usearch_sequence_describe(
        &(usearch_sequence_describe_t) {.index = loaded, .outcome = &outcome, .description = &loaded_description});
    expect(!outcome.message, outcome.message);
    expect(loaded_description.size == description.size, "A GPU checkpoint loads into RAM whole");
    usearch_sequence_free(loaded);
    remove(path);

    usearch_arena_free(arena);
    usearch_sequence_free(index);
    usearch_executor_free(executor);
    printf("Test: Sequence VRAM - PASSED\n");
}

/** Shapes round-trip the same way: a stored pose finds itself at zero, and @c get exports its
 *  points back, before the save and after the load alike. A frozen point count refuses a shape of
 *  any other. */
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
                                                      .shapes = tape_of(points, short_offsets, usearch_scalar_f32_k)});
    expect_eq(outcome.status, usearch_status_shape_mismatch_k, "A frozen point count refuses any other");
    size_t const falling_offsets[] = {0, 4, 2};
    usearch_shapes_insert(
        &(usearch_shapes_insert_t) {.index = index,
                                    .executor = executor,
                                    .outcome = &outcome,
                                    .count = 2,
                                    .keys = {keys, sizeof(usearch_key_t)},
                                    .shapes = tape_of(points, falling_offsets, usearch_scalar_f32_k)});
    expect_eq(outcome.status, usearch_status_invalid_argument_k, "Falling offsets are refused before any insert");
    usearch_shapes_insert(&(usearch_shapes_insert_t) {.index = index,
                                                      .executor = executor,
                                                      .outcome = &outcome,
                                                      .count = 3,
                                                      .keys = {keys, sizeof(usearch_key_t)},
                                                      .shapes = tape_of(points, offsets, usearch_scalar_f32_k)});
    expect(!outcome.message, outcome.message);

    char const path[] = "usearch-test-c-shapes.safegraphs";
    usearch_bytes_view_t const path_view = {(uint8_t const*)path, sizeof(path) - 1};
    for (int reloaded = 0; reloaded != 2; ++reloaded) {
        usearch_matches_view_t matches = {0};
        usearch_shapes_search(&(usearch_shapes_search_t) {.index = index,
                                                          .executor = executor,
                                                          .outcome = &outcome,
                                                          .count = 1,
                                                          .queries = tape_of(points, offsets, usearch_scalar_f32_k),
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
        expect(written == 1 && offsets_of(got)[1] - offsets_of(got)[0] == 4, "Get hands every point back");
        float const* const exported = (float const*)got.data + offsets_of(got)[0] * 3;
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
    expect(description.load_path == usearch_load_path_none_k, "A load into RAM names no device load path");

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
                                                          .queries = tape_of(points, offsets, usearch_scalar_f32_k),
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
        expect(viewed_written == 1 && offsets_of(viewed_got)[1] - offsets_of(viewed_got)[0] == 4,
               "A view from storage hands every point back");
        float const* const viewed_points = (float const*)viewed_got.data + offsets_of(viewed_got)[0] * 3;
        for (size_t coordinate = 0; coordinate != 12; ++coordinate)
            expect(fabsf(viewed_points[coordinate] - points[24 + coordinate]) < 1e-6f,
                   "A view from storage hands the stored pose back");
        usearch_shapes_cursor_t cursor = NULL;
        usearch_shapes_cursor_init(
            &(usearch_shapes_cursor_init_t) {.index = viewed, .outcome = &outcome, .cursor = &cursor});
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
                                                          .shapes = tape_of(points, offsets, usearch_scalar_f32_k)});
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

    usearch_shapes_index_t refused = NULL;
    usearch_shapes_init_vram(&(usearch_shapes_init_vram_t) {.index = &refused,
                                                            .outcome = &outcome,
                                                            .metric = usearch_shapes_metric_wasserstein_procrustes_k,
                                                            .quantization = usearch_scalar_f16_k,
                                                            .vram = vram});
    expect_eq(outcome.status, usearch_status_unsupported_k, "Only Chamfer and Hausdorff serve a GPU");
    expect(refused == NULL, "A refused initialization returns no handle");

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
                                                      .shapes = tape_of(points, offsets, usearch_scalar_f32_k)});
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
                                                      .queries = tape_of(points, offsets, usearch_scalar_f32_k),
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
    expect(!outcome.message, outcome.message);
    usearch_shapes_index_t loaded = NULL;
    usearch_shapes_load_ram(&(usearch_shapes_load_ram_t) {.index = &loaded,
                                                          .outcome = &outcome,
                                                          .path = {(uint8_t const*)path, sizeof(path) - 1}});
    expect(!outcome.message, outcome.message);
    usearch_shapes_index_description_t loaded_description = {0};
    usearch_shapes_describe(
        &(usearch_shapes_describe_t) {.index = loaded, .outcome = &outcome, .description = &loaded_description});
    expect(!outcome.message, outcome.message);
    expect(loaded_description.size == description.size && loaded_description.total_points == description.total_points,
           "A GPU checkpoint loads into RAM whole");
    usearch_shapes_free(loaded);
    remove(path);

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

/** Geospatial round-trips the same way: an observation finds itself at zero metres and @c get hands
 *  it back, before the save and after the load alike, while a ridge on the handle's terrain hides
 *  what stands behind it. */
void test_geospatial_round_trip(void) {
    printf("Test: Geospatial Round Trip\n");
    usearch_outcome_t outcome = {0};
    usearch_executor_t const executor = create_executor(0);
    usearch_arena_t const arena = create_arena(executor);

    // Six positions about 56 m apart over flat ground, with a 100 m ridge along the third; every
    // entry stands inside the grid, since one on its edge may round off it, and off the grid
    // nothing blocks a sight line.
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
    expect(description.load_path == usearch_load_path_none_k, "A load into RAM names no device load path");

    // The same checkpoint served from storage judges every relation on the entries its walks read
    // and answers reads and containment as RAM does, refusing writes and saves.
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
            &(usearch_geospatial_cursor_init_t) {.index = viewed, .outcome = &outcome, .cursor = &cursor});
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
            .shapes = tape_of(points + which * entries_per_index * points_per_shape * 3, offsets, usearch_scalar_f32_k),
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
    usearch_refine_report_t report = {0};
    usearch_shapes_absorb(&(usearch_shapes_absorb_t) {.index = shapes[0],
                                                      .executor = executor,
                                                      .outcome = &outcome,
                                                      .count = 1,
                                                      .donors = shapes + 1,
                                                      .report = &report});
    expect(!outcome.message, outcome.message);
    expect(shapes[1] == NULL && report.reachability.audited == 2 * entries_per_index, "The donor's shapes must land");
    usearch_shapes_refine(
        &(usearch_shapes_refine_t) {.index = shapes[0], .executor = executor, .outcome = &outcome, .report = &report});
    expect(!outcome.message, outcome.message);
    expect(report.reachability.reachable == 2 * entries_per_index, "Every absorbed shape must stay reachable");
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
                                                              .report = &report});
    expect(!outcome.message, outcome.message);
    expect(
        report.reachability.audited == 2 * entries_per_index && report.reachability.reachable == 2 * entries_per_index,
        "Every absorbed observation must stay reachable");
    usearch_geospatial_free(places[0]);
    usearch_executor_free(executor);
    printf("Test: Shapes & Geospatial Absorb - PASSED\n");
}

/** Leaves every height of a terrain window unknown. */
static void fill_unknown_heights(size_t first_row, size_t first_column, size_t rows, size_t columns, float* heights,
                                 size_t heights_stride, void* state) {
    (void)first_row;
    (void)first_column;
    (void)state;
    for (size_t row = 0; row != rows; ++row)
        for (size_t column = 0; column != columns; ++column)
            ((float*)((char*)heights + row * heights_stride))[column] = NAN;
}

/** A GPU geospatial index refuses a terrain filled by a callback and lines of sight without a
 *  terrain, inserts and describes its entries, and finds every observation itself first, at zero
 *  metres, on host pages; the verbs it lacks refuse. Skipped without a GPU. */
void test_geospatial_vram(void) {
    usearch_outcome_t outcome = {0};
    size_t devices_count = 0;
    usearch_gpu_devices(&(usearch_gpu_devices_t) {.outcome = &outcome, .devices_count = &devices_count});
    if (!devices_count) return;
    printf("Test: Geospatial VRAM\n");
    int32_t const devices[1] = {0};
    usearch_vram_options_t const vram = {.devices = devices, .devices_count = 1};

    usearch_geospatial_index_t index = NULL;
    usearch_geospatial_init_vram(
        &(usearch_geospatial_init_vram_t) {.index = &index, .outcome = &outcome, .vram = vram});
    if (outcome.status == usearch_status_device_k) {
        printf("Test: Geospatial VRAM - SKIPPED: %s\n", outcome.message);
        return;
    }
    expect(!outcome.message, outcome.message);

    usearch_geospatial_terrain_t const terrain = {.rows = 1,
                                                  .columns = 1,
                                                  .row_step_degrees = 1,
                                                  .column_step_degrees = 1,
                                                  .callback = fill_unknown_heights};
    usearch_geospatial_index_t refused = NULL;
    usearch_geospatial_init_vram(
        &(usearch_geospatial_init_vram_t) {.index = &refused, .outcome = &outcome, .terrain = terrain, .vram = vram});
    expect_eq(outcome.status, usearch_status_unsupported_k, "A GPU terrain needs its heights, not a callback");
    expect(refused == NULL, "A refused initialization returns no handle");

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
    expect_eq(outcome.status, usearch_status_unsupported_k, "Lines of sight need a handle with terrain");

    char const path[] = "usearch-test-c-geospatial-vram.safegraphs";
    usearch_geospatial_save(&(usearch_geospatial_save_t) {.index = index,
                                                          .executor = executor,
                                                          .outcome = &outcome,
                                                          .path = {(uint8_t const*)path, sizeof(path) - 1}});
    expect(!outcome.message, outcome.message);
    usearch_geospatial_index_t loaded = NULL;
    usearch_geospatial_load_ram(&(usearch_geospatial_load_ram_t) {.index = &loaded,
                                                                  .outcome = &outcome,
                                                                  .path = {(uint8_t const*)path, sizeof(path) - 1}});
    expect(!outcome.message, outcome.message);
    usearch_geospatial_index_description_t loaded_description = {0};
    usearch_geospatial_describe(
        &(usearch_geospatial_describe_t) {.index = loaded, .outcome = &outcome, .description = &loaded_description});
    expect(!outcome.message, outcome.message);
    expect(loaded_description.size == description.size, "A GPU checkpoint loads into RAM whole");
    usearch_geospatial_free(loaded);
    remove(path);

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

/** The @c int8_t value table an attend test resolves keys through, its first @c rows resolvable. */
typedef struct attend_values_t {
    int8_t const* data;
    size_t rows;
    size_t columns;
    usearch_scalar_t scalar;
} attend_values_t;

/** Resolves a key made by @c key_of to its row of the table, NULL past the resolvable rows. */
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
                                     .carry = carry,
                                     .outputs = &outputs,
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

/** Every written run of @p matches against @p exact, <tt>[query][document]</tt> totals: as many keys
 *  as @p expected_count per query, largest first, each at its exact total, even keys alone when
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
            expect(document < documents, "A written key names a document");
            expect(!even_only || document % 2 == 0, "A predicate narrows the written keys");
            expect(fabs(matches.scores[entry] - exact[query * documents + document]) < 1e-3,
                   "A written key scores its exact summed maxima");
            expect(entry == first || matches.scores[entry] <= matches.scores[entry - 1], "Keys land largest first");
        }
    }
}

/** Documents of 2 to 4 token rows hold small integers, which @c bf16 keeps exactly, so every MaxSim
 *  total is exact: queries of 1, 2 and 3 rows land a ragged tape, walks and a scan alike, every
 *  document without a bound, the best one with one while @c required counts them all, and even
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

/** A RAM index saved as one monolith serves the same keys and vectors from storage: an NVMe
 *  executor views the file, its search, count, get, health and cursor answer as the RAM index did,
 *  and every mutation is refused. An implementation without an NVMe residency says so, and the test
 *  skips. */
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
            &(usearch_dense_view_nvme_t) {.index = &view,
                                          .executor = storage,
                                          .outcome = &outcome,
                                          .path = path_view,
                                          .ram = &(usearch_ram_options_t) {.memory_limit = (size_t)1 << 20}});
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
    usearch_dense_cursor_init(&(usearch_dense_cursor_init_t) {.index = view, .outcome = &outcome, .cursor = &cursor});
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

    // A checkpoint cut inside its header holds no index in any format the library reads.
    FILE* const truncated = fopen(path, "wb");
    expect(truncated != NULL && fwrite(path, 1, 12, truncated) == 12, "Failed to truncate the checkpoint");
    fclose(truncated);
    usearch_dense_index_t truncated_view = NULL;
    usearch_dense_view_nvme(&(usearch_dense_view_nvme_t) {.index = &truncated_view,
                                                          .executor = storage,
                                                          .outcome = &outcome,
                                                          .path = path_view});
    expect(outcome.status != usearch_status_success_k && truncated_view == NULL,
           "A view must refuse a checkpoint cut inside its header");

    usearch_executor_free(storage);
    remove(path);
    free(data);
    usearch_executor_free(executor);
    usearch_dense_free(index);
    printf("Test: NVMe View - PASSED\n");
}

/** A SafeGraphs monolith shaped like a benchmark's save - @c u32 keys, @c bf16 rows, 16 neighbors
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
        usearch_dense_view_nvme(&(usearch_dense_view_nvme_t) {.index = &view,
                                                              .executor = storage,
                                                              .outcome = &outcome,
                                                              .path = path_view,
                                                              .ram = &(usearch_ram_options_t) {0}});
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
 *  exercised here; the failing verbs are the ones @c src/ has yet to instantiate. */
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
    expect(usearch_capabilities_compiled() != NULL, "The compiled capabilities must be reported");
    expect(usearch_capabilities_available() != NULL, "The available capabilities must be reported");

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

/** One enumerator and the spelling its @c _name function answers: a row of the naming contract. */
typedef struct named_value_t {
    int value;
    char const* spelling;
} named_value_t;

/** The bytes of @p text, viewed for a @c _from_name function. */
static usearch_bytes_view_t text_view(char const* text, size_t length) {
    return (usearch_bytes_view_t) {(uint8_t const*)text, length};
}

/** Fails, naming @p what and @p spelling, unless @p holds. */
static void expect_named(bool holds, char const* what, char const* spelling, char const* claim) {
    char message[160];
    snprintf(message, sizeof(message), "%s \"%s\": %s", what, spelling, claim);
    expect(holds, message);
}

/**
 *  @brief Checks one @c _name and @c _from_name pair against @p table, the enumerators it spells.
 *
 *  Every spelling names its value and reads back to it, and no two rows share a spelling. A near
 *  miss of a spelling - upper case, a trailing space or letter, the last letter dropped - and an
 *  empty or null view read back as @p unknown. Every value from @p unknown to eight past the
 *  largest row that no row lists spells @p fallback, which reads back as @p unknown.
 */
static void expect_names(char const* what, named_value_t const* table, size_t count, int unknown, char const* fallback,
                         char const* (*name_of)(int), int (*from_name)(usearch_bytes_view_t)) {
    int largest = unknown;
    for (size_t row = 0; row != count; ++row) {
        char const* const spelling = table[row].spelling;
        size_t const length = strlen(spelling);
        largest = table[row].value > largest ? table[row].value : largest;
        expect_named(strcmp(name_of(table[row].value), spelling) == 0, what, spelling, "the value spells it");
        expect_named(from_name(text_view(spelling, length)) == table[row].value, what, spelling, "it reads back");
        for (size_t other = 0; other != row; ++other)
            expect_named(strcmp(table[other].spelling, spelling) != 0, what, spelling, "no other row spells it");

        char near_miss[64];
        expect_named(length + 1 < sizeof(near_miss), what, spelling, "it fits the near-miss buffer");
        for (size_t letter = 0; letter != length; ++letter)
            near_miss[letter] = spelling[letter] >= 'a' && spelling[letter] <= 'z'
                                    ? (char)(spelling[letter] - 'a' + 'A')
                                    : spelling[letter];
        expect_named(from_name(text_view(near_miss, length)) == unknown, what, spelling, "upper case is unknown");
        memcpy(near_miss, spelling, length);
        near_miss[length] = ' ';
        expect_named(from_name(text_view(near_miss, length + 1)) == unknown, what, spelling,
                     "a trailing space is unknown");
        near_miss[length] = 'x';
        expect_named(from_name(text_view(near_miss, length + 1)) == unknown, what, spelling,
                     "a trailing letter is unknown");
        expect_named(from_name(text_view(spelling, length - 1)) == unknown, what, spelling,
                     "a dropped last letter is unknown");
    }
    expect_named(from_name(text_view("", 0)) == unknown, what, "", "an empty view is unknown");
    expect_named(from_name(text_view(NULL, 0)) == unknown, what, "", "a null view is unknown");
    for (int value = unknown; value <= largest + 8; ++value) {
        bool listed = false;
        for (size_t row = 0; row != count; ++row) listed = listed || table[row].value == value;
        if (!listed) expect_named(strcmp(name_of(value), fallback) == 0, what, fallback, "an unlisted value spells it");
    }
    expect_named(from_name(text_view(fallback, strlen(fallback))) == unknown, what, fallback, "it reads back unknown");
}

static char const* status_name_of(int value) { return usearch_status_name((usearch_status_t)value); }
static int status_from_name(usearch_bytes_view_t name) {
    usearch_status_t status = (usearch_status_t)0;
    return usearch_status_from_name(name, &status) == usearch_status_success_k ? (int)status : -1;
}
static char const* scalar_name_of(int value) { return usearch_scalar_name((usearch_scalar_t)value); }
static int scalar_from_name(usearch_bytes_view_t name) { return (int)usearch_scalar_from_name(name); }
static char const* residency_name_of(int value) { return usearch_residency_name((usearch_residency_t)value); }
static int residency_from_name(usearch_bytes_view_t name) {
    usearch_residency_t residency = (usearch_residency_t)0;
    return usearch_residency_from_name(name, &residency) == usearch_status_success_k ? (int)residency : -1;
}
static char const* traversal_name_of(int value) { return usearch_traversal_name((usearch_traversal_t)value); }
static int traversal_from_name(usearch_bytes_view_t name) {
    usearch_traversal_t traversal = (usearch_traversal_t)0;
    return usearch_traversal_from_name(name, &traversal) == usearch_status_success_k ? (int)traversal : -1;
}
static char const* mood_name_of(int value) { return usearch_mood_name((usearch_mood_t)value); }
static int mood_from_name(usearch_bytes_view_t name) {
    usearch_mood_t mood = (usearch_mood_t)0;
    return usearch_mood_from_name(name, &mood) == usearch_status_success_k ? (int)mood : -1;
}
static char const* cpu_class_name_of(int value) { return usearch_cpu_class_name((usearch_cpu_class_t)value); }
static int cpu_class_from_name(usearch_bytes_view_t name) { return (int)usearch_cpu_class_from_name(name); }
static char const* gpu_class_name_of(int value) { return usearch_gpu_class_name((usearch_gpu_class_t)value); }
static int gpu_class_from_name(usearch_bytes_view_t name) { return (int)usearch_gpu_class_from_name(name); }
static char const* nvme_class_name_of(int value) { return usearch_nvme_class_name((usearch_nvme_class_t)value); }
static int nvme_class_from_name(usearch_bytes_view_t name) { return (int)usearch_nvme_class_from_name(name); }
static char const* dense_metric_name_of(int value) { return usearch_dense_metric_name((usearch_dense_metric_t)value); }
static int dense_metric_from_name(usearch_bytes_view_t name) { return (int)usearch_dense_metric_from_name(name); }

/** Every @c _name and @c _from_name pair against the spellings its enumerators carry. */
void test_names(void) {
    printf("Test: Names\n");
    named_value_t const statuses[] = {
        {usearch_status_success_k, "success"},
        {usearch_status_deferred_k, "deferred"},
        {usearch_status_out_of_memory_k, "out-of-memory"},
        {usearch_status_shape_mismatch_k, "shape-mismatch"},
        {usearch_status_unsupported_layout_k, "unsupported-layout"},
        {usearch_status_unknown_kind_k, "unknown-kind"},
        {usearch_status_read_only_k, "read-only"},
        {usearch_status_io_k, "io"},
        {usearch_status_device_k, "device"},
        {usearch_status_malformed_k, "malformed"},
        {usearch_status_internal_k, "internal"},
        {usearch_status_overflow_k, "overflow"},
        {usearch_status_uninitialized_k, "uninitialized"},
        {usearch_status_config_mismatch_k, "config-mismatch"},
        {usearch_status_contended_k, "contended"},
        {usearch_status_cancelled_k, "cancelled"},
        {usearch_status_unsupported_k, "unsupported"},
        {usearch_status_invalid_argument_k, "invalid-argument"},
    };
    named_value_t const scalars[] = {
        {usearch_scalar_f64_k, "f64"},   {usearch_scalar_f32_k, "f32"},   {usearch_scalar_bf16_k, "bf16"},
        {usearch_scalar_f16_k, "f16"},   {usearch_scalar_e5m2_k, "e5m2"}, {usearch_scalar_e4m3_k, "e4m3"},
        {usearch_scalar_e3m2_k, "e3m2"}, {usearch_scalar_e2m3_k, "e2m3"}, {usearch_scalar_i8_k, "i8"},
        {usearch_scalar_i4x2_k, "i4x2"}, {usearch_scalar_i32_k, "i32"},   {usearch_scalar_u8_k, "u8"},
        {usearch_scalar_u4x2_k, "u4x2"}, {usearch_scalar_u1x8_k, "u1x8"},
    };
    named_value_t const residencies[] = {
        {usearch_residency_ram_k, "ram"},
        {usearch_residency_vram_k, "vram"},
        {usearch_residency_nvme_k, "nvme"},
    };
    named_value_t const traversals[] = {{usearch_traversal_walk_k, "walk"}, {usearch_traversal_scan_k, "scan"}};
    named_value_t const moods[] = {{usearch_mood_parked_k, "parked"}, {usearch_mood_spinning_k, "spinning"}};
    named_value_t const cpu_classes[] = {
        {usearch_cpu_class_serial_k, "serial"},
        {usearch_cpu_class_intel_haswell_k, "intel-haswell"},
        {usearch_cpu_class_intel_sierra_k, "intel-sierra"},
        {usearch_cpu_class_intel_skylake_k, "intel-skylake"},
        {usearch_cpu_class_intel_icelake_k, "intel-icelake"},
        {usearch_cpu_class_amd_genoa_k, "amd-genoa"},
        {usearch_cpu_class_intel_sapphire_k, "intel-sapphire"},
        {usearch_cpu_class_intel_granite_k, "intel-granite"},
        {usearch_cpu_class_intel_diamond_k, "intel-diamond"},
        {usearch_cpu_class_aws_g2_k, "aws-g2"},
        {usearch_cpu_class_apple_m2_k, "apple-m2"},
        {usearch_cpu_class_apple_m4_k, "apple-m4"},
        {usearch_cpu_class_aws_g5_k, "aws-g5"},
        {usearch_cpu_class_nvidia_vera_k, "nvidia-vera"},
        {usearch_cpu_class_alibaba_c930_k, "alibaba-c930"},
        {usearch_cpu_class_wasm32_v128_k, "wasm32-v128"},
        {usearch_cpu_class_wasm64_extended_k, "wasm64-extended"},
        {usearch_cpu_class_monoversioned_k, "monoversioned"},
    };
    named_value_t const gpu_classes[] = {
        {usearch_gpu_class_nvidia_ampere_k, "nvidia-ampere"},
        {usearch_gpu_class_nvidia_ada_k, "nvidia-ada"},
        {usearch_gpu_class_nvidia_hopper_k, "nvidia-hopper"},
        {usearch_gpu_class_nvidia_blackwell_k, "nvidia-blackwell"},
        {usearch_gpu_class_nvidia_blackwellrtx_k, "nvidia-blackwellrtx"},
        {usearch_gpu_class_nvidia_rubin_k, "nvidia-rubin"},
        {usearch_gpu_class_amd_cdna3_k, "amd-cdna3"},
        {usearch_gpu_class_amd_cdna4_k, "amd-cdna4"},
        {usearch_gpu_class_amd_cdna5_k, "amd-cdna5"},
        {usearch_gpu_class_apple_metal7_k, "apple-metal7"},
        {usearch_gpu_class_apple_metal9_k, "apple-metal9"},
        {usearch_gpu_class_apple_metal10_k, "apple-metal10"},
    };
    named_value_t const nvme_classes[] = {
        {usearch_nvme_class_posix_k, "posix"},
        {usearch_nvme_class_linux_aio_k, "linux-aio"},
        {usearch_nvme_class_linux_uring_k, "linux-uring"},
        {usearch_nvme_class_linux_uring_passthrough_k, "linux-uring-passthrough"},
        {usearch_nvme_class_linux_spdk_k, "linux-spdk"},
        {usearch_nvme_class_windows_ioring_k, "windows-ioring"},
    };
    named_value_t const dense_metrics[] = {
        {usearch_dense_metric_inner_k, "inner"},         {usearch_dense_metric_angular_k, "angular"},
        {usearch_dense_metric_euclidean_k, "euclidean"}, {usearch_dense_metric_hamming_k, "hamming"},
        {usearch_dense_metric_jaccard_k, "jaccard"},
    };
    size_t const row_bytes = sizeof(named_value_t);
    // Parsers answering a status read a refused spelling as -1, which no enumerator holds.
    expect_names("status", statuses, sizeof(statuses) / row_bytes, -1, "unrecognized", status_name_of,
                 status_from_name);
    expect_names("scalar", scalars, sizeof(scalars) / row_bytes, usearch_scalar_unknown_k, "unknown", scalar_name_of,
                 scalar_from_name);
    expect_names("residency", residencies, sizeof(residencies) / row_bytes, -1, "unknown", residency_name_of,
                 residency_from_name);
    expect_names("traversal", traversals, sizeof(traversals) / row_bytes, -1, "unknown", traversal_name_of,
                 traversal_from_name);
    expect_names("mood", moods, sizeof(moods) / row_bytes, -1, "unknown", mood_name_of, mood_from_name);
    usearch_bytes_view_t const bogus = text_view("bogus", 5);
    usearch_status_t status = usearch_status_success_k;
    usearch_residency_t residency = usearch_residency_ram_k;
    usearch_traversal_t traversal = usearch_traversal_walk_k;
    usearch_mood_t mood = usearch_mood_parked_k;
    expect_eq(usearch_status_from_name(bogus, &status), usearch_status_invalid_argument_k, "A bogus status is refused");
    expect_eq(usearch_residency_from_name(bogus, &residency), usearch_status_invalid_argument_k,
              "A bogus residency is refused");
    expect_eq(usearch_traversal_from_name(bogus, &traversal), usearch_status_invalid_argument_k,
              "A bogus traversal is refused");
    expect_eq(usearch_mood_from_name(bogus, &mood), usearch_status_invalid_argument_k, "A bogus mood is refused");
    expect(status == usearch_status_success_k && residency == usearch_residency_ram_k &&
               traversal == usearch_traversal_walk_k && mood == usearch_mood_parked_k,
           "A refused spelling leaves its output alone");
    expect_names("CPU class", cpu_classes, sizeof(cpu_classes) / row_bytes, usearch_cpu_class_unknown_k, "unknown",
                 cpu_class_name_of, cpu_class_from_name);
    expect_names("GPU class", gpu_classes, sizeof(gpu_classes) / row_bytes, usearch_gpu_class_unknown_k, "unknown",
                 gpu_class_name_of, gpu_class_from_name);
    expect_names("NVMe class", nvme_classes, sizeof(nvme_classes) / row_bytes, usearch_nvme_class_unknown_k, "unknown",
                 nvme_class_name_of, nvme_class_from_name);
    expect_names("dense metric", dense_metrics, sizeof(dense_metrics) / row_bytes, usearch_dense_metric_unknown_k,
                 "unknown", dense_metric_name_of, dense_metric_from_name);
    printf("Test: Names - PASSED\n");
}

enum { deferred_rows = 128, deferred_dimensions = 16, deferred_wanted = 4 };

/** One search's outputs, the arena they land in, and its outcome, which a deferred call holds until final. */
typedef struct search_outputs_t {
    usearch_matches_view_t matches;
    usearch_arena_t arena;
    usearch_outcome_t outcome;
} search_outputs_t;

/** Whether two searches of every row wrote the same matches. */
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

/** Holds the call reporting to it at its last progress report until the test releases it, so the
 *  test meets the call while it still holds its executor; a call run at submission stops waiting
 *  after a few seconds, since its submitter cannot release it. */
typedef struct gate_t {
    int volatile entered;
    int volatile released;
} gate_t;

static usearch_continuation_t hold_until_released(size_t completed, size_t total, void* state) {
    gate_t* const gate = (gate_t*)state;
    if (completed != total) return usearch_continue_k;
    gate->entered = 1;
    time_t const deadline = time(NULL) + 3;
    while (!gate->released && time(NULL) < deadline) {}
    return usearch_continue_k;
}

/** Whether @p gate saw its call arrive within a few seconds. */
static bool entered_soon(gate_t const* gate) {
    time_t const deadline = time(NULL) + 3;
    while (!gate->entered && time(NULL) < deadline) {}
    return gate->entered;
}

/** Submits a search of every row of @p rows into @p outputs, held at @p gate unless it is NULL; its
 *  descriptor dies when this returns. */
void search_every_row_through(usearch_dense_index_t index, usearch_executor_t executor, usearch_completion_t completion,
                              float const* rows, search_outputs_t* outputs, gate_t* gate) {
    usearch_progress_t const progress = {gate ? hold_until_released : NULL, gate, 1};
    usearch_dense_search(&(usearch_dense_search_t) {
        .index = index,
        .executor = executor,
        .outcome = &outputs->outcome,
        .completion = completion,
        .count = deferred_rows,
        .queries = {rows, deferred_dimensions * sizeof(float), deferred_dimensions, usearch_scalar_f32_k},
        .matches = &outputs->matches,
        .arena = outputs->arena,
        .wanted = deferred_wanted,
        .progress = progress});
}

/** Submits a search of every row of @p rows into @p outputs; its descriptor dies on return. */
void search_every_row(usearch_dense_index_t index, usearch_executor_t executor, usearch_completion_t completion,
                      float const* rows, search_outputs_t* outputs) {
    search_every_row_through(index, executor, completion, rows, outputs, NULL);
}

/** Waits until no call holds @p executor, then leaves its threads in @p mood. */
void drain(usearch_executor_t executor, usearch_mood_t mood) {
    usearch_outcome_t outcome = {0};
    usearch_executor_drain(&(usearch_executor_drain_t) {.executor = executor, .outcome = &outcome, .mood = mood});
    expect(!outcome.message, outcome.message);
}

/** A RAM index pinned to the serial class opens on any CPU and describes itself under that class
 *  and no GPU class, and an NVMe executor submits through the POSIX class it names. */
void test_pins_and_fill(void) {
    printf("Test: Pins and Fill\n");
    usearch_outcome_t outcome = {0};
    usearch_dense_index_t index = NULL;
    usearch_ram_options_t const serial = {.cpu_class = usearch_cpu_class_serial_k};
    usearch_dense_init_ram(&(usearch_dense_init_ram_t) {.index = &index,
                                                        .outcome = &outcome,
                                                        .metric = usearch_dense_metric_angular_k,
                                                        .quantization = usearch_scalar_bf16_k,
                                                        .dimensions = 8,
                                                        .ram = serial});
    expect(!outcome.message, outcome.message);
    usearch_dense_reserve(&(usearch_dense_reserve_t) {.index = index, .outcome = &outcome, .capacity = 2});
    expect(!outcome.message, outcome.message);
    float const rows[16] = {1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1};
    usearch_key_t const keys[2] = {key_of(1), key_of(2)};
    usearch_dense_insert(&(usearch_dense_insert_t) {.index = index,
                                                    .outcome = &outcome,
                                                    .count = 2,
                                                    .keys = {keys, sizeof(usearch_key_t)},
                                                    .vectors = {rows, 8 * sizeof(float), 8, usearch_scalar_f32_k}});
    expect(!outcome.message, outcome.message);
    usearch_dense_index_description_t description = {0};
    usearch_dense_describe(
        &(usearch_dense_describe_t) {.index = index, .outcome = &outcome, .description = &description});
    expect(!outcome.message, outcome.message);
    expect(description.shards_count >= 1, "A fresh index spans at least one shard");
    expect_eq(description.size, (size_t)2, "The description counts the rows inserted");
    expect(description.cpu_class == usearch_cpu_class_serial_k, "A serial pin dispatches the serial class");
    expect(description.gpu_class == usearch_gpu_class_unknown_k, "A RAM index names no GPU class");
    expect(description.load_path == usearch_load_path_none_k, "A RAM index names no load path");
    usearch_dense_free(index);

    usearch_executor_t storage = NULL;
    usearch_executor_for_nvme(&(usearch_executor_for_nvme_t) {.executor = &storage,
                                                              .outcome = &outcome,
                                                              .threads = 1,
                                                              .nvme_class = usearch_nvme_class_posix_k});
    expect(!outcome.message, outcome.message);
    usearch_executor_free(storage);
    printf("Test: Pins and Fill - PASSED\n");
}

/** A deferred search lands what a blocking one does, though its descriptor died at submission; one
 *  thread is the caller's own, so it runs at submission, and so does a call without an executor. */
void test_deferred_search(void) {
    printf("Test: Deferred Search\n");
    float* const rows = create_vectors(deferred_rows, deferred_dimensions);
    // Landmarks are drawn per thread, so only one as wide as a call without an executor
    // draws the same.
    usearch_executor_t const executor = create_executor(1);
    usearch_dense_index_t const index = create_filled_index(rows, executor);
    search_outputs_t blocking = {.arena = create_arena(executor)};
    search_outputs_t deferred = {.arena = create_arena(executor)};

    search_every_row(index, executor, usearch_completion_blocking_k, rows, &blocking);
    expect(!blocking.outcome.message, blocking.outcome.message);

    search_every_row(index, executor, usearch_completion_deferred_k, rows, &deferred);
    expect_eq(usearch_outcome_status(&deferred.outcome), usearch_status_success_k,
              "A pool needing its caller runs a deferred call at submission");
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

/** A deferred call on a pool of its own threads finishes while its submitter only polls. */
void test_deferred_polling(void) {
    printf("Test: Deferred Polling\n");
    float* const rows = create_vectors(deferred_rows, deferred_dimensions);
    usearch_executor_t const executor = create_executor(2);
    usearch_dense_index_t const index = create_filled_index(rows, executor);
    search_outputs_t outputs = {.arena = create_arena(executor)};

    search_every_row(index, executor, usearch_completion_deferred_k, rows, &outputs);
    time_t const deadline = time(NULL) + 30;
    while (usearch_outcome_status(&outputs.outcome) == usearch_status_deferred_k && time(NULL) < deadline) {}
    expect_eq(usearch_outcome_status(&outputs.outcome), usearch_status_success_k, "Polling alone finishes the call");
    for (size_t row = 0; row != deferred_rows; ++row)
        expect_eq(outputs.matches.counts[row], deferred_wanted, "A polled call lands its matches");

    usearch_arena_free(outputs.arena);
    usearch_dense_free(index);
    usearch_executor_free(executor);
    free(rows);
    printf("Test: Deferred Polling - PASSED\n");
}

/** While a deferred call holds its executor on a pool thread, a call from this thread is refused at
 *  once rather than overlapping it, and the executor serves again once the first call ended. */
void test_contended_from_another_thread(void) {
    printf("Test: Contended From Another Thread\n");
    float* const rows = create_vectors(deferred_rows, deferred_dimensions);
    usearch_executor_t const executor = create_executor(2);
    usearch_dense_index_t const index = create_filled_index(rows, executor);
    search_outputs_t holding = {.arena = create_arena(executor)};
    search_outputs_t second = {.arena = create_arena(executor)};
    gate_t gate = {0, 0};

    search_every_row_through(index, executor, usearch_completion_deferred_k, rows, &holding, &gate);
    // An executor without threads of its own, as the stand-in's, ran the call at submission.
    if (usearch_outcome_status(&holding.outcome) == usearch_status_deferred_k) {
        expect(entered_soon(&gate), "A delegated call reaches its progress handler");
        search_every_row(index, executor, usearch_completion_blocking_k, rows, &second);
        expect_eq(second.outcome.status, usearch_status_contended_k, "A second call on a held executor is refused");
        search_every_row(index, executor, usearch_completion_deferred_k, rows, &second);
        expect_eq(second.outcome.status, usearch_status_contended_k, "A deferred one is refused at submission");
        gate.released = 1;
    }
    expect_eq(usearch_outcome_wait(&holding.outcome), usearch_status_success_k, "The holding call finishes");
    search_every_row(index, executor, usearch_completion_blocking_k, rows, &second);
    expect(!second.outcome.message, second.outcome.message);

    usearch_arena_free(second.arena);
    usearch_arena_free(holding.arena);
    usearch_dense_free(index);
    usearch_executor_free(executor);
    free(rows);
    printf("Test: Contended From Another Thread - PASSED\n");
}

/** A drain returns once the deferred call ended, whichever mood it then leaves the threads in, and
 *  the executor serves the next call after either. */
void test_drain_then_mood(void) {
    printf("Test: Drain Then Mood\n");
    float* const rows = create_vectors(deferred_rows, deferred_dimensions);
    usearch_executor_t const executor = create_executor(2);
    usearch_dense_index_t const index = create_filled_index(rows, executor);
    search_outputs_t outputs = {.arena = create_arena(executor)};

    search_every_row(index, executor, usearch_completion_deferred_k, rows, &outputs);
    drain(executor, usearch_mood_parked_k);
    expect_eq(usearch_outcome_status(&outputs.outcome), usearch_status_success_k, "A drain waits out the call");
    search_every_row(index, executor, usearch_completion_deferred_k, rows, &outputs);
    drain(executor, usearch_mood_spinning_k);
    expect_eq(usearch_outcome_status(&outputs.outcome), usearch_status_success_k, "Parked threads wake for a call");
    search_every_row(index, executor, usearch_completion_blocking_k, rows, &outputs);
    expect(!outputs.outcome.message, outputs.outcome.message);

    usearch_arena_free(outputs.arena);
    usearch_dense_free(index);
    usearch_executor_free(executor);
    free(rows);
    printf("Test: Drain Then Mood - PASSED\n");
}

/** Freeing an index waits for the deferred call naming it before its memory goes. */
void test_free_waits_for_named_call(void) {
    printf("Test: Free Waits for Named Call\n");
    float* const rows = create_vectors(deferred_rows, deferred_dimensions);
    usearch_executor_t const executor = create_executor(2);
    usearch_dense_index_t const index = create_filled_index(rows, executor);
    search_outputs_t outputs = {.arena = create_arena(executor)};

    search_every_row(index, executor, usearch_completion_deferred_k, rows, &outputs);
    usearch_dense_free(index);
    expect_eq(usearch_outcome_status(&outputs.outcome), usearch_status_success_k,
              "A free returns only once the calls naming the handle ended");
    for (size_t row = 0; row != deferred_rows; ++row)
        expect_eq(outputs.matches.counts[row], deferred_wanted, "A call a free waited for lands its matches");

    usearch_arena_free(outputs.arena);
    usearch_executor_free(executor);
    free(rows);
    printf("Test: Free Waits for Named Call - PASSED\n");
}

/** A deferred call holds its arena until just before it is final, so another call naming it
 *  meanwhile is refused, and the arena serves again once the deferred call ended. */
void test_deferred_holds_arena(void) {
    printf("Test: Deferred Holds Arena\n");
    float* const rows = create_vectors(deferred_rows, deferred_dimensions);
    usearch_executor_t const executor = create_executor(2);
    usearch_dense_index_t const index = create_filled_index(rows, executor);
    search_outputs_t outputs = {.arena = create_arena(executor)};
    gate_t gate = {0, 0};

    search_every_row_through(index, executor, usearch_completion_deferred_k, rows, &outputs, &gate);
    usearch_outcome_t outcome = {0};
    usearch_rows_view_t vectors = {0};
    usearch_dense_get_t get = {.index = index,
                               .outcome = &outcome,
                               .key = key_of(3),
                               .vectors = &vectors,
                               .arena = outputs.arena,
                               .outputs_scalar = usearch_scalar_f32_k};
    // An executor without threads of its own, as the stand-in's, ran the call at submission.
    if (usearch_outcome_status(&outputs.outcome) == usearch_status_deferred_k) {
        expect(entered_soon(&gate), "A delegated call reaches its progress handler");
        // Without an executor the get runs at once, so it meets the deferred call's hold.
        usearch_dense_get(&get);
        expect_eq(outcome.status, usearch_status_contended_k, "An arena a deferred call holds refuses another call");
        expect(vectors.data == NULL, "A refused call leaves its outputs NULL");
        gate.released = 1;
    }
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

/** A three-part checkpoint loaded onto one GPU keeps a shard per part on it, loads every entry, and
 *  reaches exactly the nodes the saved graph reached. Skipped without a GPU. */
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
    if (outcome.status == usearch_status_device_k) {
        printf("Test: VRAM Parts Share a Device - SKIPPED: %s\n", outcome.message);
        return;
    }
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
    // Insertions may leave islands, so the reload is judged against the saved graph's own audit.
    usearch_reachability_t saved_reachability = {0};
    usearch_dense_health(&(usearch_dense_health_t) {.index = index,
                                                    .executor = executor,
                                                    .outcome = &outcome,
                                                    .reachability = &saved_reachability});
    expect(!outcome.message, outcome.message);
    char const path[] = "usearch-test-c-parts.safegraphs";
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
    for (int part = 1; part <= 3; ++part) {
        char part_path[64];
        snprintf(part_path, sizeof(part_path), "usearch-test-c-parts-%05d-of-00003.safegraphs", part);
        remove(part_path);
    }
    free(keys);
    free(data);
    printf("Test: VRAM Parts Share a Device - PASSED\n");
}

/** An exact scan on the first GPU scores every entry once per query and refuses a predicate as a
 *  walk does where the vendor scans, and refuses with @c usearch_status_unsupported_k elsewhere. */
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
    if (outcome.status == usearch_status_device_k) {
        printf("Test: VRAM Scan - SKIPPED: %s\n", outcome.message);
        return;
    }
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

static void release_schema_of_test(struct ArrowSchema* schema) { schema->release = NULL; }

static void release_array_of_test(struct ArrowArray* array) { array->release = NULL; }

/** Releases an exported column, the schema first, as its consumer may. */
static void release_column(struct ArrowSchema* schema, struct ArrowDeviceArray* array) {
    schema->release(schema);
    array->array.release(&array->array);
    expect(!schema->release && !array->array.release, "A release marks its structure released");
}

/** An index of four short strings, then a long one a get lands in an arena as an output would. */
static usearch_sequence_index_t create_strings(usearch_executor_t executor) {
    usearch_outcome_t outcome = {0};
    char const bytes[536] = "kittensittingmittenflask";
    size_t const offsets[] = {0, 6, 13, 19, 24, 536};
    usearch_key_t const keys[] = {key_of(0), key_of(1), key_of(2), key_of(3), key_of(4)};
    usearch_sequence_index_t index = NULL;
    usearch_sequence_init_ram(&(usearch_sequence_init_ram_t) {.index = &index,
                                                              .outcome = &outcome,
                                                              .metric = usearch_sequence_metric_levenshtein_bytes_k});
    expect(!outcome.message, outcome.message);
    usearch_sequence_insert(&(usearch_sequence_insert_t) {.index = index,
                                                          .executor = executor,
                                                          .outcome = &outcome,
                                                          .count = 5,
                                                          .keys = {keys, sizeof(usearch_key_t)},
                                                          .sequences = tape_of(bytes, offsets, usearch_scalar_u8_k)});
    expect(!outcome.message, outcome.message);
    return index;
}

/** Rewinds @p arena and answers the 512 bytes a get wrote there, which any export may name. */
static uint8_t* bytes_in_arena(usearch_sequence_index_t index, usearch_executor_t executor, usearch_arena_t arena) {
    usearch_outcome_t outcome = {0};
    usearch_tape_view_t got = {0};
    size_t written = 0;
    usearch_sequence_get(&(usearch_sequence_get_t) {.index = index,
                                                    .executor = executor,
                                                    .outcome = &outcome,
                                                    .key = key_of(4),
                                                    .sequences = &got,
                                                    .written = &written,
                                                    .arena = arena});
    expect(!outcome.message, outcome.message);
    expect_eq(written, 1, "A get finds the long string");
    return (uint8_t*)got.data;
}

void test_arrow_rows_and_keys(void) {
    printf("Test: Arrow Rows and Keys\n");
    usearch_executor_t const executor = create_executor(0);
    usearch_arena_t const arena = create_arena(executor);
    usearch_sequence_index_t const index = create_strings(executor);
    uint8_t const* const bytes = bytes_in_arena(index, executor, arena);

    struct {
        usearch_scalar_t scalar;
        size_t columns;
        size_t row_bytes;
        char const* format;
        char const* element;
    } const cases[] = {
        {usearch_scalar_f64_k, 4, 32, "+w:4", "g"},   {usearch_scalar_f32_k, 4, 16, "+w:4", "f"},
        {usearch_scalar_f16_k, 4, 8, "+w:4", "e"},    {usearch_scalar_bf16_k, 4, 8, "+w:4", "S"},
        {usearch_scalar_e5m2_k, 4, 4, "+w:4", "C"},   {usearch_scalar_e4m3_k, 4, 4, "+w:4", "C"},
        {usearch_scalar_e3m2_k, 4, 4, "+w:4", "C"},   {usearch_scalar_e2m3_k, 4, 4, "+w:4", "C"},
        {usearch_scalar_i8_k, 4, 4, "+w:4", "c"},     {usearch_scalar_u8_k, 4, 4, "+w:4", "C"},
        {usearch_scalar_i32_k, 4, 16, "+w:4", "i"},   {usearch_scalar_i4x2_k, 4, 2, "+w:2", "C"},
        {usearch_scalar_u4x2_k, 4, 2, "+w:2", "C"},   {usearch_scalar_u1x8_k, 16, 2, "w:2", NULL},
        {usearch_scalar_f64_k, 11, 88, "+w:11", "g"},
    };
    for (size_t index_of_case = 0; index_of_case != sizeof(cases) / sizeof(cases[0]); ++index_of_case) {
        struct ArrowSchema schema = {0};
        struct ArrowDeviceArray array = {0};
        usearch_rows_view_t const rows = {bytes, cases[index_of_case].row_bytes, cases[index_of_case].columns,
                                          cases[index_of_case].scalar};
        expect_eq(usearch_rows_view_to_arrow(rows, 3, arena, &schema, &array), usearch_status_success_k,
                  "Every scalar exports");
        expect(!strcmp(schema.format, cases[index_of_case].format), "A row's format spells its width");
        expect(!cases[index_of_case].element || !strcmp(schema.children[0]->format, cases[index_of_case].element),
               "An element's format spells its scalar");
        expect(array.device_type == ARROW_DEVICE_CPU && array.array.length == 3, "Host rows export as host rows");

        usearch_rows_view_t viewed = {0};
        size_t viewed_count = 0;
        expect_eq(usearch_rows_view_from_arrow(&schema, &array.array, &viewed, &viewed_count), usearch_status_success_k,
                  "Every scalar imports");
        expect(viewed.data == (void const*)bytes && viewed.stride == rows.stride && viewed.columns == rows.columns &&
                   viewed.scalar == rows.scalar && viewed_count == 3,
               "An import views the very rows an export named");

        array.array.offset = 1, array.array.length = 2;
        expect_eq(usearch_rows_view_from_arrow(&schema, &array.array, &viewed, &viewed_count), usearch_status_success_k,
                  "A slice imports");
        expect(viewed.data == (void const*)(bytes + rows.stride) && viewed_count == 2, "A slice applies its offset");
        release_column(&schema, &array);
    }

    struct ArrowSchema schema = {0};
    struct ArrowDeviceArray array = {0};
    usearch_rows_view_t const uneven = {bytes, 2, 3, usearch_scalar_u4x2_k};
    expect_eq(usearch_rows_view_to_arrow(uneven, 1, arena, &schema, &array), usearch_status_unsupported_layout_k,
              "Sub-byte rows fill whole bytes");

    usearch_keys_view_t const keys = {(usearch_key_t const*)bytes, sizeof(usearch_key_t)};
    expect_eq(usearch_keys_view_to_arrow(keys, 3, arena, &schema, &array), usearch_status_success_k, "Keys export");
    expect(!strcmp(schema.format, "w:16") && schema.metadata != NULL, "Keys export as tagged 16-byte values");
    usearch_keys_view_t viewed_keys = {0};
    size_t viewed_count = 0;
    expect_eq(usearch_keys_view_from_arrow(&schema, &array.array, &viewed_keys, &viewed_count),
              usearch_status_success_k, "Keys import");
    expect(viewed_keys.data == keys.data && viewed_keys.stride == sizeof(usearch_key_t) && viewed_count == 3,
           "An import views the very keys an export named");
    release_column(&schema, &array);

    usearch_keys_view_t const strided = {(usearch_key_t const*)bytes, 2 * sizeof(usearch_key_t)};
    expect_eq(usearch_keys_view_to_arrow(strided, 2, arena, &schema, &array), usearch_status_unsupported_layout_k,
              "An export takes packed keys alone");
    usearch_key_t const outside[2] = {key_of(1), key_of(2)};
    usearch_keys_view_t const foreign = {outside, sizeof(usearch_key_t)};
    expect_eq(usearch_keys_view_to_arrow(foreign, 2, arena, &schema, &array), usearch_status_invalid_argument_k,
              "An export takes the arena's own outputs alone");

    usearch_sequence_free(index);
    usearch_arena_free(arena);
    usearch_executor_free(executor);
}

void test_arrow_refusals(void) {
    printf("Test: Arrow Refusals\n");
    uint8_t const bytes[32] = {0};
    uint8_t const validity = 0xFF;
    void const* buffers[2] = {NULL, bytes};
    struct ArrowSchema schema = {.format = "w:16", .release = &release_schema_of_test};
    struct ArrowArray array = {.length = 2, .n_buffers = 2, .buffers = buffers, .release = &release_array_of_test};
    usearch_keys_view_t keys = {0};
    usearch_rows_view_t rows = {0};
    usearch_tape_view_t tape = {0};
    size_t count = 0;

    expect_eq(usearch_keys_view_from_arrow(&schema, &array, &keys, &count), usearch_status_success_k,
              "A column with no validity buffer holds no null");
    array.null_count = -1;
    expect_eq(usearch_keys_view_from_arrow(&schema, &array, &keys, &count), usearch_status_success_k,
              "An unknown count of nulls without a validity buffer rules them out");
    array.null_count = 1;
    expect_eq(usearch_keys_view_from_arrow(&schema, &array, &keys, &count), usearch_status_invalid_argument_k,
              "A counted null is refused");
    buffers[0] = &validity, array.null_count = -1;
    expect_eq(usearch_keys_view_from_arrow(&schema, &array, &keys, &count), usearch_status_invalid_argument_k,
              "An unknown count of nulls beside a validity buffer is refused");
    buffers[0] = NULL, array.null_count = 0;

    array.length = -1;
    expect_eq(usearch_keys_view_from_arrow(&schema, &array, &keys, &count), usearch_status_malformed_k,
              "A negative length is malformed");
    array.length = 2;
    schema.dictionary = &schema;
    expect_eq(usearch_keys_view_from_arrow(&schema, &array, &keys, &count), usearch_status_unsupported_layout_k,
              "A dictionary is refused");
    schema.dictionary = NULL;
    schema.format = "w:8";
    expect_eq(usearch_keys_view_from_arrow(&schema, &array, &keys, &count), usearch_status_unsupported_layout_k,
              "Keys are 16 bytes wide");
    expect_eq(usearch_rows_view_from_arrow(&schema, &array, &rows, &count), usearch_status_unsupported_layout_k,
              "An untagged fixed-size column holds no rows");
    expect_eq(usearch_tape_view_from_arrow(&schema, &array, &tape, &count), usearch_status_unsupported_layout_k,
              "A fixed-size column is no tape");
    schema.format = "w:16", schema.release = NULL;
    expect_eq(usearch_keys_view_from_arrow(&schema, &array, &keys, &count), usearch_status_invalid_argument_k,
              "A released schema is refused");
    schema.release = &release_schema_of_test;

    void const* list_buffers[1] = {NULL};
    struct ArrowSchema element = {.format = "b", .release = &release_schema_of_test};
    struct ArrowSchema* element_pointer = &element;
    struct ArrowArray values = {.length = 8, .n_buffers = 2, .buffers = buffers, .release = &release_array_of_test};
    struct ArrowArray* values_pointer = &values;
    struct ArrowSchema list_schema = {.format = "+w:4",
                                      .n_children = 1,
                                      .children = &element_pointer,
                                      .release = &release_schema_of_test};
    struct ArrowArray list = {.length = 2,
                              .n_buffers = 1,
                              .n_children = 1,
                              .buffers = list_buffers,
                              .children = &values_pointer,
                              .release = &release_array_of_test};
    expect_eq(usearch_rows_view_from_arrow(&list_schema, &list, &rows, &count), usearch_status_unsupported_layout_k,
              "A list of bools is refused");
    element.format = "f";
    expect_eq(usearch_rows_view_from_arrow(&list_schema, &list, &rows, &count), usearch_status_success_k,
              "A list of floats imports");
    values.null_count = 1;
    expect_eq(usearch_rows_view_from_arrow(&list_schema, &list, &rows, &count), usearch_status_invalid_argument_k,
              "A null among a row's elements is refused");
    values.null_count = 0, values.length = 7;
    expect_eq(usearch_rows_view_from_arrow(&list_schema, &list, &rows, &count), usearch_status_malformed_k,
              "Rows past their child's values are malformed");
}

void test_arrow_tapes(void) {
    printf("Test: Arrow Tapes\n");
    usearch_outcome_t outcome = {0};
    usearch_executor_t const executor = create_executor(0);
    usearch_arena_t const arena = create_arena(executor);

    // Both widths of an input reach the same entries, and a slice's offsets start anywhere
    char const bytes[] = "kittensittingmittenflask";
    uint32_t const narrow[] = {0, 6, 13, 19, 24};
    uint64_t const wide[] = {0, 6, 13, 19, 24};
    usearch_key_t const keys[] = {key_of(0), key_of(1), key_of(2), key_of(3)};
    for (int layout = 0; layout != 2; ++layout) {
        usearch_tape_view_t tape = {.data = bytes, .scalar = usearch_scalar_u8_k};
        if (layout) tape.layout = usearch_tape_u64_k, tape.offsets.u64 = wide;
        else tape.layout = usearch_tape_u32_k, tape.offsets.u32 = narrow;
        usearch_sequence_index_t index = NULL;
        usearch_sequence_init_ram(
            &(usearch_sequence_init_ram_t) {.index = &index,
                                            .outcome = &outcome,
                                            .metric = usearch_sequence_metric_levenshtein_bytes_k});
        expect(!outcome.message, outcome.message);
        usearch_sequence_insert(&(usearch_sequence_insert_t) {.index = index,
                                                              .executor = executor,
                                                              .outcome = &outcome,
                                                              .count = 4,
                                                              .keys = {keys, sizeof(usearch_key_t)},
                                                              .sequences = tape});
        expect(!outcome.message, outcome.message);
        usearch_matches_view_t matches = {0};
        usearch_sequence_search(&(usearch_sequence_search_t) {.index = index,
                                                              .executor = executor,
                                                              .outcome = &outcome,
                                                              .count = 4,
                                                              .queries = tape,
                                                              .matches = &matches,
                                                              .arena = arena,
                                                              .wanted = 1});
        expect(!outcome.message, outcome.message);
        for (size_t query = 0; query != 4; ++query)
            expect(value_of(matches.keys[query]) == query && matches.scores[query] == 0,
                   "Either width of offsets finds every stored string");
        usearch_sequence_free(index);
    }
    usearch_tape_view_t unknown = {.data = bytes, .layout = (usearch_tape_layout_t)7, .scalar = usearch_scalar_u8_k};
    unknown.offsets.u32 = narrow;
    usearch_sequence_index_t const index = create_strings(executor);
    usearch_sequence_insert(&(usearch_sequence_insert_t) {.index = index,
                                                          .executor = executor,
                                                          .outcome = &outcome,
                                                          .count = 4,
                                                          .keys = {keys, sizeof(usearch_key_t)},
                                                          .sequences = unknown});
    expect_eq(outcome.status, usearch_status_unsupported_layout_k, "An unknown width of offsets is refused");

    // Text imports as the bytes it is laid out as, at either width
    void const* narrow_buffers[3] = {NULL, narrow, bytes};
    void const* wide_buffers[3] = {NULL, wide, bytes};
    char const* const formats[] = {"z", "u", "Z", "U"};
    for (size_t format = 0; format != 4; ++format) {
        bool const large = format >= 2;
        struct ArrowSchema schema = {.format = formats[format], .release = &release_schema_of_test};
        struct ArrowArray array = {.length = 3,
                                   .offset = 1,
                                   .n_buffers = 3,
                                   .buffers = large ? wide_buffers : narrow_buffers,
                                   .release = &release_array_of_test};
        usearch_tape_view_t tape = {0};
        size_t count = 0;
        expect_eq(usearch_tape_view_from_arrow(&schema, &array, &tape, &count), usearch_status_success_k,
                  "Bytes and text import");
        expect(tape.data == (void const*)bytes && tape.scalar == usearch_scalar_u8_k && count == 3 &&
                   tape.layout == (large ? usearch_tape_u64_k : usearch_tape_u32_k),
               "A tape keeps its column's bytes and width");
        expect(large ? tape.offsets.u64 == wide + 1 : tape.offsets.u32 == narrow + 1,
               "A slice skips its first offsets");
    }

    // A get's strings leave as a column and come back in place
    usearch_tape_view_t got = {0};
    size_t written = 0;
    usearch_sequence_get(&(usearch_sequence_get_t) {.index = index,
                                                    .executor = executor,
                                                    .outcome = &outcome,
                                                    .key = key_of(2),
                                                    .sequences = &got,
                                                    .written = &written,
                                                    .arena = arena});
    expect(!outcome.message, outcome.message);
    bool const wide_outputs = sizeof(size_t) == sizeof(uint64_t);
    expect(got.layout == (wide_outputs ? usearch_tape_u64_k : usearch_tape_u32_k) && written == 1,
           "Outputs carry offsets as wide as size_t");
    struct ArrowSchema schema = {0};
    struct ArrowDeviceArray array = {0};
    expect_eq(usearch_tape_view_to_arrow(got, written, arena, &schema, &array), usearch_status_success_k,
              "Strings export");
    expect(!strcmp(schema.format, wide_outputs ? "Z" : "z") && array.array.n_buffers == 3 &&
               array.array.buffers[1] == (void const*)offsets_of(got),
           "Strings export in place, at their width");
    usearch_tape_view_t viewed = {0};
    size_t viewed_count = 0;
    expect_eq(usearch_tape_view_from_arrow(&schema, &array.array, &viewed, &viewed_count), usearch_status_success_k,
              "Strings import");
    expect(viewed.data == got.data && offsets_of(viewed) == offsets_of(got) && viewed_count == 1,
           "An import views the very tape an export named");
    release_column(&schema, &array);

    // Shapes do the same as lists of points, at the width of their offsets
    float const points[] = {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 2, 2, 2};
    size_t const point_offsets[] = {0, 2, 5};
    usearch_key_t const shape_keys[] = {key_of(0), key_of(1)};
    usearch_shapes_index_t shapes = NULL;
    usearch_shapes_init_ram(&(usearch_shapes_init_ram_t) {.index = &shapes,
                                                          .outcome = &outcome,
                                                          .metric = usearch_shapes_metric_chamfer_k,
                                                          .quantization = usearch_scalar_f32_k});
    expect(!outcome.message, outcome.message);
    usearch_shapes_insert(&(usearch_shapes_insert_t) {.index = shapes,
                                                      .executor = executor,
                                                      .outcome = &outcome,
                                                      .count = 2,
                                                      .keys = {shape_keys, sizeof(usearch_key_t)},
                                                      .shapes = tape_of(points, point_offsets, usearch_scalar_f32_k)});
    expect(!outcome.message, outcome.message);
    usearch_shapes_get(&(usearch_shapes_get_t) {.index = shapes,
                                                .executor = executor,
                                                .outcome = &outcome,
                                                .key = key_of(1),
                                                .shapes = &got,
                                                .outputs_scalar = usearch_scalar_f32_k,
                                                .written = &written,
                                                .arena = arena});
    expect(!outcome.message, outcome.message);
    expect_eq(usearch_tape_view_to_arrow(got, written, arena, &schema, &array), usearch_status_success_k,
              "Shapes export");
    expect(!strcmp(schema.format, wide_outputs ? "+L" : "+l") && !strcmp(schema.children[0]->format, "+w:3") &&
               !strcmp(schema.children[0]->children[0]->format, "f"),
           "Shapes export as lists of points");
    expect(array.array.children[0]->length == 3 && array.array.children[0]->children[0]->length == 9,
           "A list's children count points and scalars");
    expect_eq(usearch_tape_view_from_arrow(&schema, &array.array, &viewed, &viewed_count), usearch_status_success_k,
              "Shapes import");
    expect(viewed.data == got.data && viewed.scalar == usearch_scalar_f32_k && viewed_count == 1 &&
               offsets_of(viewed)[1] - offsets_of(viewed)[0] == 3,
           "An import views the very points an export named");
    release_column(&schema, &array);

    usearch_shapes_free(shapes);
    usearch_sequence_free(index);
    usearch_arena_free(arena);
    usearch_executor_free(executor);
}

void test_arrow_matches(void) {
    printf("Test: Arrow Matches\n");
    usearch_executor_t const executor = create_executor(0);
    usearch_arena_t const arena = create_arena(executor);
    usearch_sequence_index_t const index = create_strings(executor);
    struct ArrowSchema schema = {0};
    struct ArrowDeviceArray array = {0};

    // A consumer may move a child out and release its parent first
    uint8_t* const bytes = bytes_in_arena(index, executor, arena);
    size_t* const tape_offsets = (size_t*)bytes;
    tape_offsets[0] = 0, tape_offsets[1] = 1, tape_offsets[2] = 3;
    usearch_key_t const* const tape_keys = (usearch_key_t const*)(bytes + 64);
    usearch_distance_t const* const tape_scores = (usearch_distance_t const*)(bytes + 128);
    usearch_matches_tape_view_t const ragged = {tape_keys, tape_scores, tape_offsets};
    expect_eq(usearch_matches_tape_view_to_arrow(ragged, 2, arena, &schema, &array), usearch_status_success_k,
              "A tape of matches exports");
    expect(!strcmp(schema.format, "+L") && !strcmp(schema.children[0]->format, "+s") &&
               array.array.children[0]->length == 3,
           "A tape of matches exports as a large list of structs");
    struct ArrowArray moved = *array.array.children[0];
    array.array.children[0]->release = NULL;
    array.array.release(&array.array);
    expect(moved.length == 3 && moved.children[0]->buffers[1] == (void const*)tape_keys,
           "A moved child outlives its released parent");
    moved.release(&moved);
    schema.release(&schema);

    usearch_keys_tape_view_t const links = {tape_keys, tape_offsets};
    expect_eq(usearch_keys_tape_view_to_arrow(links, 2, arena, &schema, &array), usearch_status_success_k,
              "A tape of keys exports");
    expect(!strcmp(schema.format, "+L") && !strcmp(schema.children[0]->format, "w:16") &&
               array.array.children[0]->length == 3,
           "A tape of keys exports as a large list of them");
    release_column(&schema, &array);

    usearch_sequence_free(index);
    usearch_arena_free(arena);
    usearch_executor_free(executor);
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
    if (selected("names")) test_names();
    if (selected("outcome_contract")) test_outcome_contract(64);

    size_t const collection_sizes[] = {11, 512};
    size_t const dimension_counts[] = {83, 256};
    for (size_t size_index = 0; size_index != 2; ++size_index)
        for (size_t dimension_index = 0; dimension_index != 2; ++dimension_index) {
            if (selected("init")) test_init(collection_sizes[size_index], dimension_counts[dimension_index]);
            if (selected("add_and_search"))
                test_add_and_search(collection_sizes[size_index], dimension_counts[dimension_index]);
        }
    if (selected("empty_reads")) test_empty_reads(dimension_counts[0]);
    if (selected("add_progress")) test_add_progress();
    if (selected("absorb")) test_absorb(dimension_counts[0]);
    if (selected("compact")) test_compact(dimension_counts[0]);
    if (selected("sequence_absorb")) test_sequence_absorb();
    if (selected("count_and_get")) test_count_and_get(dimension_counts[0]);
    if (selected("count_and_get")) test_count_and_get(dimension_counts[1]);
    if (selected("arenas")) test_arenas(dimension_counts[0]);
    if (selected("walks_name_their_index")) test_walks_name_their_index(dimension_counts[0]);
    if (selected("sample")) test_sample();
    if (selected("cursor_steps")) test_cursor_steps();
    if (selected("sequence_round_trip")) test_sequence_round_trip();
    if (selected("sequence_shards")) test_sequence_shards();
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
    if (selected("pins_and_fill")) test_pins_and_fill();
    if (selected("deferred_search")) test_deferred_search();
    if (selected("deferred_polling")) test_deferred_polling();
    if (selected("contended_from_another_thread")) test_contended_from_another_thread();
    if (selected("drain_then_mood")) test_drain_then_mood();
    if (selected("free_waits_for_named_call")) test_free_waits_for_named_call();
    if (selected("deferred_holds_arena")) test_deferred_holds_arena();
    // An implementation without Arrow, as the stand-in, answers every exchange unsupported.
    struct ArrowSchema probed_schema = {0};
    struct ArrowDeviceArray probed_array = {0};
    if (usearch_rows_view_to_arrow((usearch_rows_view_t) {0}, 0, NULL, &probed_schema, &probed_array) ==
        usearch_status_unsupported_k)
        printf("Test: Arrow - SKIPPED: this build exchanges no Arrow\n");
    else {
        if (selected("arrow_rows_and_keys")) test_arrow_rows_and_keys();
        if (selected("arrow_refusals")) test_arrow_refusals();
        if (selected("arrow_tapes")) test_arrow_tapes();
        if (selected("arrow_matches")) test_arrow_matches();
    }

    printf("All tests passed!\n");
    return 0;
}
