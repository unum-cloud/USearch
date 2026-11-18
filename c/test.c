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
 *  A failed check prints its location and lets the case go on; a case returns where the next step
 *  would read past a call that failed. On startup @c install_test_signal_handlers arms a handler
 *  that names the case in progress and prints a native back-trace before re-raising, so CI logs
 *  pinpoint the faulting frame instead of stopping at a bare exit code.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include <signal.h> // `signal`, `raise`, `SIGSEGV`
#include <stdio.h>  // `printf`, `snprintf`
#include <stdlib.h> // `malloc`, `free`, `getenv`, `strtoull`, `srand`
#include <string.h> // `memset`, `memcmp`
#include <time.h>   // `time`, for a drawn seed

#include <inttypes.h> // `PRIu64` - the run's lines
#include <math.h>     // `fabsf`, `fabs`
#include <stdbool.h>  // `bool`
#include <stdint.h>   // `uint8_t`, `uint64_t`

#include "usearch/arrow.h"
#include "usearch/dense_index.h"
#include "usearch/geospatial_index.h"
#include "usearch/sequence_index.h"
#include "usearch/shapes_index.h"

/* `dbghelp.h` needs types `windows.h` defines, so a blank line keeps clang-format from sorting it first. */
#if USEARCH_OS_WINDOWS_
#include <winsock2.h>
#include <windows.h>

#include <dbghelp.h> // `CaptureStackBackTrace`, `SymFromAddr`
#include <io.h>      // `_write`
#pragma comment(lib, "Dbghelp.lib")
#else
#include <unistd.h> // `write`, `STDERR_FILENO`
#if __has_include(<execinfo.h>)
#include <execinfo.h> // `backtrace`, `backtrace_symbols_fd`
#endif                // __has_include(<execinfo.h>)
#endif                // USEARCH_OS_WINDOWS_

/*  region Shared C Checks  */
static size_t failed_checks = 0;
static size_t failed_checks_before_case = 0;
static size_t passed_cases = 0;
static size_t failed_cases = 0;
static size_t skipped_cases = 0;
static char const* current_case = NULL;
static char const* program_name = "";
static uint64_t test_seed = 42;
static double test_scale = 1.0;
static bool refuse_skips = false;

/** Fills the settings above from the file's own spelled-out variables and flags. */
static void read_test_settings(int argc, char** argv);

/** SplitMix64's finalizer, a bijection that spreads every input bit over all 64 output bits. */
static inline uint64_t split_mix_bits(uint64_t value) {
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ull;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBull;
    return value ^ (value >> 31);
}

/** Writes @p text to the standard error through a raw @c write, which a signal handler may call. */
static void write_text(char const* text) {
#if USEARCH_OS_WINDOWS_
    (void)!_write(2, text, (unsigned)strlen(text));
#else
    (void)!write(STDERR_FILENO, text, strlen(text));
#endif
}

/** Counts a failed check at @p file : @p line without stopping, so one run lists every mismatch. */
static inline bool expect_at(bool is_met, char const* file, int line, char const* what) {
    if (is_met) return true;
    fflush(stdout);
    char message[512];
    snprintf(message, sizeof message, "  check failed at %s:%d\n    %s\n", file, line, what);
    write_text(message);
    ++failed_checks;
    return false;
}

/** The text of a setting: its @p flag when given, else its environment @p variable , else NULL. */
static inline char const* setting_text(int argc, char** argv, char const* variable, char const* flag) {
    size_t const flag_length = strlen(flag);
    for (int index = 1; index < argc; ++index) {
        if (strcmp(argv[index], flag) == 0)
            return index + 1 < argc && strncmp(argv[index + 1], "--", 2) != 0 ? argv[index + 1] : "1";
        if (strncmp(argv[index], flag, flag_length) == 0 && argv[index][flag_length] == '=')
            return argv[index] + flag_length + 1;
    }
    char const* text = getenv(variable);
    return text && *text ? text : NULL;
}

/** Exits with one line naming a setting that does not parse, so a bad value never runs anything. */
static inline void exit_unparsed(char const* variable, char const* text, char const* expected) {
    char message[512];
    snprintf(message, sizeof message, "%s=\"%s\" does not parse, expected %s\n", variable, text, expected);
    write_text(message);
    exit(1);
}

/** The run's seed: 42 when unset, drawn afresh for @c random , else any unsigned 64-bit value. */
static inline uint64_t read_seed_setting(int argc, char** argv, char const* variable, char const* flag) {
    char const* text = setting_text(argc, argv, variable, flag);
    char* end = NULL;
    if (!text) return 42;
    if (strcmp(text, "random") == 0) return split_mix_bits((uint64_t)time(NULL) ^ (uint64_t)clock());
    uint64_t const seed = (uint64_t)strtoull(text, &end, 10);
    if (*end != '\0' || *text == '-') exit_unparsed(variable, text, "an unsigned integer or random");
    return seed;
}

/** A positive real, @p fallback when unset. */
static inline double read_real_setting(int argc, char** argv, char const* variable, char const* flag, double fallback) {
    char const* text = setting_text(argc, argv, variable, flag);
    char* end = NULL;
    if (!text) return fallback;
    double const value = strtod(text, &end);
    if (*end != '\0' || !(value > 0)) exit_unparsed(variable, text, "a positive number like 0.1 or 10");
    return value;
}

/** A switch, @p fallback when unset: 0, false, off and no turn it off, anything else on. */
static inline bool read_flag_setting(int argc, char** argv, char const* variable, char const* flag, bool fallback) {
    char const* text = setting_text(argc, argv, variable, flag);
    if (!text) return fallback;
    return strcmp(text, "0") != 0 && strcmp(text, "false") != 0 && strcmp(text, "off") != 0 && strcmp(text, "no") != 0;
}

/** Reads the settings and prints the settings line, before the first case. */
static inline void start_test_run(int argc, char** argv) {
    program_name = argc != 0 ? argv[0] : "";
    read_test_settings(argc, argv);
    printf("- Seed: %" PRIu64 ", scale: %g\n", test_seed, test_scale);
}

/** Starts the case @p label. */
static inline void start_test_case(char const* label) {
    current_case = label;
    failed_checks_before_case = failed_checks;
}

/** Ends the case @ref start_test_case began: @p skip_reason when it skipped, NULL when it ran. */
static inline void finish_test_case(char const* skip_reason) {
    size_t const failed = failed_checks - failed_checks_before_case;
    if (skip_reason) {
        ++skipped_cases;
        printf("- %s ... skipped: %s\n", current_case, skip_reason);
    }
    else if (failed == 0) {
        ++passed_cases;
        printf("- %s ... ok\n", current_case);
    }
    else {
        ++failed_cases;
        fflush(stdout);
        char message[512];
        snprintf(message, sizeof message, "- %s ... FAILED: %zu failed check(s)\n", current_case, failed);
        write_text(message);
        snprintf(message, sizeof message, "  rerun: %s --seed %" PRIu64 "\n", program_name, test_seed);
        write_text(message);
    }
    current_case = NULL;
}

/** The exit code: 1 when a case or a check outside any case failed, or nothing ran; 3 when
 *  skips are refused and one happened; else 0. */
static inline int finish_test_run(void) {
    printf("%zu passed, %zu failed, %zu skipped, seed %" PRIu64 "\n", passed_cases, failed_cases, skipped_cases,
           test_seed);
    if (failed_cases != 0 || failed_checks != 0 || passed_cases + skipped_cases == 0) return 1;
    if (skipped_cases != 0 && refuse_skips) return 3;
    return 0;
}
/*  endregion Shared C Checks  */

static void read_test_settings(int argc, char** argv) {
    test_seed = read_seed_setting(argc, argv, "USEARCH_SEED", "--seed");
    test_scale = read_real_setting(argc, argv, "USEARCH_SCALE", "--scale", 1.0);
    refuse_skips = read_flag_setting(argc, argv, "USEARCH_NO_SKIPS", "--no-skips", false);
}

#define expect_(condition) expect_at((condition), __FILE__, __LINE__, #condition)

/** Why a case skips, for the reasons several cases share. */
static char const no_gpu_device_k[] = "no GPU device";
static char const needs_nvme_k[] = "needs usearch_dense_view_nvme";
static char const needs_arrow_k[] = "needs usearch_rows_view_to_arrow";

/** Names the signal and the case in progress, prints a back-trace, then re-raises the signal, so a
 *  crash localizes itself and the shell still sees the true exit status. */
static void test_fatal_signal_handler(int signal_number) {
    char const* name = signal_number == SIGABRT  ? "SIGABRT"
                       : signal_number == SIGILL ? "SIGILL"
                       : signal_number == SIGFPE ? "SIGFPE"
                                                 : "SIGSEGV";
#if defined(SIGBUS)
    if (signal_number == SIGBUS) name = "SIGBUS";
#endif
    write_text("\n*** ");
    write_text(name);
    write_text(" in case '");
    write_text(current_case ? current_case : "none");
    write_text("' - backtrace follows ***\n");
#if USEARCH_OS_WINDOWS_
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
    fflush(stderr);
#elif __has_include(<execinfo.h>)
    enum { backtrace_depth_limit = 64 };
    void* backtrace_frames[backtrace_depth_limit];
    int backtrace_depth = backtrace(backtrace_frames, backtrace_depth_limit);
    backtrace_symbols_fd(backtrace_frames, backtrace_depth, STDERR_FILENO);
#else
    write_text("  <back-trace unavailable on this platform>\n");
#endif
    signal(signal_number, SIG_DFL);
    raise(signal_number);
}

/** Arms @ref test_fatal_signal_handler for every fatal signal but @c SIGTRAP , which a debugger
 *  keeps. */
static void install_test_signal_handlers(void) {
    // Line-buffered, so verdicts survive a crash under redirection; ucrt fast-fails on a zero size.
    fflush(stdout);
    setvbuf(stdout, NULL, _IOLBF, BUFSIZ);
    int const fatal_signals[] = {
        SIGSEGV, SIGABRT, SIGILL, SIGFPE,
#if defined(SIGBUS)
        SIGBUS,
#endif
    };
    for (unsigned signal_index = 0; signal_index < sizeof(fatal_signals) / sizeof(fatal_signals[0]); ++signal_index)
        signal(fatal_signals[signal_index], &test_fatal_signal_handler);
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
    if (!expect_at(data, __FILE__, __LINE__, "Failed to allocate memory")) return NULL;
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
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
    if (!expect_at(index != NULL, __FILE__, __LINE__, "Initialization returned no handle")) return NULL;
    return index;
}

/** A RAM executor of @p threads threads, zero taking the default, every other field derived. */
usearch_executor_t create_executor(size_t const threads) {
    usearch_outcome_t outcome = {0};
    usearch_executor_t executor = NULL;
    usearch_executor_for_ram(
        &(usearch_executor_for_ram_t) {.executor = &executor, .outcome = &outcome, .threads = threads});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
    return executor;
}

/** An arena placed where @p executor computes, or host pages for calls on this thread when it is NULL. */
usearch_arena_t create_arena(usearch_executor_t const executor) {
    usearch_outcome_t outcome = {0};
    usearch_arena_t arena = NULL;
    usearch_arena_init(&(usearch_arena_init_t) {.arena = &arena, .executor = executor, .outcome = &outcome});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
    if (!expect_at(arena != NULL, __FILE__, __LINE__, "Arena initialization returned no handle")) return NULL;
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
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
        for (size_t row = 0; row != count; ++row)
            expect_at(matches.counts[row] == 0 || matches.keys[row].bytes[15] % 2 == 0, __FILE__, __LINE__,
                      "An expensive predicate must narrow the matches");
    }
}

/** Initializes an index twice, checking the outcome at each step, and reserves capacity,
 *  verifying that size, capacity, dimensions, and connectivity describe back as configured. */
void test_init_ram_starts_empty(size_t const collection_size, size_t const dimensions) {

    usearch_outcome_t outcome = {0};
    usearch_dense_free(create_index(dimensions));
    usearch_dense_index_t const index = create_index(dimensions);
    usearch_executor_t const executor = create_executor(0);

    usearch_dense_reserve(&(usearch_dense_reserve_t) {.index = index,
                                                      .executor = executor,
                                                      .outcome = &outcome,
                                                      .capacity = collection_size});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;

    usearch_dense_index_description_t description = {0};
    usearch_dense_describe(&(usearch_dense_describe_t) {.index = index,
                                                        .executor = executor,
                                                        .outcome = &outcome,
                                                        .description = &description});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    expect_at(description.size == 0, __FILE__, __LINE__, "Fresh index must be empty");
    expect_at(description.capacity >= collection_size, __FILE__, __LINE__, "Capacity must cover the reservation");
    expect_at(description.metadata.dimensions == dimensions, __FILE__, __LINE__,
              "Dimensions must read back as configured");
    expect_at(description.metadata.connectivity == 16, __FILE__, __LINE__, "Connectivity must read back as configured");

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
        expect_at(outcome.status == usearch_status_config_mismatch_k, __FILE__, __LINE__,
                  "A malformed geometry is refused at init");
        expect_at(malformed == NULL, __FILE__, __LINE__, "A refused initialization returns no handle");
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
    expect_at(outcome.status == usearch_status_unsupported_k, __FILE__, __LINE__,
              "A storage dtype the build left out is unsupported");
    expect_at(refused == NULL, __FILE__, __LINE__, "A refused initialization returns no handle");
#endif

    usearch_executor_free(executor);
    usearch_dense_free(index);
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
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    expect_at(description.size == 0, __FILE__, __LINE__, "An empty index describes no entries");

    size_t entries = 1;
    usearch_dense_count(&(usearch_dense_count_t) {.index = index,
                                                  .executor = executor,
                                                  .outcome = &outcome,
                                                  .key = key_of(0),
                                                  .entries = &entries});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    expect_at(entries == 0, __FILE__, __LINE__, "An empty index counts no entries");

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
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    expect_at(written == 0 && !vectors.data, __FILE__, __LINE__, "An empty index gets no rows");

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
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    if (!expect_at(matches.counts != NULL, __FILE__, __LINE__, "An empty search still lands its counts")) return;
    if (host_matches) expect_at(matches.counts[0] == 0, __FILE__, __LINE__, "An empty index finds no matches");
    free(query);
    usearch_arena_free(arena);
}

/** An index fresh from init, in RAM and on the first GPU when one is listed, answers every read
 *  with nothing and success. */
void test_empty_reads(size_t const dimensions) {
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
    expect_at(device_index || !devices_count || outcome.status == usearch_status_device_k, __FILE__, __LINE__,
              outcome.message ? outcome.message : "A listed GPU opens an index or answers no device");
    if (device_index) {
        usearch_executor_t device_executor = NULL;
        usearch_executor_for_vram(
            &(usearch_executor_for_vram_t) {.executor = &device_executor, .outcome = &outcome, .vram = vram});
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
        expect_empty_reads(device_index, device_executor, dimensions, false);
        usearch_executor_free(device_executor);
        usearch_dense_free(device_index);
    }
}

/** Inserts a batch and queries every row back, which an exact self-search must answer with the
 *  row's own key. The batch is all-or-nothing, so a short size is a failure even without an
 *  error status. A predicate then narrows the same queries to the keys it admits. */
void test_add_and_search(size_t const collection_size, size_t const dimensions) {

    usearch_outcome_t outcome = {0};
    usearch_dense_index_t const index = create_index(dimensions);
    usearch_executor_t const executor = create_executor(0);
    usearch_dense_reserve(&(usearch_dense_reserve_t) {.index = index,
                                                      .executor = executor,
                                                      .outcome = &outcome,
                                                      .capacity = collection_size});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;

    float* const data = create_vectors(collection_size, dimensions);
    usearch_key_t* const keys = (usearch_key_t*)malloc(collection_size * sizeof(usearch_key_t));
    if (!expect_at(keys, __FILE__, __LINE__, "Failed to allocate keys")) return;
    for (size_t row = 0; row != collection_size; ++row) keys[row] = key_of(row);

    size_t const row_bytes = dimensions * sizeof(float);
    usearch_dense_insert(&(usearch_dense_insert_t) {.index = index,
                                                    .executor = executor,
                                                    .outcome = &outcome,
                                                    .count = collection_size,
                                                    .keys = {keys, sizeof(usearch_key_t)},
                                                    .vectors = {data, row_bytes, dimensions, usearch_scalar_f32_k}});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    usearch_dense_index_description_t description = {0};
    usearch_dense_describe(&(usearch_dense_describe_t) {.index = index,
                                                        .executor = executor,
                                                        .outcome = &outcome,
                                                        .description = &description});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    expect_at(description.size == collection_size, __FILE__, __LINE__,
              "A batch is all-or-nothing, so every row must be published");

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
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    expect_at(matches.columns == 1, __FILE__, __LINE__, "The matches must report the row width the call used");

    size_t recalled = 0;
    for (size_t row = 0; row != collection_size; ++row) {
        expect_at(matches.counts[row] != 0, __FILE__, __LINE__, "Every query must find something in a non-empty index");
        usearch_key_t const expected = key_of(row);
        recalled += memcmp(matches.keys[row].bytes, expected.bytes, sizeof(expected.bytes)) == 0;
    }
    // Approximate search may miss, but a self-query missing itself often means a broken graph.
    expect_at(recalled * 10 >= collection_size * 9, __FILE__, __LINE__, "Self-recall collapsed below 90%");

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
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    for (size_t row = 0; row != collection_size; ++row)
        expect_at(matches.counts[row] == 0 || matches.keys[row].bytes[15] % 2 == 0, __FILE__, __LINE__,
                  "A predicate must narrow the matches");
    expect_at(widest_batch != 0 && widest_batch <= 4, __FILE__, __LINE__,
              "A walk must ask its predicate about one scored quad at a time");
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
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    for (size_t row = 0; row != collection_size; ++row) {
        usearch_key_t const expected = key_of(row);
        expect_at(
            matches.counts[row] == 1 && memcmp(matches.keys[row].bytes, expected.bytes, sizeof(expected.bytes)) == 0,
            __FILE__, __LINE__, "An exact scan must find every self-query itself");
    }

    // A zero `wanted` lands only the counts.
    usearch_dense_search(&(usearch_dense_search_t) {.index = index,
                                                    .executor = executor,
                                                    .outcome = &outcome,
                                                    .count = collection_size,
                                                    .queries = {data, row_bytes, dimensions, usearch_scalar_f32_k},
                                                    .matches = &matches,
                                                    .arena = arena});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    for (size_t row = 0; row != collection_size; ++row)
        expect_at(matches.counts[row] == 0, __FILE__, __LINE__, "A zero `wanted` lands zero counts");

    usearch_arena_free(arena);
    free(keys);
    free(data);
    usearch_executor_free(executor);
    usearch_dense_free(index);
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

    // A handler asking for four reports hears four, the batch split into nearly equal slices.
    enum { rows_count = 101, dimensions = 16, steps = 4 };
    usearch_outcome_t outcome = {0};
    usearch_dense_index_t const index = create_index(dimensions);
    usearch_executor_t const executor = create_executor(0);
    usearch_dense_reserve(
        &(usearch_dense_reserve_t) {.index = index, .executor = executor, .outcome = &outcome, .capacity = rows_count});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;

    float* const data = create_vectors(rows_count, dimensions);
    usearch_key_t* const keys = (usearch_key_t*)malloc(rows_count * sizeof(usearch_key_t));
    if (!expect_at(keys, __FILE__, __LINE__, "Failed to allocate keys")) return;
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
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    expect_at(reports.ascending && reports.calls == steps, __FILE__, __LINE__,
              "Every step must be reported once, in order");
    expect_at(reports.completed == rows_count && reports.total == rows_count, __FILE__, __LINE__,
              "The last report must cover the batch");

    usearch_dense_index_description_t description = {0};
    usearch_dense_describe(&(usearch_dense_describe_t) {.index = index,
                                                        .executor = executor,
                                                        .outcome = &outcome,
                                                        .description = &description});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    expect_at(description.size == rows_count, __FILE__, __LINE__, "A reported batch must publish every row");

    free(keys);
    free(data);
    usearch_executor_free(executor);
    usearch_dense_free(index);
}

/** A RAM index bounded at a quarter of its rows holds four shards, and reshards in place, as its
 *  own only source and target, under a bound of half its rows into two through a plan whose every
 *  call is cancelled until its rounds end, each call naming the phase it stopped in: every key
 *  still resolves to one entry, and every entry stays. A placement of another residency, one naming
 *  memory domains, a source named twice, or a class ceiling resolving to another variant than the
 *  handle's is refused before any entry moves, and one resolving to the handle's own passes. */
void test_reshard(size_t const dimensions) {

    enum { rows_count = 1200, quarter = rows_count / 4 };
    usearch_outcome_t outcome = {0};
    usearch_executor_t executor = NULL;
    usearch_executor_for_ram(&(usearch_executor_for_ram_t) {.executor = &executor, .outcome = &outcome, .threads = 2});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    float* const data = create_vectors(rows_count, dimensions);
    usearch_key_t* const keys = (usearch_key_t*)malloc(rows_count * sizeof(usearch_key_t));
    if (!expect_at(keys, __FILE__, __LINE__, "Failed to allocate keys")) return;
    for (size_t row = 0; row != rows_count; ++row) keys[row] = key_of(row);

    usearch_dense_index_t index = NULL;
    usearch_dense_init_ram(&(usearch_dense_init_ram_t) {.index = &index,
                                                        .outcome = &outcome,
                                                        .metric = usearch_dense_metric_angular_k,
                                                        .quantization = usearch_scalar_bf16_k,
                                                        .dimensions = dimensions,
                                                        .connectivity = 16,
                                                        .ram = {.shard_capacity = quarter}});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    // The bounded index takes every entry, one generation of shards per quarter.
    usearch_dense_reserve(
        &(usearch_dense_reserve_t) {.index = index, .executor = executor, .outcome = &outcome, .capacity = rows_count});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    usearch_dense_insert(
        &(usearch_dense_insert_t) {.index = index,
                                   .executor = executor,
                                   .outcome = &outcome,
                                   .count = rows_count,
                                   .keys = {keys, sizeof(usearch_key_t)},
                                   .vectors = {data, dimensions * sizeof(float), dimensions, usearch_scalar_f32_k}});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;

    usearch_dense_reshard(&(usearch_dense_reshard_t) {.sources_count = 1,
                                                      .sources = &index,
                                                      .targets_count = 1,
                                                      .targets = &index,
                                                      .executor = executor,
                                                      .outcome = &outcome,
                                                      .vram = &(usearch_vram_options_t) {0}});
    expect_at(outcome.status == usearch_status_config_mismatch_k, __FILE__, __LINE__,
              "A VRAM placement names nothing a RAM index holds");
    int32_t const domain = 0;
    usearch_dense_reshard(&(usearch_dense_reshard_t) {
        .sources_count = 1,
        .sources = &index,
        .targets_count = 1,
        .targets = &index,
        .executor = executor,
        .outcome = &outcome,
        .ram = &(usearch_ram_options_t) {.memory_domains = &domain, .memory_domains_count = 1}});
    expect_at(outcome.status == usearch_status_unsupported_k, __FILE__, __LINE__,
              "Resharding places no shard on a named domain yet");
    usearch_dense_index_t const twice[2] = {index, index};
    usearch_dense_reshard(&(usearch_dense_reshard_t) {.sources_count = 2,
                                                      .sources = twice,
                                                      .targets_count = 1,
                                                      .targets = &index,
                                                      .executor = executor,
                                                      .outcome = &outcome});
    expect_at(outcome.status == usearch_status_invalid_argument_k, __FILE__, __LINE__,
              "A source named twice must be refused");

    usearch_dense_reshard_plan_t plan = NULL;
    usearch_refine_report_t report = {0};
    size_t calls = 0;
    do {
        usearch_phase_progress_t phase = {0};
        usearch_dense_reshard(
            &(usearch_dense_reshard_t) {.sources_count = 1,
                                        .sources = &index,
                                        .targets_count = 1,
                                        .targets = &index,
                                        .executor = executor,
                                        .outcome = &outcome,
                                        .ram = &(usearch_ram_options_t) {.shard_capacity = rows_count / 2},
                                        .report = &report,
                                        .progress = {cancel_progress},
                                        .plan = &plan,
                                        .phase = &phase});
        bool const cancelled = outcome.status == usearch_status_cancelled_k;
        if (!expect_at(cancelled || !outcome.message, __FILE__, __LINE__, outcome.message)) return;
        expect_at(plan != NULL && phase.phase != usearch_phase_none_k, __FILE__, __LINE__,
                  "Every call through a plan must keep it live and name a phase");
        if (!cancelled) break;
        expect_at(phase.phase != usearch_phase_done_k && phase.completed <= phase.total, __FILE__, __LINE__,
                  "A stopped round must stand in one of its phases");
    } while (++calls != rows_count);
    usearch_dense_reshard_plan_finish(&(usearch_dense_reshard_plan_finish_t) {.executor = executor,
                                                                              .outcome = &outcome,
                                                                              .plan = plan,
                                                                              .report = &report});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    expect_at(report.reachability.reachable == report.reachability.audited, __FILE__, __LINE__,
              "Every entry must stay reachable after resharding");
    usearch_dense_index_description_t description = {0};
    usearch_dense_describe(&(usearch_dense_describe_t) {.index = index,
                                                        .executor = executor,
                                                        .outcome = &outcome,
                                                        .description = &description});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    expect_at(description.size == rows_count, __FILE__, __LINE__, "Resharding keeps every entry");
    for (size_t row = 0; row != rows_count; ++row) {
        size_t entries = 0;
        usearch_dense_count(&(usearch_dense_count_t) {.index = index,
                                                      .executor = executor,
                                                      .outcome = &outcome,
                                                      .key = keys[row],
                                                      .entries = &entries});
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
        expect_at(entries == 1, __FILE__, __LINE__, "Every key must resolve to one row after resharding in place");
    }

    // A class ceiling resolving to the handle's own variant passes, and one resolving to another
    // is refused; a monoversioned build compiles no class under the serial floor.
    usearch_dense_reshard(
        &(usearch_dense_reshard_t) {.sources_count = 1,
                                    .sources = &index,
                                    .targets_count = 1,
                                    .targets = &index,
                                    .executor = executor,
                                    .outcome = &outcome,
                                    .ram = &(usearch_ram_options_t) {.cpu_class_ceiling = description.cpu_class}});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    if (description.cpu_class != usearch_cpu_class_serial_k) {
        usearch_dense_reshard(&(usearch_dense_reshard_t) {
            .sources_count = 1,
            .sources = &index,
            .targets_count = 1,
            .targets = &index,
            .executor = executor,
            .outcome = &outcome,
            .ram = &(usearch_ram_options_t) {.cpu_class_ceiling = usearch_cpu_class_serial_k}});
        usearch_status_t const refused = description.cpu_class == usearch_cpu_class_monoversioned_k
                                             ? usearch_status_unsupported_k
                                             : usearch_status_config_mismatch_k;
        expect_at(outcome.status == refused, __FILE__, __LINE__,
                  "A ceiling resolving to another variant than the handle's must be refused");
    }
    usearch_dense_describe(&(usearch_dense_describe_t) {.index = index,
                                                        .executor = executor,
                                                        .outcome = &outcome,
                                                        .description = &description});
    expect_at(description.size == rows_count, __FILE__, __LINE__, "A reshard under a ceiling keeps every entry");

    free(keys);
    free(data);
    usearch_executor_free(executor);
    usearch_dense_free(index);
}

/** The first of four indexes over the quarters of one dataset gathers the rest through reshards
 *  naming it the one target: a handler cancelling the closing round of a one-shot reshard still
 *  leaves both other sources empty, the call answering @c usearch_status_cancelled_k, and the last
 *  source lands through a reshard plan, whose calls are each cancelled until the round ends, each
 *  naming the phase it stopped in, while the target refuses insertions and the round so far never
 *  shrinks. Every emptied source stays a live handle the caller frees, the sizes add up, the
 *  reported audit matches a health call, and every key resolves to one reachable row. */
void test_reshard_gather(size_t const dimensions) {

    enum { indexes_count = 4, rows_per_index = 300, rows_count = indexes_count * rows_per_index };
    usearch_outcome_t outcome = {0};
    // Two threads, as the C++ suites drive: every key's count is a fan-out of its own.
    usearch_executor_t executor = NULL;
    usearch_executor_for_ram(&(usearch_executor_for_ram_t) {.executor = &executor, .outcome = &outcome, .threads = 2});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    float* const data = create_vectors(rows_count, dimensions);
    usearch_key_t* const keys = (usearch_key_t*)malloc(rows_count * sizeof(usearch_key_t));
    if (!expect_at(keys, __FILE__, __LINE__, "Failed to allocate keys")) return;
    for (size_t row = 0; row != rows_count; ++row) keys[row] = key_of(row);
    usearch_dense_index_t indexes[indexes_count];
    for (size_t which = 0; which != indexes_count; ++which) {
        indexes[which] = create_index(dimensions);
        usearch_dense_reserve(&(usearch_dense_reserve_t) {.index = indexes[which],
                                                          .executor = executor,
                                                          .outcome = &outcome,
                                                          .capacity = rows_per_index});
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
        usearch_dense_insert(
            &(usearch_dense_insert_t) {.index = indexes[which],
                                       .executor = executor,
                                       .outcome = &outcome,
                                       .count = rows_per_index,
                                       .keys = {keys + which * rows_per_index, sizeof(usearch_key_t)},
                                       .vectors = {data + which * rows_per_index * dimensions,
                                                   dimensions * sizeof(float), dimensions, usearch_scalar_f32_k}});
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    }

    usearch_dense_reshard(&(usearch_dense_reshard_t) {.sources_count = indexes_count - 1,
                                                      .sources = indexes,
                                                      .targets_count = 1,
                                                      .targets = indexes,
                                                      .executor = executor,
                                                      .outcome = &outcome,
                                                      .progress = {cancel_progress}});
    expect_at(outcome.status == usearch_status_cancelled_k, __FILE__, __LINE__,
              "A cancel must stop only the closing round");
    size_t sizes[indexes_count] = {0};
    for (size_t which = 0; which != indexes_count; ++which) {
        usearch_dense_index_description_t described = {0};
        usearch_dense_describe(&(usearch_dense_describe_t) {.index = indexes[which],
                                                            .executor = executor,
                                                            .outcome = &outcome,
                                                            .description = &described});
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
        sizes[which] = described.size;
    }
    expect_at(sizes[0] == 3 * rows_per_index && sizes[1] == 0 && sizes[2] == 0 && sizes[3] == rows_per_index, __FILE__,
              __LINE__, "Every other source of the call must end empty past the cancel, and no other index");

    // Every call cancels at its first barrier, so the round ends over several calls, each naming
    // where it stopped and reporting the round so far.
    usearch_dense_index_t const last_sources[2] = {indexes[0], indexes[3]};
    usearch_dense_reshard_plan_t plan = NULL;
    progress_reports_t reports = {0, 0, 0, true};
    usearch_refine_report_t running = {0};
    size_t calls = 0;
    do {
        usearch_refine_report_t pulled = {0};
        usearch_phase_progress_t phase = {0};
        usearch_dense_reshard(&(usearch_dense_reshard_t) {.sources_count = 2,
                                                          .sources = last_sources,
                                                          .targets_count = 1,
                                                          .targets = indexes,
                                                          .executor = executor,
                                                          .outcome = &outcome,
                                                          .report = &pulled,
                                                          .progress = {record_and_cancel, &reports},
                                                          .plan = &plan,
                                                          .phase = &phase});
        bool const cancelled = outcome.status == usearch_status_cancelled_k;
        if (!expect_at(cancelled || !outcome.message, __FILE__, __LINE__, outcome.message)) return;
        if (!expect_at(plan != NULL, __FILE__, __LINE__, "The first call must open the plan")) return;
        expect_at(phase.phase != usearch_phase_none_k, __FILE__, __LINE__,
                  "Every call through a plan must name a phase");
        expect_at(pulled.changed <= rows_count && pulled.reselected >= running.reselected, __FILE__, __LINE__,
                  "The round so far must never shrink, nor change more rows than the index holds");
        running = pulled;
        if (!cancelled) break;
        expect_at(phase.phase != usearch_phase_done_k && phase.completed <= phase.total, __FILE__, __LINE__,
                  "A stopped round must stand in one of its phases");
        if (++calls != 1) continue;
        usearch_key_t const late_key = key_of(rows_count);
        usearch_dense_insert(&(usearch_dense_insert_t) {
            .index = indexes[0],
            .executor = executor,
            .outcome = &outcome,
            .count = 1,
            .keys = {&late_key, sizeof(usearch_key_t)},
            .vectors = {data, dimensions * sizeof(float), dimensions, usearch_scalar_f32_k}});
        expect_at(outcome.status == usearch_status_contended_k, __FILE__, __LINE__,
                  "A live plan must hold its index closed");
    } while (calls != rows_count);
    expect_at(calls >= 1 && outcome.status == usearch_status_success_k, __FILE__, __LINE__,
              "The round must end over several calls");
    expect_at(reports.calls > 0 && reports.completed <= reports.total && reports.total >= rows_per_index, __FILE__,
              __LINE__, "The handler must hear the round's phases over every source row");

    usearch_refine_report_t finished = {0};
    usearch_dense_reshard_plan_finish(&(usearch_dense_reshard_plan_finish_t) {.executor = executor,
                                                                              .outcome = &outcome,
                                                                              .plan = plan,
                                                                              .report = &finished});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    expect_at(finished.changed >= running.changed && finished.changed > 0, __FILE__, __LINE__,
              "The plan must report at least what its calls reported");

    usearch_dense_index_description_t description = {0};
    usearch_dense_describe(&(usearch_dense_describe_t) {.index = indexes[0],
                                                        .executor = executor,
                                                        .outcome = &outcome,
                                                        .description = &description});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    expect_at(description.size == rows_count, __FILE__, __LINE__, "The gathered sizes must add up");
    for (size_t row = 0; row != rows_count; ++row) {
        size_t entries = 0;
        usearch_dense_count(&(usearch_dense_count_t) {.index = indexes[0],
                                                      .executor = executor,
                                                      .outcome = &outcome,
                                                      .key = keys[row],
                                                      .entries = &entries});
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
        expect_at(entries == 1, __FILE__, __LINE__, "Every key must resolve to one gathered row");
    }
    usearch_reachability_t reachability = {0};
    usearch_dense_health(&(usearch_dense_health_t) {.index = indexes[0],
                                                    .executor = executor,
                                                    .outcome = &outcome,
                                                    .reachability = &reachability});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    expect_at(reachability.reachable == rows_count, __FILE__, __LINE__, "Every gathered row must stay reachable");
    expect_at(finished.reachability.audited == reachability.audited &&
                  finished.reachability.reachable == reachability.reachable,
              __FILE__, __LINE__, "The closing health check must match a health call right after");

    free(keys);
    free(data);
    usearch_executor_free(executor);
    for (size_t which = 0; which != indexes_count; ++which) usearch_dense_free(indexes[which]);
}

/** A fresh index holding the first @p rows_count rows of @p data, inserted by @p executor: on one
 *  thread, equal calls build equal graphs, so their rounds compare field by field. */
static usearch_dense_index_t create_index_holding(float const* data, size_t const rows_count, size_t const dimensions,
                                                  usearch_executor_t const executor) {
    usearch_outcome_t outcome = {0};
    usearch_dense_index_t const index = create_index(dimensions);
    usearch_key_t* const keys = (usearch_key_t*)malloc(rows_count * sizeof(usearch_key_t));
    if (!expect_at(keys, __FILE__, __LINE__, "Failed to allocate keys")) return NULL;
    for (size_t row = 0; row != rows_count; ++row) keys[row] = key_of(row);
    usearch_dense_reserve(
        &(usearch_dense_reserve_t) {.index = index, .executor = executor, .outcome = &outcome, .capacity = rows_count});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
    usearch_dense_insert(
        &(usearch_dense_insert_t) {.index = index,
                                   .executor = executor,
                                   .outcome = &outcome,
                                   .count = rows_count,
                                   .keys = {keys, sizeof(usearch_key_t)},
                                   .vectors = {data, dimensions * sizeof(float), dimensions, usearch_scalar_f32_k}});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
    free(keys);
    return index;
}

/** Whether two rounds over equal graphs opened, gathered and re-stitched alike. */
static bool same_round(usearch_refine_report_t const* first, usearch_refine_report_t const* second) {
    return first->neighborhoods == second->neighborhoods && first->widest_neighborhood == second->widest_neighborhood &&
           first->added_links == second->added_links && first->dropped_links == second->dropped_links &&
           first->changed == second->changed &&
           memcmp(&first->opening, &second->opening, sizeof(usearch_reachability_t)) == 0;
}

/** Rounds through a plan match the one-shot round on graphs one thread built. A round without a
 *  plan names the closing phase, a round run to its end through a plan reports what a round without
 *  one does, and a second round through the same plan opens where the first closed, from the same
 *  root. A plan whose every call cancels at its first barrier stays live and keeps its index
 *  closed, one of its calls stops in the gather, and its last call reaches the uninterrupted
 *  round's neighborhoods, links and opening health check. A plan finished while stopped in the
 *  gather reports a stopped verdict, and its index saves, loads and finds entries. */
void test_refine_plan(size_t const dimensions, size_t const rows_count) {

    usearch_outcome_t outcome = {0};
    usearch_executor_t const executor = create_executor(1);
    float* const data = create_vectors(rows_count, dimensions);
    size_t const row_bytes = dimensions * sizeof(float);

    usearch_dense_index_t const one_shot = create_index_holding(data, rows_count, dimensions, executor);
    usearch_refine_report_t reference = {0};
    usearch_phase_progress_t unplanned = {usearch_phase_gather_k, 1, 1, 1};
    usearch_dense_refine(&(usearch_dense_refine_t) {.index = one_shot,
                                                    .executor = executor,
                                                    .outcome = &outcome,
                                                    .report = &reference,
                                                    .phase = &unplanned});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    expect_at(unplanned.phase == usearch_phase_done_k, __FILE__, __LINE__,
              "A call without a plan must name the closing phase");
    usearch_dense_free(one_shot);

    usearch_dense_index_t const planned = create_index_holding(data, rows_count, dimensions, executor);
    usearch_dense_refine_plan_t plan = NULL;
    usearch_refine_report_t first = {0}, second = {0};
    usearch_dense_refine(&(usearch_dense_refine_t) {.index = planned,
                                                    .executor = executor,
                                                    .outcome = &outcome,
                                                    .report = &first,
                                                    .plan = &plan});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    if (!expect_at(plan != NULL, __FILE__, __LINE__, "A NULL pointee must open a plan")) return;
    expect_at(same_round(&reference, &first), __FILE__, __LINE__, "A plan's round must match the round without one");
    usearch_dense_refine(&(usearch_dense_refine_t) {.index = planned,
                                                    .executor = executor,
                                                    .outcome = &outcome,
                                                    .report = &second,
                                                    .plan = &plan});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    expect_at(memcmp(&second.opening, &first.reachability, sizeof(usearch_reachability_t)) == 0, __FILE__, __LINE__,
              "The second round must open where the first closed, from the same root");
    usearch_dense_refine_plan_finish(&(usearch_dense_refine_plan_finish_t) {.index = planned,
                                                                            .executor = executor,
                                                                            .outcome = &outcome,
                                                                            .plan = plan});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    usearch_dense_free(planned);

    usearch_dense_index_t const resumed_index = create_index_holding(data, rows_count, dimensions, executor);
    usearch_refine_report_t resumed = {0};
    bool gather_stopped = false;
    size_t calls = 0;
    plan = NULL;
    do {
        usearch_phase_progress_t phase = {0};
        usearch_dense_refine(&(usearch_dense_refine_t) {.index = resumed_index,
                                                        .executor = executor,
                                                        .outcome = &outcome,
                                                        .report = &resumed,
                                                        .progress = {cancel_progress},
                                                        .plan = &plan,
                                                        .phase = &phase});
        ++calls;
        expect_at(phase.phase != usearch_phase_none_k, __FILE__, __LINE__,
                  "Every call through a plan must name a phase");
        if (outcome.status != usearch_status_cancelled_k) break;
        if (!expect_at(plan != NULL, __FILE__, __LINE__, "A cancel must leave the plan live")) return;
        expect_at(phase.phase != usearch_phase_done_k && phase.completed <= phase.total, __FILE__, __LINE__,
                  "A stopped round must stand in one of its phases");
        gather_stopped = gather_stopped || phase.phase == usearch_phase_gather_k;
        if (calls != 1) continue;
        usearch_key_t const late_key = key_of(rows_count);
        usearch_dense_insert(
            &(usearch_dense_insert_t) {.index = resumed_index,
                                       .executor = executor,
                                       .outcome = &outcome,
                                       .count = 1,
                                       .keys = {&late_key, sizeof(usearch_key_t)},
                                       .vectors = {data, row_bytes, dimensions, usearch_scalar_f32_k}});
        expect_at(outcome.status == usearch_status_contended_k, __FILE__, __LINE__,
                  "A live plan must hold its index closed");
    } while (calls != rows_count);
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    expect_at(calls > 2, __FILE__, __LINE__, "Cancelling at every barrier must take several calls");
    expect_at(gather_stopped, __FILE__, __LINE__, "Cancelling at every barrier must stop in the gather");
    expect_at(resumed.reselected == rows_count && resumed.neighborhoods != 0, __FILE__, __LINE__,
              "A round must re-stitch every entry in neighborhoods");
    expect_at(same_round(&reference, &resumed), __FILE__, __LINE__, "A resumed round must match the uninterrupted one");
    usearch_dense_refine_plan_finish(&(usearch_dense_refine_plan_finish_t) {.index = resumed_index,
                                                                            .executor = executor,
                                                                            .outcome = &outcome,
                                                                            .plan = plan});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    usearch_dense_free(resumed_index);

    usearch_dense_index_t const stopped = create_index_holding(data, rows_count, dimensions, executor);
    usearch_phase_progress_t phase = {0};
    plan = NULL;
    do {
        usearch_dense_refine(&(usearch_dense_refine_t) {.index = stopped,
                                                        .executor = executor,
                                                        .outcome = &outcome,
                                                        .progress = {cancel_progress},
                                                        .plan = &plan,
                                                        .phase = &phase});
        expect_at(outcome.status == usearch_status_cancelled_k, __FILE__, __LINE__,
                  "A round must not end before its gather");
    } while (phase.phase != usearch_phase_gather_k);
    usearch_refine_report_t closing = {0};
    usearch_dense_refine_plan_finish(&(usearch_dense_refine_plan_finish_t) {.index = stopped,
                                                                            .executor = executor,
                                                                            .outcome = &outcome,
                                                                            .plan = plan,
                                                                            .report = &closing});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    expect_at(closing.verdict == usearch_refine_stopped_k, __FILE__, __LINE__,
              "A plan finished mid-round must report it stopped");

    char const path[] = "usearch-test-c-refine-plan.safegraphs";
    usearch_bytes_view_t const path_view = {(uint8_t const*)path, sizeof(path) - 1};
    usearch_dense_save(
        &(usearch_dense_save_t) {.index = stopped, .executor = executor, .outcome = &outcome, .path = path_view});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    usearch_dense_free(stopped);
    usearch_dense_index_t loaded = NULL;
    usearch_dense_load_ram(
        &(usearch_dense_load_ram_t) {.index = &loaded, .executor = executor, .outcome = &outcome, .path = path_view});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    remove(path);
    usearch_arena_t const arena = create_arena(executor);
    usearch_matches_view_t matches = {0};
    enum { queries_count = 64 };
    usearch_dense_search(&(usearch_dense_search_t) {.index = loaded,
                                                    .executor = executor,
                                                    .outcome = &outcome,
                                                    .count = queries_count,
                                                    .queries = {data, row_bytes, dimensions, usearch_scalar_f32_k},
                                                    .matches = &matches,
                                                    .arena = arena,
                                                    .wanted = 1});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    for (size_t row = 0; row != queries_count; ++row)
        expect_at(matches.counts[row] == 1, __FILE__, __LINE__, "Every query must find an entry after a stopped round");

    usearch_arena_free(arena);
    usearch_dense_free(loaded);
    free(data);
    usearch_executor_free(executor);
}

/** An index reshards as the one source into two empty targets through a plan whose every call is
 *  cancelled until the closing rounds end, each call naming the phase it stopped in, while a target
 *  refuses insertions; the plan's finish audits every row, the source is left empty and live, and
 *  every key resolves to one row in one target. */
void test_reshard_cut_plan(size_t const dimensions) {

    enum { rows_count = 1200 };
    usearch_outcome_t outcome = {0};
    usearch_executor_t const executor = create_executor(2);
    float* const data = create_vectors(rows_count, dimensions);
    usearch_dense_index_t const index = create_index_holding(data, rows_count, dimensions, executor);
    usearch_dense_index_t const targets[2] = {create_index(dimensions), create_index(dimensions)};

    usearch_dense_reshard_plan_t plan = NULL;
    usearch_refine_report_t report = {0};
    size_t calls = 0;
    do {
        usearch_phase_progress_t phase = {0};
        usearch_dense_reshard(&(usearch_dense_reshard_t) {.sources_count = 1,
                                                          .sources = &index,
                                                          .targets_count = 2,
                                                          .targets = targets,
                                                          .executor = executor,
                                                          .outcome = &outcome,
                                                          .report = &report,
                                                          .progress = {cancel_progress},
                                                          .plan = &plan,
                                                          .phase = &phase});
        bool const cancelled = outcome.status == usearch_status_cancelled_k;
        if (!expect_at(cancelled || !outcome.message, __FILE__, __LINE__, outcome.message)) return;
        expect_at(plan != NULL && phase.phase != usearch_phase_none_k, __FILE__, __LINE__,
                  "Every call through a plan must keep it live and name a phase");
        if (!cancelled) break;
        expect_at(phase.phase != usearch_phase_done_k && phase.completed <= phase.total, __FILE__, __LINE__,
                  "A stopped round must stand in one of its phases");
        if (++calls != 1) continue;
        usearch_key_t const late_key = key_of(rows_count);
        usearch_dense_insert(&(usearch_dense_insert_t) {
            .index = targets[0],
            .executor = executor,
            .outcome = &outcome,
            .count = 1,
            .keys = {&late_key, sizeof(usearch_key_t)},
            .vectors = {data, dimensions * sizeof(float), dimensions, usearch_scalar_f32_k}});
        expect_at(outcome.status == usearch_status_contended_k, __FILE__, __LINE__,
                  "A live plan must hold its targets closed");
    } while (calls != rows_count);
    usearch_dense_reshard_plan_finish(&(usearch_dense_reshard_plan_finish_t) {.executor = executor,
                                                                              .outcome = &outcome,
                                                                              .plan = plan,
                                                                              .report = &report});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    expect_at(report.reachability.audited == rows_count, __FILE__, __LINE__,
              "The closing health checks must judge every row");

    size_t sizes[3] = {0};
    usearch_dense_index_t const parts[3] = {index, targets[0], targets[1]};
    for (size_t part = 0; part != 3; ++part) {
        usearch_dense_index_description_t description = {0};
        usearch_dense_describe(&(usearch_dense_describe_t) {.index = parts[part],
                                                            .executor = executor,
                                                            .outcome = &outcome,
                                                            .description = &description});
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
        sizes[part] = description.size;
    }
    expect_at(sizes[0] == 0 && sizes[1] + sizes[2] == rows_count, __FILE__, __LINE__,
              "A cut must move every row out of its source");
    for (size_t row = 0; row != rows_count; ++row) {
        size_t found = 0;
        for (size_t part = 1; part != 3; ++part) {
            size_t entries = 0;
            usearch_dense_count(&(usearch_dense_count_t) {.index = parts[part],
                                                          .executor = executor,
                                                          .outcome = &outcome,
                                                          .key = key_of(row),
                                                          .entries = &entries});
            if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
            found += entries;
        }
        expect_at(found == 1, __FILE__, __LINE__, "Every key must resolve to one row in one target");
    }

    for (size_t part = 0; part != 3; ++part) usearch_dense_free(parts[part]);
    free(data);
    usearch_executor_free(executor);
}

/** Draws @p count random reads of 100 to 199 nucleotides, end to end around @p count + 1
 *  @p offsets; the bytes must be @b free-ed afterwards. */
char* create_reads(size_t const count, size_t* offsets) {
    offsets[0] = 0;
    for (size_t read = 0; read != count; ++read) offsets[read + 1] = offsets[read] + 100 + (size_t)rand() % 100;
    char* const bytes = (char*)malloc(offsets[count]);
    if (!expect_at(bytes, __FILE__, __LINE__, "Failed to allocate the reads")) return NULL;
    for (size_t position = 0; position != offsets[count]; ++position) bytes[position] = "ACGT"[rand() % 4];
    return bytes;
}

/** The first of four sequence indexes over the quarters of one set of reads gathers the rest, as
 *  @c test_reshard_gather drives the dense ones: a one-shot reshard cancelled in its closing round
 *  still empties both other sources and answers @c usearch_status_cancelled_k, and the last source
 *  lands through a reshard plan, whose calls each cancel until the round ends, each naming the
 *  phase it stopped in, while the target refuses insertions and the round so far never shrinks.
 *  The finished report covers every read, a one-shot refine after it reaches every read, and every
 *  key resolves to one read. */
void test_sequence_reshard(void) {
    enum { indexes_count = 4, reads_per_index = 60, reads_count = indexes_count * reads_per_index };
    usearch_outcome_t outcome = {0};
    usearch_executor_t executor = NULL;
    usearch_executor_for_ram(&(usearch_executor_for_ram_t) {.executor = &executor, .outcome = &outcome, .threads = 2});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
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
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
        // Eight candidates wire a deliberately weak graph, which the rounds then improve.
        usearch_sequence_insert(&(usearch_sequence_insert_t) {
            .index = indexes[which],
            .executor = executor,
            .outcome = &outcome,
            .count = reads_per_index,
            .keys = {keys + which * reads_per_index, sizeof(usearch_key_t)},
            .sequences = tape_of(bytes, offsets + which * reads_per_index, usearch_scalar_u8_k),
            .expansion_factor = 8});
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    }

    usearch_sequence_reshard(&(usearch_sequence_reshard_t) {.sources_count = indexes_count - 1,
                                                            .sources = indexes,
                                                            .targets_count = 1,
                                                            .targets = indexes,
                                                            .executor = executor,
                                                            .outcome = &outcome,
                                                            .progress = {cancel_progress}});
    expect_at(outcome.status == usearch_status_cancelled_k, __FILE__, __LINE__,
              "A cancel must stop only the closing round");
    size_t sizes[indexes_count] = {0};
    for (size_t which = 0; which != indexes_count; ++which) {
        usearch_sequence_index_description_t described = {0};
        usearch_sequence_describe(&(usearch_sequence_describe_t) {.index = indexes[which],
                                                                  .executor = executor,
                                                                  .outcome = &outcome,
                                                                  .description = &described});
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
        sizes[which] = described.size;
    }
    expect_at(sizes[0] == 3 * reads_per_index && sizes[1] == 0 && sizes[2] == 0 && sizes[3] == reads_per_index,
              __FILE__, __LINE__, "Every other source of the call must end empty past the cancel, and no other index");

    // Every call cancels at its first barrier, so the round ends over several calls, each naming
    // where it stopped and reporting the round so far.
    usearch_sequence_index_t const last_sources[2] = {indexes[0], indexes[3]};
    usearch_sequence_reshard_plan_t plan = NULL;
    progress_reports_t reports = {0, 0, 0, true};
    usearch_refine_report_t running = {0};
    size_t calls = 0;
    do {
        usearch_refine_report_t pulled = {0};
        usearch_phase_progress_t phase = {0};
        usearch_sequence_reshard(&(usearch_sequence_reshard_t) {.sources_count = 2,
                                                                .sources = last_sources,
                                                                .targets_count = 1,
                                                                .targets = indexes,
                                                                .executor = executor,
                                                                .outcome = &outcome,
                                                                .report = &pulled,
                                                                .progress = {record_and_cancel, &reports},
                                                                .plan = &plan,
                                                                .phase = &phase});
        bool const cancelled = outcome.status == usearch_status_cancelled_k;
        if (!expect_at(cancelled || !outcome.message, __FILE__, __LINE__, outcome.message)) return;
        if (!expect_at(plan != NULL, __FILE__, __LINE__, "The first call must open the plan")) return;
        expect_at(phase.phase != usearch_phase_none_k, __FILE__, __LINE__,
                  "Every call through a plan must name a phase");
        expect_at(pulled.changed <= reads_count && pulled.reselected >= running.reselected, __FILE__, __LINE__,
                  "The round so far must never shrink, nor change more reads than the index holds");
        running = pulled;
        if (!cancelled) break;
        expect_at(phase.phase != usearch_phase_done_k && phase.completed <= phase.total, __FILE__, __LINE__,
                  "A stopped round must stand in one of its phases");
        if (++calls != 1) continue;
        usearch_key_t const late_key = key_of(reads_count);
        usearch_sequence_insert(
            &(usearch_sequence_insert_t) {.index = indexes[0],
                                          .executor = executor,
                                          .outcome = &outcome,
                                          .count = 1,
                                          .keys = {&late_key, sizeof(usearch_key_t)},
                                          .sequences = tape_of(bytes, offsets, usearch_scalar_u8_k)});
        expect_at(outcome.status == usearch_status_contended_k, __FILE__, __LINE__,
                  "A live plan must hold its index closed");
    } while (calls != reads_count);
    expect_at(calls >= 1 && outcome.status == usearch_status_success_k, __FILE__, __LINE__,
              "The round must end over several calls");
    expect_at(reports.calls > 0 && reports.completed <= reports.total && reports.total >= reads_per_index, __FILE__,
              __LINE__, "The handler must hear the round's phases over every read");

    usearch_refine_report_t finished = {0};
    usearch_sequence_reshard_plan_finish(&(usearch_sequence_reshard_plan_finish_t) {.executor = executor,
                                                                                    .outcome = &outcome,
                                                                                    .plan = plan,
                                                                                    .report = &finished});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    expect_at(finished.changed >= running.changed && finished.changed > 0, __FILE__, __LINE__,
              "The plan must report what its calls changed");
    expect_at(finished.reachability.audited == reads_count, __FILE__, __LINE__,
              "The closing health check must judge every read");

    // The index reopened: a one-shot refine runs to its end and leaves every read reachable.
    usearch_refine_report_t refined = {0};
    usearch_sequence_refine(&(usearch_sequence_refine_t) {.index = indexes[0],
                                                          .executor = executor,
                                                          .outcome = &outcome,
                                                          .report = &refined});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    expect_at(refined.reachability.audited == reads_count && refined.reachability.reachable == reads_count, __FILE__,
              __LINE__, "Every read must stay reachable");
    for (size_t read = 0; read != reads_count; ++read) {
        usearch_sequence_key_stats_t key_stats = {0};
        usearch_sequence_count(&(usearch_sequence_count_t) {.index = indexes[0],
                                                            .executor = executor,
                                                            .outcome = &outcome,
                                                            .key = keys[read],
                                                            .key_stats = &key_stats});
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
        expect_at(key_stats.entries == 1, __FILE__, __LINE__, "Every key must resolve to one gathered read");
    }

    free(bytes);
    for (size_t which = 0; which != indexes_count; ++which) usearch_sequence_free(indexes[which]);
    usearch_executor_free(executor);
}

/** Reads entries back by key. @c usearch_dense_get sizes its own output, writing every row under
 *  the key in an arena and reporting how many there were. */
void test_count_and_get(size_t const dimensions) {

    usearch_outcome_t outcome = {0};
    usearch_dense_index_t const index = create_index(dimensions);
    usearch_executor_t const executor = create_executor(0);
    usearch_dense_reserve(
        &(usearch_dense_reserve_t) {.index = index, .executor = executor, .outcome = &outcome, .capacity = 2});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;

    // Small integers through every nibble, which bf16, `i4x2` and `u1x8` all hold exactly.
    float* const data = (float*)malloc(2 * dimensions * sizeof(float));
    if (!expect_at(data, __FILE__, __LINE__, "Failed to allocate memory")) return;
    for (size_t index = 0; index != 2 * dimensions; ++index) data[index] = (float)((int)(index % 16) - 8);
    usearch_key_t const keys[2] = {key_of(7), key_of(8)};
    size_t const row_bytes = dimensions * sizeof(float);
    usearch_dense_insert(&(usearch_dense_insert_t) {.index = index,
                                                    .executor = executor,
                                                    .outcome = &outcome,
                                                    .count = 2,
                                                    .keys = {keys, sizeof(usearch_key_t)},
                                                    .vectors = {data, row_bytes, dimensions, usearch_scalar_f32_k}});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;

    size_t entries = 0;
    usearch_dense_count(&(usearch_dense_count_t) {.index = index,
                                                  .executor = executor,
                                                  .outcome = &outcome,
                                                  .key = key_of(7),
                                                  .entries = &entries});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    expect_at(entries == 1, __FILE__, __LINE__, "The key was inserted once");
    usearch_dense_count(&(usearch_dense_count_t) {.index = index,
                                                  .executor = executor,
                                                  .outcome = &outcome,
                                                  .key = key_of(9),
                                                  .entries = &entries});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    expect_at(entries == 0, __FILE__, __LINE__, "The key was never inserted");

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
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    expect_at(written == 1, __FILE__, __LINE__, "Retrieval must report the true entry count");
    expect_at(vectors.stride == row_bytes, __FILE__, __LINE__, "Retrieval must report the pitch of the rows it packed");
    expect_at(vectors.columns == dimensions, __FILE__, __LINE__,
              "Retrieval must report the width of the rows it packed");
    if (!expect_at(vectors.data != NULL, __FILE__, __LINE__, "Retrieval must land the rows in the arena")) return;
    float const* const retrieved = (float const*)vectors.data;
    for (size_t position = 0; position != dimensions; ++position)
        expect_at(retrieved[position] == data[position], __FILE__, __LINE__,
                  "A bf16 index must return small integers exactly");

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
        expect_at(outcome.status == usearch_status_shape_mismatch_k, __FILE__, __LINE__,
                  "A bit row ending inside a byte is refused");
    else {
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
        uint8_t const* const bits = (uint8_t const*)vectors.data;
        for (size_t position = 0; position != dimensions; ++position)
            expect_at(((bits[position / 8] >> (7 - position % 8)) & 1) == (data[position] != 0), __FILE__, __LINE__,
                      "A bit row must hold whether each value is non-zero");
    }
    get_packed.outputs_scalar = usearch_scalar_i4x2_k;
    usearch_dense_get(&get_packed);
    if (dimensions % 2)
        expect_at(outcome.status == usearch_status_shape_mismatch_k, __FILE__, __LINE__,
                  "A nibble row ending inside a byte is refused");
    else {
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
        uint8_t const* const nibbles = (uint8_t const*)vectors.data;
        for (size_t position = 0; position != dimensions; ++position) {
            int const nibble = position % 2 ? nibbles[position / 2] & 0xF : nibbles[position / 2] >> 4;
            expect_at((nibble ^ 8) - 8 == (int)data[position], __FILE__, __LINE__,
                      "A nibble row must hold each value exactly");
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
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    usearch_dense_reserve(
        &(usearch_dense_reserve_t) {.index = narrow, .executor = executor, .outcome = &outcome, .capacity = 2});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    float* const tiny = (float*)malloc(dimensions * sizeof(float));
    double* const tiny_wide = (double*)malloc(dimensions * sizeof(double));
    if (!expect_at(tiny && tiny_wide, __FILE__, __LINE__, "Failed to allocate the tiny rows")) return;
    for (size_t position = 0; position != dimensions; ++position)
        tiny[position] = data[position] * 1e-4f, tiny_wide[position] = data[dimensions + position] * 1e-4;
    usearch_dense_insert(&(usearch_dense_insert_t) {.index = narrow,
                                                    .executor = executor,
                                                    .outcome = &outcome,
                                                    .count = 1,
                                                    .keys = {&keys[0], sizeof(usearch_key_t)},
                                                    .vectors = {tiny, row_bytes, dimensions, usearch_scalar_f32_k}});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    usearch_dense_insert(&(usearch_dense_insert_t) {
        .index = narrow,
        .executor = executor,
        .outcome = &outcome,
        .count = 1,
        .keys = {&keys[1], sizeof(usearch_key_t)},
        .vectors = {tiny_wide, dimensions * sizeof(double), dimensions, usearch_scalar_f64_k}});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    for (size_t row = 0; row != 2; ++row) {
        usearch_dense_get(&(usearch_dense_get_t) {.index = narrow,
                                                  .executor = executor,
                                                  .outcome = &outcome,
                                                  .key = keys[row],
                                                  .vectors = &vectors,
                                                  .written = &written,
                                                  .arena = arena,
                                                  .outputs_scalar = usearch_scalar_f32_k});
        if (!expect_at(!outcome.message && written == 1, __FILE__, __LINE__,
                       outcome.message ? outcome.message : "A stored key gets its one row"))
            return;
        float const* const direction = (float const*)vectors.data;
        double dot = 0, original_norm = 0, direction_norm = 0;
        for (size_t position = 0; position != dimensions; ++position) {
            double const original = data[row * dimensions + position];
            dot += original * direction[position];
            original_norm += original * original, direction_norm += (double)direction[position] * direction[position];
        }
        expect_at(dot / sqrt(original_norm * direction_norm) > 0.99, __FILE__, __LINE__,
                  "A tiny row must keep its direction in e4m3");
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
    expect_at(outcome.status == usearch_status_shape_mismatch_k, __FILE__, __LINE__,
              "Rows of another width than the frozen one are refused");

    char const path[] = "usearch-test-c-view.safegraphs";
    usearch_bytes_view_t const path_view = {(uint8_t const*)path, sizeof(path) - 1};
    usearch_dense_save(
        &(usearch_dense_save_t) {.index = index, .executor = executor, .outcome = &outcome, .path = path_view});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    usearch_dense_index_t view = NULL;
    usearch_dense_view_ram(&(usearch_dense_view_ram_t) {.index = &view, .outcome = &outcome, .path = path_view});
    // An implementation that cannot serve a mapping yet says so, and has no view to hold to the contract.
    if (outcome.status != usearch_status_unsupported_k) {
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
        usearch_dense_insert(
            &(usearch_dense_insert_t) {.index = view,
                                       .executor = executor,
                                       .outcome = &outcome,
                                       .count = 1,
                                       .keys = {keys, sizeof(usearch_key_t)},
                                       .vectors = {data, row_bytes, dimensions, usearch_scalar_f32_k}});
        expect_at(outcome.status == usearch_status_read_only_k, __FILE__, __LINE__,
                  "A view must refuse every mutation");
        usearch_dense_free(view);
    }
    remove(path);

    usearch_arena_free(arena);
    free(data);
    usearch_executor_free(executor);
    usearch_dense_free(index);
}

/** Arenas come from an executor or from none, and every verb lands its outputs in whichever one it
 *  is given: a NULL arena is refused, one arena serves different verbs in turn, and a failed call
 *  leaves its outputs NULL. Each call rewinds the arena, so the views of the call before it are
 *  dead from then on; that is a contract on the caller, never read back here. */
void test_arenas(size_t const dimensions) {

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
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    usearch_dense_insert(&(usearch_dense_insert_t) {.index = index,
                                                    .executor = executor,
                                                    .outcome = &outcome,
                                                    .count = 2,
                                                    .keys = {keys, sizeof(usearch_key_t)},
                                                    .vectors = {data, row_bytes, dimensions, usearch_scalar_f32_k}});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;

    usearch_matches_view_t matches = {0};
    usearch_dense_search_t search = {.index = index,
                                     .executor = executor,
                                     .outcome = &outcome,
                                     .count = 1,
                                     .queries = {data, row_bytes, dimensions, usearch_scalar_f32_k},
                                     .matches = &matches,
                                     .wanted = 2};
    usearch_dense_search(&search);
    expect_at(outcome.status == usearch_status_invalid_argument_k, __FILE__, __LINE__,
              "A NULL arena is refused, never created");
    expect_at(matches.keys == NULL && matches.counts == NULL, __FILE__, __LINE__,
              "A refused call leaves its outputs NULL");

    search.arena = placed;
    usearch_dense_search(&search);
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    expect_at(matches.counts[0] == 2 && memcmp(matches.keys[0].bytes, keys[0].bytes, sizeof(usearch_key_t)) == 0,
              __FILE__, __LINE__, "A row must find itself first");

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
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    expect_at(written == 1 && vectors.data != NULL, __FILE__, __LINE__, "Another verb must reuse the same arena");

    get.outputs_scalar = usearch_scalar_unknown_k;
    usearch_dense_get(&get);
    expect_at(outcome.status == usearch_status_unknown_kind_k, __FILE__, __LINE__,
              "A kind naming no scalar is refused");
    expect_at(vectors.data == NULL && written == 0, __FILE__, __LINE__, "A failed call leaves its outputs NULL");

    // An arena made without an executor serves calls on this thread.
    search.executor = NULL, search.arena = local;
    usearch_dense_search(&search);
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    expect_at(matches.counts[0] == 2, __FILE__, __LINE__, "An arena of host pages serves a call on the calling thread");

    usearch_arena_description_t description = {usearch_residency_unknown_k, 1};
    usearch_arena_describe(
        &(usearch_arena_describe_t) {.arena = local, .outcome = &outcome, .description = &description});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    expect_at(description.residency == usearch_residency_ram_k && description.device == 0, __FILE__, __LINE__,
              "An arena made without an executor is host pages");
    usearch_arena_describe(
        &(usearch_arena_describe_t) {.arena = NULL, .outcome = &outcome, .description = &description});
    expect_at(outcome.status == usearch_status_invalid_argument_k, __FILE__, __LINE__,
              "Describing a NULL arena is refused");

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
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
        expect_at(description.residency == usearch_residency_vram_k && description.device == 0, __FILE__, __LINE__,
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
}

/** A cursor or a stream keeps no hold on its index, so every pull names it, and a pull naming any
 *  other handle is refused before the walk moves. */
void test_walks_name_their_index(size_t const dimensions) {

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
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    usearch_dense_insert(&(usearch_dense_insert_t) {.index = index,
                                                    .executor = executor,
                                                    .outcome = &outcome,
                                                    .count = 2,
                                                    .keys = {keys, sizeof(usearch_key_t)},
                                                    .vectors = {data, row_bytes, dimensions, usearch_scalar_f32_k}});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;

    usearch_dense_cursor_t cursor = NULL;
    usearch_dense_cursor_init(&(usearch_dense_cursor_init_t) {.index = index, .outcome = &outcome, .cursor = &cursor});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
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
    expect_at(outcome.status == usearch_status_invalid_argument_k, __FILE__, __LINE__,
              "A cursor pull must name the index it was opened on");
    expect_at(written == 0 && walked.data == NULL, __FILE__, __LINE__, "A refused pull hands out nothing");
    usearch_dense_cursor_next(&(usearch_dense_cursor_next_t) {.index = index,
                                                              .executor = executor,
                                                              .outcome = &outcome,
                                                              .cursor = cursor,
                                                              .keys = &walked,
                                                              .written = &written,
                                                              .arena = arena,
                                                              .wanted = 4});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    expect_at(written == 2, __FILE__, __LINE__, "A refused pull must leave the walk where it was");
    expect_at(memcmp(walked.data[1].bytes, keys[1].bytes, sizeof(usearch_key_t)) == 0, __FILE__, __LINE__,
              "A pull lands keys in order");
    usearch_dense_cursor_free(cursor);

    usearch_dense_search_stream_t stream = NULL;
    usearch_dense_search_stream_init(
        &(usearch_dense_search_stream_init_t) {.stream = &stream,
                                               .index = index,
                                               .outcome = &outcome,
                                               .query = {data, 0, dimensions, usearch_scalar_f32_k}});
    // An implementation that cannot stream yet says so, and has no stream to hold to the contract.
    if (outcome.status != usearch_status_unsupported_k) {
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
        usearch_matches_view_t pulled = {0};
        usearch_dense_search_stream_next(&(usearch_dense_search_stream_next_t) {.index = other,
                                                                                .executor = executor,
                                                                                .outcome = &outcome,
                                                                                .stream = stream,
                                                                                .matches = &pulled,
                                                                                .arena = arena,
                                                                                .wanted = 2});
        expect_at(outcome.status == usearch_status_invalid_argument_k, __FILE__, __LINE__,
                  "A stream pull must name the index it was opened on");
        usearch_dense_search_stream_next(&(usearch_dense_search_stream_next_t) {.index = index,
                                                                                .executor = executor,
                                                                                .outcome = &outcome,
                                                                                .stream = stream,
                                                                                .matches = &pulled,
                                                                                .arena = arena,
                                                                                .wanted = 2});
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
        expect_at(pulled.counts[0] == 2, __FILE__, __LINE__, "A refused pull must leave the stream where it was");
        usearch_dense_search_stream_free(stream);
    }

    usearch_arena_free(arena);
    free(data);
    usearch_executor_free(executor);
    usearch_dense_free(other);
    usearch_dense_free(index);
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
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
    usearch_key_t keys[neighborhood_rows_k];
    for (size_t row = 0; row != neighborhood_rows_k; ++row) keys[row] = key_of(row == 1 ? 0 : row);
    usearch_dense_reserve(&(usearch_dense_reserve_t) {.index = index,
                                                      .executor = executor,
                                                      .outcome = &outcome,
                                                      .capacity = neighborhood_rows_k});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
    usearch_dense_insert(&(usearch_dense_insert_t) {
        .index = index,
        .executor = executor,
        .outcome = &outcome,
        .count = neighborhood_rows_k,
        .keys = {keys, sizeof(usearch_key_t)},
        .vectors = {rows, neighborhood_dimensions_k * sizeof(float), neighborhood_dimensions_k, usearch_scalar_f32_k}});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
    usearch_dense_refine(&(usearch_dense_refine_t) {.index = index, .executor = executor, .outcome = &outcome});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
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
    expect_at(written <= wanted && written <= neighborhood_rows_k, __FILE__, __LINE__,
              "A sample writes at most the draws wanted");
    expect_at(coordinates.columns == 3 && coordinates.scalar == usearch_scalar_f32_k, __FILE__, __LINE__,
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
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return 0;
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
        expect_at(sampled->ordinals[draw] < get_entry(index, executor, arena, key, 0, row), __FILE__, __LINE__,
                  "A draw names an entry of its key, as get counts them");
        expect_at(!drew(sampled, draw, key, sampled->ordinals[draw]), __FILE__, __LINE__, "An entry is drawn once");
        expect_at(!shown || !drew_key(shown, shown_count, key), __FILE__, __LINE__, "A shown key is never drawn again");
        expect_at(!sampled->parented[draw] || drew_key(sampled, draw, sampled->parents[draw]) ||
                      (shown && drew_key(shown, shown_count, sampled->parents[draw])),
                  __FILE__, __LINE__, "A draw hangs from a shown or an earlier draw");
        for (size_t axis = 0; axis != 3; ++axis)
            expect_at(isfinite(sampled->coordinates[draw * 3 + axis]), __FILE__, __LINE__,
                      "A draw lies at finite coordinates");
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
char const* test_sample(void) {
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
        free(samples);
        usearch_dense_free(index);
        free(rows);
        usearch_arena_free(arena);
        usearch_executor_free(executor);
        return "needs usearch_dense_sample";
    }
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
    expect_at(whole->written == sampled_wanted_k, __FILE__, __LINE__, "A sample of the whole index draws all it wants");
    expect_draws(index, executor, arena, whole, NULL);
    outcome = sample_draws(index, executor, arena, NULL, NULL, 0, sampled_wanted_k, 42, repeated);
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
    expect_at(same_samples(whole, repeated), __FILE__, __LINE__,
              "An equal seed, shown keys and focus must draw alike, bit for bit");
    outcome = sample_draws(index, executor, arena, NULL, NULL, 0, sampled_wanted_k / 3, 42, nested);
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
    for (size_t draw = 0; draw != nested->written; ++draw)
        expect_at(drew(whole, whole->written, nested->keys[draw], nested->ordinals[draw]), __FILE__, __LINE__,
                  "A smaller sample must draw a subset of a larger one, so zoom levels nest");

    // Zooming in shows the nested draws and looks around one key.
    float shown_coordinates[sampled_wanted_k * 3];
    memcpy(shown_coordinates, nested->coordinates, sizeof(shown_coordinates));
    usearch_key_t const focus = key_of(5);
    outcome = sample_draws(index, executor, arena, nested, &focus, 1, sampled_wanted_k, 42, around);
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
    expect_at(around->written > 0, __FILE__, __LINE__, "A sample around a key draws");
    expect_draws(index, executor, arena, around, nested);
    expect_at(!memcmp(shown_coordinates, nested->coordinates, sizeof(shown_coordinates)), __FILE__, __LINE__,
              "Shown points keep theirs");

    // Every key in focus and nothing shown: every entry is drawn, placed by the projection.
    usearch_key_t every_key[neighborhood_rows_k - 1];
    for (size_t key = 0; key != neighborhood_rows_k - 1; ++key) every_key[key] = key_of(key ? key + 1 : 0);
    outcome = sample_draws(index, executor, arena, NULL, every_key, neighborhood_rows_k - 1, neighborhood_rows_k, 7,
                           golden);
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
    expect_at(golden->written == neighborhood_rows_k, __FILE__, __LINE__, "A focus on every key draws every entry");
    expect_draws(index, executor, arena, golden, NULL);
    for (size_t draw = 0; draw != golden->written; ++draw) {
        float row[neighborhood_dimensions_k], projected[3];
        get_entry(index, executor, arena, golden->keys[draw], golden->ordinals[draw], row);
        project(row, 7, projected);
        expect_at(!memcmp(projected, golden->coordinates + draw * 3, sizeof(projected)), __FILE__, __LINE__,
                  "A draw with nothing shown lies where the projection of its row places it, bit for bit");
    }

    free(samples);
    usearch_dense_free(index);
    free(rows);
    usearch_arena_free(arena);
    usearch_executor_free(executor);
    return NULL;
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
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) break;
        for (size_t entry = 0; entry != written; ++entry) {
            uint64_t const value = value_of(keys.data[entry]);
            if (!expect_at(value < neighborhood_rows_k, __FILE__, __LINE__,
                           "A cursor yields only keys the index holds"))
                continue;
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
    usearch_outcome_t outcome = {0};
    usearch_executor_t const executor = create_executor(0);
    usearch_arena_t const arena = create_arena(executor);
    float* const rows = create_vectors(neighborhood_rows_k, neighborhood_dimensions_k);
    usearch_dense_index_t const index = create_neighborhood(rows, executor);

    size_t yielded[neighborhood_rows_k] = {0};
    usearch_dense_cursor_t cursor = NULL;
    usearch_dense_cursor_init(&(usearch_dense_cursor_init_t) {.index = index, .outcome = &outcome, .cursor = &cursor});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    expect_at(pull_to_end(index, executor, arena, cursor, yielded) == neighborhood_rows_k, __FILE__, __LINE__,
              "A plain scan must yield every row");
    for (size_t row = 0; row != neighborhood_rows_k; ++row)
        expect_at(yielded[row] == (row == 0   ? 2u
                                   : row == 1 ? 0u
                                              : 1u),
                  __FILE__, __LINE__, "A plain scan must yield every row once");

    usearch_key_t const start = key_of(0);
    usearch_dense_cursor_init(&(usearch_dense_cursor_init_t) {.index = index,
                                                              .outcome = &outcome,
                                                              .starts_count = 1,
                                                              .starts = &start,
                                                              .cursor = &cursor});
    if (outcome.status != usearch_status_unsupported_k) {
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
        memset(yielded, 0, sizeof(yielded));
        expect_at(pull_to_end(index, executor, arena, cursor, yielded) > 2, __FILE__, __LINE__,
                  "A key's entries must link somewhere");
        expect_at(yielded[0] == 2, __FILE__, __LINE__, "A walk must yield both of the start key's own entries");
        for (size_t row = 1; row != neighborhood_rows_k; ++row)
            expect_at(yielded[row] <= 1, __FILE__, __LINE__, "A walk must yield each linked entry once");

        usearch_key_t const absent = key_of(neighborhood_rows_k);
        usearch_dense_cursor_init(&(usearch_dense_cursor_init_t) {.index = index,
                                                                  .outcome = &outcome,
                                                                  .starts_count = 1,
                                                                  .starts = &absent,
                                                                  .cursor = &cursor});
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
        expect_at(pull_to_end(index, executor, arena, cursor, yielded) == 0, __FILE__, __LINE__,
                  "An absent key must step nowhere");
    }

    usearch_keys_view_t keys = {0};
    size_t written = 0;
    usearch_dense_cursor_next(&(usearch_dense_cursor_next_t) {.index = index,
                                                              .executor = executor,
                                                              .outcome = &outcome,
                                                              .keys = &keys,
                                                              .written = &written,
                                                              .arena = arena,
                                                              .wanted = 1});
    expect_at(outcome.status == usearch_status_invalid_argument_k, __FILE__, __LINE__,
              "A pull from no cursor must be refused");

    usearch_dense_free(index);
    free(rows);
    usearch_arena_free(arena);
    usearch_executor_free(executor);
}

/** Sequences round-trip through a checkpoint: a stored query finds itself, then its one-edit
 *  neighbor, and @c get hands the bytes back, before the save and after the load alike. The overlap
 *  kind refuses to go without window widths. */
void test_sequence_saves_and_loads_a_file(void) {
    usearch_outcome_t outcome = {0};
    usearch_executor_t const executor = create_executor(0);
    usearch_arena_t const arena = create_arena(executor);

    usearch_sequence_index_t refused = NULL;
    usearch_sequence_init_ram(&(usearch_sequence_init_ram_t) {.index = &refused,
                                                              .outcome = &outcome,
                                                              .metric = usearch_sequence_metric_overlap_k});
    expect_at(outcome.status == usearch_status_config_mismatch_k, __FILE__, __LINE__,
              "The overlap kind needs window widths");
    expect_at(refused == NULL, __FILE__, __LINE__, "A refused initialization returns no handle");

    char const bytes[] = "kittensittingmittenflask";
    size_t const offsets[] = {0, 6, 13, 19, 24};
    usearch_key_t const keys[] = {key_of(0), key_of(1), key_of(2), key_of(3)};
    usearch_sequence_index_t index = NULL;
    usearch_sequence_init_ram(&(usearch_sequence_init_ram_t) {.index = &index,
                                                              .outcome = &outcome,
                                                              .metric = usearch_sequence_metric_levenshtein_bytes_k});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    usearch_sequence_insert(&(usearch_sequence_insert_t) {.index = index,
                                                          .executor = executor,
                                                          .outcome = &outcome,
                                                          .count = 4,
                                                          .keys = {keys, sizeof(usearch_key_t)},
                                                          .sequences = tape_of(bytes, offsets, usearch_scalar_u8_k)});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    size_t const falling_offsets[] = {0, 6, 3};
    usearch_sequence_insert(
        &(usearch_sequence_insert_t) {.index = index,
                                      .executor = executor,
                                      .outcome = &outcome,
                                      .count = 2,
                                      .keys = {keys, sizeof(usearch_key_t)},
                                      .sequences = tape_of(bytes, falling_offsets, usearch_scalar_u8_k)});
    expect_at(outcome.status == usearch_status_invalid_argument_k, __FILE__, __LINE__,
              "Falling offsets are refused before any insert");

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
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
        expect_at(matches.counts[0] == 2 && value_of(matches.keys[0]) == 0 && matches.scores[0] == 0, __FILE__,
                  __LINE__, "A stored query finds itself first");
        expect_at(value_of(matches.keys[1]) == 2 && matches.scores[1] == 1, __FILE__, __LINE__,
                  "One substitution away comes second");

        usearch_tape_view_t got = {0};
        size_t written = 0;
        usearch_sequence_get(&(usearch_sequence_get_t) {.index = index,
                                                        .executor = executor,
                                                        .outcome = &outcome,
                                                        .key = key_of(1),
                                                        .sequences = &got,
                                                        .written = &written,
                                                        .arena = arena});
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
        expect_at(written == 1 && offsets_of(got)[1] - offsets_of(got)[0] == 7 &&
                      memcmp((uint8_t const*)got.data + offsets_of(got)[0], "sitting", 7) == 0,
                  __FILE__, __LINE__, "Get hands the stored bytes back");
        if (reloaded) break;

        usearch_sequence_save(
            &(usearch_sequence_save_t) {.index = index, .executor = executor, .outcome = &outcome, .path = path_view});
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
        usearch_sequence_free(index);
        index = NULL;
        usearch_sequence_load_ram(&(usearch_sequence_load_ram_t) {.index = &index,
                                                                  .executor = executor,
                                                                  .outcome = &outcome,
                                                                  .path = path_view});
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    }
    usearch_sequence_index_description_t description = {0};
    usearch_sequence_describe(&(usearch_sequence_describe_t) {.index = index,
                                                              .executor = executor,
                                                              .outcome = &outcome,
                                                              .description = &description});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    expect_at(description.size == 4 && description.metadata.metric == usearch_sequence_metric_levenshtein_bytes_k,
              __FILE__, __LINE__, "A loaded index describes the saved one");
    expect_at(description.load_path == usearch_load_path_none_k, __FILE__, __LINE__,
              "A load into RAM names no device load path");

    // The same checkpoint served from storage answers reads as RAM does, refusing writes and saves.
    usearch_executor_t storage = NULL;
    usearch_executor_for_nvme(&(usearch_executor_for_nvme_t) {.executor = &storage, .outcome = &outcome, .threads = 1});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    usearch_sequence_index_t viewed = NULL;
    usearch_sequence_view_nvme(&(
        usearch_sequence_view_nvme_t) {.index = &viewed, .executor = storage, .outcome = &outcome, .path = path_view});
    if (outcome.status != usearch_status_unsupported_k) {
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
        usearch_matches_view_t viewed_matches = {0};
        usearch_sequence_search(&(usearch_sequence_search_t) {.index = viewed,
                                                              .executor = storage,
                                                              .outcome = &outcome,
                                                              .count = 1,
                                                              .queries = tape_of(bytes, offsets, usearch_scalar_u8_k),
                                                              .matches = &viewed_matches,
                                                              .arena = arena,
                                                              .wanted = 1});
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
        expect_at(
            viewed_matches.counts[0] == 1 && value_of(viewed_matches.keys[0]) == 0 && viewed_matches.scores[0] == 0,
            __FILE__, __LINE__, "A view from storage finds a stored query at itself");
        usearch_sequence_key_stats_t stats = {0};
        usearch_sequence_count(&(usearch_sequence_count_t) {.index = viewed,
                                                            .executor = storage,
                                                            .outcome = &outcome,
                                                            .key = key_of(1),
                                                            .key_stats = &stats});
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
        expect_at(stats.entries == 1 && stats.total_bytes == 7 && stats.largest_bytes == 7, __FILE__, __LINE__,
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
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
        expect_at(viewed_written == 1 && offsets_of(viewed_got)[1] - offsets_of(viewed_got)[0] == 7 &&
                      memcmp((uint8_t const*)viewed_got.data + offsets_of(viewed_got)[0], "sitting", 7) == 0,
                  __FILE__, __LINE__, "A view from storage hands the stored bytes back");
        usearch_sequence_refine(
            &(usearch_sequence_refine_t) {.index = viewed, .executor = storage, .outcome = &outcome});
        expect_at(outcome.status == usearch_status_read_only_k, __FILE__, __LINE__,
                  "A view from storage refines nothing");
        usearch_sequence_cursor_t cursor = NULL;
        usearch_sequence_cursor_init(
            &(usearch_sequence_cursor_init_t) {.index = viewed, .outcome = &outcome, .cursor = &cursor});
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
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
            if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
            if (pull == 0) {
                usearch_key_t const* pulled_keys = cursor_keys.data;
                expect_at(pulled[0] == 3 && value_of(pulled_keys[1]) == 1 && value_of(pulled_keys[2]) == 2, __FILE__,
                          __LINE__, "A cursor over storage reads rows in insertion order");
            }
        }
        expect_at(pulled[1] == 1, __FILE__, __LINE__, "A cursor over storage stops at the last row");
        usearch_sequence_cursor_free(cursor);
        usearch_sequence_insert(
            &(usearch_sequence_insert_t) {.index = viewed,
                                          .executor = storage,
                                          .outcome = &outcome,
                                          .count = 1,
                                          .keys = {keys, sizeof(usearch_key_t)},
                                          .sequences = tape_of(bytes, offsets, usearch_scalar_u8_k)});
        expect_at(outcome.status == usearch_status_read_only_k, __FILE__, __LINE__,
                  "A view from storage refuses every mutation");
        usearch_sequence_save(
            &(usearch_sequence_save_t) {.index = viewed, .executor = storage, .outcome = &outcome, .path = path_view});
        expect_at(outcome.status == usearch_status_unsupported_k, __FILE__, __LINE__,
                  "A view from storage already is its checkpoint");
        usearch_reachability_t reachability = {0};
        usearch_sequence_health(&(usearch_sequence_health_t) {.index = viewed,
                                                              .executor = storage,
                                                              .outcome = &outcome,
                                                              .reachability = &reachability});
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
        expect_at(reachability.audited == 4 && reachability.reachable == 4, __FILE__, __LINE__,
                  "A view from storage audits and reaches every entry");
        usearch_key_t const start = key_of(0);
        usearch_sequence_cursor_t walk = NULL;
        usearch_sequence_cursor_init(&(usearch_sequence_cursor_init_t) {.index = viewed,
                                                                        .outcome = &outcome,
                                                                        .starts_count = 1,
                                                                        .starts = &start,
                                                                        .cursor = &walk});
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
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
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
        expect_at(walked_count > 1 && value_of(walked.data[0]) == 0 && links.offsets[walked_count] > links.offsets[0],
                  __FILE__, __LINE__, "A cursor over storage starts at its start key and hands every entry's links");
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
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
        expect_at(sampled == 4, __FILE__, __LINE__, "A view from storage samples every entry it holds");
        usearch_bytes_view_t monolith = {0};
        usearch_sequence_save_buffer(&(usearch_sequence_save_buffer_t) {.index = viewed,
                                                                        .executor = storage,
                                                                        .outcome = &outcome,
                                                                        .monolith = &monolith,
                                                                        .arena = arena});
        expect_at(outcome.status == usearch_status_unsupported_k, __FILE__, __LINE__,
                  "A view from storage saves into no buffer");
        if (!expect_at(outcome.message != NULL, __FILE__, __LINE__, "A view's refusal to save says why")) return;
        usearch_sequence_free(viewed);
    }
    usearch_executor_free(storage);
    remove(path);

    usearch_sequence_insert(&(usearch_sequence_insert_t) {.index = index,
                                                          .executor = executor,
                                                          .outcome = &outcome,
                                                          .count = 4,
                                                          .keys = {keys, 0},
                                                          .sequences = tape_of(bytes, offsets, usearch_scalar_u8_k)});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    usearch_sequence_key_stats_t repeated = {0};
    usearch_sequence_count(&(usearch_sequence_count_t) {.index = index,
                                                        .executor = executor,
                                                        .outcome = &outcome,
                                                        .key = key_of(0),
                                                        .key_stats = &repeated});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    expect_at(repeated.entries == 5, __FILE__, __LINE__, "A zero key stride repeats the first key for every row");
    usearch_keys_view_t uncursored_keys = {0};
    size_t uncursored = 0;
    usearch_sequence_cursor_next(&(usearch_sequence_cursor_next_t) {.index = index,
                                                                    .executor = executor,
                                                                    .outcome = &outcome,
                                                                    .keys = &uncursored_keys,
                                                                    .written = &uncursored,
                                                                    .arena = arena,
                                                                    .wanted = 1});
    expect_at(outcome.status == usearch_status_invalid_argument_k, __FILE__, __LINE__,
              "A pull without a cursor is refused");

    usearch_sequence_free(index);
    usearch_arena_free(arena);
    usearch_executor_free(executor);
}

/** A sequence RAM index bounded at ten entries a shard takes twenty-five reads in one batch,
 *  opening three generations of shards on its one default domain: every key counts once, every read
 *  finds itself first across the shards, a cursor pulls every key once, and a checkpoint writes a
 *  manifest beside one part per shard that reloads as many shards. */
void test_sequence_shards(void) {
    enum { reads_count = 25, shard_capacity = 10 };
    usearch_outcome_t outcome = {0};
    usearch_executor_t executor = NULL;
    usearch_executor_for_ram(&(usearch_executor_for_ram_t) {.executor = &executor, .outcome = &outcome, .threads = 2});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
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
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    usearch_sequence_insert(&(usearch_sequence_insert_t) {.index = index,
                                                          .executor = executor,
                                                          .outcome = &outcome,
                                                          .count = reads_count,
                                                          .keys = {keys, sizeof(usearch_key_t)},
                                                          .sequences = tape_of(bytes, offsets, usearch_scalar_u8_k)});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    usearch_sequence_index_description_t description = {0};
    usearch_sequence_describe(&(usearch_sequence_describe_t) {.index = index,
                                                              .executor = executor,
                                                              .outcome = &outcome,
                                                              .description = &description});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    expect_at(description.size == reads_count, __FILE__, __LINE__, "Every shard's rows count once");

    for (size_t read = 0; read != reads_count; ++read) {
        usearch_sequence_key_stats_t stats = {0};
        usearch_sequence_count(&(usearch_sequence_count_t) {.index = index,
                                                            .executor = executor,
                                                            .outcome = &outcome,
                                                            .key = keys[read],
                                                            .key_stats = &stats});
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
        expect_at(stats.entries == 1, __FILE__, __LINE__, "Every key resolves to one read across the shards");
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
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    for (size_t read = 0; read != reads_count; ++read)
        expect_at(matches.counts[read] == 1 && value_of(matches.keys[read]) == read && matches.scores[read] == 0,
                  __FILE__, __LINE__, "Every read finds itself first, whichever shard holds it");

    usearch_sequence_cursor_t cursor = NULL;
    usearch_sequence_cursor_init(
        &(usearch_sequence_cursor_init_t) {.index = index, .outcome = &outcome, .cursor = &cursor});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
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
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
        for (size_t entry = 0; entry != written; ++entry) {
            uint64_t const value = value_of(pulled_keys.data[entry]);
            expect_at(value < reads_count && !pulled[value], __FILE__, __LINE__,
                      "A cursor pulls every key once, shard after shard");
            pulled[value] = true;
            ++pulled_count;
        }
    }
    expect_at(pulled_count == reads_count, __FILE__, __LINE__, "A cursor pulls every shard's keys");
    usearch_sequence_cursor_free(cursor);

    char const path[] = "usearch-test-c-sequence-shards.safegraphs";
    usearch_bytes_view_t const path_view = {(uint8_t const*)path, sizeof(path) - 1};
    usearch_sequence_save(
        &(usearch_sequence_save_t) {.index = index, .executor = executor, .outcome = &outcome, .path = path_view});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    usearch_sequence_free(index);
    for (int part = 1; part <= 3; ++part) {
        char part_path[64];
        snprintf(part_path, sizeof(part_path), "usearch-test-c-sequence-shards-%05d-of-00003.safegraphs", part);
        FILE* const part_file = fopen(part_path, "rb");
        if (!expect_at(part_file != NULL, __FILE__, __LINE__,
                       "A save of three shards writes one part per shard beside its manifest"))
            return;
        if (part_file) fclose(part_file);
    }

    index = NULL;
    usearch_sequence_load_ram(
        &(usearch_sequence_load_ram_t) {.index = &index, .executor = executor, .outcome = &outcome, .path = path_view});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    usearch_sequence_index_description_t reloaded = {0};
    usearch_sequence_describe(&(usearch_sequence_describe_t) {.index = index,
                                                              .executor = executor,
                                                              .outcome = &outcome,
                                                              .description = &reloaded});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    expect_at(reloaded.size == reads_count, __FILE__, __LINE__, "The manifest reloads every part's rows");
    expect_at(reloaded.shards_count == 3, __FILE__, __LINE__, "A load opens one shard per part");
    usearch_sequence_search(&(usearch_sequence_search_t) {.index = index,
                                                          .executor = executor,
                                                          .outcome = &outcome,
                                                          .count = reads_count,
                                                          .queries = tape_of(bytes, offsets, usearch_scalar_u8_k),
                                                          .matches = &matches,
                                                          .arena = arena,
                                                          .wanted = 1});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    for (size_t read = 0; read != reads_count; ++read)
        expect_at(matches.counts[read] == 1 && value_of(matches.keys[read]) == read && matches.scores[read] == 0,
                  __FILE__, __LINE__, "Every read finds itself first in the reloaded parts");

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
}

/** A GPU sequence index takes byte and rune Levenshtein, inserts and describes its entries, and
 *  lands a search's matches in its executor's device arena; local alignments, a global one without
 *  weights and the verbs it lacks refuse. An overlap index lands its scores in a host arena on the
 *  scale a RAM one does, and a Needleman-Wunsch one its exact costs over packed codes. Skipped
 *  without a GPU. */
char const* test_sequence_vram(void) {
    usearch_outcome_t outcome = {0};
    size_t devices_count = 0;
    usearch_gpu_devices(&(usearch_gpu_devices_t) {.outcome = &outcome, .devices_count = &devices_count});
    if (!devices_count) return NULL;
    int32_t const devices[1] = {0};
    usearch_vram_options_t const vram = {.devices = devices, .devices_count = 1};

    usearch_sequence_index_t index = NULL;
    usearch_sequence_init_vram(&(usearch_sequence_init_vram_t) {.index = &index,
                                                                .outcome = &outcome,
                                                                .metric = usearch_sequence_metric_levenshtein_bytes_k,
                                                                .vram = vram});
    if (outcome.status == usearch_status_device_k) return no_gpu_device_k;
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;

    usearch_sequence_index_t refused = NULL;
    usearch_sequence_init_vram(&(usearch_sequence_init_vram_t) {.index = &refused,
                                                                .outcome = &outcome,
                                                                .metric = usearch_sequence_metric_needleman_wunsch_k,
                                                                .vram = vram});
    expect_at(outcome.status == usearch_status_config_mismatch_k, __FILE__, __LINE__, "An alignment needs weights");
    usearch_sequence_init_vram(&(usearch_sequence_init_vram_t) {.index = &refused,
                                                                .outcome = &outcome,
                                                                .metric = usearch_sequence_metric_levenshtein_bytes_k,
                                                                .values_per_key = 2,
                                                                .vram = vram});
    expect_at(outcome.status == usearch_status_unsupported_k, __FILE__, __LINE__,
              "A GPU sequence index bounds no key's entries");

    usearch_executor_t executor = NULL;
    usearch_executor_for_vram(
        &(usearch_executor_for_vram_t) {.executor = &executor, .outcome = &outcome, .vram = vram});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;

    char const bytes[] = "kittensittingmittenflask";
    size_t const offsets[] = {0, 6, 13, 19, 24};
    usearch_key_t const keys[] = {key_of(0), key_of(1), key_of(2), key_of(3)};
    usearch_sequence_insert(&(usearch_sequence_insert_t) {.index = index,
                                                          .executor = executor,
                                                          .outcome = &outcome,
                                                          .count = 4,
                                                          .keys = {keys, sizeof(usearch_key_t)},
                                                          .sequences = tape_of(bytes, offsets, usearch_scalar_u8_k)});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
    usearch_sequence_refine(&(usearch_sequence_refine_t) {.index = index, .executor = executor, .outcome = &outcome});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
    usearch_sequence_index_description_t description = {0};
    usearch_sequence_describe(&(usearch_sequence_describe_t) {.index = index,
                                                              .executor = executor,
                                                              .outcome = &outcome,
                                                              .description = &description});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
    expect_at(description.size == 4 && description.metadata.metric == usearch_sequence_metric_levenshtein_bytes_k,
              __FILE__, __LINE__, "A GPU index describes the rows it holds");

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
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
    expect_at(matches.keys && matches.scores && matches.counts, __FILE__, __LINE__,
              "A GPU search lands its matches in the arena");

    // "mitten" holds three of the four windows "kitten" has at three bytes.
    size_t const widths[] = {3};
    usearch_sequence_index_t overlap = NULL;
    usearch_sequence_init_vram(&(usearch_sequence_init_vram_t) {.index = &overlap,
                                                                .outcome = &outcome,
                                                                .metric = usearch_sequence_metric_overlap_k,
                                                                .window_widths = widths,
                                                                .window_widths_count = 1,
                                                                .vram = vram});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
    usearch_sequence_insert(&(usearch_sequence_insert_t) {.index = overlap,
                                                          .executor = executor,
                                                          .outcome = &outcome,
                                                          .count = 4,
                                                          .keys = {keys, sizeof(usearch_key_t)},
                                                          .sequences = tape_of(bytes, offsets, usearch_scalar_u8_k)});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
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
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
    expect_at(overlaps.counts[0] == 2 && value_of(overlaps.keys[0]) == 0 && overlaps.scores[0] == 0, __FILE__, __LINE__,
              "A stored overlap query finds itself first");
    expect_at(value_of(overlaps.keys[1]) == 2 && overlaps.scores[1] == 0.25, __FILE__, __LINE__,
              "Missing a quarter of windows scores 0.25");
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
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
    usearch_sequence_insert(
        &(usearch_sequence_insert_t) {.index = aligned,
                                      .executor = executor,
                                      .outcome = &outcome,
                                      .count = 4,
                                      .keys = {keys, sizeof(usearch_key_t)},
                                      .sequences = tape_of(packed, packed_offsets, usearch_scalar_u8_k)});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
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
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
    expect_at(alignments.counts[0] == 4 && value_of(alignments.keys[0]) == 0 && alignments.scores[0] == 0, __FILE__,
              __LINE__, "A stored alignment query finds itself first");
    expect_at(alignments.scores[1] == 2 && alignments.scores[2] == 2 && alignments.scores[3] == 5, __FILE__, __LINE__,
              "A transversion and a gap cost 2, and two transversions with a transition 5");
    usearch_arena_free(host_arena);
    usearch_sequence_free(aligned);

    char const path[] = "usearch-test-c-sequence-vram.safegraphs";
    usearch_sequence_save(&(usearch_sequence_save_t) {.index = index,
                                                      .executor = executor,
                                                      .outcome = &outcome,
                                                      .path = {(uint8_t const*)path, sizeof(path) - 1}});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
    usearch_sequence_index_t loaded = NULL;
    usearch_sequence_load_ram(&(usearch_sequence_load_ram_t) {.index = &loaded,
                                                              .outcome = &outcome,
                                                              .path = {(uint8_t const*)path, sizeof(path) - 1}});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
    usearch_sequence_index_description_t loaded_description = {0};
    usearch_sequence_describe(
        &(usearch_sequence_describe_t) {.index = loaded, .outcome = &outcome, .description = &loaded_description});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
    expect_at(loaded_description.size == description.size, __FILE__, __LINE__, "A GPU checkpoint loads into RAM whole");
    usearch_sequence_free(loaded);
    remove(path);

    usearch_arena_free(arena);
    usearch_sequence_free(index);
    usearch_executor_free(executor);
    return NULL;
}

/** Shapes round-trip the same way: a stored pose finds itself at zero, and @c get exports its
 *  points back, before the save and after the load alike. A frozen point count refuses a shape of
 *  any other. */
void test_shapes_saves_and_loads_a_file(void) {
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
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;

    size_t const short_offsets[] = {0, 3};
    usearch_shapes_insert(&(usearch_shapes_insert_t) {.index = index,
                                                      .executor = executor,
                                                      .outcome = &outcome,
                                                      .count = 1,
                                                      .keys = {keys, sizeof(usearch_key_t)},
                                                      .shapes = tape_of(points, short_offsets, usearch_scalar_f32_k)});
    expect_at(outcome.status == usearch_status_shape_mismatch_k, __FILE__, __LINE__,
              "A frozen point count refuses any other");
    size_t const falling_offsets[] = {0, 4, 2};
    usearch_shapes_insert(
        &(usearch_shapes_insert_t) {.index = index,
                                    .executor = executor,
                                    .outcome = &outcome,
                                    .count = 2,
                                    .keys = {keys, sizeof(usearch_key_t)},
                                    .shapes = tape_of(points, falling_offsets, usearch_scalar_f32_k)});
    expect_at(outcome.status == usearch_status_invalid_argument_k, __FILE__, __LINE__,
              "Falling offsets are refused before any insert");
    usearch_shapes_insert(&(usearch_shapes_insert_t) {.index = index,
                                                      .executor = executor,
                                                      .outcome = &outcome,
                                                      .count = 3,
                                                      .keys = {keys, sizeof(usearch_key_t)},
                                                      .shapes = tape_of(points, offsets, usearch_scalar_f32_k)});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;

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
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
        expect_at(matches.counts[0] == 1 && value_of(matches.keys[0]) == 0 && fabs(matches.scores[0]) < 1e-5, __FILE__,
                  __LINE__, "A stored pose finds itself at zero");

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
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
        expect_at(written == 1 && offsets_of(got)[1] - offsets_of(got)[0] == 4, __FILE__, __LINE__,
                  "Get hands every point back");
        float const* const exported = (float const*)got.data + offsets_of(got)[0] * 3;
        for (size_t coordinate = 0; coordinate != 12; ++coordinate)
            expect_at(fabsf(exported[coordinate] - points[24 + coordinate]) < 1e-6f, __FILE__, __LINE__,
                      "Get hands the stored pose back");
        if (reloaded) break;

        usearch_shapes_save(
            &(usearch_shapes_save_t) {.index = index, .executor = executor, .outcome = &outcome, .path = path_view});
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
        usearch_shapes_free(index);
        index = NULL;
        usearch_shapes_load_ram(&(
            usearch_shapes_load_ram_t) {.index = &index, .executor = executor, .outcome = &outcome, .path = path_view});
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    }
    usearch_shapes_index_description_t description = {0};
    usearch_shapes_describe(&(usearch_shapes_describe_t) {.index = index,
                                                          .executor = executor,
                                                          .outcome = &outcome,
                                                          .description = &description});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    expect_at(description.size == 3 && description.total_points == 12 && description.metadata.points == 4, __FILE__,
              __LINE__, "A loaded index describes the saved one");
    expect_at(description.load_path == usearch_load_path_none_k, __FILE__, __LINE__,
              "A load into RAM names no device load path");

    // The same checkpoint served from storage answers reads as RAM does, refusing writes and saves.
    usearch_executor_t storage = NULL;
    usearch_executor_for_nvme(&(usearch_executor_for_nvme_t) {.executor = &storage, .outcome = &outcome, .threads = 1});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    usearch_shapes_index_t viewed = NULL;
    usearch_shapes_view_nvme(
        &(usearch_shapes_view_nvme_t) {.index = &viewed, .executor = storage, .outcome = &outcome, .path = path_view});
    if (outcome.status != usearch_status_unsupported_k) {
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
        usearch_matches_view_t viewed_matches = {0};
        usearch_shapes_search(&(usearch_shapes_search_t) {.index = viewed,
                                                          .executor = storage,
                                                          .outcome = &outcome,
                                                          .count = 1,
                                                          .queries = tape_of(points, offsets, usearch_scalar_f32_k),
                                                          .matches = &viewed_matches,
                                                          .arena = arena,
                                                          .wanted = 1});
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
        expect_at(viewed_matches.counts[0] == 1 && value_of(viewed_matches.keys[0]) == 0 &&
                      fabs(viewed_matches.scores[0]) < 1e-5,
                  __FILE__, __LINE__, "A view from storage finds a stored pose at zero");
        usearch_shapes_key_stats_t stats = {0};
        usearch_shapes_count(&(usearch_shapes_count_t) {.index = viewed,
                                                        .executor = storage,
                                                        .outcome = &outcome,
                                                        .key = key_of(2),
                                                        .key_stats = &stats});
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
        expect_at(stats.entries == 1 && stats.total_points == 4 && stats.largest_points == 4, __FILE__, __LINE__,
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
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
        expect_at(viewed_written == 1 && offsets_of(viewed_got)[1] - offsets_of(viewed_got)[0] == 4, __FILE__, __LINE__,
                  "A view from storage hands every point back");
        float const* const viewed_points = (float const*)viewed_got.data + offsets_of(viewed_got)[0] * 3;
        for (size_t coordinate = 0; coordinate != 12; ++coordinate)
            expect_at(fabsf(viewed_points[coordinate] - points[24 + coordinate]) < 1e-6f, __FILE__, __LINE__,
                      "A view from storage hands the stored pose back");
        usearch_shapes_cursor_t cursor = NULL;
        usearch_shapes_cursor_init(
            &(usearch_shapes_cursor_init_t) {.index = viewed, .outcome = &outcome, .cursor = &cursor});
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
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
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
        usearch_key_t const* pulled_keys = cursor_keys.data;
        expect_at(pulled == 3 && value_of(pulled_keys[1]) == 1 && value_of(pulled_keys[2]) == 2, __FILE__, __LINE__,
                  "A cursor over storage reads every pose in insertion order");
        usearch_shapes_cursor_free(cursor);
        usearch_shapes_insert(&(usearch_shapes_insert_t) {.index = viewed,
                                                          .executor = storage,
                                                          .outcome = &outcome,
                                                          .count = 1,
                                                          .keys = {keys, sizeof(usearch_key_t)},
                                                          .shapes = tape_of(points, offsets, usearch_scalar_f32_k)});
        expect_at(outcome.status == usearch_status_read_only_k, __FILE__, __LINE__,
                  "A view from storage refuses every mutation");
        usearch_shapes_save(
            &(usearch_shapes_save_t) {.index = viewed, .executor = storage, .outcome = &outcome, .path = path_view});
        expect_at(outcome.status == usearch_status_unsupported_k, __FILE__, __LINE__,
                  "A view from storage already is its checkpoint");
        usearch_shapes_free(viewed);
    }
    usearch_executor_free(storage);
    remove(path);
    usearch_keys_view_t uncursored_keys = {0};
    size_t uncursored = 0;
    usearch_shapes_cursor_next(&(usearch_shapes_cursor_next_t) {.index = index,
                                                                .executor = executor,
                                                                .outcome = &outcome,
                                                                .keys = &uncursored_keys,
                                                                .written = &uncursored,
                                                                .arena = arena,
                                                                .wanted = 1});
    expect_at(outcome.status == usearch_status_invalid_argument_k, __FILE__, __LINE__,
              "A pull without a cursor is refused");

    usearch_shapes_free(index);
    usearch_arena_free(arena);
    usearch_executor_free(executor);
}

/** A GPU shapes index takes Chamfer and Hausdorff alone, inserts and describes its clouds, and
 *  lands a search's matches in its executor's device arena; the verbs it lacks refuse. Skipped
 *  without a GPU. */
char const* test_shapes_vram(void) {
    usearch_outcome_t outcome = {0};
    size_t devices_count = 0;
    usearch_gpu_devices(&(usearch_gpu_devices_t) {.outcome = &outcome, .devices_count = &devices_count});
    if (!devices_count) return NULL;
    int32_t const devices[1] = {0};
    usearch_vram_options_t const vram = {.devices = devices, .devices_count = 1};

    usearch_shapes_index_t index = NULL;
    usearch_shapes_init_vram(&(usearch_shapes_init_vram_t) {.index = &index,
                                                            .outcome = &outcome,
                                                            .metric = usearch_shapes_metric_hausdorff_k,
                                                            .quantization = usearch_scalar_f16_k,
                                                            .vram = vram});
    if (outcome.status == usearch_status_device_k) return no_gpu_device_k;
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;

    usearch_shapes_index_t refused = NULL;
    usearch_shapes_init_vram(&(usearch_shapes_init_vram_t) {.index = &refused,
                                                            .outcome = &outcome,
                                                            .metric = usearch_shapes_metric_wasserstein_procrustes_k,
                                                            .quantization = usearch_scalar_f16_k,
                                                            .vram = vram});
    expect_at(outcome.status == usearch_status_unsupported_k, __FILE__, __LINE__,
              "Only Chamfer and Hausdorff serve a GPU");
    expect_at(refused == NULL, __FILE__, __LINE__, "A refused initialization returns no handle");

    usearch_executor_t executor = NULL;
    usearch_executor_for_vram(
        &(usearch_executor_for_vram_t) {.executor = &executor, .outcome = &outcome, .vram = vram});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;

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
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
    usearch_shapes_index_description_t description = {0};
    usearch_shapes_describe(&(usearch_shapes_describe_t) {.index = index,
                                                          .executor = executor,
                                                          .outcome = &outcome,
                                                          .description = &description});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
    expect_at(description.size == 3 && description.total_points == 13 &&
                  description.metadata.metric == usearch_shapes_metric_hausdorff_k,
              __FILE__, __LINE__, "A GPU index describes the clouds it holds");

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
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
    expect_at(matches.keys && matches.scores && matches.counts, __FILE__, __LINE__,
              "A GPU search lands its matches in the arena");

    char const path[] = "usearch-test-c-shapes-vram.safegraphs";
    usearch_shapes_save(&(usearch_shapes_save_t) {.index = index,
                                                  .executor = executor,
                                                  .outcome = &outcome,
                                                  .path = {(uint8_t const*)path, sizeof(path) - 1}});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
    usearch_shapes_index_t loaded = NULL;
    usearch_shapes_load_ram(&(usearch_shapes_load_ram_t) {.index = &loaded,
                                                          .outcome = &outcome,
                                                          .path = {(uint8_t const*)path, sizeof(path) - 1}});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
    usearch_shapes_index_description_t loaded_description = {0};
    usearch_shapes_describe(
        &(usearch_shapes_describe_t) {.index = loaded, .outcome = &outcome, .description = &loaded_description});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
    expect_at(
        loaded_description.size == description.size && loaded_description.total_points == description.total_points,
        __FILE__, __LINE__, "A GPU checkpoint loads into RAM whole");
    usearch_shapes_free(loaded);
    remove(path);

    usearch_arena_free(arena);
    usearch_shapes_free(index);
    usearch_executor_free(executor);
    return NULL;
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
void test_geospatial_saves_and_loads_a_file(void) {
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
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    usearch_geospatial_insert(&(usearch_geospatial_insert_t) {.index = index,
                                                              .executor = executor,
                                                              .outcome = &outcome,
                                                              .count = 4,
                                                              .keys = {keys, sizeof(usearch_key_t)},
                                                              .observations = observations});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;

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
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
        expect_at(matches.counts[0] == 3 && value_of(matches.keys[0]) == 0 && matches.scores[0] < 0.01 &&
                      value_of(matches.keys[1]) == 2 && value_of(matches.keys[2]) == 1,
                  __FILE__, __LINE__,
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
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
        expect_at(matches.counts[0] == 2 && value_of(matches.keys[0]) == 0 && value_of(matches.keys[1]) == 2, __FILE__,
                  __LINE__, "The ridge hides the entry behind it");

        usearch_geospatial_observation_t const* got = NULL;
        size_t written = 0;
        usearch_geospatial_get(&(usearch_geospatial_get_t) {.index = index,
                                                            .executor = executor,
                                                            .outcome = &outcome,
                                                            .key = key_of(2),
                                                            .observations = &got,
                                                            .written = &written,
                                                            .arena = arena});
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
        expect_at(written == 1 && fabs(got[0].latitude_degrees - 51.4995) < 1e-6 &&
                      fabs(got[0].longitude_degrees + 0.1) < 1e-6 && fabs(got[0].altitude_metres) < 0.02 &&
                      isnan(got[0].heading_degrees),
                  __FILE__, __LINE__, "Get hands the observation back, standing on the terrain");
        if (reloaded) break;

        usearch_geospatial_save(&(
            usearch_geospatial_save_t) {.index = index, .executor = executor, .outcome = &outcome, .path = path_view});
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
        usearch_geospatial_free(index);
        index = NULL;
        usearch_geospatial_load_ram(&(usearch_geospatial_load_ram_t) {.index = &index,
                                                                      .executor = executor,
                                                                      .outcome = &outcome,
                                                                      .path = path_view,
                                                                      .terrain = terrain});
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    }
    usearch_geospatial_index_description_t description = {0};
    usearch_geospatial_describe(&(usearch_geospatial_describe_t) {.index = index,
                                                                  .executor = executor,
                                                                  .outcome = &outcome,
                                                                  .description = &description});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    expect_at(description.size == 4 && description.metadata.connectivity == 12, __FILE__, __LINE__,
              "A loaded index describes the saved one");
    expect_at(description.load_path == usearch_load_path_none_k, __FILE__, __LINE__,
              "A load into RAM names no device load path");

    // The same checkpoint served from storage judges every relation on the entries its walks read
    // and answers reads and containment as RAM does, refusing writes and saves.
    usearch_executor_t storage = NULL;
    usearch_executor_for_nvme(&(usearch_executor_for_nvme_t) {.executor = &storage, .outcome = &outcome, .threads = 1});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    usearch_geospatial_index_t viewed = NULL;
    usearch_geospatial_view_nvme(&(usearch_geospatial_view_nvme_t) {.index = &viewed,
                                                                    .executor = storage,
                                                                    .outcome = &outcome,
                                                                    .path = path_view,
                                                                    .terrain = terrain});
    if (outcome.status != usearch_status_unsupported_k) {
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
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
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
        expect_at(matches.counts[0] == 3 && value_of(matches.keys[0]) == 0 && matches.scores[0] < 0.01 &&
                      value_of(matches.keys[1]) == 2 && value_of(matches.keys[2]) == 1,
                  __FILE__, __LINE__,
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
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
        expect_at(matches.counts[0] == 2 && value_of(matches.keys[0]) == 0 && value_of(matches.keys[1]) == 2, __FILE__,
                  __LINE__, "A view from storage hides the entry behind the ridge");

        usearch_geospatial_key_stats_t stats = {0};
        usearch_geospatial_count(&(usearch_geospatial_count_t) {.index = viewed,
                                                                .executor = storage,
                                                                .outcome = &outcome,
                                                                .key = key_of(2),
                                                                .key_stats = &stats});
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
        expect_at(stats.entries == 1, __FILE__, __LINE__, "A view from storage counts a key's entries");
        usearch_geospatial_observation_t const* got = NULL;
        size_t written = 0;
        usearch_geospatial_get(&(usearch_geospatial_get_t) {.index = viewed,
                                                            .executor = storage,
                                                            .outcome = &outcome,
                                                            .key = key_of(2),
                                                            .observations = &got,
                                                            .written = &written,
                                                            .arena = arena});
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
        expect_at(written == 1 && fabs(got[0].latitude_degrees - 51.4995) < 1e-6 && fabs(got[0].altitude_metres) < 0.02,
                  __FILE__, __LINE__, "A view from storage hands the observation back");

        usearch_geospatial_cursor_t cursor = NULL;
        usearch_geospatial_cursor_init(
            &(usearch_geospatial_cursor_init_t) {.index = viewed, .outcome = &outcome, .cursor = &cursor});
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
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
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
        usearch_key_t const* pulled_keys = cursor_keys.data;
        expect_at(pulled == 4 && value_of(pulled_keys[3]) == 3, __FILE__, __LINE__,
                  "A cursor over storage reads every observation in insertion order");
        usearch_geospatial_cursor_free(cursor);

        usearch_geospatial_insert(&(usearch_geospatial_insert_t) {.index = viewed,
                                                                  .executor = storage,
                                                                  .outcome = &outcome,
                                                                  .count = 1,
                                                                  .keys = {keys, sizeof(usearch_key_t)},
                                                                  .observations = observations});
        expect_at(outcome.status == usearch_status_read_only_k, __FILE__, __LINE__,
                  "A view from storage refuses every mutation");

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
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
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
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
        for (size_t match = 0; match != nvme_reported; ++match) nvme_keys |= 1u << value_of(balls.keys[match]);
        expect_at(ram_reported == 3 && nvme_reported == 3 && ram_keys == 7 && nvme_keys == 7, __FILE__, __LINE__,
                  "A view from storage reports the containment RAM reports, Paris left out");
        usearch_geospatial_save(&(
            usearch_geospatial_save_t) {.index = viewed, .executor = storage, .outcome = &outcome, .path = path_view});
        expect_at(outcome.status == usearch_status_unsupported_k, __FILE__, __LINE__,
                  "A view from storage already is its checkpoint");
        usearch_geospatial_free(viewed);
    }
    usearch_executor_free(storage);
    remove(path);
    usearch_keys_view_t uncursored_keys = {0};
    size_t uncursored = 0;
    usearch_geospatial_cursor_next(&(usearch_geospatial_cursor_next_t) {.index = index,
                                                                        .executor = executor,
                                                                        .outcome = &outcome,
                                                                        .keys = &uncursored_keys,
                                                                        .written = &uncursored,
                                                                        .arena = arena,
                                                                        .wanted = 1});
    expect_at(outcome.status == usearch_status_invalid_argument_k, __FILE__, __LINE__,
              "A pull without a cursor is refused");

    usearch_geospatial_free(index);
    usearch_arena_free(arena);
    usearch_executor_free(executor);
}

/** Shapes and geospatial reshard and refine alike: one index gathers another of its geometry, which
 *  ends empty and live, and refuses one of another, and a one-shot refine after it leaves every
 *  entry of the gathered index reachable. */
void test_shapes_and_geospatial_reshard(void) {
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
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
        if (which == 2) break;
        usearch_shapes_insert(&(usearch_shapes_insert_t) {
            .index = shapes[which],
            .executor = executor,
            .outcome = &outcome,
            .count = entries_per_index,
            .keys = {keys + which * entries_per_index, sizeof(usearch_key_t)},
            .shapes = tape_of(points + which * entries_per_index * points_per_shape * 3, offsets, usearch_scalar_f32_k),
            .expansion_factor = 4});
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    }
    usearch_shapes_index_t const mismatched[2] = {shapes[0], shapes[2]};
    usearch_shapes_reshard(&(usearch_shapes_reshard_t) {.sources_count = 2,
                                                        .sources = mismatched,
                                                        .targets_count = 1,
                                                        .targets = shapes,
                                                        .executor = executor,
                                                        .outcome = &outcome});
    expect_at(outcome.status == usearch_status_config_mismatch_k, __FILE__, __LINE__,
              "A source of another point count is refused");
    usearch_refine_report_t report = {0};
    usearch_shapes_reshard(&(usearch_shapes_reshard_t) {.sources_count = 2,
                                                        .sources = shapes,
                                                        .targets_count = 1,
                                                        .targets = shapes,
                                                        .executor = executor,
                                                        .outcome = &outcome,
                                                        .report = &report});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    expect_at(report.reachability.audited == 2 * entries_per_index, __FILE__, __LINE__,
              "The other source's shapes must land");
    usearch_shapes_refine(
        &(usearch_shapes_refine_t) {.index = shapes[0], .executor = executor, .outcome = &outcome, .report = &report});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    expect_at(report.reachability.reachable == 2 * entries_per_index, __FILE__, __LINE__,
              "Every gathered shape must stay reachable");
    usearch_shapes_index_description_t shapes_description = {0};
    usearch_shapes_describe(&(usearch_shapes_describe_t) {.index = shapes[0],
                                                          .executor = executor,
                                                          .outcome = &outcome,
                                                          .description = &shapes_description});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    expect_at(shapes_description.size == 2 * entries_per_index, __FILE__, __LINE__, "The gathered sizes must add up");
    usearch_shapes_describe(&(usearch_shapes_describe_t) {.index = shapes[1],
                                                          .executor = executor,
                                                          .outcome = &outcome,
                                                          .description = &shapes_description});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    expect_at(shapes_description.size == 0 && shapes_description.total_points == 0, __FILE__, __LINE__,
              "The other source must end empty, its points counted where they landed");
    for (size_t which = 0; which != 3; ++which) usearch_shapes_free(shapes[which]);

    // Observations scattered over a city, a few hundred metres apart.
    usearch_geospatial_observation_t observations[2 * entries_per_index];
    for (size_t entry = 0; entry != 2 * entries_per_index; ++entry)
        observations[entry] = observed_at(51.45 + 0.1 * rand() / RAND_MAX, -0.15 + 0.1 * rand() / RAND_MAX, NAN);
    usearch_geospatial_index_t places[2];
    for (size_t which = 0; which != 2; ++which) {
        usearch_geospatial_init_ram(&(usearch_geospatial_init_ram_t) {.index = &places[which], .outcome = &outcome});
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
        usearch_geospatial_insert(
            &(usearch_geospatial_insert_t) {.index = places[which],
                                            .executor = executor,
                                            .outcome = &outcome,
                                            .count = entries_per_index,
                                            .keys = {keys + which * entries_per_index, sizeof(usearch_key_t)},
                                            .observations = observations + which * entries_per_index,
                                            .expansion_factor = 16});
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    }
    usearch_geospatial_reshard(&(usearch_geospatial_reshard_t) {.sources_count = 2,
                                                                .sources = places,
                                                                .targets_count = 1,
                                                                .targets = places,
                                                                .executor = executor,
                                                                .outcome = &outcome});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    usearch_geospatial_refine(&(usearch_geospatial_refine_t) {.index = places[0],
                                                              .executor = executor,
                                                              .outcome = &outcome,
                                                              .report = &report});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    expect_at(
        report.reachability.audited == 2 * entries_per_index && report.reachability.reachable == 2 * entries_per_index,
        __FILE__, __LINE__, "Every gathered observation must stay reachable");
    usearch_geospatial_free(places[0]);
    usearch_geospatial_free(places[1]);
    usearch_executor_free(executor);
}

/**
 *  @brief Defines @c expect_<family>_reshard_refusals, which drives one family's long verbs through
 *      every refusal of lists, plans and executors its header documents.
 *
 *  @p full and @p also_full hold entries and @p empty and @p also_empty hold none, and no refused
 *  call may change that. A plan opened with @p executor is refused by a call or a finish naming
 *  @p other_executor, a finish still freeing it, and a refine plan is refused by another index. A
 *  finish reopens every handle its plan held, so a slot set back to @c NULL refuses a second finish
 *  and opens a new plan on the next call. The macro ends with the function's prototype, which the
 *  semicolon at its use closes.
 */
#define define_reshard_refusals(family)                                                                             \
    static size_t family##_size_of(usearch_##family##_index_t index, usearch_executor_t executor) {                 \
        usearch_outcome_t outcome = {0};                                                                            \
        usearch_##family##_index_description_t description = {0};                                                   \
        usearch_##family##_describe(&(usearch_##family##_describe_t) {.index = index,                               \
                                                                      .executor = executor,                         \
                                                                      .outcome = &outcome,                          \
                                                                      .description = &description});                \
        expect_at(!outcome.message, __FILE__, __LINE__, outcome.message);                                           \
        return description.size;                                                                                    \
    }                                                                                                               \
    static void expect_##family##_reshard_refusals(                                                                 \
        usearch_##family##_index_t full, usearch_##family##_index_t also_full, usearch_##family##_index_t empty,    \
        usearch_##family##_index_t also_empty, usearch_executor_t executor, usearch_executor_t other_executor) {    \
        typedef usearch_##family##_index_t index_t;                                                                 \
        usearch_outcome_t outcome = {0};                                                                            \
        size_t const full_size = family##_size_of(full, executor),                                                  \
                     also_full_size = family##_size_of(also_full, executor);                                        \
        index_t const fulls[2] = {full, also_full}, empties[2] = {empty, also_empty};                               \
        index_t const with_null[2] = {full, NULL}, full_twice[2] = {full, full}, empty_twice[2] = {empty, empty};   \
        index_t const full_and_empty[2] = {full, empty};                                                            \
        struct {                                                                                                    \
            size_t sources_count;                                                                                   \
            index_t const* sources;                                                                                 \
            size_t targets_count;                                                                                   \
            index_t const* targets;                                                                                 \
            usearch_status_t status;                                                                                \
            char const* claim;                                                                                      \
        } const refusals[] = {                                                                                      \
            {0, fulls, 1, empties, usearch_status_invalid_argument_k, "No sources must be refused"},                \
            {1, fulls, 0, empties, usearch_status_invalid_argument_k, "No targets must be refused"},                \
            {1, NULL, 1, empties, usearch_status_invalid_argument_k, "A NULL source list must be refused"},         \
            {1, fulls, 1, NULL, usearch_status_invalid_argument_k, "A NULL target list must be refused"},           \
            {2, with_null, 1, empties, usearch_status_invalid_argument_k, "A NULL source must be refused"},         \
            {2, full_twice, 1, fulls, usearch_status_invalid_argument_k, "A source named twice must be refused"},   \
            {1, fulls, 2, empty_twice, usearch_status_invalid_argument_k, "A target named twice must be refused"},  \
            {1, fulls, 1, &also_full, usearch_status_config_mismatch_k,                                             \
             "A target neither a source nor empty must be refused"},                                                \
            {1, fulls, 2, full_and_empty, usearch_status_config_mismatch_k,                                         \
             "A source cut into targets that include it must be refused"},                                          \
            {2, fulls, 2, full_and_empty, usearch_status_config_mismatch_k,                                         \
             "Several sources cut into targets that include one must be refused"},                                  \
        };                                                                                                          \
        for (size_t refusal = 0; refusal != sizeof(refusals) / sizeof(refusals[0]); ++refusal) {                    \
            usearch_##family##_reshard(                                                                             \
                &(usearch_##family##_reshard_t) {.sources_count = refusals[refusal].sources_count,                  \
                                                 .sources = refusals[refusal].sources,                              \
                                                 .targets_count = refusals[refusal].targets_count,                  \
                                                 .targets = refusals[refusal].targets,                              \
                                                 .executor = executor,                                              \
                                                 .outcome = &outcome});                                             \
            expect_at(outcome.status == refusals[refusal].status, __FILE__, __LINE__, refusals[refusal].claim);     \
        }                                                                                                           \
                                                                                                                    \
        usearch_##family##_reshard_plan_t plan = NULL;                                                              \
        usearch_##family##_reshard_t call = {.sources_count = 2,                                                    \
                                             .sources = full_and_empty,                                             \
                                             .targets_count = 1,                                                    \
                                             .targets = fulls,                                                      \
                                             .outcome = &outcome,                                                   \
                                             .plan = &plan};                                                        \
        usearch_##family##_reshard(&call);                                                                          \
        expect_at(outcome.status == usearch_status_invalid_argument_k && !plan, __FILE__, __LINE__,                 \
                  "A plan without an executor is refused");                                                         \
        call.executor = executor;                                                                                   \
        usearch_##family##_reshard(&call);                                                                          \
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;                              \
        if (!expect_at(plan != NULL, __FILE__, __LINE__, "A NULL pointee must open a plan")) return;                \
        call.executor = other_executor;                                                                             \
        usearch_##family##_reshard(&call);                                                                          \
        expect_at(outcome.status == usearch_status_invalid_argument_k, __FILE__, __LINE__,                          \
                  "A live plan refuses another executor");                                                          \
        usearch_##family##_reshard_plan_finish(                                                                     \
            &(usearch_##family##_reshard_plan_finish_t) {.executor = other_executor,                                \
                                                         .outcome = &outcome,                                       \
                                                         .plan = plan});                                            \
        expect_at(outcome.status == usearch_status_invalid_argument_k, __FILE__, __LINE__,                          \
                  "A finish refuses another executor");                                                             \
        plan = NULL;                                                                                                \
        usearch_##family##_reshard_plan_finish(                                                                     \
            &(usearch_##family##_reshard_plan_finish_t) {.executor = executor, .outcome = &outcome, .plan = plan}); \
        expect_at(outcome.status == usearch_status_invalid_argument_k, __FILE__, __LINE__,                          \
                  "A finish refuses a NULL plan");                                                                  \
        call.executor = executor;                                                                                   \
        usearch_##family##_reshard(&call);                                                                          \
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;                              \
        if (!expect_at(plan != NULL, __FILE__, __LINE__, "A finished plan must reopen its handles for the next"))   \
            return;                                                                                                 \
        usearch_##family##_reshard_plan_finish(                                                                     \
            &(usearch_##family##_reshard_plan_finish_t) {.executor = executor, .outcome = &outcome, .plan = plan}); \
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;                              \
                                                                                                                    \
        usearch_##family##_refine_plan_t refine_plan = NULL;                                                        \
        usearch_##family##_refine(                                                                                  \
            &(usearch_##family##_refine_t) {.index = full, .outcome = &outcome, .plan = &refine_plan});             \
        expect_at(outcome.status == usearch_status_invalid_argument_k && !refine_plan, __FILE__, __LINE__,          \
                  "A refine plan without an executor is refused");                                                  \
        usearch_##family##_refine(&(usearch_##family##_refine_t) {.index = full,                                    \
                                                                  .executor = executor,                             \
                                                                  .outcome = &outcome,                              \
                                                                  .plan = &refine_plan});                           \
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;                              \
        if (!expect_at(refine_plan != NULL, __FILE__, __LINE__, "A NULL pointee must open a refine plan")) return;  \
        usearch_##family##_refine(&(usearch_##family##_refine_t) {.index = full,                                    \
                                                                  .executor = other_executor,                       \
                                                                  .outcome = &outcome,                              \
                                                                  .plan = &refine_plan});                           \
        expect_at(outcome.status == usearch_status_invalid_argument_k, __FILE__, __LINE__,                          \
                  "A live refine plan refuses another executor");                                                   \
        usearch_##family##_refine(&(usearch_##family##_refine_t) {.index = also_full,                               \
                                                                  .executor = executor,                             \
                                                                  .outcome = &outcome,                              \
                                                                  .plan = &refine_plan});                           \
        expect_at(outcome.status == usearch_status_invalid_argument_k, __FILE__, __LINE__,                          \
                  "A live refine plan refuses another index");                                                      \
        usearch_##family##_refine_plan_finish(&(usearch_##family##_refine_plan_finish_t) {.index = also_full,       \
                                                                                          .executor = executor,     \
                                                                                          .outcome = &outcome,      \
                                                                                          .plan = refine_plan});    \
        expect_at(outcome.status == usearch_status_invalid_argument_k, __FILE__, __LINE__,                          \
                  "A refine finish refuses another index");                                                         \
                                                                                                                    \
        expect_at(family##_size_of(full, executor) == full_size &&                                                  \
                      family##_size_of(also_full, executor) == also_full_size &&                                    \
                      family##_size_of(empty, executor) == 0 && family##_size_of(also_empty, executor) == 0,        \
                  __FILE__, __LINE__, "A refused reshard must move nothing");                                       \
    }                                                                                                               \
    static void expect_##family##_reshard_refusals(                                                                 \
        usearch_##family##_index_t full, usearch_##family##_index_t also_full, usearch_##family##_index_t empty,    \
        usearch_##family##_index_t also_empty, usearch_executor_t executor, usearch_executor_t other_executor)

/**
 *  @brief Defines @c expect_<family>_reshard_cut, which cuts one modality's @p source of @p entries
 *      into @p first and @p second through a plan.
 *
 *  A handle as its only source and target is refused, having one shard to recut. Every call of the
 *  plan is cancelled until the closing rounds end, each naming the phase it stopped in; the finish
 *  audits every entry, and the source ends empty while the targets hold every entry between them.
 *  The macro ends with the function's prototype, which the semicolon at its use closes.
 */
#define define_reshard_cut(family)                                                                                 \
    static void expect_##family##_reshard_cut(usearch_##family##_index_t source, usearch_##family##_index_t first, \
                                              usearch_##family##_index_t second, usearch_executor_t executor,      \
                                              size_t entries) {                                                    \
        usearch_outcome_t outcome = {0};                                                                           \
        usearch_##family##_reshard(&(usearch_##family##_reshard_t) {.sources_count = 1,                            \
                                                                    .sources = &source,                            \
                                                                    .targets_count = 1,                            \
                                                                    .targets = &source,                            \
                                                                    .executor = executor,                          \
                                                                    .outcome = &outcome});                         \
        expect_at(outcome.status == usearch_status_unsupported_k, __FILE__, __LINE__,                              \
                  "A handle of one shard has nothing to recut");                                                   \
                                                                                                                   \
        usearch_##family##_index_t const targets[2] = {first, second};                                             \
        usearch_##family##_reshard_plan_t plan = NULL;                                                             \
        size_t calls = 0;                                                                                          \
        do {                                                                                                       \
            usearch_phase_progress_t phase = {0};                                                                  \
            usearch_##family##_reshard(&(usearch_##family##_reshard_t) {.sources_count = 1,                        \
                                                                        .sources = &source,                        \
                                                                        .targets_count = 2,                        \
                                                                        .targets = targets,                        \
                                                                        .executor = executor,                      \
                                                                        .outcome = &outcome,                       \
                                                                        .progress = {cancel_progress},             \
                                                                        .plan = &plan,                             \
                                                                        .phase = &phase});                         \
            bool const cancelled = outcome.status == usearch_status_cancelled_k;                                   \
            if (!expect_at(cancelled || !outcome.message, __FILE__, __LINE__, outcome.message)) return;            \
            expect_at(plan != NULL && phase.phase != usearch_phase_none_k, __FILE__, __LINE__,                     \
                      "Every call through a plan must keep it live and name a phase");                             \
            if (!cancelled) {                                                                                      \
                expect_at(phase.phase == usearch_phase_done_k, __FILE__, __LINE__,                                 \
                          "A call whose rounds ended names the closing phase");                                    \
                break;                                                                                             \
            }                                                                                                      \
            expect_at(phase.phase != usearch_phase_done_k && phase.completed <= phase.total, __FILE__, __LINE__,   \
                      "A stopped round must stand in one of its phases");                                          \
        } while (++calls != entries);                                                                              \
        usearch_refine_report_t report = {0};                                                                      \
        usearch_##family##_reshard_plan_finish(&(usearch_##family##_reshard_plan_finish_t) {.executor = executor,  \
                                                                                            .outcome = &outcome,   \
                                                                                            .plan = plan,          \
                                                                                            .report = &report});   \
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;                             \
        expect_at(report.reachability.audited == entries, __FILE__, __LINE__,                                      \
                  "The closing health checks must judge every entry");                                             \
        expect_at(family##_size_of(source, executor) == 0 &&                                                       \
                      family##_size_of(first, executor) + family##_size_of(second, executor) == entries,           \
                  __FILE__, __LINE__, "A cut must move every entry out of its source");                            \
    }                                                                                                              \
    static void expect_##family##_reshard_cut(usearch_##family##_index_t source, usearch_##family##_index_t first, \
                                              usearch_##family##_index_t second, usearch_executor_t executor,      \
                                              size_t entries)

define_reshard_refusals(dense);
define_reshard_refusals(sequence);
define_reshard_refusals(shapes);
define_reshard_refusals(geospatial);
define_reshard_cut(sequence);
define_reshard_cut(shapes);
define_reshard_cut(geospatial);

/** Every family refuses the reshard lists, plans and executors its header refuses, moving
 *  nothing, and every modality cuts one index into two through a plan. */
void test_reshard_refusals(void) {
    enum { entries = 120, dimensions = 83, points_per_shape = 8 };
    usearch_outcome_t outcome = {0};
    usearch_executor_t const executor = create_executor(2), other_executor = create_executor(1);
    usearch_key_t keys[entries];
    for (size_t entry = 0; entry != entries; ++entry) keys[entry] = key_of(entry);

    float* const rows = create_vectors(entries, dimensions);
    usearch_dense_index_t const dense[4] = {create_index_holding(rows, entries, dimensions, executor),
                                            create_index_holding(rows, entries, dimensions, executor),
                                            create_index(dimensions), create_index(dimensions)};
    expect_dense_reshard_refusals(dense[0], dense[1], dense[2], dense[3], executor, other_executor);
    for (size_t which = 0; which != 4; ++which) usearch_dense_free(dense[which]);
    free(rows);

    size_t offsets[entries + 1];
    char* const reads = create_reads(entries, offsets);
    usearch_sequence_index_t sequences[4];
    for (size_t which = 0; which != 4; ++which) {
        usearch_sequence_init_ram(&(usearch_sequence_init_ram_t) {.index = &sequences[which],
                                                                  .outcome = &outcome,
                                                                  .metric = usearch_sequence_metric_levenshtein_bytes_k,
                                                                  .connectivity = 8});
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
        if (which >= 2) continue;
        usearch_sequence_insert(
            &(usearch_sequence_insert_t) {.index = sequences[which],
                                          .executor = executor,
                                          .outcome = &outcome,
                                          .count = entries,
                                          .keys = {keys, sizeof(usearch_key_t)},
                                          .sequences = tape_of(reads, offsets, usearch_scalar_u8_k)});
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    }
    expect_sequence_reshard_refusals(sequences[0], sequences[1], sequences[2], sequences[3], executor, other_executor);
    expect_sequence_reshard_cut(sequences[0], sequences[2], sequences[3], executor, entries);
    for (size_t which = 0; which != 4; ++which) usearch_sequence_free(sequences[which]);
    free(reads);

    float points[entries * points_per_shape * 3];
    for (size_t coordinate = 0; coordinate != sizeof(points) / sizeof(points[0]); ++coordinate)
        points[coordinate] = (float)rand() / (float)RAND_MAX;
    for (size_t shape = 0; shape != entries + 1; ++shape) offsets[shape] = shape * points_per_shape;
    usearch_shapes_index_t shapes[4];
    for (size_t which = 0; which != 4; ++which) {
        usearch_shapes_init_ram(&(usearch_shapes_init_ram_t) {.index = &shapes[which],
                                                              .outcome = &outcome,
                                                              .metric = usearch_shapes_metric_chamfer_k,
                                                              .quantization = usearch_scalar_f32_k,
                                                              .points = points_per_shape});
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
        if (which >= 2) continue;
        usearch_shapes_insert(&(usearch_shapes_insert_t) {.index = shapes[which],
                                                          .executor = executor,
                                                          .outcome = &outcome,
                                                          .count = entries,
                                                          .keys = {keys, sizeof(usearch_key_t)},
                                                          .shapes = tape_of(points, offsets, usearch_scalar_f32_k),
                                                          .expansion_factor = 4});
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    }
    expect_shapes_reshard_refusals(shapes[0], shapes[1], shapes[2], shapes[3], executor, other_executor);
    expect_shapes_reshard_cut(shapes[0], shapes[2], shapes[3], executor, entries);
    for (size_t which = 0; which != 4; ++which) usearch_shapes_free(shapes[which]);

    usearch_geospatial_observation_t observations[entries];
    for (size_t entry = 0; entry != entries; ++entry)
        observations[entry] = observed_at(51.45 + 0.1 * rand() / RAND_MAX, -0.15 + 0.1 * rand() / RAND_MAX, NAN);
    usearch_geospatial_index_t places[4];
    for (size_t which = 0; which != 4; ++which) {
        usearch_geospatial_init_ram(&(usearch_geospatial_init_ram_t) {.index = &places[which], .outcome = &outcome});
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
        if (which >= 2) continue;
        usearch_geospatial_insert(&(usearch_geospatial_insert_t) {.index = places[which],
                                                                  .executor = executor,
                                                                  .outcome = &outcome,
                                                                  .count = entries,
                                                                  .keys = {keys, sizeof(usearch_key_t)},
                                                                  .observations = observations,
                                                                  .expansion_factor = 16});
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    }
    expect_geospatial_reshard_refusals(places[0], places[1], places[2], places[3], executor, other_executor);
    expect_geospatial_reshard_cut(places[0], places[2], places[3], executor, entries);
    for (size_t which = 0; which != 4; ++which) usearch_geospatial_free(places[which]);

    usearch_executor_free(other_executor);
    usearch_executor_free(executor);
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
char const* test_geospatial_vram(void) {
    usearch_outcome_t outcome = {0};
    size_t devices_count = 0;
    usearch_gpu_devices(&(usearch_gpu_devices_t) {.outcome = &outcome, .devices_count = &devices_count});
    if (!devices_count) return NULL;
    int32_t const devices[1] = {0};
    usearch_vram_options_t const vram = {.devices = devices, .devices_count = 1};

    usearch_geospatial_index_t index = NULL;
    usearch_geospatial_init_vram(
        &(usearch_geospatial_init_vram_t) {.index = &index, .outcome = &outcome, .vram = vram});
    if (outcome.status == usearch_status_device_k) return no_gpu_device_k;
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;

    usearch_geospatial_terrain_t const terrain = {.rows = 1,
                                                  .columns = 1,
                                                  .row_step_degrees = 1,
                                                  .column_step_degrees = 1,
                                                  .callback = fill_unknown_heights};
    usearch_geospatial_index_t refused = NULL;
    usearch_geospatial_init_vram(
        &(usearch_geospatial_init_vram_t) {.index = &refused, .outcome = &outcome, .terrain = terrain, .vram = vram});
    expect_at(outcome.status == usearch_status_unsupported_k, __FILE__, __LINE__,
              "A GPU terrain needs its heights, not a callback");
    expect_at(refused == NULL, __FILE__, __LINE__, "A refused initialization returns no handle");

    usearch_executor_t executor = NULL;
    usearch_executor_for_vram(
        &(usearch_executor_for_vram_t) {.executor = &executor, .outcome = &outcome, .vram = vram});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;

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
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
    usearch_geospatial_index_description_t description = {0};
    usearch_geospatial_describe(&(usearch_geospatial_describe_t) {.index = index,
                                                                  .executor = executor,
                                                                  .outcome = &outcome,
                                                                  .description = &description});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
    expect_at(description.size == 4 && description.metadata.connectivity == 16, __FILE__, __LINE__,
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
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
    for (size_t row = 0; row != 4; ++row) {
        expect_at(matches.counts[row] == 2, __FILE__, __LINE__, "Every observation finds itself and a neighbor");
        expect_at(memcmp(&matches.keys[row * 2], &keys[row], sizeof(usearch_key_t)) == 0 &&
                      matches.scores[row * 2] < 1e-3 && matches.scores[row * 2 + 1] > 60,
                  __FILE__, __LINE__, "An observation finds itself first, at zero metres");
    }
    search.relation = usearch_geospatial_relation_visible_k;
    usearch_geospatial_search(&search);
    expect_at(outcome.status == usearch_status_unsupported_k, __FILE__, __LINE__,
              "Lines of sight need a handle with terrain");

    char const path[] = "usearch-test-c-geospatial-vram.safegraphs";
    usearch_geospatial_save(&(usearch_geospatial_save_t) {.index = index,
                                                          .executor = executor,
                                                          .outcome = &outcome,
                                                          .path = {(uint8_t const*)path, sizeof(path) - 1}});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
    usearch_geospatial_index_t loaded = NULL;
    usearch_geospatial_load_ram(&(usearch_geospatial_load_ram_t) {.index = &loaded,
                                                                  .outcome = &outcome,
                                                                  .path = {(uint8_t const*)path, sizeof(path) - 1}});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
    usearch_geospatial_index_description_t loaded_description = {0};
    usearch_geospatial_describe(
        &(usearch_geospatial_describe_t) {.index = loaded, .outcome = &outcome, .description = &loaded_description});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
    expect_at(loaded_description.size == description.size, __FILE__, __LINE__, "A GPU checkpoint loads into RAM whole");
    usearch_geospatial_free(loaded);
    remove(path);

    usearch_arena_free(arena);
    usearch_geospatial_free(index);
    usearch_executor_free(executor);
    return NULL;
}

/** A packed kind's row is as wide as its logical dimensions, so every product spans all of them -
 *  a row read at its storage width is either refused or scored over half its dimensions. */
void test_multiply_packed(void) {

    enum { rows_count = 8, dimensions = 16 };
    usearch_outcome_t outcome = {0};
    usearch_dense_index_t index = NULL;
    usearch_dense_init_ram(&(usearch_dense_init_ram_t) {.index = &index,
                                                        .outcome = &outcome,
                                                        .metric = usearch_dense_metric_inner_k,
                                                        .quantization = usearch_scalar_u1x8_k,
                                                        .dimensions = dimensions,
                                                        .connectivity = 16});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    usearch_executor_t const executor = create_executor(0);
    usearch_dense_reserve(
        &(usearch_dense_reserve_t) {.index = index, .executor = executor, .outcome = &outcome, .capacity = rows_count});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;

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
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;

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
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;

    for (size_t row = 0; row != rows_count; ++row) {
        expect_at(products.offsets[row + 1] > products.offsets[row], __FILE__, __LINE__,
                  "Every walk scores at least its entry point");
        for (size_t pair = products.offsets[row]; pair != products.offsets[row + 1]; ++pair) {
            size_t const named = products.keys[pair].bytes[15];
            double exact = 0;
            for (size_t column = 0; column != dimensions; ++column)
                exact += data[row * dimensions + column] * data[named * dimensions + column];
            expect_at(products.scores[pair] == exact, __FILE__, __LINE__, "A product must span every packed dimension");
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
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    usearch_dense_multiply(
        &(usearch_dense_multiply_t) {.index = index,
                                     .outcome = &outcome,
                                     .count = 1,
                                     .left = {products.keys, sizeof(usearch_key_t), dimensions, usearch_scalar_u8_k},
                                     .products = &products,
                                     .arena = arena});
    expect_at(outcome.status == usearch_status_invalid_argument_k, __FILE__, __LINE__,
              "An input inside the rewound arena is refused");
    expect_at(products.keys == NULL && products.offsets == NULL, __FILE__, __LINE__,
              "A refused call leaves its outputs NULL");

    usearch_arena_free(arena);
    usearch_executor_free(executor);
    usearch_dense_free(index);
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
    expect_at(scalar == values->scalar && columns == values->columns, __FILE__, __LINE__,
              "The lookup must learn the stored row shape");
    for (size_t key = 0; key != count; ++key) {
        size_t const token = keys[key].bytes[15];
        rows[key] = token < values->rows ? values->data + token * values->columns : NULL;
    }
}

/** A graph this small walks every key, so attention reproduces the exact softmax over all of them, with one
 *  head per walk and with two sharing it; heads that do not divide the rows, a key without a value row, a
 *  device row lookup or no host one, and a distance kind refuse the call. */
void test_attention_is_exact_over_every_key(void) {

    enum { tokens = 4, dimensions = 4, heads = 2 };
    usearch_outcome_t outcome = {0};
    usearch_dense_index_t index = NULL;
    usearch_dense_init_ram(&(usearch_dense_init_ram_t) {.index = &index,
                                                        .outcome = &outcome,
                                                        .metric = usearch_dense_metric_inner_k,
                                                        .quantization = usearch_scalar_i8_k,
                                                        .dimensions = dimensions});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    usearch_executor_t const executor = create_executor(0);
    usearch_dense_reserve(
        &(usearch_dense_reserve_t) {.index = index, .executor = executor, .outcome = &outcome, .capacity = tokens});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;

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
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;

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
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
        expect_at(outputs.scalar == usearch_scalar_f32_k && outputs.columns == dimensions, __FILE__, __LINE__,
                  "A zero output scalar lands the queries' own, as wide as the keys");
        float const* const attended = (float const*)outputs.data;
        size_t const pitch = outputs.stride / sizeof(float);
        expect_at(fabsf(carry[0].maximum_score - 3 * 0.69314718f) < 1e-5f && fabsf(carry[1].maximum_score) < 1e-5f,
                  __FILE__, __LINE__, "Each head's carry must hold its own largest logit");
        expect_at(fabsf(carry[0].partition - 1.875f) < 1e-4f && fabsf(carry[1].partition - 1.875f) < 1e-4f, __FILE__,
                  __LINE__, "Each head's carry must hold its partition over every key");
        expect_at(fabsf(attended[0] - 4.25f / 1.875f) < 1e-4f && fabsf(attended[pitch] - 1.375f / 1.875f) < 1e-4f,
                  __FILE__, __LINE__, "Every key must weigh the value row it resolves to");
        expect_at(fabsf(attended[1] - 1) < 1e-4f && fabsf(attended[pitch + 1] - 1) < 1e-4f && attended[2] == 0 &&
                      attended[pitch + 3] == 0,
                  __FILE__, __LINE__, "The attended rows must be normalized and as wide as the keys");
    }

    // Narrower outputs land the same rows to their own precision, and a NULL carry keeps no state.
    attend.outputs_scalar = usearch_scalar_bf16_k;
    attend.carry = NULL;
    usearch_dense_attend(&attend);
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    expect_at(outputs.scalar == usearch_scalar_bf16_k && outputs.stride == dimensions * sizeof(uint16_t), __FILE__,
              __LINE__, "Outputs land packed in the scalar asked for");
    uint16_t const* const narrowed = (uint16_t const*)outputs.data;
    uint32_t const widened_bits = (uint32_t)narrowed[0] << 16;
    float widened;
    memcpy(&widened, &widened_bits, sizeof(widened));
    expect_at(fabsf(widened - 4.25f / 1.875f) < 1e-2f, __FILE__, __LINE__, "A bf16 output must round the f32 one");
    attend.outputs_scalar = 0;
    attend.carry = carry;

    attend.head_count = 3;
    usearch_dense_attend(&attend);
    expect_at(outcome.status == usearch_status_shape_mismatch_k, __FILE__, __LINE__,
              "Heads that do not divide the rows must refuse");
    expect_at(outputs.data == NULL, __FILE__, __LINE__, "A refused call leaves its outputs NULL");

    attend.head_count = heads;
    values.rows = tokens - 1;
    usearch_dense_attend(&attend);
    expect_at(outcome.status == usearch_status_shape_mismatch_k, __FILE__, __LINE__,
              "A key without a value row must refuse the call");

    values.rows = tokens;
    attend.value_lookup = NULL;
    usearch_dense_attend(&attend);
    expect_at(outcome.status == usearch_status_invalid_argument_k, __FILE__, __LINE__,
              "A missing value lookup must refuse the call");

    usearch_dense_index_t const angular = create_index(dimensions);
    attend.index = angular;
    attend.value_lookup = attend_value_rows;
    usearch_dense_attend(&attend);
    expect_at(outcome.status == usearch_status_config_mismatch_k, __FILE__, __LINE__,
              "A distance kind must refuse attention");

    usearch_arena_free(arena);
    usearch_executor_free(executor);
    usearch_dense_free(angular);
    usearch_dense_free(index);
}

/** Every written run of @p matches against @p exact, <tt>[query][document]</tt> totals: as many keys
 *  as @p expected_count per query, largest first, each at its exact total, even keys alone when
 *  @p even_only. */
static void expect_maxsim_tape(usearch_matches_tape_view_t matches, size_t written, size_t queries, size_t documents,
                               double const* exact, size_t const* expected_count, bool even_only) {
    if (!expect_at(matches.offsets && matches.offsets[0] == 0, __FILE__, __LINE__, "A tape starts at zero")) return;
    expect_at(written == matches.offsets[queries], __FILE__, __LINE__, "The written count is the tape's last offset");
    for (size_t query = 0; query != queries; ++query) {
        size_t const first = matches.offsets[query], end = matches.offsets[query + 1];
        expect_at(end - first == expected_count[query], __FILE__, __LINE__,
                  "Every query lands as many keys as it kept");
        for (size_t entry = first; entry != end; ++entry) {
            uint64_t const document = value_of(matches.keys[entry]);
            expect_at(document < documents, __FILE__, __LINE__, "A written key names a document");
            expect_at(!even_only || document % 2 == 0, __FILE__, __LINE__, "A predicate narrows the written keys");
            expect_at(fabs(matches.scores[entry] - exact[query * documents + document]) < 1e-3, __FILE__, __LINE__,
                      "A written key scores its exact summed maxima");
            expect_at(entry == first || matches.scores[entry] <= matches.scores[entry - 1], __FILE__, __LINE__,
                      "Keys land largest first");
        }
    }
}

/** Documents of 2 to 4 token rows hold small integers, which @c bf16 keeps exactly, so every MaxSim
 *  total is exact: queries of 1, 2 and 3 rows land a ragged tape, walks and a scan alike, every
 *  document without a bound, the best one with one while @c required counts them all, and even
 *  documents alone under a predicate. */
void test_late_interaction_scores_over_documents(void) {

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
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    usearch_executor_t const executor = create_executor(0);
    usearch_dense_reserve(
        &(usearch_dense_reserve_t) {.index = index, .executor = executor, .outcome = &outcome, .capacity = rows_count});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;

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
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;

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
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
        expect_maxsim_tape(matches, written, queries, documents, exact, every, false);

        maxsim.wanted = 1;
        usearch_dense_maxsim(&maxsim);
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
        expect_maxsim_tape(matches, written, queries, documents, exact, best_only, false);
        expect_at(required == queries * documents, __FILE__, __LINE__,
                  "Required counts every key scored before wanted cut them");
        for (size_t query = 0; query != queries; ++query)
            for (size_t document = 0; document != documents; ++document)
                expect_at(exact[query * documents + document] <= matches.scores[matches.offsets[query]] + 1e-3,
                          __FILE__, __LINE__, "The one kept key is the best");

        size_t widest_batch = 0;
        maxsim.wanted = 0;
        maxsim.predicate = (usearch_predicate_t) {admits_even, &widest_batch};
        usearch_dense_maxsim(&maxsim);
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
        expect_maxsim_tape(matches, written, queries, documents, exact, even_ones, true);
        expect_at(required == queries * documents / 2, __FILE__, __LINE__,
                  "Required counts no key the predicate rejected");
    }
    maxsim.predicate = (usearch_predicate_t) {0};
    maxsim.wanted = SIZE_MAX / queries + 1;
    usearch_dense_maxsim(&maxsim);
    expect_at(outcome.status == usearch_status_overflow_k, __FILE__, __LINE__,
              "Room for every query's wanted pairs must not wrap");

    usearch_arena_free(arena);
    usearch_executor_free(executor);
    usearch_dense_free(index);
}

/** A RAM index saved as one monolith serves the same keys and vectors from storage: an NVMe
 *  executor views the file, its search, count, get, health and cursor answer as the RAM index did,
 *  and every mutation is refused. An implementation without an NVMe residency says so, and the test
 *  skips. */
char const* test_nvme_view(size_t const dimensions) {

    enum { rows_count = 64 };
    usearch_outcome_t outcome = {0};
    usearch_dense_index_t const index = create_index(dimensions);
    usearch_executor_t const executor = create_executor(0);
    usearch_dense_reserve(
        &(usearch_dense_reserve_t) {.index = index, .executor = executor, .outcome = &outcome, .capacity = rows_count});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
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
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
    char const path[] = "usearch-test-c-nvme.safegraphs";
    usearch_bytes_view_t const path_view = {(uint8_t const*)path, sizeof(path) - 1};
    usearch_dense_save(
        &(usearch_dense_save_t) {.index = index, .executor = executor, .outcome = &outcome, .path = path_view});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;

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
        usearch_executor_free(storage);
        remove(path);
        free(data);
        usearch_executor_free(executor);
        usearch_dense_free(index);
        return needs_nvme_k;
    }
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;

    usearch_dense_index_description_t description = {0};
    usearch_dense_describe(&(usearch_dense_describe_t) {.index = view,
                                                        .executor = storage,
                                                        .outcome = &outcome,
                                                        .description = &description});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
    expect_at(description.size == rows_count, __FILE__, __LINE__, "A view must serve every saved row");
    expect_at(description.capacity == rows_count, __FILE__, __LINE__, "A view hosts exactly the rows it serves");
    expect_at(description.metadata.dimensions == dimensions, __FILE__, __LINE__,
              "A view must serve the saved geometry");

    usearch_reachability_t reachability = {0};
    usearch_dense_health(&(usearch_dense_health_t) {.index = view,
                                                    .executor = storage,
                                                    .outcome = &outcome,
                                                    .reachability = &reachability});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
    expect_at(reachability.audited == rows_count, __FILE__, __LINE__, "A view must audit every saved row");

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
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
    size_t recalled = 0;
    for (size_t row = 0; row != rows_count; ++row)
        recalled += matches.counts[row] && memcmp(matches.keys[row].bytes, keys[row].bytes, sizeof(usearch_key_t)) == 0;
    expect_at(recalled * 10 >= rows_count * 9, __FILE__, __LINE__,
              "A viewed graph must find its own rows as the RAM one does");

    usearch_dense_search(&(usearch_dense_search_t) {.index = view,
                                                    .executor = storage,
                                                    .outcome = &outcome,
                                                    .count = rows_count,
                                                    .queries = {data, row_bytes, dimensions, usearch_scalar_f32_k},
                                                    .matches = &matches,
                                                    .arena = arena,
                                                    .wanted = 1,
                                                    .traversal = usearch_traversal_scan_k});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
    for (size_t row = 0; row != rows_count; ++row)
        expect_at(
            matches.counts[row] == 1 && memcmp(matches.keys[row].bytes, keys[row].bytes, sizeof(usearch_key_t)) == 0,
            __FILE__, __LINE__, "An exact scan of a view must find every self-query itself");
    expect_expensive_narrows(view, storage, arena, data, rows_count, dimensions);

    size_t entries = 0;
    usearch_dense_count(&(usearch_dense_count_t) {.index = view,
                                                  .executor = storage,
                                                  .outcome = &outcome,
                                                  .key = key_of(5),
                                                  .entries = &entries});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
    expect_at(entries == 1, __FILE__, __LINE__, "A saved key must count once in its view");
    usearch_dense_count(&(usearch_dense_count_t) {.index = view,
                                                  .executor = storage,
                                                  .outcome = &outcome,
                                                  .key = key_of(rows_count + 7),
                                                  .entries = &entries});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
    expect_at(entries == 0, __FILE__, __LINE__, "A key never saved must count zero");

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
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
    if (!expect_at(written == 1 && vectors.data != NULL && vectors.columns == dimensions, __FILE__, __LINE__,
                   "A view must read a key's row back"))
        return NULL;
    float const* const retrieved = (float const*)vectors.data;
    for (size_t position = 0; position != dimensions; ++position)
        expect_at(fabsf(retrieved[position] - data[5 * dimensions + position]) <=
                      fabsf(data[5 * dimensions + position]) / 128,
                  __FILE__, __LINE__, "A bf16 view must return rows within its 7-bit mantissa");

    usearch_dense_cursor_t cursor = NULL;
    usearch_dense_cursor_init(&(usearch_dense_cursor_init_t) {.index = view, .outcome = &outcome, .cursor = &cursor});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
    usearch_keys_view_t walked = {0};
    usearch_dense_cursor_next(&(usearch_dense_cursor_next_t) {.index = view,
                                                              .executor = storage,
                                                              .outcome = &outcome,
                                                              .cursor = cursor,
                                                              .keys = &walked,
                                                              .written = &written,
                                                              .arena = arena,
                                                              .wanted = rows_count});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
    expect_at(written == rows_count, __FILE__, __LINE__, "A view's cursor must read every saved row");
    usearch_dense_cursor_free(cursor);

    usearch_dense_insert(&(usearch_dense_insert_t) {.index = view,
                                                    .executor = storage,
                                                    .outcome = &outcome,
                                                    .count = 1,
                                                    .keys = {keys, sizeof(usearch_key_t)},
                                                    .vectors = {data, row_bytes, dimensions, usearch_scalar_f32_k}});
    expect_at(outcome.status == usearch_status_read_only_k, __FILE__, __LINE__, "A view must refuse every mutation");
    usearch_dense_save(
        &(usearch_dense_save_t) {.index = view, .executor = storage, .outcome = &outcome, .path = path_view});
    expect_at(outcome.status == usearch_status_unsupported_k, __FILE__, __LINE__, "A view already is its checkpoint");
    usearch_arena_free(arena);
    usearch_dense_free(view);

    usearch_dense_index_t keyless_view = NULL;
    usearch_dense_view_nvme(&(usearch_dense_view_nvme_t) {.index = &keyless_view,
                                                          .executor = storage,
                                                          .outcome = &outcome,
                                                          .path = path_view});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
    usearch_dense_count(&(usearch_dense_count_t) {.index = keyless_view,
                                                  .executor = storage,
                                                  .outcome = &outcome,
                                                  .key = key_of(5),
                                                  .entries = &entries});
    expect_at(outcome.status == usearch_status_unsupported_k, __FILE__, __LINE__,
              "A view without a key table must refuse to count keys");
    usearch_dense_free(keyless_view);

    // A checkpoint cut inside its header holds no index in any format the library reads.
    FILE* const truncated = fopen(path, "wb");
    expect_at(truncated != NULL && fwrite(path, 1, 12, truncated) == 12, __FILE__, __LINE__,
              "Failed to truncate the checkpoint");
    fclose(truncated);
    usearch_dense_index_t truncated_view = NULL;
    usearch_dense_view_nvme(&(usearch_dense_view_nvme_t) {.index = &truncated_view,
                                                          .executor = storage,
                                                          .outcome = &outcome,
                                                          .path = path_view});
    expect_at(outcome.status != usearch_status_success_k && truncated_view == NULL, __FILE__, __LINE__,
              "A view must refuse a checkpoint cut inside its header");

    usearch_executor_free(storage);
    remove(path);
    free(data);
    usearch_executor_free(executor);
    usearch_dense_free(index);
    return NULL;
}

/** How many of @p count self-queries over @p rows find their own key first in @p index. */
static size_t count_self_matches(usearch_dense_index_t const index, usearch_executor_t const executor,
                                 usearch_arena_t const arena, float const* rows, size_t const count,
                                 size_t const dimensions, usearch_key_t const* keys) {
    usearch_outcome_t outcome = {0};
    usearch_matches_view_t matches = {0};
    usearch_dense_search(
        &(usearch_dense_search_t) {.index = index,
                                   .executor = executor,
                                   .outcome = &outcome,
                                   .count = count,
                                   .queries = {rows, dimensions * sizeof(float), dimensions, usearch_scalar_f32_k},
                                   .matches = &matches,
                                   .arena = arena,
                                   .wanted = 1});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return 0;
    size_t found = 0;
    for (size_t row = 0; row != count; ++row)
        found += matches.counts[row] && memcmp(matches.keys[row].bytes, keys[row].bytes, sizeof(usearch_key_t)) == 0;
    return found;
}

/** Whether the file at @p path exists. */
static int file_exists(char const* path) {
    FILE* const file = fopen(path, "rb");
    if (file) fclose(file);
    return file != NULL;
}

/** A RAM index of two shards saved as a manifest opens writable through @c usearch_dense_load_nvme:
 *  a lone part refuses, a second load of the manifest is contended, insertions, saves and a reshard
 *  naming drives answer @c usearch_status_unsupported_k while a view of the same manifest keeps
 *  answering @c usearch_status_read_only_k, even after a RAM source. A refine plan finished after
 *  the first shard committed leaves the second part in place and the handle searchable; a whole
 *  refine keeps or improves self-recall, keeps every entry reachable, and leaves only the refined
 *  parts once the handle is freed by its owner. */
char const* test_nvme_load(size_t const dimensions) {

    enum { rows_count = 600 };
    usearch_outcome_t outcome = {0};
    usearch_executor_t const executor = create_executor(0);
    usearch_dense_index_t index = NULL;
    usearch_dense_init_ram(&(usearch_dense_init_ram_t) {.index = &index,
                                                        .outcome = &outcome,
                                                        .metric = usearch_dense_metric_angular_k,
                                                        .quantization = usearch_scalar_bf16_k,
                                                        .dimensions = dimensions,
                                                        .connectivity = 16,
                                                        .ram = {.shard_capacity = rows_count / 2}});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
    usearch_dense_reserve(
        &(usearch_dense_reserve_t) {.index = index, .executor = executor, .outcome = &outcome, .capacity = rows_count});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
    float* const data = create_vectors(rows_count, dimensions);
    usearch_key_t keys[rows_count];
    for (size_t row = 0; row != rows_count; ++row) keys[row] = key_of(row);
    usearch_dense_insert(
        &(usearch_dense_insert_t) {.index = index,
                                   .executor = executor,
                                   .outcome = &outcome,
                                   .count = rows_count,
                                   .keys = {keys, sizeof(usearch_key_t)},
                                   .vectors = {data, dimensions * sizeof(float), dimensions, usearch_scalar_f32_k},
                                   .expansion_factor = 4});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
    char const manifest[] = "usearch-test-c-nvme-load.safegraphs.json";
    char const* const old_parts[2] = {"usearch-test-c-nvme-load-00001-of-00002.safegraphs",
                                      "usearch-test-c-nvme-load-00002-of-00002.safegraphs"};
    // A replacement takes the first generation no file holds, so the twice-refined first shard
    // ends at the second.
    char const* const new_parts[3] = {"usearch-test-c-nvme-load.1-00001-of-00002.safegraphs",
                                      "usearch-test-c-nvme-load.1-00002-of-00002.safegraphs",
                                      "usearch-test-c-nvme-load.2-00001-of-00002.safegraphs"};
    usearch_bytes_view_t const manifest_view = {(uint8_t const*)manifest, sizeof(manifest) - 1};
    usearch_dense_save(
        &(usearch_dense_save_t) {.index = index, .executor = executor, .outcome = &outcome, .path = manifest_view});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
    usearch_dense_free(index);
    expect_at(file_exists(old_parts[0]) && file_exists(old_parts[1]), __FILE__, __LINE__,
              "Two shards must save as two parts");

    usearch_executor_t storage = NULL;
    usearch_executor_for_nvme(&(usearch_executor_for_nvme_t) {.executor = &storage, .outcome = &outcome, .threads = 2});
    usearch_dense_index_t loaded = NULL;
    if (outcome.status == usearch_status_success_k)
        usearch_dense_load_nvme(&(usearch_dense_load_nvme_t) {.index = &loaded,
                                                              .executor = storage,
                                                              .outcome = &outcome,
                                                              .path = manifest_view});
    if (outcome.status == usearch_status_unsupported_k) {
        usearch_executor_free(storage);
        remove(manifest), remove(old_parts[0]), remove(old_parts[1]);
        free(data);
        usearch_executor_free(executor);
        return needs_nvme_k;
    }
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;

    usearch_dense_index_t refused = NULL;
    usearch_bytes_view_t const part_view = {(uint8_t const*)old_parts[0], strlen(old_parts[0])};
    usearch_dense_load_nvme(
        &(usearch_dense_load_nvme_t) {.index = &refused, .executor = storage, .outcome = &outcome, .path = part_view});
    expect_at(outcome.status == usearch_status_unsupported_k, __FILE__, __LINE__,
              "A lone part has no manifest to replace");
    usearch_dense_load_nvme(&(usearch_dense_load_nvme_t) {.index = &refused,
                                                          .executor = storage,
                                                          .outcome = &outcome,
                                                          .path = manifest_view});
    expect_at(outcome.status == usearch_status_contended_k && refused == NULL, __FILE__, __LINE__,
              "A second writer of one manifest must be refused");

    usearch_dense_index_t view = NULL;
    usearch_dense_view_nvme(
        &(usearch_dense_view_nvme_t) {.index = &view, .executor = storage, .outcome = &outcome, .path = manifest_view});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
    usearch_dense_refine(&(usearch_dense_refine_t) {.index = view, .executor = storage, .outcome = &outcome});
    expect_at(outcome.status == usearch_status_read_only_k, __FILE__, __LINE__, "A view must keep refusing refines");

    usearch_dense_insert(
        &(usearch_dense_insert_t) {.index = loaded,
                                   .executor = storage,
                                   .outcome = &outcome,
                                   .count = 1,
                                   .keys = {keys, sizeof(usearch_key_t)},
                                   .vectors = {data, dimensions * sizeof(float), dimensions, usearch_scalar_f32_k}});
    expect_at(outcome.status == usearch_status_unsupported_k, __FILE__, __LINE__,
              "A loaded handle must refuse insertions");
    usearch_dense_save(
        &(usearch_dense_save_t) {.index = loaded, .executor = storage, .outcome = &outcome, .path = manifest_view});
    expect_at(outcome.status == usearch_status_unsupported_k, __FILE__, __LINE__,
              "A loaded handle commits only through refines");
    usearch_dense_reshard(&(usearch_dense_reshard_t) {.sources_count = 1,
                                                      .sources = &loaded,
                                                      .targets_count = 1,
                                                      .targets = &loaded,
                                                      .executor = storage,
                                                      .outcome = &outcome,
                                                      .nvme = &(usearch_nvme_options_t) {&part_view, 1}});
    expect_at(outcome.status == usearch_status_unsupported_k, __FILE__, __LINE__,
              "A loaded handle's reshard must not place parts anew");
    // A loaded handle takes its writes through rewrites, so its own reshard opens; one stopped at
    // its first report and finished there commits nothing.
    usearch_dense_reshard_plan_t recut = NULL;
    usearch_dense_reshard(&(usearch_dense_reshard_t) {.sources_count = 1,
                                                      .sources = &loaded,
                                                      .targets_count = 1,
                                                      .targets = &loaded,
                                                      .executor = storage,
                                                      .outcome = &outcome,
                                                      .progress = {cancel_progress},
                                                      .plan = &recut});
    if (!expect_at(outcome.status == usearch_status_cancelled_k || !outcome.message, __FILE__, __LINE__,
                   outcome.message))
        return NULL;
    usearch_dense_reshard_plan_finish(
        &(usearch_dense_reshard_plan_finish_t) {.executor = storage, .outcome = &outcome, .plan = recut});
    if (!expect_at(outcome.status == usearch_status_cancelled_k || !outcome.message, __FILE__, __LINE__,
                   outcome.message))
        return NULL;
    usearch_dense_reshard(&(usearch_dense_reshard_t) {.sources_count = 1,
                                                      .sources = &view,
                                                      .targets_count = 1,
                                                      .targets = &view,
                                                      .executor = storage,
                                                      .outcome = &outcome});
    expect_at(outcome.status == usearch_status_read_only_k, __FILE__, __LINE__, "A view must refuse a reshard");
    // Every handle answers read-only before any answers for its kind, so a view target refuses
    // as read-only though a RAM source precedes it.
    usearch_dense_index_t foreign = NULL;
    usearch_dense_init_ram(&(usearch_dense_init_ram_t) {.index = &foreign,
                                                        .outcome = &outcome,
                                                        .metric = usearch_dense_metric_angular_k,
                                                        .quantization = usearch_scalar_bf16_k,
                                                        .dimensions = dimensions});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
    usearch_dense_index_t const mixed[2] = {loaded, foreign};
    usearch_dense_reshard(&(usearch_dense_reshard_t) {.sources_count = 2,
                                                      .sources = mixed,
                                                      .targets_count = 1,
                                                      .targets = &view,
                                                      .executor = storage,
                                                      .outcome = &outcome});
    expect_at(outcome.status == usearch_status_read_only_k, __FILE__, __LINE__,
              "A view target must refuse before a foreign source");
    usearch_dense_free(foreign);

    usearch_arena_t const arena = create_arena(storage);
    size_t const viewed_matches = count_self_matches(view, storage, arena, data, rows_count, dimensions, keys);

    // A plan stopped at every barrier reports a part only once the barrier closing its round
    // committed it, and its finish drops the rewrite of the second shard in hand.
    usearch_dense_refine_plan_t plan = NULL;
    usearch_refine_report_t committed = {0};
    do
        usearch_dense_refine(&(usearch_dense_refine_t) {.index = loaded,
                                                        .executor = storage,
                                                        .outcome = &outcome,
                                                        .report = &committed,
                                                        .progress = {cancel_progress},
                                                        .plan = &plan});
    while (outcome.status == usearch_status_cancelled_k && !committed.reselected);
    expect_at(outcome.status == usearch_status_cancelled_k, __FILE__, __LINE__, "A cancelled refine must say so");
    usearch_dense_refine_plan_finish(&(
        usearch_dense_refine_plan_finish_t) {.index = loaded, .executor = storage, .outcome = &outcome, .plan = plan});
    expect_at(outcome.status == usearch_status_cancelled_k, __FILE__, __LINE__,
              "A pass finished before its end must say so");
    expect_at(file_exists(new_parts[0]) && !file_exists(new_parts[1]), __FILE__, __LINE__,
              "Only the first shard may have been replaced");
    expect_at(file_exists(old_parts[1]), __FILE__, __LINE__, "The shard after a cancel keeps its part");
    expect_at(count_self_matches(loaded, storage, arena, data, rows_count, dimensions, keys) * 10 >= rows_count * 9,
              __FILE__, __LINE__, "A handle a cancel stopped must stay searchable");

    usearch_refine_report_t report = {0};
    usearch_dense_refine(&(usearch_dense_refine_t) {.index = loaded,
                                                    .executor = storage,
                                                    .outcome = &outcome,
                                                    .report = &report,
                                                    .refine = {.expansion_factor = 64}});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
    expect_at(report.reachability.audited == rows_count, __FILE__, __LINE__,
              "A refine must audit every entry it committed");
    expect_at(report.reachability.reachable == rows_count, __FILE__, __LINE__,
              "A refine must leave every entry reachable");
    size_t const refined_matches = count_self_matches(loaded, storage, arena, data, rows_count, dimensions, keys);
    expect_at(refined_matches >= viewed_matches, __FILE__, __LINE__, "A refine must keep or improve self-recall");
    // A view opened before the refine still reads its parts, which stay until the writer is freed.
    expect_at(count_self_matches(view, storage, arena, data, rows_count, dimensions, keys) == viewed_matches, __FILE__,
              __LINE__, "A view opened before the refine must keep answering as before");
    usearch_dense_free(view);
    usearch_dense_free(loaded);
    expect_at(!file_exists(old_parts[0]) && !file_exists(old_parts[1]) && !file_exists(new_parts[0]), __FILE__,
              __LINE__, "Freeing the writer removes superseded parts");
    expect_at(file_exists(new_parts[1]) && file_exists(new_parts[2]), __FILE__, __LINE__,
              "The manifest must name the refined parts");

    usearch_dense_index_t reopened = NULL;
    usearch_dense_view_nvme(&(usearch_dense_view_nvme_t) {.index = &reopened,
                                                          .executor = storage,
                                                          .outcome = &outcome,
                                                          .path = manifest_view});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
    expect_at(count_self_matches(reopened, storage, arena, data, rows_count, dimensions, keys) == refined_matches,
              __FILE__, __LINE__, "A view of the replaced manifest must answer as the refined handle did");
    usearch_dense_free(reopened);

    usearch_arena_free(arena);
    usearch_executor_free(storage);
    remove(manifest), remove(new_parts[1]), remove(new_parts[2]);
    free(data);
    usearch_executor_free(executor);
    return NULL;
}

/** A SafeGraphs monolith shaped like a benchmark's save - @c u32 keys, @c bf16 rows, 16 neighbors
 *  each - framed here by hand, opens through the NVMe view with every key zero-extended into the
 *  16-byte ones the ABI speaks, so it counts and ranks under the identity it was saved with. */
char const* test_nvme_view_narrow_keys(void) {

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
    if (!expect_at(file, __FILE__, __LINE__, "Failed to allocate memory")) return NULL;
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
    expect_at(stream && fwrite(file, 1, file_bytes, stream) == file_bytes && fclose(stream) == 0, __FILE__, __LINE__,
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
        usearch_executor_free(storage);
        remove(path);
        return needs_nvme_k;
    }
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;

    size_t entries = 0;
    usearch_dense_count(&(usearch_dense_count_t) {.index = view,
                                                  .executor = storage,
                                                  .outcome = &outcome,
                                                  .key = key_of(102),
                                                  .entries = &entries});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
    expect_at(entries == 1, __FILE__, __LINE__, "A `u32` key must count once under its widened identity");

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
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
    usearch_key_t const last = key_of(100 + rows_count - 1);
    expect_at(matches.counts[0] == 1 && memcmp(matches.keys[0].bytes, last.bytes, sizeof(usearch_key_t)) == 0, __FILE__,
              __LINE__, "A walk over `u32` keys must resolve them widened");

    usearch_arena_free(arena);
    usearch_dense_free(view);
    usearch_executor_free(storage);
    remove(path);
    return NULL;
}

/** A message is @c NULL exactly when the status is success, and the outcome a caller reuses is
 *  fully overwritten - the contract every binding branches on. Only the success half is
 *  exercised here; the failing verbs are the ones @c src/ has yet to instantiate. */
void test_outcome_contract(size_t const dimensions) {

    // An outcome carrying an earlier failure must come back clean, not merely unchanged.
    usearch_outcome_t outcome = {usearch_status_io_k, "a stale message the call must overwrite"};

    usearch_dense_index_t index = NULL;
    usearch_dense_init_ram(&(usearch_dense_init_ram_t) {.index = &index,
                                                        .outcome = &outcome,
                                                        .metric = usearch_dense_metric_angular_k,
                                                        .quantization = usearch_scalar_bf16_k,
                                                        .dimensions = dimensions});
    if (!expect_at(index != NULL, __FILE__, __LINE__, "Initialization returned no handle")) return;
    expect_at(outcome.status == usearch_status_success_k, __FILE__, __LINE__, "A successful call must report success");
    expect_at(outcome.message == NULL, __FILE__, __LINE__, "A successful call must carry no message");

    usearch_executor_t const executor = create_executor(0);
    outcome = (usearch_outcome_t) {usearch_status_io_k, "a stale message the call must overwrite"};
    usearch_dense_reserve(
        &(usearch_dense_reserve_t) {.index = index, .executor = executor, .outcome = &outcome, .capacity = 4});
    expect_at(outcome.status == usearch_status_success_k, __FILE__, __LINE__, "A successful call must report success");
    expect_at(outcome.message == NULL, __FILE__, __LINE__, "A successful call must carry no message");

    // A key the index never saw is a state, not a failure.
    size_t entries = 1;
    usearch_dense_count(&(usearch_dense_count_t) {.index = index,
                                                  .executor = executor,
                                                  .outcome = &outcome,
                                                  .key = key_of(4242),
                                                  .entries = &entries});
    expect_at(entries == 0, __FILE__, __LINE__, "An absent key counts zero");
    expect_at(outcome.status == usearch_status_success_k, __FILE__, __LINE__, "An absent key is not an error");

    usearch_executor_free(executor);
    usearch_dense_free(index);
}

/** The library-wide probes, which take no handle and cannot fail. */
void test_library_probes(void) {
    expect_at(usearch_capabilities_compiled() != NULL, __FILE__, __LINE__,
              "The compiled capabilities must be reported");
    expect_at(usearch_capabilities_available() != NULL, __FILE__, __LINE__,
              "The available capabilities must be reported");

    // A machine without GPUs lists none, so the call succeeds everywhere and names what it lists.
    usearch_gpu_device_t devices[8];
    size_t devices_count = 0;
    usearch_outcome_t outcome = {0};
    usearch_gpu_devices(&(usearch_gpu_devices_t) {.outcome = &outcome,
                                                  .devices = devices,
                                                  .devices_capacity = 8,
                                                  .devices_count = &devices_count});
    expect_at(outcome.status == usearch_status_success_k, __FILE__, __LINE__, "Listing the GPUs must succeed");
    for (size_t position = 0; position != devices_count && position != 8; ++position) {
        expect_at(devices[position].vendor != usearch_gpu_vendor_unknown_k, __FILE__, __LINE__,
                  "A listed GPU names its vendor");
        printf("- GPU %zu: %s, %s, %u compute units, %llu MiB\n", position, devices[position].name,
               usearch_gpu_class_name(devices[position].gpu_class), devices[position].compute_units,
               (unsigned long long)(devices[position].memory_bytes >> 20));
    }
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
    expect_at(holds, __FILE__, __LINE__, message);
}

/**
 *  @brief Checks one @c _name and @c _from_name pair against @p table, the enumerators it spells.
 *
 *  Every spelling names its value and reads back to it, and no two rows share a spelling. A near
 *  miss of a spelling - upper case, a trailing space or letter, the last letter dropped - and an
 *  empty or null view read back as @p unknown. Every value between the smallest and the largest
 *  enumerator that no row lists spells @p fallback, which reads back as @p unknown; a value past
 *  them would lie outside the range a C++ enumeration may hold.
 */
static void expect_names(char const* what, named_value_t const* table, size_t count, int unknown, char const* fallback,
                         char const* (*name_of)(int), int (*from_name)(usearch_bytes_view_t)) {
    // A negative unknown is no enumerator but the refusal of a parser answering a status.
    int smallest = unknown >= 0 ? unknown : table[0].value, largest = smallest;
    for (size_t row = 0; row != count; ++row) {
        char const* const spelling = table[row].spelling;
        size_t const length = strlen(spelling);
        smallest = table[row].value < smallest ? table[row].value : smallest;
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
    for (int value = smallest; value <= largest; ++value) {
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
static int scalar_from_name(usearch_bytes_view_t name) {
    usearch_scalar_t scalar = usearch_scalar_unknown_k;
    return usearch_scalar_from_name(name, &scalar) == usearch_status_success_k ? (int)scalar : -1;
}
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
static char const* verdict_name_of(int value) { return usearch_refine_verdict_name((usearch_refine_verdict_t)value); }
static int verdict_from_name(usearch_bytes_view_t name) {
    usearch_refine_verdict_t verdict = (usearch_refine_verdict_t)0;
    return usearch_refine_verdict_from_name(name, &verdict) == usearch_status_success_k ? (int)verdict : -1;
}
static char const* phase_name_of(int value) { return usearch_phase_name((usearch_phase_t)value); }
static int phase_from_name(usearch_bytes_view_t name) {
    usearch_phase_t phase = (usearch_phase_t)0;
    return usearch_phase_from_name(name, &phase) == usearch_status_success_k ? (int)phase : -1;
}
static char const* cpu_class_name_of(int value) { return usearch_cpu_class_name((usearch_cpu_class_t)value); }
static int cpu_class_from_name(usearch_bytes_view_t name) {
    usearch_cpu_class_t cpu_class = usearch_cpu_class_unknown_k;
    return usearch_cpu_class_from_name(name, &cpu_class) == usearch_status_success_k ? (int)cpu_class : -1;
}
static char const* gpu_class_name_of(int value) { return usearch_gpu_class_name((usearch_gpu_class_t)value); }
static int gpu_class_from_name(usearch_bytes_view_t name) {
    usearch_gpu_class_t gpu_class = usearch_gpu_class_unknown_k;
    return usearch_gpu_class_from_name(name, &gpu_class) == usearch_status_success_k ? (int)gpu_class : -1;
}
static char const* nvme_class_name_of(int value) { return usearch_nvme_class_name((usearch_nvme_class_t)value); }
static int nvme_class_from_name(usearch_bytes_view_t name) {
    usearch_nvme_class_t nvme_class = usearch_nvme_class_unknown_k;
    return usearch_nvme_class_from_name(name, &nvme_class) == usearch_status_success_k ? (int)nvme_class : -1;
}
static char const* dense_metric_name_of(int value) { return usearch_dense_metric_name((usearch_dense_metric_t)value); }
static int dense_metric_from_name(usearch_bytes_view_t name) {
    usearch_dense_metric_t metric = usearch_dense_metric_unknown_k;
    return usearch_dense_metric_from_name(name, &metric) == usearch_status_success_k ? (int)metric : -1;
}
static char const* sequence_metric_name_of(int value) {
    return usearch_sequence_metric_name((usearch_sequence_metric_t)value);
}
static int sequence_metric_from_name(usearch_bytes_view_t name) {
    usearch_sequence_metric_t metric = usearch_sequence_metric_unknown_k;
    return usearch_sequence_metric_from_name(name, &metric) == usearch_status_success_k ? (int)metric : -1;
}
static char const* shapes_metric_name_of(int value) {
    return usearch_shapes_metric_name((usearch_shapes_metric_t)value);
}
static int shapes_metric_from_name(usearch_bytes_view_t name) {
    usearch_shapes_metric_t metric = usearch_shapes_metric_unknown_k;
    return usearch_shapes_metric_from_name(name, &metric) == usearch_status_success_k ? (int)metric : -1;
}

/** Every @c _name and @c _from_name pair against the spellings its enumerators carry. */
void test_names_round_trip_through_the_library_and_refuse_strangers(void) {
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
    named_value_t const verdicts[] = {
        {usearch_refine_none_k, "none"},           {usearch_refine_improved_k, "improved"},
        {usearch_refine_converged_k, "converged"}, {usearch_refine_stalled_k, "stalled"},
        {usearch_refine_stopped_k, "stopped"},
    };
    named_value_t const phases[] = {
        {usearch_phase_none_k, "none"},       {usearch_phase_copy_k, "copy"},
        {usearch_phase_build_k, "build"},     {usearch_phase_partition_k, "partition"},
        {usearch_phase_scatter_k, "scatter"}, {usearch_phase_health_k, "health"},
        {usearch_phase_heal_k, "heal"},       {usearch_phase_hubs_k, "hubs"},
        {usearch_phase_gather_k, "gather"},   {usearch_phase_restitch_k, "restitch"},
        {usearch_phase_write_k, "write"},     {usearch_phase_done_k, "done"},
    };
    named_value_t const cpu_classes[] = {
        {usearch_cpu_class_serial_k, "serial"},
        {usearch_cpu_class_monoversioned_k, "monoversioned"},
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
        {usearch_cpu_class_ibm_power10_k, "ibm-power10"},
        {usearch_cpu_class_wasm32_v128_k, "wasm32-v128"},
        {usearch_cpu_class_wasm64_extended_k, "wasm64-extended"},
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
    named_value_t const sequence_metrics[] = {
        {usearch_sequence_metric_levenshtein_bytes_k, "levenshtein"},
        {usearch_sequence_metric_levenshtein_utf8_k, "levenshtein-utf8"},
        {usearch_sequence_metric_needleman_wunsch_k, "needleman-wunsch"},
        {usearch_sequence_metric_overlap_k, "overlap"},
    };
    named_value_t const shapes_metrics[] = {
        {usearch_shapes_metric_chamfer_k, "chamfer"},
        {usearch_shapes_metric_hausdorff_k, "hausdorff"},
        {usearch_shapes_metric_gromov_wasserstein_k, "gromov-wasserstein"},
        {usearch_shapes_metric_wasserstein_procrustes_k, "wasserstein-procrustes"},
    };
    size_t const row_bytes = sizeof(named_value_t);
    // Parsers answering a status read a refused spelling as -1, which no enumerator holds.
    expect_names("status", statuses, sizeof(statuses) / row_bytes, -1, "unrecognized", status_name_of,
                 status_from_name);
    expect_names("scalar", scalars, sizeof(scalars) / row_bytes, -1, "unknown", scalar_name_of, scalar_from_name);
    expect_names("residency", residencies, sizeof(residencies) / row_bytes, -1, "unknown", residency_name_of,
                 residency_from_name);
    expect_names("traversal", traversals, sizeof(traversals) / row_bytes, -1, "unknown", traversal_name_of,
                 traversal_from_name);
    expect_names("mood", moods, sizeof(moods) / row_bytes, -1, "unknown", mood_name_of, mood_from_name);
    expect_names("refine verdict", verdicts, sizeof(verdicts) / row_bytes, -1, "unknown", verdict_name_of,
                 verdict_from_name);
    expect_names("phase", phases, sizeof(phases) / row_bytes, -1, "unknown", phase_name_of, phase_from_name);
    usearch_bytes_view_t const bogus = text_view("bogus", 5);
    usearch_status_t status = usearch_status_success_k;
    usearch_residency_t residency = usearch_residency_ram_k;
    usearch_traversal_t traversal = usearch_traversal_walk_k;
    usearch_mood_t mood = usearch_mood_parked_k;
    expect_at(usearch_status_from_name(bogus, &status) == usearch_status_invalid_argument_k, __FILE__, __LINE__,
              "A bogus status is refused");
    expect_at(usearch_residency_from_name(bogus, &residency) == usearch_status_invalid_argument_k, __FILE__, __LINE__,
              "A bogus residency is refused");
    expect_at(usearch_traversal_from_name(bogus, &traversal) == usearch_status_invalid_argument_k, __FILE__, __LINE__,
              "A bogus traversal is refused");
    expect_at(usearch_mood_from_name(bogus, &mood) == usearch_status_invalid_argument_k, __FILE__, __LINE__,
              "A bogus mood is refused");
    expect_at(status == usearch_status_success_k && residency == usearch_residency_ram_k &&
                  traversal == usearch_traversal_walk_k && mood == usearch_mood_parked_k,
              __FILE__, __LINE__, "A refused spelling leaves its output alone");
    expect_names("CPU class", cpu_classes, sizeof(cpu_classes) / row_bytes, -1, "unknown", cpu_class_name_of,
                 cpu_class_from_name);
    expect_names("GPU class", gpu_classes, sizeof(gpu_classes) / row_bytes, -1, "unknown", gpu_class_name_of,
                 gpu_class_from_name);
    expect_names("NVMe class", nvme_classes, sizeof(nvme_classes) / row_bytes, -1, "unknown", nvme_class_name_of,
                 nvme_class_from_name);
    expect_names("dense metric", dense_metrics, sizeof(dense_metrics) / row_bytes, -1, "unknown", dense_metric_name_of,
                 dense_metric_from_name);
    expect_names("sequence metric", sequence_metrics, sizeof(sequence_metrics) / row_bytes, -1, "unknown",
                 sequence_metric_name_of, sequence_metric_from_name);
    expect_names("shapes metric", shapes_metrics, sizeof(shapes_metrics) / row_bytes, -1, "unknown",
                 shapes_metric_name_of, shapes_metric_from_name);
    usearch_scalar_t scalar = usearch_scalar_f32_k;
    usearch_nvme_class_t nvme_class = usearch_nvme_class_posix_k;
    usearch_dense_metric_t dense_metric = usearch_dense_metric_inner_k;
    expect_at(usearch_scalar_from_name(bogus, &scalar) == usearch_status_invalid_argument_k, __FILE__, __LINE__,
              "A bogus scalar is refused");
    expect_at(usearch_nvme_class_from_name(bogus, &nvme_class) == usearch_status_invalid_argument_k, __FILE__, __LINE__,
              "A bogus NVMe class is refused");
    expect_at(usearch_dense_metric_from_name(bogus, &dense_metric) == usearch_status_invalid_argument_k, __FILE__,
              __LINE__, "A bogus dense metric is refused");
    expect_at(scalar == usearch_scalar_f32_k && nvme_class == usearch_nvme_class_posix_k &&
                  dense_metric == usearch_dense_metric_inner_k,
              __FILE__, __LINE__, "A refused spelling leaves its output alone");
    usearch_shapes_metric_t shapes_metric = usearch_shapes_metric_hausdorff_k;
    expect_at(usearch_shapes_metric_from_name(text_view("gromov_wasserstein", 18), &shapes_metric) ==
                  usearch_status_invalid_argument_k,
              __FILE__, __LINE__, "An underscored shapes metric is refused");
    expect_at(shapes_metric == usearch_shapes_metric_hausdorff_k, __FILE__, __LINE__,
              "A refused spelling leaves its output alone");
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
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
    usearch_dense_insert(&(usearch_dense_insert_t) {
        .index = index,
        .executor = executor,
        .outcome = &outcome,
        .count = deferred_rows,
        .keys = {keys, sizeof(usearch_key_t)},
        .vectors = {rows, deferred_dimensions * sizeof(float), deferred_dimensions, usearch_scalar_f32_k}});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
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
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
}

/** Opens a RAM index under @p ceiling and reports the class it picked, or the refusing status. */
static usearch_status_t ceiling_picks(usearch_cpu_class_t ceiling, usearch_cpu_class_t* picked) {
    usearch_outcome_t outcome = {0};
    usearch_dense_index_t index = NULL;
    usearch_dense_init_ram(&(usearch_dense_init_ram_t) {.index = &index,
                                                        .outcome = &outcome,
                                                        .metric = usearch_dense_metric_angular_k,
                                                        .quantization = usearch_scalar_bf16_k,
                                                        .dimensions = 8,
                                                        .ram = {.cpu_class_ceiling = ceiling}});
    if (outcome.status != usearch_status_success_k) return outcome.status;
    usearch_dense_index_description_t description = {0};
    usearch_dense_describe(
        &(usearch_dense_describe_t) {.index = index, .outcome = &outcome, .description = &description});
    usearch_dense_free(index);
    *picked = description.cpu_class;
    return outcome.status;
}

/** A RAM index capped at the floor class, serial or a monoversioned build's one class, opens on any
 *  CPU the build runs on and describes itself under that class and no GPU class; a ceiling of
 *  another architecture narrows to what both share rather than refusing, while a value no class
 *  takes is refused; and an NVMe executor submits through the POSIX class it names. */
void test_ceilings_and_fill(void) {
    usearch_outcome_t outcome = {0};
    usearch_dense_index_t index = NULL;
    usearch_dense_index_description_t description = {0};
    usearch_dense_init_ram(&(usearch_dense_init_ram_t) {.index = &index,
                                                        .outcome = &outcome,
                                                        .metric = usearch_dense_metric_angular_k,
                                                        .quantization = usearch_scalar_bf16_k,
                                                        .dimensions = 8});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    usearch_dense_describe(
        &(usearch_dense_describe_t) {.index = index, .outcome = &outcome, .description = &description});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    usearch_dense_free(index);
    // A monoversioned build compiles its one class alone, so the serial floor is not there to pick.
    usearch_cpu_class_t const ceiling = description.cpu_class == usearch_cpu_class_monoversioned_k
                                            ? usearch_cpu_class_monoversioned_k
                                            : usearch_cpu_class_serial_k;
    index = NULL;
    usearch_dense_init_ram(&(usearch_dense_init_ram_t) {.index = &index,
                                                        .outcome = &outcome,
                                                        .metric = usearch_dense_metric_angular_k,
                                                        .quantization = usearch_scalar_bf16_k,
                                                        .dimensions = 8,
                                                        .ram = {.cpu_class_ceiling = ceiling}});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    usearch_dense_reserve(&(usearch_dense_reserve_t) {.index = index, .outcome = &outcome, .capacity = 2});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    float const rows[16] = {1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1};
    usearch_key_t const keys[2] = {key_of(1), key_of(2)};
    usearch_dense_insert(&(usearch_dense_insert_t) {.index = index,
                                                    .outcome = &outcome,
                                                    .count = 2,
                                                    .keys = {keys, sizeof(usearch_key_t)},
                                                    .vectors = {rows, 8 * sizeof(float), 8, usearch_scalar_f32_k}});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    usearch_dense_describe(
        &(usearch_dense_describe_t) {.index = index, .outcome = &outcome, .description = &description});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    expect_at(description.shards_count >= 1, __FILE__, __LINE__, "A fresh index spans at least one shard");
    expect_at(description.size == (size_t)2, __FILE__, __LINE__, "The description counts the rows inserted");
    expect_at(description.cpu_class == ceiling, __FILE__, __LINE__, "A floor ceiling dispatches the floor class");
    expect_at(description.gpu_class == usearch_gpu_class_unknown_k, __FILE__, __LINE__,
              "A RAM index names no GPU class");
    expect_at(description.load_path == usearch_load_path_none_k, __FILE__, __LINE__, "A RAM index names no load path");
    usearch_dense_free(index);

    usearch_cpu_class_t best = usearch_cpu_class_unknown_k, picked = usearch_cpu_class_unknown_k;
    if (!expect_at(ceiling_picks(usearch_cpu_class_unknown_k, &best) == usearch_status_success_k, __FILE__, __LINE__,
                   "No ceiling picks a class"))
        return;
    if (!expect_at(ceiling_picks(best, &picked) == usearch_status_success_k, __FILE__, __LINE__,
                   "The machine's own pick admits itself"))
        return;
    expect_at(picked == best, __FILE__, __LINE__, "A ceiling at the machine's own pick changes nothing");
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    usearch_cpu_class_t const foreign = usearch_cpu_class_aws_g2_k;
#else
    usearch_cpu_class_t const foreign = usearch_cpu_class_intel_haswell_k;
#endif
    picked = usearch_cpu_class_unknown_k;
    usearch_status_t const narrowed = ceiling_picks(foreign, &picked);
    if (ceiling == usearch_cpu_class_monoversioned_k)
        expect_at(narrowed == usearch_status_unsupported_k, __FILE__, __LINE__,
                  "A monoversioned build has no class under a foreign one");
    else {
        expect_at(narrowed == usearch_status_success_k, __FILE__, __LINE__,
                  "A ceiling the CPU lacks narrows rather than refuses");
        expect_at(picked == usearch_cpu_class_serial_k, __FILE__, __LINE__,
                  "A foreign ceiling narrows to the serial floor");
    }
    usearch_cpu_class_t const past_every_class = (usearch_cpu_class_t)(usearch_cpu_class_wasm64_extended_k + 1);
    expect_at(ceiling_picks(past_every_class, &picked) == usearch_status_invalid_argument_k, __FILE__, __LINE__,
              "A ceiling no class takes is refused");

    usearch_executor_t storage = NULL;
    usearch_executor_for_nvme(&(usearch_executor_for_nvme_t) {.executor = &storage,
                                                              .outcome = &outcome,
                                                              .threads = 1,
                                                              .nvme_class = usearch_nvme_class_posix_k});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    usearch_executor_free(storage);
}

/** A deferred search lands what a blocking one does, though its descriptor died at submission; one
 *  thread is the caller's own, so it runs at submission, and so does a call without an executor. */
void test_deferred_search(void) {
    float* const rows = create_vectors(deferred_rows, deferred_dimensions);
    // Landmarks are drawn per thread, so only one as wide as a call without an executor
    // draws the same.
    usearch_executor_t const executor = create_executor(1);
    usearch_dense_index_t const index = create_filled_index(rows, executor);
    search_outputs_t blocking = {.arena = create_arena(executor)};
    search_outputs_t deferred = {.arena = create_arena(executor)};

    search_every_row(index, executor, usearch_completion_blocking_k, rows, &blocking);
    if (!expect_at(!blocking.outcome.message, __FILE__, __LINE__, blocking.outcome.message)) return;

    search_every_row(index, executor, usearch_completion_deferred_k, rows, &deferred);
    if (!expect_at(usearch_outcome_status(&deferred.outcome) == usearch_status_success_k, __FILE__, __LINE__,
                   "A pool needing its caller runs a deferred call at submission"))
        return;
    expect_at(same_matches(&blocking.matches, &deferred.matches), __FILE__, __LINE__,
              "A deferred search must land the matches a blocking one does");

    // An arena made by an executor serves a call without one, since both compute on the host.
    search_every_row(index, NULL, usearch_completion_deferred_k, rows, &deferred);
    expect_at(deferred.outcome.status == usearch_status_success_k, __FILE__, __LINE__,
              "A call without an executor is final when it returns");
    expect_at(same_matches(&blocking.matches, &deferred.matches), __FILE__, __LINE__, "Every receiver must agree");

    usearch_arena_free(deferred.arena);
    usearch_arena_free(blocking.arena);
    usearch_dense_free(index);
    usearch_executor_free(executor);
    free(rows);
}

/** A deferred call on a pool of its own threads finishes while its submitter only polls. */
void test_deferred_polling(void) {
    float* const rows = create_vectors(deferred_rows, deferred_dimensions);
    usearch_executor_t const executor = create_executor(2);
    usearch_dense_index_t const index = create_filled_index(rows, executor);
    search_outputs_t outputs = {.arena = create_arena(executor)};

    search_every_row(index, executor, usearch_completion_deferred_k, rows, &outputs);
    time_t const deadline = time(NULL) + 30;
    while (usearch_outcome_status(&outputs.outcome) == usearch_status_deferred_k && time(NULL) < deadline) {}
    if (!expect_at(usearch_outcome_status(&outputs.outcome) == usearch_status_success_k, __FILE__, __LINE__,
                   "Polling alone finishes the call"))
        return;
    for (size_t row = 0; row != deferred_rows; ++row)
        expect_at(outputs.matches.counts[row] == deferred_wanted, __FILE__, __LINE__,
                  "A polled call lands its matches");

    usearch_arena_free(outputs.arena);
    usearch_dense_free(index);
    usearch_executor_free(executor);
    free(rows);
}

/** While a deferred call holds its executor on a pool thread, a call from this thread is refused at
 *  once rather than overlapping it, and the executor serves again once the first call ended. */
void test_contended_from_another_thread(void) {
    float* const rows = create_vectors(deferred_rows, deferred_dimensions);
    usearch_executor_t const executor = create_executor(2);
    usearch_dense_index_t const index = create_filled_index(rows, executor);
    search_outputs_t holding = {.arena = create_arena(executor)};
    search_outputs_t second = {.arena = create_arena(executor)};
    gate_t gate = {0, 0};

    search_every_row_through(index, executor, usearch_completion_deferred_k, rows, &holding, &gate);
    // An executor without threads of its own, as the stand-in's, ran the call at submission.
    if (usearch_outcome_status(&holding.outcome) == usearch_status_deferred_k) {
        expect_at(entered_soon(&gate), __FILE__, __LINE__, "A delegated call reaches its progress handler");
        search_every_row(index, executor, usearch_completion_blocking_k, rows, &second);
        expect_at(second.outcome.status == usearch_status_contended_k, __FILE__, __LINE__,
                  "A second call on a held executor is refused");
        search_every_row(index, executor, usearch_completion_deferred_k, rows, &second);
        expect_at(second.outcome.status == usearch_status_contended_k, __FILE__, __LINE__,
                  "A deferred one is refused at submission");
        gate.released = 1;
    }
    if (!expect_at(usearch_outcome_wait(&holding.outcome) == usearch_status_success_k, __FILE__, __LINE__,
                   "The holding call finishes"))
        return;
    search_every_row(index, executor, usearch_completion_blocking_k, rows, &second);
    if (!expect_at(!second.outcome.message, __FILE__, __LINE__, second.outcome.message)) return;

    usearch_arena_free(second.arena);
    usearch_arena_free(holding.arena);
    usearch_dense_free(index);
    usearch_executor_free(executor);
    free(rows);
}

/** A drain returns once the deferred call ended, whichever mood it then leaves the threads in, and
 *  the executor serves the next call after either. */
void test_drain_then_mood(void) {
    float* const rows = create_vectors(deferred_rows, deferred_dimensions);
    usearch_executor_t const executor = create_executor(2);
    usearch_dense_index_t const index = create_filled_index(rows, executor);
    search_outputs_t outputs = {.arena = create_arena(executor)};

    search_every_row(index, executor, usearch_completion_deferred_k, rows, &outputs);
    drain(executor, usearch_mood_parked_k);
    if (!expect_at(usearch_outcome_status(&outputs.outcome) == usearch_status_success_k, __FILE__, __LINE__,
                   "A drain waits out the call"))
        return;
    search_every_row(index, executor, usearch_completion_deferred_k, rows, &outputs);
    drain(executor, usearch_mood_spinning_k);
    if (!expect_at(usearch_outcome_status(&outputs.outcome) == usearch_status_success_k, __FILE__, __LINE__,
                   "Parked threads wake for a call"))
        return;
    search_every_row(index, executor, usearch_completion_blocking_k, rows, &outputs);
    if (!expect_at(!outputs.outcome.message, __FILE__, __LINE__, outputs.outcome.message)) return;

    usearch_arena_free(outputs.arena);
    usearch_dense_free(index);
    usearch_executor_free(executor);
    free(rows);
}

/** Freeing an index waits for the deferred call naming it before its memory goes. */
void test_free_waits_for_named_call(void) {
    float* const rows = create_vectors(deferred_rows, deferred_dimensions);
    usearch_executor_t const executor = create_executor(2);
    usearch_dense_index_t const index = create_filled_index(rows, executor);
    search_outputs_t outputs = {.arena = create_arena(executor)};

    search_every_row(index, executor, usearch_completion_deferred_k, rows, &outputs);
    usearch_dense_free(index);
    if (!expect_at(usearch_outcome_status(&outputs.outcome) == usearch_status_success_k, __FILE__, __LINE__,
                   "A free returns only once the calls naming the handle ended"))
        return;
    for (size_t row = 0; row != deferred_rows; ++row)
        expect_at(outputs.matches.counts[row] == deferred_wanted, __FILE__, __LINE__,
                  "A call a free waited for lands its matches");

    usearch_arena_free(outputs.arena);
    usearch_executor_free(executor);
    free(rows);
}

/** A search without an executor that a delegated call runs from its last progress report, held at
 *  its own last report, so this thread meets it in flight. */
typedef struct held_search_t {
    usearch_dense_index_t index;
    float const* rows;
    search_outputs_t outputs;
    gate_t gate;
    int volatile left;
} held_search_t;

static usearch_continuation_t hold_then_leave(size_t completed, size_t total, void* state) {
    held_search_t* const held = (held_search_t*)state;
    usearch_continuation_t const continuation = hold_until_released(completed, total, &held->gate);
    if (completed == total) held->left = 1;
    return continuation;
}

static usearch_continuation_t search_without_executor(size_t completed, size_t total, void* state) {
    held_search_t* const held = (held_search_t*)state;
    if (completed != total) return usearch_continue_k;
    usearch_dense_search(&(usearch_dense_search_t) {
        .index = held->index,
        .outcome = &held->outputs.outcome,
        .count = deferred_rows,
        .queries = {held->rows, deferred_dimensions * sizeof(float), deferred_dimensions, usearch_scalar_f32_k},
        .matches = &held->outputs.matches,
        .arena = held->outputs.arena,
        .wanted = deferred_wanted,
        .progress = {hold_then_leave, held, 1}});
    return usearch_continue_k;
}

/** Freeing an index waits for a call without an executor naming it, which a delegated call runs on
 *  a pool thread while this thread frees. */
void test_free_waits_for_call_without_executor(void) {
    float* const rows = create_vectors(deferred_rows, deferred_dimensions);
    usearch_executor_t const executor = create_executor(2);
    usearch_dense_index_t const index = create_filled_index(rows, executor);
    held_search_t held = {.index = create_filled_index(rows, NULL),
                          .rows = rows,
                          .outputs = {.arena = create_arena(NULL)}};
    search_outputs_t outer = {.arena = create_arena(executor)};

    usearch_dense_search(&(usearch_dense_search_t) {
        .index = index,
        .executor = executor,
        .outcome = &outer.outcome,
        .completion = usearch_completion_deferred_k,
        .count = deferred_rows,
        .queries = {rows, deferred_dimensions * sizeof(float), deferred_dimensions, usearch_scalar_f32_k},
        .matches = &outer.matches,
        .arena = outer.arena,
        .wanted = deferred_wanted,
        .progress = {search_without_executor, &held, 1}});
    expect_at(entered_soon(&held.gate), __FILE__, __LINE__,
              "The delegated call reaches the search it runs without an executor");
    usearch_dense_free(held.index);
    expect_at(held.left, __FILE__, __LINE__,
              "A free returns only once the call without an executor naming the handle ended");
    if (!expect_at(usearch_outcome_wait(&outer.outcome) == usearch_status_success_k, __FILE__, __LINE__,
                   "The delegated call finishes"))
        return;
    expect_at(held.outputs.outcome.status == usearch_status_success_k, __FILE__, __LINE__,
              "The call a free waited for finishes");

    usearch_arena_free(held.outputs.arena);
    usearch_arena_free(outer.arena);
    usearch_dense_free(index);
    usearch_executor_free(executor);
    free(rows);
}

/** A deferred call holds its arena until just before it is final, so another call naming it
 *  meanwhile is refused, and the arena serves again once the deferred call ended. */
void test_deferred_holds_arena(void) {
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
        expect_at(entered_soon(&gate), __FILE__, __LINE__, "A delegated call reaches its progress handler");
        // Without an executor the get runs at once, so it meets the deferred call's hold.
        usearch_dense_get(&get);
        expect_at(outcome.status == usearch_status_contended_k, __FILE__, __LINE__,
                  "An arena a deferred call holds refuses another call");
        expect_at(vectors.data == NULL, __FILE__, __LINE__, "A refused call leaves its outputs NULL");
        gate.released = 1;
    }
    if (!expect_at(usearch_outcome_wait(&outputs.outcome) == usearch_status_success_k, __FILE__, __LINE__,
                   "A waited search must finish"))
        return;
    expect_at(outputs.matches.counts[0] == deferred_wanted, __FILE__, __LINE__,
              "The deferred call lands its matches in the held arena");
    usearch_dense_get(&get);
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    if (!expect_at(vectors.data != NULL, __FILE__, __LINE__, "A final call lets go of its arena")) return;

    usearch_arena_free(outputs.arena);
    usearch_dense_free(index);
    usearch_executor_free(executor);
    free(rows);
}

/** A three-part checkpoint loaded onto one GPU keeps a shard per part on it, loads every entry, and
 *  reaches exactly the nodes the saved graph reached. Skipped without a GPU. */
char const* test_vram_parts_share_a_device(size_t const collection_size, size_t const dimensions) {
    usearch_outcome_t outcome = {0};
    size_t devices_count = 0;
    usearch_gpu_devices(&(usearch_gpu_devices_t) {.outcome = &outcome, .devices_count = &devices_count});
    if (!devices_count) return NULL;

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
    if (outcome.status == usearch_status_device_k) return no_gpu_device_k;
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
    usearch_executor_t executor = NULL;
    usearch_executor_for_vram(
        &(usearch_executor_for_vram_t) {.executor = &executor, .outcome = &outcome, .vram = thrice});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;

    float* const data = create_vectors(collection_size, dimensions);
    usearch_key_t* const keys = (usearch_key_t*)malloc(collection_size * sizeof(usearch_key_t));
    if (!expect_at(keys, __FILE__, __LINE__, "Failed to allocate keys")) return NULL;
    for (size_t row = 0; row != collection_size; ++row) keys[row] = key_of(row);
    size_t const row_bytes = dimensions * sizeof(float);
    usearch_dense_reserve(&(usearch_dense_reserve_t) {.index = index,
                                                      .executor = executor,
                                                      .outcome = &outcome,
                                                      .capacity = collection_size});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
    usearch_dense_insert(&(usearch_dense_insert_t) {.index = index,
                                                    .executor = executor,
                                                    .outcome = &outcome,
                                                    .count = collection_size,
                                                    .keys = {keys, sizeof(usearch_key_t)},
                                                    .vectors = {data, row_bytes, dimensions, usearch_scalar_f32_k}});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
    // Insertions may leave islands, so the reload is judged against the saved graph's own audit.
    usearch_reachability_t saved_reachability = {0};
    usearch_dense_health(&(usearch_dense_health_t) {.index = index,
                                                    .executor = executor,
                                                    .outcome = &outcome,
                                                    .reachability = &saved_reachability});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
    char const path[] = "usearch-test-c-parts.safegraphs";
    usearch_bytes_view_t const path_view = {(uint8_t const*)path, sizeof(path) - 1};
    usearch_dense_save(
        &(usearch_dense_save_t) {.index = index, .executor = executor, .outcome = &outcome, .path = path_view});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
    usearch_executor_free(executor);
    usearch_dense_free(index);

    // An executor over the one named device drives every shard there, all on its one stream.
    usearch_executor_for_vram(
        &(usearch_executor_for_vram_t) {.executor = &executor, .outcome = &outcome, .vram = once});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
    usearch_dense_load_vram(&(usearch_dense_load_vram_t) {.index = &index,
                                                          .executor = executor,
                                                          .outcome = &outcome,
                                                          .path = path_view,
                                                          .vram = once});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;

    usearch_dense_index_description_t description = {0};
    usearch_dense_describe(&(usearch_dense_describe_t) {.index = index,
                                                        .executor = executor,
                                                        .outcome = &outcome,
                                                        .description = &description});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
    expect_at(description.size == collection_size, __FILE__, __LINE__,
              "The parts must load every saved row onto one device");
    usearch_reachability_t reachability = {0};
    usearch_dense_health(&(usearch_dense_health_t) {.index = index,
                                                    .executor = executor,
                                                    .outcome = &outcome,
                                                    .reachability = &reachability});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
    expect_at(
        reachability.audited == saved_reachability.audited && reachability.reachable == saved_reachability.reachable,
        __FILE__, __LINE__, "The loaded parts must reach exactly the rows the saved ones did");

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
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
    expect_at(matches.keys && matches.counts, __FILE__, __LINE__, "A VRAM search must land its matches in the arena");

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
    return NULL;
}

/** An exact scan on the first GPU scores every entry once per query and refuses a predicate as a
 *  walk does where the vendor scans, and refuses with @c usearch_status_unsupported_k elsewhere. */
char const* test_vram_scan(size_t const collection_size, size_t const dimensions) {
    usearch_outcome_t outcome = {0};
    usearch_gpu_device_t device = {0};
    size_t devices_count = 0;
    usearch_gpu_devices(&(usearch_gpu_devices_t) {.outcome = &outcome,
                                                  .devices = &device,
                                                  .devices_capacity = 1,
                                                  .devices_count = &devices_count});
    if (!devices_count) return NULL;

    int32_t const devices[1] = {0};
    usearch_vram_options_t const vram = {.devices = devices, .devices_count = 1};
    usearch_dense_index_t index = NULL;
    usearch_dense_init_vram(&(usearch_dense_init_vram_t) {.index = &index,
                                                          .outcome = &outcome,
                                                          .metric = usearch_dense_metric_euclidean_k,
                                                          .quantization = usearch_scalar_bf16_k,
                                                          .dimensions = dimensions,
                                                          .vram = vram});
    if (outcome.status == usearch_status_device_k) return no_gpu_device_k;
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
    usearch_executor_t executor = NULL;
    usearch_executor_for_vram(
        &(usearch_executor_for_vram_t) {.executor = &executor, .outcome = &outcome, .vram = vram});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;

    float* const data = create_vectors(collection_size, dimensions);
    usearch_key_t* const keys = (usearch_key_t*)malloc(collection_size * sizeof(usearch_key_t));
    if (!expect_at(keys, __FILE__, __LINE__, "Failed to allocate keys")) return NULL;
    for (size_t row = 0; row != collection_size; ++row) keys[row] = key_of(row);
    size_t const row_bytes = dimensions * sizeof(float);
    usearch_dense_reserve(&(usearch_dense_reserve_t) {.index = index,
                                                      .executor = executor,
                                                      .outcome = &outcome,
                                                      .capacity = collection_size});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
    usearch_dense_insert(&(usearch_dense_insert_t) {.index = index,
                                                    .executor = executor,
                                                    .outcome = &outcome,
                                                    .count = collection_size,
                                                    .keys = {keys, sizeof(usearch_key_t)},
                                                    .vectors = {data, row_bytes, dimensions, usearch_scalar_f32_k}});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;

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
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
        expect_at(matches.keys && matches.counts, __FILE__, __LINE__, "A VRAM scan lands its matches in the arena");
        expect_at(stats.computed_distances == collection_size * collection_size, __FILE__, __LINE__,
                  "A scan scores every row per query");
        size_t widest_batch = 0;
        search.predicate = (usearch_predicate_t) {admits_even, &widest_batch};
        usearch_dense_search(&search);
        expect_at(outcome.status == usearch_status_unsupported_k, __FILE__, __LINE__,
                  "A host predicate cannot narrow a device scan");
    }
    else expect_at(outcome.status == usearch_status_unsupported_k, __FILE__, __LINE__, "Only NVIDIA devices scan");

    usearch_arena_free(arena);
    usearch_dense_free(index);
    usearch_executor_free(executor);
    free(keys);
    free(data);
    return NULL;
}

static void release_schema_of_test(struct ArrowSchema* schema) { schema->release = NULL; }

static void release_array_of_test(struct ArrowArray* array) { array->release = NULL; }

/** Releases an exported column, the schema first, as its consumer may. */
static void release_column(struct ArrowSchema* schema, struct ArrowDeviceArray* array) {
    schema->release(schema);
    array->array.release(&array->array);
    expect_at(!schema->release && !array->array.release, __FILE__, __LINE__, "A release marks its structure released");
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
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
    usearch_sequence_insert(&(usearch_sequence_insert_t) {.index = index,
                                                          .executor = executor,
                                                          .outcome = &outcome,
                                                          .count = 5,
                                                          .keys = {keys, sizeof(usearch_key_t)},
                                                          .sequences = tape_of(bytes, offsets, usearch_scalar_u8_k)});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
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
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return NULL;
    expect_at(written == 1, __FILE__, __LINE__, "A get finds the long string");
    return (uint8_t*)got.data;
}

void test_arrow_rows_and_keys(void) {
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
        if (!expect_at(usearch_rows_view_to_arrow(rows, 3, arena, &schema, &array) == usearch_status_success_k,
                       __FILE__, __LINE__, "Every scalar exports"))
            return;
        expect_at(!strcmp(schema.format, cases[index_of_case].format), __FILE__, __LINE__,
                  "A row's format spells its width");
        expect_at(!cases[index_of_case].element || !strcmp(schema.children[0]->format, cases[index_of_case].element),
                  __FILE__, __LINE__, "An element's format spells its scalar");
        expect_at(array.device_type == ARROW_DEVICE_CPU && array.array.length == 3, __FILE__, __LINE__,
                  "Host rows export as host rows");

        usearch_rows_view_t viewed = {0};
        size_t viewed_count = 0;
        if (!expect_at(
                usearch_rows_view_from_arrow(&schema, &array.array, &viewed, &viewed_count) == usearch_status_success_k,
                __FILE__, __LINE__, "Every scalar imports"))
            return;
        expect_at(viewed.data == (void const*)bytes && viewed.stride == rows.stride && viewed.columns == rows.columns &&
                      viewed.scalar == rows.scalar && viewed_count == 3,
                  __FILE__, __LINE__, "An import views the very rows an export named");

        array.array.offset = 1, array.array.length = 2;
        if (!expect_at(
                usearch_rows_view_from_arrow(&schema, &array.array, &viewed, &viewed_count) == usearch_status_success_k,
                __FILE__, __LINE__, "A slice imports"))
            return;
        expect_at(viewed.data == (void const*)(bytes + rows.stride) && viewed_count == 2, __FILE__, __LINE__,
                  "A slice applies its offset");
        release_column(&schema, &array);
    }

    struct ArrowSchema schema = {0};
    struct ArrowDeviceArray array = {0};
    usearch_rows_view_t const uneven = {bytes, 2, 3, usearch_scalar_u4x2_k};
    expect_at(usearch_rows_view_to_arrow(uneven, 1, arena, &schema, &array) == usearch_status_unsupported_layout_k,
              __FILE__, __LINE__, "Sub-byte rows fill whole bytes");

    usearch_keys_view_t const keys = {(usearch_key_t const*)bytes, sizeof(usearch_key_t)};
    if (!expect_at(usearch_keys_view_to_arrow(keys, 3, arena, &schema, &array) == usearch_status_success_k, __FILE__,
                   __LINE__, "Keys export"))
        return;
    expect_at(!strcmp(schema.format, "w:16") && schema.metadata != NULL, __FILE__, __LINE__,
              "Keys export as tagged 16-byte values");
    usearch_keys_view_t viewed_keys = {0};
    size_t viewed_count = 0;
    if (!expect_at(usearch_keys_view_from_arrow(&schema, &array.array, &viewed_keys, &viewed_count) ==
                       usearch_status_success_k,
                   __FILE__, __LINE__, "Keys import"))
        return;
    expect_at(viewed_keys.data == keys.data && viewed_keys.stride == sizeof(usearch_key_t) && viewed_count == 3,
              __FILE__, __LINE__, "An import views the very keys an export named");
    release_column(&schema, &array);

    usearch_keys_view_t const strided = {(usearch_key_t const*)bytes, 2 * sizeof(usearch_key_t)};
    expect_at(usearch_keys_view_to_arrow(strided, 2, arena, &schema, &array) == usearch_status_unsupported_layout_k,
              __FILE__, __LINE__, "An export takes packed keys alone");
    usearch_key_t const outside[2] = {key_of(1), key_of(2)};
    usearch_keys_view_t const foreign = {outside, sizeof(usearch_key_t)};
    expect_at(usearch_keys_view_to_arrow(foreign, 2, arena, &schema, &array) == usearch_status_invalid_argument_k,
              __FILE__, __LINE__, "An export takes the arena's own outputs alone");

    usearch_sequence_free(index);
    usearch_arena_free(arena);
    usearch_executor_free(executor);
}

void test_arrow_refusals(void) {
    uint8_t const bytes[32] = {0};
    uint8_t const validity = 0xFF;
    void const* buffers[2] = {NULL, bytes};
    struct ArrowSchema schema = {.format = "w:16", .release = &release_schema_of_test};
    struct ArrowArray array = {.length = 2, .n_buffers = 2, .buffers = buffers, .release = &release_array_of_test};
    usearch_keys_view_t keys = {0};
    usearch_rows_view_t rows = {0};
    usearch_tape_view_t tape = {0};
    size_t count = 0;

    if (!expect_at(usearch_keys_view_from_arrow(&schema, &array, &keys, &count) == usearch_status_success_k, __FILE__,
                   __LINE__, "A column with no validity buffer holds no null"))
        return;
    array.null_count = -1;
    if (!expect_at(usearch_keys_view_from_arrow(&schema, &array, &keys, &count) == usearch_status_success_k, __FILE__,
                   __LINE__, "An unknown count of nulls without a validity buffer rules them out"))
        return;
    array.null_count = 1;
    expect_at(usearch_keys_view_from_arrow(&schema, &array, &keys, &count) == usearch_status_invalid_argument_k,
              __FILE__, __LINE__, "A counted null is refused");
    buffers[0] = &validity, array.null_count = -1;
    expect_at(usearch_keys_view_from_arrow(&schema, &array, &keys, &count) == usearch_status_invalid_argument_k,
              __FILE__, __LINE__, "An unknown count of nulls beside a validity buffer is refused");
    buffers[0] = NULL, array.null_count = 0;

    array.length = -1;
    expect_at(usearch_keys_view_from_arrow(&schema, &array, &keys, &count) == usearch_status_malformed_k, __FILE__,
              __LINE__, "A negative length is malformed");
    array.length = 2;
    schema.dictionary = &schema;
    expect_at(usearch_keys_view_from_arrow(&schema, &array, &keys, &count) == usearch_status_unsupported_layout_k,
              __FILE__, __LINE__, "A dictionary is refused");
    schema.dictionary = NULL;
    schema.format = "w:8";
    expect_at(usearch_keys_view_from_arrow(&schema, &array, &keys, &count) == usearch_status_unsupported_layout_k,
              __FILE__, __LINE__, "Keys are 16 bytes wide");
    expect_at(usearch_rows_view_from_arrow(&schema, &array, &rows, &count) == usearch_status_unsupported_layout_k,
              __FILE__, __LINE__, "An untagged fixed-size column holds no rows");
    int32_t const negative_pairs = -1;
    schema.metadata = (char const*)&negative_pairs;
    expect_at(usearch_rows_view_from_arrow(&schema, &array, &rows, &count) == usearch_status_unsupported_layout_k,
              __FILE__, __LINE__, "A negative count in the metadata tags no extension");
    schema.metadata = NULL;
    expect_at(usearch_tape_view_from_arrow(&schema, &array, &tape, &count) == usearch_status_unsupported_layout_k,
              __FILE__, __LINE__, "A fixed-size column is no tape");
    schema.format = "w:16", schema.release = NULL;
    expect_at(usearch_keys_view_from_arrow(&schema, &array, &keys, &count) == usearch_status_invalid_argument_k,
              __FILE__, __LINE__, "A released schema is refused");
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
    expect_at(usearch_rows_view_from_arrow(&list_schema, &list, &rows, &count) == usearch_status_unsupported_layout_k,
              __FILE__, __LINE__, "A list of bools is refused");
    element.format = "f";
    if (!expect_at(usearch_rows_view_from_arrow(&list_schema, &list, &rows, &count) == usearch_status_success_k,
                   __FILE__, __LINE__, "A list of floats imports"))
        return;
    values.null_count = 1;
    expect_at(usearch_rows_view_from_arrow(&list_schema, &list, &rows, &count) == usearch_status_invalid_argument_k,
              __FILE__, __LINE__, "A null among a row's elements is refused");
    values.null_count = 0, values.length = 7;
    expect_at(usearch_rows_view_from_arrow(&list_schema, &list, &rows, &count) == usearch_status_malformed_k, __FILE__,
              __LINE__, "Rows past their child's values are malformed");
}

void test_arrow_tapes(void) {
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
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
        usearch_sequence_insert(&(usearch_sequence_insert_t) {.index = index,
                                                              .executor = executor,
                                                              .outcome = &outcome,
                                                              .count = 4,
                                                              .keys = {keys, sizeof(usearch_key_t)},
                                                              .sequences = tape});
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
        usearch_matches_view_t matches = {0};
        usearch_sequence_search(&(usearch_sequence_search_t) {.index = index,
                                                              .executor = executor,
                                                              .outcome = &outcome,
                                                              .count = 4,
                                                              .queries = tape,
                                                              .matches = &matches,
                                                              .arena = arena,
                                                              .wanted = 1});
        if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
        for (size_t query = 0; query != 4; ++query)
            expect_at(value_of(matches.keys[query]) == query && matches.scores[query] == 0, __FILE__, __LINE__,
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
    expect_at(outcome.status == usearch_status_unsupported_layout_k, __FILE__, __LINE__,
              "An unknown width of offsets is refused");

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
        if (!expect_at(usearch_tape_view_from_arrow(&schema, &array, &tape, &count) == usearch_status_success_k,
                       __FILE__, __LINE__, "Bytes and text import"))
            return;
        expect_at(tape.data == (void const*)bytes && tape.scalar == usearch_scalar_u8_k && count == 3 &&
                      tape.layout == (large ? usearch_tape_u64_k : usearch_tape_u32_k),
                  __FILE__, __LINE__, "A tape keeps its column's bytes and width");
        expect_at(large ? tape.offsets.u64 == wide + 1 : tape.offsets.u32 == narrow + 1, __FILE__, __LINE__,
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
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    bool const wide_outputs = sizeof(size_t) == sizeof(uint64_t);
    expect_at(got.layout == (wide_outputs ? usearch_tape_u64_k : usearch_tape_u32_k) && written == 1, __FILE__,
              __LINE__, "Outputs carry offsets as wide as size_t");
    struct ArrowSchema schema = {0};
    struct ArrowDeviceArray array = {0};
    if (!expect_at(usearch_tape_view_to_arrow(got, written, arena, &schema, &array) == usearch_status_success_k,
                   __FILE__, __LINE__, "Strings export"))
        return;
    expect_at(!strcmp(schema.format, wide_outputs ? "Z" : "z") && array.array.n_buffers == 3 &&
                  array.array.buffers[1] == (void const*)offsets_of(got),
              __FILE__, __LINE__, "Strings export in place, at their width");
    usearch_tape_view_t viewed = {0};
    size_t viewed_count = 0;
    if (!expect_at(
            usearch_tape_view_from_arrow(&schema, &array.array, &viewed, &viewed_count) == usearch_status_success_k,
            __FILE__, __LINE__, "Strings import"))
        return;
    expect_at(viewed.data == got.data && offsets_of(viewed) == offsets_of(got) && viewed_count == 1, __FILE__, __LINE__,
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
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    usearch_shapes_insert(&(usearch_shapes_insert_t) {.index = shapes,
                                                      .executor = executor,
                                                      .outcome = &outcome,
                                                      .count = 2,
                                                      .keys = {shape_keys, sizeof(usearch_key_t)},
                                                      .shapes = tape_of(points, point_offsets, usearch_scalar_f32_k)});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    usearch_shapes_get(&(usearch_shapes_get_t) {.index = shapes,
                                                .executor = executor,
                                                .outcome = &outcome,
                                                .key = key_of(1),
                                                .shapes = &got,
                                                .outputs_scalar = usearch_scalar_f32_k,
                                                .written = &written,
                                                .arena = arena});
    if (!expect_at(!outcome.message, __FILE__, __LINE__, outcome.message)) return;
    if (!expect_at(usearch_tape_view_to_arrow(got, written, arena, &schema, &array) == usearch_status_success_k,
                   __FILE__, __LINE__, "Shapes export"))
        return;
    expect_at(!strcmp(schema.format, wide_outputs ? "+L" : "+l") && !strcmp(schema.children[0]->format, "+w:3") &&
                  !strcmp(schema.children[0]->children[0]->format, "f"),
              __FILE__, __LINE__, "Shapes export as lists of points");
    expect_at(array.array.children[0]->length == 3 && array.array.children[0]->children[0]->length == 9, __FILE__,
              __LINE__, "A list's children count points and scalars");
    if (!expect_at(
            usearch_tape_view_from_arrow(&schema, &array.array, &viewed, &viewed_count) == usearch_status_success_k,
            __FILE__, __LINE__, "Shapes import"))
        return;
    expect_at(viewed.data == got.data && viewed.scalar == usearch_scalar_f32_k && viewed_count == 1 &&
                  offsets_of(viewed)[1] - offsets_of(viewed)[0] == 3,
              __FILE__, __LINE__, "An import views the very points an export named");
    release_column(&schema, &array);

    usearch_shapes_free(shapes);
    usearch_sequence_free(index);
    usearch_arena_free(arena);
    usearch_executor_free(executor);
}

void test_arrow_matches(void) {
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
    if (!expect_at(usearch_matches_tape_view_to_arrow(ragged, 2, arena, &schema, &array) == usearch_status_success_k,
                   __FILE__, __LINE__, "A tape of matches exports"))
        return;
    expect_at(!strcmp(schema.format, "+L") && !strcmp(schema.children[0]->format, "+s") &&
                  array.array.children[0]->length == 3,
              __FILE__, __LINE__, "A tape of matches exports as a large list of structs");
    struct ArrowArray moved = *array.array.children[0];
    array.array.children[0]->release = NULL;
    array.array.release(&array.array);
    expect_at(moved.length == 3 && moved.children[0]->buffers[1] == (void const*)tape_keys, __FILE__, __LINE__,
              "A moved child outlives its released parent");
    moved.release(&moved);
    schema.release(&schema);

    usearch_keys_tape_view_t const links = {tape_keys, tape_offsets};
    if (!expect_at(usearch_keys_tape_view_to_arrow(links, 2, arena, &schema, &array) == usearch_status_success_k,
                   __FILE__, __LINE__, "A tape of keys exports"))
        return;
    expect_at(!strcmp(schema.format, "+L") && !strcmp(schema.children[0]->format, "w:16") &&
                  array.array.children[0]->length == 3,
              __FILE__, __LINE__, "A tape of keys exports as a large list of them");
    release_column(&schema, &array);

    usearch_sequence_free(index);
    usearch_arena_free(arena);
    usearch_executor_free(executor);
}

int main(int argc, char* argv[]) {
    install_test_signal_handlers();
    printf("USearch %s\n", usearch_version_name());
    printf("- Compiled for: %s\n", usearch_capabilities_compiled());
    printf("- This machine: %s\n", usearch_capabilities_available());
    start_test_run(argc, argv);
    srand((unsigned)test_seed);

    start_test_case("test_library_probes");
    test_library_probes();
    finish_test_case(NULL);
    start_test_case("test_names_round_trip_through_the_library_and_refuse_strangers");
    test_names_round_trip_through_the_library_and_refuse_strangers();
    finish_test_case(NULL);
    start_test_case("test_outcome_contract");
    test_outcome_contract(64);
    finish_test_case(NULL);

    size_t const collection_sizes[] = {11, 512};
    size_t const dimension_counts[] = {83, 256};
    char label[128];
    for (size_t size_index = 0; size_index != 2; ++size_index)
        for (size_t dimension_index = 0; dimension_index != 2; ++dimension_index) {
            size_t const rows = collection_sizes[size_index], dimensions = dimension_counts[dimension_index];
            snprintf(label, sizeof(label), "test_init_ram_starts_empty(%zu, %zu)", rows, dimensions);
            start_test_case(label);
            test_init_ram_starts_empty(rows, dimensions);
            finish_test_case(NULL);
            snprintf(label, sizeof(label), "test_add_and_search(%zu, %zu)", rows, dimensions);
            start_test_case(label);
            test_add_and_search(rows, dimensions);
            finish_test_case(NULL);
        }
    start_test_case("test_empty_reads");
    test_empty_reads(dimension_counts[0]);
    finish_test_case(NULL);
    start_test_case("test_add_progress");
    test_add_progress();
    finish_test_case(NULL);
    start_test_case("test_reshard_gather");
    test_reshard_gather(dimension_counts[0]);
    finish_test_case(NULL);
    start_test_case("test_refine_plan");
    test_refine_plan(1536, 4096);
    finish_test_case(NULL);
    start_test_case("test_reshard_cut_plan");
    test_reshard_cut_plan(dimension_counts[0]);
    finish_test_case(NULL);
    start_test_case("test_reshard");
    test_reshard(dimension_counts[0]);
    finish_test_case(NULL);
    start_test_case("test_sequence_reshard");
    test_sequence_reshard();
    finish_test_case(NULL);
    for (size_t dimension_index = 0; dimension_index != 2; ++dimension_index) {
        snprintf(label, sizeof(label), "test_count_and_get(%zu)", dimension_counts[dimension_index]);
        start_test_case(label);
        test_count_and_get(dimension_counts[dimension_index]);
        finish_test_case(NULL);
    }
    start_test_case("test_arenas");
    test_arenas(dimension_counts[0]);
    finish_test_case(NULL);
    start_test_case("test_walks_name_their_index");
    test_walks_name_their_index(dimension_counts[0]);
    finish_test_case(NULL);
    start_test_case("test_sample");
    finish_test_case(test_sample());
    start_test_case("test_cursor_steps");
    test_cursor_steps();
    finish_test_case(NULL);
    start_test_case("test_sequence_saves_and_loads_a_file");
    test_sequence_saves_and_loads_a_file();
    finish_test_case(NULL);
    start_test_case("test_sequence_shards");
    test_sequence_shards();
    finish_test_case(NULL);
    start_test_case("test_sequence_vram");
    finish_test_case(test_sequence_vram());
    start_test_case("test_shapes_saves_and_loads_a_file");
    test_shapes_saves_and_loads_a_file();
    finish_test_case(NULL);
    start_test_case("test_shapes_vram");
    finish_test_case(test_shapes_vram());
    start_test_case("test_geospatial_saves_and_loads_a_file");
    test_geospatial_saves_and_loads_a_file();
    finish_test_case(NULL);
    start_test_case("test_shapes_and_geospatial_reshard");
    test_shapes_and_geospatial_reshard();
    finish_test_case(NULL);
    start_test_case("test_reshard_refusals");
    test_reshard_refusals();
    finish_test_case(NULL);
    start_test_case("test_geospatial_vram");
    finish_test_case(test_geospatial_vram());
#if USEARCH_DENSE_DTYPES >= 2
    start_test_case("test_multiply_packed");
    test_multiply_packed();
    finish_test_case(NULL);
#else
    start_test_case("test_multiply_packed");
    finish_test_case("built without usearch_scalar_u1x8_k in USEARCH_DENSE_DTYPES");
#endif
#if USEARCH_DENSE_DTYPES >= 3
    start_test_case("test_attention_is_exact_over_every_key");
    test_attention_is_exact_over_every_key();
    finish_test_case(NULL);
#else
    start_test_case("test_attention_is_exact_over_every_key");
    finish_test_case("built without usearch_scalar_i8_k in USEARCH_DENSE_DTYPES");
#endif
    start_test_case("test_late_interaction_scores_over_documents");
    test_late_interaction_scores_over_documents();
    finish_test_case(NULL);
    start_test_case("test_nvme_view");
    finish_test_case(test_nvme_view(dimension_counts[0]));
    start_test_case("test_nvme_load");
    finish_test_case(test_nvme_load(dimension_counts[0]));
    start_test_case("test_nvme_view_narrow_keys");
    finish_test_case(test_nvme_view_narrow_keys());
    start_test_case("test_vram_parts_share_a_device");
    finish_test_case(test_vram_parts_share_a_device(collection_sizes[1], dimension_counts[1]));
    start_test_case("test_vram_scan");
    finish_test_case(test_vram_scan(collection_sizes[1], dimension_counts[1]));
    start_test_case("test_ceilings_and_fill");
    test_ceilings_and_fill();
    finish_test_case(NULL);
    start_test_case("test_deferred_search");
    test_deferred_search();
    finish_test_case(NULL);
    start_test_case("test_deferred_polling");
    test_deferred_polling();
    finish_test_case(NULL);
    start_test_case("test_contended_from_another_thread");
    test_contended_from_another_thread();
    finish_test_case(NULL);
    start_test_case("test_drain_then_mood");
    test_drain_then_mood();
    finish_test_case(NULL);
    start_test_case("test_free_waits_for_named_call");
    test_free_waits_for_named_call();
    finish_test_case(NULL);
    start_test_case("test_free_waits_for_call_without_executor");
    test_free_waits_for_call_without_executor();
    finish_test_case(NULL);
    start_test_case("test_deferred_holds_arena");
    test_deferred_holds_arena();
    finish_test_case(NULL);
    // An implementation without Arrow, as the stand-in, answers every exchange unsupported.
    struct ArrowSchema probed_schema = {0};
    struct ArrowDeviceArray probed_array = {0};
    char const* const arrow_skip = usearch_rows_view_to_arrow((usearch_rows_view_t) {0}, 0, NULL, &probed_schema,
                                                              &probed_array) == usearch_status_unsupported_k
                                       ? needs_arrow_k
                                       : NULL;
    start_test_case("test_arrow_rows_and_keys");
    if (!arrow_skip) test_arrow_rows_and_keys();
    finish_test_case(arrow_skip);

    start_test_case("test_arrow_refusals");
    if (!arrow_skip) test_arrow_refusals();
    finish_test_case(arrow_skip);

    start_test_case("test_arrow_tapes");
    if (!arrow_skip) test_arrow_tapes();
    finish_test_case(arrow_skip);

    start_test_case("test_arrow_matches");
    if (!arrow_skip) test_arrow_matches();
    finish_test_case(arrow_skip);

    return finish_test_run();
}

#undef expect_
