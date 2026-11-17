/**
 *  @file cpp/test.cpp
 *  @author Ash Vardanian
 *  @date October 14, 2026
 *  @brief The smoke test of the C++ wrappers: one dense round trip through every handle they own,
 *      then the sequence, shapes and geospatial cases the Python modality suite runs, against
 *      both the engine and the brute-force stub.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include <csignal>   // `std::signal`, `std::raise`, `SIGSEGV`, `SIGABRT`, `SIGTRAP`
#include <cctype>    // `std::tolower` - a setting's flag spelling
#include <cinttypes> // `PRIu64` - the run's lines
#include <cmath>     // `std::abs`, `std::llround`, `std::isfinite`
#include <cstdint>   // `std::uint8_t`, `std::uint64_t`
#include <cstdio>    // `std::remove`, `std::fputc`
#include <cstdlib>   // `std::exit`, `std::getenv`
#include <cstring>   // `std::memcmp`, `std::strcmp`, `std::strlen`

#include <algorithm>       // `std::max`
#include <atomic>          // `std::atomic` - the run's failed checks
#include <charconv>        // `std::from_chars` - the settings
#include <chrono>          // `std::chrono::steady_clock` - case timings and the wait limit
#include <functional>      // `std::invoke` - the cases
#include <limits>          // `std::numeric_limits`
#include <optional>        // `std::optional` - the compiled filter
#include <random>          // `std::mt19937`, `std::normal_distribution`, `std::uniform_int_distribution`
#include <regex>           // `std::regex` - the filter
#include <source_location> // `std::source_location` - where a check failed
#include <string>          // `std::string` - the filter pattern
#include <string_view>     // `std::string_view` - the settings' names
#include <system_error>    // `std::errc`
#include <thread>          // `std::this_thread::yield` - the bounded waits
#include <type_traits>     // `std::invoke_result_t` - what a case returns

#include <fmt/format.h>

#include "usearch/dense_index.hpp"
#include "usearch/geospatial_index.hpp"
#include "usearch/sequence_index.hpp"
#include "usearch/shapes_index.hpp"

#if USEARCH_OS_WINDOWS_
#include <io.h> // `_write`
#else
#include <unistd.h> // `write`, `STDERR_FILENO`
#endif
/*  Only glibc and Darwin ship `backtrace` in libc itself; FreeBSD and musl need another library. */
#if (defined(__GLIBC__) || USEARCH_OS_APPLE_) && __has_include(<execinfo.h>)
#include <execinfo.h> // `backtrace`, `backtrace_symbols_fd`
#endif

namespace us = unum::usearch;

/** Stops in an attached debugger, or dumps core without one; no crash handler catches SIGTRAP. */
inline void trap_into_debugger() noexcept {
#if USEARCH_OS_WINDOWS_
    __debugbreak();
#else
    std::raise(SIGTRAP);
#endif
}

/** Prints the case in progress and a backtrace on a fatal signal; defined beside its run. */
inline void install_test_signal_handlers() noexcept;

/** The names the C interface reports for this build and this machine. */
struct machine_t {
    char const* version = nullptr;
    char const* compiled = nullptr;
    char const* available = nullptr;
};

/** Probes the names @c machine_t reports. */
inline machine_t probe_machine() noexcept {
    return {usearch_version_name(), usearch_capabilities_compiled(), usearch_capabilities_available()};
}

/** Prints the version line, then the capabilities as "- Compiled for:" and "- This machine:". */
inline void print(machine_t const& machine) noexcept {
    fmt::println("USearch {}", machine.version);
    fmt::println("- Compiled for: {}", machine.compiled);
    fmt::println("- This machine: {}", machine.available);
}

#pragma region Shared Settings

/** Exits with one line naming a setting that does not parse, so a bad value never runs anything. */
[[noreturn]] inline void exit_unparsed(char const* variable, char const* text, char const* expected) noexcept {
    std::fprintf(stderr, "%s=\"%s\" does not parse, expected %s\n", variable, text, expected);
    std::exit(1);
}

/** The text of a setting: its @p flag when given, else its environment @p variable , else null. */
inline char const* setting_text(int argc, char** argv, char const* variable, std::string_view flag) noexcept {
    for (int index = 1; index < argc; ++index) {
        std::string_view const argument {argv[index]};
        if (argument == flag)
            return index + 1 < argc && std::strncmp(argv[index + 1], "--", 2) != 0 ? argv[index + 1] : "1";
        if (argument.size() > flag.size() && argument.starts_with(flag) && argument[flag.size()] == '=')
            return argv[index] + flag.size() + 1;
    }
    char const* text = std::getenv(variable);
    return text && *text ? text : nullptr;
}

/** A switch, @p fallback when unset: 0, false, off and no turn it off, anything else on. */
inline bool read_flag_setting(int argc, char** argv, char const* variable, std::string_view flag,
                              bool fallback = false) noexcept {
    char const* text = setting_text(argc, argv, variable, flag);
    if (!text) return fallback;
    for (char const* off : {"0", "false", "off", "no"})
        if (std::strcmp(text, off) == 0) return false;
    return true;
}

/** A whole number, @p fallback when unset. */
inline std::size_t read_count_setting(int argc, char** argv, char const* variable, std::string_view flag,
                                      std::size_t fallback) noexcept {
    char const* text = setting_text(argc, argv, variable, flag);
    if (!text) return fallback;
    std::string_view const spelled {text};
    std::size_t count = 0;
    auto const [end, error] = std::from_chars(spelled.data(), spelled.data() + spelled.size(), count);
    if (error != std::errc {} || end != spelled.data() + spelled.size())
        exit_unparsed(variable, text, "a whole number");
    return count;
}

/** A positive real, @p fallback when unset. */
inline double read_real_setting(int argc, char** argv, char const* variable, std::string_view flag,
                                double fallback) noexcept {
    char const* text = setting_text(argc, argv, variable, flag);
    if (!text) return fallback;
    char* end = nullptr;
    double const value = std::strtod(text, &end);
    if (*end != '\0' || !(value > 0) || !std::isfinite(value))
        exit_unparsed(variable, text, "a positive number like 0.1 or 10");
    return value;
}

/** A positive duration such as 200ms, 10s or 2m, @p fallback when unset. */
inline std::chrono::milliseconds read_duration_setting(int argc, char** argv, char const* variable,
                                                       std::string_view flag,
                                                       std::chrono::milliseconds fallback) noexcept {
    char const* text = setting_text(argc, argv, variable, flag);
    if (!text) return fallback;
    std::string_view const spelled {text};
    std::uint32_t count = 0;
    auto const [end, error] = std::from_chars(spelled.data(), spelled.data() + spelled.size(), count);
    std::string_view const unit = spelled.substr(static_cast<std::size_t>(end - spelled.data()));
    if (error == std::errc {} && count != 0) {
        if (unit == "ms") return std::chrono::milliseconds(count);
        if (unit == "s") return std::chrono::seconds(count);
        if (unit == "m") return std::chrono::minutes(count);
    }
    exit_unparsed(variable, text, "a duration like 200ms, 10s or 2m");
}

/** The run's seed: 42 when unset, drawn afresh for @c random , else any unsigned 64-bit value. */
inline std::uint64_t read_seed_setting(int argc, char** argv, char const* variable, std::string_view flag) noexcept {
    char const* text = setting_text(argc, argv, variable, flag);
    if (!text) return 42;
    std::string_view const spelled {text};
    if (spelled == "random") {
        std::random_device device;
        return (static_cast<std::uint64_t>(device()) << 32) ^ device();
    }
    std::uint64_t seed = 0;
    auto const [end, error] = std::from_chars(spelled.data(), spelled.data() + spelled.size(), seed);
    if (error != std::errc {} || end != spelled.data() + spelled.size())
        exit_unparsed(variable, text, "an unsigned integer or random");
    return seed;
}

/** Which labels a run selects: a regex search when the pattern compiles, else a substring. */
struct filter_t {
    std::string pattern;
    std::optional<std::regex> regex;

    bool selects(std::string_view label) const noexcept {
        if (pattern.empty()) return true;
        if (regex) return std::regex_search(label.begin(), label.end(), *regex);
        return label.find(pattern) != std::string_view::npos;
    }
};

/** The filter a setting spells, compiled once; empty, selecting everything, when unset. */
inline filter_t read_filter_setting(int argc, char** argv, char const* variable, std::string_view flag) noexcept {
    filter_t filter;
    char const* text = setting_text(argc, argv, variable, flag);
    if (!text) return filter;
    filter.pattern = text;
#if defined(__cpp_exceptions) && __cpp_exceptions
    try {
        filter.regex.emplace(filter.pattern);
    }
    catch (std::regex_error const&) {
    }
#else
    filter.regex.emplace(filter.pattern);
#endif
    return filter;
}

/** What every suite and bench reads before its own settings; @c program starts the rerun lines. */
struct common_settings_t {
    std::uint64_t seed = 42;
    filter_t filter;
    char const* program = "";
};

/** SplitMix64's finalizer, a bijection that spreads every input bit over all 64 output bits. */
constexpr std::uint64_t split_mix_bits(std::uint64_t value) noexcept {
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ull;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBull;
    return value ^ (value >> 31);
}

/** The @p index -th stream of the family @p name under @p seed , identical in every language. */
constexpr std::uint64_t stream_key(std::uint64_t seed, std::string_view name, std::uint64_t index = 0) noexcept {
    std::uint64_t hashed = 0xCBF29CE484222325ull;
    for (char const character : name) hashed = (hashed ^ static_cast<std::uint8_t>(character)) * 0x100000001B3ull;
    return split_mix_bits(split_mix_bits(seed ^ hashed) + index);
}

/**
 *  @brief SplitMix64 over one 64-bit state, seeded by a @ref stream_key , bit-identical to the
 *      JavaScript generator NumKong ships.
 *
 *  Distributions are not specified to draw the same sequence on every standard library, so a
 *  failure would stop reproducing from its seed when the toolchain changed; @ref below is the
 *  portable bounded draw.
 */
struct split_mix_generator_t {
    using result_type = std::uint64_t;
    std::uint64_t state = 0;

    explicit split_mix_generator_t(std::uint64_t key) noexcept : state(key) {}
    static constexpr result_type(min)() noexcept { return 0; }
    static constexpr result_type(max)() noexcept { return ~result_type {0}; }
    result_type operator()() noexcept { return split_mix_bits(state += 0x9E3779B97F4A7C15ull); }

    /** A uniform draw below @p bound , zero for an empty bound, without the bias of a modulo. */
    std::uint64_t below(std::uint64_t bound) noexcept {
        if (bound == 0) return 0;
        std::uint64_t const rejection_floor = (0u - bound) % bound;
        std::uint64_t draw = (*this)();
        while (draw < rejection_floor) draw = (*this)();
        return draw % bound;
    }
};

#pragma endregion Shared Settings

/** What the suite reads: the common settings and the run's. */
struct test_settings_t : common_settings_t {
    double scale = 1.0;
    bool trap_on_failure = false;
    bool refuse_skips = false;
};

/** Reads every @c test_settings_t field. */
inline test_settings_t read_test_settings(int argc, char** argv) noexcept;

#pragma region Shared Run

/** What a case reports beyond its checks: that it ran, or why it skipped. */
struct case_result_t {
    char const* skip_reason = nullptr;

    [[nodiscard]] static constexpr case_result_t skipped(char const* reason) noexcept { return {reason}; }
};

/**
 *  @brief The run in progress: the harness's one sanctioned mutable global.
 *
 *  Checks fire on worker threads and inside helpers as well as in a case's own body, so the run
 *  counts failed checks in one place and @ref run_test_case diffs that count around each case,
 *  rather than threading a tally through every call. After a crash, printing @c test_run from a
 *  core dump names the case in progress and the last failed check, and the signal handlers print
 *  the case too.
 */
struct test_run_t {
    test_settings_t settings {};
    char const* current_case = nullptr;
    std::atomic<std::size_t> failed_checks {0};
    std::atomic<char const*> last_failed_file {nullptr};
    std::atomic<std::uint_least32_t> last_failed_line {0};
    std::size_t passed_cases = 0;
    std::size_t failed_cases = 0;
    std::size_t skipped_cases = 0;
};

inline test_run_t test_run;

/** How long any wait in a test may take; a passing case never waits it out, so it bounds hangs. */
inline constexpr std::chrono::seconds wait_limit_k {120};

/** Yields until @p predicate holds; false when @ref wait_limit_k passed first. */
template <typename predicate_type_>
bool spin_until(predicate_type_&& predicate) noexcept {
    auto const deadline = std::chrono::steady_clock::now() + wait_limit_k;
    while (!predicate()) {
        if (std::chrono::steady_clock::now() > deadline) return false;
        std::this_thread::yield();
    }
    return true;
}

/** @p baseline iterations under the @c SCALE setting, never fewer than one. */
inline std::size_t scaled_iterations(std::size_t baseline) noexcept {
    double const scaled = static_cast<double>(baseline) * test_run.settings.scale;
    return scaled < 1 ? 1 : static_cast<std::size_t>(std::llround(scaled));
}

/** The step that visits a @p coverage fraction of a sweep, at least one. */
inline std::size_t sweep_stride(double coverage) noexcept {
    double const stride = 1.0 / coverage;
    return stride < 1 ? 1 : static_cast<std::size_t>(std::llround(stride));
}

/** The @p index -th stream of the case in progress, so a filtered run draws what a full one did. */
inline std::uint64_t case_stream_key(std::uint64_t index = 0) noexcept {
    return stream_key(test_run.settings.seed, test_run.current_case ? test_run.current_case : "", index);
}

/** Line-buffers stdout, prints the machine, reads the settings, arms the crash handlers and
 *  prints the settings line, before the first case. */
inline void start_test_run(int argc, char** argv) noexcept {
    std::setvbuf(stdout, nullptr, _IOLBF, BUFSIZ); // before any output, or the C library keeps its buffering
    print(probe_machine());
    test_run.settings = read_test_settings(argc, argv);
    install_test_signal_handlers();
    std::printf("- Seed: %" PRIu64 ", scale: %g, filter: %s\n", test_run.settings.seed, test_run.settings.scale,
                test_run.settings.filter.pattern.empty() ? "none" : test_run.settings.filter.pattern.c_str());
}

/** Counts one failed check at @p where , and traps when the run asked to stop at the first. */
inline void record_failed_check(std::source_location where) noexcept {
    std::fflush(stdout); // so a log that merges both streams keeps the failure under its case
    std::fprintf(stderr, "  check failed at %s:%" PRIuLEAST32 "\n", where.file_name(), where.line());
    test_run.last_failed_file.store(where.file_name(), std::memory_order_relaxed);
    test_run.last_failed_line.store(where.line(), std::memory_order_relaxed);
    test_run.failed_checks.fetch_add(1, std::memory_order_relaxed);
    if (test_run.settings.trap_on_failure) trap_into_debugger();
}

/** Holds when @p is_met does; otherwise counts a failed check at the caller's line. */
inline bool expect_true(bool is_met, std::source_location where = std::source_location::current()) noexcept {
    if (!is_met) record_failed_check(where);
    return is_met;
}

/** Prints one case's verdict, and the line that reruns it alone when it failed. */
inline void report_test_case(char const* label, std::size_t failed_checks, char const* skip_reason,
                             double seconds) noexcept {
    if (skip_reason) {
        std::printf("- %s ... skipped: %s\n", label, skip_reason);
        return;
    }
    if (failed_checks == 0) {
        std::printf("- %s ... ok (%.2f s)\n", label, seconds);
        return;
    }
    std::fflush(stdout);
    std::fprintf(stderr, "- %s ... FAILED: %zu failed check(s)\n", label, failed_checks);
    std::fprintf(stderr, "  rerun: %s --seed %" PRIu64 " --filter '^", test_run.settings.program,
                 test_run.settings.seed);
    for (char const* character = label; *character; ++character) {
        if (std::strchr("\\^$.|?*+()[]{}", *character)) std::fputc('\\', stderr); // labels are code, the filter a regex
        std::fputc(*character, stderr);
    }
    std::fputs("$'\n", stderr);
}

/** Runs one case unless the filter leaves it out, and records whether it passed or skipped. */
template <typename case_function_type_, typename... argument_types_>
void run_test_case(char const* label, case_function_type_&& case_function, argument_types_&&... arguments) noexcept {
    if (!test_run.settings.filter.selects(label)) return;
    test_run.current_case = label;
    std::size_t const failed_before = test_run.failed_checks.load();
    auto const started = std::chrono::steady_clock::now();
    case_result_t result {};
    using returned_type = std::invoke_result_t<case_function_type_, argument_types_...>;
    if constexpr (std::is_void_v<returned_type>)
        std::invoke(case_function, std::forward<argument_types_>(arguments)...);
    else result = std::invoke(case_function, std::forward<argument_types_>(arguments)...);
    std::chrono::duration<double> const elapsed = std::chrono::steady_clock::now() - started;
    std::size_t const failed_checks = test_run.failed_checks.load() - failed_before;
    if (result.skip_reason) ++test_run.skipped_cases;
    else if (failed_checks != 0) ++test_run.failed_cases;
    else ++test_run.passed_cases;
    report_test_case(label, failed_checks, result.skip_reason, elapsed.count());
    test_run.current_case = nullptr;
}

/** The exit code: 1 when a case or a check outside any case failed, or nothing ran; 3 when
 *  skips are refused and one happened; else 0. */
inline int finish_test_run() noexcept {
    std::printf("%zu passed, %zu failed, %zu skipped, seed %" PRIu64 "\n", test_run.passed_cases, test_run.failed_cases,
                test_run.skipped_cases, test_run.settings.seed);
    bool const failed = test_run.failed_cases != 0 || test_run.failed_checks.load() != 0;
    if (failed || test_run.passed_cases + test_run.skipped_cases == 0) return 1;
    if (test_run.skipped_cases != 0 && test_run.settings.refuse_skips) return 3;
    return 0;
}

#pragma endregion Shared Run

inline test_settings_t read_test_settings(int argc, char** argv) noexcept {
    test_settings_t settings;
    settings.seed = read_seed_setting(argc, argv, "USEARCH_SEED", "--seed");
    settings.filter = read_filter_setting(argc, argv, "USEARCH_FILTER", "--filter");
    settings.program = argc != 0 ? argv[0] : "";
    settings.scale = read_real_setting(argc, argv, "USEARCH_SCALE", "--scale", 1.0);
    settings.trap_on_failure = read_flag_setting(argc, argv, "USEARCH_TRAP_ON_FAILURE", "--trap-on-failure");
    settings.refuse_skips = read_flag_setting(argc, argv, "USEARCH_NO_SKIPS", "--no-skips");
    return settings;
}

/** Names the signal and the case in progress, prints a backtrace, then re-raises the signal;
 *  writes through a raw @c write, since the crashing thread may already hold the stdio lock. */
inline void test_fatal_signal_handler(int signal_number) noexcept {
    auto const write_text = [](std::string_view text) noexcept {
#if USEARCH_OS_WINDOWS_
        [[maybe_unused]] auto const written = _write(2, text.data(), static_cast<unsigned>(text.size()));
#else
        [[maybe_unused]] auto const written = ::write(STDERR_FILENO, text.data(), text.size());
#endif
    };
    std::string_view signal_name = "SIGSEGV";
    if (signal_number == SIGABRT) signal_name = "SIGABRT";
    else if (signal_number == SIGILL) signal_name = "SIGILL";
    else if (signal_number == SIGFPE) signal_name = "SIGFPE";
#if defined(SIGBUS)
    else if (signal_number == SIGBUS) signal_name = "SIGBUS";
#endif
    write_text("\n*** ");
    write_text(signal_name);
    write_text(" in case '");
    write_text(test_run.current_case ? test_run.current_case : "none");
    write_text("' - backtrace follows ***\n");
#if (defined(__GLIBC__) || USEARCH_OS_APPLE_) && __has_include(<execinfo.h>)
    void* frames[64];
    int const frames_count = backtrace(frames, static_cast<int>(std::size(frames)));
    backtrace_symbols_fd(frames, frames_count, STDERR_FILENO);
#endif
    // Restored and re-raised, so the shell sees the real signal and any core dump is produced.
    std::signal(signal_number, SIG_DFL);
    std::raise(signal_number);
}

inline void install_test_signal_handlers() noexcept {
    std::signal(SIGSEGV, test_fatal_signal_handler);
    std::signal(SIGABRT, test_fatal_signal_handler);
    std::signal(SIGILL, test_fatal_signal_handler);
    std::signal(SIGFPE, test_fatal_signal_handler);
#if defined(SIGBUS) // Windows has no bus error to catch
    std::signal(SIGBUS, test_fatal_signal_handler);
#endif
}

/** A format string bundled with the @c std::source_location of the call site that spelled it, so a
 *  check's context trails its arguments without a macro. */
struct format_at_t {
    fmt::string_view format;
    std::source_location location;

    format_at_t(char const* format, std::source_location location = std::source_location::current()) noexcept
        : format(format), location(location) {}
};

/** Passes when @p holds; otherwise counts a failed check at the caller's line and prints @p context
 *  formatted with @p args beneath it. */
template <typename... args_types_>
bool expect_true(bool holds, format_at_t context, args_types_ const&... args) noexcept {
    if (holds) return true;
    record_failed_check(context.location);
    fmt::println(stderr, "    {}", fmt::vformat(context.format, fmt::make_format_args(args...)));
    return false;
}

/** Holds when @p reported succeeded; otherwise counts a failed check and prints @p context
 *  with its status. */
template <typename outcome_type_>
bool expect_success(outcome_type_ const& reported, format_at_t context) noexcept {
    if (reported) return true;
    record_failed_check(context.location);
    fmt::println(stderr, "    {}: {}", context.format,
                 reported.message ? reported.message : usearch_status_name(reported.status));
    return false;
}

/** The key spelling @p offset as big-endian bytes, so keys made from offsets order as they do. */
static usearch_key_t key_of(std::size_t offset) {
    usearch_key_t key {};
    for (std::size_t byte = 0; byte != sizeof(offset); ++byte)
        key.bytes[sizeof(key.bytes) - 1 - byte] = static_cast<std::uint8_t>(offset >> (byte * 8));
    return key;
}

static bool same_key(usearch_key_t const& first, usearch_key_t const& second) {
    return std::memcmp(first.bytes, second.bytes, sizeof(first.bytes)) == 0;
}

static usearch_bytes_view_t bytes_of(char const* text) {
    return {reinterpret_cast<std::uint8_t const*>(text), std::strlen(text)};
}

/** A cancel at the first barrier a call reports, so a plan stops with work still ahead. */
struct cancel_at_first_t {
    usearch_continuation_t operator()(std::size_t, std::size_t) noexcept { return usearch_cancel_k; }
};

#pragma region Modality Fixtures

/** The most entries a fixture holds, so its storage is a fixed array. */
static constexpr std::size_t fixture_capacity = 64;

/** Random lowercase words of 8 to 16 bytes, as one byte tape. */
struct sequence_fixture_t {
    using index_t = us::sequence_index_t;
    using refine_plan_t = us::sequence_refine_plan_t;
    using reshard_plan_t = us::sequence_reshard_plan_t;
    using raw_t = usearch_sequence_index_t;
    static constexpr char const* name = "sequence";

    std::uint8_t bytes[fixture_capacity * 16] {};
    std::uint64_t offsets[fixture_capacity + 1] {};
    std::size_t entries = 0;

    sequence_fixture_t(std::size_t count, std::uint32_t seed) : entries(count) {
        expect_true(count <= fixture_capacity, "A sequence fixture holds at most its capacity");
        std::mt19937 generator {seed};
        std::uniform_int_distribution<int> letter {'a', 'z'}, length {8, 16};
        for (std::size_t entry = 0; entry != count; ++entry) {
            std::uint64_t end = offsets[entry];
            for (int byte = length(generator); byte != 0; --byte) bytes[end++] = letter(generator);
            offsets[entry + 1] = end;
        }
    }
    us::tape_view<std::uint8_t> batch() const { return {bytes, std::span<std::uint64_t const>(offsets, entries + 1)}; }
    static us::outcome<index_t> init() {
        return index_t::init_ram({.metric = usearch_sequence_metric_levenshtein_bytes_k});
    }
    template <typename on_type_>
    auto insert(on_type_ const& on, std::span<usearch_key_t const> keys) const {
        return on.insert(keys, batch());
    }
};

/** Random clouds of 4 to 12 points in the unit cube, as one point tape. */
struct shapes_fixture_t {
    using index_t = us::shapes_index_t;
    using refine_plan_t = us::shapes_refine_plan_t;
    using reshard_plan_t = us::shapes_reshard_plan_t;
    using raw_t = usearch_shapes_index_t;
    static constexpr char const* name = "shapes";

    float points[fixture_capacity * 12 * 3] {};
    std::uint64_t offsets[fixture_capacity + 1] {};
    std::size_t entries = 0;

    shapes_fixture_t(std::size_t count, std::uint32_t seed) : entries(count) {
        expect_true(count <= fixture_capacity, "A shapes fixture holds at most its capacity");
        std::mt19937 generator {seed};
        std::uniform_real_distribution<float> coordinate {0, 1};
        std::uniform_int_distribution<int> length {4, 12};
        for (std::size_t entry = 0; entry != count; ++entry) {
            std::uint64_t const begin = offsets[entry], end = begin + length(generator);
            for (std::uint64_t scalar = begin * 3; scalar != end * 3; ++scalar) points[scalar] = coordinate(generator);
            offsets[entry + 1] = end;
        }
    }
    us::tape_view<float> batch() const { return {points, std::span<std::uint64_t const>(offsets, entries + 1)}; }
    static us::outcome<index_t> init() {
        return index_t::init_ram({.metric = usearch_shapes_metric_chamfer_k, .quantization = usearch_scalar_f32_k});
    }
    template <typename on_type_>
    auto insert(on_type_ const& on, std::span<usearch_key_t const> keys) const {
        return on.template insert<float>(keys, batch());
    }
};

/** An observation at @p latitude and @p longitude, every other field unrecorded. */
static usearch_geospatial_observation_t observation_at(double latitude, double longitude) {
    double const unrecorded = std::numeric_limits<double>::quiet_NaN();
    return {latitude,   longitude,  unrecorded, unrecorded, unrecorded, unrecorded,
            unrecorded, unrecorded, unrecorded, unrecorded, unrecorded};
}

/** Random positions between the 60th parallels, every other field unrecorded. */
struct geospatial_fixture_t {
    using index_t = us::geospatial_index_t;
    using refine_plan_t = us::geospatial_refine_plan_t;
    using reshard_plan_t = us::geospatial_reshard_plan_t;
    using raw_t = usearch_geospatial_index_t;
    static constexpr char const* name = "geospatial";

    usearch_geospatial_observation_t observations[fixture_capacity] {};
    std::size_t entries = 0;

    geospatial_fixture_t(std::size_t count, std::uint32_t seed) : entries(count) {
        expect_true(count <= fixture_capacity, "A geospatial fixture holds at most its capacity");
        std::mt19937 generator {seed};
        std::uniform_real_distribution<double> latitude {-60, 60}, longitude {-180, 180};
        for (std::size_t entry = 0; entry != count; ++entry)
            observations[entry] = observation_at(latitude(generator), longitude(generator));
    }
    static us::outcome<index_t> init() { return index_t::init_ram({}); }
    template <typename on_type_>
    auto insert(on_type_ const& on, std::span<usearch_key_t const> keys) const {
        return on.insert(keys, std::span<usearch_geospatial_observation_t const>(observations, entries));
    }
};

/** A host index of @p fixture_type_ holding @p count random entries under consecutive keys from
 *  @p first_key. */
template <typename fixture_type_>
static typename fixture_type_::index_t built(us::executor_t const& executor, std::size_t count,
                                             std::size_t first_key = 0, std::uint32_t seed = 0) {
    us::outcome<typename fixture_type_::index_t> index = fixture_type_::init();
    expect_success(index, "creating a modality index");
    if (count == 0) return std::move(index.value);
    usearch_key_t keys[fixture_capacity] {};
    for (std::size_t entry = 0; entry != count; ++entry) keys[entry] = key_of(first_key + entry);
    expect_success(fixture_type_(count, seed).insert(index.value.on(executor), {keys, count}),
                   "inserting modality entries");
    return std::move(index.value);
}

template <typename index_type_>
static std::size_t size_of(index_type_ const& index) {
    return index.on().describe().value.size;
}

#pragma endregion Modality Fixtures

/** A metric kind reads back from its own spelling, and a word no kind spells is refused. */
template <typename metric_type_>
static void check_metric_name(char const* (*name_of)(metric_type_),
                              us::outcome<metric_type_> (*from_name)(usearch_bytes_view_t), metric_type_ metric) {
    char const* const spelled = name_of(metric);
    expect_true(spelled && std::strlen(spelled) != 0, "Every metric kind has a name");
    us::outcome<metric_type_> const read = from_name(bytes_of(spelled));
    expect_true(read && read.value == metric, "A metric kind reads back from its own name");
    expect_true(from_name(bytes_of("no-such-metric")).status == usearch_status_invalid_argument_k,
                "A word no metric kind spells must be refused");
}

/** A sample draws distinct keys placed in three dimensions, never draws a shown key again, and
 *  refuses shown keys without their coordinates. */
template <typename index_type_>
static void check_sample(index_type_ const& index, us::executor_t const& executor, us::arena_t& arena) {
    constexpr std::size_t wanted = 8;
    auto const on = index.on(executor);
    us::outcome<us::sample_t> const first = on.sample(wanted, arena, {.seed = 3});
    expect_success(first, "sampling an index");
    expect_true(first.value.written == wanted && first.value.coordinates.columns == 3,
                "A sample of a populated index draws every key wanted, in three dimensions");
    usearch_key_t shown[wanted] {};
    float placed[wanted][3] {};
    for (std::size_t draw = 0; draw != wanted; ++draw) {
        shown[draw] = first.value.keys.data[draw];
        std::memcpy(placed[draw], first.value.coordinate(draw), sizeof(placed[draw]));
    }
    us::outcome<us::sample_t> const again = on.sample(wanted, arena, {.seed = 3});
    expect_true(again && again.value.written == wanted && same_key(again.value.keys.data[0], shown[0]),
                "Equal seeds over an unchanged index draw equally");

    us::outcome<us::sample_t> const next = on.sample(
        wanted, arena, {.shown = shown, .shown_coordinates = {&placed[0][0], wanted, 3}, .seed = 3});
    expect_success(next, "sampling around shown keys");
    for (std::size_t draw = 0; draw != next.value.written; ++draw)
        for (usearch_key_t const& key : shown)
            expect_true(!same_key(next.value.keys.data[draw], key), "A shown key must never be drawn again");
    expect_true(on.sample(wanted, arena, {.shown = shown}).status == usearch_status_invalid_argument_k,
                "Shown keys without coordinates must be refused");
}

/** The plan and reshard cases of <tt>test/modalities.py</tt>, which each modality answers alike. */
template <typename fixture_type_>
static void test_plans_and_reshards(us::executor_t const& executor) {
    using index_t = typename fixture_type_::index_t;
    using raw_t = typename fixture_type_::raw_t;

    // A sample draws keys from the graph of every modality alike.
    {
        index_t index = built<fixture_type_>(executor, 64);
        us::outcome<us::arena_t> arena = us::arena_t::make({.executor = executor.handle()});
        expect_success(arena, "making a sample arena");
        check_sample(index, executor, arena.value);
    }

    // A round reports what it did, its closing audit judging every entry.
    {
        index_t index = built<fixture_type_>(executor, 64);
        us::outcome<usearch_refine_report_t> const report = index.on(executor).refine();
        expect_success(report, "refining a modality index");
        expect_true(report.value.changed <= report.value.reselected, "A round changes only entries it reselected");
        expect_true(report.value.reachability.audited == 64, "A round's closing audit must judge every entry");
        expect_true(index.on().count(key_of(17)).value.entries == 1, "A refined index keeps every entry");
    }

    // A plan keeps a stopped round live, and its finish closes it once.
    {
        index_t index = built<fixture_type_>(executor, 64);
        typename fixture_type_::refine_plan_t plan;
        cancel_at_first_t cancel;
        us::outcome<usearch_refine_report_t> const stopped = index.on(executor).refine(
            {.progress = {cancel}, .plan = &plan});
        expect_true(stopped.status == usearch_status_cancelled_k, "A cancel must stop the planned round");
        expect_true(plan.phase().phase != usearch_phase_done_k && plan.phase().completed <= plan.phase().total,
                    "A stopped plan must sit before its end");
        expect_success(index.on(executor).refine({.plan = &plan}), "continuing the planned round");
        expect_true(plan.phase().phase == usearch_phase_done_k, "A continued round must run to its end");
        us::outcome<usearch_refine_report_t> const finished = index.on(executor).refine_plan_finish(std::move(plan));
        expect_success(finished, "finishing the refine plan");
        expect_true(finished.value.reachability.audited == 64, "A finish reports the whole plan");
        expect_true(index.on(executor).refine_plan_finish(std::move(plan)).status == usearch_status_invalid_argument_k,
                    "A finished plan must not finish again");
    }

    // A plan lives across calls on one executor, so the calling thread cannot open one.
    {
        index_t index = built<fixture_type_>(executor, 0);
        typename fixture_type_::refine_plan_t plan;
        expect_true(index.on().refine({.plan = &plan}).status == usearch_status_invalid_argument_k,
                    "A plan without an executor must be refused");
    }

    // Several sources into one target gather every entry there, leaving the others empty and live.
    {
        index_t first = built<fixture_type_>(executor, 32);
        index_t second = built<fixture_type_>(executor, 32, 32, 1);
        index_t third = built<fixture_type_>(executor, 32, 64, 2);
        raw_t const sources[3] = {first.handle(), second.handle(), third.handle()};
        expect_success(first.on(executor).reshard(sources, {sources, 1}), "gathering three sources");
        expect_true(size_of(first) == 96 && size_of(second) == 0 && size_of(third) == 0,
                    "A gather must move every entry into its target");
        expect_true(first.on().count(key_of(40)).value.entries == 1 && first.on().count(key_of(80)).value.entries == 1,
                    "A gathered target must hold every source's keys");
    }

    // One source into several targets cuts it between them, leaving the source live but empty.
    {
        index_t source = built<fixture_type_>(executor, 64);
        index_t first = built<fixture_type_>(executor, 0), second = built<fixture_type_>(executor, 0);
        raw_t const from[1] = {source.handle()};
        raw_t const into[2] = {first.handle(), second.handle()};
        expect_success(source.on(executor).reshard(from, into, {.seed = 7}), "cutting one source in two");
        expect_true(size_of(first) + size_of(second) == 64 && size_of(first) != 0 && size_of(second) != 0,
                    "A cut must give every target a part of every entry");
        expect_true(size_of(source) == 0, "A cut source must end empty");
    }

    // A stopped cut keeps every entry moved, the next call ends its rounds, and a deferred
    // receiver closes it.
    {
        index_t source = built<fixture_type_>(executor, 64);
        index_t first = built<fixture_type_>(executor, 0), second = built<fixture_type_>(executor, 0);
        raw_t const from[1] = {source.handle()};
        raw_t const into[2] = {first.handle(), second.handle()};
        typename fixture_type_::reshard_plan_t plan;
        cancel_at_first_t cancel;
        us::outcome<usearch_refine_report_t> const stopped = source.on(executor).reshard(
            from, into, {.progress = {cancel}, .plan = &plan});
        expect_true(stopped.status == usearch_status_cancelled_k || stopped.status == usearch_status_success_k,
                    "A cancelled cut either stops its rounds or had none to stop");
        expect_true(size_of(first) + size_of(second) == 64, "A stopped cut keeps every entry it moved");
        expect_success(source.on(executor).reshard(from, into, {.plan = &plan}), "continuing the planned cut");
        expect_true(plan.phase().phase == usearch_phase_done_k, "A continued cut must run to its end");
        us::deferred<usearch_refine_report_t> finishing = source.deferred(executor).reshard_plan_finish(
            std::move(plan));
        expect_success(finishing.wait(), "finishing the reshard plan deferred");
    }

    // One index as its only source and target has a single shard to recut, which is refused.
    {
        index_t index = built<fixture_type_>(executor, 16);
        raw_t const both[1] = {index.handle()};
        expect_true(index.on(executor).reshard(both, both).status == usearch_status_unsupported_k,
                    "A recut of one handle must be unsupported");
        expect_true(size_of(index) == 16, "A refused recut moves nothing");
    }

    // Empty or repeated lists are malformed and a target holding foreign entries mismatches.
    {
        index_t first = built<fixture_type_>(executor, 8);
        index_t second = built<fixture_type_>(executor, 8, 8, 1);
        raw_t const firsts[2] = {first.handle(), first.handle()};
        raw_t const seconds[1] = {second.handle()};
        auto const on = first.on(executor);
        expect_true(on.reshard({}, {firsts, 1}).status == usearch_status_invalid_argument_k,
                    "An empty source list is malformed");
        expect_true(on.reshard(firsts, {firsts, 1}).status == usearch_status_invalid_argument_k,
                    "A repeated source is malformed");
        expect_true(on.reshard({firsts, 1}, seconds).status == usearch_status_config_mismatch_k,
                    "A target holding foreign entries mismatches");
        expect_true(size_of(first) == 8 && size_of(second) == 8, "A refusal moves nothing");
    }
}

/** Every sequence searched for comes back first under its own key, at distance zero, and survives a
 *  checkpoint. */
static void test_sequences(us::executor_t const& executor, us::arena_t& arena) {
    us::outcome<us::sequence_index_t> index = us::sequence_index_t::init_ram(
        {.metric = usearch_sequence_metric_levenshtein_utf8_k});
    expect_success(index, "creating a sequence index");
    auto const on = index.value.on(executor);
    us::outcome<usearch_sequence_index_description_t> const empty = on.describe();
    expect_true(empty.value.size == 0 && empty.value.gpu_class == usearch_gpu_class_unknown_k,
                "A fresh host index is empty and on no GPU");
    expect_true(empty.value.metadata.metric == usearch_sequence_metric_levenshtein_utf8_k,
                "The frozen metric reads back");

    std::uint8_t const words[] = {'k', 'i', 't', 't', 'e', 'n', 's', 'i', 't', 't',
                                  'i', 'n', 'g', 'm', 'i', 't', 't', 'e', 'n'};
    std::uint64_t const offsets[] = {0, 6, 13, 19};
    usearch_key_t const keys[] = {key_of(1), key_of(2), key_of(3)};
    expect_success(on.insert(keys, {words, offsets}), "inserting sequences");
    expect_true(on.insert({keys, 1}, {words, offsets}).status == usearch_status_shape_mismatch_k,
                "Keys and sequences of different counts must be refused before the call crosses");

    std::uint64_t const query_offsets[] = {0, 7};
    us::outcome<usearch_matches_view_t> const found = on.search({words + 6, query_offsets}, 1, arena);
    expect_success(found, "searching a sequence");
    expect_true(same_key(found.value.keys[0], keys[1]) && found.value.scores[0] == 0,
                "A stored sequence must find itself first, at distance zero");

    us::outcome<usearch_sequence_key_stats_t> const counted = on.count(keys[1]);
    expect_true(counted && counted.value.entries == 1 && counted.value.total_bytes == 7, "A key's bytes are counted");
    us::outcome<us::sequence_entries_t> const fetched = on.get(keys[1], arena);
    expect_true(fetched && fetched.value.written == 1 && fetched.value.entry(0).size() == 7 &&
                    std::memcmp(fetched.value.entry(0).data(), "sitting", 7) == 0,
                "A key's sequence must come back whole");

    char const* const path = "usearch_test_cpp.sequences";
    expect_success(on.save(bytes_of(path)), "saving sequences");
    us::outcome<usearch_sequence_metadata_t> const metadata = us::sequence_index_t::metadata_read(
        {.path = bytes_of(path)});
    expect_true(metadata && metadata.value.metric == usearch_sequence_metric_levenshtein_utf8_k,
                "A checkpoint reports its metric");
    us::outcome<us::sequence_index_t> const loaded = us::sequence_index_t::load_ram({.path = bytes_of(path)});
    expect_success(loaded, "loading sequences");
    expect_true(size_of(loaded.value) == 3, "A loaded checkpoint keeps every entry");
    std::remove(path);
    us::outcome<usearch_bytes_view_t> const monolith = on.save_buffer(arena);
    expect_success(monolith, "saving sequences into a buffer");
    us::outcome<us::sequence_index_t> const reloaded = us::sequence_index_t::load_ram_buffer(
        {.buffer = monolith.value});
    expect_true(reloaded && size_of(reloaded.value) == 3, "A buffer reloads every entry");

    // A VRAM index either opens on a GPU or answers the library's own refusal.
    us::outcome<us::sequence_index_t> const device = us::sequence_index_t::init_vram({});
    expect_true(device ? device.value.on().describe().value.gpu_class != usearch_gpu_class_unknown_k
                       : device.status != usearch_status_deferred_k,
                "A VRAM index opens on a GPU or forwards the library's status");
    check_metric_name(us::sequence_metric_name, us::sequence_metric_from_name,
                      usearch_sequence_metric_levenshtein_utf8_k);
}

/** Shapes of different point counts share one batch, and each finds itself first. */
static void test_shapes(us::executor_t const& executor, us::arena_t& arena) {
    us::outcome<us::shapes_index_t> index = us::shapes_index_t::init_ram(
        {.metric = usearch_shapes_metric_hausdorff_k, .quantization = usearch_scalar_f32_k});
    expect_success(index, "creating a shapes index");
    auto const on = index.value.on(executor);
    expect_true(on.describe().value.metadata.metric == usearch_shapes_metric_hausdorff_k,
                "The frozen metric reads back");

    shapes_fixture_t const clouds(3, 42);
    usearch_key_t const keys[] = {key_of(1), key_of(2), key_of(3)};
    expect_success(on.insert<float>(keys, clouds.batch()), "inserting shapes");
    us::outcome<usearch_shapes_index_description_t> const described = on.describe();
    expect_true(described.value.size == 3 && described.value.total_points == clouds.offsets[clouds.entries],
                "Every point of every shape is stored");

    std::uint64_t const last_offsets[] = {0, clouds.offsets[3] - clouds.offsets[2]};
    us::outcome<usearch_matches_view_t> const found = on.search<float>(
        {clouds.points + clouds.offsets[2] * 3, last_offsets}, 1, arena);
    expect_true(found && same_key(found.value.keys[0], keys[2]), "A stored shape must find itself first");

    us::outcome<us::shapes_entries<float>> const fetched = on.get<float>(keys[2], arena);
    std::size_t const last_points = clouds.offsets[3] - clouds.offsets[2];
    expect_true(fetched && fetched.value.written == 1 && fetched.value.entry(0).size() == last_points * 3,
                "A key's shape must come back with every point");
    expect_true(on.count(keys[2]).value.largest_points == last_points, "A key's points are counted");

    char const* const path = "usearch_test_cpp.shapes";
    expect_success(on.save(bytes_of(path)), "saving shapes");
    us::outcome<us::shapes_index_t> const loaded = us::shapes_index_t::load_ram({.path = bytes_of(path)});
    expect_true(loaded && size_of(loaded.value) == 3, "A loaded checkpoint keeps every entry");
    std::remove(path);
    check_metric_name(us::shapes_metric_name, us::shapes_metric_from_name, usearch_shapes_metric_hausdorff_k);
    check_metric_name(us::shapes_metric_name, us::shapes_metric_from_name, usearch_shapes_metric_gromov_wasserstein_k);
    expect_true(
        std::strcmp(us::shapes_metric_name(usearch_shapes_metric_gromov_wasserstein_k), "gromov-wasserstein") == 0,
        "Shapes metric names are spelled with dashes");
}

/** An observation searched for comes back first at zero chord metres, and a radius admits only
 *  the entries inside it. */
static void test_geospatial(us::executor_t const& executor, us::arena_t& arena) {
    us::outcome<us::geospatial_index_t> index = us::geospatial_index_t::init_ram({});
    expect_success(index, "creating a geospatial index");
    auto const on = index.value.on(executor);

    usearch_geospatial_observation_t const places[] = {observation_at(48.8566, 2.3522), observation_at(48.8656, 2.3522),
                                                       observation_at(52.52, 13.405)};
    usearch_key_t const keys[] = {key_of(1), key_of(2), key_of(3)};
    expect_success(on.insert(keys, places), "inserting observations");
    expect_true(size_of(index.value) == 3, "Every observation is stored");

    us::outcome<usearch_matches_view_t> const found = on.search({places + 2, 1}, usearch_geospatial_relation_near_k, {},
                                                                1, arena);
    expect_true(found && same_key(found.value.keys[0], keys[2]) && found.value.scores[0] == 0,
                "A stored observation must find itself first, at zero metres");

    us::outcome<us::geospatial_within_t> const nearby = on.within({places, 1}, usearch_geospatial_relation_near_k,
                                                                  {.metres = 2000}, arena);
    expect_success(nearby, "reporting observations within a radius");
    expect_true(nearby.value.view.offsets[1] == 2, "A two-kilometre radius admits the two Paris observations only");

    us::outcome<us::geospatial_observations_t> const fetched = on.get(keys[2], arena);
    expect_true(fetched && fetched.value.written == 1 && std::abs(fetched.value.first->latitude_degrees - 52.52) < 1e-6,
                "A key's observation must come back within the stored rounding");
    expect_true(on.count(keys[2]).value.entries == 1, "A key's observations are counted");

    char const* const path = "usearch_test_cpp.geospatial";
    expect_success(on.save(bytes_of(path)), "saving observations");
    us::outcome<us::geospatial_index_t> const loaded = us::geospatial_index_t::load_ram({.path = bytes_of(path)});
    expect_true(loaded && size_of(loaded.value) == 3, "A loaded checkpoint keeps every entry");
    std::remove(path);
}

/** Mirrors the Quickstart of <tt>cpp/README.md</tt> line by line, which a C++ test cannot run
 *  verbatim. It differs in three lines: <tt>int main()</tt> becomes this function, the final
 *  @c std::printf becomes @c fmt::println and also writes the count into @p found, and the
 *  includes and the namespace alias sit at the top of this file. */
static int quickstart(std::size_t& found) {
    // Threads live in the executor, the data lives in the index, and outputs land in the arena.
    auto [executor, executor_status, executor_message] = us::executor_t::for_ram();
    auto [arena, arena_status, arena_message] = us::arena_t::make({.executor = executor.handle()});
    auto [index, index_status, index_message] = us::dense_index_t::init_ram(
        {.metric = usearch_dense_metric_angular_k, .quantization = usearch_scalar_bf16_k, .dimensions = 128});
    if (index_status != usearch_status_success_k) return 1;

    usearch_key_t keys[2] {};
    keys[0].bytes[15] = 1, keys[1].bytes[15] = 2;
    float vectors[2][128] = {{1}, {0, 1}};
    if (!index.on(executor).insert<float>(keys, {&vectors[0][0], 2, 128})) return 1;

    auto [matches, status, message] = index.on(executor).search<float>({&vectors[0][0], 1, 128}, 10, arena);
    if (status != usearch_status_success_k) return 1;
    fmt::println("Found {} matches", found = matches.counts[0]);
    return 0;
}

/** The README's Quickstart finds both of its entries. */
static void test_readme_quickstart() {
    std::size_t found = 0;
    expect_true(quickstart(found) == 0 && found == 2, "The README's Quickstart must find both entries");
}

/** One dense round trip through every handle the wrappers own. */
static void test_dense_round_trip() {
    std::size_t const rows = 1000, dimensions = 64, probe = 7;
    std::vector<float> vectors(rows * dimensions);
    std::mt19937 generator {static_cast<std::uint32_t>(case_stream_key())};
    std::normal_distribution<float> distribution;
    for (float& value : vectors) value = distribution(generator);
    std::vector<usearch_key_t> keys(rows);
    for (std::size_t row = 0; row != rows; ++row) keys[row] = key_of(row);
    us::rows_view<float> const all_rows {vectors.data(), rows, dimensions};

    us::outcome<us::executor_t> const executor = us::executor_t::for_ram();
    expect_success(executor, "spawning an executor");
    us::outcome<us::arena_t> arena = us::arena_t::make({.executor = executor.value.handle()});
    expect_success(arena, "making an arena");
    usearch_dense_init_ram_t const geometry {.metric = usearch_dense_metric_angular_k,
                                             .quantization = usearch_scalar_bf16_k,
                                             .dimensions = dimensions,
                                             .connectivity = 16};
    us::outcome<us::dense_index_t> index = us::dense_index_t::init_ram(geometry);
    expect_success(index, "creating an index");
    auto const on = index.value.on(executor.value);

    expect_success(on.reserve(rows), "reserving the entries");
    expect_success(on.insert<float>(keys, all_rows), "inserting the rows");
    us::outcome<usearch_dense_index_description_t> const described = on.describe();
    expect_success(described, "describing the index");
    expect_true(described.value.size == rows, "An insertion is all-or-nothing, so every entry must be published");
    expect_true(on.insert<float>({keys.data(), 1}, all_rows).status == usearch_status_shape_mismatch_k,
                "Keys and rows of different counts must be refused before the call crosses");

    us::outcome<usearch_matches_view_t> const matches = on.search<float>(all_rows, 1, arena.value);
    expect_success(matches, "searching every row");
    std::size_t recalled = 0;
    for (std::size_t row = 0; row != rows; ++row) recalled += same_key(matches.value.keys[row], keys[row]);
    expect_true(recalled * 10 >= rows * 9, "Self-recall collapsed below 90%");

    us::outcome<std::size_t> const counted = on.count(keys[probe]);
    expect_success(counted, "counting a key");
    expect_true(counted.value == 1, "Unique keys hold one entry apiece");
    us::outcome<us::dense_vectors<float>> const fetched = on.get<float>(keys[probe], arena.value);
    expect_success(fetched, "getting a key");
    float const* const fetched_row = fetched.value.row(0);
    expect_true(fetched.value.written == 1 && fetched.value.view.columns == dimensions && fetched_row[0] != 0,
                "A key's one row must come back whole");

    // The receiver's type picks the completion, and the deferred call's owner waits on it.
    {
        us::deferred<usearch_reachability_t> audit = index.value.deferred(executor.value).health();
        us::outcome<usearch_reachability_t> const audited = audit.wait();
        expect_success(audited, "auditing deferred");
        expect_true(audited.value.audited == rows, "A deferred audit must see every entry");
    }

    // A sample draws from the graph, and a deferred one answers what the blocking one would have.
    check_sample(index.value, executor.value, arena.value);
    {
        us::deferred<us::sample_t> drawing = index.value.deferred(executor.value).sample(16, arena.value);
        us::outcome<us::sample_t> const drawn = drawing.wait();
        expect_true(drawn && drawn.value.written == 16, "A deferred sample draws every key wanted");
    }
    check_metric_name(us::dense_metric_name, us::dense_metric_from_name, usearch_dense_metric_angular_k);
    check_metric_name(us::dense_metric_name, us::dense_metric_from_name, usearch_dense_metric_jaccard_k);

    // A plan keeps its round between calls, and its finish on the receiver closes it.
    us::dense_refine_plan_t refine_plan;
    std::size_t reports = 0;
    auto cancel_first = [&](std::size_t, std::size_t) noexcept {
        return reports++ == 0 ? usearch_cancel_k : usearch_continue_k;
    };
    us::outcome<usearch_refine_report_t> const stopped = on.refine(
        {.refine = {.memory_limit = 0}, .progress = {cancel_first}, .plan = &refine_plan});
    expect_true(stopped.status == usearch_status_cancelled_k, "A cancel must stop the planned round");
    expect_true(refine_plan.handle() != nullptr, "A cancel must leave the plan live");
    expect_success(on.refine({.plan = &refine_plan}), "continuing the planned round");
    expect_true(refine_plan.phase().phase == usearch_phase_done_k, "A continued round must run to its end");
    us::outcome<usearch_refine_report_t> const finished = on.refine_plan_finish(std::move(refine_plan));
    expect_success(finished, "finishing the refine plan");
    expect_true(std::strlen(usearch_refine_verdict_name(finished.value.verdict)) != 0, "Every verdict has a name");

    char const* const path = "usearch_test_cpp.safegraphs";
    usearch_bytes_view_t const path_view {reinterpret_cast<std::uint8_t const*>(path), std::strlen(path)};
    expect_success(on.save(path_view), "saving a monolith");
    // RAM copies a checkpoint in, so a view in place serves it from storage.
    us::outcome<us::executor_t> const storage = us::executor_t::for_nvme({.threads = 1});
    expect_success(storage, "spawning a storage executor");
    us::outcome<us::dense_index_t> viewed = us::dense_index_t::view_nvme({.path = path_view}, storage.value);
    expect_success(viewed, "viewing the monolith");
    auto const on_storage = viewed.value.on(storage.value);
    us::outcome<> const refused = on_storage.insert<float>({keys.data(), 1}, {vectors.data(), 1, dimensions});
    expect_true(refused.status == usearch_status_read_only_k, "A viewed handle must refuse every mutation");
    us::outcome<usearch_matches_view_t> const found = on_storage.search<float>(
        {vectors.data() + probe * dimensions, 1, dimensions}, 1, arena.value);
    expect_success(found, "searching the view");
    expect_true(same_key(found.value.keys[0], keys[probe]), "The view must find a stored entry at itself");

    us::outcome<us::dense_cursor_t> const cursor = index.value.cursor_init();
    expect_success(cursor, "opening a cursor");
    us::outcome<us::cursor_batch_t> const pulled = on.cursor_next(cursor.value, rows, arena.value);
    expect_success(pulled, "pulling every entry through the cursor");
    expect_true(pulled.value.written == rows, "One pull wide enough must export every entry");

    constexpr std::size_t gathered_rows = 10;
    usearch_key_t gathered_keys[gathered_rows];
    for (std::size_t row = 0; row != gathered_rows; ++row) gathered_keys[row] = key_of(rows + row);
    us::outcome<us::dense_index_t> other = us::dense_index_t::init_ram(geometry);
    expect_success(other, "creating a second index");
    expect_success(
        other.value.on(executor.value).insert<float>(gathered_keys, {vectors.data(), gathered_rows, dimensions}),
        "inserting into the second index");
    usearch_dense_index_t const gathered[2] = {index.value.handle(), other.value.handle()};
    expect_success(on.reshard({gathered, 2}, {gathered, 1}), "gathering two sources into one");
    expect_true(on.describe().value.size == rows + gathered_rows, "A gather keeps every entry of its sources");

    // A cut moves one source into several targets, emptying it.
    us::outcome<us::dense_index_t> first_part = us::dense_index_t::init_ram(geometry);
    us::outcome<us::dense_index_t> second_part = us::dense_index_t::init_ram(geometry);
    expect_success(first_part, "creating the first part");
    expect_success(second_part, "creating the second part");
    usearch_dense_index_t const source[1] = {index.value.handle()};
    usearch_dense_index_t const parts[2] = {first_part.value.handle(), second_part.value.handle()};
    expect_success(on.reshard(source, parts, {.seed = 7}), "cutting one source into two targets");
    std::size_t const first_size = first_part.value.on(executor.value).describe().value.size;
    std::size_t const second_size = second_part.value.on(executor.value).describe().value.size;
    expect_true(on.describe().value.size == 0, "A cut source must end empty");
    expect_true(first_size + second_size == rows + gathered_rows, "A cut keeps every entry of its source");
    expect_true(first_size != 0 && second_size != 0, "A cut must give every target a part");

    std::remove(path);
}

int main(int argc, char** argv) {
    start_test_run(argc, argv);
    run_test_case("test_readme_quickstart", test_readme_quickstart);
    run_test_case("test_dense_round_trip", test_dense_round_trip);
    us::outcome<us::executor_t> const executor = us::executor_t::for_ram();
    us::outcome<us::arena_t> arena = us::arena_t::make({.executor = executor.value.handle()});
    if (!expect_success(executor, "spawning an executor") || !expect_success(arena, "making an arena"))
        return finish_test_run();
    run_test_case("test_sequences", test_sequences, executor.value, arena.value);
    run_test_case("test_shapes", test_shapes, executor.value, arena.value);
    run_test_case("test_geospatial", test_geospatial, executor.value, arena.value);
    run_test_case("test_plans_and_reshards<sequence_fixture_t>", test_plans_and_reshards<sequence_fixture_t>,
                  executor.value);
    run_test_case("test_plans_and_reshards<shapes_fixture_t>", test_plans_and_reshards<shapes_fixture_t>,
                  executor.value);
    run_test_case("test_plans_and_reshards<geospatial_fixture_t>", test_plans_and_reshards<geospatial_fixture_t>,
                  executor.value);
    return finish_test_run();
}
