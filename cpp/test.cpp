/**
 *  @file cpp/test.cpp
 *  @author Ash Vardanian
 *  @date October 14, 2026
 *  @brief The smoke test of the C++ wrappers: one dense round trip through every handle they own,
 *      an awaited deferred call, then the sequence, shapes and geospatial cases the Python modality
 *      suite runs, against both the engine and the brute-force stub.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include <csignal> // `std::signal`, `std::raise`, `SIGSEGV`, `SIGABRT`, `SIGTRAP`
#include <cctype>  // `std::tolower` - a setting's flag spelling
#include <cmath>   // `std::abs`, `std::llround`, `std::isfinite`
#include <cstdint> // `std::uint8_t`, `std::uint64_t`
#include <cstdio>  // `std::FILE`, `std::fflush` - the run's lines
#include <cstdlib> // `std::exit`, `std::getenv`
#include <cstring> // `std::memcmp`, `std::strcmp`, `std::strlen`

#include <algorithm>       // `std::max`
#include <array>           // `std::array` - the test loop's ring
#include <atomic>          // `std::atomic` - the run's failed checks
#include <charconv>        // `std::from_chars` - the settings
#include <chrono>          // `std::chrono::steady_clock` - case timings and the wait limit
#include <coroutine>       // `std::suspend_never` - the awaiting test task
#include <functional>      // `std::invoke` - the cases
#include <limits>          // `std::numeric_limits`
#include <random>          // `std::mt19937`, `std::normal_distribution`, `std::uniform_int_distribution`
#include <regex>           // `std::regex` - the filter
#include <source_location> // `std::source_location` - where a check failed
#include <string_view>     // `std::string_view` - the settings' names
#include <system_error>    // `std::errc`
#include <thread>          // `std::this_thread::yield` - the bounded waits
#include <type_traits>     // `std::invoke_result_t`, `std::type_identity_t` - case results, check contexts

#include <fmt/format.h>

#include "usearch/usearch.hpp" // `us::dense_index_t`, `us::deferred`, `us::awaiter` and the other modalities

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

/** Writes one line to @p stream, formatted from @p format and @p args, then its newline. */
template <typename... args_types_>
void print_line(std::FILE* stream, fmt::format_string<args_types_...> format, args_types_&&... args) {
    fmt::println(stream, format, std::forward<args_types_>(args)...);
}

#pragma region Shared Settings

/** Exits with one line naming a setting that does not parse, so a bad value never runs anything. */
[[noreturn]] inline void exit_unparsed(char const* variable, char const* text, char const* expected) noexcept {
    print_line(stderr, "{}=\"{}\" does not parse, expected {}", variable, text, expected);
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
    auto const [end, parse_status] = std::from_chars(spelled.data(), spelled.data() + spelled.size(), count);
    if (parse_status != std::errc {} || end != spelled.data() + spelled.size())
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
    auto const [end, parse_status] = std::from_chars(spelled.data(), spelled.data() + spelled.size(), count);
    std::string_view const unit = spelled.substr(static_cast<std::size_t>(end - spelled.data()));
    if (parse_status == std::errc {} && count != 0) {
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
    auto const [end, parse_status] = std::from_chars(spelled.data(), spelled.data() + spelled.size(), seed);
    if (parse_status != std::errc {} || end != spelled.data() + spelled.size())
        exit_unparsed(variable, text, "an unsigned integer or random");
    return seed;
}

/** Which labels a run selects: those its regex matches somewhere, every label while unset. */
struct regex_filter_t {
    std::string_view text;
    std::regex regex {""};

    bool selects(std::string_view label) const noexcept { return std::regex_search(label.begin(), label.end(), regex); }
};

/** The filter a setting spells, compiled once; one that does not compile exits as a bad setting. */
inline regex_filter_t read_regex_filter_setting(int argc, char** argv, char const* variable,
                                                std::string_view flag) noexcept {
    regex_filter_t filter;
    char const* text = setting_text(argc, argv, variable, flag);
    if (!text) return filter;
    filter.text = text;
#if defined(__cpp_exceptions) && __cpp_exceptions
    try {
        filter.regex.assign(filter.text.begin(), filter.text.end());
    }
    catch (std::regex_error const&) {
        exit_unparsed(variable, text, "a regular expression");
    }
#else
    filter.regex.assign(filter.text.begin(), filter.text.end());
#endif
    return filter;
}

/** What every suite and bench reads before its own settings; @c program starts the rerun lines. */
struct common_settings_t {
    std::uint64_t seed = 42;
    regex_filter_t regex_filter;
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
 *  @brief SplitMix64 over one 64-bit counter, seeded by a @ref stream_key , bit-identical to the
 *      JavaScript generator NumKong ships.
 *
 *  Distributions are not specified to draw the same sequence on every standard library, so a
 *  failure would stop reproducing from its seed when the toolchain changed; @ref below is the
 *  portable bounded draw.
 */
struct split_mix_generator_t {
    using result_type = std::uint64_t;
    std::uint64_t counter = 0;

    explicit split_mix_generator_t(std::uint64_t key) noexcept : counter(key) {}
    static constexpr result_type(min)() noexcept { return 0; }
    static constexpr result_type(max)() noexcept { return ~result_type {0}; }
    result_type operator()() noexcept { return split_mix_bits(counter += 0x9E3779B97F4A7C15ull); }

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
    print_line(stdout, "- Seed: {}, scale: {}", test_run.settings.seed, test_run.settings.scale);
    print_line(
        stdout, "- Filter: {}",
        test_run.settings.regex_filter.text.empty() ? std::string_view("none") : test_run.settings.regex_filter.text);
}

/** Counts one failed check at @p where , and traps when the run asked to stop at the first. */
inline void record_failed_check(std::source_location where) noexcept {
    std::fflush(stdout); // so a log that merges both streams keeps the failure under its case
    print_line(stderr, "  check failed at {}:{}", where.file_name(), where.line());
    test_run.last_failed_file.store(where.file_name(), std::memory_order_relaxed);
    test_run.last_failed_line.store(where.line(), std::memory_order_relaxed);
    test_run.failed_checks.fetch_add(1, std::memory_order_relaxed);
    if (test_run.settings.trap_on_failure) trap_into_debugger();
}

/** Holds when @p is_satisfied does; otherwise counts a failed check at the caller's line. */
inline bool expect_true(bool is_satisfied, std::source_location where = std::source_location::current()) noexcept {
    if (!is_satisfied) record_failed_check(where);
    return is_satisfied;
}

/** A case label spelled as a regex that matches only itself, every metacharacter escaped. */
struct regex_escaped_t {
    std::string_view label;
};

/** Writes a @ref regex_escaped_t straight to the output, a backslash before each metacharacter. */
template <>
struct fmt::formatter<regex_escaped_t> {
    constexpr auto parse(fmt::format_parse_context& context) { return context.begin(); }
    template <typename context_type_>
    auto format(regex_escaped_t const& escaped, context_type_& context) const {
        auto output = context.out();
        for (char const character : escaped.label) {
            if (std::string_view {"\\^$.|?*+()[]{}"}.find(character) != std::string_view::npos) *output++ = '\\';
            *output++ = character;
        }
        return output;
    }
};

/** Prints one case's verdict, and the line that reruns it alone when it failed. */
inline void report_test_case(char const* label, std::size_t failed_checks, char const* skip_reason,
                             double seconds) noexcept {
    if (skip_reason) {
        print_line(stdout, "- {} ... skipped: {}", label, skip_reason);
        return;
    }
    if (failed_checks == 0) {
        print_line(stdout, "- {} ... ok ({:.2f} s)", label, seconds);
        return;
    }
    std::fflush(stdout);
    print_line(stderr, "- {} ... FAILED: {} failed check(s)", label, failed_checks);
    print_line(stderr, "  rerun: {} --seed {} --filter '^{}$'", test_run.settings.program, test_run.settings.seed,
               regex_escaped_t {label});
}

/** Runs one case unless the filter leaves it out, and records whether it passed or skipped. */
template <typename case_function_type_, typename... argument_types_>
void run_test_case(char const* label, case_function_type_&& case_function, argument_types_&&... arguments) noexcept {
    if (!test_run.settings.regex_filter.selects(label)) return;
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
    print_line(stdout, "{} passed, {} failed, {} skipped, seed {}", test_run.passed_cases, test_run.failed_cases,
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
    settings.regex_filter = read_regex_filter_setting(argc, argv, "USEARCH_FILTER", "--filter");
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

/**
 *  @brief A format string bundled with the @c std::source_location of the call that spelled it.
 *
 *  The implicit conversion from a string literal captures the caller's line behind the variadic
 *  arguments of a check's context, where a defaulted trailing parameter cannot go, and checks the
 *  pattern against those arguments at compile time, as @c fmt::format_string does.
 */
template <typename... args_types_>
struct located_format {
    fmt::format_string<args_types_...> format;
    std::source_location location;

    template <typename text_type_>
    consteval located_format(text_type_ const& text,
                             std::source_location location = std::source_location::current()) noexcept
        : format(text), location(location) {}
};

/** Prints @p context formatted with @p args , indented under a failed check's location line. */
template <typename... args_types_>
void print_context_(located_format<std::type_identity_t<args_types_>...> context, args_types_ const&... args) noexcept {
    fmt::println(stderr, "    {}", fmt::vformat(context.format.get(), fmt::make_format_args(args...)));
}

/** Holds when @p is_satisfied does; otherwise counts a failed check at the caller's line and prints
 *  @p context formatted with @p args beneath it. */
template <typename... args_types_>
bool expect_true(bool is_satisfied, located_format<std::type_identity_t<args_types_>...> context,
                 args_types_ const&... args) noexcept {
    if (is_satisfied) return true;
    record_failed_check(context.location);
    print_context_<args_types_...>(context, args...);
    return false;
}

/** Holds when @p status succeeded; otherwise counts a failed check and prints @p context. */
inline bool expect_success(us::status_t const& status, located_format<> context) noexcept {
    if (!status.failed()) return true;
    record_failed_check(context.location);
    fmt::println(stderr, "    {}: {}", context.format.get(),
                 status.note ? status.note : usearch_status_name(static_cast<usearch_status_t>(status.code)));
    return false;
}

/** Holds when @p result carries a value; otherwise counts a failed check and prints @p context. */
template <typename value_type_>
bool expect_success(us::expected<value_type_> const& result, located_format<> context) noexcept {
    return expect_success(result.status, context);
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

/** A cancel at the first barrier a call reports, so a plan stops with work still ahead. */
struct cancel_at_first_t {
    usearch_continuation_t operator()(std::size_t, std::size_t) noexcept { return usearch_cancel_k; }
};

/** A predicate counting the keys it is asked about, which only one thread could call safely, as a
 *  @c mutable lambda would. */
struct counting_predicate_t {
    std::size_t asked = 0;
    bool operator()(usearch_key_t) noexcept { return ++asked != 0; }
};

/** A predicate reading only what it was built with, safe on every executor thread at once. */
struct reading_predicate_t {
    std::size_t refused_first_byte = 0;
    bool operator()(usearch_key_t key) const noexcept { return key.bytes[0] != refused_first_byte; }
};

static_assert(!us::a_key_predicate_handler<counting_predicate_t>, "a predicate runs on several threads at once");
static_assert(us::a_key_predicate_handler<reading_predicate_t>);

/** A batch predicate counting the batches it is asked about, as a @c mutable lambda would. */
struct counting_batch_predicate_t {
    std::size_t batches = 0;
    void operator()(std::span<usearch_key_t const>, std::span<usearch_admission_t>) noexcept { ++batches; }
};

/** A batch predicate admitting every key, reading nothing it changes. */
struct admitting_batch_predicate_t {
    void operator()(std::span<usearch_key_t const>, std::span<usearch_admission_t> admissions) const noexcept {
        for (usearch_admission_t& admission : admissions) admission = usearch_admit_k;
    }
};

static_assert(!us::a_keys_predicate_handler<counting_batch_predicate_t>, "a predicate runs on several threads at once");
static_assert(us::a_keys_predicate_handler<admitting_batch_predicate_t>);

/** A terrain counting the tiles it fills, as a @c mutable lambda would. */
struct counting_terrain_t {
    std::size_t tiles = 0;
    void operator()(std::size_t, std::size_t, std::size_t, std::size_t, float*, std::size_t) noexcept { ++tiles; }
};

/** A terrain filling every tile with sea level, rows @p heights_stride bytes apart. */
struct flat_terrain_t {
    void operator()(std::size_t, std::size_t, std::size_t rows, std::size_t columns, float* heights,
                    std::size_t heights_stride) const noexcept {
        for (std::size_t row = 0; row != rows; ++row)
            std::fill_n(reinterpret_cast<float*>(reinterpret_cast<char*>(heights) + row * heights_stride), columns,
                        0.f);
    }
};

static_assert(!us::a_terrain_handler<counting_terrain_t>, "a terrain fills tiles on several threads at once");
static_assert(us::a_terrain_handler<flat_terrain_t>);

#pragma region Modality Fixtures

/** The most entries a fixture holds, so its storage is a fixed array. */
static constexpr std::size_t fixture_capacity = 64;

/** Random lowercase words of 8 to 16 bytes, as one byte tape. */
struct sequence_fixture_t {
    using index_t = us::sequence_index_t;
    using on_t = us::sequence_index_on<usearch_completion_blocking_k>;
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
    static us::expected<index_t> init() {
        return index_t::init_ram({.metric = usearch_sequence_metric_levenshtein_bytes_k});
    }
    us::status_t insert(on_t const& on, std::span<usearch_key_t const> keys) const { return on.insert(keys, batch()); }
};

/** Random clouds of 4 to 12 points in the unit cube, as one point tape. */
struct shapes_fixture_t {
    using index_t = us::shapes_index_t;
    using on_t = us::shapes_index_on<usearch_completion_blocking_k>;
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
    static us::expected<index_t> init() {
        return index_t::init_ram({.metric = usearch_shapes_metric_chamfer_k, .quantization = usearch_scalar_f32_k});
    }
    us::status_t insert(on_t const& on, std::span<usearch_key_t const> keys) const {
        return on.insert<float>(keys, batch());
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
    using on_t = us::geospatial_index_on<usearch_completion_blocking_k>;
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
    static us::expected<index_t> init() { return index_t::init_ram({}); }
    us::status_t insert(on_t const& on, std::span<usearch_key_t const> keys) const {
        return on.insert(keys, std::span<usearch_geospatial_observation_t const>(observations, entries));
    }
};

/** A host index of @p fixture_type_ holding @p count random entries under consecutive keys from
 *  @p first_key. */
template <typename fixture_type_>
static typename fixture_type_::index_t built(us::executor_t const& executor, std::size_t count,
                                             std::size_t first_key = 0, std::uint32_t seed = 0) {
    auto [index, index_status] = fixture_type_::init();
    expect_success(index_status, "creating a modality index");
    if (count == 0) return std::move(index);
    usearch_key_t keys[fixture_capacity] {};
    for (std::size_t entry = 0; entry != count; ++entry) keys[entry] = key_of(first_key + entry);
    expect_success(fixture_type_(count, seed).insert(index.on(executor), {keys, count}), "inserting modality entries");
    return std::move(index);
}

template <typename index_type_>
static std::size_t size_of(index_type_ const& index) {
    auto [description, describe_status] = index.describe();
    expect_success(describe_status, "describing an index");
    return description.size;
}

#pragma endregion Modality Fixtures

/** A metric kind reads back from its own spelling, and a word no kind spells is refused. */
template <typename metric_type_>
static void check_metric_name(char const* (*name_of)(metric_type_),
                              us::expected<metric_type_> (*from_name)(usearch_bytes_view_t), metric_type_ metric) {
    char const* const spelled = name_of(metric);
    expect_true(spelled && std::strlen(spelled) != 0, "Every metric kind has a name");
    auto [read, status] = from_name(us::bytes_view(spelled));
    expect_true(!status.failed() && read == metric, "A metric kind reads back from its own name");
    expect_true(from_name(us::bytes_view("no-such-metric")).status == us::status_code_t::invalid_argument_k,
                "A word no metric kind spells must be refused");
}

/** A sample draws distinct keys placed in three dimensions, never draws a shown key again, and
 *  refuses shown keys without their coordinates. */
template <typename index_type_>
static void check_sample(index_type_ const& index, us::executor_t const& executor, us::arena_t& arena) {
    constexpr std::size_t wanted = 8;
    usearch_key_t shown[wanted] {};
    float placed[wanted][3] {};
    {
        auto [first, status] = index.on(executor).sample(wanted, arena, {.seed = 3});
        expect_success(status, "sampling an index");
        expect_true(first.written == wanted && first.coordinates.columns == 3,
                    "A sample of a populated index draws every key wanted, in three dimensions");
        for (std::size_t draw = 0; draw != first.written; ++draw) {
            shown[draw] = first.keys.data[draw];
            std::memcpy(placed[draw], first.coordinate(draw), sizeof(placed[draw]));
        }
    }
    {
        auto [again, status] = index.on(executor).sample(wanted, arena, {.seed = 3});
        expect_true(!status.failed() && again.written == wanted && same_key(again.keys.data[0], shown[0]),
                    "Equal seeds over an unchanged index draw equally");
    }
    {
        auto [next, status] = index.on(executor).sample(
            wanted, arena, {.shown = shown, .shown_coordinates = {&placed[0][0], wanted, 3}, .seed = 3});
        expect_success(status, "sampling around shown keys");
        for (std::size_t draw = 0; draw != next.written; ++draw)
            for (usearch_key_t const& key : shown)
                expect_true(!same_key(next.keys.data[draw], key), "A shown key must never be drawn again");
    }
    expect_true(
        index.on(executor).sample(wanted, arena, {.shown = shown}).status == us::status_code_t::invalid_argument_k,
        "Shown keys without coordinates must be refused");
}

/** The plan and reshard cases of <tt>test/modalities.py</tt>, which each modality answers alike. */
template <typename fixture_type_>
static void test_plans_and_reshards(us::executor_t const& executor) {
    using index_t = typename fixture_type_::index_t;
    using on_t = typename fixture_type_::on_t;
    using raw_t = typename fixture_type_::raw_t;

    // A sample draws keys from the graph of every modality alike.
    {
        index_t index = built<fixture_type_>(executor, 64);
        auto [arena, arena_status] = us::arena_t::make({.executor = executor.handle()});
        expect_success(arena_status, "making a sample arena");
        check_sample(index, executor, arena);
    }

    // A round reports what it did, its closing audit judging every entry.
    {
        index_t index = built<fixture_type_>(executor, 64);
        auto [report, refine_status] = index.on(executor).refine();
        expect_success(refine_status, "refining a modality index");
        expect_true(report.changed <= report.reselected, "A round changes only entries it reselected");
        expect_true(report.reachability.audited == 64, "A round's closing audit must judge every entry");
        auto [counted, count_status] = index.count(key_of(17));
        expect_true(!count_status.failed() && counted.entries == 1, "A refined index keeps every entry");
    }

    // A plan keeps a stopped round live, and its finish closes it once.
    {
        index_t index = built<fixture_type_>(executor, 64);
        typename fixture_type_::refine_plan_t plan;
        cancel_at_first_t cancel;
        expect_true(
            index.on(executor).refine({.progress = {cancel}, .plan = &plan}).status == us::status_code_t::cancelled_k,
            "A cancel must stop the planned round");
        expect_true(plan.phase().phase != usearch_phase_done_k && plan.phase().completed <= plan.phase().total,
                    "A stopped plan must sit before its end");
        expect_success(index.on(executor).refine({.plan = &plan}), "continuing the planned round");
        expect_true(plan.phase().phase == usearch_phase_done_k, "A continued round must run to its end");
        auto [finished, finish_status] = index.on(executor).refine_plan_finish(std::move(plan));
        expect_success(finish_status, "finishing the refine plan");
        expect_true(finished.reachability.audited == 64, "A finish reports the whole plan");
        expect_true(
            index.on(executor).refine_plan_finish(std::move(plan)).status == us::status_code_t::invalid_argument_k,
            "A finished plan must not finish again");
    }

    // A plan lives across calls on one executor, so the calling thread cannot open one.
    {
        index_t index = built<fixture_type_>(executor, 0);
        typename fixture_type_::refine_plan_t plan;
        expect_true(index.refine({.plan = &plan}).status == us::status_code_t::invalid_argument_k,
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
        auto [fortieth, fortieth_status] = first.count(key_of(40));
        auto [eightieth, eightieth_status] = first.count(key_of(80));
        expect_true(
            !fortieth_status.failed() && fortieth.entries == 1 && !eightieth_status.failed() && eightieth.entries == 1,
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
        auto [stopped, stopped_status] = source.on(executor).reshard(from, into, {.progress = {cancel}, .plan = &plan});
        expect_true(stopped_status == us::status_code_t::cancelled_k || stopped_status == us::status_code_t::success_k,
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
        expect_true(index.on(executor).reshard(both, both).status == us::status_code_t::unsupported_k,
                    "A recut of one handle must be unsupported");
        expect_true(size_of(index) == 16, "A refused recut moves nothing");
    }

    // Empty or repeated lists are malformed and a target holding foreign entries mismatches.
    {
        index_t first = built<fixture_type_>(executor, 8);
        index_t second = built<fixture_type_>(executor, 8, 8, 1);
        raw_t const firsts[2] = {first.handle(), first.handle()};
        raw_t const seconds[1] = {second.handle()};
        on_t const on = first.on(executor);
        expect_true(on.reshard({}, {firsts, 1}).status == us::status_code_t::invalid_argument_k,
                    "An empty source list is malformed");
        expect_true(on.reshard(firsts, {firsts, 1}).status == us::status_code_t::invalid_argument_k,
                    "A repeated source is malformed");
        expect_true(on.reshard({firsts, 1}, seconds).status == us::status_code_t::config_mismatch_k,
                    "A target holding foreign entries mismatches");
        expect_true(size_of(first) == 8 && size_of(second) == 8, "A refusal moves nothing");
    }
}

/** Every sequence searched for comes back first under its own key, at distance zero, and survives a
 *  checkpoint. */
static void test_sequences(us::executor_t const& executor, us::arena_t& arena) {
    auto [index, index_status] = us::sequence_index_t::init_ram({.metric = usearch_sequence_metric_levenshtein_utf8_k});
    expect_success(index_status, "creating a sequence index");
    us::sequence_index_on<usearch_completion_blocking_k> const on = index.on(executor);
    {
        auto [empty, status] = on.describe();
        expect_true(!status.failed() && empty.size == 0 && empty.gpu_class == usearch_gpu_class_unknown_k,
                    "A fresh host index is empty and on no GPU");
        expect_true(empty.metadata.metric == usearch_sequence_metric_levenshtein_utf8_k,
                    "The frozen metric reads back");
    }

    std::uint8_t const words[] = {'k', 'i', 't', 't', 'e', 'n', 's', 'i', 't', 't',
                                  'i', 'n', 'g', 'm', 'i', 't', 't', 'e', 'n'};
    std::uint64_t const offsets[] = {0, 6, 13, 19};
    usearch_key_t const keys[] = {key_of(1), key_of(2), key_of(3)};
    expect_success(on.insert(keys, {words, offsets}), "inserting sequences");
    expect_true(on.insert({keys, 1}, {words, offsets}) == us::status_code_t::shape_mismatch_k,
                "Keys and sequences of different counts must be refused before the call crosses");

    std::uint64_t const query_offsets[] = {0, 7};
    {
        auto [found, status] = on.search({words + 6, query_offsets}, 1, arena);
        expect_success(status, "searching a sequence");
        expect_true(same_key(found.keys[0], keys[1]) && found.scores[0] == 0,
                    "A stored sequence must find itself first, at distance zero");
    }
    {
        auto [counted, status] = on.count(keys[1]);
        expect_true(!status.failed() && counted.entries == 1 && counted.total_bytes == 7, "A key's bytes are counted");
    }
    {
        auto [fetched, status] = on.get(keys[1], arena);
        expect_true(!status.failed() && fetched.written == 1 && fetched.entry(0).size() == 7 &&
                        std::memcmp(fetched.entry(0).data(), "sitting", 7) == 0,
                    "A key's sequence must come back whole");
    }

    char const* const path = "usearch_test_cpp.sequences";
    expect_success(on.save(us::bytes_view(path)), "saving sequences");
    {
        auto [metadata, status] = us::sequence_index_t::metadata_read({.path = us::bytes_view(path)});
        expect_true(!status.failed() && metadata.metric == usearch_sequence_metric_levenshtein_utf8_k,
                    "A checkpoint reports its metric");
    }
    {
        auto [loaded, status] = us::sequence_index_t::load_ram({.path = us::bytes_view(path)});
        expect_success(status, "loading sequences");
        expect_true(size_of(loaded) == 3, "A loaded checkpoint keeps every entry");
    }
    std::remove(path);
    {
        auto [monolith, status] = on.save_buffer(arena);
        expect_success(status, "saving sequences into a buffer");
        auto [reloaded, load_status] = us::sequence_index_t::load_ram_buffer({.buffer = monolith});
        expect_true(!load_status.failed() && size_of(reloaded) == 3, "A buffer reloads every entry");
    }

    // A VRAM index either opens on a GPU or answers the library's own refusal.
    {
        auto [device, status] = us::sequence_index_t::init_vram({});
        if (status.failed()) expect_true(status.code != us::status_code_t::deferred_k, "A refusal is final");
        else {
            auto [described, describe_status] = device.describe();
            expect_true(!describe_status.failed() && described.gpu_class != usearch_gpu_class_unknown_k,
                        "A VRAM index opens on a GPU or forwards the library's status");
        }
    }
    check_metric_name(us::sequence_metric_name, us::sequence_metric_from_name,
                      usearch_sequence_metric_levenshtein_utf8_k);
}

/** Shapes of different point counts share one batch, and each finds itself first. */
static void test_shapes(us::executor_t const& executor, us::arena_t& arena) {
    auto [index, index_status] = us::shapes_index_t::init_ram(
        {.metric = usearch_shapes_metric_hausdorff_k, .quantization = usearch_scalar_f32_k});
    expect_success(index_status, "creating a shapes index");
    us::shapes_index_on<usearch_completion_blocking_k> const on = index.on(executor);

    shapes_fixture_t const clouds(3, 42);
    usearch_key_t const keys[] = {key_of(1), key_of(2), key_of(3)};
    expect_success(on.insert<float>(keys, clouds.batch()), "inserting shapes");
    {
        auto [described, status] = on.describe();
        expect_true(!status.failed() && described.metadata.metric == usearch_shapes_metric_hausdorff_k,
                    "The frozen metric reads back");
        expect_true(described.size == 3 && described.total_points == clouds.offsets[clouds.entries],
                    "Every point of every shape is stored");
    }

    std::size_t const last_points = clouds.offsets[3] - clouds.offsets[2];
    std::uint64_t const last_offsets[] = {0, last_points};
    {
        auto [found, status] = on.search<float>({clouds.points + clouds.offsets[2] * 3, last_offsets}, 1, arena);
        expect_true(!status.failed() && same_key(found.keys[0], keys[2]), "A stored shape must find itself first");
    }
    {
        auto [fetched, status] = on.get<float>(keys[2], arena);
        expect_true(!status.failed() && fetched.written == 1 && fetched.entry(0).size() == last_points * 3,
                    "A key's shape must come back with every point");
    }
    {
        auto [counted, status] = on.count(keys[2]);
        expect_true(!status.failed() && counted.largest_points == last_points, "A key's points are counted");
    }

    char const* const path = "usearch_test_cpp.shapes";
    expect_success(on.save(us::bytes_view(path)), "saving shapes");
    {
        auto [loaded, status] = us::shapes_index_t::load_ram({.path = us::bytes_view(path)});
        expect_true(!status.failed() && size_of(loaded) == 3, "A loaded checkpoint keeps every entry");
    }
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
    auto [index, index_status] = us::geospatial_index_t::init_ram({});
    expect_success(index_status, "creating a geospatial index");
    us::geospatial_index_on<usearch_completion_blocking_k> const on = index.on(executor);

    usearch_geospatial_observation_t const places[] = {observation_at(48.8566, 2.3522), observation_at(48.8656, 2.3522),
                                                       observation_at(52.52, 13.405)};
    usearch_key_t const keys[] = {key_of(1), key_of(2), key_of(3)};
    expect_success(on.insert(keys, places), "inserting observations");
    expect_true(size_of(index) == 3, "Every observation is stored");

    {
        auto [found, status] = on.search({places + 2, 1}, usearch_geospatial_relation_near_k, {}, 1, arena);
        expect_true(!status.failed() && same_key(found.keys[0], keys[2]) && found.scores[0] == 0,
                    "A stored observation must find itself first, at zero metres");
    }
    {
        auto [nearby, status] = on.within({places, 1}, usearch_geospatial_relation_near_k, {.metres = 2000}, arena);
        expect_success(status, "reporting observations within a radius");
        expect_true(nearby.view.offsets[1] == 2, "A two-kilometre radius admits the two Paris observations only");
    }
    {
        auto [fetched, status] = on.get(keys[2], arena);
        expect_true(
            !status.failed() && fetched.written == 1 && std::abs(fetched.first->latitude_degrees - 52.52) < 1e-6,
            "A key's observation must come back within the stored rounding");
    }
    {
        auto [counted, status] = on.count(keys[2]);
        expect_true(!status.failed() && counted.entries == 1, "A key's observations are counted");
    }

    char const* const path = "usearch_test_cpp.geospatial";
    expect_success(on.save(us::bytes_view(path)), "saving observations");
    {
        auto [loaded, status] = us::geospatial_index_t::load_ram({.path = us::bytes_view(path)});
        expect_true(!status.failed() && size_of(loaded) == 3, "A loaded checkpoint keeps every entry");
    }
    std::remove(path);
}

/** Mirrors the Quickstart of <tt>cpp/README.md</tt> line by line, which a C++ test cannot run
 *  verbatim. It differs in three lines: <tt>int main()</tt> becomes this function, the final
 *  @c std::printf becomes @c fmt::println and also writes the count into @p found, and the
 *  includes and the namespace alias sit at the top of this file. */
static int quickstart(std::size_t& found) {
    auto [index, index_status] = us::dense_index_t::init_ram(
        {.metric = usearch_dense_metric_angular_k, .quantization = usearch_scalar_i8_k, .dimensions = 4});
    if (index_status.failed()) return 1;
    us::status_t const reserved = index.reserve(1000); // the calling thread; inserts grow on demand anyway

    usearch_key_t keys[2] {};
    keys[0].bytes[15] = 42;
    keys[1].bytes[15] = 43;
    float const slab[8] = {0.2f, 0.1f, 0.2f, 0.1f, 0.1f, 0.2f, 0.1f, 0.2f};
    if (reserved.failed() || index.insert<float>(keys, {slab, 2, 4}).failed()) return 1;

    auto [executor, executor_status] = us::executor_t::for_ram();
    auto [arena, arena_status] = us::arena_t::make({.executor = executor.handle()});
    auto [local_arena, local_arena_status] = us::arena_t::make();
    float const query[4] = {0.2f, 0.1f, 0.2f, 0.1f};
    auto [local, local_status] = index.search<float>({query, 1, 4}, 10, local_arena);          // the calling thread
    auto [pooled, pooled_status] = index.on(executor).search<float>({query, 1, 4}, 10, arena); // blocks on the executor
    if (local_status.failed() || pooled_status.failed()) return 1;
    us::deferred<usearch_matches_view_t> call = index.deferred(executor).search<float>({slab, 2, 4}, 10, arena);
    auto [matches, status] = call.wait(); // blocks until final; destruction waits too
    if (status.failed()) return 1;
    fmt::println("Found {} and {} matches", local.counts[0], found = matches.counts[1]);
    return 0;
}

/** A host arena reports host memory, a GPU-less listing answers a count, and the linked library
 *  names its version. */
static void test_arena_devices_and_version() {
    auto [arena, arena_status] = us::arena_t::make();
    expect_success(arena_status, "making a host arena");
    auto [description, describe_status] = arena.describe();
    expect_success(describe_status, "describing the arena");
    expect_true(description.residency == usearch_residency_ram_k && description.device == -1 &&
                    description.host_access == usearch_host_access_readable_k,
                "A host arena must hold readable host memory on no device");
    expect_true(us::arena_t {}.describe().status == us::status_code_t::invalid_argument_k,
                "An empty arena must be refused");
    expect_success(us::gpu_devices({}).status, "counting the GPUs");
    expect_true(!us::version_name().empty(), "The library must name its version");
}

/** The README's Quickstart finds both of its entries, skipping with the library's note in a build
 *  that leaves its @c i8 index out. */
static case_result_t test_readme_quickstart() {
    auto [probe, probe_status] = us::dense_index_t::init_ram(
        {.metric = usearch_dense_metric_angular_k, .quantization = usearch_scalar_i8_k, .dimensions = 4});
    if (probe_status.code == us::status_code_t::unsupported_k) return case_result_t::skipped(probe_status.note);
    std::size_t found = 0;
    expect_true(quickstart(found) == 0 && found == 2, "The README's Quickstart must find both entries");
    return {};
}

/** A fixed ring of posted polls, the loop a deferred call is awaited on, allocating nothing. */
struct test_loop_t {
    struct posted_t {
        void (*poll)(void*) noexcept;
        void* poll_state;
    };
    std::array<posted_t, 64> ring {};
    std::size_t head = 0, tail = 0;

    void post(void (*poll)(void*) noexcept, void* state) noexcept {
        expect_true(tail - head < ring.size(), "The test loop's ring must never overflow");
        ring[tail++ % ring.size()] = {poll, state};
    }
    void run() noexcept {
        while (head != tail) {
            posted_t const next = ring[head++ % ring.size()];
            next.poll(next.poll_state);
        }
    }
};

/** A coroutine that starts at once and runs to its end on whatever resumes it. */
struct test_task_t {
    struct promise_type {
        test_task_t get_return_object() noexcept { return {}; }
        std::suspend_never initial_suspend() noexcept { return {}; }
        std::suspend_never final_suspend() noexcept { return {}; }
        void return_void() noexcept {}
        void unhandled_exception() noexcept { std::abort(); }
    };
};

/** Plays an engine whose deferred call finishes only after @c turns_left more turns of the loop,
 *  as a real one finishes while its caller polls. */
struct late_engine_t {
    usearch_outcome_t* outcome = nullptr;
    std::size_t* written = nullptr;
    std::size_t turns_left = 3;
    test_loop_t* loop = nullptr;

    static void turn(void* state) noexcept {
        late_engine_t& engine = *static_cast<late_engine_t*>(state);
        if (--engine.turns_left != 0) return engine.loop->post(&turn, state);
        *engine.written = 42;
        engine.outcome->status = usearch_status_success_k;
    }
};

/** Awaits a deferred audit of @p index on @p loop, writing what it judged into @p audited. */
static test_task_t awaited_health(us::dense_index_t const& index, us::executor_t const& executor, test_loop_t& loop,
                                  std::size_t& audited) {
    us::deferred<usearch_reachability_t> call = index.deferred(executor).health();
    auto [reachability, status] = co_await call.wait(loop); // or: call.wait()
    expect_success(status, "awaiting a deferred audit");
    audited = reachability.audited;
}

/** Searches two shards at once on @p loop, as the README's fan-out does, writing both match counts
 *  into @p found. */
static test_task_t awaited_fan_out(us::dense_index_t const& east, us::dense_index_t const& west,
                                   us::executor_t const& east_executor, us::executor_t const& west_executor,
                                   us::rows_view<float> queries, test_loop_t& loop, std::size_t& found) {
    auto [east_arena, east_arena_status] = us::arena_t::make({.executor = east_executor.handle()});
    auto [west_arena, west_arena_status] = us::arena_t::make({.executor = west_executor.handle()});
    expect_success(east_arena_status, "making the east arena");
    expect_success(west_arena_status, "making the west arena");
    us::deferred<usearch_matches_view_t> east_call =
        east.deferred(east_executor).search<float>(queries, 10, east_arena);
    us::deferred<usearch_matches_view_t> west_call =
        west.deferred(west_executor).search<float>(queries, 10, west_arena);
    auto [east_matches, east_status] = co_await east_call.wait(loop); // both shards were already running
    auto [west_matches, west_status] = co_await west_call.wait(loop);
    expect_success(east_status, "awaiting the east shard");
    expect_success(west_status, "awaiting the west shard");
    if (east_status.failed() || west_status.failed()) co_return;
    found = east_matches.counts[0] + west_matches.counts[0];
}

/** Awaits a call @p engine finishes later on @p loop and writes its answer into @p answered. */
static test_task_t awaited_late_call(late_engine_t& engine, test_loop_t& loop, std::size_t& answered) {
    us::deferred<std::size_t> call([&](usearch_outcome_t* outcome, us::outputs<std::size_t>* written) noexcept {
        engine.outcome = outcome;
        engine.written = &written->value;
        outcome->status = usearch_status_deferred_k;
    });
    loop.post(&late_engine_t::turn, &engine);
    auto [value, status] = co_await call.wait(loop);
    expect_success(status, "awaiting a call that finishes later");
    answered = value;
}

/** A coroutine awaits a deferred call through the loop's ring, suspending only while the call is
 *  not final, and gets what a blocking call would have answered. */
static void test_awaited_calls() {
    auto [executor, executor_status] = us::executor_t::for_ram();
    expect_success(executor_status, "spawning an executor");
    auto [index, index_status] = us::dense_index_t::init_ram(
        {.metric = usearch_dense_metric_angular_k, .quantization = usearch_scalar_bf16_k, .dimensions = 4});
    expect_success(index_status, "creating an index");
    usearch_key_t const keys[2] = {key_of(1), key_of(2)};
    float const rows[2][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}};
    expect_success(index.on(executor).insert<float>(keys, {&rows[0][0], 2, 4}), "inserting the rows");

    test_loop_t loop;
    std::size_t audited = 0;
    awaited_health(index, executor, loop, audited);
    loop.run();
    expect_true(audited == 2, "An awaited audit must see every entry");

    auto [west_executor, west_executor_status] = us::executor_t::for_ram();
    expect_success(west_executor_status, "spawning a second executor");
    auto [west, west_status] = us::dense_index_t::init_ram(
        {.metric = usearch_dense_metric_angular_k, .quantization = usearch_scalar_bf16_k, .dimensions = 4});
    expect_success(west_status, "creating a second shard");
    usearch_key_t const west_keys[1] = {key_of(3)};
    expect_success(west.insert<float>(west_keys, {&rows[1][0], 1, 4}), "inserting on the calling thread");
    std::size_t found = 0;
    awaited_fan_out(index, west, executor, west_executor, {&rows[0][0], 1, 4}, loop, found);
    loop.run();
    expect_true(found == 3, "A fan-out must answer every entry of both shards");

    late_engine_t engine {.loop = &loop};
    std::size_t answered = 0;
    awaited_late_call(engine, loop, answered);
    expect_true(answered == 0, "An awaited call must not resume before it is final");
    loop.run();
    expect_true(answered == 42 && engine.turns_left == 0, "An awaited call must resume once final, with its outputs");
    expect_true(loop.head == loop.tail, "The loop must drain every poll it was posted");
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

    auto [executor, executor_status] = us::executor_t::for_ram();
    expect_success(executor_status, "spawning an executor");
    auto [arena, arena_status] = us::arena_t::make({.executor = executor.handle()});
    expect_success(arena_status, "making an arena");
    usearch_dense_init_ram_t const geometry {.metric = usearch_dense_metric_angular_k,
                                             .quantization = usearch_scalar_bf16_k,
                                             .dimensions = dimensions,
                                             .connectivity = 16};
    auto [index, index_status] = us::dense_index_t::init_ram(geometry);
    expect_success(index_status, "creating an index");
    us::dense_index_on<usearch_completion_blocking_k> const on = index.on(executor);

    expect_success(on.reserve(rows), "reserving the entries");
    expect_success(on.insert<float>(keys, all_rows), "inserting the rows");
    {
        auto [described, status] = on.describe();
        expect_success(status, "describing the index");
        expect_true(described.size == rows, "An insertion is all-or-nothing, so every entry must be published");
    }
    expect_true(on.insert<float>({keys.data(), 1}, all_rows) == us::status_code_t::shape_mismatch_k,
                "Keys and rows of different counts must be refused before the call crosses");

    {
        auto [matches, status] = on.search<float>(all_rows, 1, arena);
        expect_success(status, "searching every row");
        std::size_t recalled = 0;
        for (std::size_t row = 0; row != rows; ++row) recalled += same_key(matches.keys[row], keys[row]);
        expect_true(recalled * 10 >= rows * 9, "Self-recall collapsed below 90%");
    }
    {
        auto [counted, status] = on.count(keys[probe]);
        expect_success(status, "counting a key");
        expect_true(counted == 1, "Unique keys hold one entry apiece");
    }
    {
        auto [fetched, status] = on.get<float>(keys[probe], arena);
        expect_success(status, "getting a key");
        float const* const fetched_row = fetched.row(0);
        expect_true(fetched.written == 1 && fetched.view.columns == dimensions && fetched_row[0] != 0,
                    "A key's one row must come back whole");
    }

    // The receiver's type picks the completion, and the deferred call's owner waits on it.
    {
        us::deferred<usearch_reachability_t> audit = index.deferred(executor).health();
        auto [audited, status] = audit.wait();
        expect_success(status, "auditing deferred");
        expect_true(audited.audited == rows, "A deferred audit must see every entry");
    }

    // A sample draws from the graph, and a deferred one answers what the blocking one would have.
    check_sample(index, executor, arena);
    {
        us::deferred<us::sample_t> drawing = index.deferred(executor).sample(16, arena);
        auto [drawn, status] = drawing.wait();
        expect_true(!status.failed() && drawn.written == 16, "A deferred sample draws every key wanted");
    }
    check_metric_name(us::dense_metric_name, us::dense_metric_from_name, usearch_dense_metric_angular_k);
    check_metric_name(us::dense_metric_name, us::dense_metric_from_name, usearch_dense_metric_jaccard_k);

    // A plan keeps its round between calls, and its finish on the receiver closes it.
    us::dense_refine_plan_t refine_plan;
    std::size_t reports = 0;
    auto cancel_first = [&](std::size_t, std::size_t) noexcept {
        return reports++ == 0 ? usearch_cancel_k : usearch_continue_k;
    };
    expect_true(on.refine({.refine = {.memory_limit = 0}, .progress = {cancel_first}, .plan = &refine_plan}).status ==
                    us::status_code_t::cancelled_k,
                "A cancel must stop the planned round");
    expect_true(refine_plan.handle() != nullptr, "A cancel must leave the plan live");
    expect_success(on.refine({.plan = &refine_plan}), "continuing the planned round");
    expect_true(refine_plan.phase().phase == usearch_phase_done_k, "A continued round must run to its end");
    {
        auto [finished, status] = on.refine_plan_finish(std::move(refine_plan));
        expect_success(status, "finishing the refine plan");
        expect_true(std::strlen(usearch_refine_verdict_name(finished.verdict)) != 0, "Every verdict has a name");
    }

    char const* const path = "usearch_test_cpp.safegraphs";
    expect_success(on.save(us::bytes_view(path)), "saving a monolith");
    // RAM copies a checkpoint in, so a view in place serves it from storage.
    {
        auto [storage, storage_status] = us::executor_t::for_nvme({.threads = 1});
        expect_success(storage_status, "spawning a storage executor");
        auto [viewed, view_status] = us::dense_index_t::view_nvme({.path = us::bytes_view(path)}, storage);
        expect_success(view_status, "viewing the monolith");
        us::dense_index_on<usearch_completion_blocking_k> const on_storage = viewed.on(storage);
        expect_true(on_storage.insert<float>({keys.data(), 1}, {vectors.data(), 1, dimensions}) ==
                        us::status_code_t::read_only_k,
                    "A viewed handle must refuse every mutation");
        auto [found, status] = on_storage.search<float>({vectors.data() + probe * dimensions, 1, dimensions}, 1, arena);
        expect_success(status, "searching the view");
        expect_true(same_key(found.keys[0], keys[probe]), "The view must find a stored entry at itself");
    }
    {
        auto [cursor, cursor_status] = index.cursor_init();
        expect_success(cursor_status, "opening a cursor");
        auto [pulled, status] = on.cursor_next(cursor, rows, arena);
        expect_success(status, "pulling every entry through the cursor");
        expect_true(pulled.written == rows, "One pull wide enough must export every entry");
    }

    constexpr std::size_t gathered_rows = 10;
    usearch_key_t gathered_keys[gathered_rows];
    for (std::size_t row = 0; row != gathered_rows; ++row) gathered_keys[row] = key_of(rows + row);
    auto [other, other_status] = us::dense_index_t::init_ram(geometry);
    expect_success(other_status, "creating a second index");
    expect_success(other.on(executor).insert<float>(gathered_keys, {vectors.data(), gathered_rows, dimensions}),
                   "inserting into the second index");
    usearch_dense_index_t const gathered[2] = {index.handle(), other.handle()};
    expect_success(on.reshard({gathered, 2}, {gathered, 1}), "gathering two sources into one");
    expect_true(size_of(index) == rows + gathered_rows, "A gather keeps every entry of its sources");

    // A cut moves one source into several targets, emptying it.
    auto [first_part, first_part_status] = us::dense_index_t::init_ram(geometry);
    auto [second_part, second_part_status] = us::dense_index_t::init_ram(geometry);
    expect_success(first_part_status, "creating the first part");
    expect_success(second_part_status, "creating the second part");
    usearch_dense_index_t const source[1] = {index.handle()};
    usearch_dense_index_t const parts[2] = {first_part.handle(), second_part.handle()};
    expect_success(on.reshard(source, parts, {.seed = 7}), "cutting one source into two targets");
    std::size_t const first_size = size_of(first_part);
    std::size_t const second_size = size_of(second_part);
    expect_true(size_of(index) == 0, "A cut source must end empty");
    expect_true(first_size + second_size == rows + gathered_rows, "A cut keeps every entry of its source");
    expect_true(first_size != 0 && second_size != 0, "A cut must give every target a part");

    std::remove(path);
}

int main(int argc, char** argv) {
    start_test_run(argc, argv);
    run_test_case("test_readme_quickstart", test_readme_quickstart);
    run_test_case("test_arena_devices_and_version", test_arena_devices_and_version);
    run_test_case("test_awaited_calls", test_awaited_calls);
    run_test_case("test_dense_round_trip", test_dense_round_trip);
    auto [executor, executor_status] = us::executor_t::for_ram();
    if (!expect_success(executor_status, "spawning an executor")) return finish_test_run();
    auto [arena, arena_status] = us::arena_t::make({.executor = executor.handle()});
    if (!expect_success(arena_status, "making an arena")) return finish_test_run();
    run_test_case("test_sequences", test_sequences, executor, arena);
    run_test_case("test_shapes", test_shapes, executor, arena);
    run_test_case("test_geospatial", test_geospatial, executor, arena);
    run_test_case("test_plans_and_reshards<sequence_fixture_t>", test_plans_and_reshards<sequence_fixture_t>, executor);
    run_test_case("test_plans_and_reshards<shapes_fixture_t>", test_plans_and_reshards<shapes_fixture_t>, executor);
    run_test_case("test_plans_and_reshards<geospatial_fixture_t>", test_plans_and_reshards<geospatial_fixture_t>,
                  executor);
    return finish_test_run();
}
