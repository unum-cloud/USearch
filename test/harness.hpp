/**
 *  @file test/harness.hpp
 *  @author Ash Vardanian
 *  @date July 17, 2026
 *  @brief Shared helpers for the C++ test suites: the run and its checks, ledgered allocators, and
 *      universally-representable synthetic data generation.
 *
 *  Every setting is an environment variable that a flag of the same name overrides, each spelled
 *  out where @ref read_host_settings and @ref read_session_settings read it, like @c USEARCH_SEED
 *  and @c --seed. A filter that matches no case fails the binary rather than passing an empty
 *  suite, so a mistyped filter is never mistaken for success.
 *
 *  A failed check prints its location and lets the case go on, so one run lists every mismatch it
 *  reaches. It counts into @c test_run, the harness's one sanctioned mutable global, which
 *  @c run_test_case diffs around each case. Where the next step after a failed check would read
 *  past what it checked, the case returns on the check's answer. Checks on worker threads count
 *  the same way as those on the main thread.
 */
#pragma once

#include <csignal> // `std::signal`, `std::raise`, `SIGSEGV`, `SIGABRT`, `SIGTRAP`
#include <cctype>  // `std::tolower` - a setting's flag spelling
#include <cmath>   // `std::fabs`, `std::fmax`, `std::pow`, `std::sqrt`, `std::llround` - error statistics, scaling
#include <cstdio>  // `std::FILE`, `std::fflush`, `std::remove` - the run's lines, a viewed checkpoint's files
#include <cstdlib> // `std::malloc`, `std::free`, `std::abort`, `std::getenv`
#include <cstring> // `std::memset`, `std::memcmp`, `std::strcmp` - a freed block's scribble, the keys, the settings

#include <algorithm>       // `std::sort`, `std::unique`, `std::any_of`, `std::find` - the dispensed IDs, draws, cursors
#include <array>           // `std::array` - one latent point of the sign sketches
#include <atomic>          // `std::atomic` - the run's failed checks
#include <bit>             // `std::bit_cast` - the units in the last place of `error_stats_t`
#include <charconv>        // `std::from_chars` - counts, seeds and device ordinals
#include <chrono>          // `std::chrono::steady_clock` - case timings and the wait limit
#include <filesystem>      // `std::filesystem::temp_directory_path` - the scratch directories
#include <functional>      // `std::invoke` - the cases
#include <limits>          // `std::numeric_limits` - the unbounded allocation budget
#include <map>             // `std::map` - the links each entry holds
#include <optional>        // `std::optional` - a yielded key's minted offset
#include <random>          // `std::random_device` - a drawn seed, a scratch directory's name
#include <regex>           // `std::regex` - the filter
#include <source_location> // `std::source_location` - where a check failed
#include <span>            // `std::span` - the dispensed-ID density check
#include <string>          // `std::string` - a cell's checkpoint stem
#include <string_view>     // `std::string_view` - the stream names
#include <system_error>    // `std::errc`, `std::error_code`
#include <thread>          // `std::this_thread::yield` - bounded spins and the flaky waiter's stalls
#include <type_traits>     // `std::invoke_result_t`, `std::type_identity_t` - case results, check contexts
#include <vector>          // `std::vector` - the gathered predicate items

#include <fmt/format.h>

#include <numkong/tensor.hpp>
#include <numkong/vector.hpp>

#include "usearch/types.h" // `usearch_version_name`, `usearch_capabilities_compiled`, `usearch_capabilities_available`
#include "usearch/utilities/ram_buffers.hpp" // `growing_arena` - a sharded cursor's links
#include "usearch/utilities/types.hpp"

#include "abi.hpp" // `tools::settings_t`, `tools::copied_matches_t` - what every suite shares

#if USEARCH_OS_WINDOWS_
#include <io.h> // `_write`
#else
#include <unistd.h> // `write`, `STDERR_FILENO`
#endif
/*  Only glibc and Darwin ship `backtrace` in libc itself; FreeBSD and musl need another library. */
#if (defined(__GLIBC__) || USEARCH_OS_APPLE_) && __has_include(<execinfo.h>)
#include <execinfo.h> // `backtrace`, `backtrace_symbols_fd`
#endif

namespace util = unum::usearch::utilities;

/** Prints a status by name, as a failed comparison of the most compared enum shows it. */
namespace us = unum::usearch;

template <>
struct fmt::formatter<util::status_t> : fmt::formatter<char const*> {
    auto format(util::status_t status, format_context& context) const {
        return fmt::formatter<char const*>::format(util::status_to_string(status), context);
    }
};

template <>
struct fmt::formatter<util::status_code_t> : fmt::formatter<char const*> {
    auto format(util::status_code_t code, format_context& context) const {
        return fmt::formatter<char const*>::format(util::status_to_string(code), context);
    }
};

/** The short spelling every test reaches the library through where its `using` directive is
 *  ambiguous or unwanted, as inside foreign specializations. */
namespace us = unum::usearch;

namespace unum::usearch::test {

/** Stops in an attached debugger, or dumps core without one; no crash handler catches SIGTRAP. */
inline void trap_into_debugger() noexcept {
#if USEARCH_OS_WINDOWS_
    __debugbreak();
#else
    std::raise(SIGTRAP);
#endif
}

/** Prints the case in progress and a backtrace on a fatal signal; defined beside the run. */
inline void install_test_signal_handlers() noexcept;

/** The names the C interface reports for this build and machine, which users see. */
struct machine_t {
    char const* version = nullptr;
    char const* compiled = nullptr;
    char const* available = nullptr;
};

/** Probes the names @c machine_t reports; each binary defines it, as only library linkers can. */
machine_t probe_machine() noexcept;

/** The names the library reports, for the binaries that link it. */
inline machine_t probe_library_machine() noexcept {
    return {usearch_version_name(), usearch_capabilities_compiled(), usearch_capabilities_available()};
}

/** Prints the version line, then the capabilities as "- Compiled for:" and "- This machine:". */
inline void print(machine_t const& machine, std::FILE* stream = stdout) noexcept {
    fmt::println(stream, "USearch {}", machine.version);
    fmt::println(stream, "- Compiled for: {}", machine.compiled);
    fmt::println(stream, "- This machine: {}", machine.available);
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

/** What every suite reads: the common settings and the run's, then the devices, NVMe paths,
 *  residency, classes and threads the sessions open with. */
struct test_settings_t : common_settings_t {
    double scale = 1.0;
    bool trap_on_failure = false;
    bool refuse_skips = false;

    /** USearch device ordinals the GPU cases run on; none takes the first device of each vendor. */
    st::vector<std::int32_t> devices;

    /** NVMe namespaces a view serves from, in the grammar of @c usearch_nvme_options_t::paths;
     *  none serves each part where its checkpoint points. */
    st::vector<std::string_view> nvme_paths;

    /** Where an index of the tools' cases lives. */
    usearch_residency_t residency = usearch_residency_ram_k;

    /** The newest classes to dispatch as and the NVMe class to use; unknown sets none. */
    usearch_cpu_class_t cpu_class_ceiling = usearch_cpu_class_unknown_k;
    usearch_gpu_class_t gpu_class_ceiling = usearch_gpu_class_unknown_k;
    usearch_nvme_class_t nvme_class = usearch_nvme_class_unknown_k;

    /** Worker threads of a RAM or NVMe executor; zero derives them from the domains or devices. */
    std::size_t threads = 0;
};

/** Reads every @c test_settings_t field; each binary defines it, since only those linking the
 *  library parse the class names, through @ref read_session_settings . */
test_settings_t read_test_settings(int argc, char** argv) noexcept;

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

} // namespace unum::usearch::test

/** Writes a @ref regex_escaped_t straight to the output, a backslash before each metacharacter. */
template <>
struct fmt::formatter<us::test::regex_escaped_t> {
    constexpr auto parse(fmt::format_parse_context& context) { return context.begin(); }
    template <typename context_type_>
    auto format(us::test::regex_escaped_t const& escaped, context_type_& context) const {
        auto output = context.out();
        for (char const character : escaped.label) {
            if (std::string_view {"\\^$.|?*+()[]{}"}.find(character) != std::string_view::npos) *output++ = '\\';
            *output++ = character;
        }
        return output;
    }
};

namespace unum::usearch::test {

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

/** The USearch device ordinals @p text lists, comma-separated, as positions in
 *  @c usearch_gpu_devices ; none when @p text is null. */
inline st::vector<std::int32_t> parse_devices(char const* text) noexcept {
    st::vector<std::int32_t> devices;
    if (!text) return devices;
    for (std::string_view const item : sz::split(std::string_view {text}, ",")) {
        std::uint32_t ordinal = 0;
        auto const [parsed_end, parse_status] = std::from_chars(item.data(), item.data() + item.size(), ordinal);
        if (item.empty() || parse_status != std::errc {} || parsed_end != item.data() + item.size() ||
            ordinal > static_cast<std::uint32_t>((std::numeric_limits<std::int32_t>::max)()))
            exit_unparsed("USEARCH_DEVICES", text, "unsigned integers like 0,1");
        if (st::failed(devices.push_back(static_cast<std::int32_t>(ordinal)))) tools::exit_out_of_memory("the devices");
    }
    return devices;
}

/** The NVMe namespaces @p text lists, comma-separated; none when @p text is null. */
inline st::vector<std::string_view> parse_nvme_paths(char const* text) noexcept {
    st::vector<std::string_view> paths;
    if (!text) return paths;
    for (std::string_view const path : sz::split(std::string_view {text}, ",")) {
        if (path.empty())
            exit_unparsed("USEARCH_NVME_PATHS", text, "comma-separated files, block devices or transport IDs");
        if (st::failed(paths.push_back(std::string_view {path}))) tools::exit_out_of_memory("the NVMe paths");
    }
    return paths;
}

/** The settings a binary reads without the library: the common ones, the run's, the devices, NVMe
 *  paths and threads. */
inline test_settings_t read_host_settings(int argc, char** argv) noexcept {
    test_settings_t settings;
    settings.seed = read_seed_setting(argc, argv, "USEARCH_SEED", "--seed");
    settings.regex_filter = read_regex_filter_setting(argc, argv, "USEARCH_FILTER", "--filter");
    settings.program = argc != 0 ? argv[0] : "";
    settings.scale = read_real_setting(argc, argv, "USEARCH_SCALE", "--scale", 1.0);
    settings.trap_on_failure = read_flag_setting(argc, argv, "USEARCH_TRAP_ON_FAILURE", "--trap-on-failure");
    settings.refuse_skips = read_flag_setting(argc, argv, "USEARCH_NO_SKIPS", "--no-skips");
    settings.devices = parse_devices(setting_text(argc, argv, "USEARCH_DEVICES", "--devices"));
    settings.nvme_paths = parse_nvme_paths(setting_text(argc, argv, "USEARCH_NVME_PATHS", "--nvme-paths"));
    settings.threads = read_count_setting(argc, argv, "USEARCH_THREADS", "--threads", 0);
    return settings;
}

/** Writes the enumerator the library names the setting's text as into @p value , or leaves it. */
template <typename enum_type_>
void read_named_setting(int argc, char** argv, char const* variable, std::string_view flag, enum_type_& value,
                        usearch_status_t (*from_name)(usearch_bytes_view_t, enum_type_*),
                        char const* expected) noexcept {
    char const* const text = setting_text(argc, argv, variable, flag);
    if (text && from_name(tools::bytes_view(text), &value) != usearch_status_success_k)
        exit_unparsed(variable, text, expected);
}

/** Adds to @p settings the residency and the classes, spelled as the library names them. */
inline void read_session_settings(int argc, char** argv, test_settings_t& settings) noexcept {
    read_named_setting(argc, argv, "USEARCH_RESIDENCY", "--residency", settings.residency, usearch_residency_from_name,
                       "ram, vram or nvme");
    read_named_setting(argc, argv, "USEARCH_CPU_CLASS_CEILING", "--cpu-class-ceiling", settings.cpu_class_ceiling,
                       usearch_cpu_class_from_name, "a CPU class like intel-icelake or serial");
    read_named_setting(argc, argv, "USEARCH_GPU_CLASS_CEILING", "--gpu-class-ceiling", settings.gpu_class_ceiling,
                       usearch_gpu_class_from_name, "a GPU class like nvidia-ampere");
    read_named_setting(argc, argv, "USEARCH_NVME_CLASS", "--nvme-class", settings.nvme_class,
                       usearch_nvme_class_from_name, "an NVMe class like posix, linux-uring or linux-spdk");
}

/** The settings a session of the run opens with, in the tools' shape. */
inline tools::settings_t session_settings() {
    test_settings_t const& run = test_run.settings;
    tools::settings_t settings;
    // Keys are minted under 32 bits of the seed, as the tools mint them.
    settings.seed = static_cast<std::uint32_t>(run.seed);
    settings.residency.selected = tools::selected_of(tools::residencies_k, run.residency);
    settings.cpu_class_ceiling.selected = tools::selected_of(tools::cpu_class_ceilings_k, run.cpu_class_ceiling);
    settings.gpu_class_ceiling.selected = tools::selected_of(tools::gpu_class_ceilings_k, run.gpu_class_ceiling);
    settings.nvme_class.selected = tools::selected_of(tools::nvme_classes_k, run.nvme_class);
    settings.threads.value = run.threads;
    if (st::failed(settings.devices.assign(run.devices)) || st::failed(settings.nvme_paths.assign(run.nvme_paths)))
        tools::exit_out_of_memory("the settings");
    return settings;
}

/** The exact spelling of a residency's constant, for the labels of the cases that sweep it. */
constexpr char const* constant_of(usearch_residency_t value) noexcept {
    switch (value) {
    case usearch_residency_unknown_k: return "usearch_residency_unknown_k";
    case usearch_residency_ram_k: return "usearch_residency_ram_k";
    case usearch_residency_vram_k: return "usearch_residency_vram_k";
    case usearch_residency_nvme_k: return "usearch_residency_nvme_k";
    default: return "usearch_residency_t";
    }
}

/** The exact spelling of a scalar's constant, for the labels of the cases that sweep it. */
constexpr char const* constant_of(usearch_scalar_t value) noexcept {
    switch (value) {
    case usearch_scalar_unknown_k: return "usearch_scalar_unknown_k";
    case usearch_scalar_f64_k: return "usearch_scalar_f64_k";
    case usearch_scalar_f32_k: return "usearch_scalar_f32_k";
    case usearch_scalar_bf16_k: return "usearch_scalar_bf16_k";
    case usearch_scalar_f16_k: return "usearch_scalar_f16_k";
    case usearch_scalar_e5m2_k: return "usearch_scalar_e5m2_k";
    case usearch_scalar_e4m3_k: return "usearch_scalar_e4m3_k";
    case usearch_scalar_e3m2_k: return "usearch_scalar_e3m2_k";
    case usearch_scalar_e2m3_k: return "usearch_scalar_e2m3_k";
    case usearch_scalar_i8_k: return "usearch_scalar_i8_k";
    case usearch_scalar_i4x2_k: return "usearch_scalar_i4x2_k";
    case usearch_scalar_i32_k: return "usearch_scalar_i32_k";
    case usearch_scalar_u8_k: return "usearch_scalar_u8_k";
    case usearch_scalar_u4x2_k: return "usearch_scalar_u4x2_k";
    case usearch_scalar_u1x8_k: return "usearch_scalar_u1x8_k";
    default: return "usearch_scalar_t";
    }
}

/** The exact spelling of a CPU class's constant, for the labels of the cases that sweep it. */
constexpr char const* constant_of(usearch_cpu_class_t value) noexcept {
    switch (value) {
    case usearch_cpu_class_unknown_k: return "usearch_cpu_class_unknown_k";
    case usearch_cpu_class_serial_k: return "usearch_cpu_class_serial_k";
    case usearch_cpu_class_monoversioned_k: return "usearch_cpu_class_monoversioned_k";
    case usearch_cpu_class_intel_haswell_k: return "usearch_cpu_class_intel_haswell_k";
    case usearch_cpu_class_intel_sierra_k: return "usearch_cpu_class_intel_sierra_k";
    case usearch_cpu_class_intel_skylake_k: return "usearch_cpu_class_intel_skylake_k";
    case usearch_cpu_class_intel_icelake_k: return "usearch_cpu_class_intel_icelake_k";
    case usearch_cpu_class_amd_genoa_k: return "usearch_cpu_class_amd_genoa_k";
    case usearch_cpu_class_intel_sapphire_k: return "usearch_cpu_class_intel_sapphire_k";
    case usearch_cpu_class_intel_granite_k: return "usearch_cpu_class_intel_granite_k";
    case usearch_cpu_class_intel_diamond_k: return "usearch_cpu_class_intel_diamond_k";
    case usearch_cpu_class_aws_g2_k: return "usearch_cpu_class_aws_g2_k";
    case usearch_cpu_class_apple_m2_k: return "usearch_cpu_class_apple_m2_k";
    case usearch_cpu_class_apple_m4_k: return "usearch_cpu_class_apple_m4_k";
    case usearch_cpu_class_aws_g5_k: return "usearch_cpu_class_aws_g5_k";
    case usearch_cpu_class_nvidia_vera_k: return "usearch_cpu_class_nvidia_vera_k";
    case usearch_cpu_class_alibaba_c930_k: return "usearch_cpu_class_alibaba_c930_k";
    case usearch_cpu_class_ibm_power10_k: return "usearch_cpu_class_ibm_power10_k";
    case usearch_cpu_class_wasm32_v128_k: return "usearch_cpu_class_wasm32_v128_k";
    case usearch_cpu_class_wasm64_extended_k: return "usearch_cpu_class_wasm64_extended_k";
    default: return "usearch_cpu_class_t";
    }
}

/**
 *  @brief Names the signal and the case in progress, prints a backtrace, then re-raises the signal,
 *      so a crash localizes itself rather than dying silently under CI's output redirection.
 *
 *  Writes through a raw @c write, since the crashing thread may already hold the stdio lock that
 *  @c std::fprintf takes.
 */
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
    // The `_fd` form writes without allocating, which is what makes it usable from a handler.
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

/** Prints both sides of a failed equality under its location line, when both can be named. */
template <typename actual_type_, typename wanted_type_>
void print_unequal_(actual_type_ const& actual, wanted_type_ const& wanted) noexcept {
    if constexpr (fmt::is_formattable<actual_type_>::value && fmt::is_formattable<wanted_type_>::value)
        fmt::println(stderr, "    expected {}, got {}", wanted, actual);
}

/** Holds when @p actual equals @p wanted ; otherwise counts a failed check at the caller's line
 *  and prints both. */
template <typename actual_type_, typename wanted_type_>
bool expect_equal(actual_type_ const& actual, wanted_type_ const& wanted,
                  std::source_location where = std::source_location::current()) noexcept {
    if (actual == wanted) return true;
    record_failed_check(where);
    print_unequal_(actual, wanted);
    return false;
}

/** The same, printing @p context formatted with @p args beneath both sides on failure. */
template <typename actual_type_, typename wanted_type_, typename... args_types_>
bool expect_equal(actual_type_ const& actual, wanted_type_ const& wanted,
                  located_format<std::type_identity_t<args_types_>...> context, args_types_ const&... args) noexcept {
    if (actual == wanted) return true;
    record_failed_check(context.location);
    print_unequal_(actual, wanted);
    print_context_<args_types_...>(context, args...);
    return false;
}

/** Holds when @p outcome answered @p wanted ; otherwise counts a failed check and prints both
 *  statuses by name and the outcome's message. */
inline bool expect_status(usearch_outcome_t const& outcome, usearch_status_t wanted,
                          std::source_location where = std::source_location::current()) noexcept {
    if (outcome.status == wanted) return true;
    record_failed_check(where);
    fmt::println(stderr, "    expected {}, got {}: {}", usearch_status_name(wanted),
                 usearch_status_name(outcome.status), outcome.message ? outcome.message : "no message");
    return false;
}

/** @ref expect_status of a success. */
inline bool expect_success(usearch_outcome_t const& outcome,
                           std::source_location where = std::source_location::current()) noexcept {
    return expect_status(outcome, usearch_status_success_k, where);
}

/** Why a GPU case skips on a machine with no device of its vendor. */
inline constexpr char no_gpu_device_k[] = "no GPU device";

/** Why an AMD case skips on a device whose wavefronts are not the 64 lanes the CDNA3 kits run. */
inline constexpr char needs_64_lane_waves_k[] = "needs gpu_device_t::warp_lanes of 64";

/** Why an AMD case skips in a build that keeps no CDNA3 class. */
inline constexpr char built_without_cdna3_k[] = "built without USEARCH_TARGET_AMD_CDNA3";

/** Whether every check since it was made held, so a scenario stops before a broken step. */
struct checks_held_t {
    std::size_t failed_before = test_run.failed_checks.load();

    bool operator()() const noexcept { return test_run.failed_checks.load() == failed_before; }
};

/** A directory of its own under @c TMPDIR , <tt>usearch-CASE-XXXXXX</tt>, removed with its
 *  contents when dropped. */
struct scratch_directory_t {
    std::filesystem::path root;

    scratch_directory_t() noexcept {
        std::error_code error;
        std::filesystem::path const parent = std::filesystem::temp_directory_path(error);
        std::string label = test_run.current_case ? test_run.current_case : "run";
        for (char& character : label)
            if (character == '.') character = '-';
        std::random_device device;
        do root = parent / fmt::format("usearch-{}-{:06x}", label, device() & 0xFFFFFFu);
        while (!std::filesystem::create_directory(root, error) && !error);
    }
    scratch_directory_t(scratch_directory_t const&) = delete;
    scratch_directory_t& operator=(scratch_directory_t const&) = delete;
    ~scratch_directory_t() noexcept {
        std::error_code error;
        std::filesystem::remove_all(root, error);
    }

    /** The file @p name inside. */
    std::string path(std::string_view name) const { return (root / name).string(); }
};

/** The relative error a @c -ffast-math build leaves, which @c expect_errors allows. */
inline constexpr util::f64_t approximate_tolerance_k = 1e-3;

/** Recall an index may lose against its reference - a rewritten, refined, re-sharded or stopped
 *  graph against its source, a VRAM index against its RAM twin - as a share of the results, one
 *  bound for every modality and residency. */
inline constexpr util::f64_t recall_margin_k = 0.05;

/** Queries every recall is measured over, enough that @ref recall_margin_k spans three standard
 *  deviations of a 95% recall: 3 * sqrt(0.95 * 0.05 / 192) is about 0.047. A test holding fewer
 *  rows than this samples all of them. */
inline constexpr std::size_t recall_samples_k = 192;

/** Matches every recall at ten judges per query, the depth the margin's share is taken over. */
inline constexpr std::size_t recall_wanted_k = 10;

/** How a cell's results relate to their exact reference, the families NumKong's harness judges its
 *  kernels by: equal, within rounding, within rounding of the largest reference for weighted
 *  averages that pass through zero, or only statistically close for paths approximate by design. */
enum class comparison_family_t { exact_k, approximate_k, normalized_reduction_k, bounded_k };

/** Whether a cell beyond its family's bound fails the suite or only warns: sparse paths warn, exact
 *  paths assert. */
enum class comparison_verdict_t { assert_k, warn_k };

/** Errors of one cell against its exact reference: max absolute and relative errors, the absolute
 *  errors' mean and Welford sum of squares that sigma bounds read, and the largest distance in
 *  @c f32 units in the last place. */
struct error_stats_t {
    std::size_t count = 0;
    util::f64_t max_absolute_error = 0;
    util::f64_t max_relative_error = 0;
    util::f64_t max_reference = 0;
    util::f64_t mean = 0;
    util::f64_t sum_of_squares = 0;
    util::u64_t max_ulp = 0;
    std::size_t exact_matches = 0;

    void add(util::f64_t actual, util::f64_t reference) noexcept {
        util::f64_t const difference = actual == reference ? 0.0 : std::fabs(actual - reference);
        // NaN counts as the largest error, which every family's bound refuses.
        util::f64_t const error = std::isnan(difference) ? std::numeric_limits<util::f64_t>::infinity() : difference;
        ++count;
        exact_matches += error == 0;
        max_absolute_error = std::fmax(max_absolute_error, error);
        max_relative_error = std::fmax(max_relative_error,
                                       error / std::fmax(std::fmax(std::fabs(actual), std::fabs(reference)), 1.0));
        max_reference = std::fmax(max_reference, std::fabs(reference));
        util::f64_t const delta = error - mean;
        mean += delta / static_cast<util::f64_t>(count);
        sum_of_squares += delta * (error - mean);
        max_ulp = util::max_of(max_ulp,
                               ulp_distance(static_cast<util::f32_t>(actual), static_cast<util::f32_t>(reference)));
    }

    util::f64_t deviation() const noexcept {
        return count > 1 ? std::sqrt(sum_of_squares / static_cast<util::f64_t>(count - 1)) : 0.0;
    }

    static util::u64_t ulp_distance(util::f32_t first, util::f32_t second) noexcept {
        auto const ordered = [](util::f32_t value) noexcept {
            util::u32_t const bits = std::bit_cast<util::u32_t>(value);
            return bits >> 31 ? -static_cast<util::i64_t>(bits & 0x7FFFFFFFu) : static_cast<util::i64_t>(bits);
        };
        util::i64_t const gap = ordered(first) - ordered(second);
        return static_cast<util::u64_t>(gap < 0 ? -gap : gap);
    }
};

/** Judges the statistics of @p cell by its @p family's bound and, when @p sigmas is positive, by no
 *  error past rounding lying beyond @p sigmas standard deviations past the mean; on @c warn_k
 *  prints a warning instead of failing. A cell of no samples judged nothing and always fails. */
inline bool expect_errors(char const* cell, comparison_family_t family, comparison_verdict_t verdict,
                          error_stats_t const& errors, double sigmas,
                          std::source_location where = std::source_location::current()) noexcept {
    located_format<char const*> no_samples {"{}: no samples were compared"};
    no_samples.location = where; // a consteval constructor cannot take the caller's runtime location
    if (!expect_true(errors.count != 0, no_samples, cell)) return false;
    bool within = std::isfinite(errors.max_absolute_error);
    switch (family) {
    case comparison_family_t::exact_k: within = within && errors.exact_matches == errors.count; break;
    case comparison_family_t::approximate_k:
        within = within && errors.max_relative_error <= approximate_tolerance_k;
        break;
    case comparison_family_t::normalized_reduction_k:
        within = within && errors.max_absolute_error <= approximate_tolerance_k * std::fmax(errors.max_reference, 1.0);
        break;
    case comparison_family_t::bounded_k: break;
    }
    // Rounding-level outliers say nothing about a path, so the sigma bound reads only the errors
    // beyond rounding level.
    if (sigmas > 0 && errors.max_relative_error > approximate_tolerance_k)
        within = within && errors.max_absolute_error <= errors.mean + sigmas * errors.deviation();
    if (within) return true;
    if (verdict == comparison_verdict_t::assert_k) record_failed_check(where);
    fmt::println(stderr,
                 "    {}{}: {} of {} exact, max abs {:.4g}, max rel {:.4g}, max ulp {}, mean abs {:.4g} ± {:.4g}",
                 verdict == comparison_verdict_t::warn_k ? "warning in " : "", cell, errors.exact_matches, errors.count,
                 errors.max_absolute_error, errors.max_relative_error, errors.max_ulp, errors.mean, errors.deviation());
    return verdict == comparison_verdict_t::warn_k;
}

/** A per-item test filter in the quad shape the engines ask, each item answered on its own. */
template <typename per_item_type_>
auto quad_predicate(per_item_type_ per_item) noexcept {
    return [per_item]<typename item_type_, std::size_t extent_>(
               std::span<item_type_ const, extent_> items, std::span<util::admission_t, extent_> admissions) noexcept {
        for (std::size_t lane = 0; lane != items.size(); ++lane) admissions[lane] = per_item(items[lane]);
    };
}

/** A per-item test filter in the aggregating shape: it gathers every item an engine pushes beside
 *  its payload and answers them all at the drain. */
template <typename per_item_type_, typename item_type_, typename payload_type_>
struct gathering_predicate {
    per_item_type_ per_item;
    std::vector<item_type_> items;
    std::vector<payload_type_> payloads;

    util::status_t reserve(std::size_t capacity) noexcept {
        items.reserve(capacity);
        payloads.reserve(capacity);
        return util::status_code_t::success_k;
    }

    template <std::size_t extent_>
    void push(std::span<item_type_ const, extent_> asked, std::span<payload_type_ const, extent_> held) noexcept {
        items.insert(items.end(), asked.begin(), asked.end());
        payloads.insert(payloads.end(), held.begin(), held.end());
    }

    template <typename on_admitted_type_>
    void drain(on_admitted_type_&& on_admitted) noexcept {
        for (std::size_t position = 0; position != items.size(); ++position)
            if (per_item(items[position]) == util::admission_t::admit_k) on_admitted(payloads[position]);
        items.clear();
        payloads.clear();
    }
};

/** One @ref gathering_predicate per executor thread, the family an engine driving its own threads
 *  hands each thread its own of. */
template <typename per_item_type_, typename item_type_, typename payload_type_>
struct gathering_predicates {
    using thread_predicate_t = gathering_predicate<per_item_type_, item_type_, payload_type_>;
    std::vector<thread_predicate_t> threads;

    thread_predicate_t& of_thread(std::size_t thread) noexcept { return threads[thread]; }
};

/** A @ref gathering_predicates of @p threads, each gathering @p item_type_ beside @p payload_type_
 *  and admitting what @p per_item admits. */
template <typename item_type_, typename payload_type_, typename per_item_type_>
gathering_predicates<per_item_type_, item_type_, payload_type_> gathering_per_thread(per_item_type_ per_item,
                                                                                     std::size_t threads) {
    return {std::vector(threads, gathering_predicate<per_item_type_, item_type_, payload_type_> {per_item, {}, {}})};
}

/** One unit's cases, as it registers them with the runner: a GPU vendor's, run once per selected
 *  device of that vendor, or a host unit's under @c gpu_vendor_t::unknown_k , run once with no
 *  device, so the runner names neither the vendors nor the dtypes a build compiled. */
struct registered_suite_t {
    util::gpu_vendor_t vendor = util::gpu_vendor_t::unknown_k;
    void (*run_cases)(util::gpu_device_t const& device) = nullptr;
};

/** The exact spelling of @p vendor 's constant, for the labels of the cases that run on it. */
constexpr char const* constant_of(util::gpu_vendor_t vendor) noexcept {
    switch (vendor) {
    case util::gpu_vendor_t::nvidia_k: return "gpu_vendor_t::nvidia_k";
    case util::gpu_vendor_t::amd_k: return "gpu_vendor_t::amd_k";
    case util::gpu_vendor_t::apple_k: return "gpu_vendor_t::apple_k";
    default: return "gpu_vendor_t::unknown_k";
    }
}

/** Every vendor's driver table, shared with the tools' read-backs. */
using us::tools::gpu_drivers;
using us::tools::spawn_harness_executor;

/** Every device the C interface numbers, in its order, which @c USEARCH_DEVICES indexes. */
std::span<util::gpu_device_t const> gpu_devices() noexcept;

/** The class the library lists for @p device, the one its kernels run, matched by vendor and
 *  ordinal. */
inline usearch_gpu_class_t listed_class(util::gpu_device_t const& device) noexcept {
    for (usearch_gpu_device_t const& entry : tools::listed_devices())
        if (entry.vendor == static_cast<usearch_gpu_vendor_t>(device.vendor) &&
            entry.vendor_ordinal == device.vendor_ordinal)
            return entry.gpu_class;
    return usearch_gpu_class_unknown_k;
}

/** Registers @p added when it names cases, and answers every suite registered so far. */
inline std::span<registered_suite_t const> registered_suites(registered_suite_t added = {}) noexcept {
    static registered_suite_t suites[16];
    static std::size_t count = 0;
    if (added.run_cases && count != std::size(suites)) suites[count++] = added;
    return {suites, count};
}

/** IDs dispensed to racing lanes must come out dense & duplicate-free: sorted in
 *  place, unique, and running from @p first_id through the half-open bound
 *  @p first_id plus the ID count. */
template <typename node_id_type_>
inline void expect_contiguous_ids(std::span<node_id_type_> ids, std::size_t first_id) {
    std::sort(ids.begin(), ids.end());
    expect_true(std::unique(ids.begin(), ids.end()) == ids.end());
    expect_true(!ids.empty() && static_cast<std::size_t>(ids.front()) == first_id &&
                static_cast<std::size_t>(ids.back()) == first_id + ids.size() - 1);
}

/** The reader-lane invariants over one search's scored results: every ID below the watermark
 *  sampled after the search, distances ascending, the candidate breadth bound honored. Answers the
 *  violation count silently - racing threads add it to their shared counter, since printing from
 *  inside a race would interleave & flood - and the status verdict stays at the call site: threads
 *  differ in how they record it. */
template <typename scored_type_>
inline std::size_t scored_violations(scored_type_ const* results, std::size_t results_count,
                                     std::size_t published_after, std::size_t results_bound) noexcept {
    std::size_t violations = results_count > results_bound;
    for (std::size_t position = 0; position != results_count; ++position) {
        violations += static_cast<std::size_t>(results[position].node_id) >= published_after;
        if (position) violations += results[position].distance < results[position - 1].distance;
    }
    return violations;
}

/** Shared counts of live bytes & allocations, mutated by every @c counting_allocator. */
struct allocation_ledger_t {

    /** Lifetime numbers of granted @c allocate and of @c deallocate calls, for coarse
     *  allocation-pressure checks. */
    std::size_t allocations = 0;
    std::size_t deallocations = 0;

    /** Bytes allocated minus bytes freed; must balance to zero at scope exit. */
    std::ptrdiff_t live_bytes = 0;

    /** Requests refused past @ref byte_budget . */
    std::size_t refusals = 0;

    /** Allocation ceiling in live bytes; scenarios lower it to drive the all-or-
     *  nothing refusal paths, so a request past it is refused, never counted. */
    std::ptrdiff_t byte_budget = (std::numeric_limits<std::ptrdiff_t>::max)();
};

/**
 *  @brief Byte-counting allocator proving propagation through a container: every allocation and
 *      deallocation is tallied in the shared ledger, which must balance to zero at scope exit.
 *
 *  The default constructor exists only because @c nk::tensor requires a default-constructible
 *  allocator for its empty instances; a defaulted copy aborts on any real allocation, so
 *  every allocation on an audited matrix provably flows through ledgered instances. The audit
 *  covers the matrices instantiated with this allocator - the CPU cells ledger their vectors
 *  & neighbors matrices, while per-context scratch stays on the plain heap.
 */
template <typename value_type_>
struct counting_allocator {
    using value_type = value_type_;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_copy_assignment = std::true_type;
    using propagate_on_container_swap = std::true_type;
    allocation_ledger_t* ledger_ = nullptr;

    counting_allocator() noexcept = default;
    explicit counting_allocator(allocation_ledger_t& ledger) noexcept : ledger_(&ledger) {}

    template <typename other_type_>
    counting_allocator(counting_allocator<other_type_> const& other) noexcept : ledger_(other.ledger_) {}

    value_type_* allocate(std::size_t count) noexcept {
        if (!ledger_) std::abort(); // A default-constructed instance must never allocate.
        std::ptrdiff_t const wanted = static_cast<std::ptrdiff_t>(count * sizeof(value_type_));
        fu::standard_atomic_ref<std::ptrdiff_t> const live_bytes {ledger_->live_bytes};
        if (live_bytes.load() + wanted > fu::standard_atomic_ref<std::ptrdiff_t>(ledger_->byte_budget).load())
            return fu::standard_atomic_ref<std::size_t>(ledger_->refusals).store_add(1), nullptr;
        live_bytes.store_add(wanted);
        fu::standard_atomic_ref<std::size_t>(ledger_->allocations).store_add(1);
        return static_cast<value_type_*>(std::malloc(count * sizeof(value_type_)));
    }
    void deallocate(value_type_* pointer, std::size_t count) noexcept {
        if (ledger_) {
            fu::standard_atomic_ref<std::ptrdiff_t>(ledger_->live_bytes)
                .store_sub(static_cast<std::ptrdiff_t>(count * sizeof(value_type_)));
            fu::standard_atomic_ref<std::size_t>(ledger_->deallocations).store_add(1);
        }
        // Scribbled first, so a read after free sees garbage in every build rather than stale bytes.
        if (pointer) std::memset(static_cast<void*>(pointer), 0xA5, count * sizeof(value_type_));
        std::free(pointer);
    }
    bool operator==(counting_allocator const& other) const noexcept { return ledger_ == other.ledger_; }
};

/** The byte specialization every fixture places its index through. */
using counting_allocator_t = counting_allocator<util::byte_t>;

/**
 *  @brief Distinct levels one component of @p scalar_ draws from: a packed bit has two, since
 *      its proxy sets the bit for any nonzero and a wider draw would bias it toward ones; every
 *      other type keeps the eight levels the fixtures are built around.
 *  @sa @c nk::fill_uniform, which ranges every dtype by its own @c finite_min & @c finite_max.
 */
template <typename scalar_>
constexpr util::u32_t random_component_levels() noexcept {
    if constexpr (std::is_same_v<scalar_, nk::u1x8_t>) return 2;
    else return 8;
}

/** Builds a scalar from a small unsigned - sub-byte types go through the raw nibble path,
 *  float wrappers via @c from_f32, byte-or-larger integers via the integral constructor. */
template <typename scalar_>
auto make_scalar(unsigned value) noexcept {
    if constexpr (nk::dimensions_per_value<scalar_>() > 1) return static_cast<util::i8_t>(value);
    else if constexpr (requires { scalar_::from_f32(0.0f); }) return scalar_::from_f32(static_cast<float>(value));
    else return scalar_(value);
}

/** Waiter that misbehaves on purpose: mostly yields once, but one wait in 64 yields thousands of
 *  times, stretching the lock & watermark contention windows far beyond anything hardware timing
 *  produces - a poor man's schedule fuzzer. The draw hashes the watched word, its value and the
 *  thread, so a stalled wait stays stalled on every retry until another thread moves the word. */
struct flaky_waiter_t {
    template <typename value_type_, typename thread_index_type_, typename bound_type_ = util::wait_capped_t>
    void operator()(value_type_ const* watched, value_type_ observed, thread_index_type_ thread,
                    bound_type_ = {}) const noexcept {
        split_mix_generator_t draw {reinterpret_cast<util::u64_t>(watched) ^ static_cast<util::u64_t>(observed) ^
                                    (static_cast<util::u64_t>(thread) << 48)};
        std::size_t const yields = util::uniform_below<util::u32_t> {64}(draw) == 0 ? 4096 : 1;
        for (std::size_t yielded = 0; yielded != yields; ++yielded) std::this_thread::yield();
    }
};

/** A uniform real in @b [lower,upper) - the real-valued sibling of @ref uniform_below, identical
 *  on every standard library, where @c std::uniform_real_distribution is implementation-defined
 *  and would draw different test data on each. */
template <typename real_type_>
struct uniform_between;

template <>
struct uniform_between<util::f64_t> {
    util::f64_t lower = 0, upper = 1;

    /** One draw from the top 53 random bits, the double's whole mantissa. */
    template <typename generator_type_>
    constexpr util::f64_t operator()(generator_type_& generator) const noexcept {
        return lower + static_cast<util::f64_t>(generator() >> 11) * 0x1p-53 * (upper - lower);
    }
};

/** Factory function to allocate zeroed vectors, aborting on exhaustion. */
template <typename type_>
[[nodiscard]] nk::vector<type_> make_vector(std::size_t count) {
    auto [made, status] = nk::vector<type_>::zeros(count);
    if (nk::failed(status)) std::abort();
    return std::move(made);
}

/** Factory function to allocate zeroed row-major matrices, aborting on exhaustion. */
template <typename type_>
[[nodiscard]] nk::matrix<type_> make_matrix(std::size_t rows, std::size_t columns) {
    auto [made, status] = nk::matrix<type_>::zeros({rows, columns});
    if (nk::failed(status)) std::abort();
    return std::move(made);
}

/** Builds binary vectors as sign sketches of points in a small latent space: bit @c b of a row
 *  is the side of the @c b-th random hyperplane its point falls on, the way binary embeddings
 *  arise. Hamming distance then tracks the angle between points, so the landscape stays graded,
 *  where independent bits - or clusters of independently flipped ones - leave every pair
 *  near-equidistant and no walk can navigate. Eight latent dimensions keep neighboring rows
 *  distinct at the scenarios' row counts. */
template <typename scalar_>
nk::matrix<scalar_> make_sketched_vectors(std::size_t count, std::size_t dimensions, util::u64_t seed) {
    constexpr std::size_t latent_dimensions_k = 8;
    split_mix_generator_t points_generator {seed};
    uniform_between<util::f64_t> const centered {-1, 1};
    auto [made, status] = nk::matrix<scalar_>::zeros({count, dimensions});
    if (nk::failed(status)) std::abort();
    std::array<util::f64_t, latent_dimensions_k> point;
    // Filled before the move, as in `make_vectors`, past the same MSVC 14.51 `/O2` miscompile.
    for (auto row : made.rows()) {
        for (util::f64_t& coordinate : point) coordinate = centered(points_generator);
        // One set of hyperplanes for every call, as one embedding model serves every batch: slices
        // drawn under different seeds, like the concurrent builds', still fall in one sketch space.
        // Redrawn per row in the same order, so nothing holds them.
        split_mix_generator_t hyperplanes_generator {0x5E7C4ull};
        auto bits = row.as_vector();
        for (std::size_t dimension = 0; dimension != dimensions; ++dimension) {
            util::f64_t side = 0;
            for (std::size_t latent = 0; latent != latent_dimensions_k; ++latent)
                side += centered(hyperplanes_generator) * point[latent];
            bits[dimension] = make_scalar<scalar_>(side > 0);
        }
    }
    return std::move(made);
}

/** Fills a row-major matrix with pseudo-random small values from {0..7}, exactly
 *  representable in every supported scalar type down to i4 nibbles. Binary vectors
 *  route through the sign-sketch generator: their flat draw has no navigable structure. */
template <typename scalar_>
nk::matrix<scalar_> make_vectors(std::size_t count, std::size_t dimensions, util::u64_t seed) {
    if constexpr (std::is_same_v<scalar_, nk::u1x8_t>) return make_sketched_vectors<scalar_>(count, dimensions, seed);
    auto [made, status] = nk::matrix<scalar_>::zeros({count, dimensions});
    if (nk::failed(status)) std::abort();
    split_mix_generator_t random_generator {seed};
    util::uniform_below<util::u32_t> component_distribution {random_component_levels<scalar_>()};
    // Filled before the move: MSVC 14.51 `/O2` reads the moved rank from `rcx`, not `xmm1`.
    for (auto row : made.rows()) {
        auto values = row.as_vector();
        for (std::size_t dimension = 0; dimension != dimensions; ++dimension)
            values[dimension] = make_scalar<scalar_>(component_distribution(random_generator));
    }
    return std::move(made);
}

/** Fills a row-major matrix with rows drawn around the origin, each scaled by a length of its own
 *  drawn log-uniformly across a hundredfold, so an inner product favors the long rows far past
 *  what @ref make_vectors draws. Only floats of sixteen bits or more hold that spread: an integer
 *  or a byte-wide float folds the short rows into one. */
template <typename scalar_>
    requires(sizeof(scalar_) >= 2 && requires { scalar_::from_f32(0.0f); })
nk::matrix<scalar_> make_spread_vectors(std::size_t count, std::size_t dimensions, util::u64_t seed) {
    auto [made, status] = nk::matrix<scalar_>::zeros({count, dimensions});
    if (nk::failed(status)) std::abort();
    split_mix_generator_t random_generator {seed};
    uniform_between<util::f64_t> const centered {-1, 1}, decades {0, 2};
    // Filled before the move, as in `make_vectors`, past the same MSVC 14.51 `/O2` miscompile.
    for (auto row : made.rows()) {
        util::f64_t const length = std::pow(10.0, decades(random_generator));
        auto values = row.as_vector();
        for (std::size_t dimension = 0; dimension != dimensions; ++dimension)
            values[dimension] = scalar_::from_f32(static_cast<float>(length * centered(random_generator)));
    }
    return std::move(made);
}

/** A cursor over the sharded @p indexes, pulled @p chunk keys at a time, yields each of its
 *  @p entries once as stored, and from @p start breadth-first that key first and each reached entry
 *  once, every link a key @p indexes holds once and each entry's run closing at ordered offsets. */
template <typename indexes_type_>
void expect_sharded_cursor(indexes_type_& indexes, typename indexes_type_::key_t const& start, std::size_t entries,
                           std::size_t chunk) {
    using key_t = typename indexes_type_::key_t;
    using cursor_t = typename indexes_type_::cursor_t;
    for (std::span<key_t const> const starts : {std::span<key_t const> {}, std::span<key_t const> {&start, 1}}) {
        util::growing_arena<util::forkunion_allocator_t> arena {util::forkunion_allocator_t {util::no_memory_domain_k}};
        typename cursor_t::links_allocator_t const links_allocator {arena};
        typename cursor_t::links_keys_t links {links_allocator};
        typename cursor_t::links_offsets_t offsets {links_allocator};
        cursor_t cursor {util::forkunion_allocator_t {util::no_memory_domain_k}};
        expect_equal(cursor.open(indexes, starts), util::status_code_t::success_k);
        nk::vector<key_t> pulled_keys = make_vector<key_t>(chunk);
        std::vector<key_t> pulled;
        while (true) {
            auto const [read, status] = cursor.pull(indexes, util::to_writable_span(pulled_keys), &links, &offsets);
            expect_equal(status, util::status_code_t::success_k);
            if (!read || failed(status)) break;
            pulled.insert(pulled.end(), pulled_keys.data(), pulled_keys.data() + read);
        }
        std::size_t repeated = 0, foreign = 0;
        for (auto yielded = pulled.begin(); yielded != pulled.end(); ++yielded)
            repeated += std::find(pulled.begin(), yielded, *yielded) != yielded;
        for (key_t const& linked : links) foreign += indexes.count(linked) != 1;
        expect_equal(repeated, 0u);
        expect_equal(foreign, 0u);
        expect_equal(offsets.size(), pulled.size());
        expect_true(std::is_sorted(offsets.begin(), offsets.end()) &&
                    (offsets.empty() || offsets[offsets.size() - 1] == links.size()));
        starts.empty() ? expect_equal(pulled.size(), entries)
                       : expect_true(pulled.size() > 1 && pulled.size() <= entries && pulled[0] == start);
    }
}

using us::tools::copied_matches_t;
using us::tools::copy_matches;

namespace dense {
void run_ram_metric_cases();
void run_nvme_indexes_cases();
} // namespace dense

namespace utilities {
void run_ram_buffers_cases();
void run_ram_tables_cases();
void run_safegraphs_format_cases();
} // namespace utilities

namespace basic {
void run_sentences_cases();
void run_bioinformatics_cases();
void run_meshes_cases();
} // namespace basic

namespace sequence {
void run_sequence_cases();

/** The sequence VRAM cases on @p device , of the vendor of @p executor_type_ , which each
 *  vendor's units specialize. */
template <typename executor_type_>
void run_sequence_vram_cases(util::gpu_device_t const& device);
} // namespace sequence

namespace shapes {
void run_shapes_cases();

/** The shapes VRAM cases on @p device , of the vendor of @p executor_type_ , which each vendor's
 *  units specialize. */
template <typename executor_type_>
void run_shapes_vram_cases(util::gpu_device_t const& device);
} // namespace shapes

namespace geospatial {
void run_geospatial_cases();

/** The geospatial VRAM cases on @p device , of the vendor of @p executor_type_ , which each
 *  vendor's units specialize. */
template <typename executor_type_>
void run_geospatial_vram_cases(util::gpu_device_t const& device);
} // namespace geospatial

/** The suites over the C interface, each sweeping its modality's diagonal, and the checks every
 *  modality's cells share. */
namespace abi {
using namespace unum::usearch::tools;

void run_dense_abi_cases();
void run_sequence_abi_cases();
void run_shapes_abi_cases();
void run_geospatial_abi_cases();

/** Entries or queries each call of a cell takes, so a build spans several calls. */
inline constexpr std::size_t cell_batch_k = 384;

/** Matches a cell judges per query. */
inline constexpr std::size_t cell_wanted_k = 10;

/** Whether @p first and @p second are the same key, byte for byte. */
inline bool same_key(usearch_key_t const& first, usearch_key_t const& second) noexcept {
    return !std::memcmp(first.bytes, second.bytes, sizeof(first.bytes));
}

/** The mean distance between the scores of @p first and @p second, rank by rank, zero when neither
 *  holds a match. */
inline double mean_score_deviation(copied_matches_t const& first, copied_matches_t const& second) {
    double moved = 0;
    for (std::size_t match = 0; match != first.scores.size(); ++match)
        moved += std::fabs(first.scores[match] - second.scores[match]);
    return first.scores.empty() ? 0 : moved / static_cast<double>(first.scores.size());
}

/** Whether @p first and @p second land as many matches per query, at the same distances rank by
 *  rank: on average within @p score_band of each other, so an iterative solver's rarest move does
 *  not hinge on how many queries ran, or with no band each within the rounding two scoring paths
 *  differ by; keys are not compared, as ties reorder them. */
inline bool equivalent_matches(copied_matches_t const& first, copied_matches_t const& second, double score_band) {
    if (!std::ranges::equal(first.counts, second.counts) || first.scores.size() != second.scores.size()) return false;
    if (score_band) return mean_score_deviation(first, second) <= score_band;
    for (std::size_t match = 0; match != first.scores.size(); ++match) {
        double const one = first.scores[match], other = second.scores[match];
        if (!within(one, other) || !within(other, one)) return false;
    }
    return true;
}

/** Whether the files at @p first and @p second both open and hold the same bytes. */
inline bool same_file_bytes(std::string_view first, std::string_view second) {
    auto [first_file, first_status] = util::mapped_file_t::read(first);
    auto [second_file, second_status] = util::mapped_file_t::read(second);
    return util::succeeded(first_status) && util::succeeded(second_status) &&
           std::ranges::equal(first_file.view(), second_file.view());
}

/** Whether a reloaded checkpoint must save again byte for byte. */
enum class saved_bytes_t {
    identical_k,
    differing_k,
};

/**
 *  @brief Saves @p index under @p stem, reloads it into its own residency, into RAM, onto the
 *      session's devices when it has any, and as an NVMe view, and judges every copy.
 *
 *  A checkpoint of one shard is one monolith, and one of @p parts shards is that many parts beside
 *  a manifest. The copy in the index's own residency must land matches equivalent to the original's
 *  under @p search, as <tt>search(handle) → copied_matches_t</tt>, within @p score_band when the stored
 *  values round on the way; every loaded copy must save again, byte for byte when @p bytes says so;
 *  the view must answer the search. Every file the round trip wrote is removed.
 */
template <typename verbs_type_, typename search_type_>
void expect_checkpoint_round_trip(handle<verbs_type_>& index, std::string_view stem, std::size_t parts,
                                  saved_bytes_t bytes, double score_band, search_type_&& search) {
    session_t& session = *index.session;
    sz::string_t const path = formatted("{}.safegraphs", stem);
    usearch_outcome_t outcome {};
    verbs_type_::save(index.get(), index.executor(), bytes_view(path), &outcome);
    if (!expect_success(outcome)) return;
    expect_true(describe(index).shards_count == parts);
    st::vector<sz::string_t> const saved = files_starting_with(stem);
    expect_true(saved.size() == (parts == 1 ? 1 : parts + 1));
    copied_matches_t const original_matches = search(index);

    // Every loaded copy saves again under its own stem, and its files compare to the original's.
    auto const same_checkpoint = [&](std::string_view other_stem) {
        st::vector<sz::string_t> const resaved = files_starting_with(other_stem);
        if (resaved.size() != saved.size()) return false;
        bool same = true;
        // A manifest names its parts by their own stem, so only the parts compare.
        for (std::size_t file = 0; file != saved.size(); ++file)
            if (parts == 1 || saved[file] != path) same = same && same_file_bytes(saved[file], resaved[file]);
        return same;
    };
    auto const judge_copy = [&](typename verbs_type_::index_t copy, usearch_residency_t residency,
                                std::string_view copy_stem) {
        if (!copy) return;
        handle<verbs_type_> reopened {copy, session, residency, index.geometry};
        if (residency == index.residency) {
            copied_matches_t const reloaded = search(reopened);
            bool const equivalent = equivalent_matches(reloaded, original_matches, score_band);
            if (!equivalent && reloaded.scores.size() == original_matches.scores.size())
                fmt::println(stderr, "  reloaded scores moved {:.3g} on average against a band of {:.3g}",
                             mean_score_deviation(reloaded, original_matches), score_band);
            expect_true(equivalent);
        }
        if (residency == usearch_residency_nvme_k) return;
        sz::string_t const copy_path = formatted("{}.safegraphs", copy_stem);
        verbs_type_::save(reopened.get(), reopened.executor(), bytes_view(copy_path), &outcome);
        expect_success(outcome);
        if (bytes == saved_bytes_t::identical_k) expect_true(same_checkpoint(copy_stem));
        for (sz::string_t const& file : files_starting_with(copy_stem)) std::remove(file.c_str());
    };

    st::vector<std::int32_t> placement;
    typename verbs_type_::index_t copy = nullptr;
    verbs_type_::load_ram(&copy, session.executor_for(usearch_residency_ram_k), bytes_view(path),
                          session.ram_options(0, placement), index.geometry, &outcome);
    expect_success(outcome);
    judge_copy(copy, usearch_residency_ram_k, formatted("{}-ram", stem));

    if (!session.devices.empty()) {
        copy = nullptr;
        verbs_type_::load_vram(&copy, session.executor, bytes_view(path), session.vram_options(parts, placement),
                               index.geometry, &outcome);
        expect_success(outcome);
        judge_copy(copy, usearch_residency_vram_k, formatted("{}-vram", stem));
    }

    if (session.storage) {
        copy = nullptr;
        verbs_type_::view_nvme(&copy, session.storage, bytes_view(path), session.nvme_options(),
                               session.nvme_ram_options(), index.geometry, &outcome);
        expect_success(outcome);
        judge_copy(copy, usearch_residency_nvme_k, formatted("{}-nvme", stem));
    }

    for (sz::string_t const& file : saved) std::remove(file.c_str());
}

/** How far one coordinate of a cloud spanning @p extent may move through the two roundings a stored
 *  pose takes at @p unit_roundoff, the pack at insert and the unpack at save. */
inline double pose_rounding_band(double extent, double unit_roundoff) noexcept { return 2 * unit_roundoff * extent; }

/**
 *  @brief Every coordinate of @p reloaded lies within the rounding band of its twin in
 *      @p original , and their lengths agree.
 *
 *  Clouds sit end to end, cloud @c i spanning coordinates <tt>offsets[i]</tt> up to
 *  <tt>offsets[i + 1]</tt>, and each one's band follows its own extent under the storage dtype's
 *  rounding error @p unit_roundoff.
 */
inline void expect_poses_within_rounding(std::span<float const> original, std::span<float const> reloaded,
                                         std::span<std::size_t const> offsets, double unit_roundoff) {
    if (!expect_true(original.size() == reloaded.size() && !offsets.empty() && offsets.back() == original.size()))
        return;
    std::size_t moved = 0;
    for (std::size_t cloud = 0; cloud + 1 < offsets.size(); ++cloud) {
        std::span<float const> const coordinates = original.subspan(offsets[cloud],
                                                                    offsets[cloud + 1] - offsets[cloud]);
        auto const extremes = std::minmax_element(coordinates.begin(), coordinates.end());
        double const band = coordinates.empty() ? 0
                                                : pose_rounding_band(*extremes.second - *extremes.first, unit_roundoff);
        for (std::size_t coordinate = offsets[cloud]; coordinate != offsets[cloud + 1]; ++coordinate)
            moved += std::fabs(static_cast<double>(original[coordinate]) - reloaded[coordinate]) > band;
    }
    expect_true(moved == 0, "{} coordinates moved past their clouds' rounding bands", moved);
}

/** Inserts @p count entries of @p inputs into the empty @p index under their keys, which must
 *  spread across the cell's shards and each count once where the residency serves a count; false
 *  when the insertion itself failed, leaving nothing for a cell to judge. */
template <typename verbs_type_>
bool check_built(sweep_cell_t const& cell, handle<verbs_type_>& index, typename verbs_type_::values_t inputs,
                 std::size_t count) {
    usearch_traversal_stats_t stats {};
    if (!expect_success(insert_entries(index, inputs, count, 0, cell_batch_k, 0, stats))) return false;
    typename verbs_type_::description_t const built = describe(index);
    expect_true(built.size == count && built.shards_count == cell.shards);
    expect_true(stats.computed_distances > 0);
    std::size_t miscounted = 0;
    for (std::size_t offset = 0; offset != count; ++offset) {
        usearch_outcome_t outcome {};
        std::size_t entries = 0;
        verbs_type_::count(index.get(), index.executor(), key_of_offset(index.session->seed, offset), &entries,
                           &outcome);
        // A residency that serves no count yet leaves the keys to the walks' recall.
        if (outcome.status == usearch_status_unsupported_k) break;
        if (outcome.status == usearch_status_success_k && entries == 1) continue;
        if (!miscounted++)
            fmt::println(stderr, "  a key counts {} entries: {}", entries, outcome.message ? outcome.message : "");
    }
    expect_true(miscounted == 0);
    return true;
}

/** Writes into @p recall the share of the exact scan's top ten in @p oracle a walk at @p expansion
 *  keeps over the first @p count queries of @p queries, ties at the cut counting however the scan
 *  happened to break them, and into @p spent, when named, what the walk spent. */
template <typename verbs_type_>
usearch_outcome_t walk_recall(handle<verbs_type_>& index, typename verbs_type_::queries_t queries, std::size_t count,
                              std::size_t expansion, copied_matches_t const& oracle, double& recall,
                              usearch_traversal_stats_t* spent = nullptr) {
    copied_matches_t found;
    usearch_traversal_stats_t stats {};
    usearch_outcome_t const walked = search_queries(index, queries, count, cell_batch_k, cell_wanted_k, expansion,
                                                    usearch_traversal_walk_k, stats, found);
    if (spent) *spent = stats;
    if (walked.status == usearch_status_success_k) recall = k_recall_at(cell_wanted_k, found, oracle);
    return walked;
}

/** A VRAM index's exact scan of the first @p count queries of @p queries scores every rank as the
 *  same index, saved under @p stem and reloaded on RAM, scans it, within @p score_band, ties free
 *  to name other keys; and it scores every query against every entry. */
template <typename verbs_type_>
void expect_device_scan(handle<verbs_type_>& index, typename verbs_type_::queries_t queries, std::size_t count,
                        std::string_view stem, double score_band) {
    session_t& session = *index.session;
    sz::string_t const path = formatted("{}.safegraphs", stem);
    usearch_outcome_t outcome {};
    verbs_type_::save(index.get(), index.executor(), bytes_view(path), &outcome);
    if (!expect_success(outcome)) return;
    st::vector<std::int32_t> placement;
    typename verbs_type_::index_t copy = nullptr;
    verbs_type_::load_ram(&copy, session.executor_for(usearch_residency_ram_k), bytes_view(path),
                          session.ram_options(0, placement), index.geometry, &outcome);
    for (sz::string_t const& file : files_starting_with(stem)) std::remove(file.c_str());
    if (!expect_success(outcome)) return;
    handle<verbs_type_> reloaded {copy, session, usearch_residency_ram_k, index.geometry};
    copied_matches_t on_device, on_host;
    usearch_traversal_stats_t device_stats {}, host_stats {};
    if (!expect_success(search_queries(index, queries, count, cell_batch_k, cell_wanted_k, 0, usearch_traversal_scan_k,
                                       device_stats, on_device)) ||
        !expect_success(search_queries(reloaded, queries, count, cell_batch_k, cell_wanted_k, 0,
                                       usearch_traversal_scan_k, host_stats, on_host)))
        return;
    expect_true(equivalent_matches(on_device, on_host, score_band));
    expect_true(device_stats.computed_distances >= count * describe(index).size);
}

/** A walk at @p expansion keeps @p floor of the exact scan's top ten over the first @p count
 *  queries of @p queries, as @ref walk_recall judges it; a residency that serves no scan yet has no
 *  oracle to judge by. */
template <typename verbs_type_>
void expect_walk_recall(handle<verbs_type_>& index, typename verbs_type_::queries_t queries, std::size_t count,
                        std::size_t expansion, double floor) {
    copied_matches_t oracle;
    usearch_outcome_t const scanned = scan_oracle(index, queries, count, cell_wanted_k, cell_batch_k, oracle);
    if (scanned.status == usearch_status_unsupported_k) return;
    double recall = 0;
    if (!expect_success(scanned) || !expect_success(walk_recall(index, queries, count, expansion, oracle, recall)))
        return;
    expect_true(recall >= floor, "recall at ten {:.3f} under the floor {:.3f}", recall, floor);
}

/** The first @p count queries' first matches hint a second search of them, which answers
 *  @p accepted: on success a key no entry holds is counted once per query, and each query ranks at
 *  least as near as its hint. */
template <typename verbs_type_>
void expect_hints(handle<verbs_type_>& index, typename verbs_type_::queries_t queries, std::size_t count,
                  std::size_t expansion, usearch_status_t accepted) {
    copied_matches_t found;
    usearch_traversal_stats_t stats {};
    if (!expect_success(search_queries(index, queries, count, cell_batch_k, cell_wanted_k, expansion,
                                       usearch_traversal_walk_k, stats, found)))
        return;
    st::vector<std::size_t> const offsets = match_offsets(found);
    nk::matrix<usearch_key_t> keys = make_matrix<usearch_key_t>(count, 2);
    nk::vector<std::size_t> counts = make_vector<std::size_t>(count);
    std::fill_n(counts.data(), count, 2);
    for (std::size_t query = 0; query != count; ++query) {
        keys(query, 0) = std::bit_cast<usearch_key_t>(found.keys[offsets[query]]);
        keys(query, 1) = key_of_offset(index.session->seed, (std::numeric_limits<util::u64_t>::max)() - query);
    }
    usearch_matches_view_t const hints {keys.data(), nullptr, counts.data(), 2};
    copied_matches_t hinted;
    usearch_traversal_stats_t hinted_stats {};
    // Several batches, so each takes its own rows of the hints.
    usearch_outcome_t const outcome = search_queries(index, queries, count,
                                                     util::divide_round_up<std::size_t>(count, 4), cell_wanted_k,
                                                     expansion, usearch_traversal_walk_k, hinted_stats, hinted, &hints);
    if (accepted != usearch_status_success_k) return void(expect_status(outcome, accepted));
    if (!expect_success(outcome)) return;
    st::vector<std::size_t> const hinted_offsets = match_offsets(hinted);
    std::size_t farther = 0;
    for (std::size_t query = 0; query != count; ++query)
        farther += !hinted.counts[query] || !within(hinted.scores[hinted_offsets[query]], found.scores[offsets[query]]);
    // A warm second call, its scratch already sized, ranks alike.
    copied_matches_t warm;
    usearch_traversal_stats_t warm_stats {};
    expect_success(search_queries(index, queries, count, util::divide_round_up<std::size_t>(count, 4), cell_wanted_k,
                                  expansion, usearch_traversal_walk_k, warm_stats, warm, &hints));
    expect_true(std::ranges::equal(warm.counts, hinted.counts) && warm_stats.missing_hints == count);
    expect_true(hinted_stats.missing_hints == count);
    expect_true(farther == 0);
}

/** An audit reaches every one of @p count entries. */
template <typename verbs_type_>
void expect_health(handle<verbs_type_>& index, std::size_t count) {
    usearch_outcome_t outcome {};
    usearch_reachability_t reachability {};
    verbs_type_::health(index.get(), index.executor(), &reachability, &outcome);
    expect_success(outcome);
    expect_true(reachability.audited == count && reachability.reachable == count);
}

/** A cursor opened on the executor of @p index yields each of its @p count entries once as stored,
 *  and from the first entry's key breadth-first that key first and then only what links lead to,
 *  once each; every yielded and linked key was minted, and a walk past one entry hands links. */
template <typename verbs_type_>
void expect_cursor(handle<verbs_type_>& index, std::size_t count) {
    usearch_key_t const start = key_of_offset(index.session->seed, 0);
    for (std::span<usearch_key_t const> const starts : {std::span<usearch_key_t const> {}, std::span {&start, 1}}) {
        usearch_outcome_t outcome {};
        typename verbs_type_::cursor_t cursor = nullptr;
        verbs_type_::cursor_init(index.get(), index.executor(), starts, &cursor, &outcome);
        expect_success(outcome);
        if (!cursor) continue;
        std::vector<bool> yielded(count);
        std::size_t pulled = 0, repeated = 0, unminted = 0, unordered = 0, written = 0, linked_count = 0;
        bool started = starts.empty();
        while (true) {
            usearch_keys_view_t keys {};
            usearch_keys_tape_view_t links {};
            verbs_type_::cursor_next(index.get(), index.executor(), index.arena(), cursor, cell_batch_k, &keys, &links,
                                     &written, &outcome);
            expect_success(outcome);
            if (outcome.status != usearch_status_success_k || !written) break;
            nk::vector<usearch_key_t> batch = make_vector<usearch_key_t>(written);
            nk::vector<std::size_t> offsets = make_vector<std::size_t>(written + 1);
            expect_true(download(batch.data(), keys.data, written * sizeof(usearch_key_t)));
            expect_true(download(offsets.data(), links.offsets, offsets.size() * sizeof(std::size_t)));
            nk::vector<usearch_key_t> linked = make_vector<usearch_key_t>(offsets[written]);
            expect_true(download(linked.data(), links.keys, linked.size() * sizeof(usearch_key_t)));
            if (!pulled && !starts.empty()) started = same_key(batch[0], start);
            for (std::size_t entry = 0; entry != written; ++entry) {
                std::optional<util::u64_t> const offset = offset_of_key(index.session->seed, batch[entry]);
                unminted += !offset || *offset >= count;
                if (offset && *offset < count) repeated += yielded[*offset], yielded[*offset] = true;
                unordered += offsets[entry] > offsets[entry + 1];
            }
            for (usearch_key_t const& key : linked) unminted += !offset_of_key(index.session->seed, key);
            pulled += written, linked_count += linked.size();
        }
        verbs_type_::cursor_free(cursor);
        expect_true(started);
        expect_true(repeated == 0);
        expect_true(unminted == 0);
        expect_true(unordered == 0);
        expect_true(pulled < 2 || linked_count > 0);
        starts.empty() ? expect_true(pulled == count) : expect_true(pulled > 1 && pulled <= count);
    }
}

/** A sample of the whole of @p index draws as many of its @p count entries as it wants, each a
 *  distinct minted key, its key's only entry, hanging from an earlier draw at finite coordinates,
 *  and an equal seed draws the same keys again; a GPU index refuses it and says why. */
template <typename verbs_type_>
void expect_sample(handle<verbs_type_>& index, std::size_t count) {
    std::size_t const wanted = util::min_of(cell_wanted_k, count);
    std::array<std::vector<usearch_key_t>, 2> drawn;
    for (std::vector<usearch_key_t>& keys : drawn) {
        usearch_outcome_t outcome {};
        sample_outputs_view_t outputs {};
        verbs_type_::sample(index.get(), index.executor(), index.arena(), {}, {}, {}, wanted, index.session->seed,
                            outputs, &outcome);
        if (index.residency == usearch_residency_vram_k)
            return void(expect_true(outcome.status == usearch_status_unsupported_k && outcome.message));
        if (!expect_success(outcome)) return;
        expect_true(outputs.written == wanted);
        keys.resize(outputs.written);
        nk::vector<std::size_t> ordinals = make_vector<std::size_t>(outputs.written);
        nk::vector<std::size_t> parented = make_vector<std::size_t>(outputs.written);
        nk::matrix<usearch_key_t> parents = make_matrix<usearch_key_t>(outputs.written, outputs.parents.columns);
        nk::matrix<util::f32_t> coordinates = make_matrix<util::f32_t>(
            outputs.written, outputs.coordinates.stride / sizeof(util::f32_t));
        expect_true(download(keys.data(), outputs.keys.data, keys.size() * sizeof(usearch_key_t)));
        expect_true(download(ordinals.data(), outputs.entry_ordinals, ordinals.size() * sizeof(std::size_t)));
        expect_true(download(parented.data(), outputs.parents.counts, parented.size() * sizeof(std::size_t)));
        expect_true(download(parents.data(), outputs.parents.keys, parents.size() * sizeof(usearch_key_t)));
        expect_true(download(coordinates.data(), outputs.coordinates.data, coordinates.size() * sizeof(util::f32_t)));
        auto const drawn_before = [&](usearch_key_t const& key, std::size_t draw) {
            return std::any_of(keys.begin(), keys.begin() + draw,
                               [&](usearch_key_t const& earlier) { return same_key(earlier, key); });
        };
        std::size_t misdrawn = 0;
        for (std::size_t draw = 0; draw != keys.size(); ++draw) {
            std::optional<util::u64_t> const offset = offset_of_key(index.session->seed, keys[draw]);
            misdrawn += !offset || *offset >= count || ordinals[draw] != 0 || drawn_before(keys[draw], draw);
            misdrawn += parented[draw] && !drawn_before(parents(draw, 0), draw);
            misdrawn += !std::isfinite(coordinates(draw, 0)) || !std::isfinite(coordinates(draw, 1)) ||
                        !std::isfinite(coordinates(draw, 2));
        }
        expect_true(misdrawn == 0);
    }
    expect_true(drawn[0].size() == drawn[1].size() &&
                std::equal(drawn[0].begin(), drawn[0].end(), drawn[1].begin(), same_key));
}

/** Calls @p call with a handler cancelling at its first report until a call answers other than
 *  cancelled or @p limit calls ran, each cancelled call leaving @p plan live and naming one of its
 *  phases, @p outcome keeping the last call's. */
template <typename plan_type_, typename call_type_>
void drive_cancelled_plan(plan_type_ const& plan, usearch_outcome_t const& outcome, std::size_t limit,
                          call_type_ const& call) {
    usearch_progress_t const cancelling {[](std::size_t, std::size_t, void*) noexcept { return usearch_cancel_k; },
                                         nullptr, 0};
    for (std::size_t calls = 0; calls <= limit; ++calls) {
        usearch_phase_progress_t progress {};
        call(cancelling, &progress);
        if (outcome.status != usearch_status_cancelled_k) break;
        expect_true(plan != nullptr && progress.phase != usearch_phase_none_k &&
                        progress.phase != usearch_phase_done_k && progress.completed <= progress.total,
                    "Cancelled call {} of {} left a {} plan in phase {} at {} of {}", calls, limit,
                    plan != nullptr ? "live" : "null", static_cast<int>(progress.phase), progress.completed,
                    progress.total);
    }
}

/** Resharding @p index , whose entries are keyed by its first offsets, as the one source into two
 *  empty targets of its own session and shard count through a plan every call of which is
 *  cancelled leaves @p index empty and live and every key in one target once, where the residency
 *  counts entries, and the plan's finish audits every entry; both targets then gather and cut again
 *  into @p index and a third index, several sources into several targets, every key landing once
 *  more. Malformed lists, and a source among several targets, refuse first, an NVMe view refuses it
 *  as read-only and a GPU index answers @p on_device, a refusal saying why. */
template <typename verbs_type_>
void expect_reshard_cut(handle<verbs_type_>& index, usearch_status_t on_device) {
    usearch_status_t const wanted = index.residency == usearch_residency_nvme_k   ? usearch_status_read_only_k
                                    : index.residency == usearch_residency_vram_k ? on_device
                                                                                  : usearch_status_success_k;
    typename verbs_type_::description_t const before = describe(index);
    std::array<handle<verbs_type_>, 2> targets {
        open_index<verbs_type_>(*index.session, index.geometry, before.shards_count),
        open_index<verbs_type_>(*index.session, index.geometry, before.shards_count)};
    if (!targets[0] || !targets[1]) return void(expect_true(false));
    typename verbs_type_::index_t const source = index.get();
    std::array<typename verbs_type_::index_t, 2> const handles {targets[0].get(), targets[1].get()};
    usearch_outcome_t outcome {};
    usearch_refine_report_t report {};
    if (wanted != usearch_status_success_k) {
        verbs_type_::reshard({&source, 1}, handles, index.executor(), index.session->seed, &report, &outcome);
        return void(expect_true(outcome.status == wanted && outcome.message));
    }
    // Malformed lists refuse as arguments before anything closes: no targets, a null slot, or a
    // target named twice; a source among several targets would be cut into itself.
    using index_t = typename verbs_type_::index_t;
    std::array<index_t, 2> const holding_null {targets[0].get(), nullptr};
    std::array<index_t, 2> const repeated {targets[0].get(), targets[0].get()};
    for (std::span<index_t const> malformed :
         {std::span<index_t const> {}, std::span<index_t const> {holding_null}, std::span<index_t const> {repeated}}) {
        verbs_type_::reshard({&source, 1}, malformed, index.executor(), index.session->seed, &report, &outcome);
        expect_true(outcome.status == usearch_status_invalid_argument_k && outcome.message);
    }
    std::array<index_t, 2> const several {source, targets[1].get()};
    verbs_type_::reshard(several, handles, index.executor(), index.session->seed, &report, &outcome);
    expect_true(outcome.status == usearch_status_config_mismatch_k && outcome.message);
    expect_true(describe(index).size == before.size);

    typename verbs_type_::reshard_plan_t plan = nullptr;
    drive_cancelled_plan(plan, outcome, before.size, [&](usearch_progress_t progress, usearch_phase_progress_t* phase) {
        verbs_type_::reshard({&source, 1}, handles, index.executor(), index.session->seed, &report, &outcome, 0, 0,
                             progress, &plan, phase);
    });
    expect_success(outcome);
    expect_true(plan != nullptr);
    verbs_type_::reshard_plan_finish(index.executor(), plan, &report, &outcome);
    expect_success(outcome);
    expect_true(report.reachability.audited == before.size);
    // The finish freed the plan, so finishing again can only name none.
    verbs_type_::reshard_plan_finish(index.executor(), nullptr, nullptr, &outcome);
    expect_status(outcome, usearch_status_invalid_argument_k);
    expect_true(describe(index).size == 0 && describe(targets[0]).size + describe(targets[1]).size == before.size);
    auto const miscounted = [&](handle<verbs_type_>& first, handle<verbs_type_>& second) {
        std::size_t wrong = 0;
        for (std::size_t offset = 0; offset != before.size; ++offset) {
            usearch_key_t const key = key_of_offset(index.session->seed, offset);
            std::size_t landed = 0;
            for (handle<verbs_type_>* target : {&first, &second}) {
                std::size_t entries = 0;
                verbs_type_::count(target->get(), target->executor(), key, &entries, &outcome);
                if (outcome.status == usearch_status_unsupported_k) return std::size_t {0};
                landed += entries;
            }
            wrong += landed != 1;
        }
        return wrong;
    };
    expect_true(miscounted(targets[0], targets[1]) == 0);

    handle<verbs_type_> third = open_index<verbs_type_>(*index.session, index.geometry, before.shards_count);
    if (!third) return void(expect_true(false));
    std::array<typename verbs_type_::index_t, 2> const recut {source, third.get()};
    verbs_type_::reshard(handles, recut, index.executor(), index.session->seed + 1, &report, &outcome);
    expect_success(outcome);
    expect_true(report.reachability.audited == before.size);
    expect_true(describe(targets[0]).size == 0 && describe(targets[1]).size == 0 &&
                describe(index).size + describe(third).size == before.size);
    expect_true(miscounted(index, third) == 0);
}

/** A RAM index saves into its session's arena, whose bytes name their metadata, and loads from them
 *  into RAM again with as many entries over as many shards, a walk at @p expansion keeping @p floor
 *  of the exact scan's top ten over the first @p count queries of @p queries; an NVMe view refuses
 *  the save as unsupported and a GPU index answers @p on_device, a refusal saying why. */
template <typename verbs_type_>
void expect_buffer_round_trip(handle<verbs_type_>& index, typename verbs_type_::queries_t queries, std::size_t count,
                              std::size_t expansion, double floor, usearch_status_t on_device) {
    usearch_status_t const wanted = index.residency == usearch_residency_nvme_k   ? usearch_status_unsupported_k
                                    : index.residency == usearch_residency_vram_k ? on_device
                                                                                  : usearch_status_success_k;
    usearch_outcome_t outcome {};
    usearch_bytes_view_t monolith {};
    verbs_type_::save_buffer(index.get(), index.executor(), index.arena(), &monolith, &outcome);
    if (wanted != usearch_status_success_k) return void(expect_true(outcome.status == wanted && outcome.message));
    if (!expect_success(outcome)) return;
    std::string bytes(monolith.length, '\0');
    expect_true(download(bytes.data(), monolith.data, bytes.size()));
    typename verbs_type_::metadata_t metadata {};
    verbs_type_::metadata_read_buffer(index.session->executor, bytes_view(bytes), &metadata, &outcome);
    expect_success(outcome);
    expect_true(metadata.connectivity != 0);
    st::vector<std::int32_t> placement;
    typename verbs_type_::index_t copy = nullptr;
    verbs_type_::load_ram_buffer(&copy, index.session->executor_for(usearch_residency_ram_k), bytes_view(bytes),
                                 index.session->ram_options(0, placement), index.geometry, &outcome);
    expect_success(outcome);
    handle<verbs_type_> reloaded {copy, *index.session, usearch_residency_ram_k, index.geometry};
    if (!reloaded) return void(expect_true(false));
    typename verbs_type_::description_t const original = describe(index), restored = describe(reloaded);
    expect_true(restored.size == original.size && restored.shards_count == original.shards_count);
    expect_walk_recall(reloaded, queries, count, expansion, floor);
}

/** The offsets each entry of @p index links to, sorted, under the entry's own offset, as a cursor
 *  over the whole index reads them. */
template <typename verbs_type_>
std::map<util::u64_t, std::vector<util::u64_t>> links_by_offset(handle<verbs_type_>& index) {
    std::map<util::u64_t, std::vector<util::u64_t>> rows;
    usearch_outcome_t outcome {};
    typename verbs_type_::cursor_t cursor = nullptr;
    verbs_type_::cursor_init(index.get(), index.executor(), std::span<usearch_key_t const> {}, &cursor, &outcome);
    if (!expect_success(outcome) || !cursor) return rows;
    std::size_t written = 0;
    while (true) {
        usearch_keys_view_t keys {};
        usearch_keys_tape_view_t links {};
        verbs_type_::cursor_next(index.get(), index.executor(), index.arena(), cursor, cell_batch_k, &keys, &links,
                                 &written, &outcome);
        expect_success(outcome);
        if (outcome.status != usearch_status_success_k || !written) break;
        std::vector<usearch_key_t> batch(written);
        std::vector<std::size_t> offsets(written + 1);
        expect_true(download(batch.data(), keys.data, written * sizeof(usearch_key_t)));
        expect_true(download(offsets.data(), links.offsets, offsets.size() * sizeof(std::size_t)));
        std::vector<usearch_key_t> linked(offsets.back());
        expect_true(download(linked.data(), links.keys, linked.size() * sizeof(usearch_key_t)));
        for (std::size_t entry = 0; entry != written; ++entry) {
            std::optional<util::u64_t> const offset = offset_of_key(index.session->seed, batch[entry]);
            if (!expect_true(offset.has_value())) continue;
            std::vector<util::u64_t>& row = rows[*offset];
            for (std::size_t link = offsets[entry]; link != offsets[entry + 1]; ++link)
                row.push_back(offset_of_key(index.session->seed, linked[link]).value_or(~util::u64_t {}));
            std::sort(row.begin(), row.end());
        }
    }
    verbs_type_::cursor_free(cursor);
    return rows;
}

/** How many entries of @p reloaded hold other links than the same entry of @p original, or are
 *  missing from one of them. */
inline std::size_t rows_that_differ(std::map<util::u64_t, std::vector<util::u64_t>> const& original,
                                    std::map<util::u64_t, std::vector<util::u64_t>> const& reloaded) {
    std::size_t differing = 0;
    for (auto const& [offset, row] : original) {
        auto const found = reloaded.find(offset);
        differing += found == reloaded.end() || found->second != row;
    }
    return differing + (reloaded.size() > original.size() ? reloaded.size() - original.size() : 0);
}

/** Widening: a checkpoint of @p index saved under @p stem , stored at some degree, loads into RAM
 *  at twice it with every entry's links kept and the rest of its row empty, and one refinement
 *  then fills more; at half of it the load is refused as @c config_mismatch. Where the session
 *  has GPUs, a VRAM load at twice it saves and loads back into RAM with the same links, and
 *  NVIDIA refuses an odd degree. */
template <typename verbs_type_>
void expect_widened(handle<verbs_type_>& index, std::string const& stem) {
    session_t& session = *index.session;
    sz::string_t const path = formatted("{}.safegraphs", stem);
    usearch_outcome_t outcome {};
    verbs_type_::save(index.get(), index.executor(), bytes_view(path), &outcome);
    if (!expect_success(outcome)) return;
    typename verbs_type_::metadata_t stored {};
    verbs_type_::metadata_read(session.executor, bytes_view(path), &stored, &outcome);
    if (!expect_success(outcome) || !expect_true(stored.connectivity > 1)) return;
    std::size_t const parts = describe(index).shards_count;
    std::size_t const widened_degree = 2 * stored.connectivity;
    std::map<util::u64_t, std::vector<util::u64_t>> const original = links_by_offset(index);

    st::vector<std::int32_t> placement;
    auto const load_at = [&](std::size_t connectivity, usearch_residency_t residency, sz::string_t const& from,
                             handle<verbs_type_>& loaded) {
        typename verbs_type_::geometry_t geometry = index.geometry;
        metadata_of<verbs_type_>(geometry).connectivity = connectivity;
        typename verbs_type_::index_t opened = nullptr;
        usearch_outcome_t loading {};
        if (residency == usearch_residency_ram_k)
            verbs_type_::load_ram(&opened, session.executor_for(residency), bytes_view(from),
                                  session.ram_options(0, placement), geometry, &loading);
        else
            verbs_type_::load_vram(&opened, session.executor, bytes_view(from), session.vram_options(parts, placement),
                                   geometry, &loading);
        loaded = {opened, session, residency, geometry};
        return loading;
    };

    handle<verbs_type_> widened;
    if (expect_success(load_at(widened_degree, usearch_residency_ram_k, path, widened)) && widened) {
        std::map<util::u64_t, std::vector<util::u64_t>> const kept = links_by_offset(widened);
        std::size_t const differing = rows_that_differ(original, kept);
        expect_true(differing == 0, "{} of {} rows changed or grew past their stored links on load_ram at {}",
                    differing, original.size(), widened_degree);
        usearch_refine_report_t report {};
        verbs_type_::refine(widened.get(), widened.executor(), &report, &outcome);
        expect_success(outcome);
        // Wider rows re-select, dropping links as well as adding them, yet reach every entry
        expect_true(report.reachability.audited == original.size() &&
                        report.reachability.reachable == report.reachability.audited,
                    "refine at degree {} reached {} of {} entries", widened_degree, report.reachability.reachable,
                    original.size());
    }

    handle<verbs_type_> narrowed;
    expect_status(load_at(stored.connectivity / 2, usearch_residency_ram_k, path, narrowed),
                  usearch_status_config_mismatch_k);
    expect_true(!narrowed, "load_ram at {} of a degree {} checkpoint returned an index", stored.connectivity / 2,
                stored.connectivity);

    if (!session.devices.empty()) {
        st::vector<usearch_gpu_device_t> const listed = listed_devices();
        bool const nvidia = listed[session.devices[0]].vendor == usearch_gpu_vendor_nvidia_k;
        usearch_status_t const wanted = nvidia && !std::has_single_bit(widened_degree)
                                            ? usearch_status_config_mismatch_k
                                            : usearch_status_success_k;
        handle<verbs_type_> on_device;
        expect_status(load_at(widened_degree, usearch_residency_vram_k, path, on_device), wanted);
        if (wanted == usearch_status_success_k && on_device) {
            sz::string_t const resaved = formatted("{}-vram.safegraphs", stem);
            verbs_type_::save(on_device.get(), on_device.executor(), bytes_view(resaved), &outcome);
            handle<verbs_type_> returned;
            if (expect_success(outcome) && expect_success(load_at(0, usearch_residency_ram_k, resaved, returned)) &&
                returned) {
                std::size_t const differing = rows_that_differ(original, links_by_offset(returned));
                expect_true(differing == 0, "{} of {} rows changed through load_vram at {} and save", differing,
                            original.size(), widened_degree);
            }
        }
        if (nvidia) {
            handle<verbs_type_> odd;
            expect_status(load_at(widened_degree + 1, usearch_residency_vram_k, path, odd),
                          usearch_status_config_mismatch_k);
        }
    }
    for (sz::string_t const& file : files_starting_with(stem)) std::remove(file.c_str());
}

/** Every metric kind the modality spells reads back by its name, "unknown" included, and
 *  "unrecognized" is refused with @c invalid_argument, leaving the output alone. */
template <typename verbs_type_>
void test_metric_names() {
    using metric_t = typename verbs_type_::metric_t;
    std::size_t misread = 0;
    expect_true(std::string_view {verbs_type_::metric_name(static_cast<metric_t>(0))} == "unknown");
    expect_true(std::string_view {verbs_type_::metric_name(static_cast<metric_t>(1))} != "unrecognized");
    // Every kind fits three bits, and a number no kind takes reads "unrecognized".
    for (util::u32_t kind = 0; kind != 8; ++kind) {
        metric_t const metric = static_cast<metric_t>(kind);
        if (std::string_view {verbs_type_::metric_name(metric)} == "unrecognized") continue;
        metric_t parsed = static_cast<metric_t>(1);
        misread += verbs_type_::metric_from_name(bytes_view(verbs_type_::metric_name(metric)), &parsed) !=
                       usearch_status_success_k ||
                   parsed != metric;
    }
    metric_t untouched = static_cast<metric_t>(1);
    expect_true(misread == 0);
    expect_true(verbs_type_::metric_from_name(bytes_view("unrecognized"), &untouched) ==
                usearch_status_invalid_argument_k);
    expect_true(untouched == static_cast<metric_t>(1));
}

/** The index round-trips through all residencies under @p stem, as
 *  @ref expect_checkpoint_round_trip judges it on the first @p count queries of @p queries, within
 *  @p score_band where stored entries round; in an NVMe cell its saved view then answers the same
 *  walk, audits, walks, samples and refuses every write, one entry of @p inputs included. */
template <typename verbs_type_>
void expect_persistence(sweep_cell_t const& cell, handle<verbs_type_>& index, typename verbs_type_::values_t inputs,
                        typename verbs_type_::queries_t queries, std::size_t count, std::size_t expansion, double floor,
                        std::string const& stem, saved_bytes_t bytes, double score_band) {
    // Entries that round on the way may steer a walk down another path, so they compare by exact
    // scans wherever the residency serves one.
    auto const search = [&](handle<verbs_type_>& searched) {
        copied_matches_t found;
        usearch_traversal_stats_t stats {};
        usearch_outcome_t outcome {};
        if (score_band)
            outcome = search_queries(searched, queries, count, cell_batch_k, cell_wanted_k, 0, usearch_traversal_scan_k,
                                     stats, found);
        if (!score_band || outcome.status == usearch_status_unsupported_k)
            found = {}, outcome = search_queries(searched, queries, count, cell_batch_k, cell_wanted_k, expansion,
                                                 usearch_traversal_walk_k, stats, found);
        if (outcome.status != usearch_status_success_k) return copied_matches_t {};
        return found;
    };
    expect_checkpoint_round_trip(index, stem, cell.shards, bytes, score_band, search);
    if (cell.residency != usearch_residency_nvme_k) return;
    std::size_t const entries = describe(index).size;
    handle<verbs_type_> view = save_and_reopen(std::move(index), stem + "-view.safegraphs", usearch_residency_nvme_k);
    expect_true(bool(view));
    if (view) {
        expect_walk_recall(view, queries, count, expansion, floor);
        expect_health(view, entries);
        expect_cursor(view, entries);
        expect_sample(view, entries);
        expect_reshard_cut(view, usearch_status_unsupported_k);
        expect_buffer_round_trip(view, queries, count, expansion, floor, usearch_status_unsupported_k);
        usearch_outcome_t outcome {};
        usearch_key_t const key = key_of_offset(view.session->seed, 0);
        verbs_type_::insert(view.get(), view.executor(), 1, keys_view({&key, 1}), inputs, 0, nullptr, &outcome);
        expect_status(outcome, usearch_status_read_only_k);
        verbs_type_::clear(view.get(), view.executor(), &outcome);
        expect_status(outcome, usearch_status_read_only_k);
    }
    view = {};
    for (sz::string_t const& file : files_starting_with(stem + "-view")) std::remove(file.c_str());
}

/** Two refinements settle: each changes no more entries than it re-selected, the second keeps every
 *  entry the first left reachable, and a walk spending as many distances as one at @p expansion did
 *  before keeps as much of the exact scan's top ten over the first @p count queries of @p queries,
 *  so the scan and the walks that judge it must all succeed, where the residency serves a scan; a
 *  VRAM index whose tower still declares its refinement answers @p on_device instead and is left
 *  out of the judging. */
template <typename verbs_type_>
void expect_refined(handle<verbs_type_>& index, typename verbs_type_::queries_t queries, std::size_t count,
                    std::size_t expansion, usearch_status_t on_device) {
    copied_matches_t oracle;
    double recall_before = 0, recall_after = 0;
    usearch_traversal_stats_t spent_before {}, spent_after {};
    char const* refused = nullptr;
    usearch_outcome_t oracle_outcome = scan_oracle(index, queries, count, cell_wanted_k, cell_batch_k, oracle);
    usearch_status_t const scanned = oracle_outcome.status;
    if (scanned != usearch_status_success_k) refused = "scan_oracle";
    else if (oracle_outcome = walk_recall(index, queries, count, expansion, oracle, recall_before, &spent_before);
             oracle_outcome.status != usearch_status_success_k)
        refused = "walk_recall";
    usearch_refine_report_t reports[2] {};
    for (usearch_refine_report_t& report : reports) {
        usearch_outcome_t outcome {};
        verbs_type_::refine(index.get(), index.executor(), &report, &outcome);
        if (index.residency == usearch_residency_vram_k && on_device != usearch_status_success_k)
            return void(expect_status(outcome, on_device));
        expect_success(outcome);
        expect_true(report.changed <= report.reselected);
    }
    // A round draws new hubs, so its changed count need not shrink; its reach must not.
    expect_true(reports[1].reachability.reachable >= reports[0].reachability.reachable);
    if (scanned == usearch_status_unsupported_k) return;
    if (refused) {
        fmt::println(stderr, "  the recall oracle's {} refused with {}: {}", refused,
                     usearch_status_name(oracle_outcome.status),
                     oracle_outcome.message ? oracle_outcome.message : "no message");
        return void(expect_true(false));
    }
    // A round keeps only diverse links, so a walk of one breadth now spends fewer distances.
    std::size_t breadth = expansion;
    expect_success(walk_recall(index, queries, count, breadth, oracle, recall_after, &spent_after));
    while (recall_after + recall_margin_k < recall_before &&
           spent_after.computed_distances < spent_before.computed_distances && breadth < 2 * expansion) {
        breadth += util::max_of<std::size_t>(expansion / 8, 1);
        expect_success(walk_recall(index, queries, count, breadth, oracle, recall_after, &spent_after));
    }
    if (recall_after + recall_margin_k < recall_before)
        fmt::println(stderr, "  recall at ten fell from {:.3f} to {:.3f} over two refinements, at breadth {}",
                     recall_before, recall_after, breadth);
    expect_true(recall_after + recall_margin_k >= recall_before);
}

/** A refine plan driven by calls each cancelled at their first report ends its round, holds
 *  @p index closed to an insertion of the first entry of @p inputs under a fresh key until it
 *  finishes, and its finish audits all @p count entries, the refused one never landing; a VRAM
 *  index whose tower still declares its plan answers @p on_device at the opening call. */
template <typename verbs_type_>
void expect_refine_plan(handle<verbs_type_>& index, typename verbs_type_::values_t inputs, std::size_t count,
                        usearch_status_t on_device) {
    usearch_outcome_t outcome {};
    usearch_refine_report_t report {};
    typename verbs_type_::refine_plan_t plan = nullptr;
    drive_cancelled_plan(plan, outcome, count, [&](usearch_progress_t progress, usearch_phase_progress_t* phase) {
        verbs_type_::refine(index.get(), index.executor(), &report, &outcome, 0, 0, nullptr, progress, &plan, phase);
    });
    if (index.residency == usearch_residency_vram_k && on_device != usearch_status_success_k)
        return void(expect_true(outcome.status == on_device && !plan));
    expect_success(outcome);
    expect_true(plan != nullptr && report.changed <= report.reselected);
    usearch_key_t const fresh = key_of_offset(index.session->seed, count);
    verbs_type_::insert(index.get(), index.executor(), 1, keys_view({&fresh, 1}), inputs, 0, nullptr, &outcome);
    expect_status(outcome, usearch_status_contended_k);
    verbs_type_::refine_plan_finish(index.get(), index.executor(), plan, &report, &outcome);
    expect_success(outcome);
    expect_true(report.reachability.audited == count);
    expect_true(describe(index).size == count);
}

/** A second source of @p count entries of @p inputs, keyed past the index's own @p first_offset
 *  entries, gathers whole into @p index , the one target, through a reshard plan every call of
 *  which is cancelled: the source ends empty and live, the plan holds the index closed to an
 *  insertion until it finishes, its finish audits every entry, and the sizes add up; a device
 *  index whose tower does not reshard yet answers @p on_device instead. */
template <typename verbs_type_>
void expect_reshard_gather(handle<verbs_type_>& index, typename verbs_type_::values_t inputs, std::size_t count,
                           std::size_t first_offset, usearch_status_t on_device) {
    handle<verbs_type_> other = open_index<verbs_type_>(*index.session, index.geometry, 1);
    if (!expect_true(bool(other))) return;
    usearch_traversal_stats_t stats {};
    if (!expect_success(insert_entries(other, inputs, count, first_offset, cell_batch_k, 0, stats))) return;
    std::array<typename verbs_type_::index_t, 2> const sources {index.get(), other.get()};
    typename verbs_type_::index_t const target = index.get();
    usearch_outcome_t outcome {};
    usearch_refine_report_t report {};
    typename verbs_type_::reshard_plan_t plan = nullptr;
    drive_cancelled_plan(plan, outcome, first_offset + count,
                         [&](usearch_progress_t progress, usearch_phase_progress_t* phase) {
                             verbs_type_::reshard(sources, {&target, 1}, index.executor(), 0, &report, &outcome, 0, 0,
                                                  progress, &plan, phase);
                         });
    if (index.residency == usearch_residency_vram_k && on_device != usearch_status_success_k)
        return void(expect_true(outcome.status == on_device && !plan));
    expect_success(outcome);
    expect_true(plan != nullptr);
    expect_true(describe(other).size == 0);
    usearch_key_t const fresh = key_of_offset(index.session->seed, first_offset + count);
    verbs_type_::insert(index.get(), index.executor(), 1, keys_view({&fresh, 1}), inputs, 0, nullptr, &outcome);
    expect_status(outcome, usearch_status_contended_k);
    verbs_type_::reshard_plan_finish(index.executor(), plan, &report, &outcome);
    expect_success(outcome);
    expect_true(report.reachability.audited == first_offset + count);
    expect_true(describe(index).size == first_offset + count);
}

/** A cleared index holds nothing, and a cursor opened over its entries before the clear answers
 *  @c contended_k after it; an index a reshard emptied first takes the first entry of @p inputs
 *  back, since clearing nothing need not end a cursor. */
template <typename verbs_type_>
void expect_cleared(handle<verbs_type_>& index, typename verbs_type_::values_t inputs) {
    usearch_traversal_stats_t stats {};
    if (!describe(index).size) expect_success(insert_entries(index, inputs, 1, 0, cell_batch_k, 0, stats));
    usearch_outcome_t outcome {};
    typename verbs_type_::cursor_t cursor = nullptr;
    verbs_type_::cursor_init(index.get(), index.executor(), {}, &cursor, &outcome);
    expect_success(outcome);
    verbs_type_::clear(index.get(), index.executor(), &outcome);
    expect_success(outcome);
    expect_true(describe(index).size == 0);
    if (!cursor) return void(expect_true(false));
    usearch_keys_view_t keys {};
    std::size_t written = 0;
    verbs_type_::cursor_next(index.get(), index.executor(), index.arena(), cursor, cell_batch_k, &keys, nullptr,
                             &written, &outcome);
    verbs_type_::cursor_free(cursor);
    expect_true(outcome.status == usearch_status_contended_k && written == 0);
}

} // namespace abi

} // namespace unum::usearch::test
