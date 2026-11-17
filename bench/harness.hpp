/**
 *  @file bench/harness.hpp
 *  @author Ash Vardanian
 *  @date July 17, 2026
 *  @brief Settings and logging helpers shared by the benchmarking harnesses.
 *
 *  Every setting a benchmark reads itself is an environment variable that a flag of the same name
 *  overrides, spelled out in @ref read_bench_settings ; the session flags and their @c RETRIEVAL_
 *  defaults stay with @c tools/abi.hpp. The rest is presentation-only: progress that redraws one
 *  line on a terminal and prints deciles when piped, human-readable count formatting, enum
 *  pretty-printers, and process-level memory telemetry. No benchmark logic belongs in this header.
 */
#pragma once

#include <cctype>  // `std::tolower` - a setting's flag spelling
#include <cmath>   // `std::isfinite` - a real setting
#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint64_t` - the seed and its streams
#include <cstdio>  // `std::fopen` for `/proc/self/status`
#include <cstdlib> // `std::strtoull`, `std::getenv`
#include <cstring> // `std::strncmp`

#include <charconv>     // `std::from_chars` - counts, seeds and durations
#include <chrono>       // `std::chrono::steady_clock` for progress rates
#include <optional>     // `std::optional` - the compiled filter
#include <random>       // `std::random_device` - a drawn seed
#include <regex>        // `std::regex` - the filter
#include <string>       // `std::string` - the filter's pattern
#include <string_view>  // `std::string_view` - setting names, row names
#include <system_error> // `std::errc`

#include <fmt/format.h>

#include <numkong/types.h>

#include "usearch/utilities/types.hpp"

namespace unum::usearch::bench {
using namespace unum::usearch::utilities;

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

/** What a benchmark reads itself beyond the session flags: the common settings, then each row's
 *  warm-up and timed run. */
struct bench_settings_t : common_settings_t {
    std::chrono::milliseconds warmup = std::chrono::seconds(1);
    std::chrono::milliseconds time_limit = std::chrono::seconds(10);
};

/** @c USEARCH_SEED , @c USEARCH_FILTER , @c USEARCH_WARMUP and @c USEARCH_TIME_LIMIT , each
 *  overridden by its flag. */
inline bench_settings_t read_bench_settings(int argc, char** argv) noexcept {
    bench_settings_t settings;
    settings.seed = read_seed_setting(argc, argv, "USEARCH_SEED", "--seed");
    settings.filter = read_filter_setting(argc, argv, "USEARCH_FILTER", "--filter");
    settings.program = argc != 0 ? argv[0] : "";
    settings.warmup = read_duration_setting(argc, argv, "USEARCH_WARMUP", "--warmup", settings.warmup);
    settings.time_limit = read_duration_setting(argc, argv, "USEARCH_TIME_LIMIT", "--time-limit", settings.time_limit);
    return settings;
}

/** A count or rate rendered as "1.23 G" / "45.6 M" / "789 k" for scannable tables. */
template <typename value_type_>
struct humanized {
    value_type_ value;
    explicit humanized(value_type_ value) noexcept : value(value) {}
};

} // namespace unum::usearch::bench

namespace us = unum::usearch;
namespace nk = ashvardanian::numkong;

/** Renders straight into the output, so a progress tick allocates nothing. */
template <typename value_type_>
struct fmt::formatter<unum::usearch::bench::humanized<value_type_>> {
    constexpr auto parse(format_parse_context& context) { return context.begin(); }
    auto format(us::bench::humanized<value_type_> wrapped, format_context& context) const {
        double const value = static_cast<double>(wrapped.value);
        if (value >= 1e9) return fmt::format_to(context.out(), "{:.2f} G", value / 1e9);
        if (value >= 1e6) return fmt::format_to(context.out(), "{:.2f} M", value / 1e6);
        if (value >= 1e3) return fmt::format_to(context.out(), "{:.2f} k", value / 1e3);
        return fmt::format_to(context.out(), "{:.0f}", value);
    }
};

namespace unum::usearch::bench {

namespace stdc = std::chrono;
using steady_clock_t = stdc::steady_clock;
using time_point_t = steady_clock_t::time_point;

/** A byte count in the gigabytes that reports print as "GB". */
constexpr double gigabytes(std::size_t bytes) noexcept { return double(bytes) / (1ull << 30); }

/**
 *  @brief Phase progress for every usage form: an in-place line with a bar, rate, and ETA
 *      on a TTY, or one plain line per completed decile when piped or logged - short and
 *      ANSI-free where scrollback is permanent.
 *
 *  Tick with the completed count from the driving loop; @c finish clears the TTY line so
 *  the phase report takes its place.
 */
enum class progress_style_t { bar_k, deciles_k };

class progress_bar_t {
  public:
    progress_bar_t(char const* label, std::size_t total, progress_style_t style) noexcept
        : label_(label), total_(total), terminal_(style == progress_style_t::bar_k) {}

    void tick(std::size_t completed) noexcept {
        if (terminal_) render_bar_(completed);
        else render_decile_(completed);
    }
    void finish() noexcept {
        if (terminal_) fmt::print("\r\033[K");
    }

  private:
    inline static constexpr std::size_t bar_cells_k = 24;

    char const* label_;
    std::size_t total_;
    bool terminal_;
    time_point_t started_ {steady_clock_t::now()};
    time_point_t rendered_ {};
    std::size_t reported_deciles_ = 0;

    double rate_per_second_(std::size_t completed) const noexcept {
        double const seconds = stdc::duration<double>(steady_clock_t::now() - started_).count();
        return seconds > 0 ? static_cast<double>(completed) / seconds : 0;
    }

    void render_bar_(std::size_t completed) noexcept {
        auto const now = steady_clock_t::now();
        bool const final = completed == total_;
        if (!final && now - rendered_ < stdc::milliseconds(100)) return; // The work outruns the terminal.
        rendered_ = now;
        std::size_t const filled_cells = total_ ? bar_cells_k * completed / total_ : bar_cells_k;
        double const rate = rate_per_second_(completed);
        double const eta_seconds = rate > 0 ? static_cast<double>(total_ - completed) / rate : 0;
        fmt::print("\r\033[K  {} {:█<{}}{:░<{}} {} / {} · {}/s · eta {:.0f} s", label_, "", filled_cells, "",
                   bar_cells_k - filled_cells, humanized(completed), humanized(total_), humanized(rate), eta_seconds);
        std::fflush(stdout);
    }

    void render_decile_(std::size_t completed) noexcept {
        std::size_t const deciles = total_ ? 10 * completed / total_ : 10;
        if (completed != total_ && deciles == reported_deciles_) return;
        reported_deciles_ = deciles;
        fmt::println("  {}: {} / {} · {}/s", label_, humanized(completed), humanized(total_),
                     humanized(rate_per_second_(completed)));
        // A piped stdout is block-buffered: without the flush a decile line can lag its
        // phase by minutes in `tee`-style logs, which exist precisely to watch progress.
        std::fflush(stdout);
    }
};

/** Names the benchmarked dtype subset; NumKong exposes no C-level name table yet. */
inline char const* dtype_name(nk_dtype_t dtype) noexcept {
    switch (dtype) {
    // Sorted by decreasing dynamic range, grouped floats, signed integers, unsigned.
    case nk_f32_k: return "f32";
    case nk_bf16_k: return "bf16";
    case nk_f16_k: return "f16";
    case nk_e5m2_k: return "e5m2";
    case nk_e4m3_k: return "e4m3";
    case nk_e3m2_k: return "e3m2";
    case nk_e2m3_k: return "e2m3";
    case nk_i8_k: return "i8";
    case nk_i4_k: return "i4";
    case nk_u32_k: return "u32";
    case nk_u8_k: return "u8";
    case nk_u4_k: return "u4";
    case nk_u1_k: return "u1";
    default: return "unknown";
    }
}

/** Peak resident set size from @c /proc/self/status in bytes, or zero when unknown. */
inline std::size_t peak_resident_bytes() noexcept {
    std::FILE* status = std::fopen("/proc/self/status", "r");
    if (!status) return 0;
    char line[256];
    std::size_t resident_bytes = 0;
    while (std::fgets(line, sizeof(line), status))
        if (std::strncmp(line, "VmHWM:", 6) == 0) {
            resident_bytes = std::strtoull(line + 6, nullptr, 10) * 1024;
            break;
        }
    std::fclose(status);
    return resident_bytes;
}

} // namespace unum::usearch::bench
