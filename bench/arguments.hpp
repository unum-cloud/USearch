/**
 *  @file bench/arguments.hpp
 *  @author Ash Vardanian
 *  @date September 15, 2026
 *  @brief Command-line flags for every benchmark binary: one table parses the arguments and prints the usage.
 *
 *  Flags are long kebab-case words whose value follows as the next argument or after @c =. Counts are plain
 *  integers, durations read like @c 200ms or @c 10s, seeds are integers or @c random, and lists are
 *  comma-separated. A text value views the tail of its @c argv entry, so it is null-terminated; list items are
 *  not. Copied from UStore's @c tools/arguments.hpp, keeping the destinations these binaries take.
 *
 *  @code
 *  count_t top_k {.value = 10};
 *  std::string_view vectors_path;
 *  std::array<flag_t, 2> const flags {{
 *      {"--vectors", "path", "base vectors, `.fbin` family", &vectors_path},
 *      {"--top-k", "count", "matches per query to judge", &top_k},
 *  }};
 *  parse_or_exit(argc, argv, "usearch_bench_dense_ram_index --vectors <path> [flags]", flags);
 *  @endcode
 */
#pragma once

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint64_t`
#include <cstdlib> // `std::exit`, `std::strtod`

#include <array>        // `std::array`
#include <charconv>     // `std::from_chars`
#include <chrono>       // `std::chrono::milliseconds`
#include <limits>       // `std::numeric_limits`
#include <optional>     // `std::optional`
#include <random>       // `std::random_device`
#include <span>         // `std::span`
#include <string_view>  // `std::string_view`
#include <system_error> // `std::errc`
#include <variant>      // `std::variant`

#include <fmt/format.h>

namespace unum::usearch::bench {

/** Outcome of parsing one value or a whole argument list. */
enum class argument_status_t {

    /** Every argument matched its flag. */
    parsed_k,

    /** `--help` was given, so the caller prints the usage and exits. */
    help_k,

    /** An argument starting with `--` names no flag. */
    unknown_flag_k,

    /** A flag taking a value ended the argument list. */
    missing_value_k,

    /** A switch was given a value through `=`. */
    unexpected_value_k,

    /** The digits do not form a non-negative integer. */
    invalid_number_k,

    /** The value overflows its destination or falls outside its bounds. */
    out_of_range_k,

    /** A list or the positional arguments exceed their capacity. */
    too_many_items_k,

    /** A list holds fewer items than it requires. */
    too_few_items_k,
};

/** A plain non-negative integer, accepted only within @b [minimum,maximum]. */
struct count_t {
    std::size_t value = 0;
    std::size_t minimum = 1;
    std::size_t maximum = std::numeric_limits<std::size_t>::max();
};

/** Comma-separated counts like @c 16,256,4096, each within @b [minimum,maximum].
 *  Between @c minimum_size and @c maximum_size of them are accepted. */
struct counts_t {
    static constexpr std::size_t capacity_k = 16;
    std::array<std::size_t, capacity_k> values {};
    std::size_t size = 0;
    std::size_t minimum = 1;
    std::size_t maximum = std::numeric_limits<std::size_t>::max();
    std::size_t minimum_size = 1;
    std::size_t maximum_size = capacity_k;

    [[nodiscard]] std::span<std::size_t const> view() const noexcept { return {values.data(), size}; }
};

/** Whether a flag without a value appeared. */
enum class switch_t : bool {
    absent_k,
    present_k,
};

/** One flag: its name with both dashes, the usage placeholder, one help line, and the value it writes. */
struct flag_t {
    std::string_view name;
    std::string_view placeholder;
    std::string_view help;
    std::variant<std::string_view*, count_t*, counts_t*, switch_t*, std::chrono::milliseconds*, std::uint32_t*, double*>
        destination;
};

/** Result of @c parse_arguments: the status, the argument and flag it concerns, the value that failed, and how
 *  many positionals were collected. */
struct arguments_t {
    argument_status_t status = argument_status_t::parsed_k;
    std::string_view argument;
    std::string_view value;
    flag_t const* flag = nullptr;
    std::size_t positionals_count = 0;
};

#pragma region Values

[[nodiscard]] inline argument_status_t parse_integer_(std::string_view digits, std::uint64_t& value) noexcept {
    if (digits.empty()) return argument_status_t::invalid_number_k;
    auto const [end, parse_status] = std::from_chars(digits.data(), digits.data() + digits.size(), value);
    if (parse_status == std::errc::result_out_of_range) return argument_status_t::out_of_range_k;
    if (parse_status != std::errc {} || end != digits.data() + digits.size())
        return argument_status_t::invalid_number_k;
    return argument_status_t::parsed_k;
}

[[nodiscard]] inline argument_status_t parse_bounded_(std::string_view text, std::size_t minimum, std::size_t maximum,
                                                      std::size_t& value) noexcept {
    std::uint64_t parsed = 0;
    argument_status_t const status = parse_integer_(text, parsed);
    if (status != argument_status_t::parsed_k) return status;
    if (parsed < minimum || parsed > maximum) return argument_status_t::out_of_range_k;
    value = static_cast<std::size_t>(parsed);
    return argument_status_t::parsed_k;
}

/** Parses a plain integer within the bounds @p count already carries. */
[[nodiscard]] inline argument_status_t parse_count(std::string_view text, count_t& count) noexcept {
    return parse_bounded_(text, count.minimum, count.maximum, count.value);
}

/** Parses comma-separated counts within the bounds @p counts carries, replacing the items it held. */
[[nodiscard]] inline argument_status_t parse_counts(std::string_view text, counts_t& counts) noexcept {
    std::size_t const maximum_size = counts.maximum_size < counts_t::capacity_k ? counts.maximum_size
                                                                                : counts_t::capacity_k;
    counts.size = 0;
    while (true) {
        std::size_t const comma = text.find(',');
        if (counts.size == maximum_size) return argument_status_t::too_many_items_k;
        argument_status_t const status = parse_bounded_(text.substr(0, comma), counts.minimum, counts.maximum,
                                                        counts.values[counts.size]);
        if (status != argument_status_t::parsed_k) return status;
        ++counts.size;
        if (comma == std::string_view::npos) break;
        text.remove_prefix(comma + 1);
    }
    return counts.size < counts.minimum_size ? argument_status_t::too_few_items_k : argument_status_t::parsed_k;
}

/** A positive duration in whole milliseconds or seconds, like "200ms" or "10s". */
inline std::optional<std::chrono::milliseconds> parse_duration(std::string_view text) noexcept {
    std::uint32_t count = 0;
    auto const [end, parse_status] = std::from_chars(text.data(), text.data() + text.size(), count);
    if (parse_status != std::errc {} || !count) return std::nullopt;
    std::string_view const unit = text.substr(end - text.data());
    if (unit == "ms") return std::chrono::milliseconds(count);
    if (unit == "s") return std::chrono::seconds(count);
    return std::nullopt;
}

/** A 32-bit seed, or "random" for a fresh draw from @c std::random_device. */
inline std::optional<std::uint32_t> parse_seed(std::string_view text) noexcept {
    if (text == "random") return static_cast<std::uint32_t>(std::random_device {}());
    std::uint32_t seed = 0;
    auto const [end, parse_status] = std::from_chars(text.data(), text.data() + text.size(), seed);
    if (parse_status != std::errc {} || end != text.data() + text.size()) return std::nullopt;
    return seed;
}

/** A non-negative real, like "0.125" or "1e-3". */
inline std::optional<double> parse_real(std::string_view text) noexcept {
    // Text values view the tail of their `argv` entry, so `strtod` stops at its terminator.
    char* end = nullptr;
    double const real = std::strtod(text.data(), &end);
    if (text.empty() || end != text.data() + text.size() || !(real >= 0)) return std::nullopt;
    return real;
}

#pragma endregion Values

#pragma region Arguments

[[nodiscard]] inline argument_status_t assign_(flag_t const& flag, std::string_view value) noexcept {
    if (auto const* text = std::get_if<std::string_view*>(&flag.destination)) {
        **text = value;
        return argument_status_t::parsed_k;
    }
    if (auto const* count = std::get_if<count_t*>(&flag.destination)) return parse_count(value, **count);
    if (auto const* counts = std::get_if<counts_t*>(&flag.destination)) return parse_counts(value, **counts);
    if (auto const* duration = std::get_if<std::chrono::milliseconds*>(&flag.destination)) {
        std::optional<std::chrono::milliseconds> const parsed = parse_duration(value);
        if (!parsed) return argument_status_t::invalid_number_k;
        **duration = *parsed;
        return argument_status_t::parsed_k;
    }
    if (auto const* seed = std::get_if<std::uint32_t*>(&flag.destination)) {
        std::optional<std::uint32_t> const parsed = parse_seed(value);
        if (!parsed) return argument_status_t::invalid_number_k;
        **seed = *parsed;
        return argument_status_t::parsed_k;
    }
    if (auto const* real = std::get_if<double*>(&flag.destination)) {
        std::optional<double> const parsed = parse_real(value);
        if (!parsed) return argument_status_t::invalid_number_k;
        **real = *parsed;
        return argument_status_t::parsed_k;
    }
    return argument_status_t::unexpected_value_k;
}

/**
 *  @brief Parses @p argv against @p flags, collecting arguments without leading dashes into @p positionals.
 *
 *  Parsing stops at the first failure, leaving earlier destinations written.
 *  @param[in] argc Entries of @p argv, the first of which is the program name and is skipped.
 *  @param[in] argv The arguments, which every text destination and list item goes on viewing.
 *  @param[in] flags Every flag the binary accepts; @c --help is always accepted and never listed.
 *  @param[out] positionals Receives positional arguments in order, viewing into @p argv.
 */
[[nodiscard]] inline arguments_t parse_arguments(int argc, char const* const* argv, std::span<flag_t const> flags,
                                                 std::span<std::string_view> positionals) noexcept {
    arguments_t result;
    for (int index = 1; index < argc; ++index) {
        std::string_view const argument = argv[index];
        result.argument = argument, result.value = {}, result.flag = nullptr;

        if (!argument.starts_with("--")) {
            if (result.positionals_count == positionals.size()) {
                result.status = argument_status_t::too_many_items_k;
                return result;
            }
            positionals[result.positionals_count++] = argument;
            continue;
        }
        if (argument == "--help") {
            result.status = argument_status_t::help_k;
            return result;
        }

        std::size_t const equals = argument.find('=');
        std::string_view const name = argument.substr(0, equals);
        flag_t const* flag = nullptr;
        for (flag_t const& candidate : flags)
            if (candidate.name == name) flag = &candidate;
        if (!flag) {
            result.status = argument_status_t::unknown_flag_k;
            return result;
        }
        result.flag = flag;

        auto const* toggle = std::get_if<switch_t*>(&flag->destination);
        if (toggle && equals != std::string_view::npos) {
            result.value = argument.substr(equals + 1);
            result.status = argument_status_t::unexpected_value_k;
            return result;
        }
        if (toggle) {
            **toggle = switch_t::present_k;
            continue;
        }
        if (equals == std::string_view::npos && index + 1 == argc) {
            result.status = argument_status_t::missing_value_k;
            return result;
        }

        result.value = equals != std::string_view::npos ? argument.substr(equals + 1) : argv[++index];
        result.status = assign_(*flag, result.value);
        if (result.status != argument_status_t::parsed_k) return result;
    }
    result = {.positionals_count = result.positionals_count};
    return result;
}

/** The grammar a value of @p flag parses in, as the error line names it. */
[[nodiscard]] inline std::string_view grammar(flag_t const& flag) noexcept {
    if (auto const* count = std::get_if<count_t*>(&flag.destination))
        return (*count)->minimum ? "a positive count" : "an unsigned integer";
    if (auto const* counts = std::get_if<counts_t*>(&flag.destination))
        return (*counts)->minimum ? "positive counts like 64,128" : "unsigned integers like 0,1";
    if (std::holds_alternative<std::chrono::milliseconds*>(flag.destination)) return "a duration like 200ms or 10s";
    if (std::holds_alternative<std::uint32_t*>(flag.destination)) return "an unsigned integer or random";
    if (std::holds_alternative<double*>(flag.destination)) return "a non-negative real";
    if (std::holds_alternative<switch_t*>(flag.destination)) return "no value";
    return "a value";
}

/** Prints @p synopsis and one aligned line per flag, generated from the same table that parses. */
inline void print_usage(std::string_view synopsis, std::span<flag_t const> flags) {
    constexpr std::size_t column_k = 34;
    fmt::println("Usage: {}\n", synopsis);
    for (flag_t const& flag : flags) {
        std::size_t const width = flag.name.size() + (flag.placeholder.empty() ? 0 : flag.placeholder.size() + 3);
        std::size_t const padding = width < column_k ? column_k - width : 1;
        if (flag.placeholder.empty()) fmt::println("  {}{:{}}{}", flag.name, "", padding, flag.help);
        else fmt::println("  {} <{}>{:{}}{}", flag.name, flag.placeholder, "", padding, flag.help);
    }
    fmt::println("  --help{:{}}print this message", "", column_k - 6);
}

/** Parses @p argv against @p flags. On `--help` prints the usage and exits with status 0; on the first argument that
 *  fails prints one line, like `--top-k="0" does not parse, expected a positive count`, and exits with status 1. */
inline void parse_or_exit(int argc, char const* const* argv, std::string_view synopsis, std::span<flag_t const> flags) {
    arguments_t const parsed = parse_arguments(argc, argv, flags, {});
    if (parsed.status == argument_status_t::parsed_k) return;
    if (parsed.status == argument_status_t::help_k) {
        print_usage(synopsis, flags);
        std::exit(0);
    }
    if (!parsed.flag) fmt::println(stderr, "{} is not a flag, see --help", parsed.argument);
    else
        fmt::println(stderr, "{}=\"{}\" does not parse, expected {}", parsed.flag->name, parsed.value,
                     grammar(*parsed.flag));
    std::exit(1);
}

#pragma endregion Arguments

} // namespace unum::usearch::bench
