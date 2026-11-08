/**
 *  @file tools/arguments.hpp
 *  @author Ash Vardanian
 *  @date October 25, 2026
 *  @brief Command-line flags for every USearch binary: one table parses the arguments, reads their
 *      variables and prints the usage.
 *
 *  Flags are long kebab-case words whose value follows as the next argument or after @c =. Byte
 *  counts carry a binary suffix, durations carry @c ms or @c s, counts are plain integers, and
 *  lists are comma-separated. A flag naming a library enumerator writes that enumerator, parsed by
 *  the C interface's own names. A text value views the tail of its @c argv entry, so it is
 *  null-terminated; list items are not. A value that does not parse prints @c --flag="value"
 *  @c does @c not @c parse, @c expected @c <grammar>.
 *
 *  @code{.cpp}
 *  bytes_t memory_limit;
 *  std::string_view save;
 *  usearch_residency_t residency = usearch_residency_ram_k;
 *  std::array<flag_t, 3> const flags {{
 *      {"--save", "address", "where the index is saved", &save},
 *      {"--memory-limit", "bytes", "bytes a pass holds past the index, like 40G", &memory_limit},
 *      {"--residency", "name", "where an index lives", &residency},
 *  }};
 *  arguments_t const parsed = parse_arguments(argc, argv, flags, {});
 *  @endcode
 */
#pragma once

#include <cctype>  // `std::toupper` - the variable names
#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint32_t`, `std::uint64_t`
#include <cstdio>  // `stderr`
#include <cstdlib> // `std::exit`, `std::strtod`, `std::getenv`

#include <algorithm>    // `std::min`
#include <array>        // `std::array`
#include <charconv>     // `std::from_chars`
#include <chrono>       // `std::chrono::milliseconds`
#include <limits>       // `std::numeric_limits`
#include <optional>     // `std::optional`
#include <random>       // `std::random_device`
#include <regex>        // `std::regex`, `std::regex_search` - the filter
#include <span>         // `std::span`
#include <string>       // `std::string`
#include <string_view>  // `std::string_view`
#include <system_error> // `std::errc`
#include <variant>      // `std::variant`

#include <fmt/format.h>

#include "usearch/usearch.h" // the enumerators flags name and their spellings

namespace unum::usearch::tools {

/** Outcome of parsing one value or a whole argument list. */
enum class argument_status_t {

    /** Every argument matched its flag. */
    parsed_k,

    /** <tt>--help</tt> was given, so the caller prints the usage and exits. */
    help_k,

    /** An argument starting with <tt>--</tt> names no flag. */
    unknown_flag_k,

    /** A flag taking a value ended the argument list. */
    missing_value_k,

    /** A switch was given a value through @c =. */
    unexpected_value_k,

    /** The digits do not form a non-negative integer. */
    invalid_number_k,

    /** The value overflows its destination or falls outside its bounds. */
    out_of_range_k,

    /** A list or the positional arguments exceed their capacity. */
    too_many_items_k,

    /** A list holds fewer items than it requires. */
    too_few_items_k,

    /** The value names no enumerator of its destination. */
    unknown_name_k,
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

/** A byte count: an integer, or one suffixed K, M, G or T for its binary multiple, like 24G. */
struct bytes_t {
    std::size_t value = 0;
};

/** Whether a flag without a value appeared. */
enum class switch_t : bool {
    absent_k,
    present_k,
};

/** Which named checks run: every one for an empty pattern, else those the pattern finds as an
 *  ECMAScript regex, or as a substring when it is no valid regex. */
struct filter_t {
    std::string_view pattern;
    std::optional<std::regex> regex;

    /** Whether @p name passes the filter. */
    bool selects(std::string_view name) const noexcept {
        if (pattern.empty()) return true;
        if (regex) return std::regex_search(name.begin(), name.end(), *regex);
        return name.find(pattern) != std::string_view::npos;
    }
};

/** The filter @p pattern spells, compiled once. */
inline filter_t filter_of(std::string_view pattern) {
    filter_t filter {pattern, std::nullopt};
    if (pattern.empty()) return filter;
#if defined(__cpp_exceptions) && __cpp_exceptions
    try {
        filter.regex.emplace(pattern.begin(), pattern.end());
    }
    catch (std::regex_error const&) {
    }
#else
    filter.regex.emplace(pattern.begin(), pattern.end());
#endif
    return filter;
}

/** One flag: its name with both dashes, the usage placeholder, one help line, and the value it
 *  writes. */
struct flag_t {
    std::string_view name;
    std::string_view placeholder;
    std::string_view help;
    std::variant<std::string_view*, count_t*, counts_t*, switch_t*, std::chrono::milliseconds*, std::uint32_t*, double*,
                 filter_t*, usearch_residency_t*, usearch_cpu_class_t*, usearch_gpu_class_t*, usearch_nvme_class_t*,
                 usearch_traversal_t*, bytes_t*>
        destination;
};

/** Result of @c parse_arguments: the status, the argument and flag it concerns, the value that
 *  failed, and how many positionals were collected. */
struct arguments_t {
    argument_status_t status = argument_status_t::parsed_k;
    std::string_view argument;
    std::string_view value;
    flag_t const* flag = nullptr;
    std::size_t positionals_count = 0;
};

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

/** Parses comma-separated counts within the bounds @p counts carries, replacing the items it
 *  held. */
[[nodiscard]] inline argument_status_t parse_counts(std::string_view text, counts_t& counts) noexcept {
    std::size_t const maximum_size = std::min(counts.maximum_size, counts_t::capacity_k);
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

/** Parses a byte count into @p bytes: digits, then an optional K, M, G or T. */
[[nodiscard]] inline argument_status_t parse_bytes(std::string_view text, bytes_t& bytes) noexcept {
    std::size_t shift = 0;
    if (!text.empty()) switch (text.back()) {
        case 'K': shift = 10; break;
        case 'M': shift = 20; break;
        case 'G': shift = 30; break;
        case 'T': shift = 40; break;
        default: break;
        }
    if (shift) text.remove_suffix(1);
    std::uint64_t parsed = 0;
    if (argument_status_t const status = parse_integer_(text, parsed); status != argument_status_t::parsed_k)
        return status;
    if (parsed > (std::numeric_limits<std::size_t>::max() >> shift)) return argument_status_t::out_of_range_k;
    bytes.value = static_cast<std::size_t>(parsed) << shift;
    return argument_status_t::parsed_k;
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

/** The bytes of @p text as the C interface borrows a path or a name. */
inline usearch_bytes_view_t bytes_view(std::string_view text) noexcept {
    return {reinterpret_cast<usearch_byte_t const*>(text.data()), text.size()};
}

/** Writes the enumerator @p value spells into @p destination, read back by @p from_name and
 *  accepted only when @p name spells it the same way; the unknown enumerator at zero is no value a
 *  flag names, since leaving the flag out derives it. */
template <typename enumeration_type_>
[[nodiscard]] argument_status_t assign_named_(std::string_view value, enumeration_type_& destination,
                                              enumeration_type_ (*from_name)(usearch_bytes_view_t),
                                              char const* (*name)(enumeration_type_)) noexcept {
    enumeration_type_ const parsed = from_name(bytes_view(value));
    if (parsed == enumeration_type_ {} || value != name(parsed)) return argument_status_t::unknown_name_k;
    destination = parsed;
    return argument_status_t::parsed_k;
}

[[nodiscard]] inline argument_status_t assign_(flag_t const& flag, std::string_view value) noexcept {
    if (std::string_view* const* text = std::get_if<std::string_view*>(&flag.destination)) {
        **text = value;
        return argument_status_t::parsed_k;
    }
    if (count_t* const* count = std::get_if<count_t*>(&flag.destination)) return parse_count(value, **count);
    if (counts_t* const* counts = std::get_if<counts_t*>(&flag.destination)) return parse_counts(value, **counts);
    if (std::chrono::milliseconds* const* duration = std::get_if<std::chrono::milliseconds*>(&flag.destination)) {
        std::optional<std::chrono::milliseconds> const parsed = parse_duration(value);
        if (!parsed) return argument_status_t::invalid_number_k;
        **duration = *parsed;
        return argument_status_t::parsed_k;
    }
    if (std::uint32_t* const* seed = std::get_if<std::uint32_t*>(&flag.destination)) {
        std::optional<std::uint32_t> const parsed = parse_seed(value);
        if (!parsed) return argument_status_t::invalid_number_k;
        **seed = *parsed;
        return argument_status_t::parsed_k;
    }
    if (double* const* real = std::get_if<double*>(&flag.destination)) {
        std::optional<double> const parsed = parse_real(value);
        if (!parsed) return argument_status_t::invalid_number_k;
        **real = *parsed;
        return argument_status_t::parsed_k;
    }
    if (filter_t* const* filter = std::get_if<filter_t*>(&flag.destination)) {
        **filter = filter_of(value);
        return argument_status_t::parsed_k;
    }
    if (usearch_residency_t* const* residency = std::get_if<usearch_residency_t*>(&flag.destination))
        return assign_named_(value, **residency, usearch_residency_from_name, usearch_residency_name);
    if (usearch_cpu_class_t* const* cpu_class = std::get_if<usearch_cpu_class_t*>(&flag.destination))
        return assign_named_(value, **cpu_class, usearch_cpu_class_from_name, usearch_cpu_class_name);
    if (usearch_gpu_class_t* const* gpu_class = std::get_if<usearch_gpu_class_t*>(&flag.destination))
        return assign_named_(value, **gpu_class, usearch_gpu_class_from_name, usearch_gpu_class_name);
    if (usearch_nvme_class_t* const* nvme_class = std::get_if<usearch_nvme_class_t*>(&flag.destination))
        return assign_named_(value, **nvme_class, usearch_nvme_class_from_name, usearch_nvme_class_name);
    if (usearch_traversal_t* const* traversal = std::get_if<usearch_traversal_t*>(&flag.destination))
        return assign_named_(value, **traversal, usearch_traversal_from_name, usearch_traversal_name);
    if (bytes_t* const* bytes = std::get_if<bytes_t*>(&flag.destination)) return parse_bytes(value, **bytes);
    return argument_status_t::unexpected_value_k;
}

/**
 *  @brief Parses @p argv against @p flags, collecting undashed arguments into @p positionals.
 *
 *  Parsing stops at the first failure, leaving earlier destinations written.
 *
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

        switch_t* const* toggle = std::get_if<switch_t*>(&flag->destination);
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
    if (count_t* const* count = std::get_if<count_t*>(&flag.destination))
        return (*count)->minimum ? "a positive count" : "an unsigned integer";
    if (counts_t* const* counts = std::get_if<counts_t*>(&flag.destination))
        return (*counts)->minimum ? "positive counts like 64,128" : "unsigned integers like 0,1";
    if (std::holds_alternative<std::chrono::milliseconds*>(flag.destination)) return "a duration like 200ms or 10s";
    if (std::holds_alternative<std::uint32_t*>(flag.destination)) return "an unsigned integer or random";
    if (std::holds_alternative<double*>(flag.destination)) return "a non-negative real";
    if (std::holds_alternative<filter_t*>(flag.destination)) return "an ECMAScript regex or a substring";
    if (std::holds_alternative<usearch_residency_t*>(flag.destination)) return "ram, vram or nvme";
    if (std::holds_alternative<usearch_cpu_class_t*>(flag.destination))
        return "a CPU class like intel-icelake or serial";
    if (std::holds_alternative<usearch_gpu_class_t*>(flag.destination)) return "a GPU class like nvidia-ampere";
    if (std::holds_alternative<usearch_nvme_class_t*>(flag.destination))
        return "an NVMe class like posix or linux-uring";
    if (std::holds_alternative<usearch_traversal_t*>(flag.destination)) return "walk or scan";
    if (std::holds_alternative<switch_t*>(flag.destination)) return "no value";
    if (std::holds_alternative<bytes_t*>(flag.destination)) return "bytes like 4096 or 24G";
    return "a value";
}

/** The variable that sets @p flag's default, the name RetriEval reads too: <tt>--top-k</tt> reads
 *  @c RETRIEVAL_TOP_K. */
[[nodiscard]] inline std::string retrieval_variable(flag_t const& flag) {
    std::string variable = "RETRIEVAL_";
    for (char const letter : flag.name.substr(2))
        variable += letter == '-' ? '_' : static_cast<char>(std::toupper(static_cast<unsigned char>(letter)));
    return variable;
}

/** Writes @p value, a variable's text, into @p flag: a switch takes 1, 0, true or false, the rest
 *  parse as on the command line. */
[[nodiscard]] inline argument_status_t assign_variable_(flag_t const& flag, std::string_view value) noexcept {
    switch_t* const* toggle = std::get_if<switch_t*>(&flag.destination);
    if (!toggle) return assign_(flag, value);
    bool const present = value == "1" || value == "true";
    if (!present && value != "0" && value != "false") return argument_status_t::unexpected_value_k;
    **toggle = present ? switch_t::present_k : switch_t::absent_k;
    return argument_status_t::parsed_k;
}

/** Prints @p synopsis and one aligned line per flag, generated from the same table that parses. */
inline void print_usage(std::string_view synopsis, std::span<flag_t const> flags) {
    constexpr std::size_t column_k = 34;
    fmt::println("Usage: {}\n", synopsis);
    for (flag_t const& flag : flags) {
        std::size_t const width = flag.name.size() + (flag.placeholder.empty() ? 0 : flag.placeholder.size() + 3);
        std::size_t const padding = width < column_k ? column_k - width : 1;
        if (flag.placeholder.empty())
            fmt::println("  {}{:{}}{} [env: {}]", flag.name, "", padding, flag.help, retrieval_variable(flag));
        else
            fmt::println("  {} <{}>{:{}}{} [env: {}]", flag.name, flag.placeholder, "", padding, flag.help,
                         retrieval_variable(flag));
    }
    fmt::println("  --help{:{}}print this message", "", column_k - 6);
}

/** Parses @p argv against @p flags, each defaulting to its @ref retrieval_variable when set, so one
 *  environment drives these binaries and RetriEval alike. On <tt>--help</tt> prints the usage and exits
 *  with status 0; on the first variable or argument that fails prints one line, like
 *  <tt>--top-k="0" does not parse, expected a positive count</tt>, and exits with status 1. */
inline void parse_or_exit(int argc, char const* const* argv, std::string_view synopsis, std::span<flag_t const> flags) {
    for (flag_t const& flag : flags) {
        std::string const variable = retrieval_variable(flag);
        char const* const value = std::getenv(variable.c_str());
        if (!value || !*value || assign_variable_(flag, value) == argument_status_t::parsed_k) continue;
        bool const toggle = std::holds_alternative<switch_t*>(flag.destination);
        fmt::println(stderr, "{}=\"{}\" does not parse, expected {}", variable, value,
                     toggle ? "1, 0, true or false" : grammar(flag));
        std::exit(1);
    }
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

} // namespace unum::usearch::tools
