/**
 *  @file tools/abi.hpp
 *  @author Ash Vardanian
 *  @date October 25, 2026
 *  @brief What the suites, benchmarks and tools share over the C interface: flags and settings,
 *      keys minted from rows, sessions of executors and arenas, owned index handles, checkpoint
 *      round trips, mapped datasets, and recall against an exact scan.
 *
 *  Every helper reaches the library through its C verbs alone, and reads GPU outputs back through
 *  the vendor's runtime, so a suite or a benchmark built on this header exercises exactly what a
 *  binding does. Rows travel as opaque version-4 UUIDs minted from the seed by @ref key_of_row, and
 *  come back through @ref row_of_key, which refuses a key it never minted.
 *
 *  Settings come from the environment first and the command line second:
 *
 *  @verbatim
 *  Variable            Default          Meaning
 *  USEARCH_SEED        42               Seed every scenario's data derives from, an unsigned integer or random
 *  USEARCH_FILTER      none             Checks to run, by name, as an ECMAScript regex or else as a substring
 *  USEARCH_DEVICES     first of each    Comma-separated USearch device ordinals the GPU work runs on
 *  USEARCH_NVME_PATHS  none             Comma-separated files, block devices or transport IDs NVMe serves from
 *  --filter            none             Checks to run, overriding USEARCH_FILTER
 *  --residency         ram              Where an index lives: ram, vram or nvme
 *  --cpu-class         derived          The CPU class to dispatch as, like intel-icelake or serial
 *  --gpu-class         derived          The GPU class to dispatch as, like nvidia-ampere
 *  --nvme-class        derived          The NVMe class to submit through, like posix, linux-uring or linux-spdk
 *  --threads           derived          Worker threads of a RAM or NVMe executor
 *  --shards            one per domain   Shards each index opens on, over the memory domains or the devices
 *  --shard-capacity    unbounded        Entries one shard holds before another shard generation opens
 *  --checkpoint        none             Path to save a built index under and reopen it from
 *  @endverbatim
 *
 *  Every flag also reads its default from a @c RETRIEVAL_ variable named after it, the one
 *  RetriEval reads, so @c --top-k defaults to @c RETRIEVAL_TOP_K and the command line overrides it.
 */
#pragma once

#include <cctype>  // `std::toupper` - the variable names
#include <cmath>   // `std::fabs`, `std::log2` - the score tolerance, the ranking discounts
#include <cstddef> // `std::size_t`
#include <cstdint> // `std::int32_t`, `std::uint32_t`
#include <cstdio>  // `std::remove`, `stdout` - the round-tripped checkpoints, the terminal probe
#include <cstdlib> // `std::exit`, `std::strtod`, `std::getenv`
#include <cstring> // `std::memcpy`, `std::memcmp` - the counters, the keys

#if !USEARCH_OS_WINDOWS_
#include <unistd.h> // `isatty`
#endif

#include <algorithm>       // `std::min`, `std::max`
#include <array>           // `std::array`
#include <bit>             // `std::bit_cast` - keys between the C struct and `uuid_t`
#include <charconv>        // `std::from_chars`
#include <chrono>          // `std::chrono::milliseconds`, `std::chrono::steady_clock`
#include <filesystem>      // `std::filesystem::temp_directory_path` - where an NVMe benchmark saves
#include <limits>          // `std::numeric_limits`
#include <optional>        // `std::optional`
#include <random>          // `std::random_device`
#include <regex>           // `std::regex`, `std::regex_search` - the filter
#include <source_location> // `std::source_location` - where a check failed
#include <span>            // `std::span`
#include <string>          // `std::string`
#include <string_view>     // `std::string_view`
#include <system_error>    // `std::errc`
#include <utility>         // `std::exchange`, `std::move`
#include <variant>         // `std::variant`
#include <vector>          // `std::vector`

#include <fmt/format.h>
#include <fmt/ranges.h> // `fmt::join` - the devices, the shard sizes

#include <forkunion.h> // `fu_topology_new`, `fu_memory_domain_id_at_index` - the domains shards repeat over

#include "usearch/usearch.h"
#include "usearch/utilities/blocking_files.hpp" // `mapped_file_t` - the dataset parts, the compared checkpoints
#include "usearch/utilities/types.hpp"          // `uuid_t`, `u64_t`
#include "usearch/utilities/vram_device.hpp"    // `cuda_device_of`, `hip_device_of` - device outputs read back

namespace unum::usearch::tools {
using namespace unum::usearch::utilities;

#pragma region Flags

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
                 filter_t*, usearch_residency_t*, usearch_cpu_class_t*, usearch_gpu_class_t*, usearch_nvme_class_t*>
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
    return argument_status_t::unexpected_value_k;
}

/**
 *  @brief Parses @p argv against @p flags, collecting arguments without leading dashes into
 *      @p positionals.
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
    if (std::holds_alternative<switch_t*>(flag.destination)) return "no value";
    return "a value";
}

/** The variable that sets @p flag's default, the name RetriEval reads too: `--top-k` reads
 *  `RETRIEVAL_TOP_K`. */
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
 *  environment drives these binaries and RetriEval alike. On `--help` prints the usage and exits
 *  with status 0; on the first variable or argument that fails prints one line, like
 *  `--top-k="0" does not parse, expected a positive count`, and exits with status 1. */
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

/** Rows below this bound get keys: a version-4 UUID leaves 62 bits beside its variant. */
inline constexpr u64_t key_rows_k = u64_t {1} << 62;

/** The odd multipliers of the row mix, SplitMix64's. */
inline constexpr u64_t key_multipliers_k[2] = {0xbf58476d1ce4e5b9ull, 0x94d049bb133111ebull};

/** The inverse of the odd @p multiplier modulo 2^64, by Newton's iteration. */
constexpr u64_t inverse_of_(u64_t multiplier) noexcept {
    u64_t inverse = multiplier;
    for (int step = 0; step != 5; ++step) inverse *= 2 - multiplier * inverse;
    return inverse;
}

/** SplitMix64's finalizer. */
constexpr u64_t mix_(u64_t word) noexcept {
    word = (word ^ word >> 30) * key_multipliers_k[0];
    word = (word ^ word >> 27) * key_multipliers_k[1];
    return word ^ word >> 31;
}

/** The leading eight bytes of a key whose trailing eight are @p trailing : a version-4 tag over a
 *  mix of those and @p seed, which @ref row_of_key checks. */
constexpr u64_t leading_word_(u64_t seed, u64_t trailing) noexcept {
    return (mix_(trailing ^ mix_(seed)) & ~u64_t {0xF000}) | 0x4000;
}

/**
 *  @brief The key the harness mints for @p row under @p seed : a random-looking version-4 UUID with
 *      no table behind it.
 *
 *  The trailing eight bytes hold the variant over an invertible mix of the row within 62 bits, and
 *  the leading eight a mix of them, so @ref row_of_key inverts the first and checks the second.
 */
inline usearch_key_t key_of_row(u64_t seed, u64_t row) noexcept {
    u64_t mixed = (row ^ mix_(seed)) & (key_rows_k - 1);
    for (u64_t const multiplier : key_multipliers_k) mixed = (mixed ^ mixed >> 31) * multiplier & (key_rows_k - 1);
    u64_t const trailing = u64_t {2} << 62 | (mixed ^ mixed >> 31);
    u64_t const leading = leading_word_(seed, trailing);
    usearch_key_t key;
    for (std::size_t byte = 0; byte != 8; ++byte) {
        key.bytes[7 - byte] = static_cast<usearch_byte_t>(leading >> (byte * 8));
        key.bytes[15 - byte] = static_cast<usearch_byte_t>(trailing >> (byte * 8));
    }
    return key;
}

/** The row @ref key_of_row minted @p key for under @p seed , or none for a key it never minted. */
inline std::optional<u64_t> row_of_key(u64_t seed, usearch_key_t const& key) noexcept {
    u64_t leading = 0, trailing = 0;
    for (std::size_t byte = 0; byte != 8; ++byte)
        leading = leading << 8 | key.bytes[byte], trailing = trailing << 8 | key.bytes[8 + byte];
    if (trailing >> 62 != 2 || leading != leading_word_(seed, trailing)) return std::nullopt;
    u64_t mixed = trailing & (key_rows_k - 1);
    mixed ^= mixed >> 31;
    for (std::size_t round = std::size(key_multipliers_k); round--;) {
        mixed = mixed * inverse_of_(key_multipliers_k[round]) & (key_rows_k - 1);
        mixed ^= mixed >> 31;
    }
    return (mixed ^ mix_(seed)) & (key_rows_k - 1);
}

/** The packed run of @p keys as a batch names its identities. */
inline usearch_keys_view_t keys_view(std::span<usearch_key_t const> keys) noexcept {
    return {keys.data(), sizeof(usearch_key_t)};
}

#pragma endregion Flags

#pragma region Settings

/** The text of the environment variable @p name, or nothing when it is unset or empty. */
inline std::optional<std::string_view> environment_text(char const* name) noexcept {
    char const* const text = std::getenv(name);
    if (!text || !*text) return std::nullopt;
    return std::string_view(text);
}

/** Exits with status 1 after printing `NAME="text" does not parse, expected <expected>`. */
[[noreturn]] inline void exit_unparsed(std::string_view name, std::string_view text, std::string_view expected) {
    fmt::println(stderr, "{}=\"{}\" does not parse, expected {}", name, text, expected);
    std::exit(1);
}

/** Comma-separated items of @p text, each non-empty, or none when an item is empty. */
inline std::optional<std::vector<std::string_view>> split_items(std::string_view text) {
    std::vector<std::string_view> items;
    for (std::size_t begin = 0; begin <= text.size();) {
        std::size_t const end = std::min(text.find(',', begin), text.size());
        if (end == begin) return std::nullopt;
        items.push_back(text.substr(begin, end - begin));
        begin = end + 1;
    }
    return items;
}

/** The USearch device ordinals @c USEARCH_DEVICES lists, comma-separated, as positions in
 *  @c usearch_gpu_devices; none when it is unset or empty, and exits with status 1 on text that
 *  does not parse. */
inline std::vector<std::int32_t> devices_from_environment() {
    std::vector<std::int32_t> devices;
    std::optional<std::string_view> const text = environment_text("USEARCH_DEVICES");
    if (!text) return devices;
    std::optional<std::vector<std::string_view>> const items = split_items(*text);
    if (!items) exit_unparsed("USEARCH_DEVICES", *text, "unsigned integers like 0,1");
    for (std::string_view const item : *items) {
        std::uint32_t ordinal = 0;
        auto const [end, parse_status] = std::from_chars(item.data(), item.data() + item.size(), ordinal);
        if (parse_status != std::errc {} || end != item.data() + item.size() ||
            ordinal > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()))
            exit_unparsed("USEARCH_DEVICES", *text, "unsigned integers like 0,1");
        devices.push_back(static_cast<std::int32_t>(ordinal));
    }
    return devices;
}

/**
 *  @brief Every setting the suites, the benchmarks and the tools share, its default as the
 *      initializer, filled once by @ref read_settings.
 *
 *  The flags a binary reads beyond these live in its own settings, parsed in the same pass.
 */
struct settings_t {

    /** Seed every scenario family and every minted key derives from. Override: @c USEARCH_SEED. */
    std::uint32_t seed = 42;

    /** Checks to run. Override: @c USEARCH_FILTER. */
    filter_t filter;

    /** Where an index lives. */
    usearch_residency_t residency = usearch_residency_ram_k;

    /** USearch device ordinals the GPU work runs on; none takes device zero, or every vendor's
     *  first in a suite. Override: @c USEARCH_DEVICES. */
    std::vector<std::int32_t> devices;

    /** NVMe namespaces to serve from, in the grammar of @c usearch_nvme_options_t::paths; none
     *  serves every shard where its checkpoint points. Override: @c USEARCH_NVME_PATHS. */
    std::vector<std::string_view> nvme_paths;

    /** The CPU class to dispatch as; unknown derives the CPU's own. */
    usearch_cpu_class_t cpu_class = usearch_cpu_class_unknown_k;

    /** The GPU class to dispatch as; unknown derives the devices'. */
    usearch_gpu_class_t gpu_class = usearch_gpu_class_unknown_k;

    /** The NVMe class to submit through; unknown derives it. */
    usearch_nvme_class_t nvme_class = usearch_nvme_class_unknown_k;

    /** Worker threads of a RAM or NVMe executor; absent derives from the domains or devices. */
    count_t threads {.minimum = 0};

    /** Shards each index opens on, repeating the domains or devices named; absent takes one per
     *  memory domain on RAM and one per device on VRAM. */
    count_t shards {.minimum = 0};

    /** Entries one shard holds before another shard generation opens; absent bounds shards by the
     *  32-bit ID space alone. */
    count_t shard_capacity {.minimum = 0};

    /** Path to save a built index under and reopen it from in its residency; empty keeps it. */
    std::string_view checkpoint;

    /** Whether stdout is a terminal, so progress redraws one line instead of printing deciles. */
    bool terminal = false;
};

/** The flags every binary reading a @ref settings_t takes. */
inline std::array<flag_t, 9> settings_flags(settings_t& settings) noexcept {
    return {{
        {"--filter", "pattern", "checks to run, as a regex or a substring; every one by default", &settings.filter},
        {"--residency", "name", "where an index lives: ram, vram or nvme; ram by default", &settings.residency},
        {"--cpu-class", "name", "CPU class to dispatch as, like intel-icelake or serial; derived by default",
         &settings.cpu_class},
        {"--gpu-class", "name", "GPU class to dispatch as, like nvidia-ampere; derived by default",
         &settings.gpu_class},
        {"--nvme-class", "name",
         "NVMe class to submit through, like posix, linux-uring or linux-spdk; derived by default",
         &settings.nvme_class},
        {"--threads", "count", "worker threads; derived from the domains or devices by default", &settings.threads},
        {"--shards", "count", "shards per index; one per memory domain or device by default", &settings.shards},
        {"--shard-capacity", "entries",
         "entries one shard holds before another shard generation opens; unbounded by default",
         &settings.shard_capacity},
        {"--checkpoint", "path", "path to save the built index under and reopen it from", &settings.checkpoint},
    }};
}

/** Reads the settings the environment holds, exiting with status 1 on the first that does not
 *  parse; calls nothing of the library, so header-only suites read it too. */
inline settings_t read_environment() {
    settings_t settings;
    settings.filter = filter_of(environment_text("USEARCH_FILTER").value_or(""));
    if (std::optional<std::string_view> const seed_text = environment_text("USEARCH_SEED")) {
        std::optional<std::uint32_t> const seed = parse_seed(*seed_text);
        if (!seed) exit_unparsed("USEARCH_SEED", *seed_text, "an unsigned integer or random");
        settings.seed = *seed;
    }
    settings.devices = devices_from_environment();
    if (std::optional<std::string_view> const paths_text = environment_text("USEARCH_NVME_PATHS")) {
        std::optional<std::vector<std::string_view>> paths = split_items(*paths_text);
        if (!paths)
            exit_unparsed("USEARCH_NVME_PATHS", *paths_text, "comma-separated files, block devices or transport IDs");
        settings.nvme_paths = *std::move(paths);
    }
#if !USEARCH_OS_WINDOWS_
    settings.terminal = isatty(fileno(stdout)) != 0;
#endif
    return settings;
}

/** Reads the environment, then parses @p argv against the shared flags and @p binary_flags. On
 *  `--help` prints the usage and exits with status 0; on a setting that does not parse prints one
 *  line and exits with status 1. */
inline settings_t read_settings(int argc, char const* const* argv, std::string_view synopsis,
                                std::span<flag_t const> binary_flags = {}) {
    settings_t settings = read_environment();
    std::vector<flag_t> flags(binary_flags.begin(), binary_flags.end());
    std::array<flag_t, 9> const shared = settings_flags(settings);
    flags.insert(flags.end(), shared.begin(), shared.end());
    parse_or_exit(argc, argv, synopsis, flags);
    return settings;
}

/** Prints each setting @ref read_environment reads as "- Name: value", in the grammar it parses
 *  from; calls nothing of the library, so header-only suites print it too. */
inline void print_environment(settings_t const& settings) {
    fmt::println("- Seed: {}", settings.seed);
    fmt::println("- Filter: {}", settings.filter.pattern.empty() ? "none" : settings.filter.pattern);
    if (settings.devices.empty()) fmt::println("- Devices: derived");
    else fmt::println("- Devices: {}", fmt::join(settings.devices, ","));
    if (settings.nvme_paths.empty()) fmt::println("- NVMe paths: none");
    else fmt::println("- NVMe paths: {}", fmt::join(settings.nvme_paths, ","));
}

/** Prints each setting as "- Name: value", in the grammar it parses from. */
inline void print(settings_t const& settings) {
    print_environment(settings);
    fmt::println("- Residency: {}", usearch_residency_name(settings.residency));
    fmt::println("- CPU class: {}", usearch_cpu_class_name(settings.cpu_class));
    fmt::println("- GPU class: {}", usearch_gpu_class_name(settings.gpu_class));
    fmt::println("- NVMe class: {}", usearch_nvme_class_name(settings.nvme_class));
}

#pragma endregion Settings

#pragma region Sessions

/** Prints a failed call's status and message after @p verb; true when it failed. */
inline bool report_failure(usearch_outcome_t const& outcome, std::string_view verb) {
    if (outcome.status == usearch_status_success_k) return false;
    fmt::println(stderr, "{} failed with {}: {}", verb, usearch_status_name(outcome.status),
                 outcome.message ? outcome.message : "no message");
    return true;
}

/** Every GPU this process drives, in USearch ordinal order. */
inline std::vector<usearch_gpu_device_t> listed_devices() {
    usearch_outcome_t outcome {};
    std::size_t count = 0;
    usearch_gpu_devices_t counting {.outcome = &outcome,
                                    .devices = nullptr,
                                    .devices_capacity = 0,
                                    .devices_count = &count};
    usearch_gpu_devices(&counting);
    std::vector<usearch_gpu_device_t> devices(count);
    usearch_gpu_devices_t listing {.outcome = &outcome,
                                   .devices = devices.data(),
                                   .devices_capacity = count,
                                   .devices_count = &count};
    usearch_gpu_devices(&listing);
    devices.resize(std::min(count, devices.size()));
    return devices;
}

/**
 *  @brief The executors and the arena one residency's calls run on, freed together.
 *
 *  A RAM session builds and serves on one RAM executor. A VRAM session builds and serves on the
 *  executor of its devices, landing outputs in an arena on the first of them. An NVMe session
 *  builds on a RAM executor and serves views on an NVMe executor of its own, so a built index is
 *  saved and viewed before it is searched there.
 */
struct session_t {
    usearch_residency_t residency = usearch_residency_ram_k;

    /** The seed every key this session's entries take is minted under, by @ref key_of_row. */
    std::uint32_t seed = 0;

    /** The class every index this session opens dispatches as. */
    usearch_cpu_class_t cpu_class = usearch_cpu_class_unknown_k;
    usearch_gpu_class_t gpu_class = usearch_gpu_class_unknown_k;

    /** The USearch device ordinals a VRAM index places shards on, in turn. */
    std::vector<std::int32_t> devices;

    /** The NVMe namespaces views serve from, borrowed from the settings. */
    std::vector<usearch_bytes_view_t> paths;

    /** The memory domains RAM shards repeat over, in turn. */
    std::vector<std::int32_t> memory_domains;

    /** Entries one shard holds before another shard generation opens; zero leaves it unbounded. */
    std::size_t shard_capacity = 0;

    /** The executor building every index: RAM threads, or the devices' streams. */
    usearch_executor_t executor = nullptr;

    /** The NVMe executor views serve on; null outside an NVMe session. */
    usearch_executor_t storage = nullptr;

    /** Where every call's outputs land: host pages, or the first device of a VRAM session. */
    usearch_arena_t arena = nullptr;

    session_t() = default;
    session_t(session_t const&) = delete;
    session_t& operator=(session_t const&) = delete;
    session_t(session_t&& other) noexcept { swap(other); }
    session_t& operator=(session_t&& other) noexcept { return swap(other), *this; }
    ~session_t() noexcept {
        usearch_arena_free(arena);
        usearch_executor_free(storage);
        usearch_executor_free(executor);
    }

    void swap(session_t& other) noexcept {
        std::swap(residency, other.residency), std::swap(seed, other.seed), std::swap(cpu_class, other.cpu_class);
        std::swap(gpu_class, other.gpu_class), std::swap(devices, other.devices);
        std::swap(paths, other.paths), std::swap(memory_domains, other.memory_domains);
        std::swap(shard_capacity, other.shard_capacity), std::swap(executor, other.executor);
        std::swap(storage, other.storage), std::swap(arena, other.arena);
    }

    /** Whether every executor and the arena the residency needs opened. */
    explicit operator bool() const noexcept {
        return executor && arena && (residency != usearch_residency_nvme_k || storage);
    }

    /** The executor serving an index of @p index_residency : the storage one for a view, and none
     *  for a RAM copy of a VRAM session's index, which runs on the calling thread. */
    usearch_executor_t executor_for(usearch_residency_t index_residency) const noexcept {
        if (index_residency == usearch_residency_nvme_k) return storage;
        return index_residency == usearch_residency_ram_k && residency == usearch_residency_vram_k ? nullptr : executor;
    }

    /** The RAM placement of @p shards shards over the session's memory domains in turn, backed by
     *  @p domains; zero shards names no domains, so the library derives one shard per domain. */
    usearch_ram_options_t ram_options(std::size_t shards, std::vector<std::int32_t>& domains) const {
        domains.resize(shards);
        for (std::size_t shard = 0; shard != shards; ++shard)
            domains[shard] = memory_domains[shard % memory_domains.size()];
        return {domains.empty() ? nullptr : domains.data(), domains.size(), shard_capacity, cpu_class};
    }

    /** The VRAM placement of @p shards shards over the session's devices in turn, backed by
     *  @p ordinals; zero shards takes one per device. */
    usearch_vram_options_t vram_options(std::size_t shards, std::vector<std::int32_t>& ordinals) const {
        std::size_t const count = shards ? shards : devices.size();
        ordinals.resize(count);
        for (std::size_t shard = 0; shard != count; ++shard) ordinals[shard] = devices[shard % devices.size()];
        return {ordinals.data(), ordinals.size(), shard_capacity, gpu_class};
    }

    /** The NVMe placement views serve from. */
    usearch_nvme_options_t nvme_options() const noexcept {
        return {paths.empty() ? nullptr : paths.data(), paths.size(), cpu_class};
    }
};

/** Opens the executors and the arena of @p settings ' residency, every index of the session
 *  dispatching as @p pinned on the CPU; prints what failed and answers a session that tests
 *  false. */
inline session_t open_session(settings_t const& settings, usearch_cpu_class_t pinned) {
    session_t session;
    session.residency = settings.residency;
    session.seed = settings.seed;
    session.cpu_class = pinned;
    session.gpu_class = settings.gpu_class;
    session.shard_capacity = settings.shard_capacity.value;
    fu_topology_t topology = nullptr;
    std::size_t domains_count = 0;
    if (fu_topology_new(&topology) == fu_success_k && fu_memory_domains_count(topology, &domains_count) == fu_success_k)
        for (std::size_t index = 0; index != domains_count; ++index) {
            fu_memory_domain_id_t domain = 0;
            if (fu_memory_domain_id_at_index(topology, index, &domain) == fu_success_k)
                session.memory_domains.push_back(domain);
        }
    if (topology) fu_topology_delete(topology);
    if (session.memory_domains.empty()) session.memory_domains.push_back(0);
    for (std::string_view const path : settings.nvme_paths) session.paths.push_back(bytes_view(path));
    usearch_outcome_t outcome {};
    if (session.residency == usearch_residency_vram_k) {
        std::vector<usearch_gpu_device_t> const listed = listed_devices();
        session.devices = settings.devices.empty() ? std::vector<std::int32_t> {0} : settings.devices;
        for (std::int32_t const ordinal : session.devices)
            if (static_cast<std::size_t>(ordinal) >= listed.size()) {
                fmt::println(stderr, "Device {} is past the {} listed", ordinal, listed.size());
                return session;
            }
        std::vector<std::int32_t> ordinals;
        usearch_executor_for_vram_t placement {.executor = &session.executor,
                                               .outcome = &outcome,
                                               .vram = session.vram_options(0, ordinals)};
        usearch_executor_for_vram(&placement);
    }
    else {
        usearch_executor_for_ram_t placement {.executor = &session.executor,
                                              .outcome = &outcome,
                                              .ram = {nullptr, 0, session.shard_capacity, pinned},
                                              .threads = settings.threads.value};
        usearch_executor_for_ram(&placement);
    }
    if (report_failure(outcome, "Opening the executor")) return session;
    if (session.residency == usearch_residency_nvme_k) {
        usearch_executor_for_nvme_t placement {.executor = &session.storage,
                                               .outcome = &outcome,
                                               .nvme = session.nvme_options(),
                                               .threads = settings.threads.value,
                                               .deferred_limit = 0,
                                               .nvme_class = settings.nvme_class};
        usearch_executor_for_nvme(&placement);
        if (report_failure(outcome, "Opening the NVMe executor")) return session;
    }
    usearch_arena_init_t arena {
        .arena = &session.arena,
        .executor = session.residency == usearch_residency_vram_k ? session.executor : nullptr,
        .outcome = &outcome,
    };
    usearch_arena_init(&arena);
    report_failure(outcome, "Opening the arena");
    return session;
}

/** Every vendor's driver table and Apple's contexts, loaded once by the first harness step that
 *  touches a device. */
inline gpu_drivers_t& gpu_drivers() noexcept {
    static gpu_drivers_t drivers;
    return drivers;
}

/** Copies @p bytes of a call's output at @p source into @p destination: through the runtime of the
 *  vendor whose device memory holds it, else directly, as host memory, pinned or not, reads; false
 *  when the copy failed. */
inline bool download(void* destination, void const* source, std::size_t bytes) noexcept {
    if (!bytes) return true;
    gpu_drivers_t const& drivers = gpu_drivers();
    if (cuda_device_of(drivers.cuda, source) >= 0)
        return cuda_copy(drivers.cuda, destination, source, bytes) == cuda_success_k;
    if (hip_device_of(drivers.hip, source) >= 0)
        return drivers.hip.copy(destination, source, bytes, hip_copy_device_to_host_k) == hip_success_k;
    std::memcpy(destination, source, bytes);
    return true;
}

#pragma endregion Sessions

#pragma region Verbs

/** The C functions of the dense modality, behind the signatures every modality shares. */
struct dense_verbs_t {
    using index_t = usearch_dense_index_t;
    using description_t = usearch_dense_index_description_t;
    using geometry_t = usearch_dense_metadata_t;
    using inputs_t = usearch_rows_view_t;
    using queries_t = usearch_rows_view_t;
    static constexpr std::string_view name_k = "dense";

    static inputs_t slice(inputs_t rows, std::size_t first) noexcept {
        rows.data = static_cast<usearch_byte_t const*>(rows.data) + first * rows.stride;
        return rows;
    }
    static void init_ram(index_t* index, geometry_t const& geometry, usearch_ram_options_t ram,
                         usearch_outcome_t* outcome) noexcept {
        usearch_dense_init_ram_t arguments {
            index, outcome, geometry.metric, geometry.quantization, geometry.dimensions, geometry.connectivity, 0, ram};
        usearch_dense_init_ram(&arguments);
    }
    static void init_vram(index_t* index, geometry_t const& geometry, usearch_vram_options_t vram,
                          usearch_outcome_t* outcome) noexcept {
        usearch_dense_init_vram_t arguments {
            index, outcome, geometry.metric, geometry.quantization, geometry.dimensions, geometry.connectivity,
            0,     vram};
        usearch_dense_init_vram(&arguments);
    }
    static void insert(index_t index, usearch_executor_t executor, std::size_t count, usearch_keys_view_t keys,
                       inputs_t rows, std::size_t expansion, usearch_traversal_stats_t* stats,
                       usearch_outcome_t* outcome) noexcept {
        usearch_dense_insert_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome;
        arguments.count = count, arguments.keys = keys, arguments.vectors = rows;
        arguments.expansion_factor = expansion, arguments.stats = stats;
        usearch_dense_insert(&arguments);
    }
    static void search(index_t index, usearch_executor_t executor, usearch_arena_t arena, std::size_t count,
                       queries_t queries, std::size_t wanted, std::size_t expansion, usearch_traversal_t traversal,
                       usearch_matches_view_t* matches, usearch_traversal_stats_t* stats,
                       usearch_outcome_t* outcome) noexcept {
        usearch_dense_search_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome;
        arguments.count = count, arguments.queries = queries, arguments.matches = matches, arguments.arena = arena;
        arguments.wanted = wanted, arguments.expansion_factor = expansion, arguments.traversal = traversal;
        arguments.stats = stats;
        usearch_dense_search(&arguments);
    }
    static void save(index_t index, usearch_executor_t executor, usearch_bytes_view_t path,
                     usearch_outcome_t* outcome) noexcept {
        usearch_dense_save_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome, arguments.path = path;
        usearch_dense_save(&arguments);
    }
    static void load_ram(index_t* index, usearch_executor_t executor, usearch_bytes_view_t path,
                         usearch_ram_options_t ram, geometry_t const&, usearch_outcome_t* outcome) noexcept {
        usearch_dense_load_ram_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome, arguments.path = path;
        arguments.ram = ram;
        usearch_dense_load_ram(&arguments);
    }
    static void load_vram(index_t* index, usearch_executor_t executor, usearch_bytes_view_t path,
                          usearch_vram_options_t vram, geometry_t const&, usearch_outcome_t* outcome) noexcept {
        usearch_dense_load_vram_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome, arguments.path = path;
        arguments.vram = vram;
        usearch_dense_load_vram(&arguments);
    }
    static void view_nvme(index_t* index, usearch_executor_t executor, usearch_bytes_view_t path,
                          usearch_nvme_options_t nvme, geometry_t const&, usearch_outcome_t* outcome) noexcept {
        usearch_dense_view_nvme_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome, arguments.path = path;
        arguments.nvme = nvme;
        usearch_dense_view_nvme(&arguments);
    }
    static void describe(index_t index, usearch_executor_t executor, description_t* description,
                         usearch_outcome_t* outcome) noexcept {
        usearch_dense_describe_t arguments {index, executor, outcome, usearch_completion_blocking_k, description};
        usearch_dense_describe(&arguments);
    }
    static void count(index_t index, usearch_executor_t executor, usearch_key_t key, std::size_t* entries,
                      usearch_outcome_t* outcome) noexcept {
        usearch_dense_count_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome;
        arguments.key = key, arguments.entries = entries;
        usearch_dense_count(&arguments);
    }
    static void clear(index_t index, usearch_executor_t executor, usearch_outcome_t* outcome) noexcept {
        usearch_dense_clear_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome;
        usearch_dense_clear(&arguments);
    }
    static void refine(index_t index, usearch_executor_t executor, std::size_t* changed,
                       usearch_outcome_t* outcome) noexcept {
        usearch_dense_refine_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome,
        arguments.changed = changed;
        usearch_dense_refine(&arguments);
    }
    static void absorb(index_t index, usearch_executor_t executor, index_t* donors, std::size_t count,
                       usearch_outcome_t* outcome) noexcept {
        usearch_dense_absorb_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome;
        arguments.count = count, arguments.donors = donors;
        usearch_dense_absorb(&arguments);
    }
    static void free(index_t index) noexcept { usearch_dense_free(index); }
};

/** What a sequence index is opened with beyond its metadata: the weights and window widths,
 *  borrowed. */
struct sequence_geometry_t {
    usearch_sequence_metadata_t metadata {};
    usearch_sequence_weights_t const* weights = nullptr;
    std::span<std::size_t const> window_widths;
};

/** The C functions of the sequence modality, behind the signatures every modality shares. */
struct sequence_verbs_t {
    using index_t = usearch_sequence_index_t;
    using description_t = usearch_sequence_index_description_t;
    using geometry_t = sequence_geometry_t;
    using inputs_t = usearch_tape_view_t;
    using queries_t = usearch_tape_view_t;
    static constexpr std::string_view name_k = "sequence";

    static inputs_t slice(inputs_t tape, std::size_t first) noexcept {
        tape.offsets += first;
        return tape;
    }
    static void init_ram(index_t* index, geometry_t const& geometry, usearch_ram_options_t ram,
                         usearch_outcome_t* outcome) noexcept {
        usearch_sequence_init_ram_t arguments {index,
                                               outcome,
                                               geometry.metadata.metric,
                                               geometry.weights,
                                               geometry.window_widths.data(),
                                               geometry.window_widths.size(),
                                               geometry.metadata.connectivity,
                                               0,
                                               ram};
        usearch_sequence_init_ram(&arguments);
    }
    static void init_vram(index_t* index, geometry_t const& geometry, usearch_vram_options_t vram,
                          usearch_outcome_t* outcome) noexcept {
        usearch_sequence_init_vram_t arguments {index,
                                                outcome,
                                                geometry.metadata.metric,
                                                geometry.weights,
                                                geometry.window_widths.data(),
                                                geometry.window_widths.size(),
                                                geometry.metadata.connectivity,
                                                0,
                                                vram};
        usearch_sequence_init_vram(&arguments);
    }
    static void insert(index_t index, usearch_executor_t executor, std::size_t count, usearch_keys_view_t keys,
                       inputs_t sequences, std::size_t expansion, usearch_traversal_stats_t* stats,
                       usearch_outcome_t* outcome) noexcept {
        usearch_sequence_insert_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome;
        arguments.count = count, arguments.keys = keys, arguments.sequences = sequences;
        arguments.expansion_factor = expansion, arguments.stats = stats;
        usearch_sequence_insert(&arguments);
    }
    static void search(index_t index, usearch_executor_t executor, usearch_arena_t arena, std::size_t count,
                       queries_t queries, std::size_t wanted, std::size_t expansion, usearch_traversal_t traversal,
                       usearch_matches_view_t* matches, usearch_traversal_stats_t* stats,
                       usearch_outcome_t* outcome) noexcept {
        usearch_sequence_search_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome;
        arguments.count = count, arguments.queries = queries, arguments.matches = matches, arguments.arena = arena;
        arguments.wanted = wanted, arguments.expansion_factor = expansion, arguments.traversal = traversal;
        arguments.stats = stats;
        usearch_sequence_search(&arguments);
    }
    static void save(index_t index, usearch_executor_t executor, usearch_bytes_view_t path,
                     usearch_outcome_t* outcome) noexcept {
        usearch_sequence_save_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome, arguments.path = path;
        usearch_sequence_save(&arguments);
    }
    static void load_ram(index_t* index, usearch_executor_t executor, usearch_bytes_view_t path,
                         usearch_ram_options_t ram, geometry_t const& geometry, usearch_outcome_t* outcome) noexcept {
        usearch_sequence_load_ram_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome, arguments.path = path;
        arguments.ram = ram, arguments.weights = geometry.weights;
        usearch_sequence_load_ram(&arguments);
    }
    static void load_vram(index_t* index, usearch_executor_t executor, usearch_bytes_view_t path,
                          usearch_vram_options_t vram, geometry_t const& geometry,
                          usearch_outcome_t* outcome) noexcept {
        usearch_sequence_load_vram_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome, arguments.path = path;
        arguments.vram = vram, arguments.weights = geometry.weights;
        usearch_sequence_load_vram(&arguments);
    }
    static void view_nvme(index_t* index, usearch_executor_t executor, usearch_bytes_view_t path,
                          usearch_nvme_options_t nvme, geometry_t const& geometry,
                          usearch_outcome_t* outcome) noexcept {
        usearch_sequence_view_nvme_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome, arguments.path = path;
        arguments.nvme = nvme, arguments.weights = geometry.weights;
        usearch_sequence_view_nvme(&arguments);
    }
    static void describe(index_t index, usearch_executor_t executor, description_t* description,
                         usearch_outcome_t* outcome) noexcept {
        usearch_sequence_describe_t arguments {index, executor, outcome, usearch_completion_blocking_k, description};
        usearch_sequence_describe(&arguments);
    }
    static void count(index_t index, usearch_executor_t executor, usearch_key_t key, std::size_t* entries,
                      usearch_outcome_t* outcome) noexcept {
        usearch_sequence_key_stats_t key_stats {};
        usearch_sequence_count_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome;
        arguments.key = key, arguments.key_stats = &key_stats;
        usearch_sequence_count(&arguments);
        *entries = key_stats.entries;
    }
    static void clear(index_t index, usearch_executor_t executor, usearch_outcome_t* outcome) noexcept {
        usearch_sequence_clear_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome;
        usearch_sequence_clear(&arguments);
    }
    static void refine(index_t index, usearch_executor_t executor, std::size_t* changed,
                       usearch_outcome_t* outcome) noexcept {
        usearch_sequence_refine_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome,
        arguments.changed = changed;
        usearch_sequence_refine(&arguments);
    }
    static void absorb(index_t index, usearch_executor_t executor, index_t* donors, std::size_t count,
                       usearch_outcome_t* outcome) noexcept {
        usearch_sequence_absorb_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome;
        arguments.count = count, arguments.donors = donors;
        usearch_sequence_absorb(&arguments);
    }
    static void free(index_t index) noexcept { usearch_sequence_free(index); }
};

/** The C functions of the shapes modality, behind the signatures every modality shares. */
struct shapes_verbs_t {
    using index_t = usearch_shapes_index_t;
    using description_t = usearch_shapes_index_description_t;
    using geometry_t = usearch_shapes_metadata_t;
    using inputs_t = usearch_tape_view_t;
    using queries_t = usearch_tape_view_t;
    static constexpr std::string_view name_k = "shapes";

    static inputs_t slice(inputs_t tape, std::size_t first) noexcept {
        tape.offsets += first;
        return tape;
    }
    static void init_ram(index_t* index, geometry_t const& geometry, usearch_ram_options_t ram,
                         usearch_outcome_t* outcome) noexcept {
        usearch_shapes_init_ram_t arguments {
            index, outcome, geometry.metric, geometry.quantization, geometry.points, geometry.connectivity, 0, ram};
        usearch_shapes_init_ram(&arguments);
    }
    static void init_vram(index_t* index, geometry_t const& geometry, usearch_vram_options_t vram,
                          usearch_outcome_t* outcome) noexcept {
        usearch_shapes_init_vram_t arguments {
            index, outcome, geometry.metric, geometry.quantization, geometry.points, geometry.connectivity, 0, vram};
        usearch_shapes_init_vram(&arguments);
    }
    static void insert(index_t index, usearch_executor_t executor, std::size_t count, usearch_keys_view_t keys,
                       inputs_t shapes, std::size_t expansion, usearch_traversal_stats_t* stats,
                       usearch_outcome_t* outcome) noexcept {
        usearch_shapes_insert_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome;
        arguments.count = count, arguments.keys = keys, arguments.shapes = shapes;
        arguments.expansion_factor = expansion, arguments.stats = stats;
        usearch_shapes_insert(&arguments);
    }
    static void search(index_t index, usearch_executor_t executor, usearch_arena_t arena, std::size_t count,
                       queries_t queries, std::size_t wanted, std::size_t expansion, usearch_traversal_t traversal,
                       usearch_matches_view_t* matches, usearch_traversal_stats_t* stats,
                       usearch_outcome_t* outcome) noexcept {
        usearch_shapes_search_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome;
        arguments.count = count, arguments.queries = queries, arguments.matches = matches, arguments.arena = arena;
        arguments.wanted = wanted, arguments.expansion_factor = expansion, arguments.traversal = traversal;
        arguments.stats = stats;
        usearch_shapes_search(&arguments);
    }
    static void save(index_t index, usearch_executor_t executor, usearch_bytes_view_t path,
                     usearch_outcome_t* outcome) noexcept {
        usearch_shapes_save_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome, arguments.path = path;
        usearch_shapes_save(&arguments);
    }
    static void load_ram(index_t* index, usearch_executor_t executor, usearch_bytes_view_t path,
                         usearch_ram_options_t ram, geometry_t const&, usearch_outcome_t* outcome) noexcept {
        usearch_shapes_load_ram_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome, arguments.path = path;
        arguments.ram = ram;
        usearch_shapes_load_ram(&arguments);
    }
    static void load_vram(index_t* index, usearch_executor_t executor, usearch_bytes_view_t path,
                          usearch_vram_options_t vram, geometry_t const&, usearch_outcome_t* outcome) noexcept {
        usearch_shapes_load_vram_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome, arguments.path = path;
        arguments.vram = vram;
        usearch_shapes_load_vram(&arguments);
    }
    static void view_nvme(index_t* index, usearch_executor_t executor, usearch_bytes_view_t path,
                          usearch_nvme_options_t nvme, geometry_t const&, usearch_outcome_t* outcome) noexcept {
        usearch_shapes_view_nvme_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome, arguments.path = path;
        arguments.nvme = nvme;
        usearch_shapes_view_nvme(&arguments);
    }
    static void describe(index_t index, usearch_executor_t executor, description_t* description,
                         usearch_outcome_t* outcome) noexcept {
        usearch_shapes_describe_t arguments {index, executor, outcome, usearch_completion_blocking_k, description};
        usearch_shapes_describe(&arguments);
    }
    static void count(index_t index, usearch_executor_t executor, usearch_key_t key, std::size_t* entries,
                      usearch_outcome_t* outcome) noexcept {
        usearch_shapes_key_stats_t key_stats {};
        usearch_shapes_count_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome;
        arguments.key = key, arguments.key_stats = &key_stats;
        usearch_shapes_count(&arguments);
        *entries = key_stats.entries;
    }
    static void clear(index_t index, usearch_executor_t executor, usearch_outcome_t* outcome) noexcept {
        usearch_shapes_clear_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome;
        usearch_shapes_clear(&arguments);
    }
    static void refine(index_t index, usearch_executor_t executor, std::size_t* changed,
                       usearch_outcome_t* outcome) noexcept {
        usearch_shapes_refine_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome,
        arguments.changed = changed;
        usearch_shapes_refine(&arguments);
    }
    static void absorb(index_t index, usearch_executor_t executor, index_t* donors, std::size_t count,
                       usearch_outcome_t* outcome) noexcept {
        usearch_shapes_absorb_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome;
        arguments.count = count, arguments.donors = donors;
        usearch_shapes_absorb(&arguments);
    }
    static void free(index_t index) noexcept { usearch_shapes_free(index); }
};

/** What a geospatial index is opened with beyond its metadata: its terrain, whose heights are
 *  borrowed. */
struct geospatial_geometry_t {
    usearch_geospatial_metadata_t metadata {};
    usearch_geospatial_terrain_t terrain {};
};

/** One batch of geospatial queries: the observations, and the relation and tolerance they ask
 *  by. */
struct geospatial_queries_t {
    usearch_geospatial_observation_t const* observations = nullptr;
    usearch_geospatial_relation_t relation = usearch_geospatial_relation_near_k;
    usearch_geospatial_tolerance_t tolerance {};
};

/** The C functions of the geospatial modality, behind the signatures every modality shares. */
struct geospatial_verbs_t {
    using index_t = usearch_geospatial_index_t;
    using description_t = usearch_geospatial_index_description_t;
    using geometry_t = geospatial_geometry_t;
    using inputs_t = usearch_geospatial_observation_t const*;
    using queries_t = geospatial_queries_t;
    static constexpr std::string_view name_k = "geospatial";

    static inputs_t slice(inputs_t observations, std::size_t first) noexcept { return observations + first; }
    static queries_t slice(queries_t queries, std::size_t first) noexcept {
        queries.observations += first;
        return queries;
    }
    static void init_ram(index_t* index, geometry_t const& geometry, usearch_ram_options_t ram,
                         usearch_outcome_t* outcome) noexcept {
        usearch_geospatial_init_ram_t arguments {index, outcome, geometry.terrain, geometry.metadata.connectivity,
                                                 0,     ram};
        usearch_geospatial_init_ram(&arguments);
    }
    static void init_vram(index_t* index, geometry_t const& geometry, usearch_vram_options_t vram,
                          usearch_outcome_t* outcome) noexcept {
        usearch_geospatial_init_vram_t arguments {index, outcome, geometry.terrain, geometry.metadata.connectivity,
                                                  0,     vram};
        usearch_geospatial_init_vram(&arguments);
    }
    static void insert(index_t index, usearch_executor_t executor, std::size_t count, usearch_keys_view_t keys,
                       inputs_t observations, std::size_t expansion, usearch_traversal_stats_t* stats,
                       usearch_outcome_t* outcome) noexcept {
        usearch_geospatial_insert_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome;
        arguments.count = count, arguments.keys = keys, arguments.observations = observations;
        arguments.expansion_factor = expansion, arguments.stats = stats;
        usearch_geospatial_insert(&arguments);
    }
    static void search(index_t index, usearch_executor_t executor, usearch_arena_t arena, std::size_t count,
                       queries_t queries, std::size_t wanted, std::size_t expansion, usearch_traversal_t traversal,
                       usearch_matches_view_t* matches, usearch_traversal_stats_t* stats,
                       usearch_outcome_t* outcome) noexcept {
        usearch_geospatial_search_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome;
        arguments.count = count, arguments.queries = queries.observations, arguments.relation = queries.relation;
        arguments.tolerance = queries.tolerance, arguments.matches = matches, arguments.arena = arena;
        arguments.wanted = wanted, arguments.expansion_factor = expansion, arguments.traversal = traversal;
        arguments.stats = stats;
        usearch_geospatial_search(&arguments);
    }
    static void save(index_t index, usearch_executor_t executor, usearch_bytes_view_t path,
                     usearch_outcome_t* outcome) noexcept {
        usearch_geospatial_save_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome, arguments.path = path;
        usearch_geospatial_save(&arguments);
    }
    static void load_ram(index_t* index, usearch_executor_t executor, usearch_bytes_view_t path,
                         usearch_ram_options_t ram, geometry_t const& geometry, usearch_outcome_t* outcome) noexcept {
        usearch_geospatial_load_ram_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome, arguments.path = path;
        arguments.ram = ram, arguments.terrain = geometry.terrain;
        usearch_geospatial_load_ram(&arguments);
    }
    static void load_vram(index_t* index, usearch_executor_t executor, usearch_bytes_view_t path,
                          usearch_vram_options_t vram, geometry_t const& geometry,
                          usearch_outcome_t* outcome) noexcept {
        usearch_geospatial_load_vram_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome, arguments.path = path;
        arguments.vram = vram, arguments.terrain = geometry.terrain;
        usearch_geospatial_load_vram(&arguments);
    }
    static void view_nvme(index_t* index, usearch_executor_t executor, usearch_bytes_view_t path,
                          usearch_nvme_options_t nvme, geometry_t const& geometry,
                          usearch_outcome_t* outcome) noexcept {
        usearch_geospatial_view_nvme_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome, arguments.path = path;
        arguments.nvme = nvme, arguments.terrain = geometry.terrain;
        usearch_geospatial_view_nvme(&arguments);
    }
    static void describe(index_t index, usearch_executor_t executor, description_t* description,
                         usearch_outcome_t* outcome) noexcept {
        usearch_geospatial_describe_t arguments {index, executor, outcome, usearch_completion_blocking_k, description};
        usearch_geospatial_describe(&arguments);
    }
    static void count(index_t index, usearch_executor_t executor, usearch_key_t key, std::size_t* entries,
                      usearch_outcome_t* outcome) noexcept {
        usearch_geospatial_key_stats_t key_stats {};
        usearch_geospatial_count_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome;
        arguments.key = key, arguments.key_stats = &key_stats;
        usearch_geospatial_count(&arguments);
        *entries = key_stats.entries;
    }
    static void clear(index_t index, usearch_executor_t executor, usearch_outcome_t* outcome) noexcept {
        usearch_geospatial_clear_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome;
        usearch_geospatial_clear(&arguments);
    }
    static void refine(index_t index, usearch_executor_t executor, std::size_t* changed,
                       usearch_outcome_t* outcome) noexcept {
        usearch_geospatial_refine_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome,
        arguments.changed = changed;
        usearch_geospatial_refine(&arguments);
    }
    static void absorb(index_t index, usearch_executor_t executor, index_t* donors, std::size_t count,
                       usearch_outcome_t* outcome) noexcept {
        usearch_geospatial_absorb_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome;
        arguments.count = count, arguments.donors = donors;
        usearch_geospatial_absorb(&arguments);
    }
    static void free(index_t index) noexcept { usearch_geospatial_free(index); }
};

#pragma endregion Verbs

#pragma region Handles

/**
 *  @brief One index of one modality, freed through @p verbs_type_ when it goes, beside the session
 *      that drives it and the geometry it was opened with, which a reopening needs again.
 *
 *  @tparam verbs_type_ One of @ref dense_verbs_t, @ref sequence_verbs_t, @ref shapes_verbs_t or
 *      @ref geospatial_verbs_t.
 */
template <typename verbs_type_>
struct handle {
    using index_t = typename verbs_type_::index_t;
    using geometry_t = typename verbs_type_::geometry_t;

    index_t index = nullptr;
    session_t* session = nullptr;

    /** Where the index lives, which can differ from its session's residency while an NVMe session
     *  builds in RAM. */
    usearch_residency_t residency = usearch_residency_ram_k;
    geometry_t geometry {};

    handle() = default;
    handle(index_t index, session_t& session, usearch_residency_t residency, geometry_t const& geometry) noexcept
        : index(index), session(&session), residency(residency), geometry(geometry) {}
    handle(handle const&) = delete;
    handle& operator=(handle const&) = delete;
    handle(handle&& other) noexcept
        : index(std::exchange(other.index, nullptr)), session(other.session), residency(other.residency),
          geometry(other.geometry) {}
    handle& operator=(handle&& other) noexcept {
        std::swap(index, other.index), std::swap(session, other.session);
        std::swap(residency, other.residency), std::swap(geometry, other.geometry);
        return *this;
    }
    ~handle() noexcept { verbs_type_::free(index); }

    index_t get() const noexcept { return index; }
    explicit operator bool() const noexcept { return index != nullptr; }

    /** The executor this index's calls run on. */
    usearch_executor_t executor() const noexcept { return session->executor_for(residency); }
};

/** Opens an empty index of @p geometry on @p shards shards of @p session, in RAM for an NVMe
 *  session; zero shards takes one per domain or device. Prints what failed and answers an empty
 *  handle. */
template <typename verbs_type_>
handle<verbs_type_> open_index(session_t& session, typename verbs_type_::geometry_t const& geometry,
                               std::size_t shards) {
    typename verbs_type_::index_t index = nullptr;
    usearch_outcome_t outcome {};
    std::vector<std::int32_t> placement;
    usearch_residency_t const residency = session.residency == usearch_residency_vram_k ? usearch_residency_vram_k
                                                                                        : usearch_residency_ram_k;
    if (residency == usearch_residency_vram_k)
        verbs_type_::init_vram(&index, geometry, session.vram_options(shards, placement), &outcome);
    else verbs_type_::init_ram(&index, geometry, session.ram_options(shards, placement), &outcome);
    report_failure(outcome, "Opening the index");
    return {index, session, residency, geometry};
}

/** Reserves room for @p count more entries, then inserts @p count entries of @p inputs into
 *  @p index in batches of @p batch, keyed by @ref key_of_row from row @p first_row on, summing
 *  every batch's telemetry into @p totals and reporting each landed count to @p progress. */
template <typename verbs_type_, typename progress_type_ = dummy_progress_t>
usearch_outcome_t insert_entries(handle<verbs_type_>& index, typename verbs_type_::inputs_t inputs, std::size_t count,
                                 std::size_t first_row, std::size_t batch, std::size_t expansion,
                                 usearch_traversal_stats_t& totals, progress_type_&& progress = {}) {
    usearch_outcome_t outcome {};
    std::size_t const batch_rows = std::max<std::size_t>(1, std::min(batch, count));
    std::vector<usearch_key_t> keys(batch_rows);
    for (std::size_t first = 0; first < count; first += batch_rows) {
        std::size_t const rows = std::min(batch_rows, count - first);
        std::span<usearch_key_t> const batch_keys = std::span {keys}.first(rows);
        for (std::size_t row = 0; row != rows; ++row)
            batch_keys[row] = key_of_row(index.session->seed, first_row + first + row);
        usearch_traversal_stats_t stats {};
        verbs_type_::insert(index.get(), index.executor(), rows, keys_view(batch_keys),
                            verbs_type_::slice(inputs, first), expansion, &stats, &outcome);
        if (outcome.status != usearch_status_success_k) return outcome;
        totals.visited_members += stats.visited_members, totals.computed_distances += stats.computed_distances;
        totals.restarts += stats.restarts;
        progress(first + rows);
    }
    return outcome;
}

/** Describes @p index; prints what failed and answers an empty description. */
template <typename verbs_type_>
typename verbs_type_::description_t describe(handle<verbs_type_> const& index) {
    typename verbs_type_::description_t described {};
    usearch_outcome_t outcome {};
    verbs_type_::describe(index.get(), index.executor(), &described, &outcome);
    report_failure(outcome, "Describing");
    return described;
}

/** Prints the entries, the bytes, the shards and the classes @p index reports, as one "- Index:"
 *  line. */
template <typename verbs_type_>
void print_fill(handle<verbs_type_> const& index) {
    typename verbs_type_::description_t const described = describe(index);
    fmt::println("- Index: {} entries · {} bytes · {} shards · {} · {}", described.size, described.memory_usage,
                 described.shards_count, usearch_cpu_class_name(described.cpu_class),
                 usearch_gpu_class_name(described.gpu_class));
}

/** Saves @p index under @p path and reopens it in @p residency on the same session: loaded into RAM
 *  or onto the session's devices, or viewed on its NVMe executor. Prints what failed and answers an
 *  empty handle. */
template <typename verbs_type_>
handle<verbs_type_> save_and_reopen(handle<verbs_type_>&& index, std::string_view path, usearch_residency_t residency) {
    session_t& session = *index.session;
    typename verbs_type_::geometry_t const geometry = index.geometry;
    usearch_outcome_t outcome {};
    verbs_type_::save(index.get(), index.executor(), bytes_view(path), &outcome);
    if (report_failure(outcome, fmt::format("Saving {}", path))) return {};
    std::size_t const shards = describe(index).shards_count;
    index = {};
    typename verbs_type_::index_t reopened = nullptr;
    std::vector<std::int32_t> placement;
    if (residency == usearch_residency_nvme_k)
        verbs_type_::view_nvme(&reopened, session.storage, bytes_view(path), session.nvme_options(), geometry,
                               &outcome);
    else if (residency == usearch_residency_vram_k)
        verbs_type_::load_vram(&reopened, session.executor, bytes_view(path), session.vram_options(shards, placement),
                               geometry, &outcome);
    else
        verbs_type_::load_ram(&reopened, session.executor, bytes_view(path), session.ram_options(0, placement),
                              geometry, &outcome);
    report_failure(outcome, fmt::format("Reopening {}", path));
    return {reopened, session, residency, geometry};
}

#pragma endregion Handles

#pragma region Datasets

/** Whether @p name matches @p pattern, where `*` spans any run of characters and `?` any one. */
inline bool wildcard_matches(std::string_view pattern, std::string_view name) noexcept {
    std::size_t pattern_at = 0, name_at = 0;
    std::size_t star = std::string_view::npos, resumed = 0;
    while (name_at != name.size()) {
        if (pattern_at != pattern.size() && (pattern[pattern_at] == '?' || pattern[pattern_at] == name[name_at]))
            ++pattern_at, ++name_at;
        else if (pattern_at != pattern.size() && pattern[pattern_at] == '*') star = pattern_at++, resumed = name_at;
        else if (star != std::string_view::npos) pattern_at = star + 1, name_at = ++resumed;
        else return false;
    }
    while (pattern_at != pattern.size() && pattern[pattern_at] == '*') ++pattern_at;
    return pattern_at == pattern.size();
}

/** The regular files in @p pattern's directory whose names match its last component, sorted, so
 *  parts numbered with leading zeros come in order; wildcards only in that last component. */
inline std::vector<std::string> matching_files(std::string_view pattern) {
    std::size_t const slash = pattern.find_last_of('/');
    std::string_view const directory = slash == std::string_view::npos ? std::string_view {}
                                                                       : pattern.substr(0, slash + 1);
    std::string_view const filename_pattern = pattern.substr(directory.size());
    std::vector<std::string> files;
    std::error_code error;
    std::filesystem::directory_iterator entries {
        directory.empty() ? std::filesystem::path {"."} : std::filesystem::path {directory}, error};
    for (; !error && entries != std::filesystem::directory_iterator {}; entries.increment(error)) {
        std::filesystem::directory_entry const& entry = *entries;
        std::string const name = entry.path().filename().string();
        std::error_code kind_error;
        if (entry.is_regular_file(kind_error) && wildcard_matches(filename_pattern, name))
            files.push_back(std::string {directory} + name);
    }
    std::sort(files.begin(), files.end());
    return files;
}

/** Every file whose name starts with @p prefix, in sorted order. */
inline std::vector<std::string> files_starting_with(std::string const& prefix) { return matching_files(prefix + "*"); }

/** The scalar a RetriEval extension names, or @c usearch_scalar_unknown_k; `.ibin` neighbor IDs
 *  name none, as no scalar holds a row. */
inline usearch_scalar_t scalar_of_extension(std::string_view path) noexcept {
    if (path.ends_with(".f64bin")) return usearch_scalar_f64_k;
    if (path.ends_with(".fbin")) return usearch_scalar_f32_k;
    if (path.ends_with(".bf16bin")) return usearch_scalar_bf16_k;
    if (path.ends_with(".f16bin")) return usearch_scalar_f16_k;
    if (path.ends_with(".i8bin")) return usearch_scalar_i8_k;
    if (path.ends_with(".u8bin")) return usearch_scalar_u8_k;
    if (path.ends_with(".u1x8bin") || path.ends_with(".b1bin")) return usearch_scalar_u1x8_k;
    return usearch_scalar_unknown_k;
}

/** Bytes one row of @p columns logical values of @p scalar takes, or of @c u32 neighbor IDs for the
 *  unknown scalar. */
inline std::size_t row_bytes_of(usearch_scalar_t scalar, std::size_t columns) noexcept {
    switch (scalar) {
    case usearch_scalar_f64_k: return columns * 8;
    case usearch_scalar_f32_k:
    case usearch_scalar_unknown_k: return columns * 4;
    case usearch_scalar_bf16_k:
    case usearch_scalar_f16_k: return columns * 2;
    case usearch_scalar_u1x8_k: return divide_round_up<std::size_t>(columns, 8);
    default: return columns;
    }
}

/** One mapped RetriEval part file and the rows of the whole matrix it holds. */
struct mapped_part_t {
    mapped_file_t mapping;
    std::size_t first_row = 0;
    std::size_t rows = 0;
};

/**
 *  @brief The RetriEval part files a wildcard pattern names, mapped in sorted order as one matrix
 *      split by rows, never copying a byte.
 *
 *  Each file opens with two little-endian @c u32 counters, rows then columns, before its row-major
 *  payload, and the parts number their rows on across each other, so a set past the counter's range
 *  splits into whole-row parts. A path without wildcards names itself.
 */
class mapped_rows_t {
    std::vector<mapped_part_t> parts_;
    usearch_scalar_t scalar_ = usearch_scalar_unknown_k;
    std::size_t rows_ = 0, columns_ = 0, row_bytes_ = 0;

    status_t append_(std::string_view path) noexcept {
        auto [mapping, read_status] = mapped_file_t::read(path);
        if (failed(read_status)) return read_status;
        bytes_view_t const bytes = mapping.view();
        std::uint32_t counters[2] = {};
        if (bytes.size() < sizeof(counters)) return status_t::malformed_k;
        std::memcpy(counters, bytes.data(), sizeof(counters));
        if (!parts_.empty() && counters[1] != columns_) return status_t::config_mismatch_k;
        columns_ = counters[1];
        row_bytes_ = row_bytes_of(scalar_, columns_);
        if (bytes.size() < sizeof(counters) + static_cast<std::size_t>(counters[0]) * row_bytes_)
            return status_t::malformed_k;
        parts_.push_back({std::move(mapping), rows_, counters[0]});
        rows_ += counters[0];
        return status_t::success_k;
    }

  public:

    /** Maps the parts @p pattern names, failing on an unreadable part or a column mismatch. */
    static expected<mapped_rows_t> open(std::string_view pattern) noexcept {
        mapped_rows_t mapped;
        mapped.scalar_ = scalar_of_extension(pattern);
        std::vector<std::string> parts = matching_files(pattern);
        // With no match the pattern stands for itself, so a missing file fails like its mapping.
        if (parts.empty()) parts.emplace_back(pattern);
        status_t status = status_t::success_k;
        for (std::size_t part = 0; succeeded(status) && part != parts.size(); ++part)
            status = mapped.append_(parts[part]);
        return {std::move(mapped), status};
    }

    /** Rows `[first_row, first_row + count)` as the C interface takes them, viewed in place when
     *  one part holds them all; null data when they straddle two parts or run past the last. */
    usearch_rows_view_t view(std::size_t first_row, std::size_t count) const noexcept {
        for (mapped_part_t const& part : parts_) {
            if (first_row < part.first_row || first_row + count > part.first_row + part.rows) continue;
            byte_t const* const payload = part.mapping.view().data() + 2 * sizeof(std::uint32_t);
            return {payload + (first_row - part.first_row) * row_bytes_, row_bytes_, columns_, scalar_};
        }
        return {nullptr, row_bytes_, columns_, scalar_};
    }

    std::size_t rows() const noexcept { return rows_; }
    std::size_t columns() const noexcept { return columns_; }
    usearch_scalar_t scalar() const noexcept { return scalar_; }
    explicit operator bool() const noexcept { return !parts_.empty(); }
};

#pragma endregion Datasets

#pragma region Recall

/** The matches searches landed, copied off their arena: every row's keys and scores end to end, and
 *  how many each row holds, so whole searches compare and judge. */
struct landed_matches_t {
    std::vector<uuid_t> keys;
    std::vector<usearch_distance_t> scores;
    std::vector<std::size_t> counts;

    bool operator==(landed_matches_t const&) const = default;
};

/** Appends the matches each of the @p queries rows of @p matches holds to @p landed, read back from
 *  wherever the arena placed them; false when the copy failed. */
inline bool append_landed(usearch_matches_view_t const& matches, std::size_t queries, landed_matches_t& landed) {
    std::vector<usearch_key_t> keys(queries * matches.columns);
    std::vector<usearch_distance_t> scores(queries * matches.columns);
    std::vector<std::size_t> counts(queries);
    if (!download(keys.data(), matches.keys, keys.size() * sizeof(usearch_key_t)) ||
        !download(scores.data(), matches.scores, scores.size() * sizeof(usearch_distance_t)) ||
        !download(counts.data(), matches.counts, counts.size() * sizeof(std::size_t)))
        return false;
    for (std::size_t row = 0; row != queries; ++row) {
        landed.counts.push_back(counts[row]);
        for (std::size_t match = row * matches.columns; match != row * matches.columns + counts[row]; ++match) {
            landed.keys.push_back(std::bit_cast<uuid_t>(keys[match]));
            landed.scores.push_back(scores[match]);
        }
    }
    return true;
}

/** The matches each of the @p queries rows of @p matches holds, copied off a host arena. */
inline landed_matches_t landed_matches(usearch_matches_view_t const& matches, std::size_t queries) {
    landed_matches_t landed {{}, {}, {matches.counts, matches.counts + queries}};
    for (std::size_t row = 0; row != queries; ++row)
        for (std::size_t match = row * matches.columns; match != row * matches.columns + matches.counts[row]; ++match) {
            landed.keys.push_back(std::bit_cast<uuid_t>(matches.keys[match]));
            landed.scores.push_back(matches.scores[match]);
        }
    return landed;
}

/** Searches @p count queries of @p queries in batches of @p batch, landing every match in @p landed
 *  and summing the telemetry into @p totals. */
template <typename verbs_type_>
usearch_outcome_t search_queries(handle<verbs_type_>& index, typename verbs_type_::queries_t queries, std::size_t count,
                                 std::size_t batch, std::size_t wanted, std::size_t expansion,
                                 usearch_traversal_t traversal, usearch_traversal_stats_t& totals,
                                 landed_matches_t& landed) {
    usearch_outcome_t outcome {};
    std::size_t const batch_rows = std::max<std::size_t>(1, std::min(batch, count));
    for (std::size_t first = 0; first < count; first += batch_rows) {
        std::size_t const rows = std::min(batch_rows, count - first);
        usearch_matches_view_t matches {};
        usearch_traversal_stats_t stats {};
        verbs_type_::search(index.get(), index.executor(), index.session->arena, rows,
                            verbs_type_::slice(queries, first), wanted, expansion, traversal, &matches, &stats,
                            &outcome);
        if (outcome.status != usearch_status_success_k) return outcome;
        if (!append_landed(matches, rows, landed))
            return {usearch_status_device_k, "The matches could not be read back"};
        totals.visited_members += stats.visited_members, totals.computed_distances += stats.computed_distances;
        totals.restarts += stats.restarts;
    }
    return outcome;
}

/** Fills @p oracle with the exact top @p wanted of the first @p judged queries of @p queries,
 *  nearest first with their distances, by a scan of every row @p index holds. */
template <typename verbs_type_>
usearch_outcome_t scan_oracle(handle<verbs_type_>& index, typename verbs_type_::queries_t queries, std::size_t judged,
                              std::size_t wanted, std::size_t batch, landed_matches_t& oracle) {
    usearch_traversal_stats_t stats {};
    return search_queries(index, queries, judged, batch, wanted, 0, usearch_traversal_scan_k, stats, oracle);
}

/** The truth a ground-truth file lists: @p depth rows per query, each keyed by @ref key_of_row
 *  under @p seed, without distances. */
inline landed_matches_t truth_oracle(u64_t seed, std::span<std::uint32_t const> rows, std::size_t depth) {
    landed_matches_t oracle;
    std::size_t const queries = depth ? rows.size() / depth : 0;
    oracle.counts.assign(queries, depth);
    for (std::uint32_t const row : rows.first(queries * depth))
        oracle.keys.push_back(std::bit_cast<uuid_t>(key_of_row(seed, row)));
    return oracle;
}

/** Whether @p found ranks no further than @p exact, within the rounding two kits may differ by. */
inline bool within(double found, double exact) noexcept {
    return found <= exact + 1e-6 * std::max(1.0, std::fabs(exact));
}

/** The offset of every row's first match in @p landed. */
inline std::vector<std::size_t> row_offsets(landed_matches_t const& landed) {
    std::vector<std::size_t> offsets(landed.counts.size() + 1);
    for (std::size_t row = 0; row != landed.counts.size(); ++row) offsets[row + 1] = offsets[row] + landed.counts[row];
    return offsets;
}

/**
 *  @brief Calls @p visit with every judged query's verdict, as `visit(expected, hits)`: how many
 *      matches the exact @p oracle expects among the first @p k, and whether each of the first @p k
 *      matches @p found holds is a hit.
 *
 *  An oracle with distances, as a scan lands it, counts a match within the exact k-th distance, so
 *  entries tied at the cut count wherever the scan broke the tie; one without, as a ground-truth
 *  file lists it, counts a match whose key it lists.
 */
template <typename visit_type_>
void judge_rows(std::size_t k, landed_matches_t const& found, landed_matches_t const& oracle, visit_type_&& visit) {
    std::vector<std::size_t> const found_offsets = row_offsets(found), truth_offsets = row_offsets(oracle);
    bool const scored = !oracle.scores.empty();
    std::vector<bool> hits;
    for (std::size_t query = 0; query != std::min(found.counts.size(), oracle.counts.size()); ++query) {
        std::size_t const expected = std::min(k, oracle.counts[query]);
        if (!expected) continue;
        std::size_t const truth_first = truth_offsets[query], found_first = found_offsets[query];
        std::size_t const ranked = std::min(k, found.counts[query]);
        hits.assign(ranked, false);
        for (std::size_t rank = 0; rank != ranked; ++rank)
            if (scored)
                hits[rank] = within(found.scores[found_first + rank], oracle.scores[truth_first + expected - 1]);
            else
                for (std::size_t truth = 0; truth != expected; ++truth)
                    hits[rank] = hits[rank] || found.keys[found_first + rank] == oracle.keys[truth_first + truth];
        visit(expected, hits);
    }
}

/** The share of each judged query's top @p k the matches @p found hold among their own first @p k,
 *  averaged over the queries the oracle expects any match for. */
inline double recall_at(std::size_t k, landed_matches_t const& found, landed_matches_t const& oracle) {
    double recall = 0;
    std::size_t answerable = 0;
    judge_rows(k, found, oracle, [&](std::size_t expected, std::vector<bool> const& hits) {
        std::size_t hit_count = 0;
        for (bool const hit : hits) hit_count += hit;
        recall += static_cast<double>(std::min(hit_count, expected)) / static_cast<double>(expected);
        ++answerable;
    });
    return answerable ? recall / static_cast<double>(answerable) : 0;
}

/** The normalized discounted cumulative gain of @p found 's first @p k matches against @p oracle,
 *  every hit earning its rank's discount, averaged over the queries the oracle expects any match
 *  for. */
inline double ndcg_at(std::size_t k, landed_matches_t const& found, landed_matches_t const& oracle) {
    double gain_sum = 0;
    std::size_t answerable = 0;
    judge_rows(k, found, oracle, [&](std::size_t expected, std::vector<bool> const& hits) {
        double gain = 0, ideal = 0;
        std::size_t counted = 0;
        for (std::size_t rank = 0; rank != hits.size() && counted != expected; ++rank)
            if (hits[rank]) gain += 1.0 / std::log2(static_cast<double>(rank) + 2.0), ++counted;
        for (std::size_t rank = 0; rank != expected; ++rank) ideal += 1.0 / std::log2(static_cast<double>(rank) + 2.0);
        gain_sum += gain / ideal;
        ++answerable;
    });
    return answerable ? gain_sum / static_cast<double>(answerable) : 0;
}

/** Searches @p count queries of @p queries at every breadth of @p expansions, printing each
 *  breadth's recall at one and at @p wanted and its nDCG against @p oracle, its throughput and its
 *  distances per query, then handing its matches to @p on_swept as `(expansion, matches)`. */
template <typename verbs_type_, typename swept_type_ = dummy_progress_t>
usearch_outcome_t sweep_recall(handle<verbs_type_>& index, typename verbs_type_::queries_t queries, std::size_t count,
                               landed_matches_t const& oracle, std::span<std::size_t const> expansions,
                               std::size_t wanted, std::size_t batch, swept_type_&& on_swept = {}) {
    usearch_outcome_t outcome {};
    for (std::size_t const expansion : expansions) {
        landed_matches_t found;
        usearch_traversal_stats_t stats {};
        std::chrono::steady_clock::time_point const started = std::chrono::steady_clock::now();
        outcome = search_queries(index, queries, count, batch, wanted, expansion, usearch_traversal_walk_k, stats,
                                 found);
        double const seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        if (outcome.status != usearch_status_success_k) return outcome;
        fmt::println("- Expansion {}: recall@1 {:.4f} · recall@{} {:.4f} · nDCG@{} {:.4f} over {} judged", expansion,
                     recall_at(1, found, oracle), wanted, recall_at(wanted, found, oracle), wanted,
                     ndcg_at(wanted, found, oracle), oracle.counts.size());
        double const per_query = 1.0 / static_cast<double>(count);
        fmt::println("  {} queries in {:.2f} s · {:.0f} QPS", count, seconds, static_cast<double>(count) / seconds);
        fmt::println("  per query: {:.0f} distances · {:.0f} visited", stats.computed_distances * per_query,
                     stats.visited_members * per_query);
        on_swept(expansion, found);
    }
    return outcome;
}

#pragma endregion Recall

#pragma region Sweeps

/** Whether a dense index of @p scalar opens in @p residency, pinned to @p cpu_class, on device zero
 *  for VRAM; a probe of what the library compiled and this machine admits. */
inline bool dense_opens(usearch_residency_t residency, usearch_scalar_t scalar,
                        usearch_cpu_class_t cpu_class) noexcept {
    usearch_dense_index_t index = nullptr;
    usearch_outcome_t outcome {};
    usearch_dense_metadata_t const geometry {usearch_dense_metric_inner_k, scalar, 64, 0};
    std::int32_t const device = 0;
    if (residency == usearch_residency_vram_k)
        dense_verbs_t::init_vram(&index, geometry, {&device, 1, 0, usearch_gpu_class_unknown_k}, &outcome);
    else dense_verbs_t::init_ram(&index, geometry, {nullptr, 0, 0, cpu_class}, &outcome);
    dense_verbs_t::free(index);
    return outcome.status == usearch_status_success_k;
}

/** The residencies this build and machine serve: RAM and NVMe always, and VRAM when a GPU is
 *  listed. */
inline std::vector<usearch_residency_t> compiled_residencies() {
    std::vector<usearch_residency_t> residencies {usearch_residency_ram_k, usearch_residency_nvme_k};
    if (!listed_devices().empty() && dense_opens(usearch_residency_vram_k, usearch_scalar_bf16_k, {}))
        residencies.push_back(usearch_residency_vram_k);
    return residencies;
}

/** The dense scalars @p residency opens, in the enumeration's order; an NVMe index serves what RAM
 *  built. */
inline std::vector<usearch_scalar_t> compiled_scalars(usearch_residency_t residency) {
    std::vector<usearch_scalar_t> scalars;
    for (int code = usearch_scalar_f64_k; code <= usearch_scalar_u1x8_k; ++code) {
        usearch_scalar_t const scalar = static_cast<usearch_scalar_t>(code);
        if (dense_opens(residency == usearch_residency_vram_k ? residency : usearch_residency_ram_k, scalar, {}))
            scalars.push_back(scalar);
    }
    return scalars;
}

/** The CPU classes the library accepts as pins on this machine, from the portable floor up. */
inline std::vector<usearch_cpu_class_t> admitted_classes() {
    std::vector<usearch_cpu_class_t> classes;
    for (int code = usearch_cpu_class_serial_k; code <= usearch_cpu_class_wasm64_extended_k; ++code) {
        usearch_cpu_class_t const cpu_class = static_cast<usearch_cpu_class_t>(code);
        if (dense_opens(usearch_residency_ram_k, usearch_scalar_bf16_k, cpu_class)) classes.push_back(cpu_class);
    }
    return classes;
}

/** One cell of a suite's sweep: where its index lives, what it stores, the class it pins and how
 *  many shards it spans. */
struct sweep_cell_t {
    usearch_residency_t residency = usearch_residency_ram_k;
    usearch_scalar_t scalar = usearch_scalar_unknown_k;
    usearch_cpu_class_t cpu_class = usearch_cpu_class_unknown_k;
    std::size_t shards = 1;
};

/**
 *  @brief The diagonal a suite sweeps rather than the full product: every scalar at the best class,
 *      every admitted class at @p first, and serial at every scalar, each over the compiled
 *      residencies at one and at three shards.
 *
 *  A basic modality passes @c usearch_scalar_unknown_k, whose cells vary the class alone, so the
 *  suite runs its first metric at each and every metric at the best class. A VRAM cell pins no CPU
 *  class, as no CPU walks it.
 */
inline std::vector<sweep_cell_t> diagonal_sweep(usearch_scalar_t first) {
    std::vector<sweep_cell_t> cells;
    std::vector<usearch_cpu_class_t> const classes = admitted_classes();
    for (usearch_residency_t const residency : compiled_residencies()) {
        std::vector<usearch_scalar_t> const scalars = first == usearch_scalar_unknown_k
                                                          ? std::vector<usearch_scalar_t> {first}
                                                          : compiled_scalars(residency);
        for (std::size_t const shards : {std::size_t {1}, std::size_t {3}}) {
            for (usearch_scalar_t const scalar : scalars)
                cells.push_back({residency, scalar, usearch_cpu_class_unknown_k, shards});
            if (residency == usearch_residency_vram_k) continue;
            for (usearch_cpu_class_t const cpu_class : classes) cells.push_back({residency, first, cpu_class, shards});
            for (usearch_scalar_t const scalar : scalars)
                if (scalar != first) cells.push_back({residency, scalar, usearch_cpu_class_serial_k, shards});
        }
    }
    return cells;
}

/** The name a filter selects @p cell by, like "ram bf16 intel-icelake 3-shard". */
inline std::string name_of(sweep_cell_t const& cell) {
    return fmt::format("{} {} {} {}-shard", usearch_residency_name(cell.residency),
                       cell.scalar ? usearch_scalar_name(cell.scalar) : "metrics",
                       cell.cpu_class ? usearch_cpu_class_name(cell.cpu_class) : "best", cell.shards);
}

/** The settings of @p settings with @p cell 's residency, for the session the cell opens. */
inline settings_t settings_of(settings_t const& settings, sweep_cell_t const& cell) {
    settings_t of_cell = settings;
    of_cell.residency = cell.residency;
    return of_cell;
}

#pragma endregion Sweeps

#pragma region Checks

/** Answers one failure when @p condition does not hold, printing where. */
inline std::size_t check(bool condition, std::source_location where = std::source_location::current()) {
    if (!condition) fmt::println(stderr, "  check failed at {}:{}", where.file_name(), where.line());
    return condition ? 0u : 1u;
}

/** Answers one failure when @p outcome is no success, printing its status and message and where. */
inline std::size_t check(usearch_outcome_t const& outcome,
                         std::source_location where = std::source_location::current()) {
    if (outcome.status == usearch_status_success_k) return 0;
    fmt::println(stderr, "  check failed at {}:{}: {}: {}", where.file_name(), where.line(),
                 usearch_status_name(outcome.status), outcome.message ? outcome.message : "no message");
    return 1;
}

/** Whether @p first and @p second land as many matches per query, at the same distances rank by
 *  rank: within @p score_band of each other, or with no band within the rounding two scoring paths
 *  differ by; keys are not compared, as ties reorder them. */
inline bool equivalent_matches(landed_matches_t const& first, landed_matches_t const& second, double score_band) {
    if (first.counts != second.counts || first.scores.size() != second.scores.size()) return false;
    for (std::size_t match = 0; match != first.scores.size(); ++match) {
        double const one = first.scores[match], other = second.scores[match];
        bool const equivalent = score_band ? std::fabs(one - other) <= score_band
                                           : within(one, other) && within(other, one);
        if (!equivalent) return false;
    }
    return true;
}

/** Whether the files at @p first and @p second both open and hold the same bytes. */
inline bool same_file_bytes(std::string const& first, std::string const& second) {
    auto [first_file, first_status] = mapped_file_t::read(first);
    auto [second_file, second_status] = mapped_file_t::read(second);
    return succeeded(first_status) && succeeded(second_status) &&
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
 *  a manifest. The copy in the index's own residency must land matches equivalent to those
 *  @p search lands on the original, as `search(handle) → landed_matches_t`, within @p score_band
 *  when the stored rows round on the way; every loaded copy must save again, byte for byte when
 *  @p bytes says so; the view must answer the search. Every file the round trip wrote is removed.
 */
template <typename verbs_type_, typename search_type_>
std::size_t check_checkpoint_round_trip(handle<verbs_type_>& index, std::string const& stem, std::size_t parts,
                                        saved_bytes_t bytes, double score_band, search_type_&& search) {
    session_t& session = *index.session;
    std::string const path = stem + ".safegraphs";
    std::size_t failures = 0;
    usearch_outcome_t outcome {};
    verbs_type_::save(index.get(), index.executor(), bytes_view(path), &outcome);
    failures += check(outcome);
    if (failures) return failures;
    failures += check(describe(index).shards_count == parts);
    std::vector<std::string> const saved = files_starting_with(stem);
    failures += check(saved.size() == (parts == 1 ? 1 : parts + 1));
    landed_matches_t const original_matches = search(index);

    // Every loaded copy saves again under its own stem, and its files compare to the original's.
    auto const same_checkpoint = [&](std::string const& other_stem) {
        std::vector<std::string> const resaved = files_starting_with(other_stem);
        if (resaved.size() != saved.size()) return false;
        bool same = true;
        // A manifest names its parts by their own stem, so only the parts compare.
        for (std::size_t file = 0; file != saved.size(); ++file)
            if (parts == 1 || saved[file] != path) same = same && same_file_bytes(saved[file], resaved[file]);
        return same;
    };
    auto const judge_copy = [&](typename verbs_type_::index_t copy, usearch_residency_t residency,
                                std::string const& copy_stem) {
        if (!copy) return;
        handle<verbs_type_> reopened {copy, session, residency, index.geometry};
        if (residency == index.residency) {
            landed_matches_t const reloaded = search(reopened);
            bool const equivalent = equivalent_matches(reloaded, original_matches, score_band);
            if (!equivalent && reloaded.scores.size() == original_matches.scores.size()) {
                double widest = 0;
                for (std::size_t match = 0; match != reloaded.scores.size(); ++match)
                    widest = std::max(widest, std::fabs(reloaded.scores[match] - original_matches.scores[match]));
                fmt::println(stderr, "  a reloaded score moved {:.3g} against a band of {:.3g}", widest, score_band);
            }
            failures += check(equivalent);
        }
        if (residency == usearch_residency_nvme_k) return;
        std::string const copy_path = copy_stem + ".safegraphs";
        verbs_type_::save(reopened.get(), reopened.executor(), bytes_view(copy_path), &outcome);
        failures += check(outcome);
        if (bytes == saved_bytes_t::identical_k) failures += check(same_checkpoint(copy_stem));
        for (std::string const& file : files_starting_with(copy_stem)) std::remove(file.c_str());
    };

    std::vector<std::int32_t> placement;
    typename verbs_type_::index_t copy = nullptr;
    verbs_type_::load_ram(&copy, index.residency == usearch_residency_vram_k ? nullptr : session.executor,
                          bytes_view(path), session.ram_options(0, placement), index.geometry, &outcome);
    failures += check(outcome);
    judge_copy(copy, usearch_residency_ram_k, stem + "-ram");

    if (!session.devices.empty()) {
        copy = nullptr;
        verbs_type_::load_vram(&copy, session.executor, bytes_view(path), session.vram_options(parts, placement),
                               index.geometry, &outcome);
        failures += check(outcome);
        judge_copy(copy, usearch_residency_vram_k, stem + "-vram");
    }

    if (session.storage) {
        copy = nullptr;
        verbs_type_::view_nvme(&copy, session.storage, bytes_view(path), session.nvme_options(), index.geometry,
                               &outcome);
        failures += check(outcome);
        judge_copy(copy, usearch_residency_nvme_k, stem + "-nvme");
    }

    for (std::string const& file : saved) std::remove(file.c_str());
    return failures;
}

/** How far one coordinate of a cloud spanning @p extent may move through the two roundings a stored
 *  pose takes at @p unit_roundoff, the pack at insert and the unpack at save. */
inline double pose_rounding_band(double extent, double unit_roundoff) noexcept { return 2 * unit_roundoff * extent; }

/**
 *  @brief Answers one failure per coordinate of @p reloaded outside the rounding band of its twin
 *      in @p original, and one more when their lengths differ.
 *
 *  Clouds sit end to end, cloud @c i spanning coordinates `offsets[i]` up to `offsets[i + 1]`, and
 *  each one's band follows its own extent under the storage dtype's @p unit_roundoff.
 */
inline std::size_t expect_poses_within_rounding(std::span<float const> original, std::span<float const> reloaded,
                                                std::span<std::size_t const> offsets, double unit_roundoff) {
    std::size_t failures = check(original.size() == reloaded.size() && !offsets.empty() &&
                                 offsets.back() == original.size());
    if (failures) return failures;
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
    if (moved) fmt::println(stderr, "  {} coordinates moved past their clouds' rounding bands", moved);
    return failures + check(moved == 0);
}

#pragma endregion Checks

#pragma region Benchmarks

/** Every flag the search benchmarks of the basic modalities share, its default as the
 *  initializer. */
struct search_bench_settings_t {

    /** The distance kind as its checkpoint token spells it; empty takes the binary's default. */
    std::string_view metric_name;

    /** Neighbors-list width per node; zero derives. */
    count_t connectivity {.minimum = 0};

    /** How many candidates an insertion keeps; zero derives. */
    count_t expansion_add {.value = 256};

    /** Search breadths, each judged over the one graph the run builds. */
    counts_t expansion_search {.values = {96}, .size = 1};

    /** Matches per query to return and judge. */
    count_t top_k {.value = 10};

    /** Cap on ingested entries; absent ingests every entry the queries leave. */
    count_t max_base_entries {.minimum = 0};

    /** Entries held out of the file's tail as queries. */
    count_t queries {.value = 100000};

    /** Queries the exact scan judges. */
    count_t oracle_queries {.value = 256};

    /** Entries or queries per insert or search call. */
    count_t batch {.value = 65536};
};

/** The flags @ref search_bench_settings_t holds. */
inline std::array<flag_t, 9> search_bench_flags(search_bench_settings_t& search) noexcept {
    return {{
        {"--metric", "name", "distance kind, by its checkpoint token", &search.metric_name},
        {"--connectivity", "degree", "graph degree; derived by default", &search.connectivity},
        {"--expansion-add", "breadth", "insertion candidates; 256 by default", &search.expansion_add},
        {"--expansion-search", "breadths", "search breadths, like 32,64,128; 96 by default", &search.expansion_search},
        {"--top-k", "count", "matches per query to judge; 10 by default", &search.top_k},
        {"--max-base-entries", "count", "cap on ingested entries; every entry the queries leave by default",
         &search.max_base_entries},
        {"--queries", "count", "entries held out of the file's tail as queries; 100,000 by default", &search.queries},
        {"--oracle-queries", "count", "queries the exact scan judges; 256 by default", &search.oracle_queries},
        {"--batch", "entries", "entries per insert or search call; 65536 by default", &search.batch},
    }};
}

/**
 *  @brief Builds one index of @p geometry over the first @p base_count entries of @p entries and
 *      searches @p queries_count @p queries against one exact scan at every breadth.
 *
 *  The entries are keyed by their rows, and the index is served from @p checkpoint in the session's
 *  residency. An NVMe session serves a view of what it built from @p checkpoint, or from the
 *  temporary directory when that is empty, and any other session reopens what it saved only when
 *  @p checkpoint names a path; the reopened index takes @p served, so a search may score under
 *  other weights than the build. Each breadth's matches go to @p on_swept as
 *  `(expansion, matches)`.
 *
 *  @return The process exit code.
 */
template <typename verbs_type_, typename swept_type_ = dummy_progress_t>
int run_search_bench(settings_t const& settings, search_bench_settings_t const& search,
                     typename verbs_type_::geometry_t const& geometry, typename verbs_type_::geometry_t const& served,
                     typename verbs_type_::inputs_t entries, std::size_t base_count,
                     typename verbs_type_::queries_t queries, std::size_t queries_count, std::string_view checkpoint,
                     swept_type_&& on_swept = {}) {
    session_t session = open_session(settings, settings.cpu_class);
    if (!session) return 1;
    handle<verbs_type_> index = open_index<verbs_type_>(session, geometry, settings.shards.value);
    if (!index) return 1;
    fmt::println("- Base entries: {} · queries: {} · top-k: {}", base_count, queries_count, search.top_k.value);

    usearch_traversal_stats_t built {};
    std::chrono::steady_clock::time_point const build_start = std::chrono::steady_clock::now();
    if (report_failure(
            insert_entries(index, entries, base_count, 0, search.batch.value, search.expansion_add.value, built),
            "Inserting"))
        return 1;
    double const build_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - build_start).count();
    fmt::println("- Build: {} entries in {:.2f} s · {:.0f}/s · {:.0f} distances per entry", base_count, build_seconds,
                 static_cast<double>(base_count) / build_seconds,
                 static_cast<double>(built.computed_distances) / static_cast<double>(base_count));
    print_fill(index);

    std::string served_path(checkpoint);
    if (served_path.empty() && settings.residency == usearch_residency_nvme_k)
        served_path = (std::filesystem::temp_directory_path() /
                       fmt::format("usearch-bench-{}.safegraphs", verbs_type_::name_k))
                          .string();
    if (!served_path.empty()) {
        index.geometry = served;
        index = save_and_reopen(std::move(index), served_path, settings.residency);
        if (!index) return 1;
        fmt::println("- Reopened {} as {}", served_path, usearch_residency_name(settings.residency));
    }

    landed_matches_t oracle;
    std::size_t const judged = std::min(search.oracle_queries.value, queries_count);
    std::chrono::steady_clock::time_point const scan_start = std::chrono::steady_clock::now();
    if (report_failure(scan_oracle(index, queries, judged, search.top_k.value, search.batch.value, oracle), "Scanning"))
        return 1;
    fmt::println("- Exact scan: {} queries in {:.2f} s", judged,
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - scan_start).count());
    usearch_outcome_t const swept = sweep_recall(index, queries, queries_count, oracle, search.expansion_search.view(),
                                                 search.top_k.value, search.batch.value, on_swept);
    if (checkpoint.empty() && !served_path.empty()) {
        index = {};
        for (std::string const& file : files_starting_with(served_path.substr(0, served_path.rfind(".safegraphs"))))
            std::remove(file.c_str());
    }
    return report_failure(swept, "Searching") ? 1 : 0;
}

#pragma endregion Benchmarks

} // namespace unum::usearch::tools
