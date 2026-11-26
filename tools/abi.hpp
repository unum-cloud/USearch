/**
 *  @file tools/abi.hpp
 *  @author Ash Vardanian
 *  @date October 25, 2026
 *  @brief What the suites, benchmarks and tools share over the C interface: flags and settings,
 *      keys minted from offsets, sessions of executors and arenas, owned index handles, checkpoint
 *      round trips, mapped datasets, and recall against an exact scan.
 *
 *  Every helper reaches the library through its C verbs alone, and reads GPU outputs back through
 *  the vendor's runtime, so a suite or a benchmark built on this header exercises exactly what a
 *  binding does. Entries travel as opaque version-4 UUIDs minted from the seed by
 *  @ref key_of_offset, and come back through @ref offset_of_key, which refuses a key it never
 *  minted.
 *
 *  Settings come from the environment first and the command line second:
 *
 *  @verbatim
 *  Variable            Default          Meaning
 *  USEARCH_SEED        42               Seed every scenario's data derives from, an unsigned integer or random
 *  USEARCH_FILTER      none             Checks to run, by name, as an ECMAScript regex or else as a substring
 *  USEARCH_DEVICES     first of each    Comma-separated USearch device ordinals the GPU work runs on
 *  USEARCH_NVME_PATHS  none             Comma-separated paths or transport IDs an NVMe view serves parts from
 *  --filter            none             Checks to run, overriding USEARCH_FILTER
 *  --residency         ram              Where an index lives: ram, vram or nvme
 *  --cpu-class-ceiling none             The newest CPU class to dispatch as, like intel-icelake or serial
 *  --gpu-class-ceiling none             The newest GPU class to dispatch as, like nvidia-ampere
 *  --nvme-class        derived          The NVMe class to submit through, like posix, linux-uring or linux-spdk
 *  --threads           derived          Worker threads of a RAM or NVMe executor
 *  --shards            one per domain   Shards each index opens on, over the memory domains or the devices
 *  --shard-capacity    unbounded        Entries one shard holds before another shard generation opens
 *  --checkpoint        none             File or device address to save a built index to and reopen it from
 *  --huge-pages        0                Free 1 GiB huge pages the host must hold at launch, enforced
 *  --nvme-ram          0                RAM an NVMe view's key table may take, like 24G; zero holds none
 *  @endverbatim
 *
 *  Every flag also reads its default from a @c RETRIEVAL_ variable named after it, the one
 *  RetriEval reads, so @c --top-k defaults to @c RETRIEVAL_TOP_K and the command line overrides it.
 */
#pragma once

#include <cmath>   // `std::fabs`, `std::log2`, `std::sqrt` - the score tolerance, the ranking discounts, rooted truths
#include <cstddef> // `std::size_t`
#include <cstdint> // `std::int32_t`, `std::uint32_t`, `std::uint64_t`
#include <cstdio>  // `std::remove`, `stdout`, `std::fopen` - the round-tripped checkpoints, the terminal probe, io.stat
#include <cstdlib> // `std::exit`, `std::strtod`, `std::getenv`
#include <cstring> // `std::memcpy`, `std::memcmp` - the counters, the keys

#if !USEARCH_OS_WINDOWS_
#include <unistd.h> // `isatty`
#endif

#include <algorithm>        // `std::equal`, `std::find`, `std::minmax_element`, `std::nth_element`, `std::sort`
#include <array>            // `std::array`
#include <bit>              // `std::bit_cast` - keys between the C struct and `uuid128_t`
#include <charconv>         // `std::from_chars`
#include <chrono>           // `std::chrono::milliseconds`, `std::chrono::steady_clock`
#include <filesystem>       // `std::filesystem::temp_directory_path` - where an NVMe benchmark saves
#include <initializer_list> // `std::initializer_list` - the flag groups a parse joins
#include <limits>           // `std::numeric_limits`
#include <optional>         // `std::optional`
#include <ranges>           // `std::views::transform` - how each NVMe path reads
#include <source_location>  // `std::source_location` - where a check failed
#include <span>             // `std::span`
#include <string_view>      // `std::string_view`
#include <system_error>     // `std::errc`
#include <type_traits>      // `std::is_same_v` - the metadata a geometry carries
#include <utility>          // `std::exchange`, `std::move`

#include <fmt/format.h>
#include <fmt/ranges.h> // `fmt::join` - the devices, the shard sizes

#include <forkunion/forkunion.h> // `fu_topology_new`, `fu_memory_domain_id_at_index` - the domains shards repeat over
#include <smashtable/basic_vector.hpp> // `st::vector`
#include <stringzilla/stringzilla.hpp> // `sz::split`

#include "usearch/usearch.h"
#include "usearch/utilities/blocking_files.hpp" // `mapped_file_t` - the dataset parts, the compared checkpoints
#include "usearch/utilities/nvme_device.hpp"    // `file_medium_t`, `block_bytes_of`, `reads_directly` - views' reads
#include "usearch/utilities/types.hpp"          // `uuid128_t`, `u64_t`
#include "usearch/utilities/vram_device.hpp"    // `cuda_device_of`, `hip_device_of` - device outputs read back

#include "arguments.hpp" // `flag_t`, `count_t`, `parse_or_exit`, `bytes_view`, `formatted`, `exit_out_of_memory`

namespace unum::usearch::tools {

#pragma region Keys

/** Offsets below this bound get keys: a version-4 UUID leaves 62 bits beside its variant. */
inline constexpr util::u64_t key_offsets_k = util::u64_t {1} << 62;

/** The odd multipliers of the offset mix, SplitMix64's. */
inline constexpr util::u64_t key_multipliers_k[2] = {0xbf58476d1ce4e5b9ull, 0x94d049bb133111ebull};

/** The inverse of the odd @p multiplier modulo 2^64, by Newton's iteration. */
constexpr util::u64_t inverse_of_(util::u64_t multiplier) noexcept {
    util::u64_t inverse = multiplier;
    for (int step = 0; step != 5; ++step) inverse *= 2 - multiplier * inverse;
    return inverse;
}

/** SplitMix64's finalizer. */
constexpr util::u64_t mix_(util::u64_t word) noexcept {
    word = (word ^ word >> 30) * key_multipliers_k[0];
    word = (word ^ word >> 27) * key_multipliers_k[1];
    return word ^ word >> 31;
}

/** The leading eight bytes of a key whose trailing eight are @p trailing : a version-4 tag over a
 *  mix of those and @p seed, which @ref offset_of_key checks. */
constexpr util::u64_t leading_word_(util::u64_t seed, util::u64_t trailing) noexcept {
    return (mix_(trailing ^ mix_(seed)) & ~util::u64_t {0xF000}) | 0x4000;
}

/**
 *  @brief The key the harness mints for @p offset under @p seed : a random-looking version-4 UUID
 *      with no table behind it.
 *
 *  The trailing eight bytes hold the variant over an invertible mix of the offset within 62 bits,
 *  and the leading eight a mix of them, so @ref offset_of_key inverts the first and checks the
 *  second.
 */
inline usearch_key_t key_of_offset(util::u64_t seed, util::u64_t offset) noexcept {
    util::u64_t mixed = (offset ^ mix_(seed)) & (key_offsets_k - 1);
    for (util::u64_t const multiplier : key_multipliers_k)
        mixed = (mixed ^ mixed >> 31) * multiplier & (key_offsets_k - 1);
    util::u64_t const trailing = util::u64_t {2} << 62 | (mixed ^ mixed >> 31);
    util::u64_t const leading = leading_word_(seed, trailing);
    usearch_key_t key;
    for (std::size_t byte = 0; byte != 8; ++byte) {
        key.bytes[7 - byte] = static_cast<usearch_byte_t>(leading >> (byte * 8));
        key.bytes[15 - byte] = static_cast<usearch_byte_t>(trailing >> (byte * 8));
    }
    return key;
}

/** The offset @ref key_of_offset minted @p key for under @p seed , or none for a key it never
 *  minted. */
inline std::optional<util::u64_t> offset_of_key(util::u64_t seed, usearch_key_t const& key) noexcept {
    util::u64_t leading = 0, trailing = 0;
    for (std::size_t byte = 0; byte != 8; ++byte)
        leading = leading << 8 | key.bytes[byte], trailing = trailing << 8 | key.bytes[8 + byte];
    if (trailing >> 62 != 2 || leading != leading_word_(seed, trailing)) return std::nullopt;
    util::u64_t mixed = trailing & (key_offsets_k - 1);
    mixed ^= mixed >> 31;
    for (std::size_t round = std::size(key_multipliers_k); round--;) {
        mixed = mixed * inverse_of_(key_multipliers_k[round]) & (key_offsets_k - 1);
        mixed ^= mixed >> 31;
    }
    return (mixed ^ mix_(seed)) & (key_offsets_k - 1);
}

/** The packed run of @p keys as a batch names its identities. */
inline usearch_keys_view_t keys_view(std::span<usearch_key_t const> keys) noexcept {
    return {keys.data(), sizeof(usearch_key_t)};
}

/** @p bytes as the C interface views a buffer. */
inline usearch_bytes_view_t bytes_view(util::bytes_view_t bytes) noexcept {
    return {reinterpret_cast<usearch_byte_t const*>(bytes.data()), bytes.size()};
}

#pragma endregion Keys

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

/** The USearch device ordinals @c USEARCH_DEVICES lists, comma-separated, as positions in
 *  @c usearch_gpu_devices; none when it is unset or empty, and exits with status 1 on text that
 *  does not parse. */
inline st::vector<std::int32_t> devices_from_environment() noexcept {
    st::vector<std::int32_t> devices;
    std::optional<std::string_view> const text = environment_text("USEARCH_DEVICES");
    if (!text) return devices;
    for (std::string_view const item : sz::split(*text, ",")) {
        std::uint32_t ordinal = 0;
        auto const [end, parse_status] = std::from_chars(item.data(), item.data() + item.size(), ordinal);
        if (item.empty() || parse_status != std::errc {} || end != item.data() + item.size() ||
            ordinal > static_cast<std::uint32_t>((std::numeric_limits<std::int32_t>::max)()))
            exit_unparsed("USEARCH_DEVICES", *text, "unsigned integers like 0,1");
        if (st::failed(devices.push_back(static_cast<std::int32_t>(ordinal)))) exit_out_of_memory("the devices");
    }
    return devices;
}

/** The names @c --residency takes, the library's own after "derived", in the order of
 *  @c residencies_k. */
inline constexpr std::array<std::string_view, 4> residency_names_k {"derived", "ram", "vram", "nvme"};

/** The residency each name of @c residency_names_k selects; "derived" leaves the verb to pick
 *  one. */
inline constexpr std::array<usearch_residency_t, 4> residencies_k {usearch_residency_unknown_k, usearch_residency_ram_k,
                                                                   usearch_residency_vram_k, usearch_residency_nvme_k};

/** The names @c --cpu-class-ceiling takes, the library's own after "none", in the order of
 *  @c cpu_class_ceilings_k. */
inline constexpr std::array<std::string_view, 20> cpu_class_ceiling_names_k {
    "none",      "serial",         "monoversioned", "intel-haswell", "intel-sierra", "intel-skylake",   "intel-icelake",
    "amd-genoa", "intel-sapphire", "intel-granite", "intel-diamond", "aws-g2",       "apple-m2",        "apple-m4",
    "aws-g5",    "nvidia-vera",    "alibaba-c930",  "ibm-power10",   "wasm32-v128",  "wasm64-extended",
};

/** The ceiling each name of @c cpu_class_ceiling_names_k selects. */
inline constexpr std::array<usearch_cpu_class_t, 20> cpu_class_ceilings_k {
    usearch_cpu_class_unknown_k,       usearch_cpu_class_serial_k,          usearch_cpu_class_monoversioned_k,
    usearch_cpu_class_intel_haswell_k, usearch_cpu_class_intel_sierra_k,    usearch_cpu_class_intel_skylake_k,
    usearch_cpu_class_intel_icelake_k, usearch_cpu_class_amd_genoa_k,       usearch_cpu_class_intel_sapphire_k,
    usearch_cpu_class_intel_granite_k, usearch_cpu_class_intel_diamond_k,   usearch_cpu_class_aws_g2_k,
    usearch_cpu_class_apple_m2_k,      usearch_cpu_class_apple_m4_k,        usearch_cpu_class_aws_g5_k,
    usearch_cpu_class_nvidia_vera_k,   usearch_cpu_class_alibaba_c930_k,    usearch_cpu_class_ibm_power10_k,
    usearch_cpu_class_wasm32_v128_k,   usearch_cpu_class_wasm64_extended_k,
};

/** The names @c --gpu-class-ceiling takes, the library's own after "none", in the order of
 *  @c gpu_class_ceilings_k. */
inline constexpr std::array<std::string_view, 13> gpu_class_ceiling_names_k {
    "none",          "nvidia-ampere", "nvidia-ada", "nvidia-hopper", "nvidia-blackwell", "nvidia-blackwellrtx",
    "nvidia-rubin",  "amd-cdna3",     "amd-cdna4",  "amd-cdna5",     "apple-metal7",     "apple-metal9",
    "apple-metal10",
};

/** The ceiling each name of @c gpu_class_ceiling_names_k selects. */
inline constexpr std::array<usearch_gpu_class_t, 13> gpu_class_ceilings_k {
    usearch_gpu_class_unknown_k,       usearch_gpu_class_nvidia_ampere_k,    usearch_gpu_class_nvidia_ada_k,
    usearch_gpu_class_nvidia_hopper_k, usearch_gpu_class_nvidia_blackwell_k, usearch_gpu_class_nvidia_blackwellrtx_k,
    usearch_gpu_class_nvidia_rubin_k,  usearch_gpu_class_amd_cdna3_k,        usearch_gpu_class_amd_cdna4_k,
    usearch_gpu_class_amd_cdna5_k,     usearch_gpu_class_apple_metal7_k,     usearch_gpu_class_apple_metal9_k,
    usearch_gpu_class_apple_metal10_k,
};

/** The names @c --nvme-class takes, the library's own after "derived", in the order of
 *  @c nvme_classes_k. */
inline constexpr std::array<std::string_view, 7> nvme_class_names_k {
    "derived", "posix", "linux-aio", "linux-uring", "linux-uring-passthrough", "linux-spdk", "windows-ioring",
};

/** The class each name of @c nvme_class_names_k selects. */
inline constexpr std::array<usearch_nvme_class_t, 7> nvme_classes_k {
    usearch_nvme_class_unknown_k,
    usearch_nvme_class_posix_k,
    usearch_nvme_class_linux_aio_k,
    usearch_nvme_class_linux_uring_k,
    usearch_nvme_class_linux_uring_passthrough_k,
    usearch_nvme_class_linux_spdk_k,
    usearch_nvme_class_windows_ioring_k,
};

/** The index @p table holds @p value at, which a @ref choice_t selects it by. */
template <typename value_type_, std::size_t count_>
constexpr std::size_t selected_of(std::array<value_type_, count_> const& table, value_type_ value) noexcept {
    std::size_t index = 0;
    while (index != count_ && table[index] != value) ++index;
    return index;
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
    regex_filter_t regex_filter;

    /** Where an index lives, one of @c residency_names_k. */
    choice_t residency {residency_names_k, selected_of(residencies_k, usearch_residency_ram_k)};

    /** USearch device ordinals the GPU work runs on; none takes device zero, or every vendor's
     *  first in a suite. Override: @c USEARCH_DEVICES. */
    st::vector<std::int32_t> devices;

    /** NVMe namespaces to serve from, in the grammar of @c usearch_nvme_options_t::paths; none
     *  serves every shard where its checkpoint points. Override: @c USEARCH_NVME_PATHS. */
    st::vector<std::string_view> nvme_paths;

    /** The newest CPU class to dispatch as, intersected with the CPU's own, one of
     *  @c cpu_class_ceiling_names_k; "none" sets none. */
    choice_t cpu_class_ceiling {cpu_class_ceiling_names_k};

    /** The newest GPU class to dispatch as, capping the devices' own, one of
     *  @c gpu_class_ceiling_names_k; "none" sets none. */
    choice_t gpu_class_ceiling {gpu_class_ceiling_names_k};

    /** The NVMe class to submit through, one of @c nvme_class_names_k; "derived" derives it. */
    choice_t nvme_class {nvme_class_names_k};

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

    /** Free 1 GiB huge pages the host must hold at launch; zero expects none, any other value is
     *  enforced, so a base-page fallback cannot pass as a huge-page result. */
    count_t huge_pages {.minimum = 0};

    /** RAM an NVMe view's key table may take; zero holds none, so key verbs and hints refuse. */
    bytes_t nvme_ram;

    /** Whether stdout is a terminal, so progress redraws one line instead of printing deciles. */
    bool terminal = false;
};

/** The flag every binary checking @ref huge_pages_provisioned takes. */
inline flag_t huge_pages_flag(count_t& huge_pages) noexcept {
    return {"--huge-pages", "count",
            "free 1 GiB huge pages required at launch; enforced, so base-page fallbacks fail fast", &huge_pages};
}

/** The flags a session opens with, every binary's and verb's one spelling of them. */
inline std::array<flag_t, 4> session_flags(settings_t& settings) noexcept {
    return {{
        {"--residency", "name",
         "where an index lives: ram, vram or nvme; ram by default, vram for an insert with a GPU listed, " //
         "nvme for health",                                                                                //
         &settings.residency},
        {"--nvme-class", "name",
         "NVMe class to submit through, like posix, linux-uring or linux-spdk; derived by default",
         &settings.nvme_class},
        {"--threads", "count", "worker threads; derived from the domains or devices by default", &settings.threads},
        {"--nvme-ram", "bytes", "RAM an NVMe view's key table may take, like 24G; zero, the default, holds none",
         &settings.nvme_ram},
    }};
}

/** The flags every binary reading a @ref settings_t takes. */
inline std::array<flag_t, 11> settings_flags(settings_t& settings) noexcept {
    std::array<flag_t, 4> const session = session_flags(settings);
    return {{
        {"--filter", "pattern", "checks to run, as an ECMAScript regex; every one by default", &settings.regex_filter},
        session[0],
        {"--cpu-class-ceiling", "name",
         "newest CPU class to dispatch as, like intel-icelake or serial, within what the CPU runs; none by default",
         &settings.cpu_class_ceiling},
        {"--gpu-class-ceiling", "name", "newest GPU class to dispatch as, like nvidia-ampere; none by default",
         &settings.gpu_class_ceiling},
        session[1],
        session[2],
        {"--shards", "count", "shards per index; one per memory domain or device by default", &settings.shards},
        {"--shard-capacity", "entries",
         "entries one shard holds before another shard generation opens; unbounded by default",
         &settings.shard_capacity},
        {"--checkpoint", "address",
         "file or device address, like /dev/nvme1n1@64GiB, to save the built index to and reopen",
         &settings.checkpoint},
        huge_pages_flag(settings.huge_pages),
        session[3],
    }};
}

/** The whole text of the file at @p path, none where it cannot be opened. */
inline std::optional<sz::string_t> read_text(char const* path) noexcept {
    std::FILE* const file = std::fopen(path, "r");
    if (!file) return std::nullopt;
    sz::string_t text;
    std::array<char, 4096> chunk;
    for (std::size_t read = 0; (read = std::fread(chunk.data(), 1, chunk.size(), file)) != 0;)
        if (text.append(chunk.data(), read) != sz::status_t::success_k) exit_out_of_memory(path);
    std::fclose(file);
    return text;
}

/** Whether the host's 1 GiB pool holds the @p expected free pages a run declares, which the
 *  launcher provisions outside the process; prints why not. */
inline bool huge_pages_provisioned(std::size_t expected) {
    if (!expected) return true;
    std::optional<sz::string_t> const text = read_text("/sys/kernel/mm/hugepages/hugepages-1048576kB/free_hugepages");
    std::size_t free_pages = 0;
    if (text) std::from_chars(text->data(), text->data() + text->size(), free_pages);
    if (free_pages < expected)
        fmt::println("--huge-pages expects {} free 1 GiB pages, but the host offers {}", expected, free_pages);
    return free_pages >= expected;
}

/** Reads the settings the environment holds, exiting with status 1 on the first that does not
 *  parse; calls nothing of the library, so header-only suites read it too. */
inline settings_t read_environment() {
    settings_t settings;
    if (std::optional<std::string_view> const filter_text = environment_text("USEARCH_FILTER"))
        if (filter_of(*filter_text, settings.regex_filter) != argument_status_t::parsed_k)
            exit_unparsed("USEARCH_FILTER", *filter_text, "an ECMAScript regex");
    if (std::optional<std::string_view> const seed_text = environment_text("USEARCH_SEED")) {
        std::optional<std::uint32_t> const seed = parse_seed(*seed_text);
        if (!seed) exit_unparsed("USEARCH_SEED", *seed_text, "an unsigned integer or random");
        settings.seed = *seed;
    }
    settings.devices = devices_from_environment();
    if (std::optional<std::string_view> const paths_text = environment_text("USEARCH_NVME_PATHS"))
        for (std::string_view const path : sz::split(*paths_text, ",")) {
            if (path.empty())
                exit_unparsed("USEARCH_NVME_PATHS", *paths_text,
                              "comma-separated files, block devices or transport IDs");
            if (st::failed(settings.nvme_paths.push_back(std::string_view {path})))
                exit_out_of_memory("the NVMe paths");
        }
#if !USEARCH_OS_WINDOWS_
    settings.terminal = isatty(fileno(stdout)) != 0;
#endif
    return settings;
}

/** Every flag of @p groups in turn, as the one table a parse takes. */
inline st::vector<flag_t> joined_flags(std::initializer_list<std::span<flag_t const>> groups) noexcept {
    st::vector<flag_t> flags;
    for (std::span<flag_t const> const group : groups)
        for (flag_t const& flag : group)
            if (st::failed(flags.push_back(flag_t {flag}))) exit_out_of_memory("the flags");
    return flags;
}

/** Reads the environment, then parses @p argv against the shared flags and @p binary_flags. On
 *  <tt>--help</tt> prints the usage and exits with status 0; on a setting that does not parse, or huge
 *  pages the host does not have, prints one line and exits with status 1. */
inline settings_t read_settings(int argc, char const* const* argv, std::string_view synopsis,
                                std::span<flag_t const> binary_flags = {}) {
    settings_t settings = read_environment();
    std::array<flag_t, 11> const shared = settings_flags(settings);
    st::vector<flag_t> const flags = joined_flags({binary_flags, shared});
    parse_or_exit(argc, argv, synopsis, flags);
    if (!huge_pages_provisioned(settings.huge_pages.value)) std::exit(1);
    return settings;
}

/** How a view reads @p path: "direct" past the page cache, "buffered" through it, or "unopened"
 *  before it opens; a transport ID reads directly and stays closed, as SPDK starts once. */
inline std::string_view nvme_reads_name(std::string_view path) noexcept {
    if (util::names_transport_id(path)) return "direct";
    auto const [medium, open_status] = util::file_medium_t::open(path);
    if (util::failed(open_status)) return "unopened";
    return util::reads_directly(medium) ? "direct" : "buffered";
}

/** Prints each setting @ref read_environment reads as "- Name: value", in the grammar it parses
 *  from, and how views read each NVMe path; calls nothing of the library, so header-only suites
 *  print it too. */
inline void print_environment(settings_t const& settings) {
    fmt::println("- Seed: {}", settings.seed);
    fmt::println("- Filter: {}",
                 settings.regex_filter.text.empty() ? std::string_view("none") : settings.regex_filter.text);
    if (settings.devices.empty()) fmt::println("- Devices: derived");
    else fmt::println("- Devices: {}", fmt::join(settings.devices, ","));
    if (settings.nvme_paths.empty()) return fmt::println("- NVMe paths: none");
    fmt::println("- NVMe paths: {}", fmt::join(settings.nvme_paths, ","));
    fmt::println("- NVMe reads: {}", fmt::join(settings.nvme_paths | std::views::transform(nvme_reads_name), ","));
}

/** Prints each setting as "- Name: value", in the grammar it parses from. */
inline void print(settings_t const& settings) {
    print_environment(settings);
    fmt::println("- Residency: {}", residency_names_k[settings.residency.selected]);
    // The index's "- Index:" line names the class dispatch picked under these ceilings.
    fmt::println("- CPU class ceiling: {}", cpu_class_ceiling_names_k[settings.cpu_class_ceiling.selected]);
    fmt::println("- GPU class ceiling: {}", gpu_class_ceiling_names_k[settings.gpu_class_ceiling.selected]);
    fmt::println("- NVMe class: {}", usearch_nvme_class_name(nvme_classes_k[settings.nvme_class.selected]));
}

#pragma endregion Settings

#pragma region Sessions

/** Prints a failed call's status and message after the verb @p verb formats from @p args; true when
 *  it failed. */
template <typename... args_types_>
bool report_failure(usearch_outcome_t const& outcome, fmt::format_string<args_types_ const&...> verb,
                    args_types_ const&... args) {
    if (outcome.status == usearch_status_success_k) return false;
    fmt::println(stderr, "{} failed with {}: {}", formatted(verb, args...), usearch_status_name(outcome.status),
                 outcome.message ? outcome.message : "no message");
    return true;
}

/** Every GPU this process drives, in USearch ordinal order. */
inline st::vector<usearch_gpu_device_t> listed_devices() noexcept {
    usearch_outcome_t outcome {};
    std::size_t count = 0;
    usearch_gpu_devices_t counting {.outcome = &outcome,
                                    .devices = nullptr,
                                    .devices_capacity = 0,
                                    .devices_count = &count};
    usearch_gpu_devices(&counting);
    st::vector<usearch_gpu_device_t> devices;
    if (st::failed(devices.resize(count))) exit_out_of_memory("the devices");
    usearch_gpu_devices_t listing {.outcome = &outcome,
                                   .devices = devices.data(),
                                   .devices_capacity = count,
                                   .devices_count = &count};
    usearch_gpu_devices(&listing);
    if (st::failed(devices.resize(util::min_of(count, devices.size())))) exit_out_of_memory("the devices");
    return devices;
}

/**
 *  @brief The executors and the arena one residency's calls run on, freed together.
 *
 *  A RAM session builds and serves on one RAM executor. A VRAM session builds and serves on the
 *  executor of its devices, writing outputs to an arena on the first of them, and runs its RAM
 *  copies on a RAM executor of its own. An NVMe session builds on a RAM executor and serves views
 *  on an NVMe executor of its own, so a built index is saved and viewed before any search.
 */
struct session_t {
    usearch_residency_t residency = usearch_residency_ram_k;

    /** The seed every key this session's entries take is minted under, by @ref key_of_offset. */
    std::uint32_t seed = 0;

    /** The newest classes every index this session opens may dispatch as. */
    usearch_cpu_class_t cpu_class_ceiling = usearch_cpu_class_unknown_k;
    usearch_gpu_class_t gpu_class_ceiling = usearch_gpu_class_unknown_k;

    /** The USearch device ordinals a VRAM index places shards on, in turn. */
    st::vector<std::int32_t> devices;

    /** The NVMe namespaces a save places its parts on, one per part, viewed from the settings. */
    st::vector<std::string_view> places;

    /** What views read those parts through: the places themselves, or under passthrough each block
     *  node's generic character node. */
    st::vector<sz::string_t> served;

    /** The served namespaces as the C interface views them. */
    st::vector<usearch_bytes_view_t> paths;

    /** The memory domains RAM shards repeat over, in turn. */
    st::vector<std::int32_t> memory_domains;

    /** Entries one shard holds before another shard generation opens; zero leaves it unbounded. */
    std::size_t shard_capacity = 0;

    /** The executor building every index: RAM threads, or the devices' streams. */
    usearch_executor_t executor = nullptr;

    /** The NVMe executor views serve on; null outside an NVMe session. */
    usearch_executor_t storage = nullptr;

    /** The RAM executor a VRAM session's RAM copies run on; null outside a VRAM session. */
    usearch_executor_t host = nullptr;

    /** The RAM every view's key table may take, @c memory_limit zero holding none. */
    usearch_ram_options_t nvme_ram {};

    /** The RAM options a view takes: none while @c nvme_ram holds nothing. */
    usearch_ram_options_t const* nvme_ram_options() const noexcept {
        return nvme_ram.memory_limit ? &nvme_ram : nullptr;
    }

    /** Where every call's outputs land: host pages, or the first device of a VRAM session. */
    usearch_arena_t arena = nullptr;

    /** Where the outputs of a VRAM session's RAM copies land; null outside a VRAM session. */
    usearch_arena_t host_arena = nullptr;

    session_t() = default;
    session_t(session_t const&) = delete;
    session_t& operator=(session_t const&) = delete;
    session_t(session_t&& other) noexcept { swap(other); }
    session_t& operator=(session_t&& other) noexcept { return swap(other), *this; }
    ~session_t() noexcept {
        usearch_arena_free(host_arena);
        usearch_arena_free(arena);
        usearch_executor_free(host);
        usearch_executor_free(storage);
        usearch_executor_free(executor);
    }

    void swap(session_t& other) noexcept {
        std::swap(residency, other.residency), std::swap(seed, other.seed);
        std::swap(cpu_class_ceiling, other.cpu_class_ceiling), std::swap(gpu_class_ceiling, other.gpu_class_ceiling);
        std::swap(devices, other.devices);
        std::swap(places, other.places), std::swap(served, other.served), std::swap(paths, other.paths);
        std::swap(memory_domains, other.memory_domains);
        std::swap(shard_capacity, other.shard_capacity), std::swap(executor, other.executor);
        std::swap(storage, other.storage), std::swap(host, other.host);
        std::swap(arena, other.arena), std::swap(host_arena, other.host_arena);
    }

    /** Whether every executor and the arena the residency needs opened. */
    explicit operator bool() const noexcept {
        return executor && arena && (residency != usearch_residency_nvme_k || storage) &&
               (residency != usearch_residency_vram_k || (host && host_arena));
    }

    /** The executor serving an index of @p index_residency : the storage one for a view, and the
     *  host one for a RAM copy of a VRAM session's index. */
    usearch_executor_t executor_for(usearch_residency_t index_residency) const noexcept {
        if (index_residency == usearch_residency_nvme_k) return storage;
        return index_residency == usearch_residency_ram_k && residency == usearch_residency_vram_k ? host : executor;
    }

    /** The arena serving an index of @p index_residency , which @ref executor_for pairs it with. */
    usearch_arena_t arena_for(usearch_residency_t index_residency) const noexcept {
        return index_residency == usearch_residency_ram_k && residency == usearch_residency_vram_k ? host_arena : arena;
    }

    /** The RAM placement of @p shards shards over the session's memory domains in turn, backed by
     *  @p domains; zero shards names no domains, so the library derives one shard per domain. */
    usearch_ram_options_t ram_options(std::size_t shards, st::vector<std::int32_t>& domains) const noexcept {
        if (st::failed(domains.resize(shards))) exit_out_of_memory("the placement");
        for (std::size_t shard = 0; shard != shards; ++shard)
            domains[shard] = memory_domains[shard % memory_domains.size()];
        return {domains.empty() ? nullptr : domains.data(), domains.size(), shard_capacity, cpu_class_ceiling};
    }

    /** The VRAM placement of @p shards shards over the session's devices in turn, backed by
     *  @p ordinals; zero shards takes one per device. */
    usearch_vram_options_t vram_options(std::size_t shards, st::vector<std::int32_t>& ordinals) const noexcept {
        std::size_t const count = shards ? shards : devices.size();
        if (st::failed(ordinals.resize(count))) exit_out_of_memory("the placement");
        for (std::size_t shard = 0; shard != count; ++shard) ordinals[shard] = devices[shard % devices.size()];
        return {ordinals.data(), ordinals.size(), shard_capacity, gpu_class_ceiling};
    }

    /** The NVMe placement views serve from. */
    usearch_nvme_options_t nvme_options() const noexcept {
        return {paths.empty() ? nullptr : paths.data(), paths.size(), cpu_class_ceiling};
    }
};

/** The generic character node serving the namespace at @p address, a block node of the kernel's
 *  NVMe driver, at the same base and length; empty for any other address. */
inline sz::string_t character_sibling(std::string_view address) noexcept {
    // The kernel names a namespace's character node after its block node: nvmeXnY and ngXnY.
    std::string_view const block_prefix = "/dev/nvme";
    if (!address.starts_with(block_prefix)) return {};
    return formatted("/dev/ng{}", address.substr(block_prefix.size()));
}

/** Opens the executors and the arena of @p settings ' residency, every index of the session
 *  dispatching on the CPU under @p cpu_class_ceiling; prints what failed and answers a session
 *  that tests false. */
inline session_t open_session(settings_t const& settings, usearch_cpu_class_t cpu_class_ceiling) {
    session_t session;
    session.residency = residencies_k[settings.residency.selected];
    session.seed = settings.seed;
    session.cpu_class_ceiling = cpu_class_ceiling;
    session.gpu_class_ceiling = gpu_class_ceilings_k[settings.gpu_class_ceiling.selected];
    session.shard_capacity = settings.shard_capacity.value;
    session.nvme_ram.memory_limit = settings.nvme_ram.value;
    fu_topology_t topology = nullptr;
    std::size_t domains_count = 0;
    if (fu_topology_new(&topology) == fu_success_k && fu_memory_domains_count(topology, &domains_count) == fu_success_k)
        for (std::size_t index = 0; index != domains_count; ++index) {
            fu_memory_domain_id_t domain = 0;
            if (fu_memory_domain_id_at_index(topology, index, &domain) == fu_success_k &&
                st::failed(session.memory_domains.push_back(static_cast<std::int32_t>(domain))))
                exit_out_of_memory("the memory domains");
        }
    if (topology) fu_topology_delete(topology);
    if (session.memory_domains.empty() && st::failed(session.memory_domains.push_back(0)))
        exit_out_of_memory("the memory domains");
    // Parts are written through block nodes, and passthrough reads them through character nodes.
    bool const passthrough = nvme_classes_k[settings.nvme_class.selected] ==
                             usearch_nvme_class_linux_uring_passthrough_k;
    for (std::string_view const place : settings.nvme_paths) {
        sz::string_t sibling = character_sibling(place);
        if (!passthrough || sibling.empty()) sibling = formatted("{}", place);
        if (st::failed(session.places.push_back(std::string_view {place})) ||
            st::failed(session.served.push_back(std::move(sibling))))
            exit_out_of_memory("the NVMe paths");
    }
    for (sz::string_t const& path : session.served)
        if (st::failed(session.paths.push_back(bytes_view(path)))) exit_out_of_memory("the NVMe paths");
    usearch_outcome_t outcome {};
    if (session.residency == usearch_residency_vram_k) {
        st::vector<usearch_gpu_device_t> const listed = listed_devices();
        if (st::failed(settings.devices.empty() ? session.devices.push_back(0)
                                                : session.devices.assign(settings.devices)))
            exit_out_of_memory("the devices");
        for (std::int32_t const ordinal : session.devices)
            if (static_cast<std::size_t>(ordinal) >= listed.size()) {
                fmt::println(stderr, "Device {} is past the {} listed", ordinal, listed.size());
                return session;
            }
        st::vector<std::int32_t> ordinals;
        usearch_executor_for_vram_t placement {.executor = &session.executor,
                                               .outcome = &outcome,
                                               .vram = session.vram_options(0, ordinals)};
        usearch_executor_for_vram(&placement);
    }
    else {
        usearch_executor_for_ram_t placement {.executor = &session.executor,
                                              .outcome = &outcome,
                                              .ram = {nullptr, 0, session.shard_capacity, cpu_class_ceiling},
                                              .threads = settings.threads.value};
        usearch_executor_for_ram(&placement);
    }
    if (report_failure(outcome, "Opening the executor")) return session;
    if (session.residency == usearch_residency_vram_k) {
        usearch_executor_for_ram_t placement {.executor = &session.host,
                                              .outcome = &outcome,
                                              .ram = {nullptr, 0, session.shard_capacity, cpu_class_ceiling},
                                              .threads = settings.threads.value};
        usearch_executor_for_ram(&placement);
        if (report_failure(outcome, "Opening the host executor")) return session;
    }
    if (session.residency == usearch_residency_nvme_k) {
        usearch_executor_for_nvme_t placement {.executor = &session.storage,
                                               .outcome = &outcome,
                                               .nvme = session.nvme_options(),
                                               .threads = settings.threads.value,
                                               .nvme_class = nvme_classes_k[settings.nvme_class.selected]};
        usearch_executor_for_nvme(&placement);
        if (report_failure(outcome, "Opening the NVMe executor")) return session;
    }
    usearch_arena_init_t arena {
        .arena = &session.arena,
        .executor = session.residency == usearch_residency_vram_k ? session.executor : nullptr,
        .outcome = &outcome,
    };
    usearch_arena_init(&arena);
    if (report_failure(outcome, "Opening the arena")) return session;
    if (session.residency == usearch_residency_vram_k) {
        usearch_arena_init_t host_arena {.arena = &session.host_arena, .executor = nullptr, .outcome = &outcome};
        usearch_arena_init(&host_arena);
        report_failure(outcome, "Opening the host arena");
    }
    return session;
}

/** The harness's stand-in for the library's GPU runtime, which the library keeps hidden: every
 *  vendor's driver table and Apple's contexts, and the kernels the harness's own executors launch,
 *  loaded once by the first harness step that touches a device. */
struct gpu_harness_t {
    util::gpu_drivers_t drivers;
    util::kernel_tables<util::cuda_module_t> cuda_kernels;
    util::kernel_tables<util::hip_module_t> hip_kernels;
    util::kernel_tables<util::metal_module_t> metal_kernels;

    /** The ceiling a harness executor's scratch grows to, zero for none. */
    std::size_t scratch_limit = 0;
};

/** The one harness GPU state, built by the first step that touches a device. */
inline gpu_harness_t& gpu_harness() noexcept {
    static gpu_harness_t harness;
    return harness;
}

/** Every vendor's driver table and Apple's contexts. */
inline util::gpu_drivers_t& gpu_drivers() noexcept { return gpu_harness().drivers; }

/** Spawns an executor of @p executor_type_ over @p devices through the harness's drivers, its
 *  streams launching the harness's kernels. */
template <typename executor_type_>
util::expected<executor_type_> spawn_harness_executor(std::span<int const> devices) noexcept {
    gpu_harness_t& harness = gpu_harness();
    return executor_type_::spawn(util::driver_of(harness.drivers, std::type_identity<executor_type_> {}), devices, {},
                                 harness, harness.scratch_limit);
}

/** Copies @p bytes of a call's output at @p source into @p destination: through the runtime of the
 *  vendor whose device memory holds it, else directly, as host memory, pinned or not, reads; false
 *  when the copy failed. */
inline bool download(void* destination, void const* source, std::size_t bytes) noexcept {
    if (!bytes) return true;
    util::gpu_drivers_t const& drivers = gpu_drivers();
    if (util::cuda_device_of(drivers.cuda, source) >= 0)
        return util::cuda_copy(drivers.cuda, destination, source, bytes) == util::cuda_success_k;
    if (util::hip_device_of(drivers.hip, source) >= 0)
        return drivers.hip.copy(destination, source, bytes, util::hip_copy_device_to_host_k) == util::hip_success_k;
    std::memcpy(destination, source, bytes);
    return true;
}

#pragma endregion Sessions

#pragma region Verbs

/** Where a sample lands, every view inside the call's arena. */
struct sample_outputs_view_t {
    usearch_keys_view_t keys {};
    std::size_t const* entry_ordinals = nullptr;
    usearch_matches_view_t parents {};
    usearch_rows_view_t coordinates {};
    std::size_t written = 0;
    usearch_traversal_stats_t stats {};
};

/** The C functions of the dense modality, behind the signatures every modality shares. */
struct dense_verbs_t {
    using index_t = usearch_dense_index_t;
    using description_t = usearch_dense_index_description_t;
    using geometry_t = usearch_dense_metadata_t;
    using metadata_t = usearch_dense_metadata_t;
    using metric_t = usearch_dense_metric_t;
    using values_t = usearch_rows_view_t;
    using queries_t = usearch_rows_view_t;
    static constexpr std::string_view name_k = "dense";

    static values_t slice(values_t rows, std::size_t first) noexcept {
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
                       values_t rows, std::size_t expansion, usearch_traversal_stats_t* stats,
                       usearch_outcome_t* outcome) noexcept {
        usearch_dense_insert_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome;
        arguments.count = count, arguments.keys = keys, arguments.vectors = rows;
        arguments.expansion_factor = expansion, arguments.stats = stats;
        usearch_dense_insert(&arguments);
    }
    static void search(index_t index, usearch_executor_t executor, usearch_arena_t arena, std::size_t count,
                       queries_t queries, std::size_t wanted, std::size_t expansion, usearch_traversal_t traversal,
                       usearch_matches_view_t* matches, usearch_traversal_stats_t* stats, usearch_outcome_t* outcome,
                       usearch_matches_view_t const* hints = nullptr) noexcept {
        usearch_dense_search_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome;
        arguments.count = count, arguments.queries = queries, arguments.matches = matches, arguments.arena = arena;
        arguments.wanted = wanted, arguments.expansion_factor = expansion, arguments.traversal = traversal;
        arguments.hints = hints, arguments.stats = stats;
        usearch_dense_search(&arguments);
    }
    static void save(index_t index, usearch_executor_t executor, usearch_bytes_view_t path,
                     usearch_outcome_t* outcome) noexcept {
        usearch_dense_save_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome, arguments.path = path;
        usearch_dense_save(&arguments);
    }
    static void load_ram(index_t* index, usearch_executor_t executor, usearch_bytes_view_t path,
                         usearch_ram_options_t ram, geometry_t const& geometry, usearch_outcome_t* outcome) noexcept {
        usearch_dense_load_ram_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome, arguments.path = path;
        arguments.ram = ram, arguments.connectivity = geometry.connectivity;
        usearch_dense_load_ram(&arguments);
    }
    static void save_buffer(index_t index, usearch_executor_t executor, usearch_arena_t arena,
                            usearch_bytes_view_t* monolith, usearch_outcome_t* outcome) noexcept {
        usearch_dense_save_buffer_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome;
        arguments.monolith = monolith, arguments.arena = arena;
        usearch_dense_save_buffer(&arguments);
    }
    static void load_ram_buffer(index_t* index, usearch_executor_t executor, usearch_bytes_view_t buffer,
                                usearch_ram_options_t ram, geometry_t const& geometry,
                                usearch_outcome_t* outcome) noexcept {
        usearch_dense_load_ram_buffer_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome, arguments.buffer = buffer;
        arguments.ram = ram, arguments.connectivity = geometry.connectivity;
        usearch_dense_load_ram_buffer(&arguments);
    }
    static void metadata_read(usearch_executor_t executor, usearch_bytes_view_t path, metadata_t* metadata,
                              usearch_outcome_t* outcome) noexcept {
        usearch_dense_metadata_read_t arguments {};
        arguments.executor = executor, arguments.outcome = outcome, arguments.path = path;
        arguments.metadata = metadata;
        usearch_dense_metadata_read(&arguments);
    }
    static void metadata_read_buffer(usearch_executor_t executor, usearch_bytes_view_t buffer, metadata_t* metadata,
                                     usearch_outcome_t* outcome) noexcept {
        usearch_dense_metadata_read_buffer_t arguments {};
        arguments.executor = executor, arguments.outcome = outcome, arguments.buffer = buffer;
        arguments.metadata = metadata;
        usearch_dense_metadata_read_buffer(&arguments);
    }
    static char const* metric_name(metric_t metric) noexcept { return usearch_dense_metric_name(metric); }
    static usearch_status_t metric_from_name(usearch_bytes_view_t name, metric_t* metric) noexcept {
        return usearch_dense_metric_from_name(name, metric);
    }
    static void load_vram(index_t* index, usearch_executor_t executor, usearch_bytes_view_t path,
                          usearch_vram_options_t vram, geometry_t const& geometry,
                          usearch_outcome_t* outcome) noexcept {
        usearch_dense_load_vram_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome, arguments.path = path;
        arguments.vram = vram, arguments.connectivity = geometry.connectivity;
        usearch_dense_load_vram(&arguments);
    }
    static void view_nvme(index_t* index, usearch_executor_t executor, usearch_bytes_view_t path,
                          usearch_nvme_options_t nvme, usearch_ram_options_t const* ram, geometry_t const&,
                          usearch_outcome_t* outcome) noexcept {
        usearch_dense_view_nvme_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome, arguments.path = path;
        arguments.nvme = nvme, arguments.ram = ram;
        usearch_dense_view_nvme(&arguments);
    }
    static void load_nvme(index_t* index, usearch_executor_t executor, usearch_bytes_view_t path,
                          usearch_nvme_options_t nvme, usearch_ram_options_t const* ram,
                          usearch_outcome_t* outcome) noexcept {
        usearch_dense_load_nvme_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome, arguments.path = path;
        arguments.nvme = nvme, arguments.ram = ram;
        usearch_dense_load_nvme(&arguments);
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
    using refine_plan_t = usearch_dense_refine_plan_t;
    static void refine(index_t index, usearch_executor_t executor, usearch_refine_report_t* report,
                       usearch_outcome_t* outcome, std::size_t expansion = 0, std::size_t memory_limit = 0,
                       usearch_traversal_stats_t* stats = nullptr, usearch_progress_t progress = {},
                       refine_plan_t* plan = nullptr, usearch_phase_progress_t* phase = nullptr) noexcept {
        usearch_dense_refine_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome, arguments.report = report;
        arguments.refine = {expansion, memory_limit};
        arguments.stats = stats, arguments.progress = progress, arguments.plan = plan, arguments.phase = phase;
        usearch_dense_refine(&arguments);
    }
    static void refine_plan_finish(index_t index, usearch_executor_t executor, refine_plan_t plan,
                                   usearch_refine_report_t* report, usearch_outcome_t* outcome) noexcept {
        usearch_dense_refine_plan_finish_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome;
        arguments.plan = plan, arguments.report = report;
        usearch_dense_refine_plan_finish(&arguments);
    }
    static void health(index_t index, usearch_executor_t executor, usearch_reachability_t* reachability,
                       usearch_outcome_t* outcome) noexcept {
        usearch_dense_health_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome;
        arguments.reachability = reachability;
        usearch_dense_health(&arguments);
    }
    using reshard_plan_t = usearch_dense_reshard_plan_t;
    static void reshard(std::span<index_t const> sources, std::span<index_t const> targets, usearch_executor_t executor,
                        std::uint64_t seed, usearch_refine_report_t* report, usearch_outcome_t* outcome,
                        std::size_t expansion = 0, std::size_t memory_limit = 0, usearch_progress_t progress = {},
                        reshard_plan_t* plan = nullptr, usearch_phase_progress_t* phase = nullptr,
                        usearch_ram_options_t const* ram = nullptr) noexcept {
        usearch_dense_reshard_t arguments {};
        arguments.sources_count = sources.size(), arguments.sources = sources.data();
        arguments.targets_count = targets.size(), arguments.targets = targets.data();
        arguments.executor = executor, arguments.outcome = outcome, arguments.seed = seed, arguments.report = report;
        arguments.refine = {expansion, memory_limit}, arguments.progress = progress;
        arguments.plan = plan, arguments.phase = phase;
        arguments.ram = ram;
        usearch_dense_reshard(&arguments);
    }
    static void reshard_plan_finish(usearch_executor_t executor, reshard_plan_t plan, usearch_refine_report_t* report,
                                    usearch_outcome_t* outcome) noexcept {
        usearch_dense_reshard_plan_finish_t arguments {};
        arguments.executor = executor, arguments.outcome = outcome;
        arguments.plan = plan, arguments.report = report;
        usearch_dense_reshard_plan_finish(&arguments);
    }
    using cursor_t = usearch_dense_cursor_t;
    static void cursor_init(index_t index, usearch_executor_t executor, std::span<usearch_key_t const> starts,
                            cursor_t* cursor, usearch_outcome_t* outcome) noexcept {
        usearch_dense_cursor_init_t arguments {};
        arguments.cursor = cursor, arguments.index = index, arguments.executor = executor, arguments.outcome = outcome;
        arguments.starts_count = starts.size(), arguments.starts = starts.data();
        usearch_dense_cursor_init(&arguments);
    }
    static void cursor_next(index_t index, usearch_executor_t executor, usearch_arena_t arena, cursor_t cursor,
                            std::size_t wanted, usearch_keys_view_t* keys, usearch_keys_tape_view_t* links,
                            std::size_t* written, usearch_outcome_t* outcome) noexcept {
        usearch_dense_cursor_next_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome;
        arguments.cursor = cursor, arguments.keys = keys, arguments.links = links, arguments.written = written;
        arguments.arena = arena, arguments.wanted = wanted;
        usearch_dense_cursor_next(&arguments);
    }
    static void sample(index_t index, usearch_executor_t executor, usearch_arena_t arena,
                       std::span<usearch_key_t const> shown, usearch_rows_view_t shown_coordinates,
                       std::span<usearch_key_t const> focus, std::size_t wanted, std::uint64_t seed,
                       sample_outputs_view_t& outputs, usearch_outcome_t* outcome) noexcept {
        usearch_dense_sample_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome, arguments.arena = arena;
        arguments.shown_count = shown.size(), arguments.shown = keys_view(shown);
        arguments.shown_coordinates = shown_coordinates;
        arguments.focus_count = focus.size(), arguments.focus = keys_view(focus);
        arguments.keys = &outputs.keys, arguments.entry_ordinals = &outputs.entry_ordinals,
        arguments.parents = &outputs.parents;
        arguments.coordinates = &outputs.coordinates, arguments.written = &outputs.written;
        arguments.stats = &outputs.stats, arguments.wanted = wanted, arguments.seed = seed;
        usearch_dense_sample(&arguments);
    }
    static void cursor_free(cursor_t cursor) noexcept { usearch_dense_cursor_free(cursor); }
    static void free(index_t index) noexcept { usearch_dense_free(index); }
};

/** What a sequence index is opened with beyond its metadata: the costs an init folds, borrowed
 *  until the index opens, or none for the metric's own. */
struct sequence_geometry_t {
    usearch_sequence_metadata_t metadata {};
    usearch_sequence_costs_t const* costs = nullptr;
};

/** A tape over @p offsets as a caller holding @c std::size_t offsets passes them, in their
 *  width. */
inline usearch_tape_view_t tape_of(void const* data, std::size_t const* offsets, usearch_scalar_t scalar) noexcept {
    usearch_tape_view_t tape {};
    tape.data = data, tape.scalar = scalar;
    if constexpr (sizeof(std::size_t) == sizeof(std::uint64_t))
        tape.layout = usearch_tape_u64_k, tape.offsets.u64 = reinterpret_cast<std::uint64_t const*>(offsets);
    else tape.offsets.u32 = reinterpret_cast<std::uint32_t const*>(offsets);
    return tape;
}

/** @p tape without its first @p first entries. */
inline usearch_tape_view_t tape_past(usearch_tape_view_t tape, std::size_t first) noexcept {
    if (tape.layout == usearch_tape_u64_k) tape.offsets.u64 += first;
    else tape.offsets.u32 += first;
    return tape;
}

/** Offset @p position of @p tape, in the width a call wrote it. */
inline std::size_t tape_offset(usearch_tape_view_t tape, std::size_t position) noexcept {
    return tape.layout == usearch_tape_u64_k ? static_cast<std::size_t>(tape.offsets.u64[position])
                                             : tape.offsets.u32[position];
}

/** The C functions of the sequence modality, behind the signatures every modality shares. */
struct sequence_verbs_t {
    using index_t = usearch_sequence_index_t;
    using description_t = usearch_sequence_index_description_t;
    using geometry_t = sequence_geometry_t;
    using metadata_t = usearch_sequence_metadata_t;
    using metric_t = usearch_sequence_metric_t;
    using values_t = usearch_tape_view_t;
    using queries_t = usearch_tape_view_t;
    static constexpr std::string_view name_k = "sequence";

    static values_t slice(values_t tape, std::size_t first) noexcept { return tape_past(tape, first); }
    static void init_ram(index_t* index, geometry_t const& geometry, usearch_ram_options_t ram,
                         usearch_outcome_t* outcome) noexcept {
        usearch_sequence_init_ram_t arguments {};
        arguments.index = index, arguments.outcome = outcome, arguments.metric = geometry.metadata.metric;
        arguments.costs = geometry.costs, arguments.connectivity = geometry.metadata.connectivity;
        arguments.sketch_bytes = geometry.metadata.sketch_bytes, arguments.ram = ram;
        usearch_sequence_init_ram(&arguments);
    }
    static void init_vram(index_t* index, geometry_t const& geometry, usearch_vram_options_t vram,
                          usearch_outcome_t* outcome) noexcept {
        usearch_sequence_init_vram_t arguments {};
        arguments.index = index, arguments.outcome = outcome, arguments.metric = geometry.metadata.metric;
        arguments.costs = geometry.costs, arguments.connectivity = geometry.metadata.connectivity;
        arguments.sketch_bytes = geometry.metadata.sketch_bytes, arguments.vram = vram;
        usearch_sequence_init_vram(&arguments);
    }
    static void insert(index_t index, usearch_executor_t executor, std::size_t count, usearch_keys_view_t keys,
                       values_t sequences, std::size_t expansion, usearch_traversal_stats_t* stats,
                       usearch_outcome_t* outcome) noexcept {
        usearch_sequence_insert_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome;
        arguments.count = count, arguments.keys = keys, arguments.sequences = sequences;
        arguments.expansion_factor = expansion, arguments.stats = stats;
        usearch_sequence_insert(&arguments);
    }
    static void search(index_t index, usearch_executor_t executor, usearch_arena_t arena, std::size_t count,
                       queries_t queries, std::size_t wanted, std::size_t expansion, usearch_traversal_t traversal,
                       usearch_matches_view_t* matches, usearch_traversal_stats_t* stats, usearch_outcome_t* outcome,
                       usearch_matches_view_t const* hints = nullptr) noexcept {
        usearch_sequence_search_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome;
        arguments.count = count, arguments.queries = queries, arguments.matches = matches, arguments.arena = arena;
        arguments.wanted = wanted, arguments.expansion_factor = expansion, arguments.traversal = traversal;
        arguments.stats = stats, arguments.hints = hints;
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
        arguments.ram = ram;
        arguments.connectivity = geometry.metadata.connectivity;
        arguments.sketch_bytes = geometry.metadata.sketch_bytes;
        usearch_sequence_load_ram(&arguments);
    }
    static void save_buffer(index_t index, usearch_executor_t executor, usearch_arena_t arena,
                            usearch_bytes_view_t* monolith, usearch_outcome_t* outcome) noexcept {
        usearch_sequence_save_buffer_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome;
        arguments.monolith = monolith, arguments.arena = arena;
        usearch_sequence_save_buffer(&arguments);
    }
    static void load_ram_buffer(index_t* index, usearch_executor_t executor, usearch_bytes_view_t buffer,
                                usearch_ram_options_t ram, geometry_t const& geometry,
                                usearch_outcome_t* outcome) noexcept {
        usearch_sequence_load_ram_buffer_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome, arguments.buffer = buffer;
        arguments.ram = ram;
        arguments.connectivity = geometry.metadata.connectivity;
        arguments.sketch_bytes = geometry.metadata.sketch_bytes;
        usearch_sequence_load_ram_buffer(&arguments);
    }
    static void metadata_read(usearch_executor_t executor, usearch_bytes_view_t path, metadata_t* metadata,
                              usearch_outcome_t* outcome) noexcept {
        usearch_sequence_metadata_read_t arguments {};
        arguments.executor = executor, arguments.outcome = outcome, arguments.path = path;
        arguments.metadata = metadata;
        usearch_sequence_metadata_read(&arguments);
    }
    static void metadata_read_buffer(usearch_executor_t executor, usearch_bytes_view_t buffer, metadata_t* metadata,
                                     usearch_outcome_t* outcome) noexcept {
        usearch_sequence_metadata_read_buffer_t arguments {};
        arguments.executor = executor, arguments.outcome = outcome, arguments.buffer = buffer;
        arguments.metadata = metadata;
        usearch_sequence_metadata_read_buffer(&arguments);
    }
    static char const* metric_name(metric_t metric) noexcept { return usearch_sequence_metric_name(metric); }
    static usearch_status_t metric_from_name(usearch_bytes_view_t name, metric_t* metric) noexcept {
        return usearch_sequence_metric_from_name(name, metric);
    }
    static void load_vram(index_t* index, usearch_executor_t executor, usearch_bytes_view_t path,
                          usearch_vram_options_t vram, geometry_t const& geometry,
                          usearch_outcome_t* outcome) noexcept {
        usearch_sequence_load_vram_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome, arguments.path = path;
        arguments.vram = vram;
        arguments.connectivity = geometry.metadata.connectivity;
        arguments.sketch_bytes = geometry.metadata.sketch_bytes;
        usearch_sequence_load_vram(&arguments);
    }
    static void view_nvme(index_t* index, usearch_executor_t executor, usearch_bytes_view_t path,
                          usearch_nvme_options_t nvme, usearch_ram_options_t const*, geometry_t const& geometry,
                          usearch_outcome_t* outcome) noexcept {
        usearch_sequence_view_nvme_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome, arguments.path = path;
        arguments.nvme = nvme, arguments.sketch_bytes = geometry.metadata.sketch_bytes;
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
    using refine_plan_t = usearch_sequence_refine_plan_t;
    static void refine(index_t index, usearch_executor_t executor, usearch_refine_report_t* report,
                       usearch_outcome_t* outcome, std::size_t expansion = 0, std::size_t memory_limit = 0,
                       usearch_traversal_stats_t* stats = nullptr, usearch_progress_t progress = {},
                       refine_plan_t* plan = nullptr, usearch_phase_progress_t* phase = nullptr) noexcept {
        usearch_sequence_refine_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome, arguments.report = report;
        arguments.refine = {expansion, memory_limit};
        arguments.stats = stats, arguments.progress = progress, arguments.plan = plan, arguments.phase = phase;
        usearch_sequence_refine(&arguments);
    }
    static void refine_plan_finish(index_t index, usearch_executor_t executor, refine_plan_t plan,
                                   usearch_refine_report_t* report, usearch_outcome_t* outcome) noexcept {
        usearch_sequence_refine_plan_finish_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome;
        arguments.plan = plan, arguments.report = report;
        usearch_sequence_refine_plan_finish(&arguments);
    }
    static void health(index_t index, usearch_executor_t executor, usearch_reachability_t* reachability,
                       usearch_outcome_t* outcome) noexcept {
        usearch_sequence_health_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome;
        arguments.reachability = reachability;
        usearch_sequence_health(&arguments);
    }
    using reshard_plan_t = usearch_sequence_reshard_plan_t;
    static void reshard(std::span<index_t const> sources, std::span<index_t const> targets, usearch_executor_t executor,
                        std::uint64_t seed, usearch_refine_report_t* report, usearch_outcome_t* outcome,
                        std::size_t expansion = 0, std::size_t memory_limit = 0, usearch_progress_t progress = {},
                        reshard_plan_t* plan = nullptr, usearch_phase_progress_t* phase = nullptr) noexcept {
        usearch_sequence_reshard_t arguments {};
        arguments.sources_count = sources.size(), arguments.sources = sources.data();
        arguments.targets_count = targets.size(), arguments.targets = targets.data();
        arguments.executor = executor, arguments.outcome = outcome, arguments.seed = seed, arguments.report = report;
        arguments.refine = {expansion, memory_limit}, arguments.progress = progress;
        arguments.plan = plan, arguments.phase = phase;
        usearch_sequence_reshard(&arguments);
    }
    static void reshard_plan_finish(usearch_executor_t executor, reshard_plan_t plan, usearch_refine_report_t* report,
                                    usearch_outcome_t* outcome) noexcept {
        usearch_sequence_reshard_plan_finish_t arguments {};
        arguments.executor = executor, arguments.outcome = outcome;
        arguments.plan = plan, arguments.report = report;
        usearch_sequence_reshard_plan_finish(&arguments);
    }
    using cursor_t = usearch_sequence_cursor_t;
    static void cursor_init(index_t index, usearch_executor_t executor, std::span<usearch_key_t const> starts,
                            cursor_t* cursor, usearch_outcome_t* outcome) noexcept {
        usearch_sequence_cursor_init_t arguments {};
        arguments.cursor = cursor, arguments.index = index, arguments.executor = executor, arguments.outcome = outcome;
        arguments.starts_count = starts.size(), arguments.starts = starts.data();
        usearch_sequence_cursor_init(&arguments);
    }
    static void cursor_next(index_t index, usearch_executor_t executor, usearch_arena_t arena, cursor_t cursor,
                            std::size_t wanted, usearch_keys_view_t* keys, usearch_keys_tape_view_t* links,
                            std::size_t* written, usearch_outcome_t* outcome) noexcept {
        usearch_sequence_cursor_next_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome;
        arguments.cursor = cursor, arguments.keys = keys, arguments.links = links, arguments.written = written;
        arguments.arena = arena, arguments.wanted = wanted;
        usearch_sequence_cursor_next(&arguments);
    }
    static void sample(index_t index, usearch_executor_t executor, usearch_arena_t arena,
                       std::span<usearch_key_t const> shown, usearch_rows_view_t shown_coordinates,
                       std::span<usearch_key_t const> focus, std::size_t wanted, std::uint64_t seed,
                       sample_outputs_view_t& outputs, usearch_outcome_t* outcome) noexcept {
        usearch_sequence_sample_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome, arguments.arena = arena;
        arguments.shown_count = shown.size(), arguments.shown = keys_view(shown);
        arguments.shown_coordinates = shown_coordinates;
        arguments.focus_count = focus.size(), arguments.focus = keys_view(focus);
        arguments.keys = &outputs.keys, arguments.entry_ordinals = &outputs.entry_ordinals,
        arguments.parents = &outputs.parents;
        arguments.coordinates = &outputs.coordinates, arguments.written = &outputs.written;
        arguments.stats = &outputs.stats, arguments.wanted = wanted, arguments.seed = seed;
        usearch_sequence_sample(&arguments);
    }
    static void cursor_free(cursor_t cursor) noexcept { usearch_sequence_cursor_free(cursor); }
    static void free(index_t index) noexcept { usearch_sequence_free(index); }
};

/** The C functions of the shapes modality, behind the signatures every modality shares. */
struct shapes_verbs_t {
    using index_t = usearch_shapes_index_t;
    using description_t = usearch_shapes_index_description_t;
    using geometry_t = usearch_shapes_metadata_t;
    using metadata_t = usearch_shapes_metadata_t;
    using metric_t = usearch_shapes_metric_t;
    using values_t = usearch_tape_view_t;
    using queries_t = usearch_tape_view_t;
    static constexpr std::string_view name_k = "shapes";

    static values_t slice(values_t tape, std::size_t first) noexcept { return tape_past(tape, first); }
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
                       values_t shapes, std::size_t expansion, usearch_traversal_stats_t* stats,
                       usearch_outcome_t* outcome) noexcept {
        usearch_shapes_insert_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome;
        arguments.count = count, arguments.keys = keys, arguments.shapes = shapes;
        arguments.expansion_factor = expansion, arguments.stats = stats;
        usearch_shapes_insert(&arguments);
    }
    static void search(index_t index, usearch_executor_t executor, usearch_arena_t arena, std::size_t count,
                       queries_t queries, std::size_t wanted, std::size_t expansion, usearch_traversal_t traversal,
                       usearch_matches_view_t* matches, usearch_traversal_stats_t* stats, usearch_outcome_t* outcome,
                       usearch_matches_view_t const* hints = nullptr) noexcept {
        usearch_shapes_search_t arguments {};
        arguments.hints = hints;
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
                         usearch_ram_options_t ram, geometry_t const& geometry, usearch_outcome_t* outcome) noexcept {
        usearch_shapes_load_ram_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome, arguments.path = path;
        arguments.ram = ram, arguments.connectivity = geometry.connectivity;
        usearch_shapes_load_ram(&arguments);
    }
    static void save_buffer(index_t index, usearch_executor_t executor, usearch_arena_t arena,
                            usearch_bytes_view_t* monolith, usearch_outcome_t* outcome) noexcept {
        usearch_shapes_save_buffer_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome;
        arguments.monolith = monolith, arguments.arena = arena;
        usearch_shapes_save_buffer(&arguments);
    }
    static void load_ram_buffer(index_t* index, usearch_executor_t executor, usearch_bytes_view_t buffer,
                                usearch_ram_options_t ram, geometry_t const& geometry,
                                usearch_outcome_t* outcome) noexcept {
        usearch_shapes_load_ram_buffer_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome, arguments.buffer = buffer;
        arguments.ram = ram, arguments.connectivity = geometry.connectivity;
        usearch_shapes_load_ram_buffer(&arguments);
    }
    static void metadata_read(usearch_executor_t executor, usearch_bytes_view_t path, metadata_t* metadata,
                              usearch_outcome_t* outcome) noexcept {
        usearch_shapes_metadata_read_t arguments {};
        arguments.executor = executor, arguments.outcome = outcome, arguments.path = path;
        arguments.metadata = metadata;
        usearch_shapes_metadata_read(&arguments);
    }
    static void metadata_read_buffer(usearch_executor_t executor, usearch_bytes_view_t buffer, metadata_t* metadata,
                                     usearch_outcome_t* outcome) noexcept {
        usearch_shapes_metadata_read_buffer_t arguments {};
        arguments.executor = executor, arguments.outcome = outcome, arguments.buffer = buffer;
        arguments.metadata = metadata;
        usearch_shapes_metadata_read_buffer(&arguments);
    }
    static char const* metric_name(metric_t metric) noexcept { return usearch_shapes_metric_name(metric); }
    static usearch_status_t metric_from_name(usearch_bytes_view_t name, metric_t* metric) noexcept {
        return usearch_shapes_metric_from_name(name, metric);
    }
    static void load_vram(index_t* index, usearch_executor_t executor, usearch_bytes_view_t path,
                          usearch_vram_options_t vram, geometry_t const& geometry,
                          usearch_outcome_t* outcome) noexcept {
        usearch_shapes_load_vram_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome, arguments.path = path;
        arguments.vram = vram, arguments.connectivity = geometry.connectivity;
        usearch_shapes_load_vram(&arguments);
    }
    static void view_nvme(index_t* index, usearch_executor_t executor, usearch_bytes_view_t path,
                          usearch_nvme_options_t nvme, usearch_ram_options_t const*, geometry_t const&,
                          usearch_outcome_t* outcome) noexcept {
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
    using refine_plan_t = usearch_shapes_refine_plan_t;
    static void refine(index_t index, usearch_executor_t executor, usearch_refine_report_t* report,
                       usearch_outcome_t* outcome, std::size_t expansion = 0, std::size_t memory_limit = 0,
                       usearch_traversal_stats_t* stats = nullptr, usearch_progress_t progress = {},
                       refine_plan_t* plan = nullptr, usearch_phase_progress_t* phase = nullptr) noexcept {
        usearch_shapes_refine_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome, arguments.report = report;
        arguments.refine = {expansion, memory_limit};
        arguments.stats = stats, arguments.progress = progress, arguments.plan = plan, arguments.phase = phase;
        usearch_shapes_refine(&arguments);
    }
    static void refine_plan_finish(index_t index, usearch_executor_t executor, refine_plan_t plan,
                                   usearch_refine_report_t* report, usearch_outcome_t* outcome) noexcept {
        usearch_shapes_refine_plan_finish_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome;
        arguments.plan = plan, arguments.report = report;
        usearch_shapes_refine_plan_finish(&arguments);
    }
    static void health(index_t index, usearch_executor_t executor, usearch_reachability_t* reachability,
                       usearch_outcome_t* outcome) noexcept {
        usearch_shapes_health_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome;
        arguments.reachability = reachability;
        usearch_shapes_health(&arguments);
    }
    using reshard_plan_t = usearch_shapes_reshard_plan_t;
    static void reshard(std::span<index_t const> sources, std::span<index_t const> targets, usearch_executor_t executor,
                        std::uint64_t seed, usearch_refine_report_t* report, usearch_outcome_t* outcome,
                        std::size_t expansion = 0, std::size_t memory_limit = 0, usearch_progress_t progress = {},
                        reshard_plan_t* plan = nullptr, usearch_phase_progress_t* phase = nullptr) noexcept {
        usearch_shapes_reshard_t arguments {};
        arguments.sources_count = sources.size(), arguments.sources = sources.data();
        arguments.targets_count = targets.size(), arguments.targets = targets.data();
        arguments.executor = executor, arguments.outcome = outcome, arguments.seed = seed, arguments.report = report;
        arguments.refine = {expansion, memory_limit}, arguments.progress = progress;
        arguments.plan = plan, arguments.phase = phase;
        usearch_shapes_reshard(&arguments);
    }
    static void reshard_plan_finish(usearch_executor_t executor, reshard_plan_t plan, usearch_refine_report_t* report,
                                    usearch_outcome_t* outcome) noexcept {
        usearch_shapes_reshard_plan_finish_t arguments {};
        arguments.executor = executor, arguments.outcome = outcome;
        arguments.plan = plan, arguments.report = report;
        usearch_shapes_reshard_plan_finish(&arguments);
    }
    using cursor_t = usearch_shapes_cursor_t;
    static void cursor_init(index_t index, usearch_executor_t executor, std::span<usearch_key_t const> starts,
                            cursor_t* cursor, usearch_outcome_t* outcome) noexcept {
        usearch_shapes_cursor_init_t arguments {};
        arguments.cursor = cursor, arguments.index = index, arguments.executor = executor, arguments.outcome = outcome;
        arguments.starts_count = starts.size(), arguments.starts = starts.data();
        usearch_shapes_cursor_init(&arguments);
    }
    static void cursor_next(index_t index, usearch_executor_t executor, usearch_arena_t arena, cursor_t cursor,
                            std::size_t wanted, usearch_keys_view_t* keys, usearch_keys_tape_view_t* links,
                            std::size_t* written, usearch_outcome_t* outcome) noexcept {
        usearch_shapes_cursor_next_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome;
        arguments.cursor = cursor, arguments.keys = keys, arguments.links = links, arguments.written = written;
        arguments.arena = arena, arguments.wanted = wanted;
        usearch_shapes_cursor_next(&arguments);
    }
    static void sample(index_t index, usearch_executor_t executor, usearch_arena_t arena,
                       std::span<usearch_key_t const> shown, usearch_rows_view_t shown_coordinates,
                       std::span<usearch_key_t const> focus, std::size_t wanted, std::uint64_t seed,
                       sample_outputs_view_t& outputs, usearch_outcome_t* outcome) noexcept {
        usearch_shapes_sample_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome, arguments.arena = arena;
        arguments.shown_count = shown.size(), arguments.shown = keys_view(shown);
        arguments.shown_coordinates = shown_coordinates;
        arguments.focus_count = focus.size(), arguments.focus = keys_view(focus);
        arguments.keys = &outputs.keys, arguments.entry_ordinals = &outputs.entry_ordinals,
        arguments.parents = &outputs.parents;
        arguments.coordinates = &outputs.coordinates, arguments.written = &outputs.written;
        arguments.stats = &outputs.stats, arguments.wanted = wanted, arguments.seed = seed;
        usearch_shapes_sample(&arguments);
    }
    static void cursor_free(cursor_t cursor) noexcept { usearch_shapes_cursor_free(cursor); }
    static void free(index_t index) noexcept { usearch_shapes_free(index); }
};

/** What a geospatial index is opened with beyond its metadata: its terrain, whose heights are
 *  viewed. */
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
    using metadata_t = usearch_geospatial_metadata_t;
    using values_t = usearch_geospatial_observation_t const*;
    using queries_t = geospatial_queries_t;
    static constexpr std::string_view name_k = "geospatial";

    static values_t slice(values_t observations, std::size_t first) noexcept { return observations + first; }
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
                       values_t observations, std::size_t expansion, usearch_traversal_stats_t* stats,
                       usearch_outcome_t* outcome) noexcept {
        usearch_geospatial_insert_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome;
        arguments.count = count, arguments.keys = keys, arguments.observations = observations;
        arguments.expansion_factor = expansion, arguments.stats = stats;
        usearch_geospatial_insert(&arguments);
    }
    static void search(index_t index, usearch_executor_t executor, usearch_arena_t arena, std::size_t count,
                       queries_t queries, std::size_t wanted, std::size_t expansion, usearch_traversal_t traversal,
                       usearch_matches_view_t* matches, usearch_traversal_stats_t* stats, usearch_outcome_t* outcome,
                       usearch_matches_view_t const* hints = nullptr) noexcept {
        if (hints) return void(*outcome = {usearch_status_unsupported_k, "Geospatial searches take no hints"});
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
        arguments.connectivity = geometry.metadata.connectivity;
        usearch_geospatial_load_ram(&arguments);
    }
    static void save_buffer(index_t index, usearch_executor_t executor, usearch_arena_t arena,
                            usearch_bytes_view_t* monolith, usearch_outcome_t* outcome) noexcept {
        usearch_geospatial_save_buffer_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome;
        arguments.monolith = monolith, arguments.arena = arena;
        usearch_geospatial_save_buffer(&arguments);
    }
    static void load_ram_buffer(index_t* index, usearch_executor_t executor, usearch_bytes_view_t buffer,
                                usearch_ram_options_t ram, geometry_t const& geometry,
                                usearch_outcome_t* outcome) noexcept {
        usearch_geospatial_load_ram_buffer_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome, arguments.buffer = buffer;
        arguments.ram = ram, arguments.terrain = geometry.terrain;
        arguments.connectivity = geometry.metadata.connectivity;
        usearch_geospatial_load_ram_buffer(&arguments);
    }
    static void metadata_read(usearch_executor_t executor, usearch_bytes_view_t path, metadata_t* metadata,
                              usearch_outcome_t* outcome) noexcept {
        usearch_geospatial_metadata_read_t arguments {};
        arguments.executor = executor, arguments.outcome = outcome, arguments.path = path;
        arguments.metadata = metadata;
        usearch_geospatial_metadata_read(&arguments);
    }
    static void metadata_read_buffer(usearch_executor_t executor, usearch_bytes_view_t buffer, metadata_t* metadata,
                                     usearch_outcome_t* outcome) noexcept {
        usearch_geospatial_metadata_read_buffer_t arguments {};
        arguments.executor = executor, arguments.outcome = outcome, arguments.buffer = buffer;
        arguments.metadata = metadata;
        usearch_geospatial_metadata_read_buffer(&arguments);
    }
    static void load_vram(index_t* index, usearch_executor_t executor, usearch_bytes_view_t path,
                          usearch_vram_options_t vram, geometry_t const& geometry,
                          usearch_outcome_t* outcome) noexcept {
        usearch_geospatial_load_vram_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome, arguments.path = path;
        arguments.vram = vram, arguments.terrain = geometry.terrain;
        arguments.connectivity = geometry.metadata.connectivity;
        usearch_geospatial_load_vram(&arguments);
    }
    static void view_nvme(index_t* index, usearch_executor_t executor, usearch_bytes_view_t path,
                          usearch_nvme_options_t nvme, usearch_ram_options_t const*, geometry_t const& geometry,
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
    using refine_plan_t = usearch_geospatial_refine_plan_t;
    static void refine(index_t index, usearch_executor_t executor, usearch_refine_report_t* report,
                       usearch_outcome_t* outcome, std::size_t expansion = 0, std::size_t memory_limit = 0,
                       usearch_traversal_stats_t* stats = nullptr, usearch_progress_t progress = {},
                       refine_plan_t* plan = nullptr, usearch_phase_progress_t* phase = nullptr) noexcept {
        usearch_geospatial_refine_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome, arguments.report = report;
        arguments.refine = {expansion, memory_limit};
        arguments.stats = stats, arguments.progress = progress, arguments.plan = plan, arguments.phase = phase;
        usearch_geospatial_refine(&arguments);
    }
    static void refine_plan_finish(index_t index, usearch_executor_t executor, refine_plan_t plan,
                                   usearch_refine_report_t* report, usearch_outcome_t* outcome) noexcept {
        usearch_geospatial_refine_plan_finish_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome;
        arguments.plan = plan, arguments.report = report;
        usearch_geospatial_refine_plan_finish(&arguments);
    }
    static void health(index_t index, usearch_executor_t executor, usearch_reachability_t* reachability,
                       usearch_outcome_t* outcome) noexcept {
        usearch_geospatial_health_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome;
        arguments.reachability = reachability;
        usearch_geospatial_health(&arguments);
    }
    using reshard_plan_t = usearch_geospatial_reshard_plan_t;
    static void reshard(std::span<index_t const> sources, std::span<index_t const> targets, usearch_executor_t executor,
                        std::uint64_t seed, usearch_refine_report_t* report, usearch_outcome_t* outcome,
                        std::size_t expansion = 0, std::size_t memory_limit = 0, usearch_progress_t progress = {},
                        reshard_plan_t* plan = nullptr, usearch_phase_progress_t* phase = nullptr) noexcept {
        usearch_geospatial_reshard_t arguments {};
        arguments.sources_count = sources.size(), arguments.sources = sources.data();
        arguments.targets_count = targets.size(), arguments.targets = targets.data();
        arguments.executor = executor, arguments.outcome = outcome, arguments.seed = seed, arguments.report = report;
        arguments.refine = {expansion, memory_limit}, arguments.progress = progress;
        arguments.plan = plan, arguments.phase = phase;
        usearch_geospatial_reshard(&arguments);
    }
    static void reshard_plan_finish(usearch_executor_t executor, reshard_plan_t plan, usearch_refine_report_t* report,
                                    usearch_outcome_t* outcome) noexcept {
        usearch_geospatial_reshard_plan_finish_t arguments {};
        arguments.executor = executor, arguments.outcome = outcome;
        arguments.plan = plan, arguments.report = report;
        usearch_geospatial_reshard_plan_finish(&arguments);
    }
    using cursor_t = usearch_geospatial_cursor_t;
    static void cursor_init(index_t index, usearch_executor_t executor, std::span<usearch_key_t const> starts,
                            cursor_t* cursor, usearch_outcome_t* outcome) noexcept {
        usearch_geospatial_cursor_init_t arguments {};
        arguments.cursor = cursor, arguments.index = index, arguments.executor = executor, arguments.outcome = outcome;
        arguments.starts_count = starts.size(), arguments.starts = starts.data();
        usearch_geospatial_cursor_init(&arguments);
    }
    static void cursor_next(index_t index, usearch_executor_t executor, usearch_arena_t arena, cursor_t cursor,
                            std::size_t wanted, usearch_keys_view_t* keys, usearch_keys_tape_view_t* links,
                            std::size_t* written, usearch_outcome_t* outcome) noexcept {
        usearch_geospatial_cursor_next_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome;
        arguments.cursor = cursor, arguments.keys = keys, arguments.links = links, arguments.written = written;
        arguments.arena = arena, arguments.wanted = wanted;
        usearch_geospatial_cursor_next(&arguments);
    }
    static void sample(index_t index, usearch_executor_t executor, usearch_arena_t arena,
                       std::span<usearch_key_t const> shown, usearch_rows_view_t shown_coordinates,
                       std::span<usearch_key_t const> focus, std::size_t wanted, std::uint64_t seed,
                       sample_outputs_view_t& outputs, usearch_outcome_t* outcome) noexcept {
        usearch_geospatial_sample_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome, arguments.arena = arena;
        arguments.shown_count = shown.size(), arguments.shown = keys_view(shown);
        arguments.shown_coordinates = shown_coordinates;
        arguments.focus_count = focus.size(), arguments.focus = keys_view(focus);
        arguments.keys = &outputs.keys, arguments.entry_ordinals = &outputs.entry_ordinals,
        arguments.parents = &outputs.parents;
        arguments.coordinates = &outputs.coordinates, arguments.written = &outputs.written;
        arguments.stats = &outputs.stats, arguments.wanted = wanted, arguments.seed = seed;
        usearch_geospatial_sample(&arguments);
    }
    static void cursor_free(cursor_t cursor) noexcept { usearch_geospatial_cursor_free(cursor); }
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

    /** The block every read of a view covers, which counts the bytes a process read as reads; zero
     *  outside a view. */
    util::u64_t block_bytes = 0;

    handle() = default;
    handle(index_t index, session_t& session, usearch_residency_t residency, geometry_t const& geometry) noexcept
        : index(index), session(&session), residency(residency), geometry(geometry) {}
    handle(handle const&) = delete;
    handle& operator=(handle const&) = delete;
    handle(handle&& other) noexcept
        : index(std::exchange(other.index, nullptr)), session(other.session), residency(other.residency),
          geometry(other.geometry), block_bytes(other.block_bytes) {}
    handle& operator=(handle&& other) noexcept {
        std::swap(index, other.index), std::swap(session, other.session);
        std::swap(residency, other.residency), std::swap(geometry, other.geometry);
        std::swap(block_bytes, other.block_bytes);
        return *this;
    }
    ~handle() noexcept { verbs_type_::free(index); }

    index_t get() const noexcept { return index; }
    explicit operator bool() const noexcept { return index != nullptr; }

    /** The executor this index's calls run on. */
    usearch_executor_t executor() const noexcept { return session->executor_for(residency); }

    /** The arena this index's outputs land in. */
    usearch_arena_t arena() const noexcept { return session->arena_for(residency); }
};

/** Opens an empty index of @p geometry on @p shards shards of @p session, in RAM for an NVMe
 *  session; zero shards takes one per path of an NVMe session that names paths, else one per domain
 *  or device. Prints what failed and answers an empty handle. */
template <typename verbs_type_>
handle<verbs_type_> open_index(session_t& session, typename verbs_type_::geometry_t const& geometry,
                               std::size_t shards) {
    typename verbs_type_::index_t index = nullptr;
    usearch_outcome_t outcome {};
    st::vector<std::int32_t> placement;
    usearch_residency_t const residency = session.residency == usearch_residency_vram_k ? usearch_residency_vram_k
                                                                                        : usearch_residency_ram_k;
    // An NVMe session places one part per path, so the build spreads over as many shards.
    if (!shards && session.residency == usearch_residency_nvme_k) shards = session.places.size();
    if (residency == usearch_residency_vram_k)
        verbs_type_::init_vram(&index, geometry, session.vram_options(shards, placement), &outcome);
    else verbs_type_::init_ram(&index, geometry, session.ram_options(shards, placement), &outcome);
    report_failure(outcome, "Opening the index");
    return {index, session, residency, geometry};
}

/** Reserves room for @p count more entries, then inserts @p count entries of @p inputs into
 *  @p index in batches of @p batch, keyed by @ref key_of_offset from offset @p first_offset on,
 *  summing every batch's telemetry into @p totals and reporting each inserted count to
 *  @p progress. */
template <typename verbs_type_, typename progress_type_ = util::dummy_progress_t>
usearch_outcome_t insert_entries(handle<verbs_type_>& index, typename verbs_type_::values_t inputs, std::size_t count,
                                 std::size_t first_offset, std::size_t batch, std::size_t expansion,
                                 usearch_traversal_stats_t& totals, progress_type_&& progress = {}) {
    usearch_outcome_t outcome {};
    std::size_t const batch_entries = util::max_of<std::size_t>(1, util::min_of(batch, count));
    st::vector<usearch_key_t> keys;
    if (st::failed(keys.resize(batch_entries))) exit_out_of_memory("the keys");
    for (std::size_t first = 0; first < count; first += batch_entries) {
        std::size_t const entries = util::min_of(batch_entries, count - first);
        std::span<usearch_key_t> const batch_keys = std::span {keys}.first(entries);
        for (std::size_t offset = 0; offset != entries; ++offset)
            batch_keys[offset] = key_of_offset(index.session->seed, first_offset + first + offset);
        usearch_traversal_stats_t stats {};
        verbs_type_::insert(index.get(), index.executor(), entries, keys_view(batch_keys),
                            verbs_type_::slice(inputs, first), expansion, &stats, &outcome);
        if (outcome.status != usearch_status_success_k) return outcome;
        totals.visited_members += stats.visited_members, totals.computed_distances += stats.computed_distances;
        totals.restarts += stats.restarts;
        progress(first + entries);
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

/** Reopens the checkpoint at @p path of @p geometry in @p residency on @p session : loaded into RAM
 *  or onto @p shards of the session's devices, or viewed on its NVMe executor from the session's
 *  served paths when it names any. Prints what failed and answers an empty handle. */
template <typename verbs_type_>
handle<verbs_type_> reopen(session_t& session, std::string_view path, typename verbs_type_::geometry_t const& geometry,
                           usearch_residency_t residency, std::size_t shards) {
    usearch_outcome_t outcome {};
    typename verbs_type_::index_t reopened = nullptr;
    st::vector<std::int32_t> placement;
    if (residency == usearch_residency_nvme_k)
        verbs_type_::view_nvme(&reopened, session.storage, bytes_view(path), session.nvme_options(),
                               session.nvme_ram_options(), geometry, &outcome);
    else if (residency == usearch_residency_vram_k)
        verbs_type_::load_vram(&reopened, session.executor, bytes_view(path), session.vram_options(shards, placement),
                               geometry, &outcome);
    else
        verbs_type_::load_ram(&reopened, session.executor, bytes_view(path), session.ram_options(0, placement),
                              geometry, &outcome);
    report_failure(outcome, "Reopening {}", path);
    handle<verbs_type_> opened {reopened, session, residency, geometry};
    if (residency == usearch_residency_nvme_k && !util::names_transport_id(path)) {
        auto const [medium, medium_status] = util::file_medium_t::open(path);
        if (util::succeeded(medium_status)) opened.block_bytes = util::block_bytes_of(medium);
    }
    return opened;
}

/** The checkpoint metadata @p geometry carries: the geometry itself where a modality opens with
 *  nothing more. */
template <typename verbs_type_>
typename verbs_type_::metadata_t& metadata_of(typename verbs_type_::geometry_t& geometry) noexcept {
    if constexpr (std::is_same_v<typename verbs_type_::geometry_t, typename verbs_type_::metadata_t>) return geometry;
    else return geometry.metadata;
}

template <typename verbs_type_>
typename verbs_type_::metadata_t const& metadata_of(typename verbs_type_::geometry_t const& geometry) noexcept {
    if constexpr (std::is_same_v<typename verbs_type_::geometry_t, typename verbs_type_::metadata_t>) return geometry;
    else return geometry.metadata;
}

/** Whether the checkpoint at @p path recalls the metric and connectivity @p geometry names, a zero
 *  connectivity taking any; prints what failed to read. */
template <typename verbs_type_>
bool checkpoint_matches_geometry(std::string_view path, typename verbs_type_::geometry_t const& geometry) {
    typename verbs_type_::metadata_t recalled {};
    usearch_outcome_t outcome {};
    verbs_type_::metadata_read(nullptr, bytes_view(path), &recalled, &outcome);
    if (report_failure(outcome, "Reading {}", path)) return false;
    typename verbs_type_::metadata_t const& named = metadata_of<verbs_type_>(geometry);
    bool same_metric = true;
    if constexpr (requires { named.metric; }) same_metric = recalled.metric == named.metric;
    return same_metric && (!named.connectivity || recalled.connectivity == named.connectivity);
}

/** Saves @p index under @p path and reopens it in @p residency on the same session, as @ref reopen
 *  does, first placing an NVMe save's parts on the session's places when it names any. Prints what
 *  failed and answers an empty handle. */
template <typename verbs_type_>
handle<verbs_type_> save_and_reopen(handle<verbs_type_>&& index, std::string_view path, usearch_residency_t residency) {
    session_t& session = *index.session;
    typename verbs_type_::geometry_t const geometry = index.geometry;
    usearch_outcome_t outcome {};
    verbs_type_::save(index.get(), index.executor(), bytes_view(path), &outcome);
    if (report_failure(outcome, "Saving {}", path)) return {};
    std::size_t const shards = describe(index).shards_count;
    index = {};
    return reopen<verbs_type_>(session, path, geometry, residency, shards);
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
inline st::vector<sz::string_t> matching_files(std::string_view pattern) {
    std::size_t const slash = pattern.find_last_of('/');
    std::string_view const directory = slash == std::string_view::npos ? std::string_view {}
                                                                       : pattern.substr(0, slash + 1);
    std::string_view const filename_pattern = pattern.substr(directory.size());
    st::vector<sz::string_t> files;
    std::error_code error;
    std::filesystem::directory_iterator entries {
        directory.empty() ? std::filesystem::path {"."} : std::filesystem::path {directory}, error};
    for (; !error && entries != std::filesystem::directory_iterator {}; entries.increment(error)) {
        std::filesystem::directory_entry const& entry = *entries;
        std::filesystem::path const name = entry.path().filename();
        std::error_code kind_error;
        if (entry.is_regular_file(kind_error) && wildcard_matches(filename_pattern, name.string()) &&
            st::failed(files.push_back(formatted("{}{}", directory, name.string()))))
            exit_out_of_memory("the file names");
    }
    std::sort(files.begin(), files.end());
    return files;
}

/** Every file whose name starts with @p prefix, in sorted order. */
inline st::vector<sz::string_t> files_starting_with(std::string_view prefix) {
    return matching_files(formatted("{}*", prefix));
}

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
    case usearch_scalar_u1x8_k: return util::divide_round_up<std::size_t>(columns, 8);
    default: return columns;
    }
}

/** One mapped RetriEval part file and the rows of the whole matrix it holds. */
struct mapped_part_t {
    util::mapped_file_t mapping;
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
    st::vector<mapped_part_t> parts_;
    usearch_scalar_t scalar_ = usearch_scalar_unknown_k;
    std::size_t rows_ = 0, columns_ = 0, row_bytes_ = 0;

    util::status_t append_(std::string_view path) noexcept {
        auto [mapping, read_status] = util::mapped_file_t::read(path);
        if (util::failed(read_status)) return read_status;
        util::bytes_view_t const bytes = mapping.view();
        std::uint32_t counters[2] = {};
        if (bytes.size() < sizeof(counters)) return util::status_code_t::malformed_k;
        std::memcpy(counters, bytes.data(), sizeof(counters));
        if (!parts_.empty() && counters[1] != columns_) return util::status_code_t::config_mismatch_k;
        columns_ = counters[1];
        row_bytes_ = row_bytes_of(scalar_, columns_);
        if (bytes.size() < sizeof(counters) + static_cast<std::size_t>(counters[0]) * row_bytes_)
            return util::status_code_t::malformed_k;
        if (st::failed(parts_.push_back({std::move(mapping), rows_, counters[0]})))
            return util::status_code_t::out_of_memory_k;
        rows_ += counters[0];
        return util::status_code_t::success_k;
    }

  public:

    /** Maps the parts @p pattern names, failing on an unreadable part or a column mismatch. */
    static util::expected<mapped_rows_t> open(std::string_view pattern) noexcept {
        mapped_rows_t mapped;
        mapped.scalar_ = scalar_of_extension(pattern);
        st::vector<sz::string_t> parts = matching_files(pattern);
        // With no match the pattern stands for itself, so a missing file fails like its mapping.
        if (parts.empty() && st::failed(parts.push_back(formatted("{}", pattern))))
            return {std::move(mapped), util::status_code_t::out_of_memory_k};
        util::status_t status = util::status_code_t::success_k;
        for (std::size_t part = 0; util::succeeded(status) && part != parts.size(); ++part)
            status = mapped.append_(parts[part]);
        return {std::move(mapped), status};
    }

    /** Rows `[first_row, first_row + count)` as the C interface takes them, viewed in place when
     *  one part holds them all; null data when they straddle two parts or run past the last. */
    usearch_rows_view_t view(std::size_t first_row, std::size_t count) const noexcept {
        for (mapped_part_t const& part : parts_) {
            if (first_row < part.first_row || first_row + count > part.first_row + part.rows) continue;
            util::byte_t const* const payload = part.mapping.view().data() + 2 * sizeof(std::uint32_t);
            return {payload + (first_row - part.first_row) * row_bytes_, row_bytes_, columns_, scalar_};
        }
        return {nullptr, row_bytes_, columns_, scalar_};
    }

    std::size_t rows() const noexcept { return rows_; }
    std::size_t columns() const noexcept { return columns_; }
    usearch_scalar_t scalar() const noexcept { return scalar_; }
    explicit operator bool() const noexcept { return !parts_.empty(); }

    /** The bytes past the rows of the last part, where a ground-truth file stores its distances. */
    util::bytes_view_t trailing() const noexcept {
        if (parts_.empty()) return {};
        mapped_part_t const& last = parts_.back();
        return last.mapping.view().subspan(2 * sizeof(std::uint32_t) + last.rows * row_bytes_);
    }
};

/** Entries end to end and where each starts, as the C interface takes a tape of bytes. */
struct byte_tape_t {
    sz::string_t bytes;
    st::vector<std::size_t> offsets;

    std::size_t size() const noexcept { return offsets.empty() ? 0 : offsets.size() - 1; }
    usearch_tape_view_t view() const noexcept { return tape_of(bytes.data(), offsets.data(), usearch_scalar_u8_k); }
};

/** The lines of @p text as entries, each newline dropped with a carriage return before it. */
inline byte_tape_t lines_of(util::bytes_view_t text) noexcept {
    byte_tape_t lines;
    if (lines.bytes.resize(text.size()) != sz::status_t::success_k || st::failed(lines.offsets.push_back(0)))
        exit_out_of_memory("the lines");
    std::size_t length = 0;
    for (util::byte_t const byte : text) {
        char const character = static_cast<char>(byte);
        if (character != '\n') {
            lines.bytes[length++] = character;
            continue;
        }
        if (length != lines.offsets.back() && lines.bytes[length - 1] == '\r') --length;
        if (st::failed(lines.offsets.push_back(std::size_t {length}))) exit_out_of_memory("the lines");
    }
    if (length != lines.offsets.back() && st::failed(lines.offsets.push_back(std::size_t {length})))
        exit_out_of_memory("the lines");
    if (lines.bytes.resize(length) != sz::status_t::success_k) exit_out_of_memory("the lines");
    return lines;
}

/** Costs over bytes, as the C interface reads them and folds them into classes itself. */
struct byte_costs_t {
    std::array<std::int8_t, 256 * 256> similarities {};
    std::int8_t gap = 0;

    /** The costs the C interface reads, borrowing the similarities. */
    usearch_sequence_costs_t view() const noexcept { return {similarities.data(), gap}; }
};

/** The costs in the file at @p path : 256x256 int8 similarities between bytes and an int8 gap cost;
 *  none after printing why not. */
inline std::optional<byte_costs_t> read_costs(std::string_view path) {
    byte_costs_t read;
    auto [file, status] = util::mapped_file_t::read(path);
    if (failed(status) || file.view().size() != read.similarities.size() + 1) {
        fmt::println(stderr, "--costs=\"{}\" must hold 256x256 int8 similarities and an int8 gap cost", path);
        return std::nullopt;
    }
    std::memcpy(read.similarities.data(), file.view().data(), read.similarities.size());
    std::memcpy(&read.gap, file.view().data() + read.similarities.size(), 1);
    return read;
}

/** Bytes a key takes in the <tt>.u64bin</tt> or <tt>.u128bin</tt> at @p path, zero for any other name. */
inline std::size_t key_width_of(std::string_view path) noexcept {
    return path.ends_with(".u64bin") ? 8 : path.ends_with(".u128bin") ? 16 : 0;
}

/** The key the little-endian integer of @p width bytes at @p integer names, filling a key's low
 *  bytes as row numbers do. */
inline usearch_key_t key_of_integer(usearch_byte_t const* integer, std::size_t width) noexcept {
    usearch_key_t key {};
    for (std::size_t index = 0; index != width; ++index) key.bytes[sizeof(key.bytes) - 1 - index] = integer[index];
    return key;
}

/** Reads the keys of the <tt>.u64bin</tt> or <tt>.u128bin</tt> at @p path, one little-endian integer
 *  for each of @p rows rows after its two counters, as <tt>--keys</tt> reads them; none after
 *  printing why it cannot. */
inline std::optional<st::vector<usearch_key_t>> read_keys(std::string_view path, std::size_t rows) {
    std::size_t const width = key_width_of(path);
    if (!width) return fmt::println(stderr, "--keys={} is neither .u64bin nor .u128bin", path), std::nullopt;
    auto [file, status] = util::mapped_file_t::read(path);
    std::array<std::uint32_t, 2> counters {};
    if (failed(status) || file.view().size() < sizeof(counters))
        return fmt::println(stderr, "cannot read {}", path), std::nullopt;
    std::memcpy(counters.data(), file.view().data(), sizeof(counters));
    if (counters[0] != rows || counters[1] != 1 || file.view().size() < sizeof(counters) + rows * width)
        return fmt::println(stderr, "{} holds {} rows of {} columns, not one key for each of {} rows", path,
                            counters[0], counters[1], rows),
               std::nullopt;
    st::vector<usearch_key_t> keys;
    if (st::failed(keys.resize(rows))) exit_out_of_memory("the keys");
    for (std::size_t row = 0; row != rows; ++row)
        keys[row] = key_of_integer(
            reinterpret_cast<usearch_byte_t const*>(file.view().data()) + sizeof(counters) + row * width, width);
    return keys;
}

/** Point clouds mapped from two RetriEval files, every cloud's points end to end and the points per
 *  cloud, and each cloud's first point, as a tape takes them. */
struct point_clouds_t {
    mapped_rows_t points;
    mapped_rows_t counts;
    st::vector<std::size_t> offsets;

    std::size_t size() const noexcept { return offsets.empty() ? 0 : offsets.size() - 1; }
    usearch_tape_view_t view() const noexcept {
        return tape_of(points.view(0, points.rows()).data, offsets.data(), points.scalar());
    }
};

/** Maps the clouds of @p points_path, 3 f32 or f16 columns per point, and @p counts_path, 1 u32
 *  column per cloud, into @p clouds; false after printing why it cannot. */
inline bool open_point_clouds(std::string_view points_path, std::string_view counts_path, point_clouds_t& clouds) {
    auto [points, points_status] = mapped_rows_t::open(points_path);
    if (util::failed(points_status) || points.columns() != 3 ||
        (points.scalar() != usearch_scalar_f32_k && points.scalar() != usearch_scalar_f16_k)) {
        fmt::println(stderr, "{} must hold 3 f32 or f16 columns per point", points_path);
        return false;
    }
    auto [counts, counts_status] = mapped_rows_t::open(counts_path);
    if (util::failed(counts_status) || counts.columns() != 1) {
        fmt::println(stderr, "{} must hold 1 column per cloud", counts_path);
        return false;
    }
    clouds.points = std::move(points), clouds.counts = std::move(counts);
    std::uint32_t const* const point_counts = static_cast<std::uint32_t const*>(
        clouds.counts.view(0, clouds.counts.rows()).data);
    if (st::failed(clouds.offsets.resize(clouds.counts.rows() + 1))) exit_out_of_memory("the cloud offsets");
    for (std::size_t cloud = 0; cloud != clouds.counts.rows(); ++cloud)
        clouds.offsets[cloud + 1] = clouds.offsets[cloud] + point_counts[cloud];
    if (clouds.offsets.back() != clouds.points.rows()) {
        fmt::println(stderr, "{} sums to {} points, {} holds {}", counts_path, clouds.offsets.back(), points_path,
                     clouds.points.rows());
        return false;
    }
    return true;
}

/** The relation a geospatial search asks, by the name its flag takes: near, visible, sees or seen;
 *  unknown for any other. */
inline usearch_geospatial_relation_t geospatial_relation_of(std::string_view name) noexcept {
    if (name == "near") return usearch_geospatial_relation_near_k;
    if (name == "visible") return usearch_geospatial_relation_visible_k;
    if (name == "sees") return usearch_geospatial_relation_sees_k;
    if (name == "seen") return usearch_geospatial_relation_seen_k;
    return usearch_geospatial_relation_unknown_k;
}

#pragma endregion Datasets

#pragma region Recall

/** The matches searches wrote, copied off their arena: every query's keys and scores end to end,
 *  and how many each query holds, so whole searches compare and judge. */
struct copied_matches_t {
    st::vector<util::uuid128_t> keys;
    st::vector<usearch_distance_t> scores;
    st::vector<std::size_t> counts;
};

/** Whether @p first and @p second hold the same keys at the same scores, as many per query. */
inline bool same_matches(copied_matches_t const& first, copied_matches_t const& second) noexcept {
    return std::ranges::equal(first.keys, second.keys) && std::ranges::equal(first.scores, second.scores) &&
           std::ranges::equal(first.counts, second.counts);
}

/** Appends one match, @p key at @p score, to @p copied. */
inline void append_match(copied_matches_t& copied, usearch_key_t const& key, usearch_distance_t score) noexcept {
    if (st::failed(copied.keys.push_back(std::bit_cast<util::uuid128_t>(key))) ||
        st::failed(copied.scores.push_back(usearch_distance_t {score})))
        exit_out_of_memory("the matches");
}

/** Appends the matches the first @p queries queries of @p matches hold to @p copied, read back from
 *  wherever the arena placed them; false when the copy failed. */
inline bool append_matches(usearch_matches_view_t const& matches, std::size_t queries,
                           copied_matches_t& copied) noexcept {
    st::vector<usearch_key_t> keys;
    st::vector<usearch_distance_t> scores;
    st::vector<std::size_t> counts;
    if (st::failed(keys.resize(st::assume_uninitialized, queries * matches.columns)) ||
        st::failed(scores.resize(st::assume_uninitialized, queries * matches.columns)) ||
        st::failed(counts.resize(st::assume_uninitialized, queries)))
        exit_out_of_memory("the matches");
    if (!download(keys.data(), matches.keys, keys.size() * sizeof(usearch_key_t)) ||
        !download(scores.data(), matches.scores, scores.size() * sizeof(usearch_distance_t)) ||
        !download(counts.data(), matches.counts, counts.size() * sizeof(std::size_t)))
        return false;
    for (std::size_t query = 0; query != queries; ++query) {
        if (st::failed(copied.counts.push_back(std::size_t {counts[query]}))) exit_out_of_memory("the matches");
        for (std::size_t match = query * matches.columns; match != query * matches.columns + counts[query]; ++match)
            append_match(copied, keys[match], scores[match]);
    }
    return true;
}

/** The matches the first @p queries queries of @p matches hold, copied off a host arena. */
inline copied_matches_t copy_matches(usearch_matches_view_t const& matches, std::size_t queries) noexcept {
    copied_matches_t copied;
    if (st::failed(copied.counts.assign({matches.counts, queries}))) exit_out_of_memory("the matches");
    for (std::size_t query = 0; query != queries; ++query)
        for (std::size_t match = query * matches.columns; match != query * matches.columns + matches.counts[query];
             ++match)
            append_match(copied, matches.keys[match], matches.scores[match]);
    return copied;
}

/** Searches @p count queries of @p queries in batches of @p batch, each batch handed its rows of
 *  the optional @p hints, copying every match to @p copied and its telemetry into @p totals, and
 *  reporting the queries resolved to the optional @p progress after each batch, which a cancel
 *  stops with @c usearch_status_cancelled_k. */
template <typename verbs_type_>
usearch_outcome_t search_queries(handle<verbs_type_>& index, typename verbs_type_::queries_t queries, std::size_t count,
                                 std::size_t batch, std::size_t wanted, std::size_t expansion,
                                 usearch_traversal_t traversal, usearch_traversal_stats_t& totals,
                                 copied_matches_t& copied, usearch_matches_view_t const* hints = nullptr,
                                 usearch_progress_t progress = {}) {
    usearch_outcome_t outcome {};
    std::size_t const batch_queries = util::max_of<std::size_t>(1, util::min_of(batch, count));
    for (std::size_t first = 0; first < count; first += batch_queries) {
        std::size_t const sliced = util::min_of(batch_queries, count - first);
        usearch_matches_view_t matches {};
        usearch_traversal_stats_t stats {};
        usearch_matches_view_t const sliced_hints = hints
                                                        ? usearch_matches_view_t {hints->keys + first * hints->columns,
                                                                                  nullptr, hints->counts + first,
                                                                                  hints->columns}
                                                        : usearch_matches_view_t {};
        verbs_type_::search(index.get(), index.executor(), index.arena(), sliced, verbs_type_::slice(queries, first),
                            wanted, expansion, traversal, &matches, &stats, &outcome, hints ? &sliced_hints : nullptr);
        if (outcome.status != usearch_status_success_k) return outcome;
        if (!append_matches(matches, sliced, copied))
            return {usearch_status_device_k, "The matches could not be read back"};
        totals.visited_members += stats.visited_members, totals.computed_distances += stats.computed_distances;
        totals.restarts += stats.restarts, totals.missing_hints += stats.missing_hints;
        totals.seeding_distances += stats.seeding_distances;
        if (progress.callback && progress.callback(first + sliced, count, progress.state) == usearch_cancel_k)
            return {usearch_status_cancelled_k, "The caller's handler stopped the search between batches"};
    }
    return outcome;
}

/** Fills @p oracle with the exact top @p wanted of the first @p judged queries of @p queries,
 *  nearest first with their distances, by a scan of every entry @p index holds. */
template <typename verbs_type_>
usearch_outcome_t scan_oracle(handle<verbs_type_>& index, typename verbs_type_::queries_t queries, std::size_t judged,
                              std::size_t wanted, std::size_t batch, copied_matches_t& oracle) {
    usearch_traversal_stats_t stats {};
    return search_queries(index, queries, judged, batch, wanted, 0, usearch_traversal_scan_k, stats, oracle);
}

/** How a ground-truth file stores its distances, against the ones the index reports. */
enum class truth_distances_t { reported_k, squared_k };

/** Fills @p oracle with the truth the first @p queries rows of the ground-truth file @p truth list:
 *  each row's neighbor offsets keyed by @ref key_of_offset under @p seed, at the @c f32 distances
 *  RetriEval and BigANN store past the rows, rooted when @p stored says they hold the square of
 *  the Euclidean distance the index reports; false for a file storing none. */
inline bool truth_oracle(util::u64_t seed, mapped_rows_t const& truth, std::size_t queries, truth_distances_t stored,
                         copied_matches_t& oracle) noexcept {
    std::size_t const depth = truth.columns(), listed = queries * depth;
    util::bytes_view_t const distances = truth.trailing();
    if (distances.size() < truth.rows() * depth * sizeof(float)) return false;
    util::byte_t const* const offsets = static_cast<util::byte_t const*>(truth.view(0, queries).data);
    if (st::failed(oracle.counts.resize(queries, depth)) || st::failed(oracle.keys.resize(listed)) ||
        st::failed(oracle.scores.resize(listed)))
        exit_out_of_memory("the ground truth");
    for (std::size_t match = 0; match != listed; ++match) {
        std::uint32_t offset = 0;
        float distance = 0;
        std::memcpy(&offset, offsets + match * sizeof(offset), sizeof(offset));
        std::memcpy(&distance, distances.data() + match * sizeof(distance), sizeof(distance));
        oracle.keys[match] = std::bit_cast<util::uuid128_t>(key_of_offset(seed, offset));
        oracle.scores[match] = stored == truth_distances_t::squared_k ? std::sqrt(distance) : distance;
    }
    return true;
}

/** Whether @p found ranks no further than @p exact, within the rounding two kits may differ by. */
inline bool within(double found, double exact) noexcept {
    return found <= exact + 1e-6 * util::max_of(1.0, std::fabs(exact));
}

/** The offset of every query's first match in @p copied. */
inline st::vector<std::size_t> match_offsets(copied_matches_t const& copied) noexcept {
    st::vector<std::size_t> offsets;
    if (st::failed(offsets.resize(copied.counts.size() + 1))) exit_out_of_memory("the match offsets");
    for (std::size_t query = 0; query != copied.counts.size(); ++query)
        offsets[query + 1] = offsets[query] + copied.counts[query];
    return offsets;
}

/**
 *  @brief Calls @p visit with each judged query's <tt>(expected, has_nearest, hits)</tt>.
 *
 *  Those are how many matches the exact @p oracle expects among the first @p k, whether any of
 *  the first @p k matches @p found holds is the true nearest key or lies within its distance, and
 *  whether each is among the true first @p k keys or lies within the true k-th distance.
 *
 *  A key match counts whatever distance a re-packed graph reports, and a distance match counts a
 *  tie with a true neighbor wherever the oracle broke it.
 */
template <typename visit_type_>
void judge_queries(std::size_t k, copied_matches_t const& found, copied_matches_t const& oracle, visit_type_&& visit) {
    st::vector<std::size_t> const found_offsets = match_offsets(found), truth_offsets = match_offsets(oracle);
    st::vector<bool> hits;
    for (std::size_t query = 0; query != util::min_of(found.counts.size(), oracle.counts.size()); ++query) {
        std::size_t const expected = util::min_of(k, oracle.counts[query]);
        if (!expected) continue;
        std::size_t const truth_first = truth_offsets[query], found_first = found_offsets[query];
        std::size_t const ranked = util::min_of(k, found.counts[query]);
        double const nearest = oracle.scores[truth_first], farthest = oracle.scores[truth_first + expected - 1];
        std::span<util::uuid128_t const> const true_top {oracle.keys.data() + truth_first, expected};
        bool has_nearest = false;
        hits.clear();
        if (st::failed(hits.resize(ranked, false))) exit_out_of_memory("the hits");
        for (std::size_t rank = 0; rank != ranked; ++rank) {
            util::uuid128_t const& key = found.keys[found_first + rank];
            usearch_distance_t const score = found.scores[found_first + rank];
            hits[rank] = within(score, farthest) || std::find(true_top.begin(), true_top.end(), key) != true_top.end();
            has_nearest = has_nearest || within(score, nearest) || key == true_top.front();
        }
        visit(expected, has_nearest, hits);
    }
}

/** The share of judged queries whose first @p k matches in @p found hold one within the true
 *  nearest distance, over the queries the oracle expects any match for. */
inline double one_recall_at(std::size_t k, copied_matches_t const& found, copied_matches_t const& oracle) {
    std::size_t nearest_count = 0, answerable = 0;
    judge_queries(k, found, oracle, [&](std::size_t, bool has_nearest, std::span<bool const>) {
        nearest_count += has_nearest;
        ++answerable;
    });
    return answerable ? static_cast<double>(nearest_count) / static_cast<double>(answerable) : 0;
}

/** The share of each judged query's top @p k the matches @p found hold among their own first
 *  @p k, each counted within the true k-th distance, averaged over the queries the oracle
 *  expects any match for. */
inline double k_recall_at(std::size_t k, copied_matches_t const& found, copied_matches_t const& oracle) {
    double recall = 0;
    std::size_t answerable = 0;
    judge_queries(k, found, oracle, [&](std::size_t expected, bool, std::span<bool const> hits) {
        std::size_t hit_count = 0;
        for (bool const hit : hits) hit_count += hit;
        recall += static_cast<double>(util::min_of(hit_count, expected)) / static_cast<double>(expected);
        ++answerable;
    });
    return answerable ? recall / static_cast<double>(answerable) : 0;
}

/** The normalized discounted cumulative gain of @p found 's first @p k matches against @p oracle,
 *  every hit earning its rank's discount, averaged over the queries the oracle expects any match
 *  for. */
inline double ndcg_at(std::size_t k, copied_matches_t const& found, copied_matches_t const& oracle) {
    double gain_sum = 0;
    std::size_t answerable = 0;
    judge_queries(k, found, oracle, [&](std::size_t expected, bool, std::span<bool const> hits) {
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

/** Reads and bytes the block layer counted for this process's cgroup, summed over its disks. */
struct cgroup_reads_t {
    util::u64_t reads = 0;
    util::u64_t bytes = 0;
};

/** The text of the file @p name in this process's cgroup v2 directory, none where the kernel keeps
 *  no such file, as off Linux or in the root cgroup. */
inline std::optional<sz::string_t> cgroup_text(std::string_view name) noexcept {
    std::optional<sz::string_t> const membership = read_text("/proc/self/cgroup");
    if (!membership) return std::nullopt;
    std::string_view const lines {*membership};
    // The unified hierarchy's line reads `0::<path>`.
    std::size_t const unified = lines.find("0::");
    if (unified == std::string_view::npos || (unified && lines[unified - 1] != '\n')) return std::nullopt;
    std::size_t const end = lines.find('\n', unified);
    std::string_view const scope = lines.substr(unified + 3, end - unified - 3);
    return read_text(formatted("/sys/fs/cgroup{}/{}", scope, name).c_str());
}

/** What @c /proc/self/io counts this process fetched from the block layer so far, every kernel
 *  engine's reads included, as reads of @p block_bytes each; none off Linux or without a block. */
inline std::optional<cgroup_reads_t> process_reads(util::u64_t block_bytes) noexcept {
    std::optional<sz::string_t> const io = read_text("/proc/self/io");
    if (!io) return std::nullopt;
    std::string_view const counters {*io};
    constexpr std::string_view key = "read_bytes: ";
    std::size_t const found = counters.find(key);
    if (!block_bytes || found == std::string_view::npos) return std::nullopt;
    cgroup_reads_t counted;
    std::from_chars_result const parsed = std::from_chars(counters.data() + found + key.size(),
                                                          counters.data() + counters.size(), counted.bytes);
    if (parsed.ec != std::errc {}) return std::nullopt;
    counted.reads = counted.bytes / block_bytes;
    return counted;
}

/** What this process's cgroup @c io.stat counts so far, every kernel engine's reads on every
 *  disk, or what @ref process_reads counts in @p block_bytes reads under a user scope that keeps
 *  no such file. */
inline std::optional<cgroup_reads_t> cgroup_reads(util::u64_t block_bytes) {
    std::optional<sz::string_t> const stat = cgroup_text("io.stat");
    if (!stat) return process_reads(block_bytes);
    cgroup_reads_t counted;
    std::string_view text {*stat};
    while (!text.empty()) {
        std::size_t const space = util::min_of(text.find_first_of(" \n"), text.size());
        std::string_view const token = text.substr(0, space);
        std::string_view const digits = token.substr(token.find('=') + 1);
        util::u64_t value = 0;
        std::from_chars_result const parsed = std::from_chars(digits.data(), digits.data() + digits.size(), value);
        if (parsed.ec == std::errc {} && token.starts_with("rbytes=")) counted.bytes += value;
        if (parsed.ec == std::errc {} && token.starts_with("rios=")) counted.reads += value;
        text.remove_prefix(util::min_of(space + 1, text.size()));
    }
    return counted;
}

/** The @p quantile of @p samples, reordering them around it. */
inline double percentile(std::span<util::u64_t> samples, double quantile) noexcept {
    std::size_t const rank = static_cast<std::size_t>(quantile * static_cast<double>(samples.size() - 1));
    std::nth_element(samples.begin(), samples.begin() + rank, samples.end());
    return static_cast<double>(samples[rank]);
}

/** Times single-query searches of @p queries on @p index 's executor one after another, from the
 *  first of @p count on, until @p budget has passed; @p latencies takes each call's nanoseconds. */
template <typename verbs_type_>
usearch_outcome_t time_single_queries(handle<verbs_type_>& index, typename verbs_type_::queries_t queries,
                                      std::size_t count, std::size_t wanted, std::size_t expansion,
                                      std::chrono::steady_clock::duration budget, st::vector<util::u64_t>& latencies) {
    usearch_outcome_t outcome {};
    latencies.clear();
    if (st::failed(latencies.reserve(count))) exit_out_of_memory("the latencies");
    std::chrono::steady_clock::time_point const deadline = std::chrono::steady_clock::now() + budget;
    for (std::size_t query = 0; query != count && std::chrono::steady_clock::now() < deadline; ++query) {
        usearch_matches_view_t matches {};
        usearch_traversal_stats_t stats {};
        std::chrono::steady_clock::time_point const started = std::chrono::steady_clock::now();
        verbs_type_::search(index.get(), index.executor(), index.arena(), 1, verbs_type_::slice(queries, query), wanted,
                            expansion, usearch_traversal_walk_k, &matches, &stats, &outcome);
        if (st::failed(latencies.push_back(static_cast<util::u64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started)
                    .count()))))
            exit_out_of_memory("the latencies");
        if (outcome.status != usearch_status_success_k) return outcome;
    }
    return outcome;
}

/** Searches @p count queries of @p queries at every breadth of @p expansions, printing each
 *  breadth's 1-recall and k-recall at @p wanted and its nDCG against @p oracle, its throughput and
 *  its distances per query, then handing its matches to @p on_swept as @b (expansion,matches).
 *  A view on NVMe also prints the reads and bytes its cgroup's @c io.stat counted per query, and
 *  the p50 and p99 of single-query calls over judged queries, timed after the throughput pass for
 *  as long as that pass took, since each one walks alone. */
template <typename verbs_type_, typename swept_type_ = util::dummy_progress_t>
usearch_outcome_t sweep_recall(handle<verbs_type_>& index, typename verbs_type_::queries_t queries, std::size_t count,
                               copied_matches_t const& oracle, std::span<std::size_t const> expansions,
                               std::size_t wanted, std::size_t batch, swept_type_&& on_swept = {}) {
    usearch_outcome_t outcome {};
    st::vector<util::u64_t> latencies;
    for (std::size_t const expansion : expansions) {
        copied_matches_t found;
        usearch_traversal_stats_t stats {};
        std::optional<cgroup_reads_t> const reads_before = index.residency == usearch_residency_nvme_k
                                                               ? cgroup_reads(index.block_bytes)
                                                               : std::nullopt;
        std::chrono::steady_clock::time_point const started = std::chrono::steady_clock::now();
        outcome = search_queries(index, queries, count, batch, wanted, expansion, usearch_traversal_walk_k, stats,
                                 found);
        std::chrono::steady_clock::duration const batched = std::chrono::steady_clock::now() - started;
        double const seconds = std::chrono::duration<double>(batched).count();
        std::optional<cgroup_reads_t> const reads_after = reads_before ? cgroup_reads(index.block_bytes) : std::nullopt;
        if (outcome.status != usearch_status_success_k) return outcome;
        fmt::println("- Expansion {}: 1-recall@{} {:.4f} · {}-recall@{} {:.4f} · nDCG@{} {:.4f} over {} judged",
                     expansion, wanted, one_recall_at(wanted, found, oracle), wanted, wanted,
                     k_recall_at(wanted, found, oracle), wanted, ndcg_at(wanted, found, oracle), oracle.counts.size());
        double const per_query = 1.0 / static_cast<double>(count);
        fmt::println("  {} queries in {:.2f} s · {:.0f} QPS", count, seconds, static_cast<double>(count) / seconds);
        if (!reads_after)
            fmt::println("  per query: {:.0f} distances · {:.0f} visited", stats.computed_distances * per_query,
                         stats.visited_members * per_query);
        else
            fmt::println("  per query: {:.0f} distances · {:.0f} visited · {:.1f} reads · {:.1f} KiB",
                         stats.computed_distances * per_query, stats.visited_members * per_query,
                         static_cast<double>(reads_after->reads - reads_before->reads) * per_query,
                         static_cast<double>(reads_after->bytes - reads_before->bytes) * per_query / 1024);
        if (index.residency == usearch_residency_nvme_k) {
            std::size_t const judged = util::min_of(count, oracle.counts.size());
            outcome = time_single_queries(index, queries, judged, wanted, expansion, batched, latencies);
            if (outcome.status != usearch_status_success_k) return outcome;
            if (!latencies.empty())
                fmt::println("  latency: p50 {:.0f} µs · p99 {:.0f} µs over {} single-query calls",
                             percentile(latencies, 0.50) / 1e3, percentile(latencies, 0.99) / 1e3, latencies.size());
        }
        on_swept(expansion, found);
    }
    return outcome;
}

#pragma endregion Recall

#pragma region Sweeps

/** Whether a dense index of @p scalar opens in @p residency, on device zero for VRAM; a probe of
 *  what the library compiled and this machine admits. */
inline bool dense_opens(usearch_residency_t residency, usearch_scalar_t scalar) noexcept {
    usearch_dense_index_t index = nullptr;
    usearch_outcome_t outcome {};
    usearch_dense_metadata_t const geometry {usearch_dense_metric_inner_k, scalar, 64, 0};
    std::int32_t const device = 0;
    if (residency == usearch_residency_vram_k)
        dense_verbs_t::init_vram(&index, geometry, {&device, 1, 0, usearch_gpu_class_unknown_k}, &outcome);
    else dense_verbs_t::init_ram(&index, geometry, {}, &outcome);
    dense_verbs_t::free(index);
    return outcome.status == usearch_status_success_k;
}

/** The class a bf16 RAM index picks under @p cpu_class_ceiling, described on @p executor , or
 *  unknown if it does not open. */
inline usearch_cpu_class_t picked_class(usearch_executor_t executor, usearch_cpu_class_t cpu_class_ceiling) noexcept {
    usearch_dense_index_t index = nullptr;
    usearch_outcome_t outcome {};
    usearch_dense_index_description_t described {};
    dense_verbs_t::init_ram(&index, {usearch_dense_metric_inner_k, usearch_scalar_bf16_k, 64, 0},
                            {nullptr, 0, 0, cpu_class_ceiling}, &outcome);
    if (index) dense_verbs_t::describe(index, executor, &described, &outcome);
    dense_verbs_t::free(index);
    return outcome.status == usearch_status_success_k ? described.cpu_class : usearch_cpu_class_unknown_k;
}

/** The residencies this build and machine serve: RAM and NVMe always, and VRAM when a GPU is
 *  listed. */
inline st::vector<usearch_residency_t> compiled_residencies() noexcept {
    st::vector<usearch_residency_t> residencies;
    bool const vram = !listed_devices().empty() && dense_opens(usearch_residency_vram_k, usearch_scalar_bf16_k);
    if (st::failed(residencies.push_back(usearch_residency_ram_k)) ||
        st::failed(residencies.push_back(usearch_residency_nvme_k)) ||
        (vram && st::failed(residencies.push_back(usearch_residency_vram_k))))
        exit_out_of_memory("the residencies");
    return residencies;
}

/** The dense scalars @p residency opens, in the enumeration's order; an NVMe index serves what RAM
 *  built. */
inline st::vector<usearch_scalar_t> compiled_scalars(usearch_residency_t residency) noexcept {
    st::vector<usearch_scalar_t> scalars;
    for (int code = usearch_scalar_f64_k; code <= usearch_scalar_u1x8_k; ++code) {
        usearch_scalar_t const scalar = static_cast<usearch_scalar_t>(code);
        if (dense_opens(residency == usearch_residency_vram_k ? residency : usearch_residency_ram_k, scalar) &&
            st::failed(scalars.push_back(usearch_scalar_t {scalar})))
            exit_out_of_memory("the scalars");
    }
    return scalars;
}

/** The CPU classes this machine dispatches as when each is the ceiling, from the portable floor to
 *  the last class the header declares, since a value past it is no @c usearch_cpu_class_t, and a
 *  gap in between spells its name "unrecognized"; a class the machine lacks narrows to another
 *  one, so the sweep leaves it out. */
inline st::vector<usearch_cpu_class_t> admitted_classes(usearch_executor_t executor) noexcept {
    st::vector<usearch_cpu_class_t> classes;
    for (int code = usearch_cpu_class_serial_k; code <= usearch_cpu_class_wasm64_extended_k; ++code) {
        usearch_cpu_class_t const cpu_class = static_cast<usearch_cpu_class_t>(code);
        if (std::string_view(usearch_cpu_class_name(cpu_class)) != "unrecognized" &&
            picked_class(executor, cpu_class) == cpu_class &&
            st::failed(classes.push_back(usearch_cpu_class_t {cpu_class})))
            exit_out_of_memory("the CPU classes");
    }
    return classes;
}

/** The ceiling naming the portable floor: serial, or the one class a monoversioned build compiles
 *  in its place, which a RAM index without a ceiling then describes. */
inline usearch_cpu_class_t floor_class(usearch_executor_t executor) {
    return picked_class(executor, usearch_cpu_class_unknown_k) == usearch_cpu_class_monoversioned_k
               ? usearch_cpu_class_monoversioned_k
               : usearch_cpu_class_serial_k;
}

/** One cell of a suite's sweep: where its index lives, what it stores, the class it dispatches as,
 *  set as its ceiling, and how many shards it spans. */
struct sweep_cell_t {
    usearch_residency_t residency = usearch_residency_ram_k;
    usearch_scalar_t scalar = usearch_scalar_unknown_k;
    usearch_cpu_class_t cpu_class = usearch_cpu_class_unknown_k;
    std::size_t shards = 1;
};

/**
 *  @brief The diagonal a suite sweeps rather than the full product: every scalar at the best class,
 *      every admitted class at @p first, and the floor class at every scalar, each over the
 *      compiled residencies at one and at three shards.
 *
 *  A basic modality passes @c usearch_scalar_unknown_k, whose cells vary the class alone, so the
 *  suite runs its first metric at each and every metric at the best class. A VRAM cell sets no CPU
 *  class ceiling, as no CPU walks it.
 */
inline st::vector<sweep_cell_t> diagonal_sweep(usearch_scalar_t first) noexcept {
    st::vector<sweep_cell_t> cells;
    // One thread is enough to describe each probing index.
    usearch_executor_t probing = nullptr;
    usearch_outcome_t outcome {};
    usearch_executor_for_ram_t placement {.executor = &probing, .outcome = &outcome, .threads = 1};
    usearch_executor_for_ram(&placement);
    if (report_failure(outcome, "Opening the probing executor")) return cells;
    st::vector<usearch_cpu_class_t> const classes = admitted_classes(probing);
    usearch_cpu_class_t const floor = floor_class(probing);
    usearch_executor_free(probing);
    for (usearch_residency_t const residency : compiled_residencies()) {
        st::vector<usearch_scalar_t> scalars;
        if (first != usearch_scalar_unknown_k) scalars = compiled_scalars(residency);
        else if (st::failed(scalars.push_back(usearch_scalar_t {first}))) exit_out_of_memory("the sweep");
        for (std::size_t const shards : {std::size_t {1}, std::size_t {3}}) {
            for (usearch_scalar_t const scalar : scalars)
                if (st::failed(cells.push_back({residency, scalar, usearch_cpu_class_unknown_k, shards})))
                    exit_out_of_memory("the sweep");
            if (residency == usearch_residency_vram_k) continue;
            for (usearch_cpu_class_t const cpu_class : classes)
                if (st::failed(cells.push_back({residency, first, cpu_class, shards}))) exit_out_of_memory("the sweep");
            for (usearch_scalar_t const scalar : scalars)
                if (scalar != first && st::failed(cells.push_back({residency, scalar, floor, shards})))
                    exit_out_of_memory("the sweep");
        }
    }
    return cells;
}

/** The name a filter selects @p cell by, like "ram bf16 intel-icelake 3-shard". */
inline sz::string_t name_of(sweep_cell_t const& cell) noexcept {
    return formatted("{} {} {} {}-shard", usearch_residency_name(cell.residency),
                     cell.scalar ? usearch_scalar_name(cell.scalar) : "metrics",
                     cell.cpu_class ? usearch_cpu_class_name(cell.cpu_class) : "best", cell.shards);
}

/** The settings of @p settings with @p cell 's residency, for the session the cell opens; every
 *  field of @ref settings_t is named here, as its lists copy only through their status verbs. */
inline settings_t settings_of(settings_t const& settings, sweep_cell_t const& cell) {
    settings_t of_cell {
        .seed = settings.seed,
        .regex_filter = settings.regex_filter,
        .residency = {residency_names_k, selected_of(residencies_k, cell.residency)},
        .devices = {},
        .nvme_paths = {},
        .cpu_class_ceiling = settings.cpu_class_ceiling,
        .gpu_class_ceiling = settings.gpu_class_ceiling,
        .nvme_class = settings.nvme_class,
        .threads = settings.threads,
        .shards = settings.shards,
        .shard_capacity = settings.shard_capacity,
        .checkpoint = settings.checkpoint,
        .huge_pages = settings.huge_pages,
        .nvme_ram = settings.nvme_ram,
        .terminal = settings.terminal,
    };
    if (st::failed(of_cell.devices.assign(settings.devices)) ||
        st::failed(of_cell.nvme_paths.assign(settings.nvme_paths)))
        exit_out_of_memory("the settings");
    return of_cell;
}

#pragma endregion Sweeps

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
    count_t oracle_queries {.value = 256, .minimum = 0};

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
        {"--oracle-queries", "count", "queries the exact scan judges, none for 0; 256 by default",
         &search.oracle_queries},
        {"--batch", "entries", "entries per insert or search call; 65536 by default", &search.batch},
    }};
}

/**
 *  @brief Builds one index of @p geometry over the first @p base_count entries of @p entries and
 *      searches @p queries_count @p queries against one exact scan at every breadth.
 *
 *  The entries are keyed by their offsets, and the index is served from @p checkpoint in the
 *  session's residency. An NVMe session serves a view of what it built from @p checkpoint, or from
 *  the temporary directory when that is empty, and any other session reopens what it saved only
 *  when @p checkpoint names a path; the reopened index takes @p served, so a search may score under
 *  other costs than the build. Each breadth's matches go to @p on_swept as @b (expansion,matches).
 *
 *  @return The process exit code.
 */
template <typename verbs_type_, typename swept_type_ = util::dummy_progress_t>
int run_search_bench(settings_t const& settings, search_bench_settings_t const& search,
                     typename verbs_type_::geometry_t const& geometry, typename verbs_type_::geometry_t const& served,
                     typename verbs_type_::values_t entries, std::size_t base_count,
                     typename verbs_type_::queries_t queries, std::size_t queries_count, std::string_view checkpoint,
                     swept_type_&& on_swept = {}) {
    session_t session = open_session(settings, cpu_class_ceilings_k[settings.cpu_class_ceiling.selected]);
    if (!session) return 1;
    fmt::println("- Base entries: {} · queries: {} · top-k: {}", base_count, queries_count, search.top_k.value);
    sz::string_t served_path = formatted("{}", checkpoint);
    // A saved checkpoint reopens as is, so a RAM run can judge a graph built where no scan serves.
    bool const reused = !checkpoint.empty() && std::filesystem::exists(served_path.c_str());
    if (reused && !checkpoint_matches_geometry<verbs_type_>(checkpoint, geometry)) {
        fmt::println(stderr, "{} was saved under another metric or connectivity", checkpoint);
        return 1;
    }
    handle<verbs_type_> index = reused ? reopen<verbs_type_>(session, checkpoint, served,
                                                             residencies_k[settings.residency.selected],
                                                             settings.shards.value)
                                       : open_index<verbs_type_>(session, geometry, settings.shards.value);
    if (!index) return 1;
    if (reused)
        fmt::println("- Reused {} as {}, nothing built", checkpoint, residency_names_k[settings.residency.selected]);
    else {
        usearch_traversal_stats_t built {};
        std::chrono::steady_clock::time_point const build_start = std::chrono::steady_clock::now();
        std::size_t reported_millions = 0;
        // One line per million entries, so a long build's rate reads early.
        auto const report_build = [&](std::size_t inserted) {
            if (inserted / 1000000 == reported_millions) return;
            reported_millions = inserted / 1000000;
            double const seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - build_start).count();
            fmt::println("  build: {} / {} · {:.0f}/s", inserted, base_count, static_cast<double>(inserted) / seconds);
            std::fflush(stdout);
        };
        if (report_failure(insert_entries(index, entries, base_count, 0, search.batch.value, search.expansion_add.value,
                                          built, report_build),
                           "Inserting"))
            return 1;
        double const build_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - build_start).count();
        fmt::println("- Build: {} entries in {:.2f} s · {:.0f}/s · {:.0f} distances per entry", base_count,
                     build_seconds, static_cast<double>(base_count) / build_seconds,
                     static_cast<double>(built.computed_distances) / static_cast<double>(base_count));
        print_fill(index);

        if (served_path.empty() && residencies_k[settings.residency.selected] == usearch_residency_nvme_k) {
            sz::string_t const name = formatted("usearch-bench-{}.safegraphs", verbs_type_::name_k);
            served_path = formatted("{}", (std::filesystem::temp_directory_path() / std::string_view {name}).string());
        }
        if (!served_path.empty()) {
            index.geometry = served;
            index = save_and_reopen(std::move(index), served_path, residencies_k[settings.residency.selected]);
            if (!index) return 1;
            fmt::println("- Reopened {} as {}", served_path, residency_names_k[settings.residency.selected]);
        }
    }

    copied_matches_t oracle;
    std::size_t const judged = util::min_of(search.oracle_queries.value, queries_count);
    std::chrono::steady_clock::time_point const scan_start = std::chrono::steady_clock::now();
    if (judged &&
        report_failure(scan_oracle(index, queries, judged, search.top_k.value, search.batch.value, oracle), "Scanning"))
        return 1;
    fmt::println("- Exact scan: {} queries in {:.2f} s", judged,
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - scan_start).count());
    usearch_outcome_t const swept = sweep_recall(index, queries, queries_count, oracle, search.expansion_search.view(),
                                                 search.top_k.value, search.batch.value, on_swept);
    if (checkpoint.empty() && !served_path.empty()) {
        index = {};
        std::string_view const served {served_path};
        for (sz::string_t const& file : files_starting_with(served.substr(0, served.rfind(".safegraphs"))))
            std::remove(file.c_str());
    }
    return report_failure(swept, "Searching") ? 1 : 0;
}

#pragma endregion Benchmarks

} // namespace unum::usearch::tools
