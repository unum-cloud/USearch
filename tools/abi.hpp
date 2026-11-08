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
 *  --cpu-class         derived          The CPU class to dispatch as, like intel-icelake or serial
 *  --gpu-class         derived          The GPU class to dispatch as, like nvidia-ampere
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

#include <cmath>   // `std::fabs`, `std::log2` - the score tolerance, the ranking discounts
#include <cstddef> // `std::size_t`
#include <cstdint> // `std::int32_t`, `std::uint32_t`, `std::uint64_t`
#include <cstdio>  // `std::remove`, `stdout`, `std::fopen` - the round-tripped checkpoints, the terminal probe, io.stat
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
#include <source_location> // `std::source_location` - where a check failed
#include <span>            // `std::span`
#include <string>          // `std::string`
#include <string_view>     // `std::string_view`
#include <system_error>    // `std::errc`
#include <utility>         // `std::exchange`, `std::move`
#include <vector>          // `std::vector`

#include <fmt/format.h>
#include <fmt/ranges.h> // `fmt::join` - the devices, the shard sizes

#include <forkunion/forkunion.h> // `fu_topology_new`, `fu_memory_domain_id_at_index` - the domains shards repeat over

#include "usearch/usearch.h"
#include "usearch/utilities/blocking_files.hpp" // `mapped_file_t` - the dataset parts, the compared checkpoints
#include "usearch/utilities/nvme_device.hpp"    // `file_medium_t`, `block_bytes_of`, `reads_directly` - views' reads
#include "usearch/utilities/types.hpp"          // `uuid_t`, `u64_t`
#include "usearch/utilities/vram_device.hpp"    // `cuda_device_of`, `hip_device_of` - device outputs read back

#include "arguments.hpp" // `flag_t`, `count_t`, `parse_or_exit`, `bytes_view`

namespace unum::usearch::tools {
using namespace unum::usearch::utilities;

#pragma region Keys

/** Offsets below this bound get keys: a version-4 UUID leaves 62 bits beside its variant. */
inline constexpr u64_t key_offsets_k = u64_t {1} << 62;

/** The odd multipliers of the offset mix, SplitMix64's. */
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
 *  mix of those and @p seed, which @ref offset_of_key checks. */
constexpr u64_t leading_word_(u64_t seed, u64_t trailing) noexcept {
    return (mix_(trailing ^ mix_(seed)) & ~u64_t {0xF000}) | 0x4000;
}

/**
 *  @brief The key the harness mints for @p offset under @p seed : a random-looking version-4 UUID
 *      with no table behind it.
 *
 *  The trailing eight bytes hold the variant over an invertible mix of the offset within 62 bits,
 *  and the leading eight a mix of them, so @ref offset_of_key inverts the first and checks the
 *  second.
 */
inline usearch_key_t key_of_offset(u64_t seed, u64_t offset) noexcept {
    u64_t mixed = (offset ^ mix_(seed)) & (key_offsets_k - 1);
    for (u64_t const multiplier : key_multipliers_k) mixed = (mixed ^ mixed >> 31) * multiplier & (key_offsets_k - 1);
    u64_t const trailing = u64_t {2} << 62 | (mixed ^ mixed >> 31);
    u64_t const leading = leading_word_(seed, trailing);
    usearch_key_t key;
    for (std::size_t byte = 0; byte != 8; ++byte) {
        key.bytes[7 - byte] = static_cast<usearch_byte_t>(leading >> (byte * 8));
        key.bytes[15 - byte] = static_cast<usearch_byte_t>(trailing >> (byte * 8));
    }
    return key;
}

/** The offset @ref key_of_offset minted @p key for under @p seed , or none for a key it never
 *  minted. */
inline std::optional<u64_t> offset_of_key(u64_t seed, usearch_key_t const& key) noexcept {
    u64_t leading = 0, trailing = 0;
    for (std::size_t byte = 0; byte != 8; ++byte)
        leading = leading << 8 | key.bytes[byte], trailing = trailing << 8 | key.bytes[8 + byte];
    if (trailing >> 62 != 2 || leading != leading_word_(seed, trailing)) return std::nullopt;
    u64_t mixed = trailing & (key_offsets_k - 1);
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
        {"--filter", "pattern", "checks to run, as a regex or a substring; every one by default", &settings.filter},
        session[0],
        {"--cpu-class", "name", "CPU class to dispatch as, like intel-icelake or serial; derived by default",
         &settings.cpu_class},
        {"--gpu-class", "name", "GPU class to dispatch as, like nvidia-ampere; derived by default",
         &settings.gpu_class},
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
inline std::optional<std::string> read_text(char const* path) {
    std::FILE* const file = std::fopen(path, "r");
    if (!file) return std::nullopt;
    std::string text;
    std::array<char, 4096> chunk;
    for (std::size_t read = 0; (read = std::fread(chunk.data(), 1, chunk.size(), file)) != 0;)
        text.append(chunk.data(), read);
    std::fclose(file);
    return text;
}

/** Whether the host's 1 GiB pool holds the @p expected free pages a run declares, which the
 *  launcher provisions outside the process; prints why not. */
inline bool huge_pages_provisioned(std::size_t expected) {
    if (!expected) return true;
    std::optional<std::string> const text = read_text("/sys/kernel/mm/hugepages/hugepages-1048576kB/free_hugepages");
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
 *  <tt>--help</tt> prints the usage and exits with status 0; on a setting that does not parse, or huge
 *  pages the host does not hold, prints one line and exits with status 1. */
inline settings_t read_settings(int argc, char const* const* argv, std::string_view synopsis,
                                std::span<flag_t const> binary_flags = {}) {
    settings_t settings = read_environment();
    std::vector<flag_t> flags(binary_flags.begin(), binary_flags.end());
    std::array<flag_t, 11> const shared = settings_flags(settings);
    flags.insert(flags.end(), shared.begin(), shared.end());
    parse_or_exit(argc, argv, synopsis, flags);
    if (!huge_pages_provisioned(settings.huge_pages.value)) std::exit(1);
    return settings;
}

/** How a view reads @p path: "direct" past the page cache, "buffered" through it, or "unopened"
 *  before it opens; a transport ID reads directly and stays closed, as SPDK starts once. */
inline std::string_view nvme_reads_name(std::string_view path) noexcept {
    if (names_transport_id(path)) return "direct";
    auto const [medium, open_status] = file_medium_t::open(path);
    if (failed(open_status)) return "unopened";
    return reads_directly(medium) ? "direct" : "buffered";
}

/** Prints each setting @ref read_environment reads as "- Name: value", in the grammar it parses
 *  from, and how views read each NVMe path; calls nothing of the library, so header-only suites
 *  print it too. */
inline void print_environment(settings_t const& settings) {
    fmt::println("- Seed: {}", settings.seed);
    fmt::println("- Filter: {}", settings.filter.pattern.empty() ? "none" : settings.filter.pattern);
    if (settings.devices.empty()) fmt::println("- Devices: derived");
    else fmt::println("- Devices: {}", fmt::join(settings.devices, ","));
    if (settings.nvme_paths.empty()) return fmt::println("- NVMe paths: none");
    fmt::println("- NVMe paths: {}", fmt::join(settings.nvme_paths, ","));
    std::vector<std::string_view> reads;
    for (std::string_view const path : settings.nvme_paths) reads.push_back(nvme_reads_name(path));
    fmt::println("- NVMe reads: {}", fmt::join(reads, ","));
}

/** Prints each setting as "- Name: value", in the grammar it parses from. */
inline void print(settings_t const& settings) {
    print_environment(settings);
    fmt::println("- Residency: {}", usearch_residency_name(settings.residency));
    // Unpinned runs dispatch on the index, whose "- Index:" line names the class it picked.
    fmt::println("- CPU class: {}", settings.cpu_class == usearch_cpu_class_unknown_k
                                        ? "derived"
                                        : usearch_cpu_class_name(settings.cpu_class));
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
 *  executor of its devices, writing outputs to an arena on the first of them. An NVMe session
 *  builds on a RAM executor and serves views on an NVMe executor of its own, so a built index is
 *  saved and viewed before it is searched there.
 */
struct session_t {
    usearch_residency_t residency = usearch_residency_ram_k;

    /** The seed every key this session's entries take is minted under, by @ref key_of_offset. */
    std::uint32_t seed = 0;

    /** The class every index this session opens dispatches as. */
    usearch_cpu_class_t cpu_class = usearch_cpu_class_unknown_k;
    usearch_gpu_class_t gpu_class = usearch_gpu_class_unknown_k;

    /** The USearch device ordinals a VRAM index places shards on, in turn. */
    std::vector<std::int32_t> devices;

    /** The NVMe namespaces a save places its parts on, one per part, borrowed from the settings. */
    std::vector<std::string_view> places;

    /** What views read those parts through: the places themselves, or under passthrough each block
     *  node's generic character node. */
    std::vector<std::string> served;

    /** The served namespaces as the C interface borrows them. */
    std::vector<usearch_bytes_view_t> paths;

    /** The memory domains RAM shards repeat over, in turn. */
    std::vector<std::int32_t> memory_domains;

    /** Entries one shard holds before another shard generation opens; zero leaves it unbounded. */
    std::size_t shard_capacity = 0;

    /** The executor building every index: RAM threads, or the devices' streams. */
    usearch_executor_t executor = nullptr;

    /** The NVMe executor views serve on; null outside an NVMe session. */
    usearch_executor_t storage = nullptr;

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
        usearch_executor_free(storage);
        usearch_executor_free(executor);
    }

    void swap(session_t& other) noexcept {
        std::swap(residency, other.residency), std::swap(seed, other.seed), std::swap(cpu_class, other.cpu_class);
        std::swap(gpu_class, other.gpu_class), std::swap(devices, other.devices);
        std::swap(places, other.places), std::swap(served, other.served), std::swap(paths, other.paths);
        std::swap(memory_domains, other.memory_domains);
        std::swap(shard_capacity, other.shard_capacity), std::swap(executor, other.executor);
        std::swap(storage, other.storage), std::swap(arena, other.arena), std::swap(host_arena, other.host_arena);
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

    /** The arena serving an index of @p index_residency , which @ref executor_for pairs it with. */
    usearch_arena_t arena_for(usearch_residency_t index_residency) const noexcept {
        return index_residency == usearch_residency_ram_k && residency == usearch_residency_vram_k ? host_arena : arena;
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

/** The generic character node serving the namespace at @p address, a block node of the kernel's
 *  NVMe driver, at the same base and length; empty for any other address. */
inline std::string character_sibling(std::string_view address) {
    // The kernel names a namespace's character node after its block node: nvmeXnY and ngXnY.
    std::string_view const block_prefix = "/dev/nvme";
    if (!address.starts_with(block_prefix)) return {};
    return fmt::format("/dev/ng{}", address.substr(block_prefix.size()));
}

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
    session.nvme_ram.memory_limit = settings.nvme_ram.value;
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
    // Parts are written through block nodes, and passthrough reads them through character nodes.
    bool const passthrough = settings.nvme_class == usearch_nvme_class_linux_uring_passthrough_k;
    for (std::string_view const place : settings.nvme_paths) {
        std::string const sibling = character_sibling(place);
        session.places.push_back(place);
        session.served.push_back(passthrough && !sibling.empty() ? sibling : std::string {place});
    }
    for (std::string const& path : session.served) session.paths.push_back(bytes_view(path));
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
    if (report_failure(outcome, "Opening the arena")) return session;
    if (session.residency == usearch_residency_vram_k) {
        usearch_arena_init_t host_arena {.arena = &session.host_arena, .executor = nullptr, .outcome = &outcome};
        usearch_arena_init(&host_arena);
        report_failure(outcome, "Opening the host arena");
    }
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
                                usearch_ram_options_t ram, geometry_t const&, usearch_outcome_t* outcome) noexcept {
        usearch_dense_load_ram_buffer_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome, arguments.buffer = buffer;
        arguments.ram = ram;
        usearch_dense_load_ram_buffer(&arguments);
    }
    static void metadata_read_buffer(usearch_executor_t executor, usearch_bytes_view_t buffer, metadata_t* metadata,
                                     usearch_outcome_t* outcome) noexcept {
        usearch_dense_metadata_read_buffer_t arguments {};
        arguments.executor = executor, arguments.outcome = outcome, arguments.buffer = buffer;
        arguments.metadata = metadata;
        usearch_dense_metadata_read_buffer(&arguments);
    }
    static char const* metric_name(metric_t metric) noexcept { return usearch_dense_metric_name(metric); }
    static metric_t metric_from_name(usearch_bytes_view_t name) noexcept {
        return usearch_dense_metric_from_name(name);
    }
    static void load_vram(index_t* index, usearch_executor_t executor, usearch_bytes_view_t path,
                          usearch_vram_options_t vram, geometry_t const&, usearch_outcome_t* outcome) noexcept {
        usearch_dense_load_vram_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome, arguments.path = path;
        arguments.vram = vram;
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
    static void refine(index_t index, usearch_executor_t executor, usearch_refine_report_t* report,
                       usearch_outcome_t* outcome, std::size_t expansion = 0, std::size_t memory_limit = 0,
                       usearch_traversal_stats_t* stats = nullptr) noexcept {
        usearch_dense_refine_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome, arguments.report = report;
        arguments.refine = {expansion, memory_limit};
        arguments.stats = stats;
        usearch_dense_refine(&arguments);
    }
    static void absorb(index_t index, usearch_executor_t executor, index_t* donors, std::size_t count,
                       usearch_outcome_t* outcome, usearch_refine_report_t* report = nullptr, std::size_t expansion = 0,
                       std::size_t memory_limit = 0) noexcept {
        usearch_dense_absorb_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome;
        arguments.count = count, arguments.donors = donors;
        arguments.report = report;
        arguments.refine = {expansion, memory_limit};
        usearch_dense_absorb(&arguments);
    }
    static void health(index_t index, usearch_executor_t executor, usearch_reachability_t* reachability,
                       usearch_outcome_t* outcome) noexcept {
        usearch_dense_health_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome;
        arguments.reachability = reachability;
        usearch_dense_health(&arguments);
    }
    static void split(index_t index, usearch_executor_t executor, index_t const* receivers, std::size_t count,
                      std::uint64_t seed, usearch_refine_report_t* report, usearch_outcome_t* outcome,
                      std::size_t expansion = 0, std::size_t memory_limit = 0) noexcept {
        usearch_dense_split_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome;
        arguments.count = count, arguments.receivers = receivers, arguments.seed = seed, arguments.report = report;
        arguments.refine = {expansion, memory_limit};
        usearch_dense_split(&arguments);
    }
    using refinement_t = usearch_dense_refinement_t;
    static void refinement_init(index_t index, usearch_executor_t executor, refinement_t* refinement,
                                usearch_outcome_t* outcome) noexcept {
        usearch_dense_refinement_init_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome;
        arguments.refinement = refinement;
        usearch_dense_refinement_init(&arguments);
    }
    static void refinement_next(index_t index, usearch_executor_t executor, refinement_t refinement,
                                usearch_progress_t progress, std::size_t* reselected, std::size_t* changed,
                                usearch_outcome_t* outcome) noexcept {
        usearch_dense_refinement_next_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome;
        arguments.refinement = refinement, arguments.progress = progress;
        arguments.reselected = reselected, arguments.changed = changed;
        usearch_dense_refinement_next(&arguments);
    }
    static void refinement_finish(index_t index, usearch_executor_t executor, refinement_t refinement,
                                  usearch_refine_report_t* report, usearch_outcome_t* outcome) noexcept {
        usearch_dense_refinement_finish_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome;
        arguments.refinement = refinement, arguments.report = report;
        usearch_dense_refinement_finish(&arguments);
    }
    using cursor_t = usearch_dense_cursor_t;
    static void cursor_init(index_t index, std::span<usearch_key_t const> starts, cursor_t* cursor,
                            usearch_outcome_t* outcome) noexcept {
        usearch_dense_cursor_init_t arguments {};
        arguments.cursor = cursor, arguments.index = index, arguments.outcome = outcome;
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

/** What a sequence index is opened with beyond its metadata: the weights and window widths,
 *  borrowed. */
struct sequence_geometry_t {
    usearch_sequence_metadata_t metadata {};
    usearch_sequence_weights_t const* weights = nullptr;
    std::span<std::size_t const> window_widths;
};

/** A tape over @p offsets as a caller holding @c std::size_t offsets lends them, in their width. */
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
        if (hints) return void(*outcome = {usearch_status_unsupported_k, "Sequence searches take no hints"});
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
        arguments.ram = ram, arguments.weights = geometry.weights;
        usearch_sequence_load_ram_buffer(&arguments);
    }
    static void metadata_read_buffer(usearch_executor_t executor, usearch_bytes_view_t buffer, metadata_t* metadata,
                                     usearch_outcome_t* outcome) noexcept {
        usearch_sequence_metadata_read_buffer_t arguments {};
        arguments.executor = executor, arguments.outcome = outcome, arguments.buffer = buffer;
        arguments.metadata = metadata;
        usearch_sequence_metadata_read_buffer(&arguments);
    }
    static char const* metric_name(metric_t metric) noexcept { return usearch_sequence_metric_name(metric); }
    static metric_t metric_from_name(usearch_bytes_view_t name) noexcept {
        return usearch_sequence_metric_from_name(name);
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
                          usearch_nvme_options_t nvme, usearch_ram_options_t const*, geometry_t const& geometry,
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
    static void refine(index_t index, usearch_executor_t executor, usearch_refine_report_t* report,
                       usearch_outcome_t* outcome, std::size_t expansion = 0, std::size_t memory_limit = 0,
                       usearch_traversal_stats_t* stats = nullptr) noexcept {
        usearch_sequence_refine_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome, arguments.report = report;
        arguments.refine = {expansion, memory_limit};
        arguments.stats = stats;
        usearch_sequence_refine(&arguments);
    }
    static void absorb(index_t index, usearch_executor_t executor, index_t* donors, std::size_t count,
                       usearch_outcome_t* outcome, usearch_refine_report_t* report = nullptr, std::size_t expansion = 0,
                       std::size_t memory_limit = 0) noexcept {
        usearch_sequence_absorb_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome;
        arguments.count = count, arguments.donors = donors;
        arguments.report = report;
        arguments.refine = {expansion, memory_limit};
        usearch_sequence_absorb(&arguments);
    }
    static void health(index_t index, usearch_executor_t executor, usearch_reachability_t* reachability,
                       usearch_outcome_t* outcome) noexcept {
        usearch_sequence_health_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome;
        arguments.reachability = reachability;
        usearch_sequence_health(&arguments);
    }
    static void split(index_t index, usearch_executor_t executor, index_t const* receivers, std::size_t count,
                      std::uint64_t seed, usearch_refine_report_t* report, usearch_outcome_t* outcome,
                      std::size_t expansion = 0, std::size_t memory_limit = 0) noexcept {
        usearch_sequence_split_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome;
        arguments.count = count, arguments.receivers = receivers, arguments.seed = seed, arguments.report = report;
        arguments.refine = {expansion, memory_limit};
        usearch_sequence_split(&arguments);
    }
    using refinement_t = usearch_sequence_refinement_t;
    static void refinement_init(index_t index, usearch_executor_t executor, refinement_t* refinement,
                                usearch_outcome_t* outcome) noexcept {
        usearch_sequence_refinement_init_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome;
        arguments.refinement = refinement;
        usearch_sequence_refinement_init(&arguments);
    }
    static void refinement_next(index_t index, usearch_executor_t executor, refinement_t refinement,
                                usearch_progress_t progress, std::size_t* reselected, std::size_t* changed,
                                usearch_outcome_t* outcome) noexcept {
        usearch_sequence_refinement_next_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome;
        arguments.refinement = refinement, arguments.progress = progress;
        arguments.reselected = reselected, arguments.changed = changed;
        usearch_sequence_refinement_next(&arguments);
    }
    static void refinement_finish(index_t index, usearch_executor_t executor, refinement_t refinement,
                                  usearch_refine_report_t* report, usearch_outcome_t* outcome) noexcept {
        usearch_sequence_refinement_finish_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome;
        arguments.refinement = refinement, arguments.report = report;
        usearch_sequence_refinement_finish(&arguments);
    }
    using cursor_t = usearch_sequence_cursor_t;
    static void cursor_init(index_t index, std::span<usearch_key_t const> starts, cursor_t* cursor,
                            usearch_outcome_t* outcome) noexcept {
        usearch_sequence_cursor_init_t arguments {};
        arguments.cursor = cursor, arguments.index = index, arguments.outcome = outcome;
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
                         usearch_ram_options_t ram, geometry_t const&, usearch_outcome_t* outcome) noexcept {
        usearch_shapes_load_ram_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome, arguments.path = path;
        arguments.ram = ram;
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
                                usearch_ram_options_t ram, geometry_t const&, usearch_outcome_t* outcome) noexcept {
        usearch_shapes_load_ram_buffer_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome, arguments.buffer = buffer;
        arguments.ram = ram;
        usearch_shapes_load_ram_buffer(&arguments);
    }
    static void metadata_read_buffer(usearch_executor_t executor, usearch_bytes_view_t buffer, metadata_t* metadata,
                                     usearch_outcome_t* outcome) noexcept {
        usearch_shapes_metadata_read_buffer_t arguments {};
        arguments.executor = executor, arguments.outcome = outcome, arguments.buffer = buffer;
        arguments.metadata = metadata;
        usearch_shapes_metadata_read_buffer(&arguments);
    }
    static char const* metric_name(metric_t metric) noexcept { return usearch_shapes_metric_name(metric); }
    static metric_t metric_from_name(usearch_bytes_view_t name) noexcept {
        return usearch_shapes_metric_from_name(name);
    }
    static void load_vram(index_t* index, usearch_executor_t executor, usearch_bytes_view_t path,
                          usearch_vram_options_t vram, geometry_t const&, usearch_outcome_t* outcome) noexcept {
        usearch_shapes_load_vram_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome, arguments.path = path;
        arguments.vram = vram;
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
    static void refine(index_t index, usearch_executor_t executor, usearch_refine_report_t* report,
                       usearch_outcome_t* outcome, std::size_t expansion = 0, std::size_t memory_limit = 0,
                       usearch_traversal_stats_t* stats = nullptr) noexcept {
        usearch_shapes_refine_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome, arguments.report = report;
        arguments.refine = {expansion, memory_limit};
        arguments.stats = stats;
        usearch_shapes_refine(&arguments);
    }
    static void absorb(index_t index, usearch_executor_t executor, index_t* donors, std::size_t count,
                       usearch_outcome_t* outcome, usearch_refine_report_t* report = nullptr, std::size_t expansion = 0,
                       std::size_t memory_limit = 0) noexcept {
        usearch_shapes_absorb_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome;
        arguments.count = count, arguments.donors = donors;
        arguments.report = report;
        arguments.refine = {expansion, memory_limit};
        usearch_shapes_absorb(&arguments);
    }
    static void health(index_t index, usearch_executor_t executor, usearch_reachability_t* reachability,
                       usearch_outcome_t* outcome) noexcept {
        usearch_shapes_health_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome;
        arguments.reachability = reachability;
        usearch_shapes_health(&arguments);
    }
    static void split(index_t index, usearch_executor_t executor, index_t const* receivers, std::size_t count,
                      std::uint64_t seed, usearch_refine_report_t* report, usearch_outcome_t* outcome,
                      std::size_t expansion = 0, std::size_t memory_limit = 0) noexcept {
        usearch_shapes_split_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome;
        arguments.count = count, arguments.receivers = receivers, arguments.seed = seed, arguments.report = report;
        arguments.refine = {expansion, memory_limit};
        usearch_shapes_split(&arguments);
    }
    using refinement_t = usearch_shapes_refinement_t;
    static void refinement_init(index_t index, usearch_executor_t executor, refinement_t* refinement,
                                usearch_outcome_t* outcome) noexcept {
        usearch_shapes_refinement_init_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome;
        arguments.refinement = refinement;
        usearch_shapes_refinement_init(&arguments);
    }
    static void refinement_next(index_t index, usearch_executor_t executor, refinement_t refinement,
                                usearch_progress_t progress, std::size_t* reselected, std::size_t* changed,
                                usearch_outcome_t* outcome) noexcept {
        usearch_shapes_refinement_next_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome;
        arguments.refinement = refinement, arguments.progress = progress;
        arguments.reselected = reselected, arguments.changed = changed;
        usearch_shapes_refinement_next(&arguments);
    }
    static void refinement_finish(index_t index, usearch_executor_t executor, refinement_t refinement,
                                  usearch_refine_report_t* report, usearch_outcome_t* outcome) noexcept {
        usearch_shapes_refinement_finish_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome;
        arguments.refinement = refinement, arguments.report = report;
        usearch_shapes_refinement_finish(&arguments);
    }
    using cursor_t = usearch_shapes_cursor_t;
    static void cursor_init(index_t index, std::span<usearch_key_t const> starts, cursor_t* cursor,
                            usearch_outcome_t* outcome) noexcept {
        usearch_shapes_cursor_init_t arguments {};
        arguments.cursor = cursor, arguments.index = index, arguments.outcome = outcome;
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
        usearch_geospatial_load_ram_buffer(&arguments);
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
    static void refine(index_t index, usearch_executor_t executor, usearch_refine_report_t* report,
                       usearch_outcome_t* outcome, std::size_t expansion = 0, std::size_t memory_limit = 0,
                       usearch_traversal_stats_t* stats = nullptr) noexcept {
        usearch_geospatial_refine_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome, arguments.report = report;
        arguments.refine = {expansion, memory_limit};
        arguments.stats = stats;
        usearch_geospatial_refine(&arguments);
    }
    static void absorb(index_t index, usearch_executor_t executor, index_t* donors, std::size_t count,
                       usearch_outcome_t* outcome, usearch_refine_report_t* report = nullptr, std::size_t expansion = 0,
                       std::size_t memory_limit = 0) noexcept {
        usearch_geospatial_absorb_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome;
        arguments.count = count, arguments.donors = donors;
        arguments.report = report;
        arguments.refine = {expansion, memory_limit};
        usearch_geospatial_absorb(&arguments);
    }
    static void health(index_t index, usearch_executor_t executor, usearch_reachability_t* reachability,
                       usearch_outcome_t* outcome) noexcept {
        usearch_geospatial_health_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome;
        arguments.reachability = reachability;
        usearch_geospatial_health(&arguments);
    }
    static void split(index_t index, usearch_executor_t executor, index_t const* receivers, std::size_t count,
                      std::uint64_t seed, usearch_refine_report_t* report, usearch_outcome_t* outcome,
                      std::size_t expansion = 0, std::size_t memory_limit = 0) noexcept {
        usearch_geospatial_split_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome;
        arguments.count = count, arguments.receivers = receivers, arguments.seed = seed, arguments.report = report;
        arguments.refine = {expansion, memory_limit};
        usearch_geospatial_split(&arguments);
    }
    using refinement_t = usearch_geospatial_refinement_t;
    static void refinement_init(index_t index, usearch_executor_t executor, refinement_t* refinement,
                                usearch_outcome_t* outcome) noexcept {
        usearch_geospatial_refinement_init_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome;
        arguments.refinement = refinement;
        usearch_geospatial_refinement_init(&arguments);
    }
    static void refinement_next(index_t index, usearch_executor_t executor, refinement_t refinement,
                                usearch_progress_t progress, std::size_t* reselected, std::size_t* changed,
                                usearch_outcome_t* outcome) noexcept {
        usearch_geospatial_refinement_next_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome;
        arguments.refinement = refinement, arguments.progress = progress;
        arguments.reselected = reselected, arguments.changed = changed;
        usearch_geospatial_refinement_next(&arguments);
    }
    static void refinement_finish(index_t index, usearch_executor_t executor, refinement_t refinement,
                                  usearch_refine_report_t* report, usearch_outcome_t* outcome) noexcept {
        usearch_geospatial_refinement_finish_t arguments {};
        arguments.index = index, arguments.executor = executor, arguments.outcome = outcome;
        arguments.refinement = refinement, arguments.report = report;
        usearch_geospatial_refinement_finish(&arguments);
    }
    using cursor_t = usearch_geospatial_cursor_t;
    static void cursor_init(index_t index, std::span<usearch_key_t const> starts, cursor_t* cursor,
                            usearch_outcome_t* outcome) noexcept {
        usearch_geospatial_cursor_init_t arguments {};
        arguments.cursor = cursor, arguments.index = index, arguments.outcome = outcome;
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
    u64_t block_bytes = 0;

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
    std::vector<std::int32_t> placement;
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
template <typename verbs_type_, typename progress_type_ = dummy_progress_t>
usearch_outcome_t insert_entries(handle<verbs_type_>& index, typename verbs_type_::values_t inputs, std::size_t count,
                                 std::size_t first_offset, std::size_t batch, std::size_t expansion,
                                 usearch_traversal_stats_t& totals, progress_type_&& progress = {}) {
    usearch_outcome_t outcome {};
    std::size_t const batch_entries = std::max<std::size_t>(1, std::min(batch, count));
    std::vector<usearch_key_t> keys(batch_entries);
    for (std::size_t first = 0; first < count; first += batch_entries) {
        std::size_t const entries = std::min(batch_entries, count - first);
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
    std::vector<std::int32_t> placement;
    if (residency == usearch_residency_nvme_k)
        verbs_type_::view_nvme(&reopened, session.storage, bytes_view(path), session.nvme_options(),
                               session.nvme_ram_options(), geometry, &outcome);
    else if (residency == usearch_residency_vram_k)
        verbs_type_::load_vram(&reopened, session.executor, bytes_view(path), session.vram_options(shards, placement),
                               geometry, &outcome);
    else
        verbs_type_::load_ram(&reopened, session.executor, bytes_view(path), session.ram_options(0, placement),
                              geometry, &outcome);
    report_failure(outcome, fmt::format("Reopening {}", path));
    handle<verbs_type_> opened {reopened, session, residency, geometry};
    if (residency == usearch_residency_nvme_k && !names_transport_id(path)) {
        auto const [medium, medium_status] = file_medium_t::open(path);
        if (succeeded(medium_status)) opened.block_bytes = block_bytes_of(medium);
    }
    return opened;
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
    if (report_failure(outcome, fmt::format("Saving {}", path))) return {};
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

/** Entries end to end and where each starts, as the C interface takes a tape of bytes. */
struct byte_tape_t {
    std::string bytes;
    std::vector<std::size_t> offsets {0};

    std::size_t size() const noexcept { return offsets.size() - 1; }
    usearch_tape_view_t view() const noexcept { return tape_of(bytes.data(), offsets.data(), usearch_scalar_u8_k); }
};

/** The lines of @p text as entries, each newline dropped with a carriage return before it. */
inline byte_tape_t lines_of(bytes_view_t text) {
    byte_tape_t lines;
    lines.bytes.reserve(text.size());
    for (byte_t const byte : text) {
        char const character = static_cast<char>(byte);
        if (character != '\n') {
            lines.bytes.push_back(character);
            continue;
        }
        if (lines.bytes.size() != lines.offsets.back() && lines.bytes.back() == '\r') lines.bytes.pop_back();
        lines.offsets.push_back(lines.bytes.size());
    }
    if (lines.bytes.size() != lines.offsets.back()) lines.offsets.push_back(lines.bytes.size());
    return lines;
}

/** Point clouds mapped from two RetriEval files, every cloud's points end to end and the points per
 *  cloud, and each cloud's first point, as a tape takes them. */
struct point_clouds_t {
    mapped_rows_t points;
    mapped_rows_t counts;
    std::vector<std::size_t> offsets {0};

    std::size_t size() const noexcept { return offsets.size() - 1; }
    usearch_tape_view_t view() const noexcept {
        return tape_of(points.view(0, points.rows()).data, offsets.data(), points.scalar());
    }
};

/** Maps the clouds of @p points_path, 3 f32 or f16 columns per point, and @p counts_path, 1 u32
 *  column per cloud, into @p clouds; false after printing why it cannot. */
inline bool open_point_clouds(std::string_view points_path, std::string_view counts_path, point_clouds_t& clouds) {
    auto [points, points_status] = mapped_rows_t::open(points_path);
    if (failed(points_status) || points.columns() != 3 ||
        (points.scalar() != usearch_scalar_f32_k && points.scalar() != usearch_scalar_f16_k)) {
        fmt::println(stderr, "{} must hold 3 f32 or f16 columns per point", points_path);
        return false;
    }
    auto [counts, counts_status] = mapped_rows_t::open(counts_path);
    if (failed(counts_status) || counts.columns() != 1) {
        fmt::println(stderr, "{} must hold 1 column per cloud", counts_path);
        return false;
    }
    clouds.points = std::move(points), clouds.counts = std::move(counts);
    std::uint32_t const* const point_counts = static_cast<std::uint32_t const*>(
        clouds.counts.view(0, clouds.counts.rows()).data);
    for (std::size_t cloud = 0; cloud != clouds.counts.rows(); ++cloud)
        clouds.offsets.push_back(clouds.offsets.back() + point_counts[cloud]);
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
    std::vector<uuid_t> keys;
    std::vector<usearch_distance_t> scores;
    std::vector<std::size_t> counts;

    bool operator==(copied_matches_t const&) const = default;
};

/** Appends the matches the first @p queries queries of @p matches hold to @p copied, read back from
 *  wherever the arena placed them; false when the copy failed. */
inline bool append_matches(usearch_matches_view_t const& matches, std::size_t queries, copied_matches_t& copied) {
    std::vector<usearch_key_t> keys(queries * matches.columns);
    std::vector<usearch_distance_t> scores(queries * matches.columns);
    std::vector<std::size_t> counts(queries);
    if (!download(keys.data(), matches.keys, keys.size() * sizeof(usearch_key_t)) ||
        !download(scores.data(), matches.scores, scores.size() * sizeof(usearch_distance_t)) ||
        !download(counts.data(), matches.counts, counts.size() * sizeof(std::size_t)))
        return false;
    for (std::size_t query = 0; query != queries; ++query) {
        copied.counts.push_back(counts[query]);
        for (std::size_t match = query * matches.columns; match != query * matches.columns + counts[query]; ++match) {
            copied.keys.push_back(std::bit_cast<uuid_t>(keys[match]));
            copied.scores.push_back(scores[match]);
        }
    }
    return true;
}

/** The matches the first @p queries queries of @p matches hold, copied off a host arena. */
inline copied_matches_t copy_matches(usearch_matches_view_t const& matches, std::size_t queries) {
    copied_matches_t copied {{}, {}, {matches.counts, matches.counts + queries}};
    for (std::size_t query = 0; query != queries; ++query)
        for (std::size_t match = query * matches.columns; match != query * matches.columns + matches.counts[query];
             ++match) {
            copied.keys.push_back(std::bit_cast<uuid_t>(matches.keys[match]));
            copied.scores.push_back(matches.scores[match]);
        }
    return copied;
}

/** Searches @p count queries of @p queries in batches of @p batch, each batch handed its rows of
 *  the optional @p hints, copying every match to @p copied and its telemetry into @p totals. */
template <typename verbs_type_>
usearch_outcome_t search_queries(handle<verbs_type_>& index, typename verbs_type_::queries_t queries, std::size_t count,
                                 std::size_t batch, std::size_t wanted, std::size_t expansion,
                                 usearch_traversal_t traversal, usearch_traversal_stats_t& totals,
                                 copied_matches_t& copied, usearch_matches_view_t const* hints = nullptr) {
    usearch_outcome_t outcome {};
    std::size_t const batch_queries = std::max<std::size_t>(1, std::min(batch, count));
    for (std::size_t first = 0; first < count; first += batch_queries) {
        std::size_t const sliced = std::min(batch_queries, count - first);
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

/** The truth a ground-truth file lists: @p depth offsets per query, each keyed by
 *  @ref key_of_offset under @p seed, without distances. */
inline copied_matches_t truth_oracle(u64_t seed, std::span<std::uint32_t const> offsets, std::size_t depth) {
    copied_matches_t oracle;
    std::size_t const queries = depth ? offsets.size() / depth : 0;
    oracle.counts.assign(queries, depth);
    for (std::uint32_t const offset : offsets.first(queries * depth))
        oracle.keys.push_back(std::bit_cast<uuid_t>(key_of_offset(seed, offset)));
    return oracle;
}

/** Whether @p found ranks no further than @p exact, within the rounding two kits may differ by. */
inline bool within(double found, double exact) noexcept {
    return found <= exact + 1e-6 * std::max(1.0, std::fabs(exact));
}

/** The offset of every query's first match in @p copied. */
inline std::vector<std::size_t> match_offsets(copied_matches_t const& copied) {
    std::vector<std::size_t> offsets(copied.counts.size() + 1);
    for (std::size_t query = 0; query != copied.counts.size(); ++query)
        offsets[query + 1] = offsets[query] + copied.counts[query];
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
void judge_queries(std::size_t k, copied_matches_t const& found, copied_matches_t const& oracle, visit_type_&& visit) {
    std::vector<std::size_t> const found_offsets = match_offsets(found), truth_offsets = match_offsets(oracle);
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
inline double recall_at(std::size_t k, copied_matches_t const& found, copied_matches_t const& oracle) {
    double recall = 0;
    std::size_t answerable = 0;
    judge_queries(k, found, oracle, [&](std::size_t expected, std::vector<bool> const& hits) {
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
inline double ndcg_at(std::size_t k, copied_matches_t const& found, copied_matches_t const& oracle) {
    double gain_sum = 0;
    std::size_t answerable = 0;
    judge_queries(k, found, oracle, [&](std::size_t expected, std::vector<bool> const& hits) {
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
    u64_t reads = 0;
    u64_t bytes = 0;
};

/** The text of the file @p name in this process's cgroup v2 directory, none where the kernel keeps
 *  no such file, as off Linux or in the root cgroup. */
inline std::optional<std::string> cgroup_text(std::string_view name) {
    std::optional<std::string> const membership = read_text("/proc/self/cgroup");
    // The unified hierarchy's line reads `0::<path>`.
    std::size_t const unified = membership ? membership->find("0::") : std::string::npos;
    if (unified == std::string::npos || (unified && (*membership)[unified - 1] != '\n')) return std::nullopt;
    std::size_t const end = membership->find('\n', unified);
    std::string_view const scope = std::string_view {*membership}.substr(unified + 3, end - unified - 3);
    return read_text(fmt::format("/sys/fs/cgroup{}/{}", scope, name).c_str());
}

/** What @c /proc/self/io counts this process fetched from the block layer so far, every kernel
 *  engine's reads included, as reads of @p block_bytes each; none off Linux or without a block. */
inline std::optional<cgroup_reads_t> process_reads(u64_t block_bytes) {
    std::optional<std::string> const io = read_text("/proc/self/io");
    constexpr std::string_view key = "read_bytes: ";
    std::size_t const found = io ? io->find(key) : std::string::npos;
    if (!block_bytes || found == std::string::npos) return std::nullopt;
    cgroup_reads_t counted;
    std::from_chars_result const parsed = std::from_chars(io->data() + found + key.size(), io->data() + io->size(),
                                                          counted.bytes);
    if (parsed.ec != std::errc {}) return std::nullopt;
    counted.reads = counted.bytes / block_bytes;
    return counted;
}

/** What this process's cgroup @c io.stat counts so far, every kernel engine's reads on every
 *  disk, or what @ref process_reads counts in @p block_bytes reads under a user scope that keeps
 *  no such file. */
inline std::optional<cgroup_reads_t> cgroup_reads(u64_t block_bytes) {
    std::optional<std::string> const stat = cgroup_text("io.stat");
    if (!stat) return process_reads(block_bytes);
    cgroup_reads_t counted;
    std::string_view text {*stat};
    while (!text.empty()) {
        std::size_t const space = std::min(text.find_first_of(" \n"), text.size());
        std::string_view const token = text.substr(0, space);
        std::string_view const digits = token.substr(token.find('=') + 1);
        u64_t value = 0;
        std::from_chars_result const parsed = std::from_chars(digits.data(), digits.data() + digits.size(), value);
        if (parsed.ec == std::errc {} && token.starts_with("rbytes=")) counted.bytes += value;
        if (parsed.ec == std::errc {} && token.starts_with("rios=")) counted.reads += value;
        text.remove_prefix(std::min(space + 1, text.size()));
    }
    return counted;
}

/** The @p quantile of @p samples, reordering them around it. */
inline double percentile(std::span<u64_t> samples, double quantile) noexcept {
    std::size_t const rank = static_cast<std::size_t>(quantile * static_cast<double>(samples.size() - 1));
    std::nth_element(samples.begin(), samples.begin() + rank, samples.end());
    return static_cast<double>(samples[rank]);
}

/** Times single-query searches of @p queries on @p index 's executor one after another, from the
 *  first of @p count on, until @p budget has passed; @p latencies takes each call's nanoseconds. */
template <typename verbs_type_>
usearch_outcome_t time_single_queries(handle<verbs_type_>& index, typename verbs_type_::queries_t queries,
                                      std::size_t count, std::size_t wanted, std::size_t expansion,
                                      std::chrono::steady_clock::duration budget, std::vector<u64_t>& latencies) {
    usearch_outcome_t outcome {};
    latencies.clear();
    latencies.reserve(count);
    std::chrono::steady_clock::time_point const deadline = std::chrono::steady_clock::now() + budget;
    for (std::size_t query = 0; query != count && std::chrono::steady_clock::now() < deadline; ++query) {
        usearch_matches_view_t matches {};
        usearch_traversal_stats_t stats {};
        std::chrono::steady_clock::time_point const started = std::chrono::steady_clock::now();
        verbs_type_::search(index.get(), index.executor(), index.arena(), 1, verbs_type_::slice(queries, query), wanted,
                            expansion, usearch_traversal_walk_k, &matches, &stats, &outcome);
        latencies.push_back(static_cast<u64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count()));
        if (outcome.status != usearch_status_success_k) return outcome;
    }
    return outcome;
}

/** Searches @p count queries of @p queries at every breadth of @p expansions, printing each
 *  breadth's recall at one and at @p wanted and its nDCG against @p oracle, its throughput and its
 *  distances per query, then handing its matches to @p on_swept as @b (expansion,matches). A view
 *  on NVMe also prints the reads and bytes its cgroup's @c io.stat counted per query, and the p50
 *  and p99 of single-query calls over judged queries, timed after the throughput pass for as long
 *  as that pass took, since each one walks alone. */
template <typename verbs_type_, typename swept_type_ = dummy_progress_t>
usearch_outcome_t sweep_recall(handle<verbs_type_>& index, typename verbs_type_::queries_t queries, std::size_t count,
                               copied_matches_t const& oracle, std::span<std::size_t const> expansions,
                               std::size_t wanted, std::size_t batch, swept_type_&& on_swept = {}) {
    usearch_outcome_t outcome {};
    std::vector<u64_t> latencies;
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
        fmt::println("- Expansion {}: recall@1 {:.4f} · recall@{} {:.4f} · nDCG@{} {:.4f} over {} judged", expansion,
                     recall_at(1, found, oracle), wanted, recall_at(wanted, found, oracle), wanted,
                     ndcg_at(wanted, found, oracle), oracle.counts.size());
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
            std::size_t const judged = std::min(count, oracle.counts.size());
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
    for (int code = usearch_cpu_class_serial_k; code <= usearch_cpu_class_monoversioned_k; ++code) {
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
inline bool equivalent_matches(copied_matches_t const& first, copied_matches_t const& second, double score_band) {
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
 *  a manifest. The copy in the index's own residency must land matches equivalent to the original's
 *  under @p search, as <tt>search(handle) → copied_matches_t</tt>, within @p score_band when the stored
 *  values round on the way; every loaded copy must save again, byte for byte when @p bytes says so;
 *  the view must answer the search. Every file the round trip wrote is removed.
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
    copied_matches_t const original_matches = search(index);

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
            copied_matches_t const reloaded = search(reopened);
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
    verbs_type_::load_ram(&copy, session.executor_for(usearch_residency_ram_k), bytes_view(path),
                          session.ram_options(0, placement), index.geometry, &outcome);
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
        verbs_type_::view_nvme(&copy, session.storage, bytes_view(path), session.nvme_options(),
                               session.nvme_ram_options(), index.geometry, &outcome);
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
 *  The entries are keyed by their offsets, and the index is served from @p checkpoint in the
 *  session's residency. An NVMe session serves a view of what it built from @p checkpoint, or from
 *  the temporary directory when that is empty, and any other session reopens what it saved only
 *  when @p checkpoint names a path; the reopened index takes @p served, so a search may score under
 *  other weights than the build. Each breadth's matches go to @p on_swept as
 *  `(expansion, matches)`.
 *
 *  @return The process exit code.
 */
template <typename verbs_type_, typename swept_type_ = dummy_progress_t>
int run_search_bench(settings_t const& settings, search_bench_settings_t const& search,
                     typename verbs_type_::geometry_t const& geometry, typename verbs_type_::geometry_t const& served,
                     typename verbs_type_::values_t entries, std::size_t base_count,
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

    copied_matches_t oracle;
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
