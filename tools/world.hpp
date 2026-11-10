/**
 *  @file tools/world.hpp
 *  @author Ash Vardanian
 *  @date October 18, 2026
 *  @brief The process group a verb runs on, the verbs both tools dispatch to, and the verbs every
 *      modality shares over its checkpoints.
 *
 *  Each binary links one definition of @ref world_t: @c world_local.cpp for a world of one process,
 *  @c world_mpi.cpp for an MPI world, the only unit that includes MPI, so the library and every
 *  verb stay free of it. Ranks talk only through @ref world_t::barrier and @ref world_t::gather.
 *
 *  The checkpoint verbs run on one process over any modality's verbs table from @c abi.hpp: each
 *  modality's unit reads its own inputs and queries, and the insert, absorb, refine, split, health
 *  and search steps around them are written once here.
 */
#pragma once

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint64_t`
#include <cstdio>  // `std::FILE`, `stdout`, `stderr` - where a step prints
#include <cstdlib> // `EXIT_SUCCESS`, `EXIT_FAILURE`
#include <cstring> // `std::memcpy` - a `.u128bin` header

#include <sys/stat.h> // `stat`, `S_ISREG` - a saved part's bytes

#include <algorithm>   // `std::min`
#include <bit>         // `std::bit_cast` - entry numbers into keys
#include <array>       // `std::array` - the flag tables
#include <chrono>      // `std::chrono::steady_clock`
#include <iterator>    // `std::back_inserter`
#include <optional>    // `std::optional`
#include <span>        // `std::span`
#include <string>      // `std::string`
#include <string_view> // `std::string_view`
#include <type_traits> // `std::is_same_v` - the dense-only widening
#include <utility>     // `std::move`
#include <vector>      // `std::vector`

#include <fmt/format.h>

#include "usearch/utilities/safegraphs_format.hpp" // `safegraphs_part_path`, `safegraphs_write_manifest`

#include "abi.hpp"               // `flag_t`
#include "arrow_checkpoints.hpp" // `arrow_matches_write` - what `--output` writes
#include "slots.hpp"             // `checkpoints_admit_save`, `slot_copy` - saves onto a drive's slots

namespace unum::usearch::tools {

/** The ranks a verb runs on, started from the program's arguments and finished when it dies. */
struct world_t {

    /** This process among @ref size, counted from zero. */
    std::size_t rank = 0;

    /** Processes the verb runs on. */
    std::size_t size = 1;

    /** This process among those sharing its node, which picks its GPU or memory domain. */
    std::size_t local_rank = 0;

    world_t(int& argc, char**& argv) noexcept;
    world_t(world_t const&) = delete;
    world_t& operator=(world_t const&) = delete;
    ~world_t() noexcept;

    /** Waits for every rank and answers whether any of them passed a true @p failed. */
    bool barrier(bool failed) noexcept;

    /** Gathers @p sent from every rank into @p received on rank 0, rank after rank; the other ranks
     *  pass an empty @p received. */
    void gather(std::span<std::uint64_t const> sent, std::span<std::uint64_t> received) noexcept;
};

/** What leads a rank's lines while several run, like "rank 3: ", and nothing in a world of one. */
inline std::string rank_label(world_t const& world) {
    return world.size > 1 ? fmt::format("rank {}: ", world.rank) : std::string {};
}

/** Whether a verb that runs on one process may run here, rank 0 saying why not under several. */
inline bool runs_alone(world_t const& world) {
    if (world.size == 1) return true;
    if (world.rank == 0) fmt::println(stderr, "only dense insert runs across ranks; run this verb under usearch");
    return false;
}

/** Parses @p argv against @p flags, each defaulting to its @c RETRIEVAL_ variable, and collects
 *  up to @p positionals.size() positionals, counting them into @p given. Answers the status to
 *  exit with when the verb ends here: after the usage on @c --help, or after rank 0 names what
 *  failed. */
std::optional<int> parse_verb(world_t const& world, int argc, char const* const* argv, std::string_view synopsis,
                              std::span<flag_t const> flags, std::span<std::string_view> positionals,
                              std::size_t& given);

/** Each verb takes the world, then the arguments after its name with the name where a program name
 *  stands, and answers the process's exit status. The verbs reading a modality's own entries,
 *  receivers or queries are defined in its unit, picked by its verbs table; its other verbs read
 *  checkpoints alone, so the templates below serve every modality. */
int modality_insert(dense_verbs_t, world_t& world, int argc, char const* const* argv);
int modality_split(dense_verbs_t, world_t& world, int argc, char const* const* argv);
int modality_search(dense_verbs_t, world_t& world, int argc, char const* const* argv);
int dense_get(world_t& world, int argc, char const* const* argv);
int modality_insert(sequence_verbs_t, world_t& world, int argc, char const* const* argv);
int modality_split(sequence_verbs_t, world_t& world, int argc, char const* const* argv);
int modality_search(sequence_verbs_t, world_t& world, int argc, char const* const* argv);
int modality_insert(shapes_verbs_t, world_t& world, int argc, char const* const* argv);
int modality_split(shapes_verbs_t, world_t& world, int argc, char const* const* argv);
int modality_search(shapes_verbs_t, world_t& world, int argc, char const* const* argv);
int modality_insert(geospatial_verbs_t, world_t& world, int argc, char const* const* argv);
int modality_split(geospatial_verbs_t, world_t& world, int argc, char const* const* argv);
int modality_search(geospatial_verbs_t, world_t& world, int argc, char const* const* argv);
int checkpoints_metadata(world_t& world, int argc, char const* const* argv);
int checkpoints_copy(world_t& world, int argc, char const* const* argv);
int checkpoints_export(world_t& world, int argc, char const* const* argv);
int checkpoints_import(world_t& world, int argc, char const* const* argv);
int checkpoints_recall(world_t& world, int argc, char const* const* argv);

#pragma region Checkpoint Verbs

using checkpoint_clock_t = std::chrono::steady_clock;

inline double checkpoint_seconds_since(checkpoint_clock_t::time_point started) noexcept {
    return std::chrono::duration<double>(checkpoint_clock_t::now() - started).count();
}

/** The path of part @p part, counted from zero, of the @p parts the manifest at @p manifest_path
 *  commits; empty when it cannot be formatted. */
inline std::string checkpoint_part_path(std::string_view manifest_path, std::size_t part, std::size_t parts) {
    growing_buffer<char> named {global_allocator};
    return std::string {safegraphs_part_path(named, safegraphs_stem(manifest_path), part, parts).value};
}

/** Bytes the regular file at @p path holds, or zero for a device, an SPDK address or a failure. */
inline std::uint64_t checkpoint_file_bytes(std::string const& path) noexcept {
    struct stat file_stat;
    bool const measured = stat(path.c_str(), &file_stat) == 0 && S_ISREG(file_stat.st_mode);
    return measured ? static_cast<std::uint64_t>(file_stat.st_size) : 0;
}

/** Commits the checkpoint: the manifest at @p manifest_path naming each of @p parts beside it, with
 *  its bytes and nodes. */
inline bool checkpoint_write_manifest(std::string_view manifest_path, std::span<safegraphs_part_t const> parts) {
    std::string_view const stem = safegraphs_basename(safegraphs_stem(manifest_path));
    growing_buffer<char> named {global_allocator};
    return succeeded(safegraphs_write_manifest(manifest_path, parts.size(), [&](std::size_t part) noexcept {
        safegraphs_part_t entry = parts[part];
        entry.path = safegraphs_part_path(named, stem, part, parts.size()).value;
        return entry;
    }));
}

/** The text a key prints as: the entry number it was minted from, or its 32 hex digits. */
inline std::string checkpoint_key_text(uuid128_t const& key) {
    if (std::optional<u64_t> const number = identity_of(key)) return fmt::format("{}", *number);
    std::string text;
    for (std::byte const byte : key.bytes)
        fmt::format_to(std::back_inserter(text), "{:02x}", std::to_integer<unsigned>(byte));
    return text;
}

/** Every setting the checkpoint verbs take, each spelled as its flag with dashes; @c session holds
 *  the residency, the NVMe class, the threads and the devices a session opens with. */
struct checkpoint_settings_t {
    std::string_view save;
    count_t first_entry {.minimum = 0};
    count_t entries {.minimum = 0};
    count_t parts {.minimum = 2};
    count_t connectivity {.minimum = 0};
    count_t expansion_add {.minimum = 0};
    count_t expansion_refine {.minimum = 0};
    count_t expansion_search {.minimum = 0};
    count_t max_refines {.minimum = 0};
    count_t wanted {.value = 10};
    count_t batch {.value = 65536};
    bytes_t memory_limit;
    std::string_view query_keys;
    usearch_traversal_t traversal = usearch_traversal_walk_k;
    std::string_view output;
    settings_t session;
};

/** @p verb_flags, then the flags of the session every verb opens, then @p modality_flags. */
inline std::vector<flag_t> with_session_flags(settings_t& session, std::vector<flag_t> verb_flags,
                                              std::span<flag_t const> modality_flags = {}) {
    std::array<flag_t, 4> const shared = session_flags(session);
    verb_flags.insert(verb_flags.end(), shared.begin(), shared.end());
    verb_flags.insert(verb_flags.end(), modality_flags.begin(), modality_flags.end());
    return verb_flags;
}

/** @ref with_session_flags for a verb that refines, with the two flags every such verb takes. */
inline std::vector<flag_t> with_refine_flags(checkpoint_settings_t& settings, std::vector<flag_t> verb_flags,
                                             std::span<flag_t const> modality_flags = {}) {
    verb_flags.push_back({"--expansion-refine", "count",
                          "defaults to the library's derivation: candidates while refining",
                          &settings.expansion_refine});
    verb_flags.push_back({"--memory-limit", "bytes",
                          "defaults to a storage chunk per tile: bytes a pass holds past the index, like 40G",
                          &settings.memory_limit});
    return with_session_flags(settings.session, std::move(verb_flags), modality_flags);
}

/** The flag every verb calling the library in batches takes. */
inline flag_t batch_flag(checkpoint_settings_t& settings) noexcept {
    return {"--batch", "count", "defaults to 65536: entries per insertion, or queries per search", &settings.batch};
}

/** The flags every insert takes, then @p modality_flags. */
inline std::vector<flag_t> insert_flags(checkpoint_settings_t& settings, std::span<flag_t const> modality_flags) {
    return with_refine_flags(
        settings,
        {
            {"--save", "address",
             "required: where the index is saved; several ranks save a manifest there, a part each beside it",
             &settings.save},
            {"--first-entry", "offset", "defaults to 0: the first entry of the slice to index", &settings.first_entry},
            {"--entries", "count", "defaults to every entry past --first-entry: entries in the slice",
             &settings.entries},
            {"--connectivity", "count", "defaults to the library's derivation: graph degree", &settings.connectivity},
            {"--expansion-add", "count", "defaults to the library's derivation: candidates while inserting",
             &settings.expansion_add},
            {"--max-refines", "count", "defaults to 0: refinement passes after ingestion, stopping early once settled",
             &settings.max_refines},
            batch_flag(settings),
        },
        modality_flags);
}

/** What an insert indexes: the entries as the modality's verbs take them, how many there are, and
 *  the geometry the empty index opens with. */
template <typename verbs_type_>
struct checkpoint_entries {
    typename verbs_type_::values_t values {};
    std::size_t count = 0;
    typename verbs_type_::geometry_t geometry {};
};

/** What a search asks: the queries as the modality's verbs take them, how many there are, and the
 *  key of each when they arrive with theirs, else empty. */
template <typename verbs_type_>
struct checkpoint_queries {
    typename verbs_type_::queries_t queries {};
    std::size_t count = 0;
    std::vector<usearch_key_t> keys {};
};

/** Loads the checkpoint standard input carries into RAM on @p session; prints what failed and
 *  answers an empty handle. */
template <typename verbs_type_>
handle<verbs_type_> checkpoint_read_piped(session_t& session, usearch_residency_t residency,
                                          typename verbs_type_::geometry_t const& geometry) {
    if (residency != usearch_residency_ram_k) {
        fmt::println(stderr, "a checkpoint on standard input loads into RAM alone, not --residency={}",
                     usearch_residency_name(residency));
        return {};
    }
    std::vector<byte_t> bytes;
    if (!read_standard_input(bytes)) {
        fmt::println(stderr, "cannot read a checkpoint off standard input");
        return {};
    }
    usearch_outcome_t outcome {};
    typename verbs_type_::index_t loaded = nullptr;
    std::vector<std::int32_t> placement;
    usearch_bytes_view_t const buffer {reinterpret_cast<usearch_byte_t const*>(bytes.data()), bytes.size()};
    verbs_type_::load_ram_buffer(&loaded, session.executor, buffer, session.ram_options(0, placement), geometry,
                                 &outcome);
    report_failure(outcome, "Loading the checkpoint off standard input");
    return {loaded, session, residency, geometry};
}

/** Opens the checkpoint at @p address on @p session, or standard input for @c -: viewed from
 *  storage in NVMe, else loaded into RAM or onto the session's devices, under the load options
 *  @p geometry names; prints the step to @p stream, or what failed and an empty handle. */
template <typename verbs_type_>
handle<verbs_type_> checkpoint_open(session_t& session, std::string_view address, usearch_residency_t residency,
                                    typename verbs_type_::geometry_t const& geometry = {}, std::FILE* stream = stdout) {
    checkpoint_clock_t::time_point const started = checkpoint_clock_t::now();
    handle<verbs_type_> index = address == "-" ? checkpoint_read_piped<verbs_type_>(session, residency, geometry)
                                               : reopen<verbs_type_>(session, address, geometry, residency, 0);
    if (!index) return index;
    double const seconds = checkpoint_seconds_since(started);
    if (residency == usearch_residency_vram_k)
        fmt::println(stream, "loaded {} in {:.1f} s through {}", address, seconds,
                     usearch_load_path_name(describe(index).load_path));
    else if (residency != usearch_residency_nvme_k) fmt::println(stream, "loaded {} in {:.1f} s", address, seconds);
    else if (!session.nvme_ram_options()) fmt::println(stream, "viewed {} in {:.1f} s", address, seconds);
    else {
        typename verbs_type_::description_t const described = describe(index);
        fmt::println(stream, "viewed {} in {:.1f} s, holding keys of {} rows in {} bytes", address, seconds,
                     described.size, described.memory_usage);
    }
    return index;
}

/** Writes @p index to standard output as one buffer, read back off device memory first where the
 *  session's arena sits there, and prints the step to @p stream. */
template <typename verbs_type_>
bool checkpoint_save_piped(handle<verbs_type_> const& index, std::FILE* stream) {
    checkpoint_clock_t::time_point const started = checkpoint_clock_t::now();
    usearch_outcome_t outcome {};
    usearch_bytes_view_t monolith {};
    verbs_type_::save_buffer(index.get(), index.executor(), index.session->arena, &monolith, &outcome);
    if (report_failure(outcome, "saving to standard output")) return false;
    std::vector<byte_t> copied;
    void const* bytes = monolith.data;
    if (index.residency == usearch_residency_vram_k) {
        copied.resize(monolith.length);
        if (!download(copied.data(), monolith.data, monolith.length)) {
            fmt::println(stderr, "cannot read the checkpoint back off its device");
            return false;
        }
        bytes = copied.data();
    }
    if (std::fwrite(bytes, 1, monolith.length, stdout) != monolith.length || std::fflush(stdout) != 0) {
        fmt::println(stderr, "writing standard output failed");
        return false;
    }
    fmt::println(stream, "saved to standard output: {} bytes in {:.1f} s", monolith.length,
                 checkpoint_seconds_since(started));
    return true;
}

/** Saves @p index under @p address, or to standard output for @c -, and prints the step under this
 *  rank to @p stream. */
template <typename verbs_type_>
bool checkpoint_save(world_t const& world, handle<verbs_type_> const& index, std::string const& address,
                     std::FILE* stream = stdout) {
    if (address == "-") return checkpoint_save_piped(index, stream);
    checkpoint_clock_t::time_point const started = checkpoint_clock_t::now();
    usearch_outcome_t outcome {};
    verbs_type_::save(index.get(), index.executor(), bytes_view(address), &outcome);
    if (report_failure(outcome, fmt::format("{}saving {}", rank_label(world), address))) return false;
    double const seconds = checkpoint_seconds_since(started);
    if (std::uint64_t const bytes = checkpoint_file_bytes(address))
        fmt::println(stream, "{}saved {}: {} bytes in {:.1f} s", rank_label(world), address, bytes, seconds);
    else fmt::println(stream, "{}saved {} in {:.1f} s", rank_label(world), address, seconds);
    return true;
}

/** Runs up to @p passes refinements over @p index under @p refine, stopping once one changes
 *  nothing, and prints each under this rank to @p stream. */
template <typename verbs_type_>
bool checkpoint_refine_passes(world_t const& world, handle<verbs_type_> const& index, std::size_t passes,
                              usearch_refine_options_t refine, std::FILE* stream = stdout) {
    for (std::size_t pass = 0; pass != passes; ++pass) {
        checkpoint_clock_t::time_point const started = checkpoint_clock_t::now();
        usearch_outcome_t outcome {};
        usearch_refine_report_t report {};
        usearch_traversal_stats_t stats {};
        verbs_type_::refine(index.get(), index.executor(), &report, &outcome, refine.expansion_factor,
                            refine.memory_limit, &stats);
        if (report_failure(outcome, fmt::format("{}refining", rank_label(world)))) return false;
        fmt::println(stream, "{}refined pass {}: {} changed, {} of {} reachable in {:.1f} s, {:.1f} distances per row",
                     rank_label(world), pass + 1, report.changed, report.reachability.reachable,
                     report.reachability.audited, checkpoint_seconds_since(started),
                     double(stats.computed_distances) / double(std::max<std::size_t>(report.reselected, 1)));
        if (!report.changed) break;
    }
    return true;
}

/** Inserts a slice of the entries @p open_entries reads from the one positional into a new
 *  one-shard index in RAM or VRAM, keyed by their entry numbers, refines it and saves it. */
template <typename verbs_type_, typename open_entries_type_>
int checkpoint_insert(world_t& world, int argc, char const* const* argv, std::string_view synopsis,
                      std::span<flag_t const> modality_flags, open_entries_type_&& open_entries) {
    checkpoint_settings_t settings;
    settings.session.residency = usearch_residency_unknown_k;
    std::vector<flag_t> const flags = insert_flags(settings, modality_flags);
    std::array<std::string_view, 1> positionals;
    std::size_t given = 0;
    if (std::optional<int> const exit = parse_verb(world, argc, argv, synopsis, flags, positionals, given))
        return *exit;
    if (!runs_alone(world)) return EXIT_FAILURE;
    if (given != 1 || settings.save.empty()) {
        fmt::println(stderr, "the entries and --save are required");
        return EXIT_FAILURE;
    }
    if (settings.session.residency == usearch_residency_unknown_k)
        settings.session.residency = listed_devices().empty() ? usearch_residency_ram_k : usearch_residency_vram_k;
    if (settings.session.residency == usearch_residency_nvme_k) {
        fmt::println(stderr, "--residency=nvme builds nothing; insert in ram or vram, then view the checkpoint");
        return EXIT_FAILURE;
    }
    std::optional<std::string> const admitted = checkpoints_admit_save(settings.save);
    if (!admitted) return EXIT_FAILURE;
    std::FILE* const steps = settings.save == "-" ? stderr : stdout;
    std::optional<checkpoint_entries<verbs_type_>> const entries = open_entries(positionals[0], settings);
    if (!entries) return EXIT_FAILURE;
    std::size_t const first = settings.first_entry.value;
    if (first >= entries->count || settings.entries.value > entries->count - first) {
        fmt::println(stderr, "--first-entry={} --entries={} reach past the {} entries of {}", first,
                     settings.entries.value, entries->count, positionals[0]);
        return EXIT_FAILURE;
    }
    std::size_t const count = settings.entries.value ? settings.entries.value : entries->count - first;

    settings.session.devices = devices_from_environment();
    session_t session = open_session(settings.session, usearch_cpu_class_unknown_k);
    if (!session) return EXIT_FAILURE;
    checkpoint_clock_t::time_point const started = checkpoint_clock_t::now();
    // One shard keeps the saved monolith one graph, as the dense insert does.
    handle<verbs_type_> index = open_index<verbs_type_>(session, entries->geometry, 1);
    if (!index) return EXIT_FAILURE;
    std::size_t const batch = std::min(settings.batch.value, count);
    std::vector<usearch_key_t> keys(batch);
    for (std::size_t offset = 0; offset < count; offset += batch) {
        std::size_t const sliced = std::min(batch, count - offset);
        for (std::size_t position = 0; position != sliced; ++position)
            keys[position] = std::bit_cast<usearch_key_t>(uuid_of(first + offset + position));
        usearch_outcome_t outcome {};
        verbs_type_::insert(index.get(), index.executor(), sliced, keys_view(std::span {keys}.first(sliced)),
                            verbs_type_::slice(entries->values, first + offset), settings.expansion_add.value, nullptr,
                            &outcome);
        if (report_failure(outcome, "inserting")) return EXIT_FAILURE;
    }
    fmt::println(steps, "inserted entries {} to {} in {} in {:.1f} s", first, first + count,
                 usearch_residency_name(index.residency), checkpoint_seconds_since(started));
    bool const built = checkpoint_refine_passes(world, index, settings.max_refines.value,
                                                {settings.expansion_refine.value, settings.memory_limit.value},
                                                steps) &&
                       checkpoint_save(world, index, *admitted, steps);
    return built ? EXIT_SUCCESS : EXIT_FAILURE;
}

/** Loads a base and absorbs its donors into it one at a time, then saves the merged index. */
template <typename verbs_type_>
int checkpoint_absorb(world_t& world, int argc, char const* const* argv) {
    checkpoint_settings_t settings;
    std::vector<flag_t> const flags = with_refine_flags(
        settings, {{"--save", "address", "required: where the merged index is saved", &settings.save}});
    // Every argument is at most one positional, so the arguments bound the donors.
    std::vector<std::string_view> positionals(static_cast<std::size_t>(argc));
    std::size_t given = 0;
    std::string const synopsis = fmt::format("usearch {} absorb <base> <donors>... --save <address> [flags]",
                                             verbs_type_::name_k);
    if (std::optional<int> const exit = parse_verb(world, argc, argv, synopsis, flags, positionals, given))
        return *exit;
    if (!runs_alone(world)) return EXIT_FAILURE;
    if (given < 2 || settings.save.empty()) {
        fmt::println(stderr, "a base, at least one donor and --save are required");
        return EXIT_FAILURE;
    }
    std::optional<std::string> const admitted = checkpoints_admit_save(settings.save);
    if (!admitted) return EXIT_FAILURE;
    std::FILE* const steps = settings.save == "-" ? stderr : stdout;

    settings.session.devices = devices_from_environment();
    session_t session = open_session(settings.session, usearch_cpu_class_unknown_k);
    if (!session) return EXIT_FAILURE;
    handle<verbs_type_> base = checkpoint_open<verbs_type_>(session, positionals[0], settings.session.residency, {},
                                                            steps);
    if (!base) return EXIT_FAILURE;
    for (std::string_view const donor_address : std::span {positionals}.subspan(1, given - 1)) {
        handle<verbs_type_> donor = checkpoint_open<verbs_type_>(session, donor_address, settings.session.residency, {},
                                                                 steps);
        if (!donor) return EXIT_FAILURE;
        std::size_t const donor_entries = describe(donor).size;
        checkpoint_clock_t::time_point const started = checkpoint_clock_t::now();
        usearch_outcome_t outcome {};
        usearch_refine_report_t report {};
        // A consumed donor's handle is nulled; a refused one stays for its owner to free.
        verbs_type_::absorb(base.get(), base.executor(), &donor.index, 1, &outcome, &report,
                            settings.expansion_refine.value, settings.memory_limit.value);
        if (report_failure(outcome, fmt::format("absorbing {}", donor_address))) return EXIT_FAILURE;
        fmt::println(steps, "absorbed {}: {} entries in {:.1f} s, {} changed, {} of {} reachable", donor_address,
                     donor_entries, checkpoint_seconds_since(started), report.changed, report.reachability.reachable,
                     report.reachability.audited);
    }
    return checkpoint_save(world, base, *admitted, steps) ? EXIT_SUCCESS : EXIT_FAILURE;
}

/** Loads a checkpoint, a dense one widened to a larger connectivity when asked, re-selects its
 *  neighbors pass after pass, and saves it. */
template <typename verbs_type_>
int checkpoint_refine(world_t& world, int argc, char const* const* argv) {
    checkpoint_settings_t settings;
    settings.max_refines = {.value = 1};
    std::vector<flag_t> const flags = with_refine_flags(
        settings,
        {
            {"--save", "address", "required: where the refined index is saved", &settings.save},
            {"--connectivity", "count", "defaults to the checkpoint's: a wider graph degree to refine a dense one into",
             &settings.connectivity},
            {"--max-refines", "count", "defaults to 1: refinement passes, stopping early once settled",
             &settings.max_refines},
        });
    std::array<std::string_view, 1> positionals;
    std::size_t given = 0;
    std::string const synopsis = fmt::format("usearch {} refine <address> --save <address> [flags]",
                                             verbs_type_::name_k);
    if (std::optional<int> const exit = parse_verb(world, argc, argv, synopsis, flags, positionals, given))
        return *exit;
    if (!runs_alone(world)) return EXIT_FAILURE;
    if (given != 1 || settings.save.empty()) {
        fmt::println(stderr, "a checkpoint and --save are required");
        return EXIT_FAILURE;
    }
    typename verbs_type_::geometry_t widened {};
    if constexpr (std::is_same_v<verbs_type_, dense_verbs_t>) widened.connectivity = settings.connectivity.value;
    else if (settings.connectivity.value) {
        fmt::println(stderr, "--connectivity widens dense checkpoints only");
        return EXIT_FAILURE;
    }
    std::optional<std::string> const admitted = checkpoints_admit_save(settings.save);
    if (!admitted) return EXIT_FAILURE;
    std::FILE* const steps = settings.save == "-" ? stderr : stdout;

    settings.session.devices = devices_from_environment();
    session_t session = open_session(settings.session, usearch_cpu_class_unknown_k);
    if (!session) return EXIT_FAILURE;
    handle<verbs_type_> index = checkpoint_open<verbs_type_>(session, positionals[0], settings.session.residency,
                                                             widened, steps);
    bool const refined = index &&
                         checkpoint_refine_passes(world, index, settings.max_refines.value,
                                                  {settings.expansion_refine.value, settings.memory_limit.value},
                                                  steps) &&
                         checkpoint_save(world, index, *admitted, steps);
    return refined ? EXIT_SUCCESS : EXIT_FAILURE;
}

/** Loads a checkpoint, cuts it into the empty receivers @p receiver_geometry opens from its
 *  metadata, saves each as a part and commits the manifest beside them. */
template <typename verbs_type_, typename receiver_geometry_type_>
int checkpoint_split(world_t& world, int argc, char const* const* argv, std::span<flag_t const> modality_flags,
                     receiver_geometry_type_&& receiver_geometry) {
    checkpoint_settings_t settings;
    std::vector<flag_t> const flags = with_refine_flags(
        settings,
        {
            {"--parts", "count", "required: receivers to split into, at least 2", &settings.parts},
            {"--save", "path", "required: the manifest to commit, its parts written beside it", &settings.save},
        },
        modality_flags);
    std::array<std::string_view, 1> positionals;
    std::size_t given = 0;
    std::string const synopsis = fmt::format("usearch {} split <address> --parts <count> --save <path> [flags]",
                                             verbs_type_::name_k);
    if (std::optional<int> const exit = parse_verb(world, argc, argv, synopsis, flags, positionals, given))
        return *exit;
    if (!runs_alone(world)) return EXIT_FAILURE;
    if (given != 1 || !settings.parts.value || settings.save.empty()) {
        fmt::println(stderr, "a checkpoint, --parts and --save are required");
        return EXIT_FAILURE;
    }
    if (settings.save == "-") {
        fmt::println(stderr, "--save names a manifest with its parts beside it, which one pipe cannot carry");
        return EXIT_FAILURE;
    }

    settings.session.devices = devices_from_environment();
    session_t session = open_session(settings.session, usearch_cpu_class_unknown_k);
    if (!session) return EXIT_FAILURE;
    handle<verbs_type_> index = checkpoint_open<verbs_type_>(session, positionals[0], settings.session.residency);
    if (!index) return EXIT_FAILURE;
    std::size_t const parts_count = settings.parts.value;
    typename verbs_type_::metadata_t const metadata = describe(index).metadata;
    std::vector<handle<verbs_type_>> receivers;
    std::vector<typename verbs_type_::index_t> receiver_indexes;
    for (std::size_t part = 0; part != parts_count; ++part) {
        receivers.push_back(open_index<verbs_type_>(session, receiver_geometry(metadata), 1));
        if (!receivers.back()) return EXIT_FAILURE;
        receiver_indexes.push_back(receivers.back().get());
    }

    checkpoint_clock_t::time_point const started = checkpoint_clock_t::now();
    usearch_outcome_t outcome {};
    usearch_refine_report_t report {};
    verbs_type_::split(index.get(), index.executor(), receiver_indexes.data(), parts_count, 0, &report, &outcome,
                       settings.expansion_refine.value, settings.memory_limit.value);
    if (report_failure(outcome, "splitting")) return EXIT_FAILURE;
    fmt::println("split into {} parts in {:.1f} s, {} changed, {} of {} reachable", parts_count,
                 checkpoint_seconds_since(started), report.changed, report.reachability.reachable,
                 report.reachability.audited);

    std::vector<safegraphs_part_t> parts(parts_count);
    for (std::size_t part = 0; part != parts_count; ++part) {
        std::string const part_path = checkpoint_part_path(settings.save, part, parts_count);
        std::size_t const nodes = describe(receivers[part]).size;
        if (!checkpoint_save(world, receivers[part], part_path)) return EXIT_FAILURE;
        parts[part] = {.bytes = checkpoint_file_bytes(part_path), .nodes = nodes};
    }
    if (!checkpoint_write_manifest(settings.save, parts)) {
        fmt::println(stderr, "cannot write {}", settings.save);
        return EXIT_FAILURE;
    }
    fmt::println("committed {} over {} parts", settings.save, parts_count);
    return EXIT_SUCCESS;
}

/** Views a checkpoint from storage and audits how many of its entries a search reaches. */
template <typename verbs_type_>
int checkpoint_health(world_t& world, int argc, char const* const* argv) {
    checkpoint_settings_t settings;
    settings.session.residency = usearch_residency_nvme_k;
    std::vector<flag_t> const flags = with_session_flags(settings.session, {});
    std::array<std::string_view, 1> positionals;
    std::size_t given = 0;
    std::string const synopsis = fmt::format("usearch {} health <address> [flags]", verbs_type_::name_k);
    if (std::optional<int> const exit = parse_verb(world, argc, argv, synopsis, flags, positionals, given))
        return *exit;
    if (!runs_alone(world)) return EXIT_FAILURE;
    if (given != 1) {
        fmt::println(stderr, "a checkpoint is required");
        return EXIT_FAILURE;
    }

    // A view reads the graph from storage, so the audit holds nothing that grows with the index.
    settings.session.devices = devices_from_environment();
    session_t session = open_session(settings.session, usearch_cpu_class_unknown_k);
    if (!session) return EXIT_FAILURE;
    handle<verbs_type_> index = checkpoint_open<verbs_type_>(session, positionals[0], settings.session.residency);
    if (!index) return EXIT_FAILURE;
    checkpoint_clock_t::time_point const started = checkpoint_clock_t::now();
    usearch_outcome_t outcome {};
    usearch_reachability_t reachability {};
    verbs_type_::health(index.get(), index.executor(), &reachability, &outcome);
    if (report_failure(outcome, "auditing")) return EXIT_FAILURE;
    fmt::println("{}: {} of {} reachable over {} sweeps in {:.1f} s", positionals[0], reachability.reachable,
                 reachability.audited, reachability.sweeps, checkpoint_seconds_since(started));
    return EXIT_SUCCESS;
}

/** Bytes of a <tt>.u128bin</tt> header: its rows, then its columns, each a little-endian @c u32. */
inline constexpr std::size_t key_rows_header_bytes_k = 2 * sizeof(std::uint32_t);

/** Reads @p path, a <tt>.u128bin</tt> of rows of little-endian 128-bit integers as <tt>--keys</tt> reads them,
 *  into @p keys, @p columns per row, and how many keys each row holds into @p counts: the
 *  little-endian @c u32 per row that may trail the keys, else every row full. False after printing
 *  why, or when it holds other than @p rows rows. */
inline bool read_key_rows(std::string_view path, std::size_t rows, std::vector<usearch_key_t>& keys,
                          std::size_t& columns, std::vector<std::size_t>& counts) {
    auto const [file, status] = mapped_file_t::read(path);
    bytes_view_t const bytes = file.view();
    std::array<std::uint32_t, 2> counters {};
    if (failed(status) || bytes.size() < key_rows_header_bytes_k) {
        fmt::println(stderr, "cannot read {}", path);
        return false;
    }
    std::memcpy(counters.data(), bytes.data(), key_rows_header_bytes_k);
    columns = counters[1];
    std::size_t const keys_bytes = key_rows_header_bytes_k +
                                   std::size_t {counters[0]} * columns * sizeof(usearch_key_t);
    if (counters[0] != rows || bytes.size() < keys_bytes) {
        fmt::println(stderr, "{} holds {} rows of {} keys, not one row for each of {} queries", path, counters[0],
                     counters[1], rows);
        return false;
    }
    keys.resize(rows * columns);
    byte_t const* const integers = bytes.data() + key_rows_header_bytes_k;
    // Little-endian integers into keys stored most significant byte first.
    for (std::size_t key = 0; key != keys.size(); ++key)
        for (std::size_t byte = 0; byte != sizeof(usearch_key_t); ++byte)
            keys[key].bytes[sizeof(usearch_key_t) - 1 - byte] = std::to_integer<usearch_byte_t>(
                integers[key * sizeof(usearch_key_t) + byte]);
    counts.assign(rows, columns);
    if (bytes.size() < keys_bytes + rows * sizeof(std::uint32_t)) return true;
    for (std::size_t row = 0; row != rows; ++row) {
        std::uint32_t count = 0;
        std::memcpy(&count, bytes.data() + keys_bytes + row * sizeof(std::uint32_t), sizeof(count));
        counts[row] = std::min<std::size_t>(count, columns);
    }
    return true;
}

/** Writes the matches of @p found to @p path as @ref read_key_rows reads them, @p columns per
 *  query, then each query's count, so a query that matched none hints none; a query that matched
 *  fewer repeats its last match, so readers ignoring the counts take the same keys. */
inline bool write_key_rows(std::string_view path, copied_matches_t const& found, std::size_t columns) {
    std::size_t const rows = found.counts.size();
    std::size_t const keys_bytes = key_rows_header_bytes_k + rows * columns * sizeof(usearch_key_t);
    auto [file, status] = mapped_file_t::write(path, keys_bytes + rows * sizeof(std::uint32_t));
    if (failed(status)) {
        fmt::println(stderr, "cannot write {}", path);
        return false;
    }
    byte_t* const bytes = file.span().data();
    std::array<std::uint32_t, 2> const counters {static_cast<std::uint32_t>(rows), static_cast<std::uint32_t>(columns)};
    std::memcpy(bytes, counters.data(), key_rows_header_bytes_k);
    std::vector<std::size_t> const offsets = match_offsets(found);
    for (std::size_t row = 0; row != rows; ++row) {
        std::uint32_t const count = static_cast<std::uint32_t>(std::min(found.counts[row], columns));
        std::memcpy(bytes + keys_bytes + row * sizeof(std::uint32_t), &count, sizeof(count));
        if (!count) continue;
        for (std::size_t column = 0; column != columns; ++column) {
            uuid128_t const& key = found.keys[offsets[row] + min_of<std::size_t>(column, count - 1)];
            byte_t* const integer = bytes + key_rows_header_bytes_k + (row * columns + column) * sizeof(usearch_key_t);
            for (std::size_t byte = 0; byte != sizeof(usearch_key_t); ++byte)
                integer[byte] = key.bytes[sizeof(usearch_key_t) - 1 - byte];
        }
    }
    if (failed(file.sync())) {
        fmt::println(stderr, "cannot write {}", path);
        return false;
    }
    return true;
}

/** Opens a checkpoint in the residency asked for, named by the positional or by @p index_address
 *  when a modality's flag filled it, searches the queries @p open_queries reads from the files its
 *  flags name, each walk seeded by its row of <tt>--hints</tt>, and prints every query's keys and
 *  distances, then the throughput; <tt>--matches</tt> keeps the keys as <tt>--hints</tt> reads them, and
 *  <tt>--output</tt> keeps the matches as Arrow in place of the printed ones; under <tt>--output -</tt>, standard
 *  output carries only the Arrow bytes, and every step prints to standard error instead. */
template <typename verbs_type_, typename open_queries_type_>
int checkpoint_search(world_t& world, int argc, char const* const* argv, std::string_view synopsis,
                      std::span<flag_t const> modality_flags, open_queries_type_&& open_queries,
                      std::string_view const& index_address = {}) {
    checkpoint_settings_t settings;
    std::string_view hints_path, matches_path;
    std::vector<flag_t> const flags = with_session_flags(
        settings.session,
        {
            {"--wanted", "count", "defaults to 10: matches per query", &settings.wanted},
            {"--expansion-search", "count", "defaults to the library's derivation: candidates while searching",
             &settings.expansion_search},
            {"--traversal", "name", "defaults to walk: walk the graph, or scan every entry for the exact matches",
             &settings.traversal},
            batch_flag(settings),
            {"--hints", "path",
             "optional: a .u128bin of u32 rows, u32 columns, then little-endian 128-bit keys as --keys reads them, " //
             "one row per query its walk starts from, then optionally a u32 count per row",                          //
             &hints_path},
            {"--matches", "path", "optional: writes every query's top keys as a .u128bin --hints reads", &matches_path},
            {"--query-keys", "path", "defaults to row numbers: a .u128bin of one key per query, as --output names them",
             &settings.query_keys},
            {"--output", "path",
             "optional: writes the matches as Arrow in place of printing them, a stream for - or an .arrows path, " //
             "else a file",                                                                                         //
             &settings.output},
        },
        modality_flags);
    std::array<std::string_view, 1> positionals;
    std::size_t given = 0;
    if (std::optional<int> const exit = parse_verb(world, argc, argv, synopsis, flags, positionals, given))
        return *exit;
    if (!runs_alone(world)) return EXIT_FAILURE;
    if (given + !index_address.empty() != 1) {
        fmt::println(stderr, "a checkpoint is required, once");
        return EXIT_FAILURE;
    }
    if (!hints_path.empty() &&
        (std::is_same_v<verbs_type_, sequence_verbs_t> || std::is_same_v<verbs_type_, geospatial_verbs_t>)) {
        fmt::println(stderr, "--hints seeds dense and shapes walks only");
        return EXIT_FAILURE;
    }
    std::optional<checkpoint_queries<verbs_type_>> const queries = open_queries();
    if (!queries) return EXIT_FAILURE;
    if (!queries->count) {
        fmt::println(stderr, "the queries hold none");
        return EXIT_FAILURE;
    }
    std::vector<usearch_key_t> hint_keys;
    std::vector<std::size_t> hint_counts;
    std::size_t hint_columns = 0;
    if (!hints_path.empty() && !read_key_rows(hints_path, queries->count, hint_keys, hint_columns, hint_counts))
        return EXIT_FAILURE;
    usearch_matches_view_t const hints {hint_keys.data(), nullptr, hint_counts.data(), hint_columns};
    std::vector<usearch_key_t> query_keys = queries->keys;
    std::vector<std::size_t> query_key_counts;
    std::size_t query_key_columns = 1;
    if (query_keys.empty() && settings.query_keys.empty())
        for (std::size_t query = 0; query != queries->count; ++query)
            query_keys.push_back(std::bit_cast<usearch_key_t>(uuid_of(query)));
    else if (query_keys.empty() &&
             !read_key_rows(settings.query_keys, queries->count, query_keys, query_key_columns, query_key_counts))
        return EXIT_FAILURE;
    if (query_key_columns != 1) {
        fmt::println(stderr, "{} holds {} keys per query, not one", settings.query_keys, query_key_columns);
        return EXIT_FAILURE;
    }
    std::FILE* const steps = settings.output == "-" ? stderr : stdout;

    settings.session.devices = devices_from_environment();
    session_t session = open_session(settings.session, usearch_cpu_class_unknown_k);
    if (!session) return EXIT_FAILURE;
    handle<verbs_type_> index = checkpoint_open<verbs_type_>(session, given ? positionals[0] : index_address,
                                                             settings.session.residency, {}, steps);
    if (!index) return EXIT_FAILURE;
    checkpoint_clock_t::time_point const started = checkpoint_clock_t::now();
    copied_matches_t found;
    usearch_traversal_stats_t totals {};
    usearch_outcome_t const outcome = search_queries(
        index, queries->queries, queries->count, settings.batch.value, settings.wanted.value,
        settings.expansion_search.value, settings.traversal, totals, found, hints_path.empty() ? nullptr : &hints);
    if (report_failure(outcome, "searching")) return EXIT_FAILURE;
    double const seconds = checkpoint_seconds_since(started);
    if (!matches_path.empty() && !write_key_rows(matches_path, found, settings.wanted.value)) return EXIT_FAILURE;
    if (!settings.output.empty())
        if (std::string const failure = arrow_matches_write(settings.output, query_keys, found); !failure.empty()) {
            fmt::println(stderr, "{}", failure);
            return EXIT_FAILURE;
        }

    std::vector<std::size_t> const offsets = match_offsets(found);
    std::string line;
    for (std::size_t query = 0; settings.output.empty() && query != queries->count; ++query) {
        line = fmt::format("query {}:", query);
        for (std::size_t match = offsets[query]; match != offsets[query + 1]; ++match)
            fmt::format_to(std::back_inserter(line), " {} {}", checkpoint_key_text(found.keys[match]),
                           found.scores[match]);
        fmt::println("{}", line);
    }
    std::string const missing = hints_path.empty() ? std::string {}
                                                   : fmt::format(", {} hints missing", totals.missing_hints);
    fmt::println(steps,
                 "searched {} queries for the top {} in {:.3f} s: {:.0f} queries per second, {:.1f} distances each{}",
                 queries->count, settings.wanted.value, seconds, static_cast<double>(queries->count) / seconds,
                 static_cast<double>(totals.computed_distances) / static_cast<double>(queries->count), missing);
    return EXIT_SUCCESS;
}

/** Runs the verb @p argv names after the modality @p verbs_type_ names, or answers none when
 *  @p argv names another modality or no verb of it. */
template <typename verbs_type_>
std::optional<int> run_modality_verb(world_t& world, int argc, char const* const* argv) {
    if (argc < 3 || argv[1] != verbs_type_::name_k) return std::nullopt;
    std::string_view const verb = argv[2];
    // Each verb parses the arguments after its name, reading the name where a program name stands.
    argc -= 2, argv += 2;
    if (verb == "insert") return modality_insert(verbs_type_ {}, world, argc, argv);
    if (verb == "absorb") return checkpoint_absorb<verbs_type_>(world, argc, argv);
    if (verb == "refine") return checkpoint_refine<verbs_type_>(world, argc, argv);
    if (verb == "split") return modality_split(verbs_type_ {}, world, argc, argv);
    if (verb == "health") return checkpoint_health<verbs_type_>(world, argc, argv);
    if (verb == "search") return modality_search(verbs_type_ {}, world, argc, argv);
    if constexpr (std::is_same_v<verbs_type_, dense_verbs_t>)
        if (verb == "get") return dense_get(world, argc, argv);
    // A modality's metadata is the one any checkpoint prints, its modality read from its header.
    if (verb == "metadata") return checkpoints_metadata(world, argc, argv);
    return std::nullopt;
}

#pragma endregion Checkpoint Verbs

} // namespace unum::usearch::tools
