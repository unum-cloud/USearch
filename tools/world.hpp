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

#include <csignal> // `std::sig_atomic_t` - the stop a signal asks for
#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint64_t`
#include <cstdio>  // `std::FILE`, `stdout`, `stderr` - where a step prints
#include <cstdlib> // `EXIT_SUCCESS`, `EXIT_FAILURE`
#include <cstring> // `std::memcpy` - a `.u128bin` header

#include <sys/stat.h> // `stat`, `S_ISREG` - a saved part's bytes

#include <algorithm>        // `std::min`
#include <bit>              // `std::bit_cast` - entry numbers into keys
#include <array>            // `std::array` - the flag tables
#include <chrono>           // `std::chrono::steady_clock`
#include <initializer_list> // `std::initializer_list` - a verb's own flags
#include <limits>           // `std::numeric_limits` - the digits of a rank
#include <optional>         // `std::optional`
#include <span>             // `std::span`
#include <string_view>      // `std::string_view`
#include <type_traits>      // `std::is_same_v` - the dense-only widening
#include <utility>          // `std::move`

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

/** What leads a rank's lines while several run, like "rank 3: ", formatted in place. */
struct rank_label_t {
    std::array<char, sizeof("rank : ") + std::numeric_limits<std::size_t>::digits10 + 1> characters {};
    std::size_t length = 0;
};

/** The text @p label prints as. */
inline std::string_view format_as(rank_label_t const& label) noexcept {
    return {label.characters.data(), label.length};
}

/** The label of this rank's lines, empty in a world of one. */
inline rank_label_t rank_label(world_t const& world) noexcept {
    rank_label_t label;
    if (world.size > 1)
        label.length = fmt::format_to_n(label.characters.data(), label.characters.size(), "rank {}: ", world.rank).size;
    return label;
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

/** Set by the first SIGINT or SIGTERM: every long verb stops at its next barrier, saves what stands
 *  where it saves, and the process exits with @ref exit_stopped_k. */
inline volatile std::sig_atomic_t stop_requested = 0;

/** The exit status of a verb a signal stopped, as shells report a command SIGINT ended. */
inline constexpr int exit_stopped_k = 130;

/** The phase a verb is in: what the shared progress handler prints, and what a stop names. */
struct verb_phase_t {
    std::string_view name;
    std::string_view unit;
    std::FILE* stream = stdout;
    rank_label_t label {};
    checkpoint_clock_t::time_point printed {};
};

/** The handler every long verb passes: prints the phase's advance at most once a second, and
 *  answers cancel once a signal asked the process to stop. */
inline usearch_continuation_t verb_progress(std::size_t completed, std::size_t total, void* state) noexcept {
    verb_phase_t& phase = *static_cast<verb_phase_t*>(state);
    checkpoint_clock_t::time_point const now = checkpoint_clock_t::now();
    if (now - phase.printed >= std::chrono::seconds(1)) {
        phase.printed = now;
        fmt::println(phase.stream, "{}{}: {} of {} {}", phase.label, phase.name, completed, total, phase.unit);
    }
    return stop_requested ? usearch_cancel_k : usearch_continue_k;
}

/** The C handler reporting into @p phase. */
inline usearch_progress_t verb_progress_of(verb_phase_t& phase) noexcept { return {verb_progress, &phase, 0}; }

/** Prints in which phase a signal stopped the verb. */
inline void verb_stopped(verb_phase_t const& phase) noexcept {
    fmt::println(phase.stream, "{}stopped in {}", phase.label, phase.name);
}

/** The word a refinement round's verdict prints as. */
inline std::string_view refine_verdict_name(usearch_refine_verdict_t verdict) noexcept {
    switch (verdict) {
    case usearch_refine_improved_k: return "improved";
    case usearch_refine_converged_k: return "converged";
    case usearch_refine_stalled_k: return "stalled";
    case usearch_refine_stopped_k: return "stopped";
    }
    return "unknown";
}

/** The path of part @p part, counted from zero, of the @p parts the manifest at @p manifest_path
 *  commits; empty when it cannot be formatted. */
inline sz::string_t checkpoint_part_path(std::string_view manifest_path, std::size_t part, std::size_t parts) noexcept {
    growing_buffer<char> named {global_allocator};
    return formatted("{}", safegraphs_part_path(named, safegraphs_stem(manifest_path), part, parts).value);
}

/** Appends the parts the checkpoint at @p address spans to @p parts: each part a manifest names,
 *  beside it, or the address itself for a monolith, a transport ID or standard input; answers
 *  false after printing why. */
inline bool checkpoint_parts_of(std::string_view address, st::vector<sz::string_t>& parts) noexcept {
    if (address != "-" && !names_transport_id(address)) {
        auto const [mapping, map_status] = mapped_file_t::read(address);
        if (failed(map_status)) {
            fmt::println(stderr, "cannot read {}", address);
            return false;
        }
        if (!safegraphs_is_monolith(mapping.view())) {
            std::string_view const directory = address.substr(0, address.size() - safegraphs_basename(address).size());
            auto per_part = [&](safegraphs_part_t part) noexcept {
                if (st::failed(parts.push_back(formatted("{}{}", directory, part.path))))
                    exit_out_of_memory("the parts");
            };
            auto const [summary, summary_status] = safegraphs_read_manifest(address, std::allocator<byte_t> {},
                                                                            per_part);
            if (failed(summary_status)) fmt::println(stderr, "{} holds no SafeGraphs monolith or manifest", address);
            return succeeded(summary_status);
        }
    }
    if (st::failed(parts.push_back(formatted("{}", address)))) exit_out_of_memory("the parts");
    return true;
}

/** Bytes the regular file at @p path holds, or zero for a device, an SPDK address or a failure. */
inline std::uint64_t checkpoint_file_bytes(sz::string_t const& path) noexcept {
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

/** Appends the text @p key prints as to @p text: the entry number it was minted from, or its 32 hex
 *  digits. */
inline void checkpoint_append_key(sz::string_t& text, uuid128_t const& key) noexcept {
    if (std::optional<u64_t> const number = identity_of(key)) return append_formatted(text, "{}", *number);
    for (std::byte const byte : key.bytes) append_formatted(text, "{:02x}", std::to_integer<unsigned>(byte));
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
inline st::vector<flag_t> with_session_flags(settings_t& session, std::initializer_list<flag_t> verb_flags,
                                             std::span<flag_t const> modality_flags = {}) noexcept {
    std::array<flag_t, 4> const shared = session_flags(session);
    return joined_flags({verb_flags, shared, modality_flags});
}

/** @ref with_session_flags for a verb that refines, with the two flags every such verb takes. */
inline st::vector<flag_t> with_refine_flags(checkpoint_settings_t& settings, std::initializer_list<flag_t> verb_flags,
                                            std::span<flag_t const> modality_flags = {}) noexcept {
    std::array<flag_t, 2> const refine {{
        {"--expansion-refine", "count", "defaults to the library's derivation: candidates of walks beside the rounds",
         &settings.expansion_refine},
        {"--memory-limit", "bytes",
         "defaults to the library's derivation: bytes an NVMe refine holds for one part, like 40G",
         &settings.memory_limit},
    }};
    std::array<flag_t, 4> const shared = session_flags(settings.session);
    return joined_flags({verb_flags, refine, shared, modality_flags});
}

/** The flag every verb calling the library in batches takes. */
inline flag_t batch_flag(checkpoint_settings_t& settings) noexcept {
    return {"--batch", "count", "defaults to 65536: entries per insertion, or queries per search", &settings.batch};
}

/** The flags every insert takes, then @p modality_flags. */
inline st::vector<flag_t> insert_flags(checkpoint_settings_t& settings,
                                       std::span<flag_t const> modality_flags) noexcept {
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
            {"--max-refines", "count", "defaults to 0: refinement rounds after ingestion, until one is not improved",
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
    st::vector<usearch_key_t> keys {};
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
    st::vector<byte_t> bytes;
    if (!read_standard_input(bytes)) {
        fmt::println(stderr, "cannot read a checkpoint off standard input");
        return {};
    }
    usearch_outcome_t outcome {};
    typename verbs_type_::index_t loaded = nullptr;
    st::vector<std::int32_t> placement;
    verbs_type_::load_ram_buffer(&loaded, session.executor, bytes_view(bytes), session.ram_options(0, placement),
                                 geometry, &outcome);
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
    st::vector<byte_t> copied;
    void const* bytes = monolith.data;
    if (index.residency == usearch_residency_vram_k) {
        if (st::failed(copied.resize(st::assume_uninitialized, monolith.length))) exit_out_of_memory("the checkpoint");
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
bool checkpoint_save(world_t const& world, handle<verbs_type_> const& index, sz::string_t const& address,
                     std::FILE* stream = stdout) {
    if (address == "-") return checkpoint_save_piped(index, stream);
    checkpoint_clock_t::time_point const started = checkpoint_clock_t::now();
    usearch_outcome_t outcome {};
    verbs_type_::save(index.get(), index.executor(), bytes_view(address), &outcome);
    if (report_failure(outcome, "{}saving {}", rank_label(world), address)) return false;
    double const seconds = checkpoint_seconds_since(started);
    if (std::uint64_t const bytes = checkpoint_file_bytes(address))
        fmt::println(stream, "{}saved {}: {} bytes in {:.1f} s", rank_label(world), address, bytes, seconds);
    else fmt::println(stream, "{}saved {} in {:.1f} s", rank_label(world), address, seconds);
    return true;
}

/** Runs up to @p rounds refinement rounds over @p index under @p refine, until one concludes other
 *  than improved or a signal stops one, printing each under this rank to the stream of @p phase and
 *  naming the phase a stop came in; false after printing a failure. */
template <typename verbs_type_>
bool checkpoint_refine_rounds(world_t const& world, handle<verbs_type_> const& index, std::size_t rounds,
                              usearch_refine_options_t refine, verb_phase_t& phase) {
    // Rounds run through one plan, so they share their root and a stop closes the round in hand; a
    // loaded NVMe handle rounds part by part within each call and takes none.
    typename verbs_type_::refine_plan_t plan = nullptr;
    bool const planned = index.residency != usearch_residency_nvme_k;
    auto finish_plan = [&](usearch_refine_report_t* report) noexcept {
        usearch_outcome_t outcome {};
        if (plan)
            verbs_type_::refine_plan_finish(index.get(), index.executor(), std::exchange(plan, nullptr), report,
                                            &outcome);
        return !report_failure(outcome, "{}finishing the refine plan", rank_label(world));
    };
    for (std::size_t round = 0; round != rounds && !stop_requested; ++round) {
        phase.name = "refine", phase.unit = "entries";
        checkpoint_clock_t::time_point const started = checkpoint_clock_t::now();
        usearch_outcome_t outcome {};
        usearch_refine_report_t report {};
        usearch_traversal_stats_t stats {};
        usearch_phase_t stopped {};
        verbs_type_::refine(index.get(), index.executor(), &report, &outcome, refine.expansion_factor,
                            refine.memory_limit, &stats, verb_progress_of(phase), planned ? &plan : nullptr, &stopped);
        if (outcome.status != usearch_status_cancelled_k && report_failure(outcome, "{}refining", rank_label(world)))
            return finish_plan(nullptr), false;
        if (outcome.status == usearch_status_cancelled_k && plan) {
            if (stopped.name) phase.name = stopped.name;
            if (!finish_plan(&report)) return false;
        }
        fmt::println(phase.stream,
                     "{}refined round {}: {}; {} neighborhoods, widest {}; "               //
                     "{} links added, {} dropped, {} of {} entries changed; "              //
                     "{} of {} in one component in {:.1f} s, {:.1f} distances per entry; " //
                     "{} unreached and {} stranded islands, opened with {} and {}",
                     rank_label(world), round + 1, refine_verdict_name(report.verdict), report.neighborhoods,
                     report.widest_neighborhood, report.added_links, report.dropped_links, report.changed,
                     report.reselected, report.reachability.reachable, report.reachability.audited,
                     checkpoint_seconds_since(started),
                     double(stats.computed_distances) / double(max_of<std::size_t>(report.reachability.audited, 1)),
                     report.reachability.unreached_islands, report.reachability.stranded_islands,
                     report.opening.unreached_islands, report.opening.stranded_islands);
        if (report.verdict != usearch_refine_improved_k) break;
    }
    if (!finish_plan(nullptr)) return false;
    if (stop_requested) verb_stopped(phase);
    return true;
}

/** Inserts a slice of the entries @p open_entries reads from the one positional into a new
 *  one-shard index in RAM or VRAM, keyed by their entry numbers, refines it and saves it. */
template <typename verbs_type_, typename open_entries_type_>
int checkpoint_insert(world_t& world, int argc, char const* const* argv, std::string_view synopsis,
                      std::span<flag_t const> modality_flags, open_entries_type_&& open_entries) {
    checkpoint_settings_t settings;
    settings.session.residency = usearch_residency_unknown_k;
    st::vector<flag_t> const flags = insert_flags(settings, modality_flags);
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
    std::optional<sz::string_t> const admitted = checkpoints_admit_save(settings.save);
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
    st::vector<usearch_key_t> keys;
    if (st::failed(keys.resize(batch))) exit_out_of_memory("the keys");
    verb_phase_t phase {.name = "insert", .unit = "entries", .stream = steps};
    std::size_t inserted = 0;
    while (inserted < count) {
        std::size_t const sliced = min_of(batch, count - inserted);
        for (std::size_t position = 0; position != sliced; ++position)
            keys[position] = std::bit_cast<usearch_key_t>(uuid_of(first + inserted + position));
        usearch_outcome_t outcome {};
        verbs_type_::insert(index.get(), index.executor(), sliced, keys_view(std::span {keys}.first(sliced)),
                            verbs_type_::slice(entries->values, first + inserted), settings.expansion_add.value,
                            nullptr, &outcome);
        if (report_failure(outcome, "inserting")) return EXIT_FAILURE;
        inserted += sliced;
        if (verb_progress(inserted, count, &phase) == usearch_cancel_k) break;
    }
    fmt::println(steps, "inserted entries {} to {} in {} in {:.1f} s", first, first + inserted,
                 usearch_residency_name(index.residency), checkpoint_seconds_since(started));
    bool const built = checkpoint_refine_rounds(world, index, settings.max_refines.value,
                                                {settings.expansion_refine.value, settings.memory_limit.value},
                                                phase) &&
                       checkpoint_save(world, index, *admitted, steps);
    return built ? EXIT_SUCCESS : EXIT_FAILURE;
}

/** Loads a base and absorbs every donor into it in one call, then saves the merged index; a
 *  manifest stands for its parts, so the first part named is the base. */
template <typename verbs_type_>
int checkpoint_absorb(world_t& world, int argc, char const* const* argv) {
    checkpoint_settings_t settings;
    st::vector<flag_t> const flags = with_refine_flags(
        settings, {{"--save", "address", "required: where the merged index is saved", &settings.save}});
    // Every argument is at most one positional, so the arguments bound the donors.
    st::vector<std::string_view> positionals;
    if (st::failed(positionals.resize(static_cast<std::size_t>(argc)))) exit_out_of_memory("the arguments");
    std::size_t given = 0;
    sz::string_t const synopsis = formatted("usearch {} absorb <base|manifest> [donors]... --save <address> [flags]",
                                            verbs_type_::name_k);
    if (std::optional<int> const exit = parse_verb(world, argc, argv, synopsis, flags, positionals, given))
        return *exit;
    if (!runs_alone(world)) return EXIT_FAILURE;
    if (!given || settings.save.empty()) {
        fmt::println(stderr, "a base and its donors, or a manifest of several parts, and --save are required");
        return EXIT_FAILURE;
    }
    st::vector<sz::string_t> parts;
    for (std::string_view const address : std::span {positionals}.first(given))
        if (!checkpoint_parts_of(address, parts)) return EXIT_FAILURE;
    if (parts.size() < 2) {
        fmt::println(stderr, "{} names one part, and absorbing needs a donor", positionals[0]);
        return EXIT_FAILURE;
    }
    std::optional<sz::string_t> const admitted = checkpoints_admit_save(settings.save);
    if (!admitted) return EXIT_FAILURE;
    std::FILE* const steps = settings.save == "-" ? stderr : stdout;

    settings.session.devices = devices_from_environment();
    session_t session = open_session(settings.session, usearch_cpu_class_unknown_k);
    if (!session) return EXIT_FAILURE;
    handle<verbs_type_> base = checkpoint_open<verbs_type_>(session, parts[0], settings.session.residency, {}, steps);
    if (!base) return EXIT_FAILURE;
    st::vector<handle<verbs_type_>> donors;
    st::vector<typename verbs_type_::index_t> donor_indexes;
    std::size_t donor_entries = 0;
    for (sz::string_t const& part : std::span {parts}.subspan(1)) {
        handle<verbs_type_> donor = checkpoint_open<verbs_type_>(session, part, settings.session.residency, {}, steps);
        if (!donor) return EXIT_FAILURE;
        donor_entries += describe(donor).size;
        if (st::failed(donor_indexes.push_back(donor.get())) || st::failed(donors.push_back(std::move(donor))))
            exit_out_of_memory("the donors");
    }
    checkpoint_clock_t::time_point const started = checkpoint_clock_t::now();
    usearch_outcome_t outcome {};
    usearch_refine_report_t report {};
    verb_phase_t phase {.name = "absorb", .unit = "entries", .stream = steps};
    // A plan names the phase a stop came in; a loaded NVMe base takes none.
    typename verbs_type_::absorb_plan_t plan = nullptr;
    usearch_phase_t stopped_in {};
    verbs_type_::absorb(base.get(), base.executor(), donor_indexes, &outcome, &report, settings.expansion_refine.value,
                        settings.memory_limit.value, verb_progress_of(phase),
                        base.residency == usearch_residency_nvme_k ? nullptr : &plan, &stopped_in);
    bool const stopped = outcome.status == usearch_status_cancelled_k;
    if (plan) {
        if (stopped && stopped_in.name) phase.name = stopped_in.name;
        usearch_outcome_t finished {};
        verbs_type_::absorb_plan_finish(base.get(), base.executor(), plan, &report, &finished);
        if (report_failure(finished, "finishing the absorb plan")) return EXIT_FAILURE;
    }
    // A consumed donor's slot is nulled, so its handle lets go; a refused one its handle frees.
    for (std::size_t donor = 0; donor != donors.size(); ++donor)
        if (!donor_indexes[donor]) donors[donor].index = nullptr;
    if (!stopped && report_failure(outcome, "absorbing {} donors", donors.size())) return EXIT_FAILURE;
    fmt::println(steps, "absorbed {} donors: {} entries in {:.1f} s, {} changed, {} of {} reachable", donors.size(),
                 donor_entries, checkpoint_seconds_since(started), report.changed, report.reachability.reachable,
                 report.reachability.audited);
    if (stopped) verb_stopped(phase);
    return checkpoint_save(world, base, *admitted, steps) ? EXIT_SUCCESS : EXIT_FAILURE;
}

/** Opens the manifest at @p address on @p session's NVMe executor as a writable index, which a
 *  refine rewrites part by part; only dense checkpoints have one, so the rest report what C reports
 *  for a verb it lacks. Prints the step to @p stream, or what failed and an empty handle. */
template <typename verbs_type_>
handle<verbs_type_> checkpoint_load_nvme(session_t& session, std::string_view address, std::FILE* stream) {
    checkpoint_clock_t::time_point const started = checkpoint_clock_t::now();
    usearch_outcome_t outcome {usearch_status_unsupported_k, "Only dense checkpoints load writable from NVMe"};
    typename verbs_type_::index_t loaded = nullptr;
    if constexpr (std::is_same_v<verbs_type_, dense_verbs_t>)
        verbs_type_::load_nvme(&loaded, session.storage, bytes_view(address), session.nvme_options(),
                               session.nvme_ram_options(), &outcome);
    if (report_failure(outcome, "Loading {}", address)) return {};
    fmt::println(stream, "loaded {} writable in {:.1f} s", address, checkpoint_seconds_since(started));
    return {loaded, session, usearch_residency_nvme_k, {}};
}

/** Loads a checkpoint, a dense one widened to a larger connectivity when asked, re-selects its
 *  neighbors pass after pass, and saves it; with @c --residency=nvme it refines the manifest in
 *  place, which commits every pass itself. */
template <typename verbs_type_>
int checkpoint_refine(world_t& world, int argc, char const* const* argv) {
    checkpoint_settings_t settings;
    settings.max_refines = {.value = 1};
    st::vector<flag_t> const flags = with_refine_flags(
        settings,
        {
            {"--save", "address", "required unless --residency=nvme refines in place: where the refined index is saved",
             &settings.save},
            {"--connectivity", "count", "defaults to the checkpoint's: a wider graph degree to refine a dense one into",
             &settings.connectivity},
            {"--max-refines", "count", "defaults to 1: refinement rounds, until one is not improved",
             &settings.max_refines},
        });
    std::array<std::string_view, 1> positionals;
    std::size_t given = 0;
    sz::string_t const synopsis = formatted("usearch {} refine <address> [--save <address>] [flags]",
                                            verbs_type_::name_k);
    if (std::optional<int> const exit = parse_verb(world, argc, argv, synopsis, flags, positionals, given))
        return *exit;
    if (!runs_alone(world)) return EXIT_FAILURE;
    bool const in_place = settings.session.residency == usearch_residency_nvme_k;
    if (given != 1 || settings.save.empty() != in_place) {
        fmt::println(stderr, "a checkpoint is required, and --save unless --residency=nvme refines it in place");
        return EXIT_FAILURE;
    }
    typename verbs_type_::geometry_t widened {};
    if constexpr (std::is_same_v<verbs_type_, dense_verbs_t>) widened.connectivity = settings.connectivity.value;
    if (settings.connectivity.value && (in_place || !std::is_same_v<verbs_type_, dense_verbs_t>)) {
        fmt::println(stderr, "--connectivity widens dense checkpoints loaded into RAM or VRAM only");
        return EXIT_FAILURE;
    }
    std::optional<sz::string_t> const admitted = in_place ? std::optional<sz::string_t> {}
                                                          : checkpoints_admit_save(settings.save);
    if (!in_place && !admitted) return EXIT_FAILURE;
    std::FILE* const steps = settings.save == "-" ? stderr : stdout;

    settings.session.devices = devices_from_environment();
    session_t session = open_session(settings.session, usearch_cpu_class_unknown_k);
    if (!session) return EXIT_FAILURE;
    handle<verbs_type_> index = in_place ? checkpoint_load_nvme<verbs_type_>(session, positionals[0], steps)
                                         : checkpoint_open<verbs_type_>(session, positionals[0],
                                                                        settings.session.residency, widened, steps);
    verb_phase_t phase {.name = "load", .unit = "entries", .stream = steps};
    bool const refined = index &&
                         checkpoint_refine_rounds(world, index, settings.max_refines.value,
                                                  {settings.expansion_refine.value, settings.memory_limit.value},
                                                  phase) &&
                         (in_place || checkpoint_save(world, index, *admitted, steps));
    return refined ? EXIT_SUCCESS : EXIT_FAILURE;
}

/** Loads a checkpoint, cuts it into the empty receivers @p receiver_geometry opens from its
 *  metadata, saves each as a part and commits the manifest beside them. */
template <typename verbs_type_, typename receiver_geometry_type_>
int checkpoint_split(world_t& world, int argc, char const* const* argv, std::span<flag_t const> modality_flags,
                     receiver_geometry_type_&& receiver_geometry) {
    checkpoint_settings_t settings;
    st::vector<flag_t> const flags = with_refine_flags(
        settings,
        {
            {"--parts", "count", "required: receivers to split into, at least 2", &settings.parts},
            {"--save", "path", "required: the manifest to commit, its parts written beside it", &settings.save},
        },
        modality_flags);
    std::array<std::string_view, 1> positionals;
    std::size_t given = 0;
    sz::string_t const synopsis = formatted("usearch {} split <address> --parts <count> --save <path> [flags]",
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
    st::vector<handle<verbs_type_>> receivers;
    st::vector<typename verbs_type_::index_t> receiver_indexes;
    for (std::size_t part = 0; part != parts_count; ++part) {
        handle<verbs_type_> receiver = open_index<verbs_type_>(session, receiver_geometry(metadata), 1);
        if (!receiver) return EXIT_FAILURE;
        if (st::failed(receiver_indexes.push_back(receiver.get())) ||
            st::failed(receivers.push_back(std::move(receiver))))
            exit_out_of_memory("the receivers");
    }

    checkpoint_clock_t::time_point const started = checkpoint_clock_t::now();
    usearch_outcome_t outcome {};
    usearch_refine_report_t report {};
    verb_phase_t phase {.name = "split", .unit = "entries"};
    // A plan names the phase a stop came in; a loaded NVMe index takes none.
    typename verbs_type_::split_plan_t plan = nullptr;
    usearch_phase_t stopped_in {};
    verbs_type_::split(index.get(), index.executor(), receiver_indexes.data(), parts_count, 0, &report, &outcome,
                       settings.expansion_refine.value, settings.memory_limit.value, verb_progress_of(phase),
                       index.residency == usearch_residency_nvme_k ? nullptr : &plan, &stopped_in);
    bool const stopped = outcome.status == usearch_status_cancelled_k;
    if (plan) {
        if (stopped && stopped_in.name) phase.name = stopped_in.name;
        usearch_outcome_t finished {};
        verbs_type_::split_plan_finish(index.get(), index.executor(), plan, &report, &finished);
        if (report_failure(finished, "finishing the split plan")) return EXIT_FAILURE;
    }
    if (!stopped && report_failure(outcome, "splitting")) return EXIT_FAILURE;
    fmt::println("split into {} parts in {:.1f} s, {} changed, {} of {} reachable", parts_count,
                 checkpoint_seconds_since(started), report.changed, report.reachability.reachable,
                 report.reachability.audited);
    if (stopped) verb_stopped(phase);

    st::vector<safegraphs_part_t> parts;
    if (st::failed(parts.resize(parts_count))) exit_out_of_memory("the parts");
    for (std::size_t part = 0; part != parts_count; ++part) {
        sz::string_t const part_path = checkpoint_part_path(settings.save, part, parts_count);
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
    st::vector<flag_t> const flags = with_session_flags(settings.session, {});
    std::array<std::string_view, 1> positionals;
    std::size_t given = 0;
    sz::string_t const synopsis = formatted("usearch {} health <address> [flags]", verbs_type_::name_k);
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
    fmt::println("{}: {} of {} in one component, {} unreached islands, {} stranded islands, " //
                 "depth {} from the root in {:.1f} s; {} unlinked, {} underlinked, {} beyond the most populated depth",
                 positionals[0], reachability.reachable, reachability.audited, reachability.unreached_islands,
                 reachability.stranded_islands, reachability.depth_from_root, checkpoint_seconds_since(started),
                 reachability.unlinked, reachability.underlinked, reachability.beyond_most_populated_depth);
    return EXIT_SUCCESS;
}

/** Bytes of a <tt>.u128bin</tt> header: its rows, then its columns, each a little-endian @c u32. */
inline constexpr std::size_t key_rows_header_bytes_k = 2 * sizeof(std::uint32_t);

/** Reads @p path, a <tt>.u128bin</tt> of rows of little-endian 128-bit integers as <tt>--keys</tt> reads them,
 *  into @p keys, @p columns per row, and how many keys each row holds into @p counts: the
 *  little-endian @c u32 per row that may trail the keys, else every row full. False after printing
 *  why, or when it holds other than @p rows rows. */
inline bool read_key_rows(std::string_view path, std::size_t rows, st::vector<usearch_key_t>& keys,
                          std::size_t& columns, st::vector<std::size_t>& counts) noexcept {
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
    counts.clear();
    if (st::failed(keys.resize(rows * columns)) || st::failed(counts.resize(rows, columns))) exit_out_of_memory(path);
    byte_t const* const integers = bytes.data() + key_rows_header_bytes_k;
    // Little-endian integers into keys stored most significant byte first.
    for (std::size_t key = 0; key != keys.size(); ++key)
        for (std::size_t byte = 0; byte != sizeof(usearch_key_t); ++byte)
            keys[key].bytes[sizeof(usearch_key_t) - 1 - byte] = std::to_integer<usearch_byte_t>(
                integers[key * sizeof(usearch_key_t) + byte]);
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
    st::vector<std::size_t> const offsets = match_offsets(found);
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
    st::vector<flag_t> const flags = with_session_flags(
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
    st::vector<usearch_key_t> hint_keys;
    st::vector<std::size_t> hint_counts;
    std::size_t hint_columns = 0;
    if (!hints_path.empty() && !read_key_rows(hints_path, queries->count, hint_keys, hint_columns, hint_counts))
        return EXIT_FAILURE;
    usearch_matches_view_t const hints {hint_keys.data(), nullptr, hint_counts.data(), hint_columns};
    st::vector<usearch_key_t> query_keys;
    st::vector<std::size_t> query_key_counts;
    std::size_t query_key_columns = 1;
    if (st::failed(query_keys.assign(queries->keys))) exit_out_of_memory("the query keys");
    if (query_keys.empty() && settings.query_keys.empty()) {
        for (std::size_t query = 0; query != queries->count; ++query)
            if (st::failed(query_keys.push_back(std::bit_cast<usearch_key_t>(uuid_of(query)))))
                exit_out_of_memory("the query keys");
    }
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
    verb_phase_t phase {.name = "search", .unit = "queries", .stream = steps};
    usearch_outcome_t const outcome = search_queries(index, queries->queries, queries->count, settings.batch.value,
                                                     settings.wanted.value, settings.expansion_search.value,
                                                     settings.traversal, totals, found,
                                                     hints_path.empty() ? nullptr : &hints, verb_progress_of(phase));
    // A stopped search writes none of its outputs, which would hold some of the queries alone.
    if (outcome.status == usearch_status_cancelled_k) {
        verb_stopped(phase);
        return EXIT_SUCCESS;
    }
    if (report_failure(outcome, "searching")) return EXIT_FAILURE;
    double const seconds = checkpoint_seconds_since(started);
    if (!matches_path.empty() && !write_key_rows(matches_path, found, settings.wanted.value)) return EXIT_FAILURE;
    if (!settings.output.empty())
        if (sz::string_t const failure = arrow_matches_write(settings.output, query_keys, found); !failure.empty()) {
            fmt::println(stderr, "{}", failure);
            return EXIT_FAILURE;
        }

    st::vector<std::size_t> const offsets = match_offsets(found);
    sz::string_t line;
    for (std::size_t query = 0; settings.output.empty() && query != queries->count; ++query) {
        line.clear();
        append_formatted(line, "query {}:", query);
        for (std::size_t match = offsets[query]; match != offsets[query + 1]; ++match) {
            append_formatted(line, " ");
            checkpoint_append_key(line, found.keys[match]);
            append_formatted(line, " {}", found.scores[match]);
        }
        fmt::println("{}", line);
    }
    sz::string_t const missing = hints_path.empty() ? sz::string_t {}
                                                    : formatted(", {} hints missing", totals.missing_hints);
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
