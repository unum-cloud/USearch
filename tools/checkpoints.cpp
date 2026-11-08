/**
 *  @file tools/checkpoints.cpp
 *  @author Ash Vardanian
 *  @date October 18, 2026
 *  @brief The verbs over checkpoints of every modality: their metadata, the slots of a raw drive
 *      holding them, verified copies between files and slots, Apache Arrow exports and imports, and
 *      the recall of one search's matches against another's.
 *
 *  How a drive is cut into slots, mapped and copied onto lives in @c slots.hpp, which the
 *  benchmarks share; deleting a file zeroes its first block. How a checkpoint's columns become
 *  Arrow fields and back lives in @c arrow_checkpoints.hpp, which the suites share.
 */
#include <cstdio>  // `stderr`
#include <cstdlib> // `EXIT_SUCCESS`, `EXIT_FAILURE`

#include <algorithm>   // `std::count_if`, `std::max_element`, `std::min`
#include <array>       // `std::array`
#include <iterator>    // `std::back_inserter`
#include <span>        // `std::span`
#include <string>      // `std::string`
#include <string_view> // `std::string_view`
#include <vector>      // `std::vector`

#include <fmt/format.h>

#include "usearch/usearch.h"
#include "usearch/utilities/nvme_device.hpp" // `names_transport_id`
#include "usearch/utilities/nvme_walks.hpp"  // `nvme_read_metadata`

#include "arrow_checkpoints.hpp" // `arrow_export`, `arrow_import`, `arrow_matches_read`
#include "slots.hpp"             // `slot_drive_t`, `slot_map`, `slot_copy`
#include "world.hpp"             // `world_t`, `parse_verb`

namespace unum::usearch::tools {

int checkpoints_slots(world_t& world, int argc, char const* const* argv) {
    std::array<std::string_view, 1> positionals;
    std::size_t given = 0;
    if (std::optional<int> const exit = parse_verb(world, argc, argv, "usearch slots <drive>", {}, positionals, given))
        return *exit;
    if (!runs_alone(world)) return EXIT_FAILURE;
    if (given != 1 || utilities::names_transport_id(positionals[0])) {
        fmt::println(stderr, "a drive the kernel's nvme driver holds is required, like /dev/nvme1n1");
        return EXIT_FAILURE;
    }
    std::string location;
    slot_drive_t drive;
    std::vector<slot_file_t> files;
    if (!slot_open_drive(positionals[0], location, drive, files)) return EXIT_FAILURE;

    u64_t const slots = drive.bytes / slot_bytes_k;
    u64_t live = 0;
    for (slot_file_t const& file : files) live += file.owner != slot_foreign_k;
    fmt::println("{}: {} bytes in {} slots of 1 GiB, {} live", location, drive.bytes, slots, live);
    u64_t next = 0;
    for (slot_file_t const& file : files) {
        if (file.first_slot > next)
            fmt::println("{}@{}GiB: free, {} {}", location, next, file.first_slot - next,
                         slots_noun(file.first_slot - next));
        u64_t const covered = slots_covered(file.bytes);
        if (file.owner == slot_foreign_k) fmt::println("{}@{}GiB: foreign, skipped", location, file.first_slot);
        else
            fmt::println("{}@{}GiB: {}, {} bytes over {} {}", location, file.first_slot, file.owner, file.bytes,
                         covered, slots_noun(covered));
        next = file.first_slot + covered;
    }
    if (next < slots) fmt::println("{}@{}GiB: free, {} {}", location, next, slots - next, slots_noun(slots - next));
    return EXIT_SUCCESS;
}

int checkpoints_copy(world_t& world, int argc, char const* const* argv) {
    std::array<std::string_view, 2> positionals;
    std::size_t given = 0;
    if (std::optional<int> const exit = parse_verb(world, argc, argv, "usearch copy <from> <to>", {}, positionals,
                                                   given))
        return *exit;
    if (!runs_alone(world)) return EXIT_FAILURE;
    std::string_view const from = positionals[0], to = positionals[1];
    if (given != 2 || utilities::names_transport_id(from) || utilities::names_transport_id(to)) {
        fmt::println(stderr, "a source and a destination the kernel reaches are required");
        return EXIT_FAILURE;
    }
    auto const [source_address, source_status] = utilities::parse_file_address(from);
    auto const [target_address, target_status] = utilities::parse_file_address(to);
    if (utilities::failed(source_status) || utilities::failed(target_status)) {
        fmt::println(stderr, "{} is no address", utilities::failed(source_status) ? from : to);
        return EXIT_FAILURE;
    }

    // A drive holds files back to back, so its source is the live file its header bounds.
    std::string const source_location {source_address.location};
    slot_drive_t source;
    if (!slot_open(source_location, source)) return EXIT_FAILURE;
    u64_t const source_base = source_address.byte_base;
    u64_t length = source.bytes - min_of(source_base, source.bytes);
    if (slot_names_device(source_location)) {
        slot_file_t file;
        if (!slot_probe(source, source_base, file) || file.owner == slot_foreign_k) {
            fmt::println(stderr, "{} holds no SafeGraphs monolith", from);
            return EXIT_FAILURE;
        }
        length = file.bytes;
    }

    std::string const target_location {target_address.location};
    u64_t target_base = target_address.byte_base;
    if (slot_names_device(target_location)) {
        slot_drive_t target;
        std::vector<slot_file_t> files;
        if (!slot_open(target_location, target) || !slot_map(target, target_location, files)) return EXIT_FAILURE;
        u64_t const slots = target.bytes / slot_bytes_k, needed = slots_covered(length);
        if (target_address.location.size() == to.size())
            target_base = slot_free_run(files, slots, needed) * slot_bytes_k;
        if (target_base % slot_bytes_k || target_base / slot_bytes_k + needed > slots) {
            fmt::println(stderr, "{} {} from byte {} of {} would sit off its slot boundaries or past its {} slots",
                         needed, slots_noun(needed), target_base, target_location, slots);
            return EXIT_FAILURE;
        }
        if (slot_file_t const* live = slot_overlap(files, target_base / slot_bytes_k, needed)) {
            fmt::println(stderr, "{} {} from {}@{}GiB would overlap the {} at {}@{}GiB", needed, slots_noun(needed),
                         target_location, target_base / slot_bytes_k, live->owner, target_location, live->first_slot);
            return EXIT_FAILURE;
        }
    }
    return slot_copy(from, target_location, target_base, length) ? EXIT_SUCCESS : EXIT_FAILURE;
}

/** The modalities a checkpoint's metric token tells apart. */
enum class checkpoints_modality_t { dense_k, sequence_k, shapes_k, geospatial_k };

/** The modality whose engine records @p metric, as each modality reads its checkpoints' tokens;
 *  dense for the rest, whose tokens are NumKong's kernel names. */
static checkpoints_modality_t checkpoints_modality_of(std::string_view metric) noexcept {
    if (usearch_sequence_metric_from_name(bytes_view(metric)) != usearch_sequence_metric_unknown_k)
        return checkpoints_modality_t::sequence_k;
    if (usearch_shapes_metric_from_name(bytes_view(metric)) != usearch_shapes_metric_unknown_k)
        return checkpoints_modality_t::shapes_k;
    // Geospatial indexes measure the one chord distance, so the interface names no metric to parse.
    if (metric == "chord") return checkpoints_modality_t::geospatial_k;
    return checkpoints_modality_t::dense_k;
}

int checkpoints_metadata(world_t& world, int argc, char const* const* argv) {
    std::array<std::string_view, 1> positionals;
    std::size_t given = 0;
    if (std::optional<int> const exit = parse_verb(world, argc, argv, "usearch metadata <address|->", {}, positionals,
                                                   given))
        return *exit;
    if (!runs_alone(world)) return EXIT_FAILURE;
    std::string_view const address = positionals[0];
    if (given != 1 || utilities::names_transport_id(address)) {
        fmt::println(stderr, "a monolith or manifest the kernel reaches, or - for standard input, is required");
        return EXIT_FAILURE;
    }
    bool const piped = address == "-";
    std::vector<byte_t> piped_bytes;
    if (piped && !read_standard_input(piped_bytes)) {
        fmt::println(stderr, "cannot read a checkpoint off standard input");
        return EXIT_FAILURE;
    }
    auto const [header, header_status] = piped ? utilities::safegraphs_read_metadata(bytes_view_t {piped_bytes})
                                               : utilities::nvme_read_metadata<fu::standard_yield_t>(address);
    if (utilities::failed(header_status)) {
        fmt::println(stderr, "{} holds no SafeGraphs header or manifest", address);
        return EXIT_FAILURE;
    }
    std::string_view const metric = header.metric.view();
    fmt::println("{}: {} nodes in {} parts, metric \"{}\", {} keys", address, header.nodes, header.parts, metric,
                 utilities::dtype_name(header.key_dtype));

    usearch_outcome_t outcome {};
    usearch_bytes_view_t const path = bytes_view(address);
    usearch_bytes_view_t const buffer {reinterpret_cast<usearch_byte_t const*>(piped_bytes.data()), piped_bytes.size()};
    std::string const reading = fmt::format("reading {}", address);
    switch (checkpoints_modality_of(metric)) {
    case checkpoints_modality_t::dense_k: {
        usearch_dense_metadata_t metadata {};
        usearch_dense_metadata_read_t read {.outcome = &outcome, .path = path, .metadata = &metadata};
        if (piped) dense_verbs_t::metadata_read_buffer(nullptr, buffer, &metadata, &outcome);
        else usearch_dense_metadata_read(&read);
        if (report_failure(outcome, reading)) return EXIT_FAILURE;
        fmt::println("dense: {} metric, {} dimensions of {}, connectivity {}",
                     usearch_dense_metric_name(metadata.metric), metadata.dimensions,
                     usearch_scalar_name(metadata.quantization), metadata.connectivity);
        return EXIT_SUCCESS;
    }
    case checkpoints_modality_t::sequence_k: {
        usearch_sequence_metadata_t metadata {};
        usearch_sequence_metadata_read_t read {.outcome = &outcome, .path = path, .metadata = &metadata};
        if (piped) sequence_verbs_t::metadata_read_buffer(nullptr, buffer, &metadata, &outcome);
        else usearch_sequence_metadata_read(&read);
        if (report_failure(outcome, reading)) return EXIT_FAILURE;
        fmt::println("sequence: connectivity {}", metadata.connectivity);
        return EXIT_SUCCESS;
    }
    case checkpoints_modality_t::shapes_k: {
        usearch_shapes_metadata_t metadata {};
        usearch_shapes_metadata_read_t read {.outcome = &outcome, .path = path, .metadata = &metadata};
        if (piped) shapes_verbs_t::metadata_read_buffer(nullptr, buffer, &metadata, &outcome);
        else usearch_shapes_metadata_read(&read);
        if (report_failure(outcome, reading)) return EXIT_FAILURE;
        fmt::println("shapes: {} points of {}, connectivity {}", metadata.points,
                     usearch_scalar_name(metadata.quantization), metadata.connectivity);
        return EXIT_SUCCESS;
    }
    case checkpoints_modality_t::geospatial_k: {
        usearch_geospatial_metadata_t metadata {};
        usearch_geospatial_metadata_read_t read {.outcome = &outcome, .path = path, .metadata = &metadata};
        if (piped) geospatial_verbs_t::metadata_read_buffer(nullptr, buffer, &metadata, &outcome);
        else usearch_geospatial_metadata_read(&read);
        if (report_failure(outcome, reading)) return EXIT_FAILURE;
        fmt::println("geospatial: connectivity {}", metadata.connectivity);
        return EXIT_SUCCESS;
    }
    }
    return EXIT_FAILURE;
}

int checkpoints_export(world_t& world, int argc, char const* const* argv) {
    std::array<std::string_view, 2> positionals;
    std::size_t given = 0;
    if (std::optional<int> const exit = parse_verb(world, argc, argv, "usearch export <address> <file.arrow>", {},
                                                   positionals, given))
        return *exit;
    if (!runs_alone(world)) return EXIT_FAILURE;
    std::string_view const address = positionals[0];
    if (given != 2 || utilities::names_transport_id(address)) {
        fmt::println(stderr, "a monolith or manifest the kernel reaches and an Arrow file are required");
        return EXIT_FAILURE;
    }
    std::string const arrow_path {positionals[1]};
    arrow_summary_t summary;
    if (std::string const failure = arrow_export(address, arrow_path, summary); !failure.empty()) {
        fmt::println(stderr, "{}", failure);
        return EXIT_FAILURE;
    }
    fmt::println("exported {} to {}: {} nodes in {} record batches, metric \"{}\", {} bytes", address, arrow_path,
                 summary.nodes, summary.parts, summary.metric, summary.bytes);
    return EXIT_SUCCESS;
}

/** Copies the imported @p checkpoint into a RAM index through the verbs of its modality, then saves
 *  it at @p address; true once both landed. */
template <typename verbs_type_>
static bool checkpoints_import_as(usearch_executor_t executor, std::string_view checkpoint,
                                  std::string const& address) {
    typename verbs_type_::index_t index = nullptr;
    usearch_outcome_t outcome {};
    verbs_type_::load_ram_buffer(&index, executor, bytes_view(checkpoint), {}, {}, &outcome);
    if (report_failure(outcome, "loading the imported checkpoint")) return false;
    verbs_type_::save(index, executor, bytes_view(address), &outcome);
    verbs_type_::free(index);
    return !report_failure(outcome, fmt::format("saving {}", address));
}

int checkpoints_import(world_t& world, int argc, char const* const* argv) {
    std::string_view metric;
    std::array<flag_t, 1> const flags {{
        {"--metric", "name",
         "defaults to the one the file records: refuses a file recording another, and names it for one recording none",
         &metric},
    }};
    std::array<std::string_view, 2> positionals;
    std::size_t given = 0;
    if (std::optional<int> const exit = parse_verb(world, argc, argv, "usearch import <file.arrow> <address> [flags]",
                                                   flags, positionals, given))
        return *exit;
    if (!runs_alone(world)) return EXIT_FAILURE;
    if (given != 2) {
        fmt::println(stderr, "an Arrow file and the address to save its checkpoint at are required");
        return EXIT_FAILURE;
    }
    std::optional<std::string> const admitted = checkpoints_admit_save(positionals[1]);
    if (!admitted) return EXIT_FAILURE;
    std::string checkpoint;
    arrow_summary_t summary;
    if (std::string const failure = arrow_import(std::string {positionals[0]}, metric, checkpoint, summary);
        !failure.empty()) {
        fmt::println(stderr, "{}", failure);
        return EXIT_FAILURE;
    }

    usearch_executor_t executor = nullptr;
    usearch_outcome_t outcome {};
    usearch_executor_for_ram_t placement {.executor = &executor, .outcome = &outcome};
    usearch_executor_for_ram(&placement);
    if (report_failure(outcome, "creating the CPU executor")) return EXIT_FAILURE;
    bool imported = false;
    switch (checkpoints_modality_of(summary.metric)) {
    case checkpoints_modality_t::dense_k:
        imported = checkpoints_import_as<dense_verbs_t>(executor, checkpoint, *admitted);
        break;
    case checkpoints_modality_t::sequence_k:
        imported = checkpoints_import_as<sequence_verbs_t>(executor, checkpoint, *admitted);
        break;
    case checkpoints_modality_t::shapes_k:
        imported = checkpoints_import_as<shapes_verbs_t>(executor, checkpoint, *admitted);
        break;
    case checkpoints_modality_t::geospatial_k:
        imported = checkpoints_import_as<geospatial_verbs_t>(executor, checkpoint, *admitted);
        break;
    }
    usearch_executor_free(executor);
    if (!imported) return EXIT_FAILURE;
    fmt::println("imported {} into {}: {} nodes in {} parts, metric \"{}\"", positionals[0], *admitted, summary.nodes,
                 summary.parts, summary.metric);
    return EXIT_SUCCESS;
}

/** Copies the distances of a query's matches into @p kept, past the one holding the query's own key
 *  at distance zero, which is the query finding itself; true when it dropped that one. */
static bool recall_keep(uuid_t const& query, std::span<uuid_t const> keys,
                        std::span<usearch_distance_t const> distances, std::vector<usearch_distance_t>& kept) {
    kept.clear();
    bool dropped = false;
    for (std::size_t match = 0; match != keys.size(); ++match) {
        if (!dropped && keys[match] == query && distances[match] == 0) dropped = true;
        else kept.push_back(distances[match]);
    }
    return dropped;
}

int checkpoints_recall(world_t& world, int argc, char const* const* argv) {
    counts_t wanted {.values = {1, 10}, .size = 2};
    std::array<flag_t, 1> const flags {{
        {"--wanted", "counts", "defaults to 1,10: nearest true neighbors to find among as many matches as the largest",
         &wanted},
    }};
    std::array<std::string_view, 2> positionals;
    std::size_t given = 0;
    if (std::optional<int> const exit = parse_verb(
            world, argc, argv, "usearch recall <truth.arrow> <run.arrow> [flags]", flags, positionals, given))
        return *exit;
    if (!runs_alone(world)) return EXIT_FAILURE;
    if (given != 2) {
        fmt::println(stderr, "the truth and the run, each written by a search's --output, are required");
        return EXIT_FAILURE;
    }
    arrow_matches_t truth, run;
    for (std::string const& failure : {arrow_matches_read(std::string {positionals[0]}, truth),
                                       arrow_matches_read(std::string {positionals[1]}, run)})
        if (!failure.empty()) {
            fmt::println(stderr, "{}", failure);
            return EXIT_FAILURE;
        }
    if (truth.queries != run.queries) {
        fmt::println(stderr, "{} and {} did not search the same queries", positionals[0], positionals[1]);
        return EXIT_FAILURE;
    }

    std::span<std::size_t const> const depths = wanted.view();
    std::size_t const pool = *std::max_element(depths.begin(), depths.end());
    std::vector<std::size_t> const truth_offsets = match_offsets(truth.matches),
                                   run_offsets = match_offsets(run.matches);
    std::vector<usearch_distance_t> true_distances, run_distances;
    std::vector<double> recalled(depths.size());
    std::size_t own_keys = 0;
    std::span<uuid_t const> const truth_keys {truth.matches.keys}, run_keys {run.matches.keys};
    std::span<usearch_distance_t const> const truth_scores {truth.matches.scores}, run_scores {run.matches.scores};
    for (std::size_t query = 0; query != truth.queries.size(); ++query) {
        std::size_t const truth_first = truth_offsets[query], truth_count = truth.matches.counts[query];
        std::size_t const run_first = run_offsets[query], run_count = run.matches.counts[query];
        own_keys += recall_keep(truth.queries[query], truth_keys.subspan(truth_first, truth_count),
                                truth_scores.subspan(truth_first, truth_count), true_distances);
        recall_keep(run.queries[query], run_keys.subspan(run_first, run_count),
                    run_scores.subspan(run_first, run_count), run_distances);
        if (true_distances.size() < pool) {
            fmt::println(stderr, "{} holds {} matches past query {}'s own key, fewer than the {} wanted",
                         positionals[0], true_distances.size(), query, pool);
            return EXIT_FAILURE;
        }
        run_distances.resize(std::min(pool, run_distances.size()));
        // A match within the depth's true distance counts, so ties at the cut count either way.
        for (std::size_t index = 0; index != depths.size(); ++index) {
            usearch_distance_t const bound = true_distances[depths[index] - 1];
            std::size_t const within_bound = std::count_if(run_distances.begin(), run_distances.end(),
                                                           [&](usearch_distance_t found) { return found <= bound; });
            recalled[index] += static_cast<double>(std::min(within_bound, depths[index])) /
                               static_cast<double>(depths[index]);
        }
    }
    std::string line;
    for (std::size_t index = 0; index != depths.size(); ++index)
        fmt::format_to(std::back_inserter(line), "{}recall@{} at {}: {:.4f}", index ? ", " : "", depths[index], pool,
                       recalled[index] / static_cast<double>(truth.queries.size()));
    fmt::println("{} over {} queries, ties counted, {} own keys dropped", line, truth.queries.size(), own_keys);
    return EXIT_SUCCESS;
}

} // namespace unum::usearch::tools
