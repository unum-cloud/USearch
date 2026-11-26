/**
 *  @file tools/checkpoints.cpp
 *  @author Ash Vardanian
 *  @date October 18, 2026
 *  @brief The verbs over checkpoints of every modality: their metadata, verified copies between
 *      files and slots, Apache Arrow exports and imports, and the recall of one search's
 *      matches against those of another.
 *
 *  How a drive is cut into slots, mapped and copied onto lives in @c slots.hpp, which the
 *  benchmarks share; deleting a file zeroes its first block. How a checkpoint's columns become
 *  Arrow fields and back lives in @c arrow_checkpoints.hpp, which the suites share.
 */
#include <cstdio>  // `stderr`
#include <cstdlib> // `EXIT_SUCCESS`, `EXIT_FAILURE`

#include <algorithm>   // `std::count_if`, `std::max_element`, `std::min`, `std::ranges::equal`
#include <array>       // `std::array`
#include <memory>      // `std::unique_ptr` - the metadata executor, freed on every return
#include <span>        // `std::span`
#include <string_view> // `std::string_view`

#include <fmt/format.h>

#include "usearch/usearch.h"
#include "usearch/utilities/nvme_device.hpp" // `names_transport_id`
#include "usearch/utilities/nvme_walks.hpp"  // `nvme_read_metadata`

#include "arrow_checkpoints.hpp" // `arrow_export`, `arrow_import`, `arrow_matches_read`
#include "slots.hpp"             // `slot_drive_t`, `slot_map`, `slot_copy`
#include "world.hpp"             // `world_t`, `parse_verb`

namespace unum::usearch::tools {

int checkpoints_copy(world_t& world, int argc, char const* const* argv) {
    std::array<std::string_view, 2> positionals;
    std::size_t given = 0;
    if (std::optional<int> const exit = parse_verb(world, argc, argv, "usearch copy <from> <to>", {}, positionals,
                                                   given))
        return *exit;
    if (!runs_alone(world)) return EXIT_FAILURE;
    std::string_view const from = positionals[0], to = positionals[1];
    if (given != 2 || util::names_transport_id(from) || util::names_transport_id(to)) {
        fmt::println(stderr, "a source and a destination the kernel reaches are required");
        return EXIT_FAILURE;
    }
    auto const [source_address, source_status] = util::parse_file_address(from);
    auto const [target_address, target_status] = util::parse_file_address(to);
    if (util::failed(source_status) || util::failed(target_status)) {
        fmt::println(stderr, "{} is no address", util::failed(source_status) ? from : to);
        return EXIT_FAILURE;
    }

    // A drive holds files back to back, so its source is the live file its header bounds.
    std::string_view const source_location = source_address.location;
    slot_drive_t source;
    if (!slot_open(source_location, source)) return EXIT_FAILURE;
    util::u64_t const source_base = source_address.byte_base;
    util::u64_t length = source.bytes - util::min_of(source_base, source.bytes);
    if (slot_names_device(source_location)) {
        slot_file_t file;
        if (!slot_probe(source, source_base, file) || std::string_view {file.owner} == slot_foreign_k) {
            fmt::println(stderr, "{} holds no SafeGraphs monolith", from);
            return EXIT_FAILURE;
        }
        length = file.bytes;
    }

    std::string_view const target_location = target_address.location;
    util::u64_t target_base = target_address.byte_base;
    if (slot_names_device(target_location)) {
        slot_drive_t target;
        st::vector<slot_file_t> files;
        if (!slot_open(target_location, target) || !slot_map(target, target_location, files)) return EXIT_FAILURE;
        util::u64_t const slots = target.bytes / slot_bytes_k, needed = slots_covered(length);
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
    usearch_sequence_metric_t sequence_metric = usearch_sequence_metric_unknown_k;
    if (usearch_sequence_metric_from_name(bytes_view(metric), &sequence_metric) == usearch_status_success_k)
        return checkpoints_modality_t::sequence_k;
    usearch_shapes_metric_t shapes_metric = usearch_shapes_metric_unknown_k;
    if (usearch_shapes_metric_from_name(bytes_view(metric), &shapes_metric) == usearch_status_success_k)
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
    if (given != 1 || util::names_transport_id(address)) {
        fmt::println(stderr, "a monolith or manifest the kernel reaches, or - for standard input, is required");
        return EXIT_FAILURE;
    }
    bool const piped = address == "-";
    st::vector<util::byte_t> piped_bytes;
    if (piped && !read_standard_input(piped_bytes)) {
        fmt::println(stderr, "cannot read a checkpoint off standard input");
        return EXIT_FAILURE;
    }
    auto const [header, header_status] = piped ? util::safegraphs_read_metadata(bytes_view_t {piped_bytes})
                                               : util::nvme_read_metadata<fu::standard_yield_t>(address);
    if (util::failed(header_status)) {
        fmt::println(stderr, "{} holds no SafeGraphs header or manifest", address);
        return EXIT_FAILURE;
    }
    std::string_view const metric = header.metric.view();
    fmt::println("{}: {} nodes in {} parts, metric \"{}\", {} keys", address, header.nodes, header.parts, metric,
                 dtype_name(header.key_dtype));

    usearch_outcome_t outcome {};
    usearch_executor_t executor = nullptr;
    usearch_executor_for_ram_t placement {.executor = &executor, .outcome = &outcome, .threads = 1};
    usearch_executor_for_ram(&placement);
    if (report_failure(outcome, "creating the CPU executor")) return EXIT_FAILURE;
    std::unique_ptr<usearch_executor_handle_t, void (*)(usearch_executor_t)> const owned_executor {
        executor, [](usearch_executor_t owned) { usearch_executor_free(owned); }};
    usearch_bytes_view_t const path = bytes_view(address);
    usearch_bytes_view_t const buffer = bytes_view(piped_bytes);
    switch (checkpoints_modality_of(metric)) {
    case checkpoints_modality_t::dense_k: {
        usearch_dense_metadata_t metadata {};
        usearch_dense_metadata_read_t read {.executor = executor,
                                            .outcome = &outcome,
                                            .path = path,
                                            .metadata = &metadata};
        if (piped) dense_verbs_t::metadata_read_buffer(executor, buffer, &metadata, &outcome);
        else usearch_dense_metadata_read(&read);
        if (report_failure(outcome, "reading {}", address)) return EXIT_FAILURE;
        fmt::println("dense: {} metric, {} dimensions of {}, connectivity {}",
                     usearch_dense_metric_name(metadata.metric), metadata.dimensions,
                     usearch_scalar_name(metadata.quantization), metadata.connectivity);
        return EXIT_SUCCESS;
    }
    case checkpoints_modality_t::sequence_k: {
        usearch_sequence_metadata_t metadata {};
        usearch_sequence_metadata_read_t read {.executor = executor,
                                               .outcome = &outcome,
                                               .path = path,
                                               .metadata = &metadata};
        if (piped) sequence_verbs_t::metadata_read_buffer(executor, buffer, &metadata, &outcome);
        else usearch_sequence_metadata_read(&read);
        if (report_failure(outcome, "reading {}", address)) return EXIT_FAILURE;
        fmt::println("sequence: connectivity {}", metadata.connectivity);
        return EXIT_SUCCESS;
    }
    case checkpoints_modality_t::shapes_k: {
        usearch_shapes_metadata_t metadata {};
        usearch_shapes_metadata_read_t read {.executor = executor,
                                             .outcome = &outcome,
                                             .path = path,
                                             .metadata = &metadata};
        if (piped) shapes_verbs_t::metadata_read_buffer(executor, buffer, &metadata, &outcome);
        else usearch_shapes_metadata_read(&read);
        if (report_failure(outcome, "reading {}", address)) return EXIT_FAILURE;
        fmt::println("shapes: {} points of {}, connectivity {}", metadata.points,
                     usearch_scalar_name(metadata.quantization), metadata.connectivity);
        return EXIT_SUCCESS;
    }
    case checkpoints_modality_t::geospatial_k: {
        usearch_geospatial_metadata_t metadata {};
        usearch_geospatial_metadata_read_t read {.executor = executor,
                                                 .outcome = &outcome,
                                                 .path = path,
                                                 .metadata = &metadata};
        if (piped) geospatial_verbs_t::metadata_read_buffer(executor, buffer, &metadata, &outcome);
        else usearch_geospatial_metadata_read(&read);
        if (report_failure(outcome, "reading {}", address)) return EXIT_FAILURE;
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
    if (given != 2 || util::names_transport_id(address)) {
        fmt::println(stderr, "a monolith or manifest the kernel reaches and an Arrow file are required");
        return EXIT_FAILURE;
    }
    std::string_view const arrow_path = positionals[1];
    arrow_summary_t summary;
    if (sz::string_t const failure = arrow_export(address, arrow_path, summary); !failure.empty()) {
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
static bool checkpoints_import_as(usearch_executor_t executor, bytes_view_t checkpoint, std::string_view address) {
    typename verbs_type_::index_t index = nullptr;
    usearch_outcome_t outcome {};
    verbs_type_::load_ram_buffer(&index, executor, bytes_view(checkpoint), {}, {}, &outcome);
    if (report_failure(outcome, "loading the imported checkpoint")) return false;
    verbs_type_::save(index, executor, bytes_view(address), &outcome);
    verbs_type_::free(index);
    return !report_failure(outcome, "saving {}", address);
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
    std::optional<sz::string_t> const admitted = checkpoints_admit_save(positionals[1]);
    if (!admitted) return EXIT_FAILURE;
    st::vector<util::byte_t> checkpoint;
    arrow_summary_t summary;
    if (sz::string_t const failure = arrow_import(positionals[0], metric, checkpoint, summary); !failure.empty()) {
        fmt::println(stderr, "{}", failure);
        return EXIT_FAILURE;
    }
    // A stop while reading writes nothing; once the save starts, it runs to its end.
    if (stop_requested) {
        verb_stopped({.name = "import"});
        return EXIT_SUCCESS;
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
static bool recall_keep(util::uuid128_t const& query, std::span<util::uuid128_t const> keys,
                        std::span<usearch_distance_t const> distances, st::vector<usearch_distance_t>& kept) noexcept {
    kept.clear();
    bool dropped = false;
    for (std::size_t match = 0; match != keys.size(); ++match) {
        if (!dropped && keys[match] == query && distances[match] == 0) dropped = true;
        else if (st::failed(kept.push_back(usearch_distance_t {distances[match]}))) exit_out_of_memory("the distances");
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
    for (std::size_t file = 0; file != positionals.size(); ++file)
        if (sz::string_t const failure = arrow_matches_read(positionals[file], file ? run : truth); !failure.empty()) {
            fmt::println(stderr, "{}", failure);
            return EXIT_FAILURE;
        }
    if (!std::ranges::equal(truth.queries, run.queries)) {
        fmt::println(stderr, "{} and {} did not search the same queries", positionals[0], positionals[1]);
        return EXIT_FAILURE;
    }

    std::span<std::size_t const> const depths = wanted.view();
    std::size_t const pool = *std::max_element(depths.begin(), depths.end());
    st::vector<std::size_t> const truth_offsets = match_offsets(truth.matches),
                                  run_offsets = match_offsets(run.matches);
    st::vector<usearch_distance_t> true_distances, run_distances;
    std::array<double, counts_t::capacity_k> recalled {};
    std::size_t own_keys = 0;
    std::span<util::uuid128_t const> const truth_keys {truth.matches.keys}, run_keys {run.matches.keys};
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
        if (st::failed(run_distances.resize(util::min_of(pool, run_distances.size()))))
            exit_out_of_memory("the distances");
        // A match within the depth's true distance counts, so ties at the cut count either way.
        for (std::size_t index = 0; index != depths.size(); ++index) {
            usearch_distance_t const bound = true_distances[depths[index] - 1];
            std::size_t const within_bound = std::count_if(
                run_distances.begin(), run_distances.end(),
                [&](usearch_distance_t found) { return within(found, bound); });
            recalled[index] += static_cast<double>(std::min(within_bound, depths[index])) /
                               static_cast<double>(depths[index]);
        }
    }
    sz::string_t line;
    for (std::size_t index = 0; index != depths.size(); ++index)
        append_formatted(line, "{}{}-recall@{}: {:.4f}", index ? ", " : "", depths[index], pool,
                         recalled[index] / static_cast<double>(truth.queries.size()));
    fmt::println("{} over {} queries, ties counted, {} own keys dropped", line, truth.queries.size(), own_keys);
    return EXIT_SUCCESS;
}

} // namespace unum::usearch::tools
