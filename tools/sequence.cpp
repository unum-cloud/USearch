/**
 *  @file tools/sequence.cpp
 *  @author Ash Vardanian
 *  @date November 1, 2026
 *  @brief The sequence verbs: insert, refine, reshard, health and search, each over SafeGraphs
 *      checkpoints at any address.
 *
 *  Entries and queries are text files, one entry per line, as the sequence benchmark reads them,
 *  each newline dropped with a carriage return before it. Keys are the entries' line numbers, so
 *  slices built apart keep distinct keys when merged. Every metric scores by its own costs unless
 *  an insert names others, which the checkpoint records, so a search takes raw text alone.
 */
#include <cstdio>  // `stderr`
#include <cstdlib> // `EXIT_FAILURE`

#include <array>       // `std::array`
#include <optional>    // `std::optional`
#include <string_view> // `std::string_view`

#include <fmt/format.h>

#include "usearch/sequence_index.h"
#include "usearch/utilities/blocking_files.hpp" // `mapped_file_t`

#include "abi.hpp"   // `sequence_verbs_t`, `counts_t`, `lines_of`
#include "world.hpp" // `checkpoint_insert`, `checkpoint_search`

namespace unum::usearch::tools {

/** Reads the lines of the file at @p path into @p lines; false after printing why it cannot. */
static bool sequence_read_lines(std::string_view path, byte_tape_t& lines) {
    auto [file, status] = util::mapped_file_t::read(path);
    if (failed(status)) {
        fmt::println(stderr, "cannot read {}", path);
        return false;
    }
    lines = lines_of(file.view());
    return true;
}

int modality_insert(sequence_verbs_t, world_t& world, int argc, char const* const* argv) {
    std::string_view metric_name, costs_path;
    count_t sketch_bytes {.minimum = 0};
    std::array<flag_t, 3> const flags {{
        {"--metric", "name", "required: protein-alignment, dna-overlap or utf8-containment", &metric_name},
        {"--costs", "path",
         "a file of 256x256 int8 similarities between bytes and an int8 gap cost, for the metric's own", &costs_path},
        {"--sketch-bytes", "bytes", "a multiple of four in front of each sequence; defaults to the metric's own",
         &sketch_bytes},
    }};
    byte_tape_t lines;
    // The geometry points at these costs, which outlive the reading lambda till the index opens.
    std::optional<byte_costs_t> costs;
    usearch_sequence_costs_t costs_view {};
    return checkpoint_insert<sequence_verbs_t>(
        world, argc, argv, "usearch sequence insert <texts> --metric <name> --save <address> [flags]", flags,
        [&](std::string_view path,
            checkpoint_settings_t const& settings) -> std::optional<checkpoint_entries<sequence_verbs_t>> {
            usearch_sequence_metric_t metric = usearch_sequence_metric_unknown_k;
            if (usearch_sequence_metric_from_name(bytes_view(metric_name), &metric) != usearch_status_success_k) {
                fmt::println(stderr, "--metric=\"{}\" names no metric", metric_name);
                return std::nullopt;
            }
            bool const named = !costs_path.empty();
            if (named && !(costs = read_costs(costs_path))) return std::nullopt;
            if (!sequence_read_lines(path, lines)) return std::nullopt;
            if (named) costs_view = costs->view();
            sequence_geometry_t const geometry {{metric, settings.connectivity.value, 0, sketch_bytes.value},
                                                named ? &costs_view : nullptr};
            std::optional<st::vector<usearch_key_t>> keys = checkpoint_read_keys(settings, lines.size());
            if (!keys) return std::nullopt;
            return checkpoint_entries<sequence_verbs_t> {lines.view(), lines.size(), geometry, std::move(*keys)};
        });
}

int modality_reshard(sequence_verbs_t, world_t& world, int argc, char const* const* argv) {
    std::string_view costs_path;
    std::array<flag_t, 1> const flags {{
        {"--costs", "path", "the costs the index was built with; a split opens its parts under them", &costs_path},
    }};
    // The parts of a split point at these costs, read once, until they open.
    std::optional<byte_costs_t> costs;
    usearch_sequence_costs_t costs_view {};
    auto const target_geometry = [&](usearch_sequence_metadata_t const& metadata) {
        if (costs_path.empty()) return sequence_geometry_t {metadata, nullptr};
        if (!costs) costs = read_costs(costs_path);
        // Costs that do not read open no part: no metric is named.
        if (!costs) return sequence_geometry_t {};
        costs_view = costs->view();
        return sequence_geometry_t {metadata, &costs_view};
    };
    return checkpoint_reshard<sequence_verbs_t>(world, argc, argv, flags, target_geometry);
}

int modality_search(sequence_verbs_t, world_t& world, int argc, char const* const* argv) {
    std::string_view query_texts;
    std::array<flag_t, 1> const flags {{
        {"--query-texts", "path", "required: queries, one per line", &query_texts},
    }};
    byte_tape_t lines;
    return checkpoint_search<sequence_verbs_t>(
        world, argc, argv, "usearch sequence search <address> --query-texts <path> [flags]", flags,
        [&]() -> std::optional<checkpoint_queries<sequence_verbs_t>> {
            if (query_texts.empty()) {
                fmt::println(stderr, "--query-texts is required");
                return std::nullopt;
            }
            if (!sequence_read_lines(query_texts, lines)) return std::nullopt;
            return checkpoint_queries<sequence_verbs_t> {lines.view(), lines.size()};
        });
}

} // namespace unum::usearch::tools
