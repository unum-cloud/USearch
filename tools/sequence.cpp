/**
 *  @file tools/sequence.cpp
 *  @author Ash Vardanian
 *  @date November 1, 2026
 *  @brief The sequence verbs: insert, absorb, refine, split, health and search, each over
 *      SafeGraphs checkpoints at any address.
 *
 *  Entries and queries are text files, one entry per line, as the sequence benchmark reads them,
 *  each newline dropped with a carriage return before it. Keys are the entries' line numbers, so
 *  slices built apart keep distinct keys when merged. Byte and UTF-8 Levenshtein take no settings,
 *  and window overlap takes its window widths at insert and at split.
 */
#include <cstdio> // `stderr`

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
    auto [file, status] = mapped_file_t::read(path);
    if (failed(status)) {
        fmt::println(stderr, "cannot read {}", path);
        return false;
    }
    lines = lines_of(file.view());
    return true;
}

int modality_insert(sequence_verbs_t, world_t& world, int argc, char const* const* argv) {
    std::string_view metric_name = "levenshtein";
    counts_t window_widths {.minimum_size = 0};
    std::array<flag_t, 2> const flags {{
        {"--metric", "name", "defaults to levenshtein: levenshtein, levenshtein-utf8 or overlap", &metric_name},
        {"--window-widths", "bytes", "the overlap kind's window widths, like 3,5", &window_widths},
    }};
    byte_tape_t lines;
    return checkpoint_insert<sequence_verbs_t>(
        world, argc, argv, "usearch sequence insert <texts> --save <address> [flags]", flags,
        [&](std::string_view path,
            checkpoint_settings_t const& settings) -> std::optional<checkpoint_entries<sequence_verbs_t>> {
            usearch_sequence_metric_t const metric = usearch_sequence_metric_from_name(bytes_view(metric_name));
            if (metric == usearch_sequence_metric_unknown_k) {
                fmt::println(stderr, "--metric=\"{}\" names no metric", metric_name);
                return std::nullopt;
            }
            if (!sequence_read_lines(path, lines)) return std::nullopt;
            sequence_geometry_t const geometry {{metric, settings.connectivity.value}, nullptr, window_widths.view()};
            return checkpoint_entries<sequence_verbs_t> {lines.view(), lines.size(), geometry};
        });
}

int modality_split(sequence_verbs_t, world_t& world, int argc, char const* const* argv) {
    counts_t window_widths {.minimum_size = 0};
    std::array<flag_t, 1> const flags {{
        {"--window-widths", "bytes", "the overlap kind's window widths, as the checkpoint was built with",
         &window_widths},
    }};
    return checkpoint_split<sequence_verbs_t>(world, argc, argv, flags,
                                              [&](usearch_sequence_metadata_t const& metadata) {
                                                  return sequence_geometry_t {metadata, nullptr, window_widths.view()};
                                              });
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
