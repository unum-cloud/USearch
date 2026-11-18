/**
 *  @file tools/sequence.cpp
 *  @author Ash Vardanian
 *  @date November 1, 2026
 *  @brief The sequence verbs: insert, refine, reshard, health and search, each over SafeGraphs
 *      checkpoints at any address.
 *
 *  Entries and queries are text files, one entry per line, as the sequence benchmark reads them,
 *  each newline dropped with a carriage return before it. Keys are the entries' line numbers, so
 *  slices built apart keep distinct keys when merged. Byte and UTF-8 Levenshtein take no settings,
 *  window overlap takes its window widths at insert and at a reshard splitting one, and
 *  Needleman-Wunsch takes an alphabet, which folds every byte into a class and packs the codes at
 *  insert and at search, and optionally weights replacing the alphabet's costs at insert.
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

/** The alphabet @p name spells, under the weights at @p weights_path when given; false after
 *  printing why it cannot. */
static bool sequence_alphabet(std::string_view name, std::string_view weights_path, alphabet_t& alphabet) {
    alphabet = alphabet_of(name, alphabet_costs_t::own_k);
    if (!alphabet.classes) {
        fmt::println(stderr, "--alphabet=\"{}\" must be dna, protein or text", name);
        return false;
    }
    return weights_path.empty() || read_weights(weights_path, alphabet);
}

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
    std::string_view metric_name = "levenshtein", alphabet_name, weights_path;
    counts_t window_widths {.minimum_size = 0};
    std::array<flag_t, 4> const flags {{
        {"--metric", "name", "defaults to levenshtein: levenshtein, levenshtein-utf8, overlap or needleman-wunsch",
         &metric_name},
        {"--window-widths", "bytes", "the overlap kind's window widths, like 3,5", &window_widths},
        {"--alphabet", "name", "dna, protein or text, required by needleman-wunsch alone: folds bytes into classes",
         &alphabet_name},
        {"--weights", "path", "256x256 int8 similarities and an int8 gap cost replacing the alphabet's costs",
         &weights_path},
    }};
    byte_tape_t lines;
    alphabet_t alphabet;
    // The geometry points at these weights, which outlive the reading lambda until the index opens.
    usearch_sequence_weights_t weights {};
    return checkpoint_insert<sequence_verbs_t>(
        world, argc, argv, "usearch sequence insert <texts> --save <address> [flags]", flags,
        [&](std::string_view path,
            checkpoint_settings_t const& settings) -> std::optional<checkpoint_entries<sequence_verbs_t>> {
            usearch_sequence_metric_t metric = usearch_sequence_metric_unknown_k;
            if (usearch_sequence_metric_from_name(bytes_view(metric_name), &metric) != usearch_status_success_k) {
                fmt::println(stderr, "--metric=\"{}\" names no metric", metric_name);
                return std::nullopt;
            }
            bool const aligned = metric == usearch_sequence_metric_needleman_wunsch_k;
            if (aligned != !alphabet_name.empty() || (!aligned && !weights_path.empty())) {
                fmt::println(stderr,
                             "--alphabet and --weights belong to needleman-wunsch alone, which needs --alphabet");
                return std::nullopt;
            }
            if (aligned && !sequence_alphabet(alphabet_name, weights_path, alphabet)) return std::nullopt;
            if (!sequence_read_lines(path, lines)) return std::nullopt;
            if (aligned) lines = packed_codes_of(lines, alphabet);
            weights = alphabet.weights();
            sequence_geometry_t const geometry {{metric, settings.connectivity.value},
                                                aligned ? &weights : nullptr,
                                                window_widths.view()};
            std::optional<st::vector<usearch_key_t>> keys = checkpoint_read_keys(settings, lines.size());
            if (!keys) return std::nullopt;
            return checkpoint_entries<sequence_verbs_t> {lines.view(), lines.size(), geometry, std::move(*keys)};
        });
}

int modality_reshard(sequence_verbs_t, world_t& world, int argc, char const* const* argv) {
    counts_t window_widths {.minimum_size = 0};
    std::array<flag_t, 1> const flags {{
        {"--window-widths", "bytes", "the overlap kind's window widths, as the checkpoint was built with",
         &window_widths},
    }};
    return checkpoint_reshard<sequence_verbs_t>(
        world, argc, argv, flags, [&](usearch_sequence_metadata_t const& metadata) {
            return sequence_geometry_t {metadata, nullptr, window_widths.view()};
        });
}

int modality_search(sequence_verbs_t, world_t& world, int argc, char const* const* argv) {
    std::string_view query_texts, alphabet_name;
    std::array<flag_t, 2> const flags {{
        {"--query-texts", "path", "required: queries, one per line", &query_texts},
        {"--alphabet", "name", "dna, protein or text, as a needleman-wunsch index was built with: packs the queries",
         &alphabet_name},
    }};
    byte_tape_t lines;
    return checkpoint_search<sequence_verbs_t>(
        world, argc, argv, "usearch sequence search <address> --query-texts <path> [flags]", flags,
        [&]() -> std::optional<checkpoint_queries<sequence_verbs_t>> {
            if (query_texts.empty()) {
                fmt::println(stderr, "--query-texts is required");
                return std::nullopt;
            }
            alphabet_t alphabet;
            if (!alphabet_name.empty() && !sequence_alphabet(alphabet_name, {}, alphabet)) return std::nullopt;
            if (!sequence_read_lines(query_texts, lines)) return std::nullopt;
            if (!alphabet_name.empty()) lines = packed_codes_of(lines, alphabet);
            return checkpoint_queries<sequence_verbs_t> {lines.view(), lines.size()};
        });
}

} // namespace unum::usearch::tools
