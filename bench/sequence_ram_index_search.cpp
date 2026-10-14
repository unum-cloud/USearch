/**
 *  @file bench/sequence_ram_index_search.cpp
 *  @author Ash Vardanian
 *  @date October 14, 2026
 *  @brief The sequence search benchmark: a text file, one entry per line, built into one RAM index
 *      through the C interface and judged against an exact Levenshtein or overlap scan.
 *
 *  @verbatim
 *  --base-texts     required  Entries, one per line
 *  --window-widths  none      Window widths in bytes, which the overlap kind alone takes
 *  --metric         levenshtein, levenshtein-utf8 or overlap; levenshtein by default
 *  @endverbatim
 */
#include <array>       // `std::array` - one round of the exact scan
#include <limits>      // `std::numeric_limits` - the exact scan's open radius
#include <string>      // `std::string` - the lines end to end
#include <string_view> // `std::string_view`

#include "usearch/sequence_index.h"
#include "usearch/sequence/ram_metric.hpp"
#include "usearch/utilities/blocking_files.hpp" // `mapped_file_t`

#include "basic.cuh"

namespace unum::usearch::bench {

/** Entries per round of the exact scan. */
constexpr std::size_t oracle_round_k = 4096;

/** The lines of @p text as entries end to end in @p characters, each newline dropped with a
 *  carriage return before it. */
inline tape_t lines_of(bytes_view_t text, std::string& characters) {
    tape_t tape;
    tape.scalar = usearch_scalar_u8_k;
    characters.reserve(text.size());
    for (byte_t const byte : text) {
        char const character = static_cast<char>(byte);
        if (character != '\n') {
            characters.push_back(character);
            continue;
        }
        if (characters.size() != tape.offsets.back() && characters.back() == '\r') characters.pop_back();
        tape.offsets.push_back(characters.size());
    }
    if (characters.size() != tape.offsets.back()) tape.offsets.push_back(characters.size());
    tape.elements = characters.data();
    return tape;
}

/** Fills one distance per base entry for the entry at @p query, scored by @p metric in rounds and
 *  scaled by @p unit onto the C interface's scale. */
template <typename metric_type_>
void scan_exactly(metric_type_& metric, std::string_view characters, std::span<std::size_t const> offsets,
                  std::size_t query, std::span<double> distances, double unit) {
    auto const text_of = [&](std::size_t row) noexcept {
        return characters.substr(offsets[row], offsets[row + 1] - offsets[row]);
    };
    std::array<i64_t, oracle_round_k> round;
    metric.reset(text_of(query));
    for (std::size_t first = 0; first < distances.size(); first += oracle_round_k) {
        std::size_t const count = std::min(oracle_round_k, distances.size() - first);
        for (std::size_t row = first; row != first + count; ++row) static_cast<void>(metric.push(text_of(row)));
        metric.drain(std::span {round}.first(count), (std::numeric_limits<i64_t>::max)());
        for (std::size_t slot = 0; slot != count; ++slot)
            distances[first + slot] = static_cast<double>(round[slot]) * unit;
    }
}

/** Runs the sequence search benchmark over the flags in @p argv. */
int bench_sequence_ram_index_search(int argc, char const* const* argv) {
    basic_settings_t settings;
    std::string_view base_texts_path;
    counts_t window_widths {.minimum_size = 0};
    std::array<flag_t, 2> const flags {{
        {"--base-texts", "path", "entries, one per line", &base_texts_path},
        {"--window-widths", "bytes", "window widths for the overlap kind, like 3,5", &window_widths},
    }};
    read_basic_settings(argc, argv, "usearch_bench_sequence_ram_index_search --base-texts <path> [flags]", settings,
                        flags);

    std::string_view const metric_name = settings.metric_name.empty() ? "levenshtein" : settings.metric_name;
    usearch_sequence_metric_t const metric = metric_name == "levenshtein" ? usearch_sequence_metric_levenshtein_bytes_k
                                             : metric_name == "levenshtein-utf8"
                                                 ? usearch_sequence_metric_levenshtein_utf8_k
                                             : metric_name == "overlap" ? usearch_sequence_metric_overlap_k
                                                                        : usearch_sequence_metric_unknown_k;
    if (metric == usearch_sequence_metric_unknown_k)
        return fmt::println(stderr, "--metric=\"{}\" does not parse, expected levenshtein, levenshtein-utf8 or overlap",
                            metric_name),
               1;
    auto [file, read_status] = mapped_file_t::read(base_texts_path);
    if (failed(read_status)) return fmt::println(stderr, "--base-texts=\"{}\" could not be read", base_texts_path), 1;
    std::string characters;
    tape_t const tape = lines_of(file.view(), characters);

    usearch_outcome_t outcome {};
    usearch_sequence_index_t index = nullptr;
    usearch_sequence_init_ram_t init {.index = &index,
                                      .outcome = &outcome,
                                      .metric = metric,
                                      .window_widths = window_widths.values.data(),
                                      .window_widths_count = window_widths.size,
                                      .connectivity = settings.connectivity.value};
    usearch_sequence_init_ram(&init);
    if (outcome.message) return fmt::println(stderr, "Init: {}", outcome.message), 1;
    usearch_sequence_index_description_t description {};
    usearch_sequence_describe_t describe {.index = index, .outcome = &outcome, .description = &description};
    usearch_sequence_describe(&describe);
    fmt::println("- Metric: {} · CPU class: {}", metric_name, description.capabilities);

    auto const insert = [&](usearch_executor_t executor, usearch_keys_view_t keys, usearch_tape_view_t entries,
                            std::size_t count, usearch_sequence_traversal_stats_t* stats, usearch_outcome_t* reached) {
        usearch_sequence_insert_t arguments {.index = index,
                                             .executor = executor,
                                             .outcome = reached,
                                             .count = count,
                                             .keys = keys,
                                             .sequences = entries,
                                             .expansion_factor = settings.expansion_add.value,
                                             .stats = stats};
        usearch_sequence_insert(&arguments);
    };
    auto const search = [&](usearch_executor_t executor, usearch_arena_t arena, usearch_tape_view_t queries,
                            std::size_t count, usearch_matches_view_t* matches,
                            usearch_sequence_traversal_stats_t* stats, usearch_outcome_t* reached) {
        usearch_sequence_search_t arguments {.index = index,
                                             .executor = executor,
                                             .outcome = reached,
                                             .count = count,
                                             .queries = queries,
                                             .matches = matches,
                                             .arena = arena,
                                             .wanted = settings.top_k.value,
                                             .expansion_factor = settings.expansion_search.value,
                                             .stats = stats};
        usearch_sequence_search(&arguments);
    };
    auto const run_with = [&](auto& oracle, status_t prepared, double unit) {
        if (failed(prepared)) return fmt::println(stderr, "The exact scan could not size its rounds"), 1;
        return run_basic_bench<usearch_sequence_traversal_stats_t>(
            settings, tape, insert, search, [&](std::size_t query, std::span<double> distances) {
                scan_exactly(oracle, characters, tape.offsets, query, distances, unit);
            });
    };

    int exit_code = 0;
    if (metric == usearch_sequence_metric_overlap_k) {
        sequence::overlap_metric<> oracle;
        exit_code = run_with(oracle, oracle.init(oracle_round_k, window_widths.view()),
                             1.0 / sequence::overlap_metric<>::resolution_k);
    }
    else if (metric == usearch_sequence_metric_levenshtein_utf8_k) {
        sequence::levenshtein_metric<sz_cap_any_k, sz_levenshtein_runes_k> oracle;
        exit_code = run_with(oracle, oracle.init(oracle_round_k), 1);
    }
    else {
        sequence::levenshtein_metric<> oracle;
        exit_code = run_with(oracle, oracle.init(oracle_round_k), 1);
    }
    usearch_sequence_free(index);
    return exit_code;
}

} // namespace unum::usearch::bench

int main(int argc, char** argv) { return us::bench::bench_sequence_ram_index_search(argc, argv); }
