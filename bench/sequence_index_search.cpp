/**
 *  @file bench/sequence_index_search.cpp
 *  @author Ash Vardanian
 *  @date October 14, 2026
 *  @brief The sequence search benchmark: a text file, one entry per line, built into one index of
 *      any residency through the C interface and judged against an exact scan of its score.
 *
 *  Queries are the file's last entries, kept out of the index, and a hit counts when its distance
 *  is within the exact k-th, so integer edit distances, where many entries tie, are not
 *  undercounted. An NVMe run builds in RAM, saves a checkpoint and searches a view of it.
 *  A metric names what the entries are: proteins under BLOSUM62 with a gap of 7, DNA over A, C, G
 *  and T on either strand under transition 1, transversion 2 and gap 2, or UTF-8 text, whose case
 *  folds before its windows are counted. Costs of the caller's own replace the metric's.
 *
 *  @verbatim
 *  Flag             Default   Meaning
 *  --base-texts     required  Entries, one per line
 *  --metric         required  protein-alignment, dna-overlap or utf8-containment
 *  --costs          none      A file of 256x256 int8 similarities and an int8 gap cost, for the metric's own
 *  --query-targets  none      The base row each held-out query intends, one decimal per line
 *  --matches-output none      A file per search breadth, <path>.<breadth>, each query's matched base rows on a line
 *  @endverbatim
 */
#include <cstdio> // `std::fopen` - the matches output

#include <array>       // `std::array` - the flags
#include <optional>    // `std::optional` - the costs a metric reads
#include <charconv>    // `std::from_chars` - the queries' intended rows
#include <string>      // `std::string` - the matches output, the checkpoint
#include <string_view> // `std::string_view`
#include <vector>      // `std::vector` - the offsets

#include "usearch/sequence_index.h"
#include "usearch/utilities/blocking_files.hpp" // `mapped_file_t`

#include "abi.hpp"

namespace unum::usearch::bench {
using namespace unum::usearch::tools;

/** Runs the sequence search benchmark over the flags in @p argv. */
int bench_sequence_index_search(int argc, char const* const* argv) {
    search_bench_settings_t search;
    std::string_view base_texts_path, costs_path, query_targets_path, matches_output_path;
    count_t sketch_bytes {.minimum = 0};
    std::array<flag_t, 9> const shared = search_bench_flags(search);
    std::vector<flag_t> flags(shared.begin(), shared.end());
    std::array<flag_t, 5> const own {{
        {"--base-texts", "path", "entries, one per line", &base_texts_path},
        {"--costs", "path", "a file of 256x256 int8 similarities and an int8 gap cost, for the metric's own",
         &costs_path},
        {"--query-targets", "path", "the base row each query intends, one per line, judged at rank 1 and top-k",
         &query_targets_path},
        {"--matches-output", "path", "a file per search breadth, <path>.<breadth>, each query's matched base rows",
         &matches_output_path},
        {"--sketch-bytes", "bytes", "a multiple of four in front of each sequence; defaults to the metric's own",
         &sketch_bytes},
    }};
    flags.insert(flags.end(), own.begin(), own.end());
    settings_t const settings = read_settings(argc, argv,
                                              "usearch_bench_sequence_index_search --base-texts <path> [flags]", flags);
    print(settings);

    usearch_sequence_metric_t metric = usearch_sequence_metric_unknown_k;
    if (usearch_sequence_metric_from_name(bytes_view(search.metric_name), &metric) != usearch_status_success_k)
        return fmt::println(stderr,
                            "--metric=\"{}\" does not parse, expected protein-alignment, dna-overlap, " //
                            "utf8-containment",
                            search.metric_name),
               1;
    bool const named = !costs_path.empty();
    std::optional<byte_costs_t> costs;
    if (named && !(costs = read_costs(costs_path))) return 1;
    usearch_sequence_costs_t const costs_view = costs ? costs->view() : usearch_sequence_costs_t {};
    auto [file, read_status] = util::mapped_file_t::read(base_texts_path);
    if (failed(read_status)) return fmt::println(stderr, "--base-texts=\"{}\" could not be read", base_texts_path), 1;
    byte_tape_t const tape = lines_of(file.view());
    if (tape.size() <= search.queries.value)
        return fmt::println(stderr, "The file holds {} entries, too few to hold out {} queries", tape.size(),
                            search.queries.value),
               1;
    std::size_t const query_first = tape.size() - search.queries.value;
    std::size_t const base_count = search.max_base_entries.value
                                       ? util::min_of(search.max_base_entries.value, query_first)
                                       : query_first;
    if (named) fmt::println("- Costs: {}", costs_path);

    // The base row each held-out query intends, judged by key at rank 1 and among the top-k.
    std::vector<util::u64_t> query_targets;
    if (!query_targets_path.empty()) {
        auto [targets_file, targets_status] = util::mapped_file_t::read(query_targets_path);
        if (failed(targets_status))
            return fmt::println(stderr, "--query-targets=\"{}\" could not be read", query_targets_path), 1;
        byte_tape_t const targets = lines_of(targets_file.view());
        for (std::size_t row = 0; row != targets.size(); ++row) {
            util::u64_t target = 0;
            std::from_chars(targets.bytes.data() + targets.offsets[row],
                            targets.bytes.data() + targets.offsets[row + 1], target);
            query_targets.push_back(target);
        }
        if (query_targets.size() != search.queries.value)
            return fmt::println(stderr, "--query-targets holds {} rows for {} queries", query_targets.size(),
                                search.queries.value),
                   1;
    }
    auto const judge_task = [&](std::size_t expansion, copied_matches_t const& found) {
        st::vector<std::size_t> const offsets = match_offsets(found);
        if (!matches_output_path.empty()) {
            std::string rows;
            for (std::size_t query = 0; query + 1 < offsets.size(); ++query) {
                for (std::size_t match = offsets[query]; match != offsets[query + 1]; ++match) {
                    if (match != offsets[query]) rows += ' ';
                    rows += fmt::format("{}",
                                        offset_of_key(settings.seed, std::bit_cast<usearch_key_t>(found.keys[match]))
                                            .value_or(~util::u64_t {0}));
                }
                rows += '\n';
            }
            std::string const path = fmt::format("{}.{}", matches_output_path, expansion);
            std::FILE* const file = std::fopen(path.c_str(), "wb");
            if (!file) fmt::println(stderr, "--matches-output=\"{}\" could not be written", path);
            else std::fwrite(rows.data(), 1, rows.size(), file), std::fclose(file);
        }
        if (query_targets.empty()) return;
        std::size_t task_first = 0, task_kept = 0;
        for (std::size_t query = 0; query != query_targets.size(); ++query)
            for (std::size_t match = offsets[query]; match != offsets[query + 1]; ++match)
                if (offset_of_key(settings.seed, std::bit_cast<usearch_key_t>(found.keys[match])) ==
                    query_targets[query])
                    task_first += match == offsets[query], task_kept += 1;
        fmt::println("- Task at expansion {}: intended entry at rank 1 {:.4f} · in the top {} {:.4f} over {} queries",
                     expansion, static_cast<double>(task_first) / query_targets.size(), search.top_k.value,
                     static_cast<double>(task_kept) / query_targets.size(), query_targets.size());
    };

    sequence_geometry_t const geometry {{metric, search.connectivity.value, 0, sketch_bytes.value},
                                        named ? &costs_view : nullptr};
    usearch_tape_view_t const entries = tape.view();
    return run_search_bench<sequence_verbs_t>(settings, search, geometry, geometry, entries, base_count,
                                              sequence_verbs_t::slice(entries, query_first), search.queries.value,
                                              settings.checkpoint, judge_task);
}

} // namespace unum::usearch::bench

namespace us = unum::usearch;

int main(int argc, char** argv) { return us::bench::bench_sequence_index_search(argc, argv); }
