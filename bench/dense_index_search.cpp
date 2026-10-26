/**
 *  @file bench/dense_index_search.cpp
 *  @author Ash Vardanian
 *  @date October 25, 2026
 *  @brief The dense search benchmark over the C interface: a RetriEval dataset built into one index
 *      of any residency, pinned to any CPU or GPU class, searched at every breadth and judged
 *      against ground truth, against itself, or against an exact scan.
 *
 *  The index runs as users run it, through a session of the residency `--residency` names. An NVMe
 *  run builds in RAM, saves to `--checkpoint` or the temporary directory, and searches a view of
 *  the checkpoint on storage threads submitting through `--nvme-class`. Every row is keyed by its
 *  global dataset row, so slices built apart keep distinct keys: `--base-vectors` takes a glob of
 *  parts numbering their rows on across each other, and `--first-vector` with `--max-base-vectors`
 *  picks a slice that must lie inside one part, viewed in place. Ground truth names global rows,
 *  and is ignored once a slice reshapes the graph it was computed for.
 *
 *  @verbatim
 *  Flag                Default                      Meaning
 *  --base-vectors      required                     Base vectors, a RetriEval file or a glob of its parts
 *  --query-vectors     the first 1000 base vectors  Query vectors of the same dimensions
 *  --query-neighbors   none                         Ground truth, an .ibin of global base rows per query
 *  --stored-dtype      the file's, else bf16        Stored scalar, cast on ingest
 *  --metric            angular                      Distance kind: inner, angular, euclidean, hamming or jaccard
 *  --connectivity      derived                      Graph degree
 *  --expansion-add     derived                      Insertion breadth
 *  --expansion-search  96                           Search breadths, like 32,64,128
 *  --max-refines       0                            Refinement passes, stopping once one changes nothing
 *  --compact-capacity  none                         Entries per shard after compacting in place
 *  --top-k             10                           Matches per query to judge
 *  --first-vector      0                            First dataset row viewed, keyed by its global row
 *  --max-base-vectors  the whole file               Cap on ingested rows
 *  --self-search       off                          Query every stored row for itself
 *  --oracle-queries    256                          Queries the exact scan judges without ground truth
 *  --batch             65536                        Rows per insert or search call
 *  @endverbatim
 */
#include <cstdio> // `std::remove` - the checkpoint an NVMe run saved

#include <algorithm>  // `std::min`
#include <chrono>     // `std::chrono::steady_clock` - the phase clocks
#include <filesystem> // `std::filesystem::temp_directory_path` - where an NVMe run saves
#include <string>     // `std::string` - the checkpoint path

#include <fmt/format.h>

#include "abi.hpp"
#include "bench.hpp" // `humanized`, `progress_bar_t`

namespace unum::usearch::bench {
using namespace unum::usearch::tools;

/** Every flag this binary reads beyond the shared settings, its default as the initializer. */
struct search_settings_t {
    std::string_view base_vectors, query_vectors, query_neighbors, stored_dtype_name, metric_name = "angular";
    count_t connectivity {.minimum = 0};
    count_t expansion_add {.minimum = 0};
    counts_t expansion_search {.values = {96}, .size = 1};
    count_t max_refines {.minimum = 0};
    count_t compact_capacity {.minimum = 0};
    count_t top_k {.value = 10};
    count_t first_vector {.minimum = 0};
    count_t max_base_vectors {.minimum = 0};
    switch_t self_search = switch_t::absent_k;
    count_t oracle_queries {.value = 256};
    count_t batch {.value = 65536};
};

/** The flags @ref search_settings_t holds. */
inline std::array<flag_t, 16> search_flags(search_settings_t& search) noexcept {
    return {{
        {"--base-vectors", "path", "base vectors, a RetriEval file or a glob of its parts", &search.base_vectors},
        {"--query-vectors", "path", "query vectors; the first 1000 base vectors by default", &search.query_vectors},
        {"--query-neighbors", "path", "ground truth, an .ibin of global base rows per query", &search.query_neighbors},
        {"--stored-dtype", "name", "stored scalar, cast on ingest; the file's, else bf16, by default",
         &search.stored_dtype_name},
        {"--metric", "name", "inner, angular, euclidean, hamming or jaccard; angular by default", &search.metric_name},
        {"--connectivity", "degree", "graph degree; derived by default", &search.connectivity},
        {"--expansion-add", "breadth", "insertion breadth; derived by default", &search.expansion_add},
        {"--expansion-search", "breadths", "search breadths, like 32,64,128; 96 by default", &search.expansion_search},
        {"--max-refines", "passes", "refinement passes, stopping once one changes nothing; 0 by default",
         &search.max_refines},
        {"--compact-capacity", "entries", "entries per shard after compacting in place; no compaction by default",
         &search.compact_capacity},
        {"--top-k", "count", "matches per query to judge; 10 by default", &search.top_k},
        {"--first-vector", "row", "first dataset row viewed, keyed by its global row; 0 by default",
         &search.first_vector},
        {"--max-base-vectors", "count", "cap on ingested rows; the whole file by default", &search.max_base_vectors},
        {"--self-search", "", "query every stored row for itself", &search.self_search},
        {"--oracle-queries", "count", "queries the exact scan judges without ground truth; 256 by default",
         &search.oracle_queries},
        {"--batch", "rows", "rows per insert or search call; 65536 by default", &search.batch},
    }};
}

/** Seconds since @p started. */
inline double seconds_since(std::chrono::steady_clock::time_point started) noexcept {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
}

/** Refines @p index up to @p passes times, stopping once a pass changes nothing; false after
 *  printing a failure. */
inline bool refine_passes(handle<dense_verbs_t>& index, std::size_t passes) {
    for (std::size_t pass = 0; pass != passes; ++pass) {
        std::chrono::steady_clock::time_point const started = std::chrono::steady_clock::now();
        std::size_t changed = 0;
        usearch_outcome_t outcome {};
        dense_verbs_t::refine(index.get(), index.executor(), &changed, &outcome);
        if (report_failure(outcome, "Refining")) return false;
        fmt::println("- Refine pass {}: {} changed in {:.2f} s", pass + 1, humanized(changed), seconds_since(started));
        if (!changed) break;
    }
    return true;
}

/** Compacts @p index in place under @p capacity entries per shard; false after printing a
 *  failure. */
inline bool compact_in_place(handle<dense_verbs_t>& index, std::size_t capacity) {
    std::chrono::steady_clock::time_point const started = std::chrono::steady_clock::now();
    usearch_outcome_t outcome {};
    usearch_traversal_stats_t stats {};
    usearch_dense_index_t sources[1] = {index.get()};
    usearch_dense_compact_t compact {};
    compact.target = index.get(), compact.executor = index.executor(), compact.outcome = &outcome;
    compact.count = 1, compact.sources = sources, compact.shard_capacity = capacity, compact.stats = &stats;
    usearch_dense_compact(&compact);
    if (report_failure(outcome, "Compacting")) return false;
    fmt::println("- Compact: {} distances in {:.2f} s", humanized(stats.computed_distances), seconds_since(started));
    return true;
}

/** Runs the dense search benchmark over the flags in @p argv. */
int bench_dense_index_search(int argc, char const* const* argv) {
    search_settings_t search;
    std::array<flag_t, 16> const flags = search_flags(search);
    settings_t const settings = read_settings(argc, argv,
                                              "usearch_bench_dense_index_search --base-vectors <path> [flags]", flags);
    print(settings);
    fmt::println("- Version: {} · compiled for {} · this machine {}", usearch_version_name(),
                 usearch_capabilities_compiled(), usearch_capabilities_available());

    auto [base, base_status] = mapped_rows_t::open(search.base_vectors);
    if (failed(base_status) || base.scalar() == usearch_scalar_unknown_k)
        return fmt::println(stderr, "--base-vectors=\"{}\" maps no RetriEval vectors", search.base_vectors), 1;
    std::size_t const first_row = search.first_vector.value;
    if (first_row >= base.rows())
        return fmt::println(stderr, "--first-vector {} is past the {} rows", first_row, base.rows()), 1;
    std::size_t const rows = search.max_base_vectors.value
                                 ? std::min(search.max_base_vectors.value, base.rows() - first_row)
                                 : base.rows() - first_row;
    usearch_rows_view_t const vectors = base.view(first_row, rows);
    if (!vectors.data)
        return fmt::println(stderr, "--first-vector {} and --max-base-vectors {} span two parts", first_row, rows), 1;

    usearch_scalar_t stored = base.scalar() == usearch_scalar_f32_k || base.scalar() == usearch_scalar_f64_k
                                  ? usearch_scalar_bf16_k
                                  : base.scalar();
    if (!search.stored_dtype_name.empty()) stored = usearch_scalar_from_name(bytes_view(search.stored_dtype_name));
    usearch_dense_metric_t const metric = usearch_dense_metric_from_name(bytes_view(search.metric_name));
    if (stored == usearch_scalar_unknown_k || metric == usearch_dense_metric_unknown_k)
        return fmt::println(stderr, "--stored-dtype=\"{}\" or --metric=\"{}\" names nothing", search.stored_dtype_name,
                            search.metric_name),
               1;
    fmt::println("- Dataset: {} rows from {} · {} columns of {} · stored as {} · {}", humanized(rows), first_row,
                 base.columns(), usearch_scalar_name(base.scalar()), usearch_scalar_name(stored),
                 usearch_dense_metric_name(metric));

    session_t session = open_session(settings, settings.cpu_class);
    if (!session) return 1;
    usearch_dense_metadata_t const geometry {metric, stored, base.columns(), search.connectivity.value};
    handle<dense_verbs_t> index = open_index<dense_verbs_t>(session, geometry, settings.shards.value);
    if (!index) return 1;

    usearch_traversal_stats_t built {};
    progress_bar_t building {"build", rows, settings.terminal};
    std::chrono::steady_clock::time_point const build_start = std::chrono::steady_clock::now();
    usearch_outcome_t const inserted = insert_entries(index, vectors, rows, first_row, search.batch.value,
                                                      search.expansion_add.value, built,
                                                      [&](std::size_t landed) { building.tick(landed); });
    building.finish();
    if (report_failure(inserted, "Inserting")) return 1;
    double const build_seconds = seconds_since(build_start);
    fmt::println("- Build: {} rows in {:.2f} s · {}/s · {:.0f} distances per row", humanized(rows), build_seconds,
                 humanized(rows / build_seconds),
                 static_cast<double>(built.computed_distances) / static_cast<double>(rows));
    if (!refine_passes(index, search.max_refines.value)) return 1;
    if (search.compact_capacity.value && !compact_in_place(index, search.compact_capacity.value)) return 1;
    print_fill(index);

    // An NVMe run serves a view of what it built, from the checkpoint it saves.
    std::string const checkpoint =
        !settings.checkpoint.empty() ? std::string(settings.checkpoint)
        : settings.residency == usearch_residency_nvme_k
            ? (std::filesystem::temp_directory_path() / "usearch-bench-dense.safegraphs").string()
            : std::string();
    if (!checkpoint.empty()) {
        std::chrono::steady_clock::time_point const reopen_start = std::chrono::steady_clock::now();
        index = save_and_reopen(std::move(index), checkpoint, settings.residency);
        if (!index) return 1;
        fmt::println("- Reopened {} as {} in {:.2f} s", checkpoint, usearch_residency_name(settings.residency),
                     seconds_since(reopen_start));
        print_fill(index);
    }

    mapped_rows_t query_file;
    usearch_rows_view_t queries = vectors;
    std::size_t queries_count = std::min<std::size_t>(rows, 1000);
    if (search.self_search == switch_t::present_k) queries_count = rows;
    else if (!search.query_vectors.empty()) {
        auto [opened, query_status] = mapped_rows_t::open(search.query_vectors);
        if (failed(query_status) || opened.columns() != base.columns())
            return fmt::println(stderr, "--query-vectors=\"{}\" maps no queries of {} columns", search.query_vectors,
                                base.columns()),
                   1;
        query_file = std::move(opened);
        queries = query_file.view(0, query_file.rows()), queries_count = query_file.rows();
    }

    landed_matches_t oracle;
    std::size_t const top_k = search.top_k.value;
    if (search.self_search == switch_t::present_k) {
        // Every stored row is its own nearest neighbor.
        oracle.counts.assign(rows, 1);
        for (std::size_t row = 0; row != rows; ++row)
            oracle.keys.push_back(std::bit_cast<uuid_t>(key_of_row(settings.seed, first_row + row)));
    }
    else if (!search.query_neighbors.empty() && !first_row && !search.max_base_vectors.value) {
        auto [truth, truth_status] = mapped_rows_t::open(search.query_neighbors);
        if (failed(truth_status) || truth.rows() < queries_count)
            return fmt::println(stderr, "--query-neighbors=\"{}\" holds no row per query", search.query_neighbors), 1;
        usearch_rows_view_t const truth_rows = truth.view(0, queries_count);
        oracle = truth_oracle(settings.seed,
                              {static_cast<std::uint32_t const*>(truth_rows.data), queries_count * truth.columns()},
                              truth.columns());
    }
    else {
        if (!search.query_neighbors.empty())
            fmt::println("- Ground truth ignored: --first-vector or --max-base-vectors reshapes its graph");
        std::chrono::steady_clock::time_point const scan_start = std::chrono::steady_clock::now();
        std::size_t const judged = std::min(search.oracle_queries.value, queries_count);
        if (report_failure(scan_oracle(index, queries, judged, top_k, search.batch.value, oracle), "Scanning"))
            return 1;
        fmt::println("- Exact scan: {} queries in {:.2f} s", judged, seconds_since(scan_start));
    }
    usearch_outcome_t const swept = sweep_recall(index, queries, queries_count, oracle, search.expansion_search.view(),
                                                 top_k, search.batch.value);
    if (report_failure(swept, "Searching")) return 1;
    if (settings.residency == usearch_residency_nvme_k && settings.checkpoint.empty())
        for (std::string const& file : files_starting_with(checkpoint.substr(0, checkpoint.rfind(".safegraphs"))))
            std::remove(file.c_str());
    return 0;
}

} // namespace unum::usearch::bench

int main(int argc, char** argv) { return us::bench::bench_dense_index_search(argc, argv); }
