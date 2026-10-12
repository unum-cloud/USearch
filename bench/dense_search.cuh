/**
 *  @file bench/dense_search.cuh
 *  @author Ash Vardanian
 *  @date July 22, 2026
 *  @brief The backend-neutral surface of the search benchmarks, shared by every dense-index residency.
 *
 *  The options, the dataset streaming and the process setup of @c dense.cuh drive the build; this header adds the
 *  telemetry every build and search phase reports, the post-clock judgment of the merged hits, and the lightning
 *  indexer heads a search may rank by, judged against an exact oracle. Like its sibling it names no index, so the
 *  same surface compiles under host compilers and nvcc.
 */
#pragma once

#include <cmath>   // `std::log2` for ranking gains
#include <cstring> // `std::memmove` - the sorted expected keys

#include <algorithm> // `std::nth_element`, `std::partial_sort`
#include <limits>    // `std::numeric_limits` - the oracle's unfilled ranks
#include <span>      // `std::span`
#include <vector>    // `std::vector`

#include <fmt/format.h>
#include <fmt/ranges.h> // `fmt::join`

#include "usearch/dense/types.hpp" // `dense::indexer_heads`

#include "dense.cuh"

namespace unum::usearch::bench {
using namespace unum::usearch::dense;

#pragma region Telemetry and Judgment

/** Telemetry sums each thread accumulates privately, padded away from its neighbors. */
struct alignas(default_alignment_k) telemetry_lane_t {

    /** Frontier candidates whose neighbor lists were expanded. */
    std::size_t visited_members = 0;

    /** Total distance computations, padding lanes excluded. */
    std::size_t computed_distances = 0;

    /** Subset of @c computed_distances spent in diversity selection. */
    std::size_t computed_distances_in_refines = 0;

    /** Subset of @c computed_distances spent re-selecting while back-linking. */
    std::size_t computed_distances_in_reverse_refines = 0;

    /** Starvation restarts - a graph-connectivity health signal. */
    std::size_t restarts = 0;

    /** Walks a scratch refusal ended early on the CPU; the GPU counts its starved visited-set
     *  claims here. */
    std::size_t saturated_walks = 0;
};

/** Sums the per-thread telemetry after a phase; the lanes stay untouched. */
inline telemetry_lane_t summed_lanes(std::vector<telemetry_lane_t> const& lanes) noexcept {
    telemetry_lane_t sums;
    for (telemetry_lane_t const& lane : lanes) {
        sums.visited_members += lane.visited_members;
        sums.computed_distances += lane.computed_distances;
        sums.computed_distances_in_refines += lane.computed_distances_in_refines;
        sums.computed_distances_in_reverse_refines += lane.computed_distances_in_reverse_refines;
        sums.restarts += lane.restarts;
        sums.saturated_walks += lane.saturated_walks;
    }
    return sums;
}

/** The backend-neutral judgment state every backend's phases fill: the merged hits with their distances and
 *  latencies, and the failure tally. */
struct bench_results_t {

    /** The @b [queries,top_k] merged global rows of every query's gathered hits. */
    nk::matrix<u32_t> hit_rows;

    /** The @b [queries,top_k] distances matching @c hit_rows entry-for-entry. */
    nk::matrix<distance_t> hit_distances;

    /** Merged hits per query, sizing each row for the post-clock judgment. */
    std::vector<u32_t> hit_counts;

    /** Per-query end-to-end lookup latencies, in microseconds. */
    std::vector<float> latencies_us;

    /** Tasks that returned an error in either phase; any non-zero fails the run. */
    std::size_t failed_tasks = 0;
};

/** Sizes the per-query hit matrices for @p queries_count queries of @p top_k matches each. */
inline bool reserve_results(bench_results_t& results, std::size_t queries_count, std::size_t top_k) {
    results.hit_rows = nk::matrix<u32_t>::zeros({queries_count, top_k}).value;
    results.hit_distances = nk::matrix<distance_t>::zeros({queries_count, top_k}).value;
    if (results.hit_rows.empty() || results.hit_distances.empty())
        return fmt::println("Failed to allocate the per-query hit matrices"), false;
    results.hit_counts.resize(queries_count);
    return true;
}

/** What the refinement phase measured, whichever backend ran it: the passes, the audit that
 *  followed them, and the traversal telemetry they cost. */
struct refinement_report_t {

    /** Published nodes one pass sweeps. */
    std::size_t nodes = 0;

    /** The candidate pool every pass re-selects under. */
    std::size_t expansion = 0;

    /** Nodes each pass re-wired; its length is the passes that actually ran. */
    std::span<std::size_t const> changed_per_pass;

    /** Threads the passes fanned over; one on a device. */
    std::size_t threads = 0;

    /** Wall clock of the passes alone. */
    double refining_seconds = 0;

    /** Wall clock of the heal and its final audit. */
    double healing_seconds = 0;

    /** Traversal telemetry summed across the passes. */
    telemetry_lane_t sums;

    /** Nodes the heal's first audit found unreachable. */
    std::size_t islands_found = 0;

    /** Islands the heal wired into a reachable pick. */
    std::size_t islands_reconnected = 0;

    /** Nodes a search still reaches once the heal returned. */
    std::size_t reachable_nodes = 0;

    /** Level-synchronous sweeps the final audit took - a diameter proxy. */
    std::size_t sweeps = 0;
};

/** Prints the refinement phase - the passes, their cost, and the reachability the heal left
 *  behind. */
inline void report_refinement_metrics(refinement_report_t const& report) {
    std::size_t const passes = report.changed_per_pass.size();
    fmt::println("\nRefinement");
    fmt::println("  passes: {} · expansion {}", passes, report.expansion);
    fmt::println("  changed: {}", fmt::join(report.changed_per_pass, " → "));
    fmt::println("  threads: {}", report.threads);
    fmt::println("  wall: {:.2f} s refining · {:.2f} s healing", report.refining_seconds, report.healing_seconds);
    fmt::println("  throughput: {} nodes/s", humanized(double(report.nodes * passes) / report.refining_seconds));
    fmt::println("  distances: {}", humanized(report.sums.computed_distances));
    fmt::println("  visited: {}", humanized(report.sums.visited_members));
    fmt::println("  islands: {} found, {} reconnected", report.islands_found, report.islands_reconnected);
    fmt::println("  reachable: {} of {} · {} sweeps", report.reachable_nodes, report.nodes, report.sweeps);
}

/** Prints the search phase's wall clock, throughput, latency percentiles and traversal
 *  telemetry; @p latencies_us reorders in place. */
inline void report_search_metrics(settings_t const& settings, std::size_t query_count, double seconds,
                                  std::size_t threads, telemetry_lane_t const& sums, std::span<float> latencies_us) {
    auto percentile = [&](double share) {
        std::size_t const rank = std::min(query_count - 1, std::size_t(share * double(query_count)));
        std::nth_element(latencies_us.begin(), latencies_us.begin() + rank, latencies_us.end());
        return latencies_us[rank];
    };
    fmt::println("\nQueries");
    fmt::println("  queries: {} at top-{}{}", query_count, settings.top_k.value,
                 settings.self_recall == switch_t::present_k ? " · self-recall" : "");
    fmt::println("  threads: {}", threads);
    fmt::println("  wall: {:.2f} s", seconds);
    fmt::println("  throughput: {} QPS", humanized(double(query_count) / seconds));
    // An empty latency span means a batch-shaped backend with no per-query clock.
    if (latencies_us.empty())
        fmt::println("  latency: {:.0f} µs mean per query, batch-uniform", seconds * 1e6 / double(query_count));
    else
        fmt::println("  latency: p50 {:.0f} µs · p90 {:.0f} µs · p99 {:.0f} µs", percentile(0.50), percentile(0.90),
                     percentile(0.99));
    fmt::println("  distances/query: {}", humanized(double(sums.computed_distances) / double(query_count)));
    fmt::println("  visited/query: {}", humanized(double(sums.visited_members) / double(query_count)));
    fmt::println("  restarts: {}", sums.restarts);
    fmt::println("  saturated: {}", sums.saturated_walks);
}

/**
 *  @brief Precomputes the position-discount tables shared by every query's judgment, so
 *      per-query normalization is one lookup instead of a chain of @c log2 calls.
 *
 *  @param[out] discounts Per-position pay-offs: @c discounts[i] rewards a hit returned at position i.
 *  @param[out] ideal Prefix sums of @p discounts: @c ideal[i] is the gain of a perfect ranking
 *      of length i + 1.
 */
inline void fill_discounts(std::span<double> discounts, std::span<double> ideal) noexcept {
    double running = 0;
    for (std::size_t position = 0; position != discounts.size(); ++position) {
        discounts[position] = 1.0 / std::log2(double(position) + 2.0);
        running += discounts[position];
        if (position < ideal.size()) ideal[position] = running;
    }
}

/** The verdicts of one query's judgment, produced together by @c judge_ranking. */
struct ranking_verdict_t {

    /** 1 when the true nearest neighbor appears among the returned keys. Averaged
     *  over queries this is the production Recall@K - the probability of finding @b the
     *  nearest neighbor within the top K - monotone in K, so Recall@1 never exceeds
     *  Recall@10. */
    std::size_t found_nearest = 0;

    /** The intersection size between the returned and expected sets - "overlap",
     *  the set-coverage metric big-ANN leaderboards report. Stricter than Recall@K and
     *  not monotone against it. */
    std::size_t overlap = 0;

    /** Unnormalized discounted cumulative gain: every returned key present in the
     *  expected set earns its position's discount, so early hits weigh more. */
    double gain = 0;
};

/**
 *  @brief All ranking verdicts for one query in one sweep: the expected keys are sorted once,
 *      then every returned position pays one binary search - O(k·log k) where the naive
 *      nested probes go O(k²), which matters from top-100 recalls up.
 *
 *  @param[in] returned Keys the index produced, best first.
 *  @param[in] expected Ground-truth keys, ascending by true distance - @c expected[0] is the
 *      true nearest neighbor.
 *  @param[inout] scratch Clobbered workspace for sorting the expected keys; its length caps
 *      how many of them are judged.
 *  @param[in] discounts Position pay-offs from @c fill_discounts.
 */
template <typename key_type_>
ranking_verdict_t judge_ranking(std::span<key_type_ const> returned, std::span<key_type_ const> expected,
                                std::span<key_type_> scratch, std::span<double const> discounts) noexcept {
    std::size_t const expected_count = expected.size() < scratch.size() ? expected.size() : scratch.size();
    for (std::size_t sorted = 0; sorted != expected_count; ++sorted) {
        key_type_ const key = expected[sorted];
        key_type_* const slot = lower_bound(scratch.subspan(0, sorted), key);
        std::memmove(slot + 1, slot, (scratch.data() + sorted - slot) * sizeof(key_type_));
        *slot = key;
    }

    ranking_verdict_t verdict;
    if (!expected_count) return verdict;
    key_type_ const expected_nearest = expected.front();
    std::span<key_type_ const> const sorted {scratch.data(), expected_count};
    for (std::size_t position = 0; position != returned.size(); ++position) {
        key_type_ const key = returned[position];
        verdict.found_nearest |= static_cast<std::size_t>(key == expected_nearest);
        key_type_ const* const slot = lower_bound(sorted, key);
        if (slot != sorted.data() + sorted.size() && *slot == key) {
            verdict.overlap += 1;
            if (position < discounts.size()) verdict.gain += discounts[position];
        }
    }
    return verdict;
}

/** Judges the merged hits after the clocks stopped - self-recall against own rows, or
 *  recall/overlap/ndcg against the ground truth - one sweep per query either way. */
inline void judge_hits(settings_t const& settings, bench_results_t const& results, nk::matrix_view<u32_t> ground_truth,
                       std::size_t query_count) {
    if (settings.self_recall == switch_t::present_k) {
        std::size_t found_at_1 = 0, found_at_k = 0;
        for (std::size_t query = 0; query != query_count; ++query) {
            auto const returned = to_span(results.hit_rows.row(query).as_vector()).first(results.hit_counts[query]);
            u32_t const own_row = static_cast<u32_t>(query);
            found_at_1 += !returned.empty() && returned.front() == own_row;
            bool within = false;
            for (u32_t row : returned) within |= row == own_row;
            found_at_k += within;
        }
        double const queries = double(query_count);
        double const self_at_1 = 100.0 * double(found_at_1) / queries;
        double const self_at_k = 100.0 * double(found_at_k) / queries;
        fmt::println("  self-recall@1: {:.2f}%", self_at_1);
        fmt::println("  self-recall@{}: {:.2f}%", settings.top_k.value, self_at_k);
    }
    else if (!ground_truth.empty()) {
        std::size_t const judged_depth = std::min(settings.top_k.value, ground_truth.extent(1));
        std::vector<u32_t> scratch(judged_depth);
        std::vector<double> discounts(settings.top_k.value), ideal(settings.top_k.value);
        fill_discounts(discounts, ideal);

        std::size_t found_at_1 = 0, found_at_k = 0, shared = 0;
        double gain = 0;
        for (std::size_t query = 0; query != query_count; ++query) {
            auto expected = ground_truth.row(query).as_vector();
            auto const returned = to_span(results.hit_rows.row(query).as_vector()).first(results.hit_counts[query]);
            found_at_1 += !returned.empty() && returned.front() == expected[0];
            auto verdict = judge_ranking(std::span<u32_t const> {returned}, to_span(expected).first(judged_depth),
                                         std::span<u32_t> {scratch}, std::span<double const> {discounts});
            found_at_k += verdict.found_nearest;
            shared += verdict.overlap;
            gain += verdict.gain;
        }

        double const queries = double(query_count);
        double const recall_at_1 = 100.0 * double(found_at_1) / queries;
        double const recall_at_k = 100.0 * double(found_at_k) / queries;
        double const overlap_at_k = 100.0 * double(shared) / (queries * double(judged_depth));
        double const gain_at_k = 100.0 * gain / (queries * ideal[judged_depth - 1]);
        fmt::println("  recall@1: {:.3f}%", recall_at_1);
        fmt::println("  recall@{}: {:.3f}%", judged_depth, recall_at_k);
        fmt::println("  overlap@{}: {:.3f}%", judged_depth, overlap_at_k);
        fmt::println("  ndcg@{}: {:.3f}%", judged_depth, gain_at_k);
    }
    else { fmt::println("  recall: skipped, no ground-truth neighbors provided"); }
}

#pragma endregion

#pragma region Indexer Heads

/** A lightning indexer's heads as `--heads` and `--weights` name them, landed in the stored scalar @p scalar_type_
 *  beside the rows its walks navigate by. */
template <typename scalar_type_>
struct bench_indexer_heads {

    /** The heads widened to @c f32, @b [queries·heads,dimensions], and their signed weights, @b [queries,heads]. */
    nk::matrix<f32_t> widened_heads, weights;

    /** The heads in the stored scalar, and without `--queries` every query's `Σₕ wₕ·qₕ` navigation row. */
    std::vector<scalar_type_> heads, navigation;

    bool engaged() const noexcept { return weights.extent(0) != 0; }
    std::size_t count() const noexcept { return weights.extent(1); }

    /** The stored heads of queries @b [first,first+queries), as the search verbs take them. */
    indexer_heads<scalar_type_> rows(std::size_t first, std::size_t queries) const noexcept {
        nk::matrix_view<scalar_type_> const stored {heads.data(), widened_heads.extent(0), widened_heads.extent(1)};
        return {stored(nk::range(first * count(), (first + queries) * count()), nk::slice),
                weights(nk::range(first, first + queries), nk::slice)};
    }
};

/** Queries a search phase answers: the searched rows, or one per `--weights` row under indexer heads. */
template <typename streamed_dataset_type_, typename scalar_type_>
std::size_t searched_count(streamed_dataset_type_ const& streamed,
                           bench_indexer_heads<scalar_type_> const& heads) noexcept {
    return heads.engaged() ? heads.weights.extent(0) : searched_rows(streamed).extent(0);
}

/** Whether @p index ranks by indexer heads, as its `search` overload requires: dot-kernel
 *  floating-point keys. */
template <typename index_type_>
bool scores_indexer(index_type_ const& index) noexcept {
    return index.kernel() == nk_kernel_dot_k && !nk::is_integral_dtype<typename index_type_::scalar_t>();
}

/** Reads the RetriEval matrix @p path through the opener of `--vectors`, in whichever scalar its extension names,
 *  widened into @p widened. */
inline bool read_widened_rows(std::string_view path, nk::matrix<f32_t>& widened) {
    return open_dataset(path, {}, {},
                        [&]<typename file_scalar_type_>(utilities::persisted_dataset<file_scalar_type_> const& file) {
                            nk::matrix_view<file_scalar_type_> const rows = file.vectors();
                            widened = nk::matrix<f32_t>::zeros({rows.extent(0), rows.extent(1)}).value;
                            if (widened.empty()) return fmt::println("Failed to widen {} into f32", path), 1;
                            for (std::size_t row = 0; row != rows.extent(0); ++row)
                                widen_row(rows.row(row).as_vector(),
                                          std::span<f32_t> {widened.row(row).data(), rows.extent(1)});
                            return 0;
                        }) == 0;
}

/** Opens `--heads` and `--weights` over the searched queries of @p streamed as @p index ranks them,
 *  casting the heads, and without `--queries` their `Σₕ wₕ·qₕ` navigation rows, into the stored
 *  scalar under the staging scale, so the walks score against the same keys; without `--heads` it
 *  opens nothing. */
template <typename index_type_, typename streamed_dataset_type_>
bool open_indexer_heads(index_type_ const& index, streamed_dataset_type_ const& streamed,
                        bench_indexer_heads<typename index_type_::scalar_t>& heads) {
    using scalar_t = typename index_type_::scalar_t;
    settings_t const& settings = streamed.bench.settings;
    if (settings.heads_path.empty() && settings.weights_path.empty()) return true;
    if (!scores_indexer(index)) return fmt::println("--heads ranks floating-point keys under --metric dot"), false;
    if (settings.self_recall == switch_t::present_k)
        return fmt::println("--heads brings its own queries, so --self-recall has none to judge"), false;
    if (!read_widened_rows(settings.heads_path, heads.widened_heads) ||
        !read_widened_rows(settings.weights_path, heads.weights))
        return false;
    std::size_t const dimensions = streamed.queries.extent(1), queries_count = heads.weights.extent(0);
    if (heads.widened_heads.extent(1) != dimensions || heads.widened_heads.extent(0) != queries_count * heads.count())
        return fmt::println("--heads must hold {} rows of {}d per --weights row", heads.count(), dimensions), false;
    if (!settings.queries_path.empty() && streamed.queries.extent(0) < queries_count)
        return fmt::println("--queries must hold a row per --weights row"), false;
    if (settings.queries_path.empty()) {
        std::vector<f32_t> navigation(queries_count * dimensions, 0);
        for (std::size_t query = 0; query != queries_count; ++query)
            for (std::size_t head = 0; head != heads.count(); ++head)
                for (std::size_t dimension = 0; dimension != dimensions; ++dimension)
                    navigation[query * dimensions + dimension] +=
                        heads.weights(query, head) * heads.widened_heads(query * heads.count() + head, dimension);
        heads.navigation = stage_widened<scalar_t>(std::move(navigation), streamed.scale);
    }
    f32_t const* const widened = heads.widened_heads.data();
    heads.heads = stage_widened<scalar_t>(
        std::vector<f32_t>(widened, widened + heads.widened_heads.extent(0) * dimensions), streamed.scale);
    return true;
}

/** One query's exact scorer, in @c f64: the dot of its row and a key, or `Σₕ wₕ·ReLU(qₕ·k)` over its heads. */
struct exact_scorer_t {

    /** The query's row, or its heads back to back. */
    std::vector<f64_t> rows;

    /** The query's head weights; empty scores the plain dot. */
    std::span<f32_t const> weights;

    /** Per-key scratch, one row wide. */
    std::vector<f32_t> widened;

    /** The exact score of @p key. */
    template <typename dataset_scalar_type_>
    f64_t operator()(nk::vector_view<dataset_scalar_type_> key) noexcept {
        std::size_t const dimensions = widened.size();
        widen_row(key, std::span {widened});
        auto dot = [&](std::size_t offset) noexcept {
            f64_t sum = 0;
            for (std::size_t dimension = 0; dimension != dimensions; ++dimension)
                sum += rows[offset + dimension] * widened[dimension];
            return sum;
        };
        if (weights.empty()) return dot(0);
        f64_t score = 0;
        for (std::size_t head = 0; head != weights.size(); ++head)
            score += weights[head] * std::max(dot(head * dimensions), f64_t {});
        return score;
    }
};

/** The exact scorer of one plain @p query row. */
template <typename dataset_scalar_type_>
exact_scorer_t exact_scorer_of(nk::vector_view<dataset_scalar_type_> query) {
    exact_scorer_t scorer {{}, {}, std::vector<f32_t>(query.size())};
    widen_row(query, std::span {scorer.widened});
    scorer.rows.assign(scorer.widened.begin(), scorer.widened.end());
    return scorer;
}

/** The exact scorer of query @p query under its indexer @p heads. */
template <typename scalar_type_>
exact_scorer_t exact_scorer_of(bench_indexer_heads<scalar_type_> const& heads, std::size_t query) {
    std::size_t const dimensions = heads.widened_heads.extent(1);
    f32_t const* const first = heads.widened_heads.row(query * heads.count()).data();
    return {std::vector<f64_t>(first, first + heads.count() * dimensions),
            {heads.weights.row(query).data(), heads.count()},
            std::vector<f32_t>(dimensions)};
}

/** One exactly scored row, ranked best first by score. */
struct exact_row_t {
    f64_t score;
    u32_t row;
};

/** The best @p top_k rows of @p vectors under @p scorer, best first. */
template <typename dataset_scalar_type_>
std::vector<exact_row_t> exact_best_rows(exact_scorer_t& scorer, nk::matrix_view<dataset_scalar_type_> vectors,
                                         std::size_t top_k) {
    std::vector<exact_row_t> best(vectors.extent(0));
    for (std::size_t row = 0; row != best.size(); ++row)
        best[row] = {scorer(vectors.row(row).as_vector()), static_cast<u32_t>(row)};
    std::size_t const kept = std::min(top_k, best.size());
    std::partial_sort(best.begin(), best.begin() + kept, best.end(),
                      [](exact_row_t a, exact_row_t b) noexcept { return a.score > b.score; });
    best.resize(kept);
    return best;
}

/** Judges the hits against `--neighbors`, or an indexer run without them against the exact oracle over its first
 *  `--oracle-queries`. */
template <typename scalar_type_, typename dataset_scalar_type_>
void judge_search(settings_t const& settings, bench_indexer_heads<scalar_type_> const& heads,
                  bench_results_t const& results, nk::matrix_view<u32_t> ground_truth,
                  nk::matrix_view<dataset_scalar_type_> vectors, std::size_t query_count) {
    if (!ground_truth.empty() || !heads.engaged()) return judge_hits(settings, results, ground_truth, query_count);
    std::size_t const judged = std::min(settings.oracle_queries.value, query_count);
    std::size_t const top_k = settings.top_k.value;
    std::vector<u32_t> truth(judged * top_k, std::numeric_limits<u32_t>::max());
    for (std::size_t query = 0; query != judged; ++query) {
        exact_scorer_t scorer = exact_scorer_of(heads, query);
        std::vector<exact_row_t> const best = exact_best_rows(scorer, vectors, top_k);
        for (std::size_t rank = 0; rank != best.size(); ++rank) truth[query * top_k + rank] = best[rank].row;
    }
    fmt::println("  oracle: exact top-{} of {} queries", top_k, judged);
    judge_hits(settings, results, nk::matrix_view<u32_t> {truth.data(), judged, top_k}, judged);
}

#pragma endregion

} // namespace unum::usearch::bench
