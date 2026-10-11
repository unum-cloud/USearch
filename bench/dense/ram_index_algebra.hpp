/**
 *  @file bench/dense/ram_index_algebra.hpp
 *  @author Ash Vardanian
 *  @date September 27, 2026
 *  @brief Profiling harness for graph products and graph attention over the CPU-resident @c dense_ram_index,
 *      shared by the per-dtype algebra shards.
 *
 *  The build phases and the compile-time cascade are the search harness's, from @c dense/ram_index_search.hpp; this
 *  header swaps its search phase for @c multiply and @c attend, pinned to the dot kernel both are defined over. Every
 *  query row multiplies, then every `--head-count` consecutive rows attend as one token's heads over every key their
 *  shared walk scores, and the first `--oracle-queries` rows are judged against the exact attention oracle of
 *  @c dense_algebra.cuh:
 *
 *  @code{.sh}
 *  usearch_bench_dense_ram_index_algebra --vectors data/wiki-1m/base.1M.fbin \
 *  --queries data/wiki-1m/query.public.100K.fbin --stored-dtype bf16 --oracle-queries 16
 *  @endcode
 */
#pragma once

#include <chrono> // `std::chrono::steady_clock`
#include <span>   // `std::span`
#include <vector> // `std::vector`

#include <fmt/format.h>

#include "dense/ram_index_search.hpp"
#include "dense_algebra.cuh"

namespace unum::usearch::bench::dense {

#pragma region Pipeline

/** Runs @p body over @p count tasks on @p pool, handing it each task and the thread running it. */
template <typename body_type_>
void for_each_task(fu_pool_t pool, std::size_t count, body_type_& body) noexcept {
    fu_pool_for_n(
        pool, count,
        [](fu_lambda_context_t punned, std::size_t task, std::size_t thread, std::size_t) noexcept {
            (*static_cast<body_type_*>(punned))(task, thread);
        },
        &body);
}

/** Both phases over every query: graph products, then attention of each token's `--head-count` heads over every
 *  key their shared walk scores, judged against the exact oracle on the first `--oracle-queries` rows. */
template <typename streamed_dataset_type_>
bool run_algebra(streamed_dataset_type_& streamed, fu_pool_t pool, algebra_settings_t const& algebra) {
    using bench_t = typename streamed_dataset_type_::bench_t;
    using index_t = typename bench_t::index_t;
    using dot_result_t = typename index_t::dot_result_t;
    auto& bench = streamed.bench;
    settings_t const& settings = bench.settings;
    std::size_t const nodes_count = bench.index->size(), head_count = algebra.head_count.value;
    algebra_inputs<typename bench_t::scalar_t> inputs;
    if (!open_algebra_inputs(streamed, algebra, {streamed.id_to_row.data(), nodes_count}, inputs)) return false;
    std::size_t const query_count = inputs.rows.extent(0), columns = inputs.value_rows.extent(1),
                      judged = inputs.judged;
    std::size_t threads = 0;
    [[maybe_unused]] fu_status_t const counted = fu_pool_threads_count(pool, &threads);

    typename index_t::search_config_t config;
    config.expansion_factor = static_cast<typename bench_t::small_size_t>(settings.expansion_search.value);
    config.synchronization = synchronization_t::immutable_k;
    // A pair dump keeps every scored pair, not just the breadth's worth.
    std::size_t const wanted = algebra.pairs_path.empty() ? settings.expansion_search.value : std::size_t {1} << 18;
    std::vector<dense_node_id_t> node_ids(threads * wanted);
    std::vector<dot_result_t> products(threads * wanted);
    std::vector<std::vector<scored_pair_t>> pairs(algebra.pairs_path.empty() ? 0 : judged);
    std::vector<f32_t> outputs(query_count * columns);
    std::vector<dense_softmax_state_t> carry(threads * head_count);
    auto const values = inputs.node_values_view();
    auto const value_of = [&](std::span<dense_node_id_t const> nodes,
                              std::span<typename index_t::vector_view_t> rows) noexcept {
        for (std::size_t lane = 0; lane != nodes.size(); ++lane) rows[lane] = values.row(nodes[lane]).as_vector();
    };
    auto refuse = [&]() noexcept {
        standard_atomic_ref<std::size_t>(bench.results.failed_tasks).fetch_add(1, std::memory_order_relaxed);
    };

    auto multiply_row = [&](std::size_t query, std::size_t thread) noexcept {
        std::span const kept_ids {node_ids.data() + thread * wanted, wanted};
        std::span const kept_products {products.data() + thread * wanted, wanted};
        auto const [result, status] = bench.index->multiply(inputs.rows.row(query).as_vector(), kept_ids, kept_products,
                                                            *bench.traversals[thread], config);
        if (failed(status)) return refuse();
        if (query < pairs.size())
            for (std::size_t pair = 0; pair != min_of(result.count, wanted); ++pair)
                pairs[query].push_back(
                    {streamed.id_to_row[kept_ids[pair]], static_cast<f32_t>(static_cast<f64_t>(kept_products[pair]))});
        bench.lanes[thread].computed_distances += result.computed_distances;
        bench.lanes[thread].visited_members += result.visited_members;
    };
    auto attend_token = [&](std::size_t token, std::size_t thread) noexcept {
        std::size_t const first = token * head_count;
        auto const [result, status] = bench.index->attend(
            inputs.rows(nk::range(first, first + head_count), nk::slice), head_count, value_of, inputs.score_scale,
            std::span {outputs.data() + first * columns, head_count * columns},
            std::span {carry.data() + thread * head_count, head_count}, *bench.traversals[thread], config);
        if (failed(status)) return refuse();
        bench.lanes[thread].computed_distances += result.computed_distances;
        bench.lanes[thread].visited_members += result.visited_members;
    };

    for (telemetry_lane_t& lane : bench.lanes) lane = telemetry_lane_t {};
    auto started = steady_clock_t::now();
    for_each_task(pool, query_count, multiply_row);
    double const multiply_seconds = stdc::duration<double>(steady_clock_t::now() - started).count();
    report_multiply_metrics(query_count, nodes_count, multiply_seconds, summed_lanes(bench.lanes).computed_distances);
    if (!algebra.pairs_path.empty() && !save_scored_pairs(algebra.pairs_path, pairs))
        return fmt::println("Failed to write --pairs {}", algebra.pairs_path), false;

    for (telemetry_lane_t& lane : bench.lanes) lane = telemetry_lane_t {};
    started = steady_clock_t::now();
    for_each_task(pool, query_count / head_count, attend_token);
    double const attend_seconds = stdc::duration<double>(steady_clock_t::now() - started).count();
    if (std::size_t failures = bench.results.failed_tasks; failures)
        return fmt::println("Multiply or attend failed on {} queries", failures), false;
    report_attend_metrics(query_count, head_count, nodes_count, attend_seconds,
                          summed_lanes(bench.lanes).computed_distances);
    return judge_attention(streamed, algebra, inputs, std::span<f32_t> {outputs});
}

/** The algebra workload the cascade ends in: graph products and graph attention over the built index, judged
 *  against the exact oracle. */
struct algebra_workload_t {

    /** Both verbs are defined over the dot kernel alone, so the cascade pins it. */
    static constexpr bool dot_only_k = true;

    /** The algebra knobs, borrowed for the run's lifetime. */
    algebra_settings_t const& algebra;

    template <typename streamed_dataset_type_>
    bool run(streamed_dataset_type_& streamed, fu_pool_t pool) const {
        return run_algebra(streamed, pool, algebra);
    }
};

/** The algebra shard hop: everything below one stored dtype under the dot kernel, so a per-dtype shard
 *  translation unit carries the whole instantiation grid of its dtype. */
template <typename dataset_scalar_type_, nk_dtype_t stored_dtype_>
int run_algebra_with_stored_dtype(bench_launch<dataset_scalar_type_> const& launch, algebra_settings_t const& algebra) {
    return run_with_compute<dataset_scalar_type_, stored_dtype_>(launch, algebra_workload_t {algebra});
}

#pragma endregion

} // namespace unum::usearch::bench::dense
