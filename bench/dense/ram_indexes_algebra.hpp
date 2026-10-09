/**
 *  @file bench/dense/ram_indexes_algebra.hpp
 *  @author Ash Vardanian
 *  @date October 9, 2026
 *  @brief Profiling harness for graph products and graph attention over the sharded @c dense_ram_indexes, shared
 *      by the per-dtype algebra shards.
 *
 *  The build phases, the compile-time cascade and the process setup are the sharded search harness's, from
 *  @c dense/ram_indexes_search.hpp; this header swaps its search phase for the batched @c multiply and @c attend
 *  through the executor, pinned to the dot kernel both are defined over. Every token's walks cross every shard, and
 *  the first `--oracle-queries` rows are judged against the exact attention oracle of @c dense_algebra.cuh.
 */
#pragma once

#include <chrono>  // `std::chrono::steady_clock`
#include <numeric> // `std::iota` - the keys as value rows
#include <span>    // `std::span`
#include <vector>  // `std::vector`

#include <fmt/format.h>

#include "dense/ram_indexes_search.hpp"
#include "dense_algebra.cuh"

namespace unum::usearch::bench::dense {

#pragma region Pipeline

/** Both phases over every query, batched through the executor: graph products, then attention of each token's
 *  `--head-count` heads over every key their walks score across the shards, judged against the exact oracle on the
 *  first `--oracle-queries` rows. */
template <typename streamed_dataset_type_>
bool run_algebra(streamed_dataset_type_& streamed, algebra_settings_t const& algebra) {
    using bench_t = typename streamed_dataset_type_::bench_t;
    using index_t = typename bench_t::index_t;
    using key_t = typename bench_t::key_t;
    using vector_view_t = typename bench_t::vector_view_t;
    using dot_result_t = typename bench_t::indexes_t::dot_result_t;
    auto& bench = streamed.bench;
    settings_t const& settings = bench.settings;
    std::size_t const nodes_count = bench.indexes.size(), head_count = algebra.head_count.value;
    // Keys are viewed rows, so every key finds its value row in place.
    std::vector<u32_t> rows(nodes_count);
    std::iota(rows.begin(), rows.end(), u32_t {0});
    algebra_inputs<typename bench_t::scalar_t> inputs;
    if (!open_algebra_inputs(streamed, algebra, rows, inputs)) return false;
    std::size_t const query_count = inputs.rows.extent(0), columns = inputs.values.columns, judged = inputs.judged;

    typename index_t::search_config_t config;
    config.expansion_factor = static_cast<typename bench_t::small_size_t>(settings.expansion_search.value);
    config.synchronization = synchronization_t::immutable_k;
    // A pair dump keeps every scored pair, not just the breadth's worth, so its rows go a thread's worth per call.
    std::size_t const wanted = algebra.pairs_path.empty() ? settings.expansion_search.value : std::size_t {1} << 18;
    std::size_t const batch_rows = algebra.pairs_path.empty() ? query_count : bench.executor.size();
    std::vector<key_t> keys(batch_rows * wanted);
    std::vector<dot_result_t> products(batch_rows * wanted);
    std::vector<std::size_t> counts(batch_rows);
    std::vector<std::vector<scored_pair_t>> pairs(algebra.pairs_path.empty() ? 0 : judged);
    std::size_t scored_pairs = 0;

    auto started = steady_clock_t::now();
    for (std::size_t base = 0; base != query_count;) {
        std::size_t const count = min_of(batch_rows, query_count - base);
        auto const [result, status] = bench.indexes.multiply(
            inputs.rows(nk::range(base, base + count), nk::slice), bench.executor, config, wanted,
            std::span {keys.data(), count * wanted}, std::span {products.data(), count * wanted},
            std::span {counts.data(), count});
        if (failed(status))
            return fmt::println("Multiply refused a batch of {} rows at row {}: {}", count, base,
                                status_to_string(status)),
                   false;
        for (std::size_t row = 0; row != count && base + row < pairs.size(); ++row)
            for (std::size_t pair = 0; pair != min_of(counts[row], wanted); ++pair)
                pairs[base + row].push_back(
                    {keys[row * wanted + pair], static_cast<f32_t>(static_cast<f64_t>(products[row * wanted + pair]))});
        scored_pairs += result.computed_distances;
        base += count;
    }
    double const multiply_seconds = stdc::duration<double>(steady_clock_t::now() - started).count();
    report_multiply_metrics(query_count, nodes_count, multiply_seconds, scored_pairs);
    if (!algebra.pairs_path.empty() && !save_scored_pairs(algebra.pairs_path, pairs))
        return fmt::println("Failed to write --pairs {}", algebra.pairs_path), false;

    auto const values = inputs.values.view(inputs.values.stored.data());
    auto const value_of = [&](std::span<key_t const> value_keys, std::span<vector_view_t> value_rows) noexcept {
        for (std::size_t lane = 0; lane != value_keys.size(); ++lane)
            value_rows[lane] = values.row(value_keys[lane]).as_vector();
    };
    std::vector<f32_t> outputs(query_count * columns);
    std::vector<dense_softmax_state_t> carry(query_count);
    started = steady_clock_t::now();
    auto const [attended, attend_status] = bench.indexes.attend(inputs.rows, head_count, value_of, inputs.score_scale,
                                                                std::span {outputs}, std::span {carry}, bench.executor,
                                                                config);
    double const attend_seconds = stdc::duration<double>(steady_clock_t::now() - started).count();
    if (failed(attend_status))
        return fmt::println("Attend refused {} queries: {}", query_count / head_count, status_to_string(attend_status)),
               false;
    report_attend_metrics(query_count, head_count, nodes_count, attend_seconds, attended.computed_distances);
    return judge_attention(streamed, algebra, inputs, std::span<f32_t> {outputs});
}

/** The algebra workload the cascade ends in: graph products and graph attention over the built shards, judged
 *  against the exact oracle. */
struct algebra_workload_t {

    /** Both verbs are defined over the dot kernel alone, so the cascade pins it. */
    static constexpr bool dot_only_k = true;

    /** The algebra knobs, borrowed for the run's lifetime. */
    algebra_settings_t const& algebra;

    template <typename streamed_dataset_type_>
    bool run(streamed_dataset_type_& streamed) const {
        return run_algebra(streamed, algebra);
    }
};

#pragma endregion

#pragma region Dispatch Cascade

/** The algebra shard hop: everything below one stored dtype under the dot kernel, so a per-dtype shard translation
 *  unit carries the whole instantiation grid of its dtype. */
template <typename dataset_scalar_type_, nk_dtype_t stored_dtype_>
int run_algebra_with_stored_dtype(bench_launch<dataset_scalar_type_> const& launch, algebra_settings_t const& algebra) {
    return run_with_compute<dataset_scalar_type_, stored_dtype_>(launch, algebra_workload_t {algebra});
}

#pragma endregion

} // namespace unum::usearch::bench::dense
