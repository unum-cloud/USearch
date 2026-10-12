/**
 *  @file bench/dense/nvme_indexes_algebra.hpp
 *  @author Ash Vardanian
 *  @date October 10, 2026
 *  @brief Profiling harness for graph products and graph attention over the sharded
 *      @c dense::nvme_indexes, shared by the per-dtype algebra shards.
 *
 *  The view, the cascade and the process setup are the NVMe search harness's, from
 *  @c dense/nvme_indexes_search.hpp; this header swaps its search phase for the batched @c multiply
 *  and @c attend from storage, pinned to the dot kernel both are defined over. Value rows are
 *  addressed by key, a dataset row, and the first `--oracle-queries` rows are judged against the
 *  exact attention oracle of @c dense_algebra.cuh, over `--vectors` or read from `--reference`.
 */
#pragma once

#include <algorithm> // `std::max`, `std::ranges::max`
#include <array>     // `std::array`
#include <chrono>    // `std::chrono::steady_clock`
#include <span>      // `std::span`
#include <vector>    // `std::vector`

#include <fmt/format.h>

#include "dense/nvme_indexes_search.hpp"
#include "dense_algebra.cuh"

namespace unum::usearch::bench::dense {

#pragma region Pipeline

/** The flags only the NVMe algebra binary takes beside @ref algebra_flags: the base rows the
 *  oracle attends over, and how many queries it judges. */
inline std::array<flag_t, 2> nvme_algebra_flags(settings_t& settings) noexcept {
    return {{
        {"--vectors", "path", "base vectors, a row per key, `.fbin` family; the oracle's alone, without --reference",
         &settings.vectors_path},
        {"--oracle-queries", "count", "queries the exact oracle judges; 16 by default", &settings.oracle_queries},
    }};
}

/** Writes `--pairs` from untimed calls over every row of @p rows, each wide enough for every pair
 *  its walks scored, widening until no row is cut short. */
template <typename nvme_type_>
bool save_nvme_pairs(nvme_context<nvme_type_>& nvme, algebra_settings_t const& algebra,
                     typename nvme_type_::vectors_view_t rows, typename nvme_type_::search_config_t config,
                     std::size_t width) {
    std::size_t const judged = rows.extent(0);
    std::vector<u32_t> keys;
    std::vector<typename nvme_type_::dot_result_t> products;
    std::vector<std::size_t> counts(judged);
    std::size_t widest = width;
    // Walks over storage need not repeat exactly, so a row scored past its width widens the call.
    do {
        width = widest;
        keys.resize(judged * width), products.resize(judged * width);
        auto const [result, status] = nvme.viewed.multiply(
            rows, nvme.executor, config, nk::matrix_span<u32_t> {keys.data(), judged, width},
            nk::matrix_span<typename nvme_type_::dot_result_t> {products.data(), judged, width}, std::span {counts});
        if (failed(status))
            return fmt::println("Multiply refused the judged rows: {}", status_to_string(status)), false;
        for (std::size_t const count : counts) widest = std::max(widest, count);
    } while (widest > width);
    std::vector<std::vector<scored_pair_t>> pairs(judged);
    for (std::size_t row = 0; row != judged; ++row)
        for (std::size_t pair = 0; pair != counts[row]; ++pair)
            pairs[row].push_back(
                {keys[row * width + pair], static_cast<f32_t>(static_cast<f64_t>(products[row * width + pair]))});
    if (save_scored_pairs(algebra.pairs_path, pairs)) return true;
    return fmt::println("Failed to write --pairs {}", algebra.pairs_path), false;
}

/** Both phases over every query from storage, one batched call each: graph products, then
 *  attention of each token's `--head-count` heads over every key their walks score across the
 *  parts, judged against the exact oracle on the first `--oracle-queries` rows. */
template <typename streamed_dataset_type_>
bool run_nvme_algebra(streamed_dataset_type_& streamed, algebra_settings_t const& algebra) {
    using nvme_t = typename streamed_dataset_type_::bench_t::indexes_t;
    using key_t = typename nvme_t::key_t;
    using vector_view_t = typename nvme_t::vector_view_t;
    using dot_result_t = typename nvme_t::dot_result_t;
    auto& nvme = streamed.bench;
    settings_t const& settings = nvme.settings;
    std::size_t const nodes_count = nvme.viewed.size(), head_count = algebra.head_count.value;

    // The keys are the dataset rows the value table and the oracle's base vectors are addressed by.
    std::vector<key_t> keys(nodes_count);
    typename nvme_t::cursor_t cursor {nvme.viewed};
    auto const [listed, list_status] = cursor.read(nvme.viewed, std::span {keys}, {}, nvme.executor);
    if (failed(list_status) || listed.count != nodes_count)
        return fmt::println("Failed to read the {} keys: {}", nodes_count, status_to_string(list_status)), false;
    std::size_t const key_rows = nodes_count ? std::size_t {std::ranges::max(keys)} + 1 : 0;
    if (algebra.reference_path.empty()) {
        if (key_rows != nodes_count)
            return fmt::println("The {} keys skip rows below {}, which the oracle would attend over; pass --reference",
                                nodes_count, key_rows),
                   false;
        if (streamed.vectors.extent(0) < key_rows)
            return fmt::println("--vectors holds {} rows, but the keys reach row {}", streamed.vectors.extent(0),
                                key_rows - 1),
                   false;
        streamed.vectors = streamed.vectors(nk::range(std::size_t {0}, key_rows), nk::slice);
    }
    else streamed.vectors = {};

    algebra_inputs<typename nvme_t::scalar_t> inputs;
    if (!open_algebra_inputs(streamed, algebra, keys, inputs)) return false;
    std::size_t const query_count = inputs.rows.extent(0), columns = inputs.value_rows.extent(1);
    // Attend looks values up by key, while the inputs hold them in viewed order.
    std::vector<u32_t> node_of_key(inputs.value_rows.extent(0));
    for (std::size_t node = 0; node != nodes_count; ++node) node_of_key[keys[node]] = static_cast<u32_t>(node);

    typename nvme_t::search_config_t config;
    config.expansion_factor = static_cast<u32_t>(settings.expansion_search.value);
    std::size_t const wanted = settings.expansion_search.value;
    std::vector<key_t> multiplied(query_count * wanted);
    std::vector<dot_result_t> products(query_count * wanted);
    std::vector<std::size_t> counts(query_count);
    std::vector<u64_t> latencies_ns(query_count);
    auto started = steady_clock_t::now();
    auto const [multiply_result, multiply_status] = nvme.viewed.multiply(
        inputs.rows, nvme.executor, config, nk::matrix_span<key_t> {multiplied.data(), query_count, wanted},
        nk::matrix_span<dot_result_t> {products.data(), query_count, wanted}, std::span {counts},
        std::span {latencies_ns});
    double const multiply_seconds = stdc::duration<double>(steady_clock_t::now() - started).count();
    if (failed(multiply_status))
        return fmt::println("Multiply refused {} rows: {}", query_count, status_to_string(multiply_status)), false;
    report_multiply_metrics(query_count, nodes_count, multiply_seconds, multiply_result.computed_distances);
    report_storage_metrics("row", query_count, multiply_seconds, multiply_result, std::span {latencies_ns});
    if (!algebra.pairs_path.empty()) {
        std::size_t width = 0;
        for (std::size_t row = 0; row != inputs.judged; ++row) width = std::max(width, counts[row]);
        if (!save_nvme_pairs(nvme, algebra, inputs.rows(nk::range(std::size_t {0}, inputs.judged), nk::slice), config,
                             width))
            return false;
    }

    auto const values = inputs.node_values_view();
    auto const value_of = [&](std::span<key_t const> value_keys, std::span<vector_view_t> value_rows) noexcept {
        for (std::size_t lane = 0; lane != value_keys.size(); ++lane)
            value_rows[lane] = values.row(node_of_key[value_keys[lane]]).as_vector();
    };
    std::size_t const token_count = query_count / head_count;
    std::vector<f32_t> outputs(query_count * columns);
    std::vector<softmax_state_t> carry(query_count);
    std::vector<u64_t> token_latencies_ns(token_count);
    started = steady_clock_t::now();
    nk::matrix_span<f32_t> const attended_rows {outputs.data(), query_count, columns};
    auto const [attended, attend_status] = nvme.viewed.attend(inputs.rows, head_count, value_of, inputs.score_scale,
                                                              attended_rows, std::span {carry}, nvme.executor, config,
                                                              std::span {token_latencies_ns});
    double const attend_seconds = stdc::duration<double>(steady_clock_t::now() - started).count();
    if (failed(attend_status))
        return fmt::println("Attend refused {} queries: {}", token_count, status_to_string(attend_status)), false;
    report_attend_metrics(query_count, head_count, nodes_count, attend_seconds, attended.computed_distances);
    report_storage_metrics("query", token_count, attend_seconds, attended, std::span {token_latencies_ns});
    return judge_attention(streamed, algebra, inputs, attended_rows);
}

/** The algebra workload the cascade ends in: graph products and graph attention over the viewed
 *  parts, judged against the exact oracle. */
struct nvme_algebra_workload_t {

    /** The algebra knobs, borrowed for the run's lifetime. */
    algebra_settings_t const& algebra;

    template <typename streamed_dataset_type_>
    bool run(streamed_dataset_type_& streamed) const {
        return run_nvme_algebra(streamed, algebra);
    }
};

/** The NVMe algebra shard hop: everything below one stored dtype under the dot kernel, so a
 *  per-dtype shard translation unit carries the whole instantiation grid of its dtype. */
template <typename dataset_scalar_type_, nk_dtype_t stored_dtype_>
int run_nvme_algebra_with_stored_dtype(bench_launch<dataset_scalar_type_> const& launch,
                                       algebra_settings_t const& algebra, nvme_settings_t const& nvme) {
    return view_with_compute<dataset_scalar_type_, stored_dtype_>(launch, nvme, nvme_algebra_workload_t {algebra});
}

#pragma endregion

} // namespace unum::usearch::bench::dense
