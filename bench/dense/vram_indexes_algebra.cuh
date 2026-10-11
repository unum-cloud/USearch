/**
 *  @file bench/dense/vram_indexes_algebra.cuh
 *  @author Ash Vardanian
 *  @date October 9, 2026
 *  @brief Profiling harness for graph products and graph attention over the sharded @c dense_vram_indexes, shared
 *      by the per-dtype algebra shards.
 *
 *  The build phases and the cascade below the stored dtype come from @c dense/vram_indexes_search.cuh, and the
 *  value lookup from @c dense/vram_index_algebra.cuh. Every token walks every shard, and the first device merges the
 *  shards' attention. Both verbs score dot products only, so the workload pins the kernel and no shard carries the
 *  bit-packed dtype.
 */
#pragma once

#include <algorithm> // `std::min`
#include <chrono>    // `std::chrono::steady_clock`
#include <numeric>   // `std::iota` - the keys as value rows
#include <span>      // `std::span`
#include <vector>    // `std::vector`

#include <cuda_runtime.h>
#include <fmt/format.h>

#include "dense/vram_index_algebra.cuh"
#include "dense/vram_indexes_search.cuh"

namespace unum::usearch::bench::dense {

#pragma region Pipeline

/** Both phases over every query, batched across the shards: graph products landed on the host, then attention of
 *  each token's `--head-count` heads over every key their walks score, value rows looked up by key in a table every
 *  device reads, into rows on the first device. The first `--oracle-queries` rows are judged against the exact
 *  oracle. */
template <typename streamed_dataset_type_>
bool run_indexes_algebra(streamed_dataset_type_& streamed, algebra_settings_t const& algebra) {
    using bench_t = typename streamed_dataset_type_::bench_t;
    using index_t = typename bench_t::index_t;
    using key_t = typename bench_t::key_t;
    using dot_result_t = typename bench_t::indexes_t::dot_result_t;
    auto& bench = streamed.bench;
    settings_t const& settings = bench.settings;
    if (!algebra.pairs_path.empty()) return fmt::println("--pairs dumps host walks; device walks log no pairs"), false;
    std::size_t const nodes_count = bench.indexes.size(), head_count = algebra.head_count.value;
    std::size_t const wanted = settings.expansion_search.value;
    // Keys are viewed rows, so every key finds its value row in place.
    std::vector<u32_t> rows(nodes_count);
    std::iota(rows.begin(), rows.end(), u32_t {0});
    algebra_inputs<typename bench_t::scalar_t> inputs;
    if (!open_algebra_inputs(streamed, algebra, rows, inputs)) return false;
    std::size_t const query_count = inputs.rows.extent(0), columns = inputs.value_rows.extent(1),
                      judged = inputs.judged;
    typename index_t::search_config_t config;
    config.expansion_factor = static_cast<typename bench_t::small_size_t>(settings.expansion_search.value);
    config.synchronization = synchronization_t::immutable_k;
    std::size_t const batch_rows = streamed_batch_rows(settings, query_count);

    std::vector<key_t> keys(batch_rows * wanted);
    std::vector<dot_result_t> products(batch_rows * wanted);
    std::vector<std::size_t> counts(batch_rows);
    telemetry_lane_t sums;
    double multiply_seconds = 0;
    for (std::size_t offset = 0; offset < query_count;) {
        std::size_t const batch = std::min(batch_rows, query_count - offset);
        auto started = steady_clock_t::now();
        auto const [result, status] = bench.indexes.multiply(
            inputs.rows(nk::range(offset, offset + batch), nk::slice), bench.executor, config, wanted,
            std::span {keys.data(), batch * wanted}, std::span {products.data(), batch * wanted},
            std::span {counts.data(), batch});
        multiply_seconds += stdc::duration<double>(steady_clock_t::now() - started).count();
        if (failed(status)) {
            fmt::println("Multiply refused a batch of {} rows at row {}: {}", batch, offset, status_to_string(status));
            return false;
        }
        sums.computed_distances += result.computed_distances;
        offset += batch;
    }
    report_multiply_metrics(query_count, nodes_count, multiply_seconds, sums.computed_distances);

    // Every shard's device reads the one table, so it lives in managed memory, each device keeping a read-only copy.
    std::size_t const row_bytes = inputs.value_row_bytes(), pitch = round_up_to_multiple<std::size_t>(row_bytes, 16);
    std::size_t const attend_rows = round_up_to_multiple(batch_rows, head_count);
    cuda_managed_allocator<byte_t> device_values;
    auto device_outputs = cuda_make_unique(cuda_device_allocator_t {nullptr, bench.devices.front()},
                                           attend_rows * columns * sizeof(f32_t));
    if (failed(device_values.reserve_range(nodes_count * pitch, bench.devices.front())) || !device_outputs ||
        cudaMemcpy2D(device_values.base(), pitch, inputs.node_values.data(), row_bytes, row_bytes, nodes_count,
                     cudaMemcpyDefault) != cudaSuccess ||
        cudaMemAdvise(device_values.base(), nodes_count * pitch, cudaMemAdviseSetReadMostly, cudaMemLocation {}) !=
            cudaSuccess) {
        fmt::println("Failed to place {:.2f} GB of values in managed memory", gigabytes(nodes_count * pitch));
        return false;
    }
    attention_value_lookup<typename index_t::fragment_t> const lookup {device_values.base(), pitch};
    std::vector<dense_softmax_state_t> carry(attend_rows);
    std::vector<f32_t> judged_outputs(judged * columns);
    f32_t* const outputs = reinterpret_cast<f32_t*>(device_outputs.get());
    sums = telemetry_lane_t {};
    double attend_seconds = 0;
    for (std::size_t offset = 0; offset < query_count;) {
        std::size_t const batch = std::min(attend_rows, query_count - offset);
        auto started = steady_clock_t::now();
        auto const [result, status] = bench.indexes.attend(
            inputs.rows(nk::range(offset, offset + batch), nk::slice), head_count, lookup, inputs.score_scale,
            std::span {outputs, batch * columns}, std::span {carry.data(), batch}, bench.executor, config);
        attend_seconds += stdc::duration<double>(steady_clock_t::now() - started).count();
        if (failed(status)) {
            fmt::println("Attend refused a batch of {} queries at query {}: {}", batch, offset,
                         status_to_string(status));
            return false;
        }
        sums.computed_distances += result.computed_distances;
        std::size_t const judged_rows = offset < judged ? std::min(batch, judged - offset) : 0;
        if (judged_rows && cudaMemcpy(judged_outputs.data() + offset * columns, outputs,
                                      judged_rows * columns * sizeof(f32_t), cudaMemcpyDeviceToHost) != cudaSuccess) {
            fmt::println("Failed to read back the attended rows");
            return false;
        }
        offset += batch;
    }
    report_attend_metrics(query_count, head_count, nodes_count, attend_seconds, sums.computed_distances);
    return judge_attention(streamed, algebra, inputs, std::span<f32_t> {judged_outputs});
}

/** The algebra workload the cascade ends in: graph products and graph attention over the built shards, judged
 *  against the exact oracle. */
struct indexes_algebra_workload_t {

    /** Both verbs are defined over the dot kernel alone, so the cascade pins it. */
    static constexpr bool dot_only_k = true;

    /** The algebra knobs, borrowed for the run's lifetime. */
    algebra_settings_t const& algebra;

    template <typename streamed_dataset_type_>
    bool run(streamed_dataset_type_& streamed) const {
        return run_indexes_algebra(streamed, algebra);
    }
};

/** The algebra shard hop: the compute ceiling below one stored dtype under the dot kernel, so the per-dtype shards
 *  carry the whole device instantiation grid of their dtype. */
template <typename dataset_scalar_type_, nk_dtype_t stored_dtype_>
int run_indexes_algebra_with_stored_dtype(utilities::persisted_dataset<dataset_scalar_type_> const& dataset,
                                          settings_t const& settings, algebra_settings_t const& algebra,
                                          std::span<int const> devices, cuda_arch_t max_compute) {
    return run_indexes_with_compute<dataset_scalar_type_, stored_dtype_>(dataset, settings, devices, max_compute,
                                                                         indexes_algebra_workload_t {algebra});
}

#pragma endregion

} // namespace unum::usearch::bench::dense
