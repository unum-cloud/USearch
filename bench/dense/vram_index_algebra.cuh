/**
 *  @file bench/dense/vram_index_algebra.cuh
 *  @author Ash Vardanian
 *  @date September 27, 2026
 *  @brief Profiling harness for graph products and graph attention over the GPU-resident
 *      @c dense::vram_nvidia_index, shared by the per-dtype algebra shards.
 *
 *  The build phases and the cascade below the stored dtype come from @c dense/vram_index_search.cuh, and the value
 * tables and the oracle from @c dense_algebra.cuh. Both verbs score dot products only, so the workload pins the kernel
 * and no shard carries the bit-packed dtype.
 */
#pragma once

#include <algorithm> // `std::min`
#include <chrono>    // `std::chrono::steady_clock`
#include <span>      // `std::span`
#include <vector>    // `std::vector`

#include <fmt/format.h>

#include "dense/vram_index_search.cuh"
#include "dense_algebra.cuh"

namespace unum::usearch::bench::dense {

#pragma region Pipeline

/** Resolves an ID, a node on one device or a key across the shards, to its value row in a device table of rows
 *  @c pitch bytes apart. */
template <typename fragment_type_>
struct attention_value_lookup {

    /** Names the lookup to the attend kernels the binary's fatbin carries. */
    inline static constexpr u32_t device_lookup_k = 1;

    byte_t const* rows;
    std::size_t pitch;

#if defined(__CUDACC__)
    __device__ fragment_type_ const* operator()(std::size_t id) const noexcept {
        return reinterpret_cast<fragment_type_ const*>(rows + id * pitch);
    }
#endif
};

/** Both phases over every query, batched: graph products landed on the host, then attention of each token's
 *  `--head-count` heads over every key their shared walk scores, value rows looked up by node ID in a device-resident
 *  table, into device rows. The first `--oracle-queries` rows are judged against the exact oracle. */
template <typename streamed_dataset_type_>
bool run_algebra(streamed_dataset_type_& streamed, algebra_settings_t const& algebra) {
    using bench_t = typename streamed_dataset_type_::bench_t;
    using index_t = typename bench_t::index_t;
    using node_id_t = typename bench_t::node_id_t;
    using dot_result_t = typename index_t::dot_result_t;
    auto& bench = streamed.bench;
    settings_t const& settings = bench.settings;
    if (!algebra.pairs_path.empty()) return fmt::println("--pairs dumps host walks; device walks log no pairs"), false;
    std::size_t const nodes_count = bench.index->size(), head_count = algebra.head_count.value;
    std::size_t const wanted = settings.expansion_search.value;
    algebra_inputs<typename bench_t::scalar_t> inputs;
    if (!open_algebra_inputs(streamed, algebra, {streamed.id_to_row.data(), nodes_count}, inputs)) return false;
    std::size_t const query_count = inputs.rows.extent(0), columns = inputs.value_rows.extent(1),
                      judged = inputs.judged;
    typename index_t::search_config_t config;
    config.expansion_factor = static_cast<typename bench_t::small_size_t>(settings.expansion_search.value);
    config.synchronization = synchronization_t::immutable_k;
    std::size_t const batch_rows = streamed_batch_rows(settings, query_count);

    auto node_ids = nk::matrix<node_id_t>::zeros({batch_rows, wanted}).value;
    auto products = nk::matrix<dot_result_t>::zeros({batch_rows, wanted}).value;
    std::vector<u32_t> counts(batch_rows);
    bench.lane = telemetry_lane_t {};
    double multiply_seconds = 0;
    for (std::size_t offset = 0; offset < query_count;) {
        std::size_t const batch = std::min(batch_rows, query_count - offset);
        auto started = steady_clock_t::now();
        nk::range_t const landing = nk::range(std::size_t {0}, batch);
        auto const [result, status] = bench.index->multiply(inputs.rows(nk::range(offset, offset + batch), nk::slice),
                                                            node_ids(landing, nk::slice), products(landing, nk::slice),
                                                            std::span {counts.data(), batch}, *bench.traversal, config);
        multiply_seconds += stdc::duration<double>(steady_clock_t::now() - started).count();
        if (failed(status)) {
            fmt::println("Multiply refused a batch of {} rows at row {}: {}", batch, offset, status_to_string(status));
            return false;
        }
        bench.lane.computed_distances += result.computed_distances;
        offset += batch;
    }
    report_multiply_metrics(query_count, nodes_count, multiply_seconds, bench.lane.computed_distances);

    // Rows a 16-byte pitch apart, the alignment the lookup's fragment rows promise.
    std::size_t const row_bytes = inputs.value_row_bytes(), pitch = round_up_to_multiple<std::size_t>(row_bytes, 16);
    std::size_t const attend_rows = round_up_to_multiple(batch_rows, head_count);
    auto device_values = cuda_make_unique(cuda_device_allocator_t {nullptr, bench.device}, nodes_count * pitch);
    auto device_outputs = cuda_make_unique(cuda_device_allocator_t {nullptr, bench.device},
                                           attend_rows * columns * sizeof(f32_t));
    if (!device_values || !device_outputs ||
        cuda_copy_2d_async(device_values.get(), pitch, inputs.node_values.data(), row_bytes, row_bytes, nodes_count,
                           nullptr) != CUDA_SUCCESS ||
        cuda_api().stream_synchronize(nullptr) != CUDA_SUCCESS) {
        fmt::println("Failed to place {:.2f} GB of values on device {}", gigabytes(nodes_count * pitch), bench.device);
        return false;
    }
    attention_value_lookup<typename index_t::fragment_t> const lookup {device_values.get(), pitch};
    std::vector<softmax_state_t> carry(attend_rows);
    auto judged_outputs = nk::matrix<f32_t>::zeros({judged, columns}).value;
    nk::matrix_span<f32_t> const outputs {reinterpret_cast<f32_t*>(device_outputs.get()), attend_rows, columns};
    bench.lane = telemetry_lane_t {};
    double attend_seconds = 0;
    for (std::size_t offset = 0; offset < query_count;) {
        std::size_t const batch = std::min(attend_rows, query_count - offset);
        auto started = steady_clock_t::now();
        auto const [result, status] = bench.index->attend(inputs.rows(nk::range(offset, offset + batch), nk::slice),
                                                          head_count, lookup, inputs.score_scale,
                                                          outputs(nk::range(std::size_t {0}, batch), nk::slice),
                                                          std::span {carry.data(), batch}, *bench.traversal, config);
        attend_seconds += stdc::duration<double>(steady_clock_t::now() - started).count();
        if (failed(status)) {
            fmt::println("Attend refused a batch of {} queries at query {}: {}", batch, offset,
                         status_to_string(status));
            return false;
        }
        bench.lane.computed_distances += result.computed_distances;
        std::size_t const judged_rows = offset < judged ? std::min(batch, judged - offset) : 0;
        if (judged_rows && cuda_copy(judged_outputs.row(offset).data(), outputs.data(),
                                     judged_rows * columns * sizeof(f32_t)) != CUDA_SUCCESS) {
            fmt::println("Failed to read back the attended rows");
            return false;
        }
        offset += batch;
    }
    report_attend_metrics(query_count, head_count, nodes_count, attend_seconds, bench.lane.computed_distances);
    return judge_attention(streamed, algebra, inputs, judged_outputs.span());
}

/** The algebra workload the cascade ends in: graph products and graph attention over the built index, judged
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

/** The algebra shard hop: the memory-axis split of @ref run_with_stored_dtype, with the cascade ending in the
 *  algebra workload, so the per-dtype shards carry the whole device instantiation grid of their dtype. */
template <typename dataset_scalar_type_, nk_dtype_t stored_dtype_>
int run_algebra_with_stored_dtype(utilities::persisted_dataset<dataset_scalar_type_> const& dataset,
                                  settings_t const& settings, algebra_settings_t const& algebra, int device,
                                  nk_capability_t max_compute, cuda_address_space_t plane_space) {
    return plane_space == cuda_address_space_t::managed_k
               ? run_with_compute<dataset_scalar_type_, stored_dtype_, cuda_managed_allocator<std::byte>>( //
                     dataset, settings, device, max_compute, algebra_workload_t {algebra})
               : run_with_compute<dataset_scalar_type_, stored_dtype_, cuda_virtual_memory_allocator<std::byte>>( //
                     dataset, settings, device, max_compute, algebra_workload_t {algebra});
}

#pragma endregion

} // namespace unum::usearch::bench::dense

template <>
struct us::dense::cuda_value_lookup_of<us::bench::dense::attention_value_lookup<uint4>::device_lookup_k> {
    using type = us::bench::dense::attention_value_lookup<uint4>;
};
