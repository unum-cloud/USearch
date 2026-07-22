/**
 *  @brief Profiling harness for the GPU-resident inner index over real persisted datasets.
 *
 *  @author Ash Vardanian
 *  @file cpp/inner_index_bench.cu
 *  @date July 22, 2026
 *
 *  The backend-neutral surface - options, checkpoints, dataset streaming, telemetry, result
 *  judgment - lives in @c inner_index_bench.cuh; this translation unit owns everything CUDA:
 *  one device, one @c inner_cuda_index instantiation, and batch-shaped run loops - the GPU
 *  ingests and answers thousands of rows per launch, so the pipeline streams the dataset in
 *  `--batch`-row slabs instead of one row per task. Checkpoints share the CPU bench's
 *  RetriEval-layout matrix files: the type-hash in the file name differs per instantiation,
 *  but the matrices themselves are backend-neutral, so a GPU-built graph can be inspected -
 *  or reloaded - by any tool that reads the format.
 */
#include <algorithm> // `std::max`, `std::min`, `std::copy_n`
#include <chrono>    // `std::chrono::steady_clock`
#include <cmath>     // `std::abs`
#include <cstdint>   // `std::uint32_t`
#include <memory>    // `std::unique_ptr`
#include <string>    // `std::string`
#include <vector>    // `std::vector`

#include <clipp.h>
#include <cuda_runtime.h>
#include <fmt/format.h>

#include <usearch/inner_cuda_index.cuh>

#include "inner_index_bench.cuh"
#include "persisted_dataset.hpp"

using namespace unum::usearch;

#pragma region CUDA Pipeline

/** @brief The GPU-side state one fully-typed benchmark run shares between its phases: the
 *      single device index, its one driving traversal context, and the embedded
 *      backend-neutral judgment results. One device, one submission lane - the batch shape,
 *      not thread count, is what fills the machine. */
template <typename index_type_>
struct cuda_bench_context {
    using index_t = index_type_;
    using scalar_t = typename index_t::scalar_t;
    using small_size_t = typename index_t::small_size_t;
    using traversal_t = typename index_t::inner_traversal_context_t;
    using vector_view_t = typename index_t::vector_view_t;
    using vectors_view_t = typename index_t::vectors_view_t;
    using node_id_t = typename index_t::node_id_t;

    static constexpr nk_kernel_kind_t kernel_kind_k = index_t::kernel_kind_k;

    bench_options_t const& options;
    int device = 0;
    std::unique_ptr<index_t> index;
    std::unique_ptr<traversal_t> traversal;
    /** @brief Dense node id → global dataset row, the recall bookkeeping. */
    std::vector<std::uint32_t> id_to_row;
    /** @brief One accumulation lane - the launches are sequential on one stream. */
    telemetry_lane_t lane;
    bench_results_t results;

    cuda_bench_context(bench_options_t const& options, int device) noexcept : options(options), device(device) {}
};

/** @brief Builds the device index, reserves it for the viewed rows, restores an optional
 *      checkpoint, and sizes the judgment state - everything fallible, before any clock. */
template <typename streamed_dataset_type_>
static bool prepare(streamed_dataset_type_& streamed) {
    using bench_t = typename streamed_dataset_type_::bench_t;
    using index_t = typename bench_t::index_t;
    using traversal_t = typename bench_t::traversal_t;
    using scalar_t = typename index_t::scalar_t;
    auto& bench = streamed.bench;
    std::size_t const rows = streamed.vectors.extent(0);
    std::size_t const dimensions = streamed.vectors.extent(1);

    connectivity_t const connectivity = connectivity_of(bench.options.connectivity);
    if (connectivity == connectivity_unknown_k) {
        fmt::print("Unsupported --connectivity {}; pick 16, 32, or 128\n", bench.options.connectivity);
        return false;
    }
    if (bench.options.skip_vectors > rows) {
        fmt::print("--skip-vectors {} exceeds the {} viewed rows\n", bench.options.skip_vectors, rows);
        return false;
    }

    using vectors_allocator_t = typename index_t::vectors_allocator_t;
    using neighbors_allocator_t = typename index_t::neighbors_allocator_t;
    bench.index = std::make_unique<index_t>(vectors_allocator_t {bench.device}, neighbors_allocator_t {bench.device});
    index_t& index = *bench.index;
    bench.id_to_row.resize(rows);

    // Reserving first freezes the CLI geometry and pre-maps capacity; a checkpoint then
    // loads against that frozen geometry, so a header mismatch fails loudly below instead
    // of restoring an incompatible graph.
    auto reserved = index.reserve({rows, connectivity, dimensions});
    if (reserved.status != status_t::success_k) {
        fmt::print("Failed to reserve {} × {}d on device {}\n", rows, dimensions, bench.device);
        return false;
    }

    if (!bench.options.load_name.empty() &&
        !load_checkpoint(bench.options, index, dimensions, std::span {bench.id_to_row}, 0, 1))
        return false;

    // The loaded base must cover exactly the skipped prefix - anything else would judge
    // recall against rows the graph never saw.
    if (index.size() != bench.options.skip_vectors) {
        fmt::print("Loaded index covers {} rows, but --skip-vectors says {}\n", index.size(),
                   bench.options.skip_vectors);
        return false;
    }

    bench.traversal = std::make_unique<traversal_t>(typename traversal_t::allocator_t {nullptr, bench.device});
    std::size_t const search_count = bench.options.self_recall ? streamed.vectors.extent(0)
                                                               : streamed.queries.extent(0);
    bench.results.hit_rows = nk::matrix<std::uint32_t>::try_zeros({search_count, bench.options.top_k});
    bench.results.hit_distances = nk::matrix<inner_distance_t>::try_zeros({search_count, bench.options.top_k});
    if (bench.results.hit_rows.empty() || bench.results.hit_distances.empty()) {
        fmt::print("Failed to allocate the per-query hit matrices\n");
        return false;
    }
    bench.results.hit_counts.resize(search_count);
    bench.results.latencies_us.resize(search_count);
    return true;
}

/** @brief `--batch` rows per launch, or a launch-saturating default: the insert path drains
 *      through a bounded admission window, so bigger batches only amortize transfers. */
inline std::size_t streamed_batch_rows(bench_options_t const& options, std::size_t total_rows) noexcept {
    std::size_t const preferred = options.batch ? options.batch : 65536;
    return std::min(std::max<std::size_t>(preferred, 1), total_rows);
}

/** @brief Stages up to @p batch_rows source rows starting at @p first_row into one
 *      device-ready host matrix - a zero-copy view when the scalars already match, or the
 *      trampoline's contiguous plane after per-row rescale & cast. */
template <typename streamed_dataset_type_>
static typename streamed_dataset_type_::bench_t::vectors_view_t stage_batch(streamed_dataset_type_& streamed,
                                                                            std::size_t first_row,
                                                                            std::size_t batch_rows) {
    using vectors_view_t = typename streamed_dataset_type_::bench_t::vectors_view_t;
    std::size_t const dimensions = streamed.vectors.extent(1);
    if constexpr (streamed_dataset_type_::zero_copy_k)
        return vectors_view_t {streamed.vectors.row(first_row).data(), {batch_rows, dimensions}};
    else {
        for (std::size_t row = 0; row != batch_rows; ++row)
            streamed.stage(streamed.vectors.row(first_row + row).as_vector(), 0, row);
        return vectors_view_t {streamed.staging.row(0).row(0).data(), {batch_rows, dimensions}};
    }
}

/** @brief Streams the un-ingested topping into the index, `--batch` rows per launch - the
 *      GPU's natural task shape - and keeps the id → row bookkeeping dense. */
template <typename streamed_dataset_type_>
static bool run_insertion(streamed_dataset_type_& streamed) {
    using clock_t = std::chrono::steady_clock;
    using bench_t = typename streamed_dataset_type_::bench_t;
    using index_t = typename bench_t::index_t;
    using node_id_t = typename bench_t::node_id_t;
    using small_size_t = typename bench_t::small_size_t;
    using scalar_t = typename streamed_dataset_type_::scalar_t;
    auto& bench = streamed.bench;
    std::size_t const rows = streamed.vectors.extent(0);
    std::size_t const inserted_rows = rows - bench.options.skip_vectors;
    if (!inserted_rows) {
        fmt::print(header_style(), "\nInsertion\n");
        fmt::print("  skipped: the loaded base already covers all {} rows\n", rows);
        return true;
    }

    typename index_t::insert_config_t config;
    config.expansion_factor = static_cast<small_size_t>(bench.options.expansion_add);
    std::size_t const batch_rows = streamed_batch_rows(bench.options, inserted_rows);
    std::vector<node_id_t> batch_ids(batch_rows);

    auto started = clock_t::now();
    for (std::size_t offset = 0; offset < inserted_rows;) {
        std::size_t const first_row = bench.options.skip_vectors + offset;
        std::size_t const batch = std::min(batch_rows, inserted_rows - offset);
        auto const staged = stage_batch(streamed, first_row, batch);
        auto inserted = bench.index->insert(staged, {batch_ids.data(), batch}, *bench.traversal, config);
        if (!inserted) {
            fmt::print("Insertion failed at row {}: status {}\n", first_row, int(inserted.status));
            return false;
        }
        for (std::size_t row = 0; row != batch; ++row)
            bench.id_to_row[batch_ids[row]] = static_cast<std::uint32_t>(first_row + row);
        bench.lane.visited_members += inserted.visited_members;
        bench.lane.computed_distances += inserted.computed_distances;
        bench.lane.restarts += inserted.restarts;
        offset += batch;
    }
    double const seconds = std::chrono::duration<double>(clock_t::now() - started).count();

    fmt::print(header_style(), "\nInsertion\n");
    fmt::print("  shape: {} × {}d\n", inserted_rows, streamed.vectors.extent(1));
    if (bench.options.skip_vectors) fmt::print("  topping: {} loaded rows\n", bench.options.skip_vectors);
    fmt::print("  kit: {} · {} · device {}\n", scalar_t::dtype_name(), kernel_name(bench.options.kernel), bench.device);
    fmt::print("  batch: {} rows per launch\n", batch_rows);
    fmt::print("  wall: {:.2f} s\n", seconds);
    fmt::print("  throughput: {}/s\n", humanize(double(inserted_rows) / seconds));
    fmt::print("  distances: {}\n", humanize(double(bench.lane.computed_distances)));
    fmt::print("  visited: {}\n", humanize(double(bench.lane.visited_members)));
    fmt::print("  restarts: ");
    fmt::print(verdict_style(bench.lane.restarts < inserted_rows / 100 + 1), "{}\n", bench.lane.restarts);
    return true;
}

/** @brief Serializes the device index as the same three RetriEval-layout matrices the CPU
 *      bench writes - vectors, neighbors, and the id → row map. */
template <typename streamed_dataset_type_>
static bool save_index(streamed_dataset_type_& streamed) {
    using clock_t = std::chrono::steady_clock;
    auto& bench = streamed.bench;
    auto started = clock_t::now();
    std::size_t const total_bytes = save_checkpoint(bench.options, *bench.index, std::span {bench.id_to_row}, 0, 1);
    if (!total_bytes) return false;
    double const seconds = std::chrono::duration<double>(clock_t::now() - started).count();
    fmt::print("Saved the index: {:.2f} GB in {:.2f} s\n", double(total_bytes) / 1e9, seconds);
    return true;
}

/** @brief Answers every query through the batch verb - thousands of queries per launch, the
 *      shape the device is built for - then translates node ids into dataset rows for the
 *      shared judgment. Latency is reported per batch, spread over its queries: the GPU has
 *      no per-query clock to read honestly from the host. */
template <typename streamed_dataset_type_>
static bool run_search(streamed_dataset_type_& streamed) {
    using clock_t = std::chrono::steady_clock;
    using bench_t = typename streamed_dataset_type_::bench_t;
    using index_t = typename bench_t::index_t;
    using node_id_t = typename bench_t::node_id_t;
    using small_size_t = typename bench_t::small_size_t;
    auto& bench = streamed.bench;
    auto& source = bench.options.self_recall ? streamed.vectors : streamed.queries;
    std::size_t const query_count = source.extent(0);
    std::size_t const top_k = bench.options.top_k;
    std::size_t const node_count = bench.index->size();
    bench.lane = telemetry_lane_t {};

    typename index_t::search_config_t config;
    config.expansion_factor = static_cast<small_size_t>(bench.options.expansion_search);
    config.count = static_cast<small_size_t>(top_k);
    config.synchronization = synchronization_t::immutable_k;
    std::size_t const batch_rows = streamed_batch_rows(bench.options, query_count);
    auto found_ids = nk::matrix<node_id_t>::try_zeros({batch_rows, top_k});
    auto found_distances = nk::matrix<inner_distance_t>::try_zeros({batch_rows, top_k});
    if (found_ids.empty() || found_distances.empty()) {
        fmt::print("Failed to allocate the batch result matrices\n");
        return false;
    }

    double seconds = 0;
    for (std::size_t offset = 0; offset < query_count;) {
        std::size_t const batch = std::min(batch_rows, query_count - offset);
        typename bench_t::vectors_view_t staged;
        if constexpr (streamed_dataset_type_::zero_copy_k)
            staged = {source.row(offset).data(), {batch, source.extent(1)}};
        else {
            for (std::size_t row = 0; row != batch; ++row) streamed.stage(source.row(offset + row).as_vector(), 0, row);
            staged = {streamed.staging.row(0).row(0).data(), {batch, source.extent(1)}};
        }

        auto started = clock_t::now();
        auto found = bench.index->search(staged, nk::matrix_span<node_id_t> {found_ids.data(), {batch, top_k}},
                                         nk::matrix_span<inner_distance_t> {found_distances.data(), {batch, top_k}},
                                         *bench.traversal, config);
        double const batch_seconds = std::chrono::duration<double>(clock_t::now() - started).count();
        if (!found) {
            fmt::print("Search failed at query {}: status {}\n", offset, int(found.status));
            return false;
        }
        seconds += batch_seconds;
        bench.lane.visited_members += found.visited_members;
        bench.lane.computed_distances += found.computed_distances;
        bench.lane.restarts += found.restarts;

        float const shared_latency_us = float(batch_seconds * 1e6 / double(batch));
        for (std::size_t row = 0; row != batch; ++row) {
            std::uint32_t* const merged_rows = bench.results.hit_rows.row(offset + row).data();
            auto* const merged_distances = bench.results.hit_distances.row(offset + row).data();
            std::size_t merged_count = 0;
            for (std::size_t rank = 0; rank != top_k; ++rank) {
                node_id_t const id = found_ids.view().row(row).data()[rank];
                if (static_cast<std::size_t>(id) >= node_count) break; // Sentinel-padded tail.
                merged_rows[merged_count] = bench.id_to_row[id];
                merged_distances[merged_count] = found_distances.view().row(row).data()[rank];
                ++merged_count;
            }
            bench.results.hit_counts[offset + row] = static_cast<std::uint32_t>(merged_count);
            bench.results.latencies_us[offset + row] = shared_latency_us;
        }
        offset += batch;
    }

    report_search_metrics(bench.options, query_count, seconds, 1, bench.lane,
                          {bench.results.latencies_us.data(), bench.results.latencies_us.size()});
    judge_hits(bench.options, bench.results, query_count);
    return true;
}

#pragma endregion

#pragma region Dispatch Cascade

/** @brief The innermost hop: binds the concrete CUDA-index instantiation, wires the typed
 *      source views and the staging trampoline, and runs the phases in checkpoint order. */
template <typename dataset_scalar_type_, nk_kernel_kind_t kernel_>
static int stage_and_run(persisted_dataset<dataset_scalar_type_> const& dataset, bench_options_t const& options,
                         int device) {
    using index_t = inner_cuda_index<nk_i8_k, kernel_>;
    using bench_context_t = cuda_bench_context<index_t>;
    using streamed_t = streamed_dataset<dataset_scalar_type_, bench_context_t>;
    using scalar_t = typename bench_context_t::scalar_t;

    std::size_t const dimensions = dataset.dimensions();
    std::size_t rows = dataset.vectors().extent(0);
    if (options.max_vectors) rows = std::min(rows, options.max_vectors);

    // Self-queries keep the harness usable on a lone vectors file, without recall.
    nk::matrix_view<dataset_scalar_type_> const source_queries =
        dataset.has_queries() ? dataset.queries()
                              : nk::matrix_view<dataset_scalar_type_> {dataset.vectors().data(),
                                                                       std::min<std::size_t>(1000, rows), dimensions};

    bench_context_t bench {options, device};
    streamed_t streamed {bench, {dataset.vectors().data(), rows, dimensions}, source_queries};

    /** @brief Owns the staging trampoline plane - `[1, batch, dimensions]`, one lane. */
    nk::tensor<scalar_t, std::allocator<scalar_t>, 3> staging;
    nk::tensor<nk::f32_t, std::allocator<nk::f32_t>, 3> rescaling_planes;
    if constexpr (!streamed_t::zero_copy_k) {
        std::size_t const batch_rows = streamed_batch_rows(options, rows);
        staging = nk::tensor<scalar_t, std::allocator<scalar_t>, 3>::try_zeros({1, batch_rows, dimensions});
        if (staging.empty()) {
            fmt::print("Failed to allocate {} staging rows of {}d {}\n", batch_rows, dimensions,
                       scalar_t::dtype_name());
            return 1;
        }
        streamed.staging = staging.span();

        if constexpr (streamed_t::scaled_k) {
            rescaling_planes = nk::tensor<nk::f32_t, std::allocator<nk::f32_t>, 3>::try_zeros(
                {1, batch_rows, dimensions});
            if (rescaling_planes.empty()) {
                fmt::print("Failed to allocate {} rescaling rows of {}d f32\n", batch_rows, dimensions);
                return 1;
            }
            streamed.rescaling = rescaling_planes.span();

            // One SIMD `reduce_minmax` pass over the contiguous payload finds the amplitude;
            // the same factor then rescales every vector AND every query, keeping their
            // geometry consistent.
            typename dataset_scalar_type_::reduce_minmax_value_t min_value {}, max_value {};
            std::size_t min_index = 0, max_index = 0;
            nk::vector_view<dataset_scalar_type_> const flat {dataset.vectors().data(), rows * dimensions};
            nk::reduce_minmax(flat, &min_value, &min_index, &max_value, &max_index);
            double const max_amplitude = std::max(std::abs(double(float(min_value))),
                                                  std::abs(double(float(max_value))));
            if (max_amplitude > 0) streamed.scale = float(streamed_t::target_amplitude_k / max_amplitude);
            fmt::print("Rescaling {} → {} by ×{:.4f} before the downcast\n", dataset_scalar_type_::dtype_name(),
                       scalar_t::dtype_name(), streamed.scale);
        }
    }

    if (dataset.has_ground_truth() && !options.max_vectors) bench.results.ground_truth = dataset.neighbors();
    else if (dataset.has_ground_truth())
        fmt::print("Ground truth ignored: --max-vectors truncates the graph it was computed for\n");

    if (!prepare(streamed)) return 1;
    if (!run_insertion(streamed)) return 1;
    if (!options.save_name.empty() && !save_index(streamed)) return 1;
    if (!run_search(streamed)) return 1;
    return 0;
}

template <typename dataset_scalar_type_>
static int run_with_metric(persisted_dataset<dataset_scalar_type_> const& dataset, bench_options_t const& options,
                           int device) {
    switch (options.kernel) {
    case nk_kernel_angular_k: return stage_and_run<dataset_scalar_type_, nk_kernel_angular_k>(dataset, options, device);
    case nk_kernel_euclidean_k:
        return stage_and_run<dataset_scalar_type_, nk_kernel_euclidean_k>(dataset, options, device);
    default: fmt::print("Unsupported metric for the CUDA kits; pick --angular or --euclidean\n"); return 1;
    }
}

/** @brief Opens the dataset with the file-native scalar and enters the typed cascade. */
template <typename dataset_scalar_type_>
static int open_and_run(bench_options_t const& options, int device) {
    auto opened = persisted_dataset<dataset_scalar_type_>::try_open(
        options.vectors_path.c_str(), options.queries_path.empty() ? nullptr : options.queries_path.c_str(),
        options.neighbors_path.empty() ? nullptr : options.neighbors_path.c_str());
    if (opened.status != status_t::success_k) {
        fmt::print("Failed to open the dataset: {}\n", options.vectors_path);
        return 1;
    }
    return run_with_metric<dataset_scalar_type_>(opened.value, options, device);
}

#pragma endregion

int main(int argc, char** argv) {
    bench_options_t options;
    int device = 0;

    auto cli = ( //
        (clipp::option("--vectors") & clipp::value("path", options.vectors_path)) % "base vectors, `.fbin` family",
        (clipp::option("--queries") & clipp::value("path", options.queries_path)) % "query vectors, same dtype",
        (clipp::option("--neighbors") & clipp::value("path", options.neighbors_path)) % "ground truth, `.ibin`",
        (clipp::option("--device") & clipp::value("id", device)) % "CUDA device ordinal; 0 by default",
        (clipp::option("--angular").set(options.kernel, nk_kernel_angular_k) |
         clipp::option("--euclidean").set(options.kernel, nk_kernel_euclidean_k)) %
            "metric; angular by default",
        (clipp::option("--batch") & clipp::value("rows", options.batch)) %
            "rows & queries per launch; 65536 by default",
        (clipp::option("--connectivity") & clipp::value("degree", options.connectivity)) % "16, 32, or 128",
        (clipp::option("--expansion-add") & clipp::value("breadth", options.expansion_add)) %
            "insertion candidate pool",
        (clipp::option("--expansion-search") & clipp::value("breadth", options.expansion_search)) %
            "search candidate pool",
        (clipp::option("--top-k") & clipp::value("count", options.top_k)) % "matches per query to judge",
        (clipp::option("--max-vectors") & clipp::value("count", options.max_vectors)) % "cap on ingested rows",
        clipp::option("--self-recall").set(options.self_recall, true) % //
            "query every stored vector for itself instead of the query file",
        (clipp::option("--save") & clipp::value("name", options.save_name)) % //
            "checkpoint name to serialize the index into after ingestion",
        (clipp::option("--load") & clipp::value("name", options.load_name)) % //
            "checkpoint name to restore the index from instead of building",
        (clipp::option("--checkpoints-dir") & clipp::value("dir", options.checkpoints_dir)) % //
            "directory for checkpoint blobs; /tmp by default",
        (clipp::option("--skip-vectors") & clipp::value("count", options.skip_vectors)) % //
            "rows the loaded index already covers",
        clipp::option("--help").set(options.help) % "print this message");

    if (!clipp::parse(argc, argv, cli) || options.help || options.vectors_path.empty()) {
        for (auto const& section : clipp::make_man_page(cli, "inner_index_bench_cu"))
            fmt::print("{}\n{}\n\n", section.title(), section.content());
        return options.help ? 0 : 1;
    }

    int devices_count = 0;
    if (cudaGetDeviceCount(&devices_count) != cudaSuccess || device >= devices_count) {
        fmt::print("CUDA device {} unreachable; {} devices visible\n", device, devices_count);
        return 1;
    }
    cudaSetDevice(device);
    cudaDeviceProp properties = {};
    cudaGetDeviceProperties(&properties, device);
    fmt::print(header_style(), "Machine\n");
    fmt::print("  device {}: {} · {} SMs · {:.0f} GB\n", device, properties.name, properties.multiProcessorCount,
               double(properties.totalGlobalMem) / 1e9);
    fmt::print("  kernels requested: i8 · {}\n", kernel_name(options.kernel));

    nk_dtype_t const file_dtype = dtype_of_extension(options.vectors_path.c_str());
    switch (file_dtype) {
    case nk_f32_k: return open_and_run<nk::f32_t>(options, device);
    case nk_i8_k: return open_and_run<nk::i8_t>(options, device);
    case nk_u8_k: return open_and_run<nk::u8_t>(options, device);
    default: fmt::print("Unsupported dataset dtype for the CUDA kits\n"); return 1;
    }
}
