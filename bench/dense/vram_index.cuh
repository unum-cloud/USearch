/**
 *  @file bench/dense/vram_index.cuh
 *  @author Ash Vardanian
 *  @date July 22, 2026
 *  @brief The CUDA pipeline both dense-index benchmark families share: the single-device phases,
 *      the compile-time cascade below the stored dtype, and the process setup of both entries.
 *
 *  The backend-neutral surface - settings, checkpoints, dataset streaming, telemetry, result
 *  judgment - lives in @c dense.cuh; this header owns everything CUDA: one device, one
 *  @c dense_vram_index instantiation, batch-shaped run loops, and the compile-time cascade
 *  lowering memory space, generation ceiling, and metric below the stored dtype - the axis the
 *  shard translation units split. The cascade ends in a workload, so one pipeline serves both
 *  families; the retrieval one lives here.
 */
#pragma once

#include <cmath> // `std::abs`

#include <algorithm> // `std::max`, `std::min`, `std::copy_n`, `std::ranges::find`
#include <chrono>    // `std::chrono::steady_clock`
#include <memory>    // `std::unique_ptr`
#include <string>    // `std::string`
#include <vector>    // `std::vector`

#include <cuda_runtime.h>
#include <fmt/format.h>

#include <usearch/dense/vram_indexes.cuh>

#include "dense.cuh"
#include "utilities/persisted_dataset.hpp"

namespace unum::usearch::bench::dense {

#pragma region CUDA Pipeline

/** Compute ceilings the device kit chains adapt, ascending and named by NumKong's GPU tiers; the floor
 *  the table never lists is the Ampere kits. */
inline constexpr capability_choice<cuda_arch_t> cuda_compute_choices_k[] = {
    {"ada", cuda_arch_ada_k},
    {"hopper", cuda_arch_hopper_k},
    {"blackwell", cuda_arch_blackwell_k},
    {"blackwellrtx", cuda_arch_blackwellrtx_k},
};

/**
 *  @brief The GPU-side state one fully-typed benchmark run shares between its phases: the single
 *      device index, its one driving traversal context, and the embedded backend-neutral
 *      judgment results.
 *
 *  One device, one submission lane - the batch shape, not the thread count, fills the machine.
 */
template <typename index_type_>
struct cuda_bench_context {
    using index_t = index_type_;
    using scalar_t = typename index_t::scalar_t;
    using small_size_t = typename index_t::small_size_t;
    using traversal_t = typename index_t::dense_traversal_context_t;
    using vector_view_t = typename index_t::vector_view_t;
    using vectors_view_t = typename index_t::vectors_view_t;
    using node_id_t = typename index_t::node_id_t;

    static constexpr nk_kernel_kind_t kernel_kind_k = index_t::kernel_kind_k;

    /** The parsed CLI, borrowed for the run's lifetime. */
    settings_t const& settings;

    /** The CUDA ordinal every plane and launch of this run lands on. */
    int device = 0;

    /** The one index under measurement. */
    std::unique_ptr<index_t> index;

    /** The one context driving every batched verb. */
    std::unique_ptr<traversal_t> traversal;

    /** One accumulation lane - the launches are sequential on one stream. */
    telemetry_lane_t lane;

    /** The backend-neutral judgment state both phases fill. */
    bench_results_t results;

    cuda_bench_context(settings_t const& settings, int device) noexcept : settings(settings), device(device) {}
};

/** Builds the device index, reserves it for the viewed rows, restores an optional checkpoint, and
 *  sizes the judgment state - everything fallible, before any clock. */
template <typename streamed_dataset_type_>
bool prepare(streamed_dataset_type_& streamed) {
    using bench_t = typename streamed_dataset_type_::bench_t;
    using index_t = typename bench_t::index_t;
    using traversal_t = typename bench_t::traversal_t;
    using scalar_t = typename index_t::scalar_t;
    auto& bench = streamed.bench;
    std::size_t const rows = streamed.vectors.extent(0);
    std::size_t const dimensions = streamed.vectors.extent(1);

    connectivity_t const connectivity = connectivity_of(bench.settings.connectivity.value);
    if (connectivity == connectivity_unknown_k) {
        fmt::println("Unsupported --connectivity {}; pick 16, 32, or 128", bench.settings.connectivity.value);
        return false;
    }
    if (bench.settings.skip_vectors.value > rows) {
        fmt::println("--skip-vectors {} exceeds the {} viewed rows", bench.settings.skip_vectors.value, rows);
        return false;
    }

    using vectors_allocator_t = typename index_t::vectors_allocator_t;
    using neighbors_allocator_t = typename index_t::neighbors_allocator_t;
    bench.index = std::make_unique<index_t>(vectors_allocator_t {bench.device}, neighbors_allocator_t {bench.device});
    index_t& index = *bench.index;
    streamed.id_to_row.resize(rows);

    // Reserving first freezes the CLI geometry a checkpoint then has to agree with.
    auto const [capacity, status] = index.reserve({rows, connectivity, dimensions});
    if (failed(status)) {
        fmt::println("Failed to reserve {} × {}d on device {}", rows, dimensions, bench.device);
        return false;
    }

    if (!bench.settings.load_name.empty() &&
        !load_checkpoint(bench.settings, index, dimensions, std::span {streamed.id_to_row}, 0, 1))
        return false;

    // A base that misses the skipped prefix would judge recall against rows the graph never saw.
    if (index.size() != bench.settings.skip_vectors.value) {
        fmt::println("Loaded index covers {} rows, but --skip-vectors says {}", index.size(),
                     bench.settings.skip_vectors.value);
        return false;
    }

    bench.traversal = std::make_unique<traversal_t>(typename traversal_t::allocator_t {nullptr, bench.device});
    std::size_t const search_count = bench.settings.self_recall == switch_t::present_k ? streamed.vectors.extent(0)
                                                                                       : streamed.queries.extent(0);
    bench.results.hit_rows = nk::matrix<u32_t>::zeros({search_count, bench.settings.top_k.value}).value;
    bench.results.hit_distances = nk::matrix<dense_distance_t>::zeros({search_count, bench.settings.top_k.value}).value;
    if (bench.results.hit_rows.empty() || bench.results.hit_distances.empty()) {
        fmt::println("Failed to allocate the per-query hit matrices");
        return false;
    }
    bench.results.hit_counts.resize(search_count);
    bench.results.latencies_us.resize(search_count);
    return true;
}

/** `--batch` rows per launch, capped by @p total_rows: the insert path drains through a bounded admission window, so
 *  bigger batches only amortize transfers. */
inline std::size_t streamed_batch_rows(settings_t const& settings, std::size_t total_rows) noexcept {
    return std::min(settings.batch.value, total_rows);
}

/** Stages up to @p batch_rows rows of @p source starting at @p first_row into one device-ready
 *  host matrix - a zero-copy view when the scalars match, or the trampoline's contiguous plane
 *  after per-row rescale and cast. Both phases share it. */
template <typename streamed_dataset_type_>
typename streamed_dataset_type_::bench_t::vectors_view_t stage_batch(
    streamed_dataset_type_& streamed, nk::matrix_view<typename streamed_dataset_type_::dataset_scalar_t> source,
    std::size_t first_row, std::size_t batch_rows) {
    using vectors_view_t = typename streamed_dataset_type_::bench_t::vectors_view_t;
    std::size_t const dimensions = source.extent(1);
    if constexpr (streamed_dataset_type_::zero_copy_k)
        return vectors_view_t {source.row(first_row).data(), {batch_rows, dimensions}};
    else {
        for (std::size_t row = 0; row != batch_rows; ++row)
            streamed.stage(source.row(first_row + row).as_vector(), 0, row);
        return vectors_view_t {streamed.staging.row(0).row(0).data(), {batch_rows, dimensions}};
    }
}

/** Streams the un-ingested topping into the index, `--batch` rows per launch, and keeps the
 *  ID → row bookkeeping dense. */
template <typename streamed_dataset_type_>
bool run_insertion(streamed_dataset_type_& streamed) {
    using bench_t = typename streamed_dataset_type_::bench_t;
    using index_t = typename bench_t::index_t;
    using node_id_t = typename bench_t::node_id_t;
    using small_size_t = typename bench_t::small_size_t;
    using scalar_t = typename streamed_dataset_type_::scalar_t;
    auto& bench = streamed.bench;
    std::size_t const rows = streamed.vectors.extent(0);
    std::size_t const inserted_rows = rows - bench.settings.skip_vectors.value;
    if (!inserted_rows) {
        fmt::println("\nInsertions");
        fmt::println("  skipped: the loaded base already covers all {} rows", rows);
        return true;
    }

    typename index_t::insert_config_t config;
    config.expansion_factor = static_cast<small_size_t>(bench.settings.expansion_add.value);
    std::size_t const batch_rows = streamed_batch_rows(bench.settings, inserted_rows);
    std::vector<node_id_t> batch_ids(batch_rows);

    progress_bar_t progress {"inserting", inserted_rows, bench.settings.terminal};
    auto started = steady_clock_t::now();
    for (std::size_t offset = 0; offset < inserted_rows;) {
        std::size_t const first_row = bench.settings.skip_vectors.value + offset;
        std::size_t const batch = std::min(batch_rows, inserted_rows - offset);
        auto const staged = stage_batch(streamed, streamed.vectors, first_row, batch);
        auto const [result, status] = bench.index->insert(staged, {batch_ids.data(), batch}, *bench.traversal, config);
        if (failed(status)) {
            fmt::println("Insertion failed at row {}: status {}", first_row, int(status));
            return false;
        }
        for (std::size_t row = 0; row != batch; ++row)
            streamed.id_to_row[batch_ids[row]] = static_cast<u32_t>(first_row + row);
        bench.lane.visited_members += result.visited_members;
        bench.lane.computed_distances += result.computed_distances;
        bench.lane.restarts += result.restarts;
        bench.lane.saturated_walks += result.visited_saturations;
        offset += batch;
        progress.tick(offset);
    }
    progress.finish();
    double const seconds = stdc::duration<double>(steady_clock_t::now() - started).count();

    fmt::println("\nInsertions");
    fmt::println("  shape: {} × {}d", inserted_rows, streamed.vectors.extent(1));
    if (bench.settings.skip_vectors.value) fmt::println("  topping: {} loaded rows", bench.settings.skip_vectors.value);
    fmt::println("  kit: {} · {} · {} kit · device {}", scalar_t::dtype_name(), nk_kernel_name(bench.settings.kernel),
                 bit_name(cuda_compute_choices_k, index_t::picked_capability_k, "ampere"), bench.device);
    fmt::println("  batch: {} rows per launch", batch_rows);
    fmt::println("  wall: {:.2f} s", seconds);
    fmt::println("  throughput: {} IPS", humanized(double(inserted_rows) / seconds));
    fmt::println("  distances: {}", humanized(bench.lane.computed_distances));
    fmt::println("  visited: {}", humanized(bench.lane.visited_members));
    fmt::println("  restarts: {}", bench.lane.restarts);
    fmt::println("  saturated: {}", bench.lane.saturated_walks);
    return true;
}

/** Re-selects every node's neighbors, pass after pass, then re-links whatever no search can
 *  reach; a pass that re-wires nothing ends the phase. */
template <typename streamed_dataset_type_>
bool run_refinement(streamed_dataset_type_& streamed) {
    using bench_t = typename streamed_dataset_type_::bench_t;
    using index_t = typename bench_t::index_t;
    using node_id_t = typename bench_t::node_id_t;
    using small_size_t = typename bench_t::small_size_t;
    auto& bench = streamed.bench;
    std::size_t const max_refines = bench.settings.max_refines.value;
    std::size_t const nodes = bench.index->size();
    if (!max_refines || !nodes) return true;

    typename index_t::refine_config_t config;
    config.expansion_factor = static_cast<small_size_t>(bench.settings.expansion_refine.value);

    refinement_report_t report;
    report.nodes = nodes;
    report.expansion = bench.index->non_zero_expansion_factor(config);
    report.threads = 1;
    std::vector<std::size_t> changed_per_pass;
    changed_per_pass.reserve(max_refines);
    progress_bar_t progress {"refining", max_refines, bench.settings.terminal};
    auto started = steady_clock_t::now();
    for (std::size_t pass = 0; pass != max_refines; ++pass) {
        auto const [refinement, refine_status] = bench.index->refine({0, static_cast<node_id_t>(nodes)},
                                                                     *bench.traversal, config);
        if (failed(refine_status)) {
            fmt::println("Refinement failed on pass {}: status {}", pass, int(refine_status));
            return false;
        }
        changed_per_pass.push_back(refinement.changed_nodes);
        report.sums.visited_members += refinement.visited_members;
        report.sums.computed_distances += refinement.computed_distances;
        report.sums.restarts += refinement.restarts;
        report.islands_found += refinement.islands_found;
        report.islands_reconnected += refinement.islands_reconnected;
        report.reachable_nodes = refinement.reachable_nodes;
        report.sweeps = refinement.sweeps;
        progress.tick(pass + 1);
        if (!refinement.changed_nodes) break;
    }
    progress.finish();
    // Every pass heals what it strands, so the passes' wall time holds the healing too.
    report.refining_seconds = stdc::duration<double>(steady_clock_t::now() - started).count();
    report.changed_per_pass = changed_per_pass;
    report_refinement_metrics(report);
    return true;
}

/** Serializes the device index as three RetriEval-layout matrices - vectors, neighbors, and the
 *  ID → row map. */
template <typename streamed_dataset_type_>
bool save_index(streamed_dataset_type_& streamed) {
    auto& bench = streamed.bench;
    auto started = steady_clock_t::now();
    std::size_t const total_bytes = save_checkpoint(bench.settings, *bench.index, std::span {streamed.id_to_row}, 0, 1);
    if (!total_bytes) return false;
    double const seconds = stdc::duration<double>(steady_clock_t::now() - started).count();
    fmt::println("Saved the index: {:.2f} GB in {:.2f} s", gigabytes(total_bytes), seconds);
    return true;
}

/** Answers every query through the batch verb, then translates node IDs into dataset rows for the
 *  shared judgment; a batch-shaped verb has no per-query clock, so the report prints the mean. */
template <typename streamed_dataset_type_>
bool run_search(streamed_dataset_type_& streamed) {
    using bench_t = typename streamed_dataset_type_::bench_t;
    using index_t = typename bench_t::index_t;
    using node_id_t = typename bench_t::node_id_t;
    using small_size_t = typename bench_t::small_size_t;
    auto& bench = streamed.bench;
    auto& source = bench.settings.self_recall == switch_t::present_k ? streamed.vectors : streamed.queries;
    std::size_t const query_count = source.extent(0);
    std::size_t const top_k = bench.settings.top_k.value;
    std::size_t const node_count = bench.index->size();
    bench.lane = telemetry_lane_t {};

    typename index_t::search_config_t config;
    config.expansion_factor = static_cast<small_size_t>(bench.settings.expansion_search.value);
    config.count = static_cast<small_size_t>(top_k);
    config.synchronization = synchronization_t::immutable_k;
    std::size_t const batch_rows = streamed_batch_rows(bench.settings, query_count);
    auto found_ids = nk::matrix<node_id_t>::zeros({batch_rows, top_k}).value;
    auto found_distances = nk::matrix<dense_distance_t>::zeros({batch_rows, top_k}).value;
    if (found_ids.empty() || found_distances.empty()) {
        fmt::println("Failed to allocate the batch result matrices");
        return false;
    }

    progress_bar_t progress {"searching", query_count, bench.settings.terminal};
    double seconds = 0;
    for (std::size_t offset = 0; offset < query_count;) {
        std::size_t const batch = std::min(batch_rows, query_count - offset);
        auto const staged = stage_batch(streamed, source, offset, batch);

        auto started = steady_clock_t::now();
        auto const [result, status] = bench.index->search(
            staged, nk::matrix_span<node_id_t> {found_ids.data(), {batch, top_k}},
            nk::matrix_span<dense_distance_t> {found_distances.data(), {batch, top_k}}, *bench.traversal, config);
        double const batch_seconds = stdc::duration<double>(steady_clock_t::now() - started).count();
        if (failed(status)) {
            fmt::println("Search failed at query {}: status {}", offset, int(status));
            return false;
        }
        seconds += batch_seconds;
        bench.lane.visited_members += result.visited_members;
        bench.lane.computed_distances += result.computed_distances;
        bench.lane.restarts += result.restarts;
        bench.lane.saturated_walks += result.visited_saturations;

        for (std::size_t row = 0; row != batch; ++row) {
            u32_t* const merged_rows = bench.results.hit_rows.row(offset + row).data();
            auto* const merged_distances = bench.results.hit_distances.row(offset + row).data();
            auto const found_row_ids = found_ids.view().row(row);
            auto const found_row_distances = found_distances.view().row(row);
            std::size_t merged_count = 0;
            for (std::size_t rank = 0; rank != top_k; ++rank) {
                node_id_t const id = found_row_ids[rank];
                if (static_cast<std::size_t>(id) >= node_count) break; // Sentinel-padded tail.
                merged_rows[merged_count] = streamed.id_to_row[id];
                merged_distances[merged_count] = found_row_distances[rank];
                ++merged_count;
            }
            bench.results.hit_counts[offset + row] = static_cast<u32_t>(merged_count);
        }
        offset += batch;
        progress.tick(offset);
    }
    progress.finish();

    // The empty latency span asks the shared report for the batch-uniform mean.
    report_search_metrics(bench.settings, query_count, seconds, 1, bench.lane, {});
    judge_hits(bench.settings, bench.results, streamed.ground_truth, query_count);
    return true;
}

/** The retrieval workload the cascade ends in: batched search over the built index, judged against
 *  the ground truth. A workload says whether it pins the dot kernel and runs once the index is
 *  built. */
struct retrieval_workload_t {
    static constexpr bool dot_only_k = false;

    template <typename streamed_dataset_type_>
    bool run(streamed_dataset_type_& streamed) const {
        return run_search(streamed);
    }
};

#pragma endregion

#pragma region Dispatch Cascade

/** The innermost hop: binds the concrete CUDA-index instantiation, wires the typed source views
 *  and the staging trampoline, builds the index in checkpoint order, then runs @p workload. */
template <typename dataset_scalar_type_, nk_dtype_t stored_dtype_, nk_kernel_kind_t kernel_, cuda_arch_t capability_,
          typename plane_allocator_, typename workload_type_>
int stage_and_run(utilities::persisted_dataset<dataset_scalar_type_> const& dataset, settings_t const& settings,
                  int device, workload_type_ const& workload) {
    using index_t = dense_vram_index<stored_dtype_, kernel_, capability_, plane_allocator_, plane_allocator_,
                                     cuda_device_allocator_t>;
    using bench_context_t = cuda_bench_context<index_t>;
    using streamed_t = streamed_dataset<dataset_scalar_type_, bench_context_t>;
    using scalar_t = typename bench_context_t::scalar_t;

    std::size_t const dimensions = dataset.dimensions();
    std::size_t rows = dataset.vectors().extent(0);
    if (settings.max_vectors.value) rows = std::min(rows, settings.max_vectors.value);

    // Self-queries keep the harness usable on a lone vectors file, without recall.
    nk::matrix_view<dataset_scalar_type_> const source_queries =
        dataset.has_queries() ? dataset.queries()
                              : nk::matrix_view<dataset_scalar_type_> {dataset.vectors().data(),
                                                                       std::min<std::size_t>(1000, rows), dimensions};

    bench_context_t bench {settings, device};
    streamed_t streamed {bench, {dataset.vectors().data(), rows, dimensions}, source_queries};

    // Owns the staging trampoline plane, @b [1,batch,dimensions].
    nk::tensor<scalar_t, std::allocator<scalar_t>, 3> staging;
    // Owns the rescaling plane @c streamed_dataset::rescaling views.
    nk::tensor<nk::f32_t, std::allocator<nk::f32_t>, 3> rescaling_planes;
    if constexpr (!streamed_t::zero_copy_k) {
        // The trampoline serves both phases, so it follows the larger of the two row counts.
        std::size_t const batch_rows = streamed_batch_rows(settings, std::max(rows, source_queries.extent(0)));
        staging = nk::tensor<scalar_t, std::allocator<scalar_t>, 3>::zeros({1, batch_rows, dimensions}).value;
        if (staging.empty()) {
            fmt::println("Failed to allocate {} staging rows of {}d {}", batch_rows, dimensions,
                         scalar_t::dtype_name());
            return 1;
        }
        streamed.staging = staging.span();

        if constexpr (streamed_t::scaled_k) {
            rescaling_planes =
                nk::tensor<nk::f32_t, std::allocator<nk::f32_t>, 3>::zeros({1, batch_rows, dimensions}).value;
            if (rescaling_planes.empty()) {
                fmt::println("Failed to allocate {} rescaling rows of {}d f32", batch_rows, dimensions);
                return 1;
            }
            streamed.rescaling = rescaling_planes.span();
            streamed.establish_scale();
        }
    }

    if (dataset.has_ground_truth() && !settings.max_vectors.value) streamed.ground_truth = dataset.neighbors();
    else if (dataset.has_ground_truth())
        fmt::println("Ground truth ignored: --max-vectors truncates the graph it was computed for");

    if (!prepare(streamed)) return 1;
    if (!run_insertion(streamed)) return 1;
    if (!run_refinement(streamed)) return 1;
    if (!settings.save_name.empty() && !save_index(streamed)) return 1;
    return workload.run(streamed) ? 0 : 1;
}

/** Kernel axis: lowers the runtime metric to a compile-time kernel kind, the terminal hop into
 *  @ref stage_and_run; a dot-only workload pins the dot kernel instead. */
template <typename dataset_scalar_type_, nk_dtype_t stored_dtype_, cuda_arch_t capability_, typename plane_allocator_,
          typename workload_type_>
int run_with_metric(utilities::persisted_dataset<dataset_scalar_type_> const& dataset, settings_t const& settings,
                    int device, workload_type_ const& workload) {
    if constexpr (workload_type_::dot_only_k)
        return stage_and_run<dataset_scalar_type_, stored_dtype_, nk_kernel_dot_k, capability_, plane_allocator_>(
            dataset, settings, device, workload);
    else
        return dispatch_kernel<stored_dtype_>(
            settings.kernel,
            [&](auto kernel) {
                return stage_and_run<dataset_scalar_type_, stored_dtype_, kernel.value, capability_, plane_allocator_>(
                    dataset, settings, device, workload);
            },
            [&](auto metric) {
                fmt::println("Unsupported metric {} for {} vectors", nk_kernel_name(metric), dtype_name(stored_dtype_));
                return 1;
            });
}

/** Compute axis: the named table row's ceiling instantiates every cell below, so two runs on one device
 *  compare generations side by side; past the tail sit the Ampere kits every GPU runs. */
template <typename dataset_scalar_type_, nk_dtype_t stored_dtype_, typename plane_allocator_, typename workload_type_>
int run_with_compute(utilities::persisted_dataset<dataset_scalar_type_> const& dataset, settings_t const& settings,
                     int device, cuda_arch_t max_compute, workload_type_ const& workload) {
    auto const named = std::ranges::find(cuda_compute_choices_k, max_compute, &capability_choice<cuda_arch_t>::bit);
    return dispatch_index<std::size(cuda_compute_choices_k)>(
        static_cast<std::size_t>(named - std::begin(cuda_compute_choices_k)),
        [&](auto row) {
            return run_with_metric<dataset_scalar_type_, stored_dtype_, cuda_compute_choices_k[row.value].bit,
                                   plane_allocator_>(dataset, settings, device, workload);
        },
        [&] {
            return run_with_metric<dataset_scalar_type_, stored_dtype_, cuda_arch_ampere_k, plane_allocator_>(
                dataset, settings, device, workload);
        });
}

/**
 *  @brief The shard hop: everything below one stored dtype - the memory axis picking the plane
 *      allocator, the generation ceiling, and the metric - so the per-dtype shard translation
 *      units carry the whole device instantiation grid of their dtype.
 *
 *  Device-space planes stay in VRAM through the virtual-memory backend; managed-space planes
 *  oversubscribe it through the pager, the bigger-than-VRAM build path. The cascade ends in the
 *  retrieval workload.
 */
template <typename dataset_scalar_type_, nk_dtype_t stored_dtype_>
int run_with_stored_dtype(utilities::persisted_dataset<dataset_scalar_type_> const& dataset, settings_t const& settings,
                          int device, cuda_arch_t max_compute, cuda_address_space_t plane_space) {
    return plane_space == cuda_address_space_t::managed_k
               ? run_with_compute<dataset_scalar_type_, stored_dtype_, cuda_managed_allocator<std::byte>>( //
                     dataset, settings, device, max_compute, retrieval_workload_t {})
               : run_with_compute<dataset_scalar_type_, stored_dtype_, cuda_virtual_memory_allocator<std::byte>>( //
                     dataset, settings, device, max_compute, retrieval_workload_t {});
}

#pragma endregion

#pragma region Command Line

/** The device knobs the CUDA binary parses beside @c settings_t, before any dtype is known. */
struct cuda_cli_t {

    /** The CUDA ordinal every plane and launch lands on. */
    count_t device {.minimum = 0};

    /** The `--max-compute` name, resolved against @ref cuda_compute_choices_k once parsed. */
    std::string_view max_compute_name;

    /** Whether the index planes live in managed memory oversubscribing VRAM, instead of VRAM. */
    switch_t managed_memory = switch_t::absent_k;
};

/**
 *  @brief The flags only the CUDA binary takes: the device and its memory, the device group, and the launch shape.
 *
 *  @verbatim
 *  --device           0                    CUDA device ordinal
 *  --devices          none                 Ordinals for the multi-index pipeline, like 0,1
 *  --merge            off                  Stitch the shards into one graph after ingest
 *  --expansion-merge  --expansion-add      Candidate pool while merging
 *  --max-compute      the newest compiled  Compute ceiling by GPU generation, or ampere
 *  --managed-memory   off                  Managed-memory planes oversubscribing VRAM
 *  --batch            65536                Rows and queries per launch
 *  @endverbatim
 */
inline std::array<flag_t, 7> cuda_flags(settings_t& settings, cuda_cli_t& device_cli) noexcept {
    return {{
        {"--device", "id", "CUDA device ordinal; 0 by default", &device_cli.device},
        {"--devices", "list", "comma-separated ordinals for the multi-index pipeline: one shard per device",
         &settings.devices},
        {"--merge", "", "stitch the shards into one graph after ingest; single-walk queries", &settings.merge},
        {"--expansion-merge", "breadth", "merge candidate pool; inherits --expansion-add by default",
         &settings.expansion_merge},
        {"--max-compute", "name", "compute ceiling by GPU generation, like `hopper`, or `ampere`; newest by default",
         &device_cli.max_compute_name},
        {"--managed-memory", "", "managed-memory planes: oversubscribe VRAM through the pager, host-RAM ceiling",
         &device_cli.managed_memory},
        {"--batch", "rows", "rows & queries per launch; 65536 by default", &settings.batch},
    }};
}

/** The process setup of the CUDA benchmark main: reads the flags of @p binary_name, probes the device, resolves the
 *  compute ceiling, prints both, and opens the dataset with its file-native scalar for @p on_dataset - handed the
 *  dataset, the settings, the device, the ceiling, and the plane space. */
int run_cuda_bench(int argc, char const* const* argv, char const* binary_name, auto on_dataset) {
    settings_t settings;
    cuda_cli_t device_cli;
    std::array<flag_t, 17> const common = common_flags(settings);
    std::array<flag_t, 7> const cuda = cuda_flags(settings, device_cli);
    std::vector<flag_t> flags(common.begin(), common.end());
    flags.insert(flags.end(), cuda.begin(), cuda.end());
    parse_or_exit(argc, argv, fmt::format("{} --vectors <path> [flags]", binary_name), flags);
    resolve_common_names(settings);

    // A named ceiling caps which kit-chain rows the cascade may select; "ampere" keeps just the floor.
    // Capping above what this binary compiled is harmless: a row also needs its tier compiled in.
    cuda_arch_t max_compute = cuda_compute_choices_k[std::size(cuda_compute_choices_k) - 1].bit;
    if (device_cli.max_compute_name == "ampere") max_compute = cuda_arch_ampere_k;
    else if (!device_cli.max_compute_name.empty()) {
        std::size_t const named = index_of_name(cuda_compute_choices_k, device_cli.max_compute_name);
        if (named == std::size(cuda_compute_choices_k)) {
            fmt::println(
                stderr, "--max-compute=\"{}\" does not parse, expected one of ampere, {}", device_cli.max_compute_name,
                fmt::join(cuda_compute_choices_k | std::views::transform(&capability_choice<cuda_arch_t>::name), ", "));
            std::exit(1);
        }
        max_compute = cuda_compute_choices_k[named].bit;
    }

    int devices_count = 0;
    int const device = static_cast<int>(device_cli.device.value);
    if (cudaGetDeviceCount(&devices_count) != cudaSuccess || device_cli.device.value >= std::size_t(devices_count)) {
        fmt::println(stderr, "--device=\"{}\" does not parse, expected an ordinal below {}", device_cli.device.value,
                     devices_count);
        return 1;
    }
    cudaSetDevice(device);
    cudaDeviceProp properties = {};
    cudaGetDeviceProperties(&properties, device);
    fmt::println("USearch {}.{}.{}", USEARCH_VERSION_MAJOR, USEARCH_VERSION_MINOR, USEARCH_VERSION_PATCH);
    fmt::println("- Device {}: {} · {} SMs · {:.2f} GB", device, properties.name, properties.multiProcessorCount,
                 gigabytes(properties.totalGlobalMem));
    print(settings);
    fmt::println("- Device: {}", device);
    fmt::println("- Devices: {}", settings.devices.size ? fmt::format("{}", fmt::join(settings.devices.view(), ","))
                                                        : std::string("none"));
    fmt::println("- Merge: {}", settings.merge == switch_t::present_k);
    fmt::println("- Max compute: {}", bit_name(cuda_compute_choices_k, max_compute, "ampere"));
    fmt::println("- Managed memory: {}", device_cli.managed_memory == switch_t::present_k);
    fmt::println("- Batch: {}", settings.batch.value);

    cuda_address_space_t const plane_space = device_cli.managed_memory == switch_t::present_k
                                                 ? cuda_address_space_t::managed_k
                                                 : cuda_address_space_t::device_k;

    // File-scalar axis: lower the file's dtype to the C++ wrapper the run is instantiated over,
    // through NumKong's own `type_for` - the same paradigm the typed cascade rides.
    return dispatch_file_dtype(
        utilities::dtype_of_extension(settings.vectors_path),
        [&](auto dtype) {
            using dataset_scalar_t = typename nk::type_for<dtype.value>::type;
            auto opened = utilities::persisted_dataset<dataset_scalar_t>::open(
                settings.vectors_path, settings.queries_path, settings.neighbors_path);
            if (opened.status != status_t::success_k) {
                fmt::println("Failed to open the dataset: {}", settings.vectors_path);
                return 1;
            }
            return on_dataset(opened.value, settings, device, max_compute, plane_space);
        },
        [&](auto dtype) {
            fmt::println("Unsupported dataset dtype {} for the CUDA kits", dtype_name(dtype));
            return 1;
        });
}

#pragma endregion

} // namespace unum::usearch::bench::dense
