/**
 *  @file bench/dense/vram_index_search.cuh
 *  @author Ash Vardanian
 *  @date July 22, 2026
 *  @brief Profiling harness for searching the GPU-resident @c dense::vram_nvidia_index over real
 *      persisted datasets, shared by the per-dtype benchmark shards.
 *
 *  The backend-neutral surface - settings, checkpoints and dataset streaming - lives in @c dense.cuh,
 *  the telemetry and result judgment in @c dense_search.cuh; this header owns everything CUDA: the
 *  device options and the process setup every CUDA binary runs, one device, one
 *  @c dense::vram_nvidia_index instantiation, batch-shaped run loops, and the compile-time cascade
 *  lowering memory space, generation ceiling, and metric below the stored dtype - the axis the
 *  shard translation units split. The cascade ends in a workload, so one pipeline serves the search
 *  and algebra families, and the multi-device binaries reuse its streaming.
 *
 *  The shards compile with the host compiler alone; each binary's entry unit compiles host-only
 *  through Clang's CUDA mode and once more with NVCC device-only, into the binary's own fatbin of
 *  every ceiling's kernels, which every unit finds through the @c dense::cuda_fatbin
 *  specializations below.
 */
#pragma once

#include <algorithm> // `std::max`, `std::min`, `std::ranges::find`, `std::ranges::any_of`
#include <chrono>    // `std::chrono::steady_clock`
#include <memory>    // `std::unique_ptr`
#include <span>      // `std::span`
#include <utility>   // `std::index_sequence` - every compiled dtype of the device pass
#include <vector>    // `std::vector`

#include <fmt/format.h>

#include "usearch/dense/vram_nvidia_indexes.hpp"
#include "usearch/utilities/vram_device.hpp"

#if defined(__CUDACC__)
#include "usearch/dense/vram_nvidia_index.cuh"
#endif

#include "dense_search.cuh"
#include "utilities/persisted_dataset.hpp"

/** The binary's own fatbin, linked in as raw bytes from its entry unit's device pass. */
extern "C" us::utilities::byte_t const bench_vram_nvidia_bundle_start[], bench_vram_nvidia_bundle_end[];

namespace unum::usearch::bench::dense {

#pragma region Options and Styling

/** NumKong's NVIDIA capabilities the device kit chains adapt, ascending; the floor the table never
 *  lists is the Ampere kits, and @c cuda_ceiling_of turns a row into the mask it instantiates. */
inline constexpr capability_choice<nk_capability_t> cuda_compute_choices_k[] = {
    {"ada", nk_cap_ada_k},
    {"hopper", nk_cap_hopper_k},
    {"blackwell", nk_cap_blackwell_k},
    {"blackwellrtx", nk_cap_blackwellrtx_k},
};

/** The NVIDIA capabilities one row instantiates: the Ampere & Ada bits accumulate, as NumKong
 *  detects them, while @p generation alone stands for Hopper, Blackwell or RTX Blackwell, which run
 *  only their own kits. */
constexpr nk_capability_t cuda_ceiling_of(nk_capability_t generation) noexcept {
    nk_capability_t const ampere = nk_cap_cuda_k | nk_cap_ampere_k;
    return generation == nk_cap_ampere_k ? ampere : ampere | nk_cap_ada_k | generation;
}

/** The binary's fatbin, which carries every ceiling's kernels. */
inline bytes_view_t cuda_bench_fatbin() noexcept {
    return {bench_vram_nvidia_bundle_start,
            static_cast<std::size_t>(bench_vram_nvidia_bundle_end - bench_vram_nvidia_bundle_start)};
}

/** The storage dtypes in the library's ladder order, of which a build compiles the first
 *  @c USEARCH_DENSE_DTYPES. */
inline constexpr nk_dtype_t cuda_dtypes_ladder_k[] = {nk_bf16_k, nk_u1_k, nk_i8_k, nk_e4m3_k, nk_f16_k, nk_u8_k,
                                                      nk_e5m2_k, nk_i4_k, nk_u4_k, nk_e3m2_k, nk_e2m3_k};

#if defined(__CUDACC__)

/** Instantiates every ceiling's kernels for @p dtype_, with the attends over each of
 *  @p lookups_. */
template <nk_dtype_t dtype_, u32_t... lookups_>
__device__ void instantiate_cuda_ceilings() noexcept {
    cuda_instantiate<cuda_ceiling_of(nk_cap_ampere_k), dtype_, lookups_...>();
    [&]<std::size_t... rows_>(std::index_sequence<rows_...>) {
        (cuda_instantiate<cuda_ceiling_of(cuda_compute_choices_k[rows_].bit), dtype_, lookups_...>(), ...);
    }(std::make_index_sequence<std::size(cuda_compute_choices_k)> {});
}

/** Instantiates the kernels of every compiled dtype under every ceiling, with the attends over each
 *  of @p lookups_ - what a binary's entry unit compiles into its fatbin. */
template <u32_t... lookups_>
__device__ void instantiate_cuda_kernels() noexcept {
    [&]<std::size_t... dtypes_>(std::index_sequence<dtypes_...>) {
        (instantiate_cuda_ceilings<cuda_dtypes_ladder_k[dtypes_], lookups_...>(), ...);
    }(std::make_index_sequence<USEARCH_DENSE_DTYPES> {});
}

#endif // defined(__CUDACC__)

/** The device knobs the CUDA binaries parse beside @c settings_t, before any dtype is known; the
 *  devices themselves come from @c USEARCH_DEVICES. */
struct cuda_cli_t {

    /** The `--max-compute` name, resolved against @ref cuda_compute_choices_k once parsed. */
    std::string_view max_compute_name;

    /** Whether the index's flat matrices live in managed memory oversubscribing VRAM, instead of
     *  VRAM. */
    switch_t managed_memory = switch_t::absent_k;
};

/**
 *  @brief The flags every CUDA binary takes: the compute ceiling and the device memory.
 *
 *  @verbatim
 *  --max-compute      the newest compiled  Compute ceiling by GPU generation, or ampere
 *  --managed-memory   off                  Managed-memory flat matrices oversubscribing VRAM
 *  @endverbatim
 */
inline std::array<flag_t, 2> cuda_flags(cuda_cli_t& device_cli) noexcept {
    return {{
        {"--max-compute", "name", "compute ceiling by GPU generation, like `hopper`, or `ampere`; newest by default",
         &device_cli.max_compute_name},
        {"--managed-memory", "", "managed-memory flat matrices: oversubscribe VRAM through the pager, host-RAM ceiling",
         &device_cli.managed_memory},
    }};
}

#pragma endregion

#pragma region Pipeline

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
    using traversal_t = typename index_t::traversal_context_t;
    using vector_view_t = typename index_t::vector_view_t;
    using vectors_view_t = typename index_t::vectors_view_t;
    using node_id_t = typename index_t::node_id_t;

    /** The parsed CLI, borrowed for the run's lifetime. */
    settings_t const& settings;

    /** The CUDA ordinal every flat matrix and launch of this run lands on. */
    int device = 0;

    /** The metric the index scores by - the CLI's, or dot where the workload pins it. */
    nk_kernel_kind_t kernel;

    /** The one index under measurement. */
    std::unique_ptr<index_t> index;

    /** The one stream on @c device every transfer and launch of this run rides. */
    cuda_executor_t executor;

    /** The one context driving every batched verb, over the executor's stream. */
    std::unique_ptr<traversal_t> traversal;

    /** One accumulation lane - the launches are sequential on one stream. */
    telemetry_lane_t lane;

    /** The backend-neutral judgment state both phases fill. */
    bench_results_t results;

    cuda_bench_context(settings_t const& settings, int device, nk_kernel_kind_t kernel) noexcept
        : settings(settings), device(device), kernel(kernel) {}
};

/** Builds the device index, reserves it for the viewed rows, restores an optional checkpoint, and
 *  builds the driving context - everything fallible, before any clock. */
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
    bench.index = std::make_unique<index_t>(bench.kernel, vectors_allocator_t {bench.device},
                                            neighbors_allocator_t {bench.device});
    index_t& index = *bench.index;
    streamed.id_to_row.resize(rows);

    // Reserving first freezes the CLI geometry a checkpoint then has to agree with.
    auto const [capacity, status] = index.reserve({rows, connectivity, dimensions});
    if (failed(status)) {
        fmt::println("Failed to reserve {} × {}d on device {}", rows, dimensions, bench.device);
        return false;
    }

    auto [executor, spawn_status] = cuda_executor_t::spawn(std::span<int const> {&bench.device, 1});
    if (failed(spawn_status)) {
        fmt::println("Failed to open a stream on device {}", bench.device);
        return false;
    }
    bench.executor = std::move(executor);
    if (!bench.settings.load_name.empty() &&
        !load_checkpoint(bench.settings, index, std::span {streamed.id_to_row}, bench.executor.stream(0)))
        return false;

    // A base that misses the skipped prefix would judge recall against rows the graph never saw.
    if (index.size() != bench.settings.skip_vectors.value) {
        fmt::println("Loaded index covers {} rows, but --skip-vectors says {}", index.size(),
                     bench.settings.skip_vectors.value);
        return false;
    }

    bench.traversal = std::make_unique<traversal_t>(typename traversal_t::allocator_t {bench.executor.stream(0)});
    return true;
}

/** Views up to @p batch_rows rows of @p source starting at @p first_row in the file's scalar,
 *  which the index casts on the device; only a rescaled source stages on the host, through the
 *  trampoline's contiguous rows. Both phases share it. */
template <typename streamed_dataset_type_>
rows_view_t stage_batch(streamed_dataset_type_& streamed,
                        nk::matrix_view<typename streamed_dataset_type_::dataset_scalar_t> source,
                        std::size_t first_row, std::size_t batch_rows) {
    using dataset_scalar_t = typename streamed_dataset_type_::dataset_scalar_t;
    using vectors_view_t = typename streamed_dataset_type_::bench_t::vectors_view_t;
    std::size_t const dimensions = source.extent(1);
    if constexpr (!streamed_dataset_type_::scaled_k)
        return nk::matrix_view<dataset_scalar_t> {source.row(first_row).data(), {batch_rows, dimensions}};
    else {
        for (std::size_t row = 0; row != batch_rows; ++row)
            streamed.stage(source.row(first_row + row).as_vector(), 0, row);
        return vectors_view_t {streamed.staging.row(0).row(0).data(), {batch_rows, dimensions}};
    }
}

/** Streams the un-ingested topping into the index, `--inserts-per-core` rows per multiprocessor in
 *  each launch, and keeps the ID → row bookkeeping dense. */
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
    std::size_t const batch_rows = insert_batch_rows(bench.settings, inserted_rows);
    std::vector<node_id_t> batch_ids(batch_rows);

    progress_bar_t progress {"inserting", inserted_rows, bench.settings.terminal};
    auto const phase_started = steady_clock_t::now();
    double seconds = 0;
    for (std::size_t offset = 0; offset < inserted_rows;) {
        std::size_t const first_row = bench.settings.skip_vectors.value + offset;
        std::size_t const batch = std::min(batch_rows, inserted_rows - offset);
        auto const staged = stage_batch(streamed, streamed.vectors, first_row, batch);
        auto const started = steady_clock_t::now();
        auto const [result, status] = bench.index->insert(staged, {batch_ids.data(), batch}, *bench.traversal, config);
        seconds += stdc::duration<double>(steady_clock_t::now() - started).count();
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
    double const preparing = stdc::duration<double>(steady_clock_t::now() - phase_started).count() - seconds;

    fmt::println("\nInsertions");
    fmt::println("  shape: {} × {}d", inserted_rows, streamed.vectors.extent(1));
    if (bench.settings.skip_vectors.value) fmt::println("  topping: {} loaded rows", bench.settings.skip_vectors.value);
    fmt::println("  kit: {} · {} · {} kit · device {}", scalar_t::dtype_name(), nk_kernel_name(bench.settings.kernel),
                 bit_name(cuda_compute_choices_k, index_t::picked_quads_capability_k, "ampere"), bench.device);
    fmt::println("  batch: {} rows per launch", batch_rows);
    fmt::println("  wall: {:.4f} s", seconds);
    fmt::println("  preparation: {:.4f} s", preparing);
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
        report.reachable_nodes = refinement.health.reachable_nodes;
        report.sweeps = refinement.health.sweeps;
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

/** Serializes the device index as one SafeGraphs monolith keyed by the ID → row map, the file every
 *  residency loads and the NVMe benches view. */
template <typename streamed_dataset_type_>
bool save_index(streamed_dataset_type_& streamed) {
    auto& bench = streamed.bench;
    auto started = steady_clock_t::now();
    std::size_t const total_bytes = save_checkpoint(bench.settings, *bench.index,
                                                    std::span<u32_t const> {streamed.id_to_row}, streamed.scale,
                                                    bench.executor.stream(0));
    if (!total_bytes) return false;
    double const seconds = stdc::duration<double>(steady_clock_t::now() - started).count();
    fmt::println("Saved the index: {:.2f} GB in {:.2f} s", gigabytes(total_bytes), seconds);
    return true;
}

/** Answers every query through the batch verb, ranked by its heads under an indexer, then translates node IDs into
 *  dataset rows for the shared judgment; a batch-shaped verb has no per-query clock, so the report prints the mean. */
template <typename streamed_dataset_type_>
bool run_search(streamed_dataset_type_& streamed) {
    using bench_t = typename streamed_dataset_type_::bench_t;
    using index_t = typename bench_t::index_t;
    using node_id_t = typename bench_t::node_id_t;
    using small_size_t = typename bench_t::small_size_t;
    using vectors_view_t = typename bench_t::vectors_view_t;
    auto& bench = streamed.bench;
    settings_t const& settings = bench.settings;
    std::size_t const dimensions = streamed.vectors.extent(1);
    std::size_t const top_k = settings.top_k.value;
    std::size_t const node_count = bench.index->size();
    bench_indexer_heads<typename index_t::scalar_t> heads;
    if (!open_indexer_heads(*bench.index, streamed, heads)) return false;
    std::size_t const query_count = searched_count(streamed, heads);
    if (!reserve_results(bench.results, query_count, top_k)) return false;

    typename index_t::search_config_t config;
    config.expansion_factor = static_cast<small_size_t>(settings.expansion_search.value);
    config.count = static_cast<small_size_t>(top_k);
    config.synchronization = synchronization_t::immutable_k;
    if (settings.scan == switch_t::present_k) config.traversal = traversal_t::scan_k;
    std::size_t const batch_rows = query_batch_rows(settings, query_count);
    auto found_ids = nk::matrix<node_id_t>::zeros({batch_rows, top_k}).value;
    auto found_distances = nk::matrix<distance_t>::zeros({batch_rows, top_k}).value;
    if (found_ids.empty() || found_distances.empty()) {
        fmt::println("Failed to allocate the batch result matrices");
        return false;
    }
    bench.lane = telemetry_lane_t {};

    progress_bar_t progress {"searching", query_count, settings.terminal};
    auto const phase_started = steady_clock_t::now();
    double seconds = 0;
    for (std::size_t offset = 0; offset < query_count;) {
        std::size_t const batch = std::min(batch_rows, query_count - offset);
        rows_view_t const staged = !heads.navigation.empty()
                                       ? rows_view_t {vectors_view_t {heads.navigation.data() + offset * dimensions,
                                                                      {batch, dimensions}}}
                                       : stage_batch(streamed, searched_rows(streamed), offset, batch);
        auto search = [&](auto const&... ranking) noexcept {
            return bench.index->search(
                staged, ranking..., nk::matrix_span<node_id_t> {found_ids.data(), {batch, top_k}},
                nk::matrix_span<distance_t> {found_distances.data(), {batch, top_k}}, *bench.traversal, config);
        };

        auto started = steady_clock_t::now();
        auto const [result, status] = [&]() noexcept {
            if constexpr (!nk::is_integral_dtype<typename index_t::scalar_t>())
                if (heads.engaged()) return search(heads.rows(offset, batch));
            return search();
        }();
        seconds += stdc::duration<double>(steady_clock_t::now() - started).count();
        if (failed(status)) {
            fmt::println("Search refused a batch of {} queries at query {}: {}", batch, offset,
                         status_to_string(status));
            return false;
        }
        bench.lane.visited_members += result.visited_members;
        bench.lane.computed_distances += result.computed_distances;
        bench.lane.restarts += result.restarts;
        bench.lane.saturated_walks += result.visited_saturations;

        for (std::size_t row = 0; row != batch; ++row) {
            std::size_t const query = offset + row;
            std::span<u32_t> const rows {bench.results.hit_rows.row(query).data(), top_k};
            std::span<distance_t> const distances {bench.results.hit_distances.row(query).data(), top_k};
            auto const found_row_ids = found_ids.view().row(row);
            auto const found_row_distances = found_distances.view().row(row);
            std::size_t count = 0;
            for (; count != top_k; ++count) {
                node_id_t const id = found_row_ids[count];
                if (static_cast<std::size_t>(id) >= node_count) break; // Sentinel-padded tail.
                rows[count] = streamed.id_to_row[id], distances[count] = found_row_distances[count];
            }
            bench.results.hit_counts[query] = static_cast<u32_t>(count);
        }
        offset += batch;
        progress.tick(offset);
    }
    progress.finish();
    double const preparing = stdc::duration<double>(steady_clock_t::now() - phase_started).count() - seconds;

    // The empty latency span asks the shared report for the batch-uniform mean.
    report_search_metrics(settings, query_count, seconds, 1, bench.lane, {});
    fmt::println("  preparation: {:.4f} s", preparing);
    judge_search(settings, heads, bench.results, streamed.ground_truth, streamed.vectors, query_count);
    return true;
}

/** The search workload the cascade ends in: batched search over the built index, judged against
 *  the ground truth. */
struct search_workload_t {

    /** Whether the workload is defined over the dot kernel alone, so the cascade pins it instead of
     *  dispatching the CLI's metric. */
    static constexpr bool dot_only_k = false;

    template <typename streamed_dataset_type_>
    bool run(streamed_dataset_type_& streamed) const {
        return run_search(streamed);
    }
};

#pragma endregion

#pragma region Dispatch Cascade

/** Wires @p bench to the typed source views and the staging trampoline a rescaled source casts
 *  through, then hands the streamed dataset to @p phases - the half of the innermost hop every
 *  device binary shares. */
template <typename dataset_scalar_type_, typename bench_context_type_, typename phases_type_>
int stream_and_run(utilities::persisted_dataset<dataset_scalar_type_> const& dataset, bench_context_type_& bench,
                   phases_type_&& phases) {
    using streamed_t = streamed_dataset<dataset_scalar_type_, bench_context_type_>;
    using scalar_t = typename bench_context_type_::scalar_t;
    settings_t const& settings = bench.settings;

    std::size_t const dimensions = dataset.dimensions();
    std::size_t rows = dataset.vectors().extent(0);
    if (settings.max_base_vectors.value) rows = std::min(rows, settings.max_base_vectors.value);

    // Self-queries keep the harness usable on a lone vectors file, without recall.
    nk::matrix_view<dataset_scalar_type_> const source_queries =
        dataset.has_queries() ? dataset.queries()
                              : nk::matrix_view<dataset_scalar_type_> {dataset.vectors().data(),
                                                                       std::min<std::size_t>(1000, rows), dimensions};

    streamed_t streamed {bench, {dataset.vectors().data(), rows, dimensions}, source_queries};

    // Owns the staging trampoline rows, `[1,batch,dimensions]`.
    nk::tensor<scalar_t, std::allocator<scalar_t>, 3> staging;
    // Owns the rescaling rows `streamed_dataset::rescaling` views.
    nk::tensor<nk::f32_t, std::allocator<nk::f32_t>, 3> rescaling_rows;
    // Only a rescaled source stages on the host; the device casts every other one on import.
    if constexpr (streamed_t::scaled_k) {
        // The trampoline serves both phases, so it follows the larger of the two batches.
        std::size_t const searchable_rows = std::max(rows, source_queries.extent(0));
        std::size_t const batch_rows = std::max(insert_batch_rows(settings, rows),
                                                query_batch_rows(settings, searchable_rows));
        staging = nk::tensor<scalar_t, std::allocator<scalar_t>, 3>::zeros({1, batch_rows, dimensions}).value;
        rescaling_rows = nk::tensor<nk::f32_t, std::allocator<nk::f32_t>, 3>::zeros({1, batch_rows, dimensions}).value;
        if (staging.empty() || rescaling_rows.empty()) {
            fmt::println("Failed to allocate {} staging rows of {}d {}", batch_rows, dimensions,
                         scalar_t::dtype_name());
            return 1;
        }
        streamed.staging = staging.span();
        streamed.rescaling = rescaling_rows.span();
        streamed.establish_scale();
    }

    if (dataset.has_ground_truth() && !settings.max_base_vectors.value) streamed.ground_truth = dataset.neighbors();
    else if (dataset.has_ground_truth())
        fmt::println("Ground truth ignored: --max-base-vectors truncates the graph it was computed for");
    return phases(streamed) ? 0 : 1;
}

/** The innermost hop: binds the concrete CUDA-index instantiation, builds the index under @p kernel
 *  in checkpoint order, then runs @p workload. */
template <typename dataset_scalar_type_, nk_dtype_t stored_dtype_, nk_capability_t capability_,
          typename storage_allocator_, typename workload_type_>
int stage_and_run(utilities::persisted_dataset<dataset_scalar_type_> const& dataset, settings_t const& settings,
                  int device, nk_kernel_kind_t kernel, workload_type_ const& workload) {
    using index_t =
        vram_nvidia_index<stored_dtype_, capability_, storage_allocator_, storage_allocator_, cuda_device_allocator_t>;
    cuda_bench_context<index_t> bench {settings, device, kernel};
    return stream_and_run(dataset, bench, [&](auto& streamed) {
        return prepare(streamed) && run_insertion(streamed) && run_refinement(streamed) &&
               (settings.save_name.empty() || save_index(streamed)) && workload.run(streamed);
    });
}

/** Kernel axis: admits the runtime metric the stored dtype scores by, the terminal hop into
 *  @ref stage_and_run; a dot-only workload pins the dot kernel instead. */
template <typename dataset_scalar_type_, nk_dtype_t stored_dtype_, nk_capability_t capability_,
          typename storage_allocator_, typename workload_type_>
int run_with_metric(utilities::persisted_dataset<dataset_scalar_type_> const& dataset, settings_t const& settings,
                    int device, workload_type_ const& workload) {
    if constexpr (workload_type_::dot_only_k)
        return stage_and_run<dataset_scalar_type_, stored_dtype_, capability_, storage_allocator_>(
            dataset, settings, device, nk_kernel_dot_k, workload);
    else
        return dispatch_kernel<stored_dtype_>(
            settings.kernel,
            [&](auto kernel) {
                return stage_and_run<dataset_scalar_type_, stored_dtype_, capability_, storage_allocator_>(
                    dataset, settings, device, kernel.value, workload);
            },
            [&](auto metric) {
                fmt::println("Unsupported metric {} for {} vectors", nk_kernel_name(metric), dtype_name(stored_dtype_));
                return 1;
            });
}

/** Compute axis: the named table row's ceiling instantiates every cell below, so two runs on one device
 *  compare generations side by side; past the tail sit the Ampere kits every GPU runs. */
template <typename dataset_scalar_type_, nk_dtype_t stored_dtype_, typename storage_allocator_, typename workload_type_>
int run_with_compute(utilities::persisted_dataset<dataset_scalar_type_> const& dataset, settings_t const& settings,
                     int device, nk_capability_t max_compute, workload_type_ const& workload) {
    auto const named = std::ranges::find(cuda_compute_choices_k, max_compute, &capability_choice<nk_capability_t>::bit);
    return dispatch_index<std::size(cuda_compute_choices_k)>(
        static_cast<std::size_t>(named - std::begin(cuda_compute_choices_k)),
        [&](auto row) {
            return run_with_metric<dataset_scalar_type_, stored_dtype_,
                                   cuda_ceiling_of(cuda_compute_choices_k[row.value].bit), storage_allocator_>(
                dataset, settings, device, workload);
        },
        [&] {
            return run_with_metric<dataset_scalar_type_, stored_dtype_, cuda_ceiling_of(nk_cap_ampere_k),
                                   storage_allocator_>(dataset, settings, device, workload);
        });
}

/**
 *  @brief The shard hop: everything below one stored dtype - the memory axis picking the storage
 *      allocator, the generation ceiling, and the metric - so the per-dtype shard translation
 *      units carry the whole device instantiation grid of their dtype.
 *
 *  Device-space flat matrices stay in VRAM through the virtual-memory backend; managed-space ones
 *  oversubscribe it through the pager, the bigger-than-VRAM build path. The cascade ends in the
 *  search workload.
 */
template <typename dataset_scalar_type_, nk_dtype_t stored_dtype_>
int run_with_stored_dtype(utilities::persisted_dataset<dataset_scalar_type_> const& dataset, settings_t const& settings,
                          int device, nk_capability_t max_compute, cuda_address_space_t storage_space) {
    return storage_space == cuda_address_space_t::managed_k
               ? run_with_compute<dataset_scalar_type_, stored_dtype_, cuda_managed_allocator<std::byte>>( //
                     dataset, settings, device, max_compute, search_workload_t {})
               : run_with_compute<dataset_scalar_type_, stored_dtype_, cuda_virtual_memory_allocator<std::byte>>( //
                     dataset, settings, device, max_compute, search_workload_t {});
}

#pragma endregion

#pragma region Process Setup

/** The process every CUDA binary runs around its cascade: reads the flags of @p binary_name and
 *  the ones each of @p binary_flags adds, probes the devices - every NVIDIA one
 *  @c USEARCH_DEVICES names, or the first listed - resolves the compute ceiling, prints them, and
 *  opens the dataset for @p on_dataset - handed the dataset, the settings, the devices' CUDA
 *  ordinals, the ceiling, and the storage space. */
template <typename on_dataset_type_, typename... binary_flags_types_>
int run_cuda_bench(int argc, char const* const* argv, char const* binary_name, on_dataset_type_&& on_dataset,
                   binary_flags_types_... binary_flags) {
    settings_t settings;
    cuda_cli_t device_cli;
    std::vector<flag_t> flags;
    auto const append = [&](auto const& added) { flags.insert(flags.end(), added.begin(), added.end()); };
    append(common_flags(settings));
    append(cuda_flags(device_cli));
    append(indexer_flags(settings));
    (append(binary_flags(settings)), ...);
    parse_or_exit(argc, argv, fmt::format("{} --base-vectors <path> [flags]", binary_name), flags);
    resolve_common_names(settings);

    // A named ceiling caps which kit-chain rows the cascade may select; "ampere" keeps just the floor.
    // Capping above what this binary compiled is harmless: a row also needs its tier compiled in.
    nk_capability_t max_compute = nk_cap_ampere_k;
    if (!device_cli.max_compute_name.empty() && device_cli.max_compute_name != "ampere") {
        std::size_t const named = index_of_name(cuda_compute_choices_k, device_cli.max_compute_name);
        if (named == std::size(cuda_compute_choices_k)) {
            fmt::println(
                stderr, "--max-compute=\"{}\" does not parse, expected one of ampere, {}", device_cli.max_compute_name,
                fmt::join(cuda_compute_choices_k | std::views::transform(&capability_choice<nk_capability_t>::name),
                          ", "));
            std::exit(1);
        }
        max_compute = cuda_compute_choices_k[named].bit;
    }

    // `USEARCH_DEVICES` names USearch ordinals; the driver takes each NVIDIA device's own.
    std::span<gpu_device_t const> const listed = gpu_devices();
    counts_t const named = env_devices();
    std::vector<int> devices;
    std::vector<gpu_device_t> engaged;
    for (std::size_t const ordinal : named.view())
        if (ordinal < listed.size() && listed[ordinal].vendor == gpu_vendor_t::nvidia_k)
            devices.push_back(static_cast<int>(listed[ordinal].vendor_ordinal)), engaged.push_back(listed[ordinal]);
    if (!named.size)
        for (gpu_device_t const& device : listed)
            if (device.vendor == gpu_vendor_t::nvidia_k && devices.empty())
                devices.push_back(static_cast<int>(device.vendor_ordinal)), engaged.push_back(device);
    if (devices.empty() || devices.size() != (named.size ? named.size : 1)) {
        fmt::println(stderr,
                     "USEARCH_DEVICES=\"{}\" names no NVIDIA device of the {} listed:", fmt::join(named.view(), ","),
                     listed.size());
        for (std::size_t position = 0; position != listed.size(); ++position)
            fmt::println(stderr, "- {}: {} #{}", position,
                         usearch_gpu_class_name(static_cast<usearch_gpu_class_t>(listed[position].gpu_class)),
                         listed[position].vendor_ordinal);
        return 1;
    }
    // Unnamed, the ceiling is the first device's own generation, as Hopper and both Blackwells run
    // only their own kits.
    if (device_cli.max_compute_name.empty()) switch (engaged.front().gpu_class) {
        case gpu_class_t::nvidia_ada_k: max_compute = nk_cap_ada_k; break;
        case gpu_class_t::nvidia_hopper_k: max_compute = nk_cap_hopper_k; break;
        case gpu_class_t::nvidia_blackwell_k: max_compute = nk_cap_blackwell_k; break;
        case gpu_class_t::nvidia_blackwellrtx_k: max_compute = nk_cap_blackwellrtx_k; break;
        default: break;
        }
    cuda_set_device(devices.front());
    for (gpu_device_t const& device : engaged) settings.cores += device.compute_units;
    fmt::println("USearch {}.{}.{}", USEARCH_VERSION_MAJOR, USEARCH_VERSION_MINOR, USEARCH_VERSION_PATCH);
    for (gpu_device_t const& device : engaged)
        fmt::println("- Device {}: {} · {} SMs · {:.2f} GB", device.vendor_ordinal, device.name, device.compute_units,
                     gigabytes(device.memory_bytes));
    print(settings);
    fmt::println("- Max compute: {}", bit_name(cuda_compute_choices_k, max_compute, "ampere"));
    fmt::println("- Managed memory: {}", device_cli.managed_memory == switch_t::present_k);

    cuda_address_space_t const storage_space = device_cli.managed_memory == switch_t::present_k
                                                   ? cuda_address_space_t::managed_k
                                                   : cuda_address_space_t::device_k;
    return open_dataset(settings.base_vectors_path, settings.query_vectors_path, settings.query_neighbors_path,
                        [&](auto const& dataset) {
                            return on_dataset(dataset, settings, std::span<int const> {devices}, max_compute,
                                              storage_space);
                        });
}

#pragma endregion

} // namespace unum::usearch::bench::dense

template <>
inline us::utilities::bytes_view_t
us::dense::cuda_fatbin<us::bench::dense::cuda_ceiling_of(nk_cap_ampere_k)>() noexcept {
    return us::bench::dense::cuda_bench_fatbin();
}
template <>
inline us::utilities::bytes_view_t us::dense::cuda_fatbin<us::bench::dense::cuda_ceiling_of(nk_cap_ada_k)>() noexcept {
    return us::bench::dense::cuda_bench_fatbin();
}
template <>
inline us::utilities::bytes_view_t
us::dense::cuda_fatbin<us::bench::dense::cuda_ceiling_of(nk_cap_hopper_k)>() noexcept {
    return us::bench::dense::cuda_bench_fatbin();
}
template <>
inline us::utilities::bytes_view_t
us::dense::cuda_fatbin<us::bench::dense::cuda_ceiling_of(nk_cap_blackwell_k)>() noexcept {
    return us::bench::dense::cuda_bench_fatbin();
}
template <>
inline us::utilities::bytes_view_t
us::dense::cuda_fatbin<us::bench::dense::cuda_ceiling_of(nk_cap_blackwellrtx_k)>() noexcept {
    return us::bench::dense::cuda_bench_fatbin();
}
