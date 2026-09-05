/**
 *  @brief Profiling harness for the sharded @c dense_ram_indexes over real persisted
 *      datasets, shared by the per-dtype benchmark shards.
 *
 *  @author Ash Vardanian
 *  @file bench/dense/ram_indexes.hpp
 *  @date August 30, 2026
 *
 *  The multi-index category: one sharded index holding a shard per engaged memory domain,
 *  driven through one @c forkunion_executor_t with batched insertions and batched searches -
 *  the handle the C surface exposes, measured end to end. The sibling `usearch_bench_dense_ram_index`
 *  measures one shard with everything local; this binary measures what the whole plural
 *  costs, fill balance and cross-domain rotation included.
 *
 *  Keys are dataset rows, so a search's merged ranking judges directly against the ground
 *  truth and no id-to-row bookkeeping exists. Checkpoints ride the sharded index's own
 *  SafeGraphs verbs - a monolith per shard, the manifest committed last - so `--save` and
 *  `--load` exercise exactly the persistence path the C surface exposes.
 */
#pragma once

#include <algorithm> // `std::max`, `std::min`
#include <chrono>    // `std::chrono::steady_clock`
#include <string>    // `std::string`
#include <tuple>     // `std::tuple`, `std::tuple_element_t` for the schedule waiter list
#include <vector>    // `std::vector`

#include <clipp.h>
#include <fmt/format.h>

#include <usearch/dense/ram_indexes.hpp>
#include <usearch/utilities/ram_device.hpp>

#include "dense.cuh"
#include "utilities/persisted_dataset.hpp"

namespace unum {
namespace usearch {

#pragma region CPU Pipeline

/** @brief The STL floor below the schedule table, correct on every machine. */
using yield_waiter_t = fu::standard_yield_t;
#if FU_DETECT_ARCH_X86_64_
using hint_waiter_t = fu::x86_pause_t;
using monitor_waiter_t = fu::x86_tpause_t;
#elif FU_DETECT_ARCH_ARM64_
using hint_waiter_t = fu::arm64_yield_t;
using monitor_waiter_t = fu::arm64_wfet_t;
#elif FU_DETECT_ARCH_RISC5_
using hint_waiter_t = fu::risc5_pause_t;
using monitor_waiter_t = fu::risc5_wrs_t;
#else
using hint_waiter_t = fu::standard_yield_t;
using monitor_waiter_t = fu::standard_yield_t;
#endif

/** @brief The waiter type each schedule table row instantiates, positionally - the data list the
 *      schedule axis indexes, mirroring how the compute axis indexes @c prefix_mask. Every real
 *      table has two rows; on the fallback arch both entries collapse onto the yield floor, so
 *      only row 0 is ever reachable. */
using schedule_waiters_t = std::tuple<hint_waiter_t, monitor_waiter_t>;

/**
 *  @brief The state one fully-typed sharded-index run shares between its phases: the indexes,
 *      the executor driving them, and the embedded backend-neutral judgment results.
 *
 *  The dataset never appears here - the data side lives in @c streamed_dataset, which bridges
 *  the file's scalar into the stored one batch by batch. The sharded index owns its shards and
 *  key planes; the executor owns its pool and per-thread arenas; nothing else exists.
 */
template <typename indexes_type_>
struct bench_indexes_context {
    using indexes_t = indexes_type_;
    using index_t = typename indexes_t::index_t;
    using key_t = typename indexes_t::key_t;
    using scalar_t = typename index_t::scalar_t;
    using small_size_t = typename index_t::small_size_t;
    using vector_view_t = typename indexes_t::vector_view_t;
    using vectors_view_t = typename indexes_t::vectors_view_t;

    static constexpr nk_kernel_kind_t kernel_kind_k = index_t::kernel_kind_k;

    bench_options_t const& options;
    /** @brief The sharded index under measurement: one shard per engaged memory domain. */
    indexes_t indexes;
    /** @brief The worker threads and per-thread arenas every batched verb runs on. */
    forkunion_executor_t executor;
    /** @brief The backend-neutral judgment state both phases fill. */
    bench_results_t results;

    /** @brief The host tables follow the first named memory domain, or the heap when none is named. */
    bench_indexes_context(bench_options_t const& options) noexcept
        : options(options),
          indexes(forkunion_allocator_t {options.memory_domains.empty() ? no_memory_domain_k
                                                                        : options.memory_domains.front()}) {}
};

/** @brief The checkpoint manifest path: dtype, kernel & connectivity stamped into the name, so
 *      a geometry-mismatched run finds nothing rather than restoring the wrong graph. */
template <typename indexes_type_>
std::string checkpoint_path_of(bench_options_t const& options, std::string const& name) {
    using index_t = typename indexes_type_::index_t;
    return fmt::format("{}/{}-{}-{}-c{}.safegraphs.json", options.checkpoints_dir, name,
                       index_t::scalar_t::dtype_name(), nk_kernel_name(index_t::kernel_kind_k), options.connectivity);
}

/** @brief Serializes the sharded index as one SafeGraphs checkpoint - a monolith per shard and
 *      the manifest committed last - so later runs @c --load it instead of rebuilding. */
template <typename streamed_dataset_type_>
bool save_indexes(streamed_dataset_type_& streamed) {
    using clock_t = stdc::steady_clock;
    using bench_t = typename streamed_dataset_type_::bench_t;
    auto& bench = streamed.bench;
    std::string const path = checkpoint_path_of<typename bench_t::indexes_t>(bench.options, bench.options.save_name);
    save_config_t config;
    config.layout = checkpoint_layout_t::distributed_k;
    auto started = clock_t::now();
    if (status_t const saved = bench.indexes.save(path, config, bench.executor); failed(saved)) {
        fmt::println("Failed to save {}: {}", path, status_to_string(saved));
        return false;
    }
    double const seconds = stdc::duration<double>(clock_t::now() - started).count();
    fmt::println("Saved {} shards: {:.2f} GB in {:.2f} s", bench.indexes.shards_count(),
                 double(bench.indexes.storage_requirements(config).value) / 1e9, seconds);
    return true;
}

/** @brief Places one shard per engaged memory domain, restores a checkpoint or reserves for the
 *      viewed rows, and spawns & warms the executor - everything fallible, before any clock. */
template <typename streamed_dataset_type_>
bool prepare(streamed_dataset_type_& streamed, fu_topology_t topology) {
    using bench_t = typename streamed_dataset_type_::bench_t;
    using index_t = typename bench_t::index_t;
    using small_size_t = typename bench_t::small_size_t;
    auto& bench = streamed.bench;
    std::size_t const rows = streamed.vectors.extent(0);
    std::size_t const dimensions = streamed.vectors.extent(1);

    connectivity_t const connectivity = connectivity_of(bench.options.connectivity);
    if (connectivity == connectivity_unknown_k) {
        fmt::println("Unsupported --connectivity {}; pick 16, 32, or 128", bench.options.connectivity);
        return false;
    }

    // Placement: one shard per named memory domain, or a single heap shard when none is
    // named. The sharded index splits the reservation evenly and refuses to grow it later, so the
    // full row count goes in here.
    if (bench.options.memory_domains.empty()) {
        if (failed(bench.indexes.emplace(fu_memory_domain_id_t {-1}))) {
            fmt::println("Failed to place the heap shard");
            return false;
        }
    }
    else
        for (fu_memory_domain_id_t const domain : bench.options.memory_domains)
            if (failed(bench.indexes.emplace(domain))) {
                fmt::println("Failed to place the shard on memory domain {}", domain);
                return false;
            }
    // The executor spawns machine-wide and owns its pool outright.
    auto spawned = forkunion_executor_t::spawn(topology, bench.options.threads);
    if (failed(spawned.status)) {
        fmt::println("Failed to spawn the executor's worker threads");
        return false;
    }
    bench.executor = std::move(spawned.value);

    // AMX tile state needs a per-thread OS permission handshake before first use; NumKong's
    // thread configurator performs it and no-ops everywhere else.
    if (failed(bench.executor.for_threads(
            [](std::size_t, std::size_t) noexcept { nk_configure_thread(nk_capabilities_available()); }))) {
        fmt::println("Failed to configure the worker threads");
        return false;
    }

    // Restoring replaces building outright: the checkpoint's own summary sizes the reservation
    // inside `load`, so the fresh shards reach it unreserved; a build reserves here instead.
    if (!bench.options.load_name.empty()) {
        std::string const path = checkpoint_path_of<typename bench_t::indexes_t>(bench.options,
                                                                                 bench.options.load_name);
        if (status_t const restored = bench.indexes.load(path, load_config_t {}, bench.executor); failed(restored)) {
            fmt::println("Failed to load {}: {}", path, status_to_string(restored));
            return false;
        }
        if (bench.indexes.size() != rows) {
            fmt::println("Loaded checkpoint covers {} rows, but the view holds {}", bench.indexes.size(), rows);
            return false;
        }
    }
    else if (failed(bench.indexes.reserve({rows, connectivity, dimensions}))) {
        fmt::println("Failed to reserve {} × {}d over {} shards", rows, dimensions, bench.indexes.shards_count());
        return false;
    }

    // Warming the arenas here moves every allocation ahead of the clocks; a walk that outgrows the
    // estimate re-tables into a chunk the arena grows by, so the estimate is the whole ask.
    typename index_t::insert_config_t warm_config;
    warm_config.expansion_factor = static_cast<small_size_t>(bench.options.expansion_add);
    if (failed(bench.executor.reserve(bench.indexes.shard(0).traversal_memory_usage(warm_config)))) {
        fmt::println("Failed to reserve the executor's per-thread arenas");
        return false;
    }

    // Self-recall probes every ingested row, so the per-query buffers follow the larger count.
    std::size_t const search_count = bench.options.self_recall ? streamed.vectors.extent(0)
                                                               : streamed.queries.extent(0);
    bench.results.hit_rows = nk::matrix<u32_t>::try_zeros({search_count, bench.options.top_k});
    bench.results.hit_distances = nk::matrix<dense_distance_t>::try_zeros({search_count, bench.options.top_k});
    if (bench.results.hit_rows.empty() || bench.results.hit_distances.empty()) {
        fmt::println("Failed to allocate the per-query hit matrices");
        return false;
    }
    bench.results.hit_counts.resize(search_count);
    return true;
}

/** @brief Streams the whole dataset through batched @c add calls: a zero-copy dataset goes in
 *      as one call - the cleanest measure of the sharded index's own machinery - and a casting one in
 *      staging-bounded batches, each cast fanned over the executor before its call. */
template <typename streamed_dataset_type_>
bool run_insertion(streamed_dataset_type_& streamed) {
    using clock_t = stdc::steady_clock;
    using bench_t = typename streamed_dataset_type_::bench_t;
    using index_t = typename bench_t::index_t;
    using key_t = typename bench_t::key_t;
    using small_size_t = typename bench_t::small_size_t;
    using scalar_t = typename bench_t::scalar_t;
    using vectors_view_t = typename bench_t::vectors_view_t;
    auto& bench = streamed.bench;
    std::size_t const rows = streamed.vectors.extent(0);
    std::size_t const dimensions = streamed.vectors.extent(1);

    typename index_t::insert_config_t config;
    config.expansion_factor = static_cast<small_size_t>(bench.options.expansion_add);
    std::size_t const batch_ceiling = streamed_dataset_type_::zero_copy_k ? rows : streamed.staging.extent(1);
    std::size_t const batch_rows = bench.options.batch ? std::min(batch_ceiling, bench.options.batch) : batch_ceiling;
    std::vector<key_t> keys(std::min(batch_rows, rows));
    typename index_t::insert_result_t sums;

    auto started = clock_t::now();
    for (std::size_t base = 0; base != rows;) {
        std::size_t const count = std::min(batch_rows, rows - base);
        for (std::size_t row = 0; row != count; ++row) keys[row] = static_cast<key_t>(base + row);

        vectors_view_t batch;
        if constexpr (streamed_dataset_type_::zero_copy_k)
            batch = streamed.vectors(nk::range {base, base + count}, nk::slice);
        else {
            // The cast fans over the same workers the insertion will use; every task lands in
            // its own staging row, so the batch materializes contiguous and clash-free.
            auto cast_row = [&](std::size_t task, std::size_t, std::size_t) noexcept {
                [[maybe_unused]] auto staged = streamed.stage(streamed.vectors.row(base + task).as_vector(), 0, task);
            };
            if (failed(bench.executor.for_tasks(count, cast_row))) {
                fmt::println("Failed to dispatch the casting pass");
                return false;
            }
            batch = vectors_view_t {streamed.staging.row(0).data(), count, dimensions};
        }

        auto added = bench.indexes.add(std::span<key_t const> {keys.data(), count}, batch, bench.executor, config);
        if (!added) {
            fmt::println("Insertion refused a batch of {} rows at row {}: {}", count, base,
                         status_to_string(added.status));
            return false;
        }
        sums.visited_members += added.visited_members;
        sums.computed_distances += added.computed_distances;
        sums.computed_distances_in_refines += added.computed_distances_in_refines;
        sums.computed_distances_in_reverse_refines += added.computed_distances_in_reverse_refines;
        sums.restarts += added.restarts;
        sums.saturated_walks += added.saturated_walks;
        base += count;
    }
    double const seconds = stdc::duration<double>(clock_t::now() - started).count();

    double const refine_share = sums.computed_distances ? 100.0 * double(sums.computed_distances_in_refines) /
                                                              double(sums.computed_distances)
                                                        : 0.0;
    double const reverse_share = sums.computed_distances ? 100.0 * double(sums.computed_distances_in_reverse_refines) /
                                                               double(sums.computed_distances)
                                                         : 0.0;
    fmt::print(header_style(), "\nInsertions\n");
    fmt::println("  shape: {} × {}d", rows, dimensions);
    fmt::println("  quads: {} · {} · {} kit", scalar_t::dtype_name(),
                 nk_kernel_name(streamed_dataset_type_::bench_t::kernel_kind_k),
                 bit_name(compute_span_k, index_t::picked_quads_capability_k, "serial"));
    if constexpr (index_t::has_tiles_k)
        fmt::println("  tiles: {} · reverse refinement",
                     bit_name(compute_span_k, index_t::picked_tiles_capability_k, "none"));
    if (index_t::picked_quads_capability_k == nk_cap_serial_k &&
        (bench.options.max_compute & nk_capabilities_available() & ~nk_cap_serial_k))
        fmt::print(verdict_style(false), "  ⚠ scalar fallback: no SIMD kit is adapted for this dtype & metric\n");
    fmt::println("  threads: {}", bench.executor.size());
    auto shard_sizes = std::views::iota(std::size_t {0}, bench.indexes.shards_count()) |
                       std::views::transform([&](std::size_t shard) { return bench.indexes.shard(shard).size(); });
    fmt::println("  shards: {}, one per memory domain, filled {}", bench.indexes.shards_count(),
                 fmt::join(shard_sizes, " "));
    fmt::println("  wall: {:.2f} s", seconds);
    fmt::println("  throughput: {} IPS", humanize(double(rows) / seconds));
    fmt::println("  distances: {} ({:.1f}% refines, {:.1f}% reverse)", humanize(double(sums.computed_distances)),
                 refine_share, reverse_share);
    fmt::println("  visited: {}", humanize(double(sums.visited_members)));
    fmt::println("  restarts: {}", fmt::styled(sums.restarts, verdict_style(sums.restarts < rows / 100 + 1)));
    fmt::println("  saturated: {}", fmt::styled(sums.saturated_walks, verdict_style(sums.saturated_walks == 0)));
    if (std::size_t resident = peak_resident_bytes(); resident)
        fmt::println("  peak resident: {:.2f} GB", double(resident) / 1e9);
    return true;
}

/** @brief Streams every query through batched @c search calls, the merged rankings landing
 *      straight in the judgment matrices - keys are dataset rows, so no translation exists.
 *      A batch-shaped verb has no honest per-query clock, so the report prints the mean. */
template <typename streamed_dataset_type_>
bool run_search(streamed_dataset_type_& streamed) {
    using clock_t = stdc::steady_clock;
    using bench_t = typename streamed_dataset_type_::bench_t;
    using index_t = typename bench_t::index_t;
    using key_t = typename bench_t::key_t;
    using small_size_t = typename bench_t::small_size_t;
    using vectors_view_t = typename bench_t::vectors_view_t;
    auto& bench = streamed.bench;
    std::size_t const dimensions = streamed.vectors.extent(1);
    std::size_t const search_count = bench.options.self_recall ? streamed.vectors.extent(0)
                                                               : streamed.queries.extent(0);
    std::size_t const wanted = bench.options.top_k;

    typename index_t::search_config_t config;
    config.expansion_factor = static_cast<small_size_t>(bench.options.expansion_search);
    // The search phase runs strictly after ingestion, so the mutation-free contract holds by
    // construction - certify it and measure the atomics-free traversal.
    config.synchronization = synchronization_t::immutable_k;

    std::size_t const batch_rows = streamed_dataset_type_::zero_copy_k ? search_count : streamed.staging.extent(1);
    std::vector<std::size_t> counts(std::min(batch_rows, search_count));
    telemetry_lane_t sums;

    auto started = clock_t::now();
    for (std::size_t base = 0; base != search_count;) {
        std::size_t const count = std::min(batch_rows, search_count - base);
        auto const source = bench.options.self_recall ? streamed.vectors : streamed.queries;

        vectors_view_t batch;
        if constexpr (streamed_dataset_type_::zero_copy_k) batch = source(nk::range {base, base + count}, nk::slice);
        else {
            auto cast_row = [&](std::size_t task, std::size_t, std::size_t) noexcept {
                [[maybe_unused]] auto staged = streamed.stage(source.row(base + task).as_vector(), 0, task);
            };
            if (failed(bench.executor.for_tasks(count, cast_row))) {
                fmt::println("Failed to dispatch the casting pass");
                return false;
            }
            batch = vectors_view_t {streamed.staging.row(0).data(), count, dimensions};
        }

        auto found = bench.indexes.search(
            batch, bench.executor, config, wanted,
            std::span<key_t> {bench.results.hit_rows.row(base).data(), count * wanted},
            std::span<dense_distance_t> {bench.results.hit_distances.row(base).data(), count * wanted},
            std::span<std::size_t> {counts.data(), count});
        if (!found) {
            fmt::println("Search refused a batch of {} queries at query {}: {}", count, base,
                         status_to_string(found.status));
            return false;
        }
        sums.visited_members += found.visited_members;
        sums.computed_distances += found.computed_distances;
        sums.restarts += found.restarts;
        sums.saturated_walks += found.saturated_walks;
        for (std::size_t query = 0; query != count; ++query)
            bench.results.hit_counts[base + query] = static_cast<u32_t>(counts[query]);
        base += count;
    }
    double const seconds = stdc::duration<double>(clock_t::now() - started).count();

    report_search_metrics(bench.options, search_count, seconds, bench.executor.size(), sums, {});
    judge_hits(bench.options, bench.results, search_count);
    return true;
}

#pragma endregion

#pragma region Dispatch Cascade

/** @brief Bundles the launch state the cascade threads through its compile-time hops; the
 *      executor spawns inside the run, so no pool travels here. */
template <typename dataset_scalar_type_>
struct bench_launch {
    persisted_dataset<dataset_scalar_type_> const& dataset;
    bench_options_t const& options;
    fu_topology_t topology;
};

/** @brief Gate: a (stored dtype, kernel) cell is runnable once its kit chain is adapted in
 *      @c usearch/dense/ram_metric.hpp - unadapted combinations report as skipped and light up
 *      automatically when their kits land, exactly like the CUDA cascade. */
template <nk_dtype_t stored_dtype_, nk_kernel_kind_t kernel_, nk_capability_t capability_>
concept ram_kit_adapted = requires {
    sizeof(dense_quads_chain<kernel_, typename nk::type_for<stored_dtype_>::type, capability_>);
};

/** @brief No-op sibling picked when the cell's CPU kit chain is not adapted yet. */
template <typename dataset_scalar_type_, nk_dtype_t stored_dtype_, nk_kernel_kind_t kernel_,
          nk_capability_t capability_, typename wait_type_>
    requires(!ram_kit_adapted<stored_dtype_, kernel_, capability_>)
int stage_and_run(bench_launch<dataset_scalar_type_> const&) {
    fmt::println("skipped: no CPU kit adapted for {} · {}", dtype_name(stored_dtype_), nk_kernel_name(kernel_));
    return 0;
}

/**
 *  @brief The innermost hop: binds the concrete sharded-index instantiation, wires the typed source
 *      views and the batch-shaped staging into a @c streamed_dataset, and runs both phases.
 *
 *  Staging is one contiguous plane the batched verbs view whole, sized after the executor
 *  spawns so its row count can follow the thread count. When the file scalar already matches
 *  the stored one, everything is a zero-copy view into the mappings.
 */
template <typename dataset_scalar_type_, nk_dtype_t stored_dtype_, nk_kernel_kind_t kernel_,
          nk_capability_t capability_, typename wait_type_>
    requires(ram_kit_adapted<stored_dtype_, kernel_, capability_>)
int stage_and_run(bench_launch<dataset_scalar_type_> const& launch) {
    using indexes_t = dense_ram_indexes<u32_t, stored_dtype_, kernel_, capability_, wait_type_>;
    using bench_context_type_ = bench_indexes_context<indexes_t>;
    using streamed_t = streamed_dataset<dataset_scalar_type_, bench_context_type_>;
    using scalar_t = typename bench_context_type_::scalar_t;

    persisted_dataset<dataset_scalar_type_> const& dataset = launch.dataset;
    std::size_t const dimensions = dataset.dimensions();
    std::size_t rows = dataset.vectors().extent(0);
    if (launch.options.max_vectors) rows = std::min(rows, launch.options.max_vectors);

    // Self-queries keep the harness usable on a lone vectors file, without recall.
    nk::matrix_view<dataset_scalar_type_> const source_queries =
        dataset.has_queries() ? dataset.queries()
                              : nk::matrix_view<dataset_scalar_type_> {dataset.vectors().data(),
                                                                       std::min<std::size_t>(1000, rows), dimensions};

    bench_context_type_ bench {launch.options};
    streamed_t streamed {bench, {dataset.vectors().data(), rows, dimensions}, source_queries};
    if (!prepare(streamed, launch.topology)) return 1;

    // Owns the one staging plane both batched phases view.
    nk::tensor<scalar_t, std::allocator<scalar_t>, 3> staging;
    // Owns the rescaling plane `streamed_dataset::rescaling` views.
    nk::tensor<nk::f32_t, std::allocator<nk::f32_t>, 3> rescaling_plane;
    if constexpr (!streamed_t::zero_copy_k) {
        std::size_t const threads = bench.executor.size();
        std::size_t const batch_rows = launch.options.batch ? std::max(launch.options.batch, threads) : 4 * threads;
        staging = nk::tensor<scalar_t, std::allocator<scalar_t>, 3>::try_zeros({1, batch_rows, dimensions});
        if (staging.empty()) {
            fmt::println("Failed to allocate {} staging rows of {}d {}", batch_rows, dimensions,
                         scalar_t::dtype_name());
            return 1;
        }
        streamed.staging = staging.span();

        if constexpr (streamed_t::scaled_k) {
            rescaling_plane = nk::tensor<nk::f32_t, std::allocator<nk::f32_t>, 3>::try_zeros(
                {1, batch_rows, dimensions});
            if (rescaling_plane.empty()) {
                fmt::println("Failed to allocate {} rescaling rows of {}d f32", batch_rows, dimensions);
                return 1;
            }
            streamed.rescaling = rescaling_plane.span();
            streamed.establish_scale();
        }
    }

    if (dataset.has_ground_truth() && !launch.options.max_vectors) bench.results.ground_truth = dataset.neighbors();
    else if (dataset.has_ground_truth())
        fmt::println("Ground truth ignored: --max-vectors truncates the graph it was computed for");

    // A restored checkpoint replaces ingestion outright; the search phase runs either way.
    if (launch.options.load_name.empty() && !run_insertion(streamed)) return 1;
    if (!launch.options.save_name.empty() && !save_indexes(streamed)) return 1;
    if (!run_search(streamed)) return 1;
    return 0;
}

/** @brief Kernel axis: lowers the runtime metric to a compile-time kernel kind, the terminal
 *      hop into @ref stage_and_run. */
template <typename dataset_scalar_type_, nk_dtype_t stored_dtype_, nk_capability_t capability_, typename wait_type_>
int run_with_metric(bench_launch<dataset_scalar_type_> const& launch) {
    return dispatch_kernel<stored_dtype_>(
        launch.options.kernel,
        [&](auto kernel) {
            return stage_and_run<dataset_scalar_type_, stored_dtype_, kernel.value, capability_, wait_type_>(launch);
        },
        [&](auto metric) {
            fmt::println("Unsupported metric {} for {} vectors", nk_kernel_name(metric), dtype_name(stored_dtype_));
            return 1;
        });
}

/** @brief Schedule axis: the newest schedule row the allow-mask admits - intersected with what
 *      this machine actually offers - picks the waiter from @ref schedule_waiters_t. Past the
 *      table's tail sits the STL yield floor. */
template <typename dataset_scalar_type_, nk_dtype_t stored_dtype_, nk_capability_t capability_>
int run_with_schedule(bench_launch<dataset_scalar_type_> const& launch) {
    return dispatch_index<schedule_choices_count_k>(
        highest_index(schedule_span_k, effective_schedule(launch.options.max_schedule)),
        [&](auto row) {
            using wait_type_ = std::tuple_element_t<row.value, schedule_waiters_t>;
            return run_with_metric<dataset_scalar_type_, stored_dtype_, capability_, wait_type_>(launch);
        },
        [&] { return run_with_metric<dataset_scalar_type_, stored_dtype_, capability_, yield_waiter_t>(launch); });
}

/** @brief Compute axis: the allow-mask intersected with the runtime detection picks the newest
 *      table row, whose @ref prefix_mask ceiling instantiates @ref run_with_schedule. Past the
 *      table's tail sits the one ceiling every machine owns: the serial kernels. */
template <typename dataset_scalar_type_, nk_dtype_t stored_dtype_>
int run_with_compute(bench_launch<dataset_scalar_type_> const& launch) {
    nk_capability_t const effective = launch.options.max_compute & nk_capabilities_available();
    return dispatch_index<compute_choices_count_k>(
        highest_index(compute_span_k, effective),
        [&](auto row) {
            return run_with_schedule<dataset_scalar_type_, stored_dtype_, prefix_mask(compute_span_k, row.value)>(
                launch);
        },
        [&] { return run_with_schedule<dataset_scalar_type_, stored_dtype_, nk_cap_serial_k>(launch); });
}

/** @brief The shard hop: everything below one stored dtype, so a per-dtype shard translation unit
 *      carries the whole instantiation grid of its dtype. */
template <typename dataset_scalar_type_, nk_dtype_t stored_dtype_>
int run_with_stored_dtype(bench_launch<dataset_scalar_type_> const& launch) {
    return run_with_compute<dataset_scalar_type_, stored_dtype_>(launch);
}

#pragma endregion

} // namespace usearch
} // namespace unum
