/**
 *  @file bench/dense/ram_indexes_search.hpp
 *  @author Ash Vardanian
 *  @date August 30, 2026
 *  @brief Profiling harness for searching the sharded @c dense_ram_indexes over real persisted
 *      datasets, shared by the per-dtype benchmark shards.
 *
 *  The multi-index category: one sharded index holding a shard per engaged memory domain,
 *  driven through one @c forkunion_executor_t with batched insertions and batched searches -
 *  the handle the C surface exposes, measured end to end, fill balance and cross-domain
 *  rotation included. The cascade ends in the workload its entry hop names, and the process
 *  setup every sharded CPU binary runs lives here too.
 *
 *  Keys are dataset rows, so a search's merged ranking judges directly against the ground
 *  truth and no ID-to-row bookkeeping exists. Checkpoints ride the sharded index's own
 *  SafeGraphs verbs - a monolith per shard, the manifest committed last.
 */
#pragma once

#include <algorithm> // `std::max`, `std::min`
#include <charconv>  // `std::from_chars` - the `--memory-domain` ordinal
#include <chrono>    // `std::chrono::steady_clock`
#include <string>    // `std::string`
#include <tuple>     // `std::tuple`, `std::tuple_element_t` for the schedule waiter list
#include <vector>    // `std::vector`

#include <fmt/format.h>

#include "usearch/dense/ram_indexes.hpp"
#include "usearch/utilities/ram_device.hpp"

#include "dense_search.cuh"
#include "utilities/persisted_dataset.hpp"

namespace unum::usearch::bench::dense {

#pragma region Pipeline

/** The STL floor below the schedule table, correct on every machine. */
using yield_waiter_t = fu::standard_yield_t;

/** The spin hint every core of this architecture runs. */
#if FORKUNION_ARCH_X86_64_
using hint_waiter_t = fu::x86_pause_t;
#elif FORKUNION_ARCH_ARM64_
using hint_waiter_t = fu::arm64_yield_t;
#elif FORKUNION_ARCH_RISCV64_
using hint_waiter_t = fu::risc5_pause_t;
#else
using hint_waiter_t = fu::standard_yield_t;
#endif

/** The monitored wait this unit compiles, admitted against @c fu_runtime_capabilities before its
 *  row runs; with none compiled the row collapses onto the hint. */
#if FORKUNION_TARGET_X86_TPAUSE
using monitor_waiter_t = fu::x86_tpause_t;
#elif FORKUNION_TARGET_ARM64_WFET
using monitor_waiter_t = fu::arm64_wfet_t;
#elif FORKUNION_TARGET_RISC5_WRS
using monitor_waiter_t = fu::risc5_wrs_t;
#else
using monitor_waiter_t = hint_waiter_t;
#endif

/** The waiter type each schedule table row instantiates, positionally; on the fallback arch both
 *  entries collapse onto the yield floor, so only row 0 is reachable. */
using schedule_waiters_t = std::tuple<hint_waiter_t, monitor_waiter_t>;

/**
 *  @brief The state one fully-typed sharded-index run shares between its phases: the indexes,
 *      the executor driving them, and the embedded backend-neutral judgment results.
 *
 *  The dataset never appears here - the data side lives in @c streamed_dataset, which bridges
 *  the file's scalar into the stored one batch by batch.
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

    /** The parsed CLI, borrowed for the run's lifetime. */
    settings_t const& settings;

    /** The sharded index under measurement: one shard per engaged memory domain. */
    indexes_t indexes;

    /** The worker threads and per-thread arenas every batched verb runs on. */
    forkunion_executor_t executor;

    /** The backend-neutral judgment state both phases fill. */
    bench_results_t results;

    /** The host tables follow the first named memory domain, or the heap when none is named. */
    bench_indexes_context(settings_t const& settings) noexcept
        : settings(settings),
          indexes(forkunion_allocator_t {settings.memory_domains.empty() ? no_memory_domain_k
                                                                         : settings.memory_domains.front()}) {}
};

/** The checkpoint manifest path: dtype, kernel and connectivity stamped into the name, so a
 *  geometry-mismatched run finds nothing rather than restoring the wrong graph. */
template <typename indexes_type_>
std::string checkpoint_path_of(settings_t const& settings, std::string_view name) {
    using index_t = typename indexes_type_::index_t;
    return fmt::format("{}/{}-{}-{}-c{}.safegraphs.json", settings.checkpoints_dir, name,
                       index_t::scalar_t::dtype_name(), nk_kernel_name(index_t::kernel_kind_k),
                       settings.connectivity.value);
}

/** Serializes the sharded index as one SafeGraphs checkpoint - a monolith per shard and the
 *  manifest committed last - so later runs @c --load it instead of rebuilding. */
template <typename streamed_dataset_type_>
bool save_indexes(streamed_dataset_type_& streamed) {
    using bench_t = typename streamed_dataset_type_::bench_t;
    auto& bench = streamed.bench;
    std::string const path = checkpoint_path_of<typename bench_t::indexes_t>(bench.settings, bench.settings.save_name);
    // A view of the parts must stage its queries under the factor the rows were rescaled by.
    std::string const scale = fmt::format("{}", streamed.scale);
    safegraphs_custom_field_t const provenance[] = {{"scale", scale}};
    save_config_t config;
    config.layout = checkpoint_layout_t::distributed_k;
    config.provenance = provenance;
    auto started = steady_clock_t::now();
    if (status_t const saved = bench.indexes.save(path, config, bench.executor); failed(saved)) {
        fmt::println("Failed to save {}: {}", path, status_to_string(saved));
        return false;
    }
    double const seconds = stdc::duration<double>(steady_clock_t::now() - started).count();
    fmt::println("Saved {} shards: {:.2f} GB in {:.2f} s", bench.indexes.shards_count(),
                 gigabytes(bench.indexes.storage_requirements(config).value), seconds);
    return true;
}

/** Places one shard per engaged memory domain, restores a checkpoint, reserves for the viewed rows it
 *  lacks, and spawns and warms the executor - everything fallible, before any clock. */
template <typename streamed_dataset_type_>
bool prepare(streamed_dataset_type_& streamed, fu_topology_t topology) {
    using bench_t = typename streamed_dataset_type_::bench_t;
    using index_t = typename bench_t::index_t;
    using small_size_t = typename bench_t::small_size_t;
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

    // The reservation splits evenly and never grows later, so the full row count goes in here.
    if (bench.settings.memory_domains.empty()) {
        if (failed(bench.indexes.emplace(fu_memory_domain_id_t {-1}))) {
            fmt::println("Failed to place the heap shard");
            return false;
        }
    }
    else
        for (fu_memory_domain_id_t const domain : bench.settings.memory_domains)
            if (failed(bench.indexes.emplace(domain))) {
                fmt::println("Failed to place the shard on memory domain {}", domain);
                return false;
            }
    // The executor spawns machine-wide and owns its pool outright.
    auto [executor, status] = forkunion_executor_t::spawn(topology, bench.settings.threads.value);
    if (failed(status)) {
        fmt::println("Failed to spawn the executor's worker threads");
        return false;
    }
    bench.executor = std::move(executor);

    // AMX tile state needs a per-thread OS permission handshake before first use.
    nk_capability_t available = 0;
    if (nk_cpu_capabilities_enabled(&available) != nk_success_k) available = nk_cap_serial_k;
    if (failed(bench.executor.for_threads([available](std::size_t, std::size_t) noexcept {
            [[maybe_unused]] nk_status_t const configured = nk_cpu_configure_thread(available);
        }))) {
        fmt::println("Failed to configure the worker threads");
        return false;
    }

    // The checkpoint's own summary sizes the reservation inside @c load, so shards reach it bare.
    if (!bench.settings.load_name.empty()) {
        std::string const path = checkpoint_path_of<typename bench_t::indexes_t>(bench.settings,
                                                                                 bench.settings.load_name);
        if (status_t const restored = bench.indexes.load(path, load_config_t {}, bench.executor); failed(restored)) {
            fmt::println("Failed to load {}: {}", path, status_to_string(restored));
            return false;
        }
    }

    // A base that misses the skipped prefix would judge recall against rows the graph never saw.
    if (bench.indexes.size() != bench.settings.skip_vectors.value) {
        fmt::println("Loaded index covers {} rows, but --skip-vectors says {}", bench.indexes.size(),
                     bench.settings.skip_vectors.value);
        return false;
    }
    if (rows != bench.settings.skip_vectors.value &&
        failed(bench.indexes.reserve({rows, connectivity, dimensions}).status)) {
        fmt::println("Failed to reserve {} × {}d over {} shards", rows, dimensions, bench.indexes.shards_count());
        return false;
    }

    // Warming the arenas here moves every allocation ahead of the clocks.
    typename index_t::insert_config_t warm_config;
    warm_config.expansion_factor = static_cast<small_size_t>(bench.settings.expansion_add.value);
    if (failed(bench.executor.reserve(bench.indexes.shard(0).traversal_memory_usage(warm_config)))) {
        fmt::println("Failed to reserve the executor's per-thread arenas");
        return false;
    }

    return true;
}

/** Streams the viewed rows past the loaded base through batched @c add calls: a zero-copy dataset goes in as
 *  one call, a casting one in staging-bounded batches, each cast fanned out before its call. */
template <typename streamed_dataset_type_>
bool run_insertion(streamed_dataset_type_& streamed) {
    using bench_t = typename streamed_dataset_type_::bench_t;
    using index_t = typename bench_t::index_t;
    using key_t = typename bench_t::key_t;
    using small_size_t = typename bench_t::small_size_t;
    using scalar_t = typename bench_t::scalar_t;
    using vectors_view_t = typename bench_t::vectors_view_t;
    auto& bench = streamed.bench;
    std::size_t const rows = streamed.vectors.extent(0);
    std::size_t const dimensions = streamed.vectors.extent(1);

    std::size_t const inserted_rows = rows - bench.settings.skip_vectors.value;
    if (!inserted_rows) {
        fmt::println("\nInsertions");
        fmt::println("  skipped: the loaded base already covers all {} rows", rows);
        return true;
    }

    typename index_t::insert_config_t config;
    config.expansion_factor = static_cast<small_size_t>(bench.settings.expansion_add.value);
    std::size_t const batch_rows = streamed_dataset_type_::zero_copy_k ? rows : streamed.staging.extent(1);
    std::vector<key_t> keys(std::min(batch_rows, rows));
    typename index_t::insert_result_t sums;

    auto started = steady_clock_t::now();
    for (std::size_t base = bench.settings.skip_vectors.value; base != rows;) {
        std::size_t const count = std::min(batch_rows, rows - base);
        for (std::size_t row = 0; row != count; ++row) keys[row] = static_cast<key_t>(base + row);

        vectors_view_t batch;
        if constexpr (streamed_dataset_type_::zero_copy_k)
            batch = streamed.vectors(nk::range(base, base + count), nk::slice);
        else {
            // Every task lands in its own staging row, so the batch materializes clash-free.
            auto cast_row = [&](std::size_t task, std::size_t, std::size_t) noexcept {
                [[maybe_unused]] auto staged = streamed.stage(streamed.vectors.row(base + task).as_vector(), 0, task);
            };
            if (failed(bench.executor.for_tasks(count, cast_row))) {
                fmt::println("Failed to dispatch the casting pass");
                return false;
            }
            batch = vectors_view_t {streamed.staging.row(0).data(), count, dimensions};
        }

        auto const [result, status] = bench.indexes.insert(std::span<key_t const> {keys.data(), count}, batch,
                                                           bench.executor, config);
        if (failed(status)) {
            fmt::println("Insertion refused a batch of {} rows at row {}: {}", count, base, status_to_string(status));
            return false;
        }
        sums.visited_members += result.visited_members;
        sums.computed_distances += result.computed_distances;
        sums.computed_distances_in_refines += result.computed_distances_in_refines;
        sums.computed_distances_in_reverse_refines += result.computed_distances_in_reverse_refines;
        sums.restarts += result.restarts;
        sums.saturated_walks += result.saturated_walks;
        base += count;
    }
    double const seconds = stdc::duration<double>(steady_clock_t::now() - started).count();

    double const refine_share = sums.computed_distances ? 100.0 * double(sums.computed_distances_in_refines) /
                                                              double(sums.computed_distances)
                                                        : 0.0;
    double const reverse_share = sums.computed_distances ? 100.0 * double(sums.computed_distances_in_reverse_refines) /
                                                               double(sums.computed_distances)
                                                         : 0.0;
    fmt::println("\nInsertions");
    fmt::println("  shape: {} × {}d", inserted_rows, dimensions);
    if (bench.settings.skip_vectors.value) fmt::println("  topping: {} loaded rows", bench.settings.skip_vectors.value);
    fmt::println("  quads: {} · {} · {} kit", scalar_t::dtype_name(),
                 nk_kernel_name(streamed_dataset_type_::bench_t::kernel_kind_k),
                 bit_name(compute_choices_k, index_t::picked_quads_capability_k, "serial"));
    if constexpr (index_t::has_tiles_k)
        fmt::println("  tiles: {} · reverse refinement",
                     bit_name(compute_choices_k, index_t::picked_tiles_capability_k, "none"));
    nk_capability_t available = 0;
    if (nk_cpu_capabilities_enabled(&available) != nk_success_k) available = nk_cap_serial_k;
    if (index_t::picked_quads_capability_k == nk_cap_serial_k &&
        (bench.settings.max_compute & available & ~nk_cap_serial_k))
        fmt::println("  ⚠ scalar fallback: no SIMD kit is adapted for this dtype & metric");
    fmt::println("  threads: {}", bench.executor.size());
    auto shard_sizes = std::views::iota(std::size_t {0}, bench.indexes.shards_count()) |
                       std::views::transform([&](std::size_t shard) { return bench.indexes.shard(shard).size(); });
    fmt::println("  shards: {}, one per memory domain, filled {}", bench.indexes.shards_count(),
                 fmt::join(shard_sizes, " "));
    fmt::println("  wall: {:.2f} s", seconds);
    fmt::println("  throughput: {} IPS", humanized(double(inserted_rows) / seconds));
    fmt::println("  distances: {} ({:.1f}% refines, {:.1f}% reverse)", humanized(sums.computed_distances), refine_share,
                 reverse_share);
    fmt::println("  visited: {}", humanized(sums.visited_members));
    fmt::println("  restarts: {}", sums.restarts);
    fmt::println("  saturated: {}", sums.saturated_walks);
    if (std::size_t resident = peak_resident_bytes(); resident)
        fmt::println("  peak resident: {:.2f} GB", gigabytes(resident));
    return true;
}

/** Re-selects every node's neighbors across the shards, pass after pass, then re-links whatever
 *  no search can reach; a pass that re-wires nothing ends the phase. */
template <typename streamed_dataset_type_>
bool run_refinement(streamed_dataset_type_& streamed) {
    using bench_t = typename streamed_dataset_type_::bench_t;
    using indexes_t = typename bench_t::indexes_t;
    using small_size_t = typename bench_t::small_size_t;
    auto& bench = streamed.bench;
    std::size_t const max_refines = bench.settings.max_refines.value;
    std::size_t const nodes = bench.indexes.size();
    if (!max_refines || !nodes) return true;

    typename indexes_t::refine_config_t config;
    config.expansion_factor = static_cast<small_size_t>(bench.settings.expansion_refine.value);

    refinement_report_t report;
    report.nodes = nodes;
    report.expansion = config.expansion_factor ? config.expansion_factor : 2 * bench.indexes.connectivity();
    report.threads = bench.executor.size();

    std::vector<std::size_t> changed_per_pass;
    changed_per_pass.reserve(max_refines);
    progress_bar_t progress {"refining", nodes * max_refines, bench.settings.terminal};
    auto started = steady_clock_t::now();
    for (std::size_t pass = 0; pass != max_refines; ++pass) {
        std::size_t const swept_before = pass * nodes;
        auto const [refined, refined_status] = bench.indexes.refine(
            bench.executor, config, [&](std::size_t done, std::size_t) noexcept {
                return progress.tick(swept_before + done), continuation_t::continue_k;
            });
        if (failed(refined_status)) {
            fmt::println("Refinement failed on pass {}: status {}", pass, int(refined_status));
            return false;
        }
        changed_per_pass.push_back(refined.changed_nodes);
        report.sums.visited_members += refined.visited_members;
        report.sums.computed_distances += refined.computed_distances;
        report.sums.restarts += refined.restarts;
        report.islands_found += refined.islands_found;
        report.islands_reconnected += refined.islands_reconnected;
        report.reachable_nodes = refined.reachable_nodes;
        report.sweeps = refined.sweeps;
        if (!refined.changed_nodes) break;
    }
    progress.finish();
    // Every pass heals what it strands, so the passes' wall time holds the healing too.
    report.refining_seconds = stdc::duration<double>(steady_clock_t::now() - started).count();
    report.changed_per_pass = changed_per_pass;
    report_refinement_metrics(report);
    return true;
}

/** Streams every query through batched @c search calls, the merged rankings landing straight in
 *  the judgment matrices; a batch-shaped verb has no per-query clock, so the report prints the
 *  mean. */
template <typename streamed_dataset_type_>
bool run_search(streamed_dataset_type_& streamed) {
    using bench_t = typename streamed_dataset_type_::bench_t;
    using index_t = typename bench_t::index_t;
    using key_t = typename bench_t::key_t;
    using small_size_t = typename bench_t::small_size_t;
    using vectors_view_t = typename bench_t::vectors_view_t;
    auto& bench = streamed.bench;
    std::size_t const dimensions = streamed.vectors.extent(1);
    std::size_t const wanted = bench.settings.top_k.value;
    bench_indexer_heads<typename index_t::scalar_t> heads;
    if (!open_indexer_heads<index_t>(streamed, heads)) return false;
    std::size_t const search_count = searched_count(streamed, heads);
    if (!reserve_results(bench.results, search_count, wanted)) return false;

    typename index_t::search_config_t config;
    config.expansion_factor = static_cast<small_size_t>(bench.settings.expansion_search.value);
    // The search phase runs strictly after ingestion, so the mutation-free contract holds.
    config.synchronization = synchronization_t::immutable_k;

    std::size_t const batch_rows = streamed_dataset_type_::zero_copy_k || !heads.navigation.empty()
                                       ? search_count
                                       : streamed.staging.extent(1);
    std::vector<std::size_t> counts(std::min(batch_rows, search_count));
    telemetry_lane_t sums;

    auto started = steady_clock_t::now();
    for (std::size_t base = 0; base != search_count;) {
        std::size_t const count = std::min(batch_rows, search_count - base);
        auto const source = searched_rows(streamed);

        vectors_view_t batch;
        if (!heads.navigation.empty())
            batch = vectors_view_t {heads.navigation.data() + base * dimensions, count, dimensions};
        else if constexpr (streamed_dataset_type_::zero_copy_k)
            batch = source(nk::range(base, base + count), nk::slice);
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

        std::span<key_t> const keys {bench.results.hit_rows.row(base).data(), count * wanted};
        std::span<dense_distance_t> const distances {bench.results.hit_distances.row(base).data(), count * wanted};
        auto search = [&](auto const&... ranking) noexcept {
            return bench.indexes.search(batch, ranking..., bench.executor, config, wanted, keys, distances,
                                        std::span<std::size_t> {counts.data(), count});
        };
        auto const [result, status] = [&]() noexcept {
            if constexpr (scores_indexer<index_t>())
                if (heads.engaged()) return search(heads.rows(base, count));
            return search();
        }();
        if (failed(status)) {
            fmt::println("Search refused a batch of {} queries at query {}: {}", count, base, status_to_string(status));
            return false;
        }
        sums.visited_members += result.visited_members;
        sums.computed_distances += result.computed_distances;
        sums.restarts += result.restarts;
        sums.saturated_walks += result.saturated_walks;
        for (std::size_t query = 0; query != count; ++query)
            bench.results.hit_counts[base + query] = static_cast<u32_t>(counts[query]);
        base += count;
    }
    double const seconds = stdc::duration<double>(steady_clock_t::now() - started).count();

    report_search_metrics(bench.settings, search_count, seconds, bench.executor.size(), sums, {});
    judge_search(bench.settings, heads, bench.results, streamed.ground_truth, streamed.vectors, search_count);
    return true;
}

/** The search workload the cascade ends in: batched search over the built shards and its judgment, under
 *  whichever kernel the CLI picked. */
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

/** Gate: a (stored dtype, kernel) cell is runnable once its kit chain is adapted in
 *  @c usearch/dense/ram_metric.hpp; unadapted combinations report as skipped. */
template <nk_dtype_t stored_dtype_, nk_kernel_kind_t kernel_, nk_capability_t capability_>
concept ram_kit_adapted = requires {
    sizeof(dense_quads_chain<kernel_, typename nk::type_for<stored_dtype_>::type, capability_>);
};

/** No-op sibling picked when the cell's CPU kit chain is not adapted yet. */
template <typename dataset_scalar_type_, nk_dtype_t stored_dtype_, nk_kernel_kind_t kernel_,
          nk_capability_t capability_, typename wait_type_, typename workload_type_>
int stage_and_run(bench_launch<dataset_scalar_type_> const&, workload_type_ const&) {
    fmt::println("skipped: no CPU kit adapted for {} · {}", dtype_name(stored_dtype_), nk_kernel_name(kernel_));
    return 0;
}

/** The innermost hop: binds the concrete sharded-index instantiation, wires the typed source views
 *  and the batch-shaped staging into a @c streamed_dataset, builds the shards, and hands them to
 *  @p workload. Staging is one contiguous plane, sized after the executor spawns so its rows follow
 *  the thread count. */
template <typename dataset_scalar_type_, nk_dtype_t stored_dtype_, nk_kernel_kind_t kernel_,
          nk_capability_t capability_, typename wait_type_, typename workload_type_>
    requires ram_kit_adapted<stored_dtype_, kernel_, capability_>
int stage_and_run(bench_launch<dataset_scalar_type_> const& launch, workload_type_ const& workload) {
    using indexes_t =
        dense_ram_indexes<u32_t, stored_dtype_, kernel_, capability_, wait_type_, fu::preferred_atomic_ref>;
    using bench_context_type_ = bench_indexes_context<indexes_t>;
    using streamed_t = streamed_dataset<dataset_scalar_type_, bench_context_type_>;
    using scalar_t = typename bench_context_type_::scalar_t;

    utilities::persisted_dataset<dataset_scalar_type_> const& dataset = launch.dataset;
    std::size_t const dimensions = dataset.dimensions();
    std::size_t const total_rows = dataset.vectors().extent(0);
    std::size_t const first_row = launch.settings.first_vector.value;
    if (first_row >= total_rows) {
        fmt::println("--first-vector {} exceeds the {} dataset rows", first_row, total_rows);
        return 1;
    }
    std::size_t rows = total_rows - first_row;
    if (launch.settings.max_vectors.value) rows = std::min(rows, launch.settings.max_vectors.value);
    nk::matrix_view<dataset_scalar_type_> const dataset_vectors = dataset.vectors();
    nk::matrix_view<dataset_scalar_type_> const sliced_vectors = dataset_vectors(nk::range(first_row, first_row + rows),
                                                                                 nk::slice);

    // Self-queries keep the harness usable on a lone vectors file, without recall.
    nk::matrix_view<dataset_scalar_type_> const source_queries =
        dataset.has_queries()
            ? dataset.queries()
            : sliced_vectors(nk::range(std::size_t {0}, std::min<std::size_t>(1000, rows)), nk::slice);

    bench_context_type_ bench {launch.settings};
    streamed_t streamed {bench, sliced_vectors, source_queries};
    if (!prepare(streamed, launch.machine.topology)) return 1;

    // Owns the one staging plane both batched phases view.
    nk::tensor<scalar_t, std::allocator<scalar_t>, 3> staging;
    // Owns the rescaling plane @c streamed_dataset::rescaling views.
    nk::tensor<nk::f32_t, std::allocator<nk::f32_t>, 3> rescaling_plane;
    if constexpr (!streamed_t::zero_copy_k) {
        std::size_t const threads = bench.executor.size();
        std::size_t const batch_rows = launch.settings.batch_per_core.value * threads;
        staging = nk::tensor<scalar_t, std::allocator<scalar_t>, 3>::zeros({1, batch_rows, dimensions}).value;
        if (staging.empty()) {
            fmt::println("Failed to allocate {} staging rows of {}d {}", batch_rows, dimensions,
                         scalar_t::dtype_name());
            return 1;
        }
        streamed.staging = staging.span();

        if constexpr (streamed_t::scaled_k) {
            rescaling_plane =
                nk::tensor<nk::f32_t, std::allocator<nk::f32_t>, 3>::zeros({1, batch_rows, dimensions}).value;
            if (rescaling_plane.empty()) {
                fmt::println("Failed to allocate {} rescaling rows of {}d f32", batch_rows, dimensions);
                return 1;
            }
            streamed.rescaling = rescaling_plane.span();
            streamed.establish_scale();
        }
    }

    if (dataset.has_ground_truth() && !launch.settings.max_vectors.value && !first_row)
        streamed.ground_truth = dataset.neighbors();
    else if (dataset.has_ground_truth())
        fmt::println("Ground truth ignored: --max-vectors or --first-vector reshapes the graph it was computed for");

    if (!run_insertion(streamed)) return 1;
    if (!run_refinement(streamed)) return 1;
    if (!launch.settings.save_name.empty() && !save_indexes(streamed)) return 1;
    return workload.run(streamed) ? 0 : 1;
}

/** Kernel axis: lowers the runtime metric to a compile-time kernel kind, the terminal hop into
 *  @ref stage_and_run; a dot-only workload pins the dot kernel instead. */
template <typename dataset_scalar_type_, nk_dtype_t stored_dtype_, nk_capability_t capability_, typename wait_type_,
          typename workload_type_>
int run_with_metric(bench_launch<dataset_scalar_type_> const& launch, workload_type_ const& workload) {
    if constexpr (workload_type_::dot_only_k)
        return stage_and_run<dataset_scalar_type_, stored_dtype_, nk_kernel_dot_k, capability_, wait_type_>(launch,
                                                                                                            workload);
    else
        return dispatch_kernel<stored_dtype_>(
            launch.settings.kernel,
            [&](auto kernel) {
                return stage_and_run<dataset_scalar_type_, stored_dtype_, kernel.value, capability_, wait_type_>(
                    launch, workload);
            },
            [&](auto metric) {
                fmt::println("Unsupported metric {} for {} vectors", nk_kernel_name(metric), dtype_name(stored_dtype_));
                return 1;
            });
}

/** Schedule axis: the newest schedule row the allow-mask admits, intersected with what this
 *  machine offers, picks the waiter from @ref schedule_waiters_t; past the tail sits the yield
 *  floor. */
template <typename dataset_scalar_type_, nk_dtype_t stored_dtype_, nk_capability_t capability_, typename workload_type_>
int run_with_schedule(bench_launch<dataset_scalar_type_> const& launch, workload_type_ const& workload) {
    return dispatch_index<std::size(schedule_choices_k)>(
        highest_index(schedule_choices_k, effective_schedule(launch.settings.max_schedule)),
        [&](auto row) {
            using wait_type_ = std::tuple_element_t<row.value, schedule_waiters_t>;
            return run_with_metric<dataset_scalar_type_, stored_dtype_, capability_, wait_type_>(launch, workload);
        },
        [&] {
            return run_with_metric<dataset_scalar_type_, stored_dtype_, capability_, yield_waiter_t>(launch, workload);
        });
}

/** Compute axis: the allow-mask intersected with the runtime detection picks the newest table
 *  row, whose @ref prefix_mask ceiling instantiates @ref run_with_schedule; past the tail sit the
 *  serial kernels every machine owns. */
template <typename dataset_scalar_type_, nk_dtype_t stored_dtype_, typename workload_type_>
int run_with_compute(bench_launch<dataset_scalar_type_> const& launch, workload_type_ const& workload) {
    nk_capability_t available = 0;
    if (nk_cpu_capabilities_enabled(&available) != nk_success_k) available = nk_cap_serial_k;
    nk_capability_t const effective = launch.settings.max_compute & available;
    return dispatch_index<std::size(compute_choices_k)>(
        highest_index(compute_choices_k, effective),
        [&](auto row) {
            return run_with_schedule<dataset_scalar_type_, stored_dtype_, prefix_mask(compute_choices_k, row.value)>(
                launch, workload);
        },
        [&] { return run_with_schedule<dataset_scalar_type_, stored_dtype_, nk_cap_serial_k>(launch, workload); });
}

/** The shard hop: everything below one stored dtype, so a per-dtype shard translation unit
 *  carries the whole instantiation grid of its dtype. */
template <typename dataset_scalar_type_, nk_dtype_t stored_dtype_>
int run_with_stored_dtype(bench_launch<dataset_scalar_type_> const& launch) {
    return run_with_compute<dataset_scalar_type_, stored_dtype_>(launch, search_workload_t {});
}

#pragma endregion

#pragma region Process Setup

/** The process every sharded CPU binary runs around its cascade: reads the flags of @p binary_name and the ones each
 *  of @p binary_flags adds, probes the machine, resolves the engaged memory domains and the executor's thread count
 *  against it, prints both, then opens the dataset and exits with what @p on_launch makes of its @ref bench_launch.
 *  No pool is created here: the executor spawns inside the run and owns its own. */
template <typename on_launch_type_, typename... binary_flags_types_>
int run_cpu_indexes_bench(int argc, char const* const* argv, char const* binary_name, on_launch_type_&& on_launch,
                          binary_flags_types_... binary_flags) {
    settings_t settings = read_settings(argc, argv, binary_name, binary_flags...);
    machine_t machine = probe_machine();
    if (!machine.topology || !machine.fabric) {
        fmt::println("Failed to inspect the topology");
        return 1;
    }

    // Resolve `--memory-domain` into the engaged-domain list: empty means one heap shard,
    // a single ID one pinned shard, and `all` one shard per memory domain - the list IS the
    // placement, no modes.
    if (settings.memory_domain_name == "all")
        for (std::size_t domain = 0; domain != machine.memory_domains; ++domain)
            settings.memory_domains.push_back(static_cast<fu_memory_domain_id_t>(domain));
    else if (!settings.memory_domain_name.empty()) {
        std::size_t domain = 0;
        char const* const end = settings.memory_domain_name.data() + settings.memory_domain_name.size();
        auto const [parsed_end, parse_status] = std::from_chars(settings.memory_domain_name.data(), end, domain);
        if (parse_status != std::errc {} || parsed_end != end || domain >= machine.memory_domains) {
            fmt::println(stderr, "--memory-domain=\"{}\" does not parse, expected all or a domain below {}",
                         settings.memory_domain_name, machine.memory_domains);
            return 1;
        }
        settings.memory_domains.push_back(static_cast<fu_memory_domain_id_t>(domain));
    }

    // The executor spawns inside the run; resolving its default here lets the report name it.
    if (!settings.threads.value) {
        auto const [threads, threads_status] = forkunion_executor_t::default_threads(machine.topology);
        if (failed(threads_status)) {
            fmt::println("Failed to count the logical cores");
            return 1;
        }
        settings.threads.value = threads;
    }
    print(machine);
    print(settings);
    if (!huge_pages_provisioned(machine, settings)) return 1;

    int const exit_code = open_dataset(
        settings.vectors_path, settings.queries_path, settings.neighbors_path,
        [&]<typename dataset_scalar_type_>(utilities::persisted_dataset<dataset_scalar_type_> const& dataset) {
            return on_launch(bench_launch<dataset_scalar_type_> {dataset, settings, machine});
        });
    fu_fabric_delete(machine.fabric);
    fu_topology_delete(machine.topology);
    return exit_code;
}

#pragma endregion

} // namespace unum::usearch::bench::dense
