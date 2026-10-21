/**
 *  @file bench/dense/ram_index_search.hpp
 *  @author Ash Vardanian
 *  @date July 17, 2026
 *  @brief Profiling harness for searching the CPU-resident @c dense::ram_index over real persisted
 *      datasets, shared by the per-dtype benchmark shards.
 *
 *  The backend-neutral surface - settings, checkpoints, dataset streaming and the process setup -
 *  lives in @c dense.cuh, the telemetry and result judgment in @c dense_search.cuh; this header owns
 *  everything CPU: the ForkUnion run loops streaming one row or query per task, the NUMA-domain
 *  placement, the waiter dispatch, and the compile-time cascade resolving the CLI into one concrete
 *  @c dense::ram_index instantiation, sharded per stored dtype. The cascade ends in the workload
 *  its entry hop names, the timed search for this binary.
 *
 *  This binary holds exactly one index; plurality belongs to the operating system or to the
 *  sharded sibling. N independent builds, one per memory domain over its own slice of one
 *  dataset, is an invocation rather than a code path:
 *
 *  @code{.sh}
 *  for domain in 0 1; do
 *  usearch_bench_dense_ram_index_search --memory-domain $domain --base-vectors base.10M.bf16bin \
 *  --first-vector $((domain * 5000000)) --max-base-vectors 5000000 --self-search &
 *  done; wait
 *  @endcode
 */
#pragma once

#include <algorithm> // `std::min`
#include <chrono>    // `std::chrono::steady_clock`
#include <memory>    // `std::unique_ptr`
#include <tuple>     // `std::tuple`, `std::tuple_element_t` for the schedule waiter list
#include <vector>    // `std::vector`

#include <fmt/format.h>

#include "usearch/dense/ram_index.hpp"
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
 *  @brief The CPU-side state one fully-typed benchmark run shares between its phases: the one
 *      index, its per-thread traversal contexts, and the embedded backend-neutral judgment
 *      results.
 *
 *  The dataset never appears here - the data side lives in @c streamed_dataset, which bridges
 *  the file's scalar into the stored one vector-by-vector.
 */
template <typename index_type_>
struct bench_context {
    using allocator_t = forkunion_allocator<std::byte>;
    using index_t = index_type_;
    using scalar_t = typename index_t::scalar_t;
    using small_size_t = typename index_t::small_size_t;
    using traversal_t = typename index_t::traversal_context_t;
    using vector_view_t = typename index_t::vector_view_t;
    using vectors_view_t = typename index_t::vectors_view_t;
    using distance_t = typename index_t::distance_t;

    /** The parsed CLI, borrowed for the run's lifetime. */
    settings_t const& settings;

    /** The metric the index scores by - the CLI's, or dot where the workload pins it. */
    nk_kernel_kind_t kernel;

    /** The one index under measurement; the indirection buys address stability. */
    std::unique_ptr<index_t> index;

    /** One graph-walking context per pool thread, built by that thread on its own domain. */
    std::vector<std::unique_ptr<traversal_t>> traversals;

    /** Per-thread telemetry sums, padded apart to stay contention-free. */
    std::vector<telemetry_lane_t> lanes;

    /** The backend-neutral judgment state both phases fill. */
    bench_results_t results;

    bench_context(settings_t const& settings, nk_kernel_kind_t kernel) noexcept : settings(settings), kernel(kernel) {}
};

/** Builds the index on the chosen memory domain, reserves it for the viewed rows, and sizes all
 *  per-thread state off the pool's thread count - everything fallible, before any clock starts. */
template <typename streamed_dataset_type_>
bool prepare(streamed_dataset_type_& streamed, fu_pool_t pool) {
    using bench_t = typename streamed_dataset_type_::bench_t;
    using index_t = typename bench_t::index_t;
    using allocator_t = typename bench_t::allocator_t;
    using traversal_t = typename bench_t::traversal_t;
    auto& bench = streamed.bench;
    std::size_t const rows = streamed.vectors.extent(0);
    std::size_t const dimensions = streamed.vectors.extent(1);
    std::size_t threads = 0;
    if (fu_pool_threads_count(pool, &threads) != fu_success_k) {
        fmt::println("Failed to read the pool's shape");
        return false;
    }

    connectivity_t const connectivity = connectivity_of(bench.settings.connectivity.value);
    if (connectivity == connectivity_unknown_k) {
        fmt::println("Unsupported --connectivity {}; pick 16, 32, or 128", bench.settings.connectivity.value);
        return false;
    }

    if (bench.settings.skip_vectors.value > rows) {
        fmt::println("--skip-vectors {} exceeds the {} viewed rows", bench.settings.skip_vectors.value, rows);
        return false;
    }

    // Reserving the full row count is cheap: chunk storage allocates lazily.
    fu_memory_domain_id_t const domain = bench.settings.memory_domains.empty() //
                                             ? fu_memory_domain_id_t {-1}
                                             : bench.settings.memory_domains.front();
    bench.index = std::make_unique<index_t>(bench.kernel, allocator_t {domain}, allocator_t {domain},
                                            machine_cache_line_bytes());
    streamed.id_to_row.resize(rows);

    // Reserving first freezes the CLI geometry a checkpoint then has to agree with.
    auto const [capacity, status] = bench.index->reserve({rows, connectivity, dimensions});
    if (failed(status)) {
        fmt::println("Failed to reserve {} × {}d", rows, dimensions);
        return false;
    }

    // The keys are the ID → row map, the recall bookkeeping the index itself never stores.
    if (!bench.settings.load_name.empty() &&
        !load_checkpoint(bench.settings, *bench.index, std::span {streamed.id_to_row}))
        return false;

    // A base that misses the skipped prefix would judge recall against rows the graph never saw.
    std::size_t const loaded_rows = bench.index->size();
    if (loaded_rows != bench.settings.skip_vectors.value) {
        fmt::println("Loaded index covers {} rows, but --skip-vectors says {}", loaded_rows,
                     bench.settings.skip_vectors.value);
        return false;
    }

    // Each thread builds its own context, so the scratch lands on its local memory domain.
    bench.traversals.resize(threads);
    auto traversals_init = [](fu_lambda_context_t punned, std::size_t thread, std::size_t) noexcept {
        auto& bench = *static_cast<bench_t*>(punned);
        // AMX tile state needs a per-thread OS permission handshake before first use.
        nk_capability_t available = 0;
        if (nk_cpu_capabilities_enabled(&available) != nk_success_k) available = nk_cap_serial_k;
        [[maybe_unused]] nk_status_t const configured = nk_cpu_configure_thread(available);
        fu_memory_domain_id_t const domain = bench.settings.memory_domains.empty()
                                                 ? fu_memory_domain_id_t {-1}
                                                 : bench.settings.memory_domains.front();
        bench.traversals[thread] = std::make_unique<traversal_t>(
            typename traversal_t::allocator_t {allocator_t {domain}});
    };
    fu_pool_for_threads(pool, traversals_init, &bench);
    for (std::size_t thread = 0; thread != threads; ++thread)
        if (!bench.traversals[thread]) {
            fmt::println("Failed to build the per-thread traversal contexts");
            return false;
        }
    bench.lanes.resize(threads);
    return true;
}

/** One vector per task: viewed in place when the scalars match, or cast through the thread's
 *  staging area, then inserted into the one index under measurement. */
template <typename streamed_dataset_type_>
void insertion_callback(fu_lambda_context_t punned, std::size_t task, std::size_t thread, std::size_t) noexcept {
    using bench_t = typename streamed_dataset_type_::bench_t;
    using index_t = typename bench_t::index_t;
    using small_size_t = typename bench_t::small_size_t;
    using vector_view_t = typename bench_t::vector_view_t;
    using vectors_view_t = typename bench_t::vectors_view_t;
    streamed_dataset_type_& streamed = *static_cast<streamed_dataset_type_*>(punned);
    auto& bench = streamed.bench;

    typename index_t::insert_config_t config;
    config.expansion_factor = static_cast<small_size_t>(bench.settings.expansion_add.value);
    // With a loaded base, tasks cover only the topping - shift into absolute dataset rows.
    std::size_t const row = task + bench.settings.skip_vectors.value;
    vector_view_t vector;
    if constexpr (streamed_dataset_type_::zero_copy_k) vector = streamed.vectors.row(row).as_vector();
    else vector = streamed.stage(streamed.vectors.row(row).as_vector(), thread, task);

    node_id_t local_id = 0;
    vectors_view_t const one_row {vector.data(), 1, streamed.vectors.extent(1)};
    auto const [result, status] = bench.index->insert(one_row, {&local_id, std::size_t {1}}, *bench.traversals[thread],
                                                      config);
    if (failed(status)) {
        standard_atomic_ref<std::size_t>(bench.results.failed_tasks).fetch_add(1, std::memory_order_relaxed);
        return;
    }

    telemetry_lane_t& lane = bench.lanes[thread];
    lane.visited_members += result.visited_members;
    lane.computed_distances += result.computed_distances;
    lane.computed_distances_in_refines += result.computed_distances_in_refines;
    lane.computed_distances_in_reverse_refines += result.computed_distances_in_reverse_refines;
    lane.restarts += result.restarts;
    lane.saturated_walks += result.saturated_walks;
    streamed.id_to_row[local_id] = static_cast<u32_t>(bench.settings.first_vector.value + row);
}

template <typename streamed_dataset_type_>
bool run_insertion(streamed_dataset_type_& streamed, fu_pool_t pool) {
    using scalar_t = typename streamed_dataset_type_::scalar_t;
    auto& bench = streamed.bench;
    std::size_t const rows = streamed.vectors.extent(0);
    std::size_t const inserted_rows = rows - bench.settings.skip_vectors.value;
    if (!inserted_rows) {
        fmt::println("\nInsertions");
        fmt::println("  skipped: the loaded base already covers all {} rows", rows);
        return true;
    }

    auto started = steady_clock_t::now();
    fu_pool_for_n_dynamic(pool, inserted_rows, &insertion_callback<streamed_dataset_type_>, &streamed);
    double const seconds = stdc::duration<double>(steady_clock_t::now() - started).count();

    if (std::size_t failures = bench.results.failed_tasks; failures) {
        fmt::println("Insertion failed on {} vectors", failures);
        return false;
    }
    telemetry_lane_t const sums = summed_lanes(bench.lanes);

    using index_t = typename streamed_dataset_type_::bench_t::index_t;
    double const refine_share = sums.computed_distances ? 100.0 * double(sums.computed_distances_in_refines) /
                                                              double(sums.computed_distances)
                                                        : 0.0;
    double const reverse_share = sums.computed_distances ? 100.0 * double(sums.computed_distances_in_reverse_refines) /
                                                               double(sums.computed_distances)
                                                         : 0.0;
    fmt::println("\nInsertions");
    fmt::println("  shape: {} × {}d", inserted_rows, streamed.vectors.extent(1));
    if (bench.settings.skip_vectors.value) fmt::println("  topping: {} loaded rows", bench.settings.skip_vectors.value);
    fmt::println("  quads: {} · {} · {} kit", scalar_t::dtype_name(), nk_kernel_name(bench.kernel),
                 bit_name(compute_choices_k, index_t::quads_kit_t::capability_k, "serial"));
    nk_capability_t available = 0;
    if (nk_cpu_capabilities_enabled(&available) != nk_success_k) available = nk_cap_serial_k;
    if (index_t::quads_kit_t::capability_k == nk_cap_serial_k &&
        (bench.settings.max_compute & available & ~nk_cap_serial_k))
        fmt::println("  ⚠ scalar fallback: no SIMD kit is adapted for this dtype & metric");
    std::size_t reported_threads = 0;
    [[maybe_unused]] fu_status_t const counted = fu_pool_threads_count(pool, &reported_threads);
    fmt::println("  threads: {}", reported_threads);
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

/** The state one refinement pass shares with its threads: the plan it drains, and the tasks that refused. */
template <typename bench_type_>
struct refinement_fanout {
    using index_t = typename bench_type_::index_t;

    /** The index under measurement and its per-thread traversal contexts. */
    bench_type_& bench;

    /** The pass in flight. */
    typename index_t::refine_plan_t& plan;

    /** Tasks that returned an error; any non-zero fails the run. */
    std::size_t failures = 0;
};

/** One step of the plan's current phase per task, on the thread that pulled it. */
template <typename bench_type_>
void refinement_callback(fu_lambda_context_t punned, std::size_t task, std::size_t thread, std::size_t) noexcept {
    auto& fanout = *static_cast<refinement_fanout<bench_type_>*>(punned);
    if (failed(fanout.plan.run(fanout.plan[task], *fanout.bench.traversals[thread])))
        standard_atomic_ref<std::size_t>(fanout.failures).fetch_add(1, std::memory_order_relaxed);
}

/** Refines every node, pass after pass - each pass re-selects the graph, then heals whatever that strands - and
 *  stops at the first pass that re-wires nothing. */
template <typename streamed_dataset_type_>
bool run_refinement(streamed_dataset_type_& streamed, fu_pool_t pool) {
    using bench_t = typename streamed_dataset_type_::bench_t;
    using index_t = typename bench_t::index_t;
    using node_id_t = typename index_t::node_id_t;
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
    [[maybe_unused]] fu_status_t const counted = fu_pool_threads_count(pool, &report.threads);

    // The passes' steps cannot fail on contexts reserved for their walks.
    for (std::unique_ptr<typename bench_t::traversal_t> const& traversal : bench.traversals)
        if (failed(bench.index->reserve_traversal(*traversal, {config.expansion_factor}))) {
            fmt::println("Failed to reserve the refinement walks");
            return false;
        }

    std::vector<std::size_t> changed_per_pass;
    changed_per_pass.reserve(max_refines);
    progress_bar_t progress {"refining", max_refines, bench.settings.terminal};
    for (std::size_t pass = 0; pass != max_refines; ++pass) {
        auto [plan, plan_status] = bench.index->plan_refine({0, static_cast<node_id_t>(nodes)}, config);
        if (failed(plan_status)) {
            fmt::println("Failed to open refinement pass {}: status {}", pass, int(plan_status));
            return false;
        }
        refinement_fanout<bench_t> fanout {bench, plan};
        auto started = steady_clock_t::now();
        do {
            // The re-selection is the refining share of the wall time, the audit and the rounds after it the healing.
            bool const reselecting = plan.reselecting();
            fu_pool_for_n_dynamic(pool, plan.steps().size(), &refinement_callback<bench_t>, &fanout);
            (reselecting ? report.refining_seconds
                         : report.healing_seconds) += stdc::duration<double>(steady_clock_t::now() - started).count();
            started = steady_clock_t::now();
        } while (plan.advance());
        if (std::size_t failures = fanout.failures; failures) {
            fmt::println("Refinement failed on {} steps", failures);
            return false;
        }
        auto const [pass_result, pass_status] = plan.finish();
        if (failed(pass_status)) {
            fmt::println("Refinement pass {} failed: status {}", pass, int(pass_status));
            return false;
        }
        changed_per_pass.push_back(pass_result.changed_nodes);
        report.islands_found += pass_result.islands_found;
        report.islands_reconnected += pass_result.islands_reconnected;
        report.reachable_nodes = pass_result.health.reachable_nodes;
        report.sweeps = pass_result.health.sweeps;
        report.sums.visited_members += pass_result.visited_members;
        report.sums.computed_distances += pass_result.computed_distances;
        report.sums.restarts += pass_result.restarts;
        progress.tick(pass + 1);
        if (!pass_result.changed_nodes) break;
    }
    progress.finish();
    report.changed_per_pass = changed_per_pass;
    report_refinement_metrics(report);
    return true;
}

/** Serializes the index as one SafeGraphs monolith keyed by the ID → row map, so later runs
 *  @c --load the base and stream only the topping, and the NVMe benches view it. */
template <typename streamed_dataset_type_>
bool save_index(streamed_dataset_type_& streamed) {
    auto& bench = streamed.bench;
    auto started = steady_clock_t::now();
    std::size_t const checkpoint_bytes = save_checkpoint(bench.settings, *bench.index,
                                                         std::span<u32_t const> {streamed.id_to_row}, streamed.scale);
    if (!checkpoint_bytes) return false;
    double const seconds = stdc::duration<double>(steady_clock_t::now() - started).count();
    fmt::println("Saved the index: {:.2f} GB in {:.2f} s", gigabytes(checkpoint_bytes), seconds);
    return true;
}

/** The search phase's shared state: the streamed dataset beside the indexer heads every walk ranks by, if any. */
template <typename streamed_dataset_type_>
struct search_task {
    streamed_dataset_type_& streamed;
    bench_indexer_heads<typename streamed_dataset_type_::scalar_t> const& heads;
};

/** Every query is one task: cast through the thread's staging area if needed - before the clock - then searched,
 *  ranked by its heads under an indexer, and translated through the ID → row map into the judged hits. */
template <typename streamed_dataset_type_>
void search_callback(fu_lambda_context_t punned, std::size_t query, std::size_t thread, std::size_t) noexcept {
    using bench_t = typename streamed_dataset_type_::bench_t;
    using index_t = typename bench_t::index_t;
    using small_size_t = typename bench_t::small_size_t;
    using vector_view_t = typename bench_t::vector_view_t;
    auto& task = *static_cast<search_task<streamed_dataset_type_>*>(punned);
    streamed_dataset_type_& streamed = task.streamed;
    auto& bench = streamed.bench;
    telemetry_lane_t& lane = bench.lanes[thread];

    typename index_t::search_config_t config;
    config.expansion_factor = static_cast<small_size_t>(bench.settings.expansion_search.value);
    // The search phase runs strictly after ingestion, so the mutation-free contract holds.
    config.synchronization = synchronization_t::immutable_k;
    std::size_t const dimensions = streamed.vectors.extent(1);
    vector_view_t query_vector;
    if (!task.heads.navigation.empty()) query_vector = {task.heads.navigation.data() + query * dimensions, dimensions};
    else if constexpr (streamed_dataset_type_::zero_copy_k)
        query_vector = searched_rows(streamed).row(query).as_vector();
    else query_vector = streamed.stage(searched_rows(streamed).row(query).as_vector(), thread, query);

    u32_t* const merged_rows = bench.results.hit_rows.row(query).data();
    auto* const merged_distances = bench.results.hit_distances.row(query).data();

    // The index stores no keys, so matches translate through the ID → row map on the way out.
    typename bench_t::traversal_t& traversal = *bench.traversals[thread];
    std::span<u32_t> const rows_span {merged_rows, bench.settings.top_k.value};
    std::span<distance_t> const distances_span {merged_distances, bench.settings.top_k.value};
    auto search = [&]() noexcept {
        if constexpr (!nk::is_integral_dtype<typename index_t::scalar_t>())
            if (task.heads.engaged())
                return bench.index->search(query_vector, task.heads.rows(query, 1), traversal, config);
        return bench.index->search(query_vector, traversal, config);
    };
    auto started = steady_clock_t::now();
    auto const [result, status] = search();
    if (failed(status)) {
        standard_atomic_ref<std::size_t>(bench.results.failed_tasks).fetch_add(1, std::memory_order_relaxed);
        return;
    }
    lane.visited_members += result.visited_members;
    lane.computed_distances += result.computed_distances;
    lane.restarts += result.restarts;
    lane.saturated_walks += result.saturated_walks;
    std::size_t const merged_count = merge_search_results(to_span(traversal.results), streamed.id_to_row.data(),
                                                          rows_span, distances_span, std::size_t {0});
    bench.results.latencies_us[query] = stdc::duration<float, std::micro>(steady_clock_t::now() - started).count();
    bench.results.hit_counts[query] = static_cast<u32_t>(merged_count);
}

template <typename streamed_dataset_type_>
bool run_search(streamed_dataset_type_& streamed, fu_pool_t pool) {
    using index_t = typename streamed_dataset_type_::bench_t::index_t;
    auto& bench = streamed.bench;
    bench_indexer_heads<typename index_t::scalar_t> heads;
    if (!open_indexer_heads(*bench.index, streamed, heads)) return false;
    std::size_t const query_count = searched_count(streamed, heads);
    if (!reserve_results(bench.results, query_count, bench.settings.top_k.value)) return false;
    bench.results.latencies_us.resize(query_count);
    search_task<streamed_dataset_type_> task {streamed, heads};
    for (telemetry_lane_t& lane : bench.lanes) lane = telemetry_lane_t {};

    auto started = steady_clock_t::now();
    fu_pool_for_n(pool, query_count, &search_callback<streamed_dataset_type_>, &task);
    double const seconds = stdc::duration<double>(steady_clock_t::now() - started).count();

    if (std::size_t failures = bench.results.failed_tasks; failures) {
        fmt::println("Search failed on {} queries", failures);
        return false;
    }
    telemetry_lane_t const sums = summed_lanes(bench.lanes);
    std::size_t searching_threads = 0;
    [[maybe_unused]] fu_status_t const counted = fu_pool_threads_count(pool, &searching_threads);
    report_search_metrics(bench.settings, query_count, seconds, searching_threads, bench.index->memory_usage(), sums,
                          bench.results.latencies_us);
    judge_search(bench.settings, heads, bench.results, streamed.ground_truth, streamed.vectors, streamed.queries,
                 query_count);
    return true;
}

/** The search workload the cascade ends in: the timed search phase and its judgment, under whichever
 *  kernel the CLI picked. */
struct search_workload_t {

    /** Whether the workload is defined over the dot kernel alone, so the cascade pins it instead of
     *  dispatching the CLI's metric. */
    static constexpr bool dot_only_k = false;

    template <typename streamed_dataset_type_>
    bool run(streamed_dataset_type_& streamed, fu_pool_t pool) const {
        return run_search(streamed, pool);
    }
};

#pragma endregion

#pragma region Dispatch Cascade

/** No-op sibling picked when the cell's CPU kit chain is not adapted yet. */
template <typename dataset_scalar_type_, nk_dtype_t stored_dtype_, nk_capability_t capability_, typename wait_type_,
          typename workload_type_>
int stage_and_run(bench_launch<dataset_scalar_type_> const&, nk_kernel_kind_t kernel, workload_type_ const&) {
    fmt::println("skipped: no CPU kit adapted for {} · {}", dtype_name(stored_dtype_), nk_kernel_name(kernel));
    return 0;
}

/** The innermost hop: binds the concrete volatile-index instantiation, wires the typed source
 *  views and the bounded staging trampoline into a @c streamed_dataset, builds the index under
 *  @p kernel, and hands it to @p workload. */
template <typename dataset_scalar_type_, nk_dtype_t stored_dtype_, nk_capability_t capability_, typename wait_type_,
          typename workload_type_>
    requires quads_adapted<typename nk::type_for<stored_dtype_>::type, capability_>
int stage_and_run(bench_launch<dataset_scalar_type_> const& launch, nk_kernel_kind_t kernel,
                  workload_type_ const& workload) {
    using allocator_t = forkunion_allocator<std::byte>;
    using index_t = ram_index<typename quads_ladder<typename nk::type_for<stored_dtype_>::type, capability_>::kit_t,
                              typename tiles_ladder<typename nk::type_for<stored_dtype_>::type, capability_>::kit_t,
                              typename elementwise_ladder<capability_>::kit_t,
                              typename rmscast_ladder<typename nk::type_for<stored_dtype_>::type, capability_>::kit_t,
                              typename visits_hash_set_ladder<capability_>::kit_t, wait_type_, fu::preferred_atomic_ref,
                              fu::preferred_cache_hints_t, allocator_t, allocator_t, allocator_t>;
    using bench_context_type_ = bench_context<index_t>;
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
    if (launch.settings.max_base_vectors.value) rows = std::min(rows, launch.settings.max_base_vectors.value);
    nk::matrix_view<dataset_scalar_type_> const dataset_vectors = dataset.vectors();
    nk::matrix_view<dataset_scalar_type_> const sliced_vectors = dataset_vectors(nk::range(first_row, first_row + rows),
                                                                                 nk::slice);

    // Self-queries keep the harness usable on a lone vectors file, without recall.
    nk::matrix_view<dataset_scalar_type_> const source_queries =
        dataset.has_queries()
            ? dataset.queries()
            : sliced_vectors(nk::range(std::size_t {0}, std::min<std::size_t>(1000, rows)), nk::slice);

    bench_context_type_ bench {launch.settings, kernel};
    streamed_t streamed {bench, sliced_vectors, source_queries};

    // Owns the staging trampoline rows for the whole run.
    nk::tensor<scalar_t, std::allocator<scalar_t>, 3> staging;
    // Owns the rescaling rows `streamed_dataset::rescaling` views.
    nk::tensor<nk::f32_t, std::allocator<nk::f32_t>, 3> rescaling_rows;
    if constexpr (!streamed_t::zero_copy_k) {
        std::size_t threads = 0;
        if (fu_pool_threads_count(launch.machine.pool, &threads) != fu_success_k || !threads) {
            fmt::println("Failed to read the pool's thread count");
            return 1;
        }
        // Every task inserts or searches its one staged row before the thread stages the next.
        staging = nk::tensor<scalar_t, std::allocator<scalar_t>, 3>::zeros({threads, 1, dimensions}).value;
        if (staging.empty()) {
            fmt::println("Failed to allocate {} staging rows of {}d {}", threads, dimensions, scalar_t::dtype_name());
            return 1;
        }
        streamed.staging = staging.span();

        if constexpr (streamed_t::scaled_k) {
            rescaling_rows = nk::tensor<nk::f32_t, std::allocator<nk::f32_t>, 3>::zeros({threads, 1, dimensions}).value;
            if (rescaling_rows.empty()) {
                fmt::println("Failed to allocate {} rescaling rows of {}d f32", threads, dimensions);
                return 1;
            }
            streamed.rescaling = rescaling_rows.span();
            streamed.establish_scale();
        }
    }

    if (dataset.has_ground_truth() && !launch.settings.max_base_vectors.value && !first_row)
        streamed.ground_truth = dataset.neighbors();
    else if (dataset.has_ground_truth())
        fmt::println(
            "Ground truth ignored: --max-base-vectors or --first-vector reshapes the graph it was computed for");

    if (!prepare(streamed, launch.machine.pool)) return 1;
    if (!run_insertion(streamed, launch.machine.pool)) return 1;
    if (!run_refinement(streamed, launch.machine.pool)) return 1;
    if (!launch.settings.save_name.empty() && !save_index(streamed)) return 1;
    return workload.run(streamed, launch.machine.pool) ? 0 : 1;
}

/** Kernel axis: admits the runtime metric the stored dtype scores by, the terminal hop into
 *  @ref stage_and_run; a dot-only workload pins the dot kernel instead. */
template <typename dataset_scalar_type_, nk_dtype_t stored_dtype_, nk_capability_t capability_, typename wait_type_,
          typename workload_type_>
int run_with_metric(bench_launch<dataset_scalar_type_> const& launch, workload_type_ const& workload) {
    if constexpr (workload_type_::dot_only_k)
        return stage_and_run<dataset_scalar_type_, stored_dtype_, capability_, wait_type_>(launch, nk_kernel_dot_k,
                                                                                           workload);
    else
        return dispatch_kernel<stored_dtype_>(
            launch.settings.kernel,
            [&](auto kernel) {
                return stage_and_run<dataset_scalar_type_, stored_dtype_, capability_, wait_type_>(launch, kernel.value,
                                                                                                   workload);
            },
            [&](auto metric) {
                fmt::println("Unsupported metric {} for {} vectors", nk_kernel_name(metric), dtype_name(stored_dtype_));
                return 1;
            });
}

/** Schedule axis: the newest schedule row the allow-mask admits, intersected with what this
 *  machine offers, picks the waiter from @ref schedule_waiters_t; past the tail sits the yield
 *  floor. The same mask already reached @c fu_pool_new, so the pool waits the same way. */
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

} // namespace unum::usearch::bench::dense
