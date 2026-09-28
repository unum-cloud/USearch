/**
 *  @file bench/dense/ram_index.hpp
 *  @author Ash Vardanian
 *  @date July 17, 2026
 *  @brief Profiling harness for the CPU-resident @c dense_ram_index over real persisted datasets,
 *      shared by the per-dtype benchmark shards.
 *
 *  The backend-neutral surface - options, checkpoints, dataset streaming, telemetry, result
 *  judgment, and the machine report - lives in @c dense.cuh; this header owns everything CPU:
 *  the ForkUnion run loops streaming one row or query per task, the NUMA-domain placement,
 *  the waiter dispatch, the compile-time cascade resolving the CLI into one concrete
 *  @c dense_ram_index instantiation, sharded per stored dtype, and the process setup every
 *  CPU single-index binary shares. The cascade ends in the workload its entry hop names, the
 *  timed search for this binary. A ready-made playground is the Unum Wiki 1M dataset:
 *
 *  @code{.sh}
 *  mkdir -p ~/datasets/wiki_1M && cd ~/datasets/wiki_1M &&
 *  wget -nc https://huggingface.co/datasets/unum-cloud/ann-wiki-1m/resolve/main/base.1M.fbin &&
 *  wget -nc https://huggingface.co/datasets/unum-cloud/ann-wiki-1m/resolve/main/query.public.100K.fbin &&
 *  wget -nc https://huggingface.co/datasets/unum-cloud/ann-wiki-1m/resolve/main/groundtruth.public.100K.ibin
 *  usearch_bench_dense_ram_index --vectors ~/datasets/wiki_1M/base.1M.fbin \
 *  --queries ~/datasets/wiki_1M/query.public.100K.fbin \
 *  --neighbors ~/datasets/wiki_1M/groundtruth.public.100K.ibin --bf16
 *  @endcode
 *
 *  Huge pages are provisioned outside the process; `--huge-pages` declares the expectation,
 *  and a mismatch fails the run instead of silently benchmarking base pages:
 *
 *  @code{.sh}
 *  sudo sysctl vm.nr_hugepages=16384        # 32 GB of 2 MB pages, split across the domains
 *  usearch_bench_dense_ram_index --huge-pages 16384 \
 *  --vectors ~/datasets/sift_10M/base.10M.u8bin --memory-domain 0
 *  @endcode
 *
 *  This binary holds exactly one index; plurality belongs to the operating system or to the
 *  sharded sibling. N independent builds, one per memory domain over its own slice of one
 *  dataset, is an invocation rather than a code path:
 *
 *  @code{.sh}
 *  for domain in 0 1; do
 *  usearch_bench_dense_ram_index --memory-domain $domain --vectors base.10M.bf16bin \
 *  --first-vector $((domain * 5000000)) --max-vectors 5000000 --self-recall &
 *  done; wait
 *  @endcode
 *
 *  Defaults are inferred and each is overridable:
 *  - the stored dtype matches the dataset file, unless `--f32`, `--bf16`, `--f16`, `--i8`, `--i4`,
 *  `--u8`, or `--u1` asks for a streaming `nk::cast` through a `--batch`-row trampoline on ingest;
 *  - the newest runtime compute capability drives the distance kernels, unless
 *  `--max-compute` caps it by NumKong's microarchitecture names, like `icelake`;
 *  - the strongest offered wait instruction parks stalled threads, unless `--max-schedule`
 *  caps it by ForkUnion's instruction names, like `tpause`, down to the `standard` yield;
 *  - 7/8 of the logical cores participate, unless `--threads` overrides - full occupancy
 *  starves the ordered publication path and collapses throughput.
 */
#pragma once

#include <cmath>   // `std::abs`
#include <cstdlib> // `std::strtol` - the `--memory-domain` ordinal

#include <algorithm> // `std::max`, `std::min`, `std::copy_n`
#include <chrono>    // `std::chrono::steady_clock`
#include <memory>    // `std::unique_ptr`
#include <string>    // `std::string`
#include <tuple>     // `std::tuple`, `std::tuple_element_t` for the schedule waiter list
#include <vector>    // `std::vector`

#include <clipp.h>
#include <fmt/format.h>

#include <usearch/dense/ram_index.hpp>
#include <usearch/utilities/ram_device.hpp>

#include "dense.cuh"
#include "utilities/persisted_dataset.hpp"

namespace unum::usearch::bench::dense {

#pragma region CPU Pipeline

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
    using traversal_t = typename index_t::dense_traversal_context_t;
    using vector_view_t = typename index_t::vector_view_t;
    using vectors_view_t = typename index_t::vectors_view_t;
    using distance_t = typename index_t::distance_t;

    static constexpr nk_kernel_kind_t kernel_kind_k = index_t::kernel_kind_k;

    /** The parsed CLI, borrowed for the run's lifetime. */
    bench_options_t const& options;

    /** The one index under measurement; the indirection buys address stability. */
    std::unique_ptr<index_t> index;

    /** One graph-walking context per pool thread, built by that thread on its own domain. */
    std::vector<std::unique_ptr<traversal_t>> traversals;

    /** Per-thread telemetry sums, padded apart to stay contention-free. */
    std::vector<telemetry_lane_t> lanes;

    /** The backend-neutral judgment state both phases fill. */
    bench_results_t results;

    bench_context(bench_options_t const& options) noexcept : options(options) {}
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

    connectivity_t const connectivity = connectivity_of(bench.options.connectivity);
    if (connectivity == connectivity_unknown_k) {
        fmt::println("Unsupported --connectivity {}; pick 16, 32, or 128", bench.options.connectivity);
        return false;
    }

    if (bench.options.skip_vectors > rows) {
        fmt::println("--skip-vectors {} exceeds the {} viewed rows", bench.options.skip_vectors, rows);
        return false;
    }

    // Reserving the full row count is cheap: chunk storage allocates lazily.
    fu_memory_domain_id_t const domain = bench.options.memory_domains.empty() //
                                             ? fu_memory_domain_id_t {-1}
                                             : bench.options.memory_domains.front();
    bench.index = std::make_unique<index_t>(allocator_t {domain}, allocator_t {domain});
    bench.results.id_to_row.resize(rows);

    // Reserving first freezes the CLI geometry a checkpoint then has to agree with.
    auto const [capacity, status] = bench.index->reserve({rows, connectivity, dimensions});
    if (failed(status)) {
        fmt::println("Failed to reserve {} × {}d", rows, dimensions);
        return false;
    }

    // The third dump is the ID → row map, the recall bookkeeping the index itself never stores.
    if (!bench.options.load_name.empty() &&
        !load_checkpoint(bench.options, *bench.index, dimensions, std::span {bench.results.id_to_row}, 0, 1))
        return false;

    // A base that misses the skipped prefix would judge recall against rows the graph never saw.
    std::size_t const loaded_rows = bench.index->size();
    if (loaded_rows != bench.options.skip_vectors) {
        fmt::println("Loaded index covers {} rows, but --skip-vectors says {}", loaded_rows,
                     bench.options.skip_vectors);
        return false;
    }

    // Each thread builds its own context, so the scratch lands on its local memory domain.
    bench.traversals.resize(threads);
    auto traversals_init = [](fu_lambda_context_t punned, std::size_t thread, std::size_t) noexcept {
        auto& bench = *static_cast<bench_t*>(punned);
        // AMX tile state needs a per-thread OS permission handshake before first use.
        nk_configure_thread(nk_capabilities_available());
        fu_memory_domain_id_t const domain = bench.options.memory_domains.empty()
                                                 ? fu_memory_domain_id_t {-1}
                                                 : bench.options.memory_domains.front();
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
    bench.results.latencies_us.resize(search_count);
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
    config.expansion_factor = static_cast<small_size_t>(bench.options.expansion_add);
    // With a loaded base, tasks cover only the topping - shift into absolute dataset rows.
    std::size_t const row = task + bench.options.skip_vectors;
    vector_view_t vector;
    if constexpr (streamed_dataset_type_::zero_copy_k) vector = streamed.vectors.row(row).as_vector();
    else vector = streamed.stage(streamed.vectors.row(row).as_vector(), thread, task);

    dense_node_id_t local_id = 0;
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
    bench.results.id_to_row[local_id] = static_cast<u32_t>(row);
}

template <typename streamed_dataset_type_>
bool run_insertion(streamed_dataset_type_& streamed, fu_pool_t pool) {
    using clock_t = stdc::steady_clock;
    using scalar_t = typename streamed_dataset_type_::scalar_t;
    auto& bench = streamed.bench;
    std::size_t const rows = streamed.vectors.extent(0);
    std::size_t const inserted_rows = rows - bench.options.skip_vectors;
    if (!inserted_rows) {
        fmt::print(header_style(), "\nInsertions\n");
        fmt::println("  skipped: the loaded base already covers all {} rows", rows);
        return true;
    }

    auto started = clock_t::now();
    fu_pool_for_n_dynamic(pool, inserted_rows, &insertion_callback<streamed_dataset_type_>, &streamed);
    double const seconds = stdc::duration<double>(clock_t::now() - started).count();

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
    fmt::print(header_style(), "\nInsertions\n");
    fmt::println("  shape: {} × {}d", inserted_rows, streamed.vectors.extent(1));
    if (bench.options.skip_vectors) fmt::println("  topping: {} loaded rows", bench.options.skip_vectors);
    fmt::println("  quads: {} · {} · {} kit", scalar_t::dtype_name(),
                 nk_kernel_name(streamed_dataset_type_::bench_t::kernel_kind_k),
                 bit_name(compute_choices_k, index_t::picked_quads_capability_k, "serial"));
    if constexpr (index_t::has_tiles_k)
        fmt::println("  tiles: {} · reverse refinement",
                     bit_name(compute_choices_k, index_t::picked_tiles_capability_k, "none"));
    if (index_t::picked_quads_capability_k == nk_cap_serial_k &&
        (bench.options.max_compute & nk_capabilities_available() & ~nk_cap_serial_k))
        fmt::print(verdict_style(false), "  ⚠ scalar fallback: no SIMD kit is adapted for this dtype & metric\n");
    std::size_t reported_threads = 0;
    [[maybe_unused]] fu_status_t const counted = fu_pool_threads_count(pool, &reported_threads);
    fmt::println("  threads: {}", reported_threads);
    fmt::println("  wall: {:.2f} s", seconds);
    fmt::println("  throughput: {} IPS", humanized(double(inserted_rows) / seconds));
    fmt::println("  distances: {} ({:.1f}% refines, {:.1f}% reverse)", humanized(sums.computed_distances), refine_share,
                 reverse_share);
    fmt::println("  visited: {}", humanized(sums.visited_members));
    fmt::println("  restarts: {}", fmt::styled(sums.restarts, verdict_style(sums.restarts < inserted_rows / 100 + 1)));
    fmt::println("  saturated: {}", fmt::styled(sums.saturated_walks, verdict_style(sums.saturated_walks == 0)));
    if (std::size_t resident = peak_resident_bytes(); resident)
        fmt::println("  peak resident: {:.2f} GB", double(resident) / 1e9);
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
    using clock_t = stdc::steady_clock;
    using bench_t = typename streamed_dataset_type_::bench_t;
    using index_t = typename bench_t::index_t;
    using node_id_t = typename index_t::node_id_t;
    using small_size_t = typename bench_t::small_size_t;
    auto& bench = streamed.bench;
    std::size_t const max_refines = bench.options.max_refines;
    std::size_t const nodes = bench.index->size();
    if (!max_refines || !nodes) return true;

    typename index_t::refine_config_t config;
    config.expansion_factor = static_cast<small_size_t>(bench.options.expansion_refine);
    refinement_report_t report;
    report.nodes = nodes;
    report.expansion = bench.index->non_zero_expansion_factor(config);
    [[maybe_unused]] fu_status_t const counted = fu_pool_threads_count(pool, &report.threads);

    std::vector<std::size_t> changed_per_pass;
    changed_per_pass.reserve(max_refines);
    progress_bar_t progress {"refining", max_refines};
    for (std::size_t pass = 0; pass != max_refines; ++pass) {
        auto [plan, plan_status] = bench.index->plan_refine({0, static_cast<node_id_t>(nodes)}, *bench.traversals[0],
                                                            config);
        if (failed(plan_status)) {
            fmt::println("Failed to open refinement pass {}: status {}", pass, int(plan_status));
            return false;
        }
        refinement_fanout<bench_t> fanout {bench, plan};
        auto started = clock_t::now();
        do {
            // The re-selection is the refining share of the wall time, the audit and the rounds after it the healing.
            bool const reselecting = plan.reselecting();
            fu_pool_for_n_dynamic(pool, plan.steps().size(), &refinement_callback<bench_t>, &fanout);
            (reselecting ? report.refining_seconds
                         : report.healing_seconds) += stdc::duration<double>(clock_t::now() - started).count();
            started = clock_t::now();
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
        report.reachable_nodes = pass_result.reachable_nodes;
        report.sweeps = pass_result.sweeps;
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

/** Serializes the index as three RetriEval-layout matrices - vectors, neighbors, and the id → row
 *  map - so later runs @c --load the base and stream only the topping. */
template <typename streamed_dataset_type_>
bool save_index(streamed_dataset_type_& streamed) {
    using clock_t = stdc::steady_clock;
    auto& bench = streamed.bench;
    auto started = clock_t::now();
    std::size_t const checkpoint_bytes = save_checkpoint(bench.options, *bench.index,
                                                         std::span {bench.results.id_to_row}, 0, 1);
    if (!checkpoint_bytes) return false;
    double const seconds = stdc::duration<double>(clock_t::now() - started).count();
    fmt::println("Saved the index: {:.2f} GB in {:.2f} s", double(checkpoint_bytes) / 1e9, seconds);
    return true;
}

/** Every query is one task: cast through the thread's staging area if needed - before the clock -
 *  then searched and translated through the ID → row map into the judged hits. */
template <typename streamed_dataset_type_>
void search_callback(fu_lambda_context_t punned, std::size_t query, std::size_t thread, std::size_t) noexcept {
    using bench_t = typename streamed_dataset_type_::bench_t;
    using index_t = typename bench_t::index_t;
    using small_size_t = typename bench_t::small_size_t;
    using vector_view_t = typename bench_t::vector_view_t;
    using clock_t = stdc::steady_clock;
    streamed_dataset_type_& streamed = *static_cast<streamed_dataset_type_*>(punned);
    auto& bench = streamed.bench;
    telemetry_lane_t& lane = bench.lanes[thread];

    typename index_t::search_config_t config;
    config.expansion_factor = static_cast<small_size_t>(bench.options.expansion_search);
    // The search phase runs strictly after ingestion, so the mutation-free contract holds.
    config.synchronization = synchronization_t::immutable_k;
    // Under self-recall the query file is the base itself: every stored vector must find its row.
    auto const source_row = bench.options.self_recall ? streamed.vectors.row(query) : streamed.queries.row(query);
    vector_view_t query_vector;
    if constexpr (streamed_dataset_type_::zero_copy_k) query_vector = source_row.as_vector();
    else query_vector = streamed.stage(source_row.as_vector(), thread, query);

    u32_t* const merged_rows = bench.results.hit_rows.row(query).data();
    auto* const merged_distances = bench.results.hit_distances.row(query).data();

    // The index stores no keys, so matches translate through the ID → row map on the way out.
    typename bench_t::traversal_t& traversal = *bench.traversals[thread];
    std::span<u32_t> const rows_span {merged_rows, bench.options.top_k};
    std::span<dense_distance_t> const distances_span {merged_distances, bench.options.top_k};
    auto started = clock_t::now();
    auto const [result, status] = bench.index->search(query_vector, traversal, config);
    if (failed(status)) {
        standard_atomic_ref<std::size_t>(bench.results.failed_tasks).fetch_add(1, std::memory_order_relaxed);
        return;
    }
    lane.visited_members += result.visited_members;
    lane.computed_distances += result.computed_distances;
    lane.restarts += result.restarts;
    lane.saturated_walks += result.saturated_walks;
    std::size_t const merged_count = merge_search_results(to_span(traversal.results), bench.results.id_to_row.data(),
                                                          rows_span, distances_span, std::size_t {0});
    bench.results.latencies_us[query] = stdc::duration<float, std::micro>(clock_t::now() - started).count();
    bench.results.hit_counts[query] = static_cast<u32_t>(merged_count);
}

template <typename streamed_dataset_type_>
bool run_search(streamed_dataset_type_& streamed, fu_pool_t pool) {
    using clock_t = stdc::steady_clock;
    auto& bench = streamed.bench;
    std::size_t const query_count = bench.options.self_recall ? streamed.vectors.extent(0) : streamed.queries.extent(0);
    for (telemetry_lane_t& lane : bench.lanes) lane = telemetry_lane_t {};

    auto started = clock_t::now();
    fu_pool_for_n(pool, query_count, &search_callback<streamed_dataset_type_>, &streamed);
    double const seconds = stdc::duration<double>(clock_t::now() - started).count();

    if (std::size_t failures = bench.results.failed_tasks; failures) {
        fmt::println("Search failed on {} queries", failures);
        return false;
    }
    telemetry_lane_t const sums = summed_lanes(bench.lanes);
    std::size_t searching_threads = 0;
    [[maybe_unused]] fu_status_t const counted = fu_pool_threads_count(pool, &searching_threads);
    report_search_metrics(bench.options, query_count, seconds, searching_threads, sums,
                          {bench.results.latencies_us.data(), bench.results.latencies_us.size()});
    judge_hits(bench.options, bench.results, query_count);
    return true;
}

/** The retrieval workload the cascade ends in: the timed search phase and its judgment, under whichever
 *  kernel the CLI picked. */
struct retrieval_workload_t {

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

/** Bundles the launch state the cascade threads through its compile-time hops. */
template <typename dataset_scalar_type_>
struct bench_launch {
    utilities::persisted_dataset<dataset_scalar_type_> const& dataset;
    bench_options_t const& options;
    fu_pool_t pool;
    fu_topology_t topology;
};

/** Gate: a (stored dtype, kernel) cell is runnable once its kit chain is adapted in
 *  @c usearch/dense/ram_metric.hpp; unadapted combinations report as skipped. */
template <nk_dtype_t stored_dtype_, nk_kernel_kind_t kernel_, nk_capability_t capability_>
concept ram_kit_adapted = requires {
    sizeof(dense_quads_chain<kernel_, typename nk::type_for<stored_dtype_>::type, capability_>);
};

/** No-op sibling picked when the cell's CPU kit chain is not adapted yet. */
template <typename dataset_scalar_type_, nk_dtype_t stored_dtype_, nk_kernel_kind_t kernel_,
          nk_capability_t capability_, typename wait_type_, typename workload_type_>
    requires(!ram_kit_adapted<stored_dtype_, kernel_, capability_>)
int stage_and_run(bench_launch<dataset_scalar_type_> const&, workload_type_ const&) {
    fmt::println("skipped: no CPU kit adapted for {} · {}", dtype_name(stored_dtype_), nk_kernel_name(kernel_));
    return 0;
}

/** The innermost hop: binds the concrete volatile-index instantiation, wires the typed source
 *  views and the bounded staging trampoline into a @c streamed_dataset, builds the index, and
 *  hands it to @p workload. */
template <typename dataset_scalar_type_, nk_dtype_t stored_dtype_, nk_kernel_kind_t kernel_,
          nk_capability_t capability_, typename wait_type_, typename workload_type_>
    requires(ram_kit_adapted<stored_dtype_, kernel_, capability_>)
int stage_and_run(bench_launch<dataset_scalar_type_> const& launch, workload_type_ const& workload) {
    using allocator_t = forkunion_allocator<std::byte>;
    using index_t = dense_ram_index<stored_dtype_, kernel_, capability_, wait_type_, fu::preferred_atomic_ref,
                                    allocator_t, allocator_t, allocator_t>;
    using bench_context_type_ = bench_context<index_t>;
    using streamed_t = streamed_dataset<dataset_scalar_type_, bench_context_type_>;
    using scalar_t = typename bench_context_type_::scalar_t;

    utilities::persisted_dataset<dataset_scalar_type_> const& dataset = launch.dataset;
    std::size_t const dimensions = dataset.dimensions();
    std::size_t const total_rows = dataset.vectors().extent(0);
    std::size_t const first_row = launch.options.first_vector;
    if (first_row >= total_rows) {
        fmt::println("--first-vector {} exceeds the {} dataset rows", first_row, total_rows);
        return 1;
    }
    std::size_t rows = total_rows - first_row;
    if (launch.options.max_vectors) rows = std::min(rows, launch.options.max_vectors);
    nk::matrix_view<dataset_scalar_type_> const dataset_vectors = dataset.vectors();
    nk::matrix_view<dataset_scalar_type_> const sliced_vectors = dataset_vectors(
        nk::range {first_row, first_row + rows}, nk::slice);

    // Self-queries keep the harness usable on a lone vectors file, without recall.
    nk::matrix_view<dataset_scalar_type_> const source_queries =
        dataset.has_queries()
            ? dataset.queries()
            : sliced_vectors(nk::range {std::size_t {0}, std::min<std::size_t>(1000, rows)}, nk::slice);

    bench_context_type_ bench {launch.options};
    streamed_t streamed {bench, sliced_vectors, source_queries};

    // Owns the staging trampoline planes for the whole run.
    nk::tensor<scalar_t, std::allocator<scalar_t>, 3> staging;
    // Owns the rescaling planes @c streamed_dataset::rescaling views.
    nk::tensor<nk::f32_t, std::allocator<nk::f32_t>, 3> rescaling_planes;
    if constexpr (!streamed_t::zero_copy_k) {
        std::size_t threads = 0;
        if (fu_pool_threads_count(launch.pool, &threads) != fu_success_k || !threads) {
            fmt::println("Failed to read the pool's thread count");
            return 1;
        }
        std::size_t const batch_rows = launch.options.batch ? std::max(launch.options.batch, threads) : 4 * threads;
        std::size_t const rows_per_thread = divide_round_up(batch_rows, threads);
        staging = nk::tensor<scalar_t, std::allocator<scalar_t>, 3>::try_zeros({threads, rows_per_thread, dimensions});
        if (staging.empty()) {
            fmt::println("Failed to allocate {} staging rows of {}d {}", batch_rows, dimensions,
                         scalar_t::dtype_name());
            return 1;
        }
        streamed.staging = staging.span();

        if constexpr (streamed_t::scaled_k) {
            rescaling_planes = nk::tensor<nk::f32_t, std::allocator<nk::f32_t>, 3>::try_zeros(
                {threads, rows_per_thread, dimensions});
            if (rescaling_planes.empty()) {
                fmt::println("Failed to allocate {} rescaling rows of {}d f32", batch_rows, dimensions);
                return 1;
            }
            streamed.rescaling = rescaling_planes.span();
            streamed.establish_scale();
        }
    }

    if (dataset.has_ground_truth() && !launch.options.max_vectors && !first_row)
        bench.results.ground_truth = dataset.neighbors();
    else if (dataset.has_ground_truth())
        fmt::println("Ground truth ignored: --max-vectors or --first-vector reshapes the graph it was computed for");

    if (!prepare(streamed, launch.pool)) return 1;
    if (!run_insertion(streamed, launch.pool)) return 1;
    if (!run_refinement(streamed, launch.pool)) return 1;
    if (!launch.options.save_name.empty() && !save_index(streamed)) return 1;
    return workload.run(streamed, launch.pool) ? 0 : 1;
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
            launch.options.kernel,
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
 *  floor. The same mask already reached @c fu_pool_new, so the pool waits the same way. */
template <typename dataset_scalar_type_, nk_dtype_t stored_dtype_, nk_capability_t capability_, typename workload_type_>
int run_with_schedule(bench_launch<dataset_scalar_type_> const& launch, workload_type_ const& workload) {
    return dispatch_index<std::size(schedule_choices_k)>(
        highest_index(schedule_choices_k, effective_schedule(launch.options.max_schedule)),
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
    nk_capability_t const effective = launch.options.max_compute & nk_capabilities_available();
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
    return run_with_compute<dataset_scalar_type_, stored_dtype_>(launch, retrieval_workload_t {});
}

#pragma endregion

#pragma region Process Setup

/** The CLI names resolved only once the machine is known: the two capability ceilings and the pinned
 *  memory domain. */
struct bench_names_t {
    std::string max_compute;
    std::string max_schedule;
    std::string memory_domain;
};

/** The flags every CPU single-index binary takes: the dataset, the stored dtype, the ceilings, the executor, the
 *  graph and its refinement, the judged depth, the view, and the checkpoints. */
inline clipp::group common_cli(bench_options_t& options, bench_names_t& names) {
    return ( //
        (clipp::option("--vectors") & clipp::value("path", options.vectors_path)) % "base vectors, `.fbin` family",
        (clipp::option("--queries") & clipp::value("path", options.queries_path)) % "query vectors, same dtype",
        (clipp::option("--f32").set(options.stored_dtype, nk_f32_k) |
         clipp::option("--bf16").set(options.stored_dtype, nk_bf16_k) |
         clipp::option("--f16").set(options.stored_dtype, nk_f16_k) |
         clipp::option("--e5m2").set(options.stored_dtype, nk_e5m2_k) |
         clipp::option("--e4m3").set(options.stored_dtype, nk_e4m3_k) |
         clipp::option("--e3m2").set(options.stored_dtype, nk_e3m2_k) |
         clipp::option("--e2m3").set(options.stored_dtype, nk_e2m3_k) |
         clipp::option("--i8").set(options.stored_dtype, nk_i8_k) |
         clipp::option("--i4").set(options.stored_dtype, nk_i4_k) |
         clipp::option("--u8").set(options.stored_dtype, nk_u8_k) |
         clipp::option("--u4").set(options.stored_dtype, nk_u4_k) |
         clipp::option("--u1").set(options.stored_dtype, nk_u1_k)) %
            "stored dtype override, staging `nk::cast` on ingest",
        (clipp::option("--max-compute") & clipp::value("name", names.max_compute)) %
            "compute ceiling by microarchitecture, like `icelake`, or `serial`; newest detected by default",
        (clipp::option("--max-schedule") & clipp::value("name", names.max_schedule)) %
            "scheduling ceiling by wait instruction, like `tpause`, or `standard`; newest offered by default",
        (clipp::option("--threads") & clipp::value("count", options.threads)) %
            "threads; 7/8 of the logical cores by default",
        (clipp::option("--memory-domain") & clipp::value("id", names.memory_domain)) %
            "pin memory & threads to one NUMA domain; the heap otherwise",
        (clipp::option("--batch") & clipp::value("rows", options.batch)) %
            "staging trampoline rows shared by all threads; 4 × threads by default",
        (clipp::option("--connectivity") & clipp::value("degree", options.connectivity)) % "16, 32, or 128",
        (clipp::option("--expansion-add") & clipp::value("breadth", options.expansion_add)) %
            "insertion candidate pool; 256 by default",
        (clipp::option("--expansion-search") & clipp::value("breadth", options.expansion_search)) %
            "search candidate pool; 96 by default",
        (clipp::option("--max-refines") & clipp::value("passes", options.max_refines)) %
            "upper bound on whole-graph refinement passes after ingestion; 0 by default",
        (clipp::option("--expansion-refine") & clipp::value("breadth", options.expansion_refine)) %
            "refinement candidate pool; twice the connectivity by default",
        (clipp::option("--top-k") & clipp::value("count", options.top_k)) % "matches per query to judge",
        (clipp::option("--max-vectors") & clipp::value("count", options.max_vectors)) % "cap on ingested rows",
        (clipp::option("--first-vector") & clipp::value("row", options.first_vector)) %
            "first dataset row to view; slices one file across independent runs",
        clipp::option("--self-recall").set(options.self_recall, true) % //
            "query every stored vector for itself instead of the query file",
        (clipp::option("--save") & clipp::value("name", options.save_name)) % //
            "checkpoint name to serialize every domain index into after ingestion",
        (clipp::option("--load") & clipp::value("name", options.load_name)) % //
            "checkpoint name to restore domain indexes from instead of building",
        (clipp::option("--checkpoints-dir") & clipp::value("dir", options.checkpoints_dir)) % //
            "directory for checkpoint blobs; /tmp by default",
        (clipp::option("--skip-vectors") & clipp::value("count", options.skip_vectors)) % //
            "rows the loaded indexes already cover",
        (clipp::option("--huge-pages") & clipp::value("count", options.huge_pages)) %
            "huge pages provisioned before launch; enforced, so base-page fallbacks fail fast");
}

/** The process every CPU single-index binary runs around its workload: parses @p cli into @p options and
 *  @p names, resolves them against the machine, spawns the pool titled @p binary_name, prints the machine,
 *  then opens the dataset in its own scalar and exits with what @p on_launch makes of its @ref bench_launch. */
template <typename on_launch_type_>
int bench_main(int argc, char** argv, char const* binary_name, clipp::group const& cli, bench_options_t& options,
               bench_names_t const& names, on_launch_type_&& on_launch) {
    if (!clipp::parse(argc, argv, cli) || options.help || options.vectors_path.empty()) {
        for (auto const& section : clipp::make_man_page(cli, binary_name))
            fmt::println("{}\n{}\n", section.title(), section.content());
        return options.help ? 0 : 1;
    }

    nk_dtype_t const file_dtype = utilities::dtype_of_extension(options.vectors_path.c_str());
    if (file_dtype == nk_dtype_unknown_k || file_dtype == nk_u32_k) {
        fmt::println("Unrecognized vectors extension: {}", options.vectors_path);
        return 1;
    }
    if (options.stored_dtype == nk_dtype_unknown_k) options.stored_dtype = file_dtype;

    // Ceiling names resolve into allow-masks - the native capability currency on both
    // sides. A named compute ceiling keeps its table prefix plus serial; "serial" alone
    // keeps just the fallback bit. Capping above the detection is harmless: the dispatch
    // intersects with the runtime bits anyway.
    if (names.max_compute == "serial") { options.max_compute = nk_cap_serial_k; }
    else if (!names.max_compute.empty()) {
        std::size_t const named = index_of_name(compute_choices_k, names.max_compute);
        if (named == std::size(compute_choices_k)) {
            fmt::println("Unknown --max-compute {}; this table offers: {}", names.max_compute,
                         fmt::join(named_choices("serial", compute_choices_k, //
                                                 prefix_mask(compute_choices_k, std::size(compute_choices_k))),
                                   " "));
            return 1;
        }
        options.max_compute = nk_cap_serial_k | prefix_mask(compute_choices_k, named);
    }

    // A schedule ceiling caps only the waiter bits; the pool-management bits - topology,
    // placement, huge pages - stay allowed, so `fu_pool_new` keeps its full toolbox. The
    // floor keyword is "standard", after `fu::standard_yield_t` - Arm's own YIELD hint
    // legitimately owns the "yield" row.
    if (names.max_schedule == "standard") { options.max_schedule = schedule_floor_k; }
    else if (!names.max_schedule.empty()) {
        std::size_t const named = index_of_name(schedule_choices_k, names.max_schedule);
        if (named == std::size(schedule_choices_k)) {
            fmt::println("Unknown --max-schedule {}; this table offers: {}", names.max_schedule,
                         fmt::join(named_choices("standard", schedule_choices_k, //
                                                 prefix_mask(schedule_choices_k, std::size(schedule_choices_k))),
                                   " "));
            return 1;
        }
        options.max_schedule = static_cast<fu_capabilities_t>(schedule_floor_k |
                                                              prefix_mask(schedule_choices_k, named));
    }

    fu_topology_t topology = nullptr;
    fu_fabric_t fabric = nullptr;
    fu_pool_t pool = nullptr;
    if (fu_topology_new(&topology) != fu_success_k || fu_fabric_new(&fabric) != fu_success_k ||
        fu_pool_new(binary_name, options.max_schedule, &pool) != fu_success_k) {
        fmt::println("Failed to inspect the topology or create the pool");
        return 1;
    }

    std::size_t memory_domains = 0;
    if (fu_memory_domains_count(topology, &memory_domains) != fu_success_k) {
        fmt::println("Failed to count the memory domains");
        return 1;
    }

    // Resolve `--memory-domain`: empty means one machine-wide heap index, and a single ID
    // pins memory & threads to that domain. This benchmark measures exactly one index - the
    // multi-domain arrangement is `usearch_bench_dense_ram_indexes`, driving the sharded index.
    if (!names.memory_domain.empty()) {
        fu_memory_domain_id_t const domain = static_cast<fu_memory_domain_id_t>(
            std::strtol(names.memory_domain.c_str(), nullptr, 10));
        if (domain < 0 || static_cast<std::size_t>(domain) >= memory_domains) {
            fmt::println("Unknown --memory-domain {}; this machine has {} memory domains", names.memory_domain,
                         memory_domains);
            return 1;
        }
        options.memory_domains.push_back(domain);
    }

    // Default to 7/8 of the available logical cores: at full occupancy the strictly-ordered
    // publication convoy serializes on scheduler latency, and throughput collapses by an order
    // of magnitude. A single pinned domain carves the core budget from every compute domain
    // whose local memory it is - several clusters can share one NUMA node.
    bool spawned = false;
    std::size_t threads = options.threads;
    if (options.memory_domains.size() == 1) {
        std::size_t compute_domains = 0;
        if (fu_compute_domains_count(topology, &compute_domains) != fu_success_k) {
            fmt::println("Failed to count the compute domains");
            return 1;
        }
        if (!threads) {
            std::size_t local_cores = 0;
            for (std::size_t compute_domain = 0; compute_domain != compute_domains; ++compute_domain) {
                std::size_t local_memory = 0, domain_cores = 0;
                if (fu_local_memory_of(topology, compute_domain, &local_memory) != fu_success_k) continue;
                if (local_memory != static_cast<std::size_t>(options.memory_domains.front())) continue;
                if (fu_logical_cores_count_in(topology, compute_domain, &domain_cores) != fu_success_k) continue;
                local_cores += domain_cores;
            }
            threads = std::max<std::size_t>(1, local_cores * 7 / 8);
        }
        spawned = fu_pool_spawn_near_memory_domain(topology, pool, options.memory_domains.front(), threads,
                                                   fu_caller_inclusive_k) == fu_success_k;
    }
    else {
        if (!threads) {
            std::size_t logical_cores = 0;
            if (fu_logical_cores_count(topology, &logical_cores) != fu_success_k) {
                fmt::println("Failed to count the logical cores");
                return 1;
            }
            threads = std::max<std::size_t>(1, logical_cores * 7 / 8);
        }
        spawned = fu_pool_spawn(topology, pool, threads, fu_caller_inclusive_k) == fu_success_k;
    }
    if (!spawned) {
        fmt::println("Failed to spawn {} threads", threads);
        return 1;
    }
    print_machine(topology, fabric, options, threads);

    // Huge pages are provisioned outside the process - see the header's shell snippet - and
    // `--huge-pages` declares the expectation, enforced against what the domains report.
    if (options.huge_pages) {
        std::size_t volume = 0;
        for (std::size_t domain = 0; domain != memory_domains; ++domain) {
            std::size_t domain_volume = 0;
            if (fu_volume_huge_pages_in(topology, domain, &domain_volume) != fu_success_k) continue;
            volume += domain_volume;
        }
        fmt::println("  huge pages expected: {}, available: {:.2f} GB", options.huge_pages, double(volume) / 1e9);
        if (!volume) {
            fmt::println("--huge-pages expects {}, but the domains report none", options.huge_pages);
            return 1;
        }
    }

    // File-scalar axis: the newest hop of the compile-time cascade, lowering the file's dtype
    // to the C++ wrapper the whole run is instantiated over through NumKong's own `type_for`.
    int const exit_code = dispatch_file_dtype(
        file_dtype,
        [&](auto dtype) {
            using dataset_scalar_t = typename nk::type_for<dtype.value>::type;
            auto opened = utilities::persisted_dataset<dataset_scalar_t>::open(
                options.vectors_path, options.queries_path, options.neighbors_path);
            if (opened.status != status_t::success_k) {
                fmt::println("Failed to open the dataset: {}", options.vectors_path);
                return 1;
            }
            return on_launch(bench_launch<dataset_scalar_t> {opened.value, options, pool, topology});
        },
        [&](auto dtype) {
            fmt::println("Unsupported dataset dtype {}", dtype_name(dtype));
            return 1;
        });

    fu_pool_delete(pool);
    fu_fabric_delete(fabric);
    fu_topology_delete(topology);
    return exit_code;
}

#pragma endregion

} // namespace unum::usearch::bench::dense
