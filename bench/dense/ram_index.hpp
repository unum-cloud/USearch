/**
 *  @brief Profiling harness for the CPU-resident @c dense_ram_index over real persisted
 *      datasets, shared by the per-dtype benchmark shards.
 *
 *  @author Ash Vardanian
 *  @file bench/dense/ram_index.hpp
 *  @date July 17, 2026
 *
 *  The backend-neutral surface - options, checkpoints, dataset streaming, telemetry, result
 *  judgment, and the machine report - lives in @c dense.cuh; this header owns
 *  everything CPU: the ForkUnion run loops streaming one row or query per task, the
 *  NUMA-domain placement, the waiter dispatch, and the compile-time cascade resolving
 *  the CLI into one concrete @c dense_ram_index instantiation - sharded per stored dtype,
 *  so parallel translation units split the instantiation grid. A ready-made playground
 *  is the Unum Wiki 1M dataset:
 *
 *  @code{.sh}
 *  mkdir -p ~/datasets/wiki_1M && cd ~/datasets/wiki_1M &&
 *      wget -nc https://huggingface.co/datasets/unum-cloud/ann-wiki-1m/resolve/main/base.1M.fbin &&
 *      wget -nc https://huggingface.co/datasets/unum-cloud/ann-wiki-1m/resolve/main/query.public.100K.fbin &&
 *      wget -nc https://huggingface.co/datasets/unum-cloud/ann-wiki-1m/resolve/main/groundtruth.public.100K.ibin
 *  usearch_bench_dense_ram_index --vectors ~/datasets/wiki_1M/base.1M.fbin \
 *      --queries ~/datasets/wiki_1M/query.public.100K.fbin \
 *      --neighbors ~/datasets/wiki_1M/groundtruth.public.100K.ibin --bf16
 *  @endcode
 *
 *  Huge pages are provisioned outside the process, before it starts; `--huge-pages` then
 *  declares the expectation, which the harness logs and enforces - a mismatch fails the
 *  run instead of silently benchmarking base pages. The effect is anything but cosmetic:
 *  random vector gathers over a multi-GB pool miss the STLB on nearly every access under
 *  4 KB pages, stacking a page walk onto each already-cold load, and 2 MB pages have
 *  measured ~30% higher insertion throughput and ~20% higher search QPS on SIFT-10M on a
 *  dual-socket Sapphire Rapids box - the largest single-switch effect this harness knows:
 *
 *  @code{.sh}
 *  sudo sysctl vm.nr_hugepages=16384        # 32 GB of 2 MB pages, split across the domains
 *  usearch_bench_dense_ram_index --huge-pages 16384 \
 *      --vectors ~/datasets/sift_10M/base.10M.u8bin --memory-domain 0
 *  @endcode
 *
 *  This binary holds exactly one index; plurality belongs to the operating system or to the
 *  sharded sibling. The zero-cross-traffic upper bound - N independent builds, one per memory
 *  domain, each pinned to every compute domain local to it and ingesting its own slice of one
 *  dataset - is an invocation rather than a code path:
 *
 *  @code{.sh}
 *  for domain in 0 1; do
 *      usearch_bench_dense_ram_index --memory-domain $domain --vectors base.10M.bf16bin \
 *          --first-vector $((domain * 5000000)) --max-vectors 5000000 --self-recall &
 *  done; wait
 *  @endcode
 *
 *  Defaults are inferred and each is overridable:
 *  - the stored dtype matches the dataset file, unless `--f32`, `--bf16`, `--f16`, `--i8`, `--i4`,
 *    `--u8`, or `--u1` asks for a streaming `nk::cast` through a `--batch`-row trampoline on ingest;
 *  - the newest runtime compute capability drives the distance kernels, unless
 *    `--max-compute` caps it by NumKong's microarchitecture names, like `icelake`;
 *  - the strongest offered wait instruction parks stalled threads, unless `--max-schedule`
 *    caps it by ForkUnion's instruction names, like `tpause`, down to the `standard` yield;
 *  - 7/8 of the logical cores participate, unless `--threads` overrides - full occupancy
 *    starves the ordered publication path and collapses throughput.
 */
#pragma once

#include <cmath> // `std::abs`

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
 *  @brief The CPU-side state one fully-typed benchmark run shares between its phases: the one
 *      index, its per-thread traversal contexts, and the embedded backend-neutral judgment
 *      results.
 *
 *  The dataset never appears here - the data side lives in @c streamed_dataset, which
 *  bridges the file's scalar into the stored one vector-by-vector. One index, on one memory
 *  domain or the plain heap - the multi-domain arrangement is a different benchmark category
 *  with a different binary, driving @c dense_ram_indexes instead.
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

    bench_options_t const& options;
    /** @brief The one index under measurement; the indirection buys address stability, as an
     *      index holds atomics and never moves. */
    std::unique_ptr<index_t> index;
    /** @brief One graph-walking context per pool thread, built by that thread itself with
     *      its local memory domain's allocator. */
    std::vector<std::unique_ptr<traversal_t>> traversals;
    /** @brief Per-thread telemetry sums, padded apart to stay contention-free. */
    std::vector<telemetry_lane_t> lanes;
    /** @brief The backend-neutral judgment state both phases fill. */
    bench_results_t results;

    bench_context(bench_options_t const& options) noexcept : options(options) {}
};

/** @brief Builds the index on the chosen memory domain, reserves it for the viewed rows, and
 *      sizes all per-thread state off the pool's actual thread count - everything fallible,
 *      before any clock starts. */
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

    // Placement: the chosen memory domain, or the plain heap. The index is reserved for the
    // full row count: capacity is a counter and chunk storage allocates lazily, so the slack
    // costs directory slots, not memory.
    fu_memory_domain_id_t const domain = bench.options.memory_domains.empty() //
                                             ? fu_memory_domain_id_t {-1}
                                             : bench.options.memory_domains.front();
    bench.index = std::make_unique<index_t>(allocator_t {domain}, allocator_t {domain});
    bench.results.id_to_row.resize(rows);

    // Reserving first freezes the CLI geometry and pre-grows capacity; a checkpoint then
    // loads against that frozen geometry, so a header mismatch fails loudly below instead
    // of restoring an incompatible graph.
    auto reserved = bench.index->reserve({rows, connectivity, dimensions});
    if (reserved.status != status_t::success_k) {
        fmt::println("Failed to reserve {} × {}d", rows, dimensions);
        return false;
    }

    // Three RetriEval-layout matrix dumps: the file name's dtype-kernel-connectivity stamp
    // carries compatibility across backends, the two-counter headers carry the shapes, and
    // any dataset tool opens them. The id → row map restores the recall bookkeeping the
    // index itself never stores.
    if (!bench.options.load_name.empty() &&
        !load_checkpoint(bench.options, *bench.index, dimensions, std::span {bench.results.id_to_row}, 0, 1))
        return false;

    // The loaded base must cover exactly the skipped prefix - anything else would judge
    // recall against rows the graph never saw.
    std::size_t const loaded_rows = bench.index->size();
    if (loaded_rows != bench.options.skip_vectors) {
        fmt::println("Loaded index covers {} rows, but --skip-vectors says {}", loaded_rows,
                     bench.options.skip_vectors);
        return false;
    }

    // Every pool thread constructs its own traversal context, so the scratch - results,
    // frontier, visited set - lands on the thread's local memory domain, not the caller's.
    bench.traversals.resize(threads);
    auto traversals_init = [](fu_lambda_context_t punned, std::size_t thread, std::size_t) noexcept {
        auto& bench = *static_cast<bench_t*>(punned);
        // AMX tile state needs a per-thread OS permission handshake before first use;
        // NumKong's thread configurator performs it and no-ops everywhere else.
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

/** @brief One vector per task, streamed: viewed in place when the scalars match, or cast
 *      through the thread's staging area, then inserted into the one index under measurement. */
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
    auto inserted = bench.index->insert(one_row, {&local_id, std::size_t {1}}, *bench.traversals[thread], config);
    if (!inserted) {
        bench.results.failed_tasks.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    telemetry_lane_t& lane = bench.lanes[thread];
    lane.visited_members += inserted.visited_members;
    lane.computed_distances += inserted.computed_distances;
    lane.computed_distances_in_refines += inserted.computed_distances_in_refines;
    lane.computed_distances_in_reverse_refines += inserted.computed_distances_in_reverse_refines;
    lane.restarts += inserted.restarts;
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

    if (std::size_t failures = bench.results.failed_tasks.load(); failures) {
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
                 bit_name(compute_span_k, index_t::picked_quads_capability_k, "serial"));
    if constexpr (index_t::has_tiles_k)
        fmt::println("  tiles: {} · reverse refinement",
                     bit_name(compute_span_k, index_t::picked_tiles_capability_k, "none"));
    if (index_t::picked_quads_capability_k == nk_cap_serial_k &&
        (bench.options.max_compute & nk_capabilities_available() & ~nk_cap_serial_k))
        fmt::print(verdict_style(false), "  ⚠ scalar fallback: no SIMD kit is adapted for this dtype & metric\n");
    std::size_t reported_threads = 0;
    [[maybe_unused]] fu_status_t const counted = fu_pool_threads_count(pool, &reported_threads);
    fmt::println("  threads: {}", reported_threads);
    fmt::println("  wall: {:.2f} s", seconds);
    fmt::println("  throughput: {} IPS", humanize(double(inserted_rows) / seconds));
    fmt::println("  distances: {} ({:.1f}% refines, {:.1f}% reverse)", humanize(double(sums.computed_distances)),
                 refine_share, reverse_share);
    fmt::println("  visited: {}", humanize(double(sums.visited_members)));
    fmt::print("  restarts: ");
    fmt::print(verdict_style(sums.restarts < inserted_rows / 100 + 1), "{}\n", sums.restarts);
    if (std::size_t resident = peak_resident_bytes(); resident)
        fmt::println("  peak resident: {:.2f} GB", double(resident) / 1e9);
    return true;
}

/** @brief Serializes the index as three RetriEval-layout matrices - vectors, neighbors, and
 *      the id → row map - so later runs @c --load the base and stream only the topping: a
 *      100M experiment then costs an mmap, not a half-hour rebuild. */
template <typename streamed_dataset_type_>
bool save_index(streamed_dataset_type_& streamed) {
    using clock_t = stdc::steady_clock;
    auto& bench = streamed.bench;
    auto started = clock_t::now();
    std::size_t const saved_bytes = save_checkpoint(bench.options, *bench.index, std::span {bench.results.id_to_row}, 0,
                                                    1);
    if (!saved_bytes) return false;
    double const seconds = stdc::duration<double>(clock_t::now() - started).count();
    fmt::println("Saved the index: {:.2f} GB in {:.2f} s", double(saved_bytes) / 1e9, seconds);
    return true;
}

/** @brief Every query is one task: cast through the thread's staging area if needed - before
 *      the clock - then searched and translated through the id → row map into the judged
 *      hits. The clock covers the whole lookup - the honest end-to-end latency. */
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
    // The search phase runs strictly after ingestion, so the mutation-free contract holds by
    // construction - certify it and measure the atomics-free traversal.
    config.synchronization = synchronization_t::immutable_k;
    // Under self-recall the "query file" is the base itself: every stored vector must find
    // its own row - theorem-grade navigability probing with zero ground-truth files.
    auto const source_row = bench.options.self_recall ? streamed.vectors.row(query) : streamed.queries.row(query);
    vector_view_t query_vector;
    if constexpr (streamed_dataset_type_::zero_copy_k) query_vector = source_row.as_vector();
    else query_vector = streamed.stage(source_row.as_vector(), thread, query);

    u32_t* const merged_rows = bench.results.hit_rows.row(query).data();
    auto* const merged_distances = bench.results.hit_distances.row(query).data();

    // The dense ids of the matches translate through the id → row map on the way out, since
    // the index stores no keys and recall is judged in dataset rows.
    typename bench_t::traversal_t& traversal = *bench.traversals[thread];
    std::span<u32_t> const rows_span {merged_rows, bench.options.top_k};
    std::span<dense_distance_t> const distances_span {merged_distances, bench.options.top_k};
    auto started = clock_t::now();
    auto const found = bench.index->search(query_vector, traversal, config);
    if (!found) {
        bench.results.failed_tasks.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    lane.visited_members += found.visited_members;
    lane.computed_distances += found.computed_distances;
    lane.restarts += found.restarts;
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

    if (std::size_t failures = bench.results.failed_tasks.load(); failures) {
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

#pragma endregion

#pragma region Dispatch Cascade

/** @brief Bundles the launch state the cascade threads through its compile-time hops. */
template <typename dataset_scalar_type_>
struct bench_launch {
    persisted_dataset<dataset_scalar_type_> const& dataset;
    bench_options_t const& options;
    fu_pool_t pool;
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
 *  @brief The innermost hop: binds the concrete volatile-index instantiation, wires the
 *      typed source views and the bounded staging trampoline into a @c streamed_dataset,
 *      and runs both phases.
 *
 *  When the file scalar already matches the stored one, everything is a zero-copy view
 *  into the mappings; otherwise the staging matrix holds just `--batch` rows - typically a
 *  few per thread - and each vector streams through it on its way into the index. The
 *  dataset is never copied whole.
 */
template <typename dataset_scalar_type_, nk_dtype_t stored_dtype_, nk_kernel_kind_t kernel_,
          nk_capability_t capability_, typename wait_type_>
    requires(ram_kit_adapted<stored_dtype_, kernel_, capability_>)
int stage_and_run(bench_launch<dataset_scalar_type_> const& launch) {
    using allocator_t = forkunion_allocator<std::byte>;
    using index_t =
        dense_ram_index<stored_dtype_, kernel_, capability_, wait_type_, allocator_t, allocator_t, allocator_t>;
    using bench_context_type_ = bench_context<index_t>;
    using streamed_t = streamed_dataset<dataset_scalar_type_, bench_context_type_>;
    using scalar_t = typename bench_context_type_::scalar_t;

    persisted_dataset<dataset_scalar_type_> const& dataset = launch.dataset;
    std::size_t const dimensions = dataset.dimensions();
    std::size_t const total_rows = dataset.vectors().extent(0);
    std::size_t const first_row = launch.options.first_vector;
    if (first_row >= total_rows) {
        fmt::println("--first-vector {} exceeds the {} dataset rows", first_row, total_rows);
        return 1;
    }
    std::size_t rows = total_rows - first_row;
    if (launch.options.max_vectors) rows = std::min(rows, launch.options.max_vectors);
    nk::matrix_view<dataset_scalar_type_> const sliced_vectors = dataset.vectors()(
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
    // Owns the rescaling planes `streamed_dataset::rescaling` views.
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
    if (!launch.options.save_name.empty() && !save_index(streamed)) return 1;
    if (!run_search(streamed, launch.pool)) return 1;
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
 *      this machine actually offers, which is what gates Arm's runtime-only `WFET` - picks the
 *      waiter from @ref schedule_waiters_t; the same mask already reached `fu_pool_new`, so the
 *      pool waits the same way. Past the table's tail sits the STL yield floor. */
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
 *      table's tail sits the one ceiling every machine owns: the serial kernels - serving an
 *      explicit `--max-compute serial` and any CPU with no SIMD bits. */
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
