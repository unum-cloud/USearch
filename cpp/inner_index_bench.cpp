/**
 *  @brief Profiling harness for the CPU-resident volatile index over real persisted datasets.
 *
 *  @author Ash Vardanian
 *  @file cpp/inner_index_bench.cpp
 *  @date July 17, 2026
 *
 *  The backend-neutral surface - options, checkpoints, dataset streaming, telemetry, result
 *  judgment, and the machine report - lives in @c inner_index_bench.cuh; this translation
 *  unit owns everything CPU: the ForkUnion run loops streaming one row or query per task,
 *  the NUMA-domain placement, the waiter dispatch, and the compile-time cascade resolving
 *  the CLI into one concrete @c inner_volatile_index instantiation. A ready-made playground
 *  is the Unum Wiki 1M dataset:
 *
 *  @code{.sh}
 *  mkdir -p ~/datasets/wiki_1M && cd ~/datasets/wiki_1M &&
 *      wget -nc https://huggingface.co/datasets/unum-cloud/ann-wiki-1m/resolve/main/base.1M.fbin &&
 *      wget -nc https://huggingface.co/datasets/unum-cloud/ann-wiki-1m/resolve/main/query.public.100K.fbin &&
 *      wget -nc https://huggingface.co/datasets/unum-cloud/ann-wiki-1m/resolve/main/groundtruth.public.100K.ibin
 *  inner_index_bench_cpp --vectors ~/datasets/wiki_1M/base.1M.fbin \
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
 *  inner_index_bench_cpp --huge-pages 16384 \
 *      --vectors ~/datasets/sift_10M/base.10M.u8bin --memory-domain all
 *  @endcode
 *
 *  Defaults are inferred and each is overridable:
 *  - the stored dtype matches the dataset file, unless `--bf16`, `--i8`, `--u8`, or `--i4`
 *    asks for a streaming `nk::cast` through a `--batch`-row trampoline on ingest;
 *  - the newest runtime compute capability drives the distance kernels, unless
 *    `--max-compute` caps it by NumKong's microarchitecture names, like `icelake`;
 *  - the strongest offered wait instruction parks stalled threads, unless `--max-schedule`
 *    caps it by ForkUnion's instruction names, like `tpause`, down to the `standard` yield;
 *  - 7/8 of the logical cores participate, unless `--threads` overrides - full occupancy
 *    starves the ordered publication path and collapses throughput.
 */
#include <algorithm> // `std::max`, `std::min`, `std::copy_n`
#include <chrono>    // `std::chrono::steady_clock`
#include <cmath>     // `std::abs`
#include <cstdint>   // `std::uint32_t`
#include <memory>    // `std::unique_ptr`
#include <string>    // `std::string`
#include <tuple>     // `std::tuple`, `std::tuple_element_t` for the schedule waiter list
#include <vector>    // `std::vector`

#include <clipp.h>
#include <fmt/format.h>

#include <usearch/inner_volatile_index.hpp>

#include "inner_index_bench.cuh"
#include "persisted_dataset.hpp"

using namespace unum::usearch;

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
 *  @brief Allocates on a ForkUnion memory domain when one is chosen at runtime, and on the
 *      plain heap otherwise - one type either way, so the index instantiation grid doesn't
 *      double and a single binary serves both placements.
 */
template <typename value_type_>
struct domain_or_default_allocator {
    using value_type = value_type_;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_copy_assignment = std::true_type;
    using propagate_on_container_swap = std::true_type;
    using is_always_equal = std::false_type;

    /** @brief The pinned ForkUnion memory domain; negative means "plain heap". */
    fu_memory_domain_id_t memory_domain = -1;

    domain_or_default_allocator() noexcept = default;
    explicit domain_or_default_allocator(fu_memory_domain_id_t domain) noexcept : memory_domain(domain) {}
    template <typename other_type_>
    domain_or_default_allocator(domain_or_default_allocator<other_type_> const& other) noexcept
        : memory_domain(other.memory_domain) {}

    value_type_* allocate(std::size_t count) noexcept {
        if (memory_domain < 0) return static_cast<value_type_*>(std::malloc(count * sizeof(value_type_)));
        return static_cast<value_type_*>(fu_allocate_on_domain_id(memory_domain, count * sizeof(value_type_)));
    }
    void deallocate(value_type_* pointer, std::size_t count) noexcept {
        if (memory_domain < 0) return std::free(pointer);
        fu_free_on_domain_id(memory_domain, pointer, count * sizeof(value_type_));
    }
    bool operator==(domain_or_default_allocator const& other) const noexcept {
        return memory_domain == other.memory_domain;
    }
};

/**
 *  @brief The CPU-side state one fully-typed benchmark run shares between its phases: the
 *      indexes - one per engaged memory domain - their per-thread traversal contexts, and
 *      the embedded backend-neutral judgment results.
 *
 *  The dataset never appears here - the data side lives in @c streamed_dataset, which
 *  bridges the file's scalar into the stored one vector-by-vector. In the default and
 *  pinned placements exactly one index exists; under `--memory-domain all` every memory
 *  domain owns one, every prong routes to its local index through
 *  @c index_of_compute_domain, and searches gather through @c merge_into.
 */
template <typename index_type_>
struct bench_context {
    using allocator_t = domain_or_default_allocator<std::byte>;
    using index_t = index_type_;
    using scalar_t = typename index_t::scalar_t;
    using small_size_t = typename index_t::small_size_t;
    using traversal_t = typename index_t::inner_traversal_context_t;
    using vector_view_t = typename index_t::vector_view_t;
    using vectors_view_t = typename index_t::vectors_view_t;
    using distance_t = typename index_t::distance_t;

    static constexpr nk_kernel_kind_t kernel_kind_k = index_t::kernel_kind_k;

    bench_options_t const& options;
    /** @brief One index per engaged memory domain; the indirection buys address stability,
     *      as an index holds atomics and never moves. */
    std::vector<std::unique_ptr<index_t>> indexes;
    /** @brief Routes every prong to the index local to its compute domain. */
    std::vector<std::size_t> index_of_compute_domain;
    /** @brief One graph-walking context per pool thread, built by that thread itself with
     *      its local memory domain's allocator. */
    std::vector<std::unique_ptr<traversal_t>> traversals;
    /** @brief Per-thread telemetry sums, padded apart to stay contention-free. */
    std::vector<telemetry_lane_t> lanes;
    /** @brief The backend-neutral judgment state both phases fill. */
    bench_results_t results;

    bench_context(bench_options_t const& options) noexcept : options(options) {}
};

/** @brief Builds one index per engaged memory domain, reserves each for the viewed rows, and
 *      sizes all per-thread state off the pool's actual thread count - everything fallible,
 *      before any clock starts. */
template <typename streamed_dataset_type_>
bool prepare(streamed_dataset_type_& streamed, fu_pool_t pool, fu_topology_t topology) {
    using bench_t = typename streamed_dataset_type_::bench_t;
    using index_t = typename bench_t::index_t;
    using allocator_t = typename bench_t::allocator_t;
    using traversal_t = typename bench_t::traversal_t;
    auto& bench = streamed.bench;
    std::size_t const rows = streamed.vectors.extent(0);
    std::size_t const dimensions = streamed.vectors.extent(1);
    std::size_t const threads = fu_pool_threads_count(pool);
    std::size_t const compute_domains = fu_compute_domains_count(topology);

    connectivity_t const connectivity = connectivity_of(bench.options.connectivity);
    if (connectivity == connectivity_unknown_k) {
        fmt::print("Unsupported --connectivity {}; pick 16, 32, or 128\n", bench.options.connectivity);
        return false;
    }

    // Placement: one index per memory domain under `--memory-domain all`, one otherwise -
    // pinned to the chosen domain or on the plain heap. Every index is reserved for the full
    // row count: capacity is a counter and chunk storage allocates lazily, so the slack costs
    // directory slots, not memory.
    std::vector<fu_memory_domain_id_t> const& memory_domains = bench.options.memory_domains;
    std::size_t const indexes_count = std::max<std::size_t>(1, memory_domains.size());
    bench.index_of_compute_domain.resize(compute_domains, 0);
    if (memory_domains.size() > 1)
        for (std::size_t compute_domain = 0; compute_domain != compute_domains; ++compute_domain) {
            std::size_t const local_memory = fu_local_memory_of(topology, compute_domain);
            for (std::size_t slot = 0; slot != memory_domains.size(); ++slot)
                if (static_cast<std::size_t>(memory_domains[slot]) == local_memory)
                    bench.index_of_compute_domain[compute_domain] = slot;
        }

    if (bench.options.skip_vectors > rows) {
        fmt::print("--skip-vectors {} exceeds the {} viewed rows\n", bench.options.skip_vectors, rows);
        return false;
    }
    bench.indexes.reserve(indexes_count);
    bench.results.id_to_row.resize(indexes_count);
    for (std::size_t slot = 0; slot != indexes_count; ++slot) {
        fu_memory_domain_id_t const domain = memory_domains.empty() ? -1 : memory_domains[slot];
        bench.indexes.push_back(std::make_unique<index_t>(allocator_t {domain}, allocator_t {domain}));
        index_t& index = *bench.indexes.back();
        bench.results.id_to_row[slot].resize(rows);

        // Reserving first freezes the CLI geometry and pre-grows capacity; a checkpoint then
        // loads against that frozen geometry, so a header mismatch fails loudly below instead
        // of restoring an incompatible graph.
        auto reserved = index.reserve({rows, connectivity, dimensions});
        if (reserved.status != status_t::success_k) {
            fmt::print("Failed to reserve {} × {}d\n", rows, dimensions);
            return false;
        }

        // Three RetriEval-layout matrix dumps per slot: the file name's dtype-kernel-
        // connectivity stamp carries compatibility across backends, the two-counter headers
        // carry the shapes, and any dataset tool opens them. The id → row map restores the
        // recall bookkeeping the index itself never stores.
        if (!bench.options.load_name.empty() &&
            !load_checkpoint(bench.options, index, dimensions, std::span {bench.results.id_to_row[slot]}, slot,
                             indexes_count))
            return false;
    }

    // The loaded base must cover exactly the skipped prefix - anything else would judge
    // recall against rows the graph never saw.
    std::size_t loaded_rows = 0;
    for (auto const& index : bench.indexes) loaded_rows += index->size();
    if (loaded_rows != bench.options.skip_vectors) {
        fmt::print("Loaded indexes cover {} rows, but --skip-vectors says {}\n", loaded_rows,
                   bench.options.skip_vectors);
        return false;
    }

    // Every pool thread constructs its own traversal context, so the scratch - results,
    // frontier, visited set - lands on the thread's local memory domain, not the caller's.
    bench.traversals.resize(threads);
    auto traversals_init = [](fu_lambda_context_t punned, std::size_t thread, std::size_t compute_domain) noexcept {
        auto& bench = *static_cast<bench_t*>(punned);
        // AMX tile state needs a per-thread OS permission handshake before first use;
        // NumKong's thread configurator performs it and no-ops everywhere else.
        nk_configure_thread(nk_capabilities_available());
        fu_memory_domain_id_t const domain =
            bench.options.memory_domains.empty()
                ? fu_memory_domain_id_t {-1}
                : bench.options.memory_domains[bench.index_of_compute_domain[compute_domain]];
        bench.traversals[thread] = std::make_unique<traversal_t>(
            typename traversal_t::allocator_t {allocator_t {domain}});
    };
    fu_pool_for_threads(pool, traversals_init, &bench);
    for (std::size_t thread = 0; thread != threads; ++thread)
        if (!bench.traversals[thread]) {
            fmt::print("Failed to build the per-thread traversal contexts\n");
            return false;
        }
    bench.lanes.resize(threads);
    // Self-recall probes every ingested row, so the per-query buffers follow the larger count.
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

/** @brief One vector per task, streamed: viewed in place when the scalars match, or cast
 *      through the thread's staging area, then inserted into the prong's own memory-domain
 *      index - sharding is emergent, memory follows the worker that pulled the row. */
template <typename streamed_dataset_type_>
void insertion_callback(fu_lambda_context_t punned, std::size_t task, std::size_t thread,
                        std::size_t compute_domain) noexcept {
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

    std::size_t const slot = bench.index_of_compute_domain[compute_domain];
    index_t& index = *bench.indexes[slot];
    inner_node_id_t local_id = 0;
    vectors_view_t const one_row {vector.data(), 1, streamed.vectors.extent(1)};
    auto inserted = index.insert(one_row, {&local_id, std::size_t {1}}, *bench.traversals[thread], config);
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
    bench.results.id_to_row[slot][local_id] = static_cast<std::uint32_t>(row);
}

template <typename streamed_dataset_type_>
bool run_insertion(streamed_dataset_type_& streamed, fu_pool_t pool) {
    using clock_t = std::chrono::steady_clock;
    using scalar_t = typename streamed_dataset_type_::scalar_t;
    auto& bench = streamed.bench;
    std::size_t const rows = streamed.vectors.extent(0);
    std::size_t const inserted_rows = rows - bench.options.skip_vectors;
    if (!inserted_rows) {
        fmt::print(header_style(), "\nInsertion\n");
        fmt::print("  skipped: the loaded base already covers all {} rows\n", rows);
        return true;
    }

    auto started = clock_t::now();
    fu_pool_for_n_dynamic(pool, inserted_rows, &insertion_callback<streamed_dataset_type_>, &streamed);
    double const seconds = std::chrono::duration<double>(clock_t::now() - started).count();

    if (std::size_t failures = bench.results.failed_tasks.load(); failures) {
        fmt::print("Insertion failed on {} vectors\n", failures);
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
    fmt::print(header_style(), "\nInsertion\n");
    fmt::print("  shape: {} × {}d\n", inserted_rows, streamed.vectors.extent(1));
    if (bench.options.skip_vectors) fmt::print("  topping: {} loaded rows\n", bench.options.skip_vectors);
    fmt::print("  quads: {} · {} · {} kit\n", scalar_t::dtype_name(),
               kernel_name(streamed_dataset_type_::bench_t::kernel_kind_k),
               bit_name(compute_span_k, index_t::picked_quads_capability_k, "serial"));
    if constexpr (index_t::has_tiles_k)
        fmt::print("  tiles: {} · reverse refinement\n",
                   bit_name(compute_span_k, index_t::picked_tiles_capability_k, "none"));
    if (index_t::picked_quads_capability_k == nk_cap_serial_k &&
        (bench.options.max_compute & nk_capabilities_available() & ~nk_cap_serial_k))
        fmt::print(verdict_style(false), "  ⚠ scalar fallback: no SIMD kit is adapted for this dtype & metric\n");
    fmt::print("  threads: {}\n", fu_pool_threads_count(pool));
    if (bench.indexes.size() > 1) fmt::print("  indexes: {}, one per memory domain\n", bench.indexes.size());
    fmt::print("  wall: {:.2f} s\n", seconds);
    fmt::print("  throughput: {}/s\n", humanize(double(inserted_rows) / seconds));
    fmt::print("  distances: {} ({:.1f}% refines, {:.1f}% reverse)\n", humanize(double(sums.computed_distances)),
               refine_share, reverse_share);
    fmt::print("  visited: {}\n", humanize(double(sums.visited_members)));
    fmt::print("  restarts: ");
    fmt::print(verdict_style(sums.restarts < inserted_rows / 100 + 1), "{}\n", sums.restarts);
    if (std::size_t resident = peak_resident_bytes(); resident)
        fmt::print("  peak resident: {:.2f} GB\n", double(resident) / 1e9);
    return true;
}

/** @brief Serializes every domain index as three RetriEval-layout matrices - vectors,
 *      neighbors, and the id → row map - so later runs @c --load the base and stream only
 *      the topping: a 100M experiment then costs an mmap, not a half-hour rebuild. */
template <typename streamed_dataset_type_>
bool save_indexes(streamed_dataset_type_& streamed) {
    using clock_t = std::chrono::steady_clock;
    auto& bench = streamed.bench;
    auto started = clock_t::now();
    std::size_t const slots = bench.indexes.size();
    std::size_t total_bytes = 0;
    for (std::size_t slot = 0; slot != slots; ++slot) {
        std::size_t const saved_bytes = save_checkpoint(bench.options, *bench.indexes[slot],
                                                        std::span {bench.results.id_to_row[slot]}, slot, slots);
        if (!saved_bytes) return false;
        total_bytes += saved_bytes;
    }
    double const seconds = std::chrono::duration<double>(clock_t::now() - started).count();
    fmt::print("Saved {} indexes: {:.2f} GB in {:.2f} s\n", bench.indexes.size(), double(total_bytes) / 1e9, seconds);
    return true;
}

/** @brief Every query is one task: cast through the thread's staging area if needed - before
 *      the clock - then scattered over all memory-domain indexes, gathered through
 *      `merge_into` with the id → row translation, and judged on the merged hits. The clock
 *      covers the whole multi-domain lookup - the honest end-to-end latency. */
template <typename streamed_dataset_type_>
void search_callback(fu_lambda_context_t punned, std::size_t query, std::size_t thread, std::size_t) noexcept {
    using bench_t = typename streamed_dataset_type_::bench_t;
    using index_t = typename bench_t::index_t;
    using small_size_t = typename bench_t::small_size_t;
    using vector_view_t = typename bench_t::vector_view_t;
    using clock_t = std::chrono::steady_clock;
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

    std::uint32_t* const merged_rows = bench.results.hit_rows.row(query).data();
    auto* const merged_distances = bench.results.hit_distances.row(query).data();
    std::size_t merged_count = 0;

    auto started = clock_t::now();
    for (std::size_t slot = 0; slot != bench.indexes.size(); ++slot) {
        index_t& index = *bench.indexes[slot];
        auto found = index.search(query_vector, *bench.traversals[thread], config);
        if (!found) {
            bench.results.failed_tasks.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        lane.visited_members += found.visited_members;
        lane.computed_distances += found.computed_distances;
        lane.restarts += found.restarts;
        auto const& results = bench.traversals[thread]->results;
        merged_count = merge_search_results(to_span(results), bench.results.id_to_row[slot],
                                            std::span {merged_rows, bench.options.top_k},
                                            std::span {merged_distances, bench.options.top_k}, merged_count);
    }
    bench.results.latencies_us[query] = std::chrono::duration<float, std::micro>(clock_t::now() - started).count();
    bench.results.hit_counts[query] = static_cast<std::uint32_t>(merged_count);
}

template <typename streamed_dataset_type_>
bool run_search(streamed_dataset_type_& streamed, fu_pool_t pool) {
    using clock_t = std::chrono::steady_clock;
    auto& bench = streamed.bench;
    std::size_t const query_count = bench.options.self_recall ? streamed.vectors.extent(0) : streamed.queries.extent(0);
    for (telemetry_lane_t& lane : bench.lanes) lane = telemetry_lane_t {};

    auto started = clock_t::now();
    fu_pool_for_n(pool, query_count, &search_callback<streamed_dataset_type_>, &streamed);
    double const seconds = std::chrono::duration<double>(clock_t::now() - started).count();

    if (std::size_t failures = bench.results.failed_tasks.load(); failures) {
        fmt::print("Search failed on {} queries\n", failures);
        return false;
    }
    telemetry_lane_t const sums = summed_lanes(bench.lanes);
    report_search_metrics(bench.options, query_count, seconds, fu_pool_threads_count(pool), sums,
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
int stage_and_run(bench_launch<dataset_scalar_type_> const& launch) {
    using allocator_t = domain_or_default_allocator<std::byte>;
    using index_t =
        inner_volatile_index<stored_dtype_, kernel_, capability_, wait_type_, allocator_t, allocator_t, allocator_t>;
    using bench_context_type_ = bench_context<index_t>;
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

    /** @brief Owns the staging trampoline planes for the whole run. */
    nk::tensor<scalar_t, std::allocator<scalar_t>, 3> staging;
    /** @brief Owns the rescaling planes @c streamed_dataset::rescaling views. */
    nk::tensor<nk::f32_t, std::allocator<nk::f32_t>, 3> rescaling_planes;
    if constexpr (!streamed_t::zero_copy_k) {
        std::size_t const threads = fu_pool_threads_count(launch.pool);
        std::size_t const batch_rows = launch.options.batch ? std::max(launch.options.batch, threads) : 4 * threads;
        std::size_t const rows_per_thread = divide_round_up(batch_rows, threads);
        staging = nk::tensor<scalar_t, std::allocator<scalar_t>, 3>::try_zeros({threads, rows_per_thread, dimensions});
        if (staging.empty()) {
            fmt::print("Failed to allocate {} staging rows of {}d {}\n", batch_rows, dimensions,
                       scalar_t::dtype_name());
            return 1;
        }
        streamed.staging = staging.span();

        if constexpr (streamed_t::scaled_k) {
            rescaling_planes = nk::tensor<nk::f32_t, std::allocator<nk::f32_t>, 3>::try_zeros(
                {threads, rows_per_thread, dimensions});
            if (rescaling_planes.empty()) {
                fmt::print("Failed to allocate {} rescaling rows of {}d f32\n", batch_rows, dimensions);
                return 1;
            }
            streamed.rescaling = rescaling_planes.span();
            streamed.establish_scale();
        }
    }

    if (dataset.has_ground_truth() && !launch.options.max_vectors) bench.results.ground_truth = dataset.neighbors();
    else if (dataset.has_ground_truth())
        fmt::print("Ground truth ignored: --max-vectors truncates the graph it was computed for\n");

    if (!prepare(streamed, launch.pool, launch.topology)) return 1;
    if (!run_insertion(streamed, launch.pool)) return 1;
    if (!launch.options.save_name.empty() && !save_indexes(streamed)) return 1;
    if (!run_search(streamed, launch.pool)) return 1;
    return 0;
}

/** @brief Kernel axis: lowers the runtime metric to a compile-time kernel kind, the terminal
 *      hop into @ref stage_and_run. */
template <typename dataset_scalar_type_, nk_dtype_t stored_dtype_, nk_capability_t capability_, typename wait_type_>
int run_with_metric(bench_launch<dataset_scalar_type_> const& launch) {
    return dispatch_value<nk_kernel_dot_k, nk_kernel_angular_k, nk_kernel_euclidean_k>(
        launch.options.kernel,
        [&](auto kernel) {
            return stage_and_run<dataset_scalar_type_, stored_dtype_, kernel.value, capability_, wait_type_>(launch);
        },
        [&](auto metric) {
            fmt::print("Unsupported metric {}\n", kernel_name(metric));
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

/** @brief Stored-dtype axis: the CLI's chosen store instantiates the compute chain, or reports
 *      the dtypes that carry adapted CPU kits. */
template <typename dataset_scalar_type_>
int run_with_stored_dtype(bench_launch<dataset_scalar_type_> const& launch) {
    return dispatch_value<nk_bf16_k, nk_i8_k, nk_u8_k, nk_i4_k>(
        launch.options.stored_dtype,
        [&](auto dtype) { return run_with_compute<dataset_scalar_type_, dtype.value>(launch); },
        [&](auto dtype) {
            fmt::print("Stored dtype {} has no adapted kernels; override with --bf16, --i8, --u8, or --i4\n",
                       dtype_name(dtype));
            return 1;
        });
}

/** @brief Opens the dataset with the file-native scalar and enters the typed cascade. */
template <typename dataset_scalar_type_>
int open_and_run(bench_options_t const& options, fu_pool_t pool, fu_topology_t topology) {
    auto opened = persisted_dataset<dataset_scalar_type_>::try_open(
        options.vectors_path.c_str(), options.queries_path.empty() ? nullptr : options.queries_path.c_str(),
        options.neighbors_path.empty() ? nullptr : options.neighbors_path.c_str());
    if (opened.status != status_t::success_k) {
        fmt::print("Failed to open the dataset: {}\n", options.vectors_path);
        return 1;
    }
    return run_with_stored_dtype(bench_launch<dataset_scalar_type_> {opened.value, options, pool, topology});
}

#pragma endregion

int main(int argc, char** argv) {
    bench_options_t options;
    std::string max_compute_name, max_schedule_name, memory_domain_name;

    auto cli = ( //
        (clipp::option("--vectors") & clipp::value("path", options.vectors_path)) % "base vectors, `.fbin` family",
        (clipp::option("--queries") & clipp::value("path", options.queries_path)) % "query vectors, same dtype",
        (clipp::option("--neighbors") & clipp::value("path", options.neighbors_path)) % "ground truth, `.ibin`",
        (clipp::option("--bf16").set(options.stored_dtype, nk_bf16_k) |
         clipp::option("--i8").set(options.stored_dtype, nk_i8_k) |
         clipp::option("--u8").set(options.stored_dtype, nk_u8_k) |
         clipp::option("--i4").set(options.stored_dtype, nk_i4_k)) %
            "stored dtype override, staging `nk::cast` on ingest",
        (clipp::option("--max-compute") & clipp::value("name", max_compute_name)) %
            "compute ceiling by microarchitecture, like `icelake`, or `serial`; newest detected by default",
        (clipp::option("--max-schedule") & clipp::value("name", max_schedule_name)) %
            "scheduling ceiling by wait instruction, like `tpause`, or `standard`; newest offered by default",
        (clipp::option("--dot").set(options.kernel, nk_kernel_dot_k) |
         clipp::option("--angular").set(options.kernel, nk_kernel_angular_k) |
         clipp::option("--euclidean").set(options.kernel, nk_kernel_euclidean_k)) %
            "metric; angular by default",
        (clipp::option("--threads") & clipp::value("count", options.threads)) %
            "worker threads; 7/8 of the logical cores by default",
        (clipp::option("--memory-domain") & clipp::value("id", memory_domain_name)) %
            "pin memory & threads to one NUMA domain, or `all` for one index per domain",
        (clipp::option("--batch") & clipp::value("rows", options.batch)) %
            "staging trampoline rows shared by all threads; 4 × threads by default",
        (clipp::option("--connectivity") & clipp::value("degree", options.connectivity)) % "16, 32, or 128",
        (clipp::option("--expansion-add") & clipp::value("breadth", options.expansion_add)) %
            "insertion candidate pool; 128 by default, tuned for 99% recall@10",
        (clipp::option("--expansion-search") & clipp::value("breadth", options.expansion_search)) %
            "search candidate pool; 192 by default, tuned for 99% recall@10",
        (clipp::option("--top-k") & clipp::value("count", options.top_k)) % "matches per query to judge",
        (clipp::option("--max-vectors") & clipp::value("count", options.max_vectors)) % "cap on ingested rows",
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
            "huge pages provisioned before launch; enforced, so base-page fallbacks fail fast",
        clipp::option("--help").set(options.help) % "print this message");

    if (!clipp::parse(argc, argv, cli) || options.help || options.vectors_path.empty()) {
        for (auto const& section : clipp::make_man_page(cli, "inner_index_bench_cpp"))
            fmt::print("{}\n{}\n\n", section.title(), section.content());
        return options.help ? 0 : 1;
    }

    nk_dtype_t const file_dtype = dtype_of_extension(options.vectors_path.c_str());
    if (file_dtype == nk_dtype_unknown_k || file_dtype == nk_u32_k) {
        fmt::print("Unrecognized vectors extension: {}\n", options.vectors_path);
        return 1;
    }
    if (options.stored_dtype == nk_dtype_unknown_k) options.stored_dtype = file_dtype;

    // Ceiling names resolve into allow-masks - the native capability currency on both
    // sides. A named compute ceiling keeps its table prefix plus serial; "serial" alone
    // keeps just the fallback bit. Capping above the detection is harmless: the dispatch
    // intersects with the runtime bits anyway.
    if (max_compute_name == "serial") { options.max_compute = nk_cap_serial_k; }
    else if (!max_compute_name.empty()) {
        std::size_t const named = index_of_name(compute_span_k, max_compute_name);
        if (named == compute_choices_count_k) {
            fmt::print("Unknown --max-compute {}; this table offers: serial", max_compute_name);
            for (capability_choice<nk_capability_t> const& choice : compute_choices_k) fmt::print(" {}", choice.name);
            fmt::print("\n");
            return 1;
        }
        options.max_compute = nk_cap_serial_k | prefix_mask(compute_span_k, named);
    }

    // A schedule ceiling caps only the waiter bits; the pool-management bits - topology,
    // placement, huge pages - stay allowed, so `fu_pool_new` keeps its full toolbox. The
    // floor keyword is "standard", after @c fu::standard_yield_t - Arm's own YIELD hint
    // legitimately owns the "yield" row.
    if (max_schedule_name == "standard") { options.max_schedule = schedule_floor_k; }
    else if (!max_schedule_name.empty()) {
        std::size_t const named = index_of_name(schedule_span_k, max_schedule_name);
        if (named == schedule_choices_count_k) {
            fmt::print("Unknown --max-schedule {}; this table offers: standard", max_schedule_name);
            for (capability_choice<fu_capabilities_t> const& choice : schedule_choices_k)
                fmt::print(" {}", choice.name);
            fmt::print("\n");
            return 1;
        }
        options.max_schedule = static_cast<fu_capabilities_t>(schedule_floor_k | prefix_mask(schedule_span_k, named));
    }

    fu_topology_t topology = fu_topology_new();
    fu_fabric_t fabric = fu_fabric_new();
    fu_pool_t pool = fu_pool_new("usearch_bench", options.max_schedule);
    if (!topology || !pool) {
        fmt::print("Failed to inspect the topology or create the pool\n");
        return 1;
    }

    // Resolve `--memory-domain` into the engaged-domain list: empty means one machine-wide
    // heap index, a single id pins memory & threads to that domain, and `all` builds one
    // index per memory domain - the list IS the placement, no modes.
    if (memory_domain_name == "all")
        for (std::size_t domain = 0; domain != fu_memory_domains_count(topology); ++domain)
            options.memory_domains.push_back(static_cast<fu_memory_domain_id_t>(domain));
    else if (!memory_domain_name.empty()) {
        fu_memory_domain_id_t const domain = static_cast<fu_memory_domain_id_t>(
            std::strtol(memory_domain_name.c_str(), nullptr, 10));
        if (domain < 0 || static_cast<std::size_t>(domain) >= fu_memory_domains_count(topology)) {
            fmt::print("Unknown --memory-domain {}; this machine has {} memory domains\n", memory_domain_name,
                       fu_memory_domains_count(topology));
            return 1;
        }
        options.memory_domains.push_back(domain);
    }

    // Default to 7/8 of the available logical cores: at full occupancy the strictly-ordered
    // publication convoy serializes on scheduler latency, and throughput collapses by an order
    // of magnitude. A single pinned domain carves both the core budget and the spawn from the
    // compute domain whose local memory it is.
    bool spawned = false;
    std::size_t threads = options.threads;
    if (options.memory_domains.size() == 1) {
        std::size_t const compute_domains = fu_compute_domains_count(topology);
        for (std::size_t compute_domain = 0; compute_domain != compute_domains && !spawned; ++compute_domain) {
            if (fu_local_memory_of(topology, compute_domain) !=
                static_cast<std::size_t>(options.memory_domains.front()))
                continue;
            if (!threads)
                threads = std::max<std::size_t>(1, fu_logical_cores_count_in(topology, compute_domain) * 7 / 8);
            spawned = fu_pool_spawn_on(topology, pool, compute_domain, threads, fu_caller_inclusive_k);
        }
    }
    else {
        if (!threads) threads = std::max<std::size_t>(1, fu_logical_cores_count(topology) * 7 / 8);
        spawned = fu_pool_spawn(topology, pool, threads, fu_caller_inclusive_k);
    }
    if (!spawned) {
        fmt::print("Failed to spawn {} threads\n", threads);
        return 1;
    }
    print_machine(topology, fabric, options, threads);

    // Huge pages are provisioned outside the process - see the header's shell snippet - and
    // `--huge-pages` declares the expectation, enforced against what the domains report.
    if (options.huge_pages) {
        std::size_t volume = 0;
        for (std::size_t domain = 0; domain != fu_memory_domains_count(topology); ++domain)
            volume += fu_volume_huge_pages_in(topology, domain);
        fmt::print("  huge pages expected: {}, available: {:.2f} GB\n", options.huge_pages, double(volume) / 1e9);
        if (!volume) {
            fmt::print("--huge-pages expects {}, but the domains report none\n", options.huge_pages);
            return 1;
        }
    }

    // File-scalar axis: the newest hop of the compile-time cascade, lowering the file's dtype
    // to the C++ wrapper the whole run is instantiated over through NumKong's own `type_for`.
    int const exit_code = dispatch_value<nk_f32_k, nk_bf16_k, nk_i8_k, nk_u8_k>(
        file_dtype,
        [&](auto dtype) { return open_and_run<typename nk::type_for<dtype.value>::type>(options, pool, topology); },
        [&](auto dtype) {
            fmt::print("Unsupported dataset dtype {}\n", dtype_name(dtype));
            return 1;
        });

    fu_pool_delete(pool);
    fu_fabric_delete(fabric);
    fu_topology_delete(topology);
    return exit_code;
}
