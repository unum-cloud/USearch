/**
 *  @brief Profiling harness for the volatile index over real persisted datasets.
 *
 *  @author Ash Vardanian
 *  @file cpp/inner_volatile_index_bench.cpp
 *  @date July 17, 2026
 *
 *  Unlike the test suite, which sweeps synthetic combinations for correctness, this harness
 *  profiles insertion & search throughput on RetriEval-style binary datasets. It builds on
 *  the ForkUnion C-level topology, fabric, and pool abstractions, and prints with {fmt}.
 *  A ready-made playground is the Unum Wiki 1M dataset:
 *
 *  @code{.sh}
 *  mkdir -p ~/datasets/wiki_1M && cd ~/datasets/wiki_1M &&
 *      wget -nc https://huggingface.co/datasets/unum-cloud/ann-wiki-1m/resolve/main/base.1M.fbin &&
 *      wget -nc https://huggingface.co/datasets/unum-cloud/ann-wiki-1m/resolve/main/query.public.100K.fbin &&
 *      wget -nc https://huggingface.co/datasets/unum-cloud/ann-wiki-1m/resolve/main/groundtruth.public.100K.ibin
 *  inner_volatile_index_bench_cpp --vectors ~/datasets/wiki_1M/base.1M.fbin \
 *      --queries ~/datasets/wiki_1M/query.public.100K.fbin \
 *      --neighbors ~/datasets/wiki_1M/groundtruth.public.100K.ibin --bf16
 *  @endcode
 *
 *  Defaults are inferred and each is overridable:
 *  - the stored dtype matches the dataset file, unless `--bf16`, `--i8`, `--u8`, or `--i4`
 *    asks for a streaming `nk::cast` through a `--batch`-row trampoline on ingest;
 *  - the newest runtime-detected compute capability drives the distance kernels, unless
 *    `--max-compute` caps it, and threads wait on monitor instructions like `tpause`,
 *    unless `--max-schedule` degrades them to plain pauses or STL yields;
 *  - 7/8 of the logical cores participate, unless `--threads` overrides - full occupancy
 *    starves the ordered publication path and collapses throughput.
 */
#include <algorithm> // `std::nth_element`
#include <atomic>    // `std::atomic`
#include <chrono>    // `std::chrono::steady_clock`
#include <cmath>     // `std::abs`
#include <cstdint>   // `std::uint32_t`
#include <string>    // `std::string`
#include <vector>    // `std::vector`

#include <clipp.h>
#include <fmt/color.h>
#include <fmt/format.h>

#include <forkunion.h>
#include <forkunion/capabilities.hpp>
#include <numkong/numkong.h>
#include <numkong/numkong.hpp>

#include <usearch/inner_volatile_index.hpp>

#include "bench.hpp"
#include "persisted_dataset.hpp"

namespace us = unum::usearch;
namespace fu = ashvardanian::forkunion;
namespace nk = ashvardanian::numkong;

using us::dtype_name;
using us::header_style;
using us::humanize;
using us::kernel_name;
using us::peak_resident_bytes;
using us::verdict_style;

#pragma region Options and Styling

/** @brief Compute-capability ceilings the kernel chains adapt, in ascending preference;
 *      `--max-compute <name>` caps the effective set at everything at or below it. */
struct compute_choice_t {
    char const* name;
    nk_capability_t bit;
};
inline constexpr compute_choice_t compute_choices_k[] = {{"serial", nk_cap_serial_k},
                                                         {"haswell", nk_cap_haswell_k},
                                                         {"skylake", nk_cap_skylake_k},
                                                         {"icelake", nk_cap_icelake_k},
                                                         {"genoa", nk_cap_genoa_k}};
inline constexpr std::size_t compute_choices_count_k = sizeof(compute_choices_k) / sizeof(compute_choices_k[0]);

/** @brief Everything at or below the @p index-th compute ceiling, as one chain mask. */
constexpr nk_capability_t compute_prefix_mask(std::size_t index) noexcept {
    nk_capability_t mask = 0;
    for (std::size_t choice = 0; choice <= index; ++choice) mask |= compute_choices_k[choice].bit;
    return mask;
}

/** @brief Names a single capability bit - for reporting which kit a chain actually picked. */
static char const* compute_bit_name(nk_capability_t bit) noexcept {
    for (compute_choice_t const& choice : compute_choices_k)
        if (choice.bit == bit) return choice.name;
    return "unknown";
}

/** @brief Scheduling-capability ceilings: how hard a stalled thread may lean on the core -
 *      portable STL yields, the architecture's pause hint, or monitor-waits like `tpause`. */
enum class schedule_ceiling_t { yield_k, pause_k, monitor_k };

using yield_waiter_t = fu::standard_yield_t;
#if defined(__x86_64__) || defined(_M_X64)
using pause_waiter_t = fu::x86_pause_t;
#elif defined(__aarch64__) || defined(_M_ARM64)
using pause_waiter_t = fu::arm64_yield_t;
#else
using pause_waiter_t = fu::standard_yield_t;
#endif
using monitor_waiter_t = fu::preferred_yield_t;

static char const* schedule_name(schedule_ceiling_t ceiling) noexcept {
    switch (ceiling) {
    case schedule_ceiling_t::yield_k: return "yield";
    case schedule_ceiling_t::pause_k: return "pause";
    default: return "monitor";
    }
}

/** @brief Everything the CLI can configure; zeroes mean "infer from the dataset & machine". */
struct bench_options_t {
    std::string vectors_path;
    std::string queries_path;
    std::string neighbors_path;
    nk_dtype_t stored_dtype = nk_dtype_unknown_k;
    std::size_t max_compute_index = compute_choices_count_k; // The sentinel means "newest detected".
    schedule_ceiling_t max_schedule = schedule_ceiling_t::monitor_k;
    nk_kernel_kind_t kernel = nk_kernel_angular_k;
    std::size_t threads = 0;
    std::size_t batch = 0;
    std::size_t connectivity = 32;
    std::size_t expansion_add = 0;
    std::size_t expansion_search = 0;
    std::size_t top_k = 10;
    std::size_t max_vectors = 0;
    bool help = false;
};

#pragma endregion

#pragma region Typed Runs

/** @brief Telemetry sums each thread accumulates privately, padded away from its neighbors. */
struct alignas(64) thread_totals_t {
    std::size_t visited_members = 0;
    std::size_t computed_distances = 0;
    std::size_t computed_distances_in_refines = 0;
    std::size_t computed_distances_in_reverse_refines = 0;
    std::size_t restarts = 0;
    std::size_t hits_at_1 = 0;
    std::size_t hits_at_k = 0;
    std::size_t judged_queries = 0;
};

/**
 *  @brief The index-side state one fully-typed benchmark run shares between its phases:
 *      the index, its per-thread traversal contexts, and the recall bookkeeping.
 *
 *  The dataset never appears here - the data side lives in @c streamed_dataset_t, which
 *  bridges the file's scalar into the stored one vector-by-vector.
 */
template <nk_dtype_t stored_dtype_, nk_kernel_kind_t kernel_, nk_capability_t capability_, typename wait_type_>
struct bench_context_t {
    using index_t = us::inner_volatile_index<stored_dtype_, kernel_, capability_, wait_type_, std::allocator<std::byte>,
                                             std::allocator<std::byte>, std::allocator<std::byte>>;
    using scalar_t = typename index_t::scalar_t;
    using small_size_t = typename index_t::small_size_t;
    using traversal_t = typename index_t::inner_traversal_context_t;
    using vector_view_t = typename index_t::vector_view_t;
    using vectors_view_t = typename index_t::vectors_view_t;

    static constexpr nk_kernel_kind_t kernel_kind_k = kernel_;

    bench_options_t const& options;
    index_t index;
    nk::matrix_view<std::uint32_t> ground_truth; // Empty when absent or truncated by `--max-vectors`.
    std::vector<traversal_t> traversals;         // One graph-walking context per pool thread.
    std::vector<us::inner_node_id_t> assigned_ids;
    std::vector<std::uint32_t> id_to_row; // Recall's node-id → dataset-row inverse permutation.
    std::vector<thread_totals_t> lanes;
    std::vector<float> latencies_us;
    std::atomic<std::size_t> failed_tasks {0};

    bench_context_t(bench_options_t const& options) noexcept : options(options) {}
};

/**
 *  @brief The data-side of a run: typed views over the source rows and the bounded staging
 *      matrix the streaming `nk::cast`s trampoline through, one vector at a time.
 *
 *  The staging matrix holds `--batch` rows for ALL threads - never a whole-dataset copy.
 *  Each thread owns a contiguous area of it and rotates through that area as its tasks
 *  stream by, so concurrent tasks never share a row and no `malloc` follows the clocks.
 *  When the file scalar already matches the stored one, the staging is empty and every
 *  access is a zero-copy view into the mapping.
 */
template <typename dataset_scalar_type_, typename bench_context_type_>
struct streamed_dataset_t {
    using bench_t = bench_context_type_;
    using scalar_t = typename bench_context_type_::scalar_t;
    using vector_view_t = typename bench_context_type_::vector_view_t;
    static constexpr bool zero_copy_k = std::is_same_v<dataset_scalar_type_, scalar_t>;

    /** @brief Float sources downcast into integer targets only after a global symmetric
     *      rescale into the target's range - a raw saturating cast collapses the value
     *      distribution, degenerates the distances, and loops the graph traversal. */
    static constexpr bool float_source_k = dataset_scalar_type_::dtype() == nk_f32_k ||
                                           dataset_scalar_type_::dtype() == nk_bf16_k;
    static constexpr bool integer_target_k = scalar_t::dtype() == nk_i8_k || scalar_t::dtype() == nk_u8_k ||
                                             scalar_t::dtype() == nk_i4_k;
    static constexpr bool scaled_k = !zero_copy_k && float_source_k && integer_target_k;
    static constexpr double target_amplitude_k = scalar_t::dtype() == nk_i8_k   ? 127.0
                                                 : scalar_t::dtype() == nk_u8_k ? 255.0
                                                                                : 7.0;

    bench_context_type_& bench;
    nk::matrix_view<dataset_scalar_type_> vectors;
    nk::matrix_view<dataset_scalar_type_> queries;
    nk::matrix_span<scalar_t> staging;
    nk::matrix_span<nk::f32_t> scaling_temp; // Sized like `staging`; only allocated when `scaled_k`.
    std::size_t staging_rows_per_thread = 1;
    float scale = 1.0f;

    /** @brief Converts one source row into the calling thread's own staging area and views
     *      it back; @p task rotates the row within the area, so in-flight tasks never clash. */
    vector_view_t stage(nk::vector_view<dataset_scalar_type_> source, std::size_t thread, std::size_t task) noexcept {
        std::size_t const dimensions = vectors.extent(1);
        std::size_t const landing_row = thread * staging_rows_per_thread + task % staging_rows_per_thread;
        auto landing = staging.row(landing_row).as_vector();
        if constexpr (scaled_k) {
            auto rescaled = scaling_temp.row(landing_row).as_vector();
            nk::cast(source, rescaled);
            for (std::size_t dimension = 0; dimension != dimensions; ++dimension) rescaled[dimension] *= scale;
            nk::cast(nk::vector_view<nk::f32_t> {rescaled.data(), dimensions}, landing);
        }
        else { nk::cast(source, landing); }
        return vector_view_t {landing.data(), dimensions};
    }
};

/** @brief Sums the per-thread telemetry after a phase; the lanes stay untouched. */
static thread_totals_t summed_lanes(std::vector<thread_totals_t> const& lanes) noexcept {
    thread_totals_t sums;
    for (thread_totals_t const& lane : lanes) {
        sums.visited_members += lane.visited_members;
        sums.computed_distances += lane.computed_distances;
        sums.computed_distances_in_refines += lane.computed_distances_in_refines;
        sums.computed_distances_in_reverse_refines += lane.computed_distances_in_reverse_refines;
        sums.restarts += lane.restarts;
        sums.hits_at_1 += lane.hits_at_1;
        sums.hits_at_k += lane.hits_at_k;
        sums.judged_queries += lane.judged_queries;
    }
    return sums;
}

/** @brief Reserves the index for the viewed rows and sizes all per-thread state off the
 *      pool's actual thread count - everything fallible, before any clock starts. */
template <typename streamed_dataset_type_>
bool prepare(streamed_dataset_type_& streamed, fu_pool_t pool) {
    using traversal_t = typename streamed_dataset_type_::bench_t::traversal_t;
    auto& bench = streamed.bench;
    std::size_t const rows = streamed.vectors.extent(0);
    std::size_t const dimensions = streamed.vectors.extent(1);
    std::size_t const threads = fu_pool_threads_count(pool);

    us::connectivity_t connectivity = us::connectivity_unknown_k;
    if (bench.options.connectivity == 16) connectivity = us::connectivity_16_k;
    else if (bench.options.connectivity == 32) connectivity = us::connectivity_32_k;
    else if (bench.options.connectivity == 128) connectivity = us::connectivity_128_k;
    else {
        fmt::print("Unsupported --connectivity {}; pick 16, 32, or 128\n", bench.options.connectivity);
        return false;
    }

    auto reserved = bench.index.reserve({rows, connectivity, dimensions});
    if (reserved.status != us::status_t::success_k) {
        fmt::print("Failed to reserve {} × {}d\n", rows, dimensions);
        return false;
    }

    bench.traversals.reserve(threads);
    for (std::size_t thread = 0; thread != threads; ++thread)
        bench.traversals.emplace_back(typename traversal_t::allocator_t {});
    bench.assigned_ids.resize(rows);
    bench.id_to_row.resize(rows);
    bench.lanes.resize(threads);
    bench.latencies_us.resize(streamed.queries.extent(0));
    return true;
}

/** @brief One vector per task, streamed: viewed in place when the scalars match, or cast
 *      through the thread's staging area, then inserted and folded into its lane. */
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
    vector_view_t vector;
    if constexpr (streamed_dataset_type_::zero_copy_k) vector = streamed.vectors.row(task).as_vector();
    else vector = streamed.stage(streamed.vectors.row(task).as_vector(), thread, task);

    vectors_view_t const one_row {vector.data(), 1, streamed.vectors.extent(1)};
    auto inserted = bench.index.insert(one_row, {bench.assigned_ids.data() + task, std::size_t {1}},
                                       bench.traversals[thread], config);
    if (!inserted) {
        bench.failed_tasks.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    thread_totals_t& lane = bench.lanes[thread];
    lane.visited_members += inserted.visited_members;
    lane.computed_distances += inserted.computed_distances;
    lane.computed_distances_in_refines += inserted.computed_distances_in_refines;
    lane.computed_distances_in_reverse_refines += inserted.computed_distances_in_reverse_refines;
    lane.restarts += inserted.restarts;
    bench.id_to_row[bench.assigned_ids[task]] = static_cast<std::uint32_t>(task);
}

template <typename streamed_dataset_type_>
bool run_insertion(streamed_dataset_type_& streamed, fu_pool_t pool) {
    using clock_t = std::chrono::steady_clock;
    using scalar_t = typename streamed_dataset_type_::scalar_t;
    auto& bench = streamed.bench;
    std::size_t const rows = streamed.vectors.extent(0);

    auto started = clock_t::now();
    fu_pool_for_n_dynamic(pool, rows, &insertion_callback<streamed_dataset_type_>, &streamed);
    double const seconds = std::chrono::duration<double>(clock_t::now() - started).count();

    if (std::size_t failures = bench.failed_tasks.load(); failures) {
        fmt::print("Insertion failed on {} vectors\n", failures);
        return false;
    }
    thread_totals_t const sums = summed_lanes(bench.lanes);

    using index_t = typename streamed_dataset_type_::bench_t::index_t;
    fmt::print(header_style(), "\nInsertion  {} × {}d  {} · {} · {} kit · {} threads\n", rows,
               streamed.vectors.extent(1), scalar_t::dtype_name(),
               kernel_name(streamed_dataset_type_::bench_t::kernel_kind_k),
               compute_bit_name(index_t::picked_capability_k), fu_pool_threads_count(pool));
    if (index_t::picked_capability_k == nk_cap_serial_k && bench.options.max_compute_index != 0)
        fmt::print(verdict_style(false), "  ⚠ scalar fallback: no SIMD kit is adapted for this dtype & metric\n");
    fmt::print("  wall {:.2f} s → {}/s\n", seconds, humanize(double(rows) / seconds));
    double const refine_share = sums.computed_distances ? 100.0 * double(sums.computed_distances_in_refines) /
                                                              double(sums.computed_distances)
                                                        : 0.0;
    double const reverse_share = sums.computed_distances ? 100.0 * double(sums.computed_distances_in_reverse_refines) /
                                                               double(sums.computed_distances)
                                                         : 0.0;
    fmt::print("  distances {} ({:.1f}% refines, {:.1f}% reverse), visited {}, restarts ",
               humanize(double(sums.computed_distances)), refine_share, reverse_share,
               humanize(double(sums.visited_members)));
    fmt::print(verdict_style(sums.restarts < rows / 100 + 1), "{}\n", sums.restarts);
    if (std::size_t resident = peak_resident_bytes(); resident)
        fmt::print("  peak resident {:.2f} GB\n", double(resident) / 1e9);
    return true;
}

/** @brief Every query is one task: cast through the thread's staging area if needed - before
 *      the clock - then searched with only the walk itself timed, and judged right away. */
template <typename streamed_dataset_type_>
void search_callback(fu_lambda_context_t punned, std::size_t query, std::size_t thread, std::size_t) noexcept {
    using bench_t = typename streamed_dataset_type_::bench_t;
    using index_t = typename bench_t::index_t;
    using small_size_t = typename bench_t::small_size_t;
    using vector_view_t = typename bench_t::vector_view_t;
    using clock_t = std::chrono::steady_clock;
    streamed_dataset_type_& streamed = *static_cast<streamed_dataset_type_*>(punned);
    auto& bench = streamed.bench;
    thread_totals_t& lane = bench.lanes[thread];

    typename index_t::search_config_t config;
    config.expansion_factor = static_cast<small_size_t>(bench.options.expansion_search);
    vector_view_t query_vector;
    if constexpr (streamed_dataset_type_::zero_copy_k) query_vector = streamed.queries.row(query).as_vector();
    else query_vector = streamed.stage(streamed.queries.row(query).as_vector(), thread, query);

    auto started = clock_t::now();
    auto found = bench.index.search(query_vector, bench.traversals[thread], config);
    bench.latencies_us[query] = std::chrono::duration<float, std::micro>(clock_t::now() - started).count();

    if (!found) {
        bench.failed_tasks.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    lane.visited_members += found.visited_members;
    lane.computed_distances += found.computed_distances;
    lane.restarts += found.restarts;
    if (!bench.ground_truth.data()) return;

    auto expected_rows = bench.ground_truth.row(query).as_vector();
    std::size_t const judged_depth = std::min(bench.options.top_k, bench.ground_truth.extent(1));
    std::size_t const matches = std::min(found.count, bench.options.top_k);
    us::inner_scored_node_t const* results = bench.traversals[thread].results.data();
    lane.judged_queries += 1;
    if (matches && bench.id_to_row[results[0].node_id] == expected_rows[0]) lane.hits_at_1 += 1;
    for (std::size_t match = 0; match != matches; ++match) {
        std::uint32_t const returned_row = bench.id_to_row[results[match].node_id];
        for (std::size_t expected = 0; expected != judged_depth; ++expected)
            if (expected_rows[expected] == returned_row) {
                lane.hits_at_k += 1;
                break;
            }
    }
}

template <typename streamed_dataset_type_>
bool run_search(streamed_dataset_type_& streamed, fu_pool_t pool) {
    using clock_t = std::chrono::steady_clock;
    auto& bench = streamed.bench;
    std::size_t const query_count = streamed.queries.extent(0);
    for (thread_totals_t& lane : bench.lanes) lane = thread_totals_t {};

    auto started = clock_t::now();
    fu_pool_for_n(pool, query_count, &search_callback<streamed_dataset_type_>, &streamed);
    double const seconds = std::chrono::duration<double>(clock_t::now() - started).count();

    if (std::size_t failures = bench.failed_tasks.load(); failures) {
        fmt::print("Search failed on {} queries\n", failures);
        return false;
    }
    thread_totals_t const sums = summed_lanes(bench.lanes);

    auto percentile = [&](double share) {
        std::size_t const rank = std::min(query_count - 1, std::size_t(share * double(query_count)));
        std::nth_element(bench.latencies_us.begin(), bench.latencies_us.begin() + rank, bench.latencies_us.end());
        return bench.latencies_us[rank];
    };

    fmt::print(header_style(), "\nSearch  {} queries  top-{}  {} threads\n", query_count, bench.options.top_k,
               fu_pool_threads_count(pool));
    fmt::print("  wall {:.2f} s → {} QPS,  latency p50 {:.0f} µs  p90 {:.0f} µs  p99 {:.0f} µs\n", seconds,
               humanize(double(query_count) / seconds), percentile(0.50), percentile(0.90), percentile(0.99));
    fmt::print("  distances/query {},  visited/query {},  restarts {}\n",
               humanize(double(sums.computed_distances) / double(query_count)),
               humanize(double(sums.visited_members) / double(query_count)), sums.restarts);
    if (sums.judged_queries) {
        std::size_t const judged_depth = std::min(bench.options.top_k, bench.ground_truth.extent(1));
        double const recall_at_1 = 100.0 * double(sums.hits_at_1) / double(sums.judged_queries);
        double const recall_at_k = 100.0 * double(sums.hits_at_k) / double(sums.judged_queries * judged_depth);
        fmt::print("  recall@1 ");
        fmt::print(verdict_style(recall_at_1 >= 90.0), "{:.1f}%", recall_at_1);
        fmt::print("  recall@{} ", judged_depth);
        fmt::print(verdict_style(recall_at_k >= 90.0), "{:.1f}%\n", recall_at_k);
    }
    else { fmt::print("  recall skipped: no ground-truth neighbors provided\n"); }
    return true;
}

#pragma endregion

#pragma region Dispatch Cascade

/** @brief Bundles the launch state the cascade threads through its compile-time hops. */
template <typename dataset_scalar_type_>
struct bench_launch_t {
    us::persisted_dataset<dataset_scalar_type_> const& dataset;
    bench_options_t const& options;
    fu_pool_t pool;
};

/**
 *  @brief The innermost hop: wires the typed source views and the bounded staging
 *      trampoline into a @c streamed_dataset_t, and runs both phases.
 *
 *  When the file scalar already matches the stored one, everything is a zero-copy view
 *  into the mappings; otherwise the staging matrix holds just `--batch` rows - typically a
 *  few per thread - and each vector streams through it on its way into the index. The
 *  dataset is never copied whole.
 */
template <typename dataset_scalar_type_, nk_dtype_t stored_dtype_, nk_kernel_kind_t kernel_,
          nk_capability_t capability_, typename wait_type_>
int stage_and_run(bench_launch_t<dataset_scalar_type_> const& launch) {
    using bench_context_type_ = bench_context_t<stored_dtype_, kernel_, capability_, wait_type_>;
    using streamed_t = streamed_dataset_t<dataset_scalar_type_, bench_context_type_>;
    using scalar_t = typename bench_context_type_::scalar_t;

    us::persisted_dataset<dataset_scalar_type_> const& dataset = launch.dataset;
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

    nk::matrix<scalar_t> staging;         // Owns the trampoline rows for the whole run.
    nk::matrix<nk::f32_t> rescaling_rows; // Owns the pre-downcast rescaling rows, when needed.
    if constexpr (!streamed_t::zero_copy_k) {
        std::size_t const threads = fu_pool_threads_count(launch.pool);
        std::size_t const batch_rows = launch.options.batch ? std::max(launch.options.batch, threads) : 4 * threads;
        staging = nk::matrix<scalar_t>::try_zeros({batch_rows, dimensions});
        if (!staging.data()) {
            fmt::print("Failed to allocate {} staging rows of {}d {}\n", batch_rows, dimensions,
                       scalar_t::dtype_name());
            return 1;
        }
        streamed.staging = staging.span();
        streamed.staging_rows_per_thread = batch_rows / threads;

        if constexpr (streamed_t::scaled_k) {
            rescaling_rows = nk::matrix<nk::f32_t>::try_zeros({batch_rows, dimensions});
            if (!rescaling_rows.data()) {
                fmt::print("Failed to allocate {} rescaling rows of {}d f32\n", batch_rows, dimensions);
                return 1;
            }
            streamed.scaling_temp = rescaling_rows.span();

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

    if (dataset.has_ground_truth() && !launch.options.max_vectors) bench.ground_truth = dataset.neighbors();
    else if (dataset.has_ground_truth())
        fmt::print("Ground truth ignored: --max-vectors truncates the graph it was computed for\n");

    if (!prepare(streamed, launch.pool)) return 1;
    if (!run_insertion(streamed, launch.pool)) return 1;
    if (!run_search(streamed, launch.pool)) return 1;
    return 0;
}

template <typename dataset_scalar_type_, nk_dtype_t stored_dtype_, nk_capability_t capability_, typename wait_type_>
int run_with_metric(bench_launch_t<dataset_scalar_type_> const& launch) {
    switch (launch.options.kernel) {
    case nk_kernel_dot_k:
        return stage_and_run<dataset_scalar_type_, stored_dtype_, nk_kernel_dot_k, capability_, wait_type_>(launch);
    case nk_kernel_angular_k:
        return stage_and_run<dataset_scalar_type_, stored_dtype_, nk_kernel_angular_k, capability_, wait_type_>(launch);
    case nk_kernel_euclidean_k:
        return stage_and_run<dataset_scalar_type_, stored_dtype_, nk_kernel_euclidean_k, capability_, wait_type_>(
            launch);
    default: fmt::print("Unsupported metric\n"); return 1;
    }
}

template <typename dataset_scalar_type_, nk_dtype_t stored_dtype_, nk_capability_t capability_>
int run_with_schedule(bench_launch_t<dataset_scalar_type_> const& launch) {
    switch (launch.options.max_schedule) {
    case schedule_ceiling_t::yield_k:
        return run_with_metric<dataset_scalar_type_, stored_dtype_, capability_, yield_waiter_t>(launch);
    case schedule_ceiling_t::pause_k:
        return run_with_metric<dataset_scalar_type_, stored_dtype_, capability_, pause_waiter_t>(launch);
    default: return run_with_metric<dataset_scalar_type_, stored_dtype_, capability_, monitor_waiter_t>(launch);
    }
}

template <typename dataset_scalar_type_, nk_dtype_t stored_dtype_>
int run_with_compute(bench_launch_t<dataset_scalar_type_> const& launch) {
    switch (launch.options.max_compute_index) {
    case 0: return run_with_schedule<dataset_scalar_type_, stored_dtype_, compute_prefix_mask(0)>(launch);
    case 1: return run_with_schedule<dataset_scalar_type_, stored_dtype_, compute_prefix_mask(1)>(launch);
    case 2: return run_with_schedule<dataset_scalar_type_, stored_dtype_, compute_prefix_mask(2)>(launch);
    case 3: return run_with_schedule<dataset_scalar_type_, stored_dtype_, compute_prefix_mask(3)>(launch);
    case 4: return run_with_schedule<dataset_scalar_type_, stored_dtype_, compute_prefix_mask(4)>(launch);
    default: fmt::print("The compute ceiling was not resolved\n"); return 1;
    }
}

template <typename dataset_scalar_type_>
int run_with_stored_dtype(bench_launch_t<dataset_scalar_type_> const& launch) {
    switch (launch.options.stored_dtype) {
    case nk_bf16_k: return run_with_compute<dataset_scalar_type_, nk_bf16_k>(launch);
    case nk_i8_k: return run_with_compute<dataset_scalar_type_, nk_i8_k>(launch);
    case nk_u8_k: return run_with_compute<dataset_scalar_type_, nk_u8_k>(launch);
    case nk_i4_k: return run_with_compute<dataset_scalar_type_, nk_i4_k>(launch);
    default:
        fmt::print("Stored dtype {} has no adapted kernels; override with --bf16, --i8, --u8, or --i4\n",
                   dtype_name(launch.options.stored_dtype));
        return 1;
    }
}

/** @brief Opens the dataset with the file-native scalar and enters the typed cascade. */
template <typename dataset_scalar_type_>
int open_and_run(bench_options_t const& options, fu_pool_t pool) {
    auto opened = us::persisted_dataset<dataset_scalar_type_>::try_open(
        options.vectors_path.c_str(), options.queries_path.empty() ? nullptr : options.queries_path.c_str(),
        options.neighbors_path.empty() ? nullptr : options.neighbors_path.c_str());
    if (opened.status != us::status_t::success_k) {
        fmt::print("Failed to open the dataset: {}\n", options.vectors_path);
        return 1;
    }
    return run_with_stored_dtype(bench_launch_t<dataset_scalar_type_> {opened.value, options, pool});
}

#pragma endregion

/** @brief Prints the machine as ForkUnion sees it: cores, compute & memory domains, and the
 *      NUMA fabric levels, next to the NumKong capability the kernels will actually use. */
static void print_machine(fu_topology_t topology, fu_fabric_t fabric, bench_options_t const& options,
                          std::size_t threads) {
    std::size_t const compute_domains = fu_compute_domains_count(topology);
    std::size_t const memory_domains = fu_memory_domains_count(topology);
    fmt::print(header_style(), "Machine\n");
    fmt::print("  {} logical cores, {} compute domains, {} memory domains, {} threads engaged\n",
               fu_logical_cores_count(topology), compute_domains, memory_domains, threads);
    for (std::size_t domain = 0; domain != memory_domains; ++domain)
        fmt::print("  memory domain {} at fabric level {}\n", domain, fu_fabric_memory_level_in(fabric, domain));

    nk_capability_t const available = nk_capabilities_available();
    fmt::print("  compute detected:");
    for (compute_choice_t const& choice : compute_choices_k)
        if (available & choice.bit) fmt::print(" {}", choice.name);
    fmt::print("\n  kernels: {} · {} · {} compute · {} scheduling\n", dtype_name(options.stored_dtype),
               kernel_name(options.kernel), compute_choices_k[options.max_compute_index].name,
               schedule_name(options.max_schedule));
}

int main(int argc, char** argv) {
    bench_options_t options;
    std::string max_compute_name, max_schedule_name;

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
            "compute ceiling: serial, haswell, skylake, icelake, genoa; newest detected by default",
        (clipp::option("--max-schedule") & clipp::value("name", max_schedule_name)) %
            "scheduling ceiling: yield, pause, monitor; monitor by default",
        (clipp::option("--dot").set(options.kernel, nk_kernel_dot_k) |
         clipp::option("--angular").set(options.kernel, nk_kernel_angular_k) |
         clipp::option("--euclidean").set(options.kernel, nk_kernel_euclidean_k)) %
            "metric; angular by default",
        (clipp::option("--threads") & clipp::value("count", options.threads)) %
            "worker threads; 7/8 of the logical cores by default",
        (clipp::option("--batch") & clipp::value("rows", options.batch)) %
            "staging trampoline rows shared by all threads; 4 × threads by default",
        (clipp::option("--connectivity") & clipp::value("degree", options.connectivity)) % "16, 32, or 128",
        (clipp::option("--expansion-add") & clipp::value("breadth", options.expansion_add)) %
            "insertion candidate pool; 2 × connectivity by default",
        (clipp::option("--expansion-search") & clipp::value("breadth", options.expansion_search)) %
            "search candidate pool; 2 × connectivity by default",
        (clipp::option("--top-k") & clipp::value("count", options.top_k)) % "matches per query to judge",
        (clipp::option("--max-vectors") & clipp::value("count", options.max_vectors)) % "cap on ingested rows",
        clipp::option("--help").set(options.help) % "print this message");

    if (!clipp::parse(argc, argv, cli) || options.help || options.vectors_path.empty()) {
        for (auto const& section : clipp::make_man_page(cli, "inner_volatile_index_bench_cpp"))
            fmt::print("{}\n{}\n\n", section.title(), section.content());
        return options.help ? 0 : 1;
    }

    nk_dtype_t const file_dtype = us::dtype_of_extension(options.vectors_path.c_str());
    if (file_dtype == nk_dtype_unknown_k || file_dtype == nk_u32_k) {
        fmt::print("Unrecognized vectors extension: {}\n", options.vectors_path);
        return 1;
    }
    if (options.stored_dtype == nk_dtype_unknown_k) options.stored_dtype = file_dtype;

    nk_capability_t const available = nk_capabilities_available();
    if (!max_compute_name.empty()) {
        for (std::size_t choice = 0; choice != compute_choices_count_k; ++choice)
            if (max_compute_name == compute_choices_k[choice].name) options.max_compute_index = choice;
        if (options.max_compute_index == compute_choices_count_k ||
            !(available & compute_choices_k[options.max_compute_index].bit)) {
            fmt::print("Unknown or unsupported --max-compute {}; this CPU offers:", max_compute_name);
            for (compute_choice_t const& choice : compute_choices_k)
                if (available & choice.bit) fmt::print(" {}", choice.name);
            fmt::print("\n");
            return 1;
        }
    }
    else // The newest detected ceiling wins when the CLI stays silent.
        for (std::size_t choice = 0; choice != compute_choices_count_k; ++choice)
            if (available & compute_choices_k[choice].bit) options.max_compute_index = choice;

    if (!max_schedule_name.empty()) {
        if (max_schedule_name == "yield") options.max_schedule = schedule_ceiling_t::yield_k;
        else if (max_schedule_name == "pause") options.max_schedule = schedule_ceiling_t::pause_k;
        else if (max_schedule_name == "monitor") options.max_schedule = schedule_ceiling_t::monitor_k;
        else {
            fmt::print("Unknown --max-schedule {}; pick yield, pause, or monitor\n", max_schedule_name);
            return 1;
        }
    }

    fu_topology_t topology = fu_topology_new();
    fu_fabric_t fabric = fu_fabric_new();
    fu_pool_t pool = fu_pool_new("usearch_bench", fu_capabilities_all_k);
    // Default to 7/8 of the logical cores: at full occupancy the strictly-ordered publication
    // convoy serializes on scheduler latency, and throughput collapses by an order of magnitude.
    std::size_t const threads = options.threads ? options.threads
                                                : std::max<std::size_t>(1, fu_logical_cores_count(topology) * 7 / 8);
    if (!topology || !pool || !fu_pool_spawn(topology, pool, threads, fu_caller_inclusive_k)) {
        fmt::print("Failed to spawn {} threads\n", threads);
        return 1;
    }
    print_machine(topology, fabric, options, threads);

    int exit_code = 1;
    switch (file_dtype) {
    case nk_f32_k: exit_code = open_and_run<nk::f32_t>(options, pool); break;
    case nk_bf16_k: exit_code = open_and_run<nk::bf16_t>(options, pool); break;
    case nk_i8_k: exit_code = open_and_run<nk::i8_t>(options, pool); break;
    case nk_u8_k: exit_code = open_and_run<nk::u8_t>(options, pool); break;
    default: fmt::print("Unsupported dataset dtype\n"); break;
    }

    fu_pool_delete(pool);
    fu_fabric_delete(fabric);
    fu_topology_delete(topology);
    return exit_code;
}
