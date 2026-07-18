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
 *  Huge pages are provisioned outside the process, before it starts; `--huge-pages` then
 *  declares the expectation, which the harness logs and enforces - a mismatch fails the
 *  run instead of silently benchmarking base pages:
 *
 *  @code{.sh}
 *  sudo sysctl vm.nr_hugepages=16384        # 32 GB of 2 MB pages, split across the domains
 *  inner_volatile_index_bench_cpp --huge-pages 16384 \
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
#include <algorithm> // `std::nth_element`
#include <atomic>    // `std::atomic`
#include <chrono>    // `std::chrono::steady_clock`
#include <cmath>     // `std::abs`
#include <cstdint>   // `std::uint32_t`
#include <memory>    // `std::unique_ptr`
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
#include "test.hpp"

namespace fu = ashvardanian::forkunion;
namespace nk = ashvardanian::numkong;

// The bench lives in the project's namespace, like the test suites - its types and helpers
// are project vocabulary, so no alias or qualification noise is needed.
namespace unum::usearch {

#pragma region Options and Styling

/** @brief One named capability ceiling - the row shape the compute and schedule tables
 *      share, so their prefix-masks, name lookups, and prints are one machinery. */
template <typename capability_type_>
struct capability_choice {
    char const* name;
    capability_type_ bit;
};

/** @brief Compute-capability ceilings the kernel chains adapt, in ascending preference per
 *      architecture, named by NumKong's microarchitecture vocabulary and split by its own
 *      target macros. The implicit floor the table never lists is the serial kernels. */
inline constexpr capability_choice<nk_capability_t> compute_choices_k[] = {
#if NK_TARGET_X8664
    {"haswell", nk_cap_haswell_k},
    {"skylake", nk_cap_skylake_k},
    {"icelake", nk_cap_icelake_k},
    {"genoa", nk_cap_genoa_k},
#elif NK_TARGET_ARM64
    {"neon", nk_cap_neon_k},
    {"neonsdot", nk_cap_neonsdot_k},
    {"neonbfdot", nk_cap_neonbfdot_k},
#else
    {"serial", nk_cap_serial_k},
#endif
};
inline constexpr std::size_t compute_choices_count_k = sizeof(compute_choices_k) / sizeof(compute_choices_k[0]);

/** @brief Scheduling ceilings the harness instantiates waiters for, in ascending strength
 *      per architecture, named by ForkUnion's own instruction vocabulary and split by its
 *      own detection macros. The implicit floor is the STL yield, like serial for compute;
 *      Arm's `WFET` has no compile-time feature macro, so its row - like every row - is
 *      gated by the runtime capability mask at dispatch. */
inline constexpr capability_choice<fu_capabilities_t> schedule_choices_k[] = {
#if FU_DETECT_ARCH_X86_64_
    {"pause", fu_capability_x86_pause_k},
    {"tpause", fu_capability_x86_tpause_k},
#elif FU_DETECT_ARCH_ARM64_
    {"yield", fu_capability_arm64_yield_k},
    {"wfet", fu_capability_arm64_wfet_k},
#elif FU_DETECT_ARCH_RISC5_
    {"pause", fu_capability_risc5_pause_k},
    {"wrs", fu_capability_risc5_wrs_k},
#else
    {"standard", fu_capabilities_unknown_k},
#endif
};
inline constexpr std::size_t schedule_choices_count_k = sizeof(schedule_choices_k) / sizeof(schedule_choices_k[0]);

/** @brief The tables as spans - the currency of the shared ceiling helpers below. */
inline constexpr std::span<capability_choice<nk_capability_t> const> compute_span_k {compute_choices_k};
inline constexpr std::span<capability_choice<fu_capabilities_t> const> schedule_span_k {schedule_choices_k};

/** @brief Everything at or below the @p index-th ceiling of @p choices, as one allow-mask. */
template <typename capability_type_>
constexpr capability_type_ prefix_mask(std::span<capability_choice<capability_type_> const> choices,
                                       std::size_t index) noexcept {
    capability_type_ mask {};
    for (std::size_t choice = 0; choice <= index && choice < choices.size(); ++choice)
        mask = static_cast<capability_type_>(mask | choices[choice].bit);
    return mask;
}

/** @brief Names a single capability bit of @p choices; the unlisted @p floor backs them all. */
template <typename capability_type_>
constexpr char const* bit_name(std::span<capability_choice<capability_type_> const> choices, capability_type_ bit,
                               char const* floor) noexcept {
    for (capability_choice<capability_type_> const& choice : choices)
        if (choice.bit == bit) return choice.name;
    return floor;
}

/** @brief The newest table bit of @p choices inside @p mask, or @p floor when none is. */
template <typename capability_type_>
constexpr capability_type_ highest_bit(std::span<capability_choice<capability_type_> const> choices,
                                       capability_type_ mask, capability_type_ floor) noexcept {
    capability_type_ highest = floor;
    for (capability_choice<capability_type_> const& choice : choices)
        if (mask & choice.bit) highest = choice.bit;
    return highest;
}

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

/** @brief Everything the CLI can configure; zeroes mean "infer from the dataset & machine". */
struct bench_options_t {
    /** @brief The base vectors file, in the RetriEval `.fbin` extension family. */
    std::string vectors_path;
    /** @brief Optional query vectors of the same dtype; absent means self-queries. */
    std::string queries_path;
    /** @brief Optional `.ibin` ground-truth neighbors; absent skips the quality judgment. */
    std::string neighbors_path;
    /** @brief The representation vectors are stored in; unknown means "as the file". */
    nk_dtype_t stored_dtype = nk_dtype_unknown_k;
    /** @brief Allow-mask over NumKong kernel capabilities, intersected with the runtime
     *      detection - the native ceiling shape, so new bits need no new plumbing. */
    nk_capability_t max_compute = nk_cap_any_k;
    /** @brief Allow-mask over ForkUnion scheduling capabilities, forwarded verbatim to
     *      @c fu_pool_new and mirrored by the waiter type the index instantiates. */
    fu_capabilities_t max_schedule = fu_capabilities_all_k;
    /** @brief The distance kernel driving both construction & search. */
    nk_kernel_kind_t kernel = nk_kernel_angular_k;
    /** @brief Engaged memory domains: empty is one machine-wide heap index, a single entry
     *      pins memory & threads to it, and several build one index per listed domain. */
    std::vector<fu_memory_domain_id_t> memory_domains;
    /** @brief Worker threads; zero takes 7/8 of the available logical cores. */
    std::size_t threads = 0;
    /** @brief Staging trampoline rows shared by all threads; zero takes 4 × threads. */
    std::size_t batch = 0;
    /** @brief Neighbors-list width per node; must match a `connectivity_t` tiling. */
    std::size_t connectivity = 32;
    /** @brief Insertion candidate pool. Calibrated on SIFT 10M with @c expansion_search to
     *      clear 99% recall@10 with margin: 128/192 lands at 99.3%, and the leaner 96/192
     *      rides the line at 99.1% with ~30% faster ingestion. */
    std::size_t expansion_add = 128;
    /** @brief Search candidate pool; see @c expansion_add for the calibration. */
    std::size_t expansion_search = 192;
    /** @brief Matches per query to return & judge. */
    std::size_t top_k = 10;
    /** @brief Cap on ingested rows; zero ingests the whole file. */
    std::size_t max_vectors = 0;
    /** @brief Huge pages the launcher provisioned before starting the process; zero expects
     *      none, any other value is enforced - a run that silently fell back to base pages
     *      must never masquerade as a huge-page result. */
    std::size_t huge_pages = 0;
    /** @brief Print the manual and exit. */
    bool help = false;
};

#pragma endregion

#pragma region Typed Runs

/** @brief Telemetry sums each thread accumulates privately, padded away from its neighbors. */
struct alignas(64) telemetry_lane_t {
    /** @brief Frontier candidates whose neighbor lists were expanded. */
    std::size_t visited_members = 0;
    /** @brief Total distance computations, padding lanes excluded. */
    std::size_t computed_distances = 0;
    /** @brief Subset of @c computed_distances spent in diversity selection. */
    std::size_t computed_distances_in_refines = 0;
    /** @brief Subset of @c computed_distances spent re-selecting while back-linking. */
    std::size_t computed_distances_in_reverse_refines = 0;
    /** @brief Starvation restarts - a graph-connectivity health signal. */
    std::size_t restarts = 0;
};

/**
 *  @brief The index-side state one fully-typed benchmark run shares between its phases:
 *      the indexes - one per engaged memory domain - their per-thread traversal contexts,
 *      and the recall bookkeeping.
 *
 *  The dataset never appears here - the data side lives in @c streamed_dataset, which
 *  bridges the file's scalar into the stored one vector-by-vector. In the default and
 *  pinned placements exactly one index exists; under `--memory-domain all` every memory
 *  domain owns one, every prong routes to its local index through
 *  @c index_of_compute_domain, and searches gather through @c merge_into.
 */
template <nk_dtype_t stored_dtype_, nk_kernel_kind_t kernel_, nk_capability_t capability_, typename wait_type_>
struct bench_context {
    using allocator_t = domain_or_default_allocator<std::byte>;
    using index_t =
        inner_volatile_index<stored_dtype_, kernel_, capability_, wait_type_, allocator_t, allocator_t, allocator_t>;
    using scalar_t = typename index_t::scalar_t;
    using small_size_t = typename index_t::small_size_t;
    using traversal_t = typename index_t::inner_traversal_context_t;
    using vector_view_t = typename index_t::vector_view_t;
    using vectors_view_t = typename index_t::vectors_view_t;
    using distance_t = typename index_t::distance_t;

    static constexpr nk_kernel_kind_t kernel_kind_k = kernel_;

    bench_options_t const& options;
    /** @brief One index per engaged memory domain; the indirection buys address stability,
     *      as an index holds atomics and never moves. */
    std::vector<std::unique_ptr<index_t>> indexes;
    /** @brief The @b [indexes,rows] recall bookkeeping: dense node id → global dataset row. */
    std::vector<std::vector<std::uint32_t>> id_to_row;
    /** @brief Routes every prong to the index local to its compute domain. */
    std::vector<std::size_t> index_of_compute_domain;
    /** @brief The @b [queries,depth] expected neighbors; empty when absent or truncated
     *      by `--max-vectors`. */
    nk::matrix_view<std::uint32_t> ground_truth;
    /** @brief One graph-walking context per pool thread, built by that thread itself with
     *      its local memory domain's allocator. */
    std::vector<std::unique_ptr<traversal_t>> traversals;
    /** @brief Per-thread telemetry sums, padded apart to stay contention-free. */
    std::vector<telemetry_lane_t> lanes;
    /** @brief The @b [queries,top_k] merged global rows of every query's gathered hits. */
    nk::matrix<std::uint32_t> hit_rows;
    /** @brief The @b [queries,top_k] distances matching @c hit_rows entry-for-entry. */
    nk::matrix<distance_t> hit_distances;
    /** @brief Merged hits per query, sizing each row for the post-clock judgment. */
    std::vector<std::uint32_t> hit_counts;
    /** @brief Per-query end-to-end lookup latencies, in microseconds. */
    std::vector<float> latencies_us;
    /** @brief Tasks that returned an error in either phase; any non-zero fails the run. */
    std::atomic<std::size_t> failed_tasks {0};

    bench_context(bench_options_t const& options) noexcept : options(options) {}
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
struct streamed_dataset {
    using bench_t = bench_context_type_;
    using scalar_t = typename bench_context_type_::scalar_t;
    using vector_view_t = typename bench_context_type_::vector_view_t;
    static constexpr bool zero_copy_k = std::is_same_v<dataset_scalar_type_, scalar_t>;

    /** @brief Float sources downcast into integer targets only after a global symmetric
     *      rescale into the target's range - a raw saturating cast collapses the value
     *      distribution, degenerates the distances, and loops the graph traversal.
     *      NumKong's own reflection drives both the classification and the range. */
    static constexpr bool scaled_k = !zero_copy_k && !nk::is_integral_dtype<dataset_scalar_type_>() &&
                                     nk::is_integral_dtype<scalar_t>();

    /** @brief The largest magnitude one stored dimension can carry, in the stored dtype. */
    static constexpr float target_amplitude_k = dimension_amplitude<scalar_t>();

    /** @brief The index-side state both phases feed - punned through the pool callbacks. */
    bench_context_type_& bench;
    /** @brief The @b [rows,dimensions] mapped source rows, in the file's own scalar. */
    nk::matrix_view<dataset_scalar_type_> vectors;
    /** @brief The @b [queries,dimensions] mapped query rows - or a prefix of @c vectors. */
    nk::matrix_view<dataset_scalar_type_> queries;
    /** @brief The @b [threads,rows_per_thread,dimensions] staging trampoline planes. */
    nk::tensor_span<scalar_t, 3> staging;
    /** @brief The @b [threads,rows_per_thread,dimensions] `f32` rescaling planes; only
     *      set when @c scaled_k. */
    nk::tensor_span<nk::f32_t, 3> rescaling;
    /** @brief The global symmetric factor applied before integer downcasts. */
    float scale = 1.0f;

    /** @brief Converts one source row into the calling thread's own staging plane and views
     *      it back; @p task rotates the row within the plane, so in-flight tasks never clash. */
    vector_view_t stage(nk::vector_view<dataset_scalar_type_> source, std::size_t thread, std::size_t task) noexcept {
        std::size_t const dimensions = vectors.extent(1);
        std::size_t const rotation = task % staging.extent(1);
        auto landing = staging.row(thread).row(rotation).as_vector();
        if constexpr (scaled_k) {
            // One SIMD `nk::scale` affine pass replaces a scalar multiply loop; an f32
            // source feeds it straight from the mapping, skipping the widening cast.
            auto rescaled = rescaling.row(thread).row(rotation).as_vector();
            nk_f32_t const alpha = scale;
            nk_f32_t const beta = 0.0f;
            if constexpr (std::is_same_v<dataset_scalar_type_, nk::f32_t>)
                nk::scale(source.data(), dimensions, &alpha, &beta, rescaled.data());
            else {
                nk::cast(source, rescaled);
                nk::scale(rescaled.data(), dimensions, &alpha, &beta, rescaled.data());
            }
            nk::cast(rescaled.data(), dimensions, landing.data());
        }
        else { nk::cast(source, landing); }
        return vector_view_t {landing.data(), dimensions};
    }
};

/** @brief Sums the per-thread telemetry after a phase; the lanes stay untouched. */
static telemetry_lane_t summed_lanes(std::vector<telemetry_lane_t> const& lanes) noexcept {
    telemetry_lane_t sums;
    for (telemetry_lane_t const& lane : lanes) {
        sums.visited_members += lane.visited_members;
        sums.computed_distances += lane.computed_distances;
        sums.computed_distances_in_refines += lane.computed_distances_in_refines;
        sums.computed_distances_in_reverse_refines += lane.computed_distances_in_reverse_refines;
        sums.restarts += lane.restarts;
    }
    return sums;
}

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

    connectivity_t connectivity = connectivity_unknown_k;
    if (bench.options.connectivity == 16) connectivity = connectivity_16_k;
    else if (bench.options.connectivity == 32) connectivity = connectivity_32_k;
    else if (bench.options.connectivity == 128) connectivity = connectivity_128_k;
    else {
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

    bench.indexes.reserve(indexes_count);
    bench.id_to_row.resize(indexes_count);
    for (std::size_t slot = 0; slot != indexes_count; ++slot) {
        fu_memory_domain_id_t const domain = memory_domains.empty() ? -1 : memory_domains[slot];
        bench.indexes.push_back(std::make_unique<index_t>(allocator_t {domain}, allocator_t {domain}));
        auto reserved = bench.indexes.back()->reserve({rows, connectivity, dimensions});
        if (reserved.status != status_t::success_k) {
            fmt::print("Failed to reserve {} × {}d\n", rows, dimensions);
            return false;
        }
        bench.id_to_row[slot].resize(rows);
    }

    // Every pool thread constructs its own traversal context, so the scratch - results,
    // frontier, visited set - lands on the thread's local memory domain, not the caller's.
    bench.traversals.resize(threads);
    auto traversals_init = [](fu_lambda_context_t punned, std::size_t thread, std::size_t compute_domain) noexcept {
        auto& bench = *static_cast<bench_t*>(punned);
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
    bench.hit_rows = nk::matrix<std::uint32_t>::try_zeros({streamed.queries.extent(0), bench.options.top_k});
    bench.hit_distances = nk::matrix<typename bench_t::distance_t>::try_zeros(
        {streamed.queries.extent(0), bench.options.top_k});
    if (bench.hit_rows.empty() || bench.hit_distances.empty()) {
        fmt::print("Failed to allocate the per-query hit matrices\n");
        return false;
    }
    bench.hit_counts.resize(streamed.queries.extent(0));
    bench.latencies_us.resize(streamed.queries.extent(0));
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
    vector_view_t vector;
    if constexpr (streamed_dataset_type_::zero_copy_k) vector = streamed.vectors.row(task).as_vector();
    else vector = streamed.stage(streamed.vectors.row(task).as_vector(), thread, task);

    std::size_t const slot = bench.index_of_compute_domain[compute_domain];
    index_t& index = *bench.indexes[slot];
    inner_node_id_t local_id = 0;
    vectors_view_t const one_row {vector.data(), 1, streamed.vectors.extent(1)};
    auto inserted = index.insert(one_row, {&local_id, std::size_t {1}}, *bench.traversals[thread], config);
    if (!inserted) {
        bench.failed_tasks.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    telemetry_lane_t& lane = bench.lanes[thread];
    lane.visited_members += inserted.visited_members;
    lane.computed_distances += inserted.computed_distances;
    lane.computed_distances_in_refines += inserted.computed_distances_in_refines;
    lane.computed_distances_in_reverse_refines += inserted.computed_distances_in_reverse_refines;
    lane.restarts += inserted.restarts;
    bench.id_to_row[slot][local_id] = static_cast<std::uint32_t>(task);
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
    telemetry_lane_t const sums = summed_lanes(bench.lanes);

    using index_t = typename streamed_dataset_type_::bench_t::index_t;
    double const refine_share = sums.computed_distances ? 100.0 * double(sums.computed_distances_in_refines) /
                                                              double(sums.computed_distances)
                                                        : 0.0;
    double const reverse_share = sums.computed_distances ? 100.0 * double(sums.computed_distances_in_reverse_refines) /
                                                               double(sums.computed_distances)
                                                         : 0.0;
    fmt::print(header_style(), "\nInsertion\n");
    fmt::print("  shape: {} × {}d\n", rows, streamed.vectors.extent(1));
    fmt::print("  kernels: {} · {} · {} kit\n", scalar_t::dtype_name(),
               kernel_name(streamed_dataset_type_::bench_t::kernel_kind_k),
               bit_name(compute_span_k, index_t::picked_capability_k, "serial"));
    if (index_t::picked_capability_k == nk_cap_serial_k &&
        (bench.options.max_compute & nk_capabilities_available() & ~nk_cap_serial_k))
        fmt::print(verdict_style(false), "  ⚠ scalar fallback: no SIMD kit is adapted for this dtype & metric\n");
    fmt::print("  threads: {}\n", fu_pool_threads_count(pool));
    if (bench.indexes.size() > 1) fmt::print("  indexes: {}, one per memory domain\n", bench.indexes.size());
    fmt::print("  wall: {:.2f} s\n", seconds);
    fmt::print("  throughput: {}/s\n", humanize(double(rows) / seconds));
    fmt::print("  distances: {} ({:.1f}% refines, {:.1f}% reverse)\n", humanize(double(sums.computed_distances)),
               refine_share, reverse_share);
    fmt::print("  visited: {}\n", humanize(double(sums.visited_members)));
    fmt::print("  restarts: ");
    fmt::print(verdict_style(sums.restarts < rows / 100 + 1), "{}\n", sums.restarts);
    if (std::size_t resident = peak_resident_bytes(); resident)
        fmt::print("  peak resident: {:.2f} GB\n", double(resident) / 1e9);
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
    vector_view_t query_vector;
    if constexpr (streamed_dataset_type_::zero_copy_k) query_vector = streamed.queries.row(query).as_vector();
    else query_vector = streamed.stage(streamed.queries.row(query).as_vector(), thread, query);

    std::uint32_t* const merged_rows = bench.hit_rows.row(query).data();
    auto* const merged_distances = bench.hit_distances.row(query).data();
    std::size_t merged_count = 0;

    auto started = clock_t::now();
    for (std::size_t slot = 0; slot != bench.indexes.size(); ++slot) {
        index_t& index = *bench.indexes[slot];
        auto found = index.search(query_vector, *bench.traversals[thread], config);
        if (!found) {
            bench.failed_tasks.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        lane.visited_members += found.visited_members;
        lane.computed_distances += found.computed_distances;
        lane.restarts += found.restarts;
        auto const& results = bench.traversals[thread]->results;
        merged_count = merge_search_results(to_span(results), bench.id_to_row[slot],
                                            std::span {merged_rows, bench.options.top_k},
                                            std::span {merged_distances, bench.options.top_k}, merged_count);
    }
    bench.latencies_us[query] = std::chrono::duration<float, std::micro>(clock_t::now() - started).count();
    bench.hit_counts[query] = static_cast<std::uint32_t>(merged_count);
}

template <typename streamed_dataset_type_>
bool run_search(streamed_dataset_type_& streamed, fu_pool_t pool) {
    using clock_t = std::chrono::steady_clock;
    auto& bench = streamed.bench;
    std::size_t const query_count = streamed.queries.extent(0);
    for (telemetry_lane_t& lane : bench.lanes) lane = telemetry_lane_t {};

    auto started = clock_t::now();
    fu_pool_for_n(pool, query_count, &search_callback<streamed_dataset_type_>, &streamed);
    double const seconds = std::chrono::duration<double>(clock_t::now() - started).count();

    if (std::size_t failures = bench.failed_tasks.load(); failures) {
        fmt::print("Search failed on {} queries\n", failures);
        return false;
    }
    telemetry_lane_t const sums = summed_lanes(bench.lanes);

    auto percentile = [&](double share) {
        std::size_t const rank = std::min(query_count - 1, std::size_t(share * double(query_count)));
        std::nth_element(bench.latencies_us.begin(), bench.latencies_us.begin() + rank, bench.latencies_us.end());
        return bench.latencies_us[rank];
    };

    fmt::print(header_style(), "\nSearch\n");
    fmt::print("  queries: {} at top-{}\n", query_count, bench.options.top_k);
    fmt::print("  threads: {}\n", fu_pool_threads_count(pool));
    fmt::print("  wall: {:.2f} s\n", seconds);
    fmt::print("  throughput: {} QPS\n", humanize(double(query_count) / seconds));
    fmt::print("  latency: p50 {:.0f} µs · p90 {:.0f} µs · p99 {:.0f} µs\n", percentile(0.50), percentile(0.90),
               percentile(0.99));
    fmt::print("  distances/query: {}\n", humanize(double(sums.computed_distances) / double(query_count)));
    fmt::print("  visited/query: {}\n", humanize(double(sums.visited_members) / double(query_count)));
    fmt::print("  restarts: {}\n", sums.restarts);
    // Judgment runs after the clock stopped, so QPS never depends on how much we measure.
    // One sweep per query produces every verdict; the discount tables are built once.
    if (!bench.ground_truth.empty()) {
        std::size_t const judged_depth = std::min(bench.options.top_k, bench.ground_truth.extent(1));
        std::vector<std::uint32_t> scratch(judged_depth);
        std::vector<double> discounts(bench.options.top_k), ideal(bench.options.top_k);
        fill_discounts(discounts, ideal);

        std::size_t found_at_1 = 0, found_at_k = 0, shared = 0;
        double gain = 0;
        for (std::size_t query = 0; query != query_count; ++query) {
            auto expected = bench.ground_truth.row(query).as_vector();
            auto const returned = to_span(bench.hit_rows.row(query).as_vector()).first(bench.hit_counts[query]);
            found_at_1 += !returned.empty() && returned.front() == expected[0];
            auto verdict = judge_ranking(std::span<std::uint32_t const> {returned},
                                         to_span(expected).first(judged_depth), std::span<std::uint32_t> {scratch},
                                         std::span<double const> {discounts});
            found_at_k += verdict.found_nearest;
            shared += verdict.overlap;
            gain += verdict.gain;
        }

        double const queries = double(query_count);
        double const recall_at_1 = 100.0 * double(found_at_1) / queries;
        double const recall_at_k = 100.0 * double(found_at_k) / queries;
        double const overlap_at_k = 100.0 * double(shared) / (queries * double(judged_depth));
        double const gain_at_k = 100.0 * gain / (queries * ideal[judged_depth - 1]);
        fmt::print("  recall@1: ");
        fmt::print(verdict_style(recall_at_1 >= 90.0), "{:.1f}%\n", recall_at_1);
        fmt::print("  recall@{}: ", judged_depth);
        fmt::print(verdict_style(recall_at_k >= 99.0), "{:.1f}%\n", recall_at_k);
        fmt::print("  overlap@{}: {:.1f}%\n", judged_depth, overlap_at_k);
        fmt::print("  ndcg@{}: {:.1f}%\n", judged_depth, gain_at_k);
    }
    else { fmt::print("  recall: skipped, no ground-truth neighbors provided\n"); }
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
 *  @brief The innermost hop: wires the typed source views and the bounded staging
 *      trampoline into a @c streamed_dataset, and runs both phases.
 *
 *  When the file scalar already matches the stored one, everything is a zero-copy view
 *  into the mappings; otherwise the staging matrix holds just `--batch` rows - typically a
 *  few per thread - and each vector streams through it on its way into the index. The
 *  dataset is never copied whole.
 */
template <typename dataset_scalar_type_, nk_dtype_t stored_dtype_, nk_kernel_kind_t kernel_,
          nk_capability_t capability_, typename wait_type_>
int stage_and_run(bench_launch<dataset_scalar_type_> const& launch) {
    using bench_context_type_ = bench_context<stored_dtype_, kernel_, capability_, wait_type_>;
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

    if (!prepare(streamed, launch.pool, launch.topology)) return 1;
    if (!run_insertion(streamed, launch.pool)) return 1;
    if (!run_search(streamed, launch.pool)) return 1;
    return 0;
}

template <typename dataset_scalar_type_, nk_dtype_t stored_dtype_, nk_capability_t capability_, typename wait_type_>
int run_with_metric(bench_launch<dataset_scalar_type_> const& launch) {
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
int run_with_schedule(bench_launch<dataset_scalar_type_> const& launch) {
    // The newest schedule row the allow-mask admits - intersected with what this machine
    // actually offers, which is what gates Arm's runtime-only `WFET` - picks the waiter;
    // the same mask already reached `fu_pool_new`, so the pool waits the same way.
    fu_capabilities_t const effective = static_cast<fu_capabilities_t>(launch.options.max_schedule &
                                                                       fu_runtime_capabilities());
    std::size_t chosen = schedule_choices_count_k;
    for (std::size_t choice = 0; choice != schedule_choices_count_k; ++choice)
        if (effective & schedule_choices_k[choice].bit) chosen = choice;
    switch (chosen) {
    case 0: return run_with_metric<dataset_scalar_type_, stored_dtype_, capability_, hint_waiter_t>(launch);
    case 1:
        if constexpr (schedule_choices_count_k > 1)
            return run_with_metric<dataset_scalar_type_, stored_dtype_, capability_, monitor_waiter_t>(launch);
        break;
    default: break;
    }
    return run_with_metric<dataset_scalar_type_, stored_dtype_, capability_, yield_waiter_t>(launch);
}

template <typename dataset_scalar_type_, nk_dtype_t stored_dtype_>
int run_with_compute(bench_launch<dataset_scalar_type_> const& launch) {
    // The effective set is the allow-mask intersected with the runtime detection; the
    // newest table row inside it picks the instantiation. The per-architecture tables
    // differ in length, so every case beyond a table's tail is discarded at compile time
    // rather than indexing the choices out of bounds.
    nk_capability_t const effective = launch.options.max_compute & nk_capabilities_available();
    std::size_t chosen = compute_choices_count_k;
    for (std::size_t choice = 0; choice != compute_choices_count_k; ++choice)
        if (effective & compute_choices_k[choice].bit) chosen = choice;
    switch (chosen) {
    case 0: return run_with_schedule<dataset_scalar_type_, stored_dtype_, prefix_mask(compute_span_k, 0)>(launch);
    case 1:
        if constexpr (compute_choices_count_k > 1)
            return run_with_schedule<dataset_scalar_type_, stored_dtype_, prefix_mask(compute_span_k, 1)>(launch);
        break;
    case 2:
        if constexpr (compute_choices_count_k > 2)
            return run_with_schedule<dataset_scalar_type_, stored_dtype_, prefix_mask(compute_span_k, 2)>(launch);
        break;
    case 3:
        if constexpr (compute_choices_count_k > 3)
            return run_with_schedule<dataset_scalar_type_, stored_dtype_, prefix_mask(compute_span_k, 3)>(launch);
        break;
    case 4:
        if constexpr (compute_choices_count_k > 4)
            return run_with_schedule<dataset_scalar_type_, stored_dtype_, prefix_mask(compute_span_k, 4)>(launch);
        break;
    default: break;
    }
    // Past the table's tail sits the one ceiling every machine owns: the serial kernels.
    // This branch serves an explicit `--max-compute serial`, and any CPU with no SIMD bits.
    return run_with_schedule<dataset_scalar_type_, stored_dtype_, nk_cap_serial_k>(launch);
}

template <typename dataset_scalar_type_>
int run_with_stored_dtype(bench_launch<dataset_scalar_type_> const& launch) {
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

/** @brief Prints the machine as ForkUnion sees it: cores, compute & memory domains, and the
 *      NUMA fabric levels, next to the NumKong capability the kernels will actually use. */
static void print_machine(fu_topology_t topology, fu_fabric_t fabric, bench_options_t const& options,
                          std::size_t threads) {
    std::size_t const compute_domains = fu_compute_domains_count(topology);
    std::size_t const memory_domains = fu_memory_domains_count(topology);
    fmt::print(header_style(), "Machine\n");
    fmt::print("  logical cores: {}\n", fu_logical_cores_count(topology));
    fmt::print("  compute domains: {}\n", compute_domains);
    fmt::print("  memory domains: {}\n", memory_domains);
    for (std::size_t domain = 0; domain != memory_domains; ++domain)
        fmt::print("  memory domain {}: fabric level {}, huge pages {:.2f} GB\n", domain,
                   fu_fabric_memory_level_in(fabric, domain), double(fu_volume_huge_pages_in(topology, domain)) / 1e9);
    if (!options.memory_domains.empty()) {
        fmt::print("  engaged memory domains:");
        for (fu_memory_domain_id_t domain : options.memory_domains) fmt::print(" {}", domain);
        fmt::print("\n");
    }
    fmt::print("  threads engaged: {}\n", threads);

    char scheduling_names[256] = {0};
    fu_name_capabilities(fu_runtime_capabilities(), scheduling_names, sizeof(scheduling_names));
    fmt::print("  scheduling detected: {}\n", scheduling_names);

    nk_capability_t const available = nk_capabilities_available();
    fmt::print("  compute detected: serial");
    for (capability_choice<nk_capability_t> const& choice : compute_choices_k)
        if (available & choice.bit) fmt::print(" {}", choice.name);
    fmt::print("\n");
    fmt::print("  compute ceiling: {}\n",
               bit_name(compute_span_k, highest_bit(compute_span_k, options.max_compute & available, nk_cap_serial_k),
                        "serial"));
    fmt::print("  scheduling ceiling: {}\n",
               bit_name(schedule_span_k,
                        highest_bit(schedule_span_k,
                                    static_cast<fu_capabilities_t>(options.max_schedule & fu_runtime_capabilities()),
                                    fu_capabilities_unknown_k),
                        "standard"));
    fmt::print("  kernels requested: {} · {}\n", dtype_name(options.stored_dtype), kernel_name(options.kernel));
}

} // namespace unum::usearch

int main(int argc, char** argv) {
    using namespace unum::usearch;
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
        (clipp::option("--huge-pages") & clipp::value("count", options.huge_pages)) %
            "huge pages provisioned before launch; enforced, so base-page fallbacks fail fast",
        clipp::option("--help").set(options.help) % "print this message");

    if (!clipp::parse(argc, argv, cli) || options.help || options.vectors_path.empty()) {
        for (auto const& section : clipp::make_man_page(cli, "inner_volatile_index_bench_cpp"))
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
        std::size_t named = compute_choices_count_k;
        for (std::size_t choice = 0; choice != compute_choices_count_k; ++choice)
            if (max_compute_name == compute_choices_k[choice].name) named = choice;
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
    if (max_schedule_name == "standard") {
        options.max_schedule = static_cast<fu_capabilities_t>(fu_capabilities_all_k & ~fu_capability_any_yield_k);
    }
    else if (!max_schedule_name.empty()) {
        std::size_t named = schedule_choices_count_k;
        for (std::size_t choice = 0; choice != schedule_choices_count_k; ++choice)
            if (max_schedule_name == schedule_choices_k[choice].name) named = choice;
        if (named == schedule_choices_count_k) {
            fmt::print("Unknown --max-schedule {}; this table offers: standard", max_schedule_name);
            for (capability_choice<fu_capabilities_t> const& choice : schedule_choices_k)
                fmt::print(" {}", choice.name);
            fmt::print("\n");
            return 1;
        }
        options.max_schedule = static_cast<fu_capabilities_t>((fu_capabilities_all_k & ~fu_capability_any_yield_k) |
                                                              prefix_mask(schedule_span_k, named));
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

    int exit_code = 1;
    switch (file_dtype) {
    case nk_f32_k: exit_code = open_and_run<nk::f32_t>(options, pool, topology); break;
    case nk_bf16_k: exit_code = open_and_run<nk::bf16_t>(options, pool, topology); break;
    case nk_i8_k: exit_code = open_and_run<nk::i8_t>(options, pool, topology); break;
    case nk_u8_k: exit_code = open_and_run<nk::u8_t>(options, pool, topology); break;
    default: fmt::print("Unsupported dataset dtype\n"); break;
    }

    fu_pool_delete(pool);
    fu_fabric_delete(fabric);
    fu_topology_delete(topology);
    return exit_code;
}
