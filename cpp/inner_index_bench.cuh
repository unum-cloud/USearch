/**
 *  @brief The backend-neutral benchmarking surface shared by every inner-index category -
 *      volatile in-RAM, GPU-resident, and persisted alike.
 *
 *  @author Ash Vardanian
 *  @file cpp/inner_index_bench.cuh
 *  @date July 22, 2026
 *
 *  Only what every backend genuinely shares lives here: the CLI option set, the RetriEval
 *  checkpoint naming & layout, the dataset streaming trampoline, the telemetry aggregation,
 *  the post-clock result judgment, and the host-machine report. The run loops themselves -
 *  how rows reach an index and queries leave it - are backend property: each translation
 *  unit owns its pipeline and its concrete index instantiations, so this header names no
 *  index and reads no index-specific members, and the same surface compiles under host
 *  compilers and nvcc.
 */
#pragma once

#include <algorithm> // `std::nth_element`, `std::min`
#include <atomic>    // `std::atomic`
#include <cstdint>   // `std::uint32_t`
#include <span>      // `std::span`
#include <string>    // `std::string`
#include <vector>    // `std::vector`

#include <fmt/color.h>
#include <fmt/format.h>

#include <forkunion.h>
#include <forkunion/capabilities.hpp>
#include <numkong/numkong.h>
#include <numkong/numkong.hpp>
#include <stringzilla/stringzilla.hpp>

#include <usearch/serialization.hpp>

#include "bench.hpp"
#include "persisted_dataset.hpp" // `mmap_view_t`, `parse_matrix_header`, `matrix_save`
#include "test.hpp"

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
#if NK_TARGET_X8664_
    {"haswell", nk_cap_haswell_k}, {"skylake", nk_cap_skylake_k},         {"icelake", nk_cap_icelake_k},
    {"genoa", nk_cap_genoa_k},     {"sapphireamx", nk_cap_sapphireamx_k},
#elif NK_TARGET_ARM64_
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
    /** @brief Insertion candidate pool. Calibrated on SIFT 10M against @c expansion_search for
     *      the build-once-search-forever regime: 256/96 clears 99.3% recall@10 at roughly half
     *      the per-query distance work of 128/192, collapsing the p99 latency ~4x - build
     *      compute spent once to keep every future query cheap. */
    std::size_t expansion_add = 256;
    /** @brief Search candidate pool; see @c expansion_add for the calibration. */
    std::size_t expansion_search = 96;
    /** @brief Matches per query to return & judge. */
    std::size_t top_k = 10;
    /** @brief Search every ingested vector for itself instead of the query file - the
     *      calibration-free navigability probe, with a search phase as long as the ingestion. */
    bool self_recall = false;
    /** @brief Checkpoint name to serialize every domain index into after ingestion - resolved
     *      through @ref checkpoint_paths_of, so the file name stamps the dtype, kernel, and
     *      connectivity, and any backend matching them restores it. */
    std::string save_name;
    /** @brief Checkpoint name to restore domain indexes from before ingestion; pair it with
     *      `--skip-vectors`, so only the unindexed topping streams in. */
    std::string load_name;
    /** @brief Directory the checkpoint blobs land in. The tmpfs-backed default suits small
     *      runs; point 100M-scale runs at a real disk. */
    std::string checkpoints_dir = "/tmp";
    /** @brief Rows the loaded indexes already cover; ingestion resumes right after them. */
    std::size_t skip_vectors = 0;
    /** @brief Cap on ingested rows; zero ingests the whole file. */
    std::size_t max_vectors = 0;
    /** @brief Huge pages the launcher provisioned before starting the process; zero expects
     *      none, any other value is enforced - a run that silently fell back to base pages
     *      must never masquerade as a huge-page result. */
    std::size_t huge_pages = 0;
    /** @brief Print the manual and exit. */
    bool help = false;
};

/** @brief The three matrix files one checkpointed slot spans - each a RetriEval-layout dump. */
struct checkpoint_paths_t {
    std::string vectors;
    std::string neighbors;
    std::string rows;
};

/** @brief Resolves a user-facing checkpoint name into one slot's matrix-file paths. The
 *      stamp names exactly what the payload depends on - stored dtype, distance kernel, and
 *      connectivity - so any backend instantiating the same combination computes the same
 *      names: a graph built on the GPU restores on the CPU and back. A mismatched
 *      combination resolves to different file names and simply finds nothing; geometry
 *      inside the files is re-validated at load either way. */
template <typename index_type_>
static checkpoint_paths_t checkpoint_paths_of(bench_options_t const& options, std::string_view name, std::size_t slot,
                                              std::size_t slot_count) {
    std::string const prefix = fmt::format("{}/{}-{}-{}-c{}-{}-{}", options.checkpoints_dir, name,
                                           index_type_::scalar_t::dtype_name(), kernel_name(index_type_::kernel_kind_k),
                                           options.connectivity, slot, slot_count);
    return {fmt::format("{}.vectors.{}bin", prefix, index_type_::scalar_t::dtype_name()), //
            prefix + ".neighbors.ibin", prefix + ".rows.ibin"};
}

/** @brief Maps the CLI neighbor-list degree onto the closed @c connectivity_t tiling set. */
inline connectivity_t connectivity_of(std::size_t degree) noexcept {
    if (degree == 16) return connectivity_16_k;
    if (degree == 32) return connectivity_32_k;
    if (degree == 128) return connectivity_128_k;
    return connectivity_unknown_k;
}

/**
 *  @brief Restores one index slot from its three RetriEval-layout matrix dumps: maps the
 *      files, validates their headers against the run's frozen geometry, streams the
 *      vectors & neighbors through the index's own @c load, and refills the id → row map
 *      the recall bookkeeping needs. Backend-neutral: only the conformance surface every
 *      index category exposes is touched. Prints its own diagnostics.
 *
 *  @param[in] options Supplies @c load_name, @c checkpoints_dir, and the @c connectivity
 *      the neighbor matrix must agree with.
 *  @param[inout] index A virgin index already @c reserve() -d to the run's geometry - the
 *      load validates the files against that frozen shape.
 *  @param[in] dimensions Row width the vectors matrix must match; not recoverable from
 *      @p index, whose facade may not expose it.
 *  @param[out] id_to_row Landing for the restored id → dataset-row map; must span at least
 *      as many rows as the checkpoint holds.
 *  @param[in] slot This index's position among @p slot_count sibling slots - both are baked
 *      into the file names, so a multi-domain save only restores into the same placement.
 *  @return True on a fully restored slot; false after printing what failed.
 */
template <typename index_type_>
static bool load_checkpoint(bench_options_t const& options, index_type_& index, std::size_t dimensions,
                            std::span<std::uint32_t> id_to_row, std::size_t slot, std::size_t slot_count) {
    using scalar_t = typename index_type_::scalar_t;
    checkpoint_paths_t const paths = checkpoint_paths_of<index_type_>(options, options.load_name, slot, slot_count);
    auto mapped_vectors = mmap_view_t::try_map(paths.vectors.c_str());
    auto mapped_neighbors = mmap_view_t::try_map(paths.neighbors.c_str());
    auto mapped_rows = mmap_view_t::try_map(paths.rows.c_str());
    if (mapped_vectors.status != status_t::success_k || mapped_neighbors.status != status_t::success_k ||
        mapped_rows.status != status_t::success_k) {
        fmt::print("Failed to map the {} checkpoint for slot {}\n", options.load_name, slot);
        return false;
    }

    auto const vectors_extents = parse_matrix_header<scalar_t, std::uint32_t>(mapped_vectors.value);
    auto const neighbors_extents = parse_matrix_header<std::uint32_t, std::uint32_t>(mapped_neighbors.value);
    auto const map_extents = parse_matrix_header<std::uint32_t, std::uint32_t>(mapped_rows.value);
    bool const parsed = vectors_extents.status == status_t::success_k &&
                        neighbors_extents.status == status_t::success_k && map_extents.status == status_t::success_k;
    std::size_t const loaded_nodes = vectors_extents.value.rows;
    if (!parsed || vectors_extents.value.columns != dimensions ||
        neighbors_extents.value.columns != options.connectivity || map_extents.value.columns != 1 ||
        neighbors_extents.value.rows != loaded_nodes || map_extents.value.rows != loaded_nodes ||
        loaded_nodes > id_to_row.size()) {
        fmt::print("Checkpoint {} disagrees with the run's geometry\n", options.load_name);
        return false;
    }

    nk::matrix_view<scalar_t> const vectors_view = view_matrix_payload<scalar_t, std::uint32_t>(
        mapped_vectors.value, loaded_nodes, dimensions);
    nk::matrix_view<std::uint32_t> const neighbors_view = view_matrix_payload<std::uint32_t, std::uint32_t>(
        mapped_neighbors.value, loaded_nodes, options.connectivity);
    status_t const loaded = index.load(vectors_view, neighbors_view, synchronization_t::exclusive_k);
    if (loaded != status_t::success_k) {
        fmt::print("Failed to restore {}: status {}\n", paths.vectors, int(loaded));
        return false;
    }
    nk::matrix_view<std::uint32_t> const rows_view = view_matrix_payload<std::uint32_t, std::uint32_t>(
        mapped_rows.value, loaded_nodes, 1);
    std::copy_n(rows_view.data(), loaded_nodes, id_to_row.data());
    return true;
}

/**
 *  @brief Serializes one index slot as three RetriEval-layout matrices - vectors, neighbors,
 *      and the id → row map - through the index's own @c saved_shape / @c save. Backend-
 *      neutral, like @ref load_checkpoint. Prints its own diagnostics.
 *
 *  @param[in] options Supplies @c save_name, @c checkpoints_dir, and the @c connectivity
 *      stamped into the file names.
 *  @param[inout] index The built index; @c save drains it under @c immutable_k, so no
 *      insertions may be in flight.
 *  @param[in] id_to_row The id → dataset-row map filled during ingestion; its first
 *      @c saved_shape().nodes entries ride along as the third matrix.
 *  @param[in] slot This index's position among @p slot_count sibling slots.
 *  @return The serialized byte volume, or zero on failure - a real checkpoint never is.
 */
template <typename index_type_>
static std::size_t save_checkpoint(bench_options_t const& options, index_type_& index,
                                   std::span<std::uint32_t> id_to_row, std::size_t slot, std::size_t slot_count) {
    using scalar_t = typename index_type_::scalar_t;
    using node_id_t = typename index_type_::node_id_t;
    typename index_type_::saved_shape_t const shape = index.saved_shape();
    auto vectors = nk::matrix<scalar_t>::try_zeros({shape.nodes, shape.dimensions});
    auto neighbors = nk::matrix<node_id_t>::try_zeros({shape.nodes, shape.connectivity});
    if (vectors.empty() || neighbors.empty()) {
        fmt::print("Failed to stage {} rows for serialization\n", shape.nodes);
        return 0;
    }
    status_t const saved = index.save(vectors.span(), neighbors.span(), synchronization_t::immutable_k);
    if (saved != status_t::success_k) {
        fmt::print("Failed to serialize index {}: status {}\n", slot, int(saved));
        return 0;
    }

    checkpoint_paths_t const paths = checkpoint_paths_of<index_type_>(options, options.save_name, slot, slot_count);
    nk::matrix_view<std::uint32_t> const rows_map_view {id_to_row.data(), {shape.nodes, 1}};
    bool const written = matrix_save<scalar_t>(paths.vectors.c_str(), vectors.view()) == status_t::success_k &&
                         matrix_save<node_id_t>(paths.neighbors.c_str(), neighbors.view()) == status_t::success_k &&
                         matrix_save<std::uint32_t>(paths.rows.c_str(), rows_map_view) == status_t::success_k;
    if (!written) {
        fmt::print("Failed to write the {} checkpoint for slot {}\n", options.save_name, slot);
        return 0;
    }
    return shape.nodes * (static_cast<std::size_t>(vectors.view().stride_bytes(0)) + //
                          shape.connectivity * sizeof(node_id_t) + sizeof(std::uint32_t));
}

#pragma endregion

#pragma region Telemetry and Judgment

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

/** @brief Sums the per-thread telemetry after a phase; the lanes stay untouched. */
inline telemetry_lane_t summed_lanes(std::vector<telemetry_lane_t> const& lanes) noexcept {
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

/** @brief The backend-neutral judgment state every backend's phases fill: the id → row
 *      bookkeeping, the merged hits with their distances & latencies, and the failure tally.
 *      Backend contexts embed one and add their own machinery around it. */
struct bench_results_t {
    /** @brief The @b [indexes,rows] recall bookkeeping: dense node id → global dataset row. */
    std::vector<std::vector<std::uint32_t>> id_to_row;
    /** @brief The @b [queries,depth] expected neighbors; empty when absent or truncated
     *      by `--max-vectors`. */
    nk::matrix_view<std::uint32_t> ground_truth;
    /** @brief The @b [queries,top_k] merged global rows of every query's gathered hits. */
    nk::matrix<std::uint32_t> hit_rows;
    /** @brief The @b [queries,top_k] distances matching @c hit_rows entry-for-entry. */
    nk::matrix<inner_distance_t> hit_distances;
    /** @brief Merged hits per query, sizing each row for the post-clock judgment. */
    std::vector<std::uint32_t> hit_counts;
    /** @brief Per-query end-to-end lookup latencies, in microseconds. */
    std::vector<float> latencies_us;
    /** @brief Tasks that returned an error in either phase; any non-zero fails the run. */
    std::atomic<std::size_t> failed_tasks {0};
};

/** @brief Prints the search phase's wall clock, throughput, latency percentiles & traversal
 *      telemetry - the backend-neutral half of the report; @p latencies_us reorders in place. */
inline void report_search_metrics(bench_options_t const& options, std::size_t query_count, double seconds,
                                  std::size_t threads, telemetry_lane_t const& sums, std::span<float> latencies_us) {
    auto percentile = [&](double share) {
        std::size_t const rank = std::min(query_count - 1, std::size_t(share * double(query_count)));
        std::nth_element(latencies_us.begin(), latencies_us.begin() + rank, latencies_us.end());
        return latencies_us[rank];
    };
    fmt::print(header_style(), "\nSearch\n");
    fmt::print("  queries: {} at top-{}{}\n", query_count, options.top_k, options.self_recall ? " · self-recall" : "");
    fmt::print("  threads: {}\n", threads);
    fmt::print("  wall: {:.2f} s\n", seconds);
    fmt::print("  throughput: {} QPS\n", humanize(double(query_count) / seconds));
    fmt::print("  latency: p50 {:.0f} µs · p90 {:.0f} µs · p99 {:.0f} µs\n", percentile(0.50), percentile(0.90),
               percentile(0.99));
    fmt::print("  distances/query: {}\n", humanize(double(sums.computed_distances) / double(query_count)));
    fmt::print("  visited/query: {}\n", humanize(double(sums.visited_members) / double(query_count)));
    fmt::print("  restarts: {}\n", sums.restarts);
}

/** @brief Judges the merged hits after the clocks stopped - self-recall against own rows, or
 *      recall/overlap/ndcg against the ground truth - one sweep per query either way, so QPS
 *      never depends on how much we measure. The discount tables are built once. */
inline void judge_hits(bench_options_t const& options, bench_results_t& results, std::size_t query_count) {
    if (options.self_recall) {
        std::size_t found_at_1 = 0, found_at_k = 0;
        for (std::size_t query = 0; query != query_count; ++query) {
            auto const returned = to_span(results.hit_rows.row(query).as_vector()).first(results.hit_counts[query]);
            std::uint32_t const own_row = static_cast<std::uint32_t>(query);
            found_at_1 += !returned.empty() && returned.front() == own_row;
            bool within = false;
            for (std::uint32_t row : returned) within |= row == own_row;
            found_at_k += within;
        }
        double const queries = double(query_count);
        double const self_at_1 = 100.0 * double(found_at_1) / queries;
        double const self_at_k = 100.0 * double(found_at_k) / queries;
        fmt::print("  self-recall@1: ");
        fmt::print(verdict_style(self_at_1 >= 99.0), "{:.2f}%\n", self_at_1);
        fmt::print("  self-recall@{}: ", options.top_k);
        fmt::print(verdict_style(self_at_k >= 99.0), "{:.2f}%\n", self_at_k);
    }
    else if (!results.ground_truth.empty()) {
        std::size_t const judged_depth = std::min(options.top_k, results.ground_truth.extent(1));
        std::vector<std::uint32_t> scratch(judged_depth);
        std::vector<double> discounts(options.top_k), ideal(options.top_k);
        fill_discounts(discounts, ideal);

        std::size_t found_at_1 = 0, found_at_k = 0, shared = 0;
        double gain = 0;
        for (std::size_t query = 0; query != query_count; ++query) {
            auto expected = results.ground_truth.row(query).as_vector();
            auto const returned = to_span(results.hit_rows.row(query).as_vector()).first(results.hit_counts[query]);
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
}

#pragma endregion

#pragma region Dataset Streaming

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

#pragma endregion

/** @brief Prints the machine as ForkUnion sees it: cores, compute & memory domains, and the
 *      NUMA fabric levels, next to the NumKong capability the kernels will actually use. */
inline void print_machine(fu_topology_t topology, fu_fabric_t fabric, bench_options_t const& options,
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
