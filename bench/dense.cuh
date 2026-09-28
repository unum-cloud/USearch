/**
 *  @file bench/dense.cuh
 *  @author Ash Vardanian
 *  @date July 22, 2026
 *  @brief The backend-neutral benchmarking surface shared by every dense-index residency - volatile
 *      in-RAM, GPU-resident, and persisted alike.
 *
 *  Only what every backend genuinely shares lives here: the CLI option set, the RetriEval
 *  checkpoint naming and layout, the dataset streaming trampoline, the telemetry aggregation,
 *  the post-clock result judgment, and the host-machine report. The run loops are backend
 *  property, so this header names no index and reads no index-specific members - the same
 *  surface compiles under host compilers and nvcc.
 */
#pragma once

#include <cmath>   // `std::abs` for the rescale amplitude
#include <cstring> // `std::memcpy` - the checkpoint runs

#include <algorithm>   // `std::nth_element`, `std::min`
#include <atomic>      // `std::atomic`
#include <ranges>      // `std::views::iota`, `std::views::filter`, `std::views::transform`
#include <span>        // `std::span`
#include <string>      // `std::string`
#include <type_traits> // `std::integral_constant` for the dispatch chains
#include <vector>      // `std::vector`

#include <fmt/color.h>
#include <fmt/format.h>
#include <fmt/ranges.h> // `fmt::join`

#include <forkunion.h>
#include <forkunion/capabilities.hpp>
#include <numkong/numkong.h>
#include <numkong/numkong.hpp>
#include <stringzilla/stringzilla.hpp>

#include "usearch/utilities/safegraphs_format.hpp"

#include "bench.hpp"
#include "utilities/persisted_dataset.hpp" // `mapped_file_t`, `parse_matrix_header`, `matrix_save`

namespace unum::usearch::bench {

#pragma region Options and Styling

/** One named capability ceiling - the row shape the compute and schedule tables share. */
template <typename capability_type_>
struct capability_choice {
    char const* name;
    capability_type_ bit;
};

/** Compute-capability ceilings the kernel chains adapt, ascending per architecture and named by
 *  NumKong's microarchitecture vocabulary; the floor the table never lists is the serial
 *  kernels. */
inline constexpr capability_choice<nk_capability_t> compute_choices_k[] = {
#if NUMKONG_ARCH_X86_64_
    {"haswell", nk_cap_haswell_k}, {"skylake", nk_cap_skylake_k},         {"icelake", nk_cap_icelake_k},
    {"genoa", nk_cap_genoa_k},     {"sapphireamx", nk_cap_sapphireamx_k},
#elif NUMKONG_ARCH_ARM64_
    {"neon", nk_cap_neon_k},
    {"neonsdot", nk_cap_neonsdot_k},
    {"neonbfdot", nk_cap_neonbfdot_k},
#else
    {"serial", nk_cap_serial_k},
#endif
};

/** Scheduling ceilings the harness instantiates waiters for, ascending per architecture and named
 *  by ForkUnion's instruction vocabulary; the floor is the STL yield. Every row is gated by the
 *  runtime capability mask at dispatch, since Arm's @c WFET has no compile-time feature macro. */
inline constexpr capability_choice<fu_capabilities_t> schedule_choices_k[] = {
#if FORKUNION_ARCH_X86_64_
    {"pause", fu_capability_x86_pause_k},
    {"tpause", fu_capability_x86_tpause_k},
#elif FORKUNION_ARCH_ARM64_
    {"yield", fu_capability_arm64_yield_k},
    {"wfet", fu_capability_arm64_wfet_k},
#elif FORKUNION_ARCH_RISCV64_
    {"pause", fu_capability_risc5_pause_k},
    {"wrs", fu_capability_risc5_wrs_k},
#else
    {"standard", fu_capabilities_unknown_k},
#endif
};

/** Everything at or below the @p index-th ceiling of @p choices, as one allow-mask. */
template <typename capability_type_, std::size_t count_>
constexpr capability_type_ prefix_mask(capability_choice<capability_type_> const (&choices)[count_],
                                       std::size_t index) noexcept {
    capability_type_ mask {};
    for (std::size_t choice = 0; choice <= index && choice < count_; ++choice)
        mask = static_cast<capability_type_>(mask | choices[choice].bit);
    return mask;
}

/** The @p floor name, then every choice of @p choices whose bit @p mask covers - a view the
 *  report joins in one print. */
template <typename capability_type_, std::size_t count_>
auto named_choices(char const* floor, capability_choice<capability_type_> const (&choices)[count_],
                   capability_type_ mask) noexcept {
    return std::views::iota(std::size_t {0}, count_ + 1) |
           std::views::filter([&choices, mask](std::size_t position) noexcept {
               return position == 0 || (choices[position - 1].bit & mask) != 0;
           }) |
           std::views::transform([floor, &choices](std::size_t position) noexcept {
               return position ? choices[position - 1].name : floor;
           });
}

/** Names a single capability bit of @p choices; the unlisted @p floor backs them all. */
template <typename capability_type_, std::size_t count_>
constexpr char const* bit_name(capability_choice<capability_type_> const (&choices)[count_], capability_type_ bit,
                               char const* floor) noexcept {
    for (capability_choice<capability_type_> const& choice : choices)
        if (choice.bit == bit) return choice.name;
    return floor;
}

/** The newest table bit of @p choices inside @p mask, or @p floor when none is. */
template <typename capability_type_, std::size_t count_>
constexpr capability_type_ highest_bit(capability_choice<capability_type_> const (&choices)[count_],
                                       capability_type_ mask, capability_type_ floor) noexcept {
    capability_type_ highest = floor;
    for (capability_choice<capability_type_> const& choice : choices)
        if (mask & choice.bit) highest = choice.bit;
    return highest;
}

/** The index of the newest table row of @p choices inside @p mask, or @c count_ when the
 *  mask names none - the dispatch cascade's entry point into a table. */
template <typename capability_type_, std::size_t count_>
constexpr std::size_t highest_index(capability_choice<capability_type_> const (&choices)[count_],
                                    capability_type_ mask) noexcept {
    std::size_t highest = count_;
    for (std::size_t choice = 0; choice != count_; ++choice)
        if (mask & choices[choice].bit) highest = choice;
    return highest;
}

/** The index of the @p choices row named @p name, or @c count_ when unmatched - the CLI's
 *  entry point into a table, the naming inverse of @ref highest_index. */
template <typename capability_type_, std::size_t count_>
inline std::size_t index_of_name(capability_choice<capability_type_> const (&choices)[count_],
                                 std::string_view name) noexcept {
    for (std::size_t choice = 0; choice != count_; ++choice)
        if (name == choices[choice].name) return choice;
    return count_;
}

/** The schedule allow-mask intersected with what this machine offers - the one place the
 *  @c fu_capabilities_t C-enum arithmetic is re-typed, so its callers never re-cast. */
inline fu_capabilities_t effective_schedule(fu_capabilities_t allow) noexcept {
    return static_cast<fu_capabilities_t>(allow & fu_runtime_capabilities());
}

/** Every schedule bit except the waiter hints - the floor a `--max-schedule` ceiling builds on,
 *  keeping the pool-management toolbox while capping the wait instruction. */
inline constexpr fu_capabilities_t schedule_floor_k = static_cast<fu_capabilities_t>(fu_capabilities_all_k &
                                                                                     ~fu_capability_any_yield_k);

/** Lowers a runtime table position in `[0, count_)` to a compile-time index and hands it to
 *  @p on_row as an @c integral_constant; a position at or past the tail takes @p on_floor. Every
 *  position instantiates its own @p on_row, so the whole grid is built at compile time. */
template <std::size_t count_, std::size_t choice_ = 0>
int dispatch_index(std::size_t position, auto on_row, auto on_floor) {
    if constexpr (choice_ < count_) {
        if (position == choice_) return on_row(std::integral_constant<std::size_t, choice_> {});
        return dispatch_index<count_, choice_ + 1>(position, on_row, on_floor);
    }
    else return on_floor();
}

/** Matches a runtime enum against a compile-time value set, handing the match to @p on_value as an
 *  @c integral_constant; an unlisted value goes to @p on_unknown as the raw runtime value, so its
 *  diagnostic can name what was rejected. */
template <auto value_, auto... rest_>
int dispatch_value(decltype(value_) runtime, auto on_value, auto on_unknown) {
    if (runtime == value_) return on_value(std::integral_constant<decltype(value_), value_> {});
    if constexpr (sizeof...(rest_) == 0) return on_unknown(runtime);
    else return dispatch_value<rest_...>(runtime, on_value, on_unknown);
}

/** The file scalars the harness opens: what the RetriEval extensions name. */
int dispatch_file_dtype(nk_dtype_t dtype, auto on_dtype, auto on_unknown) {
    return dispatch_value<nk_f32_k, nk_bf16_k, nk_f16_k, nk_i8_k, nk_u8_k, nk_u1_k>(dtype, on_dtype, on_unknown);
}

/** Packed bits cast into no other storage: a bit file reaches the u1 shard when @p stored_dtypes_ lists one, and
 *  every other file dispatches over @p stored_dtypes_ in full. */
template <typename dataset_scalar_type_, nk_dtype_t... stored_dtypes_>
int dispatch_stored_value(nk_dtype_t dtype, auto on_dtype, auto on_unknown) {
    if constexpr (!std::is_same_v<dataset_scalar_type_, nk::u1x8_t>)
        return dispatch_value<stored_dtypes_...>(dtype, on_dtype, on_unknown);
    else if constexpr (((stored_dtypes_ == nk_u1_k) || ...))
        return dispatch_value<nk_u1_k>(dtype, on_dtype, on_unknown);
    else return on_unknown(dtype);
}

/** The stored dtypes the CPU benchmarks carry, by shrinking dynamic range. */
template <typename dataset_scalar_type_>
int dispatch_stored_dtype(nk_dtype_t dtype, auto on_dtype, auto on_unknown) {
    return dispatch_stored_value<dataset_scalar_type_, nk_f32_k, nk_bf16_k, nk_f16_k, nk_e5m2_k, nk_e4m3_k, nk_e3m2_k,
                                 nk_e2m3_k, nk_i8_k, nk_i4_k, nk_u8_k, nk_u4_k, nk_u1_k>(dtype, on_dtype, on_unknown);
}

/** The kernels one stored dtype adapts: bit-packed vectors take the bit kernels. */
template <nk_dtype_t stored_dtype_>
    requires(stored_dtype_ == nk_u1_k)
int dispatch_kernel(nk_kernel_kind_t kernel, auto on_kernel, auto on_unknown) {
    return dispatch_value<nk_kernel_dot_k, nk_kernel_hamming_k, nk_kernel_jaccard_k>(kernel, on_kernel, on_unknown);
}

/** The kernels one stored dtype adapts: every other dtype takes the metric trio. */
template <nk_dtype_t stored_dtype_>
    requires(stored_dtype_ != nk_u1_k)
int dispatch_kernel(nk_kernel_kind_t kernel, auto on_kernel, auto on_unknown) {
    return dispatch_value<nk_kernel_dot_k, nk_kernel_angular_k, nk_kernel_euclidean_k>(kernel, on_kernel, on_unknown);
}

/** Everything the CLI can configure; zeroes mean "infer from the dataset and machine". */
struct bench_options_t {

    /** The base vectors file, in the RetriEval `.fbin` extension family. */
    std::string vectors_path;

    /** Optional query vectors of the same dtype; absent means self-queries. */
    std::string queries_path;

    /** Optional `.ibin` ground-truth neighbors; absent skips the quality judgment. */
    std::string neighbors_path;

    /** The representation vectors are stored in; unknown means "as the file". */
    nk_dtype_t stored_dtype = nk_dtype_unknown_k;

    /** Allow-mask over NumKong kernel capabilities, intersected with the runtime detection. */
    nk_capability_t max_compute = nk_cap_any_k;

    /** Allow-mask over ForkUnion scheduling capabilities, forwarded verbatim to
     *  @c fu_pool_new and mirrored by the waiter type the index instantiates. */
    fu_capabilities_t max_schedule = fu_capabilities_all_k;

    /** The distance kernel driving both construction and search. */
    nk_kernel_kind_t kernel = nk_kernel_angular_k;

    /** Engaged memory domains: empty is one machine-wide heap index, a single entry pins
     *  memory and threads to it, and several build one index per listed domain. */
    std::vector<fu_memory_domain_id_t> memory_domains;

    /** Worker threads; zero takes 7/8 of the available logical cores. */
    std::size_t threads = 0;

    /** Staging trampoline rows shared by all threads; zero takes 4 × threads. */
    std::size_t batch = 0;

    /** Neighbors-list width per node; must match a `connectivity_t` tiling. */
    std::size_t connectivity = 32;

    /** Insertion candidate pool, calibrated on SIFT 10M against @c expansion_search for the
     *  build-once-search-forever regime. */
    std::size_t expansion_add = 256;

    /** Search candidate pool; see @c expansion_add for the calibration. */
    std::size_t expansion_search = 96;

    /** Matches per query to return and judge. */
    std::size_t top_k = 10;

    /** Search every ingested vector for itself instead of the query file - the
     *  calibration-free navigability probe, with a search phase as long as the ingestion. */
    bool self_recall = false;

    /** Checkpoint name to serialize every domain index into after ingestion, resolved through
     *  @ref checkpoint_paths_of. */
    std::string save_name;

    /** Checkpoint name to restore domain indexes from before ingestion; pair it with
     *  `--skip-vectors`, so only the unindexed topping streams in. */
    std::string load_name;

    /** Directory the checkpoint blobs land in; point 100M-scale runs at a real disk. */
    std::string checkpoints_dir = "/tmp";

    /** Rows the loaded indexes already cover; ingestion resumes right after them. */
    std::size_t skip_vectors = 0;

    /** Cap on ingested rows; zero ingests the whole file. */
    std::size_t max_vectors = 0;

    /** First dataset row to view - slices one file across independent runs. */
    std::size_t first_vector = 0;

    /** Huge pages the launcher provisioned before starting the process; zero expects none, any
     *  other value is enforced, so a base-page fallback cannot pass as a huge-page result. */
    std::size_t huge_pages = 0;

    /** CUDA device group for the multi-index pipeline: one shard index per listed ordinal.
     *  Empty runs the single-device pipeline on `--device`. */
    std::vector<int> devices;

    /** Merge the shards into one stitched graph after ingest and answer queries with a single
     *  walk; false scatters every query to all shards and k-way merges the hits on the host. */
    bool merge = false;

    /** Merge candidate pool; zero inherits `--expansion-add`. */
    std::size_t expansion_merge = 0;

    /** Upper bound on refinement passes after ingestion or a load; a pass that re-selects
     *  nothing ends the phase early, and zero skips it outright. */
    std::size_t max_refines = 0;

    /** Refinement candidate pool; zero takes the index's default, twice the connectivity. */
    std::size_t expansion_refine = 0;

    /** Print the manual and exit. */
    bool help = false;
};

/** The three matrix files one checkpointed slot spans - each a RetriEval-layout dump. */
struct checkpoint_paths_t {
    std::string vectors;
    std::string neighbors;
    std::string rows;
};

/** Resolves a user-facing checkpoint name into one slot's matrix-file paths. The stamp names what
 *  the payload depends on - stored dtype, distance kernel, connectivity - so any backend
 *  instantiating the same combination computes the same names, and a mismatched one finds
 *  nothing. */
template <typename index_type_>
static checkpoint_paths_t checkpoint_paths_of(bench_options_t const& options, std::string_view name, std::size_t slot,
                                              std::size_t slot_count) {
    std::string const prefix = fmt::format(
        "{}/{}-{}-{}-c{}-{}-{}", options.checkpoints_dir, name, index_type_::scalar_t::dtype_name(),
        nk_kernel_name(index_type_::kernel_kind_k), options.connectivity, slot, slot_count);
    return {fmt::format("{}.vectors.{}bin", prefix, index_type_::scalar_t::dtype_name()), //
            prefix + ".neighbors.ibin", prefix + ".rows.ibin"};
}

/** Maps the CLI neighbor-list degree onto the closed @c connectivity_t tiling set. */
inline connectivity_t connectivity_of(std::size_t degree) noexcept {
    if (degree == 16) return connectivity_16_k;
    if (degree == 32) return connectivity_32_k;
    if (degree == 128) return connectivity_128_k;
    return connectivity_unknown_k;
}

/**
 *  @brief Restores one index slot from its three RetriEval-layout matrix dumps, refilling the
 *      ID → row map the recall bookkeeping needs. Prints its own diagnostics.
 *
 *  @param[in] options Supplies @c load_name, @c checkpoints_dir and the @c connectivity.
 *  @param[inout] index A virgin index already reserved to the run's geometry, which the mapped
 *      headers are validated against.
 *  @param[in] dimensions Row width the vectors matrix must match.
 *  @param[out] id_to_row Landing for the restored map; spans at least the checkpoint's rows.
 *  @param[in] slot This index's position among the sibling slots.
 *  @param[in] slot_count Sibling slots the save spanned; both are baked into the file names.
 *  @return True on a fully restored slot; false after printing what failed.
 */
template <typename index_type_>
static bool load_checkpoint(bench_options_t const& options, index_type_& index, std::size_t dimensions,
                            std::span<u32_t> id_to_row, std::size_t slot, std::size_t slot_count) {
    using scalar_t = typename index_type_::scalar_t;
    checkpoint_paths_t const paths = checkpoint_paths_of<index_type_>(options, options.load_name, slot, slot_count);
    auto [vectors_mapping, vectors_map_status] = mapped_file_t::read(paths.vectors);
    auto [neighbors_mapping, neighbors_map_status] = mapped_file_t::read(paths.neighbors);
    auto [rows_mapping, rows_map_status] = mapped_file_t::read(paths.rows);
    if (failed(vectors_map_status) || failed(neighbors_map_status) || failed(rows_map_status)) {
        fmt::println("Failed to map the {} checkpoint for slot {}", options.load_name, slot);
        return false;
    }

    auto const [vectors_extents,
                vectors_status] = utilities::parse_matrix_header<scalar_t, u32_t>(vectors_mapping.view());
    auto const [neighbors_extents,
                neighbors_status] = utilities::parse_matrix_header<u32_t, u32_t>(neighbors_mapping.view());
    auto const [map_extents, map_status] = utilities::parse_matrix_header<u32_t, u32_t>(rows_mapping.view());
    bool const parsed = succeeded(vectors_status) && succeeded(neighbors_status) && succeeded(map_status);
    std::size_t const loaded_nodes = vectors_extents.rows;
    if (!parsed || vectors_extents.columns != dimensions || neighbors_extents.columns != options.connectivity ||
        map_extents.columns != 1 || neighbors_extents.rows != loaded_nodes || map_extents.rows != loaded_nodes ||
        loaded_nodes > id_to_row.size()) {
        fmt::println("Checkpoint {} disagrees with the run's geometry", options.load_name);
        return false;
    }

    nk::matrix_view<scalar_t> const vectors_view = utilities::view_matrix_payload<scalar_t, u32_t>(
        vectors_mapping.view(), loaded_nodes, dimensions);
    nk::matrix_view<u32_t> const neighbors_view = utilities::view_matrix_payload<u32_t, u32_t>(
        neighbors_mapping.view(), loaded_nodes, options.connectivity);
    // Both the chunks and the mapped dumps are dense, so a chunk is one byte move.
    std::size_t const vectors_row_bytes = static_cast<std::size_t>(vectors_view.stride_bytes(0));
    std::size_t const neighbors_row_bytes = static_cast<std::size_t>(neighbors_view.stride_bytes(0));
    bytes_view_t const vectors_source {reinterpret_cast<byte_t const*>(vectors_view.byte_data()),
                                       loaded_nodes * vectors_row_bytes};
    bytes_view_t const neighbors_source {reinterpret_cast<byte_t const*>(neighbors_view.byte_data()),
                                         loaded_nodes * neighbors_row_bytes};
    using node_id_t = typename index_type_::node_id_t;
    status_t const loaded = index.load({vectors_source, neighbors_source}, loaded_nodes,
                                       {0, static_cast<node_id_t>(loaded_nodes)});
    if (loaded != status_t::success_k) {
        fmt::println("Failed to restore {}: status {}", paths.vectors, int(loaded));
        return false;
    }
    nk::matrix_view<u32_t> const rows_view = utilities::view_matrix_payload<u32_t, u32_t>(rows_mapping.view(),
                                                                                          loaded_nodes, 1);
    std::copy_n(rows_view.data(), loaded_nodes, id_to_row.data());
    return true;
}

/**
 *  @brief Serializes one index slot as three RetriEval-layout matrices - vectors, neighbors and
 *      the ID → row map. Prints its own diagnostics.
 *
 *  @param[in] options Supplies @c save_name, @c checkpoints_dir and the @c connectivity.
 *  @param[inout] index The built index; @c save drains it under @c immutable_k, so no insertions
 *      may be in flight.
 *  @param[in] id_to_row The map filled during ingestion; its first @c saved_shape().nodes entries
 *      ride along as the third matrix.
 *  @param[in] slot This index's position among the sibling slots.
 *  @param[in] slot_count Sibling slots the save spans; both are baked into the file names.
 *  @return The serialized byte volume, or zero on failure - a real checkpoint never is.
 */
template <typename index_type_>
static std::size_t save_checkpoint(bench_options_t const& options, index_type_& index, std::span<u32_t> id_to_row,
                                   std::size_t slot, std::size_t slot_count) {
    using scalar_t = typename index_type_::scalar_t;
    using node_id_t = typename index_type_::node_id_t;
    typename index_type_::saved_shape_t const shape = index.saved_shape();
    auto vectors = nk::matrix<scalar_t>::try_zeros({shape.nodes, shape.dimensions});
    auto neighbors = nk::matrix<node_id_t>::try_zeros({shape.nodes, shape.connectivity});
    if (vectors.empty() || neighbors.empty()) {
        fmt::println("Failed to stage {} rows for serialization", shape.nodes);
        return 0;
    }
    // The staged matrices are dense, so a chunk is one byte move.
    std::size_t const vectors_row_bytes = static_cast<std::size_t>(vectors.stride_bytes(0));
    std::size_t const neighbors_row_bytes = static_cast<std::size_t>(neighbors.stride_bytes(0));
    bytes_span_t const vectors_target {reinterpret_cast<byte_t*>(vectors.data()), shape.nodes * vectors_row_bytes};
    bytes_span_t const neighbors_target {reinterpret_cast<byte_t*>(neighbors.data()),
                                         shape.nodes * neighbors_row_bytes};
    // One step's rows at most: a storage chunk never holds more, and the device plan ignores it.
    auto scratch = nk::matrix<node_id_t>::try_zeros(
        {min_of(shape.nodes, power_of_two<std::size_t>(20)), shape.connectivity});
    if (shape.nodes && scratch.empty()) {
        fmt::println("Failed to stage the neighbor scratch for index {}", slot);
        return 0;
    }
    status_t const saved = index.save({vectors_target, neighbors_target}, synchronization_t::immutable_k,
                                      scratch.span());
    if (saved != status_t::success_k) {
        fmt::println("Failed to serialize index {}: status {}", slot, int(saved));
        return 0;
    }

    checkpoint_paths_t const paths = checkpoint_paths_of<index_type_>(options, options.save_name, slot, slot_count);
    nk::matrix_view<u32_t> const rows_map_view {id_to_row.data(), {shape.nodes, 1}};
    bool const written =
        utilities::matrix_save<scalar_t>(paths.vectors.c_str(), vectors.view()) == status_t::success_k &&
        utilities::matrix_save<node_id_t>(paths.neighbors.c_str(), neighbors.view()) == status_t::success_k &&
        utilities::matrix_save<u32_t>(paths.rows.c_str(), rows_map_view) == status_t::success_k;
    if (!written) {
        fmt::println("Failed to write the {} checkpoint for slot {}", options.save_name, slot);
        return 0;
    }
    return shape.nodes * (static_cast<std::size_t>(vectors.view().stride_bytes(0)) + //
                          shape.connectivity * sizeof(node_id_t) + sizeof(u32_t));
}

#pragma endregion

#pragma region Telemetry and Judgment

/** Telemetry sums each thread accumulates privately, padded away from its neighbors. */
struct alignas(default_alignment_k) telemetry_lane_t {

    /** Frontier candidates whose neighbor lists were expanded. */
    std::size_t visited_members = 0;

    /** Total distance computations, padding lanes excluded. */
    std::size_t computed_distances = 0;

    /** Subset of @c computed_distances spent in diversity selection. */
    std::size_t computed_distances_in_refines = 0;

    /** Subset of @c computed_distances spent re-selecting while back-linking. */
    std::size_t computed_distances_in_reverse_refines = 0;

    /** Starvation restarts - a graph-connectivity health signal. */
    std::size_t restarts = 0;

    /** Walks a scratch refusal ended early on the CPU; the GPU folds its starved visited-set
     *  claims here. */
    std::size_t saturated_walks = 0;
};

/** Sums the per-thread telemetry after a phase; the lanes stay untouched. */
inline telemetry_lane_t summed_lanes(std::vector<telemetry_lane_t> const& lanes) noexcept {
    telemetry_lane_t sums;
    for (telemetry_lane_t const& lane : lanes) {
        sums.visited_members += lane.visited_members;
        sums.computed_distances += lane.computed_distances;
        sums.computed_distances_in_refines += lane.computed_distances_in_refines;
        sums.computed_distances_in_reverse_refines += lane.computed_distances_in_reverse_refines;
        sums.restarts += lane.restarts;
        sums.saturated_walks += lane.saturated_walks;
    }
    return sums;
}

/** The backend-neutral judgment state every backend's phases fill: the id → row bookkeeping, the
 *  merged hits with their distances and latencies, and the failure tally. */
struct bench_results_t {

    /** The @b [rows] recall bookkeeping: dense node id → global dataset row. */
    std::vector<u32_t> id_to_row;

    /** The @b [queries,depth] expected neighbors; empty when absent or truncated by
     *  `--max-vectors`. */
    nk::matrix_view<u32_t> ground_truth;

    /** The @b [queries,top_k] merged global rows of every query's gathered hits. */
    nk::matrix<u32_t> hit_rows;

    /** The @b [queries,top_k] distances matching @c hit_rows entry-for-entry. */
    nk::matrix<dense_distance_t> hit_distances;

    /** Merged hits per query, sizing each row for the post-clock judgment. */
    std::vector<u32_t> hit_counts;

    /** Per-query end-to-end lookup latencies, in microseconds. */
    std::vector<float> latencies_us;

    /** Tasks that returned an error in either phase; any non-zero fails the run. */
    std::size_t failed_tasks = 0;
};

/** What the refinement phase measured, whichever backend ran it: the passes, the audit that
 *  followed them, and the traversal telemetry they cost. */
struct refinement_report_t {

    /** Published nodes one pass sweeps. */
    std::size_t nodes = 0;

    /** The candidate pool every pass re-selects under. */
    std::size_t expansion = 0;

    /** Nodes each pass re-wired; its length is the passes that actually ran. */
    std::span<std::size_t const> changed_per_pass;

    /** Threads the passes fanned over; one on a device. */
    std::size_t threads = 0;

    /** Wall clock of the passes alone. */
    double refining_seconds = 0;

    /** Wall clock of the heal and its final audit. */
    double healing_seconds = 0;

    /** Traversal telemetry summed across the passes. */
    telemetry_lane_t sums;

    /** Nodes the heal's first audit found unreachable. */
    std::size_t islands_found = 0;

    /** Islands the heal wired into a reachable pick. */
    std::size_t islands_reconnected = 0;

    /** Nodes a search still reaches once the heal returned. */
    std::size_t reachable_nodes = 0;

    /** Level-synchronous sweeps the final audit took - a diameter proxy. */
    std::size_t sweeps = 0;
};

/** Prints the refinement phase - the passes, their cost, and the reachability the heal left
 *  behind. */
inline void report_refinement_metrics(refinement_report_t const& report) {
    std::size_t const passes = report.changed_per_pass.size();
    fmt::print(header_style(), "\nRefinement\n");
    fmt::println("  passes: {} · expansion {}", passes, report.expansion);
    fmt::println("  changed: {}", fmt::join(report.changed_per_pass, " → "));
    fmt::println("  threads: {}", report.threads);
    fmt::println("  wall: {:.2f} s refining · {:.2f} s healing", report.refining_seconds, report.healing_seconds);
    fmt::println("  throughput: {} nodes/s", humanized(double(report.nodes * passes) / report.refining_seconds));
    fmt::println("  distances: {}", humanized(report.sums.computed_distances));
    fmt::println("  visited: {}", humanized(report.sums.visited_members));
    fmt::println("  islands: {} found, {} reconnected", report.islands_found, report.islands_reconnected);
    fmt::println("  reachable: {} of {} · {} sweeps",
                 fmt::styled(report.reachable_nodes, verdict_style(report.reachable_nodes == report.nodes)),
                 report.nodes, report.sweeps);
}

/** Prints the search phase's wall clock, throughput, latency percentiles and traversal
 *  telemetry; @p latencies_us reorders in place. */
inline void report_search_metrics(bench_options_t const& options, std::size_t query_count, double seconds,
                                  std::size_t threads, telemetry_lane_t const& sums, std::span<float> latencies_us) {
    auto percentile = [&](double share) {
        std::size_t const rank = std::min(query_count - 1, std::size_t(share * double(query_count)));
        std::nth_element(latencies_us.begin(), latencies_us.begin() + rank, latencies_us.end());
        return latencies_us[rank];
    };
    fmt::print(header_style(), "\nQueries\n");
    fmt::println("  queries: {} at top-{}{}", query_count, options.top_k, options.self_recall ? " · self-recall" : "");
    fmt::println("  threads: {}", threads);
    fmt::println("  wall: {:.2f} s", seconds);
    fmt::println("  throughput: {} QPS", humanized(double(query_count) / seconds));
    // An empty latency span means a batch-shaped backend with no per-query clock.
    if (latencies_us.empty())
        fmt::println("  latency: {:.0f} µs mean per query, batch-uniform", seconds * 1e6 / double(query_count));
    else
        fmt::println("  latency: p50 {:.0f} µs · p90 {:.0f} µs · p99 {:.0f} µs", percentile(0.50), percentile(0.90),
                     percentile(0.99));
    fmt::println("  distances/query: {}", humanized(double(sums.computed_distances) / double(query_count)));
    fmt::println("  visited/query: {}", humanized(double(sums.visited_members) / double(query_count)));
    fmt::println("  restarts: {}", sums.restarts);
    fmt::println("  saturated: {}", fmt::styled(sums.saturated_walks, verdict_style(sums.saturated_walks == 0)));
}

/** Judges the merged hits after the clocks stopped - self-recall against own rows, or
 *  recall/overlap/ndcg against the ground truth - one sweep per query either way. */
inline void judge_hits(bench_options_t const& options, bench_results_t& results, std::size_t query_count) {
    if (options.self_recall) {
        std::size_t found_at_1 = 0, found_at_k = 0;
        for (std::size_t query = 0; query != query_count; ++query) {
            auto const returned = to_span(results.hit_rows.row(query).as_vector()).first(results.hit_counts[query]);
            u32_t const own_row = static_cast<u32_t>(query);
            found_at_1 += !returned.empty() && returned.front() == own_row;
            bool within = false;
            for (u32_t row : returned) within |= row == own_row;
            found_at_k += within;
        }
        double const queries = double(query_count);
        double const self_at_1 = 100.0 * double(found_at_1) / queries;
        double const self_at_k = 100.0 * double(found_at_k) / queries;
        fmt::println("  self-recall@1: {:.2f}%", fmt::styled(self_at_1, verdict_style(self_at_1 >= 99.0)));
        fmt::println("  self-recall@{}: {:.2f}%", options.top_k,
                     fmt::styled(self_at_k, verdict_style(self_at_k >= 99.0)));
    }
    else if (!results.ground_truth.empty()) {
        std::size_t const judged_depth = std::min(options.top_k, results.ground_truth.extent(1));
        std::vector<u32_t> scratch(judged_depth);
        std::vector<double> discounts(options.top_k), ideal(options.top_k);
        fill_discounts(discounts, ideal);

        std::size_t found_at_1 = 0, found_at_k = 0, shared = 0;
        double gain = 0;
        for (std::size_t query = 0; query != query_count; ++query) {
            auto expected = results.ground_truth.row(query).as_vector();
            auto const returned = to_span(results.hit_rows.row(query).as_vector()).first(results.hit_counts[query]);
            found_at_1 += !returned.empty() && returned.front() == expected[0];
            auto verdict = judge_ranking(std::span<u32_t const> {returned}, to_span(expected).first(judged_depth),
                                         std::span<u32_t> {scratch}, std::span<double const> {discounts});
            found_at_k += verdict.found_nearest;
            shared += verdict.overlap;
            gain += verdict.gain;
        }

        double const queries = double(query_count);
        double const recall_at_1 = 100.0 * double(found_at_1) / queries;
        double const recall_at_k = 100.0 * double(found_at_k) / queries;
        double const overlap_at_k = 100.0 * double(shared) / (queries * double(judged_depth));
        double const gain_at_k = 100.0 * gain / (queries * ideal[judged_depth - 1]);
        fmt::println("  recall@1: {:.3f}%", fmt::styled(recall_at_1, verdict_style(recall_at_1 >= 90.0)));
        fmt::println("  recall@{}: {:.3f}%", judged_depth,
                     fmt::styled(recall_at_k, verdict_style(recall_at_k >= 99.0)));
        fmt::println("  overlap@{}: {:.3f}%", judged_depth, overlap_at_k);
        fmt::println("  ndcg@{}: {:.3f}%", judged_depth, gain_at_k);
    }
    else { fmt::println("  recall: skipped, no ground-truth neighbors provided"); }
}

#pragma endregion

#pragma region Dataset Streaming

/**
 *  @brief The data-side of a run: typed views over the source rows and the bounded staging
 *      matrix the streaming @c nk::cast trampolines through, one vector at a time.
 *
 *  The staging matrix holds `--batch` rows for every thread together, never a whole-dataset
 *  copy; each thread rotates through its own contiguous area, so concurrent tasks never share a
 *  row. When the file scalar matches the stored one the staging is empty and every access is a
 *  zero-copy view into the mapping.
 */
template <typename dataset_scalar_type_, typename bench_context_type_>
struct streamed_dataset {
    using bench_t = bench_context_type_;
    using dataset_scalar_t = dataset_scalar_type_;
    using scalar_t = typename bench_context_type_::scalar_t;
    using vector_view_t = typename bench_context_type_::vector_view_t;
    static constexpr bool zero_copy_k = std::is_same_v<dataset_scalar_type_, scalar_t>;

    /** Float sources downcast into integer or byte-sized float targets only after a global
     *  symmetric rescale into the target's range; a raw cast collapses the value distribution,
     *  saturating it in integers and flushing most of an embedding to zero in fp8 and fp6. */
    static constexpr bool scaled_k = !zero_copy_k && !nk::is_integral_dtype<dataset_scalar_type_>() &&
                                     (nk::is_integral_dtype<scalar_t>() || sizeof(scalar_t) == 1);

    /** The index-side state both phases feed - punned through the pool callbacks. */
    bench_context_type_& bench;

    /** The @b [rows,dimensions] mapped source rows, in the file's own scalar. */
    nk::matrix_view<dataset_scalar_type_> vectors;

    /** The @b [queries,dimensions] mapped query rows - or a prefix of @c vectors. */
    nk::matrix_view<dataset_scalar_type_> queries;

    /** The @b [threads,rows_per_thread,dimensions] staging trampoline planes. */
    nk::tensor_span<scalar_t, 3> staging;

    /** The @b [threads,rows_per_thread,dimensions] `f32` rescaling planes; only set when
     *  @c scaled_k. */
    nk::tensor_span<nk::f32_t, 3> rescaling;

    /** The global symmetric factor applied before integer downcasts. */
    float scale = 1.0f;

    /** One @c reduce_minmax pass over the source payload finds the amplitude; the same factor
     *  rescales every vector and every query, keeping their geometry consistent. A no-op unless
     *  this run downcasts a non-integral source into an integral store. */
    void establish_scale() {
        if constexpr (scaled_k) {
            typename dataset_scalar_type_::reduce_minmax_value_t min_value {}, max_value {};
            std::size_t min_index = 0, max_index = 0;
            // `as_vector` answers only rank-1 views, so the contiguous payload is flattened by hand.
            nk::vector_view<dataset_scalar_type_> const payload {vectors.data(), vectors.extent(0) * vectors.extent(1)};
            nk::reduce_minmax(payload, &min_value, &min_index, &max_value, &max_index);
            float const max_amplitude = std::max(std::abs(float(min_value)), std::abs(float(max_value)));
            // The stored dtype's largest magnitude; not a constant expression for nvcc's byte floats.
            float const target_amplitude = component_max<scalar_t>();
            if (max_amplitude > 0) scale = target_amplitude / max_amplitude;
            fmt::println("Rescaling {} → {} by ×{:.4f} before the downcast", dataset_scalar_type_::dtype_name(),
                         scalar_t::dtype_name(), scale);
        }
    }

    /** Converts one source row into the calling thread's own staging plane and views it back;
     *  @p task rotates the row within the plane, so in-flight tasks never clash. */
    vector_view_t stage(nk::vector_view<dataset_scalar_type_> source, std::size_t thread, std::size_t task) noexcept {
        std::size_t const dimensions = vectors.extent(1);
        std::size_t const rotation = task % staging.extent(1);
        auto landing = staging.row(thread).row(rotation).as_vector();
        if constexpr (scaled_k) {
            // An f32 source feeds @c nk::scale straight from the mapping, skipping the widening cast.
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

/** Prints the machine as ForkUnion sees it: cores, compute and memory domains, and the NUMA
 *  fabric levels, next to the NumKong capability the kernels will use. */
inline void print_machine(fu_topology_t topology, fu_fabric_t fabric, bench_options_t const& options,
                          std::size_t threads) {
    // A machine description is worth printing even where one query declines to answer.
    std::size_t compute_domains = 0, memory_domains = 0, logical_cores = 0;
    [[maybe_unused]] fu_status_t const compute_status = fu_compute_domains_count(topology, &compute_domains);
    [[maybe_unused]] fu_status_t const memory_status = fu_memory_domains_count(topology, &memory_domains);
    [[maybe_unused]] fu_status_t const cores_status = fu_logical_cores_count(topology, &logical_cores);
    fmt::print(header_style(), "Machine\n");
    fmt::println("  logical cores: {}", logical_cores);
    fmt::println("  compute domains: {}", compute_domains);
    fmt::println("  memory domains: {}", memory_domains);
    for (std::size_t domain = 0; domain != memory_domains; ++domain) {
        std::size_t fabric_level = 0, huge_pages_bytes = 0;
        [[maybe_unused]] fu_status_t const level_status = fu_fabric_memory_level_in(fabric, domain, &fabric_level);
        [[maybe_unused]] fu_status_t const pages_status = fu_volume_huge_pages_in(topology, domain, &huge_pages_bytes);
        fmt::println("  memory domain {}: fabric level {}, huge pages {:.2f} GB", domain, fabric_level,
                     double(huge_pages_bytes) / 1e9);
    }
    if (!options.memory_domains.empty())
        fmt::println("  engaged memory domains: {}", fmt::join(options.memory_domains, " "));
    fmt::println("  threads engaged: {}", threads);

    char scheduling_names[256] = {0};
    std::size_t scheduling_names_length = 0;
    [[maybe_unused]] fu_status_t const named = fu_name_capabilities(fu_runtime_capabilities(), scheduling_names,
                                                                    sizeof(scheduling_names), &scheduling_names_length);
    fmt::println("  scheduling detected: {}", scheduling_names);

    nk_capability_t const available = nk_capabilities_available();
    fmt::println("  compute detected: {}", fmt::join(named_choices("serial", compute_choices_k, available), " "));
    fmt::println("  compute ceiling: {}",
                 bit_name(compute_choices_k,
                          highest_bit(compute_choices_k, options.max_compute & available, nk_cap_serial_k), "serial"));
    fmt::println("  scheduling ceiling: {}",
                 bit_name(schedule_choices_k,
                          highest_bit(schedule_choices_k, effective_schedule(options.max_schedule), //
                                      fu_capabilities_unknown_k),
                          "standard"));
    fmt::println("  kernels requested: {} · {}", dtype_name(options.stored_dtype), nk_kernel_name(options.kernel));
}

} // namespace unum::usearch::bench
