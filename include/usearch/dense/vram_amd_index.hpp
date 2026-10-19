/**
 *  @file include/usearch/dense/vram_amd_index.hpp
 *  @author Ash Vardanian
 *  @date October 11, 2026
 *  @brief Vector Search index for a single uniform latency memory space on AMD GPUs.
 *
 *  The host face of the AMD family, compiled by any C++ compiler: it fills the plain argument
 *  records of @c vram_index.hpp and launches the kernels of @c vram_amd_index.hip by name through
 *  the HIP runtime it opens at first use, so it never sees a kit, a shared view or HIP beyond its
 *  runtime's declarations. Every kernel is one instantiation of a single template over five
 *  arguments - the kernel, the capability ceiling, the dtype, the register rung and the attend
 *  value lookup - so @c vram_kernel_symbol spells its symbol here from those arguments alone, and
 *  one code object per class, which the face's kernels type names, carries every dtype.
 */
#pragma once

#include <cstddef> // `std::size_t`
#include <cstring> // `std::memcpy` - an attend's value lookup travels as bytes

#include <bit>         // `std::bit_ceil` - the lanes sharing a row
#include <limits>      // `std::numeric_limits` - the unit-RMS import's epsilon
#include <numbers>     // `std::numbers::log2e_v` - the base-2 softmax
#include <type_traits> // `std::is_same_v` - which heads a walk stages
#include <utility>     // `std::move` - the loaded module

#include <numkong/capabilities.h> // `nk_kernel_kind_t`
#include <numkong/cast/serial.h>  // `nk_cast_plan_serial_` - the device casts' plans
#include <numkong/types.h>

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/vram_amd_device.hpp"
#include "usearch/utilities/vram_device.hpp" // `vram_kernel_symbol` - a kernel's symbol
#include "usearch/dense/types.hpp"           // `vram_telemetry_block_t` & the launch geometry constants
#include "usearch/dense/vram_index.hpp"

namespace unum::usearch::dense {

#pragma region Kernels

/** The device value lookup an attend kernel resolves rows through, by the @c device_lookup_k a
 *  lookup type names itself with: whichever unit compiles a code object specializes it with
 *  @c type for every lookup that code object's attend kernels take. */
template <u32_t lookup_>
struct amd_value_lookup_of;

/** Blocks of @p kernel on its @p slots rung resident per compute unit under @p capability_ over
 *  @p dtype_ - the register budget its launch bounds set, the load window its body sizes, and its
 *  grid's share of every compute unit: three for the builders and the back-links riding their grid
 *  and arena, as many as saturate the path at its registers; a walk's by its rung; eight for the
 *  sweeps striding the graph, oversubscribed so enough rows stay in flight; and one for the rest,
 *  whose grids follow their work. */
template <nk_capability_t capability_, nk_dtype_t dtype_>
constexpr u32_t hip_resident_blocks(vram_kernel_t kernel, u32_t slots) noexcept {
    switch (kernel) {
    case vram_kernel_t::insert_k:
    case vram_kernel_t::back_link_k:
    case vram_kernel_t::refine_k:
    case vram_kernel_t::merge_back_links_k: return 3;
    case vram_kernel_t::search_k:
    case vram_kernel_t::search_indexer_k:
    case vram_kernel_t::multiply_k: return slots <= 8 ? 4 : 3;
    // The attend fold carries per-head state, which spills at 8 slots under four blocks' budget.
    case vram_kernel_t::attend_k: return slots <= 4 ? 4 : 3;
    case vram_kernel_t::scatter_back_links_k:
    case vram_kernel_t::derive_norms_k:
    case vram_kernel_t::rebase_links_k:
    case vram_kernel_t::reachability_sweep_k:
    case vram_kernel_t::reached_in_links_k:
    case vram_kernel_t::cast_k:
    case vram_kernel_t::rmscast_k: return 8;
    default: return 1;
    }
}

/** Waves each SIMD keeps when @p blocks blocks of @c vram_warps_per_block_k wavefronts share a
 *  compute unit's four SIMDs - the unit HIP-Clang reads the second argument of
 *  @c __launch_bounds__ in, where CUDA reads blocks per multiprocessor. */
constexpr u32_t hip_waves_per_simd(u32_t blocks) noexcept { return blocks * vram_warps_per_block_k / 4; }

/** One instantiated kernel: the kernel and its register rung, zero where it keeps no ladder or
 *  walks the arena frontier. */
struct amd_rung_t {
    vram_kernel_t kernel;
    u32_t slots;
};

/** Every instantiation one dtype's code exports bar the attends, which the device pass instantiates
 *  and the host looks up as it loads the code object. */
inline constexpr amd_rung_t amd_rungs[] = {
    {vram_kernel_t::insert_k, 0},
    {vram_kernel_t::insert_k, 2},
    {vram_kernel_t::insert_k, 4},
    {vram_kernel_t::insert_k, 8},
    {vram_kernel_t::insert_k, 16},
    {vram_kernel_t::back_link_k, 0},
    {vram_kernel_t::refine_k, 0},
    {vram_kernel_t::refine_k, 2},
    {vram_kernel_t::refine_k, 4},
    {vram_kernel_t::refine_k, 8},
    {vram_kernel_t::refine_k, 16},
    {vram_kernel_t::search_k, 0},
    {vram_kernel_t::search_k, 2},
    {vram_kernel_t::search_k, 4},
    {vram_kernel_t::search_k, 8},
    {vram_kernel_t::search_k, 16},
    {vram_kernel_t::search_indexer_k, 0},
    {vram_kernel_t::search_indexer_k, 2},
    {vram_kernel_t::search_indexer_k, 4},
    {vram_kernel_t::search_indexer_k, 8},
    {vram_kernel_t::search_indexer_k, 16},
    {vram_kernel_t::multiply_k, 0},
    {vram_kernel_t::multiply_k, 2},
    {vram_kernel_t::multiply_k, 4},
    {vram_kernel_t::multiply_k, 8},
    {vram_kernel_t::multiply_k, 16},
    {vram_kernel_t::merge_back_links_k, 0},
    {vram_kernel_t::scatter_back_links_k, 0},
    {vram_kernel_t::derive_norms_k, 0},
    {vram_kernel_t::rebase_links_k, 0},
    {vram_kernel_t::mark_landmarks_k, 0},
    {vram_kernel_t::reachability_sweep_k, 0},
    {vram_kernel_t::reachability_islands_k, 0},
    {vram_kernel_t::reached_in_links_k, 0},
    {vram_kernel_t::reconnect_island_k, 0},
    {vram_kernel_t::merge_attention_k, 0},
    {vram_kernel_t::resolve_matches_k, 0},
    {vram_kernel_t::merge_matches_k, 0},
    {vram_kernel_t::cast_k, 0},
    {vram_kernel_t::rmscast_k, 0},
};

/** The rungs every walk family, attends included, is instantiated over. */
inline constexpr u32_t amd_walk_rungs[] = {0, 2, 4, 8, 16};

#pragma endregion Kernels

/**
 *  @brief Vector Search index for a single uniform latency memory space on AMD GPUs.
 *
 *  Everything but the kernels is @c vram_index; a wavefront plays the part an NVIDIA warp plays,
 *  64 lanes wide before CDNA5 and 32 on it. The kernels come from the code object @p kernels_type_
 *  hands out through its static @c bytes, compiled under @p capability_.
 */
template <                                                                     //
    nk_dtype_t vector_dtype_,                                                  //
    nk_capability_t capability_,                                               //
    typename kernels_type_,                                                    //
    typename vectors_allocator_type_ = hip_virtual_memory_allocator<byte_t>,   //
    typename neighbors_allocator_type_ = hip_virtual_memory_allocator<byte_t>, //
    typename traversals_allocator_type_ = hip_arena_allocator_t>               //
struct vram_amd_index : vram_index<vram_amd_index<vector_dtype_, capability_, kernels_type_, vectors_allocator_type_,
                                                  neighbors_allocator_type_, traversals_allocator_type_>,
                                   hip_vendor_t, vector_dtype_, vectors_allocator_type_, neighbors_allocator_type_,
                                   traversals_allocator_type_> {

    using base_t = vram_index<vram_amd_index, hip_vendor_t, vector_dtype_, vectors_allocator_type_,
                              neighbors_allocator_type_, traversals_allocator_type_>;
    using base_t::arena_shape_for;
    using base_t::base_t;
    using base_t::build_scratch_bytes_per_warp;
    using base_t::connectivity;
    using base_t::dimensions;
    using base_t::non_zero_expansion_factor;
    using base_t::packed_row_bytes;
    using base_t::row_bytes;
    using base_t::row_fragments;
    using base_t::size;
    using base_t::status;
    using base_t::storage;
    using base_t::traversal_memory_usage;
    using typename base_t::distance_t;
    using typename base_t::dot_result_t;
    using typename base_t::dot_t;
    using typename base_t::frontier_kind_t;
    using typename base_t::node_id_t;
    using typename base_t::norms_t;
    using typename base_t::refine_config_t;
    using typename base_t::scalar_t;
    using typename base_t::search_config_t;
    using typename base_t::search_result_t;
    using typename base_t::traversal_context_t;

    /** The capability of the quads kit the ladder picked under @c capability_, surfaced in the
     *  bench reports; read only where the AMD kits are visible, as its initializer is. */
    inline static constexpr nk_capability_t picked_quads_capability_k = quads_ladder<scalar_t, capability_>::picked_k;

    /** Lanes cooperating on one row: the smallest power of two covering its fragments, at most a
     *  warp. */
    u32_t lanes_per_row() const noexcept { return min_of(std::bit_ceil(row_fragments()), wave_lanes()); }

    /** Lanes of this class's wavefronts: 32 on CDNA5, 64 before it. */
    static constexpr u32_t wave_lanes() noexcept { return hip_wave_lanes(capability_); }

    /** Threads per block, four wavefronts. */
    u32_t block_threads() const noexcept { return vram_warps_per_block_k * wave_lanes(); }

    /** Whether a search, multiply or attend launch seats on this device: per warp, the query span
     *  and the @p fold_bytes an attend keeps past it, beside the static batch buffers. Defined in
     *  @c vram_amd_index.hip, where the kits are visible. */
    bool walk_scratch_fits(std::size_t fold_bytes = 0) const noexcept;

    /** Whether a builder launch seats @p scratch_bytes_per_warp of per-warp scratch in shared
     *  memory beside the batch buffers and schedule state, on every block a multiprocessor keeps
     *  resident; when it cannot, the warp's arena slice keeps it, as Apple's always does. Defined
     *  in @c vram_amd_index.hip. */
    bool build_scratch_fits(u32_t scratch_bytes_per_warp) const noexcept;

    /** The bytes an executor's run must hold before a context carved over it can search or multiply
     *  @p rows rows under @p config, landing @c config.count columns per row, one aligned term per
     *  carve the verb makes. */
    std::size_t traversal_memory_usage(std::size_t rows, search_config_t config) const noexcept {
        static_assert(sizeof(dot_t) == sizeof(distance_t));
        if (!dimensions() || !device_.multiprocessor_count) return 0;
        auto aligned = [](std::size_t bytes) noexcept {
            return divide_round_up<std::size_t>(bytes, default_alignment_k) * default_alignment_k;
        };
        // Search widens the candidates to cover its count while multiply walks the bare factor, so
        // the wider one sizes.
        u32_t const expansion = max_of<u32_t>(gpu_walk_breadth(config.expansion_factor, config.count),
                                              non_zero_expansion_factor(config));
        bool const wide = expansion > vram_widest_candidate_slots_k * wave_lanes();
        u32_t const count_slots = divide_round_up<u32_t>(config.count, 32);
        u32_t const candidate_slots = candidate_slots_for_(expansion, count_slots > 2 ? count_slots : 2);
        gpu_arena_shape_t const arena_shape = arena_shape_for(
            walk_selection_rows_(expansion, candidate_slots),
            wide ? frontier_kind_t::wide_arena_k : frontier_kind_t::register_ladder_k);
        // The walks seat as many blocks as a search on the same rung, an attend no more.
        u32_t const resident_blocks = resident_blocks_(vram_kernel_t::search_k, wide ? 0 : candidate_slots);
        std::size_t const arena_warps = std::size_t {walk_blocks_(rows, resident_blocks)} * vram_warps_per_block_k;
        std::size_t const staging_bytes = rows * row_bytes() + import_bytes_(rows, nk_f64_k) +
                                          rows * config.count * (sizeof(node_id_t) + sizeof(distance_t)) +
                                          rows * sizeof(u32_t);
        return aligned(sizeof(vram_telemetry_block_t)) + 2 * aligned(sizeof(u32_t)) +
               aligned(arena_warps * arena_shape.slice_bytes<node_id_t>()) + aligned(staging_bytes);
    }

  private:
    friend base_t;

    /** The sharded index walks its shards under one snapshot of their watermarks, merges their
     *  attends on its first device, and publishes or fails their batches together. */
    template <typename, nk_dtype_t, nk_capability_t, typename, typename>
    friend struct vram_amd_indexes;
    template <typename, typename, typename, typename, typename, template <typename> class>
    friend struct sharded_indexes;

    using base_t::build_arena_shape_;
    using base_t::builder_scratch_bytes_per_warp_;
    using base_t::candidate_slots_for_;
    using base_t::device_;
    using base_t::fail_;
    using base_t::import_bytes_;
    using base_t::import_rows_;
    using base_t::kernel_;
    using base_t::land_counters_;
    using base_t::neighbor_rows_;
    using base_t::neighbors_matrix_;
    using base_t::reselection_rows_;
    using base_t::rows_refusal_;
    using base_t::vectors_matrix_;
    using base_t::walk_blocks_;
    using base_t::walk_selection_rows_;
    using typename base_t::attend_heads_t;
    using typename base_t::audit_landing_t;
    using typename base_t::refine_scratch_t;

#pragma region Search

    /** What every walking verb stages ahead of its sink: the device view, the frontier the walks
     *  keep, the staging past the uploads, and the rescoring the walks score under. */
    struct walk_launch_t {

        /** The device view every walk kernel reads. */
        vram_search_t view;

        /** The register ladder's width, when the ladder serves. */
        u32_t candidate_slots;

        /** Which frontier the walks keep. */
        frontier_kind_t frontier;

        /** The staging bytes past the queries and any heads, where the sink writes its outputs. */
        byte_t* outputs;

        /** The rescoring of every walk, over the heads staged beside the queries. */
        vram_rescoring_t rescoring;
    };

    /** Validates and uploads @p queries, then any indexer or attend @p heads and the indexer's
     *  weights, sizes the arena for @p expansion, and reserves @p outputs_bytes of staging past
     *  them: everything a walking verb stages before its sink, the walks bound below the @p visible
     *  rows. Until the context drains, the queries, the heads and the context stay live. */
    template <typename heads_type_>
    expected<walk_launch_t> stage_walk_(rows_view_t queries, [[maybe_unused]] heads_type_ const& heads,
                                        traversal_context_t& context, u32_t expansion, u32_t minimum_slots,
                                        std::size_t outputs_bytes, std::size_t visible) noexcept {
        constexpr bool indexer_k = std::is_same_v<heads_type_, indexer_heads<scalar_t>>;
        constexpr bool attend_k = std::is_same_v<heads_type_, attend_heads_t>;
        std::size_t const queries_count = queries.rows;
        if (!vectors_matrix_.reserved_bytes()) return {{}, status_t::uninitialized_k};
        if (context.device() != device_.ordinal || !context) return {{}, status_t::config_mismatch_k};
        // Validated with the rest: past here a refusal would already have burned epoch space.
        if (status_t const refusal = rows_refusal_(queries); failed(refusal)) return {{}, refusal};
        std::size_t heads_rows = 0;
        if constexpr (indexer_k) {
            heads_rows = heads.heads.extent(0);
            if (heads.weights.extent(0) != queries_count || heads_rows != queries_count * heads.weights.extent(1))
                return {{}, status_t::config_mismatch_k};
            if (status_t const refusal = rows_refusal_(heads.heads); failed(refusal)) return {{}, refusal};
        }
        if constexpr (attend_k) {
            heads_rows = heads.rows.rows;
            if (status_t const refusal = rows_refusal_(heads.rows); failed(refusal)) return {{}, refusal};
        }
        hip_stream_handle_t const stream = context.stream();

        // Past the register ladder's widest rung the arena-backed walk serves.
        frontier_kind_t const frontier = expansion > vram_widest_candidate_slots_k * wave_lanes()
                                             ? frontier_kind_t::wide_arena_k
                                             : frontier_kind_t::register_ladder_k;
        u32_t const candidate_slots = candidate_slots_for_(expansion, minimum_slots);
        gpu_arena_shape_t const arena_shape = arena_shape_for(walk_selection_rows_(expansion, candidate_slots),
                                                              frontier);
        // A search, an indexed search and a multiply seat as many blocks on one rung.
        u32_t const resident_blocks = resident_blocks_(attend_k ? vram_kernel_t::attend_k : vram_kernel_t::search_k,
                                                       frontier == frontier_kind_t::wide_arena_k ? 0 : candidate_slots);
        // Every warp the device seats on this rung, and the rows whose neighbors fill a walk's
        // candidates.
        u32_t const seats = device_.multiprocessor_count * resident_blocks * vram_warps_per_block_k;
        u32_t const filling = filling_rows<u32_t>(expansion, static_cast<u32_t>(connectivity()));
        std::size_t const arena_warps = std::size_t {walk_blocks_(queries_count, resident_blocks)} *
                                        vram_warps_per_block_k;
        std::size_t const arena_bytes_before = context.arena_bytes_;
        byte_t* const arena = context.ensure_arena(arena_warps * arena_shape.slice_bytes<node_id_t>());
        if (!arena) return {{}, status_t::bad_alloc_k};
        // The staging holds the query rows and any head rows first, since they carry the widest
        // alignment, then the window importing them, the heads' weights, then the outputs; stream
        // order serializes its reuse across calls.
        std::size_t const rows_bytes = (queries_count + heads_rows) * row_bytes();
        std::size_t const import_bytes = import_bytes_(attend_k ? heads_rows : queries_count, queries.dtype);
        std::size_t const weights_bytes = indexer_k ? heads_rows * sizeof(f32_t) : 0;
        byte_t* const staging = context.ensure_staging(rows_bytes + import_bytes + weights_bytes + outputs_bytes);
        if (!staging) return {{}, status_t::bad_alloc_k};
        byte_t* const window = staging + rows_bytes;
        auto const [epoch_base, epochs_status] = context.take_epochs(queries_count, arena_bytes_before);
        if (failed(epochs_status)) return {{}, epochs_status};
        // Pitched uploads: device rows pad to whole fragments, and the pad bytes must be zero for
        // the zero-tail distance identity to hold.
        if ((import_bytes || packed_row_bytes() != row_bytes()) &&
            hip_api().fill_async(staging, 0, rows_bytes + import_bytes, stream) != hip_success_k)
            return {{}, status_t::device_error_k};
        if (status_t const imported = import_rows_(staging, queries, window, stream); failed(imported))
            return {{}, imported};
        if (hip_api().fill_async(context.telemetry_.get(), 0, sizeof(vram_telemetry_block_t), stream) !=
                hip_success_k ||
            hip_api().fill_async(context.assigned_rows_.get(), 0, 2 * sizeof(u32_t), stream) != hip_success_k)
            return {{}, status_t::device_error_k};

        vram_rescoring_t rescoring {};
        if constexpr (indexer_k) {
            byte_t* const staged_heads = staging + queries_count * row_bytes();
            byte_t* const staged_weights = staging + rows_bytes + import_bytes;
            std::size_t const weights_row_bytes = heads.weights.extent(1) * sizeof(f32_t);
            if (failed(import_rows_(staged_heads, heads.heads, window, stream)) ||
                hip_api().copy_2d_async(staged_weights, weights_row_bytes, heads.weights.byte_data(),
                                        heads.weights.stride_bytes(0), weights_row_bytes, queries_count,
                                        hip_copy_default_k, stream) != hip_success_k)
                return {{}, status_t::device_error_k};
            rescoring = {staged_heads, reinterpret_cast<f32_t const*>(staged_weights),
                         static_cast<u32_t>(heads.weights.extent(1)), row_fragments()};
        }
        if constexpr (attend_k) {
            byte_t* const staged_heads = staging + queries_count * row_bytes();
            if (status_t const imported = import_rows_(staged_heads, heads.rows, window, stream); failed(imported))
                return {{}, imported};
            rescoring = {staged_heads, nullptr, static_cast<u32_t>(heads.head_count), row_fragments()};
        }

        vram_search_t const view {storage(),
                                  staging,
                                  static_cast<u32_t>(queries_count),
                                  static_cast<u32_t>(visible),
                                  static_cast<u32_t>(row_bytes()),
                                  expansion,
                                  arena,
                                  arena_shape,
                                  epoch_base,
                                  context.telemetry_.get(),
                                  context.assigned_rows_.get(),
                                  context.assigned_rows_.get() + 1,
                                  seats,
                                  filling};
        return {{view, candidate_slots, frontier, staging + rows_bytes + import_bytes + weights_bytes, rescoring},
                status_t::success_k};
    }

    /** The submission half of @c search: stages the queries and @p heads, launches walks over the
     *  @p visible rows, and queues the pitched writeback. Until the context drains, the queries,
     *  the heads, both outputs and the context stay live; across contexts they must not share a
     *  buffer at all. */
    template <typename heads_type_ = dummy_heads_t>
    expected<search_result_t> submit_search_(rows_view_t queries, nk::matrix_span<node_id_t> result_ids,
                                             nk::matrix_span<distance_t> result_distances, traversal_context_t& context,
                                             search_config_t config, std::size_t visible,
                                             heads_type_ const& heads = {}) noexcept {
        std::size_t const queries_count = queries.rows;
        if (config.traversal != traversal_t::walk_k) return {{}, status_t::unsupported_k};
        if (!queries_count) return {{}, status_t::success_k};
        bool const wants_distances = result_distances.extent(0) != 0;
        if (result_ids.extent(0) != queries_count || result_ids.extent(1) < config.count ||
            (wants_distances &&
             (result_distances.extent(0) != queries_count || result_distances.extent(1) < config.count)))
            return {{}, status_t::config_mismatch_k};
        if (result_ids.stride_bytes(0) < static_cast<std::ptrdiff_t>(config.count * sizeof(node_id_t)) ||
            (wants_distances &&
             result_distances.stride_bytes(0) < static_cast<std::ptrdiff_t>(config.count * sizeof(distance_t))))
            return {{}, status_t::unsupported_layout_k};

        // The expansion covers the count, so the ladder's candidates hold every rank, and past it
        // the arena frontier ranks everything it saw.
        u32_t const expansion = gpu_walk_breadth(config.expansion_factor, config.count);
        // The candidates must cover the requested count, whatever the expansion asked for.
        u32_t const count_slots = divide_round_up<u32_t>(config.count, 32);
        std::size_t const results_count = queries_count * config.count;
        std::size_t const ids_bytes = results_count * sizeof(node_id_t);
        std::size_t const distances_bytes = wants_distances ? results_count * sizeof(distance_t) : 0;
        auto const staged = stage_walk_(queries, heads, context, expansion, count_slots > 2 ? count_slots : 2,
                                        ids_bytes + distances_bytes, visible);
        if (!staged) return {{}, staged.status};
        node_id_t* const device_ids = reinterpret_cast<node_id_t*>(staged.value.outputs);
        distance_t* const device_distances = wants_distances
                                                 ? reinterpret_cast<distance_t*>(staged.value.outputs + ids_bytes)
                                                 : nullptr;
        hip_stream_handle_t const stream = context.stream();
        if (status_t const launched = launch_walk_(
                staged.value, vram_search_sink_t {device_ids, device_distances, config.count, kernel_}, stream);
            failed(launched))
            return {{}, launched};
        // The landing may be wider than the count, so a multi-index can hand every shard its own
        // columns of one results matrix.
        if (hip_api().copy_2d_async(result_ids.data(), result_ids.stride_bytes(0), device_ids,
                                    config.count * sizeof(node_id_t), config.count * sizeof(node_id_t), queries_count,
                                    hip_copy_default_k, stream) != hip_success_k)
            return {{}, status_t::device_error_k};
        if (wants_distances &&
            hip_api().copy_2d_async(result_distances.data(), result_distances.stride_bytes(0), device_distances,
                                    config.count * sizeof(distance_t), config.count * sizeof(distance_t), queries_count,
                                    hip_copy_default_k, stream) != hip_success_k)
            return {{}, status_t::device_error_k};
        return {{config.count}, status_t::success_k};
    }

    /** The submission half of @c multiply: stages the rows, launches walks over the @p visible
     *  rows, and queues the writeback of the kept pairs and the scored counts. */
    expected<search_result_t> submit_multiply_(rows_view_t left, nk::matrix_span<node_id_t> node_ids,
                                               nk::matrix_span<dot_result_t> products, std::span<u32_t> counts,
                                               traversal_context_t& context, search_config_t config,
                                               std::size_t visible) noexcept {
        std::size_t const rows_count = left.rows;
        std::size_t const wanted = node_ids.extent(1);
        if (config.traversal != traversal_t::walk_k) return {{}, status_t::unsupported_k};
        if (node_ids.extent(0) != rows_count || products.extent(0) != rows_count || products.extent(1) != wanted ||
            counts.size() != rows_count)
            return {{}, status_t::config_mismatch_k};
        if (!rows_count) return {{}, status_t::success_k};
        if (node_ids.stride_bytes(0) < static_cast<std::ptrdiff_t>(wanted * sizeof(node_id_t)) ||
            products.stride_bytes(0) < static_cast<std::ptrdiff_t>(wanted * sizeof(dot_result_t)))
            return {{}, status_t::unsupported_layout_k};
        std::size_t const ids_bytes = rows_count * wanted * sizeof(node_id_t);
        std::size_t const products_bytes = rows_count * wanted * sizeof(dot_t);
        auto const staged = stage_walk_(left, dummy_heads_t {}, context, non_zero_expansion_factor(config), 2,
                                        ids_bytes + products_bytes + rows_count * sizeof(u32_t), visible);
        if (!staged) return {{}, staged.status};
        node_id_t* const device_ids = reinterpret_cast<node_id_t*>(staged.value.outputs);
        byte_t* const device_products = staged.value.outputs + ids_bytes;
        u32_t* const device_counts = reinterpret_cast<u32_t*>(staged.value.outputs + ids_bytes + products_bytes);
        hip_stream_handle_t const stream = context.stream();
        vram_multiply_sink_t const sink {device_ids, device_products, device_counts, static_cast<u32_t>(wanted)};
        if (status_t const launched = launch_walk_(staged.value, sink, stream); failed(launched)) return {{}, launched};
        if (wanted &&
            (hip_api().copy_2d_async(node_ids.data(), node_ids.stride_bytes(0), device_ids, wanted * sizeof(node_id_t),
                                     wanted * sizeof(node_id_t), rows_count, hip_copy_default_k,
                                     stream) != hip_success_k ||
             hip_api().copy_2d_async(products.data(), products.stride_bytes(0), device_products, wanted * sizeof(dot_t),
                                     wanted * sizeof(dot_t), rows_count, hip_copy_default_k, stream) != hip_success_k))
            return {{}, status_t::device_error_k};
        if (hip_api().copy_async(counts.data(), device_counts, rows_count * sizeof(u32_t), hip_copy_default_k,
                                 stream) != hip_success_k)
            return {{}, status_t::device_error_k};
        return {{}, status_t::success_k};
    }

    /** The submission half of @c attend: stages each token's first head to navigate by, then every
     *  head, and launches one walk per token over the @p visible rows, which lands the normalized
     *  rows, then the carries, in the first of @p blocks such blocks of staging, returned
     *  device-resident, with @p exported_bytes more past them for the rows cast on their way out;
     *  the sharded index gathers its other shards' into the rest. */
    template <typename value_lookup_type_>
    expected<f32_t*> submit_attend_(rows_view_t queries, std::size_t head_count, value_lookup_type_ const& value_lookup,
                                    f32_t score_scale, traversal_context_t& context, search_config_t config,
                                    std::size_t blocks, std::size_t exported_bytes, std::size_t visible) noexcept {
        std::size_t const rows_count = queries.rows;
        if (config.traversal != traversal_t::walk_k) return {nullptr, status_t::unsupported_k};
        if (!head_count || rows_count % head_count) return {nullptr, status_t::shape_mismatch_k};
        if (!rows_count) return {nullptr, status_t::success_k};
        std::size_t const scratch_bytes = vram_attend_scratch_bytes(head_count, dimensions(), wave_lanes());
        if (!walk_scratch_fits(scratch_bytes)) return {nullptr, status_t::config_mismatch_k};
        std::size_t const outputs_bytes = rows_count * dimensions() * sizeof(f32_t);
        rows_view_t const first_heads {queries.data, rows_count / head_count, queries.columns,
                                       queries.stride * head_count, queries.dtype};
        auto staged = stage_walk_(
            first_heads, attend_heads_t {queries, head_count}, context, non_zero_expansion_factor(config), 2,
            blocks * (outputs_bytes + rows_count * sizeof(softmax_state_t)) + exported_bytes, visible);
        if (!staged) return {nullptr, staged.status};
        staged.value.view.scratch_bytes_per_warp += static_cast<u32_t>(scratch_bytes);
        if (hip_api().fill_async(context.missed_.get(), 0, sizeof(u32_t), context.stream()) != hip_success_k)
            return {nullptr, status_t::device_error_k};
        f32_t* const device_outputs = reinterpret_cast<f32_t*>(staged.value.outputs);
        if (status_t const launched = launch_attend_(
                staged.value, value_lookup, device_outputs,
                reinterpret_cast<softmax_state_t*>(staged.value.outputs + outputs_bytes),
                score_scale * std::numbers::log2e_v<f32_t>, context);
            failed(launched))
            return {nullptr, launched};
        return {device_outputs, status_t::success_k};
    }

#pragma endregion Search

#pragma region Refinement

    /** Queues one pass of the three-pass refinement engine behind @c refine_plan_t into its carved
     *  @p scratch: the walk covers @p nodes, or under a device-resident @p node_ids the listed IDs
     *  with @p nodes spanning the array's offsets, each node first walking across the neighboring
     *  group in round @p round when @p parts parts are given; the back-link passes re-select the
     *  walked rows and the rows they pick, and leave every other row as it was. Nothing waits: the
     *  landing queued after it checks its claims. */
    status_t refine_pass_(traversal_context_t& context, refine_scratch_t const& scratch, refine_config_t config,
                          iota_view<node_id_t> nodes, std::span<u32_t const> node_ids, u32_t parts = 0,
                          u32_t round = 0) noexcept {
        if (status_t const kept = status(); failed(kept)) return kept;
        std::size_t const nodes_count = size();
        std::size_t const walk_count = nodes.size();
        hip_stream_handle_t const stream = context.stream();
        // The claims, the reverse buckets' counters and the bitset of rows the merge re-selects lie
        // back to back.
        if (hip_api().fill_async(scratch.claims, 0, (2 + nodes_count + scratch.words) * sizeof(u32_t), stream) !=
            hip_success_k)
            return status_t::device_error_k;

        // Twice the connectivity by default, like every CPU walk, rounded up to the candidate sizes
        // the kernel carries, or walked exactly past them.
        u32_t const expansion = non_zero_expansion_factor(config);
        u32_t const candidate_slots = candidate_slots_for_(expansion, 4);
        // Walking across makes two walks into one visited set, so the slice is shaped for both.
        u32_t const walks = parts > 1 ? 2 : 1;
        gpu_arena_shape_t const arena_shape = build_arena_shape_(expansion, candidate_slots, walks);
        std::size_t const arena_warps = device_.multiprocessor_count *
                                        resident_blocks_(vram_kernel_t::refine_k, candidate_slots) *
                                        vram_warps_per_block_k;
        std::size_t const arena_bytes_before = context.arena_bytes_;
        byte_t* const arena = context.ensure_arena(arena_warps * arena_shape.slice_bytes<node_id_t>());
        if (!arena) return status_t::bad_alloc_k;
        auto const [epoch_base, epochs_status] = context.take_epochs(walk_count, arena_bytes_before);
        if (failed(epochs_status)) return epochs_status;

        u32_t const selection_rows = walk_selection_rows_(expansion, candidate_slots);
        vram_refine_t view {storage(), //
                            scratch.shadow_neighbors,
                            nodes.bound,
                            arena_shape.scratch_bytes ? 0u : build_scratch_bytes_per_warp(selection_rows),
                            selection_rows,
                            nodes.first,
                            arena,
                            arena_shape,
                            expansion,
                            epoch_base,
                            context.telemetry_.get(),
                            scratch.claims,
                            node_ids.data(),
                            static_cast<u32_t>(nodes_count),
                            parts > 1 ? scratch.part_bases : nullptr,
                            parts > 1 ? parts : 0,
                            round};
        // A range short of the whole graph starts from the live neighbors copied into the shadow
        // verbatim, so a row it never rewrites but one of its picks lands in merges its own
        // neighbors with the new reverse links; islands scatter nothing, so only their own rows
        // merge.
        if (node_ids.empty() && walk_count != nodes_count)
            if (hip_api().copy_async(scratch.shadow_neighbors, neighbors_matrix_.base(),
                                     nodes_count * connectivity() * sizeof(u32_t), hip_copy_device_to_device_k,
                                     stream) != hip_success_k)
                return status_t::device_error_k;
        if (status_t const launched = launch_refine_(view, candidate_slots, stream); failed(launched)) return launched;

        // Scatter the walked rows' shadow neighbors in reverse, then merge each touched node's
        // forward & reverse candidates into its final live row. A heal round over island IDs
        // re-selects the islands' own rows alone: re-judging the rows they pick would drop the
        // links that reconnected earlier islands.
        return launch_refine_back_links_(view, scratch, static_cast<u32_t>(walk_count), stream);
    }

    /** Queues a reachability audit from as many stride-spread landmarks as one search draws - the
     *  connectivity - and its first sweep, through @ref queue_sweeps_. Nothing waits; the caller
     *  reads @p landing once its event fired, and queues further sweeps from sweep 2 while the last
     *  one still expanded. */
    status_t queue_audit_(traversal_context_t& context, refine_scratch_t const& scratch,
                          audit_landing_t* landing) noexcept {
        std::size_t const rows = size();
        hip_stream_handle_t const stream = context.stream();
        std::size_t const seed_count = min_of<std::size_t>(connectivity(), rows);
        // The bitsets, the levels, the sweeps' expanded counters and the unreached counter lie back
        // to back.
        if (hip_api().fill_async(scratch.reached.words, 0, (scratch.words * 3 + 4) * sizeof(u32_t), stream) !=
            hip_success_k)
            return status_t::device_error_k;
        vram_arguments_t seeds {};
        seeds.graph.neighbors = neighbor_rows_();
        seeds.graph.rows_count = static_cast<u32_t>(rows);
        seeds.graph.connectivity = static_cast<u32_t>(connectivity());
        seeds.graph.seed_count = static_cast<u32_t>(seed_count);
        seeds.graph.reached = scratch.reached;
        seeds.graph.frontier = scratch.frontier;
        if (status_t const launched = launch_(vram_kernel_t::mark_landmarks_k, 0,
                                              divide_round_up<u32_t>(static_cast<u32_t>(seed_count), block_threads()),
                                              0, stream, seeds);
            failed(launched))
            return launched;
        return queue_sweeps_(context, scratch, landing, 1);
    }

    /** Queues the audit's sweeps from @p first_sweep up to twice it, so the batches double until
     *  one ends past convergence, then a count of the rows no sweep reached so far, collecting
     *  their IDs when @p scratch has room for them, then the landing of every counter, the last
     *  sweep's expanded count among them. */
    status_t queue_sweeps_(traversal_context_t& context, refine_scratch_t const& scratch, audit_landing_t* landing,
                           u32_t first_sweep) noexcept {
        std::size_t const rows = size();
        hip_stream_handle_t const stream = context.stream();
        vram_arguments_t sweep {};
        sweep.graph.neighbors = neighbor_rows_();
        sweep.graph.rows_count = static_cast<u32_t>(rows);
        sweep.graph.connectivity = static_cast<u32_t>(connectivity());
        sweep.graph.reached = scratch.reached;
        sweep.graph.levels = scratch.levels;
        sweep.graph.island_ids = scratch.island_ids;
        // A sweep queued past convergence finds the one before it expanded nothing and leaves at
        // entry; odd sweeps grow the first frontier, even ones the second.
        u32_t const grid = device_.multiprocessor_count * resident_blocks_(vram_kernel_t::reachability_sweep_k, 0);
        for (u32_t sweep_number = first_sweep; sweep_number != 2 * first_sweep; ++sweep_number) {
            bool const odd = sweep_number & 1u;
            sweep.graph.frontier = odd ? scratch.frontier : scratch.next_frontier;
            sweep.graph.next_frontier = odd ? scratch.next_frontier : scratch.frontier;
            sweep.graph.counter = scratch.expanded + odd;
            sweep.graph.previous_counter = sweep_number == 1 ? nullptr : scratch.expanded + !odd;
            if (hip_api().fill_async(sweep.graph.next_frontier.words, 0, scratch.words * sizeof(u32_t), stream) !=
                    hip_success_k ||
                hip_api().fill_async(sweep.graph.counter, 0, sizeof(u32_t), stream) != hip_success_k)
                return status_t::device_error_k;
            if (status_t const launched = launch_(vram_kernel_t::reachability_sweep_k, 0, grid, 0, stream, sweep);
                failed(launched))
                return launched;
        }
        sweep.graph.counter = scratch.unreached;
        if (hip_api().fill_async(scratch.unreached, 0, sizeof(u32_t), stream) != hip_success_k)
            return status_t::device_error_k;
        if (status_t const launched = launch_(vram_kernel_t::reachability_islands_k, 0,
                                              divide_round_up<u32_t>(static_cast<u32_t>(rows), block_threads()), 0,
                                              stream, sweep);
            failed(launched))
            return launched;
        return land_counters_(scratch, landing, 2 * first_sweep - 1, stream);
    }

    /** Queues the reconnection of the @p islands islands the last audit in @p scratch collected
     *  over @p rows rows: the reached in-links first, so no reconnection takes the last link a
     *  reached node is found through, then each island into a reached pick or its landmark. */
    status_t queue_reconnect_(refine_scratch_t const& scratch, u32_t islands, std::size_t rows,
                              hip_stream_handle_t stream) noexcept {
        if (hip_api().fill_async(scratch.in_links, 0, rows * sizeof(u32_t), stream) != hip_success_k)
            return status_t::device_error_k;
        vram_arguments_t heal {};
        heal.graph.neighbors = neighbor_rows_();
        heal.graph.rows_count = static_cast<u32_t>(rows);
        heal.graph.connectivity = static_cast<u32_t>(connectivity());
        heal.graph.reached = scratch.reached;
        heal.graph.in_links = scratch.in_links;
        heal.graph.island_ids = scratch.island_ids;
        heal.graph.islands_count = islands;
        heal.graph.seed_count = static_cast<u32_t>(min_of<std::size_t>(connectivity(), rows));
        heal.graph.reconnected = scratch.reconnected;
        if (status_t const counted = launch_(
                vram_kernel_t::reached_in_links_k, 0,
                device_.multiprocessor_count * resident_blocks_(vram_kernel_t::reached_in_links_k, 0), 0, stream, heal);
            failed(counted))
            return counted;
        return launch_(vram_kernel_t::reconnect_island_k, 0, divide_round_up<u32_t>(islands, block_threads()), 0,
                       stream, heal);
    }

#pragma endregion Refinement

#pragma region Launches

    /** The kernels the base names when it sizes an arena for their grid. */
    using kernel_t = vram_kernel_t;

    /** Blocks of @p kernel on its @p slots rung resident per compute unit, as its launch bounds
     *  read them; every grid and the arena behind it are sized by it. */
    static constexpr u32_t resident_blocks_(vram_kernel_t kernel, u32_t slots) noexcept {
        return hip_resident_blocks<capability_, vector_dtype_>(kernel, slots);
    }

    /** Loads the code object of @p kernels_type_ onto the device and finds every kernel this dtype
     *  exports bar the attends, by the symbol @ref vram_kernel_symbol spells. */
    status_t load_kernels_() noexcept {
        auto [module, status] = hip_module_t::load(device_.ordinal, kernels_type_::bytes());
        if (failed(status)) return status;
        module_ = std::move(module);
        vram_kernel_symbol_t symbol;
        for (amd_rung_t const rung : amd_rungs) {
            vram_kernel_symbol(symbol, "dense", "amd_kernel", "vram_arguments_t", rung.kernel, capability_,
                               vector_dtype_, rung.slots, u32_t {0});
            hip_function_handle_t& function = kernels_[static_cast<u32_t>(rung.kernel)][gpu_rung_index(rung.slots)];
            if (!(function = module_.function(symbol.text))) return status_t::device_error_k;
        }
        return status_t::success_k;
    }

    /** Launches @p kernel on its @p slots rung over @p blocks of four wavefronts on @p stream. */
    status_t launch_(vram_kernel_t kernel, u32_t slots, u32_t blocks, u32_t shared_bytes, hip_stream_handle_t stream,
                     vram_arguments_t const& arguments) const noexcept {
        return hip_launch(kernels_[static_cast<u32_t>(kernel)][gpu_rung_index(slots)], blocks, block_threads(),
                          shared_bytes, stream, arguments);
    }

    /** Launches the rows of @p build in waves, each queued behind the last and as wide as the IDs
     *  below it, never narrower than @c vram_admission_ramp_floor_k, so every ID a wave admits
     *  walks rows already published, and each followed by the back-link launch that links its
     *  newcomers from the nodes they picked. A wave failing after an earlier one launched keeps
     *  the failure, since those rows can no longer be un-claimed. */
    status_t launch_insert_(vram_build_t const& build, u32_t candidate_slots, traversal_context_t& context) noexcept {
        hip_stream_handle_t const stream = context.stream();
        vram_arguments_t arguments {};
        // Both launches carve their warps' slices from the one arena, sized for the insert's grid,
        // which the back-link's resident count matches.
        static_assert(resident_blocks_(vram_kernel_t::back_link_k, 0) == resident_blocks_(vram_kernel_t::insert_k, 0));
        u32_t const blocks = device_.multiprocessor_count * resident_blocks_(vram_kernel_t::insert_k, candidate_slots);
        u32_t const back_link_rows = reselection_rows_();
        u32_t const back_link_scratch = build.arena_shape.scratch_bytes ? 0u
                                                                        : build_scratch_bytes_per_warp(back_link_rows);
        auto zeroed = [&](u32_t* counter) noexcept {
            return hip_api().fill_async(counter, 0, sizeof(u32_t), stream) == hip_success_k;
        };
        for (u32_t wave_first = build.first_id; wave_first != build.end_id;) {
            u32_t const wave_end = static_cast<u32_t>(
                vram_admission_wave_end(wave_first, build.end_id, vram_dense_admission_share_log2_k));
            u32_t const offset = wave_first - build.first_id;
            arguments.build = build;
            arguments.build.staging = build.staging + std::size_t {offset} * row_bytes();
            arguments.build.first_id = wave_first;
            arguments.build.end_id = wave_end;
            arguments.build.epoch_base = build.epoch_base + offset;
            status_t launched = zeroed(build.assigned_rows) && zeroed(build.touched_count) ? status_t::success_k
                                                                                           : status_t::device_error_k;
            if (succeeded(launched))
                launched = launch_(vram_kernel_t::insert_k, candidate_slots, blocks,
                                   vram_warps_per_block_k * build.scratch_bytes_per_warp, stream, arguments);
            if (failed(launched)) return offset ? fail_(launched) : launched;
            // The wave's rows are queued now, so a failure past here can no longer be un-claimed.
            arguments.build.scratch_bytes_per_warp = back_link_scratch;
            arguments.build.selection_rows = back_link_rows;
            if (!zeroed(build.assigned_rows)) return fail_(status_t::device_error_k);
            if (status_t const linked = launch_(vram_kernel_t::back_link_k, 0, blocks,
                                                vram_warps_per_block_k * back_link_scratch, stream, arguments);
                failed(linked))
                return fail_(linked);
            wave_first = wave_end;
        }
        return status_t::success_k;
    }
    status_t launch_refine_(vram_refine_t const& view, u32_t candidate_slots, hip_stream_handle_t stream) noexcept {
        vram_arguments_t arguments {};
        arguments.refine = view;
        return launch_(vram_kernel_t::refine_k, candidate_slots,
                       device_.multiprocessor_count * resident_blocks_(vram_kernel_t::refine_k, candidate_slots),
                       vram_warps_per_block_k * view.scratch_bytes_per_warp, stream, arguments);
    }

    /** Scatters the @p walk_count walked rows' shadow neighbors into the reverse buckets of
     *  @p scratch, then merges every touched row's forward and reverse candidates into its live
     *  neighbors. */
    status_t launch_refine_back_links_(vram_refine_t const& view, refine_scratch_t const& scratch, u32_t walk_count,
                                       hip_stream_handle_t stream) noexcept {
        vram_arguments_t arguments {};
        vram_graph_t& graph = arguments.graph;
        graph.neighbors = scratch.shadow_neighbors;
        graph.rows_count = view.sampling_bound;
        graph.first_id = view.first_id;
        graph.walk_count = walk_count;
        graph.node_ids = view.node_ids;
        graph.connectivity = static_cast<u32_t>(connectivity());
        graph.bucket_width = static_cast<u32_t>(connectivity());
        graph.bucket_ids = scratch.buckets;
        graph.bucket_counts = scratch.bucket_counts;
        graph.touched = scratch.touched;
        if (status_t const launched = launch_(
                vram_kernel_t::scatter_back_links_k, 0,
                device_.multiprocessor_count * resident_blocks_(vram_kernel_t::scatter_back_links_k, 0), 0, stream,
                arguments);
            failed(launched))
            return launched;
        // The merge sweeps the whole ID space and skips the rows the scatter left untouched,
        // carving its slices from the arena sized for the refine's grid, which its count matches.
        static_assert(resident_blocks_(vram_kernel_t::merge_back_links_k, 0) ==
                      resident_blocks_(vram_kernel_t::refine_k, 0));
        arguments.refine = view;
        arguments.refine.scratch_bytes_per_warp = view.arena_shape.scratch_bytes
                                                      ? 0u
                                                      : build_scratch_bytes_per_warp(reselection_rows_());
        arguments.refine.selection_rows = reselection_rows_();
        arguments.refine.assigned_rows = scratch.claims + 1;
        arguments.refine.node_ids = nullptr;
        arguments.refine.first_id = 0;
        arguments.refine.end_id = view.sampling_bound;
        return launch_(vram_kernel_t::merge_back_links_k, 0,
                       device_.multiprocessor_count * resident_blocks_(vram_kernel_t::merge_back_links_k, 0),
                       vram_warps_per_block_k * arguments.refine.scratch_bytes_per_warp, stream, arguments);
    }

    /** Rebuilds the norms of rows @b [first_row,first_row+rows) from their vectors. */
    status_t launch_derive_norms_(std::size_t first_row, std::size_t rows, hip_stream_handle_t stream) noexcept {
        vram_arguments_t arguments {};
        arguments.graph.storage = storage();
        arguments.graph.first_id = static_cast<u32_t>(first_row);
        arguments.graph.rows_count = static_cast<u32_t>(rows);
        return launch_(vram_kernel_t::derive_norms_k, 0,
                       device_.multiprocessor_count * resident_blocks_(vram_kernel_t::derive_norms_k, 0), 0, stream,
                       arguments);
    }

    /** Rebases rows @p rows of the neighbors matrix from @p source_range onto the IDs from @p shift
     *  and compacts them, as a load through the bounce does on the host. */
    status_t launch_rebase_links_(iota_view<node_id_t> rows, iota_view<node_id_t> source_range, node_id_t shift,
                                  hip_stream_handle_t stream) noexcept {
        vram_arguments_t arguments {};
        vram_graph_t& graph = arguments.graph;
        graph.neighbors = neighbor_rows_();
        graph.connectivity = static_cast<u32_t>(connectivity());
        graph.first_id = rows.first;
        graph.rows_count = static_cast<u32_t>(rows.size());
        graph.source_first = source_range.first;
        graph.source_bound = source_range.bound;
        graph.shift = shift;
        return launch_(vram_kernel_t::rebase_links_k, 0,
                       device_.multiprocessor_count * resident_blocks_(vram_kernel_t::rebase_links_k, 0), 0, stream,
                       arguments);
    }

    /** Merges the attended rows and carries of @p shards walks into the first's on @p stream - the
     *  sharded attend's gather. */
    status_t launch_merge_attention_(byte_t* blocks, u32_t rows, u32_t columns, u32_t shards,
                                     hip_stream_handle_t stream) noexcept {
        vram_arguments_t arguments {};
        arguments.attend.outputs = reinterpret_cast<f32_t*>(blocks);
        arguments.attend.rows = rows;
        arguments.attend.columns = columns;
        arguments.attend.shards = shards;
        return launch_(vram_kernel_t::merge_attention_k, 0, divide_round_up<u32_t>(rows, vram_warps_per_block_k), 0,
                       stream, arguments);
    }

    /** Resolves the @p walked matches per query of @p queries_count at @p ids and @p distances
     *  through the device @p keys column into @p ranking on @p stream - a sharded search's own
     *  ranking, gathered or landed as is. */
    status_t launch_resolve_matches_(node_id_t const* ids, distance_t const* distances, u32_t walked,
                                     byte_t const* keys, vram_ranking_t const& ranking, u32_t queries_count,
                                     hip_stream_handle_t stream) const noexcept {
        vram_arguments_t arguments {};
        arguments.matches.ids = ids;
        arguments.matches.distances = distances;
        arguments.matches.keys = keys;
        arguments.matches.ranking = ranking;
        arguments.matches.walked = walked;
        arguments.matches.queries_count = queries_count;
        return launch_(vram_kernel_t::resolve_matches_k, 0,
                       divide_round_up<u32_t>(queries_count, vram_warps_per_block_k), 0, stream, arguments);
    }

    /** Merges @p shards rankings of @p queries_count queries, gathered @p block_bytes apart from
     *  @p gathered, into @p ranking on @p stream - the sharded search's device gather. */
    status_t launch_merge_matches_(vram_ranking_t const& gathered, std::size_t block_bytes, u32_t shards,
                                   vram_ranking_t const& ranking, u32_t queries_count,
                                   hip_stream_handle_t stream) const noexcept {
        vram_arguments_t arguments {};
        arguments.matches.gathered = gathered;
        arguments.matches.ranking = ranking;
        arguments.matches.block_bytes = block_bytes;
        arguments.matches.queries_count = queries_count;
        arguments.matches.shards = shards;
        return launch_(vram_kernel_t::merge_matches_k, 0, divide_round_up<u32_t>(queries_count, vram_warps_per_block_k),
                       0, stream, arguments);
    }

    /** Casts @p count values at @p from in @p from_dtype into @p to in @p to_dtype on @p stream,
     *  through NumKong's plan and its per-unit device conversions. */
    status_t launch_cast_(void const* from, nk_dtype_t from_dtype, void* to, nk_dtype_t to_dtype, std::size_t count,
                          hip_stream_handle_t stream) const noexcept {
        vram_arguments_t arguments {};
        arguments.cast.from = static_cast<unsigned char const*>(from);
        arguments.cast.to = static_cast<unsigned char*>(to);
        if (!nk_cast_plan_serial_(from_dtype, to_dtype, count,
                                  reinterpret_cast<std::uintptr_t>(from) | reinterpret_cast<std::uintptr_t>(to),
                                  &arguments.cast.plan))
            return status_t::unsupported_k;
        u32_t const blocks = static_cast<u32_t>(
            min_of<std::size_t>(divide_round_up<std::size_t>(arguments.cast.plan.units, block_threads()),
                                device_.multiprocessor_count * resident_blocks_(vram_kernel_t::cast_k, 0)));
        return launch_(vram_kernel_t::cast_k, 0, blocks, 0, stream, arguments);
    }

    /** Narrows @p rows rows of the index's columns at @p from, @p from_stride bytes apart, into
     *  @p to at the vectors matrix pitch at unit RMS on @p stream, one block per row. */
    status_t launch_rmscast_(f32_t const* from, std::size_t from_stride, byte_t* to, std::size_t rows,
                             hip_stream_handle_t stream) const noexcept {
        vram_arguments_t arguments {};
        nk_each_rmsnorm_arguments_t& rmscast = arguments.rmscast;
        rmscast.x = reinterpret_cast<unsigned char const*>(from);
        rmscast.y = reinterpret_cast<unsigned char*>(to);
        rmscast.groups = 1;
        rmscast.columns = dimensions();
        rmscast.vectors = rows;
        rmscast.x_stride = from_stride;
        rmscast.y_stride = row_bytes();
        rmscast.epsilon = std::numeric_limits<f32_t>::min();
        u32_t const blocks = static_cast<u32_t>(
            min_of<std::size_t>(rows, device_.multiprocessor_count * resident_blocks_(vram_kernel_t::rmscast_k, 0)));
        return launch_(vram_kernel_t::rmscast_k, 0, blocks, 0, stream, arguments);
    }

    /** Launches one walk per query of @p launch, each written out through @p sink: the search,
     *  indexed search and multiply families, each on its register rung or the arena frontier. */
    template <typename sink_>
    status_t launch_walk_(walk_launch_t const& launch, sink_ sink, hip_stream_handle_t stream) noexcept {
        vram_arguments_t arguments {};
        arguments.search = launch.view;
        arguments.rescoring = launch.rescoring;
        vram_kernel_t kernel = vram_kernel_t::search_k;
        if constexpr (std::is_same_v<sink_, vram_multiply_sink_t>) {
            arguments.multiply_sink = sink;
            kernel = vram_kernel_t::multiply_k;
        }
        else {
            arguments.search_sink = sink;
            if (launch.rescoring.heads) kernel = vram_kernel_t::search_indexer_k;
        }
        u32_t const slots = launch.frontier == frontier_kind_t::wide_arena_k ? 0 : launch.candidate_slots;
        u32_t const blocks = walk_blocks_(launch.view.queries_count, resident_blocks_(kernel, slots));
        return launch_(kernel, slots, blocks, vram_warps_per_block_k * launch.view.scratch_bytes_per_warp, stream,
                       arguments);
    }

    /** Walks every token of @p launch under its staged heads, folding the value rows
     *  @p value_lookup resolves into the normalized @p outputs and their @p states; the lookup
     *  travels as bytes to the attend kernel its @c device_lookup_k names, found in the code object
     *  by name. */
    template <typename value_lookup_type_>
    status_t launch_attend_(walk_launch_t const& launch, value_lookup_type_ const& value_lookup, f32_t* outputs,
                            softmax_state_t* states, f32_t logit_scale, traversal_context_t& context) noexcept {
        static_assert(std::is_trivially_copyable_v<value_lookup_type_> &&
                          sizeof(value_lookup_type_) <= vram_value_lookup_capacity_k &&
                          alignof(value_lookup_type_) <= 16,
                      "A device value lookup travels to the kernels as its own bytes");
        vram_arguments_t arguments {};
        arguments.search = launch.view;
        arguments.rescoring = launch.rescoring;
        vram_attend_t& attend = arguments.attend;
        attend.outputs = outputs;
        attend.states = states;
        attend.missed = context.missed_.get();
        attend.logit_scale = logit_scale;
        attend.heads_count = launch.rescoring.heads_count;
        attend.columns = static_cast<u32_t>(dimensions());
        std::memcpy(attend.value_lookup, &value_lookup, sizeof(value_lookup));
        u32_t const slots = launch.frontier == frontier_kind_t::wide_arena_k ? 0 : launch.candidate_slots;
        vram_kernel_symbol_t symbol;
        vram_kernel_symbol(symbol, "dense", "amd_kernel", "vram_arguments_t", vram_kernel_t::attend_k, capability_,
                           vector_dtype_, slots, u32_t {value_lookup_type_::device_lookup_k});
        return hip_launch(module_.function(symbol.text),
                          walk_blocks_(launch.view.queries_count, resident_blocks_(vram_kernel_t::attend_k, slots)),
                          block_threads(), vram_warps_per_block_k * launch.view.scratch_bytes_per_warp,
                          context.stream(), arguments);
    }

#pragma endregion Launches

    /** This class's code object on this device, and its kernels in @ref vram_kernel_t order, the
     *  attends left to their launches. */
    hip_module_t module_;
    hip_function_handle_t kernels_[static_cast<u32_t>(vram_kernel_t::count_k)][gpu_rungs_per_kernel_k] = {};
};

} // namespace unum::usearch::dense
