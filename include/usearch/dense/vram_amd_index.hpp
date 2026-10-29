/**
 *  @file include/usearch/dense/vram_amd_index.hpp
 *  @author Ash Vardanian
 *  @date October 11, 2026
 *  @brief Vector Search index for a single uniform latency memory space on AMD GPUs.
 *
 *  The host face of the AMD family, compiled by any C++ compiler: it fills the plain argument
 *  records of @c vram_index.hpp and launches the kernels of @c vram_amd_index.hip by name through
 *  the HIP runtime table its storage was placed with, so it never sees a kit, a shared view or HIP
 *  beyond its runtime's declarations. Every kernel is one instantiation of @c amd_kernel, for the
 *  walks basic graphs share, or of @c amd_dense_kernel, for the rest, both over five arguments -
 *  the kernel, the capability ceiling, the dtype, the register rung and the attend value lookup -
 *  so @c vram_kernel_symbol spells its symbol here from those arguments alone, and one code object
 *  per class, which the face's kernels type names, carries every dtype.
 */
#pragma once

#include <cstddef> // `std::size_t`
#include <cstring> // `std::memcpy` - an attend's value lookup travels as bytes

#include <bit>         // `std::bit_ceil` - the lanes sharing a row
#include <numbers>     // `std::numbers::log2e_v` - the base-2 softmax
#include <type_traits> // `std::is_same_v` - which heads a walk stages
#include <utility>     // `std::move` - the loaded module, `std::forward` - a forwarded lookup

#include <numkong/capabilities.h> // `nk_kernel_kind_t`
#include <numkong/cast/serial.h>  // `nk_cast_plan_serial_` - the device casts' plans
#include <numkong/types.h>

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/vram_amd_device.hpp"
#include "usearch/utilities/vram_device.hpp" // `vram_kernel_symbol` - a kernel's symbol
#include "usearch/utilities/vram_shards.hpp" // `vram_shard_fanout` - befriended to read the shards
#include "usearch/dense/types.hpp"           // `vram_telemetry_block_t` & the launch geometry constants
#include "usearch/dense/vram_index.hpp"

namespace unum::usearch::dense {

#pragma region Kernels

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
    case vram_kernel_t::refine_k: return 3;
    case vram_kernel_t::search_k: return slots <= 8 ? 4 : 3;
    case vram_kernel_t::rebase_links_k:
    case vram_kernel_t::reachability_sweep_k:
    case vram_kernel_t::reached_in_links_k: return 8;
    default: return 1;
    }
}

template <nk_capability_t capability_, nk_dtype_t dtype_>
constexpr u32_t hip_resident_blocks(vram_dense_kernel_t kernel, u32_t slots) noexcept {
    switch (kernel) {
    case vram_dense_kernel_t::fold_k:
    case vram_dense_kernel_t::merge_back_links_k: return 3;
    case vram_dense_kernel_t::search_indexer_k:
    case vram_dense_kernel_t::multiply_k: return slots <= 8 ? 4 : 3;
    // The attend fold carries per-head state, which spills at 8 slots under four blocks' budget.
    case vram_dense_kernel_t::attend_k: return slots <= 4 ? 4 : 3;
    case vram_dense_kernel_t::derive_norms_k:
    case vram_dense_kernel_t::cast_k:
    case vram_dense_kernel_t::rmscast_k: return 8;
    default: return 1;
    }
}

/** Waves each SIMD keeps when @p blocks blocks of @c vram_warps_per_block_k wavefronts share a
 *  compute unit's four SIMDs - the unit HIP-Clang reads the second argument of
 *  @c __launch_bounds__ in, where CUDA reads blocks per multiprocessor. The four is the CDNA
 *  compute unit's, fixed by its ISA, and the attribute needs it at compile time, out of any device
 *  query's reach. */
constexpr u32_t hip_waves_per_simd(u32_t blocks) noexcept { return blocks * vram_warps_per_block_k / 4; }

/** One instantiated kernel of @p kernel_enum_type_: the kernel and its register rung, zero where it
 *  keeps no ladder or walks the arena frontier. */
template <typename kernel_enum_type_>
struct amd_rung {
    kernel_enum_type_ kernel;
    u32_t slots;
};

/** Every instantiation of the shared @c amd_kernel one dtype exports, which the device pass
 *  instantiates and the host looks up as it loads the code object. */
inline constexpr amd_rung<vram_kernel_t> amd_rungs[] = {
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
    {vram_kernel_t::rebase_links_k, 0},
    {vram_kernel_t::mark_landmarks_k, 0},
    {vram_kernel_t::reachability_sweep_k, 0},
    {vram_kernel_t::reachability_islands_k, 0},
    {vram_kernel_t::reached_in_links_k, 0},
    {vram_kernel_t::reconnect_island_k, 0},
};

/** Every instantiation of @c amd_dense_kernel one dtype exports bar the attends, which the device
 *  pass instantiates and the host looks up as it loads the code object. */
inline constexpr amd_rung<vram_dense_kernel_t> amd_dense_rungs[] = {
    {vram_dense_kernel_t::search_indexer_k, 0},
    {vram_dense_kernel_t::search_indexer_k, 2},
    {vram_dense_kernel_t::search_indexer_k, 4},
    {vram_dense_kernel_t::search_indexer_k, 8},
    {vram_dense_kernel_t::search_indexer_k, 16},
    {vram_dense_kernel_t::multiply_k, 0},
    {vram_dense_kernel_t::multiply_k, 2},
    {vram_dense_kernel_t::multiply_k, 4},
    {vram_dense_kernel_t::multiply_k, 8},
    {vram_dense_kernel_t::multiply_k, 16},
    {vram_dense_kernel_t::merge_back_links_k, 0},
    {vram_dense_kernel_t::fold_k, 0},
    {vram_dense_kernel_t::derive_norms_k, 0},
    {vram_dense_kernel_t::merge_attention_k, 0},
    {vram_dense_kernel_t::resolve_matches_k, 0},
    {vram_dense_kernel_t::merge_matches_k, 0},
    {vram_dense_kernel_t::cast_k, 0},
    {vram_dense_kernel_t::rmscast_k, 0},
};

/** The rungs every walk family, attends included, is instantiated over. */
inline constexpr u32_t amd_walk_rungs[] = {0, 2, 4, 8, 16};

#pragma endregion Kernels

/** The AMD dense shard's kernels: the loaded code object and the hooks its @c vram_shard_core
 *  calls to load, launch and stage them, each reading the core it is handed. */
template <nk_dtype_t vector_dtype_, typename kernels_type_, typename cache_hints_type_,
          typename vectors_allocator_type_, typename neighbors_allocator_type_, typename traversals_allocator_type_,
          typename key_type_, typename key_hash_type_>
struct vram_amd_kernels {

    /** The capabilities the kernels compiled under, as the binary holding them states. */
    inline static constexpr nk_capability_t capability_k = kernels_type_::capability_k;

    using core_t =
        vram_shard_core<hip_vendor_t, vram_amd_kernels, vector_dtype_, cache_hints_type_, vectors_allocator_type_,
                        neighbors_allocator_type_, traversals_allocator_type_, key_type_, key_hash_type_>;
    using scalar_t = typename nk::type_for<vector_dtype_>::type;
    using dot_result_t = typename scalar_t::dot_result_t;
    using dot_t = typename dot_result_t::raw_t;
    using norms_t = vector_norms<dot_t>;
    using traversal_context_t = hip_vendor_t::traversal_context<traversals_allocator_type_>;
    using refine_scratch_t = vram_refine_scratch<traversal_context_t>;
    using frontier_kind_t = vram_frontier_kind_t;
    using search_config_t = vram_search_config_t;
    using search_result_t = vram_search_result_t;
    using refine_config_t = vram_refine_config_t;
    using attend_heads_t = vram_attend_heads_t;
    using audit_landing_t = vram_audit_landing_t;

    /** Lanes cooperating on one row: the smallest power of two covering its fragments, at most a
     *  warp. */
    u32_t lanes_per_row(core_t const& core) const noexcept {
        return min_of(std::bit_ceil(core.row_fragments()), wave_lanes());
    }

    /** Lanes of this class's wavefronts: 32 on CDNA5, 64 before it. */
    static constexpr u32_t wave_lanes() noexcept { return hip_wave_lanes(capability_k); }

    /** Threads per block, four wavefronts. */
    u32_t block_threads() const noexcept { return vram_warps_per_block_k * wave_lanes(); }

    /** Whether a search, multiply or attend launch fits on this device: per warp, the query span
     *  and the @p fold_bytes an attend keeps past it, beside the static batch buffers. Defined in
     *  @c vram_amd_index.hip, where the kits are visible. */
    bool walk_scratch_fits(core_t const& core, std::size_t fold_bytes = 0) const noexcept;

    /** Whether a builder launch fits @p scratch_bytes_per_warp of per-warp scratch in shared
     *  memory beside the batch buffers and schedule state, on every block a multiprocessor keeps
     *  resident; when it cannot, the warp's arena slice keeps it, as Apple's always does. Defined
     *  in @c vram_amd_index.hip. */
    bool build_scratch_fits(core_t const& core, u32_t scratch_bytes_per_warp) const noexcept;

    /** The bytes an executor's run must hold before a context carved over it can search or multiply
     *  @p rows rows under @p config, landing @c config.count columns per row, one aligned term per
     *  carve the verb makes. */
    std::size_t traversal_memory_usage(core_t const& core, std::size_t rows, search_config_t config) const noexcept {
        static_assert(sizeof(dot_t) == sizeof(distance_t));
        if (!core.dimensions() || !core.device_.multiprocessor_count) return 0;
        auto aligned = [](std::size_t bytes) noexcept {
            return divide_round_up<std::size_t>(bytes, default_alignment_k) * default_alignment_k;
        };
        // Search widens the candidates to cover its count while multiply walks the bare factor, so
        // the wider one sizes.
        u32_t const expansion = max_of<u32_t>(gpu_walk_breadth(config.expansion_factor, config.count),
                                              core.non_zero_expansion_factor(config));
        bool const wide = expansion > vram_widest_candidate_slots_k * wave_lanes();
        u32_t const count_slots = divide_round_up<u32_t>(config.count, 32);
        u32_t const candidate_slots = core.candidate_slots_for_(expansion, count_slots > 2 ? count_slots : 2);
        gpu_arena_shape_t const arena_shape = core.arena_shape_for(
            core.walk_selection_nodes_(expansion, candidate_slots),
            wide ? frontier_kind_t::wide_arena_k : frontier_kind_t::register_ladder_k);
        // The walks keep as many blocks resident as a search on the same rung, an attend no more.
        u32_t const resident_blocks = resident_blocks_(vram_kernel_t::search_k, wide ? 0 : candidate_slots);
        std::size_t const arena_warps = std::size_t {core.walk_blocks_(rows, resident_blocks)} * vram_warps_per_block_k;
        std::size_t const staging_bytes = rows * core.row_bytes() + core.import_bytes_(rows, nk_f64_k) +
                                          rows * config.count * (sizeof(node_id_t) + sizeof(distance_t)) +
                                          rows * sizeof(u32_t);
        return aligned(sizeof(vram_telemetry_block_t)) + 2 * aligned(sizeof(u32_t)) +
               aligned(arena_warps * arena_shape.slice_bytes<node_id_t>()) + aligned(staging_bytes);
    }

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
    expected<walk_launch_t> stage_walk_(core_t& core, rows_view_t queries, [[maybe_unused]] heads_type_ const& heads,
                                        traversal_context_t& context, u32_t expansion, u32_t minimum_slots,
                                        std::size_t outputs_bytes, std::size_t visible) noexcept {
        constexpr bool indexer_k = std::is_same_v<heads_type_, indexer_heads<scalar_t>>;
        constexpr bool attend_k = std::is_same_v<heads_type_, attend_heads_t>;
        std::size_t const queries_count = queries.rows;
        if (!core.vectors_matrix_.reserved_bytes()) return {{}, status_t::uninitialized_k};
        if (context.device() != core.device_.ordinal || !context) return {{}, status_t::config_mismatch_k};
        // Validated with the rest: past here a refusal would already have burned epoch space.
        if (status_t const refusal = core.rows_refusal_(queries); failed(refusal)) return {{}, refusal};
        std::size_t heads_rows = 0;
        if constexpr (indexer_k) {
            heads_rows = heads.heads.extent(0);
            if (heads.weights.extent(0) != queries_count || heads_rows != queries_count * heads.weights.extent(1))
                return {{}, status_t::config_mismatch_k};
            if (status_t const refusal = core.rows_refusal_(heads.heads); failed(refusal)) return {{}, refusal};
        }
        if constexpr (attend_k) {
            heads_rows = heads.rows.rows;
            if (status_t const refusal = core.rows_refusal_(heads.rows); failed(refusal)) return {{}, refusal};
        }
        hip_stream_handle_t const stream = context.stream();

        // Past the register ladder's widest rung the arena-backed walk serves.
        frontier_kind_t const frontier = expansion > vram_widest_candidate_slots_k * wave_lanes()
                                             ? frontier_kind_t::wide_arena_k
                                             : frontier_kind_t::register_ladder_k;
        u32_t const candidate_slots = core.candidate_slots_for_(expansion, minimum_slots);
        gpu_arena_shape_t const arena_shape = core.arena_shape_for(
            core.walk_selection_nodes_(expansion, candidate_slots), frontier);
        // A search, an indexed search and a multiply keep as many blocks resident on one rung.
        u32_t const rung_slots = frontier == frontier_kind_t::wide_arena_k ? 0 : candidate_slots;
        u32_t const resident_blocks = attend_k ? resident_blocks_(vram_dense_kernel_t::attend_k, rung_slots)
                                               : resident_blocks_(vram_kernel_t::search_k, rung_slots);
        u32_t const concurrent_walks = core.device_.multiprocessor_count * resident_blocks * vram_warps_per_block_k;
        u32_t const filling = filling_nodes<u32_t>(expansion, static_cast<u32_t>(core.connectivity()));
        std::size_t const arena_warps = std::size_t {core.walk_blocks_(queries_count, resident_blocks)} *
                                        vram_warps_per_block_k;
        std::size_t const arena_bytes_before = context.arena_bytes_;
        byte_t* const arena = context.ensure_arena(arena_warps * arena_shape.slice_bytes<node_id_t>());
        if (!arena) return {{}, status_t::bad_alloc_k};
        // The staging holds the query rows and any head rows first, since they carry the widest
        // alignment, then the window importing them, the heads' weights, then the outputs; stream
        // order serializes its reuse across calls.
        std::size_t const rows_bytes = (queries_count + heads_rows) * core.row_bytes();
        std::size_t const import_bytes = core.import_bytes_(attend_k ? heads_rows : queries_count, queries.dtype);
        std::size_t const weights_bytes = indexer_k ? heads_rows * sizeof(f32_t) : 0;
        byte_t* const staging = context.ensure_staging(rows_bytes + import_bytes + weights_bytes + outputs_bytes);
        if (!staging) return {{}, status_t::bad_alloc_k};
        byte_t* const window = staging + rows_bytes;
        auto const [epoch_base, epochs_status] = context.take_epochs(queries_count, arena_bytes_before);
        if (failed(epochs_status)) return {{}, epochs_status};
        // Pitched uploads: device rows pad to whole fragments, and the pad bytes must be zero for
        // the zero-tail distance identity to hold.
        if ((import_bytes || core.packed_row_bytes() != core.row_bytes()) &&
            core.api().fill_async(staging, 0, rows_bytes + import_bytes, stream) != hip_success_k)
            return {{}, status_t::device_error_k};
        if (status_t const imported = core.import_rows_(staging, queries, window, stream); failed(imported))
            return {{}, imported};
        if (core.api().fill_async(context.telemetry_.get(), 0, sizeof(vram_telemetry_block_t), stream) !=
                hip_success_k ||
            core.api().fill_async(context.assigned_nodes_.get(), 0, 2 * sizeof(u32_t), stream) != hip_success_k)
            return {{}, status_t::device_error_k};

        vram_rescoring_t rescoring {};
        if constexpr (indexer_k) {
            byte_t* const staged_heads = staging + queries_count * core.row_bytes();
            byte_t* const staged_weights = staging + rows_bytes + import_bytes;
            std::size_t const weights_row_bytes = heads.weights.extent(1) * sizeof(f32_t);
            if (failed(core.import_rows_(staged_heads, heads.heads, window, stream)) ||
                core.api().copy_2d_async(staged_weights, weights_row_bytes, heads.weights.byte_data(),
                                         heads.weights.stride_bytes(0), weights_row_bytes, queries_count,
                                         hip_copy_default_k, stream) != hip_success_k)
                return {{}, status_t::device_error_k};
            rescoring = {staged_heads, reinterpret_cast<f32_t const*>(staged_weights),
                         static_cast<u32_t>(heads.weights.extent(1)), core.row_fragments()};
        }
        if constexpr (attend_k) {
            byte_t* const staged_heads = staging + queries_count * core.row_bytes();
            if (status_t const imported = core.import_rows_(staged_heads, heads.rows, window, stream); failed(imported))
                return {{}, imported};
            rescoring = {staged_heads, nullptr, static_cast<u32_t>(heads.head_count), core.row_fragments()};
        }

        vram_search_t const view {core.storage(),
                                  staging,
                                  static_cast<u32_t>(queries_count),
                                  static_cast<u32_t>(visible),
                                  static_cast<u32_t>(core.row_bytes()),
                                  expansion,
                                  arena,
                                  arena_shape,
                                  epoch_base,
                                  context.telemetry_.get(),
                                  context.assigned_nodes_.get(),
                                  context.assigned_nodes_.get() + 1,
                                  concurrent_walks,
                                  filling};
        return {{view, candidate_slots, frontier, staging + rows_bytes + import_bytes + weights_bytes, rescoring},
                status_t::success_k};
    }

    /** The submission half of @c search: stages the queries and @p heads, launches walks over the
     *  @p visible rows, and queues the pitched writeback. Until the context drains, the queries,
     *  the heads, both outputs and the context stay live; across contexts they must not share a
     *  buffer at all. */
    template <typename heads_type_ = dummy_heads_t>
    expected<search_result_t> submit_search_(core_t& core, rows_view_t queries, nk::matrix_span<node_id_t> result_ids,
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
        auto const staged = stage_walk_(core, queries, heads, context, expansion, count_slots > 2 ? count_slots : 2,
                                        ids_bytes + distances_bytes, visible);
        if (!staged) return {{}, staged.status};
        node_id_t* const device_ids = reinterpret_cast<node_id_t*>(staged.value.outputs);
        distance_t* const device_distances = wants_distances
                                                 ? reinterpret_cast<distance_t*>(staged.value.outputs + ids_bytes)
                                                 : nullptr;
        hip_stream_handle_t const stream = context.stream();
        if (status_t const launched = launch_walk_(
                core, staged.value, vram_search_sink_t {device_ids, device_distances, config.count, core.kernel_},
                stream);
            failed(launched))
            return {{}, launched};
        // The landing may be wider than the count, so a multi-index can hand every shard its own
        // columns of one results matrix.
        if (core.api().copy_2d_async(result_ids.data(), result_ids.stride_bytes(0), device_ids,
                                     config.count * sizeof(node_id_t), config.count * sizeof(node_id_t), queries_count,
                                     hip_copy_default_k, stream) != hip_success_k)
            return {{}, status_t::device_error_k};
        if (wants_distances &&
            core.api().copy_2d_async(result_distances.data(), result_distances.stride_bytes(0), device_distances,
                                     config.count * sizeof(distance_t), config.count * sizeof(distance_t),
                                     queries_count, hip_copy_default_k, stream) != hip_success_k)
            return {{}, status_t::device_error_k};
        return {{config.count}, status_t::success_k};
    }

    /** The submission half of @c multiply: stages the rows, launches walks over the @p visible
     *  rows, and queues the writeback of the kept pairs and the scored counts. */
    expected<search_result_t> submit_multiply_(core_t& core, rows_view_t left, nk::matrix_span<node_id_t> node_ids,
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
        auto const staged = stage_walk_(core, left, dummy_heads_t {}, context, core.non_zero_expansion_factor(config),
                                        2, ids_bytes + products_bytes + rows_count * sizeof(u32_t), visible);
        if (!staged) return {{}, staged.status};
        node_id_t* const device_ids = reinterpret_cast<node_id_t*>(staged.value.outputs);
        byte_t* const device_products = staged.value.outputs + ids_bytes;
        u32_t* const device_counts = reinterpret_cast<u32_t*>(staged.value.outputs + ids_bytes + products_bytes);
        hip_stream_handle_t const stream = context.stream();
        vram_multiply_sink_t const sink {device_ids, device_products, device_counts, static_cast<u32_t>(wanted)};
        if (status_t const launched = launch_walk_(core, staged.value, sink, stream); failed(launched))
            return {{}, launched};
        if (wanted && (core.api().copy_2d_async(node_ids.data(), node_ids.stride_bytes(0), device_ids,
                                                wanted * sizeof(node_id_t), wanted * sizeof(node_id_t), rows_count,
                                                hip_copy_default_k, stream) != hip_success_k ||
                       core.api().copy_2d_async(products.data(), products.stride_bytes(0), device_products,
                                                wanted * sizeof(dot_t), wanted * sizeof(dot_t), rows_count,
                                                hip_copy_default_k, stream) != hip_success_k))
            return {{}, status_t::device_error_k};
        if (core.api().copy_async(counts.data(), device_counts, rows_count * sizeof(u32_t), hip_copy_default_k,
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
    expected<f32_t*> submit_attend_(core_t& core, rows_view_t queries, std::size_t head_count,
                                    value_lookup_type_ const& value_lookup, f32_t score_scale,
                                    traversal_context_t& context, search_config_t config, std::size_t blocks,
                                    std::size_t exported_bytes, std::size_t visible) noexcept {
        std::size_t const rows_count = queries.rows;
        if (config.traversal != traversal_t::walk_k) return {nullptr, status_t::unsupported_k};
        if (!head_count || rows_count % head_count) return {nullptr, status_t::shape_mismatch_k};
        if (!rows_count) return {nullptr, status_t::success_k};
        std::size_t const scratch_bytes = vram_attend_scratch_bytes(head_count, core.dimensions(), wave_lanes());
        if (!walk_scratch_fits(core, scratch_bytes)) return {nullptr, status_t::config_mismatch_k};
        std::size_t const outputs_bytes = rows_count * core.dimensions() * sizeof(f32_t);
        rows_view_t const first_heads {queries.data, rows_count / head_count, queries.columns,
                                       queries.stride * head_count, queries.dtype};
        auto staged = stage_walk_(
            core, first_heads, attend_heads_t {queries, head_count}, context, core.non_zero_expansion_factor(config), 2,
            blocks * (outputs_bytes + rows_count * sizeof(softmax_state_t)) + exported_bytes, visible);
        if (!staged) return {nullptr, staged.status};
        staged.value.view.scratch_bytes_per_warp += static_cast<u32_t>(scratch_bytes);
        if (core.api().fill_async(context.missed_.get(), 0, sizeof(u32_t), context.stream()) != hip_success_k)
            return {nullptr, status_t::device_error_k};
        f32_t* const device_outputs = reinterpret_cast<f32_t*>(staged.value.outputs);
        if (status_t const launched = launch_attend_(
                core, staged.value, value_lookup, device_outputs,
                reinterpret_cast<softmax_state_t*>(staged.value.outputs + outputs_bytes),
                score_scale * std::numbers::log2e_v<f32_t>, context);
            failed(launched))
            return {nullptr, launched};
        return {device_outputs, status_t::success_k};
    }

#pragma endregion Search

#pragma region Refinement

    /** Queues one pass of the three-pass refinement engine behind @c refine_plan_t into its carved
     *  @p scratch through @p kernel: the pass covers @p nodes, or under a device-resident
     *  @p node_ids the listed IDs with @p nodes spanning the array's offsets, each node first
     *  walking across the neighboring group in round @p round when @p parts parts are given; the
     *  merge folds the re-selected rows' forward picks and the rows chained onto every picked row
     *  into the live rows, and leaves every other row as it was. Nothing waits: the landing queued
     *  after it checks its claims. */
    status_t refine_pass_(core_t& core, traversal_context_t& context, refine_scratch_t const& scratch,
                          refine_config_t config, vram_kernel_t kernel, iota_view<node_id_t> nodes,
                          std::span<u32_t const> node_ids, u32_t parts = 0, u32_t round = 0) noexcept {
        if (status_t const kept = core.status(); failed(kept)) return kept;
        std::size_t const nodes_count = core.size();
        std::size_t const walk_count = nodes.size();
        hip_stream_handle_t const stream = context.stream();
        vram_refine_passes_t const passes = core.refine_passes_(context);
        if (!passes.link_heads) return status_t::bad_alloc_k;
        // Chains start empty and no row walked yet.
        if (core.api().fill_async(scratch.claims, 0, 2 * sizeof(u32_t), stream) != hip_success_k ||
            core.api().fill_async(passes.link_heads, 0xFF, nodes_count * sizeof(u32_t), stream) != hip_success_k ||
            core.api().fill_async(passes.walked.words, 0, scratch.words * sizeof(u32_t), stream) != hip_success_k)
            return status_t::device_error_k;

        // Twice the connectivity by default, like every CPU walk, rounded up to the candidate sizes
        // the kernel carries, or walked exactly past them.
        u32_t const expansion = core.non_zero_expansion_factor(config);
        u32_t const candidate_slots = core.candidate_slots_for_(expansion, 4);
        // Walking across makes two walks into one visited set, so the slice is shaped for both.
        u32_t const walks = parts > 1 ? 2 : 1;
        gpu_arena_shape_t const arena_shape = core.build_arena_shape_(expansion, candidate_slots, walks);
        std::size_t const arena_warps = core.device_.multiprocessor_count *
                                        resident_blocks_(vram_kernel_t::refine_k, candidate_slots) *
                                        vram_warps_per_block_k;
        std::size_t const arena_bytes_before = context.arena_bytes_;
        byte_t* const arena = context.ensure_arena(arena_warps * arena_shape.slice_bytes<node_id_t>());
        if (!arena) return status_t::bad_alloc_k;
        auto const [epoch_base, epochs_status] = context.take_epochs(walk_count, arena_bytes_before);
        if (failed(epochs_status)) return epochs_status;

        u32_t const selection_nodes = core.walk_selection_nodes_(expansion, candidate_slots);
        vram_refine_t view {core.storage(), //
                            passes.shadow_neighbors,
                            nodes.bound,
                            arena_shape.scratch_bytes ? 0u : core.build_scratch_bytes_per_warp(selection_nodes),
                            selection_nodes,
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
                            round,
                            passes.link_heads,
                            passes.link_next,
                            passes.walked};
        if (status_t const launched = launch_refine_(core, kernel, view, candidate_slots, stream); failed(launched))
            return launched;

        // Fold every walked row's forward picks and every chain of walkers into the live rows; a
        // heal round's islands chain nowhere, so only their own rows merge.
        return launch_refine_back_links_(core, view, stream);
    }

    /** Queues a reachability audit from as many stride-spread landmarks as one search draws - the
     *  connectivity - and its first sweep, through @ref queue_sweeps_. Nothing waits; the caller
     *  reads @p landing once its event fired, and queues further sweeps from sweep 2 while the last
     *  one still expanded. */
    status_t queue_audit_(core_t& core, traversal_context_t& context, refine_scratch_t const& scratch,
                          audit_landing_t* landing) noexcept {
        std::size_t const rows = core.size();
        hip_stream_handle_t const stream = context.stream();
        std::size_t const seed_count = min_of<std::size_t>(core.connectivity(), rows);
        // The bitsets, the levels, the sweeps' expanded counters and the unreached counter lie back
        // to back.
        if (core.api().fill_async(scratch.reached.words, 0, (scratch.words * 3 + 4) * sizeof(u32_t), stream) !=
            hip_success_k)
            return status_t::device_error_k;
        vram_arguments_t seeds {};
        seeds.graph.neighbors = core.neighbor_rows_();
        seeds.graph.rows_count = static_cast<u32_t>(rows);
        seeds.graph.connectivity = static_cast<u32_t>(core.connectivity());
        seeds.graph.seed_count = static_cast<u32_t>(seed_count);
        seeds.graph.reached = scratch.reached;
        seeds.graph.frontier = scratch.frontier;
        if (status_t const launched = launch_(vram_kernel_t::mark_landmarks_k, 0,
                                              divide_round_up<u32_t>(static_cast<u32_t>(seed_count), block_threads()),
                                              0, stream, seeds);
            failed(launched))
            return launched;
        return queue_sweeps_(core, context, scratch, landing, 1);
    }

    /** Queues the audit's sweeps from @p first_sweep up to twice it, so the batches double until
     *  one ends past convergence, then a count of the rows no sweep reached so far, collecting
     *  their IDs when @p scratch has room for them, then the landing of every counter, the last
     *  sweep's expanded count among them. */
    status_t queue_sweeps_(core_t& core, traversal_context_t& context, refine_scratch_t const& scratch,
                           audit_landing_t* landing, u32_t first_sweep) noexcept {
        std::size_t const rows = core.size();
        hip_stream_handle_t const stream = context.stream();
        vram_arguments_t sweep {};
        sweep.graph.neighbors = core.neighbor_rows_();
        sweep.graph.rows_count = static_cast<u32_t>(rows);
        sweep.graph.connectivity = static_cast<u32_t>(core.connectivity());
        sweep.graph.reached = scratch.reached;
        sweep.graph.levels = scratch.levels;
        sweep.graph.island_ids = scratch.island_ids;
        // A sweep queued past convergence finds the one before it expanded nothing and leaves at
        // entry; odd sweeps grow the first frontier, even ones the second.
        u32_t const grid = core.device_.multiprocessor_count * resident_blocks_(vram_kernel_t::reachability_sweep_k, 0);
        for (u32_t sweep_number = first_sweep; sweep_number != 2 * first_sweep; ++sweep_number) {
            bool const odd = sweep_number & 1u;
            sweep.graph.frontier = odd ? scratch.frontier : scratch.next_frontier;
            sweep.graph.next_frontier = odd ? scratch.next_frontier : scratch.frontier;
            sweep.graph.counter = scratch.expanded + odd;
            sweep.graph.previous_counter = sweep_number == 1 ? nullptr : scratch.expanded + !odd;
            if (core.api().fill_async(sweep.graph.next_frontier.words, 0, scratch.words * sizeof(u32_t), stream) !=
                    hip_success_k ||
                core.api().fill_async(sweep.graph.counter, 0, sizeof(u32_t), stream) != hip_success_k)
                return status_t::device_error_k;
            if (status_t const launched = launch_(vram_kernel_t::reachability_sweep_k, 0, grid, 0, stream, sweep);
                failed(launched))
                return launched;
        }
        sweep.graph.counter = scratch.unreached;
        if (core.api().fill_async(scratch.unreached, 0, sizeof(u32_t), stream) != hip_success_k)
            return status_t::device_error_k;
        if (status_t const launched = launch_(vram_kernel_t::reachability_islands_k, 0,
                                              divide_round_up<u32_t>(static_cast<u32_t>(rows), block_threads()), 0,
                                              stream, sweep);
            failed(launched))
            return launched;
        return core.land_counters_(scratch, landing, 2 * first_sweep - 1, stream);
    }

    /** Queues the reconnection of the @p islands islands the last audit in @p scratch collected
     *  on @p context 's stream: the reached in-links first, counted in the pass's staging, so no
     *  reconnection takes the last link a reached node is found through, then each island into a
     *  reached pick or its landmark. */
    status_t queue_reconnect_(core_t& core, traversal_context_t& context, refine_scratch_t const& scratch,
                              u32_t islands) noexcept {
        std::size_t const rows = core.size();
        hip_stream_handle_t const stream = context.stream();
        u32_t* const in_links = core.refine_passes_(context).link_heads;
        if (!in_links) return status_t::bad_alloc_k;
        if (core.api().fill_async(in_links, 0, rows * sizeof(u32_t), stream) != hip_success_k)
            return status_t::device_error_k;
        vram_arguments_t heal {};
        heal.graph.neighbors = core.neighbor_rows_();
        heal.graph.rows_count = static_cast<u32_t>(rows);
        heal.graph.connectivity = static_cast<u32_t>(core.connectivity());
        heal.graph.reached = scratch.reached;
        heal.graph.in_links = in_links;
        heal.graph.island_ids = scratch.island_ids;
        heal.graph.islands_count = islands;
        heal.graph.seed_count = static_cast<u32_t>(min_of<std::size_t>(core.connectivity(), rows));
        heal.graph.reconnected = scratch.reconnected;
        if (status_t const counted = launch_(
                vram_kernel_t::reached_in_links_k, 0,
                core.device_.multiprocessor_count * resident_blocks_(vram_kernel_t::reached_in_links_k, 0), 0, stream,
                heal);
            failed(counted))
            return counted;
        return launch_(vram_kernel_t::reconnect_island_k, 0, divide_round_up<u32_t>(islands, block_threads()), 0,
                       stream, heal);
    }

#pragma endregion Refinement

#pragma region Launches

    /** Blocks of @p kernel on its @p slots rung resident per compute unit, as its launch bounds
     *  read them; every grid and the arena behind it are sized by it. */
    static constexpr u32_t resident_blocks_(vram_kernel_t kernel, u32_t slots) noexcept {
        return hip_resident_blocks<capability_k, vector_dtype_>(kernel, slots);
    }
    static constexpr u32_t resident_blocks_(vram_dense_kernel_t kernel, u32_t slots) noexcept {
        return hip_resident_blocks<capability_k, vector_dtype_>(kernel, slots);
    }

    /** Loads the code object of @p kernels_type_ onto the device and finds every kernel this dtype
     *  exports bar the attends, by the symbol @ref vram_kernel_symbol spells. */
    status_t load_kernels_(core_t& core) noexcept {
        auto [module, status] = hip_module_t::load(core.api(), core.device_.ordinal, kernels_type_::bytes());
        if (failed(status)) return status;
        module_ = std::move(module);
        vram_kernel_symbol_t symbol;
        for (amd_rung<vram_kernel_t> const rung : amd_rungs) {
            vram_kernel_symbol(symbol, "dense", "amd_kernel", "vram_arguments_t", rung.kernel, capability_k,
                               vector_dtype_, rung.slots, itanium_type<void> {});
            hip_function_handle_t& function = kernels_[static_cast<u32_t>(rung.kernel)][gpu_rung_index(rung.slots)];
            if (!(function = module_.function(symbol.text))) return status_t::device_error_k;
        }
        for (amd_rung<vram_dense_kernel_t> const rung : amd_dense_rungs) {
            vram_kernel_symbol(symbol, "dense", "amd_dense_kernel", "vram_arguments_t", rung.kernel, capability_k,
                               vector_dtype_, rung.slots, itanium_type<void> {});
            hip_function_handle_t& function =
                dense_kernels_[static_cast<u32_t>(rung.kernel)][gpu_rung_index(rung.slots)];
            if (!(function = module_.function(symbol.text))) return status_t::device_error_k;
        }
        return status_t::success_k;
    }

    /** Launches @p kernel on its @p slots rung over @p blocks of four wavefronts on @p stream. */
    status_t launch_(vram_kernel_t kernel, u32_t slots, u32_t blocks, u32_t shared_bytes, hip_stream_handle_t stream,
                     vram_arguments_t const& arguments) const noexcept {
        return hip_launch(module_.api(), kernels_[static_cast<u32_t>(kernel)][gpu_rung_index(slots)], blocks,
                          block_threads(), shared_bytes, stream, arguments);
    }
    status_t launch_(vram_dense_kernel_t kernel, u32_t slots, u32_t blocks, u32_t shared_bytes,
                     hip_stream_handle_t stream, vram_arguments_t const& arguments) const noexcept {
        return hip_launch(module_.api(), dense_kernels_[static_cast<u32_t>(kernel)][gpu_rung_index(slots)], blocks,
                          block_threads(), shared_bytes, stream, arguments);
    }

    /** Launches the rows of @p build in waves, each queued behind the last and as wide as the IDs
     *  below it, never narrower than @c vram_admission_ramp_floor_k, so every ID a wave admits
     *  walks rows already published, and each followed by the back-link launch that links its
     *  newcomers from the nodes they picked. A wave failing after an earlier one launched keeps
     *  the failure, since those rows can no longer be un-claimed. */
    status_t launch_insert_(core_t& core, vram_build_t const& build, u32_t candidate_slots,
                            traversal_context_t& context) noexcept {
        hip_stream_handle_t const stream = context.stream();
        vram_arguments_t arguments {};
        // Both launches carve their warps' slices from the one arena, sized for the insert's grid,
        // which the back-link's resident count matches.
        static_assert(resident_blocks_(vram_kernel_t::back_link_k, 0) == resident_blocks_(vram_kernel_t::insert_k, 0));
        u32_t const blocks = core.device_.multiprocessor_count *
                             resident_blocks_(vram_kernel_t::insert_k, candidate_slots);
        u32_t const back_link_rows = core.reselection_nodes_();
        u32_t const back_link_scratch = build.arena_shape.scratch_bytes
                                            ? 0u
                                            : core.build_scratch_bytes_per_warp(back_link_rows);
        auto zeroed = [&](u32_t* counter) noexcept {
            return core.api().fill_async(counter, 0, sizeof(u32_t), stream) == hip_success_k;
        };
        for (u32_t wave_first = build.first_id; wave_first != build.end_id;) {
            u32_t const wave_end = static_cast<u32_t>(
                vram_admission_wave_end(wave_first, build.end_id, vram_dense_admission_share_log2_k));
            u32_t const offset = wave_first - build.first_id;
            arguments.build = build;
            arguments.build.staging = build.staging + std::size_t {offset} * core.row_bytes();
            arguments.build.first_id = wave_first;
            arguments.build.end_id = wave_end;
            arguments.build.epoch_base = build.epoch_base + offset;
            status_t launched = zeroed(build.assigned_nodes) && zeroed(build.touched_count) ? status_t::success_k
                                                                                            : status_t::device_error_k;
            if (succeeded(launched))
                launched = launch_(vram_kernel_t::insert_k, candidate_slots, blocks,
                                   vram_warps_per_block_k * build.scratch_bytes_per_warp, stream, arguments);
            if (failed(launched)) return offset ? core.fail_(launched) : launched;
            // The wave's rows are queued now, so a failure past here can no longer be un-claimed.
            arguments.build.scratch_bytes_per_warp = back_link_scratch;
            arguments.build.selection_nodes = back_link_rows;
            if (!zeroed(build.assigned_nodes)) return core.fail_(status_t::device_error_k);
            if (status_t const linked = launch_(vram_kernel_t::back_link_k, 0, blocks,
                                                vram_warps_per_block_k * back_link_scratch, stream, arguments);
                failed(linked))
                return core.fail_(linked);
            wave_first = wave_end;
        }
        return status_t::success_k;
    }
    template <typename kernel_enum_type_>
    status_t launch_refine_(core_t& core, kernel_enum_type_ kernel, vram_refine_t const& view, u32_t candidate_slots,
                            hip_stream_handle_t stream) noexcept {
        vram_arguments_t arguments {};
        arguments.refine = view;
        return launch_(kernel, candidate_slots,
                       core.device_.multiprocessor_count * resident_blocks_(kernel, candidate_slots),
                       vram_warps_per_block_k * view.scratch_bytes_per_warp, stream, arguments);
    }

    /** Merges every walked row's forward picks and every chain of walkers the walks of @p view
     *  threaded into the live neighbors. */
    status_t launch_refine_back_links_(core_t& core, vram_refine_t const& view, hip_stream_handle_t stream) noexcept {
        // The merge sweeps the whole ID space and skips the rows the pass neither walked nor
        // picked, carving its slices from the arena sized for the refine's grid, which its count
        // matches.
        static_assert(resident_blocks_(vram_dense_kernel_t::merge_back_links_k, 0) ==
                      resident_blocks_(vram_kernel_t::refine_k, 0));
        vram_arguments_t arguments {};
        arguments.refine = view;
        arguments.refine.scratch_bytes_per_warp = view.arena_shape.scratch_bytes
                                                      ? 0u
                                                      : core.build_scratch_bytes_per_warp(core.reselection_nodes_());
        arguments.refine.selection_nodes = core.reselection_nodes_();
        // The merge's claim counter follows the walk's.
        arguments.refine.assigned_nodes = view.assigned_nodes + 1;
        arguments.refine.node_ids = nullptr;
        arguments.refine.first_id = 0;
        arguments.refine.end_id = view.sampling_bound;
        return launch_(vram_dense_kernel_t::merge_back_links_k, 0,
                       core.device_.multiprocessor_count * resident_blocks_(vram_dense_kernel_t::merge_back_links_k, 0),
                       vram_warps_per_block_k * arguments.refine.scratch_bytes_per_warp, stream, arguments);
    }

    /** Queues one phase of @p fold over the rows of its launching shard on @p context 's stream,
     *  each warp's re-selection scratch in shared memory when it fits there and in the context's
     *  arena otherwise; the caller fills everything but the scratch, the counters and the
     *  telemetry, which zero ahead of the launch. */
    status_t launch_fold_(core_t const& core, vram_fold_t fold, traversal_context_t& context) const noexcept {
        static_assert(resident_blocks_(vram_dense_kernel_t::fold_k, 0) <= resident_blocks_(vram_kernel_t::refine_k, 0),
                      "A fold keeps no more blocks resident than the builders its scratch is checked against");
        u32_t const scratch_bytes = core.build_scratch_bytes_per_warp(fold.selection_nodes);
        bool const shared = build_scratch_fits(core, scratch_bytes);
        u32_t const blocks = core.device_.multiprocessor_count * resident_blocks_(vram_dense_kernel_t::fold_k, 0);
        fold.arena_shape = gpu_arena_shape(0, 0, shared ? 0u : scratch_bytes);
        fold.scratch_bytes_per_warp = shared ? scratch_bytes : 0u;
        fold.arena = context.ensure_arena(std::size_t {blocks} * vram_warps_per_block_k *
                                          fold.arena_shape.slice_bytes<node_id_t>());
        if (!fold.arena && fold.arena_shape.scratch_bytes) return status_t::bad_alloc_k;
        fold.telemetry = context.telemetry_.get();
        fold.assigned_nodes = context.assigned_nodes_.get();
        if (core.api().fill_async(fold.telemetry, 0, sizeof(vram_telemetry_block_t), context.stream()) !=
                hip_success_k ||
            core.api().fill_async(fold.assigned_nodes, 0, sizeof(u32_t), context.stream()) != hip_success_k)
            return status_t::device_error_k;
        vram_arguments_t arguments {};
        arguments.fold = fold;
        return launch_(vram_dense_kernel_t::fold_k, 0, blocks, vram_warps_per_block_k * fold.scratch_bytes_per_warp,
                       context.stream(), arguments);
    }

    /** Casts @p count values at @p from in @p from_dtype into @p to in @p to_dtype on @p stream,
     *  through NumKong's plan and its per-unit device conversions. */
    status_t launch_cast_(core_t const& core, void const* from, nk_dtype_t from_dtype, void* to, nk_dtype_t to_dtype,
                          std::size_t count, hip_stream_handle_t stream) const noexcept {
        vram_arguments_t arguments {};
        arguments.cast.from = static_cast<unsigned char const*>(from);
        arguments.cast.to = static_cast<unsigned char*>(to);
        if (!nk_cast_plan_serial_(from_dtype, to_dtype, count,
                                  reinterpret_cast<std::uintptr_t>(from) | reinterpret_cast<std::uintptr_t>(to),
                                  &arguments.cast.plan))
            return status_t::unsupported_k;
        u32_t const blocks = static_cast<u32_t>(
            min_of<std::size_t>(divide_round_up<std::size_t>(arguments.cast.plan.units, block_threads()),
                                core.device_.multiprocessor_count * resident_blocks_(vram_dense_kernel_t::cast_k, 0)));
        return launch_(vram_dense_kernel_t::cast_k, 0, blocks, 0, stream, arguments);
    }

    /** Launches one walk per query of @p launch, each written out through @p sink: the search,
     *  indexed search and multiply families, each on its register rung or the arena frontier. */
    template <typename sink_>
    status_t launch_walk_(core_t& core, walk_launch_t const& launch, sink_ sink, hip_stream_handle_t stream) noexcept {
        vram_arguments_t arguments {};
        arguments.search = launch.view;
        arguments.rescoring = launch.rescoring;
        u32_t const slots = launch.frontier == frontier_kind_t::wide_arena_k ? 0 : launch.candidate_slots;
        auto launch_kernel = [&](auto kernel) noexcept {
            u32_t const blocks = core.walk_blocks_(launch.view.queries_count, resident_blocks_(kernel, slots));
            return launch_(kernel, slots, blocks, vram_warps_per_block_k * launch.view.scratch_bytes_per_warp, stream,
                           arguments);
        };
        if constexpr (std::is_same_v<sink_, vram_multiply_sink_t>) {
            arguments.multiply_sink = sink;
            return launch_kernel(vram_dense_kernel_t::multiply_k);
        }
        else {
            arguments.search_sink = sink;
            if (launch.rescoring.heads) return launch_kernel(vram_dense_kernel_t::search_indexer_k);
            return launch_kernel(vram_kernel_t::search_k);
        }
    }

    /** Walks every token of @p launch under its staged heads, folding the value rows
     *  @p value_lookup resolves into the normalized @p outputs and their @p states; the lookup
     *  travels as bytes to the attend kernel instantiated over its type, found in the code object
     *  by name. */
    template <typename value_lookup_type_>
    status_t launch_attend_(core_t& core, walk_launch_t const& launch, value_lookup_type_ const& value_lookup,
                            f32_t* outputs, softmax_state_t* states, f32_t logit_scale,
                            traversal_context_t& context) noexcept {
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
        attend.columns = static_cast<u32_t>(core.dimensions());
        std::memcpy(attend.value_lookup, &value_lookup, sizeof(value_lookup));
        u32_t const slots = launch.frontier == frontier_kind_t::wide_arena_k ? 0 : launch.candidate_slots;
        vram_kernel_symbol_t symbol;
        vram_kernel_symbol(symbol, "dense", "amd_dense_kernel", "vram_arguments_t", vram_dense_kernel_t::attend_k,
                           capability_k, vector_dtype_, slots, itanium_type<value_lookup_type_> {});
        return hip_launch(
            module_.api(), module_.function(symbol.text),
            core.walk_blocks_(launch.view.queries_count, resident_blocks_(vram_dense_kernel_t::attend_k, slots)),
            block_threads(), vram_warps_per_block_k * launch.view.scratch_bytes_per_warp, context.stream(), arguments);
    }

#pragma endregion Launches

  private:

    /** This class's code object on this device, its shared kernels in @ref vram_kernel_t order and
     *  its dense ones in @ref vram_dense_kernel_t order, the attends left to their launches. */
    hip_module_t module_;
    hip_function_handle_t kernels_[static_cast<u32_t>(vram_kernel_t::count_k)][gpu_rungs_per_kernel_k] = {};
    hip_function_handle_t dense_kernels_[static_cast<u32_t>(vram_dense_kernel_t::count_k)][gpu_rungs_per_kernel_k] = {};
};

/**
 *  @brief Vector Search index for a single uniform latency memory space on AMD GPUs.
 *
 *  Owns the @c vram_shard_core that holds everything but the kernels, and forwards to it; a
 *  wavefront plays the part an NVIDIA warp plays, 64 lanes wide before CDNA5 and 32 on it. The
 *  kernels come from the code object @p kernels_type_ hands out through its static @c bytes,
 *  compiled under its @c capability_k, and a checkpoint load's bounce halves demote through
 *  @p cache_hints_type_.
 */
template <                                                                     //
    nk_dtype_t vector_dtype_,                                                  //
    typename kernels_type_,                                                    //
    typename cache_hints_type_ = dummy_cache_hints_t,                          //
    typename vectors_allocator_type_ = hip_virtual_memory_allocator<byte_t>,   //
    typename neighbors_allocator_type_ = hip_virtual_memory_allocator<byte_t>, //
    typename traversals_allocator_type_ = hip_arena_allocator_t,               //
    typename key_type_ = uuid_t,                                               //
    typename key_hash_type_ = std::hash<key_type_>                             //
    >
struct vram_amd_index {

    using kernels_t =
        vram_amd_kernels<vector_dtype_, kernels_type_, cache_hints_type_, vectors_allocator_type_,
                         neighbors_allocator_type_, traversals_allocator_type_, key_type_, key_hash_type_>;
    using core_t = typename kernels_t::core_t;
    using vendor_t = typename core_t::vendor_t;
    using vectors_allocator_t = typename core_t::vectors_allocator_t;
    using neighbors_allocator_t = typename core_t::neighbors_allocator_t;
    using traversals_allocator_t = typename core_t::traversals_allocator_t;
    using stream_handle_t = typename core_t::stream_handle_t;
    using event_t = typename core_t::event_t;
    using executor_t = typename core_t::executor_t;
    using device_specs_t = typename core_t::device_specs_t;
    using scalar_t = typename core_t::scalar_t;
    using dot_result_t = typename core_t::dot_result_t;
    using dot_t = typename core_t::dot_t;
    using norms_t = typename core_t::norms_t;
    using vectors_view_t = typename core_t::vectors_view_t;
    using vector_view_t = typename core_t::vector_view_t;
    using node_id_t = typename core_t::node_id_t;
    using node_ids_span_t = typename core_t::node_ids_span_t;
    using distance_t = typename core_t::distance_t;
    using small_size_t = typename core_t::small_size_t;
    using traversal_context_t = typename core_t::traversal_context_t;
    using saved_shape_t = typename core_t::saved_shape_t;
    using columns_t = typename core_t::columns_t;
    using columns_view_t = typename core_t::columns_view_t;
    using reserve_capacity_t = typename core_t::reserve_capacity_t;
    using insert_config_t = typename core_t::insert_config_t;
    using insert_result_t = typename core_t::insert_result_t;
    using search_config_t = typename core_t::search_config_t;
    using search_result_t = typename core_t::search_result_t;
    using refine_config_t = typename core_t::refine_config_t;
    using refine_result_t = typename core_t::refine_result_t;
    using health_result_t = typename core_t::health_result_t;
    using refine_plan_t = typename core_t::refine_plan_t;
    using health_plan_t = typename core_t::health_plan_t;
    using save_plan_t = typename core_t::save_plan_t;
    using load_plan_t = typename core_t::load_plan_t;
    using geometry_t = typename core_t::geometry_t;
    using key_t = typename core_t::key_t;
    using keyed_t = typename core_t::keyed_t;
    using keys_view_t = typename core_t::keys_view_t;

    inline static constexpr nk_dtype_t vector_dtype_k = vector_dtype_;
    inline static constexpr std::size_t batch_rows_log2_k = core_t::batch_rows_log2_k;
    inline static constexpr std::size_t max_node_count_k = core_t::max_node_count_k;
    inline static constexpr synchronization_t synchronization_ceiling_k = core_t::synchronization_ceiling_k;

    /** The capability of the quads kit the ladder picked under the code object's, surfaced in the
     *  bench reports; read only where the AMD kits are visible, as its initializer is. */
    inline static constexpr nk_capability_t picked_quads_capability_k =
        quads_ladder<scalar_t, kernels_t::capability_k>::picked_k;

    /** The @p kernel and @p geometry every row keeps, the device storage's allocators, and the
     *  host allocator the keys and the key map sit on, as the core takes them. */
    explicit vram_amd_index(nk_kernel_kind_t kernel, geometry_t geometry,
                            vectors_allocator_t const& vectors_allocator = {},
                            neighbors_allocator_t const& neighbors_allocator = {},
                            forkunion_allocator_t const& keys_allocator = {}) noexcept
        : core_(kernel, geometry, vectors_allocator, neighbors_allocator, keys_allocator) {}

    std::size_t size() const noexcept { return core_.size(); }

    /** The first failure that left claimed rows unpublished, as the core's @c status keeps it. */
    status_t status() const noexcept { return core_.status(); }

    std::size_t capacity() const noexcept { return core_.capacity(); }
    std::size_t connectivity() const noexcept { return core_.connectivity(); }
    geometry_t geometry() const noexcept { return core_.geometry(); }
    std::size_t dimensions() const noexcept { return core_.dimensions(); }
    device_specs_t const& device_specs() const noexcept { return core_.device_specs(); }

    /** The runtime every call on this shard goes through, the one its storage was placed by. */
    hip_api_t const& api() const noexcept { return core_.api(); }
    nk_kernel_kind_t kernel() const noexcept { return core_.kernel(); }
    bool stores_norms() const noexcept { return core_.stores_norms(); }
    std::size_t row_bytes() const noexcept { return core_.row_bytes(); }
    saved_shape_t saved_shape() const noexcept { return core_.saved_shape(); }
    std::size_t memory_usage() const noexcept { return core_.memory_usage(); }

    /** Every node's key on the host and every key's nodes, as the core keeps them. */
    keyed_t const& keys() const noexcept { return core_.keys(); }

    /**
     *  The key of the visible @p node.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The @p node is not visible.
     */
    expected<key_t> key_of(node_id_t node) const noexcept { return core_.key_of(node); }

    /** Invokes @p callback for every published node stored under @p key, as of one acquire of the
     *  watermark. */
    template <typename callback_type_>
    void find_each(key_t const& key, callback_type_&& callback) const noexcept {
        core_.find_each(key, callback);
    }

    /** The visible nodes stored under @p key. */
    std::size_t count(key_t const& key) const noexcept { return core_.count(key); }

    /** Whether any visible node is stored under @p key. */
    bool contains(key_t const& key) const noexcept { return core_.contains(key); }

    template <typename config_type_>
    small_size_t non_zero_expansion_factor(config_type_ const& config) const noexcept {
        return core_.non_zero_expansion_factor(config);
    }

    u32_t build_scratch_bytes_per_warp(u32_t selection_nodes) const noexcept {
        return core_.build_scratch_bytes_per_warp(selection_nodes);
    }

    bool build_scratch_fits(u32_t scratch_bytes_per_warp) const noexcept {
        return core_.kernels().build_scratch_fits(core_, scratch_bytes_per_warp);
    }

    /**
     *  Queues the copy of the stored row of @p node into @p destination on @p stream.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The @p node is not visible, or @p destination is not as wide as a
     *    row.
     *  - @c unsupported_layout_k The @p destination is not contiguous.
     */
    status_t vector(node_id_t node, nk::vector_span<scalar_t> destination, stream_handle_t stream) const noexcept {
        return core_.vector(node, destination, stream);
    }

    /**
     *  Ensures the index can host at least @p config.capacity entries - a hint, since every claim
     *  grows the storage it needs; the first growth binds the device.
     *
     *  @par Statuses:
     *  - @c unknown_kind_k The metric is not one the kits score for this scalar.
     *  - @c config_mismatch_k The neighbors live on another device, the dimensions are zero, the
     *    connectivity is not a power of two, @p config names another geometry, or no walk fits the
     *    geometry.
     *  - @c contended_k A batch or load is in flight.
     *  - @c overflow_k The capacity passes the ID space or the reserved range.
     */
    expected<std::size_t> reserve(reserve_capacity_t config) noexcept { return core_.reserve(config); }

    std::size_t traversal_memory_usage(std::size_t rows, insert_config_t config = {}) const noexcept {
        return core_.traversal_memory_usage(rows, config);
    }

    std::size_t traversal_memory_usage(std::size_t rows, search_config_t config) const noexcept {
        return core_.kernels().traversal_memory_usage(core_, rows, config);
    }

    std::size_t traversal_memory_usage(std::size_t rows, search_config_t config,
                                       indexer_heads<scalar_t> const& heads) const noexcept {
        return core_.traversal_memory_usage(rows, config, heads);
    }

    std::size_t traversal_memory_usage(std::size_t rows, search_config_t config, std::size_t head_count,
                                       std::size_t shards = 1) const noexcept {
        return core_.traversal_memory_usage(rows, config, head_count, shards);
    }

    std::size_t traversal_memory_usage(refine_config_t config, u32_t walks = 1) const noexcept {
        return core_.traversal_memory_usage(config, walks);
    }

    /**
     *  Inserts @p new_rows under @p keys and contiguous IDs claimed from the watermark, growing
     *  the storage on demand, as the core's @c insert does.
     *
     *  @par Statuses:
     *  - @c cancelled_k The index keeps it, as @c status says.
     *  - @c overflow_k The rows pass the ceiling or the reserved range, or @c status keeps it.
     *  - @c contended_k Another batch or load is in flight, or the claim word is closed.
     *  - @c uninitialized_k The @c status keeps it.
     *  - @c config_mismatch_k The @p keys are not one per row, the rows are not as wide as the
     *    index, @p assigned_ids are not one per row, @p context rides another device or cannot
     *    walk, or @c status keeps it.
     *  - @c unsupported_k The rows' scalar has no device cast into the stored one, or @c status
     *    keeps it.
     *  - @c unsupported_layout_k Rows sit closer than one packed row apart, or @c status keeps it.
     */
    expected<insert_result_t> insert(keys_view_t keys, rows_view_t new_rows, node_ids_span_t assigned_ids,
                                     traversal_context_t& context, insert_config_t config = {}) noexcept {
        return core_.insert(keys, new_rows, assigned_ids, context, config);
    }

    /**
     *  Finds the nearest published neighbors of every query row, as the core's @c search does.
     *
     *  @par Statuses:
     *  - @c unsupported_k A scan is asked, which this vendor has none of, or the queries' scalar
     *    has no device cast into the stored one.
     *  - @c config_mismatch_k The outputs hold other than one row per query or fewer than
     *    @c count columns, the queries are not as wide as the index, or @p context rides another
     *    device or cannot walk.
     *  - @c unsupported_layout_k An output's rows stride closer than @c count values, or query
     *    rows sit closer than one packed row apart.
     *  - @c uninitialized_k The index never grew.
     */
    expected<search_result_t> search(rows_view_t queries, nk::matrix_span<node_id_t> result_ids,
                                     nk::matrix_span<distance_t> result_distances, traversal_context_t& context,
                                     search_config_t config = {},
                                     completion_t completion = completion_t::blocking_k) noexcept {
        return core_.search(queries, result_ids, result_distances, context, config, completion);
    }

    /**
     *  The batched @c search ranked by a lightning indexer over @p heads, as the core's does. A
     *  dot verb.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The index scores by another metric than dot, the outputs hold other
     *    than one row per query or fewer than @c count columns, the heads or weights disagree with
     *    the queries, rows are not as wide as the index, or @p context rides another device or
     *    cannot walk.
     *  - @c unsupported_k A scan is asked, which this vendor has none of, or the queries' or heads'
     *    scalar has no device cast into the stored one.
     *  - @c unsupported_layout_k An output's rows stride closer than @c count values, or query or
     *    head rows sit closer than one packed row apart.
     *  - @c uninitialized_k The index never grew.
     */
    expected<search_result_t> search(rows_view_t queries, indexer_heads<scalar_t> const& heads,
                                     nk::matrix_span<node_id_t> result_ids,
                                     nk::matrix_span<distance_t> result_distances, traversal_context_t& context,
                                     search_config_t config = {},
                                     completion_t completion = completion_t::blocking_k) noexcept
        requires floating_scalar<scalar_t>
    {
        return core_.search(queries, heads, result_ids, result_distances, context, config, completion);
    }

    /**
     *  One query in, the nearest matches ascending in @c context.results, as the core's @c search
     *  does.
     *
     *  @par Statuses:
     *  - @c unsupported_k A scan is asked, which this vendor has none of.
     *  - @c config_mismatch_k The query is not as wide as the index, or @p context rides another
     *    device or cannot walk.
     *  - @c uninitialized_k The index never grew.
     */
    expected<search_result_t> search(vector_view_t query_vector, traversal_context_t& context,
                                     search_config_t config = {}) noexcept {
        return core_.search(query_vector, context, config);
    }

    /**
     *  Filtered search, refused since a host predicate cannot run inside the walk.
     *
     *  @par Statuses:
     *  - @c device_error_k Every call.
     */
    template <typename predicate_type_>
    expected<search_result_t> search_if(vector_view_t query_vector, predicate_type_&& predicate,
                                        traversal_context_t& context, search_config_t config = {}) noexcept {
        return core_.search_if(query_vector, std::forward<predicate_type_>(predicate), context, config);
    }

    /**
     *  Graph products of every left-hand row, as the core's @c multiply keeps them. A dot verb.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The index scores by another metric than dot, the outputs or
     *    @p counts disagree with the rows, the rows are not as wide as the index, or @p context
     *    rides another device or cannot walk.
     *  - @c unsupported_k A scan is asked, which this vendor has none of, or the rows' scalar has
     *    no device cast into the stored one.
     *  - @c unsupported_layout_k An output's rows stride closer than its width, or rows sit closer
     *    than one packed row apart.
     *  - @c uninitialized_k The index never grew.
     */
    expected<search_result_t> multiply(rows_view_t left, nk::matrix_span<node_id_t> node_ids,
                                       nk::matrix_span<dot_result_t> products, std::span<u32_t> counts,
                                       traversal_context_t& context, search_config_t config = {},
                                       completion_t completion = completion_t::blocking_k) noexcept {
        return core_.multiply(left, node_ids, products, counts, context, config, completion);
    }

    /**
     *  Grouped-query attention of every token over the published nodes, as the core's @c attend
     *  folds it.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The index scores by another metric than dot, @p outputs or
     *    @p carry disagree with the rows, the queries are not as wide as the index, the heads'
     *    scratch fits no walk, or @p context rides another device or cannot walk.
     *  - @c unsupported_k A scan is asked, which this vendor has none of, or the queries' or
     *    outputs' scalar has no device cast.
     *  - @c unsupported_layout_k Output or query rows sit closer than one packed row apart.
     *  - @c shape_mismatch_k The @p head_count is zero or does not divide the rows, or a walk met
     *    a node with no value row.
     *  - @c uninitialized_k The index never grew.
     */
    template <typename value_lookup_type_>
    expected<search_result_t> attend(rows_view_t queries, std::size_t head_count, value_lookup_type_&& value_lookup,
                                     f32_t score_scale, rows_span_t outputs, std::span<softmax_state_t> carry,
                                     traversal_context_t& context, search_config_t config = {},
                                     completion_t completion = completion_t::blocking_k) noexcept {
        return core_.attend(queries, head_count, std::forward<value_lookup_type_>(value_lookup), score_scale, outputs,
                            carry, context, config, completion);
    }

    /**
     *  Opens a refinement of the nodes in @p nodes on @p context 's stream, as the core's
     *  @c plan_refine does.
     *
     *  @par Statuses:
     *  - @c uninitialized_k The index never grew.
     *  - @c config_mismatch_k The @p context rides another device or cannot walk.
     */
    expected<refine_plan_t> plan_refine(iota_view<node_id_t> nodes, traversal_context_t& context,
                                        refine_config_t config = {},
                                        std::span<node_id_t const> part_bases = {}) noexcept {
        return core_.plan_refine(nodes, context, config, part_bases);
    }

    /**
     *  One refinement of the nodes in @p nodes driven to its end on the calling thread.
     *
     *  @par Statuses:
     *  - @c uninitialized_k The index never grew, or @c status keeps it.
     *  - @c config_mismatch_k The @p context rides another device or cannot walk, or @c status
     *    keeps it.
     *  - @c cancelled_k, @c unsupported_k, @c unsupported_layout_k or @c overflow_k The index
     *    keeps it, as @c status says.
     */
    expected<refine_result_t> refine(iota_view<node_id_t> nodes, traversal_context_t& context,
                                     refine_config_t config = {}, std::span<node_id_t const> part_bases = {}) noexcept {
        return core_.refine(nodes, context, config, part_bases);
    }

    /**
     *  A split of this index into @p receivers, which no dense GPU face serves yet.
     *
     *  @par Statuses:
     *  - @c unsupported_k Always, as no dense GPU face splits.
     */
    expected<refine_plan_t> plan_split(std::span<vram_amd_index*> receivers, traversal_context_t& context,
                                       refine_config_t config = {}, u64_t seed = 0) noexcept {
        return core_.plan_split(receivers, context, config, seed);
    }

    /**
     *  Opens a reachability audit of the rows published now on @p context 's stream.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The @p context rides another device.
     */
    expected<health_plan_t> plan_health(traversal_context_t& context) noexcept { return core_.plan_health(context); }

    /**
     *  Measures forward reachability from the landmarks one search draws, on the calling thread.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The @p context rides another device.
     */
    expected<health_result_t> health(traversal_context_t& context) noexcept { return core_.health(context); }

    /**
     *  Opens a save into @p columns under @p synchronization on @p stream.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The @p synchronization is @c concurrent_k, or @p columns are too
     *    short for the rows.
     *  - @c uninitialized_k The index never grew.
     */
    expected<save_plan_t> plan_save(columns_t columns, synchronization_t synchronization,
                                    typename executor_t::stream_t const& stream) const noexcept {
        return core_.plan_save(columns, synchronization, stream);
    }

    /**
     *  Serializes every row under the watermark into @p columns on the calling thread.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The @p synchronization is @c concurrent_k, or @p columns are too
     *    short for the rows.
     *  - @c uninitialized_k The index never grew.
     */
    status_t save(columns_t columns, synchronization_t synchronization, nk::matrix_span<node_id_t> scratch,
                  typename executor_t::stream_t const& stream) const noexcept {
        return core_.save(columns, synchronization, scratch, stream);
    }

    /**
     *  Opens a load of @p rows at the watermark on @p stream, as the core's @c plan_load does.
     *
     *  @par Statuses:
     *  - @c uninitialized_k The index never grew, or @c status keeps it.
     *  - @c overflow_k The rows pass the ceiling or the reserved range, or @c status keeps it.
     *  - @c contended_k A batch or load is in flight.
     *  - @c cancelled_k, @c config_mismatch_k, @c unsupported_k or @c unsupported_layout_k
     *    The index keeps it, as @c status says.
     */
    expected<load_plan_t> plan_load(std::size_t rows, typename executor_t::stream_t const& stream) noexcept {
        return core_.plan_load(rows, stream);
    }

    /**
     *  Appends the whole source on the calling thread, as the core's one-shot @c load does.
     *
     *  @par Statuses:
     *  - @c uninitialized_k The index never grew, or @c status keeps it.
     *  - @c config_mismatch_k The @p columns or @p keys are shorter than @p rows, refused before a
     *    row is claimed, or @c status keeps it.
     *  - @c overflow_k The rows pass the ceiling or the reserved range, or @c status keeps it.
     *  - @c contended_k A batch or load is in flight.
     *  - @c cancelled_k, @c unsupported_k or @c unsupported_layout_k The index keeps it,
     *    as @c status says.
     */
    status_t load(columns_view_t columns, keys_view_t keys, std::size_t rows, iota_view<node_id_t> source_range,
                  typename executor_t::stream_t const& stream) noexcept {
        return core_.load(columns, keys, rows, source_range, stream);
    }

    /** The core, through which a load plan reads another shard's rows. */
    core_t const& core() const noexcept { return core_; }

  private:

    /** The sharded index walks its shards under one snapshot of their watermarks, merges their
     *  attends on its first device, and publishes or fails their batches together; a compaction
     *  maps every shard of a fold bin for the bin's other devices. */
    template <typename, nk_dtype_t, typename, typename, typename, typename>
    friend struct vram_amd_indexes;
    template <typename, template <typename> class, typename>
    friend struct utilities::shard_set;
    template <typename, typename, typename>
    friend struct utilities::vram_shard_fanout;
    template <typename, typename>
    friend struct vram_amd_compact_plan;

    expected<node_id_t> claim_(std::size_t rows) noexcept { return core_.claim_(rows); }
    std::size_t claimed_() const noexcept { return core_.claimed_(); }
    void unclaim_(std::size_t first, std::size_t rows) noexcept { core_.unclaim_(first, rows); }
    status_t publish_(std::size_t first, std::size_t rows) noexcept { return core_.publish_(first, rows); }
    status_t fail_(status_t failure) noexcept { return core_.fail_(failure); }
    status_t close_() noexcept { return core_.close_(); }
    void reopen_() noexcept { core_.reopen_(); }
    void forget_() noexcept { core_.forget_(); }
    void publish_keys_(iota_view<node_id_t> range) noexcept { core_.publish_keys_(range); }
    key_t* device_keys_() const noexcept { return core_.device_keys_(); }
    status_t map_nodes_(std::size_t rows) noexcept { return core_.map_nodes_(rows); }
    void share_with_(std::bitset<gpu_devices_capacity_k> devices) noexcept { core_.share_with_(devices); }
    status_t grant_(std::bitset<gpu_devices_capacity_k> devices) noexcept { return core_.grant_(devices); }
    std::size_t exported_bytes_(rows_span_t outputs) const noexcept { return core_.exported_bytes_(outputs); }

    expected<insert_result_t> submit_insert_(keys_view_t keys, rows_view_t new_rows, node_id_t first_id,
                                             node_ids_span_t assigned_ids, traversal_context_t& context,
                                             insert_config_t config) noexcept {
        return core_.submit_insert_(keys, new_rows, first_id, assigned_ids, context, config);
    }

    template <typename heads_type_ = dummy_heads_t>
    expected<search_result_t> submit_search_(rows_view_t queries, nk::matrix_span<node_id_t> result_ids,
                                             nk::matrix_span<distance_t> result_distances, traversal_context_t& context,
                                             search_config_t config, std::size_t visible,
                                             heads_type_ const& heads = {}) noexcept {
        return core_.kernels().submit_search_(core_, queries, result_ids, result_distances, context, config, visible,
                                              heads);
    }

    expected<search_result_t> submit_multiply_(rows_view_t left, nk::matrix_span<node_id_t> node_ids,
                                               nk::matrix_span<dot_result_t> products, std::span<u32_t> counts,
                                               traversal_context_t& context, search_config_t config,
                                               std::size_t visible) noexcept {
        return core_.kernels().submit_multiply_(core_, left, node_ids, products, counts, context, config, visible);
    }

    template <typename value_lookup_type_>
    expected<f32_t*> submit_attend_(rows_view_t queries, std::size_t head_count, value_lookup_type_ const& value_lookup,
                                    f32_t score_scale, traversal_context_t& context, search_config_t config,
                                    std::size_t blocks, std::size_t exported_bytes, std::size_t visible) noexcept {
        return core_.kernels().submit_attend_(core_, queries, head_count, value_lookup, score_scale, context, config,
                                              blocks, exported_bytes, visible);
    }

    status_t launch_resolve_matches_(node_id_t const* ids, distance_t const* distances, u32_t walked,
                                     byte_t const* keys, vram_ranking_t const& ranking, u32_t queries_count,
                                     stream_handle_t stream) const noexcept {
        return core_.launch_resolve_matches_(ids, distances, walked, keys, ranking, queries_count, stream);
    }

    status_t launch_merge_matches_(vram_ranking_t const& gathered, std::size_t block_bytes, u32_t shards,
                                   vram_ranking_t const& ranking, u32_t queries_count,
                                   stream_handle_t stream) const noexcept {
        return core_.launch_merge_matches_(gathered, block_bytes, shards, ranking, queries_count, stream);
    }

    status_t launch_merge_attention_(byte_t* blocks, u32_t rows, u32_t columns, u32_t shards,
                                     stream_handle_t stream) noexcept {
        return core_.launch_merge_attention_(blocks, rows, columns, shards, stream);
    }

    status_t landing_refusal_(std::size_t rows, rows_span_t outputs,
                              std::span<softmax_state_t const> carry) const noexcept {
        return core_.landing_refusal_(rows, outputs, carry);
    }

    status_t land_attended_(f32_t* landed, std::size_t blocks, rows_span_t outputs, std::span<softmax_state_t> carry,
                            stream_handle_t stream) const noexcept {
        return core_.land_attended_(landed, blocks, outputs, carry, stream);
    }

    core_t core_;
};

} // namespace unum::usearch::dense
