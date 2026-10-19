/**
 *  @file include/usearch/dense/vram_nvidia_index.hpp
 *  @author Ash Vardanian
 *  @date July 22, 2026
 *  @brief Vector Search index for a single uniform latency memory space on NVIDIA GPUs.
 *
 *  The host face of the NVIDIA family, compiled by any C++ compiler: it fills the plain argument
 *  records of @c vram_index.hpp and launches the kernels of @c vram_nvidia_index.cuh by name
 *  through the driver it opens at first use, so it never sees a kit, a shared view or the CUDA
 *  runtime. Every kernel is one instantiation of a single template over five arguments - the
 *  kernel, the capability ceiling, the dtype, the register rung and the attend value lookup - so
 *  @c vram_kernel_symbol spells its symbol here from those arguments alone, and one fatbin per
 *  class, which the face's kernels type names, carries every dtype.
 */
#pragma once

#include <cstddef> // `std::size_t`
#include <cstring> // `std::memcpy` - an attend's value lookup travels as bytes

#include <bit>         // `std::bit_ceil` - the lanes sharing a row
#include <limits>      // `std::numeric_limits` - the unit-RMS import's epsilon
#include <numbers>     // `std::numbers::log2e_v` - the base-2 softmax
#include <type_traits> // `std::is_same_v` - which heads a walk stages
#include <utility>     // `std::move` - the loaded module

#include <numkong/capabilities.h> // `nk_kernel_kind_t` & the kernel-kind constants
#include <numkong/cast/serial.h>  // `nk_cast_plan_serial_` - the device casts' plans
#include <numkong/types.hpp>      // `nk::type_for` - the stored scalar of a dtype

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/vram_device.hpp" // `vram_kernel_symbol` - a kernel's symbol
#include "usearch/utilities/vram_nvidia_device.hpp"
#include "usearch/dense/types.hpp"
#include "usearch/dense/vram_index.hpp"

namespace unum::usearch::dense {

/** The device value lookup an attend kernel resolves rows through, by the @c device_lookup_k a
 *  lookup type names itself with: whichever unit compiles a fatbin specializes it with @c type for
 *  every lookup that fatbin's attend kernels take. */
template <u32_t lookup_>
struct cuda_value_lookup_of;

#pragma region Kernels

/** Queries and rows a scanning block multiplies per step, the edge of its square tile, which four
 *  warps split into 32 by 32. */
inline constexpr u32_t cuda_scan_tile_k = 64;

/** Bytes of one scanning block's staged candidates, two tiles of rows for each of its queries. */
inline constexpr u32_t cuda_scan_staged_bytes_k = 2 * cuda_scan_tile_k * cuda_scan_tile_k *
                                                  static_cast<u32_t>(gpu_scored_node_bytes<node_id_t>);

/** Row ranges a scan of @p blocks blocks of queries over @p visible rows splits into on a
 *  persistent grid of @p grid blocks, @p resident_blocks of them per multiprocessor: enough work
 *  items for every block to claim one per block its multiprocessor seats, none narrower than a tile
 *  of rows, and no more than the merge's heads and their taken counts fit in the staged candidates'
 *  shared memory. */
constexpr u32_t cuda_scan_ranges(u32_t grid, u32_t resident_blocks, u32_t blocks, u32_t visible) noexcept {
    u32_t const claimed = divide_round_up(grid * resident_blocks, blocks);
    u32_t const tiles = max_of(divide_round_up(visible, cuda_scan_tile_k), 1u);
    u32_t const merged = cuda_scan_staged_bytes_k /
                         (vram_warps_per_block_k *
                          static_cast<u32_t>(gpu_scored_node_bytes<node_id_t> + sizeof(u32_t)));
    return min_of(min_of(claimed, tiles), merged);
}

/** Blocks of @p kernel on its @p slots rung resident per multiprocessor under @p capability_ over
 *  @p dtype_ - the register budget its launch bounds set, the load window its body sizes, and its
 *  grid's share of every multiprocessor: three for the builders up to eight slots and two past
 *  them, the back-links riding their grid and arena, the most whose share of the register file
 *  holds the walk and its selection without spilling; a walk's by its rung; three for the scan, as
 *  many as the shared memory of the widest parts holds the staged candidates of; eight for the
 *  sweeps striding the graph, oversubscribed so enough rows stay in flight; and one for the rest,
 *  whose grids follow their work. */
template <nk_capability_t capability_, nk_dtype_t dtype_>
constexpr u32_t cuda_resident_blocks(vram_kernel_t kernel, u32_t slots) noexcept {
    switch (kernel) {
    // Three blocks' budget spills the 16-slot walks, and the back-links ride their walks' grids.
    case vram_kernel_t::insert_k:
    case vram_kernel_t::back_link_k:
    case vram_kernel_t::refine_k:
    case vram_kernel_t::merge_back_links_k: return slots <= 8 ? 3 : 2;
    case vram_kernel_t::search_k:
    case vram_kernel_t::search_indexer_k:
    case vram_kernel_t::multiply_k: return slots <= 8 ? 4 : 3;
    case vram_kernel_t::scan_k: return 3;
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

/** One instantiated kernel: the kernel and its register rung, zero where it keeps no ladder or
 *  walks the arena frontier. */
struct cuda_rung_t {
    vram_kernel_t kernel;
    u32_t slots;
};

/** Every instantiation one dtype's code exports bar the attends, which the device pass instantiates
 *  and the host looks up as it loads the fatbin. */
inline constexpr cuda_rung_t cuda_rungs[] = {
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
    {vram_kernel_t::scan_k, 0},
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
inline constexpr u32_t cuda_walk_rungs[] = {0, 2, 4, 8, 16};

/** Whether kernels under @p capability wait on their stream predecessor at entry, so their launches
 *  may overlap its drain: programmatic dependent launch, Hopper on. */
constexpr bool cuda_overlaps_launches(nk_capability_t capability) noexcept {
    return (capability & (nk_cap_hopper_k | nk_cap_blackwell_k | nk_cap_blackwellrtx_k)) != 0;
}

/** Whether @p kernel walks, scoring its batches through the schedule, which may gather their rows
 *  through the tensor map the kernel's arguments carry. */
constexpr bool cuda_gathers_rows(vram_kernel_t kernel) noexcept {
    switch (kernel) {
    case vram_kernel_t::insert_k:
    case vram_kernel_t::refine_k:
    case vram_kernel_t::search_k:
    case vram_kernel_t::search_indexer_k:
    case vram_kernel_t::multiply_k:
    case vram_kernel_t::attend_k: return true;
    default: return false;
    }
}

/** The one argument every NVIDIA dense kernel takes: the records every vendor's kernels read, and
 *  the vectors matrix as the tensor map a walk may gather its rows through. */
struct cuda_arguments_t : vram_arguments_t {
    cuda_rows_map_t rows_map;
};

#pragma endregion Kernels

/**
 *  @brief Vector Search index for a single uniform latency memory space on NVIDIA GPUs.
 *
 *  Everything but the kernels is @c vram_index; see @ref dense_vram_nvidia_index_concurrency for
 *  how a batch's rows publish on the device. The kernels come from the fatbin @p kernels_type_
 *  hands out through its static @c bytes, compiled under @p capability_.
 */
template <                                                                      //
    nk_dtype_t vector_dtype_,                                                   //
    nk_capability_t capability_,                                                //
    typename kernels_type_,                                                     //
    typename vectors_allocator_type_ = cuda_virtual_memory_allocator<byte_t>,   //
    typename neighbors_allocator_type_ = cuda_virtual_memory_allocator<byte_t>, //
    typename traversals_allocator_type_ = cuda_arena_allocator_t>               //
struct vram_nvidia_index
    : vram_index<vram_nvidia_index<vector_dtype_, capability_, kernels_type_, vectors_allocator_type_,
                                   neighbors_allocator_type_, traversals_allocator_type_>,
                 cuda_vendor_t, vector_dtype_, vectors_allocator_type_, neighbors_allocator_type_,
                 traversals_allocator_type_> {

    using base_t = vram_index<vram_nvidia_index, cuda_vendor_t, vector_dtype_, vectors_allocator_type_,
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

    /** Sixteen bytes every NVIDIA kit loads at once - the host's spelling of the kernels' @c uint4,
     *  which only the CUDA SDK declares. */
    struct alignas(16) fragment_t {
        u32_t x, y, z, w;
    };
    static_assert(sizeof(fragment_t) == base_t::fragment_bytes_k);

    /** The capability of the quads kit the ladder picked under @c capability_, surfaced in the
     *  bench reports; read only where the NVIDIA kits are visible, as its initializer is. */
    inline static constexpr nk_capability_t picked_quads_capability_k = quads_ladder<scalar_t, capability_>::picked_k;

    /** Lanes of an NVIDIA warp, which the register ladder's rungs count ranks in. */
    static constexpr u32_t wave_lanes() noexcept { return 32; }

    /** Lanes cooperating on one row: the smallest power of two covering its fragments, at most one
     *  per gather in flight, past which a wider row costs reductions where it could cost steps. */
    u32_t lanes_per_row() const noexcept {
        return min_of(std::bit_ceil(row_fragments()), vram_max_gathers_in_flight_k);
    }

    /** Whether a search, multiply or attend launch seats on this device: per warp, the query span
     *  and the @p fold_bytes an attend keeps past it, beside the static batch buffers. Defined in
     *  @c vram_nvidia_index.cuh, where the kits are visible. */
    bool walk_scratch_fits(std::size_t fold_bytes = 0) const noexcept;

    /** Whether a builder launch seats @p scratch_bytes_per_warp of per-warp scratch in shared
     *  memory beside the batch buffers and schedule state, on every block a multiprocessor keeps
     *  resident; when it cannot, the warp's arena slice keeps it, as Apple's always does. Defined
     *  in @c vram_nvidia_index.cuh. */
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
        bool const wide = expansion > vram_widest_candidate_slots_k * 32;
        u32_t const count_slots = divide_round_up<u32_t>(config.count, 32);
        u32_t const candidate_slots = candidate_slots_for_(expansion, count_slots > 2 ? count_slots : 2);
        gpu_arena_shape_t const arena_shape = arena_shape_for(
            walk_selection_rows_(expansion, candidate_slots),
            wide ? frontier_kind_t::wide_arena_k : frontier_kind_t::register_ladder_k);
        // A scan walks no arena, and keeps its matches past the outputs on a fragment boundary; the
        // walks seat as many blocks as a search on the same rung, an attend no more.
        bool const scans = config.traversal == traversal_t::scan_k;
        u32_t const resident_blocks = resident_blocks_(vram_kernel_t::search_k, wide ? 0 : candidate_slots);
        std::size_t const arena_warps =
            scans ? 0 : std::size_t {walk_blocks_(rows, resident_blocks)} * vram_warps_per_block_k;
        std::size_t const scan_bytes =
            scans ? sizeof(fragment_t) + scan_scratch_bytes_(rows, config.count, base_t::max_node_count_k) : 0;
        std::size_t const staging_bytes = rows * row_bytes() + import_bytes_(rows, nk_f64_k) +
                                          rows * config.count * (sizeof(node_id_t) + sizeof(distance_t)) +
                                          rows * sizeof(u32_t) + scan_bytes;
        return aligned(sizeof(vram_telemetry_block_t)) + 2 * aligned(sizeof(u32_t)) +
               aligned(arena_warps * arena_shape.slice_bytes<node_id_t>()) + aligned(staging_bytes);
    }

  private:
    friend base_t;

    /** The sharded index walks its shards under one snapshot of their watermarks, merges their
     *  attends on its first device, and publishes or fails their batches together. */
    template <typename, nk_dtype_t, nk_capability_t, typename, typename>
    friend struct vram_nvidia_indexes;
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

    /** What every walking verb stages ahead of its sink: the queries, the walk's breadth and
     *  arena, the frontier the walks keep, the staging past the uploads, and any heads the walks
     *  rescore by - which the kernel side turns into its search view and rescoring. */
    struct walk_launch_t {

        /** The staged query rows, @b [queries,row_fragments], device-resident. */
        fragment_t const* queries = nullptr;
        u32_t queries_count = 0;

        /** The watermark snapshotted at staging: landmarks are drawn below it and no neighbor at or
         *  past it is followed. */
        u32_t visible = 0;

        /** Bytes of the per-warp dynamic-shared slice: the query span, and an attend's fold past
         *  it. */
        u32_t scratch_bytes_per_warp = 0;

        /** Requested expansion; past the register ladder it is the wide walk's budget. */
        u32_t expansion = 0;

        /** The traversal arena every warp carves its slice from, and the slice's shape. */
        byte_t* arena = nullptr;
        gpu_arena_shape_t arena_shape {};

        /** The context's epoch floor at launch. */
        u32_t epoch_base = 0;

        /** The register ladder's width, when the ladder serves. */
        u32_t candidate_slots = 0;

        /** Warps the device seats on the walk's rung, and the rows whose neighbors fill a
         *  walk's candidates: the two shares @ref walk_width splits. */
        u32_t seats = 0;
        u32_t filling_rows = 0;

        /** Which frontier the walks keep. */
        frontier_kind_t frontier = frontier_kind_t::register_ladder_k;

        /** The staging bytes past the queries and any heads, where the sink writes its outputs. */
        byte_t* outputs = nullptr;

        /** The heads staged beside the queries, their weights under an indexer, and the heads per
         *  query - null and zero for a plain walk. */
        fragment_t const* heads = nullptr;
        f32_t const* weights = nullptr;
        u32_t heads_count = 0;
    };

    /** Validates and uploads @p queries, then any indexer or attend @p heads and the indexer's
     *  weights, sizes the arena for @p expansion unless the @p traversal scans, and reserves
     *  @p outputs_bytes of staging past them: everything a walking or scanning verb stages before
     *  its sink, bound below the @p visible rows. Until the context drains, the queries, the heads
     *  and the context stay live. */
    template <typename heads_type_>
    expected<walk_launch_t> stage_walk_(rows_view_t queries, [[maybe_unused]] heads_type_ const& heads,
                                        traversal_context_t& context, traversal_t traversal, u32_t expansion,
                                        u32_t minimum_slots, std::size_t outputs_bytes, std::size_t visible) noexcept {
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
        cuda_stream_handle_t const stream = context.stream();

        // Past the register ladder's widest rung the arena-backed walk serves.
        frontier_kind_t const frontier = expansion > vram_widest_candidate_slots_k * 32
                                             ? frontier_kind_t::wide_arena_k
                                             : frontier_kind_t::register_ladder_k;
        u32_t const candidate_slots = candidate_slots_for_(expansion, minimum_slots);
        gpu_arena_shape_t const arena_shape = arena_shape_for(walk_selection_rows_(expansion, candidate_slots),
                                                              frontier);
        // A scan walks nothing, so it neither sizes nor zeroes an arena; a search, an indexed
        // search and a multiply seat as many blocks on one rung.
        bool const walks = traversal == traversal_t::walk_k;
        u32_t const resident_blocks = resident_blocks_(attend_k ? vram_kernel_t::attend_k : vram_kernel_t::search_k,
                                                       frontier == frontier_kind_t::wide_arena_k ? 0 : candidate_slots);
        // Every warp the device seats on this rung, and the rows whose neighbors fill a walk's
        // candidates.
        u32_t const seats = device_.multiprocessor_count * resident_blocks * vram_warps_per_block_k;
        u32_t const filling = filling_rows<u32_t>(expansion, static_cast<u32_t>(connectivity()));
        std::size_t const arena_warps = std::size_t {walk_blocks_(queries_count, resident_blocks)} *
                                        vram_warps_per_block_k;
        std::size_t const arena_bytes_before = context.arena_bytes_;
        byte_t* const arena = walks ? context.ensure_arena(arena_warps * arena_shape.slice_bytes<node_id_t>())
                                    : nullptr;
        if (walks && !arena) return {{}, status_t::bad_alloc_k};
        // The staging holds the query rows and any head rows first, since they carry the widest
        // alignment, then the window importing them, the heads' weights, then the outputs; stream
        // order serializes its reuse across calls.
        std::size_t const rows_bytes = (queries_count + heads_rows) * row_bytes();
        std::size_t const import_bytes = import_bytes_(attend_k ? heads_rows : queries_count, queries.dtype);
        std::size_t const weights_bytes = indexer_k ? heads_rows * sizeof(f32_t) : 0;
        byte_t* const staging = context.ensure_staging(rows_bytes + import_bytes + weights_bytes + outputs_bytes);
        if (!staging) return {{}, status_t::bad_alloc_k};
        byte_t* const window = staging + rows_bytes;
        u32_t epoch_base = 0;
        if (walks) {
            auto const [first_epoch, epochs_status] = context.take_epochs(queries_count, arena_bytes_before);
            if (failed(epochs_status)) return {{}, epochs_status};
            epoch_base = first_epoch;
        }
        // The counters zero ahead of the imports, so a cast kernel importing the queries leads
        // straight into the walk, which may then overlap its drain.
        if (cuda_fill_async(context.telemetry_.get(), 0, sizeof(vram_telemetry_block_t), stream) != cuda_success_k ||
            cuda_fill_async(context.assigned_rows_.get(), 0, 2 * sizeof(u32_t), stream) != cuda_success_k)
            return {{}, status_t::device_error_k};
        // Pitched uploads: device rows pad to whole fragments, and the pad bytes must be zero for
        // the zero-tail distance identity to hold.
        if ((import_bytes || packed_row_bytes() != row_bytes()) &&
            cuda_fill_async(staging, 0, rows_bytes + import_bytes, stream) != cuda_success_k)
            return {{}, status_t::device_error_k};
        if (status_t const imported = import_rows_(staging, queries, window, stream); failed(imported))
            return {{}, imported};

        walk_launch_t walk;
        walk.queries = reinterpret_cast<fragment_t const*>(staging);
        walk.queries_count = static_cast<u32_t>(queries_count);
        walk.visible = static_cast<u32_t>(visible);
        walk.scratch_bytes_per_warp = static_cast<u32_t>(row_bytes());
        walk.expansion = expansion;
        walk.arena = arena;
        walk.arena_shape = arena_shape;
        walk.epoch_base = epoch_base;
        walk.candidate_slots = candidate_slots;
        walk.seats = seats;
        walk.filling_rows = filling;
        walk.frontier = frontier;
        walk.outputs = staging + rows_bytes + import_bytes + weights_bytes;
        if constexpr (indexer_k) {
            byte_t* const staged_heads = staging + queries_count * row_bytes();
            byte_t* const staged_weights = staging + rows_bytes + import_bytes;
            std::size_t const weights_row_bytes = heads.weights.extent(1) * sizeof(f32_t);
            if (failed(import_rows_(staged_heads, heads.heads, window, stream)) ||
                cuda_copy_2d_async(staged_weights, weights_row_bytes, heads.weights.byte_data(),
                                   heads.weights.stride_bytes(0), weights_row_bytes, queries_count,
                                   stream) != cuda_success_k)
                return {{}, status_t::device_error_k};
            walk.heads = reinterpret_cast<fragment_t const*>(staged_heads);
            walk.weights = reinterpret_cast<f32_t const*>(staged_weights);
            walk.heads_count = static_cast<u32_t>(heads.weights.extent(1));
        }
        if constexpr (attend_k) {
            byte_t* const staged_heads = staging + queries_count * row_bytes();
            if (status_t const imported = import_rows_(staged_heads, heads.rows, window, stream); failed(imported))
                return {{}, imported};
            walk.heads = reinterpret_cast<fragment_t const*>(staged_heads);
            walk.heads_count = static_cast<u32_t>(heads.head_count);
        }
        return {walk, status_t::success_k};
    }

    /** The submission half of @c search: stages the queries and @p heads, launches walks over the
     *  @p visible rows, or a scan of them past its outputs, and queues the pitched writeback. Until
     *  the context drains, the queries, the heads, both outputs and the context stay live; across
     *  contexts they must not share a buffer at all. */
    template <typename heads_type_ = dummy_heads_t>
    expected<search_result_t> submit_search_(rows_view_t queries, nk::matrix_span<node_id_t> result_ids,
                                             nk::matrix_span<distance_t> result_distances, traversal_context_t& context,
                                             search_config_t config, std::size_t visible,
                                             heads_type_ const& heads = {}) noexcept {
        std::size_t const queries_count = queries.rows;
        bool const scans = config.traversal == traversal_t::scan_k;
        // Indexer heads rescore the batches a walk gathers, and a scan gathers none.
        if (scans && !std::is_same_v<heads_type_, dummy_heads_t>) return {{}, status_t::unsupported_k};
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
        // A scan's matches follow the outputs on the next fragment boundary.
        std::size_t const scratch_offset = round_up_to_multiple<std::size_t>(ids_bytes + distances_bytes,
                                                                             sizeof(fragment_t));
        std::size_t const outputs_bytes = scans ? scratch_offset +
                                                      scan_scratch_bytes_(queries_count, config.count, visible)
                                                : ids_bytes + distances_bytes;
        auto const staged = stage_walk_(queries, heads, context, config.traversal, expansion,
                                        count_slots > 2 ? count_slots : 2, outputs_bytes, visible);
        if (!staged) return {{}, staged.status};
        node_id_t* const device_ids = reinterpret_cast<node_id_t*>(staged.value.outputs);
        distance_t* const device_distances = wants_distances
                                                 ? reinterpret_cast<distance_t*>(staged.value.outputs + ids_bytes)
                                                 : nullptr;
        cuda_stream_handle_t const stream = context.stream();
        status_t const launched = scans ? launch_scan_(staged.value, staged.value.outputs + scratch_offset, device_ids,
                                                       device_distances, config.count, context)
                                        : launch_search_(staged.value, device_ids, device_distances, config.count,
                                                         context);
        if (failed(launched)) return {{}, launched};
        // The landing may be wider than the count, so a multi-index can hand every shard its own
        // columns of one results matrix.
        if (cuda_copy_2d_async(result_ids.data(), result_ids.stride_bytes(0), device_ids,
                               config.count * sizeof(node_id_t), config.count * sizeof(node_id_t), queries_count,
                               stream) != cuda_success_k)
            return {{}, status_t::device_error_k};
        if (wants_distances &&
            cuda_copy_2d_async(result_distances.data(), result_distances.stride_bytes(0), device_distances,
                               config.count * sizeof(distance_t), config.count * sizeof(distance_t), queries_count,
                               stream) != cuda_success_k)
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
        auto const staged = stage_walk_(left, dummy_heads_t {}, context, config.traversal,
                                        non_zero_expansion_factor(config), 2,
                                        ids_bytes + products_bytes + rows_count * sizeof(u32_t), visible);
        if (!staged) return {{}, staged.status};
        node_id_t* const device_ids = reinterpret_cast<node_id_t*>(staged.value.outputs);
        dot_t* const device_products = reinterpret_cast<dot_t*>(staged.value.outputs + ids_bytes);
        u32_t* const device_counts = reinterpret_cast<u32_t*>(staged.value.outputs + ids_bytes + products_bytes);
        cuda_stream_handle_t const stream = context.stream();
        if (status_t const launched = launch_multiply_(staged.value, device_ids, device_products, device_counts,
                                                       static_cast<u32_t>(wanted), context);
            failed(launched))
            return {{}, launched};
        if (wanted &&
            (cuda_copy_2d_async(node_ids.data(), node_ids.stride_bytes(0), device_ids, wanted * sizeof(node_id_t),
                                wanted * sizeof(node_id_t), rows_count, stream) != cuda_success_k ||
             cuda_copy_2d_async(products.data(), products.stride_bytes(0), device_products, wanted * sizeof(dot_t),
                                wanted * sizeof(dot_t), rows_count, stream) != cuda_success_k))
            return {{}, status_t::device_error_k};
        if (cuda_copy_async(counts.data(), device_counts, rows_count * sizeof(u32_t), stream) != cuda_success_k)
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
        auto staged = stage_walk_(first_heads, attend_heads_t {queries, head_count}, context, config.traversal,
                                  non_zero_expansion_factor(config), 2,
                                  blocks * (outputs_bytes + rows_count * sizeof(softmax_state_t)) + exported_bytes,
                                  visible);
        if (!staged) return {nullptr, staged.status};
        staged.value.scratch_bytes_per_warp += static_cast<u32_t>(scratch_bytes);
        if (cuda_fill_async(context.missed_.get(), 0, sizeof(u32_t), context.stream()) != cuda_success_k)
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

    /** What one refinement pass hands its launches: the walked rows, the pass-local buffers, and
     *  the walk's arena, which the kernel side turns into its refine views. */
    struct refine_launch_t {

        /** The walked rows: an ID range, or under @c node_ids the offsets into that array. */
        iota_view<node_id_t> nodes;

        /** Optional device-resident IDs to walk instead of the range. */
        u32_t const* node_ids = nullptr;

        /** Rows published when the pass began: the walks' sampling bound and the merge's sweep. */
        u32_t nodes_count = 0;

        /** The shadow neighbors the walks land, @b [rows,connectivity]. */
        u32_t* shadow_neighbors = nullptr;

        /** The reverse buckets @b [rows,bucket_width], their fill counters, and the bitset of rows
         *  the merge re-selects. */
        u32_t* buckets = nullptr;
        u32_t* bucket_counts = nullptr;
        vram_bitset_t touched;
        u32_t bucket_width = 0;

        /** The walk's and the merge's claim counters, zeroed with the buckets', read once the pass
         *  landed. */
        u32_t* claims = nullptr;

        /** The traversal arena, the shape of one warp's slice, the arena frontier's expansion, and
         *  the epoch floor at launch. */
        byte_t* arena = nullptr;
        gpu_arena_shape_t arena_shape {};
        u32_t expansion = 0;
        u32_t epoch_base = 0;

        /** The register ladder's width the walks keep, zero for the arena frontier. */
        u32_t candidate_slots = 0;

        /** The parts' bases on the device, how many, and the round of their merge; fewer than two
         *  parts for a plain refine. */
        u32_t const* part_bases = nullptr;
        u32_t parts = 0;
        u32_t round = 0;
    };

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
        cuda_stream_handle_t const stream = context.stream();
        // The claims, the reverse buckets' counters and the bitset of rows the merge re-selects lie
        // back to back.
        if (cuda_fill_async(scratch.claims, 0, (2 + nodes_count + scratch.words) * sizeof(u32_t), stream) !=
            cuda_success_k)
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

        refine_launch_t launch;
        launch.nodes = nodes;
        launch.node_ids = node_ids.data();
        launch.nodes_count = static_cast<u32_t>(nodes_count);
        launch.shadow_neighbors = scratch.shadow_neighbors;
        launch.buckets = scratch.buckets;
        launch.bucket_counts = scratch.bucket_counts;
        launch.touched = scratch.touched;
        launch.claims = scratch.claims;
        launch.bucket_width = static_cast<u32_t>(connectivity());
        launch.arena = arena;
        launch.arena_shape = arena_shape;
        launch.expansion = expansion;
        launch.epoch_base = epoch_base;
        launch.candidate_slots = candidate_slots;
        launch.part_bases = parts > 1 ? scratch.part_bases : nullptr;
        launch.parts = parts > 1 ? parts : 0;
        launch.round = round;
        // A range short of the whole graph starts from the live neighbors copied into the shadow
        // verbatim, so a row it never rewrites but one of its picks lands in merges its own
        // neighbors with the new reverse links; islands scatter nothing, so only their own rows
        // merge.
        if (node_ids.empty() && walk_count != nodes_count)
            if (cuda_copy_async(scratch.shadow_neighbors, neighbors_matrix_.base(),
                                nodes_count * connectivity() * sizeof(u32_t), stream) != cuda_success_k)
                return status_t::device_error_k;
        if (status_t const launched = launch_refine_(launch, context); failed(launched)) return launched;

        // Scatter the walked rows' shadow neighbors in reverse, then merge each touched node's
        // forward & reverse candidates into its final live row. A heal round over island IDs
        // re-selects the islands' own rows alone: re-judging the rows they pick would drop the
        // links that reconnected earlier islands.
        return launch_refine_back_links_(launch, context);
    }

    /** Queues a reachability audit from as many stride-spread landmarks as one search draws - the
     *  connectivity - and its first sweep, through @ref queue_sweeps_. Nothing waits; the caller
     *  reads @p landing once its event fired, and queues further sweeps from sweep 2 while the last
     *  one still expanded. */
    status_t queue_audit_(traversal_context_t& context, refine_scratch_t const& scratch,
                          audit_landing_t* landing) noexcept {
        std::size_t const rows = size();
        cuda_stream_handle_t const stream = context.stream();
        // The bitsets, the levels, the sweeps' expanded counters and the unreached counter lie back
        // to back.
        if (cuda_fill_async(scratch.reached.words, 0, (scratch.words * 3 + 4) * sizeof(u32_t), stream) !=
            cuda_success_k)
            return status_t::device_error_k;
        if (status_t const launched = launch_audit_landmarks_(scratch.reached, scratch.frontier, rows,
                                                              min_of<std::size_t>(connectivity(), rows), stream);
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
        cuda_stream_handle_t const stream = context.stream();
        // A sweep queued past convergence finds the one before it expanded nothing and leaves at
        // entry; odd sweeps grow the first frontier, even ones the second.
        for (u32_t sweep_number = first_sweep; sweep_number != 2 * first_sweep; ++sweep_number) {
            bool const odd = sweep_number & 1u;
            vram_bitset_t const frontier = odd ? scratch.frontier : scratch.next_frontier;
            vram_bitset_t const next_frontier = odd ? scratch.next_frontier : scratch.frontier;
            u32_t* const expanded = scratch.expanded + odd;
            u32_t const* const expanded_before = sweep_number == 1 ? nullptr : scratch.expanded + !odd;
            if (cuda_fill_async(next_frontier.words, 0, scratch.words * sizeof(u32_t), stream) != cuda_success_k ||
                cuda_fill_async(expanded, 0, sizeof(u32_t), stream) != cuda_success_k)
                return status_t::device_error_k;
            if (status_t const launched = launch_audit_sweep_(scratch.reached, frontier, next_frontier, expanded,
                                                              expanded_before, scratch.levels, rows, stream);
                failed(launched))
                return launched;
        }
        if (cuda_fill_async(scratch.unreached, 0, sizeof(u32_t), stream) != cuda_success_k)
            return status_t::device_error_k;
        if (status_t const launched = launch_audit_islands_(scratch.reached, rows, scratch.island_ids,
                                                            scratch.unreached, stream);
            failed(launched))
            return launched;
        return land_counters_(scratch, landing, 2 * first_sweep - 1, stream);
    }

    /** Queues the reconnection of the @p islands islands the last audit in @p scratch collected
     *  over @p rows rows: the reached in-links first, so no reconnection takes the last link a
     *  reached node is found through, then each island into a reached pick or its landmark. */
    status_t queue_reconnect_(refine_scratch_t const& scratch, u32_t islands, std::size_t rows,
                              cuda_stream_handle_t stream) noexcept {
        if (cuda_fill_async(scratch.in_links, 0, rows * sizeof(u32_t), stream) != cuda_success_k)
            return status_t::device_error_k;
        vram_arguments_t arguments {};
        vram_graph_t& graph = arguments.graph;
        graph.neighbors = neighbor_rows_();
        graph.rows_count = static_cast<u32_t>(rows);
        graph.connectivity = static_cast<u32_t>(connectivity());
        graph.reached = scratch.reached;
        graph.in_links = scratch.in_links;
        graph.island_ids = scratch.island_ids;
        graph.islands_count = islands;
        graph.seed_count = static_cast<u32_t>(min_of<std::size_t>(connectivity(), rows));
        graph.reconnected = scratch.reconnected;
        if (status_t const counted = launch_(
                kernel_function_(vram_kernel_t::reached_in_links_k, 0),
                device_.multiprocessor_count * resident_blocks_(vram_kernel_t::reached_in_links_k, 0), 0, stream,
                arguments);
            failed(counted))
            return counted;
        return launch_(kernel_function_(vram_kernel_t::reconnect_island_k, 0),
                       divide_round_up<u32_t>(islands, block_threads_k), 0, stream, arguments);
    }

#pragma endregion Refinement

#pragma region Launches

    /** The kernels the base names when it sizes an arena for their grid. */
    using kernel_t = vram_kernel_t;

    /** Blocks of @p kernel on its @p slots rung resident per multiprocessor, as its launch bounds
     *  read them; every grid and the arena behind it are sized by it. */
    static constexpr u32_t resident_blocks_(vram_kernel_t kernel, u32_t slots) noexcept {
        return cuda_resident_blocks<capability_, vector_dtype_>(kernel, slots);
    }

    /** Bytes of a scan's scratch over @p queries queries keeping @p count matches apiece among
     *  @p visible rows: every block of queries' count of finished ranges, then each query's matches
     *  in every range @ref cuda_scan_ranges splits the rows into. */
    std::size_t scan_scratch_bytes_(std::size_t queries, std::size_t count, std::size_t visible) const noexcept {
        u32_t const blocks = divide_round_up<u32_t>(static_cast<u32_t>(queries), cuda_scan_tile_k);
        u32_t const resident_blocks = resident_blocks_(vram_kernel_t::scan_k, 0);
        u32_t const ranges = cuda_scan_ranges(device_.multiprocessor_count * resident_blocks, resident_blocks, blocks,
                                              static_cast<u32_t>(visible));
        return round_up_to_multiple<std::size_t>(blocks, 4) * sizeof(u32_t) +
               queries * ranges * count * gpu_scored_node_bytes<node_id_t>;
    }

    /** Threads per block, one warp's worth per resident warp. */
    inline static constexpr u32_t block_threads_k = vram_warps_per_block_k * 32;

    /** Loads the fatbin of @p kernels_type_ onto the device and finds every kernel this dtype
     *  exports bar the attends, by the symbol @ref vram_kernel_symbol spells. */
    status_t load_kernels_() noexcept {
        auto [module, status] = cuda_module_t::load(device_.ordinal, kernels_type_::bytes());
        if (failed(status)) return status;
        module_ = std::move(module);
        vram_kernel_symbol_t symbol;
        for (cuda_rung_t const rung : cuda_rungs) {
            vram_kernel_symbol(symbol, "dense", "cuda_kernel", "cuda_arguments_t", rung.kernel, capability_,
                               vector_dtype_, rung.slots, u32_t {0});
            cuda_function_handle_t& function = kernels_[static_cast<u32_t>(rung.kernel)][gpu_rung_index(rung.slots)];
            if (!(function = module_.function(symbol.text))) return status_t::device_error_k;
        }
        return status_t::success_k;
    }

    /** The loaded @p kernel on its @p slots rung. */
    cuda_function_handle_t kernel_function_(vram_kernel_t kernel, u32_t slots) const noexcept {
        return kernels_[static_cast<u32_t>(kernel)][gpu_rung_index(slots)];
    }

    /** Launches @p function over @p blocks of the standard warp complement on @p stream, raising
     *  its dynamic-shared ceiling first when @p shared_bytes outgrow the default window, with
     *  @p rows_map beside @p arguments for a walk to gather its rows through. */
    static status_t launch_(cuda_function_handle_t function, u32_t blocks, u32_t shared_bytes,
                            cuda_stream_handle_t stream, vram_arguments_t const& arguments,
                            cuda_rows_map_t const& rows_map = {}) noexcept {
        return cuda_launch(
            function, blocks, block_threads_k, shared_bytes, stream,
            cuda_overlaps_launches(capability_) ? cuda_overlap_t::overlapping_k : cuda_overlap_t::serialized_k,
            cuda_arguments_t {arguments, rows_map});
    }

    /** The vectors matrix as the tensor map the walks gather rows through, spanning the ceiling the
     *  first reservation froze, so the storage grows under it: encoded by the first walk launch and
     *  kept, while launches racing it use their own. Zeroed where the schedule gathers no rows or
     *  the driver cannot encode the map. Defined in @c vram_nvidia_index.cuh. */
    cuda_rows_map_t rows_map_() noexcept;

    /** Launches the rows of @p build in waves, each queued behind the last and as wide as the IDs
     *  below it, never narrower than @c vram_admission_ramp_floor_k, so every ID a wave admits
     *  walks rows already published, and each followed by the back-link launch that links its
     *  newcomers from the nodes they picked. A wave failing after an earlier one launched keeps
     *  the failure, since those rows can no longer be un-claimed. */
    status_t launch_insert_(vram_build_t const& build, u32_t candidate_slots, traversal_context_t& context) noexcept {
        vram_arguments_t arguments {};
        cuda_stream_handle_t const stream = context.stream();
        cuda_function_handle_t const insert = kernel_function_(vram_kernel_t::insert_k, candidate_slots);
        cuda_function_handle_t const back_link = kernel_function_(vram_kernel_t::back_link_k, 0);
        // Both launches carve their warps' slices from the one arena, sized for the insert's grid,
        // which the back-link's resident count matches.
        static_assert(resident_blocks_(vram_kernel_t::back_link_k, 0) == resident_blocks_(vram_kernel_t::insert_k, 0));
        u32_t const blocks = device_.multiprocessor_count * resident_blocks_(vram_kernel_t::insert_k, candidate_slots);
        u32_t const back_link_rows = reselection_rows_();
        u32_t const back_link_scratch = build.arena_shape.scratch_bytes ? 0u
                                                                        : build_scratch_bytes_per_warp(back_link_rows);
        cuda_rows_map_t const rows_map = rows_map_();
        auto zeroed = [&](u32_t* counters, u32_t count) noexcept {
            return cuda_fill_async(counters, 0, count * sizeof(u32_t), stream) == cuda_success_k;
        };
        // Overlapping launches claim from three counters a wave, the insert's, the back-link's and
        // the touched count, which one fill zeroes ahead of the waves they serve, so no fill sits
        // between two launches and keeps the later one from overlapping the earlier one's drain.
        constexpr bool chained_k = cuda_overlaps_launches(capability_);
        constexpr u32_t chained_waves_k = cuda_claim_counters_k / 3;
        for (u32_t wave_first = build.first_id, wave = 0; wave_first != build.end_id; ++wave) {
            u32_t const wave_end = static_cast<u32_t>(
                vram_admission_wave_end(wave_first, build.end_id, vram_dense_admission_share_log2_k));
            u32_t const offset = wave_first - build.first_id;
            u32_t const slot = wave % chained_waves_k;
            u32_t* const counters = build.assigned_rows + (chained_k ? 3 * slot : 0);
            arguments.build = build;
            arguments.build.staging = build.staging + std::size_t {offset} * row_bytes();
            arguments.build.first_id = wave_first;
            arguments.build.end_id = wave_end;
            arguments.build.epoch_base = build.epoch_base + offset;
            arguments.build.assigned_rows = counters;
            if constexpr (chained_k) arguments.build.touched_count = counters + 2;
            bool const counters_zeroed = chained_k ? slot != 0 || zeroed(build.assigned_rows, cuda_claim_counters_k)
                                                   : zeroed(build.assigned_rows, 1) && zeroed(build.touched_count, 1);
            status_t launched = counters_zeroed ? status_t::success_k : status_t::device_error_k;
            if (succeeded(launched))
                launched = launch_(insert, blocks, vram_warps_per_block_k * build.scratch_bytes_per_warp, stream,
                                   arguments, rows_map);
            if (failed(launched)) return offset ? fail_(launched) : launched;
            // The wave's rows are queued now, so a failure past here can no longer be un-claimed.
            arguments.build.scratch_bytes_per_warp = back_link_scratch;
            arguments.build.selection_rows = back_link_rows;
            if constexpr (chained_k) arguments.build.assigned_rows = counters + 1;
            else if (!zeroed(build.assigned_rows, 1)) return fail_(status_t::device_error_k);
            if (status_t const linked = launch_(back_link, blocks, vram_warps_per_block_k * back_link_scratch, stream,
                                                arguments);
                failed(linked))
                return fail_(linked);
            wave_first = wave_end;
        }
        return status_t::success_k;
    }

    /** Walks the rows of @p launch, landing their re-selected neighbors in its shadow matrix. */
    status_t launch_refine_(refine_launch_t const& launch, traversal_context_t& context) noexcept {
        vram_arguments_t arguments {};
        u32_t const selection_rows = walk_selection_rows_(launch.expansion, launch.candidate_slots);
        arguments.refine = {storage(), //
                            launch.shadow_neighbors,
                            launch.nodes.bound,
                            launch.arena_shape.scratch_bytes ? 0u : build_scratch_bytes_per_warp(selection_rows),
                            selection_rows,
                            launch.nodes.first,
                            launch.arena,
                            launch.arena_shape,
                            launch.expansion,
                            launch.epoch_base,
                            context.telemetry_.get(),
                            launch.claims,
                            launch.node_ids,
                            launch.nodes_count,
                            launch.part_bases,
                            launch.parts,
                            launch.round};
        return launch_(kernel_function_(vram_kernel_t::refine_k, launch.candidate_slots),
                       device_.multiprocessor_count * resident_blocks_(vram_kernel_t::refine_k, launch.candidate_slots),
                       vram_warps_per_block_k * arguments.refine.scratch_bytes_per_warp, context.stream(), arguments,
                       rows_map_());
    }

    /** Scatters the walked rows' shadow neighbors into the reverse buckets, then merges every
     *  touched row's forward and reverse candidates into its live neighbors. */
    status_t launch_refine_back_links_(refine_launch_t const& launch, traversal_context_t& context) noexcept {
        cuda_stream_handle_t const stream = context.stream();
        vram_arguments_t arguments {};
        vram_graph_t& graph = arguments.graph;
        graph.neighbors = launch.shadow_neighbors;
        graph.rows_count = launch.nodes_count;
        graph.connectivity = static_cast<u32_t>(connectivity());
        graph.first_id = launch.nodes.first;
        graph.walk_count = static_cast<u32_t>(launch.nodes.size());
        graph.node_ids = launch.node_ids;
        graph.bucket_width = launch.bucket_width;
        graph.bucket_ids = launch.buckets;
        graph.bucket_counts = launch.bucket_counts;
        graph.touched = launch.touched;
        if (status_t const launched = launch_(
                kernel_function_(vram_kernel_t::scatter_back_links_k, 0),
                device_.multiprocessor_count * resident_blocks_(vram_kernel_t::scatter_back_links_k, 0), 0, stream,
                arguments);
            failed(launched))
            return launched;
        // The merge sweeps the whole ID space and skips the rows the scatter left untouched,
        // carving its slices from the arena sized for the refine's grid, so it launches that grid.
        static_assert(resident_blocks_(vram_kernel_t::merge_back_links_k, 0) ==
                      resident_blocks_(vram_kernel_t::refine_k, 0));
        arguments.refine = {storage(),
                            launch.shadow_neighbors,
                            launch.nodes_count,
                            launch.arena_shape.scratch_bytes ? 0u : build_scratch_bytes_per_warp(reselection_rows_()),
                            reselection_rows_(),
                            0,
                            launch.arena,
                            launch.arena_shape,
                            launch.expansion,
                            launch.epoch_base,
                            context.telemetry_.get(),
                            launch.claims + 1,
                            nullptr,
                            launch.nodes_count,
                            0};
        return launch_(kernel_function_(vram_kernel_t::merge_back_links_k, 0),
                       device_.multiprocessor_count * resident_blocks_(vram_kernel_t::refine_k, launch.candidate_slots),
                       vram_warps_per_block_k * arguments.refine.scratch_bytes_per_warp, stream, arguments);
    }

    /** The search view every walking verb launches over. */
    vram_search_t search_view_(walk_launch_t const& walk, traversal_context_t& context) const noexcept {
        return {storage(),
                reinterpret_cast<byte_t const*>(walk.queries),
                walk.queries_count,
                walk.visible,
                walk.scratch_bytes_per_warp,
                walk.expansion,
                walk.arena,
                walk.arena_shape,
                walk.epoch_base,
                context.telemetry_.get(),
                context.assigned_rows_.get(),
                context.assigned_rows_.get() + 1,
                walk.seats,
                walk.filling_rows};
    }

    /** The rung a walk runs on: its register ladder's width, or zero for the arena frontier. */
    static u32_t walk_slots_(walk_launch_t const& walk) noexcept {
        return walk.frontier == frontier_kind_t::wide_arena_k ? 0 : walk.candidate_slots;
    }

    /** Launches one walk per query of @p walk through @p function, the @p kernel on the walk's
     *  rung, over @p arguments, and reports the launch's own verdict. */
    status_t launch_walk_(vram_kernel_t kernel, cuda_function_handle_t function, walk_launch_t const& walk,
                          vram_arguments_t const& arguments, traversal_context_t& context) noexcept {
        return launch_(function, walk_blocks_(walk.queries_count, resident_blocks_(kernel, walk_slots_(walk))),
                       vram_warps_per_block_k * walk.scratch_bytes_per_warp, context.stream(), arguments, rows_map_());
    }

    /** Walks every query of @p walk, landing its nearest @p count IDs and, unless null, their
     *  distances; staged indexer weights pick the indexer-rescored walk. */
    status_t launch_search_(walk_launch_t const& walk, node_id_t* ids, distance_t* distances, u32_t count,
                            traversal_context_t& context) noexcept {
        vram_arguments_t arguments {};
        arguments.search = search_view_(walk, context);
        arguments.search_sink = {ids, distances, count, kernel_};
        vram_kernel_t kernel = vram_kernel_t::search_k;
        // Only the floating dtypes take indexer heads, so only they rescore.
        if constexpr (floating_scalar<scalar_t>)
            if (walk.weights) {
                arguments.rescoring = {reinterpret_cast<byte_t const*>(walk.heads), walk.weights, walk.heads_count,
                                       row_fragments()};
                kernel = vram_kernel_t::search_indexer_k;
            }
        return launch_walk_(kernel, kernel_function_(kernel, walk_slots_(walk)), walk, arguments, context);
    }

    /** Scans every query of @p walk against every visible row on a persistent grid of every block
     *  the device seats, claiming the work items @ref cuda_scan_ranges splits it into and keeping
     *  their matches in @p scratch, and lands each query's nearest @p count IDs and, unless null,
     *  their distances. */
    status_t launch_scan_(walk_launch_t const& walk, byte_t* scratch, node_id_t* ids, distance_t* distances,
                          u32_t count, traversal_context_t& context) noexcept {
        u32_t const blocks = divide_round_up<u32_t>(walk.queries_count, cuda_scan_tile_k);
        // The finished counts start at zero, so the last range of each block to finish merges it.
        if (cuda_fill_async(scratch, 0, round_up_to_multiple<std::size_t>(blocks, 4) * sizeof(u32_t),
                            context.stream()) != cuda_success_k)
            return status_t::device_error_k;
        vram_arguments_t arguments {};
        arguments.search = search_view_(walk, context);
        arguments.search.arena = scratch;
        arguments.search_sink = {ids, distances, count, kernel_};
        return launch_(kernel_function_(vram_kernel_t::scan_k, 0),
                       device_.multiprocessor_count * resident_blocks_(vram_kernel_t::scan_k, 0),
                       cuda_scan_staged_bytes_k, context.stream(), arguments);
    }

    /** Walks every row of @p walk, landing up to @p wanted kept pairs per row and the pairs each
     *  scored. */
    status_t launch_multiply_(walk_launch_t const& walk, node_id_t* ids, dot_t* products, u32_t* counts, u32_t wanted,
                              traversal_context_t& context) noexcept {
        vram_arguments_t arguments {};
        arguments.search = search_view_(walk, context);
        arguments.multiply_sink = {ids, reinterpret_cast<byte_t*>(products), counts, wanted};
        return launch_walk_(vram_kernel_t::multiply_k, kernel_function_(vram_kernel_t::multiply_k, walk_slots_(walk)),
                            walk, arguments, context);
    }

    /** Walks every token of @p walk under its staged heads, folding the value rows @p value_lookup
     *  resolves into the normalized @p outputs and their @p states; the lookup travels as bytes to
     *  the attend kernel its @c device_lookup_k names, found in the fatbin by name. */
    template <typename value_lookup_type_>
    status_t launch_attend_(walk_launch_t const& walk, value_lookup_type_ const& value_lookup, f32_t* outputs,
                            softmax_state_t* states, f32_t logit_scale, traversal_context_t& context) noexcept {
        static_assert(std::is_trivially_copyable_v<value_lookup_type_> &&
                          sizeof(value_lookup_type_) <= vram_value_lookup_capacity_k &&
                          alignof(value_lookup_type_) <= 16,
                      "A device value lookup travels to the kernels as its own bytes");
        vram_arguments_t arguments {};
        arguments.search = search_view_(walk, context);
        arguments.rescoring = {reinterpret_cast<byte_t const*>(walk.heads), nullptr, walk.heads_count, row_fragments()};
        vram_attend_t& attend = arguments.attend;
        attend.outputs = outputs;
        attend.states = states;
        attend.missed = context.missed_.get();
        attend.logit_scale = logit_scale;
        attend.heads_count = walk.heads_count;
        attend.columns = static_cast<u32_t>(dimensions());
        std::memcpy(attend.value_lookup, &value_lookup, sizeof(value_lookup));
        vram_kernel_symbol_t symbol;
        vram_kernel_symbol(symbol, "dense", "cuda_kernel", "cuda_arguments_t", vram_kernel_t::attend_k, capability_,
                           vector_dtype_, walk_slots_(walk), u32_t {value_lookup_type_::device_lookup_k});
        return launch_walk_(vram_kernel_t::attend_k, module_.function(symbol.text), walk, arguments, context);
    }

    /** Rebuilds the norms of rows @b [first_row,first_row+rows) from their vectors. */
    status_t launch_derive_norms_(std::size_t first_row, std::size_t rows, cuda_stream_handle_t stream) noexcept {
        vram_arguments_t arguments {};
        arguments.graph.storage = storage();
        arguments.graph.first_id = static_cast<u32_t>(first_row);
        arguments.graph.rows_count = static_cast<u32_t>(rows);
        return launch_(kernel_function_(vram_kernel_t::derive_norms_k, 0),
                       device_.multiprocessor_count * resident_blocks_(vram_kernel_t::derive_norms_k, 0), 0, stream,
                       arguments);
    }

    /** Rebases rows @p rows of the neighbors matrix from @p source_range onto the IDs from @p shift
     *  and compacts them, as a load through the bounce does on the host. */
    status_t launch_rebase_links_(iota_view<node_id_t> rows, iota_view<node_id_t> source_range, node_id_t shift,
                                  cuda_stream_handle_t stream) noexcept {
        vram_arguments_t arguments {};
        vram_graph_t& graph = arguments.graph;
        graph.neighbors = neighbor_rows_();
        graph.connectivity = static_cast<u32_t>(connectivity());
        graph.first_id = rows.first;
        graph.rows_count = static_cast<u32_t>(rows.size());
        graph.source_first = source_range.first;
        graph.source_bound = source_range.bound;
        graph.shift = shift;
        return launch_(kernel_function_(vram_kernel_t::rebase_links_k, 0),
                       device_.multiprocessor_count * resident_blocks_(vram_kernel_t::rebase_links_k, 0), 0, stream,
                       arguments);
    }

    /** Marks the audit's stride-spread landmarks over @p rows rows reached and in @p frontier. */
    status_t launch_audit_landmarks_(vram_bitset_t reached, vram_bitset_t frontier, std::size_t rows,
                                     std::size_t seed_count, cuda_stream_handle_t stream) noexcept {
        vram_arguments_t arguments {};
        arguments.graph.reached = reached;
        arguments.graph.frontier = frontier;
        arguments.graph.rows_count = static_cast<u32_t>(rows);
        arguments.graph.seed_count = static_cast<u32_t>(seed_count);
        return launch_(kernel_function_(vram_kernel_t::mark_landmarks_k, 0),
                       divide_round_up<u32_t>(static_cast<u32_t>(seed_count), block_threads_k), 0, stream, arguments);
    }

    /** Expands one level of reachability from @p frontier into @p next_frontier, counting the rows
     *  it expanded in @p expanded and the level in @p levels; a level whose @p expanded_before
     *  reads zero leaves at entry. */
    status_t launch_audit_sweep_(vram_bitset_t reached, vram_bitset_t frontier, vram_bitset_t next_frontier,
                                 u32_t* expanded, u32_t const* expanded_before, u32_t* levels, std::size_t rows,
                                 cuda_stream_handle_t stream) noexcept {
        vram_arguments_t arguments {};
        vram_graph_t& graph = arguments.graph;
        graph.neighbors = neighbor_rows_();
        graph.rows_count = static_cast<u32_t>(rows);
        graph.connectivity = static_cast<u32_t>(connectivity());
        graph.reached = reached;
        graph.frontier = frontier;
        graph.next_frontier = next_frontier;
        graph.counter = expanded;
        graph.previous_counter = expanded_before;
        graph.levels = levels;
        return launch_(kernel_function_(vram_kernel_t::reachability_sweep_k, 0),
                       device_.multiprocessor_count * resident_blocks_(vram_kernel_t::reachability_sweep_k, 0), 0,
                       stream, arguments);
    }

    /** Counts the rows @p reached lacks in @p islands_cursor, collecting them into @p island_ids
     *  when it is given. */
    status_t launch_audit_islands_(vram_bitset_t reached, std::size_t rows, u32_t* island_ids, u32_t* islands_cursor,
                                   cuda_stream_handle_t stream) noexcept {
        vram_arguments_t arguments {};
        arguments.graph.reached = reached;
        arguments.graph.rows_count = static_cast<u32_t>(rows);
        arguments.graph.island_ids = island_ids;
        arguments.graph.counter = islands_cursor;
        return launch_(kernel_function_(vram_kernel_t::reachability_islands_k, 0),
                       divide_round_up<u32_t>(static_cast<u32_t>(rows), block_threads_k), 0, stream, arguments);
    }

    /** Merges the attended rows and carries of @p shards walks into the first's on @p stream - the
     *  sharded attend's gather. */
    status_t launch_merge_attention_(byte_t* blocks, u32_t rows, u32_t columns, u32_t shards,
                                     cuda_stream_handle_t stream) noexcept {
        vram_arguments_t arguments {};
        arguments.attend.outputs = reinterpret_cast<f32_t*>(blocks);
        arguments.attend.rows = rows;
        arguments.attend.columns = columns;
        arguments.attend.shards = shards;
        return launch_(kernel_function_(vram_kernel_t::merge_attention_k, 0),
                       divide_round_up<u32_t>(rows, vram_warps_per_block_k), 0, stream, arguments);
    }

    /** Resolves the @p walked matches per query of @p queries_count at @p ids and @p distances
     *  through the device @p keys column into @p ranking on @p stream - a sharded search's own
     *  ranking, gathered or landed as is. */
    status_t launch_resolve_matches_(node_id_t const* ids, distance_t const* distances, u32_t walked,
                                     byte_t const* keys, vram_ranking_t const& ranking, u32_t queries_count,
                                     cuda_stream_handle_t stream) const noexcept {
        vram_arguments_t arguments {};
        arguments.matches.ids = ids;
        arguments.matches.distances = distances;
        arguments.matches.keys = keys;
        arguments.matches.ranking = ranking;
        arguments.matches.walked = walked;
        arguments.matches.queries_count = queries_count;
        return launch_(kernel_function_(vram_kernel_t::resolve_matches_k, 0),
                       divide_round_up<u32_t>(queries_count, vram_warps_per_block_k), 0, stream, arguments);
    }

    /** Merges @p shards rankings of @p queries_count queries, gathered @p block_bytes apart from
     *  @p gathered, into @p ranking on @p stream - the sharded search's device gather. */
    status_t launch_merge_matches_(vram_ranking_t const& gathered, std::size_t block_bytes, u32_t shards,
                                   vram_ranking_t const& ranking, u32_t queries_count,
                                   cuda_stream_handle_t stream) const noexcept {
        vram_arguments_t arguments {};
        arguments.matches.gathered = gathered;
        arguments.matches.ranking = ranking;
        arguments.matches.block_bytes = block_bytes;
        arguments.matches.queries_count = queries_count;
        arguments.matches.shards = shards;
        return launch_(kernel_function_(vram_kernel_t::merge_matches_k, 0),
                       divide_round_up<u32_t>(queries_count, vram_warps_per_block_k), 0, stream, arguments);
    }

    /** Casts @p count values at @p from in @p from_dtype into @p to in @p to_dtype on @p stream,
     *  through NumKong's plan and its per-unit device conversions. */
    status_t launch_cast_(void const* from, nk_dtype_t from_dtype, void* to, nk_dtype_t to_dtype, std::size_t count,
                          cuda_stream_handle_t stream) const noexcept {
        vram_arguments_t arguments {};
        arguments.cast.from = static_cast<unsigned char const*>(from);
        arguments.cast.to = static_cast<unsigned char*>(to);
        if (!nk_cast_plan_serial_(from_dtype, to_dtype, count, cuda_address(from) | cuda_address(to),
                                  &arguments.cast.plan))
            return status_t::unsupported_k;
        u32_t const blocks = static_cast<u32_t>(
            min_of<std::size_t>(divide_round_up<std::size_t>(arguments.cast.plan.units, block_threads_k),
                                device_.multiprocessor_count * resident_blocks_(vram_kernel_t::cast_k, 0)));
        return launch_(kernel_function_(vram_kernel_t::cast_k, 0), blocks, 0, stream, arguments);
    }

    /** Narrows @p rows rows of the index's columns at @p from, @p from_stride bytes apart, into
     *  @p to at the vectors matrix pitch at unit RMS on @p stream, one block per row. */
    status_t launch_rmscast_(f32_t const* from, std::size_t from_stride, byte_t* to, std::size_t rows,
                             cuda_stream_handle_t stream) const noexcept {
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
        return launch_(kernel_function_(vram_kernel_t::rmscast_k, 0), blocks, 0, stream, arguments);
    }

#pragma endregion Launches

    /** This ceiling's fatbin on this device, and its kernels in @ref vram_kernel_t order, the
     *  attends left to their launches. */
    cuda_module_t module_;
    cuda_function_handle_t kernels_[static_cast<u32_t>(vram_kernel_t::count_k)][gpu_rungs_per_kernel_k] = {};

    /** The map @ref rows_map_ keeps once published. */
    cuda_rows_map_t rows_map_kept_ {};

    /** Zero until a launch claims the map's encoding, one while it writes @ref rows_map_kept_, and
     *  two once published, read and written through atomic references. */
    u32_t rows_map_state_ = 0;
};

} // namespace unum::usearch::dense
