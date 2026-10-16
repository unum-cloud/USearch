/**
 *  @file include/usearch/basic/vram_nvidia_index.cuh
 *  @author Ash Vardanian
 *  @date October 15, 2026
 *  @brief Similarity Search over variable-length values in a single uniform latency memory space on
 *      NVIDIA GPUs.
 *
 *  The kernel side of the basic index: the one kernel template over three integers, the kernel, the
 *  kit and the register rung, whose instantiations a fatbin carries. Only NVCC's device-only passes
 *  include it, once per modality unit over every kit it ships; each @c cuda_kernel turns the plain
 *  records of @c vram_nvidia_index.hpp into the shared views of @c vram_index.cuh, and the host
 *  finds it by the mangled name @c cuda_kernel_name spells.
 *
 *  A kit whose lane scores a row alone walks a query or a node per warp. A kit splitting a row over
 *  several lanes walks one per block: warp zero owns the frontier and the visited set and posts
 *  each hop's rows, and the block's four warps score them together, so long rows cost a hop a
 *  quarter of a warp's latency.
 */
#pragma once

#include <limits>      // `std::numeric_limits` - an absent row's distance
#include <type_traits> // `std::conditional_t` - a walk's frontier and schedule
#include <utility>     // `std::index_sequence` - every rung of one kit

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/vram_nvidia_device.cuh" // `cuda_warp_t`
#include "usearch/basic/vram_index.cuh"
#include "usearch/basic/vram_nvidia_index.hpp"

namespace unum::usearch::basic {

/**
 *  @brief Scores each round of a walk's rows with the whole block: warp zero posts the query and a
 *      row per lane, and every warp scores its share, @c kit_::lanes_per_row_k lanes to a row, in
 *      turns when a row takes more lanes than the block's warps split evenly.
 *
 *  Only warp zero walks, so the frontier, the visited set and the query's span stay its own; the
 *  other warps @ref serve rounds until it @ref finish es. Two named barriers bracket every round,
 *  so the posted rows and the span are read between them alone and rewritten outside them.
 */
template <vram_basic_kit kit_>
struct cuda_block_schedule {
    using query_t = typename kit_::query_t;

    /** Rows one warp scores per round. */
    inline static constexpr u32_t rows_per_warp_k = cuda_warp_t::width_k / kit_::lanes_per_row_k;
    inline static constexpr u32_t rows_per_block_k = rows_per_warp_k * vram_warps_per_block_k;
    static_assert(cuda_warp_t::width_k % rows_per_block_k == 0, "A block's warps split one warp's rows evenly");

    /** The round warp zero posts and the distances the block lands for it. */
    struct shared_t {
        query_t query;
        node_id_t ids[cuda_warp_t::width_k];
        distance_t distances[cuda_warp_t::width_k];
        distance_t radius;
        u32_t finished;
    };

    shared_t& round;

    __device__ __forceinline__ explicit cuda_block_schedule(shared_t& shared) noexcept : round(shared) {
        if (threadIdx.x == 0) round.finished = 0;
    }

    /** The distance from @p query to the row this lane of warp zero names, exact within @p radius;
     *  infinite for the sentinel. Every lane of warp zero calls it. */
    __device__ __forceinline__ distance_t score(vram_ragged_view const& storage, query_t const& query,
                                                node_id_t node_id, distance_t radius, u32_t lane) noexcept {
        round.ids[lane] = node_id;
        if (lane == 0) round.query = query, round.radius = radius;
        barrier_();
        score_share_(storage, 0, lane);
        barrier_();
        return round.distances[lane];
    }

    /** Scores the rounds warp zero posts until it finishes, from warp @p warp. */
    __device__ __forceinline__ void serve(vram_ragged_view const& storage, u32_t warp, u32_t lane) noexcept {
        while (true) {
            barrier_();
            if (round.finished) return;
            score_share_(storage, warp, lane);
            barrier_();
        }
    }

    /** Releases the serving warps once warp zero has walked its last row. */
    __device__ __forceinline__ void finish(u32_t lane) noexcept {
        if (lane == 0) round.finished = 1;
        barrier_();
    }

  private:
    __device__ __forceinline__ void score_share_(vram_ragged_view const& storage, u32_t warp, u32_t lane) noexcept {
        for (u32_t first = warp * rows_per_warp_k; first < cuda_warp_t::width_k; first += rows_per_block_k) {
            u32_t const position = first + lane / kit_::lanes_per_row_k;
            node_id_t const node_id = round.ids[position];
            distance_t distance = std::numeric_limits<distance_t>::infinity();
            // A row's lanes share its ID, so every group takes one side of each branch whole.
            if (node_id != node_id_unknown_k) {
                vram_row_t const row = storage.row(node_id);
                distance = kit_::bound(round.query, row);
                if (!(distance > round.radius)) distance = kit_::score(round.query, row, round.radius, lane);
            }
            if (lane % kit_::lanes_per_row_k == 0) round.distances[position] = distance;
        }
    }

    /** Barrier 1 counts every thread of the block, wherever each one reaches it from. */
    static __device__ __forceinline__ void barrier_() noexcept {
        asm volatile("bar.sync 1, %0;" ::"n"(vram_warps_per_block_k * cuda_warp_t::width_k) : "memory");
    }
};

/** The neighbor selection the kernels of the kit named by @p kit_ keep rows with: the diversity
 *  rule, unless the unit compiling a modality specializes it with @c type for its kits. */
template <u32_t kit_>
struct cuda_selector_of {
    using type = vram_keep_diverse_t;
};

/** The schedule a kit scores through: in place when a lane scores a row alone, with the whole block
 *  when a row takes several lanes. */
template <vram_basic_kit kit_>
using cuda_schedule =
    std::conditional_t<kit_::lanes_per_row_k == 1, vram_lane_schedule<kit_>, cuda_block_schedule<kit_>>;

/** Hands @p walk every ID in @b [first,end) the calling warp assigns itself from @p assigned: every
 *  warp walks under a lane schedule, warp zero alone under a block one, the other warps serving its
 *  rounds until it runs out. */
template <vram_basic_kit kit_, typename walk_type_>
__device__ __forceinline__ void cuda_for_each_assigned(cuda_schedule<kit_>& schedule, vram_ragged_view const& storage,
                                                       u32_t* assigned, u32_t first, u32_t end,
                                                       vram_warp_slice_t const& slice, walk_type_&& walk) noexcept {
    if constexpr (kit_::lanes_per_row_k != 1)
        if (slice.warp != 0) return schedule.serve(storage, slice.warp, slice.lane);
    while (true) {
        u32_t assigned_id = slice.lane == 0 ? cuda_warp_t::fetch_add_relaxed(*assigned, 1u) : 0;
        assigned_id = first + cuda_warp_t::shuffle(assigned_id, 0);
        if (assigned_id >= end) break;
        walk(assigned_id);
    }
    if constexpr (kit_::lanes_per_row_k != 1) schedule.finish(slice.lane);
}

/**
 *  @brief One walker inserts or re-selects one node at a time, and walkers never synchronize beyond
 *      reading slot-0 words: no device watermark, no publication order, no lock across a distance.
 *
 *  Within a wave, IDs are batch-local: each walker assigns itself the next offset of the wave,
 *  walks the published rows, and publishes its forward row with one release. The host launches each
 *  wave behind the last one's back-link passes, so a wave walks a graph whose reverse links are
 *  already merged, and re-selects behind its own, so its rows reach one another.
 */
template <vram_basic_kit kit_, typename selector_, u32_t pool_slots_, cuda_kernel_t kernel_>
__device__ __forceinline__ void cuda_insert(vram_build_view const& view) {
    // The scratch slice hosts the query span, then the pivot span; the visited set is in the arena.
    extern __shared__ byte_t dynamic_scratch[];
    __shared__ u32_t gathered_ids[vram_warps_per_block_k][vram_cache_rows_k];
    __shared__ u32_t selected_ids[vram_warps_per_block_k][vram_cache_rows_k];
    __shared__ u32_t kept_positions[vram_warps_per_block_k][vram_cache_rows_k];
    __shared__ f32_t candidate_distances[vram_warps_per_block_k][vram_cache_rows_k];
    __shared__ typename cuda_schedule<kit_>::shared_t schedule_state;

    vram_warp_slice_t const slice = vram_carve_slice<cuda_warp_t>(view, dynamic_scratch);
    u32_t const warp = slice.warp, lane = slice.lane;
    vram_selection_scratch_t const selection {selected_ids[warp], nullptr, kept_positions[warp],
                                              candidate_distances[warp]};
    using frontier_t = std::conditional_t<pool_slots_ == 0, vram_arena_frontier<cuda_warp_t>,
                                          vram_register_frontier<pool_slots_, cuda_warp_t>>;
    cuda_schedule<kit_> schedule {schedule_state};
    cuda_for_each_assigned<kit_>( //
        schedule, view.storage, view.assigned_rows, view.first_id, view.end_id, slice, [&](node_id_t node_id) {
            vram_warp_context<frontier_t> context;
            if constexpr (pool_slots_ == 0) {
                context.frontier.tail_.entries_ = view.arena_shape.tail_span(view.arena, slice.warp_index);
                context.frontier.tail_.capacity_ = view.arena_shape.tail_entries;
                context.frontier.expansion_target_ = view.expansion;
            }
            context.random_generator = vram_lane_generator<cuda_warp_t>(node_id, lane);
            vram_visited_set<cuda_warp_t> visited {slice.visited_slots, view.arena_shape.visited_hash_shift,
                                                   view.epoch_base + (node_id - view.first_id) + 1};
            if constexpr (kernel_ == cuda_kernel_t::reselect_k)
                vram_reselect_one<kit_, selector_>( //
                    view, schedule, node_id, visited, slice.scratch, slice.scratch + view.span_bytes,
                    gathered_ids[warp], selection, context, lane);
            else
                vram_insert_one<kit_, selector_>( //
                    view, schedule, node_id, visited, slice.scratch, slice.scratch + view.span_bytes,
                    gathered_ids[warp], selection, context, lane);
            context.telemetry.tail_drops = context.frontier.dropped();
            context.telemetry.visited_saturations = visited.saturations_;
            context.telemetry.template flush<cuda_warp_t, vram_traversal_t::inserting_k>(view.telemetry, lane);
        });
}

/** One walker back-links one node the walk launch before it touched, assigning itself the touched
 *  list's offsets from the launch counter, and resets the node's chain for the next walk launch. */
template <vram_basic_kit kit_, typename selector_>
__device__ __forceinline__ void cuda_back_link(vram_build_view const& view) {
    extern __shared__ byte_t dynamic_scratch[];
    __shared__ u32_t candidate_ids[vram_warps_per_block_k][vram_cache_rows_k];
    __shared__ vram_scored_node_t scored_entries[vram_warps_per_block_k][vram_cache_rows_k];
    __shared__ u32_t kept_positions[vram_warps_per_block_k][vram_cache_rows_k];
    __shared__ f32_t candidate_distances[vram_warps_per_block_k][vram_cache_rows_k];
    __shared__ typename cuda_schedule<kit_>::shared_t schedule_state;

    vram_warp_slice_t const slice = vram_carve_slice<cuda_warp_t>(view, dynamic_scratch);
    u32_t const warp = slice.warp, lane = slice.lane;
    vram_selection_scratch_t const selection {nullptr, scored_entries[warp], kept_positions[warp],
                                              candidate_distances[warp]};
    cuda_schedule<kit_> schedule {schedule_state};
    cuda_for_each_assigned<kit_>( //
        schedule, view.storage, view.assigned_rows, 0, *view.touched_count, slice, [&](u32_t offset) {
            node_id_t const node_id = view.touched_ids[offset];
            vram_telemetry_t telemetry;
            vram_back_link_one<kit_, selector_>( //
                view, schedule, node_id, slice.scratch, slice.scratch + view.span_bytes, candidate_ids[warp], selection,
                telemetry, lane);
            if (lane == 0) view.link_heads[node_id] = node_id_unknown_k;
            telemetry.template flush<cuda_warp_t, vram_traversal_t::inserting_k>(view.telemetry, lane);
        });
}

/** One walker per query on the rung of @p slots_, the arena frontier at zero; walkers take queries
 *  from the batch counter, as builders take rows. */
template <vram_basic_kit kit_, u32_t slots_>
__device__ __forceinline__ void cuda_walk(vram_search_view const& view) {
    extern __shared__ byte_t dynamic_scratch[];
    __shared__ u32_t gathered_ids[vram_warps_per_block_k][vram_cache_rows_k];
    __shared__ typename cuda_schedule<kit_>::shared_t schedule_state;

    vram_warp_slice_t const slice = vram_carve_slice<cuda_warp_t>(view, dynamic_scratch);
    std::conditional_t<slots_ == 0, vram_arena_frontier<cuda_warp_t>, vram_register_frontier<slots_, cuda_warp_t>>
        frontier;
    if constexpr (slots_ == 0) {
        frontier.tail_.entries_ = view.arena_shape.tail_span(view.arena, slice.warp_index);
        frontier.tail_.capacity_ = view.arena_shape.tail_entries;
        frontier.expansion_target_ = view.expansion;
    }
    cuda_schedule<kit_> schedule {schedule_state};
    cuda_for_each_assigned<kit_>( //
        schedule, view.storage, view.assigned_queries, 0, view.queries_count, slice, [&](u32_t query_index) {
            vram_visited_set<cuda_warp_t> visited {slice.visited_slots, view.arena_shape.visited_hash_shift,
                                                   view.epoch_base + query_index + 1};
            vram_walk_one<kit_>(view, schedule, query_index, slice.scratch, visited, frontier, gathered_ids[slice.warp],
                                slice.lane);
        });
}

#pragma region Views

/** The shared views, field by field from the host's records. */
__device__ __forceinline__ vram_ragged_view cuda_view(cuda_ragged_storage_t const& storage) noexcept {
    return {storage.values, storage.offsets, storage.neighbors, storage.connectivity};
}
__device__ __forceinline__ vram_build_view cuda_view(cuda_build_t const& build) noexcept {
    return {cuda_view(build.storage),     build.staged_offsets, build.tape_first, build.end_id,
            build.scratch_bytes_per_warp, build.span_bytes,     build.first_id,   build.arena,
            vram_view(build.arena_shape), build.expansion,      build.epoch_base, build.telemetry,
            build.assigned_rows,          build.link_heads,     build.link_next,  build.touched_ids,
            build.touched_count};
}
__device__ __forceinline__ vram_search_view cuda_view(cuda_search_t const& search) noexcept {
    return {cuda_view(search.storage),
            search.queries,
            search.query_offsets,
            search.queries_count,
            search.sampling_bound,
            search.scratch_bytes_per_warp,
            search.expansion,
            search.arena,
            vram_view(search.arena_shape),
            search.epoch_base,
            search.telemetry,
            search.assigned_queries,
            search.result_ids,
            search.result_distances,
            search.count};
}

#pragma endregion Views

#pragma region Kernels

/** Blocks of each kernel resident per multiprocessor, the register budget its launch bounds set:
 *  the builders' and the back-link's saturating count, and a walk's by its rung. */
constexpr u32_t cuda_resident_blocks(cuda_kernel_t kernel, u32_t slots) noexcept {
    if (kernel == cuda_kernel_t::search_k) return slots <= 8 ? 4 : 3;
    return cuda_insert_blocks_per_multiprocessor_k;
}

/** Every NVIDIA basic kernel: @p kernel_ picks the body, @p kit_ the kit through @ref cuda_kit_of,
 *  and @p slots_ the rung. */
template <u32_t kernel_, u32_t kit_, u32_t slots_>
__global__ void __launch_bounds__(vram_warps_per_block_k * 32,
                                  cuda_resident_blocks(static_cast<cuda_kernel_t>(kernel_), slots_))
    cuda_kernel(cuda_arguments_t arguments) {
    using kit_t = typename cuda_kit_of<kit_>::type;
    using selector_t = typename cuda_selector_of<kit_>::type;
    constexpr cuda_kernel_t kernel = static_cast<cuda_kernel_t>(kernel_);
    cuda_build_t const& build = arguments.build;
    if constexpr (kernel == cuda_kernel_t::insert_k || kernel == cuda_kernel_t::reselect_k)
        cuda_insert<kit_t, selector_t, slots_, kernel>(cuda_view(build));
    else if constexpr (kernel == cuda_kernel_t::back_link_k) cuda_back_link<kit_t, selector_t>(cuda_view(build));
    else cuda_walk<kit_t, slots_>(cuda_view(arguments.search));
}

template <u32_t kit_, std::size_t... rungs_>
__device__ void cuda_instantiate_rungs(std::index_sequence<rungs_...>) noexcept {
    (static_cast<void>(&cuda_kernel<static_cast<u32_t>(cuda_rungs[rungs_].kernel), kit_, cuda_rungs[rungs_].slots>),
     ...);
}

/** Instantiates every kernel one kit exports; a unit compiling a fatbin names it once per kit.
 *  Taking the kernels' addresses in device code makes NVCC emit them, while a host-only pass emits
 *  no launch stub. */
template <u32_t kit_>
__device__ void cuda_instantiate() noexcept {
    cuda_instantiate_rungs<kit_>(std::make_index_sequence<sizeof(cuda_rungs) / sizeof(cuda_rungs[0])> {});
}

#pragma endregion Kernels

} // namespace unum::usearch::basic
