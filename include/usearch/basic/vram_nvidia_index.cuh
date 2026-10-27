/**
 *  @file include/usearch/basic/vram_nvidia_index.cuh
 *  @author Ash Vardanian
 *  @date October 15, 2026
 *  @brief Similarity Search over variable-length values in a single uniform latency memory space on
 *      NVIDIA GPUs.
 *
 *  The kernel side of the basic index: the body every modality's kernel runs, over the kernel, the
 *  kit, its rung, the register rung and the node-ID type. Only the modalities' NVIDIA metrics
 *  include it, whose @c cuda_kernel templates take their ladders' inputs and hand the kit their
 *  ladder picks to @ref cuda_kernel_body, which turns the plain records of @c vram_nvidia_index.hpp
 *  into the shared views of @c vram_index.cuh; the host finds each kernel by the symbol
 *  @c vram_kernel_symbol spells.
 *
 *  A kit whose lane scores a row alone walks a query or a node per warp. A kit splitting a row over
 *  several lanes walks one per block: warp zero owns the frontier and the visited set and posts
 *  each hop's rows, and the block's four warps score them together, so long rows cost a hop a
 *  quarter of a warp's latency.
 */
#pragma once

#include <limits>      // `std::numeric_limits` - an absent row's distance
#include <type_traits> // `std::conditional_t` - a walk's frontier and schedule

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/vram_nvidia_device.cuh" // `cuda_warp_t`
#include "usearch/basic/vram_index.cuh"
#include "usearch/basic/vram_nvidia_index.hpp"

namespace unum::usearch::basic {

// Checked where only CUDA builds compile, since wasm32 hosts carry the header but never a device.
static_assert(sizeof(std::size_t) == sizeof(u64_t), "The caller's offsets cross as the device's words");

/**
 *  @brief Scores each round of a walk's rows with the whole block: warp zero posts the query and a
 *      row per lane, and every warp scores its share on @p rung_ of @p kit_, @c lanes_per_row_k
 *      lanes to a row, in turns when a row takes more lanes than the block's warps split evenly.
 *
 *  Only warp zero walks, so the frontier, the visited set and the query's span stay its own; the
 *  other warps @ref serve rounds until it @ref finish es. Two named barriers bracket every round,
 *  so the posted rows and the span are read between them alone and rewritten outside them.
 */
template <vram_basic_kit kit_, u32_t rung_, typename node_id_type_>
struct cuda_block_schedule {
    using query_t = typename kit_::query_t;
    inline static constexpr u32_t lanes_per_row_k = vram_lanes_per_row<kit_, rung_>();

    /** Rows one warp scores per round. */
    inline static constexpr u32_t rows_per_warp_k = cuda_warp_t::width_k / lanes_per_row_k;
    inline static constexpr u32_t rows_per_block_k = rows_per_warp_k * vram_warps_per_block_k;
    static_assert(cuda_warp_t::width_k % rows_per_block_k == 0, "A block's warps split one warp's rows evenly");

    /** The round warp zero posts, a warp's width of rows or two, and the distances the block lands
     *  for it. */
    struct shared_t {
        query_t query;
        node_id_type_ ids[2 * cuda_warp_t::width_k];
        distance_t distances[2 * cuda_warp_t::width_k];
        distance_t radius;
        u32_t rows;
        u32_t finished;
    };

    shared_t& round;

    __device__ __forceinline__ explicit cuda_block_schedule(shared_t& shared) noexcept : round(shared) {
        if (threadIdx.x == 0) round.finished = 0;
    }

    /** The distance from @p query to the row this lane of warp zero names, exact within @p radius;
     *  infinite for the sentinel. Every lane of warp zero calls it. */
    __device__ __forceinline__ distance_t score(vram_ragged_view<node_id_type_> const& storage, query_t const& query,
                                                node_id_type_ node_id, distance_t radius, u32_t lane) noexcept {
        round.ids[lane] = node_id;
        if (lane == 0) round.query = query, round.radius = radius, round.rows = cuda_warp_t::width_k;
        barrier_();
        score_share_(storage, 0, lane);
        barrier_();
        return round.distances[lane];
    }

    /** Scores a round of @p count rows of @p ids, past a warp's width and up to two, into
     *  @p scored with the whole block: the longest rows first, so every warp's turns pair one of
     *  the longest groups of rows with one of the shortest. */
    __device__ __forceinline__ void score_round(vram_ragged_view<node_id_type_> const& storage, query_t const& query,
                                                node_id_type_ const* ids, u32_t count, distance_t radius, u32_t lane,
                                                vram_scored_node<node_id_type_> (&scored)[2]) noexcept {
        vram_split_by_bytes<cuda_warp_t>(storage, ids, count, lane, scored);
        round.ids[lane] = scored[0].id(), round.ids[cuda_warp_t::width_k + lane] = scored[1].id();
        if (lane == 0) round.query = query, round.radius = radius, round.rows = 2 * cuda_warp_t::width_k;
        barrier_();
        score_share_(storage, 0, lane);
        barrier_();
        for (u32_t half = 0; half != 2; ++half)
            if (!(scored[half] == vram_scored_node<node_id_type_>::sentinel()))
                scored[half] = vram_scored_node<node_id_type_>::make(
                    round.distances[half * cuda_warp_t::width_k + lane], scored[half].id());
    }

    /** Scores the rounds warp zero posts until it finishes, from warp @p warp. */
    __device__ __forceinline__ void serve(vram_ragged_view<node_id_type_> const& storage, u32_t warp,
                                          u32_t lane) noexcept {
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
    __device__ __forceinline__ void score_share_(vram_ragged_view<node_id_type_> const& storage, u32_t warp,
                                                 u32_t lane) noexcept {
        for (u32_t first = warp * rows_per_warp_k; first < round.rows; first += rows_per_block_k) {
            u32_t const position = first + lane / lanes_per_row_k;
            node_id_type_ const node_id = round.ids[position];
            distance_t distance = std::numeric_limits<distance_t>::infinity();
            // A row's lanes share its ID, so every group takes one side of each branch whole.
            if (node_id != std::numeric_limits<node_id_type_>::max()) {
                vram_row_t const row = storage.row(node_id);
                distance = kit_::lower_bound(storage.kernel, round.query, row);
                if (!(distance > round.radius))
                    distance = vram_score<kit_, rung_>(storage.kernel, round.query, row, round.radius, lane);
            }
            if (lane % lanes_per_row_k == 0) round.distances[position] = distance;
        }
    }

    /** Barrier 1 counts every thread of the block, wherever each one reaches it from. */
    static __device__ __forceinline__ void barrier_() noexcept {
        asm volatile("bar.sync 1, %0;" ::"n"(vram_warps_per_block_k * cuda_warp_t::width_k) : "memory");
    }
};

/** The schedule a kit scores through on @p rung_: in place when a lane scores a row alone, with the
 *  whole block when a row takes several lanes. */
template <vram_basic_kit kit_, u32_t rung_, typename node_id_type_>
using cuda_schedule =
    std::conditional_t<vram_lanes_per_row<kit_, rung_>() == 1, vram_lane_schedule<kit_, rung_, node_id_type_>,
                       cuda_block_schedule<kit_, rung_, node_id_type_>>;

/** Hands @p walk every ID in @b [first,end) the calling warp assigns itself from @p assigned, a
 *  node ID, a touched offset or a query index: every warp walks under a lane schedule, warp zero
 *  alone under a block one, the other warps serving its rounds until it runs out. */
template <typename schedule_, typename node_id_type_, typename assigned_type_, typename walk_type_>
__device__ __forceinline__ void cuda_for_each_assigned(schedule_& schedule,
                                                       vram_ragged_view<node_id_type_> const& storage, u32_t* assigned,
                                                       assigned_type_ first, assigned_type_ end,
                                                       vram_warp_slice_t const& slice, walk_type_&& walk) noexcept {
    if constexpr (schedule_::lanes_per_row_k != 1)
        if (slice.warp != 0) return schedule.serve(storage, slice.warp, slice.lane);
    while (true) {
        u32_t offset = slice.lane == 0 ? cuda_warp_t::fetch_add_relaxed(*assigned, 1u) : 0;
        assigned_type_ const assigned_id = first + cuda_warp_t::shuffle(offset, 0);
        if (assigned_id >= end) break;
        walk(assigned_id);
    }
    if constexpr (schedule_::lanes_per_row_k != 1) schedule.finish(slice.lane);
}

/**
 *  @brief One walker inserts or re-selects one node at a time, and walkers never synchronize beyond
 *      reading slot-0 words: no device watermark, no publication order, no lock across a distance.
 *
 *  Within a wave, IDs are batch-local: each walker assigns itself the next offset of the wave,
 *  walks the published rows, and publishes its forward row with one release. The host launches each
 *  wave behind the last one's back-link passes, so a wave walks a graph whose reverse links are
 *  already merged, and re-selects behind its own, so its rows reach one another. Rows keep the
 *  neighbors the kit's @c selector_t picks.
 */
template <vram_basic_kit kit_, u32_t rung_, u32_t candidate_slots_, vram_kernel_t kernel_, typename node_id_type_>
__device__ __forceinline__ void cuda_insert(vram_build_view<node_id_type_> const& view) {
    using selector_t = typename kit_::selector_t;
    using schedule_t = cuda_schedule<kit_, rung_, node_id_type_>;
    // The scratch slice hosts the query span, the pivot span and the selection spans over the
    // walk's candidates; the visited set is in the arena.
    extern __shared__ byte_t dynamic_scratch[];
    __shared__ node_id_type_ gathered_ids[vram_warps_per_block_k][3 * cuda_warp_t::width_k];
    __shared__ typename schedule_t::shared_t schedule_state;

    vram_warp_slice_t const slice = vram_carve_slice<cuda_warp_t, node_id_type_>(view, dynamic_scratch);
    u32_t const warp = slice.warp, lane = slice.lane;
    vram_selection_scratch<node_id_type_> const selection = vram_selection_scratch<node_id_type_>::carve(
        slice.scratch + 2 * view.span_bytes, view.selection_rows);
    using frontier_t = std::conditional_t<candidate_slots_ == 0, vram_arena_frontier<cuda_warp_t, node_id_type_>,
                                          vram_register_frontier<candidate_slots_, cuda_warp_t, node_id_type_>>;
    schedule_t schedule {schedule_state};
    cuda_for_each_assigned( //
        schedule, view.storage, view.assigned_rows, view.first_id, view.end_id, slice, [&](node_id_type_ node_id) {
            vram_warp_context<frontier_t> context;
            if constexpr (candidate_slots_ == 0) {
                context.frontier.tail_.entries_ = view.arena_shape.template tail_span<node_id_type_>(view.arena,
                                                                                                     slice.warp_index);
                context.frontier.tail_.capacity_ = view.arena_shape.tail_entries;
                context.frontier.expansion_target_ = view.expansion;
            }
            context.random_generator = vram_lane_generator<cuda_warp_t>(node_id, lane);
            vram_visited_set<cuda_warp_t, node_id_type_> visited {
                slice.visited_slots, view.arena_shape.visited_hash_shift,
                view.epoch_base + static_cast<u32_t>(node_id - view.first_id) + 1};
            if constexpr (kernel_ == vram_kernel_t::refine_k)
                vram_reselect_one<kit_, selector_t>( //
                    view, schedule, node_id, visited, slice.scratch, slice.scratch + view.span_bytes,
                    gathered_ids[warp], selection, context, lane);
            else
                vram_insert_one<kit_, selector_t>( //
                    view, schedule, node_id, visited, slice.scratch, slice.scratch + view.span_bytes,
                    gathered_ids[warp], selection, context, lane);
            context.telemetry.tail_compactions = context.frontier.compactions();
            context.telemetry.tail_dropped = context.frontier.dropped();
            context.telemetry.visited_saturations = visited.saturations_;
            context.telemetry.template flush<cuda_warp_t, vram_traversal_t::inserting_k>(view.telemetry, lane);
        });
}

/** One walker back-links one node the walk launch before it touched, assigning itself the touched
 *  list's offsets from the launch counter, and resets the node's chain for the next walk launch. */
template <vram_basic_kit kit_, u32_t rung_, typename node_id_type_>
__device__ __forceinline__ void cuda_back_link(vram_build_view<node_id_type_> const& view) {
    using schedule_t = cuda_schedule<kit_, rung_, node_id_type_>;
    extern __shared__ byte_t dynamic_scratch[];
    __shared__ typename schedule_t::shared_t schedule_state;

    vram_warp_slice_t const slice = vram_carve_slice<cuda_warp_t, node_id_type_>(view, dynamic_scratch);
    u32_t const lane = slice.lane;
    vram_selection_scratch<node_id_type_> const selection = vram_selection_scratch<node_id_type_>::carve(
        slice.scratch + 2 * view.span_bytes, view.selection_rows);
    schedule_t schedule {schedule_state};
    cuda_for_each_assigned( //
        schedule, view.storage, view.assigned_rows, 0u, *view.touched_count, slice, [&](u32_t offset) {
            node_id_type_ const node_id = view.touched_ids[offset];
            vram_telemetry_t telemetry;
            vram_back_link_one<kit_, typename kit_::selector_t>( //
                view, schedule, node_id, slice.scratch, slice.scratch + view.span_bytes, selection, telemetry, lane);
            if (lane == 0) view.link_heads[node_id] = std::numeric_limits<node_id_type_>::max();
            telemetry.template flush<cuda_warp_t, vram_traversal_t::inserting_k>(view.telemetry, lane);
        });
}

/** One walker per query on the rung of @p slots_, the arena frontier at zero; walkers take queries
 *  from the batch counter, as builders take rows. */
template <vram_basic_kit kit_, u32_t rung_, u32_t slots_, typename node_id_type_>
__device__ __forceinline__ void cuda_walk(vram_search_view<node_id_type_> const& view) {
    using schedule_t = cuda_schedule<kit_, rung_, node_id_type_>;
    extern __shared__ byte_t dynamic_scratch[];
    __shared__ node_id_type_ gathered_ids[vram_warps_per_block_k][3 * cuda_warp_t::width_k];
    __shared__ typename schedule_t::shared_t schedule_state;

    vram_warp_slice_t const slice = vram_carve_slice<cuda_warp_t, node_id_type_>(view, dynamic_scratch);
    std::conditional_t<slots_ == 0, vram_arena_frontier<cuda_warp_t, node_id_type_>,
                       vram_register_frontier<slots_, cuda_warp_t, node_id_type_>>
        frontier;
    if constexpr (slots_ == 0) {
        frontier.tail_.entries_ = view.arena_shape.template tail_span<node_id_type_>(view.arena, slice.warp_index);
        frontier.tail_.capacity_ = view.arena_shape.tail_entries;
        frontier.expansion_target_ = view.expansion;
    }
    schedule_t schedule {schedule_state};
    cuda_for_each_assigned( //
        schedule, view.storage, view.assigned_queries, 0u, view.queries_count, slice, [&](u32_t query_index) {
            vram_visited_set<cuda_warp_t, node_id_type_> visited {
                slice.visited_slots, view.arena_shape.visited_hash_shift, view.epoch_base + query_index + 1};
            vram_walk_one<kit_>(view, schedule, query_index, slice.scratch, visited, frontier, gathered_ids[slice.warp],
                                slice.lane);
        });
}

#pragma region Views

/** The shared views, field by field from the host's records. */
template <typename node_id_type_>
__device__ __forceinline__ vram_ragged_view<node_id_type_> vram_view(
    vram_ragged_storage<node_id_type_> const& storage) noexcept {
    return {storage.values,       storage.offsets, storage.neighbors,
            storage.connectivity, storage.kernel,  storage.parameters};
}
template <typename node_id_type_>
__device__ __forceinline__ vram_build_view<node_id_type_> vram_view(vram_build<node_id_type_> const& build) noexcept {
    return {vram_view(build.storage),
            build.staged_offsets,
            build.tape_first,
            build.end_id,
            build.sampling_bound,
            build.scratch_bytes_per_warp,
            build.span_bytes,
            build.first_id,
            build.arena,
            vram_view(build.arena_shape),
            build.expansion,
            build.selection_rows,
            build.epoch_base,
            build.telemetry,
            build.assigned_rows,
            build.link_heads,
            build.link_next,
            build.touched_ids,
            build.touched_count};
}
template <typename node_id_type_>
__device__ __forceinline__ vram_search_view<node_id_type_> vram_view(
    vram_search<node_id_type_> const& search) noexcept {
    return {vram_view(search.storage),
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
            search.live_walks,
            search.concurrent_walks,
            search.filling_rows,
            search.result_ids,
            search.result_distances,
            search.count};
}

#pragma endregion Views

/** The body of every NVIDIA basic kernel: @p kernel_ picks the walk, @p kit_ scores on its
 *  @p rung_, @p slots_ is the register rung, and @p node_id_type_ the IDs of the records. Each
 *  modality's @c cuda_kernel names the kit its ladder picks and calls it under its launch
 *  bounds. */
template <vram_kernel_t kernel_, vram_basic_kit kit_, u32_t rung_, u32_t slots_, typename node_id_type_>
__device__ __forceinline__ void cuda_kernel_body(vram_arguments<node_id_type_> const& arguments) {
    vram_build<node_id_type_> const& build = arguments.build;
    if constexpr (kernel_ == vram_kernel_t::insert_k || kernel_ == vram_kernel_t::refine_k)
        cuda_insert<kit_, rung_, slots_, kernel_>(vram_view(build));
    else if constexpr (kernel_ == vram_kernel_t::back_link_k) cuda_back_link<kit_, rung_>(vram_view(build));
    else cuda_walk<kit_, rung_, slots_>(vram_view(arguments.search));
}

} // namespace unum::usearch::basic
