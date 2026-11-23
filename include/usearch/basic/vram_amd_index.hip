/**
 *  @file include/usearch/basic/vram_amd_index.hip
 *  @author Ash Vardanian
 *  @date November 16, 2026
 *  @brief Similarity Search over variable-length values in a single uniform latency memory space on
 *      AMD GPUs: the kernels' side.
 *
 *  The kernel side of the basic index: the body every modality's kernel runs, over the kernel, the
 *  kit, its rung, the register rung and the node-ID type. Only the modalities' AMD metrics include
 *  it, whose @c amd_kernel templates take their ladders' inputs and hand the kit their ladder picks
 *  to @ref unum::usearch::basic::hip_kernel_body, which turns the plain records of
 *  @c vram_amd_index.hpp into the shared views of @c vram_index.cuh; the host finds each kernel by
 *  the symbol @c vram_kernel_symbol spells.
 *
 *  A kit whose lane scores an entry alone walks a query or a node per warp. A kit splitting an
 *  entry over several lanes walks one per block: warp zero owns the frontier and the visited set
 *  and posts each hop's entries, and the block's four warps score them together, so long entries
 *  cost a hop a quarter of a warp's latency.
 */
#pragma once

#include <limits>      // `std::numeric_limits` - an absent entry's distance
#include <type_traits> // `std::conditional_t` - a walk's frontier and schedule

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/vram_amd_device.hip" // `hip_warp`, `cdna3_scan_t`, `cdna5_scan_t`
#include "usearch/basic/vram_index.cuh"
#include "usearch/basic/vram_amd_index.hpp"

namespace unum::usearch::basic {

/* Checked where only HIP builds compile, since wasm32 hosts see the header but never a device. */
static_assert(sizeof(std::size_t) == sizeof(u64_t), "The caller's offsets cross as the device's words");

/**
 *  @brief Scores each round of a walk's entries with the whole block: warp zero posts the query
 *      and an entry per lane, and every warp scores its share on @p rung_ of @p kit_.
 *
 *  Warps score @c lanes_per_entry_k lanes to an entry, in turns when an entry takes more lanes than
 *  the block's warps split evenly. Only warp zero walks, so the frontier, the visited set and the
 *  query's span stay its own; the other warps @ref serve rounds until it @ref finish es. Two named
 *  barriers bracket every round, so the posted entries and the span are read between them alone and
 *  rewritten outside them.
 */
template <vram_basic_kit kit_, u32_t rung_, typename node_id_type_>
struct hip_block_schedule {
    using warp_t = typename kit_::warp_t;
    using query_t = typename kit_::query_t;
    inline static constexpr u32_t lanes_per_entry_k = vram_lanes_per_entry<kit_, rung_>();

    /** Entries one warp scores per round. */
    inline static constexpr u32_t entries_per_warp_k = warp_t::width_k / lanes_per_entry_k;
    inline static constexpr u32_t entries_per_block_k = entries_per_warp_k * vram_warps_per_block_k;
    static_assert(warp_t::width_k % entries_per_block_k == 0, "A block's warps split one warp's entries evenly");

    /** The round warp zero posts, a warp's width of entries or two, and the distances the block
     *  writes for it. */
    struct shared_t {
        query_t query;
        node_id_type_ ids[2 * warp_t::width_k];
        distance_t distances[2 * warp_t::width_k];
        distance_t radius;
        u32_t entries;
        u32_t finished;
    };

    shared_t& round;

    __device__ __forceinline__ explicit hip_block_schedule(shared_t& shared) noexcept : round(shared) {
        if (threadIdx.x == 0) round.finished = 0;
    }

    /** The distance from @p query to the entry this lane of warp zero names, exact within
     *  @p radius; infinite for the sentinel. Every lane of warp zero calls it. */
    __device__ __forceinline__ distance_t score(vram_ragged_view<node_id_type_> const& storage, query_t const& query,
                                                node_id_type_ node_id, distance_t radius, u32_t lane) noexcept {
        round.ids[lane] = node_id;
        if (lane == 0) round.query = query, round.radius = radius, round.entries = warp_t::width_k;
        barrier_();
        score_share_(storage, 0, lane);
        barrier_();
        return round.distances[lane];
    }

    /** Scores a round of @p count entries of @p ids, past a warp's width and up to two, into
     *  @p scored with the whole block: the longest entries first, so every warp's turns pair one of
     *  the longest groups of entries with one of the shortest. */
    __device__ __forceinline__ void score_round(vram_ragged_view<node_id_type_> const& storage, query_t const& query,
                                                node_id_type_ const* ids, u32_t count, distance_t radius, u32_t lane,
                                                vram_scored_node<node_id_type_> (&scored)[2]) noexcept {
        vram_split_by_bytes<warp_t>(storage, ids, count, lane, scored);
        round.ids[lane] = scored[0].id(), round.ids[warp_t::width_k + lane] = scored[1].id();
        if (lane == 0) round.query = query, round.radius = radius, round.entries = 2 * warp_t::width_k;
        barrier_();
        score_share_(storage, 0, lane);
        barrier_();
        for (u32_t half = 0; half != 2; ++half)
            if (!(scored[half] == vram_scored_node<node_id_type_>::sentinel()))
                scored[half] = vram_scored_node<node_id_type_>::make(round.distances[half * warp_t::width_k + lane],
                                                                     scored[half].id());
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

    /** Releases the serving warps once warp zero has walked its last entry. */
    __device__ __forceinline__ void finish(u32_t lane) noexcept {
        if (lane == 0) round.finished = 1;
        barrier_();
    }

  private:
    __device__ __forceinline__ void score_share_(vram_ragged_view<node_id_type_> const& storage, u32_t warp,
                                                 u32_t lane) noexcept {
        for (u32_t first = warp * entries_per_warp_k; first < round.entries; first += entries_per_block_k) {
            u32_t const position = first + lane / lanes_per_entry_k;
            node_id_type_ const node_id = round.ids[position];
            distance_t distance = std::numeric_limits<distance_t>::infinity();
            // An entry's lanes share its ID, so every group takes one side of each branch whole.
            if (node_id != (std::numeric_limits<node_id_type_>::max)()) {
                vram_entry_t const entry = storage.entry(node_id);
                distance = kit_::lower_bound(storage.kernel, round.query, entry);
                if (!(distance > round.radius))
                    distance = vram_score<kit_, rung_>(storage.kernel, round.query, entry, round.radius, lane);
            }
            if (lane % lanes_per_entry_k == 0) round.distances[position] = distance;
        }
    }

    /** Every wavefront of the block reaches both barriers of every round, warp zero from its walk
     *  and the others from @ref serve, so the block barrier counts them all. */
    static __device__ __forceinline__ void barrier_() noexcept { __syncthreads(); }
};

/** The schedule a kit scores through on @p rung_: in place when a lane scores an entry alone, with
 *  the whole block when an entry takes several lanes. */
template <vram_basic_kit kit_, u32_t rung_, typename node_id_type_>
using hip_schedule =
    std::conditional_t<vram_lanes_per_entry<kit_, rung_>() == 1, vram_lane_schedule<kit_, rung_, node_id_type_>,
                       hip_block_schedule<kit_, rung_, node_id_type_>>;

/** Hands @p walk every ID in @b [first,end) the calling warp assigns itself from @p assigned, a
 *  node ID, a touched offset or a query index: every warp walks under a lane schedule, warp zero
 *  alone under a block one, the other warps serving its rounds until it runs out. */
template <typename warp_type_, typename schedule_, typename node_id_type_, typename assigned_type_, typename walk_type_>
__device__ __forceinline__ void hip_for_each_assigned(schedule_& schedule,
                                                      vram_ragged_view<node_id_type_> const& storage, u32_t* assigned,
                                                      assigned_type_ first, assigned_type_ end,
                                                      vram_warp_slice_t const& slice, walk_type_&& walk) noexcept {
    if constexpr (schedule_::lanes_per_entry_k != 1)
        if (slice.warp != 0) return schedule.serve(storage, slice.warp, slice.lane);
    while (true) {
        u32_t offset = slice.lane == 0 ? warp_type_::fetch_add_relaxed(*assigned, 1u) : 0;
        assigned_type_ const assigned_id = first + warp_type_::shuffle(offset, 0);
        if (assigned_id >= end) break;
        walk(assigned_id);
    }
    if constexpr (schedule_::lanes_per_entry_k != 1) schedule.finish(slice.lane);
}

/**
 *  @brief One walker inserts or re-selects one node at a time, and walkers never synchronize beyond
 *      reading slot-0 words: no device watermark, no publication order, no lock across a distance.
 *
 *  Within a wave, IDs are batch-local: each walker assigns itself the next offset of the wave,
 *  walks the published nodes, and publishes its forward neighbor row with one release. The host
 *  launches each wave behind the last one's back-link passes, so a wave walks a graph whose reverse
 *  links are already merged, and re-selects behind its own, so its nodes reach one another. Nodes
 *  keep the neighbors the kit's @c selector_t picks.
 */
template <vram_basic_kit kit_, u32_t rung_, u32_t candidate_slots_, vram_kernel_t kernel_, typename node_id_type_>
__device__ __forceinline__ void hip_insert(vram_build_view<node_id_type_> const& view) {
    using warp_t = typename kit_::warp_t;
    using scan_t = typename kit_::scan_t;
    using selector_t = typename kit_::selector_t;
    using schedule_t = hip_schedule<kit_, rung_, node_id_type_>;
    // The scratch slice hosts the query span, the pivot span and the selection spans over the
    // walk's candidates; the visited set is in the arena.
    extern __shared__ byte_t dynamic_scratch[];
    __shared__ node_id_type_ gathered_ids[vram_warps_per_block_k][3 * warp_t::width_k];
    __shared__ typename schedule_t::shared_t schedule_state;

    vram_warp_slice_t const slice = vram_carve_slice<warp_t, node_id_type_>(view, dynamic_scratch);
    u32_t const warp = slice.warp, lane = slice.lane;
    vram_selection_scratch<node_id_type_> const selection = vram_selection_scratch<node_id_type_>::carve(
        slice.scratch + 2 * view.span_bytes, view.selection_nodes);
    using frontier_t = std::conditional_t<candidate_slots_ == 0, vram_arena_frontier<warp_t, scan_t, node_id_type_>,
                                          vram_register_frontier<candidate_slots_, warp_t, node_id_type_>>;
    schedule_t schedule {schedule_state};
    hip_for_each_assigned<warp_t>( //
        schedule, view.storage, view.assigned_nodes, view.first_id, view.end_id, slice, [&](node_id_type_ node_id) {
            vram_warp_context<frontier_t> context;
            if constexpr (candidate_slots_ == 0) {
                context.frontier.tail_.entries_ = view.arena_shape.template tail_span<node_id_type_>(view.arena,
                                                                                                     slice.warp_index);
                context.frontier.tail_.capacity_ = view.arena_shape.tail_entries;
                context.frontier.expansion_target_ = view.expansion;
            }
            context.random_generator = vram_lane_generator<warp_t>(node_id, lane);
            node_id_type_ const walked = node_id - view.first_id;
            vram_visited_set<warp_t, node_id_type_> visited {slice.visited_slots, view.arena_shape.visited_hash_shift,
                                                             view.epoch_base + static_cast<u32_t>(walked) + 1};
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
            context.telemetry.template flush<warp_t, vram_traversal_t::inserting_k>(view.telemetry, lane);
        });
}

/** One walker back-links one node the walk launch before it touched, assigning itself the touched
 *  list's offsets from the launch counter, and resets the node's chain for the next walk launch. */
template <vram_basic_kit kit_, u32_t rung_, typename node_id_type_>
__device__ __forceinline__ void hip_back_link(vram_build_view<node_id_type_> const& view) {
    using warp_t = typename kit_::warp_t;
    using schedule_t = hip_schedule<kit_, rung_, node_id_type_>;
    extern __shared__ byte_t dynamic_scratch[];
    __shared__ typename schedule_t::shared_t schedule_state;

    vram_warp_slice_t const slice = vram_carve_slice<warp_t, node_id_type_>(view, dynamic_scratch);
    u32_t const lane = slice.lane;
    vram_selection_scratch<node_id_type_> const selection = vram_selection_scratch<node_id_type_>::carve(
        slice.scratch + 2 * view.span_bytes, view.selection_nodes);
    schedule_t schedule {schedule_state};
    hip_for_each_assigned<warp_t>( //
        schedule, view.storage, view.assigned_nodes, 0u, *view.touched_count, slice, [&](u32_t offset) {
            node_id_type_ const node_id = view.touched_ids[offset];
            vram_telemetry_t telemetry;
            vram_back_link_one<kit_, typename kit_::selector_t>( //
                view, schedule, node_id, slice.scratch, slice.scratch + view.span_bytes, selection, telemetry, lane);
            if (lane == 0) view.link_heads[node_id] = (std::numeric_limits<node_id_type_>::max)();
            telemetry.template flush<warp_t, vram_traversal_t::inserting_k>(view.telemetry, lane);
        });
}

/** One walker files one entry of a gather's runs at a time under its nearest hub, assigning itself
 *  the runs' positions from the launch counter. */
template <vram_basic_kit kit_, u32_t rung_, typename node_id_type_>
__device__ __forceinline__ void hip_gather_neighborhoods(vram_round_view<node_id_type_> const& view) {
    using warp_t = typename kit_::warp_t;
    using schedule_t = hip_schedule<kit_, rung_, node_id_type_>;
    extern __shared__ byte_t dynamic_scratch[];
    __shared__ typename schedule_t::shared_t schedule_state;

    vram_warp_slice_t const slice = vram_carve_slice<warp_t, node_id_type_>(view, dynamic_scratch);
    schedule_t schedule {schedule_state};
    vram_telemetry_t telemetry;
    hip_for_each_assigned<warp_t>( //
        schedule, view.storage, view.assigned, 0u, view.run_entries[view.runs_count], slice,
        [&](u32_t position) { vram_gather_one<kit_>(view, schedule, position, slice.scratch, telemetry, slice.lane); });
    telemetry.template flush<warp_t, vram_traversal_t::searching_k>(view.telemetry, slice.lane);
}

/** One walker re-stitches one entry of the neighborhoods at a time, assigning itself their
 *  positions from the launch counter; an entry sits in one neighborhood and its walker alone writes
 *  its row, so no row is locked, and the links added and dropped and the rows changed add to the
 *  counters once per warp. */
template <vram_basic_kit kit_, u32_t rung_, typename node_id_type_>
__device__ __forceinline__ void hip_restitch(vram_round_view<node_id_type_> const& view) {
    using warp_t = typename kit_::warp_t;
    using schedule_t = hip_schedule<kit_, rung_, node_id_type_>;
    extern __shared__ byte_t dynamic_scratch[];
    __shared__ typename schedule_t::shared_t schedule_state;

    vram_warp_slice_t const slice = vram_carve_slice<warp_t, node_id_type_>(view, dynamic_scratch);
    u32_t const lane = slice.lane;
    vram_selection_scratch<node_id_type_> const selection = vram_selection_scratch<node_id_type_>::carve(
        slice.scratch + 2 * view.span_bytes, view.selection_nodes);
    schedule_t schedule {schedule_state};
    vram_telemetry_t telemetry;
    u32_t added = 0, dropped = 0, changed = 0;
    hip_for_each_assigned<warp_t>( //
        schedule, view.storage, view.assigned, 0u, view.neighborhood_offsets[view.neighborhoods_count], slice,
        [&](u32_t position) {
            vram_restitch_one<kit_, typename kit_::selector_t>( //
                view, schedule, position, slice.scratch, slice.scratch + view.span_bytes, selection, telemetry, added,
                dropped, changed, lane);
        });
    vram_round_counters_t* const counters = view.health.counters;
    if (lane == 0 && added) warp_t::fetch_add_relaxed(counters->added_links, added);
    if (lane == 0 && dropped) warp_t::fetch_add_relaxed(counters->dropped_links, dropped);
    if (lane == 0 && changed) warp_t::fetch_add_relaxed(counters->changed, changed);
    telemetry.template flush<warp_t, vram_traversal_t::searching_k>(view.telemetry, lane);
}

/** One walker per query on the rung of @p slots_, the arena frontier at zero; walkers take queries
 *  from the batch counter, as builders take entries. */
template <vram_basic_kit kit_, u32_t rung_, u32_t slots_, typename node_id_type_>
__device__ __forceinline__ void hip_walk(vram_search_view<node_id_type_> const& view) {
    using warp_t = typename kit_::warp_t;
    using scan_t = typename kit_::scan_t;
    using schedule_t = hip_schedule<kit_, rung_, node_id_type_>;
    extern __shared__ byte_t dynamic_scratch[];
    __shared__ node_id_type_ gathered_ids[vram_warps_per_block_k][3 * warp_t::width_k];
    __shared__ typename schedule_t::shared_t schedule_state;

    vram_warp_slice_t const slice = vram_carve_slice<warp_t, node_id_type_>(view, dynamic_scratch);
    std::conditional_t<slots_ == 0, vram_arena_frontier<warp_t, scan_t, node_id_type_>,
                       vram_register_frontier<slots_, warp_t, node_id_type_>>
        frontier;
    if constexpr (slots_ == 0) {
        frontier.tail_.entries_ = view.arena_shape.template tail_span<node_id_type_>(view.arena, slice.warp_index);
        frontier.tail_.capacity_ = view.arena_shape.tail_entries;
        frontier.expansion_target_ = view.expansion;
    }
    schedule_t schedule {schedule_state};
    hip_for_each_assigned<warp_t>( //
        schedule, view.storage, view.assigned_queries, 0u, view.queries_count, slice, [&](u32_t query_index) {
            vram_visited_set<warp_t, node_id_type_> visited {slice.visited_slots, view.arena_shape.visited_hash_shift,
                                                             view.epoch_base + query_index + 1};
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
            build.selection_nodes,
            build.epoch_base,
            build.telemetry,
            build.assigned_nodes,
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
            search.filling_nodes,
            search.result_ids,
            search.result_distances,
            search.count};
}

template <typename node_id_type_>
__device__ __forceinline__ vram_round_view<node_id_type_> vram_view(vram_round<node_id_type_> const& round) noexcept {
    return {vram_view(round.storage),
            round.span_bytes,
            round.scratch_bytes_per_warp,
            round.arena,
            vram_view(round.arena_shape),
            round.selection_nodes,
            round.telemetry,
            round.assigned,
            round.entries,
            round.hub_nodes,
            round.runs,
            round.run_entries,
            round.runs_count,
            round.entry_neighborhoods,
            round.neighborhood_offsets,
            round.neighborhoods_count,
            round.health};
}

#pragma endregion Views

/** The body of every AMD basic kernel: @p kernel_ picks the walk, @p kit_ scores on its
 *  @p rung_, @p slots_ is the register rung, and @p node_id_type_ the IDs of the records. Each
 *  modality's @c amd_kernel names the kit its ladder picks, and calls the body under the launch
 *  bounds of that kit's wavefront. */
template <vram_kernel_t kernel_, vram_basic_kit kit_, u32_t rung_, u32_t slots_, typename node_id_type_>
__device__ __forceinline__ void hip_kernel_body(vram_arguments<node_id_type_> const& arguments) {
    vram_build<node_id_type_> const& build = arguments.build;
    // Kits without a load hook store values as saved and size spans by length; theirs never launch.
    constexpr bool loads_values = requires { kit_::load_values(arguments.load); };
    constexpr bool sizes_spans = requires { kit_::size_spans(arguments.load); };
    if constexpr (kernel_ == vram_kernel_t::insert_k || kernel_ == vram_kernel_t::refine_k)
        hip_insert<kit_, rung_, slots_, kernel_>(vram_view(build));
    else if constexpr (kernel_ == vram_kernel_t::back_link_k) hip_back_link<kit_, rung_>(vram_view(build));
    else if constexpr (kernel_ == vram_kernel_t::gather_neighborhoods_k)
        hip_gather_neighborhoods<kit_, rung_>(vram_view(arguments.round));
    else if constexpr (kernel_ == vram_kernel_t::restitch_k) hip_restitch<kit_, rung_>(vram_view(arguments.round));
    else if constexpr (kernel_ == vram_kernel_t::load_values_k && loads_values) kit_::load_values(arguments.load);
    else if constexpr (kernel_ == vram_kernel_t::size_spans_k && sizes_spans) kit_::size_spans(arguments.load);
    else if constexpr (kernel_ == vram_kernel_t::load_values_k || kernel_ == vram_kernel_t::size_spans_k) return;
    else hip_walk<kit_, rung_, slots_>(vram_view(arguments.search));
}

} // namespace unum::usearch::basic
