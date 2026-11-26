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
 *  ladder picks to @ref unum::usearch::basic::cuda_kernel_body, which turns the plain records of
 *  @c vram_nvidia_index.hpp into the shared views of @c vram_index.cuh; the host finds each kernel
 *  by the symbol @c vram_kernel_symbol spells.
 *
 *  Every kit rung compiles its kernels for a walker per warp and, where an entry's lanes split over
 *  the block's warps, for one walker per block; the host launches the first whose walkers' scratch
 *  fits shared memory at the kernel's resident blocks. With a walker per warp, every warp walks a
 *  query or a node of its own and scores its own entries: a kit whose lane scores an entry alone
 *  scores a warp's width at once, and a kit splitting an entry over several lanes scores them in
 *  turns, so a long entry delays its own warp and never the block's others. With one walker per
 *  block, warp zero owns the frontier and the visited set and posts each hop's entries, and all
 *  four warps of the block score them together, so a hop over long entries takes a quarter of a
 *  lone warp's latency.
 */
#pragma once

#include <limits>      // `std::numeric_limits` - an absent entry's distance
#include <type_traits> // `std::conditional_t` - a walk's frontier and schedule

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/vram_nvidia_device.cuh" // `cuda_warp_t`, `ampere_scan_t`
#include "usearch/basic/vram_index.cuh"
#include "usearch/basic/vram_nvidia_index.hpp"

namespace unum::usearch::basic {

/* Checked where only CUDA builds compile, since wasm32 hosts see the header but never a device. */
static_assert(sizeof(std::size_t) == sizeof(u64_t), "The caller's offsets cross as the device's words");

/** A kit rung whose entries take @p role_ at least a lane per warp of the block, so every warp can
 *  take a share of one walker's rounds; never a builder by sketches, a lane to an entry. */
template <typename kit_, score_role_t role_, u32_t rung_>
concept a_cuda_block_scorable_kit = vram_lanes_per_entry<kit_, role_, rung_>() >= vram_warps_per_block_k;

/** The round warp zero of a block schedule posts for @p role_, a warp's width of entries or two,
 *  and the distances the block writes for it; alike on every rung. */
template <a_vram_basic_kit kit_, score_role_t role_, typename node_id_type_>
struct cuda_block_round {
    vram_query<kit_, role_> query;
    node_id_type_ ids[2 * cuda_warp_t::width_k];
    distance_t distances[2 * cuda_warp_t::width_k];
    distance_t radius;
    u32_t entries;
    u32_t finished;
};

/**
 *  @brief Scores each round of a walk's entries with the whole block: warp zero posts the query
 *      and an entry per lane, and every warp scores its share on @p rung_ of @p kit_ for @p role_.
 *
 *  Warps score @c lanes_per_entry_k lanes to an entry, in turns when an entry takes more lanes than
 *  the block's warps split evenly, and two neighbouring entries a turn for a paired kit. Only warp
 *  zero walks, so the frontier, the visited set and the query's span stay its own; the other warps
 *  @ref serve rounds until it @ref finish es. Two named barriers bracket every round, so the posted
 *  entries and the span are read between them alone and rewritten outside them.
 */
template <a_vram_basic_kit kit_, score_role_t role_, u32_t rung_, typename node_id_type_>
struct cuda_block_schedule {
    inline static constexpr score_role_t role_k = role_;
    using query_t = vram_query<kit_, role_>;
    inline static constexpr u32_t lanes_per_entry_k = vram_lanes_per_entry<kit_, role_, rung_>();
    static_assert(a_cuda_block_scorable_kit<kit_, role_, rung_>, "Every warp of the block takes a share of an entry");

    /** Entries one warp scores per round. */
    inline static constexpr u32_t entries_per_warp_k = cuda_warp_t::width_k / lanes_per_entry_k;
    inline static constexpr u32_t entries_per_block_k = entries_per_warp_k * vram_warps_per_block_k;
    static_assert(cuda_warp_t::width_k % entries_per_block_k == 0, "A block's warps split one warp's entries evenly");

    using shared_t = cuda_block_round<kit_, role_, node_id_type_>;

    shared_t& round;

    __device__ __forceinline__ explicit cuda_block_schedule(shared_t& shared) noexcept : round(shared) {
        if (threadIdx.x == 0) round.finished = 0;
    }

    /** The distance from @p query to the entry this lane of warp zero names, exact within
     *  @p radius; infinite for the sentinel. Every lane of warp zero calls it. */
    __device__ __forceinline__ distance_t score(vram_ragged_view<node_id_type_> const& storage, query_t const& query,
                                                node_id_type_ node_id, distance_t radius, u32_t lane) noexcept {
        round.ids[lane] = node_id;
        if (lane == 0) round.query = query, round.radius = radius, round.entries = cuda_warp_t::width_k;
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
        vram_split_by_bytes<cuda_warp_t>(storage, ids, count, lane, scored);
        round.ids[lane] = scored[0].id(), round.ids[cuda_warp_t::width_k + lane] = scored[1].id();
        if (lane == 0) round.query = query, round.radius = radius, round.entries = 2 * cuda_warp_t::width_k;
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

    /** Releases the serving warps once warp zero has walked its last entry. */
    __device__ __forceinline__ void finish(u32_t lane) noexcept {
        if (lane == 0) round.finished = 1;
        barrier_();
    }

  private:
    __device__ __forceinline__ void score_share_(vram_ragged_view<node_id_type_> const& storage, u32_t warp,
                                                 u32_t lane) noexcept {
        if constexpr (a_vram_paired_kit<kit_, role_, rung_>) {
            // Each warp takes neighbours of the round's order, alike in length once split.
            for (u32_t first = 2 * warp; first < round.entries; first += 2 * vram_warps_per_block_k) {
                vram_distance_pair_t const distances = vram_score_pair<kit_, role_, rung_>(
                    storage, round.query, round.ids[first], round.ids[first + 1], round.radius, lane);
                if (lane == 0) round.distances[first] = distances.first, round.distances[first + 1] = distances.second;
            }
            return;
        }
        for (u32_t first = warp * entries_per_warp_k; first < round.entries; first += entries_per_block_k) {
            u32_t const position = first + lane / lanes_per_entry_k;
            node_id_type_ const node_id = round.ids[position];
            distance_t const distance = vram_score_entry<kit_, role_, rung_>(storage, round.query, node_id,
                                                                             round.radius, lane);
            if (lane % lanes_per_entry_k == 0) round.distances[position] = distance;
        }
    }

    /** Barrier 1 counts every thread of the block, wherever each one reaches it from. */
    static __device__ __forceinline__ void barrier_() noexcept {
        asm volatile("bar.sync 1, %0;" ::"n"(vram_warps_per_block_k * cuda_warp_t::width_k) : "memory");
    }
};

/** The schedule @p kit_ scores through for @p role_ on @p rung_ with @p walkers_ walkers per block:
 *  the whole block's for one, each walking warp's own for a warp each. */
template <a_vram_basic_kit kit_, score_role_t role_, u32_t rung_, u32_t walkers_, typename node_id_type_>
using cuda_schedule = std::conditional_t<walkers_ == 1, cuda_block_schedule<kit_, role_, rung_, node_id_type_>,
                                         vram_lane_schedule<kit_, role_, rung_, node_id_type_>>;

/** Each of @p walkers_ walkers' batches of three lane-widths of IDs, a round of two scored together
 *  and its carry. */
template <u32_t walkers_, typename node_id_type_>
using cuda_gathered_ids = node_id_type_[walkers_][3 * cuda_warp_t::width_k];

/** The layout of @p kit_ 's kernels for @p role_ with @p walkers walkers per block over
 *  @p node_id_type_ IDs, @p lanes_per_entry lanes to an entry of the kit's canonical score: the
 *  walkers' gathered IDs and the schedule's state, and empty for one walker where an entry takes
 *  fewer lanes than the block has warps, as every builder by sketches does. */
template <a_vram_basic_kit kit_, score_role_t role_, typename node_id_type_>
constexpr cuda_block_layout_t cuda_block_layout(u32_t walkers, u32_t lanes_per_entry) noexcept {
    if (walkers != 1)
        return {vram_warps_per_block_k,
                static_cast<u32_t>(sizeof(cuda_gathered_ids<vram_warps_per_block_k, node_id_type_>) +
                                   sizeof(vram_lane_round))};
    if (vram_scores_by_sketch<kit_, role_>() || lanes_per_entry < vram_warps_per_block_k) return {};
    return {1, static_cast<u32_t>(sizeof(cuda_gathered_ids<1, node_id_type_>) +
                                  sizeof(cuda_block_round<kit_, role_, node_id_type_>))};
}

/** A schedule whose other warps serve the walking warp's rounds until it finishes. */
template <typename schedule_>
concept a_cuda_serving_schedule = requires(schedule_& schedule, u32_t lane) { schedule.finish(lane); };

/** Hands @p walk every ID in @b [first,end) the calling warp assigns itself from @p assigned, a
 *  wave's offset, a touched offset or a query index: every warp walks under a lane schedule, warp
 *  zero alone under a block one, the other warps serving its rounds until it runs out. */
template <typename schedule_, typename node_id_type_, typename assigned_type_, typename walk_type_>
__device__ __forceinline__ void cuda_for_each_assigned(schedule_& schedule,
                                                       vram_ragged_view<node_id_type_> const& storage, u32_t* assigned,
                                                       assigned_type_ first, assigned_type_ end,
                                                       vram_warp_slice_t const& slice, walk_type_&& walk) noexcept {
    if constexpr (a_cuda_serving_schedule<schedule_>)
        if (slice.warp != 0) return schedule.serve(storage, slice.warp, slice.lane);
    while (true) {
        u32_t offset = slice.lane == 0 ? cuda_warp_t::fetch_add_relaxed(*assigned, 1u) : 0;
        assigned_type_ const assigned_id = first + cuda_warp_t::shuffle(offset, 0);
        if (assigned_id >= end) break;
        walk(assigned_id);
    }
    if constexpr (a_cuda_serving_schedule<schedule_>) schedule.finish(slice.lane);
}

/**
 *  @brief One walker inserts or re-selects one node at a time, and walkers never synchronize beyond
 *      reading slot-0 words: no device watermark, no publication order, no lock across a distance.
 *
 *  Within a wave, IDs are batch-local: each walker assigns itself the next offset of the wave,
 *  walks the published nodes, and publishes its forward neighbor row with one release. The host
 *  launches each wave behind the last one's back-link passes, so a wave walks a graph whose reverse
 *  links are already merged, and re-selects behind its own, so its nodes reach one another. Nodes
 *  keep the neighbors the kit's @c selector_t picks. A re-selection over the view's listed IDs
 *  walks the ID at each offset, every ID once, so each row still has one writer in the launch.
 */
template <a_vram_basic_kit kit_, u32_t rung_, u32_t candidate_slots_, vram_kernel_t kernel_, u32_t walkers_,
          typename node_id_type_>
__device__ __forceinline__ void cuda_insert(vram_build_view<node_id_type_> const& view) {
    using selector_t = typename kit_::selector_t;
    using schedule_t = cuda_schedule<kit_, score_role_t::build_k, rung_, walkers_, node_id_type_>;
    // The scratch slice hosts the query span, the pivot span and the selection spans over the
    // walk's candidates; the visited set is in the arena.
    extern __shared__ byte_t dynamic_scratch[];
    __shared__ cuda_gathered_ids<walkers_, node_id_type_> gathered_ids;
    __shared__ typename schedule_t::shared_t schedule_state;

    vram_warp_slice_t const slice = vram_carve_slice<cuda_warp_t, node_id_type_, walkers_>(view, dynamic_scratch);
    u32_t const warp = slice.warp, lane = slice.lane;
    vram_selection_scratch<node_id_type_> const selection = vram_selection_scratch<node_id_type_>::carve(
        slice.scratch + 2 * view.span_bytes, view.selection_nodes);
    using frontier_t =
        std::conditional_t<candidate_slots_ == 0, vram_arena_frontier<cuda_warp_t, ampere_scan_t, node_id_type_>,
                           vram_register_frontier<candidate_slots_, cuda_warp_t, node_id_type_>>;
    schedule_t schedule {schedule_state};
    cuda_for_each_assigned( //
        schedule, view.storage, view.assigned_nodes, 0u, static_cast<u32_t>(view.end_id - view.first_id), slice,
        [&](u32_t offset) {
            node_id_type_ const node_id = view.listed ? view.listed[offset] : view.first_id + offset;
            vram_warp_context<frontier_t> context;
            if constexpr (candidate_slots_ == 0) {
                context.frontier.tail_.entries_ = view.arena_shape.template tail_span<node_id_type_>(view.arena,
                                                                                                     slice.warp_index);
                context.frontier.tail_.capacity_ = view.arena_shape.tail_entries;
                context.frontier.expansion_target_ = view.expansion;
            }
            context.random_generator = vram_lane_generator<cuda_warp_t>(node_id, lane);
            vram_visited_set<cuda_warp_t, node_id_type_> visited {
                slice.visited_slots, view.arena_shape.visited_hash_shift, view.epoch_base + offset + 1};
            if constexpr (kernel_ == vram_kernel_t::refine_k)
                vram_reselect_one<kit_, selector_t>( //
                    view, schedule, node_id, offset, visited, slice.scratch, slice.scratch + view.span_bytes,
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
template <a_vram_basic_kit kit_, u32_t rung_, u32_t walkers_, typename node_id_type_>
__device__ __forceinline__ void cuda_back_link(vram_build_view<node_id_type_> const& view) {
    using schedule_t = cuda_schedule<kit_, score_role_t::build_k, rung_, walkers_, node_id_type_>;
    extern __shared__ byte_t dynamic_scratch[];
    __shared__ typename schedule_t::shared_t schedule_state;

    vram_warp_slice_t const slice = vram_carve_slice<cuda_warp_t, node_id_type_, walkers_>(view, dynamic_scratch);
    u32_t const lane = slice.lane;
    vram_selection_scratch<node_id_type_> const selection = vram_selection_scratch<node_id_type_>::carve(
        slice.scratch + 2 * view.span_bytes, view.selection_nodes);
    schedule_t schedule {schedule_state};
    cuda_for_each_assigned( //
        schedule, view.storage, view.assigned_nodes, 0u, *view.touched_count, slice, [&](u32_t offset) {
            node_id_type_ const node_id = view.touched_ids[offset];
            vram_telemetry_t telemetry;
            vram_back_link_one<kit_, typename kit_::selector_t>( //
                view, schedule, node_id, slice.scratch, slice.scratch + view.span_bytes, selection, telemetry, lane);
            if (lane == 0) view.link_heads[node_id] = (std::numeric_limits<node_id_type_>::max)();
            telemetry.template flush<cuda_warp_t, vram_traversal_t::inserting_k>(view.telemetry, lane);
        });
}

/** One walker files one entry of a gather's runs at a time under its nearest hub, assigning itself
 *  the runs' positions from the launch counter. */
template <a_vram_basic_kit kit_, u32_t rung_, u32_t walkers_, typename node_id_type_>
__device__ __forceinline__ void cuda_gather_neighborhoods(vram_round_view<node_id_type_> const& view) {
    using schedule_t = cuda_schedule<kit_, score_role_t::build_k, rung_, walkers_, node_id_type_>;
    extern __shared__ byte_t dynamic_scratch[];
    __shared__ typename schedule_t::shared_t schedule_state;

    vram_warp_slice_t const slice = vram_carve_slice<cuda_warp_t, node_id_type_, walkers_>(view, dynamic_scratch);
    schedule_t schedule {schedule_state};
    vram_telemetry_t telemetry;
    cuda_for_each_assigned( //
        schedule, view.storage, view.assigned, 0u, view.run_entries[view.runs_count], slice,
        [&](u32_t position) { vram_gather_one<kit_>(view, schedule, position, slice.scratch, telemetry, slice.lane); });
    telemetry.template flush<cuda_warp_t, vram_traversal_t::searching_k>(view.telemetry, slice.lane);
}

/** One walker re-stitches one entry of the neighborhoods at a time, assigning itself their
 *  positions from the launch counter; an entry sits in one neighborhood and its walker alone writes
 *  its row, so no row is locked, and the links added and dropped and the rows changed add to the
 *  counters once per warp. */
template <a_vram_basic_kit kit_, u32_t rung_, u32_t walkers_, typename node_id_type_>
__device__ __forceinline__ void cuda_restitch(vram_round_view<node_id_type_> const& view) {
    using schedule_t = cuda_schedule<kit_, score_role_t::build_k, rung_, walkers_, node_id_type_>;
    extern __shared__ byte_t dynamic_scratch[];
    __shared__ typename schedule_t::shared_t schedule_state;

    vram_warp_slice_t const slice = vram_carve_slice<cuda_warp_t, node_id_type_, walkers_>(view, dynamic_scratch);
    u32_t const lane = slice.lane;
    vram_selection_scratch<node_id_type_> const selection = vram_selection_scratch<node_id_type_>::carve(
        slice.scratch + 2 * view.span_bytes, view.selection_nodes);
    schedule_t schedule {schedule_state};
    vram_telemetry_t telemetry;
    u32_t added = 0, dropped = 0, changed = 0;
    cuda_for_each_assigned( //
        schedule, view.storage, view.assigned, 0u, view.neighborhood_offsets[view.neighborhoods_count], slice,
        [&](u32_t position) {
            vram_restitch_one<kit_, typename kit_::selector_t>( //
                view, schedule, position, slice.scratch, slice.scratch + view.span_bytes, selection, telemetry, added,
                dropped, changed, lane);
        });
    vram_round_counters_t* const counters = view.health.counters;
    if (lane == 0 && added) cuda_warp_t::fetch_add_relaxed(counters->added_links, added);
    if (lane == 0 && dropped) cuda_warp_t::fetch_add_relaxed(counters->dropped_links, dropped);
    if (lane == 0 && changed) cuda_warp_t::fetch_add_relaxed(counters->changed, changed);
    telemetry.template flush<cuda_warp_t, vram_traversal_t::searching_k>(view.telemetry, lane);
}

/** One walker per query for @p kernel_, a search or its seeding walk, or per range of a query for
 *  a scan, on the rung of @p slots_, the arena frontier at zero; walkers take queries or ranges
 *  from the batch counter, as builders take entries. */
template <a_vram_basic_kit kit_, vram_kernel_t kernel_, u32_t rung_, u32_t slots_, u32_t walkers_,
          typename node_id_type_>
__device__ __forceinline__ void cuda_walk(vram_search_view<node_id_type_> const& view) {
    constexpr score_role_t role_k = vram_kernel_role(kernel_);
    using schedule_t = cuda_schedule<kit_, role_k, rung_, walkers_, node_id_type_>;
    extern __shared__ byte_t dynamic_scratch[];
    __shared__ cuda_gathered_ids<walkers_, node_id_type_> gathered_ids;
    __shared__ typename schedule_t::shared_t schedule_state;

    vram_warp_slice_t const slice = vram_carve_slice<cuda_warp_t, node_id_type_, walkers_>(view, dynamic_scratch);
    std::conditional_t<slots_ == 0, vram_arena_frontier<cuda_warp_t, ampere_scan_t, node_id_type_>,
                       vram_register_frontier<slots_, cuda_warp_t, node_id_type_>>
        frontier;
    if constexpr (slots_ == 0) {
        frontier.tail_.entries_ = view.arena_shape.template tail_span<node_id_type_>(view.arena, slice.warp_index);
        frontier.tail_.capacity_ = view.arena_shape.tail_entries;
        frontier.expansion_target_ = view.expansion;
    }
    schedule_t schedule {schedule_state};
    u32_t const items = kernel_ == vram_kernel_t::scan_k ? view.queries_count * view.ranges : view.queries_count;
    cuda_for_each_assigned( //
        schedule, view.storage, view.assigned_queries, 0u, items, slice, [&](u32_t item) {
            if constexpr (kernel_ == vram_kernel_t::scan_k)
                vram_scan_one<kit_>(view, schedule, item, slice.scratch, frontier, gathered_ids[slice.warp],
                                    slice.lane);
            else {
                vram_visited_set<cuda_warp_t, node_id_type_> visited {
                    slice.visited_slots, view.arena_shape.visited_hash_shift, view.epoch_base + item + 1};
                vram_walk_one<kit_, role_k>(view, schedule, item, slice.scratch, visited, frontier,
                                            gathered_ids[slice.warp], slice.lane);
            }
        });
}

#pragma region Stored Values

/** Fills the sketch of every value of a staged batch: a warp per value, striding the grid, each
 *  sketch written from its own value's sequence, in front of a stored value or into a bare query's
 *  row, through the warp's share of the launch's shared scratch. */
template <a_vram_sketching_kit kit_>
__device__ __forceinline__ void cuda_sketch_values(vram_sketch_t const& batch) {
    extern __shared__ byte_t dynamic_scratch[];
    vram_grid_warp_t const warp = vram_grid_warp<typename kit_::warp_t>();
    byte_t* const scratch = dynamic_scratch +
                            threadIdx.x / kit_::warp_t::width_k * kit_::sketch_scratch_bytes(batch.sketch_bytes);
    for (u32_t value = warp.index; value < batch.count; value += warp.count) {
        u64_t const first = batch.offsets[value] + batch.shift, end = batch.offsets[value + 1] + batch.shift;
        if (batch.sketches)
            kit_::sketch_write(batch.tables, {batch.values + first, static_cast<u32_t>(end - first)}, warp.lane,
                               batch.sketches + std::size_t {value} * batch.sketch_bytes, batch.sketch_bytes, scratch);
        else
            kit_::sketch_write(batch.tables,
                               vram_part<kit_, score_role_t::search_k>(batch.values, first, end, batch.sketch_bytes),
                               warp.lane, batch.values + first, batch.sketch_bytes, scratch);
    }
}

/**
 *  @brief Stores a load's run for a kit closing its values.
 *
 *  A warp per value copies the saved sequence behind the gap its sketch of the load's bytes takes,
 *  writes the closing padding and its count, and under a sketching kit fills the sketch from the
 *  stored sequence through the warp's share of the launch's shared scratch.
 *
 *  The saved sums count bytes from the run's first, and every stored value starts on a word of the
 *  tape, as its planned start says.
 */
template <typename kit_, typename node_id_type_>
__device__ __forceinline__ void cuda_load_closed(vram_load<node_id_type_> const& load) {
    using warp_t = typename kit_::warp_t;
    vram_grid_warp_t const warp = vram_grid_warp<warp_t>();
    u64_t const saved_first = load.saved_offsets[0];
    u32_t const sketch_bytes = load.sketch_bytes;
    for (u32_t value = warp.index; value < load.count; value += warp.count) {
        byte_t const* const saved = load.saved + (load.saved_offsets[value] - saved_first);
        u32_t const bytes = static_cast<u32_t>(load.saved_offsets[value + 1] - load.saved_offsets[value]);
        u64_t const first = load.stored_offsets[value], end = load.stored_offsets[value + 1];
        byte_t* const sequence = load.tape + first + sketch_bytes;
        u32_t const closing = static_cast<u32_t>(end - first) - sketch_bytes - bytes;
        for (u32_t byte = warp.lane; byte < bytes; byte += warp_t::width_k) sequence[byte] = saved[byte];
        if (warp.lane < closing)
            sequence[bytes + warp.lane] = static_cast<byte_t>(warp.lane + 1 == closing ? closing : 0);
        if constexpr (a_vram_sketching_kit<kit_>) {
            extern __shared__ byte_t dynamic_scratch[];
            byte_t* const scratch = dynamic_scratch +
                                    threadIdx.x / warp_t::width_k * kit_::sketch_scratch_bytes(sketch_bytes);
            warp_t::sync();
            kit_::sketch_write(load.tables, {sequence, bytes}, warp.lane, load.tape + first, sketch_bytes, scratch);
        }
    }
}

#pragma endregion Stored Values

#pragma region Views

/** The shared views, field by field from the host's records. */
template <typename node_id_type_>
__device__ __forceinline__ vram_ragged_view<node_id_type_> vram_view(
    vram_ragged_storage<node_id_type_> const& storage) noexcept {
    return {storage.values, storage.offsets,    storage.neighbors,   storage.connectivity,
            storage.kernel, storage.parameters, storage.sketch_bytes};
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
            build.listed,
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
            search.query_sketches,
            search.seed_ids,
            search.seeds_count,
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
            search.count,
            search.ranges,
            reinterpret_cast<vram_scored_node<node_id_type_>*>(search.range_matches),
            search.finished_ranges};
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

/** The body of every NVIDIA basic kernel: @p kernel_ picks the walk, @p kit_ scores on its
 *  @p rung_, @p slots_ is the register rung, @p walkers_ the walkers a block hosts, and
 *  @p node_id_type_ the IDs of the records. Each modality's @c cuda_kernel names the kit its
 *  ladder picks and calls it under its launch bounds. */
template <vram_kernel_t kernel_, a_vram_basic_kit kit_, u32_t rung_, u32_t slots_, u32_t walkers_,
          typename node_id_type_>
__device__ __forceinline__ void cuda_kernel_body(vram_arguments<node_id_type_> const& arguments) {
    vram_build<node_id_type_> const& build = arguments.build;
    // Kits without a load hook store values as saved; theirs never launch.
    constexpr bool loads_values = requires { kit_::load_values(arguments.load); };
    if constexpr (kernel_ == vram_kernel_t::insert_k || kernel_ == vram_kernel_t::refine_k)
        cuda_insert<kit_, rung_, slots_, kernel_, walkers_>(vram_view(build));
    else if constexpr (kernel_ == vram_kernel_t::back_link_k) cuda_back_link<kit_, rung_, walkers_>(vram_view(build));
    else if constexpr (kernel_ == vram_kernel_t::gather_neighborhoods_k)
        cuda_gather_neighborhoods<kit_, rung_, walkers_>(vram_view(arguments.round));
    else if constexpr (kernel_ == vram_kernel_t::restitch_k)
        cuda_restitch<kit_, rung_, walkers_>(vram_view(arguments.round));
    else if constexpr (kernel_ == vram_kernel_t::load_values_k && a_vram_kit_closing_values<kit_>)
        cuda_load_closed<kit_>(arguments.load);
    else if constexpr (kernel_ == vram_kernel_t::load_values_k && loads_values) kit_::load_values(arguments.load);
    else if constexpr (kernel_ == vram_kernel_t::sketch_values_k && a_vram_sketching_kit<kit_>)
        cuda_sketch_values<kit_>(arguments.sketch);
    else if constexpr (kernel_ == vram_kernel_t::load_values_k || kernel_ == vram_kernel_t::sketch_values_k) return;
    else if constexpr (kernel_ == vram_kernel_t::seed_k) {
        static_assert(a_vram_sketching_kit<kit_>, "Only a kit building by sketches seeds its searches");
        cuda_walk<kit_, kernel_, rung_, slots_, walkers_>(vram_view(arguments.search));
    }
    else cuda_walk<kit_, kernel_, rung_, slots_, walkers_>(vram_view(arguments.search));
}

} // namespace unum::usearch::basic
