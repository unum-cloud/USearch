/**
 *  @file include/usearch/basic/vram_index.cuh
 *  @author Ash Vardanian
 *  @date October 15, 2026
 *  @brief The device bodies every basic VRAM index compiled by NVCC or HIP-Clang shares: the ragged
 *      view, the greedy walk, the diversity selection, insertion, re-selection and the back-link
 *      merge.
 *
 *  Every body is templated on a basic kit, which names its vendor's warp policy as @c kit_::warp_t,
 *  and walks over the dense design's frontiers, visited set, slot-0 neighbor protocol, selection
 *  spans and telemetry. A basic kit prepares a stored row as a query once per walk, into a span of
 *  the walking warp's scratch, and scores whole rows, which a schedule hands it: a lane each in
 *  place, or the vendor's wider split, so the bodies only name a row per lane and read back its
 *  distance.
 *
 *  Insertion publishes forward rows alone. A back-link re-selection costs a full diversity pass of
 *  distances, far past any lock budget for the costly kits, so the host scatters each admission
 *  wave's picks into reverse buckets and merges them afterwards, and no device lock is held across
 *  a distance. A wave's rows walk the graph again once that merge lets them reach one another, and
 *  re-select their rows. The host classes, the kernels and the launch bounds stay in each vendor's
 *  own @c vram_<vendor>_index file, beside the kits in each modality's @c vram_<vendor>_metric.
 */
#pragma once

#include <concepts>    // `std::same_as` - the kit concept's verbs
#include <limits>      // `std::numeric_limits` - an unbounded radius
#include <type_traits> // `std::is_convertible_v` - the kit's lanes per row

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/vram_queue.cuh"

namespace unum::usearch::basic {
using namespace unum::usearch::utilities;

/** One stored value as a kit reads it: its first byte and its length. */
struct vram_row_t {
    byte_t const* data;
    u32_t bytes;
};

/** What a basic VRAM kit binds: one modality's distance on one GPU generation, with the warp policy
 *  its walks shuffle through. @c prepare lays a row out as a query in a span, every lane of the
 *  warp taking part; @c bound answers a lower bound from the two rows' headers alone; @c score runs
 *  on @c lanes_per_row_k lanes per candidate, every one of them returning the distance, exact
 *  within @p radius and above it past. */
template <typename kit_>
concept vram_basic_kit = requires(typename kit_::query_t const& query, vram_row_t row, byte_t* span, distance_t radius,
                                  u32_t lane) {
    typename kit_::warp_t;
    typename kit_::query_t;
    requires std::is_convertible_v<decltype(kit_::lanes_per_row_k), u32_t>;
    { kit_::prepare(row, span, lane) } -> std::same_as<typename kit_::query_t>;
    { kit_::bound(query, row) } -> std::same_as<distance_t>;
    { kit_::score(query, row, radius, lane) } -> std::same_as<distance_t>;
};

/** Typed view over a basic index's device storage: the values tape, its row offsets, and the
 *  fixed-capacity neighbors matrix. */
struct vram_ragged_view {

    /** Every value back to back, from a word-aligned base, with a word of margin past the last, so
     *  word loads never fault. */
    byte_t const* values;

    /** Row starts into @c values, @b [rows+1]; an inserting warp lands its own row's two before
     *  publishing it. */
    u64_t* offsets;

    /** Neighbors matrix, @b [rows,connectivity] IDs; slot 0 doubles as the lock word. */
    u32_t* neighbors;

    /** Neighbor slots per node: the stride of @c neighbors and the connectivity. */
    u32_t connectivity;

    __device__ __forceinline__ vram_row_t row(node_id_t node_id) const noexcept {
        u64_t const first = offsets[node_id];
        return {values + first, static_cast<u32_t>(offsets[node_id + 1] - first)};
    }
    __device__ __forceinline__ u32_t* neighbors_for(node_id_t node_id) const noexcept {
        return vram_row(neighbors, node_id, connectivity);
    }
};

/** Geometry & storage one insertion wave operates on, and the reverse buckets the wave's back-link
 *  passes scatter into and merge from. Every storage row an assignable ID can touch is mapped and
 *  sentinel-filled ahead of the launch. */
struct vram_build_view {
    vram_ragged_view storage;

    /** The wave's row starts as the caller spelled them, @b [rows+1] from any origin. */
    u64_t const* staged_offsets;

    /** Where the wave's first byte landed in the values tape. */
    u64_t tape_first;

    /** IDs strictly below this bound may be claimed & inserted by this launch. */
    u32_t end_id;

    /** Bytes of the walking warp's dynamic-shared slice: the query span and the pivot span; zero
     *  when the arena shape carries them instead. */
    u32_t scratch_bytes_per_warp;

    /** Bytes of one span, a multiple of 16. */
    u32_t span_bytes;

    /** The launch's base ID: staged row 0 lands there, and the warps assign themselves offsets
     *  within @b [first_id,end_id). */
    node_id_t first_id;

    /** The traversal arena every warp carves its slice from, and the shape of one slice. */
    byte_t* arena;
    vram_arena_shape_t arena_shape;

    /** The arena frontier's expansion budget on the rung past the register ladder. */
    u32_t expansion;

    /** The context's epoch floor at launch. */
    u32_t epoch_base;
    vram_telemetry_block_t* telemetry;

    /** Rows of the launch warps have assigned themselves so far, zeroed per launch. */
    u32_t* assigned_rows;

    /** Reverse buckets, @b [rows,bucket_width] IDs, their fill counters, and the bitset of rows the
     *  merge re-selects. */
    u32_t* bucket_ids;
    u32_t* bucket_counts;
    u32_t* touched;
    u32_t bucket_width;
};

/** Inputs & outputs of one search launch: a tape of queries in, the nearest IDs and their distances
 *  out, both ascending per query. */
struct vram_search_view {
    vram_ragged_view storage;

    /** The queries' bytes, word-aligned, with a word of margin past the last. */
    byte_t const* queries;

    /** Query starts as the caller spelled them, @b [queries+1] from any origin. */
    u64_t const* query_offsets;
    u32_t queries_count;

    /** The watermark snapshotted at launch: landmarks are drawn below it and no neighbor at or past
     *  it is followed. */
    u32_t sampling_bound;

    /** Bytes of the walking warp's dynamic-shared slice, the query span. */
    u32_t scratch_bytes_per_warp;

    /** Requested expansion; past the register ladder it is the wide walk's budget. */
    u32_t expansion;

    /** The traversal arena every warp carves its slice from, and the shape of one slice. */
    byte_t* arena;
    vram_arena_shape_t arena_shape;

    /** The context's epoch floor at launch. */
    u32_t epoch_base;
    vram_telemetry_block_t* telemetry;

    /** Queries of the batch warps have assigned themselves so far, zeroed per launch. */
    u32_t* assigned_queries;

    /** The landing for the matched IDs and distances, @b [queries,count] each. */
    node_id_t* result_ids;
    distance_t* result_distances;
    u32_t count;
};

/** Scores each lane's row in place, the schedule of every kit whose lane scores a row alone, so a
 *  warp walks one query by itself. */
template <vram_basic_kit kit_>
struct vram_lane_schedule {

    /** Nothing beside the walking warp's own spans. */
    struct shared_t {};

    __device__ __forceinline__ explicit vram_lane_schedule(shared_t&) noexcept {}

    /** The distance from @p query to the row this lane names, exact within @p radius; infinite for
     *  the sentinel. Every lane of the warp calls it. */
    __device__ __forceinline__ distance_t score(vram_ragged_view const& storage, typename kit_::query_t const& query,
                                                node_id_t node_id, distance_t radius, u32_t lane) const noexcept {
        if (node_id == node_id_unknown_k) return std::numeric_limits<distance_t>::infinity();
        vram_row_t const row = storage.row(node_id);
        distance_t const bound = kit_::bound(query, row);
        return bound > radius ? bound : kit_::score(query, row, radius, lane);
    }
};

/** Samples one lane-wide batch of distinct, published landmarks in
 *  @b [sampling_first,sampling_bound) into @p candidate_ids, drawn as the dense walk draws them,
 *  and answers how many it staged: zero once every published row of the range is visited, or the
 *  visited set saturated. */
template <vram_basic_kit kit_>
__device__ __forceinline__ u32_t vram_sample_landmarks( //
    vram_ragged_view const& storage, node_id_t sampling_first, node_id_t sampling_bound,
    vram_visited_set<typename kit_::warp_t>& visited, splitmix64_t& random_generator, u32_t* candidate_ids,
    u32_t lane) noexcept {

    using warp_t = typename kit_::warp_t;
    // Rows still inserting gather at the range's top, which a redraw leaves far sooner than a lap.
    constexpr u64_t redraws_k = 16;
    node_id_t const span = sampling_bound - sampling_first;
    uniform_below<node_id_t> const uniform {span};
    node_id_t draw = sampling_first + uniform(random_generator);
    bool fresh = false;
    u32_t const saturations = visited.saturations_;
    u64_t const redraws = min_of<u64_t>(span, redraws_k);
    for (u64_t probes = 0; probes != redraws + span && !warp_t::all(fresh) && visited.saturations_ == saturations;
         ++probes) {
        bool published = false;
        if (!fresh) {
            vram_neighbors_ref<warp_t> const row {storage.neighbors_for(draw), storage.connectivity};
            published = row.peek_first() != node_id_unknown_k;
        }
        bool const claimed = visited.test_and_set(published ? draw : node_id_unknown_k, lane);
        if (!fresh && !claimed)
            draw = !published && probes < redraws ? sampling_first + uniform(random_generator)
                   : draw + 1 == sampling_bound   ? sampling_first
                                                  : draw + 1;
        fresh = fresh || claimed;
    }
    typename warp_t::mask_t const fresh_mask = warp_t::ballot(fresh);
    u32_t const ordinal = warp_t::count(fresh_mask & warp_t::below(lane));
    if (fresh) candidate_ids[ordinal] = draw;
    warp_t::sync();
    return warp_t::count(fresh_mask);
}

/** Scores one gathered batch against @p query through @p schedule and folds it into the pool. Past
 *  a full pool's worst rank nothing enters, so each row is scored within it; an unfilled pool
 *  bounds nothing. No dedup: the exact visited set upstream stages every ID once. */
template <vram_basic_kit kit_, typename schedule_, typename frontier_>
__device__ __forceinline__ void vram_score_into_frontier( //
    vram_ragged_view const& storage, schedule_& schedule, typename kit_::query_t const& query,
    u32_t const* gathered_ids, u32_t gathered_count, vram_warp_context<frontier_>& context, u32_t lane) noexcept {
    context.telemetry.scored += gathered_count;
    vram_scored_node_t const worst = context.frontier.worst();
    distance_t const radius = worst == vram_scored_node_t::sentinel() ? std::numeric_limits<distance_t>::infinity()
                                                                      : worst.distance();
    node_id_t const node_id = lane < gathered_count ? gathered_ids[lane] : node_id_unknown_k;
    distance_t const distance = schedule.score(storage, query, node_id, radius, lane);
    context.frontier.push_batch(
        lane < gathered_count ? vram_scored_node_t::make(distance, node_id) : vram_scored_node_t::sentinel(), lane);
}

/** Continues a greedy best-first walk toward @p query over whatever the frontier holds, as the
 *  dense walk does: landmarks from @b [sampling_first,sampling_bound), a transiently locked
 *  candidate held through one other expansion and dropped if still locked then, and no neighbor at
 *  or past @p first_invisible_id followed. Read-only on the graph. */
template <vram_basic_kit kit_, typename schedule_, typename frontier_>
__device__ __forceinline__ void vram_search_ahead( //
    vram_ragged_view const& storage, schedule_& schedule, typename kit_::query_t const& query, node_id_t sampling_first,
    node_id_t sampling_bound, node_id_t first_invisible_id, vram_visited_set<typename kit_::warp_t>& visited,
    u32_t* gathered_ids, vram_warp_context<frontier_>& context, u32_t lane) noexcept {

    using warp_t = typename kit_::warp_t;
    frontier_& frontier = context.frontier;
    vram_telemetry_t& telemetry = context.telemetry;
    u32_t hops_cap = telemetry.hops + frontier.hops_budget();
    vram_scored_node_t held = vram_scored_node_t::sentinel();
    u32_t held_at_hop = 0;
    bool seeding = true;
    while (true) {
        if (seeding) {
            u32_t const gathered_count = vram_sample_landmarks<kit_>( //
                storage, sampling_first, sampling_bound, visited, context.random_generator, gathered_ids, lane);
            if (!gathered_count) break; // Every published row of the range is already visited.
            vram_score_into_frontier<kit_>(storage, schedule, query, gathered_ids, gathered_count, context, lane);
            seeding = false;
            continue;
        }
        bool const revisiting = !(held == vram_scored_node_t::sentinel()) && telemetry.hops != held_at_hop;
        vram_scored_node_t const candidate = revisiting ? held : frontier.pop_nearest(lane);
        if (revisiting) held = vram_scored_node_t::sentinel();
        if (candidate == vram_scored_node_t::sentinel()) {
            if (!(held == vram_scored_node_t::sentinel())) {
                held_at_hop = ~0u;
                continue;
            }
            // A drained frontier before an exhausted walk re-seeds at one hop of the budget.
            if (frontier.exhausted() || telemetry.hops >= hops_cap) break;
            seeding = true;
            ++telemetry.restarts, --hops_cap;
            continue;
        }
        if (frontier.beyond_radius(candidate)) break;
        if (telemetry.hops >= hops_cap) break;
        ++telemetry.hops;

        // Only the leader reads the racing lock word, so every lane agrees on one unlock.
        vram_neighbors_ref<warp_t> const row {storage.neighbors_for(candidate.id()), storage.connectivity};
        node_id_t first_value = lane == 0 ? row.peek_first() : node_id_unknown_k;
        first_value = warp_t::shuffle(first_value, 0);
        warp_t::sync();
        if (first_value == node_id_unknown_k) {
            if (revisiting || !(held == vram_scored_node_t::sentinel())) ++telemetry.locked_drops;
            else held = candidate, held_at_hop = telemetry.hops;
            continue;
        }
        // A warp-width of slots at a time; the bound is warp-uniform for the collectives.
        for (u32_t slot_base = 0; slot_base < storage.connectivity; slot_base += warp_t::width_k) {
            u32_t const slot = slot_base + lane;
            node_id_t const neighbor_id = slot == 0                     ? first_value
                                          : slot < storage.connectivity ? row.load_slot(slot)
                                                                        : node_id_unknown_k;
            // The bound never exceeds the ID space, so it rejects the sentinel too.
            bool const usable = neighbor_id < first_invisible_id && neighbor_id != candidate.id();
            bool const fresh = visited.test_and_set(usable ? neighbor_id : node_id_unknown_k, lane);
            typename warp_t::mask_t const fresh_mask = warp_t::ballot(fresh);
            u32_t const ordinal = warp_t::count(fresh_mask & warp_t::below(lane));
            if (fresh) gathered_ids[ordinal] = neighbor_id;
            warp_t::sync();
            u32_t const gathered_count = warp_t::count(fresh_mask);
            if (gathered_count)
                vram_score_into_frontier<kit_>(storage, schedule, query, gathered_ids, gathered_count, context, lane);
            warp_t::sync(); // The next slots' staging overwrites the IDs every lane just scored.
        }
    }
    telemetry.locked_drops += !(held == vram_scored_node_t::sentinel());
}

/** Greedy best-first walk from random published landmarks toward @p query: a fresh frontier handed
 *  to @ref vram_search_ahead. */
template <vram_basic_kit kit_, typename schedule_, typename frontier_>
__device__ __forceinline__ void vram_search_around( //
    vram_ragged_view const& storage, schedule_& schedule, typename kit_::query_t const& query, node_id_t sampling_bound,
    node_id_t first_invisible_id, vram_visited_set<typename kit_::warp_t>& visited, u32_t* gathered_ids,
    vram_warp_context<frontier_>& context, u32_t lane) noexcept {
    context.frontier.reset(lane);
    vram_search_ahead<kit_>(storage, schedule, query, 0, sampling_bound, first_invisible_id, visited, gathered_ids,
                            context, lane);
}

/**
 *  @brief The diversity rule of the CPU's @c keep_diverse_t: a candidate is kept only when it is
 *      closer to the node than to every kept neighbor, and the slots the rule leaves empty take the
 *      nearest rejected candidates, after the diverse picks.
 *
 *  A selector keeps rows of at most @c connectivity neighbors from candidates ascending by distance
 *  to the node, whose prepared @p query it may read; it writes the kept candidates' positions into
 *  @p kept_positions and answers their count, every lane of the walking warp taking part.
 */
struct vram_keep_diverse_t {

    /** Prepares each candidate as the pivot in @p pivot_span, which scores the kept ones through
     *  @p schedule, exiting each candidate on its first violation. */
    template <vram_basic_kit kit_, typename schedule_>
    static __device__ __forceinline__ u32_t select( //
        vram_ragged_view const& storage, schedule_& schedule, typename kit_::query_t const&, byte_t* pivot_span,
        u32_t const* candidate_ids, f32_t const* candidate_distances, u32_t candidates_count, u32_t* kept_positions,
        vram_telemetry_t& telemetry, u32_t lane) noexcept {

        using warp_t = typename kit_::warp_t;
        u32_t kept_count = 0;
#pragma unroll 1
        for (u32_t candidate_index = 0; candidate_index < candidates_count && kept_count < storage.connectivity;
             ++candidate_index) {
            f32_t const to_query = candidate_distances[candidate_index];
            typename kit_::query_t const pivot = kept_count ? kit_::prepare(storage.row(candidate_ids[candidate_index]),
                                                                            pivot_span, lane)
                                                            : typename kit_::query_t {};
            bool diverse = true;
            for (u32_t base = 0; base < kept_count && diverse; base += warp_t::width_k) { // Warp-uniform bounds.
                node_id_t const kept_id = base + lane < kept_count ? candidate_ids[kept_positions[base + lane]]
                                                                   : node_id_unknown_k;
                diverse = warp_t::all(!(schedule.score(storage, pivot, kept_id, to_query, lane) < to_query));
                telemetry.scored += min_of(kept_count - base, warp_t::width_k);
            }
            if (diverse) {
                if (lane == 0) kept_positions[kept_count] = candidate_index;
                ++kept_count;
                warp_t::sync();
            }
        }
        // The kept prefix is a subsequence of the candidates, so one cursor tells the pruned apart.
        u32_t const diverse_count = kept_count;
        for (u32_t candidate_index = 0, matched = 0;
             candidate_index < candidates_count && kept_count < storage.connectivity; ++candidate_index) {
            if (matched != diverse_count && kept_positions[matched] == candidate_index) {
                ++matched;
                continue;
            }
            if (lane == 0) kept_positions[kept_count] = candidate_index;
            ++kept_count;
            warp_t::sync();
        }
        warp_t::sync();
        return kept_count;
    }
};

/** Hands the ascending pool prefix of a finished walk toward @p query to @p selector_, staged as
 *  candidate IDs and distances, and rewrites the kept positions as node IDs in
 *  @p selection.selected_ids. @return The kept count. */
template <vram_basic_kit kit_, typename selector_, typename schedule_, typename frontier_>
__device__ __forceinline__ u32_t vram_select_neighbors( //
    vram_ragged_view const& storage, schedule_& schedule, typename kit_::query_t const& query, byte_t* pivot_span,
    u32_t* candidate_ids, vram_selection_scratch_t selection, vram_warp_context<frontier_>& context,
    u32_t lane) noexcept {

    using warp_t = typename kit_::warp_t;
    frontier_& frontier = context.frontier;
    frontier.settle(lane);

    u32_t candidates_count = 0;
#pragma unroll
    for (u32_t base = 0; base < vram_cache_rows_k; base += warp_t::width_k) {
        vram_scored_node_t const entry = frontier.rank_entry(base + lane);
        typename warp_t::mask_t const live = warp_t::ballot(!(entry == vram_scored_node_t::sentinel()));
        if (!(entry == vram_scored_node_t::sentinel())) {
            candidate_ids[base + lane] = entry.id();
            selection.candidate_distances[base + lane] = entry.distance();
        }
        candidates_count += warp_t::count(live);
        if (live != warp_t::all_k) break;
    }
    warp_t::sync();

    u32_t const selected_count = selector_::template select<kit_>(storage, schedule, query, pivot_span, candidate_ids,
                                                                  selection.candidate_distances, candidates_count,
                                                                  selection.selected_ids, context.telemetry, lane);
    // The caller links to nodes, not to positions in the pool.
    for (u32_t slot = lane; slot < selected_count; slot += warp_t::width_k)
        selection.selected_ids[slot] = candidate_ids[selection.selected_ids[slot]];
    warp_t::sync();
    return selected_count;
}

/** Inserts the node identified by @p new_node_id, whose bytes already sit in the values tape: lands
 *  its two row offsets, walks the published graph, selects diverse neighbors, and publishes with
 *  one release store, one warp per node. Reverse links land after the wave, from the back-link
 *  passes. */
template <vram_basic_kit kit_, typename selector_ = vram_keep_diverse_t, typename schedule_, typename frontier_>
__device__ __forceinline__ void vram_insert_one( //
    vram_build_view const& view, schedule_& schedule, node_id_t new_node_id,
    vram_visited_set<typename kit_::warp_t>& visited, byte_t* query_span, byte_t* pivot_span, u32_t* gathered_ids,
    vram_selection_scratch_t selection, vram_warp_context<frontier_>& context, u32_t lane) noexcept {

    using warp_t = typename kit_::warp_t;
    vram_ragged_view const& storage = view.storage;

    // Both bounds land, so readers past publication see them whichever neighbor lands first.
    u32_t const offset = new_node_id - view.first_id;
    u64_t const origin = view.staged_offsets[0];
    u64_t const row_begin = view.tape_first + view.staged_offsets[offset] - origin;
    u64_t const row_end = view.tape_first + view.staged_offsets[offset + 1] - origin;
    if (lane == 0) {
        warp_t::store_relaxed(storage.offsets[new_node_id], row_begin);
        warp_t::store_relaxed(storage.offsets[new_node_id + 1], row_end);
    }
    typename kit_::query_t const query = kit_::prepare(
        {storage.values + row_begin, static_cast<u32_t>(row_end - row_begin)}, query_span, lane);

    // A virgin index publishes an empty node instead of walking.
    if (new_node_id != 0)
        vram_search_around<kit_>(storage, schedule, query, new_node_id, view.end_id, visited, gathered_ids, context,
                                 lane);
    else context.frontier.reset(lane);

    // No lock on the own row: no reader arrives before the slot-0 release flips the sentinel.
    u32_t const selected_count = vram_select_neighbors<kit_, selector_>( //
        storage, schedule, query, pivot_span, gathered_ids, selection, context, lane);
    vram_neighbors_ref<warp_t> const own_neighbors {storage.neighbors_for(new_node_id), storage.connectivity};
    for (u32_t slot = lane; slot < selected_count; slot += warp_t::width_k)
        if (slot > 0) own_neighbors.store_slot(slot, selection.selected_ids[slot]);
    warp_t::sync();
    node_id_t const first_link = selected_count ? selection.selected_ids[0] : new_node_id;
    if (lane == 0) own_neighbors.unlock(first_link); // A self-loop encodes published-empty.
}

/** Re-selects the published row of @p node_id once its wave's merge made the wave's rows reachable
 *  from one another: walks from the row's own neighbors over every published row, selects again,
 *  and rewrites the row in place, tail slots relaxed and slot 0 last with a release, so walks on
 *  other streams read either row whole at every slot. */
template <vram_basic_kit kit_, typename selector_ = vram_keep_diverse_t, typename schedule_, typename frontier_>
__device__ __forceinline__ void vram_reselect_one( //
    vram_build_view const& view, schedule_& schedule, node_id_t node_id,
    vram_visited_set<typename kit_::warp_t>& visited, byte_t* query_span, byte_t* pivot_span, u32_t* gathered_ids,
    vram_selection_scratch_t selection, vram_warp_context<frontier_>& context, u32_t lane) noexcept {

    using warp_t = typename kit_::warp_t;
    vram_ragged_view const& storage = view.storage;
    typename kit_::query_t const query = kit_::prepare(storage.row(node_id), query_span, lane);
    context.frontier.reset(lane);
    visited.test_and_set(node_id, lane);

    // The walk starts from the row's own neighbors, which the wave's first walk already found.
    vram_neighbors_ref<warp_t> const own_neighbors {storage.neighbors_for(node_id), storage.connectivity};
    for (u32_t slot_base = 0; slot_base < storage.connectivity; slot_base += warp_t::width_k) {
        u32_t const slot = slot_base + lane;
        node_id_t const seed_id = slot < storage.connectivity ? own_neighbors.load_slot(slot) : node_id_unknown_k;
        bool const usable = seed_id < view.end_id && seed_id != node_id;
        bool const fresh = visited.test_and_set(usable ? seed_id : node_id_unknown_k, lane);
        typename warp_t::mask_t const fresh_mask = warp_t::ballot(fresh);
        u32_t const ordinal = warp_t::count(fresh_mask & warp_t::below(lane));
        if (fresh) gathered_ids[ordinal] = seed_id;
        warp_t::sync();
        u32_t const gathered_count = warp_t::count(fresh_mask);
        if (gathered_count)
            vram_score_into_frontier<kit_>(storage, schedule, query, gathered_ids, gathered_count, context, lane);
        warp_t::sync();
    }
    vram_search_ahead<kit_>(storage, schedule, query, 0, view.end_id, view.end_id, visited, gathered_ids, context,
                            lane);

    u32_t const selected_count = vram_select_neighbors<kit_, selector_>( //
        storage, schedule, query, pivot_span, gathered_ids, selection, context, lane);
    for (u32_t slot = lane; slot < storage.connectivity; slot += warp_t::width_k)
        if (slot)
            own_neighbors.store_slot(slot, slot < selected_count ? selection.selected_ids[slot] : node_id_unknown_k);
    warp_t::sync();
    if (lane == 0) own_neighbors.unlock(selected_count ? selection.selected_ids[0] : node_id);
}

/**
 *  @brief Merges one node's reverse bucket into its live neighbors: appended while the row has
 *      room, re-selected over the nearest of the union otherwise.
 *
 *  The append is the dense back-linker's fast path; the re-selection hands @p selector_ the union
 *  ascending by distance to the node. The bucket may hold more IDs than one selection weighs, so
 *  every one is scored and the sorted span keeps the nearest, each later one scored within the
 *  farthest kept. The pass owns the graph against every other writer, so it reads the row plainly,
 *  while walks on other streams may read it throughout: every slot changes by one relaxed store and
 *  slot 0 last, with a release.
 */
template <vram_basic_kit kit_, typename selector_ = vram_keep_diverse_t, typename schedule_>
__device__ __forceinline__ void vram_merge_back_links_one( //
    vram_build_view const& view, schedule_& schedule, node_id_t node_id, byte_t* query_span, byte_t* pivot_span,
    u32_t* candidate_ids, vram_selection_scratch_t selection, vram_telemetry_t& telemetry, u32_t lane) noexcept {

    using warp_t = typename kit_::warp_t;
    vram_ragged_view const& storage = view.storage;
    u32_t const connectivity = storage.connectivity;
    vram_neighbors_ref<warp_t> const live {storage.neighbors_for(node_id), connectivity};

    u32_t forward_count = 0;
    for (u32_t slot_base = 0; slot_base < connectivity; slot_base += warp_t::width_k) {
        u32_t const slot = slot_base + lane;
        node_id_t const forward_id = slot < connectivity ? live.load_slot(slot) : node_id_unknown_k;
        bool const usable = forward_id != node_id_unknown_k && forward_id != node_id;
        typename warp_t::mask_t const usable_mask = warp_t::ballot(usable);
        u32_t const ordinal = forward_count + warp_t::count(usable_mask & warp_t::below(lane));
        if (usable && ordinal < vram_cache_rows_k) candidate_ids[ordinal] = forward_id;
        forward_count = min_of(forward_count + warp_t::count(usable_mask), vram_cache_rows_k);
        warp_t::sync();
    }

    // The bucket's IDs the picks lack, a warp-width of slots at a time; mutual picks land in both.
    u32_t const landed = min_of(view.bucket_counts[node_id], view.bucket_width);
    u32_t const* const reverse_slots = vram_row(view.bucket_ids, node_id, view.bucket_width);
    auto const fresh_at = [&](u32_t slot) noexcept {
        node_id_t const reverse_id = slot < landed ? reverse_slots[slot] : node_id_unknown_k;
        bool usable = reverse_id != node_id_unknown_k && reverse_id != node_id;
        for (u32_t forward_index = 0; usable && forward_index != forward_count; ++forward_index)
            usable = candidate_ids[forward_index] != reverse_id;
        return usable ? reverse_id : node_id_unknown_k;
    };
    u32_t fresh_count = 0;
    for (u32_t slot_base = 0; slot_base < landed; slot_base += warp_t::width_k)
        fresh_count += warp_t::count(warp_t::ballot(fresh_at(slot_base + lane) != node_id_unknown_k));
    if (!fresh_count) return;

    if (forward_count + fresh_count <= connectivity) {
        // Room for every reverse link: appended in place, no distance needed, slot 0 last.
        node_id_t first_link = forward_count ? node_id_unknown_k : node_id;
        for (u32_t slot_base = 0, appended = forward_count; slot_base < landed; slot_base += warp_t::width_k) {
            node_id_t const reverse_id = fresh_at(slot_base + lane);
            typename warp_t::mask_t const fresh_mask = warp_t::ballot(reverse_id != node_id_unknown_k);
            u32_t const slot = appended + warp_t::count(fresh_mask & warp_t::below(lane));
            if (reverse_id != node_id_unknown_k && slot) live.store_slot(slot, reverse_id);
            if (!appended && fresh_mask) first_link = warp_t::shuffle(reverse_id, warp_t::first(fresh_mask));
            appended += warp_t::count(fresh_mask);
        }
        warp_t::sync();
        if (lane == 0 && !forward_count) live.unlock(first_link);
        return;
    }

    typename kit_::query_t const query = kit_::prepare(storage.row(node_id), query_span, lane);
    vram_sorted_buffer<warp_t> sorted_candidates {selection.scored_entries, 0, vram_cache_rows_k};
    for (u32_t slot_base = 0; slot_base < forward_count + landed; slot_base += warp_t::width_k) {
        u32_t const slot = slot_base + lane;
        node_id_t const candidate_id = slot < forward_count ? candidate_ids[slot] : fresh_at(slot - forward_count);
        typename warp_t::mask_t const live_mask = warp_t::ballot(candidate_id != node_id_unknown_k);
        if (!live_mask) continue;
        distance_t const radius = sorted_candidates.size_ == vram_cache_rows_k
                                      ? sorted_candidates.at(vram_cache_rows_k - 1).distance()
                                      : std::numeric_limits<distance_t>::infinity();
        distance_t const distance = schedule.score(storage, query, candidate_id, radius, lane);
        vram_scored_node_t const scored = candidate_id != node_id_unknown_k
                                              ? vram_scored_node_t::make(distance, candidate_id)
                                              : vram_scored_node_t::sentinel();
        sorted_candidates.insert_batch(sort_lanes<warp_t>(scored, lane), warp_t::count(live_mask), lane);
    }
    telemetry.scored += forward_count + fresh_count;

    // The picks are scored, so their IDs make way for the union's, ascending.
    for (u32_t position = lane; position < sorted_candidates.size_; position += warp_t::width_k) {
        candidate_ids[position] = sorted_candidates.at(position).id();
        selection.candidate_distances[position] = sorted_candidates.at(position).distance();
    }
    warp_t::sync();
    u32_t* const kept_positions = selection.kept_positions;
    u32_t const refined_count = selector_::template select<kit_>(storage, schedule, query, pivot_span, candidate_ids,
                                                                 selection.candidate_distances, sorted_candidates.size_,
                                                                 kept_positions, telemetry, lane);
    for (u32_t slot = lane; slot < connectivity; slot += warp_t::width_k)
        if (slot) live.store_slot(slot, slot < refined_count ? candidate_ids[kept_positions[slot]] : node_id_unknown_k);
    warp_t::sync();
    if (lane == 0) live.unlock(candidate_ids[kept_positions[0]]);
    ++telemetry.changed_nodes;
}

/** One query's full walk: prepares the query into its span, walks the graph, and lands its nearest
 *  @c count IDs and distances, ascending, sentinel IDs past the live prefix. Safe against
 *  concurrent insertions: the walk gates on every slot-0 word and never writes the graph. */
template <vram_basic_kit kit_, typename schedule_, typename frontier_>
__device__ __forceinline__ void vram_walk_one( //
    vram_search_view const& view, schedule_& schedule, u32_t query_index, byte_t* query_span,
    vram_visited_set<typename kit_::warp_t>& visited, frontier_ frontier, u32_t* gathered_ids, u32_t lane) noexcept {

    using warp_t = typename kit_::warp_t;
    u64_t const origin = view.query_offsets[0];
    u64_t const first = view.query_offsets[query_index] - origin;
    vram_row_t const row {view.queries + first,
                          static_cast<u32_t>(view.query_offsets[query_index + 1] - origin - first)};
    typename kit_::query_t const query = kit_::prepare(row, query_span, lane);

    vram_warp_context<frontier_> context;
    context.frontier = frontier;
    context.random_generator = vram_lane_generator<warp_t>(query_index, lane);
    vram_search_around<kit_>(view.storage, schedule, query, view.sampling_bound, view.sampling_bound, visited,
                             gathered_ids, context, lane);

    // The rank extraction shuffles, so every lane participates and only in-range ones store.
    node_id_t* const ids_row = vram_row(view.result_ids, query_index, view.count);
    distance_t* const distances_row = vram_row(view.result_distances, query_index, view.count);
    context.frontier.for_each_rank(view.count, lane, [&](u32_t rank, vram_scored_node_t entry) noexcept {
        if (rank >= view.count) return;
        ids_row[rank] = entry.id();
        distances_row[rank] = entry.distance();
    });
    context.telemetry.tail_drops = context.frontier.dropped();
    context.telemetry.visited_saturations = visited.saturations_;
    context.telemetry.template flush<warp_t, vram_traversal_t::searching_k>(view.telemetry, lane);
}

} // namespace unum::usearch::basic
