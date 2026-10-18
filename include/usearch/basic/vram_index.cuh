/**
 *  @file include/usearch/basic/vram_index.cuh
 *  @author Ash Vardanian
 *  @date October 15, 2026
 *  @brief The device bodies every basic VRAM index compiled by NVCC or HIP-Clang shares: the ragged
 *      view, the greedy walk, the diversity selection, insertion, re-selection and the back-link
 *      merge.
 *
 *  Every body is templated on a basic kit, which names its vendor's warp policy as @c kit_::warp_t,
 *  and on the node-ID type its views carry, and walks over the dense design's frontiers, visited
 *  set, slot-0 neighbor protocol, selection spans and telemetry. A basic kit prepares a stored row
 *  as a query once per walk, into a span of the walking warp's scratch, and scores whole rows,
 *  which a schedule hands it: a lane each in place, or the vendor's wider split, so the bodies only
 *  name a row per lane and read back its distance. A schedule scores on one rung of its kit, the
 *  Myers words of a Levenshtein kit's queries, or zero for a kit scoring every row alike, and the
 *  view carries the kernel kind a kit covering several metrics picks among, as dense views do.
 *
 *  Insertion publishes forward rows alone. A back-link re-selection costs a full diversity pass of
 *  distances, far past any lock budget for the costly kits, so each admission wave's walkers thread
 *  their picks onto per-node chains and a back-link launch behind them merges every chain into its
 *  node, and no device lock is held across a distance. A wave's rows walk the graph again once
 *  those back-links let them reach one another, and re-select their rows. The host classes, the
 *  kernels and the launch bounds stay in each vendor's own @c vram_<vendor>_index file, beside the
 *  kits in each modality's @c vram_<vendor>_metric.
 */
#pragma once

#include <concepts> // `std::same_as` - the kit concept's verbs
#include <limits>   // `std::numeric_limits` - an unbounded radius, the node-ID type's sentinel

#include <numkong/capabilities.h> // `nk_kernel_kind_t` - the views carry the kernel kind

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
 *  its walks shuffle through and the selector its rows keep neighbors by. @c prepare lays a row out
 *  as a query in a span, every lane of the warp taking part, from the row alone or, for a kit with
 *  runtime parameters, from the storage's @c parameters too, which @ref vram_prepare tells apart;
 *  @c bound answers a lower bound from the two rows' headers alone. Its @c score runs on its lanes
 *  per row, every one of them returning the distance, exact within the radius and above it past:
 *  @c lanes_per_row_k of them and a plain @c score, or, for a kit whose split follows a rung,
 *  @c lanes_per_row of the rung and a @c score templated on it, which @ref vram_lanes_per_row and
 *  @ref vram_score tell apart. */
template <typename kit_>
concept vram_basic_kit = requires(typename kit_::query_t const& query, vram_row_t row, byte_t* span,
                                  byte_t const* parameters, nk_kernel_kind_t kernel, u32_t lane) {
    typename kit_::warp_t;
    typename kit_::query_t;
    typename kit_::selector_t;
    requires requires {
        { kit_::prepare(row, span, lane) } -> std::same_as<typename kit_::query_t>;
    } || requires {
        { kit_::prepare(parameters, row, span, lane) } -> std::same_as<typename kit_::query_t>;
    };
    { kit_::bound(kernel, query, row) } -> std::same_as<distance_t>;
};

/** Lays @p row out as @p kit_ 's query in @p span, handing a kit with runtime parameters the
 *  ones at @p parameters; every lane of the warp calls it. */
template <vram_basic_kit kit_>
__device__ __forceinline__ typename kit_::query_t vram_prepare(byte_t const* parameters, vram_row_t row, byte_t* span,
                                                               u32_t lane) noexcept {
    if constexpr (requires { kit_::prepare(parameters, row, span, lane); })
        return kit_::prepare(parameters, row, span, lane);
    else return kit_::prepare(row, span, lane);
}

/** Lanes @p kit_ scores one row with on @p rung_: its @c lanes_per_row_k at rung zero, its
 *  @c lanes_per_row of the rung past it. */
template <vram_basic_kit kit_, u32_t rung_>
constexpr u32_t vram_lanes_per_row() noexcept {
    if constexpr (rung_ == 0) return kit_::lanes_per_row_k;
    else return kit_::lanes_per_row(rung_);
}

/** The distance @p kit_ scores on @p rung_ under @p kernel from @p query to @p row, exact within
 *  @p radius; every lane of the row's group calls it. */
template <vram_basic_kit kit_, u32_t rung_>
__device__ __forceinline__ distance_t vram_score(nk_kernel_kind_t kernel, typename kit_::query_t const& query,
                                                 vram_row_t row, distance_t radius, u32_t lane) noexcept {
    if constexpr (rung_ == 0) return kit_::score(kernel, query, row, radius, lane);
    else return kit_::template score<rung_>(kernel, query, row, radius, lane);
}

/** Typed view over a basic index's device storage: the values tape, its row offsets, and the
 *  fixed-capacity neighbors matrix of @p node_id_type_ IDs. */
template <typename node_id_type_>
struct vram_ragged_view {

    /** Every value back to back, from a word-aligned base, with a word of margin past the last, so
     *  word loads never fault. */
    byte_t const* values;

    /** Row starts into @c values, @b [rows+1]; an inserting warp lands its own row's two before
     *  publishing it. */
    u64_t* offsets;

    /** Neighbors matrix, @b [rows,connectivity] IDs; slot 0 doubles as the lock word. */
    node_id_type_* neighbors;

    /** Neighbor slots per node: the stride of @c neighbors and the connectivity. */
    u32_t connectivity;

    /** The metric a kit covering several picks among, unknown for a kit of one. */
    nk_kernel_kind_t kernel;

    /** The kit's runtime parameters, as its face laid them out; null for a kit taking none. */
    byte_t const* parameters;

    __device__ __forceinline__ vram_row_t row(node_id_type_ node_id) const noexcept {
        u64_t const first = offsets[node_id];
        return {values + first, static_cast<u32_t>(offsets[node_id + 1] - first)};
    }
    __device__ __forceinline__ node_id_type_* neighbors_for(node_id_type_ node_id) const noexcept {
        return vram_row(neighbors, node_id, connectivity);
    }
};

/** Geometry & storage one insertion wave operates on, and the chains its walkers thread their picks
 *  onto for the back-link launch behind them. Every storage row an assignable ID can touch is
 *  mapped and sentinel-filled ahead of the launch. */
template <typename node_id_type_>
struct vram_build_view {
    vram_ragged_view<node_id_type_> storage;

    /** The wave's row starts as the caller spelled them, @b [rows+1] from any origin. */
    u64_t const* staged_offsets;

    /** Where the wave's first byte landed in the values tape. */
    u64_t tape_first;

    /** IDs strictly below this bound may be claimed & inserted by this launch. */
    node_id_type_ end_id;

    /** The rows a re-selection walk samples landmarks below and follows neighbors below: the
     *  launch's own bound while inserting, every published row while refining. */
    node_id_type_ sampling_bound;

    /** Bytes of the walking warp's dynamic-shared slice: the query span and the pivot span; zero
     *  when the arena shape carries them instead. */
    u32_t scratch_bytes_per_warp;

    /** Bytes of one span, a multiple of 16. */
    u32_t span_bytes;

    /** The launch's base ID: staged row 0 lands there, and the warps assign themselves offsets
     *  within @b [first_id,end_id). */
    node_id_type_ first_id;

    /** The traversal arena every warp carves its slice from, and the shape of one slice. */
    byte_t* arena;
    vram_arena_shape_t arena_shape;

    /** The arena frontier's expansion budget on the rung past the register ladder. */
    u32_t expansion;

    /** Candidates the launch's walkers select from past their two spans, as the host sized them: a
     *  walk's whole pool, or a back-link round's live row and a row's worth of newcomers. */
    u32_t selection_rows;

    /** The context's epoch floor at launch. */
    u32_t epoch_base;
    vram_telemetry_block_t* telemetry;

    /** Rows of the launch warps have assigned themselves so far, zeroed per launch. */
    u32_t* assigned_rows;

    /** Each node's last pick a walker of the launch made, @b [rows], unknown for none; pick
     *  @b p is walker @c first_id + p / connectivity, its slot p % connectivity. */
    node_id_type_* link_heads;

    /** The pick each pick was threaded before, @b [launch rows,connectivity]. */
    node_id_type_* link_next;

    /** The nodes picked by any walker, each once, and their count, zeroed per launch. */
    node_id_type_* touched_ids;
    u32_t* touched_count;
};

/** Inputs & outputs of one search launch: a tape of queries in, the nearest IDs and their distances
 *  out, both ascending per query. */
template <typename node_id_type_>
struct vram_search_view {
    vram_ragged_view<node_id_type_> storage;

    /** The queries' bytes, word-aligned, with a word of margin past the last. */
    byte_t const* queries;

    /** Query starts as the caller spelled them, @b [queries+1] from any origin. */
    u64_t const* query_offsets;
    u32_t queries_count;

    /** The watermark snapshotted at launch: landmarks are drawn below it and no neighbor at or past
     *  it is followed. */
    node_id_type_ sampling_bound;

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
    node_id_type_* result_ids;
    distance_t* result_distances;
    u32_t count;
};

/** Scores each lane's row in place on @p rung_ of @p kit_, the schedule of every kit whose lane
 *  scores a row alone, so a warp walks one query by itself. */
template <vram_basic_kit kit_, u32_t rung_, typename node_id_type_>
struct vram_lane_schedule {
    inline static constexpr u32_t lanes_per_row_k = 1;

    /** Nothing beside the walking warp's own spans. */
    struct shared_t {};

    __device__ __forceinline__ explicit vram_lane_schedule(shared_t&) noexcept {}

    /** The distance from @p query to the row this lane names, exact within @p radius; infinite for
     *  the sentinel. Every lane of the warp calls it. */
    __device__ __forceinline__ distance_t score(vram_ragged_view<node_id_type_> const& storage,
                                                typename kit_::query_t const& query, node_id_type_ node_id,
                                                distance_t radius, u32_t lane) const noexcept {
        if (node_id == std::numeric_limits<node_id_type_>::max()) return std::numeric_limits<distance_t>::infinity();
        vram_row_t const row = storage.row(node_id);
        distance_t const bound = kit_::bound(storage.kernel, query, row);
        return bound > radius ? bound : vram_score<kit_, rung_>(storage.kernel, query, row, radius, lane);
    }
};

/**
 *  @brief Samples one lane-wide batch of distinct, published landmarks in
 *      @b [sampling_first,sampling_bound) into @p candidate_ids, drawn as the dense walk draws
 *      them.
 *
 *  @param[in] frontier The walk's frontier, asked whether it holds a draw once a saturated visited
 *      set re-scores it.
 *  @return How many it staged: zero once every published row of the range is visited, or the
 *      visited set saturated.
 */
template <vram_basic_kit kit_, typename node_id_type_, vram_frontier frontier_>
__device__ __forceinline__ u32_t vram_sample_landmarks( //
    vram_ragged_view<node_id_type_> const& storage, node_id_type_ sampling_first, node_id_type_ sampling_bound,
    vram_visited_set<typename kit_::warp_t, node_id_type_>& visited, frontier_ const& frontier,
    splitmix64_t& random_generator, node_id_type_* candidate_ids, u32_t lane) noexcept {

    using warp_t = typename kit_::warp_t;
    constexpr node_id_type_ unknown_k = std::numeric_limits<node_id_type_>::max();
    node_id_type_ const span = sampling_bound - sampling_first;
    uniform_below<node_id_type_> const uniform {span};
    node_id_type_ draw = sampling_first + uniform(random_generator);
    node_id_type_ steps = 0;
    bool fresh = false;
    u32_t const saturations = visited.saturations_;
    while (!warp_t::all(fresh || steps == span) && visited.saturations_ == saturations) {
        bool const searching = !fresh && steps != span;
        bool published = false;
        if (searching) {
            vram_neighbors_ref<warp_t, node_id_type_> const row {storage.neighbors_for(draw), storage.connectivity};
            published = row.peek_first() != unknown_k;
        }
        // A saturating claim re-scores, so a draw the frontier holds stages no second copy; every
        // lane takes part in that collective scan.
        bool const drawn = visited.test_and_set(searching && published ? draw : unknown_k, lane);
        bool const claimed = !(visited.saturations_ != saturations && frontier.holds(draw, lane)) && drawn;
        // An unpublished row draws again; a visited one steps forward, one lap at most.
        bool const stepping = searching && !claimed && published;
        if (searching && !claimed && !published) draw = sampling_first + uniform(random_generator);
        if (stepping) draw = draw + 1 == sampling_bound ? sampling_first : draw + 1;
        steps += stepping;
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
template <vram_basic_kit kit_, typename schedule_, vram_frontier frontier_, typename node_id_type_>
__device__ __forceinline__ void vram_score_into_frontier( //
    vram_ragged_view<node_id_type_> const& storage, schedule_& schedule, typename kit_::query_t const& query,
    node_id_type_ const* gathered_ids, u32_t gathered_count, vram_warp_context<frontier_>& context,
    u32_t lane) noexcept {
    using scored_node_t = vram_scored_node<node_id_type_>;
    context.telemetry.scored += gathered_count;
    scored_node_t const worst = context.frontier.worst();
    distance_t const radius = worst == scored_node_t::sentinel() ? std::numeric_limits<distance_t>::infinity()
                                                                 : worst.distance();
    node_id_type_ const node_id = lane < gathered_count ? gathered_ids[lane]
                                                        : std::numeric_limits<node_id_type_>::max();
    distance_t const distance = schedule.score(storage, query, node_id, radius, lane);
    context.frontier.push_batch(
        lane < gathered_count ? scored_node_t::make(distance, node_id) : scored_node_t::sentinel(), lane);
}

/** Continues a greedy best-first walk toward @p query over whatever the frontier holds, as the
 *  dense walk does: landmarks from @b [sampling_first,sampling_bound), a transiently locked
 *  candidate held through one other expansion and dropped if still locked then, and no neighbor at
 *  or past @p first_invisible_id followed. Read-only on the graph. */
template <vram_basic_kit kit_, typename schedule_, vram_frontier frontier_, typename node_id_type_>
__device__ __forceinline__ void vram_search_ahead( //
    vram_ragged_view<node_id_type_> const& storage, schedule_& schedule, typename kit_::query_t const& query,
    node_id_type_ sampling_first, node_id_type_ sampling_bound, node_id_type_ first_invisible_id,
    vram_visited_set<typename kit_::warp_t, node_id_type_>& visited, node_id_type_* gathered_ids,
    vram_warp_context<frontier_>& context, u32_t lane) noexcept {

    using warp_t = typename kit_::warp_t;
    using scored_node_t = vram_scored_node<node_id_type_>;
    constexpr node_id_type_ unknown_k = std::numeric_limits<node_id_type_>::max();
    frontier_& frontier = context.frontier;
    vram_telemetry_t& telemetry = context.telemetry;
    scored_node_t held = scored_node_t::sentinel();
    u32_t held_at_hop = 0;
    bool seeding = true;
    while (true) {
        if (seeding) {
            u32_t const gathered_count = vram_sample_landmarks<kit_>( //
                storage, sampling_first, sampling_bound, visited, frontier, context.random_generator, gathered_ids,
                lane);
            if (!gathered_count) break; // Every published row of the range is already visited.
            vram_score_into_frontier<kit_>(storage, schedule, query, gathered_ids, gathered_count, context, lane);
            seeding = false;
            continue;
        }
        bool const revisiting = !(held == scored_node_t::sentinel()) && telemetry.hops != held_at_hop;
        scored_node_t const candidate = revisiting ? held : frontier.pop_nearest(lane);
        if (revisiting) held = scored_node_t::sentinel();
        if (candidate == scored_node_t::sentinel()) {
            if (!(held == scored_node_t::sentinel())) {
                held_at_hop = ~0u;
                continue;
            }
            // A drained frontier before an exhausted walk re-seeds, until no published row of the
            // range is left unvisited.
            if (frontier.exhausted()) break;
            seeding = true;
            ++telemetry.restarts;
            continue;
        }
        if (frontier.beyond_radius(candidate)) break;
        ++telemetry.hops;

        // Only the leader reads the racing lock word, so every lane agrees on one unlock.
        vram_neighbors_ref<warp_t, node_id_type_> const row {storage.neighbors_for(candidate.id()),
                                                             storage.connectivity};
        node_id_type_ first_value = lane == 0 ? row.peek_first() : unknown_k;
        first_value = warp_t::shuffle(first_value, 0);
        warp_t::sync();
        if (first_value == unknown_k) {
            if (revisiting || !(held == scored_node_t::sentinel())) ++telemetry.locked_drops;
            else held = candidate, held_at_hop = telemetry.hops;
            continue;
        }
        // A warp-width of slots at a time; the bound is warp-uniform for the collectives.
        for (u32_t slot_base = 0; slot_base < storage.connectivity; slot_base += warp_t::width_k) {
            u32_t const slot = slot_base + lane;
            node_id_type_ const neighbor_id = slot == 0                     ? first_value
                                              : slot < storage.connectivity ? row.load_slot(slot)
                                                                            : unknown_k;
            // The bound never exceeds the ID space, so it rejects the sentinel too.
            bool const usable = neighbor_id < first_invisible_id && neighbor_id != candidate.id();
            // Past a saturation a claim re-scores, so an ID the frontier holds stages no second
            // copy; every lane takes part in that collective scan.
            bool const claimed = visited.test_and_set(usable ? neighbor_id : unknown_k, lane);
            bool const fresh = !(visited.saturations_ && frontier.holds(neighbor_id, lane)) && claimed;
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
    telemetry.locked_drops += !(held == scored_node_t::sentinel());
}

/** Greedy best-first walk from random published landmarks toward @p query: a fresh frontier handed
 *  to @ref vram_search_ahead. */
template <vram_basic_kit kit_, typename schedule_, vram_frontier frontier_, typename node_id_type_>
__device__ __forceinline__ void vram_search_around( //
    vram_ragged_view<node_id_type_> const& storage, schedule_& schedule, typename kit_::query_t const& query,
    node_id_type_ sampling_bound, node_id_type_ first_invisible_id,
    vram_visited_set<typename kit_::warp_t, node_id_type_>& visited, node_id_type_* gathered_ids,
    vram_warp_context<frontier_>& context, u32_t lane) noexcept {
    context.frontier.reset(lane);
    vram_search_ahead<kit_>(storage, schedule, query, node_id_type_ {0}, sampling_bound, first_invisible_id, visited,
                            gathered_ids, context, lane);
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
    template <vram_basic_kit kit_, typename schedule_, typename node_id_type_>
    static __device__ __forceinline__ u32_t select( //
        vram_ragged_view<node_id_type_> const& storage, schedule_& schedule, typename kit_::query_t const&,
        byte_t* pivot_span, node_id_type_ const* candidate_ids, f32_t const* candidate_distances,
        u32_t candidates_count, u32_t* kept_positions, vram_telemetry_t& telemetry, u32_t lane) noexcept {

        using warp_t = typename kit_::warp_t;
        u32_t kept_count = 0;
#pragma unroll 1
        for (u32_t candidate_index = 0; candidate_index < candidates_count && kept_count < storage.connectivity;
             ++candidate_index) {
            f32_t const to_query = candidate_distances[candidate_index];
            typename kit_::query_t const pivot = kept_count
                                                     ? vram_prepare<kit_>(storage.parameters,
                                                                          storage.row(candidate_ids[candidate_index]),
                                                                          pivot_span, lane)
                                                     : typename kit_::query_t {};
            bool diverse = true;
            for (u32_t base = 0; base < kept_count && diverse; base += warp_t::width_k) { // Warp-uniform bounds.
                node_id_type_ const kept_id = base + lane < kept_count ? candidate_ids[kept_positions[base + lane]]
                                                                       : std::numeric_limits<node_id_type_>::max();
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

/**
 *  Hands the whole pool of a finished walk toward @p query, its @p selection_rows ranks ascending,
 *  to @p selector_, staged as candidate IDs and distances, and writes the kept candidates as node
 *  IDs in @p selection.selected_ids, through their positions in @p selection.kept_positions.
 *  @return The kept count.
 */
template <vram_basic_kit kit_, typename selector_, typename schedule_, vram_frontier frontier_, typename node_id_type_>
__device__ __forceinline__ u32_t vram_select_neighbors( //
    vram_ragged_view<node_id_type_> const& storage, schedule_& schedule, typename kit_::query_t const& query,
    byte_t* pivot_span, u32_t selection_rows, vram_selection_scratch<node_id_type_> selection,
    vram_warp_context<frontier_>& context, u32_t lane) noexcept {

    using warp_t = typename kit_::warp_t;
    using scored_node_t = vram_scored_node<node_id_type_>;
    node_id_type_* const candidate_ids = selection.candidate_ids;
    u32_t candidates_count = 0;
    context.frontier.for_each_rank(selection_rows, lane, [&](u32_t rank, scored_node_t entry) noexcept {
        bool const live = rank < selection_rows && !(entry == scored_node_t::sentinel());
        if (live) {
            candidate_ids[rank] = entry.id();
            selection.candidate_distances[rank] = entry.distance();
        }
        candidates_count += warp_t::count(warp_t::ballot(live));
    });
    warp_t::sync();

    u32_t const selected_count = selector_::template select<kit_>(storage, schedule, query, pivot_span, candidate_ids,
                                                                  selection.candidate_distances, candidates_count,
                                                                  selection.kept_positions, context.telemetry, lane);
    // The caller links to nodes, not to positions in the pool.
    for (u32_t slot = lane; slot < selected_count; slot += warp_t::width_k)
        selection.selected_ids[slot] = candidate_ids[selection.kept_positions[slot]];
    warp_t::sync();
    return selected_count;
}

/** Threads the @p selected_count picks of @p node_id onto their targets' chains, listing each
 *  target the first time any walker of the launch picks it; the back-link launch behind merges them
 *  all. */
template <typename warp_type_, typename node_id_type_>
__device__ __forceinline__ void vram_thread_back_links(vram_build_view<node_id_type_> const& view,
                                                       node_id_type_ node_id, node_id_type_ const* selected_ids,
                                                       u32_t selected_count, u32_t lane) noexcept {
    for (u32_t slot = lane; slot < selected_count; slot += warp_type_::width_k) {
        node_id_type_ const target = selected_ids[slot];
        node_id_type_ const pick = (node_id - view.first_id) * view.storage.connectivity + slot;
        node_id_type_ const previous = warp_type_::exchange_acquire(view.link_heads[target], pick);
        view.link_next[pick] = previous;
        if (previous == std::numeric_limits<node_id_type_>::max())
            view.touched_ids[warp_type_::fetch_add_relaxed(*view.touched_count, 1u)] = target;
    }
    warp_type_::sync();
}

/** Inserts the node identified by @p new_node_id, whose bytes already sit in the values tape: lands
 *  its two row offsets, walks the published graph, selects diverse neighbors, publishes with one
 *  release store, and threads its picks for the back-link launch, one warp per node. */
template <vram_basic_kit kit_, typename selector_ = vram_keep_diverse_t, typename schedule_, vram_frontier frontier_,
          typename node_id_type_>
__device__ __forceinline__ void vram_insert_one( //
    vram_build_view<node_id_type_> const& view, schedule_& schedule, node_id_type_ new_node_id,
    vram_visited_set<typename kit_::warp_t, node_id_type_>& visited, byte_t* query_span, byte_t* pivot_span,
    node_id_type_* gathered_ids, vram_selection_scratch<node_id_type_> selection, vram_warp_context<frontier_>& context,
    u32_t lane) noexcept {

    using warp_t = typename kit_::warp_t;
    vram_ragged_view<node_id_type_> const& storage = view.storage;

    // Both bounds land, so readers past publication see them whichever neighbor lands first.
    u32_t const offset = static_cast<u32_t>(new_node_id - view.first_id);
    u64_t const origin = view.staged_offsets[0];
    u64_t const row_begin = view.tape_first + view.staged_offsets[offset] - origin;
    u64_t const row_end = view.tape_first + view.staged_offsets[offset + 1] - origin;
    if (lane == 0) {
        warp_t::store_relaxed(storage.offsets[new_node_id], row_begin);
        warp_t::store_relaxed(storage.offsets[new_node_id + 1], row_end);
    }
    typename kit_::query_t const query = vram_prepare<kit_>(
        storage.parameters, {storage.values + row_begin, static_cast<u32_t>(row_end - row_begin)}, query_span, lane);

    // A virgin index publishes an empty node instead of walking.
    if (new_node_id != 0)
        vram_search_around<kit_>(storage, schedule, query, new_node_id, view.end_id, visited, gathered_ids, context,
                                 lane);
    else context.frontier.reset(lane);

    // No lock on the own row: no reader arrives before the slot-0 release flips the sentinel.
    u32_t const selected_count = vram_select_neighbors<kit_, selector_>( //
        storage, schedule, query, pivot_span, view.selection_rows, selection, context, lane);
    vram_neighbors_ref<warp_t, node_id_type_> const own_neighbors {storage.neighbors_for(new_node_id),
                                                                   storage.connectivity};
    for (u32_t slot = lane; slot < selected_count; slot += warp_t::width_k)
        if (slot > 0) own_neighbors.store_slot(slot, selection.selected_ids[slot]);
    warp_t::sync();
    node_id_type_ const first_link = selected_count ? selection.selected_ids[0] : new_node_id;
    if (lane == 0) own_neighbors.unlock(first_link); // A self-loop encodes published-empty.
    vram_thread_back_links<warp_t>(view, new_node_id, selection.selected_ids, selected_count, lane);
}

/** Re-selects the published row of @p node_id once its wave's back-links made the wave's rows
 *  reachable from one another, or once a refine asks: walks from the row's own neighbors over every
 *  row below the view's sampling bound, selects again, rewrites the row in place, tail slots
 *  relaxed and slot 0 last with a release, so walks on other streams read either row whole at every
 *  slot, and threads the new picks. */
template <vram_basic_kit kit_, typename selector_ = vram_keep_diverse_t, typename schedule_, vram_frontier frontier_,
          typename node_id_type_>
__device__ __forceinline__ void vram_reselect_one( //
    vram_build_view<node_id_type_> const& view, schedule_& schedule, node_id_type_ node_id,
    vram_visited_set<typename kit_::warp_t, node_id_type_>& visited, byte_t* query_span, byte_t* pivot_span,
    node_id_type_* gathered_ids, vram_selection_scratch<node_id_type_> selection, vram_warp_context<frontier_>& context,
    u32_t lane) noexcept {

    using warp_t = typename kit_::warp_t;
    constexpr node_id_type_ unknown_k = std::numeric_limits<node_id_type_>::max();
    vram_ragged_view<node_id_type_> const& storage = view.storage;
    typename kit_::query_t const query = vram_prepare<kit_>(storage.parameters, storage.row(node_id), query_span, lane);
    context.frontier.reset(lane);
    visited.test_and_set(node_id, lane);

    // The walk starts from the row's own neighbors, which the wave's first walk already found.
    vram_neighbors_ref<warp_t, node_id_type_> const own_neighbors {storage.neighbors_for(node_id),
                                                                   storage.connectivity};
    for (u32_t slot_base = 0; slot_base < storage.connectivity; slot_base += warp_t::width_k) {
        u32_t const slot = slot_base + lane;
        node_id_type_ const seed_id = slot < storage.connectivity ? own_neighbors.load_slot(slot) : unknown_k;
        bool const usable = seed_id < view.sampling_bound && seed_id != node_id;
        bool const fresh = visited.test_and_set(usable ? seed_id : unknown_k, lane);
        typename warp_t::mask_t const fresh_mask = warp_t::ballot(fresh);
        u32_t const ordinal = warp_t::count(fresh_mask & warp_t::below(lane));
        if (fresh) gathered_ids[ordinal] = seed_id;
        warp_t::sync();
        u32_t const gathered_count = warp_t::count(fresh_mask);
        if (gathered_count)
            vram_score_into_frontier<kit_>(storage, schedule, query, gathered_ids, gathered_count, context, lane);
        warp_t::sync();
    }
    vram_search_ahead<kit_>(storage, schedule, query, node_id_type_ {0}, view.sampling_bound, view.sampling_bound,
                            visited, gathered_ids, context, lane);

    u32_t const selected_count = vram_select_neighbors<kit_, selector_>( //
        storage, schedule, query, pivot_span, view.selection_rows, selection, context, lane);
    for (u32_t slot = lane; slot < storage.connectivity; slot += warp_t::width_k)
        if (slot) own_neighbors.store_slot(slot, slot < selected_count ? selection.selected_ids[slot] : unknown_k);
    warp_t::sync();
    if (lane == 0) own_neighbors.unlock(selected_count ? selection.selected_ids[0] : node_id);
    vram_thread_back_links<warp_t>(view, node_id, selection.selected_ids, selected_count, lane);
}

/**
 *  @brief Back-links one node the launch's walkers picked, chasing their picks from @c link_heads
 *      through @c link_next in rounds.
 *
 *  Each round caches the live row and as many walkers as the view's @c selection_rows leave room
 *  for, appends them while the row has room, and re-selects over the union otherwise. The append
 *  is the dense back-linker's fast path; the re-selection hands @p selector_ the union ascending
 *  by distance to the node, and the next round starts from the row the last one wrote, so no
 *  walker is dropped. The launch hands each node to one warp, the only writer of its row, so it
 *  reads the row plainly, while walks on other streams may read it throughout: every slot changes
 *  by one relaxed store and slot 0 last, with a release.
 */
template <vram_basic_kit kit_, typename selector_ = vram_keep_diverse_t, typename schedule_, typename node_id_type_>
__device__ __forceinline__ void vram_back_link_one( //
    vram_build_view<node_id_type_> const& view, schedule_& schedule, node_id_type_ node_id, byte_t* query_span,
    byte_t* pivot_span, vram_selection_scratch<node_id_type_> selection, vram_telemetry_t& telemetry,
    u32_t lane) noexcept {

    using warp_t = typename kit_::warp_t;
    using scored_node_t = vram_scored_node<node_id_type_>;
    constexpr node_id_type_ unknown_k = std::numeric_limits<node_id_type_>::max();
    vram_ragged_view<node_id_type_> const& storage = view.storage;
    u32_t const connectivity = storage.connectivity;
    node_id_type_* const candidate_ids = selection.candidate_ids;
    vram_neighbors_ref<warp_t, node_id_type_> const live_row {storage.neighbors_for(node_id), connectivity};

    // A chain is serial, so one lane follows it, and the rounds resume where it stopped; no lane
    // writes the head before the call returns, so every lane reads it.
    node_id_type_ pick = view.link_heads[node_id];
    while (pick != unknown_k) {
        u32_t live_count = 0;
        for (u32_t slot_base = 0; slot_base < connectivity; slot_base += warp_t::width_k) {
            u32_t const slot = slot_base + lane;
            node_id_type_ const live_id = slot < connectivity ? live_row.load_slot(slot) : unknown_k;
            bool const usable = live_id != unknown_k && live_id != node_id;
            typename warp_t::mask_t const usable_mask = warp_t::ballot(usable);
            if (usable) candidate_ids[live_count + warp_t::count(usable_mask & warp_t::below(lane))] = live_id;
            live_count += warp_t::count(usable_mask);
            warp_t::sync();
        }

        // The chain's next walkers fill the slots past the live row.
        u32_t chained_end = live_count;
        if (lane == 0)
            for (; pick != unknown_k && chained_end != view.selection_rows; pick = view.link_next[pick])
                candidate_ids[chained_end++] = view.first_id + pick / connectivity;
        chained_end = warp_t::shuffle(chained_end, 0);
        pick = warp_t::shuffle(pick, 0);
        warp_t::sync();

        // The chained IDs the live row lacks, compacted behind it; a re-selection re-picks rows it
        // links.
        u32_t fresh_count = 0;
        for (u32_t slot_base = live_count; slot_base < chained_end; slot_base += warp_t::width_k) {
            u32_t const slot = slot_base + lane;
            node_id_type_ const chained_id = slot < chained_end ? candidate_ids[slot] : unknown_k;
            bool fresh = chained_id != unknown_k && chained_id != node_id;
            for (u32_t live_index = 0; fresh && live_index != live_count; ++live_index)
                fresh = candidate_ids[live_index] != chained_id;
            typename warp_t::mask_t const fresh_mask = warp_t::ballot(fresh);
            warp_t::sync();
            if (fresh)
                candidate_ids[live_count + fresh_count + warp_t::count(fresh_mask & warp_t::below(lane))] = chained_id;
            fresh_count += warp_t::count(fresh_mask);
            warp_t::sync();
        }
        if (!fresh_count) continue;
        u32_t const candidates_count = live_count + fresh_count;

        // Room for every walker of the round: appended in place, no distance needed, slot 0 last.
        // A round of duplicates may stop short of the chain's end, so the next round goes on.
        if (candidates_count <= connectivity) {
            for (u32_t slot = live_count + lane; slot < candidates_count; slot += warp_t::width_k)
                if (slot) live_row.store_slot(slot, candidate_ids[slot]);
            warp_t::sync();
            if (lane == 0 && !live_count) live_row.unlock(candidate_ids[0]);
            warp_t::sync();
            continue;
        }

        typename kit_::query_t const query = vram_prepare<kit_>(storage.parameters, storage.row(node_id), query_span,
                                                                lane);
        vram_sorted_buffer<warp_t, node_id_type_> sorted_candidates {selection.scored_entries, 0, view.selection_rows};
        for (u32_t slot_base = 0; slot_base < candidates_count; slot_base += warp_t::width_k) {
            u32_t const slot = slot_base + lane;
            node_id_type_ const candidate_id = slot < candidates_count ? candidate_ids[slot] : unknown_k;
            distance_t const distance = schedule.score(storage, query, candidate_id,
                                                       std::numeric_limits<distance_t>::infinity(), lane);
            scored_node_t const scored = candidate_id != unknown_k ? scored_node_t::make(distance, candidate_id)
                                                                   : scored_node_t::sentinel();
            sorted_candidates.insert_batch(sort_lanes<warp_t>(scored, lane),
                                           min_of<u32_t>(candidates_count - slot_base, warp_t::width_k), lane);
        }
        telemetry.scored += candidates_count;
        ++telemetry.reselections;

        // The union is scored, so its IDs take their places, ascending.
        for (u32_t position = lane; position < sorted_candidates.size_; position += warp_t::width_k) {
            candidate_ids[position] = sorted_candidates.at(position).id();
            selection.candidate_distances[position] = sorted_candidates.at(position).distance();
        }
        warp_t::sync();
        u32_t* const kept_positions = selection.kept_positions;
        u32_t const refined_count = selector_::template select<kit_>(
            storage, schedule, query, pivot_span, candidate_ids, selection.candidate_distances, sorted_candidates.size_,
            kept_positions, telemetry, lane);
        for (u32_t slot = lane; slot < connectivity; slot += warp_t::width_k)
            if (slot) live_row.store_slot(slot, slot < refined_count ? candidate_ids[kept_positions[slot]] : unknown_k);
        warp_t::sync();
        if (lane == 0) live_row.unlock(candidate_ids[kept_positions[0]]);
        warp_t::sync();
        ++telemetry.changed_nodes;
    }
}

/** One query's full walk: prepares the query into its span, walks the graph, and lands its nearest
 *  @c count IDs and distances, ascending, sentinel IDs past the live prefix. Safe against
 *  concurrent insertions: the walk gates on every slot-0 word and never writes the graph. */
template <vram_basic_kit kit_, typename schedule_, vram_frontier frontier_, typename node_id_type_>
__device__ __forceinline__ void vram_walk_one( //
    vram_search_view<node_id_type_> const& view, schedule_& schedule, u32_t query_index, byte_t* query_span,
    vram_visited_set<typename kit_::warp_t, node_id_type_>& visited, frontier_ frontier, node_id_type_* gathered_ids,
    u32_t lane) noexcept {

    using warp_t = typename kit_::warp_t;
    u64_t const origin = view.query_offsets[0];
    u64_t const first = view.query_offsets[query_index] - origin;
    vram_row_t const row {view.queries + first,
                          static_cast<u32_t>(view.query_offsets[query_index + 1] - origin - first)};
    typename kit_::query_t const query = vram_prepare<kit_>(view.storage.parameters, row, query_span, lane);

    vram_warp_context<frontier_> context;
    context.frontier = frontier;
    context.random_generator = vram_lane_generator<warp_t>(query_index, lane);
    vram_search_around<kit_>(view.storage, schedule, query, view.sampling_bound, view.sampling_bound, visited,
                             gathered_ids, context, lane);

    // The rank extraction shuffles, so every lane participates and only in-range ones store.
    node_id_type_* const ids_row = vram_row(view.result_ids, query_index, view.count);
    distance_t* const distances_row = vram_row(view.result_distances, query_index, view.count);
    context.frontier.for_each_rank(view.count, lane, [&](u32_t rank, vram_scored_node<node_id_type_> entry) noexcept {
        if (rank >= view.count) return;
        ids_row[rank] = entry.id();
        distances_row[rank] = entry.distance();
    });
    context.telemetry.tail_compactions = context.frontier.compactions();
    context.telemetry.tail_unseated = context.frontier.unseated();
    context.telemetry.visited_saturations = visited.saturations_;
    context.telemetry.template flush<warp_t, vram_traversal_t::searching_k>(view.telemetry, lane);
}

} // namespace unum::usearch::basic
