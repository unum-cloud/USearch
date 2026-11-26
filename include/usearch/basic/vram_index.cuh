/**
 *  @file include/usearch/basic/vram_index.cuh
 *  @author Ash Vardanian
 *  @date October 15, 2026
 *  @brief The device bodies every basic VRAM index compiled by NVCC or HIP-Clang shares: the ragged
 *      view, the greedy walk, the diversity selection, insertion, re-selection, the back-link
 *      merge, and a round's gather and restitch.
 *
 *  Every body is templated on a basic kit, which names its vendor's warp policy as @c kit_::warp_t,
 *  and on the node-ID type its views hold, and walks over the dense design's frontiers, visited
 *  set, slot-0 neighbor protocol, selection spans and telemetry. A basic kit prepares a stored
 *  entry as a query once per walk, into a span of the walking warp's scratch, and scores whole
 *  entries, which a schedule hands it: a lane each in place, a group of lanes each in turns, or the
 *  vendor's wider split, so the bodies only name an entry per lane and read back its distance. A
 *  kit storing a sketch in front of every sequence builds by it, the view naming its bytes: its
 *  builders read the stored sketch in place as their query and score a lane to an entry, while its
 *  searches walk by the query's sketch first and then prepare and score the sequence behind it,
 *  seeded with that walk's nearest nodes. A schedule scores on one rung of its kit, the rows a lane
 *  of an alignment kit holds, or zero for a kit scoring every entry alike, and the view names the
 *  kernel kind a kit covering several metrics picks among, as dense views do.
 *
 *  Insertion publishes forward neighbor rows alone. A back-link re-selection costs a full diversity
 *  pass of distances, far past any lock budget for the costly kits, so each admission wave's
 *  walkers thread their picks onto per-node chains and a back-link launch behind them merges every
 *  chain into its node, and no device lock is held across a distance. A wave's nodes walk the graph
 *  again once those back-links let them reach one another, and re-select their neighbors. A round
 *  walks nothing: its gather files every entry under its nearest hub, and its restitch scores each
 *  neighborhood's entries against one another and keeps the diverse links among each entry's own
 *  and its nearest mates, the health check's reach rule on top. The host classes, the kernels and
 *  the launch bounds stay in each vendor's own @c vram_<vendor>_index file, beside the kits in each
 *  modality's @c vram_<vendor>_metric.
 */
#pragma once

#include <concepts>    // `std::same_as` - the kit concept's verbs
#include <limits>      // `std::numeric_limits` - an unbounded radius, the node-ID type's sentinel
#include <type_traits> // `std::is_same_v`, `std::conditional_t` - a round's 32-bit IDs, a role's query

#include <numkong/capabilities.h> // `nk_kernel_kind_t` - the views hold the kernel kind

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/vram_queue.cuh"

namespace unum::usearch::basic {
using namespace unum::usearch::utilities;

/** One stored value as a kit reads it: its first byte and its length. */
struct vram_entry_t {
    byte_t const* data;
    u32_t bytes;
};

/** The distances a paired scorer returns, one per entry. */
struct vram_distance_pair_t {
    distance_t first;
    distance_t second;
};

/** What a basic VRAM kit binds: one modality's distance on one GPU generation, with the warp policy
 *  its walks shuffle through, the scan kit its prefix sums run on, and the selector its nodes keep
 *  neighbors by. @c canonical_prepare lays an entry out as a query in a span, every lane of the
 *  warp taking part, from the entry alone or, for a kit with runtime parameters, from the storage's
 *  @c parameters too, which @ref unum::usearch::basic::vram_prepare tells apart;
 *  @c canonical_lower_bound answers a lower bound from the two entries' headers alone. Its
 *  @c canonical_distance runs on its lanes per entry, every one of them returning the distance,
 *  exact within the radius and above it past: @c canonical_lanes_per_entry_k of them and a plain
 *  @c canonical_distance, or, for a kit whose split follows a rung, @c canonical_lanes_per_entry of
 *  the rung and a @c canonical_distance templated on it, which
 *  @ref unum::usearch::basic::vram_lanes_per_entry and @ref unum::usearch::basic::vram_score tell
 *  apart. A round's entries have no order: results go by slot, and ties break by node ID. */
template <typename kit_>
concept a_vram_basic_kit = requires(typename kit_::canonical_query_t const& query, vram_entry_t entry, byte_t* span,
                                    byte_t const* parameters, nk_kernel_kind_t kernel, u32_t lane) {
    typename kit_::warp_t;
    typename kit_::scan_t;
    typename kit_::canonical_query_t;
    typename kit_::selector_t;
    requires requires {
        { kit_::canonical_prepare(entry, span, lane) } -> std::same_as<typename kit_::canonical_query_t>;
    } || requires {
        { kit_::canonical_prepare(parameters, entry, span, lane) } -> std::same_as<typename kit_::canonical_query_t>;
    };
    { kit_::canonical_lower_bound(kernel, query, entry) } -> std::same_as<distance_t>;
};

/** A kit whose @c canonical_prepare reads its runtime parameters beside the entry it lays out. */
template <typename kit_>
concept a_parameter_preparing_kit = requires(byte_t const* parameters, vram_entry_t entry, byte_t* span, u32_t lane) {
    kit_::canonical_prepare(parameters, entry, span, lane);
};

/** A kit whose builders score a sketch stored in front of every sequence, as many bytes as the view
 *  names: its @c sketch_distance scores two of them, symmetric and exact, one lane to a pair. A kit
 *  naming @c sketch_tables_t without a distance stores the same closed form and builds by the
 *  canonical score alone. */
template <typename kit_>
concept a_vram_sketching_kit = requires(byte_t const* sketch, u32_t bytes, u32_t lane) {
    { kit_::sketch_distance(sketch, sketch, bytes, lane) } -> std::same_as<distance_t>;
};

/** A kit that closes every stored value to a word behind the gap its sketch takes. */
template <typename kit_>
concept a_vram_kit_closing_values = requires { typename kit_::sketch_tables_t; };

/** Whether @p role_ scores @p kit_ by its sketches: a builder's, for a kit building by them. */
template <a_vram_basic_kit kit_, score_role_t role_>
constexpr bool vram_scores_by_sketch() noexcept {
    return role_ == score_role_t::build_k && a_vram_sketching_kit<kit_>;
}

/** A builder's query under a kit building by sketches: the stored sketch, read in place. */
struct vram_sketch_query_t {
    byte_t const* sketch;
};

/** The query @p role_ walks @p kit_ with: the stored sketch for a builder of a kit building by
 *  sketches, the kit's canonical query otherwise. */
template <a_vram_basic_kit kit_, score_role_t role_>
using vram_query =
    std::conditional_t<vram_scores_by_sketch<kit_, role_>(), vram_sketch_query_t, typename kit_::canonical_query_t>;

/** The part of the stored value at @b [first,end) of @p values that @p role_ reads under @p kit_,
 *  behind a sketch of @p sketch_bytes: the sketch for a builder of a kit building by sketches, the
 *  sequence otherwise, which for a kit closing its values ends before the closing padding, whose
 *  length its closing byte counts. */
template <typename kit_, score_role_t role_>
__device__ __forceinline__ vram_entry_t vram_part(byte_t const* values, u64_t first, u64_t end,
                                                  u32_t sketch_bytes) noexcept {
    if constexpr (!a_vram_kit_closing_values<kit_>) return {values + first, static_cast<u32_t>(end - first)};
    else if constexpr (role_ == score_role_t::build_k && a_vram_sketching_kit<kit_>)
        return {values + first, sketch_bytes};
    else {
        u32_t const closing = static_cast<u32_t>(values[end - 1]);
        return {values + first + sketch_bytes, static_cast<u32_t>(end - first) - sketch_bytes - closing};
    }
}

/** Lays @p entry, the part @p role_ reads, out as the query @p role_ walks @p kit_ with: the sketch
 *  as it is, or the kit's canonical query in @p span, handing a kit with runtime parameters the
 *  ones at @p parameters; every lane of the warp calls it. */
template <a_vram_basic_kit kit_, score_role_t role_>
__device__ __forceinline__ vram_query<kit_, role_> vram_prepare(byte_t const* parameters, vram_entry_t entry,
                                                                byte_t* span, u32_t lane) noexcept {
    if constexpr (vram_scores_by_sketch<kit_, role_>()) return {entry.data};
    else if constexpr (a_parameter_preparing_kit<kit_>) return kit_::canonical_prepare(parameters, entry, span, lane);
    else return kit_::canonical_prepare(entry, span, lane);
}

/** Lanes @p kit_ scores one entry with for @p role_ on @p rung_: one for a sketch, its
 *  @c canonical_lanes_per_entry_k at rung zero, and past it what its @c canonical_lanes_per_entry
 *  answers for the rung. */
template <a_vram_basic_kit kit_, score_role_t role_, u32_t rung_>
constexpr u32_t vram_lanes_per_entry() noexcept {
    if constexpr (vram_scores_by_sketch<kit_, role_>()) return 1;
    else if constexpr (rung_ == 0) return kit_::canonical_lanes_per_entry_k;
    else return kit_::canonical_lanes_per_entry(rung_);
}

/** The lower bound @p kit_ answers for @p role_ under @p kernel from @p query to @p entry: none
 *  for a sketch, which always scores. */
template <a_vram_basic_kit kit_, score_role_t role_>
__device__ __forceinline__ distance_t vram_lower_bound(nk_kernel_kind_t kernel, vram_query<kit_, role_> const& query,
                                                       vram_entry_t entry) noexcept {
    if constexpr (vram_scores_by_sketch<kit_, role_>()) return 0;
    else return kit_::canonical_lower_bound(kernel, query, entry);
}

/** The distance @p kit_ scores for @p role_ on @p rung_ under @p kernel from @p query to @p entry,
 *  exact within @p radius, and at any radius for a sketch; every lane of the group scoring the
 *  entry calls it. */
template <a_vram_basic_kit kit_, score_role_t role_, u32_t rung_>
__device__ __forceinline__ distance_t vram_score(nk_kernel_kind_t kernel, vram_query<kit_, role_> const& query,
                                                 vram_entry_t entry, distance_t radius, u32_t lane) noexcept {
    if constexpr (vram_scores_by_sketch<kit_, role_>())
        return kit_::sketch_distance(query.sketch, entry.data, entry.bytes, lane);
    else if constexpr (rung_ == 0) return kit_::canonical_distance(kernel, query, entry, radius, lane);
    else return kit_::template canonical_distance<rung_>(kernel, query, entry, radius, lane);
}

/** A kit that scores two entries on @p rung_ in one pass of its whole warp, each as its
 *  @c canonical_distance would under its own radius, for a @p role_ scoring canonically. */
template <typename kit_, score_role_t role_, u32_t rung_>
concept a_vram_paired_kit =
    requires(nk_kernel_kind_t kernel, typename kit_::canonical_query_t const& query, vram_entry_t entry,
             distance_t radius, u32_t lane) {
        {
            kit_::template canonical_distance_pair<rung_>(kernel, query, entry, entry, radius, radius, lane)
        } -> std::same_as<vram_distance_pair_t>;
    } &&
    vram_lanes_per_entry<kit_, role_, rung_>() == kit_::warp_t::width_k && !vram_scores_by_sketch<kit_, role_>();

/** Typed view over a basic index's device storage: the values tape, its entry offsets, and the
 *  fixed-capacity neighbors matrix of @p node_id_type_ IDs. */
template <typename node_id_type_>
struct vram_ragged_view {

    /** Every value back to back, from a word-aligned base, with a word of margin past the last, so
     *  word loads never fault. */
    byte_t const* values;

    /** Entry starts into @c values, @b [nodes+1]; an inserting warp writes its own entry's two
     *  before publishing it. */
    u64_t* offsets;

    /** Neighbors matrix, @b [nodes,connectivity] IDs; slot 0 doubles as the lock word. */
    node_id_type_* neighbors;

    /** Neighbor slots per node: the stride of @c neighbors and the connectivity. */
    u32_t connectivity;

    /** The metric a kit covering several picks among, unknown for a kit of one. */
    nk_kernel_kind_t kernel;

    /** The kit's runtime parameters, as its face laid them out; null for a kit taking none. */
    byte_t const* parameters;

    /** Bytes of the sketch in front of every stored value; zero for a kit storing none. */
    u32_t sketch_bytes;

    /** Bytes the stored value of @p node_id takes whole: the order a round splits by, which a
     *  sketch of one size leaves as the sequences' own. */
    __device__ __forceinline__ u32_t stored_bytes(node_id_type_ node_id) const noexcept {
        return static_cast<u32_t>(offsets[node_id + 1] - offsets[node_id]);
    }

    /** The sequence of @p node_id as @p kit_ stores it. */
    template <typename kit_>
    __device__ __forceinline__ vram_entry_t canonical_entry(node_id_type_ node_id) const noexcept {
        return vram_part<kit_, score_role_t::search_k>(values, offsets[node_id], offsets[node_id + 1], sketch_bytes);
    }

    /** The part of @p node_id 's stored value that @p role_ reads under @p kit_. */
    template <typename kit_, score_role_t role_>
    __device__ __forceinline__ vram_entry_t scored(node_id_type_ node_id) const noexcept {
        return vram_part<kit_, role_>(values, offsets[node_id], offsets[node_id + 1], sketch_bytes);
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

    /** The wave's entry starts as the caller spelled them, @b [entries+1] from any origin. */
    u64_t const* staged_offsets;

    /** Where the wave's first byte went in the values tape. */
    u64_t tape_first;

    /** IDs strictly below this bound may be claimed & inserted by this launch. */
    node_id_type_ end_id;

    /** The nodes a re-selection walk samples landmarks below and follows neighbors below: the
     *  launch's own bound. */
    node_id_type_ sampling_bound;

    /** Bytes of the walking warp's dynamic-shared slice: the query span and the pivot span; zero
     *  when the arena shape holds them instead. */
    u32_t scratch_bytes_per_warp;

    /** Bytes of one span, a multiple of 16. */
    u32_t span_bytes;

    /** The launch's base ID: staged entry 0 goes there, and the warps assign themselves offsets
     *  within @b [first_id,end_id). */
    node_id_type_ first_id;

    /** The IDs a re-selection walks by assigned offset, from the launch's first; null walks
     *  @b [first_id,end_id). */
    node_id_type_ const* listed;

    /** The traversal arena every warp carves its slice from, and the shape of one slice. */
    byte_t* arena;
    vram_arena_shape_t arena_shape;

    /** The arena frontier's expansion budget on the rung past the register ladder. */
    u32_t expansion;

    /** Candidates the launch's walkers select from past their two spans, as the host sized them:
     *  all of a walk's candidates, or a back-link round's live row and its appended-nodes row. */
    u32_t selection_nodes;

    /** The context's epoch floor at launch. */
    u32_t epoch_base;
    vram_telemetry_block_t* telemetry;

    /** Entries the launch's warps have assigned themselves so far, zeroed per launch. */
    u32_t* assigned_nodes;

    /** Each node's last pick a walker of the launch made, @b [nodes], unknown for none; pick
     *  @b p is the walker at offset p / connectivity, its slot p % connectivity. */
    node_id_type_* link_heads;

    /** The pick each pick was threaded before, @b [launch entries,connectivity]. */
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

    /** The queries' sketches a seeding walk scores by, a sketch apart, and its nearest node IDs
     *  per query, @b [queries,seeds_count], which a seeding walk writes and a search walk scores
     *  first; null and zero for a search without seeds. */
    byte_t const* query_sketches;
    node_id_type_* seed_ids;
    u32_t seeds_count;

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

    /** Walks running now, zeroed with @c assigned_queries, which @ref walk_width divides
     *  @c concurrent_walks among. */
    u32_t* live_walks;

    /** Walks the device runs at once on the launch's rung: multiprocessors × resident blocks ×
     *  warps per block. */
    u32_t concurrent_walks;

    /** Nodes whose neighbors fill a walk's candidates, the widest round a walk expands. */
    u32_t filling_nodes;

    /** The outputs for the matched IDs and distances, @b [queries,count] each. */
    node_id_type_* result_ids;
    distance_t* result_distances;
    u32_t count;

    /** The ranges a scan splits the nodes into for every query, zero for a walk; each range's
     *  nearest @c count matches, @b [queries,ranges,count]; and per query the ranges finished,
     *  zeroed per launch, so the last to finish merges them. */
    u32_t ranges;
    vram_scored_node<node_id_type_>* range_matches;
    u32_t* finished_ranges;
};

/** Splits a round of @p count entries of @p ids, past a warp's width and up to two, by entry bytes
 *  through @ref sort_lanes and one bitonic half-cleaner: @p split [0] takes the warp's width of
 *  longest entries, longest at lane 0, and @p split [1] the rest, shortest at lane 0 and sentinels
 *  past them, so lane @c l pairs the @c l -th longest entry with the @c l -th shortest. */
template <typename warp_type_, typename node_id_type_>
__device__ __forceinline__ void vram_split_by_bytes(vram_ragged_view<node_id_type_> const& storage,
                                                    node_id_type_ const* ids, u32_t count, u32_t lane,
                                                    vram_scored_node<node_id_type_> (&split)[2]) noexcept {
    using warp_t = warp_type_;
    using scored_node_t = vram_scored_node<node_id_type_>;
    constexpr u32_t width_k = warp_t::width_k;
    // Entries key by their bytes plus one, so the empty slots, keyed zero, sort below every entry.
    auto const keyed = [&](u32_t slot) noexcept {
        return slot < count
                   ? scored_node_t::make(static_cast<distance_t>(storage.stored_bytes(ids[slot])) + 1, ids[slot])
                   : scored_node_t::make(0, (std::numeric_limits<node_id_type_>::max)());
    };
    scored_node_t const descending = shuffle_xor<warp_t>(sort_lanes<warp_t>(keyed(lane), lane), width_k - 1);
    scored_node_t const ascending = sort_lanes<warp_t>(keyed(width_k + lane), lane);
    scored_node_t const longer = ascending < descending ? descending : ascending;
    scored_node_t const shorter = ascending < descending ? ascending : descending;
    split[0] = shuffle_xor<warp_t>(sort_lanes<warp_t>(longer, lane), width_k - 1);
    // The empty slots lead the ascending short entries, so the entries rotate down past them.
    u32_t const empty = 2 * width_k - count;
    scored_node_t const rotated = shuffle_from<warp_t>(sort_lanes<warp_t>(shorter, lane), (lane + empty) % width_k);
    split[1] = lane + empty < width_k ? rotated : scored_node_t::sentinel();
}

/** The distance from @p query to the entry @p node_id names on @p rung_ of @p kit_ for @p role_,
 *  exact within @p radius: infinite for the sentinel, the lower bound past @p radius; an entry's
 *  lanes share its ID, so every group takes one side of each branch whole. */
template <a_vram_basic_kit kit_, score_role_t role_, u32_t rung_, typename node_id_type_>
__device__ __forceinline__ distance_t vram_score_entry(vram_ragged_view<node_id_type_> const& storage,
                                                       vram_query<kit_, role_> const& query, node_id_type_ node_id,
                                                       distance_t radius, u32_t lane) noexcept {
    if (node_id == (std::numeric_limits<node_id_type_>::max)()) return std::numeric_limits<distance_t>::infinity();
    vram_entry_t const entry = storage.template scored<kit_, role_>(node_id);
    distance_t const lower_bound = vram_lower_bound<kit_, role_>(storage.kernel, query, entry);
    return lower_bound > radius ? lower_bound
                                : vram_score<kit_, role_, rung_>(storage.kernel, query, entry, radius, lane);
}

/** The distances from @p query to the entries @p first_id and @p second_id name on @p rung_ of
 *  @p kit_ for @p role_, exact within @p radius as @ref vram_score scores them: infinite for the
 *  sentinel, the lower bound past @p radius, and in one paired pass when both entries need scoring;
 *  every lane of the warp calls it with the same IDs. */
template <a_vram_basic_kit kit_, score_role_t role_, u32_t rung_, typename node_id_type_>
    requires a_vram_paired_kit<kit_, role_, rung_>
__device__ __forceinline__ vram_distance_pair_t vram_score_pair(vram_ragged_view<node_id_type_> const& storage,
                                                                vram_query<kit_, role_> const& query,
                                                                node_id_type_ first_id, node_id_type_ second_id,
                                                                distance_t radius, u32_t lane) noexcept {
    constexpr node_id_type_ sentinel_k = (std::numeric_limits<node_id_type_>::max)();
    vram_entry_t const first = first_id == sentinel_k ? vram_entry_t {}
                                                      : storage.template canonical_entry<kit_>(first_id);
    vram_entry_t const second = second_id == sentinel_k ? vram_entry_t {}
                                                        : storage.template canonical_entry<kit_>(second_id);
    vram_distance_pair_t distances {
        first_id == sentinel_k ? std::numeric_limits<distance_t>::infinity()
                               : vram_lower_bound<kit_, role_>(storage.kernel, query, first),
        second_id == sentinel_k ? std::numeric_limits<distance_t>::infinity()
                                : vram_lower_bound<kit_, role_>(storage.kernel, query, second)};
    // An unbounded radius holds the sentinel's infinity within it, so the ID decides.
    bool const first_scored = first_id != sentinel_k && !(distances.first > radius);
    bool const second_scored = second_id != sentinel_k && !(distances.second > radius);
    if (first_scored && second_scored)
        return kit_::template canonical_distance_pair<rung_>(storage.kernel, query, first, second, radius, radius,
                                                             lane);
    if (first_scored) distances.first = vram_score<kit_, role_, rung_>(storage.kernel, query, first, radius, lane);
    if (second_scored) distances.second = vram_score<kit_, role_, rung_>(storage.kernel, query, second, radius, lane);
    return distances;
}

/** What a lane schedule keeps in shared memory beside the walking warp's own spans: nothing, on
 *  every rung. */
struct vram_lane_round {};

/** Scores each lane's entry on @p rung_ of @p kit_ for @p role_ with the walking warp alone, so a
 *  warp walks one query by itself: in place when a lane scores an entry alone, in turns of two
 *  neighbouring entries for a kit scoring a pair with the whole warp, and in turns of
 *  @c entries_per_turn_k entries otherwise, @c lanes_per_entry_k lanes to each. */
template <a_vram_basic_kit kit_, score_role_t role_, u32_t rung_, typename node_id_type_>
struct vram_lane_schedule {
    using warp_t = typename kit_::warp_t;
    inline static constexpr score_role_t role_k = role_;
    using query_t = vram_query<kit_, role_>;
    inline static constexpr u32_t lanes_per_entry_k = vram_lanes_per_entry<kit_, role_, rung_>();

    /** Entries the warp scores at once, a group of @c lanes_per_entry_k lanes each. */
    inline static constexpr u32_t entries_per_turn_k = warp_t::width_k / lanes_per_entry_k;
    static_assert(warp_t::width_k % lanes_per_entry_k == 0, "An entry's lanes split the warp evenly");

    using shared_t = vram_lane_round;

    __device__ __forceinline__ explicit vram_lane_schedule(shared_t&) noexcept {}

    /** The distance from @p query to the entry this lane names, exact within @p radius; infinite
     *  for the sentinel. Every lane of the warp calls it, so its turns run with them all. */
    __device__ __forceinline__ distance_t score(vram_ragged_view<node_id_type_> const& storage, query_t const& query,
                                                node_id_type_ node_id, distance_t radius, u32_t lane) const noexcept {
        if constexpr (lanes_per_entry_k == 1)
            return vram_score_entry<kit_, role_, rung_>(storage, query, node_id, radius, lane);
        else if constexpr (a_vram_paired_kit<kit_, role_, rung_>) {
            // Neighbouring lanes hold entries alike in length once a round splits them by bytes.
            distance_t distance = std::numeric_limits<distance_t>::infinity();
#pragma unroll 1
            for (u32_t first = 0; first != warp_t::width_k; first += 2) {
                vram_distance_pair_t const turn = vram_score_pair<kit_, role_, rung_>(
                    storage, query, warp_t::shuffle(node_id, first), warp_t::shuffle(node_id, first + 1), radius, lane);
                if (lane == first) distance = turn.first;
                if (lane == first + 1) distance = turn.second;
            }
            return distance;
        }
        else {
            distance_t distance = std::numeric_limits<distance_t>::infinity();
#pragma unroll 1
            for (u32_t first = 0; first != warp_t::width_k; first += entries_per_turn_k) {
                node_id_type_ const turn_id = warp_t::shuffle(node_id, first + lane / lanes_per_entry_k);
                // Lanes outside the turn read any group's first lane and drop it.
                distance_t const turn_distance = warp_t::shuffle(
                    vram_score_entry<kit_, role_, rung_>(storage, query, turn_id, radius, lane),
                    (lane - first) % entries_per_turn_k * lanes_per_entry_k);
                if (lane - first < entries_per_turn_k) distance = turn_distance;
            }
            return distance;
        }
    }

    /** Scores into @p scored a round of @p count entries of @p ids, past a warp's width and up to
     *  two:
     *  lane @c l scores the @c l -th longest entry and then the @c l -th shortest, so the warp's
     *  two passes cost its longest entry and the longest of the short ones. */
    __device__ __forceinline__ void score_round(vram_ragged_view<node_id_type_> const& storage, query_t const& query,
                                                node_id_type_ const* ids, u32_t count, distance_t radius, u32_t lane,
                                                vram_scored_node<node_id_type_> (&scored)[2]) const noexcept {
        vram_split_by_bytes<warp_t>(storage, ids, count, lane, scored);
        for (vram_scored_node<node_id_type_>& entry : scored) {
            distance_t const distance = score(storage, query, entry.id(), radius, lane);
            if (!(entry == vram_scored_node<node_id_type_>::sentinel()))
                entry = vram_scored_node<node_id_type_>::make(distance, entry.id());
        }
    }
};

/**
 *  @brief Samples one lane-wide batch of distinct, published landmarks in
 *      @b [sampling_first,sampling_bound) into @p candidate_ids, drawn as the dense walk draws
 *      them.
 *
 *  @param[in] storage The graph's storage.
 *  @param[in] sampling_first The first node of the range sampled.
 *  @param[in] sampling_bound One past the last node of the range sampled.
 *  @param[in,out] visited The nodes this walk has scored already.
 *  @param[in] frontier The walk's frontier, asked whether it holds a draw once a saturated visited
 *      set re-scores it.
 *  @param[in,out] random_generator The draw state of this lane.
 *  @param[out] candidate_ids Receives the staged landmarks.
 *  @param[in] lane This lane's index in its warp.
 *  @return How many it staged: zero once every published node of the range is visited, or the
 *      visited set saturated.
 */
template <a_vram_basic_kit kit_, typename node_id_type_, a_vram_frontier frontier_>
__device__ __forceinline__ u32_t vram_sample_landmarks( //
    vram_ragged_view<node_id_type_> const& storage, node_id_type_ sampling_first, node_id_type_ sampling_bound,
    vram_visited_set<typename kit_::warp_t, node_id_type_>& visited, frontier_ const& frontier,
    splitmix64_t& random_generator, node_id_type_* candidate_ids, u32_t lane) noexcept {

    using warp_t = typename kit_::warp_t;
    constexpr node_id_type_ unknown_k = (std::numeric_limits<node_id_type_>::max)();
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
        // An unpublished node draws again; a visited one steps forward, one lap at most.
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

/** Scores one gathered batch against @p query through @p schedule and folds it into the candidates:
 *  a warp's width of entries an entry per lane, and a round past it, up to two, longest entries
 *  first. Past the full candidates' worst rank nothing enters, so each entry is scored within it;
 *  unfilled candidates bound nothing. No dedup: the exact visited set upstream stages every ID
 *  once. */
template <a_vram_basic_kit kit_, typename schedule_, a_vram_frontier frontier_, typename node_id_type_>
__device__ __forceinline__ void vram_score_into_frontier( //
    vram_ragged_view<node_id_type_> const& storage, schedule_& schedule, typename schedule_::query_t const& query,
    node_id_type_ const* gathered_ids, u32_t gathered_count, vram_warp_context<frontier_>& context,
    u32_t lane) noexcept {
    using scored_node_t = vram_scored_node<node_id_type_>;
    context.telemetry.scored += gathered_count;
    scored_node_t const worst = context.frontier.worst();
    distance_t const radius = worst == scored_node_t::sentinel() ? std::numeric_limits<distance_t>::infinity()
                                                                 : worst.distance();
    if (gathered_count > kit_::warp_t::width_k) {
        scored_node_t scored[2];
        schedule.score_round(storage, query, gathered_ids, gathered_count, radius, lane, scored);
        context.frontier.insert_batch(scored[0], lane);
        context.frontier.insert_batch(scored[1], lane);
        return;
    }
    node_id_type_ const node_id = lane < gathered_count ? gathered_ids[lane]
                                                        : (std::numeric_limits<node_id_type_>::max)();
    distance_t const distance = schedule.score(storage, query, node_id, radius, lane);
    context.frontier.insert_batch(
        lane < gathered_count ? scored_node_t::make(distance, node_id) : scored_node_t::sentinel(), lane);
}

/** Scores into the frontier the seeds the warp's lanes name, one each, the sentinel for none: the
 *  ones the walk visited already drop, and the rest score as one gathered batch. */
template <a_vram_basic_kit kit_, typename schedule_, a_vram_frontier frontier_, typename node_id_type_>
__device__ __forceinline__ void vram_score_seeds( //
    vram_ragged_view<node_id_type_> const& storage, schedule_& schedule, typename schedule_::query_t const& query,
    node_id_type_ seed_id, vram_visited_set<typename kit_::warp_t, node_id_type_>& visited, node_id_type_* gathered_ids,
    vram_warp_context<frontier_>& context, u32_t lane) noexcept {
    using warp_t = typename kit_::warp_t;
    bool const fresh = visited.test_and_set(seed_id, lane);
    typename warp_t::mask_t const fresh_mask = warp_t::ballot(fresh);
    if (fresh) gathered_ids[warp_t::count(fresh_mask & warp_t::below(lane))] = seed_id;
    warp_t::sync();
    u32_t const gathered_count = warp_t::count(fresh_mask);
    if (gathered_count)
        vram_score_into_frontier<kit_>(storage, schedule, query, gathered_ids, gathered_count, context, lane);
    warp_t::sync();
}

/** Continues a greedy best-first walk toward @p query over whatever the frontier holds, as the
 *  dense walk does: landmarks from @b [sampling_first,sampling_bound), a transiently locked
 *  candidate held through one other expansion and dropped if still locked then, and no neighbor at
 *  or past @p first_invisible_id followed. Read-only on the graph. */
template <a_vram_basic_kit kit_, typename schedule_, a_vram_frontier frontier_, typename node_id_type_>
__device__ __forceinline__ void vram_search_ahead( //
    vram_ragged_view<node_id_type_> const& storage, schedule_& schedule, typename schedule_::query_t const& query,
    node_id_type_ sampling_first, node_id_type_ sampling_bound, node_id_type_ first_invisible_id,
    vram_visited_set<typename kit_::warp_t, node_id_type_>& visited, node_id_type_* gathered_ids,
    vram_warp_context<frontier_>& context, u32_t lane) noexcept {

    using warp_t = typename kit_::warp_t;
    using scored_node_t = vram_scored_node<node_id_type_>;
    constexpr node_id_type_ unknown_k = (std::numeric_limits<node_id_type_>::max)();
    frontier_& frontier = context.frontier;
    vram_telemetry_t& telemetry = context.telemetry;
    scored_node_t held = scored_node_t::sentinel();
    u32_t held_at_hop = 0, pending = 0;
    // Seeds already scored take the first random batch's place, as on RAM.
    bool seeding = frontier.nearest() == scored_node_t::sentinel();
    auto const score = [&](u32_t count) noexcept {
        vram_score_into_frontier<kit_>(storage, schedule, query, gathered_ids, count, context, lane);
        return walk_demand_t::continue_k;
    };
    while (true) {
        if (seeding) {
            u32_t const gathered_count = vram_sample_landmarks<kit_>( //
                storage, sampling_first, sampling_bound, visited, frontier, context.random_generator, gathered_ids,
                lane);
            if (!gathered_count) break; // Every published node of the range is already visited.
            vram_score_into_frontier<kit_>(storage, schedule, query, gathered_ids, gathered_count, context, lane);
            seeding = false;
            continue;
        }
        bool const revisiting = !(held == scored_node_t::sentinel()) && telemetry.hops != held_at_hop;
        scored_node_t const candidate = revisiting ? held : frontier.pop_nearest(lane);
        if (revisiting) held = scored_node_t::sentinel();
        // Neighbors staged across hops score before the walk stops or reseeds, since they may
        // refill it.
        if (pending && (candidate == scored_node_t::sentinel() || frontier.beyond_radius(candidate))) {
            score(pending);
            pending = 0;
            warp_t::sync();
            continue;
        }
        if (candidate == scored_node_t::sentinel()) {
            if (!(held == scored_node_t::sentinel())) {
                held_at_hop = ~0u;
                continue;
            }
            // A drained frontier before an exhausted walk re-seeds, until no published node of the
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
        vram_expand_round<warp_t>(frontier, visited, telemetry, storage, candidate, first_value, context.walk_width,
                                  first_invisible_id, held, held_at_hop, gathered_ids, pending, warp_t::width_k, lane,
                                  score);
    }
    telemetry.locked_drops += !(held == scored_node_t::sentinel());
}

/** Greedy best-first walk from random published landmarks toward @p query: a fresh frontier handed
 *  to @ref vram_search_ahead. */
template <a_vram_basic_kit kit_, typename schedule_, a_vram_frontier frontier_, typename node_id_type_>
__device__ __forceinline__ void vram_search_around( //
    vram_ragged_view<node_id_type_> const& storage, schedule_& schedule, typename schedule_::query_t const& query,
    node_id_type_ sampling_bound, node_id_type_ first_invisible_id,
    vram_visited_set<typename kit_::warp_t, node_id_type_>& visited, node_id_type_* gathered_ids,
    vram_warp_context<frontier_>& context, u32_t lane) noexcept {
    context.frontier.clear(lane);
    vram_search_ahead<kit_>(storage, schedule, query, node_id_type_ {0}, sampling_bound, first_invisible_id, visited,
                            gathered_ids, context, lane);
}

/**
 *  @brief The diversity rule of the CPU's @c keep_diverse_t and its @c fill_selection: a candidate
 *      is kept only when it is closer to the node than to every kept neighbor, and, where
 *      @p refill asks, the slots the rule leaves empty take the nearest rejected candidates, after
 *      the diverse picks; an insertion refills, a round's restitch never does.
 *
 *  A selector keeps neighbor rows of at most @c connectivity neighbors from candidates ascending by
 *  distance to the node, whose prepared @p query it may read; it writes the kept candidates'
 *  positions into @p kept_positions and answers their count, every lane of the walking warp taking
 *  part.
 */
struct vram_keep_diverse_t {

    /** Prepares each candidate as the pivot in @p pivot_span, which scores the kept ones through
     *  @p schedule, exiting each candidate on its first violation. */
    template <a_vram_basic_kit kit_, typename schedule_, typename node_id_type_>
    static __device__ __forceinline__ u32_t select( //
        vram_ragged_view<node_id_type_> const& storage, schedule_& schedule, typename schedule_::query_t const&,
        byte_t* pivot_span, node_id_type_ const* candidate_ids, f32_t const* candidate_distances,
        u32_t candidates_count, u32_t* kept_positions, vram_telemetry_t& telemetry, u32_t lane,
        bool refill = true) noexcept {

        using warp_t = typename kit_::warp_t;
        u32_t kept_count = 0;
#pragma unroll 1
        for (u32_t candidate_index = 0; candidate_index < candidates_count && kept_count < storage.connectivity;
             ++candidate_index) {
            f32_t const to_query = candidate_distances[candidate_index];
            typename schedule_::query_t const pivot =
                kept_count ? vram_prepare<kit_, schedule_::role_k>(
                                 storage.parameters,
                                 storage.template scored<kit_, schedule_::role_k>(candidate_ids[candidate_index]),
                                 pivot_span, lane)
                           : typename schedule_::query_t {};
            bool diverse = true;
            for (u32_t base = 0; base < kept_count && diverse; base += warp_t::width_k) { // Warp-uniform bounds.
                node_id_type_ const kept_id = base + lane < kept_count ? candidate_ids[kept_positions[base + lane]]
                                                                       : (std::numeric_limits<node_id_type_>::max)();
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
             refill && candidate_index < candidates_count && kept_count < storage.connectivity; ++candidate_index) {
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
 *  Hands all of a finished walk's candidates toward @p query, its @p selection_nodes ranks
 *  ascending, to @p selector_, staged as candidate IDs and distances, and writes the kept
 *  candidates as node IDs in @p selection.selected_ids, through their positions in
 *  @p selection.kept_positions.
 *  @return The kept count.
 */
template <a_vram_basic_kit kit_, typename selector_, typename schedule_, a_vram_frontier frontier_,
          typename node_id_type_>
__device__ __forceinline__ u32_t vram_select_neighbors( //
    vram_ragged_view<node_id_type_> const& storage, schedule_& schedule, typename schedule_::query_t const& query,
    byte_t* pivot_span, u32_t selection_nodes, vram_selection_scratch<node_id_type_> selection,
    vram_warp_context<frontier_>& context, u32_t lane) noexcept {

    using warp_t = typename kit_::warp_t;
    using scored_node_t = vram_scored_node<node_id_type_>;
    node_id_type_* const candidate_ids = selection.candidate_ids;
    u32_t candidates_count = 0;
    context.frontier.for_each_rank(selection_nodes, lane, [&](u32_t rank, scored_node_t entry) noexcept {
        bool const live = rank < selection_nodes && !(entry == scored_node_t::sentinel());
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
    // The caller links to nodes, not to positions in the candidates.
    for (u32_t slot = lane; slot < selected_count; slot += warp_t::width_k)
        selection.selected_ids[slot] = candidate_ids[selection.kept_positions[slot]];
    warp_t::sync();
    return selected_count;
}

/** Threads the @p selected_count picks of the walker at @p offset onto their targets' chains,
 *  listing each target the first time any walker of the launch picks it; the back-link launch
 *  behind merges them all. */
template <typename warp_type_, typename node_id_type_>
__device__ __forceinline__ void vram_thread_back_links(vram_build_view<node_id_type_> const& view, u32_t offset,
                                                       node_id_type_ const* selected_ids, u32_t selected_count,
                                                       u32_t lane) noexcept {
    for (u32_t slot = lane; slot < selected_count; slot += warp_type_::width_k) {
        node_id_type_ const target = selected_ids[slot];
        node_id_type_ const pick = offset * view.storage.connectivity + slot;
        node_id_type_ const previous = warp_type_::exchange_acquire(view.link_heads[target], pick);
        view.link_next[pick] = previous;
        if (previous == (std::numeric_limits<node_id_type_>::max)())
            view.touched_ids[warp_type_::fetch_add_relaxed(*view.touched_count, 1u)] = target;
    }
    warp_type_::sync();
}

/** Inserts the node identified by @p new_node_id, whose bytes already sit in the values tape:
 *  writes its two entry offsets, walks the published graph, selects diverse neighbors, publishes
 *  with one release store, and threads its picks for the back-link launch, one warp per node. */
template <a_vram_basic_kit kit_, typename selector_ = vram_keep_diverse_t, typename schedule_,
          a_vram_frontier frontier_,
          typename node_id_type_>
__device__ __forceinline__ void vram_insert_one( //
    vram_build_view<node_id_type_> const& view, schedule_& schedule, node_id_type_ new_node_id,
    vram_visited_set<typename kit_::warp_t, node_id_type_>& visited, byte_t* query_span, byte_t* pivot_span,
    node_id_type_* gathered_ids, vram_selection_scratch<node_id_type_> selection, vram_warp_context<frontier_>& context,
    u32_t lane) noexcept {

    using warp_t = typename kit_::warp_t;
    vram_ragged_view<node_id_type_> const& storage = view.storage;

    // Both bounds are written, so whichever neighbor is written first, a reader past publication
    // sees both bounds.
    u32_t const offset = static_cast<u32_t>(new_node_id - view.first_id);
    u64_t const origin = view.staged_offsets[0];
    u64_t const row_begin = view.tape_first + view.staged_offsets[offset] - origin;
    u64_t const row_end = view.tape_first + view.staged_offsets[offset + 1] - origin;
    if (lane == 0) {
        warp_t::store_relaxed(storage.offsets[new_node_id], row_begin);
        warp_t::store_relaxed(storage.offsets[new_node_id + 1], row_end);
    }
    vram_query<kit_, score_role_t::build_k> const query = vram_prepare<kit_, score_role_t::build_k>(
        storage.parameters,
        vram_part<kit_, score_role_t::build_k>(storage.values, row_begin, row_end, storage.sketch_bytes), query_span,
        lane);

    // A virgin index publishes an empty node instead of walking.
    if (new_node_id != 0)
        vram_search_around<kit_>(storage, schedule, query, new_node_id, view.end_id, visited, gathered_ids, context,
                                 lane);
    else context.frontier.clear(lane);

    // No lock on the own row: no reader arrives before the slot-0 release flips the sentinel.
    u32_t const selected_count = vram_select_neighbors<kit_, selector_>( //
        storage, schedule, query, pivot_span, view.selection_nodes, selection, context, lane);
    vram_neighbors_ref<warp_t, node_id_type_> const own_neighbors {storage.neighbors_for(new_node_id),
                                                                   storage.connectivity};
    for (u32_t slot = lane; slot < selected_count; slot += warp_t::width_k)
        if (slot > 0) own_neighbors.store_slot(slot, selection.selected_ids[slot]);
    warp_t::sync();
    node_id_type_ const first_link = selected_count ? selection.selected_ids[0] : new_node_id;
    if (lane == 0) own_neighbors.unlock(first_link); // A self-loop encodes published-empty.
    vram_thread_back_links<warp_t>(view, offset, selection.selected_ids, selected_count, lane);
}

/**
 *  @brief Back-links one node the launch's walkers picked, chasing their picks from @c link_heads
 *      through @c link_next in rounds.
 *
 *  Each round caches the live row and as many walkers as the view's @c selection_nodes leave room
 *  for, appends them while the row has room, and re-selects over the union otherwise. The append
 *  is the dense back-linker's fast path; the re-selection hands @p selector_ the union ascending
 *  by distance to the node, and the next round starts from the row the last one wrote, so no
 *  walker is dropped. The launch hands each node to one warp, the only writer of its row, so it
 *  reads the row plainly, while walks on other streams may read it throughout: every slot changes
 *  by one relaxed store and slot 0 last, with a release.
 */
template <a_vram_basic_kit kit_, typename selector_, typename schedule_, typename node_id_type_>
__device__ __forceinline__ void vram_back_link_one( //
    vram_build_view<node_id_type_> const& view, schedule_& schedule, node_id_type_ node_id, byte_t* query_span,
    byte_t* pivot_span, vram_selection_scratch<node_id_type_> selection, vram_telemetry_t& telemetry,
    u32_t lane) noexcept {

    using warp_t = typename kit_::warp_t;
    using scored_node_t = vram_scored_node<node_id_type_>;
    constexpr node_id_type_ unknown_k = (std::numeric_limits<node_id_type_>::max)();
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
            for (; pick != unknown_k && chained_end != view.selection_nodes; pick = view.link_next[pick])
                candidate_ids[chained_end++] = view.listed ? view.listed[pick / connectivity]
                                                           : view.first_id + pick / connectivity;
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

        vram_query<kit_, score_role_t::build_k> const query = vram_prepare<kit_, score_role_t::build_k>(
            storage.parameters, storage.template scored<kit_, score_role_t::build_k>(node_id), query_span, lane);
        vram_sorted_buffer<warp_t, node_id_type_> sorted_candidates {selection.scored_entries, 0, view.selection_nodes};
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
    }
}

/** Re-selects the published row of @p node_id, the walker at @p offset, once its wave's back-links
 *  made the wave's nodes reachable from one another: walks from the row's own neighbors over every
 *  node below the view's sampling bound and selects again, counts the row changed when it gains a
 *  member, threads the new picks, and rewrites the row in place, tail slots relaxed and slot 0 last
 *  with a release, so walks on other streams read either row. */
template <a_vram_basic_kit kit_, typename selector_ = vram_keep_diverse_t, typename schedule_,
          a_vram_frontier frontier_,
          typename node_id_type_>
__device__ __forceinline__ void vram_reselect_one( //
    vram_build_view<node_id_type_> const& view, schedule_& schedule, node_id_type_ node_id, u32_t offset,
    vram_visited_set<typename kit_::warp_t, node_id_type_>& visited, byte_t* query_span, byte_t* pivot_span,
    node_id_type_* gathered_ids, vram_selection_scratch<node_id_type_> selection, vram_warp_context<frontier_>& context,
    u32_t lane) noexcept {

    using warp_t = typename kit_::warp_t;
    constexpr node_id_type_ unknown_k = (std::numeric_limits<node_id_type_>::max)();
    vram_ragged_view<node_id_type_> const& storage = view.storage;
    vram_query<kit_, score_role_t::build_k> const query = vram_prepare<kit_, score_role_t::build_k>(
        storage.parameters, storage.template scored<kit_, score_role_t::build_k>(node_id), query_span, lane);
    context.frontier.clear(lane);
    visited.test_and_set(node_id, lane);

    // The walk starts from the row's own neighbors, which the wave's first walk already found.
    vram_neighbors_ref<warp_t, node_id_type_> const own_neighbors {storage.neighbors_for(node_id),
                                                                   storage.connectivity};
    for (u32_t slot_base = 0; slot_base < storage.connectivity; slot_base += warp_t::width_k) {
        u32_t const slot = slot_base + lane;
        node_id_type_ const seed_id = slot < storage.connectivity ? own_neighbors.load_slot(slot) : unknown_k;
        bool const usable = seed_id < view.sampling_bound && seed_id != node_id;
        vram_score_seeds<kit_>(storage, schedule, query, usable ? seed_id : unknown_k, visited, gathered_ids, context,
                               lane);
    }
    vram_search_ahead<kit_>(storage, schedule, query, node_id_type_ {0}, view.sampling_bound, view.sampling_bound,
                            visited, gathered_ids, context, lane);

    u32_t const selected_count = vram_select_neighbors<kit_, selector_>( //
        storage, schedule, query, pivot_span, view.selection_nodes, selection, context, lane);
    bool gained = false;
    for (u32_t slot = lane; slot < selected_count; slot += warp_t::width_k) {
        node_id_type_ const picked = selection.selected_ids[slot];
        bool held = false;
        for (u32_t live_slot = 0; live_slot != storage.connectivity && !held; ++live_slot)
            held = own_neighbors.load_slot(live_slot) == picked;
        gained = gained || !held;
    }
    context.telemetry.changed_nodes += warp_t::any(gained) ? 1u : 0u;
    for (u32_t slot = lane; slot < storage.connectivity; slot += warp_t::width_k)
        if (slot) own_neighbors.store_slot(slot, slot < selected_count ? selection.selected_ids[slot] : unknown_k);
    warp_t::sync();
    if (lane == 0) own_neighbors.unlock(selected_count ? selection.selected_ids[0] : node_id);
    vram_thread_back_links<warp_t>(view, offset, selection.selected_ids, selected_count, lane);
}

/** One query's full walk for @p role_: a search prepares the query into its span, scores its seeds
 *  first where the view holds any, walks the graph, and writes its nearest @c count IDs and
 *  distances, ascending, sentinel IDs past the live prefix; a seeding walk scores the query's
 *  sketch and writes its nearest @c seeds_count IDs as the query's seeds. Safe against concurrent
 *  insertions: the walk gates on every slot-0 word and never writes the graph. */
template <a_vram_basic_kit kit_, score_role_t role_, typename schedule_, a_vram_frontier frontier_,
          typename node_id_type_>
__device__ __forceinline__ void vram_walk_one( //
    vram_search_view<node_id_type_> const& view, schedule_& schedule, u32_t query_index, byte_t* query_span,
    vram_visited_set<typename kit_::warp_t, node_id_type_>& visited, frontier_ frontier, node_id_type_* gathered_ids,
    u32_t lane) noexcept {

    using warp_t = typename kit_::warp_t;
    using scored_node_t = vram_scored_node<node_id_type_>;
    constexpr node_id_type_ unknown_k = (std::numeric_limits<node_id_type_>::max)();
    u64_t const origin = view.query_offsets[0];
    u64_t const first = view.query_offsets[query_index] - origin;
    vram_entry_t query_entry {view.queries + first,
                              static_cast<u32_t>(view.query_offsets[query_index + 1] - origin - first)};
    if constexpr (vram_scores_by_sketch<kit_, role_>())
        query_entry = {view.query_sketches + std::size_t {query_index} * view.storage.sketch_bytes,
                       view.storage.sketch_bytes};
    vram_query<kit_, role_> const query = vram_prepare<kit_, role_>(view.storage.parameters, query_entry, query_span,
                                                                    lane);

    vram_warp_context<frontier_> context;
    context.frontier = frontier;
    context.random_generator = vram_lane_generator<warp_t>(query_index, lane);
    // Concurrent walks the batch leaves idle widen this one, never past its filling nodes.
    u32_t live = lane == 0 ? warp_t::fetch_add_relaxed(*view.live_walks, 1u) + 1 : 0;
    live = warp_t::shuffle(live, 0);
    context.walk_width = walk_width(view.concurrent_walks, live, view.queries_count - query_index, view.filling_nodes);
    context.frontier.clear(lane);
    if constexpr (role_ == score_role_t::search_k) {
        node_id_type_ const* const seeds = vram_row(view.seed_ids, query_index, view.seeds_count);
        for (u32_t slot_base = 0; slot_base < view.seeds_count; slot_base += warp_t::width_k) {
            u32_t const slot = slot_base + lane;
            node_id_type_ const seed_id = slot < view.seeds_count ? seeds[slot] : unknown_k;
            vram_score_seeds<kit_>(view.storage, schedule, query, seed_id < view.sampling_bound ? seed_id : unknown_k,
                                   visited, gathered_ids, context, lane);
        }
    }
    vram_search_ahead<kit_>(view.storage, schedule, query, node_id_type_ {0}, view.sampling_bound, view.sampling_bound,
                            visited, gathered_ids, context, lane);
    if (lane == 0) warp_t::fetch_sub_relaxed(*view.live_walks, 1u);

    // The rank extraction shuffles, so every lane participates and only in-range ones store.
    if constexpr (role_ == score_role_t::search_k) {
        node_id_type_* const ids_row = vram_row(view.result_ids, query_index, view.count);
        distance_t* const distances_row = vram_row(view.result_distances, query_index, view.count);
        context.frontier.for_each_rank(view.count, lane, [&](u32_t rank, scored_node_t entry) noexcept {
            if (rank >= view.count) return;
            ids_row[rank] = entry.id();
            distances_row[rank] = entry.distance();
        });
    }
    else {
        node_id_type_* const seeds_row = vram_row(view.seed_ids, query_index, view.seeds_count);
        context.frontier.for_each_rank(view.seeds_count, lane, [&](u32_t rank, scored_node_t entry) noexcept {
            if (rank < view.seeds_count) seeds_row[rank] = entry.id();
        });
    }
    context.telemetry.tail_compactions = context.frontier.compactions();
    context.telemetry.tail_dropped = context.frontier.dropped();
    context.telemetry.visited_saturations = visited.saturations_;
    constexpr vram_traversal_t traversal_k = role_ == score_role_t::search_k ? vram_traversal_t::searching_k
                                                                             : vram_traversal_t::seeding_k;
    context.telemetry.template flush<warp_t, traversal_k>(view.telemetry, lane);
}

/** One range of one query's exact scan, ranges counting @p item past the queries: the query
 *  prepares into its span, every published node of the range scores against it a round of two
 *  lane-widths at a time within the worst kept match, and the range's nearest @c count matches
 *  land in its row of @c range_matches. The last range of a query to finish merges every range's
 *  row into the query's outputs, ascending, sentinel IDs past the live prefix. Read-only on the
 *  graph, reading every slot-0 word as a walk does. */
template <a_vram_basic_kit kit_, typename schedule_, a_vram_frontier frontier_, typename node_id_type_>
__device__ __forceinline__ void vram_scan_one( //
    vram_search_view<node_id_type_> const& view, schedule_& schedule, u32_t item, byte_t* query_span,
    frontier_ frontier, node_id_type_* gathered_ids, u32_t lane) noexcept {

    using warp_t = typename kit_::warp_t;
    using scored_node_t = vram_scored_node<node_id_type_>;
    constexpr node_id_type_ unknown_k = (std::numeric_limits<node_id_type_>::max)();
    // Ranges lead, so the walkers in flight read the same nodes.
    u32_t const query_index = item % view.queries_count, range = item / view.queries_count;
    u64_t const origin = view.query_offsets[0];
    u64_t const first = view.query_offsets[query_index] - origin;
    vram_entry_t const query_entry {view.queries + first,
                                    static_cast<u32_t>(view.query_offsets[query_index + 1] - origin - first)};
    vram_query<kit_, score_role_t::search_k> const query = vram_prepare<kit_, score_role_t::search_k>(
        view.storage.parameters, query_entry, query_span, lane);

    vram_warp_context<frontier_> context;
    context.frontier = frontier;
    context.frontier.clear(lane);
    u64_t const range_nodes = divide_round_up<u64_t>(view.sampling_bound, view.ranges);
    u64_t const range_first = min_of<u64_t>(range * range_nodes, view.sampling_bound);
    u64_t const range_end = min_of<u64_t>(range_first + range_nodes, view.sampling_bound);
    for (u64_t base = range_first; base < range_end; base += 2 * warp_t::width_k) { // Warp-uniform bounds.
        u32_t staged = 0;
        for (u32_t half = 0; half != 2; ++half) {
            u64_t const node = base + half * warp_t::width_k + lane;
            bool published = false;
            if (node < range_end) {
                vram_neighbors_ref<warp_t, node_id_type_> const row {
                    view.storage.neighbors_for(static_cast<node_id_type_>(node)), view.storage.connectivity};
                published = row.peek_first() != unknown_k;
            }
            typename warp_t::mask_t const published_mask = warp_t::ballot(published);
            if (published)
                gathered_ids[staged + warp_t::count(published_mask & warp_t::below(lane))] = static_cast<node_id_type_>(
                    node);
            staged += warp_t::count(published_mask);
        }
        warp_t::sync();
        if (staged) vram_score_into_frontier<kit_>(view.storage, schedule, query, gathered_ids, staged, context, lane);
        warp_t::sync();
    }

    u32_t const count = view.count, ranges = view.ranges;
    scored_node_t* const query_matches = view.range_matches + std::size_t {query_index} * ranges * count;
    scored_node_t* const range_row = query_matches + std::size_t {range} * count;
    context.frontier.for_each_rank(count, lane, [&](u32_t rank, scored_node_t entry) noexcept {
        if (rank < count) range_row[rank] = entry;
    });
    context.telemetry.tail_compactions = context.frontier.compactions();
    context.telemetry.tail_dropped = context.frontier.dropped();
    // Every range's matches are written before the finish that lets the last range read them.
    warp_t::fence_release();
    warp_t::sync();
    u32_t finished = lane == 0 ? warp_t::fetch_add_relaxed(view.finished_ranges[query_index], 1u) + 1 : 0;
    finished = warp_t::shuffle(finished, 0);
    if (finished == ranges) {
        // Reading the last count acquires every other range's release.
        static_cast<void>(warp_t::load_acquire(view.finished_ranges[query_index]));
        context.frontier.clear(lane);
        for (u32_t base = 0; base < ranges * count; base += warp_t::width_k) // Warp-uniform bounds.
            context.frontier.insert_batch(
                base + lane < ranges * count ? query_matches[base + lane] : scored_node_t::sentinel(), lane);
        node_id_type_* const ids_row = vram_row(view.result_ids, query_index, count);
        distance_t* const distances_row = vram_row(view.result_distances, query_index, count);
        context.frontier.for_each_rank(count, lane, [&](u32_t rank, scored_node_t entry) noexcept {
            if (rank >= count) return;
            ids_row[rank] = entry.id();
            distances_row[rank] = entry.distance();
        });
    }
    context.telemetry.template flush<warp_t, vram_traversal_t::searching_k>(view.telemetry, lane);
}

#pragma region Rounds

/** One gather or restitch launch of a round as its warps read it: the warps' spans and selection
 *  scratch as a build lays them out, the runs a gather files, the neighborhoods a restitch reads,
 *  and the health check's columns the reach rule reads and the counters take the round's counts. */
template <typename node_id_type_>
struct vram_round_view {
    vram_ragged_view<node_id_type_> storage;

    /** Bytes of one span, a multiple of 16, and of the warp's dynamic-shared slice, zero when the
     *  arena shape holds it instead. */
    u32_t span_bytes;
    u32_t scratch_bytes_per_warp;
    byte_t* arena;
    vram_arena_shape_t arena_shape;

    /** Rows of each selection span: the first half by candidate, the second half ascending. */
    u32_t selection_nodes;
    vram_telemetry_block_t* telemetry;

    /** Positions the launch's warps assigned themselves so far, zeroed per launch. */
    u32_t* assigned;

    /** The entries in neighborhood order, or null in the first gather, whose one run spans IDs. */
    u32_t const* entries;

    /** The hubs of every run back to back, the runs, the entries before each run, one more than
     *  the runs, and per entry the hub of its run nearest to it, which a gather writes. */
    u32_t const* hub_nodes;
    vram_neighborhood_run_t const* runs;
    u32_t const* run_entries;
    u32_t runs_count;
    u16_t* entry_neighborhoods;

    /** The neighborhoods' starts, one more than the neighborhoods, which a restitch reads. */
    u32_t const* neighborhood_offsets;
    u32_t neighborhoods_count;

    /** The check's columns the reach rule reads and takes from, its changed bits and counters. */
    vram_health_t health;
};

/** The span among @p count ascending starts @p starts holding @p position, the last whose start is
 *  at or before it: a gather's run, or a restitch's neighborhood. */
__device__ __forceinline__ u32_t vram_span_of(u32_t const* starts, u32_t count, u32_t position) noexcept {
    u32_t found = 0;
    for (u32_t remaining = count; remaining;) {
        u32_t const half = remaining / 2;
        if (starts[found + half + 1] <= position) found += half + 1, remaining -= half + 1;
        else remaining = half;
    }
    return found;
}

/**
 *  @brief Files the entry at @p position of a gather's runs under its run's nearest hub, the lower
 *      one among equals, as RAM's gather files them: the entry prepares as the query, and every
 *      lane scores a hub within the nearest so far.
 *
 *  An entry that no distance reached joins its run's first hub.
 */
template <a_vram_basic_kit kit_, typename schedule_, typename node_id_type_>
__device__ __forceinline__ void vram_gather_one(vram_round_view<node_id_type_> const& view, schedule_& schedule,
                                                u32_t position, byte_t* query_span, vram_telemetry_t& telemetry,
                                                u32_t lane) noexcept {
    static_assert(std::is_same_v<node_id_type_, u32_t>, "A round's records hold 32-bit IDs");
    using warp_t = typename kit_::warp_t;
    using scored_t = vram_scored_node<node_id_type_>;
    constexpr node_id_type_ unknown_k = (std::numeric_limits<node_id_type_>::max)();
    vram_ragged_view<node_id_type_> const& storage = view.storage;
    u32_t const run_index = vram_span_of(view.run_entries, view.runs_count, position);
    vram_neighborhood_run_t const run = view.runs[run_index];
    u32_t const offset = position - view.run_entries[run_index];
    node_id_type_ const node_id = view.entries ? view.entries[run.first + offset] : run.first + offset;
    vram_query<kit_, score_role_t::build_k> const query = vram_prepare<kit_, score_role_t::build_k>(
        storage.parameters, storage.template scored<kit_, score_role_t::build_k>(node_id), query_span, lane);
    u32_t const* const hubs = view.hub_nodes + run.first_hub;
    scored_t nearest = scored_t::sentinel();
    distance_t radius = std::numeric_limits<distance_t>::infinity();
    for (u32_t base = 0; base < run.hubs; base += warp_t::width_k) { // Warp-uniform bounds.
        u32_t const hub = base + lane;
        node_id_type_ const hub_id = hub < run.hubs ? hubs[hub] : unknown_k;
        distance_t const distance = schedule.score(storage, query, hub_id, radius, lane);
        // Past the radius a score is a bound, never a nearer hub.
        if (hub < run.hubs && !(distance > radius)) {
            scored_t const scored = scored_t::make(distance, hub);
            if (scored < nearest) nearest = scored;
        }
        nearest = vram_reduce_least<warp_t>(nearest);
        if (!(nearest == scored_t::sentinel())) radius = nearest.distance();
    }
    telemetry.scored += run.hubs;
    if (lane == 0)
        view.entry_neighborhoods[node_id] = static_cast<u16_t>(nearest == scored_t::sentinel() ? 0 : nearest.id());
    warp_t::sync();
}

/**
 *  @brief Re-stitches the entry at @p position of the neighborhoods: its live links and its
 *      nearest neighborhood-mates no live link names compete, ascending; only those @p selector_
 *      keeps stay, with no refill, then those the health check's reach rule needs.
 *
 *  The live links lead the first half of the selection's candidates and the mates follow them, as
 *  @ref unum::usearch::utilities::vram_keep_reach reads them; the second half holds them ascending,
 *  as the selector reads them. The row is rewritten only where it changed, tail slots relaxed and
 *  slot 0 last with a release, so walks on other streams read it whole, and counted once in the
 *  changed bits; its additions and drops add to @p added and @p dropped, identical on every lane.
 */
template <a_vram_basic_kit kit_, typename selector_, typename schedule_, typename node_id_type_>
__device__ __forceinline__ void vram_restitch_one(vram_round_view<node_id_type_> const& view, schedule_& schedule,
                                                  u32_t position, byte_t* query_span, byte_t* pivot_span,
                                                  vram_selection_scratch<node_id_type_> selection,
                                                  vram_telemetry_t& telemetry, u32_t& added, u32_t& dropped,
                                                  u32_t& changed, u32_t lane) noexcept {
    static_assert(std::is_same_v<node_id_type_, u32_t>, "A round's records hold 32-bit IDs");
    using warp_t = typename kit_::warp_t;
    using scored_t = vram_scored_node<node_id_type_>;
    constexpr node_id_type_ unknown_k = (std::numeric_limits<node_id_type_>::max)();
    vram_ragged_view<node_id_type_> const& storage = view.storage;
    vram_health_t const& health = view.health;
    u32_t const connectivity = storage.connectivity, half = view.selection_nodes / 2;
    u32_t const neighborhood = vram_span_of(view.neighborhood_offsets, view.neighborhoods_count, position);
    u32_t const first = view.neighborhood_offsets[neighborhood];
    u32_t const count = view.neighborhood_offsets[neighborhood + 1] - first;
    node_id_type_ const node_id = view.entries[position];
    node_id_type_* const candidate_ids = selection.candidate_ids;
    f32_t* const candidate_distances = selection.candidate_distances;
    vram_neighbors_ref<warp_t, node_id_type_> const live_row {storage.neighbors_for(node_id), connectivity};

    // The live links first; a self-loop in slot 0 means none.
    u32_t live_count = 0;
    for (u32_t slot_base = 0; slot_base < connectivity; slot_base += warp_t::width_k) {
        u32_t const slot = slot_base + lane;
        node_id_type_ const live_id = slot < connectivity ? live_row.load_slot(slot) : unknown_k;
        bool const usable = live_id != unknown_k && live_id != node_id && live_id < health.rows_count;
        typename warp_t::mask_t const usable_mask = warp_t::ballot(usable);
        if (usable) candidate_ids[live_count + warp_t::count(usable_mask & warp_t::below(lane))] = live_id;
        live_count += warp_t::count(usable_mask);
        warp_t::sync();
    }
    vram_query<kit_, score_role_t::build_k> const query = vram_prepare<kit_, score_role_t::build_k>(
        storage.parameters, storage.template scored<kit_, score_role_t::build_k>(node_id), query_span, lane);

    // Then the nearest mates no live link names, kept ascending, each scored within the farthest
    // kept once the row's worth of them is full.
    vram_sorted_buffer<warp_t, node_id_type_> nearest {selection.scored_entries, 0, connectivity};
    for (u32_t base = 0; base < count; base += warp_t::width_k) { // Warp-uniform bounds.
        u32_t const at = base + lane;
        node_id_type_ mate_id = at < count ? view.entries[first + at] : unknown_k;
        bool fresh = mate_id != unknown_k && mate_id != node_id;
        for (u32_t live = 0; fresh && live != live_count; ++live) fresh = candidate_ids[live] != mate_id;
        if (!fresh) mate_id = unknown_k;
        distance_t const radius = nearest.size_ == connectivity ? nearest.at(connectivity - 1).distance()
                                                                : std::numeric_limits<distance_t>::infinity();
        distance_t const distance = schedule.score(storage, query, mate_id, radius, lane);
        u32_t const scored_count = warp_t::count(warp_t::ballot(fresh));
        nearest.insert_batch(sort_lanes<warp_t>(fresh ? scored_t::make(distance, mate_id) : scored_t::sentinel(), lane),
                             scored_count, lane);
        telemetry.scored += scored_count;
    }
    u32_t const mates_count = nearest.size_;
    for (u32_t mate = lane; mate < mates_count; mate += warp_t::width_k) {
        candidate_ids[live_count + mate] = nearest.at(mate).id();
        candidate_distances[live_count + mate] = nearest.at(mate).distance();
    }
    warp_t::sync();
    for (u32_t base = 0; base < live_count; base += warp_t::width_k) { // Warp-uniform bounds.
        u32_t const at = base + lane;
        distance_t const distance = schedule.score(storage, query, at < live_count ? candidate_ids[at] : unknown_k,
                                                   std::numeric_limits<distance_t>::infinity(), lane);
        if (at < live_count) candidate_distances[at] = distance;
    }
    telemetry.scored += live_count;
    warp_t::sync();
    u32_t const candidates_count = live_count + mates_count;
    if (!candidates_count) return;

    // Ascending by distance, positions breaking ties, into the second half, as the selector reads.
    vram_sorted_buffer<warp_t, node_id_type_> ascending {selection.scored_entries, 0, candidates_count};
    for (u32_t base = 0; base < candidates_count; base += warp_t::width_k) {
        u32_t const at = base + lane;
        scored_t const scored = at < candidates_count ? scored_t::make(candidate_distances[at], at)
                                                      : scored_t::sentinel();
        ascending.insert_batch(sort_lanes<warp_t>(scored, lane),
                               min_of<u32_t>(candidates_count - base, warp_t::width_k), lane);
    }
    node_id_type_* const sorted_ids = candidate_ids + half;
    f32_t* const sorted_distances = candidate_distances + half;
    for (u32_t rank = lane; rank < candidates_count; rank += warp_t::width_k) {
        sorted_ids[rank] = candidate_ids[ascending.at(rank).id()];
        sorted_distances[rank] = ascending.at(rank).distance();
    }
    warp_t::sync();
    u32_t* const kept_positions = selection.kept_positions;
    u32_t kept = selector_::template select<kit_>(storage, schedule, query, pivot_span, sorted_ids, sorted_distances,
                                                  candidates_count, kept_positions, telemetry, lane, false);
    // Back to the candidates' own positions, which the reach rule tells live links by.
    for (u32_t slot = lane; slot < kept; slot += warp_t::width_k)
        kept_positions[slot] = ascending.at(kept_positions[slot]).id();
    warp_t::sync();
    if (lane == 0)
        kept = vram_keep_reach<warp_t>(health, node_id, candidate_ids, candidate_distances, live_count, kept_positions,
                                       kept, connectivity);
    kept = warp_t::shuffle(kept, 0);
    warp_t::sync();

    // Positions below the live count are live links, so the kept ones among them stay.
    u32_t in_live = 0;
    for (u32_t slot = lane; slot < kept; slot += warp_t::width_k) in_live += kept_positions[slot] < live_count;
    in_live = vram_reduce_sum<warp_t>(in_live);
    added += kept - in_live, dropped += live_count - in_live;
    if (in_live == kept && in_live == live_count) return;
    for (u32_t slot_base = 0; slot_base < connectivity; slot_base += warp_t::width_k) { // Warp-uniform.
        u32_t const slot = slot_base + lane;
        node_id_type_ const value = slot < kept ? candidate_ids[kept_positions[slot]] : unknown_k;
        if (slot && slot < connectivity && live_row.load_slot(slot) != value) live_row.store_slot(slot, value);
    }
    warp_t::sync();
    if (lane == 0) live_row.unlock(kept ? candidate_ids[kept_positions[0]] : node_id);
    if (lane == 0) changed += vram_bitset_insert<warp_t>(health.changed, node_id);
    warp_t::sync();
}

#pragma endregion Rounds

} // namespace unum::usearch::basic
