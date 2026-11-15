/**
 *  @file include/usearch/basic/ram_index.hpp
 *  @author Ash Vardanian
 *  @date August 15, 2026
 *  @brief Similarity Search baseline over arbitrary payloads & caller-supplied metrics.
 *
 *  The baseline tier's RAM residency: the readable reference implementation. Arbitrary payload
 *  objects, expensive caller-supplied batched metrics, caller-visible keys, growable per-node
 *  neighbor lists. Standard-library & OS headers only - together with @c utilities/ this tier ships
 *  as a standalone header-only library. Every specialized optimization - lock-free readers, SIMD
 *  kernel kits - belongs to the dense family, not here.
 */
#pragma once

#include <cassert> // `assert` - an unfinished insertion plan's death
#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint64_t`
#include <cstring> // `std::memset` - a round's changed marks

#include <array>        // `std::array` - a filtered walk's quads
#include <bit>          // `std::popcount`, `std::bit_width` - an audit's reached count, a refine's rounds
#include <concepts>     // `std::same_as` - the sharded index takes 32-bit shards
#include <limits>       // `std::numeric_limits` - the ID-space ceiling and the unbounded radius
#include <memory>       // `std::allocator`, `std::allocator_traits`, `std::make_obj_using_allocator`
#include <mutex>        // `std::unique_lock` - the try-only unique side's guard
#include <shared_mutex> // `std::shared_lock` - the mutators' guard
#include <span>         // `std::span`
#include <type_traits>  // `std::is_nothrow_move_assignable_v`
#include <utility>      // `std::move`, `std::forward`

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/sanitizers.hpp"
#include "usearch/utilities/ram_buffers.hpp"
#include "usearch/utilities/ram_sorts.hpp" // `order_by_neighborhood`, `replace_farthest` - a round's neighborhoods
#include "usearch/utilities/ram_tables.hpp"

namespace unum::usearch::basic {
using namespace unum::usearch::utilities;

/** The sharded index spanning many basic RAM shards, defined in @c basic/ram_indexes.hpp; it walks
 *  and scans each shard below a watermark it took across all of them. */
template <typename index_type_>
    requires std::same_as<typename index_type_::node_id_t, u32_t>
struct ram_indexes;

/** One traversal candidate; both halves are the caller's chosen widths. */
template <typename node_id_type_ = u64_t, typename distance_type_ = f64_t>
struct scored_node {
    node_id_type_ node_id;
    distance_type_ distance;

    /** Orders scored nodes nearest-first; the comparator of the sorted results. */
    struct closer_t {
        bool operator()(scored_node first, scored_node second) const noexcept {
            return first.distance < second.distance;
        }
    };

    /** Orders scored nodes furthest-first; the comparator of the frontier heap. */
    struct further_t {
        bool operator()(scored_node first, scored_node second) const noexcept {
            return first.distance > second.distance;
        }
    };
};

/** Stored payloads live in records a cover builds from the index's allocator, receive their content
 *  through one move-assignment at publication, are read as @c const& after, and are destroyed at
 *  teardown or replaced by a fresh one in @c clear - that is the whole lifecycle. */
template <typename value_type_>
concept value = std::is_nothrow_default_constructible_v<value_type_> && //
                std::is_nothrow_move_assignable_v<value_type_> &&       //
                std::is_nothrow_destructible_v<value_type_>;

/** A batched metric's answer to one @c push: whether the candidate entered the
 *  staging of the current scoring round - a capacity answer, never a failure. */
enum class metric_staging_t : bool {

    /** The staging is full; the round should drain before anything else stages. */
    refused_k = false,

    /** The candidate is staged, and the round's @c drain will score it. */
    staged_k = true,
};

/** The radius of a round that wants every distance exact: nothing is ever too far, not even a
 *  distance that overflowed to infinity. */
template <typename distance_type_>
constexpr distance_type_ unbounded_radius() noexcept {
    if constexpr (std::numeric_limits<distance_type_>::has_infinity)
        return std::numeric_limits<distance_type_>::infinity();
    else return (std::numeric_limits<distance_type_>::max)();
}

/**
 *  @brief Any batched similarity measure, driven as a builder.
 *
 *  @c reset opens a scoring round for one query - a walk reopens the same query every round, so a
 *  metric that prepares the query keeps that work when nothing changed - @c push stages one
 *  candidate - the metric owns its staging, packing or uploading eagerly if it wishes, answering
 *  whether the candidate staged - and @c drain scores everything staged within the radius the caller
 *  still cares about, writing distances in push order and clearing the staging. A distance within
 *  the radius is exact; above it the metric may report any value above the radius, so a kernel can
 *  stop a candidate the moment it cannot come back under. @c unbounded_radius asks for every
 *  distance exact.
 *
 *  Every verb runs with no locks held, so a @c drain may cost milliseconds without
 *  stalling any other thread; pushed references stay valid until the round's @c drain
 *  returns, never longer. Every requirement is a call-validity check, so a metric is free
 *  to take converting parameters - @c std::string_view for @c std::string values.
 *
 *  A round's entries carry no order: results go by slot, and ties break by node ID.
 */
template <typename metric_type_, typename value_type_, typename distance_type_>
concept metric = value<value_type_> && requires(metric_type_& metric, value_type_ const& value, distance_type_ radius,
                                                std::span<distance_type_> distances) {
    { metric.reset(value) } noexcept;
    { metric.push(value) } noexcept -> std::same_as<metric_staging_t>;
    { metric.drain(distances, radius) } noexcept;
};

/** Marks a metric whose @c reset also accepts query forms other than the stored
 *  value - the standard @c is_transparent convention. */
template <typename metric_type_>
concept transparent_metric = requires { typename metric_type_::is_transparent; };

/** A query form @p metric_type_ opens rounds for: the stored value itself, or -
 *  on a @c transparent_metric - any other type its @c reset accepts. */
template <typename query_type_, typename metric_type_, typename value_type_>
concept query = (std::same_as<query_type_, value_type_> || transparent_metric<metric_type_>) &&
                requires(metric_type_& metric, query_type_ const& query) {
                    { metric.reset(query) } noexcept;
                };

/** A metric that prepares many queries at once: @c reset over a span of them prepares every one for
 *  all the rounds that follow, and each @c drain writes @b [queries,staged] distances, query @c q
 *  to staged entry @c r at <tt>distances[q * staged + r]</tt>, exact within the one radius they
 *  share. */
template <typename metric_type_, typename query_type_>
concept batched_metric = requires(metric_type_& metric, std::span<query_type_ const> queries) {
    { metric.reset(queries) } noexcept;
};

/** The query a round reads at one position of its span: the query itself, or the stored value a
 *  borrowed pointer names, as a round borrows its hubs and neighborhood entries from records. */
template <typename query_type_>
constexpr query_type_ const& borrowed_query(query_type_ const& query) noexcept {
    return query;
}
template <typename query_type_>
constexpr query_type_ const& borrowed_query(query_type_ const* query) noexcept {
    return *query;
}

/** Opens a scan of many queries: a metric scoring one query at a time reopens each one per round in
 *  @ref score_queries, so nothing is prepared here. */
template <typename metric_type_, typename query_type_>
void open_queries(metric_type_&, std::span<query_type_ const>) noexcept {}

/** A @c batched_metric prepares every query once for the whole scan. */
template <typename metric_type_, typename query_type_>
    requires batched_metric<metric_type_, query_type_>
void open_queries(metric_type_& metric, std::span<query_type_ const> queries) noexcept {
    metric.reset(queries);
}

/**
 *  @brief Scores nodes @c [first_node,end_node) of @p view_of against every query of @p queries
 *      into @p distances, @b [queries,nodes] over the nodes scored, and answers where those nodes
 *      end: @p first_node when the metric refused the first of them.
 *
 *  A metric scoring one query at a time reopens each query over every node, in as many rounds as
 *  its staging takes, each within its own of @p thresholds.
 */
template <typename metric_type_, typename query_type_, typename view_of_type_, typename distance_type_>
std::size_t score_queries(metric_type_& metric, std::span<query_type_ const> queries, view_of_type_&& view_of,
                          std::size_t first_node, std::size_t end_node, std::span<distance_type_ const> thresholds,
                          std::span<distance_type_> distances) noexcept {
    std::size_t const nodes = end_node - first_node;
    for (std::size_t query = 0; query != queries.size(); ++query)
        for (std::size_t round_first = first_node; round_first != end_node;) {
            metric.reset(borrowed_query(queries[query]));
            std::size_t round_end = round_first;
            while (round_end != end_node && metric.push(view_of(round_end)) == metric_staging_t::staged_k) ++round_end;
            if (round_end == round_first) return first_node;
            metric.drain(distances.subspan(query * nodes + round_first - first_node, round_end - round_first),
                         thresholds[query]);
            round_first = round_end;
        }
    return end_node;
}

/** A @c batched_metric opened by @ref open_queries scores every query in one round, within the
 *  loosest of @p thresholds, so the nodes end wherever its staging refused. */
template <typename metric_type_, typename query_type_, typename view_of_type_, typename distance_type_>
    requires batched_metric<metric_type_, query_type_>
std::size_t score_queries(metric_type_& metric, std::span<query_type_ const> queries, view_of_type_&& view_of,
                          std::size_t first_node, std::size_t end_node, std::span<distance_type_ const> thresholds,
                          std::span<distance_type_> distances) noexcept {
    std::size_t round_end = first_node;
    while (round_end != end_node && metric.push(view_of(round_end)) == metric_staging_t::staged_k) ++round_end;
    if (round_end == first_node) return first_node;
    distance_type_ loosest = thresholds.front();
    for (distance_type_ const threshold : thresholds) loosest = max_of(loosest, threshold);
    metric.drain(distances.first(queries.size() * (round_end - first_node)), loosest);
    return round_end;
}

/** Reads every round a walk scores - the staged node IDs in push order beside their drained
 *  distances, both valid only for the call and exact only within the round's radius - and answers
 *  whether the walk should go on. */
template <typename fold_type_, typename node_id_type_, typename distance_type_>
concept fold = requires(fold_type_& fold, std::span<node_id_type_ const> node_ids,
                        std::span<distance_type_ const> distances) {
    { fold.absorb(node_ids, distances) } noexcept -> std::same_as<walk_demand_t>;
};

/** Serializes stored values into caller-owned buffers and back - the payload half
 *  of @c save & @c load, supplied by the caller because values are arbitrary objects. */
template <typename codec_type_, typename value_type_>
concept value_codec = requires(codec_type_ const& codec, value_type_ const& immutable_value, value_type_& mutable_value,
                               bytes_span_t output, bytes_view_t input) {
    { codec.storage_requirements(immutable_value) } noexcept -> std::same_as<std::size_t>;
    { codec.save(immutable_value, output) } noexcept -> std::same_as<status_t>;
    { codec.load(input, mutable_value) } noexcept -> std::same_as<status_t>;
};

/** One round's scoring epilogue: sizes the distances to the staging fill, drains
 *  the metric within @p radius, and counts the round into the telemetry. */
template <typename metric_type_, typename context_type_, typename distance_type_>
void drain_round(metric_type_& metric, std::size_t staged_count, context_type_& context,
                 distance_type_ radius) noexcept {
    context.staged_distances.resize_reserved(staged_count);
    metric.drain({context.staged_distances.data(), staged_count}, radius);
    context.telemetry.computed_distances += staged_count;
}

/**
 *  @brief A filtered walk's fold: a round's contenders, the nodes that could still enter
 *      @c admitted, are asked of @c relation, then of @c predicate, a quad at a time, and what
 *      both admit enters @c admitted, at most @c breadth, while the walk routes through all.
 *
 *  Both take node IDs: the relation is the engine's own, like a geospatial relation, and the
 *  predicate the caller's, usually a @c node_predicate over keys. An aggregating predicate only
 *  gathers the round, answered at @ref settle.
 */
template <typename relation_type_, typename predicate_type_, typename ranking_type_>
struct predicate_fold {
    using scored_node_t = typename ranking_type_::value_type;

    relation_type_& relation;
    predicate_type_& predicate;
    ranking_type_& admitted;
    std::size_t breadth;

    template <typename node_id_type_, typename distance_type_>
    walk_demand_t absorb(std::span<node_id_type_ const> node_ids, std::span<distance_type_ const> distances) noexcept {
        quad<node_id_type_> ids;
        quad<scored_node_t> scored;
        std::size_t filled = 0;
        for (std::size_t position = 0; position != node_ids.size(); ++position) {
            if (admitted.size() == breadth && !(distances[position] < admitted.top().distance)) continue;
            ids[filled] = node_ids[position];
            scored[filled] = scored_node_t {node_ids[position], distances[position]};
            if (++filled != ids.size()) continue;
            ask_contenders_(std::span<node_id_type_ const, quad_lanes_k>(ids),
                            std::span<scored_node_t const, quad_lanes_k>(scored));
            filled = 0;
        }
        // A round's tail goes on at its own static extent, which sizes every buffer past here.
        with_static_extent<quad_lanes_k>(filled, [&]<std::size_t extent_>() noexcept {
            ask_contenders_(std::span<node_id_type_ const, extent_>(ids.data(), extent_),
                            std::span<scored_node_t const, extent_>(scored.data(), extent_));
        });
        return walk_demand_t::continue_k;
    }

    /** Answers what an aggregating predicate gathered this round; an immediate one has nothing
     *  left. */
    void settle() noexcept {
        settle_predicate(predicate, [this](scored_node_t const& match) noexcept { insert_match_(match); });
    }

  private:
    template <typename node_id_type_, std::size_t extent_>
    void ask_contenders_(std::span<node_id_type_ const, extent_> ids,
                         std::span<scored_node_t const, extent_> scored) noexcept {
        std::array<admission_t, extent_> related;
        relation(ids, std::span<admission_t, extent_>(related));
        std::array<node_id_type_, extent_> kept_ids;
        std::array<scored_node_t, extent_> kept_scored;
        std::size_t kept = 0;
        for (std::size_t lane = 0; lane != extent_; ++lane) {
            if (related[lane] != admission_t::admit_k) continue;
            kept_ids[kept] = ids[lane];
            kept_scored[kept] = scored[lane];
            ++kept;
        }
        auto const insert_match = [this](scored_node_t const& match) noexcept { insert_match_(match); };
        with_static_extent<extent_>(kept, [&]<std::size_t kept_extent_>() noexcept {
            ask_quad(predicate, std::span<node_id_type_ const, kept_extent_>(kept_ids.data(), kept_extent_),
                     std::span<scored_node_t const, kept_extent_>(kept_scored.data(), kept_extent_), insert_match);
        });
    }

    void insert_match_(scored_node_t const& match) noexcept {
        [[maybe_unused]] bool const inserted = admitted.insert(scored_node_t {match}, breadth);
    }
};

/** Plain NSW selection: the nearest @c connectivity candidates, no pruning. */
struct keep_closest_t {

    /** The neighbors-per-node budget this selector prunes to. */
    std::size_t connectivity = connectivity_32_k;

    template <typename index_type_, typename metric_type_, typename context_type_>
    void operator()(index_type_ const&, metric_type_&, context_type_& context) const noexcept {
        context.selected_neighbors.clear();
        std::size_t const kept_count = min_of(connectivity, context.results.size());
        for (std::size_t position = 0; position != kept_count; ++position)
            context.selected_neighbors.push_back_reserved(context.results[position]);
    }
};

/** Relative-neighborhood pruning: a candidate survives only if closer to the query than to every
 *  already-kept neighbor, measured from the kept side so asymmetric metrics compare like with like:
 *  one round per kept neighbor scores every later candidate no kept neighbor rejected yet, within
 *  the farthest one's distance from the query. */
struct keep_diverse_t {

    /** The neighbors-per-node budget this selector prunes to. */
    std::size_t connectivity = connectivity_32_k;

    template <typename index_type_, typename metric_type_, typename context_type_>
    void operator()(index_type_ const& index, metric_type_& metric, context_type_& context) const noexcept {
        using distance_t = typename index_type_::distance_t;
        auto& selected = context.selected_neighbors;
        selected.clear();
        // First half: each candidate's distance from its nearest kept neighbor; second: the round.
        auto& nearest_kept = context.staged_distances;
        std::size_t const candidates_count = context.results.size();
        // A refused growth keeps no pick, which an insertion's fill then covers.
        bool const scored = succeeded(nearest_kept.reserve(candidates_count * 2));
        if (scored) {
            nearest_kept.resize_reserved(candidates_count * 2);
            for (std::size_t position = 0; position != candidates_count; ++position)
                nearest_kept[position] = context.results[position].distance;
        }
        for (std::size_t position = 0; scored && position != candidates_count && selected.size() != connectivity;
             ++position) {
            auto const candidate = context.results[position];
            if (nearest_kept[position] < candidate.distance) continue;
            selected.push_back_reserved(candidate);
            if (selected.size() == connectivity) break;
            // The kept neighbor queries the later candidates, as the node queried them all: an
            // asymmetric distance compares only like with like. A rejected candidate stays
            // rejected, so it never stages again, and only a pair's own distance from the node
            // decides whether the kept neighbor rejects it, so the farthest one staged bounds the
            // round. A full staging drains and the rest stage in the next round.
            std::span<distance_t> const round {nearest_kept.data() + candidates_count, candidates_count};
            for (std::size_t first = position + 1;;) {
                while (first != candidates_count && nearest_kept[first] < context.results[first].distance) ++first;
                if (first == candidates_count) break;
                metric.reset(index.published_value_of(candidate.node_id));
                std::size_t staged_count = 0, round_end = first, farthest = first;
                for (; round_end != candidates_count; ++round_end) {
                    if (nearest_kept[round_end] < context.results[round_end].distance) continue;
                    if (metric.push(index.published_value_of(context.results[round_end].node_id)) ==
                        metric_staging_t::refused_k)
                        break;
                    ++staged_count, farthest = round_end;
                }
                if (!staged_count) break;
                // Candidates ascend, so the last one staged is the farthest from the node.
                metric.drain(round.first(staged_count), context.results[farthest].distance);
                context.telemetry.computed_distances += staged_count;
                context.telemetry.computed_distances_in_refines += staged_count;
                for (std::size_t later = first, slot = 0; later != round_end; ++later)
                    if (!(nearest_kept[later] < context.results[later].distance))
                        nearest_kept[later] = min_of(nearest_kept[later], round[slot++]);
                first = round_end;
            }
        }
    }
};

/** Appends the nearest candidates of @c context.results the selection left out to
 *  @c context.selected_neighbors until @p connectivity or the candidates run out: what an insertion
 *  wires, never what a round keeps. Every selector keeps a subsequence of the ascending candidates,
 *  so one merge-walk tells a kept candidate from a left one. */
template <typename context_type_>
void fill_selection(context_type_& context, std::size_t connectivity) noexcept {
    std::size_t const picked = context.selected_neighbors.size();
    for (std::size_t position = 0, matched = 0;
         position != context.results.size() && context.selected_neighbors.size() < connectivity; ++position) {
        if (matched != picked && context.selected_neighbors[matched].node_id == context.results[position].node_id)
            ++matched;
        else context.selected_neighbors.push_back_reserved(context.results[position]);
    }
}

/** Any link-selection heuristic: carries its own neighbors-per-node budget as @c connectivity,
 *  reads the ascending candidates in @c context.results, resolves values through the index's
 *  @c published_value_of - every candidate ID is published or wired - scores pairs through the
 *  @p metric, and fills @c context.selected_neighbors. A selector may also write
 *  @c context.staged_distances - and nothing else: @c staged_ids and @c reverse_ids belong to the
 *  calling traversal. The context's allocator is the caller's, so a selector takes any
 *  @c traversal_context; the concept checks the index's own. */
template <typename selector_type_, typename index_type_, typename metric_type_>
concept selector = requires(selector_type_ const& selector, index_type_ const& index, metric_type_& metric,
                            typename index_type_::traversal_context_t& context) {
    { selector(index, metric, context) } noexcept;
    { selector.connectivity } -> std::convertible_to<std::size_t>;
};

/** Per-thread traversal scratch: the candidate queues, the exact visited set,
 *  the staging surfaces of batched metric rounds, and running telemetry. */
template <typename node_id_type_ = u64_t, typename distance_type_ = f64_t,
          typename allocator_type_ = std::allocator<byte_t>>
struct traversal_context {

    using scored_node_t = scored_node<node_id_type_, distance_type_>;
    using allocator_t = allocator_type_;
    using scored_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<scored_node_t>;
    using visits_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<std::byte>;

    using results_array_t = sorted_buffer<scored_node_t, typename scored_node_t::closer_t, scored_allocator_t>;
    using frontier_queue_t = max_heap_dary<8, scored_node_t, typename scored_node_t::further_t, scored_allocator_t>;
    using visited_set_t = growing_hash_set<node_id_type_, std::hash<node_id_type_>, visits_allocator_t>;

    /** The buffer rebinds to its element type itself, so no alias spells it here. */
    template <typename element_type_>
    using scratch_buffer = growing_buffer<element_type_, allocator_t>;

    /** Nodes already scored, kept ascending as the best results so far. */
    results_array_t results;

    /** Nodes considered promising, to be expanded next. */
    frontier_queue_t frontier;

    /** Nodes already checked - exact, never lossy. */
    visited_set_t visited;

    /** Pseudo-random generator for picking traversal starting points. */
    splitmix64_t random_generator;

    /** IDs of one staged fan-out - the batch a single metric round scores; also the flood stack of
     *  @c health and the neighbor list copies of back-linking. */
    scratch_buffer<node_id_type_> staged_ids;

    /** Distances output of one batched metric round, parallel to @c staged_ids. */
    scratch_buffer<distance_type_> staged_distances;

    /** The selector's output: the neighbors to wire, in the selector's order of preference. */
    scratch_buffer<scored_node_t> selected_neighbors;

    /** The freshly wired node's neighbors, snapshotted for the reverse pass - the
     *  re-selections there reuse @c selected_neighbors, so the pass walks its own copy. */
    scratch_buffer<node_id_type_> reverse_ids;

    /** A filtered walk's matches: the scored nodes its relation and predicate let in, nearest
     *  first, while @c results keeps routing through every node the walk scored. */
    results_array_t admitted;

    /** The nodes one walk takes as landmarks ahead of its random draws, in the order handed in. */
    scratch_buffer<node_id_type_> hinted_ids;

    /** Per entry of the neighborhood in hand, its nearest neighborhood-mates, a max-heap each,
     *  @b [capacity,connectivity], of which a gather keeps each entry's nearest hub in the
     *  first @b [capacity]. */
    scratch_buffer<scored_node_t> neighborhood_mates;

    /** One byte per hint key of the query in flight, set once any shard staged a node under it. */
    scratch_buffer<u8_t> hinted_held;

    /** Running per-walk counters; calls report their share as snapshot deltas. */
    traversal_stats_t telemetry;

    /** The allocator the containers above rebind; a plan opened through the context places its own
     *  state with it. */
    allocator_t allocator;

    /** Every container gets a rebound copy of @p allocator, so scratch goes where
     *  the caller places it; seed distinctly per thread to decorrelate the walks. */
    explicit traversal_context(u64_t seed = 0, allocator_t allocator = {}) noexcept
        : results(scored_allocator_t(allocator)), frontier(scored_allocator_t(allocator)),
          visited(visits_allocator_t(allocator)), random_generator {seed}, staged_ids(allocator),
          staged_distances(allocator), selected_neighbors(allocator), reverse_ids(allocator),
          admitted(scored_allocator_t(allocator)), hinted_ids(allocator), neighborhood_mates(allocator),
          hinted_held(allocator), allocator(allocator) {}
};

/** Node & element counts of the five columns a snapshot needs - what a container sizes. */
struct columns_extents_t {

    /** Published nodes, the length of every node-sized column. */
    std::size_t nodes = 0;

    /** Neighbor IDs across every node. */
    std::size_t neighbors = 0;

    /** Codec bytes across every value. */
    std::size_t value_bytes = 0;
};

/** The five caller-placed columns a basic-tier snapshot is: keys, neighbor prefix sums,
 *  neighbor IDs, value prefix sums, and codec bytes - each an independent buffer, so a
 *  container lays them out wherever it likes. Both prefix-sum columns hold @b nodes+1 sums
 *  from zero, so node @c i owns the slice between sums @c i and @c i+1. */
template <typename key_type_, typename node_id_type_, typename byte_type_, typename offset_type_ = u64_t>
struct columns {

    /** One key per node. */
    std::span<key_type_> keys;

    /** @b nodes+1 sums of neighbor IDs, from zero. */
    std::span<offset_type_> neighbor_offsets;

    /** Every node's neighbor IDs, back to back. */
    std::span<node_id_type_> neighbor_ids;

    /** @b nodes+1 sums of codec bytes, from zero. */
    std::span<offset_type_> value_offsets;

    /** Every node's codec bytes, back to back. */
    std::span<byte_type_> values;

    /** The read-only view of the same columns, so a saved set loads as it is. */
    columns<key_type_ const, node_id_type_ const, byte_type_ const, offset_type_ const> view() const noexcept {
        return {keys, neighbor_offsets, neighbor_ids, value_offsets, values};
    }
};

template <typename index_type_, synchronization_t synchronization_ = synchronization_t::concurrent_k,
          typename allocator_type_ = typename index_type_::allocator_t>
struct ram_cursor;

/**
 *  @brief Similarity Search index baseline for arbitrary values & similarity measures.
 *
 *  The design optimizes for being read, and its storage is a column of node records in chunks that
 *  grow on demand and never move, plus the key multimap. Each record owns its neighbor list, which
 *  starts empty and grows with wiring - automatic through @c insert's selector, which carries the
 *  neighbors-per-node budget, or manual through @c link. Metric calls cost microseconds to
 *  milliseconds here, so brief per-node locks are noise - the only hard rule is that no metric verb
 *  ever runs while any node lock is held - the shared mutex is deliberately held across whole
 *  batches, so a parked metric starves stop-the-world verbs, and delays the publication of the
 *  batches dispensed after its own.
 *
 *  Concurrency runs on plain primitives: batch capacity is claimed by a bounded add and covered
 *  before anything is dispensed, one more add dispenses the batch's whole ID range, any number of
 *  threads place its positions, and one turn of a watermark advanced in dispensed order publishes
 *  that range at once, so a search, a cursor or an ID lookup sees all of a batch or none of it. A
 *  batch's keys go into the multimap inside its turn, just before the watermark moves, so the map
 *  has one writer and claims no slot; a key lookup drops every node at or above the watermark it
 *  acquired, so it sees a batch's keys exactly when it sees its nodes. Keys need not be unique:
 *  every holder of a key stays addressable through @c find_each, and ID → key stays authoritative.
 *  Node IDs are never recycled before a @c clear and removals are deleted, keeping every reader's
 *  stale ID valid until then.
 *
 *  One index is complete on its own: it needs no SafeGraphs files, no sharded index and no
 *  UUIDs. Keys may be any hashable type, node IDs 32 or 64 bits wide, and every record keeps its
 *  own key. Only the sharded indexes the compiled library builds on top narrow the choice, to
 *  32-bit shards whose nodes a shard table locates in 8 bytes.
 *
 *  @tparam value_type_ any payload satisfying @c basic::value; immovable once published.
 *  @tparam key_type_ trivially-copyable caller-visible identity.
 *  @tparam node_id_type_ graph-local identity, dense from zero; @c u32_t or @c u64_t.
 *  @tparam distance_type_ the metrics' result element.
 *  @tparam wait_type_ the monitored-wait engaged on contended words.
 *  @tparam atomic_ref_template_ the atomic reference owning the plain synchronization words;
 *      @c standard_atomic_ref by default, the CPU class's reference as @c src/ picks it; either
 *      binds a const word for the const readers' loads.
 *  @tparam allocator_type_ places the node records, neighbor buffers & key multimap, and builds
 *      every stored value through uses-allocator construction.
 *  @tparam cache_hints_type_ the line hints streaming the record a walk expands next toward the
 *      core.
 */
template <                                                                //
    typename value_type_,                                                 //
    typename key_type_ = uuid128_t,                                       //
    typename node_id_type_ = u64_t,                                       //
    typename distance_type_ = f64_t,                                      //
    typename wait_type_ = dummy_waiter_t,                                 //
    template <typename> class atomic_ref_template_ = standard_atomic_ref, //
    typename allocator_type_ = std::allocator<byte_t>,                    //
    typename cache_hints_type_ = dummy_cache_hints_t                      //
    >
    requires value<value_type_> && (std::same_as<node_id_type_, u32_t> || std::same_as<node_id_type_, u64_t>)
struct ram_index {

    using value_t = value_type_;
    using key_t = key_type_;
    using node_id_t = node_id_type_;
    using distance_t = distance_type_;
    using wait_t = wait_type_;
    using allocator_t = allocator_type_;
    using prefetcher_t = object_prefetcher<cache_hints_type_>;
    template <typename word_type_>
    using atomic_ref = atomic_ref_template_<word_type_>;

    using scored_node_t = scored_node<node_id_t, distance_t>;

    /** One thread's walk scratch, its containers placed by @p context_allocator_type_: any
     *  allocator, such as an executor's per-thread arena, since scratch never outlives the call. */
    template <typename context_allocator_type_>
    using traversal_context_over = traversal_context<node_id_t, distance_t, context_allocator_type_>;

    /** One thread's walk scratch placed by the index allocator. */
    using traversal_context_t = traversal_context_over<allocator_t>;

    /** The five caller-placed columns a save fills. */
    using columns_t = columns<key_t, node_id_t, byte_t, u64_t>;

    /** The read-only columns a load consumes. */
    using columns_view_t = columns<key_t const, node_id_t const, byte_t const, u64_t const>;

  private:
    template <typename cursor_index_type_, synchronization_t cursor_synchronization_, typename cursor_allocator_type_>
    friend struct ram_cursor;

    /** The sharded index walks and scans its shards below one watermark per shard, and measures,
     *  closes and reopens their claims. */
    template <typename index_type_>
        requires std::same_as<typename index_type_::node_id_t, u32_t>
    friend struct ram_indexes;

    /** The sharded index's shard table measures, closes and reopens the claims. */
    template <typename, template <typename> class, typename>
    friend struct utilities::shard_set;

    using neighbors_buffer_t = growing_buffer<node_id_t, allocator_t>;

    /** One node record: its state, the payload, its key and the neighbors it links to. The value &
     *  key are written before the node is wired and never change until a @c clear; only
     *  @c neighbors mutates after - grown or re-selected by later inserts' back-links, grown or
     *  trimmed by manual linking - with its own @c size() as the live degree. */
    struct node_t {

        /** Set with a release once an insert wired the node, ahead of its batch's turn, so another
         *  batch's walk may admit it early. A load never sets it, since its nodes are reached below
         *  the watermark alone, and only @c clear, which recycles the IDs, resets it. */
        inline static constexpr u32_t wired_k = 1;

        /** Held by whoever edits @c neighbors once the node is reachable: one bit, claimed
         *  alone. */
        inline static constexpr unsigned held_bit_k = 1;
        inline static constexpr u32_t held_k = u32_t {1} << held_bit_k;

        /** The node's wiring and its lock in one word, through @c atomic_ref_template_. */
        u32_t state {0};
        value_t value;
        key_t key {};
        neighbors_buffer_t neighbors;

        // No default form: it would seed the neighbor list with the default allocator, silently
        // escaping whatever allocator the index was constructed with.
        node_t() = delete;
        explicit node_t(allocator_t const& allocator) noexcept
            : value(std::make_obj_using_allocator<value_t>(allocator)), neighbors(allocator) {}

        /** Waits out a holder, then holds the node for an edit of @c neighbors. */
        void lock() noexcept {
            annotate_lock_started(&state);
            // Claim only after observing the node free: on Arm a losing claim still takes the line
            // exclusive, stealing it from the holder; the peek keeps contenders in Shared state.
            atomic_ref<u32_t> word {state};
            while (true) {
                u32_t observed = word.load(std::memory_order_relaxed);
                for (; observed & held_k; observed = word.load(std::memory_order_relaxed))
                    wait_t {}(&state, observed, 0u);
                if (!word.fetch_test_and_set_bit(held_bit_k, std::memory_order_acquire)) break;
            }
            annotate_lock_acquired(&state);
        }

        void unlock() noexcept {
            annotate_unlock_started(&state);
            atomic_ref<u32_t>(state).clear_bits(held_k, std::memory_order_release);
            annotate_unlock_finished(&state);
        }

        /** Publishes the node to other batches' walks; nothing else reaches it before this. */
        void wire() noexcept { atomic_ref<u32_t>(state).store(wired_k, std::memory_order_release); }

        /** Whether an insert wired the node, carrying its value and neighbors with the acquire. */
        bool wired() const noexcept { return atomic_ref<u32_t const>(state).load(std::memory_order_acquire) & wired_k; }
    };

  public:

    /** Ceiling on dispensable node IDs: every ID the type holds but the all-ones sentinel, and no
     *  more records than the address space holds. */
    inline static constexpr std::size_t max_node_count_k = static_cast<std::size_t>(
        min_of<u64_t>(min_of<u64_t>((std::numeric_limits<node_id_t>::max)(), (std::numeric_limits<std::size_t>::max)()),
                      (u64_t {1} << address_bits_k) / sizeof(node_t)));

  private:

    /** The chunks spanning @c max_node_count_k; the first one's nodes are also one plan task's. */
    using nodes_schedule_t = chunk_schedule_spanning<max_node_count_k>;

    /** The node records in chunks on @c nodes_schedule_t: a cover builds every record of a fresh
     *  chunk from the index allocator, so slots exist before publication with no per-slot
     *  lifecycle tracking, and none ever moves. */
    using nodes_column_t =
        chunked_column<node_t, nodes_schedule_t, max_node_count_k, allocator_t, atomic_ref_template_>;

    /** The claim count while a refine holds the claims closed: past every ceiling, so the bounded
     *  add of every batch refuses it. */
    inline static constexpr std::size_t closed_claims_k = (std::numeric_limits<std::size_t>::max)();

    /** The all-ones ID no node ever takes: a heal's answer for no target. */
    inline static constexpr node_id_t missing_k = (std::numeric_limits<node_id_t>::max)();

    using key_map_t = chunked_multimap<key_t, node_id_t, std::hash<key_t>, allocator_t, atomic_ref_template_>;

    using shared_mutex_t = shared_mutex<wait_t, atomic_ref_template_>;
    using shared_lock_t = std::shared_lock<shared_mutex_t>;
    using unique_lock_t = std::unique_lock<shared_mutex_t>;

  public:
#pragma region Lifecycle

    ram_index(ram_index const&) noexcept = delete;
    ram_index& operator=(ram_index const&) noexcept = delete;
    ram_index(ram_index&&) noexcept = delete;
    ram_index& operator=(ram_index&&) noexcept = delete;

    /** Constructs an empty index prefetching @p line_bytes at a time; allocates nothing until the
     *  first @c reserve or insert. */
    ram_index(allocator_t allocator = {}, std::size_t line_bytes = default_alignment_k) noexcept
        : nodes_(allocator), key_map_(allocator), allocator_(std::move(allocator)), prefetcher_ {line_bytes} {}

    /** Exclusive: the record chunks destroy every value in place, the key multimap
     *  frees its columns wholesale. */
    ~ram_index() noexcept = default;

    /** The allocator placing every record, neighbor list and the key multimap, and building every
     *  stored value - the one a caller builds the values it inserts with. */
    allocator_t get_allocator() const noexcept { return allocator_; }

    /** A reservation: the nodes to host, and the most this index may ever claim. */
    struct reserve_capacity_t {

        /** Nodes to cover ahead of the inserts. */
        std::size_t capacity = 0;

        /** Nodes the claims may reach, clamped to @c max_node_count_k; past it every batch and
         *  absorb answers @c overflow_k. */
        std::size_t ceiling = max_node_count_k;
    };

    /**
     *  Covers the node records & the key multimap for @p config.capacity entries ahead of the
     *  inserts, which cover what they claim on their own, so this is a hint callable any time
     *  beside every other verb, and bounds every later claim by @p config.ceiling. Neighbor lists
     *  start empty and grow with wiring. Answers the node capacity covered so far, and a
     *  @c bad_alloc_k keeps the chunks it did cover, so the same call retries.
     *
     *  @par Statuses:
     *  - @c overflow_k @p config.capacity passes @c max_node_count_k.
     */
    expected<std::size_t> reserve(reserve_capacity_t config) noexcept {
        atomic_ref<std::size_t>(ceiling_nodes_)
            .store(min_of(config.ceiling, max_node_count_k), std::memory_order_relaxed);
        if (status_t const covered = nodes_.cover(config.capacity); failed(covered))
            return {nodes_.capacity(), covered};
        if (status_t const keyed = key_map_.reserve(config.capacity); failed(keyed)) return {nodes_.capacity(), keyed};
        return {nodes_.capacity(), status_t::success_k};
    }

    /**
     *  Covers @p capacity entries, bounded by the ID space alone.
     *
     *  @par Statuses:
     *  - @c overflow_k @p capacity passes @c max_node_count_k.
     */
    expected<std::size_t> reserve(std::size_t capacity) noexcept { return reserve(reserve_capacity_t {capacity}); }

    /** Forgets every node and key and keeps every covered chunk for the next build: each record
     *  below the dispenser gets a fresh value from the index allocator, an empty neighbor list and
     *  no key. Takes the shared mutex's unique side - live mutators and open plans answer
     *  @c contended_k - while the caller keeps readers out, as for @c load, since the IDs restart
     *  from zero.
     *
     *  @par Statuses:
     *  - @c contended_k A mutator or an open plan holds the shared mutex.
     *  - @c contended_k A dispensed batch has yet to publish.
     */
    status_t clear() noexcept {
        unique_lock_t unique_lock {shared_mutex_, std::try_to_lock};
        if (!unique_lock) return status_t::contended_k;
        auto const [visible, status] = watermark_();
        if (failed(status)) return status;
        forget_(visible);
        return status_t::success_k;
    }

#pragma endregion Lifecycle

#pragma region Mutation

    /** Optional tuning knobs for @c insert(); zeros mean "derive from the
     *  selector's budget". */
    struct insert_config_t {

        /** How many candidates a walk keeps during wiring. */
        std::size_t expansion_factor = 0;
    };

    /** Telemetry of one @c insert() batch. */
    struct [[nodiscard]] insert_result_t {

        /** This batch's share of the context's running counters. */
        traversal_stats_t telemetry;
    };

    /**
     *  @brief Moves a batch of values into the index under their keys, walking the graph for each
     *      and wiring the selector's chosen neighbors.
     *
     *  All-or-nothing: capacity & scratch for the whole batch are secured before the first value
     *  moves, so a non-success status means the caller's batch is untouched. Safe concurrently with
     *  @c insert, @c search & @c reserve; never with @c save, @c load, @c clear, @c health, or
     *  teardown. An @c exclusive_k instantiation compiles the per-node locks out for
     *  single-threaded builds; an @c immutable_k one is rejected at compile time.
     *
     *  One plan on the calling thread: @c reserve_traversal sizes the context, @c plan_insert's one
     *  add dispenses the batch's whole ID range, one task places it, and @c finish publishes. Per
     *  value: walk the graph scoring against the caller's still-unmoved value, move the value & key
     *  into its record, wire the selected neighbors, mark the node wired, then back-link - locking
     *  one neighbor at a time, never during a metric verb. The walk admits the published nodes and
     *  the nodes other inserts wired ahead of their turns, this batch's earlier nodes among them.
     *  One watermark turn then files the range's keys and publishes it whole. A full reverse
     *  neighbor list re-selects down to the selector's budget - manual over-wiring is advisory here
     *  and may be pruned back.
     *
     *  @param[in] incoming_values The batch to consume - moved from on success only.
     *  @param[in] keys Caller identities, parallel to @p incoming_values.
     *  @param[in,out] metric This thread's scoring instance; weights may be shared,
     *      staging scratch must be thread-private.
     *  @param[out] assigned_ids Caller-owned output: either empty, or exactly one slot per
     *      incoming value, receiving that value's dispensed node ID - one contiguous range
     *      per batch, in batch order.
     *  @param[in,out] context The calling thread's scratch & telemetry, reused across batches, over
     *      any allocator.
     *  @param[in] selector The link-selection heuristic, carrying its own
     *      neighbors-per-node budget - a per-call value, so heuristics & budgets swap per
     *      batch without re-instantiating the index.
     *  @param[in] config Optional tuning knobs; zeros mean "derive from connectivity".
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The selector's connectivity is zero, or @p keys or a non-empty
     *    @p assigned_ids differs in length from @p incoming_values.
     *  - @c overflow_k A scratch capacity of @p context wraps when rounded.
     *  - @c contended_k A refine plan holds the claims closed.
     *  - @c overflow_k The batch passes the reserved ceiling.
     */
    template <synchronization_t synchronization_ = synchronization_t::concurrent_k, typename metric_type_,
              typename context_allocator_type_, typename selector_type_ = keep_diverse_t>
        requires metric<metric_type_, value_t, distance_t> && selector<selector_type_, ram_index, metric_type_>
    expected<insert_result_t> insert(std::span<value_t> incoming_values, std::span<key_t const> keys,
                                     metric_type_& metric, std::span<node_id_t> assigned_ids,
                                     traversal_context_over<context_allocator_type_>& context,
                                     selector_type_ selector = {}, insert_config_t config = {}) noexcept {
        static_assert(
            synchronization_ != synchronization_t::immutable_k,
            "Certified immutability contradicts insertion - mark_reaches_root exclusive_k or stay concurrent_k");
        std::size_t const incoming_count = incoming_values.size();
        if (!incoming_count) return {};
        if (!selector.connectivity) return {{}, status_t::config_mismatch_k};
        if (keys.size() != incoming_count) return {{}, status_t::config_mismatch_k};
        if (!assigned_ids.empty() && assigned_ids.size() != incoming_count) return {{}, status_t::config_mismatch_k};
        if (status_t const sized = reserve_traversal(context, selector.connectivity, config); failed(sized))
            return {{}, sized};
        auto [plan, status] = plan_insert(incoming_count);
        if (failed(status)) return {{}, status};

        traversal_stats_t const telemetry_before = context.telemetry;
        plan.template insert<synchronization_>(0, incoming_values, keys, metric, context, selector, config);
        for (std::size_t position = 0; position != assigned_ids.size(); ++position)
            assigned_ids[position] = static_cast<node_id_t>(plan.range.first + position);
        // The one task above placed every position, so the finish cannot refuse.
        [[maybe_unused]] status_t const finished = plan.finish();

        insert_result_t result;
        result.telemetry = context.telemetry - telemetry_before;
        return {result, status_t::success_k};
    }

    /**
     *  @brief One batched insertion in flight: the shared hold and the dispensed range, so its
     *      positions can only be placed, and @ref finish publishes them all at once.
     *
     *  Positions map onto @c range in order. Every position must be placed before @c finish: the
     *  range holds every later turn until then, so a caller fanning the tasks out over threads
     *  places a refused fan-out on the calling thread, and nothing cancels a plan once dispensed. A
     *  plan that dies unfinished only asserts, since publishing its range would expose nodes nobody
     *  wrote.
     */
    struct [[nodiscard]] insert_plan_t {

        /** The IDs the plan dispensed, one per position, in position order. */
        iota_view<node_id_t> range {};

        /** Positions the tasks placed so far, mutated through @c atomic_ref. */
        std::size_t placed = 0;

        insert_plan_t() noexcept = default;
        insert_plan_t(ram_index* index, iota_view<node_id_t> range, shared_lock_t&& hold) noexcept
            : range(range), index_(index), hold_(std::move(hold)) {}
        insert_plan_t(insert_plan_t&& other) noexcept
            : range(other.range), placed(other.placed), index_(std::exchange(other.index_, nullptr)),
              hold_(std::move(other.hold_)) {}
        insert_plan_t& operator=(insert_plan_t&&) = delete;
        ~insert_plan_t() noexcept { assert((!index_ || range.empty()) && "A dispensed plan must finish"); }

        /**
         *  @brief Places positions @p first onward, one per value of @p values under its key in
         *      @p keys; cannot fail, since a refused walk wires a thinner neighbor list.
         *
         *  Per value: a walk scoring against the still-unmoved value, the value & key moved into
         *  the record, the selector's neighbors wired, the node flagged wired, then the back-links.
         *  Any number of threads step one plan at once over disjoint positions, each with its own
         *  @p metric and a @p context that @ref reserve_traversal sized for the selector's
         *  connectivity and @p config. The walks admit the published nodes and every node wired
         *  ahead of its turn, the plan's own among them. Positions outside the plan, keys of
         *  another length, or a zero connectivity place nothing, and @c finish then refuses.
         */
        template <synchronization_t synchronization_ = synchronization_t::concurrent_k, typename metric_type_,
                  typename context_allocator_type_, typename selector_type_ = keep_diverse_t>
            requires metric<metric_type_, value_t, distance_t> && selector<selector_type_, ram_index, metric_type_>
        void insert(std::size_t first, std::span<value_t> values, std::span<key_t const> keys, metric_type_& metric,
                    traversal_context_over<context_allocator_type_>& context, selector_type_ selector = {},
                    insert_config_t config = {}) noexcept {
            static_assert(
                synchronization_ != synchronization_t::immutable_k,
                "Certified immutability contradicts insertion - mark_reaches_root exclusive_k or stay concurrent_k");
            std::size_t const count = values.size();
            if (!index_ || !selector.connectivity || keys.size() != count || first > range.size() ||
                count > range.size() - first)
                return;
            insert_config_t const walk = resolved_config_(config, selector.connectivity);
            index_->widen_(selector.connectivity);
            for (std::size_t position = 0; position != count; ++position)
                index_->template place_<synchronization_>(static_cast<node_id_t>(range.first + first + position),
                                                          values[position], keys[position], metric, context, selector,
                                                          walk);
            // Releases this task's nodes to the acquire in `finish`, whose turn carries them on.
            atomic_ref<std::size_t>(placed).store_add(count, std::memory_order_release);
        }

        /**
         *  @brief Publishes every position at once: @c config_mismatch_k while any has yet to be
         *      placed, leaving the plan whole; then the range's one watermark turn, which files its
         *      keys and moves the watermark, and the hold dropped.
         *
         *  A success spends the plan, so a second call answers @c uninitialized_k.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is empty or already spent.
         *  - @c config_mismatch_k A position has yet to be placed.
         */
        status_t finish() noexcept {
            if (!index_) return status_t::uninitialized_k;
            if (atomic_ref<std::size_t>(placed).load(std::memory_order_acquire) != range.size())
                return status_t::config_mismatch_k;
            if (!range.empty()) {
                // Take the turn strictly in dispensed order: every range dispensed ahead of this
                // one has published, which keeps the visible prefix hole-free. Holds no node lock
                // while waiting.
                node_id_t& last_visible_id = index_->last_visible_id_;
                for (node_id_t watermark = atomic_ref<node_id_t>(last_visible_id).load(std::memory_order_acquire);
                     watermark != range.first;
                     watermark = atomic_ref<node_id_t>(last_visible_id).load(std::memory_order_acquire))
                    wait_t {}(&last_visible_id, watermark, 0u);
                // The turn holder is the map's one writer; readers drop these nodes until the
                // store.
                index_->publish_keys_(range);
                atomic_ref<node_id_t>(last_visible_id).store(range.bound, std::memory_order_release);
            }
            hold_.unlock();
            index_ = nullptr;
            return status_t::success_k;
        }

      private:

        /** The index inserted into, whose shared side @c hold_ occupies; null in the empty plan a
         *  refused @ref plan_insert returns, and once @c finish published. */
        ram_index* index_ = nullptr;

        /** The shared hold spanning the plan's whole lifetime. */
        shared_lock_t hold_;
    };

    /**
     *  @brief Opens a batched insertion of @p count values: enters the shared side, claims and
     *      covers the node records and the key multimap, then dispenses the whole ID range in one
     *      add.
     *
     *  Every refusal precedes the dispense: @c overflow_k past the reserved ceiling, or a refused
     *  cover's status once the claim is given back. Everything else fallible belongs before this
     *  call, since nothing may refuse once the range is dispensed: building the values, sizing each
     *  stepping thread's context through @ref reserve_traversal, a metric's own setup. A @p count
     *  of zero dispenses nothing, and a refine plan holding the claims closed refuses with
     *  @c contended_k. Joining the shared side waits out a unique holder, so a call from a thread
     *  holding a save or load plan deadlocks on its own hold, by construction.
     *
     *  @par Statuses:
     *  - @c contended_k A refine plan holds the claims closed.
     *  - @c overflow_k The batch passes the reserved ceiling.
     */
    expected<insert_plan_t> plan_insert(std::size_t count) noexcept {
        auto [claim, status] = claim_(count);
        if (failed(status)) return {{}, status};
        return {dispense_(std::move(claim)), status_t::success_k};
    }

    /**
     *  @brief Sizes @p context for walks under @p connectivity and @p config - the fallible half of
     *      an insertion, hoisted ahead of @ref plan_insert since no task may allocate it.
     *
     *  The candidates, the staging surfaces and the selections never allocate again, and the
     *  frontier holds the full candidates' unexpanded members plus one round's admissions, which
     *  shedding past their radius keeps it within, so only ties at that radius grow it. The visited
     *  set starts at one neighbor list per candidate and alone re-tables as a walk widens - once
     *  per doubling per context lifetime, which a context kept across walks amortizes. Idempotent;
     *  a zero @p connectivity answers @c config_mismatch_k.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k @p connectivity is zero.
     *  - @c overflow_k A scratch capacity wraps when rounded.
     */
    template <typename context_allocator_type_>
    static status_t reserve_traversal(traversal_context_over<context_allocator_type_>& context,
                                      std::size_t connectivity, insert_config_t config) noexcept {
        if (!connectivity) return status_t::config_mismatch_k;
        insert_config_t const walk = resolved_config_(config, connectivity);
        std::size_t const round_capacity = ram_index::round_capacity(connectivity, config);
        // A re-selection's candidates: the walk's breadth plus the node's own neighbor list, which
        // also covers a back-link's full neighbor list plus the appended node.
        std::size_t const candidates_capacity = walk.expansion_factor + connectivity;
        if (status_t const sized = context.results.reserve(candidates_capacity); failed(sized)) return sized;
        if (status_t const sized = context.frontier.reserve(candidates_capacity + round_capacity); failed(sized))
            return sized;
        if (status_t const sized = context.visited.reserve(walk.expansion_factor * connectivity); failed(sized))
            return sized;
        if (status_t const sized = context.staged_ids.reserve(round_capacity); failed(sized)) return sized;
        // A diversity selector keeps a distance per candidate beside the round it drains.
        if (status_t const sized = context.staged_distances.reserve(max_of(round_capacity, candidates_capacity * 2));
            failed(sized))
            return sized;
        if (status_t const sized = context.selected_neighbors.reserve(connectivity + 1); failed(sized)) return sized;
        if (status_t const sized = context.admitted.reserve(candidates_capacity); failed(sized)) return sized;
        // A walk draws a list's worth of landmarks, which bounds the hints a query stages.
        if (status_t const sized = context.hinted_ids.reserve(connectivity); failed(sized)) return sized;
        if (status_t const sized = context.hinted_held.reserve(connectivity); failed(sized)) return sized;
        return context.reverse_ids.reserve(connectivity);
    }

    /** Candidates one metric round of a walk under @p connectivity and @p config stages: the nodes
     *  whose neighbors fill a walk's candidates, of the widest neighbor list, or a full one plus
     *  the appended node a back-link re-selects - what a metric's staging holds to score every
     *  round at once. */
    static std::size_t round_capacity(std::size_t connectivity, insert_config_t config) noexcept {
        return max_of(
            divide_round_up(resolved_config_(config, connectivity).expansion_factor, connectivity) * connectivity,
            connectivity + 1);
    }

    /**
     *  Appends one directed edge @p from → @p to - purely structural: no metric, no selector, no
     *  pruning, and the neighbor list grows past any prior extent when needed. Idempotent: an
     *  existing edge answers @c present_k. Both IDs must be published; unpublished IDs & self-links
     *  answer @c malformed_k. Safe concurrently with @c insert, @c search, readers & other manual
     *  wiring.
     *
     *  @par Statuses:
     *  - @c malformed_k Either ID is unpublished, or the edge is a self-link.
     *  - @c overflow_k The grown neighbor list's capacity wraps.
     */
    template <synchronization_t synchronization_ = synchronization_t::concurrent_k>
    expected<membership_t> link(node_id_t from, node_id_t to) noexcept {
        static_assert(synchronization_ != synchronization_t::immutable_k,
                      "Certified immutability contradicts wiring - mark_reaches_root exclusive_k or stay concurrent_k");
        constexpr bool locked_k = synchronization_ == synchronization_t::concurrent_k;
        std::size_t const visible = size();
        if (static_cast<std::size_t>(from) >= visible || static_cast<std::size_t>(to) >= visible || from == to)
            return {membership_t::absent_k, status_t::malformed_k};

        // Enters the mutator class: manual wiring never moves the watermark, so the shared
        // mutex is what makes it visible to stop-the-world verbs & the marked probes.
        shared_lock_t shared_lock {shared_mutex_};
        node_t& node = nodes_[from];
        if constexpr (locked_k) node.lock();
        membership_t membership = membership_t::absent_k;
        for (std::size_t position = 0; position != node.neighbors.size() && membership == membership_t::absent_k;
             ++position)
            if (node.neighbors[position] == to) membership = membership_t::present_k;
        status_t status = status_t::success_k;
        if (membership == membership_t::absent_k) {
            status = node.neighbors.reserve(node.neighbors.size() + 1);
            if (succeeded(status)) node.neighbors.push_back_reserved(to);
        }
        std::size_t const degree = node.neighbors.size();
        if constexpr (locked_k) node.unlock();
        widen_(degree);
        return {membership, status};
    }

    /**
     *  Removes one directed edge @p from → @p to; @c malformed_k when the edge or either ID is
     *  unpublished or absent. The surviving neighbor list's order is unspecified. Safe concurrently
     *  with @c insert, @c search, readers & other manual wiring.
     *
     *  @par Statuses:
     *  - @c malformed_k Either ID is unpublished.
     *  - @c malformed_k The neighbor list holds no edge to @p to.
     */
    template <synchronization_t synchronization_ = synchronization_t::concurrent_k>
    status_t unlink(node_id_t from, node_id_t to) noexcept {
        static_assert(synchronization_ != synchronization_t::immutable_k,
                      "Certified immutability contradicts wiring - mark_reaches_root exclusive_k or stay concurrent_k");
        constexpr bool locked_k = synchronization_ == synchronization_t::concurrent_k;
        std::size_t const visible = size();
        if (static_cast<std::size_t>(from) >= visible || static_cast<std::size_t>(to) >= visible)
            return status_t::malformed_k;

        shared_lock_t shared_lock {shared_mutex_};
        node_t& node = nodes_[from];
        if constexpr (locked_k) node.lock();
        status_t status = status_t::malformed_k;
        for (std::size_t position = 0; position != node.neighbors.size(); ++position)
            if (node.neighbors[position] == to) {
                node.neighbors[position] = node.neighbors.back();
                [[maybe_unused]] node_id_t const dropped = node.neighbors.pop_back();
                status = status_t::success_k;
                break;
            }
        if constexpr (locked_k) node.unlock();
        return status;
    }

    /** Removals are deleted, not pending: ID immortality is what keeps every
     *  reader's stale ID valid. */
    status_t remove(std::span<key_t const>, traversal_context_t&) noexcept = delete;

#pragma endregion Mutation

#pragma region Refinement

    /** A refine, an absorb or a split's round knobs; zero derives. */
    struct refine_config_t {

        /** Bytes of cache each core keeps to itself, half for a neighborhood's entries and half for
         *  a draw's hubs; zero takes one storage chunk of entries as a neighborhood's capacity. */
        std::size_t cache_bytes = 0;
    };

    /** A split's knobs: its refine's, and the seed that orders the parts' growth. */
    struct split_config_t {
        refine_config_t refine {};
        u64_t seed = 0;
    };

    /**
     *  @brief One reachability audit in flight: the strong component of one drawn root over the
     *      published nodes, and the islands outside it. Read-only on the graph, so concurrent
     *      insertions merely leave fresh nodes uncounted.
     *
     *  Tasks are the IDs one bitset word at a time, so any number of threads may @ref sweep
     *  different tasks of one pass at once; @ref advance_depth is the boundary, one caller. A depth
     *  expands only the entries the depth before it reached, so depths count from the root and
     *  in-link and shallower in-link counts are exact, whatever order the tasks run in; the same
     *  read marks an entry's way to the root when one of its neighbors already reaches the root.
     *  Once a depth reaches nothing, rechecks re-read the entries reached but not yet marked. One
     *  thread then gathers the islands, the entries outside the root's component, into compact
     *  columns of their links among each other and finds their strong components; a drawn root
     *  whose component is smaller than one of them gives way once to that island's first entry. A
     *  count pass then reads every neighbor list once more against the in-link counts and files the
     *  least entry of the root's component linking to each island node, and the islands' roles,
     *  entries and weak components follow.
     */
    template <typename bits_allocator_type_>
    struct health_plan {

        /** One unit of a pass's fan-out. */
        struct task_t {

            /** The task's position in the plan - what a fan-out hands to a thread. */
            std::size_t position = 0;

            /** The IDs this task sweeps. */
            iota_view<node_id_t> nodes;
        };

        /** The audit's bitsets, one run of words each in @c bits_, in this order. */
        enum bitset_t : std::size_t {

            /** Reached from the root. */
            reached_k,

            /** Reaches the root through some neighbor in its list. */
            reaches_root_k,

            /** Links to two entries of smaller depth to root, so one link may go
             *  to a reconnection. */
            two_ways_to_root_k,

            /** The bitsets in all. */
            bitsets_k,
        };

        /** An island node's roles, the bits of its byte in @c island_roles_; the first two are its
         *  strong component's. */
        enum island_role_t : u8_t {

            /** No entry outside its component links into it. */
            unreached_island_k = 1,

            /** No link leaves its component. */
            stranded_island_k = 2,

            /** It links into the root's component. */
            links_to_component_k = 4,

            /** An entry of the root's component links to it. */
            linked_from_component_k = 8,
        };

        using bits_t = growing_buffer<u1x64_t, bits_allocator_type_>;
        using byte_column_allocator_t =
            typename std::allocator_traits<bits_allocator_type_>::template rebind_alloc<u8_t>;
        using byte_column_t = growing_buffer<u8_t, byte_column_allocator_t>;
        using island_ids_allocator_t =
            typename std::allocator_traits<bits_allocator_type_>::template rebind_alloc<node_id_t>;
        using island_ids_t = growing_buffer<node_id_t, island_ids_allocator_t>;
        using island_offsets_allocator_t =
            typename std::allocator_traits<bits_allocator_type_>::template rebind_alloc<std::size_t>;
        using island_offsets_t = growing_buffer<std::size_t, island_offsets_allocator_t>;

        /** Lanes one word carries - what an ID splits on. */
        inline static constexpr std::size_t lanes_k = std::numeric_limits<u64_t>::digits;

        /** The depth of a node no sweep reached, 255; the deepest a reached node records is 254,
         *  at which the depths saturate: every node reached from there on reads 254 and counts no
         *  shallower in-links, so no merge drops a link for a depth it cannot tell. */
        inline static constexpr u8_t unknown_depth_k = (std::numeric_limits<u8_t>::max)();

        /** The watermark the audit covers; IDs strictly below it are swept. */
        node_id_t visible = 0;

        /** An unbound plan - a refused audit's answer; owning its columns, it
         *  names an allocator. */
        explicit health_plan(bits_allocator_type_ const& allocator) noexcept
            : bits_(allocator), depths_from_root_(byte_column_allocator_t {allocator}),
              depths_to_root_(byte_column_allocator_t {allocator}),
              shallower_in_links_(byte_column_allocator_t {allocator}), in_links_(byte_column_allocator_t {allocator}),
              island_nodes_(island_ids_allocator_t {allocator}),
              island_offsets_(island_offsets_allocator_t {allocator}),
              island_links_(island_ids_allocator_t {allocator}), island_components_(island_ids_allocator_t {allocator}),
              island_parents_(island_ids_allocator_t {allocator}), island_entries_(island_ids_allocator_t {allocator}),
              island_roles_(byte_column_allocator_t {allocator}), island_walk_(island_ids_allocator_t {allocator}) {}
        health_plan(health_plan&& other) noexcept
            : visible(other.visible), index_(std::exchange(other.index_, nullptr)), bits_(std::move(other.bits_)),
              depths_from_root_(std::move(other.depths_from_root_)), depths_to_root_(std::move(other.depths_to_root_)),
              shallower_in_links_(std::move(other.shallower_in_links_)), in_links_(std::move(other.in_links_)),
              depth_entries_(other.depth_entries_), island_nodes_(std::move(other.island_nodes_)),
              island_offsets_(std::move(other.island_offsets_)), island_links_(std::move(other.island_links_)),
              island_components_(std::move(other.island_components_)),
              island_parents_(std::move(other.island_parents_)), island_entries_(std::move(other.island_entries_)),
              island_roles_(std::move(other.island_roles_)), island_walk_(std::move(other.island_walk_)),
              root_(other.root_), drawn_root_(other.drawn_root_), largest_island_(other.largest_island_),
              largest_island_entries_(other.largest_island_entries_), reached_in_depth_(other.reached_in_depth_),
              marked_in_depth_(other.marked_in_depth_), swept_entries_(other.swept_entries_), tallies_(other.tallies_),
              depth_(other.depth_), phase_(other.phase_), islands_status_(other.islands_status_) {}
        health_plan& operator=(health_plan&& other) noexcept {
            visible = other.visible;
            index_ = std::exchange(other.index_, nullptr);
            bits_ = std::move(other.bits_);
            depths_from_root_ = std::move(other.depths_from_root_);
            depths_to_root_ = std::move(other.depths_to_root_);
            shallower_in_links_ = std::move(other.shallower_in_links_);
            in_links_ = std::move(other.in_links_);
            depth_entries_ = other.depth_entries_;
            island_nodes_ = std::move(other.island_nodes_);
            island_offsets_ = std::move(other.island_offsets_);
            island_links_ = std::move(other.island_links_);
            island_components_ = std::move(other.island_components_);
            island_parents_ = std::move(other.island_parents_);
            island_entries_ = std::move(other.island_entries_);
            island_roles_ = std::move(other.island_roles_);
            island_walk_ = std::move(other.island_walk_);
            root_ = other.root_;
            drawn_root_ = other.drawn_root_;
            largest_island_ = other.largest_island_;
            largest_island_entries_ = other.largest_island_entries_;
            reached_in_depth_ = other.reached_in_depth_;
            marked_in_depth_ = other.marked_in_depth_;
            swept_entries_ = other.swept_entries_;
            tallies_ = other.tallies_;
            depth_ = other.depth_;
            phase_ = other.phase_;
            islands_status_ = other.islands_status_;
            return *this;
        }
        health_plan(health_plan const&) = delete;
        health_plan& operator=(health_plan const&) = delete;

        /** Task @p position of the plan, the nodes of one bitset word, since a task marks its
         *  own nodes without contention; an empty task past the end. */
        task_t operator[](std::size_t position) const noexcept {
            task_t task;
            if (position >= tasks().size()) return task;
            std::size_t const first = position * lanes_k;
            std::size_t const bound = min_of(first + lanes_k, static_cast<std::size_t>(visible));
            task.position = position;
            task.nodes = {static_cast<node_id_t>(first), static_cast<node_id_t>(bound)};
            return task;
        }

        /** The plan's tasks, positionally, one per bitset word; every pass sweeps all of them. */
        indexed_range<health_plan> tasks() const noexcept {
            return {this, index_ ? divide_round_up<std::size_t>(visible, lanes_k) : 0u};
        }

        /** Turns the audit onto @p index 's IDs below @p visible, which its columns must span, and
         *  restarts the sweep from a fresh draw. */
        void rebind(ram_index& index, node_id_t visible) noexcept {
            index_ = &index;
            this->visible = visible;
            root_ = missing_k;
            restart_sweep();
        }

        /** Takes the IDs up to @p visible, which its columns must span, in without a sweep: the
         *  nodes appended since read unreached both ways, at no depth, with no shallower
         *  in-links or in-links. */
        void widen(node_id_t visible) noexcept {
            if (visible <= this->visible) return;
            std::size_t const added = visible - this->visible;
            std::memset(depths_from_root_.data() + this->visible, unknown_depth_k, added);
            std::memset(depths_to_root_.data() + this->visible, unknown_depth_k, added);
            std::memset(shallower_in_links_.data() + this->visible, 0, added);
            std::memset(in_links_.data() + this->visible, 0, added);
            this->visible = visible;
        }

        /** Clears every column and starts from the root an earlier sweep kept, or else from a
         *  seeded draw, marked both ways at depth zero: the next @ref sweep starts depth one. */
        void restart_sweep() noexcept {
            // Opening and closing health checks share their root, so a different draw cannot
            // read as progress.
            bool const kept = root_ < visible;
            node_id_t const root = kept      ? root_
                                   : visible ? static_cast<node_id_t>(seeded_bijection(0, visible, 0))
                                             : missing_k;
            drawn_root_ = kept ? missing_k : root;
            start_from_(root);
        }

        /**
         *  @brief Runs @p task of the current pass: expands the entries of this depth, marking
         *      their way to the root in the same read; re-reads the entries reached but not yet
         *      marked; or counts every entry's links.
         *  @return The nodes this call reached, marked or counted.
         *
         *  Safe on any thread for disjoint tasks: the reached bits, depths, depths to root and
         *  counts other tasks read are written through relaxed atomics, and a task marks only its
         *  own nodes, each once.
         */
        std::size_t sweep(task_t const& task) noexcept {
            if (task.nodes.empty()) return 0;
            if (phase_ == phase_t::expand_k) return expand_(task);
            if (phase_ == phase_t::recheck_k) return recheck_reaches_root_(task);
            if (phase_ == phase_t::count_links_k) return count_in_and_out_links_(task);
            return 0;
        }

        /** The boundary between passes, one caller after every task ran: files the entries a depth
         *  reached, turns to the rechecks once a depth reaches nothing, gathers the islands and
         *  keeps the root or re-roots once a recheck marks nothing, and labels the islands after
         *  the count; zero means the audit converged or could not gather its islands. */
        std::size_t advance_depth() noexcept {
            if (phase_ == phase_t::expand_k) {
                std::size_t const reached = std::exchange(reached_in_depth_, 0);
                swept_entries_ += reached + std::exchange(marked_in_depth_, 0);
                if (!reached) {
                    phase_ = phase_t::recheck_k;
                    return 1;
                }
                depth_entries_[min_of<std::size_t>(++depth_, unknown_depth_k - 1u)] += reached;
                ++tallies_.depth_from_root;
                return reached;
            }
            if (phase_ == phase_t::recheck_k) {
                std::size_t const marked = std::exchange(marked_in_depth_, 0);
                if (!marked) return choose_root_();
                swept_entries_ += marked;
                ++tallies_.depth_to_root;
                return 1;
            }
            if (phase_ == phase_t::count_links_k) {
                label_islands_();
                phase_ = phase_t::done_k;
            }
            return 0;
        }

        /** The depth from the root the sweep reached @p node_id at, @c unknown_depth_k
         *  for none, which any thread may read between depths. */
        u8_t depth_from_root(node_id_t node_id) const noexcept { return depths_from_root_[node_id]; }

        /** One past the depth to root of the neighbor that marked @p node_id 's way to
         *  the root, zero for the root, @c unknown_depth_k for none, which any thread may
         *  read between depths. */
        u8_t depth_to_root(node_id_t node_id) const noexcept { return depths_to_root_[node_id]; }

        /** Reached entries linking to @p node_id, saturating at 255, which any thread may read
         *  between depths. */
        u8_t in_links(node_id_t node_id) const noexcept {
            return atomic_ref<u8_t const>(in_links_[node_id]).load(std::memory_order_relaxed);
        }

        /** Takes one of @p node_id 's links from the depth before it for a merge that drops it,
         *  answering whether another remains; a refusal leaves it untouched, so no node loses
         *  the last shallower in-link it is reached through. A count saturated at 255 refuses
         *  early, never late. */
        bool take_shallower_in_link_if_spare(node_id_t node_id) noexcept {
            atomic_ref<u8_t> const shallower_in_links {shallower_in_links_[node_id]};
            u8_t observed = shallower_in_links.load(std::memory_order_relaxed);
            while (observed > 1 &&
                   !shallower_in_links.compare_exchange_weak(observed, static_cast<u8_t>(observed - 1),
                                                             std::memory_order_relaxed, std::memory_order_relaxed)) {}
            return observed > 1;
        }

        /** Whether @p node_id was reached - readable by any thread between depths. */
        bool reached(node_id_t node_id) const noexcept { return contains_(reached_k, node_id); }

        /** Whether @p node_id reaches the root - readable by any thread between depths. */
        bool reaches_root(node_id_t node_id) const noexcept { return contains_(reaches_root_k, node_id); }

        /** Whether @p node_id is in the root's strong component: reached from it, reaching it. */
        bool in_component(node_id_t node_id) const noexcept { return reached(node_id) && reaches_root(node_id); }

        /** Takes one of @p node_id 's reached in-links for a reconnection, answering
         *  whether another remains; a refusal leaves it untouched, so no node loses its
         *  last reached in-link. */
        bool take_in_link_if_spare(node_id_t node_id) noexcept {
            return atomic_ref<u8_t>(in_links_[node_id])
                       .fetch_sub_if_at_least(u8_t {1}, u8_t {1}, std::memory_order_relaxed) > 1;
        }

        /** Takes the link from @p owner to @p resident for a swap: one not from the depth before
         *  the resident goes freely, a shallower in-link only while another remains, as @ref
         *  take_shallower_in_link_if_spare takes it, so no depth ever rises. */
        bool take_link_for_swap_if_spare(node_id_t owner, node_id_t resident) noexcept {
            u8_t const owner_depth = depth_from_root(owner);
            u8_t const child_depth = owner_depth < unknown_depth_k - 1 ? static_cast<u8_t>(owner_depth + 1)
                                                                       : unknown_depth_k;
            return child_depth == unknown_depth_k || depth_from_root(resident) != child_depth ||
                   take_shallower_in_link_if_spare(resident);
        }

        /** Takes the link from @p owner to @p resident for a swap under the owner's lock,
         *  answering whether the reach rule releases it: never the resident's last shallower
         *  in-link, and a link toward the root only while the owner keeps another at that
         *  depth to root. */
        bool take_link_under_lock_if_spare(node_id_t owner, node_id_t resident) noexcept {
            bool const way_to_root = reaches_root(owner) && depth_to_root(owner) != 0 &&
                                     depth_to_root(resident) < depth_to_root(owner);
            if (way_to_root && !contains_(two_ways_to_root_k, owner)) return false;
            if (!take_link_for_swap_if_spare(owner, resident)) return false;
            if (way_to_root) release_(two_ways_to_root_k, owner);
            return true;
        }

        /** Files a link from @p owner to island entry @p member written since the sweep in the
         *  columns a merge keeps reach by: a member at no depth takes the one past the owner's,
         *  with the owner as its one shallower in-link, and an owner with no way to the root
         *  takes one past the member's depth to root, so no later restitch drops the link. Only
         *  the task healing that island writes its entry's columns. */
        void record_heal_link(node_id_t owner, node_id_t member) noexcept {
            u8_t const owner_depth = depth_from_root(owner);
            if (depth_from_root(member) == unknown_depth_k && owner_depth < unknown_depth_k - 1) {
                atomic_ref<u8_t>(depths_from_root_[member])
                    .store(static_cast<u8_t>(owner_depth + 1), std::memory_order_relaxed);
                atomic_ref<u8_t>(shallower_in_links_[member]).store(1, std::memory_order_relaxed);
            }
            u8_t const member_depth_to_root = depth_to_root(member);
            if (depth_to_root(owner) == unknown_depth_k && member_depth_to_root < unknown_depth_k - 1)
                atomic_ref<u8_t>(depths_to_root_[owner])
                    .store(static_cast<u8_t>(member_depth_to_root + 1), std::memory_order_relaxed);
        }

        /** Counts a link written into @p node_id since the sweep, saturating. */
        void count_in_link(node_id_t node_id) noexcept { static_cast<void>(count_in_link_(node_id)); }

        /** Entries the sweeps reached plus entries they marked a way to the root for, of twice the
         *  audited ones: what a caller's progress counts. */
        std::size_t swept() const noexcept { return swept_entries_; }

        /** Entries outside the root's component, the islands' nodes, ranked by ID. */
        std::size_t islands() const noexcept { return island_nodes_.size(); }

        /** Whether island node @p rank names its strong component, being its first. */
        bool island_first(std::size_t rank) const noexcept { return island_components_[rank] == rank; }

        /** Island node @p rank 's @ref island_role_t bits, its component's two first among them. */
        u8_t island_roles(std::size_t rank) const noexcept { return island_roles_[rank]; }

        /** The entry standing for island node @p rank 's component: a member a link reaches or
         *  leaves the root's component through when one does, then the least linked one, with
         *  ties broken by a keyed draw. */
        node_id_t island_entry(std::size_t rank) const noexcept {
            return island_nodes_[island_entries_[island_components_[rank]]];
        }

        /** The entry of the root's component linking to the entry standing for island node
         *  @p rank 's component, or none. */
        node_id_t island_entry_parent(std::size_t rank) const noexcept {
            return island_parents_[island_entries_[island_components_[rank]]];
        }

        /** Takes one of @p node_id 's out-links for a reconnection, answering whether its list
         *  still reaches back through a neighbor of smaller depth to root; the root, and a node
         *  with no way to the root, have none to lose. */
        bool take_way_to_root_if_spare(node_id_t node_id) noexcept {
            return !contains_(reaches_root_k, node_id) || depth_to_root(node_id) == 0 ||
                   release_(two_ways_to_root_k, node_id);
        }

        /**
         *  @brief The report: the entries of the root's component, the depths each direction took,
         *      the in-link counts, and the islands' components. @c internal_k while a pass is
         *      pending, since the counts would then be lower bounds.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The audit is bound to no index.
         *  - @c bad_alloc_k The islands' columns could not be placed.
         */
        expected<reachability_t> finish() const noexcept {
            if (!index_) return {{}, status_t::uninitialized_k};
            if (phase_ != phase_t::done_k) return {{}, status_t::internal_k};
            if (failed(islands_status_)) return {{}, islands_status_};
            reachability_t result = tallies_;
            result.audited_nodes = visible;
            result.reachable_nodes = component_entries_();
            std::size_t most_populated = 0;
            for (std::size_t depth = 1; depth != depth_entries_.size(); ++depth)
                if (depth_entries_[depth] > depth_entries_[most_populated]) most_populated = depth;
            for (std::size_t depth = most_populated + 1; depth < depth_entries_.size(); ++depth)
                result.beyond_most_populated_depth += depth_entries_[depth];
            return {result, status_t::success_k};
        }

      private:
        friend ram_index;

        enum class phase_t : u8_t { expand_k, recheck_k, count_links_k, done_k };

        /** The smallest depth to root among the links one read met, and how many links met it. */
        struct way_to_root_t {
            u8_t lowest = unknown_depth_k;
            std::size_t at_lowest = 0;

            void note(u8_t depth_to_root) noexcept {
                if (depth_to_root < lowest) lowest = depth_to_root, at_lowest = 1;
                else at_lowest += depth_to_root == lowest && depth_to_root != unknown_depth_k;
            }
        };

        /** Sizes every column for the IDs below @p visible and turns the audit onto @p index ; a
         *  refusal leaves it unbound. */
        status_t bind_(ram_index& index, node_id_t visible) noexcept {
            std::size_t const words = divide_round_up<std::size_t>(visible, lanes_k);
            if (failed(bits_.resize(bitsets_k * words)) || failed(depths_from_root_.resize(visible)) ||
                failed(depths_to_root_.resize(visible)) || failed(shallower_in_links_.resize(visible)) ||
                failed(in_links_.resize(visible)))
                return status_t::bad_alloc_k;
            rebind(index, visible);
            return status_t::success_k;
        }

        /** Words per bitset. */
        std::size_t words_() const noexcept { return bits_.size() / bitsets_k; }

        /** The lanes of @p word below the watermark. */
        u1x64_t visible_lanes_(std::size_t word) const noexcept {
            std::size_t const lanes = min_of<std::size_t>(visible - word * lanes_k, lanes_k);
            return lanes == lanes_k ? ~u1x64_t {0} : (u1x64_t {1} << lanes) - 1;
        }

        u1x64_t word_(bitset_t bitset, std::size_t word) const noexcept {
            return atomic_ref<u1x64_t const>(bits_[bitset * words_() + word]).load(std::memory_order_relaxed);
        }

        bool contains_(bitset_t bitset, node_id_t node_id) const noexcept {
            return (word_(bitset, node_id / lanes_k) >> (node_id % lanes_k)) & 1u;
        }

        /** Sets @p node_id 's bit and answers whether it was already set. */
        bool insert_(bitset_t bitset, node_id_t node_id) noexcept {
            u1x64_t const lane = u1x64_t {1} << (node_id % lanes_k);
            return atomic_ref<u1x64_t>(bits_[bitset * words_() + node_id / lanes_k])
                       .fetch_or(lane, std::memory_order_relaxed) &
                   lane;
        }

        /** Clears @p node_id 's bit and answers whether it was set. */
        bool release_(bitset_t bitset, node_id_t node_id) noexcept {
            u1x64_t const lane = u1x64_t {1} << (node_id % lanes_k);
            return atomic_ref<u1x64_t>(bits_[bitset * words_() + node_id / lanes_k])
                       .fetch_and(~lane, std::memory_order_relaxed) &
                   lane;
        }

        static u8_t depth_of_(byte_column_t const& column, node_id_t node_id) noexcept {
            return atomic_ref<u8_t const>(column[node_id]).load(std::memory_order_relaxed);
        }

        static void add_(std::size_t& into, std::size_t value) noexcept {
            atomic_ref<std::size_t>(into).store_add(value, std::memory_order_relaxed);
        }

        /** Counts one more link into @p node_id from the depth before it, saturating. */
        void count_shallower_in_link_(node_id_t node_id) noexcept {
            atomic_ref<u8_t> const shallower_in_links {shallower_in_links_[node_id]};
            u8_t observed = shallower_in_links.load(std::memory_order_relaxed);
            while (observed != (std::numeric_limits<u8_t>::max)() &&
                   !shallower_in_links.compare_exchange_weak(observed, static_cast<u8_t>(observed + 1),
                                                             std::memory_order_relaxed, std::memory_order_relaxed)) {}
        }

        /** Counts one more reached link into @p node_id, saturating; answers how many it had. */
        u8_t count_in_link_(node_id_t node_id) noexcept {
            return atomic_ref<u8_t>(in_links_[node_id])
                .fetch_add_if_at_most(u8_t {1}, (std::numeric_limits<u8_t>::max)(), std::memory_order_relaxed);
        }

        /** Calls @p visit with every neighbor of @p node_id below the watermark but itself. */
        template <typename visit_type_>
        void for_each_link_(node_id_t node_id, visit_type_&& visit) const noexcept {
            // Bits and counts only, never a metric verb, so the node lock spans the whole list.
            node_t& node = index_->nodes_[node_id];
            node.lock();
            for (node_id_t const target : node.neighbors)
                if (target < visible && target != node_id) visit(target);
            node.unlock();
        }

        /** Entries of the root's component: reached from it and reaching it. */
        std::size_t component_entries_() const noexcept {
            std::size_t const words = words_();
            std::size_t entries = 0;
            for (std::size_t word = 0; word != words; ++word)
                entries += static_cast<std::size_t>(
                    std::popcount(bits_[reached_k * words + word] & bits_[reaches_root_k * words + word]));
            return entries;
        }

        /** Clears what one root's sweep fills and marks @p root both ways at depth zero; with no
         *  root the audit is over. */
        void start_from_(node_id_t root) noexcept {
            for (u1x64_t& lanes : bits_) lanes = 0;
            if (visible) {
                std::memset(depths_from_root_.data(), unknown_depth_k, visible);
                std::memset(depths_to_root_.data(), unknown_depth_k, visible);
                std::memset(shallower_in_links_.data(), 0, visible);
                std::memset(in_links_.data(), 0, visible);
            }
            depth_entries_.fill(0);
            for (island_ids_t* column : {&island_nodes_, &island_links_, &island_components_, &island_parents_,
                                         &island_entries_, &island_walk_})
                column->clear();
            island_offsets_.clear();
            island_roles_.clear();
            largest_island_ = missing_k;
            largest_island_entries_ = 0;
            reached_in_depth_ = marked_in_depth_ = 0;
            swept_entries_ = root == missing_k ? 0 : 2;
            tallies_ = {};
            depth_ = 0;
            islands_status_ = status_t::success_k;
            root_ = root;
            phase_ = root == missing_k ? phase_t::done_k : phase_t::expand_k;
            if (root == missing_k) return;
            insert_(reached_k, root);
            insert_(reaches_root_k, root);
            depths_from_root_[root] = 0;
            depths_to_root_[root] = 0;
            depth_entries_[0] = 1;
        }

        /** Certifies @p node_id 's way to the root one past the lowest depth to root it links
         *  to, answering whether it links to any entry reaching back; two links at that depth
         *  spare one link for a reconnection to take. */
        bool mark_reaches_root_(node_id_t node_id, way_to_root_t const& way_to_root) noexcept {
            if (way_to_root.lowest == unknown_depth_k) return false;
            u8_t const depth_to_root = static_cast<u8_t>(
                min_of<unsigned>(way_to_root.lowest + 1u, unknown_depth_k - 1u));
            atomic_ref<u8_t>(depths_to_root_[node_id]).store(depth_to_root, std::memory_order_relaxed);
            insert_(reaches_root_k, node_id);
            if (way_to_root.at_lowest > 1 && way_to_root.lowest < depth_to_root) insert_(two_ways_to_root_k, node_id);
            return true;
        }

        /** Expands the entries of @p task at this depth: each link counts an in-link and the first
         *  reaches its target one depth deeper; the same read marks the entry's way to the root. */
        std::size_t expand_(task_t const& task) noexcept {
            // Past the deepest recorded level an entry marks its expansion in its parents byte,
            // which no link counts into there.
            bool const saturated = depth_ >= unknown_depth_k - 1u;
            u8_t const own_depth = static_cast<u8_t>(min_of<std::size_t>(depth_, unknown_depth_k - 1u));
            u8_t const child_depth = static_cast<u8_t>(min_of<std::size_t>(depth_ + 1, unknown_depth_k - 1u));
            std::size_t reached_count = 0, marked_count = 0;
            for (u1x64_t lanes = word_(reached_k, task.position); lanes; lanes &= lanes - 1) {
                node_id_t const node_id = static_cast<node_id_t>(task.position * lanes_k + std::countr_zero(lanes));
                if (depth_of_(depths_from_root_, node_id) != own_depth) continue;
                if (saturated && std::exchange(shallower_in_links_[node_id], u8_t {1})) continue;
                way_to_root_t way_to_root;
                for_each_link_(node_id, [&](node_id_t target) noexcept {
                    if (!count_in_link_(target) && !insert_(reached_k, target)) {
                        atomic_ref<u8_t>(depths_from_root_[target]).store(child_depth, std::memory_order_relaxed);
                        ++reached_count;
                    }
                    // A depth still unknown is one another task reached this depth and is storing.
                    u8_t const target_depth = depth_of_(depths_from_root_, target);
                    if (child_depth != unknown_depth_k - 1 &&
                        (target_depth == child_depth || target_depth == unknown_depth_k))
                        count_shallower_in_link_(target);
                    way_to_root.note(depth_of_(depths_to_root_, target));
                });
                marked_count += depth_of_(depths_to_root_, node_id) == unknown_depth_k &&
                                mark_reaches_root_(node_id, way_to_root);
            }
            add_(reached_in_depth_, reached_count);
            add_(marked_in_depth_, marked_count);
            return reached_count + marked_count;
        }

        /** Re-reads the entries of @p task reached but not yet marked, marking each that now
         *  links to an entry reaching back. */
        std::size_t recheck_reaches_root_(task_t const& task) noexcept {
            std::size_t marked_count = 0;
            for (u1x64_t lanes = word_(reached_k, task.position) & ~word_(reaches_root_k, task.position); lanes;
                 lanes &= lanes - 1) {
                node_id_t const node_id = static_cast<node_id_t>(task.position * lanes_k + std::countr_zero(lanes));
                way_to_root_t way_to_root;
                for_each_link_(
                    node_id, [&](node_id_t target) noexcept { way_to_root.note(depth_of_(depths_to_root_, target)); });
                marked_count += mark_reaches_root_(node_id, way_to_root);
            }
            add_(marked_in_depth_, marked_count);
            return marked_count;
        }

        /** Counts the entries of @p task no reached entry links to and those linked to by fewer
         *  entries than they link to, and files the least entry of the root's component linking to
         *  each island node as its parent. */
        std::size_t count_in_and_out_links_(task_t const& task) noexcept {
            std::span<node_id_t const> const islands = to_span(island_nodes_);
            std::size_t unlinked = 0, underlinked = 0;
            for (node_id_t node_id = task.nodes.first; node_id < task.nodes.bound; ++node_id) {
                bool const member = in_component(node_id);
                std::size_t out_links = 0;
                for_each_link_(node_id, [&](node_id_t target) noexcept {
                    ++out_links;
                    if (!member || in_component(target)) return;
                    std::size_t const rank = static_cast<std::size_t>(lower_bound(islands, target) - islands.data());
                    if (rank != islands.size() && islands[rank] == target)
                        atomic_ref<node_id_t>(island_parents_[rank]).store_min(node_id, std::memory_order_relaxed);
                });
                unlinked += in_links_[node_id] == 0;
                underlinked += in_links_[node_id] < out_links;
            }
            add_(tallies_.unlinked, unlinked);
            add_(tallies_.underlinked, underlinked);
            return task.nodes.size();
        }

        /** Closes one root's sweep once a recheck marked nothing: gathers the islands and their
         *  strong components, then sweeps once more from the first entry of the largest island
         *  when it outgrows the drawn root's component, or turns to the count; islands that
         *  cannot be gathered end the audit with their status. */
        std::size_t choose_root_() noexcept {
            std::size_t const component = component_entries_();
            islands_status_ = gather_islands_(visible - component);
            if (failed(islands_status_)) {
                phase_ = phase_t::done_k;
                return 0;
            }
            close_islands_();
            if (root_ == drawn_root_ && largest_island_entries_ > component) {
                drawn_root_ = missing_k;
                start_from_(largest_island_);
                return 1;
            }
            phase_ = phase_t::count_links_k;
            return 1;
        }

        /** Gathers the @p islands entries outside the root's component in ascending order, their
         *  links among each other as ranks behind offsets, marks those linking into the component,
         *  and marks the way to the root of those, which no sweep reached; every island column is
         *  placed here, so the walks after it allocate nothing. */
        status_t gather_islands_(std::size_t islands) noexcept {
            if (failed(island_nodes_.reserve_exactly(islands)) ||
                failed(island_offsets_.reserve_exactly(islands + 1)) ||
                failed(island_components_.reserve_exactly(islands)) ||
                failed(island_parents_.reserve_exactly(islands)) || failed(island_entries_.reserve_exactly(islands)) ||
                failed(island_roles_.reserve_exactly(islands)) || failed(island_walk_.reserve_exactly(5 * islands)))
                return status_t::bad_alloc_k;
            std::size_t const words = words_();
            for (std::size_t word = 0; word != divide_round_up<std::size_t>(visible, lanes_k); ++word)
                for (u1x64_t lanes = ~(bits_[reached_k * words + word] & bits_[reaches_root_k * words + word]) &
                                     visible_lanes_(word);
                     lanes; lanes &= lanes - 1)
                    island_nodes_.push_back_reserved(static_cast<node_id_t>(word * lanes_k + std::countr_zero(lanes)));
            island_parents_.extend_uninitialized(islands);
            for (node_id_t& parent : island_parents_) parent = missing_k;
            island_roles_.extend_uninitialized(islands);
            if (islands) std::memset(island_roles_.data(), 0, islands);
            // Counted before placed, so no neighbor list stays locked across an allocation.
            std::size_t links = 0;
            for (std::size_t rank = 0; rank != islands; ++rank) {
                way_to_root_t way_to_root;
                for_each_link_(island_nodes_[rank], [&](node_id_t target) noexcept {
                    if (!in_component(target)) {
                        ++links;
                        return;
                    }
                    island_roles_[rank] |= links_to_component_k;
                    way_to_root.note(depth_to_root(target));
                });
                // A reached island never links into the component, or a recheck marked it.
                mark_reaches_root_(island_nodes_[rank], way_to_root);
            }
            if (failed(island_links_.reserve_exactly(links))) return status_t::bad_alloc_k;
            island_offsets_.push_back_reserved(0);
            for (std::size_t rank = 0; rank != islands; ++rank) {
                for_each_link_(island_nodes_[rank], [&](node_id_t target) noexcept {
                    // A list an insertion grew since the count keeps the links the count placed.
                    if (!in_component(target) && island_links_.size() != links)
                        island_links_.push_back_reserved(island_rank_(target));
                });
                island_offsets_.push_back_reserved(island_links_.size());
            }
            return status_t::success_k;
        }

        /** The rank of island node @p node_id among the islands' nodes. */
        node_id_t island_rank_(node_id_t node_id) const noexcept {
            std::span<node_id_t const> const nodes = to_span(island_nodes_);
            return static_cast<node_id_t>(lower_bound(nodes, node_id) - nodes.data());
        }

        /** Pops the strong component Tarjan's walk closed at @p rank off @p stack, names it by
         *  the rank of its first node, marks its nodes done in @p lowest, and marks the way to
         *  the root of its nodes, its successors' being final. */
        void close_component_(node_id_t* stack, std::size_t& stacked, std::size_t rank, node_id_t* lowest) noexcept {
            std::size_t from = stacked;
            while (stack[--from] != rank) {}
            std::span<node_id_t const> const members {stack + from, stacked - from};
            stacked = from;
            node_id_t first = members[0];
            for (node_id_t const member : members) first = min_of(first, member);
            if (members.size() > largest_island_entries_)
                largest_island_entries_ = members.size(), largest_island_ = island_nodes_[first];
            for (node_id_t const member : members) {
                island_components_[member] = first;
                lowest[member] = missing_k;
            }
            for (bool marked = true; marked;) {
                marked = false;
                for (node_id_t const member : members) {
                    if (depth_to_root(island_nodes_[member]) != unknown_depth_k) continue;
                    way_to_root_t way_to_root;
                    for (std::size_t link = island_offsets_[member]; link != island_offsets_[member + 1]; ++link)
                        way_to_root.note(depth_to_root(island_nodes_[island_links_[link]]));
                    marked |= mark_reaches_root_(island_nodes_[member], way_to_root);
                }
            }
        }

        /** Finds the islands' strong components by Tarjan's walk over their links, on one thread
         *  and without recursion, noting the largest. */
        void close_islands_() noexcept {
            std::size_t const islands = island_nodes_.size();
            island_components_.extend_uninitialized(islands);
            island_walk_.extend_uninitialized(5 * islands);
            // Visit order from one, zero unvisited; the least order each node reaches on the
            // stack, missing once its component closed; the stack; and the walk's path beside
            // the link each task on it reads next, counted from its node's first.
            node_id_t* const order = island_walk_.data();
            node_id_t* const lowest = order + islands;
            node_id_t* const stack = lowest + islands;
            node_id_t* const path = stack + islands;
            node_id_t* const cursors = path + islands;
            if (islands) std::memset(order, 0, islands * sizeof(node_id_t));
            std::size_t visits = 0, stacked = 0, depth = 0;
            auto visit = [&](std::size_t rank) noexcept {
                order[rank] = lowest[rank] = static_cast<node_id_t>(++visits);
                stack[stacked++] = path[depth] = static_cast<node_id_t>(rank);
                cursors[depth++] = 0;
            };
            for (std::size_t start = 0; start != islands; ++start) {
                if (order[start]) continue;
                visit(start);
                while (depth) {
                    std::size_t const rank = path[depth - 1];
                    if (std::size_t const link = island_offsets_[rank] + cursors[depth - 1];
                        link != island_offsets_[rank + 1]) {
                        ++cursors[depth - 1];
                        std::size_t const next = island_links_[link];
                        if (!order[next]) visit(next);
                        else if (lowest[next] != missing_k) lowest[rank] = min_of(lowest[rank], order[next]);
                        continue;
                    }
                    --depth;
                    if (lowest[rank] == order[rank]) close_component_(stack, stacked, rank, lowest);
                    if (depth) lowest[path[depth - 1]] = min_of(lowest[path[depth - 1]], lowest[rank]);
                }
            }
        }

        /** Marks the unreached and stranded islands, picks the entry standing for each, and counts
         *  the weak components, the root's among them, by union-find. */
        void label_islands_() noexcept {
            std::size_t const islands = island_nodes_.size();
            tallies_.components = 1;
            if (!islands) return;
            constexpr u8_t component_roles_k = unreached_island_k | stranded_island_k;
            for (std::size_t rank = 0; rank != islands; ++rank)
                if (island_components_[rank] == rank) island_roles_[rank] |= component_roles_k;
            for (std::size_t rank = 0; rank != islands; ++rank) {
                node_id_t const component = island_components_[rank];
                if (island_parents_[rank] != missing_k) {
                    island_roles_[rank] |= linked_from_component_k;
                    island_roles_[component] &= static_cast<u8_t>(~unreached_island_k);
                }
                if (island_roles_[rank] & links_to_component_k)
                    island_roles_[component] &= static_cast<u8_t>(~stranded_island_k);
                for (std::size_t link = island_offsets_[rank]; link != island_offsets_[rank + 1]; ++link) {
                    node_id_t const next = island_components_[island_links_[link]];
                    if (next == component) continue;
                    island_roles_[component] &= static_cast<u8_t>(~stranded_island_k);
                    island_roles_[next] &= static_cast<u8_t>(~unreached_island_k);
                }
            }
            for (std::size_t rank = 0; rank != islands; ++rank) {
                node_id_t const component = island_components_[rank];
                u8_t const roles = island_roles_[component] & component_roles_k;
                if (component != rank) island_roles_[rank] |= roles;
                else {
                    tallies_.unreached_islands += (roles & unreached_island_k) != 0;
                    tallies_.stranded_islands += (roles & stranded_island_k) != 0;
                }
            }

            // A member a link crosses through stands first, as the heal links through it; then the
            // fewest in-links, ties by a keyed draw, so the ID order never decides.
            auto stands_before = [&](std::size_t rank, std::size_t other) noexcept {
                auto crosses = [&](std::size_t member) noexcept {
                    return island_parents_[member] != missing_k || (island_roles_[member] & links_to_component_k) != 0;
                };
                if (crosses(rank) != crosses(other)) return crosses(rank);
                node_id_t const node = island_nodes_[rank], other_node = island_nodes_[other];
                if (in_links_[node] != in_links_[other_node]) return in_links_[node] < in_links_[other_node];
                return mix_bits(node) < mix_bits(other_node);
            };
            island_entries_.extend_uninitialized(islands);
            for (std::size_t rank = 0; rank != islands; ++rank)
                if (island_components_[rank] == rank) island_entries_[rank] = static_cast<node_id_t>(rank);
            for (std::size_t rank = 0; rank != islands; ++rank) {
                node_id_t& entry = island_entries_[island_components_[rank]];
                if (stands_before(rank, entry)) entry = static_cast<node_id_t>(rank);
            }

            // The walk's columns again, as union-find parents; the last slot stands for
            // the root's one.
            node_id_t* const weak = island_walk_.data();
            for (std::size_t rank = 0; rank <= islands; ++rank) weak[rank] = static_cast<node_id_t>(rank);
            auto find = [&](std::size_t rank) noexcept {
                while (weak[rank] != rank) rank = weak[rank] = weak[weak[rank]];
                return rank;
            };
            auto unite = [&](std::size_t first, std::size_t second) noexcept {
                first = find(first), second = find(second);
                if (first != second) weak[max_of(first, second)] = static_cast<node_id_t>(min_of(first, second));
            };
            for (std::size_t rank = 0; rank != islands; ++rank) {
                if (island_roles_[rank] & (links_to_component_k | linked_from_component_k)) unite(rank, islands);
                for (std::size_t link = island_offsets_[rank]; link != island_offsets_[rank + 1]; ++link)
                    unite(rank, island_links_[link]);
            }
            tallies_.components = 0;
            for (std::size_t rank = 0; rank <= islands; ++rank) tallies_.components += weak[rank] == rank;
        }

        /** The audited index; null on a moved-from plan. */
        ram_index* index_ = nullptr;

        /** Every bitset, one bit per swept node each, laid out as @ref bitset_t orders them. */
        bits_t bits_;

        /** Per swept node, its depth from the root. */
        byte_column_t depths_from_root_;

        /** Per swept node, one past the depth to root of the neighbor that marked its way
         *  to the root. */
        byte_column_t depths_to_root_;

        /** Per swept node, its links from the depth before it, saturating. */
        byte_column_t shallower_in_links_;

        /** Per swept node, the reached entries linking to it, saturating. */
        byte_column_t in_links_;

        /** Entries each depth reached; those past the fullest are the deepest. */
        std::array<std::size_t, unknown_depth_k> depth_entries_ {};

        /** The nodes outside the root's component, ascending. */
        island_ids_t island_nodes_;

        /** Per island node and one past the last, where its links start in @c island_links_. */
        island_offsets_t island_offsets_;

        /** Every island node's links to other island nodes, as their ranks, back to back. */
        island_ids_t island_links_;

        /** Per island node, its strong component: the rank of its first node. */
        island_ids_t island_components_;

        /** Per island node, the least entry of the root's component linking to it, or none. */
        island_ids_t island_parents_;

        /** Per island node naming its strong component, the rank of the member standing for it. */
        island_ids_t island_entries_;

        /** Per island node, its @ref island_role_t bits. */
        byte_column_t island_roles_;

        /** Tarjan's walk over the islands, then their union-find. */
        island_ids_t island_walk_;

        /** The node the sweeps start from, a seeded draw or the first entry of a larger island. */
        node_id_t root_ = missing_k;

        /** The root this audit drew, which may give way once to a larger island; missing for a
         *  root kept from an earlier audit, which never does. */
        node_id_t drawn_root_ = missing_k;

        /** The first entry of the largest island the last gather found, and its entries. */
        node_id_t largest_island_ = missing_k;
        std::size_t largest_island_entries_ = 0;

        /** Nodes the current depth reached and marked so far, summed through @c atomic_ref. */
        std::size_t reached_in_depth_ = 0;
        std::size_t marked_in_depth_ = 0;

        /** Entries the sweeps reached plus entries they marked a way to the root for, so far. */
        std::size_t swept_entries_ = 0;

        /** The kept root's depths and rechecks that moved, its in-link counts and its
         *  islands' counts, summed through @c atomic_ref where tasks add to them. */
        reachability_t tallies_ {};

        /** The depth the next forward sweep expands. */
        std::size_t depth_ = 0;

        /** The pass the next tasks run. */
        phase_t phase_ = phase_t::done_k;

        /** Whether the islands' columns could be placed. */
        status_t islands_status_ = status_t::success_k;
    };

    /* The audit a refine embeds, its columns on the index's own allocator. */
    using health_plan_t = health_plan<allocator_t>;

    /** Opens a reachability audit over the published nodes both ways from one drawn root, whose
     *  strong component it measures, and the islands outside it. The caller sweeps every task per
     *  pass until @c advance_depth() answers zero, then reads @c finish(). Its columns come from
     *  @p context 's allocator. */
    template <typename context_allocator_type_>
    expected<health_plan<context_allocator_type_>> plan_health(
        traversal_context_over<context_allocator_type_>& context) noexcept {
        return plan_audit_(static_cast<node_id_t>(size()), context.allocator);
    }

    /**
     *  @brief One refinement round in flight, from its opening health check to its closing one.
     *
     *  A plan from @ref plan_round runs one round: the opening health check, the islands it named
     *  healed without a distance, a seeded draw of hubs a core's cache holds, a gather scoring
     *  every published entry against them into the neighborhood of its nearest hub, and again
     *  inside every neighborhood over its capacity, a restitch of each neighborhood in one core's
     *  cache, every entry keeping the diverse links among its own and its nearest
     *  neighborhood-mates and no others, and the closing health check from the same root, which
     *  decides the verdict; what it leaves unreached heals again, checked again, while that
     *  shrinks. No phase walks the graph. Metrics need not be symmetric, so a restitch scores every
     *  ordered pair of its entries, each entry as the query. An entry outside the root's component
     *  keeps a link to its nearest candidate inside it, which a heal then links back, and every
     *  list keeps a slot past the budget for that heal.
     *
     *  An absorb moves every donor in first, one donor after another, each a part with its links
     *  shifted and no link across, then runs the round over everything. A split moves in reverse:
     *  the donor's nodes partition into one graph-local part per receiver, scatter into the
     *  receivers with the links inside each part rebased and the links across dropped, and
     *  publish; then every receiver runs one round in turn, the counts summing over them.
     *
     *  The plan holds the shared side and the claims closed from planning to its death, so batches
     *  answer @c contended_k and stop-the-world verbs refuse meanwhile, while searches run
     *  throughout. Every entry the opening health check reached both ways stays reached both ways
     *  throughout, and a round stopped between tasks leaves the graph correct.
     *
     *  Tasks of the current phase are disjoint ranges, so any number of threads may @ref run them
     *  at once, each with its own metric and a context @ref reserve grew; @ref advance is the
     *  single-caller barrier between phases and answers whether another phase follows. Planning
     *  reserves everything the plan keeps, so neither @ref run, @ref advance nor @ref finish
     *  allocates. A plan that dies unfinished reopens the claims, and the graph is as the last
     *  task left it.
     */
    struct [[nodiscard]] refine_plan_t {
        using task_t = typename health_plan_t::task_t;

        /** The health check the restitch reads its depths from and the heal sweeps. */
        health_plan_t health;

        /** Tasks of the current phase a driver already ran, so a cancelled call resumes at the next
         *  one. */
        std::size_t taken = 0;

        /** An unbound plan placed by @p allocator - what a refused plan answers with. */
        explicit refine_plan_t(allocator_t const& allocator) noexcept
            : health(allocator), changed_(allocator), donor_holds_(allocator), receiver_holds_(allocator),
              part_slots_(allocator), part_members_(allocator), part_extents_(allocator), hub_nodes_(allocator),
              entry_neighborhoods_(allocator), neighborhood_entries_(allocator), neighborhood_offsets_(allocator) {}
        refine_plan_t(refine_plan_t&& other) noexcept
            : health(std::move(other.health)), taken(other.taken), index_(std::exchange(other.index_, nullptr)),
              hold_(std::move(other.hold_)), changed_(std::move(other.changed_)), donors_(other.donors_),
              donor_holds_(std::move(other.donor_holds_)), absorbed_donors_(other.absorbed_donors_),
              split_donor_(std::exchange(other.split_donor_, nullptr)),
              split_donor_hold_(std::move(other.split_donor_hold_)), receivers_(other.receivers_),
              receiver_holds_(std::move(other.receiver_holds_)), part_slots_(std::move(other.part_slots_)),
              part_members_(std::move(other.part_members_)), part_extents_(std::move(other.part_extents_)),
              seed_(other.seed_), refined_receivers_(other.refined_receivers_),
              receivers_tally_(other.receivers_tally_), connectivity_(other.connectivity_),
              widest_list_(other.widest_list_), phase_(other.phase_), swept_after_(other.swept_after_),
              tally_(other.tally_), status_(other.status_), hubs_per_draw_(other.hubs_per_draw_),
              neighborhood_capacity_(other.neighborhood_capacity_), hub_nodes_(std::move(other.hub_nodes_)),
              entry_neighborhoods_(std::move(other.entry_neighborhoods_)),
              neighborhood_entries_(std::move(other.neighborhood_entries_)),
              neighborhood_offsets_(std::move(other.neighborhood_offsets_)), round_seed_(other.round_seed_),
              islands_left_(other.islands_left_) {}
        refine_plan_t(refine_plan_t const&) = delete;
        refine_plan_t& operator=(refine_plan_t const&) = delete;
        ~refine_plan_t() noexcept { release_(); }

        /** Task @p position of the current phase: a donor node moved, a health check's bitset word,
         *  the whole partition, the islands healed, a neighborhood capacity of entries gathered, or
         *  a neighborhood; an empty task past the end. */
        task_t operator[](std::size_t position) const noexcept {
            if (phase_ == phase_t::health_k) return health[position];
            task_t task;
            if (position >= tasks().size()) return task;
            iota_view<node_id_t> const range = phase_nodes_();
            std::size_t const width = phase_width_();
            std::size_t const first = range.first + position * width;
            std::size_t const bound = min_of(first + width, static_cast<std::size_t>(range.bound));
            task.position = position;
            task.nodes = {static_cast<node_id_t>(first), static_cast<node_id_t>(bound)};
            return task;
        }

        /** The current phase's tasks, positionally; none once the plan is over. */
        indexed_range<refine_plan_t> tasks() const noexcept {
            if (!index_ || phase_ == phase_t::done_k) return {this, 0u};
            if (phase_ == phase_t::partition_k) return {this, 1u};
            if (phase_ == phase_t::health_k) return {this, health.tasks().size()};
            return {this, divide_round_up<std::size_t>(phase_nodes_().size(), phase_width_())};
        }

        /**
         *  @brief Runs @p task of the current phase on the calling thread - safe for disjoint tasks
         *      of one phase, and never failing on a bound plan once @ref reserve grew @p context,
         *      since planning claimed, covered and reserved everything a task writes.
         *
         *  A copy task moves a donor's nodes in, the one partition task grows a split's parts, a
         *  scatter task moves donor nodes into their receivers, a health task expands the audit, a
         *  heal task links the round's islands, a gather task files a neighborhood capacity of
         *  entries under their nearest hubs, and a restitch task re-stitches one neighborhood, its
         *  entries' links picked by @p selector ; @p metric and @p context serve the gather and
         *  the restitch alone.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is bound to no index.
         */
        template <typename metric_type_, typename context_allocator_type_, typename selector_type_ = keep_diverse_t>
            requires metric<metric_type_, value_t, distance_t> && selector<selector_type_, ram_index, metric_type_>
        status_t run(task_t const& task, metric_type_& metric, traversal_context_over<context_allocator_type_>& context,
                     selector_type_ const& selector = {}) noexcept {
            if (!index_) return status_t::uninitialized_k;
            switch (phase_) {
            case phase_t::copy_k: copy_(task); break;
            case phase_t::partition_k: partition_(); break;
            case phase_t::scatter_k: scatter_(task); break;
            case phase_t::health_k: health.sweep(task); break;
            case phase_t::heal_k: heal_(task); break;
            case phase_t::gather_neighborhoods_k: gather_neighborhoods_(task, metric, context); break;
            case phase_t::restitch_k: restitch_neighborhoods_(task, metric, context, selector); break;
            default: break;
            }
            return status_t::success_k;
        }

        /** The barrier between phases, one caller after every task of the current one ran:
         *  publishes a copied donor, publishes a split's parts, draws the hubs, sorts the
         *  entries by neighborhood, or picks the next phase, and answers whether one follows. */
        bool advance() noexcept {
            if (!index_ || failed(status_)) return false;
            // Every donor moves in before the opening health check, which then takes all
            // of them in.
            if (phase_ == phase_t::copy_k) {
                publish_copied_();
                if (absorbed_donors_ == donors_.size()) phase_ = phase_t::health_k;
                return true;
            }
            if (phase_ == phase_t::partition_k) {
                phase_ = phase_t::scatter_k;
                return true;
            }
            if (phase_ == phase_t::scatter_k) {
                publish_parts_();
                bind_receiver_();
                return true;
            }
            if (phase_ == phase_t::health_k) {
                if (!health.advance_depth()) converge_();
            }
            else advance_round_();
            if (failed(status_)) return false;
            if (phase_ == phase_t::done_k && refined_receivers_ != receivers_.size()) settle_receiver_();
            return phase_ != phase_t::done_k;
        }

        /** Skips what is left of the round's phase in hand, at a barrier: an opening health
         *  check, or a heal not yet run, ends the round with nothing written; a hubs draw, a
         *  gather or a restitch sends the round to the closing health check, the heal's links
         *  and the neighborhoods already re-stitched standing; the closing health check runs on,
         *  heals none after it, and the verdict reads stopped. A no-op in the moves and once the
         *  round is over. */
        void stop() noexcept {
            if (rounding()) stop_round_();
        }

        /** The budget planning reserved every neighbor list for. */
        std::size_t connectivity() const noexcept { return connectivity_; }

        /** Whether the current phase copies a donor or moves a split's nodes - the phases no stop
         *  skips. */
        bool moving() const noexcept {
            return phase_ == phase_t::copy_k || phase_ == phase_t::partition_k || phase_ == phase_t::scatter_k;
        }

        /** Whether the plan is inside a round, whose every barrier a caller's progress hears: past
         *  the moves and short of its end. */
        bool rounding() const noexcept { return phase_ >= phase_t::health_k && phase_ != phase_t::done_k; }

        /** The phase the plan is in, by the name a progress line prints. */
        char const* phase_name() const noexcept {
            switch (phase_) {
            case phase_t::copy_k: return "copy";
            case phase_t::partition_k: return "partition";
            case phase_t::scatter_k: return "scatter";
            case phase_t::health_k: return "health";
            case phase_t::heal_k: return "heal";
            case phase_t::draw_hubs_k: return "hubs";
            case phase_t::gather_neighborhoods_k: return "gather";
            case phase_t::restitch_k: return "restitch";
            default: return "done";
            }
        }

        /** The position of the round's phase in hand among its six, from the opening health
         *  check to the closing one; zero outside a round's phases. */
        std::size_t ordinal() const noexcept {
            std::size_t const opening = static_cast<std::size_t>(phase_t::opening_k);
            std::size_t const closing = static_cast<std::size_t>(phase_t::restitch_k) - opening + 2;
            // A check after a closing heal closes the round too, as the one after the restitch.
            if (phase_ == phase_t::health_k) return swept_after_ == phase_t::opening_k ? 1 : closing;
            return rounding() ? static_cast<std::size_t>(phase_) - opening + 1 : 0;
        }

        /** Units of the round's phase in hand done so far, at a barrier before it advances: entries
         *  swept and marked by a health check, and otherwise all of the phase's, since each of its
         *  passes runs to its end. */
        std::size_t completed() const noexcept {
            if (phase_ == phase_t::health_k) return health.swept();
            return total();
        }

        /** Units of the round's phase in hand in all, which @ref completed counts toward:
         *  islands healed, hubs drawn, entries one gather pass files, or entries re-stitched;
         *  none in the moves. */
        std::size_t total() const noexcept {
            switch (phase_) {
            case phase_t::health_k: return 2 * std::size_t {health.visible};
            case phase_t::heal_k: return health.islands();
            case phase_t::draw_hubs_k: return hubs_for_(health.visible);
            case phase_t::gather_neighborhoods_k: return gathered_entries_();
            case phase_t::restitch_k: return health.visible;
            default: return 0;
            }
        }

        /** The first failure the plan met, which ended it; success otherwise. */
        status_t status() const noexcept { return status_; }

        /** Opens the next round once one ended, from the root the closing health check kept: the
         *  count and the changed marks start over, the health check takes in the entries the moves
         *  appended, and the opening health check sweeps again. A no-op mid-round and on a plan
         *  bound to no index. Contexts then need @ref reserve again. */
        void next_round() noexcept {
            if (!index_ || phase_ != phase_t::done_k || failed(status_)) return;
            tally_ = {};
            health.widen(static_cast<node_id_t>(index_->size()));
            if (failed(reserve_round_(health.visible))) return status_ = status_t::bad_alloc_k, void();
            open_health_(phase_t::opening_k);
        }

        /**
         *  @brief Grows @p context to what a gather or restitch task touches, which a driver calls
         *      before the plan's first task.
         *
         *  That is every entry's nearest neighborhood-mates over a neighborhood capacity, and its
         *  nearest inside the root's component, the distances of a quad of queries against one
         *  capacity of entries or of a selection's rounds, and the live links of one entry merged
         *  with its mates. An unbound plan needs none of it.
         *
         *  @par Statuses:
         *  - @c bad_alloc_k @p context cannot grow to a round's scratch.
         */
        template <typename context_allocator_type_>
        status_t reserve(traversal_context_over<context_allocator_type_>& context) const noexcept {
            if (!index_ || !neighborhood_capacity_) return status_t::success_k;
            std::size_t const capacity = neighborhood_capacity_, candidates = widest_list_ + connectivity_;
            if (failed(context.neighborhood_mates.reserve(capacity * (connectivity_ + 1))) ||
                failed(context.staged_distances.reserve(max_of(quad_lanes_k * capacity, 2 * candidates))) ||
                failed(context.results.reserve(candidates)) || failed(context.admitted.reserve(widest_list_)) ||
                failed(context.selected_neighbors.reserve(connectivity_)) ||
                failed(context.staged_ids.reserve(widest_list_)))
                return status_t::bad_alloc_k;
            return status_t::success_k;
        }

        /**
         *  @brief The round so far: complete once @ref advance answered false,
         *      otherwise @c cancelled_k.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is bound to no index.
         *  - @c cancelled_k The round stopped before @ref advance answered false, or was stopped.
         *  - @c bad_alloc_k A health check could not place its islands' adjacency.
         */
        expected<refine_report_t> finish() const noexcept {
            if (!index_) return {{}, status_t::uninitialized_k};
            if (failed(status_)) return {tally_, status_};
            bool const done = phase_ == phase_t::done_k && tally_.verdict != refine_verdict_t::stopped_k;
            return {tally_, done ? status_t::success_k : status_t::cancelled_k};
        }

      private:
        friend ram_index;

        enum class phase_t : u8_t {
            copy_k,
            partition_k,
            scatter_k,
            health_k,
            // What a round's opening health check follows, and the round's phases in their order.
            opening_k,
            heal_k,
            draw_hubs_k,
            gather_neighborhoods_k,
            restitch_k,
            done_k,
        };

        /** What a @ref add_or_swap_link_ found. */
        enum class taken_t : u8_t { written_k, present_k, refused_k };

        static void add_(std::size_t& into, std::size_t value) noexcept {
            atomic_ref<std::size_t>(into).store_add(value, std::memory_order_relaxed);
        }

        /** Reopens the claims of the refined index and of every receiver closed, and drops every
         *  hold; the plan is unbound after. */
        void release_() noexcept {
            if (index_) std::exchange(index_, nullptr)->reopen_();
            for (std::size_t receiver = 0; receiver != receiver_holds_.size(); ++receiver)
                receivers_[receiver]->reopen_();
            hold_ = {};
            donor_holds_.clear();
            receiver_holds_.clear();
            split_donor_hold_ = {};
        }

        /** The nodes the current phase tasks over: a donor's, a split donor's, the round's islands,
         *  the entries it gathers into neighborhoods, or its neighborhoods; none while it draws
         *  hubs, since the barrier closing that phase draws them. */
        iota_view<node_id_t> phase_nodes_() const noexcept {
            if (phase_ == phase_t::copy_k) return copied_nodes_();
            if (phase_ == phase_t::partition_k || phase_ == phase_t::scatter_k) return {0, split_nodes_()};
            if (phase_ == phase_t::heal_k) return {0, static_cast<node_id_t>(health.islands())};
            if (phase_ == phase_t::gather_neighborhoods_k) return {0, health.visible};
            if (phase_ == phase_t::restitch_k) return {0, static_cast<node_id_t>(neighborhoods_())};
            return {};
        }

        /** Nodes one task of the current phase spans: the whole partition, every island, a
         *  neighborhood capacity of entries gathered, or one node moved or neighborhood. */
        std::size_t phase_width_() const noexcept {
            if (phase_ == phase_t::partition_k) return max_of<std::size_t>(split_nodes_(), 1);
            // One heal task: islands sharing an owner list then link in one order.
            if (phase_ == phase_t::heal_k) return max_of<std::size_t>(health.islands(), 1);
            if (phase_ == phase_t::gather_neighborhoods_k) return neighborhood_capacity_;
            return 1;
        }

        /** Nodes the split's donor published at planning: one slot each. */
        node_id_t split_nodes_() const noexcept { return static_cast<node_id_t>(part_slots_.size()); }

        /** The most nodes one part takes: the donor's nodes over the receivers, rounded up. */
        std::size_t part_capacity_() const noexcept {
            return divide_round_up<std::size_t>(part_slots_.size(), receivers_.size());
        }

        /** The current donor's nodes, in its own ID space. */
        iota_view<node_id_t> copied_nodes_() const noexcept {
            return {0, static_cast<node_id_t>(donors_[absorbed_donors_]->size())};
        }

        /** Marks node @p node_id counted as changed, answering whether it was already. */
        bool count_changed_(node_id_t node_id) noexcept {
            return atomic_ref<u1x64_t>(changed_[node_id / health_plan_t::lanes_k])
                .fetch_test_and_set_bit(static_cast<unsigned>(node_id % health_plan_t::lanes_k),
                                        std::memory_order_relaxed);
        }

        /** Moves the nodes of @p task from the current donor into the records past the watermark,
         *  every neighbor ID shifted by it; the nodes swap buffers, so nothing allocates. */
        void copy_(task_t const& task) noexcept {
            ram_index& donor = *donors_[absorbed_donors_];
            node_id_t const base = static_cast<node_id_t>(index_->size());
            for (node_id_t donor_node = task.nodes.first; donor_node < task.nodes.bound; ++donor_node) {
                node_t& from = donor.nodes_[donor_node];
                node_t& into = index_->nodes_[base + donor_node];
                into.value = std::move(from.value);
                into.key = from.key;
                into.neighbors = std::move(from.neighbors);
                for (node_id_t& neighbor : into.neighbors) neighbor += base;
            }
        }

        /** Files the current donor's keys and publishes its nodes at once, the closed claims making
         *  this barrier the map's one writer, then empties the donor, releases it, and nulls its
         *  slot. */
        void publish_copied_() noexcept {
            ram_index& donor = *donors_[absorbed_donors_];
            node_id_t const copied = static_cast<node_id_t>(donor.size());
            node_id_t const base = static_cast<node_id_t>(index_->size());
            if (copied) {
                index_->publish_keys_({base, static_cast<node_id_t>(base + copied)});
                atomic_ref<node_id_t>(index_->next_dispensed_id_).store(base + copied, std::memory_order_relaxed);
                atomic_ref<node_id_t>(index_->last_visible_id_).store(base + copied, std::memory_order_release);
            }
            donor.forget_(copied);
            donor_holds_[absorbed_donors_].unlock();
            donors_[absorbed_donors_] = nullptr;
            ++absorbed_donors_;
        }

        /** Places donor node @p node last in @p part, its slot naming the part and the
         *  position. */
        void enlist_(node_id_t node, std::size_t part) noexcept {
            std::size_t const position = part_extents_[part].nodes++, capacity = part_capacity_();
            part_members_[part * capacity + position] = node;
            part_slots_[node] = static_cast<node_id_t>(part * capacity + position);
        }

        /** Expands the oldest unexpanded member of @p part, enlisting its unplaced neighbors while
         *  the part has room; answers whether the part could expand at all. */
        bool expand_(std::size_t part) noexcept {
            node_id_t& expanded = part_extents_[part].expanded;
            std::size_t const capacity = part_capacity_();
            if (expanded == part_extents_[part].nodes || part_extents_[part].nodes == capacity) return false;
            node_id_t const node_id = part_members_[part * capacity + expanded++];
            for (node_id_t const target : split_donor_->nodes_[node_id].neighbors)
                if (target < split_nodes_() && part_slots_[target] == missing_k &&
                    part_extents_[part].nodes != capacity)
                    enlist_(target, part);
            return true;
        }

        /** Grows one part per receiver from a root the seed picks, breadth first over the source's
         *  links and one expansion per part in turn, each capped at the part capacity; every node
         *  left unplaced then seeds a growth of the smallest part. Every node's slot names its part
         *  and its ID in the receiver, in the order the growth enlisted it. Reads the donor without
         *  locks, since its unique side is held. */
        void partition_() noexcept {
            std::size_t const parts = receivers_.size();
            node_id_t const split_nodes = split_nodes_();
            for (node_id_t& slot : part_slots_) slot = missing_k;
            for (std::size_t part = 0; part != parts; ++part) part_extents_[part] = {};
            for (std::size_t part = 0; part != parts && part != split_nodes; ++part)
                enlist_(static_cast<node_id_t>(seeded_bijection(part, split_nodes, seed_)), part);
            for (std::size_t expanded = parts; expanded;) {
                expanded = 0;
                for (std::size_t part = 0; part != parts; ++part) expanded += expand_(part);
            }
            for (node_id_t node = 0; node != split_nodes; ++node) {
                if (part_slots_[node] != missing_k) continue;
                std::size_t smallest = 0;
                for (std::size_t part = 1; part != parts; ++part)
                    if (part_extents_[part].nodes < part_extents_[smallest].nodes) smallest = part;
                enlist_(node, smallest);
                while (expand_(smallest)) {}
            }
        }

        /** Moves the donor nodes of @p task into their receivers' records, the links inside each
         *  part rebased through its slots and the links across dropped; the nodes swap buffers, so
         *  nothing allocates. */
        void scatter_(task_t const& task) noexcept {
            std::size_t const capacity = part_capacity_();
            node_id_t const split_nodes = split_nodes_();
            for (node_id_t donor_node = task.nodes.first; donor_node < task.nodes.bound; ++donor_node) {
                std::size_t const slot = part_slots_[donor_node], part = slot / capacity;
                node_t& from = split_donor_->nodes_[donor_node];
                node_t& into = receivers_[part]->nodes_[slot % capacity];
                into.value = std::move(from.value);
                into.key = from.key;
                into.neighbors = std::move(from.neighbors);
                std::size_t kept = 0;
                for (node_id_t const target : into.neighbors)
                    if (target < split_nodes && part_slots_[target] / capacity == part)
                        into.neighbors[kept++] = static_cast<node_id_t>(part_slots_[target] % capacity);
                into.neighbors.resize_reserved(kept);
            }
        }

        /** Files every receiver's keys and publishes its nodes at once, the closed claims making
         *  this barrier each map's one writer, then empties the donor and releases it. */
        void publish_parts_() noexcept {
            for (std::size_t part = 0; part != receivers_.size(); ++part) {
                ram_index& into = *receivers_[part];
                node_id_t const nodes = part_extents_[part].nodes;
                if (!nodes) continue;
                into.publish_keys_({0, nodes});
                atomic_ref<node_id_t>(into.next_dispensed_id_).store(nodes, std::memory_order_relaxed);
                atomic_ref<node_id_t>(into.last_visible_id_).store(nodes, std::memory_order_release);
            }
            split_donor_->forget_(split_nodes_());
            split_donor_hold_.unlock();
            split_donor_ = nullptr;
        }

        /** Turns the plan to the next receiver: its health check over the nodes its part received,
         *  their changed marks cleared, and its round. */
        void bind_receiver_() noexcept {
            std::size_t const receiver = refined_receivers_;
            index_ = receivers_[receiver];
            health.rebind(*index_, part_extents_[receiver].nodes);
            std::memset(changed_.data(), 0, changed_.size() * sizeof(u1x64_t));
            swept_after_ = phase_t::opening_k;
            phase_ = phase_t::health_k;
        }

        /** Folds a receiver's round into the split's, then binds the next receiver, or reports the
         *  sums once the last one's round ended. */
        void settle_receiver_() noexcept {
            if (refined_receivers_) receivers_tally_ += tally_;
            else receivers_tally_ = tally_;
            if (++refined_receivers_ == receivers_.size()) return tally_ = receivers_tally_, void();
            tally_ = {};
            bind_receiver_();
        }

        /** Opens the health check following @p phase from the root the last one kept: the opening
         *  one, or the closing one after the restitch. */
        void open_health_(phase_t phase) noexcept {
            health.restart_sweep();
            swept_after_ = phase;
            phase_ = phase_t::health_k;
        }

        /** Takes in a converged health check: the opening one opens the round, the closing one
         *  decides its verdict. */
        void converge_() noexcept {
            auto const [reachability, status] = health.finish();
            // An adjacency the audit could not place fails the plan, never reads as no islands.
            if (failed(status)) {
                status_ = status;
                phase_ = phase_t::done_k;
                return;
            }
            tally_.reachability = reachability;
            if (swept_after_ == phase_t::opening_k) return open_round_(reachability);
            // What the restitch left unreached heals with no distance, again while that shrinks.
            std::size_t const left = reachability.audited_nodes - reachability.reachable_nodes;
            if (left && left < islands_left_ && tally_.verdict != refine_verdict_t::stopped_k) {
                islands_left_ = left;
                phase_ = phase_t::heal_k;
                return;
            }
            tally_.verdict = concluded_verdict(tally_);
            phase_ = phase_t::done_k;
        }

        /** Takes in the opening health check: seeds the round's draws from what it measured, so
         *  a round over a changed graph draws anew while the ID order decides nothing, and
         *  opens the heal. */
        void open_round_(reachability_t const& opening) noexcept {
            tally_.opening = opening;
            tally_.islands_found = opening.audited_nodes - opening.reachable_nodes;
            round_seed_ = mix_bits(
                opening.audited_nodes ^
                mix_bits(opening.reachable_nodes ^
                         mix_bits(opening.unlinked ^ mix_bits(opening.underlinked ^ opening.depth_from_root))));
            islands_left_ = (std::numeric_limits<std::size_t>::max)();
            phase_ = health.visible ? phase_t::heal_k : phase_t::done_k;
        }

        /** A round's phase after the barrier closing @c phase_: the hubs draw, the entries
         *  sort by neighborhood, or the next phase opens; a heal after the closing health
         *  check checks again. */
        void advance_round_() noexcept {
            switch (phase_) {
            case phase_t::heal_k:
                if (swept_after_ == phase_t::opening_k) phase_ = phase_t::draw_hubs_k;
                else open_health_(phase_t::heal_k);
                break;
            case phase_t::draw_hubs_k:
                draw_hubs_();
                phase_ = phase_t::gather_neighborhoods_k;
                break;
            case phase_t::gather_neighborhoods_k:
                // Neighborhoods over capacity gather again among their own entries.
                order_neighborhoods_();
                if (tally_.widest_neighborhood <= neighborhood_capacity_) phase_ = phase_t::restitch_k;
                break;
            case phase_t::restitch_k: open_health_(phase_t::restitch_k); break;
            default: break;
            }
        }

        /** Skips what is left of the round's phase in hand, as @ref stop describes. */
        void stop_round_() noexcept {
            tally_.verdict = refine_verdict_t::stopped_k;
            switch (phase_) {
            case phase_t::health_k:
                if (swept_after_ == phase_t::opening_k) phase_ = phase_t::done_k;
                break;
            case phase_t::heal_k:
                if (swept_after_ == phase_t::opening_k) tally_.reachability = tally_.opening;
                phase_ = phase_t::done_k;
                break;
            case phase_t::draw_hubs_k:
            case phase_t::gather_neighborhoods_k:
            case phase_t::restitch_k: open_health_(phase_t::restitch_k); break;
            default: break;
            }
        }

        /** Sizes the hubs and the neighborhood capacity under @p cache_bytes of cache a core,
         *  half for a neighborhood's entries of @p entry_bytes each and half for a draw's hubs,
         *  or one storage chunk of entries for none; a neighborhood holds an entry and its
         *  mates at least. */
        void size_round_(std::size_t cache_bytes, std::size_t entry_bytes) noexcept {
            std::size_t const capacity = cache_bytes ? cache_bytes / (2 * entry_bytes) : nodes_schedule_t::nodes_in(0);
            neighborhood_capacity_ = max_of(capacity, connectivity_ + 1);
            hubs_per_draw_ = neighborhood_capacity_;
        }

        /** Reserves everything a round keeps across its barriers for @p entries entries: the
         *  changed marks, one draw of hubs, per entry its neighborhood and its place in the order
         *  by neighborhood, and the starts every split may write. */
        status_t reserve_round_(std::size_t entries) noexcept {
            if (failed(changed_.resize(divide_round_up(entries, health_plan_t::lanes_k))) ||
                failed(hub_nodes_.reserve_exactly(hubs_for_(entries))) ||
                failed(entry_neighborhoods_.resize(entries)) || failed(neighborhood_entries_.resize(entries)) ||
                failed(neighborhood_offsets_.reserve_exactly(split_room(entries, hubs_per_draw_))))
                return status_t::bad_alloc_k;
            std::memset(changed_.data(), 0, changed_.size() * sizeof(u1x64_t));
            return status_t::success_k;
        }

        /** Hubs one gather over @p entries draws, as @ref round_hubs counts them. */
        std::size_t hubs_for_(std::size_t entries) const noexcept {
            return round_hubs(entries, hubs_per_draw_, neighborhood_capacity_);
        }

        /** Neighborhoods the order by neighborhood holds. */
        std::size_t neighborhoods_() const noexcept {
            return neighborhood_offsets_.size() ? neighborhood_offsets_.size() - 1 : 0;
        }

        /** Entries in neighborhood @p neighborhood. */
        std::size_t neighborhood_width_(std::size_t neighborhood) const noexcept {
            return neighborhood_offsets_[neighborhood + 1] - neighborhood_offsets_[neighborhood];
        }

        /** Entries the gather pass in hand files: every audited one, then those of the
         *  neighborhoods over capacity. */
        std::size_t gathered_entries_() const noexcept {
            if (neighborhood_offsets_.empty()) return health.visible;
            std::size_t entries = 0;
            for (std::size_t neighborhood = 0; neighborhood != neighborhoods_(); ++neighborhood)
                if (neighborhood_width_(neighborhood) > neighborhood_capacity_)
                    entries += neighborhood_width_(neighborhood);
            return entries;
        }

        /** Draws one set of neighborhood hubs by seeded bijection over the visible entries,
         *  ascending. */
        void draw_hubs_() noexcept {
            std::size_t const visible = health.visible, hubs = hubs_for_(visible);
            hub_nodes_.clear();
            for (std::size_t rank = 0; rank != hubs; ++rank)
                hub_nodes_.push_back_reserved(static_cast<node_id_t>(seeded_bijection(rank, visible, round_seed_)));
            partial_sort(std::span<node_id_t> {hub_nodes_.data(), hubs}, hubs);
            neighborhood_offsets_.clear();
        }

        /** The first @p count of @p context 's neighborhood-mates, each an entry's nearest hub so
         *  far. */
        template <typename context_type_>
        static std::span<scored_node_t> nearest_hubs_(context_type_& context, std::size_t count) noexcept {
            context.neighborhood_mates.resize_reserved(count);
            std::span<scored_node_t> const nearest {context.neighborhood_mates.data(), count};
            for (scored_node_t& kept : nearest) kept = farthest_;
            return nearest;
        }

        /** The neighborhood an entry's @p nearest hub gives it; the first for one no distance
         *  reached. */
        static u16_t neighborhood_of_(scored_node_t nearest) noexcept {
            return static_cast<u16_t>(nearest.node_id == missing_k ? 0 : nearest.node_id);
        }

        /** Files the entries of @p task under their nearest hubs. Once ordered by neighborhood, a
         *  task spans positions in that order instead, and only the neighborhoods over capacity
         *  gather again, each against hubs drawn among its own entries. */
        template <typename metric_type_, typename context_type_>
        void gather_neighborhoods_(task_t const& task, metric_type_& metric, context_type_& context) noexcept {
            traversal_stats_t const before = context.telemetry;
            if (neighborhood_offsets_.empty()) {
                std::span<scored_node_t> const nearest = nearest_hubs_(context, task.nodes.size());
                file_under_nearest_(
                    hub_nodes_.size(), [&](std::size_t rank) noexcept { return hub_nodes_[rank]; },
                    [&](std::size_t at) noexcept { return static_cast<node_id_t>(task.nodes.first + at); }, nearest,
                    metric, context);
                for (node_id_t node_id = task.nodes.first; node_id < task.nodes.bound; ++node_id)
                    entry_neighborhoods_[node_id] = neighborhood_of_(nearest[node_id - task.nodes.first]);
            }
            else {
                std::span<node_id_t const> const offsets = to_span(neighborhood_offsets_);
                std::size_t neighborhood = static_cast<std::size_t>(lower_bound(offsets, task.nodes.first + 1) -
                                                                    offsets.data());
                for (--neighborhood; neighborhood + 1 < offsets.size() && offsets[neighborhood] < task.nodes.bound;
                     ++neighborhood)
                    if (neighborhood_width_(neighborhood) > neighborhood_capacity_)
                        gather_run_(neighborhood,
                                    {max_of(offsets[neighborhood], task.nodes.first),
                                     min_of(offsets[neighborhood + 1], task.nodes.bound)},
                                    metric, context);
            }
            add_spent_(context.telemetry - before);
        }

        /** Files the entries at positions @p run of wide neighborhood @p neighborhood under
         *  the nearest of the hubs that neighborhood draws among its own entries, the same
         *  in every task. */
        template <typename metric_type_, typename context_type_>
        void gather_run_(std::size_t neighborhood, iota_view<node_id_t> run, metric_type_& metric,
                         context_type_& context) noexcept {
            node_id_t const first = neighborhood_offsets_[neighborhood];
            std::size_t const entries = neighborhood_width_(neighborhood), hubs = hubs_for_(entries);
            u64_t const seed = mix_bits(round_seed_ ^ first);
            std::span<scored_node_t> const nearest = nearest_hubs_(context, run.size());
            file_under_nearest_(
                hubs,
                [&](std::size_t rank) noexcept {
                    return neighborhood_entries_[first + seeded_bijection(rank, entries, seed)];
                },
                [&](std::size_t at) noexcept { return neighborhood_entries_[run.first + at]; }, nearest, metric,
                context);
            for (std::size_t position = 0; position != run.size(); ++position)
                entry_neighborhoods_[neighborhood_entries_[run.first + position]] = neighborhood_of_(nearest[position]);
        }

        /** Scores the entries @p entry_at names, as many as @p nearest holds, against the
         *  @p hubs hubs @p hub_at names, a quad of borrowed hubs per batched round, keeping each
         *  entry's nearest hub in @p nearest, ties to the lower hub. */
        template <typename hub_at_type_, typename entry_at_type_, typename metric_type_, typename context_type_>
        void file_under_nearest_(std::size_t hubs, hub_at_type_ const& hub_at, entry_at_type_ const& entry_at,
                                 std::span<scored_node_t> nearest, metric_type_& metric,
                                 context_type_& context) noexcept {
            std::size_t const entries = nearest.size();
            auto const value_at = [&](std::size_t at) noexcept -> value_t const& {
                return index_->published_value_of(entry_at(at));
            };
            quad<distance_t> unbounded;
            unbounded.fill(unbounded_radius<distance_t>());
            for (std::size_t first_hub = 0; first_hub < hubs; first_hub += quad_lanes_k) {
                std::size_t const lanes = min_of<std::size_t>(quad_lanes_k, hubs - first_hub);
                quad<value_t const*> borrowed;
                for (std::size_t lane = 0; lane != lanes; ++lane)
                    borrowed[lane] = &index_->published_value_of(hub_at(first_hub + lane));
                std::span<value_t const* const> const queries {borrowed.data(), lanes};
                open_queries(metric, queries);
                for (std::size_t first = 0; first != entries;) {
                    context.staged_distances.resize_reserved(lanes * (entries - first));
                    std::span<distance_t> const distances {context.staged_distances.data(), lanes * (entries - first)};
                    std::size_t const end = score_queries(metric, queries, value_at, first, entries,
                                                          std::span<distance_t const> {unbounded.data(), lanes},
                                                          distances);
                    if (end == first) {
                        ++context.telemetry.saturated_walks;
                        break;
                    }
                    std::size_t const scored = end - first;
                    for (std::size_t lane = 0; lane != lanes; ++lane)
                        for (std::size_t column = 0; column != scored; ++column) {
                            distance_t const distance = distances[lane * scored + column];
                            if (distance < nearest[first + column].distance)
                                nearest[first + column] = {static_cast<node_id_t>(first_hub + lane), distance};
                        }
                    context.telemetry.computed_distances += lanes * scored;
                    first = end;
                }
            }
        }

        /** Sorts the entries by neighborhood, as @ref order_by_neighborhood does after a draw
         *  and @ref split_wide_neighborhoods after a gather over the neighborhoods over
         *  capacity, in the starts planning reserved; the tally's widest neighborhood tells
         *  whether the gather runs once more over those alone. */
        void order_neighborhoods_() noexcept {
            node_id_t const visible = health.visible;
            std::span<u16_t const> const tags {entry_neighborhoods_.data(), visible};
            std::span<node_id_t> const entries {neighborhood_entries_.data(), visible};
            std::size_t const neighborhoods = neighborhoods_();
            bool const first = neighborhood_offsets_.empty();
            neighborhood_offsets_.resize_reserved(first ? hub_nodes_.size() + 2 : split_room(visible, hubs_per_draw_));
            std::span<node_id_t> const offsets = to_writable_span(neighborhood_offsets_);
            neighborhood_order_t const order = first ? order_by_neighborhood(tags, hub_nodes_.size(), entries, offsets)
                                                     : split_wide_neighborhoods(tags, entries, offsets, neighborhoods,
                                                                                neighborhood_capacity_, hubs_per_draw_);
            neighborhood_offsets_.resize_reserved(order.neighborhoods + 1);
            tally_.neighborhoods = order.neighborhoods;
            tally_.widest_neighborhood = order.widest;
        }

        /** Re-stitches the neighborhood of @p task: scores it against itself, and re-selects every
         *  entry's links among its own and its nearest neighborhood-mates through @p selector ,
         *  keeping only what the selector keeps and what @ref keep_reach_ needs. One thread owns
         *  the neighborhood and writes only its entries' lists, each under its lock. */
        template <typename metric_type_, typename context_type_, typename selector_type_>
        void restitch_neighborhoods_(task_t const& task, metric_type_& metric, context_type_& context,
                                     selector_type_ const& selector) noexcept {
            traversal_stats_t const before = context.telemetry;
            std::size_t const count = neighborhood_width_(task.position), connectivity = connectivity_;
            std::span<node_id_t const> const entries {
                neighborhood_entries_.data() + neighborhood_offsets_[task.position], count};
            context.neighborhood_mates.resize_reserved(count * (connectivity + 1));
            std::span<scored_node_t> const mates {context.neighborhood_mates.data(), count * connectivity};
            std::span<scored_node_t> const inward {context.neighborhood_mates.data() + count * connectivity, count};
            score_mates_(entries, mates, inward, metric, context);

            snapshot_admission_t const every {health.visible};
            typename context_type_::results_array_t& results = context.results;
            typename context_type_::results_array_t& live = context.admitted;
            typename context_type_::template scratch_buffer<scored_node_t>& selected = context.selected_neighbors;
            std::size_t added = 0, dropped = 0, changed = 0;
            for (std::size_t position = 0; position != count; ++position) {
                node_id_t const node_id = entries[position];
                value_t const& value = index_->published_value_of(node_id);
                results.clear();
                index_->add_neighbor_candidates_(node_id, value, metric, every, context);
                live.clear();
                for (std::size_t rank = 0; rank != results.size(); ++rank)
                    live.insert(scored_node_t {results[rank]}, live.size() + 1);
                for (scored_node_t const mate : mates.subspan(position * connectivity, connectivity)) {
                    if (mate.node_id == missing_k) continue;
                    scored_node_t const candidate {entries[mate.node_id], mate.distance};
                    bool listed = false;
                    for (std::size_t rank = 0; rank != live.size() && !listed; ++rank)
                        listed = live[rank].node_id == candidate.node_id;
                    if (!listed) results.insert(scored_node_t {candidate}, results.size() + 1);
                }
                // The octant selector reads its base off the open round.
                metric.reset(value);
                selector(*index_, metric, context);
                index_->keep_reach_(node_id, live, selected, connectivity, health);
                if (!health.in_component(node_id))
                    keep_link_into_component_(
                        results,
                        inward[position].node_id == missing_k
                            ? inward[position]
                            : scored_node_t {entries[inward[position].node_id], inward[position].distance},
                        selected, connectivity);
                std::size_t kept = 0;
                for (std::size_t pick = 0; pick != selected.size(); ++pick)
                    for (std::size_t rank = 0; rank != live.size(); ++rank)
                        kept += live[rank].node_id == selected[pick].node_id;
                added += selected.size() - kept;
                dropped += live.size() - kept;
                if (kept == selected.size() && kept == live.size()) continue;
                index_->rewrite_list_(node_id, connectivity, context);
                changed += !count_changed_(node_id);
            }
            add_(tally_.reselected_nodes, count);
            add_(tally_.changed_nodes, changed);
            add_(tally_.added_links, added);
            add_(tally_.dropped_links, dropped);
            add_spent_(context.telemetry - before);
        }

        /** Gives an entry outside the root's component, whose @p selected picks hold no entry
         *  inside it, the nearest of its @p candidates inside it, or else its @p inward
         *  neighborhood-mate, while a slot is free: a way to the root, and an owner the closing
         *  heal links it back from. */
        template <typename candidates_type_, typename selected_type_>
        void keep_link_into_component_(candidates_type_ const& candidates, scored_node_t inward,
                                       selected_type_& selected, std::size_t connectivity) const noexcept {
            if (selected.size() >= min_of(connectivity, selected.capacity())) return;
            for (std::size_t pick = 0; pick != selected.size(); ++pick)
                if (health.in_component(selected[pick].node_id)) return;
            for (std::size_t rank = 0; rank != candidates.size(); ++rank)
                if (health.in_component(candidates[rank].node_id)) return selected.push_back_reserved(candidates[rank]);
            if (inward.node_id != missing_k) selected.push_back_reserved(inward);
        }

        /** Keeps in @p mates each entry's @c connectivity_ nearest neighborhood-mates among
         *  @p entries by position, a max-heap each, and in @p inward the nearest inside the root's
         *  component of each entry outside it: a quad of entries as borrowed queries per batched
         *  round against every entry, each round within the farthest mate its queries keep, or
         *  unbounded for a quad holding an entry outside the component. */
        template <typename metric_type_, typename context_type_>
        void score_mates_(std::span<node_id_t const> entries, std::span<scored_node_t> mates,
                          std::span<scored_node_t> inward, metric_type_& metric, context_type_& context) noexcept {
            std::size_t const count = entries.size(), connectivity = connectivity_;
            for (scored_node_t& mate : mates) mate = farthest_;
            for (scored_node_t& mate : inward) mate = farthest_;
            auto const value_at = [&](std::size_t at) noexcept -> value_t const& {
                return index_->published_value_of(entries[at]);
            };
            for (std::size_t first_row = 0; first_row < count; first_row += quad_lanes_k) {
                std::size_t const lanes = min_of<std::size_t>(quad_lanes_k, count - first_row);
                quad<value_t const*> borrowed;
                for (std::size_t lane = 0; lane != lanes; ++lane) borrowed[lane] = &value_at(first_row + lane);
                std::span<value_t const* const> const queries {borrowed.data(), lanes};
                open_queries(metric, queries);
                for (std::size_t first = 0; first != count;) {
                    // A heap's farthest only nears, so a distance past it now stays out of it.
                    quad<distance_t> thresholds;
                    for (std::size_t lane = 0; lane != lanes; ++lane)
                        thresholds[lane] = health.in_component(entries[first_row + lane])
                                               ? mates[(first_row + lane) * connectivity].distance
                                               : unbounded_radius<distance_t>();
                    context.staged_distances.resize_reserved(lanes * (count - first));
                    std::span<distance_t> const distances {context.staged_distances.data(), lanes * (count - first)};
                    std::size_t const end = score_queries(metric, queries, value_at, first, count,
                                                          std::span<distance_t const> {thresholds.data(), lanes},
                                                          distances);
                    if (end == first) {
                        ++context.telemetry.saturated_walks;
                        break;
                    }
                    std::size_t const scored = end - first;
                    for (std::size_t lane = 0; lane != lanes; ++lane) {
                        std::size_t const row = first_row + lane;
                        std::span<scored_node_t> const heap = mates.subspan(row * connectivity, connectivity);
                        bool const outside = !health.in_component(entries[row]);
                        for (std::size_t column = 0; column != scored; ++column) {
                            scored_node_t const mate {static_cast<node_id_t>(first + column),
                                                      distances[lane * scored + column]};
                            if (mate.node_id == row) continue;
                            if (further_(heap[0], mate)) replace_farthest(heap, mate, further_);
                            if (outside && health.in_component(entries[first + column]) && further_(inward[row], mate))
                                inward[row] = mate;
                        }
                    }
                    context.telemetry.computed_distances += lanes * scored;
                    first = end;
                }
            }
        }

        /** Links every unreached or stranded island whose first rank falls in @p task through the
         *  root's component, with no distance: an unreached one's entry from the first of its own
         *  links inside the component that takes it, a stranded one's entry to the entry of the
         *  component linking to it. A component with neither waits for the restitch. */
        void heal_(task_t const& task) noexcept {
            std::size_t healed = 0;
            for (std::size_t rank = task.nodes.first; rank < task.nodes.bound; ++rank) {
                if (!health.island_first(rank)) continue;
                u8_t const roles = health.island_roles(rank);
                node_id_t const entry = health.island_entry(rank);
                if (roles & health_plan_t::unreached_island_k) {
                    node_id_t owner = missing_k;
                    for (std::size_t slot = 0; owner == missing_k; ++slot) {
                        node_id_t const link = link_at_(entry, slot);
                        if (link == missing_k) break;
                        if (link < health.visible && health.in_component(link) &&
                            add_or_swap_link_(link, entry) == taken_t::written_k)
                            owner = link;
                    }
                    if (owner != missing_k) health.record_heal_link(owner, entry), ++healed;
                }
                node_id_t const parent = health.island_entry_parent(rank);
                if ((roles & health_plan_t::stranded_island_k) && parent != missing_k &&
                    add_or_swap_link_(entry, parent) == taken_t::written_k)
                    health.record_heal_link(entry, parent), ++healed;
            }
            add_(tally_.islands_reconnected, healed);
        }

        /** The link in slot @p slot of @p node_id 's list, read under its lock, or none past its
         *  end. */
        node_id_t link_at_(node_id_t node_id, std::size_t slot) noexcept {
            node_t& node = index_->nodes_[node_id];
            node.lock();
            node_id_t const link = slot < node.neighbors.size() ? node.neighbors[slot] : missing_k;
            node.unlock();
            return link;
        }

        /**
         *  @brief Links @p member from @p owner 's list under its lock: a free slot first, the one
         *      past the budget planning reserved among them; else a slot of the root's component.
         *
         *  That slot's resident is the one linked to by the most entries, taken when it stays
         *  linked to by more than @p member and the reach rule releases it. Adds go first, since an
         *  add dead ends no search, while a swap may for a target outside the round's sample. The
         *  in-link counts follow every write.
         */
        taken_t add_or_swap_link_(node_id_t owner, node_id_t member) noexcept {
            if (owner == member || owner >= health.visible || member >= health.visible) return taken_t::refused_k;
            node_t& node = index_->nodes_[owner];
            node.lock();
            neighbors_buffer_t& neighbors = node.neighbors;
            bool present = false;
            for (node_id_t const resident : neighbors) present = present || resident == member;
            taken_t taken = present ? taken_t::present_k : taken_t::refused_k;
            if (!present && neighbors.size() < neighbors.capacity()) {
                neighbors.push_back_reserved(member);
                taken = taken_t::written_k;
            }
            else if (!present && health.in_component(owner)) {
                std::size_t richest = neighbors.size();
                unsigned richest_links = health.in_links(member);
                for (std::size_t slot = 0; slot != neighbors.size(); ++slot) {
                    node_id_t const resident = neighbors[slot];
                    if (resident >= health.visible || health.in_links(resident) <= richest_links + 1u) continue;
                    richest = slot, richest_links = health.in_links(resident);
                }
                if (richest != neighbors.size() && health.take_link_under_lock_if_spare(owner, neighbors[richest])) {
                    static_cast<void>(health.take_in_link_if_spare(neighbors[richest]));
                    neighbors[richest] = member;
                    taken = taken_t::written_k;
                }
            }
            node.unlock();
            if (taken != taken_t::written_k) return taken;
            health.count_in_link(member);
            add_(tally_.changed_nodes, !count_changed_(owner));
            return taken;
        }

        /** Folds one task's spend into the count. */
        void add_spent_(traversal_stats_t const& spent) noexcept {
            add_(tally_.traversal_stats.computed_distances, spent.computed_distances);
            add_(tally_.traversal_stats.visited_members, spent.visited_members);
            add_(tally_.traversal_stats.restarts, spent.restarts);
            add_(tally_.traversal_stats.saturated_walks, spent.saturated_walks);
            add_(tally_.traversal_stats.computed_distances_in_refines, spent.computed_distances_in_refines);
            add_(tally_.traversal_stats.computed_distances_in_reverse_refines,
                 spent.computed_distances_in_reverse_refines);
        }

        /** Whether @p left lies farther than @p right, the larger ID farther among equals. */
        static constexpr auto further_ = [](scored_node_t const& left, scored_node_t const& right) noexcept {
            return left.distance > right.distance || (left.distance == right.distance && left.node_id > right.node_id);
        };

        /** An empty place in a kept heap, farther than any entry. */
        static constexpr scored_node_t farthest_ {missing_k, unbounded_radius<distance_t>()};

        /** The refined index, whose shared side @c hold_ occupies and whose claims stay closed;
         *  null in an unbound plan. */
        ram_index* index_ = nullptr;

        /** The shared hold spanning the plan's whole lifetime. */
        shared_lock_t hold_;

        /** Per node, whether the round counted it changed already. */
        growing_buffer<u1x64_t, allocator_t> changed_;

        /** The caller's donor slots, each nulled once moved in. */
        std::span<ram_index*> donors_;

        /** One unique hold per donor, each released once its donor moved in. */
        growing_buffer<unique_lock_t, allocator_t> donor_holds_;

        /** Donors moved in so far. */
        std::size_t absorbed_donors_ = 0;

        /** The index a split empties, whose unique side @c split_donor_hold_ occupies until its
         *  nodes publish in the receivers; null otherwise. */
        ram_index* split_donor_ = nullptr;

        /** The unique hold on a split's donor. */
        unique_lock_t split_donor_hold_;

        /** The caller's receiver slots of a split, one part each; empty otherwise. */
        std::span<ram_index*> receivers_;

        /** One shared hold per receiver whose claims the split closed, in receiver order. */
        growing_buffer<shared_lock_t, allocator_t> receiver_holds_;

        /** Per donor node: its part times the part capacity, plus its ID in that receiver. */
        growing_buffer<node_id_t, allocator_t> part_slots_;

        /** Per part, a run of the part capacity: its donor nodes in receiver order. */
        growing_buffer<node_id_t, allocator_t> part_members_;

        /** One part's nodes, and how many of them its growth expanded. */
        struct part_extent_t {
            node_id_t nodes = 0;
            node_id_t expanded = 0;
        };

        /** Per part, its extent. */
        growing_buffer<part_extent_t, allocator_t> part_extents_;

        /** What picks the parts' roots. */
        u64_t seed_ = 0;

        /** Receivers whose rounds ended so far; the bound one is the next. */
        std::size_t refined_receivers_ = 0;

        /** The reports of the receivers' rounds ended so far, folded. */
        refine_report_t receivers_tally_ {};

        /** The planned neighbors-per-node budget every neighbor list was reserved for. */
        std::size_t connectivity_ = 0;

        /** The widest neighbor list any entry held at planning, which a restitch scores. */
        std::size_t widest_list_ = 0;

        /** The phase whose tasks run now. */
        phase_t phase_ = phase_t::done_k;

        /** The phase the current health check follows - what decides the phase after it
         *  converges. */
        phase_t swept_after_ = phase_t::opening_k;

        /** Counts summed through @c atomic_ref by the tasks, and set by @ref advance. */
        refine_report_t tally_;

        /** The first failure the plan met: a health check whose islands' adjacency could not be
         *  placed. */
        status_t status_ = status_t::success_k;

        /** Hubs one gather draws at most: as many as half a core's cache holds. */
        std::size_t hubs_per_draw_ = 0;

        /** The most entries one neighborhood holds, and so the most one restitch takes at once. */
        std::size_t neighborhood_capacity_ = 0;

        /** The round's hubs, ascending. */
        growing_buffer<node_id_t, allocator_t> hub_nodes_;

        /** Per entry, the neighborhood its nearest hub gave it, among the hubs of its last
         *  gather. */
        growing_buffer<u16_t, allocator_t> entry_neighborhoods_;

        /** The entries by neighborhood, each one's ascending, and where each neighborhood's run
         *  starts, one past the last closing them. */
        growing_buffer<node_id_t, allocator_t> neighborhood_entries_;
        growing_buffer<node_id_t, allocator_t> neighborhood_offsets_;

        /** The seed of the round's draws. */
        u64_t round_seed_ = 0;

        /** Entries the last closing health check left unreached, which the next must lower for
         *  another heal. */
        std::size_t islands_left_ = 0;
    };

    /**
     *  @brief Opens one refinement round over the published nodes, as @ref refine_plan_t
     *      describes, with a fresh audit seeded like @ref plan_health.
     *
     *  Every fallible step happens here: the audit's bitsets and the round's columns from
     *  @p context 's allocator, about ten bytes per entry beside the audit's, and every published
     *  neighbor list reserved to @p selector 's budget, so the tasks never allocate once
     *  @c refine_plan_t::reserve grew their contexts. @p codec sizes the stored values, which sets
     *  how many entries a neighborhood takes. Enters the shared side and closes the claims,
     *  refusing with @c contended_k while a batch is in flight.
     *
     *  @param[in] config The cache a core keeps, which sizes the hubs and the neighborhoods.
     *  @param[in] selector Carries the budget every neighbor list is reserved for; the tasks take a
     *      selector of the same budget.
     *  @param[in] codec Measures the stored values.
     *  @param[in,out] context The planning thread's scratch, whose allocator places the plan.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The selector's budget is zero.
     *  - @c contended_k A batch is in flight, or another refine holds the claims closed.
     *  - @c bad_alloc_k The audit or the round's columns cannot be placed.
     */
    template <typename selector_type_, typename codec_type_>
        requires value_codec<codec_type_, value_t>
    expected<refine_plan_t> plan_round(refine_config_t config, selector_type_ const& selector, codec_type_ const& codec,
                                       traversal_context_t& context) noexcept {
        return plan_round_({}, context, selector.connectivity, config, codec);
    }

    /**
     *  @brief Opens an absorb of @p donors, whose plan's first phase moves every node of each into
     *      this index, one donor after another, as a part of its own; one round then links them in.
     *
     *  Every fallible step of every copy happens here, after the checks of @ref plan_round: each
     *  donor's unique side taken, the records and the key multimap past the watermark covered, and
     *  every donor neighbor list reserved to the budget. A copy task swaps a donor node's value and
     *  neighbors in with every ID shifted, so nothing allocates, and the phase boundary files the
     *  donor's keys and publishes its nodes at once, empties the donor and nulls its slot; at most
     *  one donor's nodes are ever unpublished. Donors must share this index's allocator, and
     *  nothing may search them meanwhile.
     *
     *  @param[in,out] donors The indexes giving their nodes up, each nulled once moved; the slots
     *      must outlive the plan.
     *  @param[in] config The cache a core keeps, which sizes the round.
     *  @param[in] selector Picks the neighbors each node keeps when the round re-stitches.
     *  @param[in] codec Measures the stored values, the donors' among them.
     *  @param[in,out] context The planning thread's scratch, whose allocator places the plan.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k A null donor, this index itself, or a zero budget.
     *  - @c contended_k A batch in flight on this index or on a donor, or another refine.
     *  - @c overflow_k Nodes past the reserved ceiling.
     */
    template <typename selector_type_, typename codec_type_>
        requires value_codec<codec_type_, value_t>
    expected<refine_plan_t> plan_absorb(std::span<ram_index*> donors, refine_config_t config,
                                        selector_type_ const& selector, codec_type_ const& codec,
                                        traversal_context_t& context) noexcept {
        return plan_round_(donors, context, selector.connectivity, config, codec);
    }

    /**
     *  @brief Opens a split of this index into @p receivers - the inverse of @c plan_absorb:
     *      its plan's first phases move every node into one graph-local part per receiver, then
     *      every receiver runs one round in turn.
     *
     *  The one partition task grows the parts breadth first over the links, one root per part
     *  through @ref seeded_bijection under the config's seed, each part capped at the nodes over
     *  the receivers rounded up; nodes no growth reached seed the smallest part. Scatter tasks move
     *  each node's value, key and neighbors into its receiver, the links inside its part rebased
     *  and the links across dropped, and the phase boundary files every receiver's keys and
     *  publishes its nodes at once, then empties this index and releases it, so its claims reopen
     *  at zero. Every receiver then runs one round as @ref plan_round does, and the result sums
     *  over the receivers.
     *
     *  Every fallible step happens here: this index's unique side taken, each receiver's shared
     *  side entered and its claims closed, the records, key multimap, audit and round of a full
     *  part covered in every receiver, and every neighbor list reserved to the budget. Receivers
     *  must share this index's allocator, the slots must outlive the plan, and nothing may search
     *  this index meanwhile.
     *
     *  @param[in] receivers The empty indexes taking the parts, at least two, all distinct.
     *  @param[in] config The cache a core keeps, which sizes the rounds, and the seed picking the
     *      parts' roots, so equal seeds over equal graphs split alike.
     *  @param[in] selector Picks the neighbors each node keeps when the rounds re-stitch.
     *  @param[in] codec Measures the stored values.
     *  @param[in,out] context The planning thread's scratch, whose allocator places the plan.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k Fewer than two receivers, a null, repeated or non-empty one, this
     *    index itself, or a zero budget.
     *  - @c contended_k A batch in flight on this index or on a receiver, or another refine.
     *  - @c overflow_k Part slots past the node ID type, or a part past a receiver's ceiling.
     */
    template <typename selector_type_, typename codec_type_>
        requires value_codec<codec_type_, value_t>
    expected<refine_plan_t> plan_split(std::span<ram_index*> receivers, split_config_t config,
                                       selector_type_ const& selector, codec_type_ const& codec,
                                       traversal_context_t& context) noexcept {
        return plan_split_(receivers, context, selector.connectivity, config.refine, config.seed, codec);
    }

    /**
     *  @brief Runs one refinement round over every published node on the calling thread: one
     *      @ref plan_round, its tasks drained in order.
     *
     *  @param[in] config The cache a core keeps, which sizes the round.
     *  @param[in] selector The link-selection heuristic and its budget.
     *  @param[in] codec Measures the stored values.
     *  @param[in,out] metric This thread's scoring instance.
     *  @param[in,out] context The calling thread's scratch & telemetry, which also places the plan.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The selector's budget is zero.
     *  - @c contended_k A batch is in flight, or another refine holds the claims closed.
     *  - @c bad_alloc_k The audit, the round's columns or @p context 's scratch cannot be placed.
     */
    template <typename metric_type_, typename selector_type_, typename codec_type_>
        requires metric<metric_type_, value_t, distance_t> && selector<selector_type_, ram_index, metric_type_> &&
                 value_codec<codec_type_, value_t>
    expected<refine_report_t> refine(refine_config_t config, selector_type_ const& selector, codec_type_ const& codec,
                                     metric_type_& metric, traversal_context_t& context) noexcept {
        auto [plan, status] = plan_round(config, selector, codec, context);
        if (failed(status)) return {{}, status};
        if (status_t const sized = plan.reserve(context); failed(sized)) return {plan.finish().value, sized};
        do
            for (typename refine_plan_t::task_t const task : plan.tasks()) [[maybe_unused]]
                status_t const ran = plan.run(task, metric, context, selector);
        while (plan.advance());
        return plan.finish();
    }

#pragma endregion Refinement

#pragma region Search

    /** Optional tuning knobs for @c search(); zeros mean "derive from @ref connectivity" - the
     *  searcher does not know the budgets the graph was wired with, so the graph tracks its
     *  widest. */
    struct search_config_t {

        /** How many candidates a walk keeps; raise to trade latency for recall. */
        std::size_t expansion_factor = 0;
    };

    /** Telemetry of one @c search(); matches stay in the caller's context. */
    struct [[nodiscard]] search_result_t {

        /** Matches left in @c context.results, ascending by distance. */
        std::size_t count = 0;

        /** This walk's share of the context's running counters. */
        traversal_stats_t telemetry;
    };

    /**
     *  @brief Finds the nearest published neighbors of @p query; matches go into
     *      @c context.results ascending by distance - the caller reranks or trims.
     *
     *  The leading template parameter is the caller's synchronization contract: the default assumes
     *  live mutators - the walk reads only nodes below the watermark snapshot and briefly locks
     *  each expanded node to copy its neighbor list - while a marked @c immutable_k or
     *  @c exclusive_k instantiation compiles the locks out entirely, bracketed by the shared
     *  mutex's wait-free probes: a violated certification answers @c contended_k rather than a
     *  silently wrong result - detected, not memory-safe, since the plain reads may meet a neighbor
     *  list a lying mutator is regrowing. Scoring runs off-lock, one @c reset / @c push / @c drain
     *  per round, and the metric need not match the one that built the graph. Once the candidates
     *  are full, their furthest member's distance drains each round as the radius, so a metric may
     *  stop scoring a candidate that cannot come under it. The query is a stored-value reference by
     *  default; a @c transparent_metric accepts other query forms.
     *
     *  @param[in] query The similarity target, alive for the whole call.
     *  @param[in] config Optional tuning knobs; zeros mean "derive from connectivity".
     *  @param[in,out] metric This thread's scoring instance.
     *  @param[in,out] context The calling thread's scratch & telemetry over any allocator; holds
     *      the matches.
     *
     *  @par Statuses:
     *  - @c contended_k A marked walk meets a mutator or a batch yet to publish.
     *  - @c overflow_k A scratch capacity of @p context wraps when rounded.
     */
    template <synchronization_t synchronization_ = synchronization_t::concurrent_k, //
              typename metric_type_, typename context_allocator_type_, typename query_type_ = value_t>
        requires metric<metric_type_, value_t, distance_t> && query<query_type_, metric_type_, value_t>
    expected<search_result_t> search(query_type_ const& query, search_config_t config, metric_type_& metric,
                                     traversal_context_over<context_allocator_type_>& context) noexcept {
        return search<synchronization_>(query, dummy_fold_t {}, config, metric, context);
    }

    /**
     *  @brief Walks exactly like @c search while @p fold reads every round the walk scores - the
     *      entry point for verbs that reduce a walk to something other than its nearest matches.
     *
     *  @param[in] query The similarity target, alive for the whole call.
     *  @param[in,out] fold The reader of every scored round - see @c basic::fold. A @c sated_k ends
     *      the walk successfully, before that round reaches the candidates.
     *  @param[in] config Optional tuning knobs; zeros mean "derive from connectivity".
     *  @param[in,out] metric This thread's scoring instance.
     *  @param[in,out] context The calling thread's scratch & telemetry over any allocator; holds
     *      the matches.
     *
     *  @par Statuses:
     *  - @c contended_k A marked walk meets a mutator or a batch yet to publish.
     *  - @c overflow_k A scratch capacity of @p context wraps when rounded.
     */
    template <synchronization_t synchronization_ = synchronization_t::concurrent_k, //
              typename metric_type_, typename fold_type_, typename context_allocator_type_,
              typename query_type_ = value_t>
        requires metric<metric_type_, value_t, distance_t> && query<query_type_, metric_type_, value_t> &&
                 fold<std::remove_cvref_t<fold_type_>, node_id_t, distance_t>
    expected<search_result_t> search(query_type_ const& query, fold_type_&& fold, search_config_t config,
                                     metric_type_& metric,
                                     traversal_context_over<context_allocator_type_>& context) noexcept {
        return fold_walk_<synchronization_>(query, metric, fold, context.results, context, config);
    }

    /**
     *  @brief Walks like @c search, keeping only the scored nodes @p relation and @p predicate both
     *      admit, ascending in @c context.results. The walk routes through every node it scores and
     *      ends once the admitted ones fill the expansion or the graph runs out.
     *
     *  @param[in] query The similarity target, alive for the whole call.
     *  @param[in] config Optional tuning knobs; zeros mean "derive from connectivity".
     *  @param[in,out] relation The engine's own relation over node IDs, or @c dummy_predicate_t.
     *  @param[in,out] predicate The caller's predicate over node IDs, asked only about the
     *      contenders @p relation kept: a quad at a time, or once per round when it aggregates.
     *  @param[in,out] metric This thread's scoring instance.
     *  @param[in,out] context The calling thread's scratch & telemetry over any allocator; holds
     *      the matches.
     *
     *  @par Statuses:
     *  - @c contended_k A marked walk meets a mutator or a batch yet to publish.
     *  - @c overflow_k A scratch capacity of @p context wraps when rounded.
     */
    template <synchronization_t synchronization_ = synchronization_t::concurrent_k, //
              typename metric_type_, typename relation_type_, typename predicate_type_,
              typename context_allocator_type_, typename query_type_ = value_t>
        requires metric<metric_type_, value_t, distance_t> && query<query_type_, metric_type_, value_t>
    expected<search_result_t> search_if(query_type_ const& query, search_config_t config, relation_type_& relation,
                                        predicate_type_& predicate, metric_type_& metric,
                                        traversal_context_over<context_allocator_type_>& context) noexcept {
        context.admitted.clear();
        predicate_fold fold {relation, predicate, context.admitted,
                             resolved_config_({config.expansion_factor}, connectivity()).expansion_factor};
        expected<search_result_t> found = fold_walk_<synchronization_>(query, metric, fold, context.admitted, context,
                                                                       config);
        std::swap(context.results, context.admitted);
        found.value.count = context.results.size();
        return found;
    }

    /**
     *  @brief The exact sibling of @c search: scores @p query against every entry published when
     *      the scan starts and keeps the nearest @p wanted that @p relation and @p predicate both
     *      admit, ascending in @c context.results.
     *
     *  Once @p wanted are held, every round drains within the worst of them, so a metric may stop
     *  scoring an entry that cannot come under it. Only an entry that would enter the matches is
     *  asked about, which makes the scan the fallback for predicates too selective for a walk to
     *  cross.
     *
     *  @param[in] query The similarity target, alive for the whole call.
     *  @param[in] wanted How many matches to keep.
     *  @param[in,out] relation The engine's own relation over node IDs, or @c dummy_predicate_t.
     *  @param[in,out] predicate The caller's predicate over node IDs, or @c dummy_predicate_t.
     *  @param[in,out] metric This thread's scoring instance.
     *  @param[in,out] context The calling thread's scratch & telemetry; holds the matches.
     */
    template <typename metric_type_, typename relation_type_, typename predicate_type_,
              typename context_allocator_type_, typename query_type_ = value_t>
        requires metric<metric_type_, value_t, distance_t> && query<query_type_, metric_type_, value_t>
    expected<search_result_t> scan(query_type_ const& query, std::size_t wanted, relation_type_& relation,
                                   predicate_type_& predicate, metric_type_& metric,
                                   traversal_context_over<context_allocator_type_>& context) noexcept {
        context.results.clear();
        if (!wanted) return {};
        auto const radius = [&]() noexcept {
            return context.results.size() == wanted ? context.results.top().distance : unbounded_radius<distance_t>();
        };
        predicate_fold fold {relation, predicate, context.results, wanted};
        return scan_below_(query, metric, radius, fold, context, wanted, static_cast<node_id_t>(size()));
    }

    /**
     *  @brief Scores @p query against every entry published when the scan starts, handing every
     *      round to @p fold: the exact sibling of the folding @c search, for verbs that reduce all
     *      entries to something other than the nearest few, such as containment.
     *
     *  @param[in] query The query every entry is scored against.
     *  @param[in,out] fold The reader of every scored round - see @c basic::fold. A @c sated_k ends
     *      the scan successfully.
     *  @param[in,out] metric The metric the entries are scored with.
     *  @param[in,out] context The traversal context the scan's buffers come from.
     */
    template <typename metric_type_, typename fold_type_, typename context_allocator_type_,
              typename query_type_ = value_t>
        requires metric<metric_type_, value_t, distance_t> && query<query_type_, metric_type_, value_t> &&
                 fold<std::remove_cvref_t<fold_type_>, node_id_t, distance_t>
    expected<search_result_t> scan(query_type_ const& query, fold_type_&& fold, metric_type_& metric,
                                   traversal_context_over<context_allocator_type_>& context) noexcept {
        context.results.clear();
        auto const radius = []() noexcept { return unbounded_radius<distance_t>(); };
        return scan_below_(query, metric, radius, fold, context, 1, static_cast<node_id_t>(size()));
    }

#pragma endregion Search

#pragma region Keys and Introspection

    /** Number of published nodes - the visibility watermark; safe any time. */
    std::size_t size() const noexcept {
        return static_cast<std::size_t>(atomic_ref<node_id_t const>(last_visible_id_).load(std::memory_order_acquire));
    }

    /** Nodes the covered node chunks hold, as inserts, loads and @c reserve grow them; safe any
     *  time. */
    std::size_t capacity() const noexcept { return nodes_.capacity(); }

    /** The widest neighbor list any wiring may have written - the largest budget an insert or a
     *  refine wired with, a manually linked one, a loaded or a moved-in one - and never under one,
     *  so an edgeless graph still draws a landmark. A search derives its unset knobs from it; safe
     *  any time. */
    std::size_t connectivity() const noexcept {
        return max_of<std::size_t>(atomic_ref<std::size_t const>(widest_list_).load(std::memory_order_relaxed), 1);
    }

    /** The walk breadth @p config asks for, or the one a search derives from @ref connectivity
     *  when it asks for none. */
    std::size_t non_zero_expansion_factor(search_config_t const& config) const noexcept {
        return resolved_config_({config.expansion_factor}, connectivity()).expansion_factor;
    }

    /** Bytes the node records' chunks and the key multimap span. The neighbor lists and whatever
     *  the values allocate on their own are left out, since counting them would walk every
     *  record. */
    std::size_t memory_usage() const noexcept { return nodes_.memory_usage() + key_map_.memory_usage(); }

    /**
     *  The key stored under @p node_id - the authoritative direction, immutable after publication.
     *  One acquire load of the watermark gates the ID, so any input is safe any time.
     *
     *  @par Statuses:
     *  - @c malformed_k @p node_id is unpublished.
     */
    expected<key_t> key_of(node_id_t node_id) const noexcept {
        if (static_cast<std::size_t>(node_id) >= size()) return {key_t {}, status_t::malformed_k};
        return {nodes_[node_id].key, status_t::success_k};
    }

    /**
     *  The payload stored under @p node_id - never relocated, valid until teardown,
     *  immutable after publication. The same watermark gate as @c key_of, so any input
     *  is safe any time.
     *
     *  @par Statuses:
     *  - @c malformed_k @p node_id is unpublished.
     */
    expected<value_t const&> value_of(node_id_t node_id) const noexcept {
        if (static_cast<std::size_t>(node_id) >= size()) return {nullptr, status_t::malformed_k};
        return {&nodes_[node_id].value, status_t::success_k};
    }

    /** The payload under a @b published or wired @p node_id, unchecked: for traversal-contract
     *  callers - selectors and the walks themselves - whose IDs come from the walk's admitted
     *  candidates, where the watermark gate of @c value_of would re-prove a known fact or refuse a
     *  node another insert wired ahead of its turn. */
    value_t const& published_value_of(node_id_t node_id) const noexcept { return nodes_[node_id].value; }

    /**
     *  The out-degree of @p node_id: the same watermark gate as @c key_of, then the
     *  node lock - every post-publication edit of its list holds it, so the read is one
     *  consistent instant even while back-linking appends.
     *
     *  @par Statuses:
     *  - @c malformed_k @p node_id is unpublished.
     */
    expected<std::size_t> degree_of(node_id_t node_id) noexcept {
        if (static_cast<std::size_t>(node_id) >= size()) return {0, status_t::malformed_k};
        node_t& node = nodes_[node_id];
        node.lock();
        std::size_t const degree = node.neighbors.size();
        node.unlock();
        return {degree, status_t::success_k};
    }

    /**
     *  One published node under @p key. Duplicates surface in an unspecified order - use
     *  @c find_each to see them all. Wait-free, concurrent with everything.
     *
     *  @par Statuses:
     *  - @c malformed_k No published node is stored under @p key.
     */
    expected<node_id_t> find(key_t const& key) const noexcept {
        expected<node_id_t> found {node_id_t {}, status_t::malformed_k};
        find_each(key, [&](node_id_t node_id) noexcept { found = {node_id, status_t::success_k}; });
        return found;
    }

    /** Invokes @p callback with the ID of every published node under @p key, as of one acquire
     *  of the watermark - wait-free, concurrent with everything. */
    template <typename callback_type_>
    void find_each(key_t const& key, callback_type_&& callback) const noexcept {
        key_map_.find_each(key, static_cast<node_id_t>(size()), callback);
    }

    /** Every published node below @p first_invisible_id stored under each of @p hints, in hint
     *  order, staged into @c context.hinted_ids for a walk's seeds as far as its reservation
     *  holds, marking in @c context.hinted_held, one byte per hint, each key holding one. */
    template <typename context_type_>
    std::span<node_id_t const> stage_hints_(std::span<key_t const> hints, node_id_t first_invisible_id,
                                            context_type_& context) const noexcept {
        context.hinted_ids.clear();
        for (std::size_t hint = 0; hint != hints.size(); ++hint)
            key_map_.find_each(hints[hint], first_invisible_id, [&](node_id_t node_id) noexcept {
                context.hinted_held[hint] = 1;
                if (context.hinted_ids.size() != context.hinted_ids.capacity())
                    context.hinted_ids.push_back_reserved(node_id);
            });
        return to_span(context.hinted_ids);
    }

    /** Whether at least one published node exists under @p key, stopping at the first. */
    bool contains(key_t const& key) const noexcept { return key_map_.contains(key, static_cast<node_id_t>(size())); }

    /** Number of published nodes under @p key, exact as of one acquire of the watermark. */
    std::size_t count(key_t const& key) const noexcept { return key_map_.count(key, static_cast<node_id_t>(size())); }

    /** Post-construction graph report - the quality signal a recall number hides.
     *  Meaningful only on @c success_k. */
    struct [[nodiscard]] graph_health_t {

        /** @c success_k, or why the report could not be taken. */
        status_t status = status_t::success_k;

        /** Forward-reachability components, flooded in ID order - equal to weak
         *  components whenever back-linking has kept the edges reciprocal. */
        std::size_t connected_components = 0;

        /** Nodes in the largest such component. */
        std::size_t largest_component = 0;

        /** The audit @ref plan_health opens, swept to convergence - the searcher's view. */
        reachability_t reachability;

        /** The smallest published out-degree. */
        std::size_t degree_min = 0;

        /** The largest published out-degree. */
        std::size_t degree_max = 0;

        /** Directed edges whose reverse edge also exists. */
        std::size_t reciprocal_edges = 0;

        /** Average published out-degree. */
        f64_t mean_degree = 0;

        explicit operator bool() const noexcept { return succeeded(status); }
    };

    /**
     *  @brief Walks the whole graph, reporting components, degree spread, edge reciprocity, and
     *      reachability as @ref plan_health audits it.
     *
     *  Takes the shared mutex's unique side for the whole walk - live mutators answer
     *  @c contended_k at entry, and none can start until the report returns.
     *
     *  @param[in,out] context The calling thread's scratch over any allocator; its visited set and
     *      staged IDs grow to the full graph size for the duration of the walk. */
    template <typename context_allocator_type_>
    graph_health_t health(traversal_context_over<context_allocator_type_>& context) noexcept {
        graph_health_t report;
        unique_lock_t unique_lock {shared_mutex_, std::try_to_lock};
        if (!unique_lock) {
            report.status = status_t::contended_k;
            return report;
        }
        auto const [visible, status] = watermark_();
        if (failed(status)) {
            report.status = status;
            return report;
        }
        std::size_t const node_count = static_cast<std::size_t>(visible);
        if (!node_count) return report;
        if (status_t const seen = context.visited.reserve(node_count); failed(seen)) {
            report.status = seen;
            return report;
        }
        if (status_t const staged = context.staged_ids.reserve(node_count); failed(staged)) {
            report.status = staged;
            return report;
        }

        // Degrees & reciprocity in one edge sweep - plain reads, the gate excludes every
        // mutator; the node-ID tripwires stay as a corruption net, never an overflow.
        report.degree_min = (std::numeric_limits<std::size_t>::max)();
        std::size_t total_neighbors = 0;
        for (std::size_t node = 0; node != node_count; ++node) {
            neighbors_buffer_t const& neighbors = nodes_[node].neighbors;
            report.degree_min = min_of(report.degree_min, neighbors.size());
            report.degree_max = max_of(report.degree_max, neighbors.size());
            total_neighbors += neighbors.size();
            for (std::size_t link = 0; link != neighbors.size(); ++link) {
                if (static_cast<std::size_t>(neighbors[link]) >= node_count) {
                    report.status = status_t::internal_k;
                    continue;
                }
                neighbors_buffer_t const& reverse = nodes_[neighbors[link]].neighbors;
                for (std::size_t back = 0; back != reverse.size(); ++back)
                    if (reverse[back] == static_cast<node_id_t>(node)) {
                        ++report.reciprocal_edges;
                        break;
                    }
            }
        }
        report.mean_degree = static_cast<f64_t>(total_neighbors) / static_cast<f64_t>(node_count);

        // Forward-reachability floodfill, reused for components and for seed coverage.
        auto flood = [&](node_id_t seed) noexcept {
            std::size_t flooded = 0;
            context.staged_ids.clear();
            context.staged_ids.push_back_reserved(seed);
            [[maybe_unused]] membership_t const seeded = context.visited.insert(seed);
            while (!context.staged_ids.empty()) {
                node_id_t const node = context.staged_ids.pop_back();
                ++flooded;
                neighbors_buffer_t const& neighbors = nodes_[node].neighbors;
                for (std::size_t link = 0; link != neighbors.size(); ++link) {
                    // The same corruption net as the edge sweep above.
                    if (static_cast<std::size_t>(neighbors[link]) >= node_count) {
                        report.status = status_t::internal_k;
                        continue;
                    }
                    if (context.visited.insert(neighbors[link]) == membership_t::absent_k)
                        context.staged_ids.push_back_reserved(neighbors[link]);
                }
            }
            return flooded;
        };

        context.visited.clear();
        for (std::size_t node = 0; node != node_count; ++node) {
            if (context.visited.contains(static_cast<node_id_t>(node))) continue;
            ++report.connected_components;
            report.largest_component = max_of(report.largest_component, flood(static_cast<node_id_t>(node)));
        }

        auto [health, health_status] = plan_audit_(visible, allocator_);
        if (failed(health_status)) {
            report.status = health_status;
            return report;
        }
        do
            for (typename health_plan_t::task_t const task : health.tasks()) [[maybe_unused]]
                std::size_t const expanded = health.sweep(task);
        while (health.advance_depth());
        auto const [reachability, reachability_status] = health.finish();
        report.reachability = reachability;
        if (failed(reachability_status)) report.status = reachability_status;
        return report;
    }

#pragma endregion Keys and Introspection

#pragma region Serialization

    /** One task-parallel save in flight: the caller's five columns, the covered-node count,
     *  and the shared mutex's unique hold - released when the plan dies or @c finish
     *  pronounces the columns complete. */
    struct [[nodiscard]] save_plan_t {

        /** The caller-placed columns every task writes disjoint slices of. */
        columns_t columns {};

        /** Nodes the tasks covered so far, mutated through @c atomic_ref. */
        std::size_t covered = 0;

        save_plan_t() noexcept = default;
        save_plan_t(ram_index const* index, columns_t columns, unique_lock_t&& hold) noexcept
            : columns(columns), index_(index), hold_(std::move(hold)) {}
        save_plan_t(save_plan_t&& other) noexcept
            : columns(other.columns), covered(other.covered), index_(std::exchange(other.index_, nullptr)),
              hold_(std::move(other.hold_)) {}
        save_plan_t& operator=(save_plan_t&&) = delete;

        /** A save publishes nothing, so a dying plan only drops its hold - the lock's own job. */
        ~save_plan_t() noexcept = default;

        /**
         *  @brief Copies the node range's keys, neighbor IDs & encoded values into the columns -
         *      disjoint slices, so any number of caller threads run their tasks concurrently; every
         *      slice starts where the plan's prefix sums say.
         *
         *  A refusal of @p codec passes through.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is empty or already spent.
         *  - @c config_mismatch_k @p node_range falls outside the plan's nodes.
         */
        template <typename value_codec_type_>
            requires value_codec<value_codec_type_, value_t>
        status_t save(iota_view<node_id_t> node_range, value_codec_type_ const& codec) noexcept {
            if (!index_) return status_t::uninitialized_k;
            std::size_t const node_count = columns.keys.size();
            if (node_range.bound < node_range.first || static_cast<std::size_t>(node_range.bound) > node_count)
                return status_t::config_mismatch_k;

            for (node_id_t const node : node_range) columns.keys[node] = index_->nodes_[node].key;

            std::size_t ids_cursor = columns.neighbor_offsets[node_range.first];
            for (node_id_t const node : node_range) {
                neighbors_buffer_t const& neighbors = index_->nodes_[node].neighbors;
                for (std::size_t link = 0; link != neighbors.size(); ++link)
                    columns.neighbor_ids[ids_cursor++] = neighbors[link];
            }

            // Each value fills exactly the slice its sum reserved; a codec needing more refuses itself.
            for (node_id_t const node : node_range) {
                std::size_t const first = columns.value_offsets[node];
                std::size_t const bound = columns.value_offsets[node + 1];
                status_t const encoded = codec.save(index_->nodes_[node].value,
                                                    columns.values.subspan(first, bound - first));
                if (failed(encoded)) return encoded;
            }
            atomic_ref<std::size_t>(covered).store_add(node_range.size(), std::memory_order_release);
            return status_t::success_k;
        }

        /**
         *  @brief Pronounces the columns complete & releases the hold: @c config_mismatch_k unless
         *      the tasks' sizes sum to the node count - overlapping tasks that still sum right are
         *      the caller's own lie.
         *
         *  A success spends the plan, so a second call answers @c uninitialized_k instead of
         *  unlocking a hold the plan has given up.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is empty or already spent.
         *  - @c config_mismatch_k The tasks covered other than every node.
         */
        status_t finish() noexcept {
            if (!index_) return status_t::uninitialized_k;
            if (atomic_ref<std::size_t>(covered).load(std::memory_order_acquire) != columns.keys.size())
                return status_t::config_mismatch_k;
            hold_.unlock();
            index_ = nullptr;
            return status_t::success_k;
        }

      private:

        /** The index being serialized, whose unique side @c hold_ occupies; null in the
         *  empty plan a refused @ref plan_save returns, and once @c finish succeeded. */
        ram_index const* index_ = nullptr;

        /** The unique hold spanning the plan's whole lifetime. */
        unique_lock_t hold_;
    };

    /**
     *  @brief Opens a task-parallel save.
     *
     *  Takes the shared mutex's unique side - held inside the returned plan until its @c finish or
     *  its death - sizes the columns against the watermark, and writes both prefix-sum columns up
     *  front, so the plan's own @c save calls copy the per-node bulk into slices already named.
     *  Live mutators answer @c contended_k; a mutator call from the plan-holding thread deadlocks
     *  on its own hold, by construction.
     *
     *  A column shorter than its count answers @c overflow_k and a longer one
     *  @c config_mismatch_k, so the key column's length always names the node count.
     *
     *  @par Statuses:
     *  - @c contended_k A mutator or an open plan holds the shared mutex, or a dispensed batch has
     *    yet to publish.
     *  - @c overflow_k The key or a prefix-sum column is shorter than its count.
     *  - @c config_mismatch_k The key or a prefix-sum column is longer than its count.
     *  - @c overflow_k The neighbor IDs or the value bytes are shorter than the sums need.
     */
    template <typename value_codec_type_>
        requires value_codec<value_codec_type_, value_t>
    expected<save_plan_t> plan_save(columns_t columns, value_codec_type_ const& codec) noexcept {
        unique_lock_t hold {shared_mutex_, std::try_to_lock};
        if (!hold) return {{}, status_t::contended_k};
        auto const [visible, status] = watermark_();
        if (failed(status)) return {{}, status};
        std::size_t const node_count = static_cast<std::size_t>(visible);
        if (status_t const sized = sized_exactly_(columns.keys.size(), node_count); failed(sized)) return {{}, sized};
        if (status_t const sized = sized_exactly_(columns.neighbor_offsets.size(), node_count + 1); failed(sized))
            return {{}, sized};
        if (status_t const sized = sized_exactly_(columns.value_offsets.size(), node_count + 1); failed(sized))
            return {{}, sized};

        // One walk writes both prefix sums, sizing every value exactly once.
        u64_t running_neighbors = 0;
        u64_t running_values = 0;
        columns.neighbor_offsets[0] = 0;
        columns.value_offsets[0] = 0;
        for (std::size_t node = 0; node != node_count; ++node) {
            node_t const& record = nodes_[node];
            running_neighbors += record.neighbors.size();
            running_values += codec.storage_requirements(record.value);
            columns.neighbor_offsets[node + 1] = running_neighbors;
            columns.value_offsets[node + 1] = running_values;
        }
        if (status_t const sized = sized_at_least_(columns.neighbor_ids.size(), running_neighbors); failed(sized))
            return {{}, sized};
        if (status_t const sized = sized_at_least_(columns.values.size(), running_values); failed(sized))
            return {{}, sized};
        return {save_plan_t {this, columns, std::move(hold)}, status_t::success_k};
    }

    /**
     *  @brief Node & element counts a save needs right now. Takes the shared mutex's unique
     *      side for the count; live mutators and open plans answer @c contended_k at entry.
     *
     *  @param[in] codec Sizes each stored value's encoded form.
     *
     *  @par Statuses:
     *  - @c contended_k A mutator or an open plan holds the shared mutex, or a dispensed batch has
     *    yet to publish.
     */
    template <typename value_codec_type_>
        requires value_codec<value_codec_type_, value_t>
    expected<columns_extents_t> columns_extents(value_codec_type_ const& codec) noexcept {
        unique_lock_t unique_lock {shared_mutex_, std::try_to_lock};
        if (!unique_lock) return {{}, status_t::contended_k};
        auto const [visible, status] = watermark_();
        if (failed(status)) return {{}, status};
        columns_extents_t extents {static_cast<std::size_t>(visible), 0, 0};
        for (std::size_t node = 0; node != extents.nodes; ++node) {
            node_t const& record = nodes_[node];
            extents.neighbors += record.neighbors.size();
            extents.value_bytes += codec.storage_requirements(record.value);
        }
        return {extents, status_t::success_k};
    }

    /**
     *  @brief Serializes the snapshot into @p columns as one plan with a single task.
     *
     *  Takes the shared mutex's unique side for the whole copy - live mutators answer
     *  @c contended_k at entry, none can start until the columns are complete, and searches keep
     *  running throughout.
     *
     *  @param[out] columns Caller-placed destinations sized from @ref columns_extents.
     *  @param[in] codec Encodes each stored value into its slice; its refusal passes through.
     *
     *  @par Statuses:
     *  - @c contended_k A mutator or an open plan holds the shared mutex, or a dispensed batch has
     *    yet to publish.
     *  - @c overflow_k A column is shorter than its count or than the sums need.
     *  - @c config_mismatch_k The key or a prefix-sum column is longer than its count.
     */
    template <typename value_codec_type_>
        requires value_codec<value_codec_type_, value_t>
    status_t save(columns_t columns, value_codec_type_ const& codec) noexcept {
        auto [plan, status] = plan_save(columns, codec);
        if (failed(status)) return status;
        node_id_t const node_count = static_cast<node_id_t>(plan.columns.keys.size());
        if (status_t const copied = plan.save({0, node_count}, codec); failed(copied)) return copied;
        return plan.finish();
    }

    /** One task-parallel load in flight - the mirror of @c save_plan_t: the vetted columns, the
     *  covered-node count, and the shared mutex's unique hold, released when the plan dies or
     *  @c finish files the keys and publishes the loaded graph. */
    struct [[nodiscard]] load_plan_t {

        /** The vetted columns being decoded. */
        columns_view_t columns {};

        /** Nodes the tasks covered so far, mutated through @c atomic_ref. */
        std::size_t covered = 0;

        load_plan_t() noexcept = default;
        load_plan_t(ram_index* index, columns_view_t columns, unique_lock_t&& hold) noexcept
            : columns(columns), index_(index), hold_(std::move(hold)) {}
        load_plan_t(load_plan_t&& other) noexcept
            : columns(other.columns), covered(other.covered), index_(std::exchange(other.index_, nullptr)),
              hold_(std::move(other.hold_)) {}
        load_plan_t& operator=(load_plan_t&&) = delete;

        /**
         *  @brief Decodes the node range's keys, neighbor lists & values into its pre-constructed
         *      records, which no other task touches, so any number of caller threads run their
         *      tasks; the keys enter the multimap in @c finish.
         *
         *  A refusal of @p codec passes through.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is empty or already published.
         *  - @c config_mismatch_k @p node_range falls outside the plan's nodes.
         */
        template <typename value_codec_type_>
            requires value_codec<value_codec_type_, value_t>
        status_t load(iota_view<node_id_t> node_range, value_codec_type_ const& codec) noexcept {
            if (!index_) return status_t::uninitialized_k;
            std::size_t const node_count = columns.keys.size();
            if (node_range.bound < node_range.first || static_cast<std::size_t>(node_range.bound) > node_count)
                return status_t::config_mismatch_k;

            // The plan vetted both prefix-sum columns and every ID, and reserved every neighbor
            // list, so the slices are trusted and the lists fit.
            std::size_t widest_list = 0;
            for (node_id_t const node : node_range) {
                node_t& record = index_->nodes_[node];
                record.key = columns.keys[node];

                std::size_t const ids_first = columns.neighbor_offsets[node];
                std::size_t const ids_bound = columns.neighbor_offsets[node + 1];
                neighbors_buffer_t& neighbors = record.neighbors;
                neighbors.clear();
                for (std::size_t link = ids_first; link != ids_bound; ++link)
                    neighbors.push_back_reserved(columns.neighbor_ids[link]);
                widest_list = max_of(widest_list, neighbors.size());

                std::size_t const bytes_first = columns.value_offsets[node];
                std::size_t const bytes_bound = columns.value_offsets[node + 1];
                if (status_t const decoded = codec.load(columns.values.subspan(bytes_first, bytes_bound - bytes_first),
                                                        record.value);
                    failed(decoded))
                    return decoded;
            }
            index_->widen_(widest_list);
            atomic_ref<std::size_t>(covered).store_add(node_range.size(), std::memory_order_release);
            return status_t::success_k;
        }

        /**
         *  @brief Publishes the loaded graph in the tier-wide order - keys, claims, identity,
         *      watermark last - and releases the hold: @c config_mismatch_k unless the tasks' sizes
         *      sum to the node count.
         *
         *  The hold makes it the map's one writer.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is empty or already published.
         *  - @c config_mismatch_k The tasks covered other than every node.
         */
        status_t finish() noexcept {
            if (!index_) return status_t::uninitialized_k;
            std::size_t const node_count = columns.keys.size();
            if (atomic_ref<std::size_t>(covered).load(std::memory_order_acquire) != node_count)
                return status_t::config_mismatch_k;
            index_->publish_keys_({0, static_cast<node_id_t>(node_count)});
            atomic_ref<std::size_t>(index_->claimed_count_).store(node_count, std::memory_order_relaxed);
            atomic_ref<node_id_t>(index_->next_dispensed_id_)
                .store(static_cast<node_id_t>(node_count), std::memory_order_relaxed);
            atomic_ref<node_id_t>(index_->last_visible_id_)
                .store(static_cast<node_id_t>(node_count), std::memory_order_release);
            hold_.unlock();
            index_ = nullptr;
            return status_t::success_k;
        }

      private:

        /** The index being repopulated, whose unique side @c hold_ occupies; null in the
         *  empty plan a refused @ref plan_load returns, and once @c finish published. */
        ram_index* index_ = nullptr;

        /** The unique hold spanning the plan's whole lifetime. */
        unique_lock_t hold_;
    };

    /**
     *  @brief Opens a task-parallel load into this empty index: takes the shared mutex's unique
     *      side - held inside the returned plan until its @c finish or its death - and vets both
     *      prefix-sum columns and every neighbor ID in full.
     *
     *  The vetted nodes' records and the key multimap are then covered and their neighbor lists
     *  reserved, so the plan's own @c load calls decode the per-node bulk and write no link or key
     *  that needs memory; nothing reaches the multimap or publishes until its @c finish.
     *
     *  @param[in] columns The five columns a prior @c save filled; the key column's length
     *      is the node count.
     *
     *  @par Statuses:
     *  - @c contended_k A mutator or an open plan holds the shared mutex.
     *  - @c config_mismatch_k The index already publishes nodes.
     *  - @c malformed_k A prefix-sum column is not one longer than the keys, does not rise from
     *    zero to its element column's length, or a neighbor ID names no node.
     *  - @c overflow_k The nodes pass @c max_node_count_k, or a node's capacity wraps.
     */
    expected<load_plan_t> plan_load(columns_view_t columns) noexcept {
        // The gate excludes mutators & flips marked probes to refusal; the exclusivity
        // contract against readers stays the caller's - key lookups race the map rebuild.
        unique_lock_t hold {shared_mutex_, std::try_to_lock};
        if (!hold) return {{}, status_t::contended_k};
        if (atomic_ref<node_id_t>(last_visible_id_).load(std::memory_order_acquire) != 0)
            return {{}, status_t::config_mismatch_k};
        if (status_t const vetted = vetted_columns_(columns); failed(vetted)) return {{}, vetted};
        // Vetted first, so hostile columns never allocate.
        std::size_t const node_count = columns.keys.size();
        if (status_t const covered = nodes_.cover(node_count); failed(covered)) return {{}, covered};
        for (std::size_t node = 0; node != node_count; ++node)
            if (status_t const grown = nodes_[node].neighbors.reserve(columns.neighbor_offsets[node + 1] -
                                                                      columns.neighbor_offsets[node]);
                failed(grown))
                return {{}, grown};
        if (status_t const keyed = key_map_.reserve(node_count); failed(keyed)) return {{}, keyed};
        return {load_plan_t {this, columns, std::move(hold)}, status_t::success_k};
    }

    /**
     *  @brief Loads a snapshot into this empty index as one plan with a single task -
     *      exclusive: decodes each value into its pre-constructed record, rebuilds the key
     *      multimap, and publishes the watermark last.
     *
     *  No key reaches the multimap before the watermark, so a failed attempt may be retried with
     *  corrected columns.
     *
     *  @param[in] columns The five columns a prior @c save filled.
     *  @param[in] codec Decodes each value slice into its record; its refusal passes through.
     *
     *  @par Statuses:
     *  - @c contended_k A mutator or an open plan holds the shared mutex.
     *  - @c config_mismatch_k The index already publishes nodes.
     *  - @c malformed_k A prefix-sum column is not one longer than the keys, does not rise from
     *    zero to its element column's length, or a neighbor ID names no node.
     *  - @c overflow_k The nodes pass @c max_node_count_k, or a node's capacity wraps.
     */
    template <typename value_codec_type_>
        requires value_codec<value_codec_type_, value_t>
    status_t load(columns_view_t columns, value_codec_type_ const& codec) noexcept {
        auto [plan, status] = plan_load(columns);
        if (failed(status)) return status;
        node_id_t const node_count = static_cast<node_id_t>(plan.columns.keys.size());
        if (status_t const decoded = plan.load({0, node_count}, codec); failed(decoded)) return decoded;
        return plan.finish();
    }

#pragma endregion Serialization

  private:
#pragma region Internals and State

    /** @p config with its zero breadth derived from @p connectivity: candidates twice the
     *  budget. */
    static insert_config_t resolved_config_(insert_config_t config, std::size_t connectivity) noexcept {
        if (!config.expansion_factor) config.expansion_factor = default_expansion_factor(connectivity);
        return config;
    }

    /** Raises @ref connectivity to @p width, a neighbor list width some wiring may write; a relaxed
     *  peek first, so a graph at its budget never writes the shared line. */
    void widen_(std::size_t width) noexcept {
        atomic_ref<std::size_t> const widest {widest_list_};
        if (widest.load(std::memory_order_relaxed) < width) widest.store_max(width, std::memory_order_relaxed);
    }

    /** Forgets every record below @p visible, every key and every counter: what @c clear does under
     *  the unique side, and what a copied donor becomes under its plan's. Open claims restart from
     *  zero, while closed ones stay closed, reopening there with @c reopen_. */
    void forget_(node_id_t visible) noexcept {
        for (std::size_t node = 0; node != static_cast<std::size_t>(visible); ++node) {
            node_t& record = nodes_[node];
            record.value = std::make_obj_using_allocator<value_t>(allocator_);
            record.key = key_t {};
            record.neighbors.clear();
            record.state = 0;
        }
        key_map_.clear();
        atomic_ref<std::size_t>(widest_list_).store(0, std::memory_order_relaxed);
        std::size_t open = visible;
        atomic_ref<std::size_t>(claimed_count_)
            .compare_exchange_strong(open, 0, std::memory_order_release, std::memory_order_relaxed);
        atomic_ref<node_id_t>(next_dispensed_id_).store(0, std::memory_order_relaxed);
        atomic_ref<node_id_t>(last_visible_id_).store(0, std::memory_order_release);
    }

    /** Drops the watermark to zero and touches no record, key or claim: what a sharded index makes
     *  of a shard it retires with its claims closed for good, so a reader still walking it reads
     *  values nothing frees, and every accessor gated by the watermark refuses its IDs. */
    void retire_() noexcept { atomic_ref<node_id_t>(last_visible_id_).store(0, std::memory_order_release); }

    /** Files every node of @p range under its key ahead of its watermark, as their one writer: the
     *  range's turn holder, a plan's barrier with the claims closed, or a load's @c finish under
     *  the unique side. Plain stores and one release byte per node. */
    void publish_keys_(iota_view<node_id_t> range) noexcept {
        for (node_id_t const node_id : range) key_map_.insert(nodes_[node_id].key, node_id);
    }

    /** Closes the claims at the watermark: one compare-exchange that succeeds only while every
     *  claimed ID is published, after which every batch refuses until @c reopen_. */
    status_t close_() noexcept {
        std::size_t visible = size();
        return atomic_ref<std::size_t>(claimed_count_)
                       .compare_exchange_strong(visible, closed_claims_k, std::memory_order_acq_rel,
                                                std::memory_order_relaxed)
                   ? status_t::success_k
                   : status_t::contended_k;
    }

    /** Reopens closed claims at the watermark, past every node a copy published meanwhile. */
    void reopen_() noexcept { atomic_ref<std::size_t>(claimed_count_).store(size(), std::memory_order_release); }

    /** Entries claimed by batches and loads, published or not, and the watermark while the claims
     *  are closed: what a shard set measures its bound against. */
    std::size_t claimed_() const noexcept {
        std::size_t const claimed = atomic_ref<std::size_t const>(claimed_count_).load(std::memory_order_acquire);
        return claimed == closed_claims_k ? size() : claimed;
    }

    /**
     *  @brief One batch's capacity, claimed and covered but not dispensed: the shared hold and the
     *      entries, so a sharded index claims on every shard before it dispenses on any.
     *
     *  Dying undispensed gives the entries back, still under the hold; @c dispense_ turns it into
     *  the batch's plan.
     */
    struct [[nodiscard]] capacity_claim_t {
        capacity_claim_t() noexcept = default;
        capacity_claim_t(ram_index* index, std::size_t entries, shared_lock_t&& hold) noexcept
            : index_(index), entries_(entries), hold_(std::move(hold)) {}
        capacity_claim_t(capacity_claim_t&& other) noexcept
            : index_(std::exchange(other.index_, nullptr)), entries_(std::exchange(other.entries_, 0)),
              hold_(std::move(other.hold_)) {}
        capacity_claim_t& operator=(capacity_claim_t&&) = delete;
        capacity_claim_t(capacity_claim_t const&) = delete;
        capacity_claim_t& operator=(capacity_claim_t const&) = delete;
        ~capacity_claim_t() noexcept {
            if (index_ && entries_)
                atomic_ref<std::size_t>(index_->claimed_count_).store_sub(entries_, std::memory_order_release);
        }

      private:
        friend ram_index;

        /** The index claimed in, whose shared side @c hold_ occupies; null in the empty claim. */
        ram_index* index_ = nullptr;

        /** Entries claimed and covered, none dispensed; zero once dispensed. */
        std::size_t entries_ = 0;

        shared_lock_t hold_;
    };

    /** The fallible half of @c plan_insert: enters the shared side, then claims @p count entries
     *  and covers their records and key map chunks, dispensing none; it refuses as @ref plan_insert
     *  does. */
    expected<capacity_claim_t> claim_(std::size_t count) noexcept {
        // The gate opens before the claim, so a batch parked behind a unique holder never
        // sits on committed capacity, spuriously overflowing its peers.
        shared_lock_t hold {shared_mutex_};
        if (!count) return {capacity_claim_t {this, 0, std::move(hold)}, status_t::success_k};
        if (atomic_ref<std::size_t>(claimed_count_).load(std::memory_order_relaxed) == closed_claims_k)
            return {{}, status_t::contended_k};

        // Claims convert one-to-one into dispensed IDs once covered, and commit only within the
        // ceiling, so peers never observe an overshoot. Covering to this claim's end suffices:
        // of the batches dispensed up to this one, the last to claim covered past this range, and
        // the dispenser's acquire-release carries its cover along.
        std::size_t const ceiling = atomic_ref<std::size_t>(ceiling_nodes_).load(std::memory_order_relaxed);
        std::size_t const claimed =
            atomic_ref<std::size_t>(claimed_count_).fetch_add_if_at_most(count, ceiling, std::memory_order_acq_rel);
        if (claimed == closed_claims_k) return {{}, status_t::contended_k};
        // A ceiling lowered under the claims refuses every batch rather than wrapping.
        if (claimed > ceiling || ceiling - claimed < count) return {{}, status_t::overflow_k};
        // The key map first, so a refused one leaves the node chunks, and the capacity, as they
        // were.
        status_t covered = key_map_.reserve(claimed + count);
        if (succeeded(covered)) covered = nodes_.cover(claimed + count);
        if (failed(covered)) {
            atomic_ref<std::size_t>(claimed_count_).store_sub(count, std::memory_order_release);
            return {{}, covered};
        }
        return {capacity_claim_t {this, count, std::move(hold)}, status_t::success_k};
    }

    /** The half of @ref plan_insert that cannot fail: one add dispenses the entries of @p claim as
     *  the range one turn will publish; the empty claim opens the empty plan. */
    insert_plan_t dispense_(capacity_claim_t&& claim) noexcept {
        if (!claim.index_) return {};
        std::size_t const count = std::exchange(claim.entries_, 0);
        if (!count) return {this, {}, std::move(claim.hold_)};
        // The tasks run between this add and the turn, so a slow scorer delays the batches
        // dispensed after this one.
        node_id_t const first_id = atomic_ref<node_id_t>(next_dispensed_id_)
                                       .fetch_add(static_cast<node_id_t>(count), std::memory_order_acq_rel);
        iota_view<node_id_t> const range {first_id, static_cast<node_id_t>(first_id + count)};
        return {this, range, std::move(claim.hold_)};
    }

    /** An audit over the IDs below @p visible, its columns placed by @p allocator. */
    template <typename bits_allocator_type_>
    expected<health_plan<bits_allocator_type_>> plan_audit_(node_id_t visible,
                                                            bits_allocator_type_ const& allocator) noexcept {
        // An empty graph is audited, not refused: no words to sweep, no tasks, zero reached.
        health_plan<bits_allocator_type_> health {allocator};
        status_t const placed = health.bind_(*this, visible);
        return {std::move(health), placed};
    }

    /** Codec bytes the published values of this index keep, summed: what a round's entries weigh
     *  in a core's cache past their records. */
    template <typename codec_type_>
    std::size_t payload_bytes_(codec_type_ const& codec) const noexcept {
        std::size_t payload = 0;
        for (node_id_t node_id = 0; node_id != static_cast<node_id_t>(size()); ++node_id)
            payload += codec.storage_requirements(nodes_[node_id].value);
        return payload;
    }

    /** What @ref plan_round and @ref plan_absorb share: every check, hold, claim and reservation
     *  of a round over the published nodes, or over the appended ones too once @p donors move in,
     *  every neighbor list reserved to @p connectivity. */
    template <typename codec_type_>
    expected<refine_plan_t> plan_round_(std::span<ram_index*> donors, traversal_context_t& context,
                                        std::size_t connectivity, refine_config_t config,
                                        codec_type_ const& codec) noexcept {
        using phase_t = typename refine_plan_t::phase_t;
        refine_plan_t plan {context.allocator};
        auto refuse = [&](status_t status) noexcept {
            plan.release_();
            return expected<refine_plan_t> {std::move(plan), status};
        };
        if (!connectivity) return refuse(status_t::config_mismatch_k);
        for (ram_index const* donor : donors)
            if (!donor || donor == this) return refuse(status_t::config_mismatch_k);

        // From here the plan reopens the claims on any refusal, and at its death.
        plan.hold_ = shared_lock_t {shared_mutex_};
        if (status_t const closed = close_(); failed(closed)) return refuse(closed);
        plan.index_ = this;
        if (failed(plan.donor_holds_.reserve_exactly(donors.size()))) return refuse(status_t::bad_alloc_k);
        node_id_t const visible = static_cast<node_id_t>(size());
        std::size_t entries = visible;
        std::size_t const ceiling = atomic_ref<std::size_t>(ceiling_nodes_).load(std::memory_order_relaxed);
        for (ram_index* donor : donors) {
            unique_lock_t donor_hold {donor->shared_mutex_, std::try_to_lock};
            if (!donor_hold) return refuse(status_t::contended_k);
            auto const [donor_entries, watermark_status] = donor->watermark_();
            if (failed(watermark_status)) return refuse(watermark_status);
            plan.donor_holds_.push_back_reserved(std::move(donor_hold));
            // Both terms stay within the address space's records, so their sum never wraps.
            if (entries + static_cast<std::size_t>(donor_entries) > ceiling) return refuse(status_t::overflow_k);
            entries += donor_entries;
        }
        if (status_t const covered = nodes_.cover(entries); failed(covered)) return refuse(covered);
        if (status_t const keyed = key_map_.reserve(entries); failed(keyed)) return refuse(keyed);
        auto [health, health_status] = plan_audit_(static_cast<node_id_t>(entries), context.allocator);
        if (failed(health_status)) return refuse(health_status);
        plan.health = std::move(health);

        // Neighbor lists last, under their locks, as searches copy them meanwhile; the donors are
        // held. Each keeps a slot past the budget, which a heal fills when a restitch left none.
        std::size_t const healed_list = connectivity + 1;
        for (node_id_t node_id = 0; node_id != visible; ++node_id) {
            node_t& node = nodes_[node_id];
            node.lock();
            status_t const grown = node.neighbors.reserve_exactly(healed_list);
            node.unlock();
            if (failed(grown)) return refuse(grown);
        }
        for (ram_index* donor : donors)
            for (std::size_t node = 0; node != donor->size(); ++node)
                if (status_t const grown = donor->nodes_[node].neighbors.reserve_exactly(healed_list); failed(grown))
                    return refuse(grown);

        widen_(healed_list);
        // Every donor's values count, since they move in before the opening health check.
        std::size_t payload = payload_bytes_(codec);
        for (ram_index const* donor : donors) {
            widen_(donor->connectivity());
            payload += donor->payload_bytes_(codec);
        }
        plan.donors_ = donors;
        plan.connectivity_ = connectivity;
        plan.widest_list_ = this->connectivity();
        plan.size_round_(config.cache_bytes,
                         sizeof(value_t) + divide_round_up<std::size_t>(payload, max_of<std::size_t>(entries, 1)));
        if (failed(plan.reserve_round_(entries))) return refuse(status_t::bad_alloc_k);
        // The audit bound above opens the round, or follows the donors' moves.
        plan.swept_after_ = phase_t::opening_k;
        plan.phase_ = donors.empty() ? phase_t::health_k : phase_t::copy_k;
        return {std::move(plan), status_t::success_k};
    }

    /** What @ref plan_split does: every check, hold, claim, cover and reservation of a split of
     *  this index into @p receivers, every neighbor list reserved to @p connectivity. */
    template <typename codec_type_>
    expected<refine_plan_t> plan_split_(std::span<ram_index*> receivers, traversal_context_t& context,
                                        std::size_t connectivity, refine_config_t config, u64_t seed,
                                        codec_type_ const& codec) noexcept {
        using phase_t = typename refine_plan_t::phase_t;
        refine_plan_t plan {context.allocator};
        auto refuse = [&](status_t status) noexcept {
            plan.release_();
            return expected<refine_plan_t> {std::move(plan), status};
        };
        std::size_t const parts = receivers.size();
        if (!connectivity || parts < 2) return refuse(status_t::config_mismatch_k);
        for (std::size_t receiver = 0; receiver != parts; ++receiver) {
            if (!receivers[receiver] || receivers[receiver] == this) return refuse(status_t::config_mismatch_k);
            for (std::size_t earlier = 0; earlier != receiver; ++earlier)
                if (receivers[earlier] == receivers[receiver]) return refuse(status_t::config_mismatch_k);
        }

        // The split's donor is held as an absorb's donor is, so no batch goes into it while its
        // nodes move out.
        unique_lock_t split_donor_hold {shared_mutex_, std::try_to_lock};
        if (!split_donor_hold) return refuse(status_t::contended_k);
        auto const [entries, watermark_status] = watermark_();
        if (failed(watermark_status)) return refuse(watermark_status);
        plan.split_donor_ = this;
        plan.split_donor_hold_ = std::move(split_donor_hold);
        plan.receivers_ = receivers;
        std::size_t const capacity = divide_round_up<std::size_t>(entries, parts);
        if (parts * capacity > static_cast<std::size_t>(missing_k)) return refuse(status_t::overflow_k);
        if (failed(plan.receiver_holds_.reserve_exactly(parts)) || failed(plan.part_extents_.resize(parts)) ||
            failed(plan.part_slots_.resize(entries)) || failed(plan.part_members_.resize(parts * capacity)))
            return refuse(status_t::bad_alloc_k);
        // One audit and one round's columns over a full part serve every receiver in turn, rebound
        // as its turn comes.
        auto [health, health_status] = plan_audit_(static_cast<node_id_t>(capacity), context.allocator);
        if (failed(health_status)) return refuse(health_status);
        plan.health = std::move(health);
        plan.connectivity_ = connectivity;
        plan.widest_list_ = max_of(connectivity + 1, this->connectivity());
        std::size_t const payload = payload_bytes_(codec);
        plan.size_round_(config.cache_bytes,
                         sizeof(value_t) + divide_round_up<std::size_t>(payload, max_of<std::size_t>(entries, 1)));
        if (failed(plan.reserve_round_(capacity))) return refuse(status_t::bad_alloc_k);

        // From here the plan reopens the claims of every receiver it closed on any refusal, and at
        // its death; the claims closed at an empty watermark prove no batch is in flight.
        for (ram_index* receiver : receivers) {
            shared_lock_t receiver_hold {receiver->shared_mutex_};
            if (status_t const closed = receiver->close_(); failed(closed)) return refuse(closed);
            plan.receiver_holds_.push_back_reserved(std::move(receiver_hold));
            if (receiver->size()) return refuse(status_t::config_mismatch_k);
            if (capacity > atomic_ref<std::size_t>(receiver->ceiling_nodes_).load(std::memory_order_relaxed))
                return refuse(status_t::overflow_k);
        }
        for (ram_index* receiver : receivers) {
            if (status_t const covered = receiver->nodes_.cover(capacity); failed(covered)) return refuse(covered);
            if (status_t const keyed = receiver->key_map_.reserve(capacity); failed(keyed)) return refuse(keyed);
        }
        // Neighbor lists last; each moves its buffer into a receiver, where re-stitching allocates
        // nothing, with a slot past the budget for a heal.
        for (node_id_t node_id = 0; node_id != entries; ++node_id)
            if (status_t const grown = nodes_[node_id].neighbors.reserve_exactly(connectivity + 1); failed(grown))
                return refuse(grown);

        for (ram_index* receiver : receivers) {
            receiver->widen_(connectivity + 1);
            receiver->widen_(this->connectivity());
        }
        plan.index_ = receivers[0];
        plan.seed_ = seed;
        plan.phase_ = phase_t::partition_k;
        return {std::move(plan), status_t::success_k};
    }

    /** Writes @c context.selected_neighbors, at most @p connectivity of them, into @p node_id 's
     *  list under its lock. */
    template <typename context_type_>
    void rewrite_list_(node_id_t node_id, std::size_t connectivity, context_type_& context) noexcept {
        node_t& node = nodes_[node_id];
        node.lock();
        std::size_t const kept_count = min_of(min_of(context.selected_neighbors.size(), connectivity),
                                              node.neighbors.capacity());
        node.neighbors.clear();
        for (std::size_t kept = 0; kept != kept_count; ++kept)
            node.neighbors.push_back_reserved(context.selected_neighbors[kept].node_id);
        node.unlock();
    }

    /**
     *  @brief Puts back what @p selected dropped of the @p live neighbors that reachability needs:
     *      a neighbor whose last shallower in-link @p node_id is, and, when no pick stands one
     *      depth to root closer to node zero, the node's nearest live neighbor that does. A list
     *      short of @p connectivity grows by each; a full one gives up its farthest pick that is
     *      kept by neither, a live neighbor only once another shallower in-link keeps it.
     *
     *  By induction over the depths every node reached both ways stays so throughout the merge: a
     *  node at a depth keeps one in-link from the depth before, and one out-link to a lower depth
     *  to root, and node zero never moves.
     */
    template <typename live_type_, typename selected_type_>
    void keep_reach_(node_id_t node_id, live_type_ const& live, selected_type_& selected, std::size_t connectivity,
                     health_plan_t& health) noexcept {
        constexpr u8_t unknown_depth_k = health_plan_t::unknown_depth_k;
        u8_t const depth = health.depth_from_root(node_id), depth_to_root = health.depth_to_root(node_id);
        // The saturated depth counts no shallower in-links, so its children are never released.
        u8_t const child_depth = depth < unknown_depth_k - 1 ? static_cast<u8_t>(depth + 1) : unknown_depth_k;
        auto position_in = []<typename among_type_>(among_type_ const& among, node_id_t member) noexcept {
            std::size_t position = 0;
            while (position != among.size() && among[position].node_id != member) ++position;
            return position;
        };
        auto releasable = [&](node_id_t member) noexcept {
            return child_depth == unknown_depth_k || health.depth_from_root(member) != child_depth ||
                   position_in(live, member) == live.size() || health.take_shallower_in_link_if_spare(member);
        };
        // Kept members gather at the tail, ahead of which the farthest releasable pick gives way.
        std::size_t kept = 0;
        auto keep = [&](scored_node_t member) noexcept {
            // A member the selection holds stays where it ranks; moving it would unsort the picks
            // the farthest of which give way.
            if (position_in(selected, member.node_id) != selected.size()) return;
            if (selected.size() < min_of(connectivity, selected.capacity())) {
                selected.push_back_reserved(member);
                ++kept;
                return;
            }
            std::size_t const tail = selected.size() - kept;
            std::size_t position = tail;
            while (position && !releasable(selected[position - 1].node_id)) --position;
            if (!position--) return;
            selected[position] = member;
            std::swap(selected[position], selected[tail - 1]);
            ++kept;
        };
        if (child_depth != unknown_depth_k)
            for (std::size_t position = 0; position != live.size(); ++position) {
                scored_node_t const member = live[position];
                if (health.depth_from_root(member.node_id) != child_depth ||
                    position_in(selected, member.node_id) != selected.size())
                    continue;
                if (!health.take_shallower_in_link_if_spare(member.node_id)) keep(member);
            }
        if (depth_to_root != unknown_depth_k && depth_to_root != 0) {
            std::size_t standing = 0;
            while (standing != selected.size() && health.depth_to_root(selected[standing].node_id) >= depth_to_root)
                ++standing;
            scored_node_t way_to_root = standing != selected.size()
                                            ? selected[standing]
                                            : scored_node_t {missing_k, unbounded_radius<distance_t>()};
            for (std::size_t position = 0; position != live.size() && standing == selected.size(); ++position)
                if (health.depth_to_root(live[position].node_id) < depth_to_root &&
                    live[position].distance < way_to_root.distance)
                    way_to_root = live[position];
            if (way_to_root.node_id != missing_k) keep(way_to_root);
        }
    }

    /** Adds @p node_id 's current neighbor list to the candidates a restitch picks from, so the
     *  long links among them still compete with the neighborhood-mates. Scored in as many rounds as
     *  the metric's staging takes; a refused copy or growth, or a metric refusing an empty round,
     *  counts a saturated walk. */
    template <typename metric_type_, typename admission_type_, typename context_type_>
    void add_neighbor_candidates_(node_id_t node_id, value_t const& value, metric_type_& metric,
                                  admission_type_ const& admission, context_type_& context) noexcept {
        context.staged_ids.clear();
        auto const [degree, copy_status] = copy_neighbors_<synchronization_t::concurrent_k>(node_id, context);
        if (failed(copy_status) || failed(context.results.reserve(context.results.size() + degree))) {
            ++context.telemetry.saturated_walks;
            return;
        }
        std::size_t missing = 0;
        for (std::size_t position = 0; position != degree; ++position) {
            node_id_t const neighbor_id = context.staged_ids[position];
            bool listed = !admission.admits(neighbor_id);
            for (std::size_t member = 0; member != context.results.size() && !listed; ++member)
                listed = context.results[member].node_id == neighbor_id;
            if (!listed) context.staged_ids[missing++] = neighbor_id;
        }
        for (std::size_t first = 0; first != missing;) {
            metric.reset(value);
            std::size_t staged_count = 0;
            while (first + staged_count != missing &&
                   metric.push(published_value_of(context.staged_ids[first + staged_count])) ==
                       metric_staging_t::staged_k)
                ++staged_count;
            if (!staged_count) {
                ++context.telemetry.saturated_walks;
                return;
            }
            drain_round(metric, staged_count, context, unbounded_radius<distance_t>());
            for (std::size_t lane = 0; lane != staged_count; ++lane)
                context.results.insert(scored_node_t {context.staged_ids[first + lane], context.staged_distances[lane]},
                                       context.results.size() + 1);
            first += staged_count;
        }
    }

    /** One position of a plan: the walk against the caller's still-unmoved value, the selection,
     *  the value & key moved into the record, the neighbor list wired and the node flagged, then
     *  the back-links wired from its neighbors. */
    template <synchronization_t synchronization_, typename metric_type_, typename context_type_,
              typename selector_type_>
    void place_(node_id_t new_node_id, value_t& incoming, key_t const& key, metric_type_& metric,
                context_type_& context, selector_type_ const& selector, insert_config_t walk) noexcept {
        std::size_t const connectivity = selector.connectivity;
        context.results.clear();
        context.frontier.clear();
        context.visited.clear();

        // Walk against the caller's still-unmoved value. The ID is already dispensed, so a refused
        // walk wires a thinner node rather than abandoning it. A short walk waits out the entries
        // dispensed ahead: wired at once, each would meet node zero alone.
        std::size_t const wanted = min_of<std::size_t>(connectivity, walk.expansion_factor);
        for (bool waited = false;; waited = true) {
            std::size_t const saturated_before = context.telemetry.saturated_walks;
            node_id_t const first_invisible_id =
                atomic_ref<node_id_t>(last_visible_id_).load(std::memory_order_acquire);
            search_around_<synchronization_>(
                incoming, metric, dispensed_admission_t {this, first_invisible_id, new_node_id}, walk.expansion_factor,
                connectivity, context.results, context, dummy_fold_t {});
            if (waited || first_invisible_id == new_node_id || context.results.size() >= wanted ||
                context.telemetry.saturated_walks != saturated_before)
                break;
            for (node_id_t ahead = first_invisible_id; ahead != new_node_id; ++ahead) {
                u32_t& state = nodes_[ahead].state;
                for (u32_t observed = atomic_ref<u32_t>(state).load(std::memory_order_acquire);
                     !(observed & node_t::wired_k); observed = atomic_ref<u32_t>(state).load(std::memory_order_acquire))
                    wait_t {}(&state, observed, 0u);
            }
        }
        selector(*this, metric, context);
        fill_selection(context, connectivity);

        node_t& record = nodes_[new_node_id];
        record.value = std::move(incoming);
        record.key = key;
        context.reverse_ids.clear();
        std::size_t const selected_count = min_of(context.selected_neighbors.size(), connectivity);
        // A failed load leaves its unpublished nodes filled; this one starts from nothing.
        record.neighbors.clear();
        // A refused neighbor list growth wires a thinner node, counted like a refused walk, since
        // the dispensed ID must still publish.
        bool const grown = succeeded(record.neighbors.reserve(selected_count));
        context.telemetry.saturated_walks += !grown;
        std::size_t const wired_count = grown ? selected_count : 0;
        for (std::size_t link = 0; link != wired_count; ++link) {
            record.neighbors.push_back_reserved(context.selected_neighbors[link].node_id);
            context.reverse_ids.push_back_reserved(context.selected_neighbors[link].node_id);
        }
        record.wire();

        back_link_<synchronization_>(new_node_id, metric, selector, context);
    }

    /**
     *  @brief Appends @p node_id's live neighbors into @c context.staged_ids - under the node's
     *      lock on the concurrent contract, as a plain copy otherwise - topping up the staging
     *      surfaces first, since manual linking grows neighbor lists past any estimate.
     *
     *  Bounded copies only, never metric verbs.
     *  @return The copied degree, or the refusal that kept the neighbor list out of staging. */
    template <synchronization_t synchronization_, typename context_type_>
    expected<std::size_t> copy_neighbors_(node_id_t node_id, context_type_& context) noexcept {
        constexpr bool locked_k = synchronization_ == synchronization_t::concurrent_k;
        node_t& node = nodes_[node_id];
        if constexpr (locked_k) node.lock();
        std::size_t const degree = node.neighbors.size();
        std::size_t const staged_bound = context.staged_ids.size() + degree;
        if (failed(context.staged_ids.reserve(staged_bound)) ||
            failed(context.staged_distances.reserve(staged_bound))) {
            if constexpr (locked_k) node.unlock();
            return {0, status_t::bad_alloc_k};
        }
        for (std::size_t position = 0; position != degree; ++position)
            context.staged_ids.push_back_reserved(node.neighbors[position]);
        if constexpr (locked_k) node.unlock();
        return {degree, status_t::success_k};
    }

    /** What a search walks: the IDs below its watermark snapshot, published for every reader. */
    struct snapshot_admission_t {

        /** The watermark snapshot. */
        node_id_t first_invisible_id = 0;

        /** Whether the walk may locate @p node_id. */
        bool admits(node_id_t node_id) const noexcept { return node_id < first_invisible_id; }

        /** Landmarks come from the IDs from this one, the first. */
        node_id_t sampling_first() const noexcept { return 0; }

        /** Landmarks come from the IDs below this bound, the snapshot. */
        node_id_t sampling_bound() const noexcept { return first_invisible_id; }
    };

    /** What an insertion walks: the IDs below its snapshot, and past it every node whose state
     *  shows it wired. The acquire of that state carries the node's value and neighbors, so batches
     *  wired side by side link to each other rather than forming islands. */
    struct dispensed_admission_t {

        /** The index the walk reads. */
        ram_index const* index = nullptr;

        /** The watermark snapshot. */
        node_id_t first_invisible_id = 0;

        /** The walking node. */
        node_id_t node_id = 0;

        /** Whether the walk may locate @p candidate_id. */
        bool admits(node_id_t candidate_id) const noexcept {
            return candidate_id < first_invisible_id || index->nodes_[candidate_id].wired();
        }

        /** Landmarks come from the IDs from this one, the first. */
        node_id_t sampling_first() const noexcept { return 0; }

        /** Landmarks come from the IDs below this bound: each was dispensed ahead of the walking
         *  node, so most are wired already. */
        node_id_t sampling_bound() const noexcept { return node_id; }
    };

    /**
     *  @brief Scores up to @p wanted fresh stochastic landmarks in one builder round,
     *      seeding the results & frontier. Visited collisions probe forward cyclically,
     *      staying exact.
     *  @return @c advanced_k, @c exhausted_k once nothing fresh stages - every admitted node visited,
     *      or the metric refusing the first one - or @c sated_k once the fold asks the walk to end. A
     *      failed status means scratch could not grow, counted as a saturated walk.
     *
     *  Draws stay within the IDs @p admission samples from. Past the snapshot a node may stay
     *  unwired for the whole walk, so the re-rolls of unadmitted draws stop once they outnumber the
     *  IDs there.
     */
    template <typename metric_type_, typename query_type_, typename admission_type_, typename context_type_,
              typename fold_type_>
    expected<walk_step_t> sample_landmarks_(query_type_ const& query, metric_type_& metric,
                                            admission_type_ const& admission, std::size_t wanted,
                                            std::size_t expansion_factor, context_type_& context, fold_type_&& fold,
                                            std::span<node_id_t const>& seeds) const noexcept {
        if (status_t const seen = context.visited.reserve(context.visited.size() + wanted); failed(seen)) {
            ++context.telemetry.saturated_walks;
            return {walk_step_t::exhausted_k, seen};
        }
        // `visited` may also hold IDs no draw names, like a later batch's nodes, so the size check
        // can only under-sample, never let the probe loop spin.
        node_id_t const sampling_first = admission.sampling_first(), sampling_bound = admission.sampling_bound();
        node_id_t unwired_draws = 0;
        context.staged_ids.clear();
        while (context.staged_ids.size() != wanted &&
               context.visited.size() < static_cast<std::size_t>(sampling_bound - sampling_first)) {
            node_id_t draw;
            // Caller-staged seeds go first, each taken once and skipped when unadmitted or visited.
            if (!seeds.empty()) {
                draw = seeds.front();
                seeds = seeds.subspan(1);
                if (!admission.admits(draw) || context.visited.contains(draw)) continue;
            }
            else {
                draw = sampling_first +
                       uniform_below<node_id_t> {sampling_bound - sampling_first}(context.random_generator);
                while (context.visited.contains(draw)) draw = draw + 1 == sampling_bound ? sampling_first : draw + 1;
                if (!admission.admits(draw)) {
                    if (++unwired_draws == sampling_bound - admission.first_invisible_id) break;
                    continue;
                }
            }
            [[maybe_unused]] membership_t const marked = context.visited.insert(draw);
            context.staged_ids.push_back_reserved(draw);
        }
        // Drawn before any scoring, so a staging size never changes which landmarks a walk sees.
        std::size_t const drawn_count = context.staged_ids.size();
        for (std::size_t first = 0; first != drawn_count;) {
            metric.reset(query);
            std::size_t staged_count = 0;
            while (first + staged_count != drawn_count &&
                   metric.push(published_value_of(context.staged_ids[first + staged_count])) ==
                       metric_staging_t::staged_k)
                ++staged_count;
            if (!staged_count) return {walk_step_t::exhausted_k, status_t::success_k};
            drain_round(metric, staged_count, context, unbounded_radius<distance_t>());
            walk_demand_t const demand = fold.absorb(
                std::span<node_id_t const>(context.staged_ids).subspan(first, staged_count),
                std::span<distance_t const>(context.staged_distances));
            settle_fold(fold);
            if (demand == walk_demand_t::sated_k) return {walk_step_t::sated_k, status_t::success_k};
            for (std::size_t lane = 0; lane != staged_count; ++lane) {
                scored_node_t const scored {context.staged_ids[first + lane], context.staged_distances[lane]};
                [[maybe_unused]] bool const ranked = context.results.insert(scored_node_t {scored}, expansion_factor);
                if (status_t const queued = context.frontier.insert(scored_node_t {scored}); failed(queued)) {
                    ++context.telemetry.saturated_walks;
                    return {walk_step_t::advanced_k, queued};
                }
            }
            first += staged_count;
        }
        if (!drawn_count) return {walk_step_t::exhausted_k, status_t::success_k};
        return {walk_step_t::advanced_k, status_t::success_k};
    }

    /** One round's expansion half: peels up to @p filling_nodes frontier entries within the
     *  stopping radius, copies their neighbor lists, and pushes the admission- and visited-filtered
     *  union into the metric, compacted in @c context.staged_ids. A full staging ends the round
     *  early. A staging that fills mid-list scores its IDs through the fold into @p scored_count of
     *  the round's distances, so a metric of small staging scores as it goes and the round ranks
     *  once; @c sated_k once the fold ends the walk there. */
    template <synchronization_t synchronization_, typename metric_type_, typename query_type_, typename admission_type_,
              typename stopping_type_, typename context_type_, typename fold_type_>
    walk_step_t expand_frontier_(query_type_ const& query, metric_type_& metric, admission_type_ const& admission,
                                 std::size_t expansion_factor, std::size_t filling_nodes,
                                 stopping_type_ const& stopping, context_type_& context, fold_type_&& fold,
                                 std::size_t& scored_count) noexcept {
        context.staged_ids.clear();
        metric.reset(query);
        scored_count = 0;
        for (std::size_t expanded_count = 0; expanded_count != filling_nodes && !context.frontier.empty();
             ++expanded_count) {
            scored_node_t const best = context.frontier.top();
            bool const stopping_full = stopping.size() == expansion_factor;
            if (stopping_full && best.distance > stopping.top().distance) {
                if (expanded_count == 0) return walk_step_t::exhausted_k;
                break;
            }

            std::size_t const segment_start = context.staged_ids.size();
            auto const [degree, status] = copy_neighbors_<synchronization_>(best.node_id, context);
            if (failed(status) || failed(context.visited.reserve(context.visited.size() + degree))) {
                // Drop the raw neighbor list copy: only pushed, filtered IDs may reach the drain.
                // The entry stays on the frontier - nothing is lost, only postponed.
                context.staged_ids.resize_reserved(segment_start);
                ++context.telemetry.saturated_walks;
                return walk_step_t::saturated_k;
            }
            context.frontier.pop();
            // The next entry's record, not its neighbor list, which a writer may move until a copy
            // locks it.
            if (!context.frontier.empty())
                prefetcher_.prefetch(&nodes_[context.frontier.top().node_id], sizeof(node_t));

            std::size_t kept = segment_start;
            for (std::size_t position = segment_start; position != segment_start + degree; ++position) {
                node_id_t const neighbor_id = context.staged_ids[position];
                // The translation frontier: an ID the admission refuses has no
                // happens-before edge to its record, so a later query picks it up.
                if (!admission.admits(neighbor_id)) continue;
                if (context.visited.contains(neighbor_id)) continue;
                if (metric.push(published_value_of(neighbor_id)) == metric_staging_t::refused_k) {
                    // A full staging scores what it holds at the round's radius and takes the rest
                    // of the list; the round still ranks once, at its end. Only a metric refusing
                    // an empty staging saturates the walk.
                    if (score_staged_(metric, scored_count, kept, expansion_factor, stopping, context, fold) ==
                        walk_demand_t::sated_k) {
                        context.staged_ids.clear();
                        return walk_step_t::sated_k;
                    }
                    scored_count = kept;
                    metric.reset(query);
                    if (metric.push(published_value_of(neighbor_id)) == metric_staging_t::refused_k) {
                        context.staged_ids.resize_reserved(kept);
                        ++context.telemetry.saturated_walks;
                        return walk_step_t::saturated_k;
                    }
                }
                [[maybe_unused]] membership_t const marked = context.visited.insert(neighbor_id);
                context.staged_ids[kept] = neighbor_id;
                ++kept;
            }
            context.staged_ids.resize_reserved(kept);
            ++context.telemetry.visited_members;
        }
        return walk_step_t::advanced_k;
    }

    /** The rounds of every scan: each node below the watermark @p first_invisible_id the caller
     *  took, the index's own or one consistent across shards, staged as many at a time as the
     *  metric takes, drained within @p radius, and handed to @p fold, which may end the scan with
     *  @c sated_k. A metric refusing an empty round can never finish, which a scan reports as a
     *  refused allocation. */
    template <typename query_type_, typename metric_type_, typename radius_type_, typename fold_type_,
              typename context_type_>
    expected<search_result_t> scan_below_(query_type_ const& query, metric_type_& metric, radius_type_ const& radius,
                                          fold_type_& fold, context_type_& context, std::size_t wanted,
                                          node_id_t first_invisible_id) noexcept {
        if (!first_invisible_id) return {};
        // A scan stages rounds and ranks, and never touches the frontier or the visited set.
        std::size_t const staging = round_capacity(connectivity(), {wanted});
        if (failed(context.staged_ids.reserve(staging)) || failed(context.staged_distances.reserve(staging)) ||
            failed(context.results.reserve(wanted)))
            return {{}, status_t::bad_alloc_k};
        traversal_stats_t const telemetry_before = context.telemetry;
        for (node_id_t first = 0; first != first_invisible_id;) {
            metric.reset(query);
            context.staged_ids.clear();
            node_id_t next = first;
            for (; next != first_invisible_id && context.staged_ids.size() != context.staged_ids.capacity(); ++next) {
                if (metric.push(published_value_of(next)) == metric_staging_t::refused_k) break;
                context.staged_ids.push_back_reserved(next);
            }
            if (next == first) return {{}, status_t::bad_alloc_k};
            drain_round(metric, context.staged_ids.size(), context, radius());
            walk_demand_t const demand = fold.absorb(std::span<node_id_t const>(context.staged_ids),
                                                     std::span<distance_t const>(context.staged_distances));
            settle_fold(fold);
            if (demand == walk_demand_t::sated_k) break;
            first = next;
        }
        return {{context.results.size(), context.telemetry - telemetry_before}, status_t::success_k};
    }

    /** The one walk behind @c search and @c search_if: @p fold reads every scored round and the
     *  walk stops on @p stopping, its own candidates or the admitted set. One watermark snapshot
     *  bounds the whole walk; a marked mode proves its own certification through the shared
     *  mutex's wait-free probes instead of trusting it. */
    template <synchronization_t synchronization_, typename metric_type_, typename fold_type_, typename stopping_type_,
              typename context_allocator_type_, typename query_type_>
    expected<search_result_t> fold_walk_(query_type_ const& query, metric_type_& metric, fold_type_&& fold,
                                         stopping_type_ const& stopping,
                                         traversal_context_over<context_allocator_type_>& context,
                                         search_config_t config) noexcept {
        if constexpr (synchronization_ == synchronization_t::concurrent_k)
            return fold_walk_below_<synchronization_>(
                query, metric, fold, stopping, context, config,
                atomic_ref<node_id_t>(last_visible_id_).load(std::memory_order_acquire));
        else {
            auto const entry_probe = shared_mutex_.snapshot();
            if (!entry_probe.unlocked()) return {{}, status_t::contended_k};
            auto const [first_invisible_id, watermark_status] = watermark_();
            if (failed(watermark_status)) return {{}, watermark_status};
            expected<search_result_t> found = fold_walk_below_<synchronization_>(query, metric, fold, stopping, context,
                                                                                 config, first_invisible_id);
            // The fence pairs with the release fence in `lock_shared`: a walk read that observed a
            // mutator's write proves its acquisition to the reload, which then reads a moved
            // word. The watermark equality is the belt against generation wrap driven by inserts.
            std::atomic_thread_fence(std::memory_order_acquire);
            auto const exit_probe = shared_mutex_.snapshot();
            if (succeeded(found.status) &&
                (!exit_probe.unlocked() || exit_probe.generation != entry_probe.generation ||
                 atomic_ref<node_id_t>(last_visible_id_).load(std::memory_order_acquire) != first_invisible_id))
                found.status = status_t::contended_k;
            return found;
        }
    }

    /** The walk behind every search below the watermark @p first_invisible_id the caller took: the
     *  index's own, or one consistent across shards, seeded by @p hinted ahead of its random draws.
     *  A marked @p synchronization_ compiles the node locks out, its caller proving it. */
    template <synchronization_t synchronization_, typename metric_type_, typename fold_type_, typename stopping_type_,
              typename context_allocator_type_, typename query_type_>
    expected<search_result_t> fold_walk_below_(query_type_ const& query, metric_type_& metric, fold_type_&& fold,
                                               stopping_type_ const& stopping,
                                               traversal_context_over<context_allocator_type_>& context,
                                               search_config_t config, node_id_t first_invisible_id,
                                               std::span<node_id_t const> hinted = {}) noexcept {
        std::size_t const widest_list = connectivity();
        insert_config_t const walk = resolved_config_({config.expansion_factor}, widest_list);
        if (status_t const sized = reserve_traversal(context, widest_list, walk); failed(sized)) return {{}, sized};
        std::size_t const expansion_factor = walk.expansion_factor;
        context.results.clear();
        context.frontier.clear();
        context.visited.clear();
        if (!first_invisible_id) return {};

        traversal_stats_t const telemetry_before = context.telemetry;
        search_around_<synchronization_>(query, metric, snapshot_admission_t {first_invisible_id}, expansion_factor,
                                         widest_list, stopping, context, fold, hinted);

        // A refused walk still ranked whatever it reached; the telemetry says the ranking is
        // short because scratch ran out, not because the graph had nothing better.
        search_result_t result;
        result.count = context.results.size();
        result.telemetry = context.telemetry - telemetry_before;
        return {result, status_t::success_k};
    }

    /**
     *  @brief The shared walk of @c search & @c insert: seed, then per round expand the
     *      nearest frontier entries and score the staged union in one builder round - every
     *      scored round, seeds included, reaching the fold before the candidates.
     *
     *  The walk locates only the IDs @p admission admits: a search's snapshot, or an insertion's
     *  snapshot widened by the nodes wired ahead of their turns. It never fails hard: an exhausted
     *  admitted set, a refused staging or a sated fold ends it normally, and a refused re-table
     *  ends it early with the ranking it has and counts a saturated walk - an insertion wires the
     *  thinner neighbor list regardless, and a search reports the count so the caller can widen the
     *  scratch. Frontier starvation with unfilled candidates re-seeds and counts a restart. The
     *  frontier-admission rule: a candidate enters the frontier iff it improved the results, or the
     *  frontier would otherwise be empty - the seedings, which take @p seeds ahead of random draws.
     */
    template <synchronization_t synchronization_, typename metric_type_, typename query_type_, typename admission_type_,
              typename stopping_type_, typename context_type_, typename fold_type_>
    void search_around_(query_type_ const& query, metric_type_& metric, admission_type_ const& admission,
                        std::size_t expansion_factor, std::size_t connectivity, stopping_type_ const& stopping,
                        context_type_& context, fold_type_&& fold, std::span<node_id_t const> seeds = {}) noexcept {
        // A round expands the nodes whose neighbors fill a walk's candidates, as `round_capacity`
        // sized.
        std::size_t const filling_nodes = divide_round_up(expansion_factor, connectivity);
        std::size_t scored_count = 0;
        while (true) {
            // An empty frontier seeds: the first round, or a restart after it drained before the
            // candidates filled. Nothing left to seed from ends the walk.
            if (context.frontier.empty()) {
                if (stopping.size() >= expansion_factor) return;
                bool const restarting = !context.results.empty();
                expected<walk_step_t> const seeded = sample_landmarks_(query, metric, admission, connectivity,
                                                                       expansion_factor, context, fold, seeds);
                if (failed(seeded.status) || seeded.value != walk_step_t::advanced_k) return;
                context.telemetry.restarts += restarting;
                continue;
            }
            walk_step_t const expanded = expand_frontier_<synchronization_>(
                query, metric, admission, expansion_factor, filling_nodes, stopping, context, fold, scored_count);
            if (expanded == walk_step_t::exhausted_k || expanded == walk_step_t::sated_k) return;
            walk_step_t const ranked = rank_round_(metric, scored_count, expansion_factor, stopping, context, fold);
            if (expanded == walk_step_t::saturated_k || ranked != walk_step_t::advanced_k) return;
        }
    }

    /** The radius of a round: full stopping candidates only admit what beats their furthest member,
     *  so that member's distance is it, which a filtered walk takes from its admitted set. */
    template <typename stopping_type_>
    static distance_t round_radius_(std::size_t expansion_factor, stopping_type_ const& stopping) noexcept {
        return stopping.size() == expansion_factor ? stopping.top().distance : unbounded_radius<distance_t>();
    }

    /** Drains the metric's staging, the round's IDs from @p scored_count to @p staged_count , at
     *  the round's radius, and hands them to the fold, which may end the walk. */
    template <typename metric_type_, typename stopping_type_, typename context_type_, typename fold_type_>
    walk_demand_t score_staged_(metric_type_& metric, std::size_t scored_count, std::size_t staged_count,
                                std::size_t expansion_factor, stopping_type_ const& stopping, context_type_& context,
                                fold_type_&& fold) const noexcept {
        std::size_t const fresh_count = staged_count - scored_count;
        context.staged_distances.resize_reserved(staged_count);
        std::span<distance_t> const fresh_distances {context.staged_distances.data() + scored_count, fresh_count};
        metric.drain(fresh_distances, round_radius_(expansion_factor, stopping));
        context.telemetry.computed_distances += fresh_count;
        walk_demand_t const demand = fold.absorb(
            std::span<node_id_t const>(context.staged_ids).subspan(scored_count, fresh_count),
            std::span<distance_t const>(fresh_distances));
        settle_fold(fold);
        return demand;
    }

    /** One round's scoring half: scores the staged IDs past the @p scored_count already scored, and
     *  pushes what improved the candidates into the frontier, shedding the frontier past the full
     *  candidates' radius first so the pushes fit. */
    template <typename metric_type_, typename stopping_type_, typename context_type_, typename fold_type_>
    walk_step_t rank_round_(metric_type_& metric, std::size_t scored_count, std::size_t expansion_factor,
                            stopping_type_ const& stopping, context_type_& context, fold_type_&& fold) const noexcept {
        std::size_t const staged_count = context.staged_ids.size();
        if (!staged_count) return walk_step_t::advanced_k;
        if (staged_count != scored_count && score_staged_(metric, scored_count, staged_count, expansion_factor,
                                                          stopping, context, fold) == walk_demand_t::sated_k)
            return walk_step_t::sated_k;
        if (context.results.size() == expansion_factor &&
            context.frontier.capacity() - context.frontier.size() < staged_count)
            context.frontier.erase_below(context.results.top());
        for (std::size_t lane = 0; lane != staged_count; ++lane) {
            scored_node_t const scored {context.staged_ids[lane], context.staged_distances[lane]};
            if (!context.results.insert(scored_node_t {scored}, expansion_factor)) continue;
            if (failed(context.frontier.insert(scored_node_t {scored}))) {
                ++context.telemetry.saturated_walks;
                return walk_step_t::saturated_k;
            }
        }
        return walk_step_t::advanced_k;
    }

    /** Wires the reverse edges of the freshly wired node before its batch's turn, walking the links
     *  snapshotted in @c context.reverse_ids: per neighbor - lock, append when room remains at zero
     *  metric cost; otherwise copy the full neighbor list, unlock, score the copy and the appended
     *  node off-lock in as many rounds as the metric's staging takes, re-select through the
     *  selector, relock, and apply only if the list is still the one copied - a lost race starts
     *  over from the lock, so the appended node competes for every list it selected. A node already
     *  linking back, as a refined node's neighbor may, is left as it is. Only a refused allocation,
     *  or a metric refusing an empty round, gives a link up, counted as a saturated walk. */
    template <synchronization_t synchronization_, typename metric_type_, typename selector_type_,
              typename context_type_>
    void back_link_(node_id_t new_node_id, metric_type_& metric, selector_type_ const& selector,
                    context_type_& context) noexcept {
        for (std::size_t reverse = 0; reverse != context.reverse_ids.size(); ++reverse)
            while (!back_link_one_<synchronization_>(context.reverse_ids[reverse], new_node_id, metric, selector,
                                                     context)) {}
    }

    /** One reverse link of @c back_link_ into @p neighbor_id 's neighbor list; @c false when the
     *  list raced the re-selection and must be tried again from the lock. */
    template <synchronization_t synchronization_, typename metric_type_, typename selector_type_,
              typename context_type_>
    bool back_link_one_(node_id_t neighbor_id, node_id_t new_node_id, metric_type_& metric,
                        selector_type_ const& selector, context_type_& context) noexcept {
        constexpr bool locked_k = synchronization_ == synchronization_t::concurrent_k;
        std::size_t const connectivity = selector.connectivity;
        value_t const& new_value = published_value_of(new_node_id);
        node_t& node = nodes_[neighbor_id];
        if constexpr (locked_k) node.lock();
        bool linked_back = false;
        for (node_id_t const linked : node.neighbors) linked_back = linked_back || linked == new_node_id;
        std::size_t const degree = node.neighbors.size();
        if (linked_back || degree < connectivity) {
            bool const grown = linked_back || succeeded(node.neighbors.reserve(degree + 1));
            if (!linked_back && grown) node.neighbors.push_back_reserved(new_node_id);
            if constexpr (locked_k) node.unlock();
            context.telemetry.saturated_walks += !grown;
            return true;
        }
        context.staged_ids.clear();
        bool const staged = succeeded(context.staged_ids.reserve(degree));
        for (std::size_t position = 0; staged && position != degree; ++position)
            context.staged_ids.push_back_reserved(node.neighbors[position]);
        if constexpr (locked_k) node.unlock();
        if (!staged || failed(context.staged_distances.reserve(degree + 1)) ||
            failed(context.results.reserve(degree + 1))) {
            ++context.telemetry.saturated_walks;
            return true;
        }

        // Off-lock: candidates = the copied neighbor list plus the appended node, scored against
        // the neighbor's own stored value; every member is ranked, so nothing is too far.
        context.staged_distances.resize_reserved(degree + 1);
        std::size_t scored_count = 0;
        while (scored_count != degree + 1) {
            metric.reset(published_value_of(neighbor_id));
            std::size_t staged_count = 0;
            for (std::size_t member = scored_count; member != degree + 1; ++member, ++staged_count)
                if (metric.push(member == degree ? new_value : published_value_of(context.staged_ids[member])) ==
                    metric_staging_t::refused_k)
                    break;
            if (!staged_count) break;
            metric.drain({context.staged_distances.data() + scored_count, staged_count},
                         unbounded_radius<distance_t>());
            context.telemetry.computed_distances += staged_count;
            context.telemetry.computed_distances_in_reverse_refines += staged_count;
            scored_count += staged_count;
        }
        if (scored_count != degree + 1) {
            ++context.telemetry.saturated_walks;
            return true;
        }

        context.results.clear();
        for (std::size_t position = 0; position != degree; ++position)
            context.results.insert(scored_node_t {context.staged_ids[position], context.staged_distances[position]},
                                   degree + 1);
        context.results.insert(scored_node_t {new_node_id, context.staged_distances[degree]}, degree + 1);
        std::size_t const refines_before = context.telemetry.computed_distances_in_refines;
        selector(*this, metric, context);
        fill_selection(context, connectivity);
        // The selector counts a refine; re-selecting a back-linked neighbor list is a reverse
        // refine.
        std::size_t const reselecting = context.telemetry.computed_distances_in_refines - refines_before;
        context.telemetry.computed_distances_in_refines = refines_before;
        context.telemetry.computed_distances_in_reverse_refines += reselecting;

        // Relock & revalidate: apply the re-selection only onto the exact neighbor list it was
        // computed from; a raced list goes round again.
        if constexpr (locked_k) node.lock();
        bool unchanged = node.neighbors.size() == degree;
        for (std::size_t position = 0; position != degree && unchanged; ++position)
            unchanged = node.neighbors[position] == context.staged_ids[position];
        if (unchanged) {
            node.neighbors.clear();
            std::size_t const kept_count = min_of(context.selected_neighbors.size(), connectivity);
            for (std::size_t kept = 0; kept != kept_count; ++kept)
                node.neighbors.push_back_reserved(context.selected_neighbors[kept].node_id);
        }
        if constexpr (locked_k) node.unlock();
        return unchanged;
    }

    /** The watermark at unique acquisition; @c contended_k when the dispenser has run ahead
     *  of the watermark, which only an exclusive-marked mutator lying mid-turn can do with
     *  the gate held. */
    expected<node_id_t> watermark_() const noexcept {
        node_id_t const visible = atomic_ref<node_id_t const>(last_visible_id_).load(std::memory_order_acquire);
        return {visible, visible == atomic_ref<node_id_t const>(next_dispensed_id_).load(std::memory_order_acquire) //
                             ? status_t::success_k
                             : status_t::contended_k};
    }

    /** Refuses a column of @p size that is not exactly @p wanted elements: a short one
     *  overflows, a long one disagrees about the node count. For the three node-sized columns,
     *  whose length is what names the node count. */
    static status_t sized_exactly_(std::size_t size, std::size_t wanted) noexcept {
        if (size < wanted) return status_t::overflow_k;
        if (size > wanted) return status_t::config_mismatch_k;
        return status_t::success_k;
    }

    /** Refuses a column of @p size shorter than @p wanted elements. For the two bulk
     *  columns, whose content the prefix sums delimit, so spare room is unused rather than
     *  wrong - and a graph that sheds edges between sizing and planning still saves. */
    static status_t sized_at_least_(std::size_t size, std::size_t wanted) noexcept {
        return size < wanted ? status_t::overflow_k : status_t::success_k;
    }

    /** Whether @p sums start at zero, never fall, never pass @p total, and end at it. */
    static status_t monotone_sums_(std::span<u64_t const> sums, std::size_t total) noexcept {
        if (sums.empty() || sums.front() != 0) return status_t::malformed_k;
        u64_t previous = 0;
        for (u64_t const next : sums.subspan(1)) {
            if (next < previous || next > total) return status_t::malformed_k;
            previous = next;
        }
        return previous == total ? status_t::success_k : status_t::malformed_k;
    }

    /** Vets untrusted columns in full - both prefix-sum columns sized to the nodes and
     *  monotone over their element columns, every neighbor ID naming a node - so the task
     *  decoders trust every slice they read. */
    static status_t vetted_columns_(columns_view_t columns) noexcept {
        std::size_t const node_count = columns.keys.size();
        if (columns.neighbor_offsets.size() != node_count + 1 || columns.value_offsets.size() != node_count + 1)
            return status_t::malformed_k;
        if (status_t const sums = monotone_sums_(columns.neighbor_offsets, columns.neighbor_ids.size()); failed(sums))
            return sums;
        if (status_t const sums = monotone_sums_(columns.value_offsets, columns.values.size()); failed(sums))
            return sums;
        for (node_id_t const neighbor : columns.neighbor_ids)
            if (static_cast<std::size_t>(neighbor) >= node_count) return status_t::malformed_k;
        return status_t::success_k;
    }

    /** The node records, covered by each claim and by @c reserve, addresses stable forever. */
    nodes_column_t nodes_;

    /** Batch capacity claims - the all-or-nothing admission, separate from identity: claims
     *  convert one-to-one into dispensed IDs, so dispensing never overflows, and a refused cover
     *  gives its claim back before anything is dispensed. The bounded add commits only within
     *  @c ceiling_nodes_, so a rejected batch never writes and the count never overshoots. */
    alignas(default_alignment_k) std::size_t claimed_count_ {0};

    /** The most the claims may reach, stored by @c reserve; on the claim word's line, which every
     *  claim reading it already holds. */
    std::size_t ceiling_nodes_ {max_node_count_k};

    /** Node-ID dispenser; one add hands a batch its whole range before the first walk.
     *  Its per-batch RMW gets its own line, so it never invalidates the watermark line
     *  every search snapshots. */
    alignas(default_alignment_k) node_id_t next_dispensed_id_ {0};

    /** The visibility watermark: every ID strictly below it is published. */
    alignas(default_alignment_k) node_id_t last_visible_id_ {0};

    /** What @ref connectivity answers, only ever raised through @c widen_; beside the watermark,
     *  since every search reads both and this one stops moving once the widest list of the graph
     *  reaches its budget. */
    std::size_t widest_list_ {0};

    /** Key → ID multimap placed by node ID, covered beside the records, and written only by
     *  whoever publishes a range: a batch's turn holder, a plan's barrier or a load's finish. */
    key_map_t key_map_;

    /** The world gate: mutators hold it shared, stop-the-world verbs try-unique, marked
     *  readers probe it wait-free. On its own line, away from the watermark, which turn loops
     *  spin on. */
    alignas(default_alignment_k) shared_mutex_t shared_mutex_;

    [[no_unique_address]] allocator_t allocator_;

    prefetcher_t prefetcher_;

#pragma endregion Internals and State
};

/**
 *  @brief External reader's cursor over the watermark of one @c basic::ram_index:
 *      ID-order iteration, random seeks, and edge-following through a privately staged
 *      copy of the current node's neighbor list.
 *
 *  Strictly read-only under every contract, so readers never stall writers. One watermark snapshot,
 *  captured at construction and widened by @c refresh, bounds everything: keys & values below it
 *  are immutable and read wait-free, neighbor lists are staged under the node lock on the
 *  concurrent mode and as plain copies on the marked ones. A staged list may name IDs at or past
 *  this cursor's snapshot - fresh back-links - and @c follow refuses those with @c malformed_k
 *  until a @c refresh admits them. The cursor starts past the end; @c seek(0) enters the graph.
 */
template <typename index_type_, synchronization_t synchronization_, typename allocator_type_>
struct ram_cursor {
    using index_t = index_type_;
    template <typename word_type_>
    using atomic_ref = typename index_t::template atomic_ref<word_type_>;
    using node_id_t = typename index_t::node_id_t;
    using key_t = typename index_t::key_t;
    using value_t = typename index_t::value_t;
    using allocator_t = allocator_type_;

    ram_cursor(ram_cursor const&) = delete;
    ram_cursor& operator=(ram_cursor const&) = delete;
    ram_cursor(ram_cursor&&) noexcept = default;
    ram_cursor& operator=(ram_cursor&&) = delete;

    /** Binds the index & captures the watermark snapshot; allocates nothing - the
     *  staged-list buffer grows on the first @c seek through @p allocator. */
    explicit ram_cursor(index_t& index, allocator_t allocator = {}) noexcept
        : index_(index), staged_neighbors_(allocator),
          visible_count_(atomic_ref<node_id_t const>(index.last_visible_id_).load(watermark_order_k)),
          current_(visible_count_) {}

    /** IDs strictly below this snapshot are readable through this cursor. */
    node_id_t visible_count() const noexcept { return visible_count_; }

    /** Widens the snapshot to admit nodes published since; a cursor standing past the
     *  end stays past the end - entering the admitted region takes an explicit @c seek. */
    void refresh() noexcept {
        bool const past_end = current_ == visible_count_;
        visible_count_ = atomic_ref<node_id_t const>(index_.last_visible_id_).load(watermark_order_k);
        if (past_end) current_ = visible_count_;
    }

    /** Whether the cursor stands on a readable node. */
    explicit operator bool() const noexcept { return current_ < visible_count_; }

    node_id_t id() const noexcept { return current_; }
    key_t key() const noexcept { return index_.nodes_[current_].key; }
    value_t const& value() const noexcept { return index_.nodes_[current_].value; }
    std::span<node_id_t const> neighbors() const noexcept {
        return {staged_neighbors_.begin(), staged_neighbors_.end()};
    }

    /**
     *  Positions on @p node & stages its neighbor list, the cursor staying where it was on either
     *  refusal.
     *
     *  @par Statuses:
     *  - @c malformed_k @p node is at or past the snapshot.
     */
    status_t seek(node_id_t node) noexcept {
        if (static_cast<std::size_t>(node) >= static_cast<std::size_t>(visible_count_)) return status_t::malformed_k;
        typename index_t::node_t& record = index_.nodes_[node];
        if constexpr (synchronization_ == synchronization_t::concurrent_k) record.lock();
        std::size_t const degree = record.neighbors.size();
        status_t const staged = staged_neighbors_.reserve(degree);
        if (succeeded(staged)) {
            staged_neighbors_.clear();
            for (std::size_t link = 0; link != degree; ++link)
                staged_neighbors_.push_back_reserved(record.neighbors[link]);
        }
        if constexpr (synchronization_ == synchronization_t::concurrent_k) record.unlock();
        if (failed(staged)) return staged;
        current_ = node;
        return status_t::success_k;
    }

    /** Advances to the next published ID, staging its neighbor list; past the last one the cursor
     *  turns false. A staging refusal parks on the new ID with an empty list - a later @c seek to
     *  the same ID retries the staging. */
    ram_cursor& operator++() noexcept {
        node_id_t const next = current_ + 1;
        if (static_cast<std::size_t>(next) >= static_cast<std::size_t>(visible_count_)) {
            current_ = visible_count_;
            staged_neighbors_.clear();
            return *this;
        }
        if (failed(seek(next))) {
            current_ = next;
            staged_neighbors_.clear();
        }
        return *this;
    }

    /** Deleted: it would copy the staged list - an allocation in disguise. */
    ram_cursor operator++(int) = delete;

    /**
     *  Hops to the staged list's @p position -th neighbor.
     *
     *  @par Statuses:
     *  - @c malformed_k @p position is past the staged degree.
     *  - @c malformed_k The neighbor sits at or past the snapshot.
     */
    status_t follow(std::size_t position) noexcept {
        if (position >= staged_neighbors_.size()) return status_t::malformed_k;
        return seek(staged_neighbors_[position]);
    }

  private:

    /** Certified contracts already ordered every publication, so no acquire. */
    inline static constexpr std::memory_order watermark_order_k = synchronization_ == synchronization_t::concurrent_k
                                                                      ? std::memory_order_acquire
                                                                      : std::memory_order_relaxed;

    index_t& index_;

    /** The private copy of the current node's neighbor list - @c follow's ground truth. */
    growing_buffer<node_id_t, allocator_t> staged_neighbors_;

    /** This cursor's watermark snapshot; IDs strictly below it are readable. */
    node_id_t visible_count_;

    /** The current ID, or @c visible_count_ when past the end. */
    node_id_t current_;
};

} // namespace unum::usearch::basic
