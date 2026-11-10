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
#include <cstring> // `std::memset` - the barrier between rounds empties the next round's picks

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
#include "usearch/utilities/ram_sorts.hpp" // `partial_sort` - a tile's proposal runs
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
            metric.reset(queries[query]);
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
 *  the farthest one's distance from the query. The slots the rule leaves empty take the nearest
 *  rejected candidates, after the diverse picks. */
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
        // A refused growth leaves the backfill alone to fill the neighbor list.
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

        // The kept picks are a subsequence of the ascending candidates, so one merge-walk tells a
        // kept candidate from a rejected one.
        std::size_t const diverse_count = selected.size();
        for (std::size_t position = 0, matched = 0;
             position != context.results.size() && selected.size() != connectivity; ++position) {
            if (matched != diverse_count && selected[matched].node_id == context.results[position].node_id) ++matched;
            else selected.push_back_reserved(context.results[position]);
        }
    }
};

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

    /** A refine walk's picks, copied for the merge that writes them, which reuses the selection. */
    scratch_buffer<scored_node_t> picks;

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
          admitted(scored_allocator_t(allocator)), hinted_ids(allocator), picks(allocator), hinted_held(allocator),
          allocator(allocator) {}
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

    /** The chunks spanning @c max_node_count_k; the first one's nodes are also one plan step's. */
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
     *  add dispenses the batch's whole ID range, one step places it, and @c finish publishes. Per
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
        static_assert(synchronization_ != synchronization_t::immutable_k,
                      "Certified immutability contradicts insertion - certify exclusive_k or stay concurrent_k");
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
        // The one step above placed every position, so the finish cannot refuse.
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
     *  range holds every later turn until then, so a caller fanning the steps out over threads
     *  places a refused fan-out on the calling thread, and nothing cancels a plan once dispensed. A
     *  plan that dies unfinished only asserts, since publishing its range would expose nodes nobody
     *  wrote.
     */
    struct [[nodiscard]] insert_plan_t {

        /** The IDs the plan dispensed, one per position, in position order. */
        iota_view<node_id_t> range {};

        /** Positions the steps placed so far, mutated through @c atomic_ref. */
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
            static_assert(synchronization_ != synchronization_t::immutable_k,
                          "Certified immutability contradicts insertion - certify exclusive_k or stay concurrent_k");
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
            // Releases this step's nodes to the acquire in `finish`, whose turn carries them on.
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

    /** Sizes @p context for walks under @p connectivity and @p config - the fallible half of an
     *  insertion, hoisted ahead of @ref plan_insert since no step may allocate it. The candidates,
     *  the staging surfaces and the selections never allocate again, and the frontier holds the
     *  full candidates' unexpanded members plus one round's admissions, which shedding past their
     *  radius keeps it within, so only ties at that radius grow it. The visited set starts at one
     *  neighbor list per candidate and alone re-tables as a walk widens - once per doubling per
     *  context lifetime, which a context kept across walks amortizes. Idempotent; a zero
     *  @p connectivity answers @c config_mismatch_k.
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
                      "Certified immutability contradicts wiring - certify exclusive_k or stay concurrent_k");
        constexpr bool locked_k = synchronization_ == synchronization_t::concurrent_k;
        std::size_t const visible = size();
        if (static_cast<std::size_t>(from) >= visible || static_cast<std::size_t>(to) >= visible || from == to)
            return {membership_t::absent_k, status_t::malformed_k};

        // Enters the mutator class: manual wiring never moves the watermark, so the shared
        // mutex is what makes it visible to stop-the-world verbs & the certified probes.
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
                      "Certified immutability contradicts wiring - certify exclusive_k or stay concurrent_k");
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

    /** Optional tuning knobs of a refine, an absorb or a split's passes; zeros mean "derive from
     *  the selector's budget". */
    struct refine_config_t {

        /** How many candidates a re-selection walk keeps, as an insertion's. */
        std::size_t expansion_factor = 0;

        /** Bytes the whole pass may hold beyond the index: the audit and a bit per node carried
         *  across tiles, then tiles of contiguous nodes at 8 B per neighbor slot plus 4 B each, as
         *  many as fit. Zero walks one storage chunk per tile. */
        std::size_t memory_limit = 0;
    };

    /** A split's knobs: its refine's, and the seed that orders the parts' growth. */
    struct split_config_t {
        refine_config_t refine {};
        u64_t seed = 0;
    };

    /**
     *  @brief One reachability audit in flight: a sweep from node zero over the published nodes,
     *      forward along the neighbor lists and back against them, one level per sweep of every
     *      step, until a level settles nothing.
     *
     *  Read-only on the graph, so concurrent insertions merely leave fresh nodes uncounted. Steps
     *  are disjoint first-chunk-sized ID ranges, so any number of threads may @ref sweep different
     *  steps of one level at once; @ref advance_level is the level boundary, one caller. Reached
     *  bits only turn on and a node expands once, so the sweep converges to the exact reachable
     *  set. A node reaches back once a neighbor in its list does, so no reverse adjacency is built;
     *  every node both reached and reaching back is reached from any other such node.
     */
    template <typename bits_allocator_type_>
    struct health_plan {

        /** One unit of a level's fan-out. */
        struct step_t {

            /** The step's position in the plan - what a fan-out hands to a thread. */
            std::size_t position = 0;

            /** The IDs this step sweeps. */
            iota_view<node_id_t> nodes;
        };

        /** The audit's bitsets, one run of words each in @c bits_, in this order. */
        enum bitset_t : std::size_t {

            /** Reached from node zero. */
            reached_k,

            /** Linked from a node settled before the level that reached it. */
            linked_once_k,

            /** Linked from two such nodes, so one link may go to a reconnection. */
            linked_twice_k,

            /** Its neighbors already swept. */
            expanded_k,

            /** Reached before the current level began. */
            settled_k,

            /** Reaches node zero through some neighbor in its list. */
            reaches_back_k,

            /** Reached node zero back before the current level began. */
            back_settled_k,

            /** Linked to two nodes that reached back before it did, so one link may go to a
             *  reconnection. */
            back_linked_twice_k,

            /** The bitsets in all. */
            bitsets_k,
        };

        using bits_t = growing_buffer<u1x64_t, bits_allocator_type_>;
        using levels_allocator_t = typename std::allocator_traits<bits_allocator_type_>::template rebind_alloc<u8_t>;
        using levels_t = growing_buffer<u8_t, levels_allocator_t>;

        /** Lanes one word carries - what an ID splits on. */
        inline static constexpr std::size_t lanes_k = std::numeric_limits<u64_t>::digits;

        /** The level of a node no sweep reached, 255; the deepest a reached node records is 254,
         *  at which the levels saturate: every node reached from there on reads 254 and counts no
         *  parents, so no merge drops a link for a level it cannot tell. */
        inline static constexpr u8_t unknown_level_k = (std::numeric_limits<u8_t>::max)();

        /** The audit's relation for a repair walk: admits the nodes reached from node zero and
         *  reaching it back. */
        struct membership_t {
            health_plan const& audit;

            template <std::size_t extent_>
            void operator()(std::span<node_id_t const, extent_> node_ids,
                            std::span<admission_t, extent_> admissions) const noexcept {
                for (std::size_t lane = 0; lane != extent_; ++lane)
                    admissions[lane] = audit.reached(node_ids[lane]) && audit.reaches_back(node_ids[lane])
                                           ? admission_t::admit_k
                                           : admission_t::reject_k;
            }
        };

        /** The watermark the audit covers; IDs strictly below it are swept. */
        node_id_t visible = 0;

        /** An unbound plan - a refused audit's answer; owning bitsets, it names an allocator. */
        explicit health_plan(bits_allocator_type_ const& allocator) noexcept
            : bits_(allocator), levels_(levels_allocator_t {allocator}), back_levels_(levels_allocator_t {allocator}),
              parents_(levels_allocator_t {allocator}) {}
        health_plan(ram_index& index, node_id_t visible, bits_t&& bits, levels_t&& levels, levels_t&& back_levels,
                    levels_t&& parents) noexcept
            : visible(visible), index_(&index), bits_(std::move(bits)), levels_(std::move(levels)),
              back_levels_(std::move(back_levels)), parents_(std::move(parents)) {
            restart_sweep();
        }
        health_plan(health_plan&& other) noexcept
            : visible(other.visible), index_(std::exchange(other.index_, nullptr)), bits_(std::move(other.bits_)),
              levels_(std::move(other.levels_)), back_levels_(std::move(other.back_levels_)),
              parents_(std::move(other.parents_)), expanded_in_level_(other.expanded_in_level_),
              reached_back_in_level_(other.reached_back_in_level_), sweeps_(other.sweeps_),
              back_sweeps_(other.back_sweeps_), level_(other.level_) {}
        health_plan& operator=(health_plan&& other) noexcept {
            visible = other.visible;
            index_ = std::exchange(other.index_, nullptr);
            bits_ = std::move(other.bits_);
            levels_ = std::move(other.levels_);
            back_levels_ = std::move(other.back_levels_);
            parents_ = std::move(other.parents_);
            expanded_in_level_ = other.expanded_in_level_;
            reached_back_in_level_ = other.reached_back_in_level_;
            sweeps_ = other.sweeps_;
            back_sweeps_ = other.back_sweeps_;
            level_ = other.level_;
            return *this;
        }
        health_plan(health_plan const&) = delete;
        health_plan& operator=(health_plan const&) = delete;

        /** Step @p position of the plan, the nodes of one bitset word, since @ref sweep writes its
         *  expanded bits without atomics; an empty step past the end. */
        step_t operator[](std::size_t position) const noexcept {
            step_t step;
            if (position >= steps().size()) return step;
            std::size_t const first = position * lanes_k;
            std::size_t const bound = min_of(first + lanes_k, static_cast<std::size_t>(visible));
            step.position = position;
            step.nodes = {static_cast<node_id_t>(first), static_cast<node_id_t>(bound)};
            return step;
        }

        /** The plan's steps, positionally, one per bitset word; every level sweeps all of them. */
        indexed_range<health_plan> steps() const noexcept {
            return {this, index_ ? divide_round_up<std::size_t>(visible, lanes_k) : 0u};
        }

        /** Turns the audit onto @p index 's IDs below @p visible, which its bitsets must span, and
         *  restarts the sweep. */
        void rebind(ram_index& index, node_id_t visible) noexcept {
            index_ = &index;
            this->visible = visible;
            restart_sweep();
        }

        /** Takes the IDs up to @p visible, which its bitsets must span, in without a sweep: the
         *  nodes appended since read unreached both ways, at no level, with no parents. */
        void widen(node_id_t visible) noexcept {
            if (visible <= this->visible) return;
            std::size_t const added = visible - this->visible;
            std::memset(levels_.data() + this->visible, unknown_level_k, added);
            std::memset(back_levels_.data() + this->visible, unknown_level_k, added);
            std::memset(parents_.data() + this->visible, 0, added);
            this->visible = visible;
        }

        /** Clears the bitsets and the levels and marks node zero both ways at level zero, so the
         *  next @ref sweep starts level one. */
        void restart_sweep() noexcept {
            for (u1x64_t& lanes : bits_) lanes = 0;
            expanded_in_level_ = 0;
            reached_back_in_level_ = 0;
            sweeps_ = 0;
            back_sweeps_ = 0;
            level_ = 0;
            if (!visible) return;
            std::memset(levels_.data(), unknown_level_k, visible);
            std::memset(back_levels_.data(), unknown_level_k, visible);
            std::memset(parents_.data(), 0, visible);
            insert_(reached_k, node_id_t {0});
            insert_(reaches_back_k, node_id_t {0});
            levels_[0] = 0;
            back_levels_[0] = 0;
            snapshot_levels_();
        }

        /**
         *  @brief Expands every node of @p step settled before this level and not yet expanded,
         *      its neighbors below the watermark turning reached at the next level, and marks
         *      every node of it a neighbor of which reached back before this level.
         *  @return The nodes this call expanded or found reaching back.
         *
         *  Safe on any thread for disjoint steps - the expanded bits are the step's own, the
         *  reached bits and the levels are set through relaxed atomics, and the settled bits stay
         *  fixed for the level. Every reached node's level is one past that of the settled nodes
         *  linking to it, its parents, which the merge keeps one of.
         */
        std::size_t sweep(step_t const& step) noexcept {
            std::size_t expanded_count = 0, reached_back_count = 0;
            u8_t const next_level = static_cast<u8_t>(min_of<std::size_t>(level_ + 1, unknown_level_k - 1));
            for (node_id_t node_id = step.nodes.first; node_id < step.nodes.bound; ++node_id) {
                bool const expanding = contains_(settled_k, node_id) && !contains_(expanded_k, node_id);
                bool const pulling = !contains_(reaches_back_k, node_id);
                if (!expanding && !pulling) continue;
                // Bits only, never a metric verb, so the node lock spans the whole neighbor list.
                node_t& node = index_->nodes_[node_id];
                node.lock();
                if (expanding) {
                    insert_exclusive_(expanded_k, node_id);
                    ++expanded_count;
                    // Only settled nodes expand, so every link to a target the level reaches
                    // counts.
                    for (node_id_t const target : node.neighbors) {
                        if (target >= visible || target == node_id || contains_(settled_k, target)) continue;
                        if (!insert_(reached_k, target))
                            atomic_ref<u8_t>(levels_[target]).store(next_level, std::memory_order_relaxed);
                        if (insert_(linked_once_k, target) && !contains_(linked_twice_k, target))
                            insert_(linked_twice_k, target);
                        // A level later than the first link's reads stale as unknown and counts
                        // no parent, nor does the saturated level: an undercount keeps a link,
                        // never drops one.
                        if (level_of_(levels_, target) == next_level && next_level != unknown_level_k - 1)
                            count_parent_(target);
                    }
                }
                if (pulling) {
                    std::size_t settled_neighbors = 0;
                    for (node_id_t const target : node.neighbors)
                        settled_neighbors += target < visible && target != node_id && contains_(back_settled_k, target);
                    if (settled_neighbors) {
                        insert_(reaches_back_k, node_id);
                        back_levels_[node_id] = next_level;
                        ++reached_back_count;
                        if (settled_neighbors > 1) insert_exclusive_(back_linked_twice_k, node_id);
                    }
                }
                node.unlock();
            }
            atomic_ref<std::size_t>(expanded_in_level_).store_add(expanded_count, std::memory_order_relaxed);
            atomic_ref<std::size_t>(reached_back_in_level_).store_add(reached_back_count, std::memory_order_relaxed);
            return expanded_count + reached_back_count;
        }

        /** The level boundary, one caller after every step swept: counts the level in each
         *  direction that moved, settles what it reached, and returns what it reached; zero means
         *  the sweep converged. */
        std::size_t advance_level() noexcept {
            std::size_t const expanded = std::exchange(expanded_in_level_, 0);
            std::size_t const reached_back = std::exchange(reached_back_in_level_, 0);
            sweeps_ += expanded != 0;
            back_sweeps_ += reached_back != 0;
            if (expanded || reached_back) snapshot_levels_();
            ++level_;
            return expanded + reached_back;
        }

        /** The level the sweep reached @p node_id at, @c unknown_level_k for none, which any thread
         *  may read between levels. */
        u8_t level(node_id_t node_id) const noexcept { return levels_[node_id]; }

        /** The level @p node_id reached node zero back at, @c unknown_level_k for none, which any
         *  thread may read between levels. */
        u8_t back_level(node_id_t node_id) const noexcept { return back_levels_[node_id]; }

        /** Takes one of @p node_id 's links from the level before it for a merge that drops
         *  it, answering whether another remains; a refusal leaves it untouched, so no node
         *  loses the last parent it is reached through. A count saturated at 255 refuses
         *  early, never late. */
        bool release_parent(node_id_t node_id) noexcept {
            atomic_ref<u8_t> const parents {parents_[node_id]};
            u8_t observed = parents.load(std::memory_order_relaxed);
            while (observed > 1 &&
                   !parents.compare_exchange_weak(observed, static_cast<u8_t>(observed - 1), std::memory_order_relaxed,
                                                  std::memory_order_relaxed)) {}
            return observed > 1;
        }

        /** Whether @p node_id was reached - readable by any thread between levels. */
        bool reached(node_id_t node_id) const noexcept { return contains_(reached_k, node_id); }

        /** Whether @p node_id reaches node zero - readable by any thread between levels. */
        bool reaches_back(node_id_t node_id) const noexcept { return contains_(reaches_back_k, node_id); }

        /** Takes one of @p node_id 's in-links for a reconnection, answering whether one from a
         *  node settled a level earlier remains; a refusal leaves it untouched. Sound by induction
         *  on levels: the kept link's source keeps a link from an earlier level in turn. */
        bool release_link(node_id_t node_id) noexcept { return release_(linked_twice_k, node_id); }

        /** Takes one of @p node_id 's out-links for a reconnection, answering whether its list
         *  still reaches back through a node that did a level earlier; a node with no way back has
         *  none to lose. */
        bool release_out_link(node_id_t node_id) noexcept {
            return !contains_(reaches_back_k, node_id) || release_(back_linked_twice_k, node_id);
        }

        /**
         *  The report: the nodes reached and reaching back, and the levels each direction took.
         *  @c internal_k when a level is still pending, since the count would then be a lower
         *  bound.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The audit is bound to no index.
         */
        expected<reachability_t> finish() const noexcept {
            if (!index_) return {{}, status_t::uninitialized_k};
            if (expanded_in_level_ || reached_back_in_level_) return {{}, status_t::internal_k};
            reachability_t result;
            result.audited_nodes = visible;
            std::size_t const words = words_();
            for (std::size_t word = 0; word != words; ++word)
                result.reachable_nodes += static_cast<std::size_t>(
                    std::popcount(bits_[reached_k * words + word] & bits_[reaches_back_k * words + word]));
            result.sweeps = sweeps_;
            result.back_sweeps = back_sweeps_;
            return {result, status_t::success_k};
        }

      private:

        /** Words per bitset. */
        std::size_t words_() const noexcept { return bits_.size() / bitsets_k; }

        bool contains_(bitset_t bitset, node_id_t node_id) const noexcept {
            u1x64_t const lanes =
                atomic_ref<u1x64_t const>(bits_[bitset * words_() + node_id / lanes_k]).load(std::memory_order_relaxed);
            return (lanes >> (node_id % lanes_k)) & 1u;
        }

        /** Sets @p node_id 's bit and answers whether it was already set. */
        bool insert_(bitset_t bitset, node_id_t node_id) noexcept {
            u1x64_t const lane = u1x64_t {1} << (node_id % lanes_k);
            return atomic_ref<u1x64_t>(bits_[bitset * words_() + node_id / lanes_k])
                       .fetch_or(lane, std::memory_order_relaxed) &
                   lane;
        }
        void insert_exclusive_(bitset_t bitset, node_id_t node_id) noexcept {
            bits_[bitset * words_() + node_id / lanes_k] |= u1x64_t {1} << (node_id % lanes_k);
        }

        /** Clears @p node_id 's bit and answers whether it was set. */
        bool release_(bitset_t bitset, node_id_t node_id) noexcept {
            u1x64_t const lane = u1x64_t {1} << (node_id % lanes_k);
            return atomic_ref<u1x64_t>(bits_[bitset * words_() + node_id / lanes_k])
                       .fetch_and(~lane, std::memory_order_relaxed) &
                   lane;
        }

        /** What the next level starts from: the nodes settled each way so far. */
        void snapshot_levels_() noexcept {
            std::size_t const words = words_();
            std::memcpy(bits_.data() + settled_k * words, bits_.data() + reached_k * words, words * sizeof(u1x64_t));
            std::memcpy(bits_.data() + back_settled_k * words, bits_.data() + reaches_back_k * words,
                        words * sizeof(u1x64_t));
        }

        static u8_t level_of_(levels_t const& levels, node_id_t node_id) noexcept {
            return atomic_ref<u8_t const>(levels[node_id]).load(std::memory_order_relaxed);
        }

        /** Counts one more link into @p node_id from the level before it, saturating. */
        void count_parent_(node_id_t node_id) noexcept {
            atomic_ref<u8_t> const parents {parents_[node_id]};
            u8_t observed = parents.load(std::memory_order_relaxed);
            while (observed != (std::numeric_limits<u8_t>::max)() &&
                   !parents.compare_exchange_weak(observed, static_cast<u8_t>(observed + 1), std::memory_order_relaxed,
                                                  std::memory_order_relaxed)) {}
        }

        /** The audited index; null on a moved-from plan. */
        ram_index* index_ = nullptr;

        /** Every bitset, one bit per swept node each, laid out as @ref bitset_t orders them. */
        bits_t bits_;

        /** Per swept node, the level the sweep reached it at. */
        levels_t levels_;

        /** Per swept node, the level it reached node zero back at. */
        levels_t back_levels_;

        /** Per swept node, its links from the level before it, saturating. */
        levels_t parents_;

        /** Nodes the current level expanded so far, summed through @c atomic_ref. */
        std::size_t expanded_in_level_ = 0;

        /** Nodes the current level found reaching back so far, summed through @c atomic_ref. */
        std::size_t reached_back_in_level_ = 0;

        /** Levels since the last @ref restart_sweep that expanded a node. */
        std::size_t sweeps_ = 0;

        /** Levels since the last @ref restart_sweep that found a node reaching back. */
        std::size_t back_sweeps_ = 0;

        /** The level the next sweep runs. */
        std::size_t level_ = 0;
    };

    /* The audit a refine embeds, its bitsets on the index's own allocator. */
    using health_plan_t = health_plan<allocator_t>;

    /** Opens a reachability audit over the published nodes, from node zero both ways, so the
     *  nodes it reaches both ways are what a walk from any of them reaches. The caller sweeps
     *  every step per level until a level marks nothing, then reads @c finish(). The bitsets come
     *  from @p context 's allocator. */
    template <typename context_allocator_type_>
    expected<health_plan<context_allocator_type_>> plan_health(
        traversal_context_over<context_allocator_type_>& context) noexcept {
        return plan_audit_(static_cast<node_id_t>(size()), context.allocator);
    }

    /**
     *  @brief One refinement pass in flight, from the first audit to the verifying one.
     *
     *  An audit records every node's levels both ways, the planned range walks one tile of nodes
     *  at a time, each node merging its picks into its list at once, and a sweep collects every
     *  link the walked nodes hold into one tile of target nodes at a time, by two scans of their
     *  lists, merging each target with the nodes linking to it, the levels keeping every merged
     *  node reached; a second audit verifies, and the islands left are re-selected and
     *  reconnected round by round while their count shrinks.
     *
     *  An absorb runs one such pass per donor: each donor moves in, its nodes walk across
     *  everything absorbed before them, seeded beside the lists of their neighbors that walked
     *  already, and merge, and the sweep reads their links alone, before the next donor moves in.
     *  An empty range heals.
     *
     *  A split runs it in reverse: the donor's nodes partition into one graph-local part per
     *  receiver, scatter into the receivers with the links inside each part rebased and the links
     *  across dropped, and publish; then every receiver is refined in turn over its picked nodes,
     *  the prefix of nodes that lost a link, and heals, the tallies summing over the receivers.
     *
     *  The plan holds the shared side and the claims closed from planning to its death, so
     *  batches answer @c contended_k and stop-the-world verbs refuse meanwhile, while
     *  searches run throughout.
     *
     *  Steps of the current phase are disjoint ID ranges, so any number of threads may @ref run
     *  them at once, each with its own metric and a context @ref reserve_traversal sized for the
     *  planned budget; @ref advance is the single-caller barrier between phases and answers whether
     *  another phase follows. A live list only ever moves from old to better under its lock, by its
     *  walk's merge and by the sweep's, and every node the first audit reached both ways stays
     *  reached both ways throughout; the heal edits under the per-node locks like an insertion.
     *  Nothing allocates, since planning reserved every neighbor list to the budget, and a pass
     *  stopped between steps leaves the graph correct. A plan that dies unfinished heals nothing:
     *  it reopens the claims, and the graph is as the last step left it.
     */
    struct [[nodiscard]] refine_plan_t {
        using step_t = typename health_plan_t::step_t;

        /** The audit the heal phases sweep. */
        health_plan_t audit;

        /** Steps of the current phase a driver already ran, so a cancelled call resumes at the next
         *  one. */
        std::size_t taken = 0;

        /** Nodes the planned phases re-selected so far, which a driver's progress reports. */
        std::size_t ticked = 0;

        /** An unbound pass placed by @p allocator - what a refused plan answers with. */
        explicit refine_plan_t(allocator_t const& allocator) noexcept
            : audit(allocator), part_bases_(allocator), back_links_(allocator), incoming_offsets_(allocator),
              changed_(allocator), donor_holds_(allocator), receiver_holds_(allocator), part_slots_(allocator),
              part_members_(allocator), part_extents_(allocator) {}
        refine_plan_t(refine_plan_t&& other) noexcept
            : audit(std::move(other.audit)), taken(other.taken), ticked(other.ticked),
              index_(std::exchange(other.index_, nullptr)), hold_(std::move(other.hold_)), nodes_(other.nodes_),
              part_bases_(std::move(other.part_bases_)), back_links_(std::move(other.back_links_)),
              incoming_offsets_(std::move(other.incoming_offsets_)), tile_(other.tile_), sweep_(other.sweep_),
              tile_nodes_(other.tile_nodes_), changed_(std::move(other.changed_)), planned_(other.planned_),
              donors_(other.donors_), donor_holds_(std::move(other.donor_holds_)),
              absorbed_donors_(other.absorbed_donors_), split_donor_(std::exchange(other.split_donor_, nullptr)),
              split_donor_hold_(std::move(other.split_donor_hold_)), receivers_(other.receivers_),
              receiver_holds_(std::move(other.receiver_holds_)), part_slots_(std::move(other.part_slots_)),
              part_members_(std::move(other.part_members_)), part_extents_(std::move(other.part_extents_)),
              seed_(other.seed_), refined_receivers_(other.refined_receivers_),
              receivers_reachability_(other.receivers_reachability_), receivers_islands_(other.receivers_islands_),
              walk_(other.walk_), connectivity_(other.connectivity_), phase_(other.phase_),
              swept_after_(other.swept_after_), islands_before_(other.islands_before_),
              audited_rows_(other.audited_rows_), tally_(other.tally_) {}
        refine_plan_t(refine_plan_t const&) = delete;
        refine_plan_t& operator=(refine_plan_t const&) = delete;
        ~refine_plan_t() noexcept { release_(); }

        /** Step @p position of the current phase: one node, a sweep's bitset word, a word of
         *  nodes whose links are counted or scattered, or the whole partition; an empty step
         *  past the end. */
        step_t operator[](std::size_t position) const noexcept {
            if (phase_ == phase_t::audit_k) return audit[position];
            step_t step;
            if (position >= steps().size()) return step;
            iota_view<node_id_t> const range = phase_nodes_();
            std::size_t const width = phase_width_();
            std::size_t const first = range.first + position * width;
            std::size_t const bound = min_of(first + width, static_cast<std::size_t>(range.bound));
            step.position = position;
            step.nodes = {static_cast<node_id_t>(first), static_cast<node_id_t>(bound)};
            return step;
        }

        /** The current phase's steps, positionally; none once the pass is over. */
        indexed_range<refine_plan_t> steps() const noexcept {
            if (!index_ || phase_ == phase_t::done_k) return {this, 0u};
            if (phase_ == phase_t::partition_k) return {this, 1u};
            if (phase_ == phase_t::audit_k) return {this, audit.steps().size()};
            return {this, divide_round_up<std::size_t>(phase_nodes_().size(), phase_width_())};
        }

        /**
         *  @brief Runs @p step of the current phase on the calling thread - safe for disjoint steps
         *      of one phase, and never failing on a bound plan, since planning claimed, covered and
         *      reserved everything a step writes.
         *
         *  A copy step moves a donor's nodes in, the one partition step grows a split's parts, a
         *  scatter step moves donor nodes into their receivers, a re-selection step walks from each
         *  node and merges what @p selector picks into its list, a count or scatter step reads the
         *  links of a word of walked nodes into the sweep's tile, a merge step writes one node of
         *  the tile from its members and the nodes linking to it, a sweep step expands the audit,
         *  and a reconnection step links islands to the nodes the audit proved both ways; @p metric
         *  and @p context serve the walks and merges alone.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is bound to no index.
         */
        template <typename metric_type_, typename context_allocator_type_, typename selector_type_ = keep_diverse_t>
            requires metric<metric_type_, value_t, distance_t> && selector<selector_type_, ram_index, metric_type_>
        status_t run(step_t const& step, metric_type_& metric, traversal_context_over<context_allocator_type_>& context,
                     selector_type_ const& selector = {}) noexcept {
            if (!index_) return status_t::uninitialized_k;
            switch (phase_) {
            case phase_t::copy_k: copy_(step); break;
            case phase_t::partition_k: partition_(); break;
            case phase_t::scatter_k: scatter_(step); break;
            case phase_t::reselect_k:
            case phase_t::reselect_islands_k: reselect_(step, metric, context, selector); break;
            case phase_t::audit_k: audit.sweep(step); break;
            case phase_t::count_back_links_k: count_back_links_(step); break;
            case phase_t::scatter_back_links_k: scatter_back_links_(step); break;
            case phase_t::merge_back_links_k: merge_back_links_(step, metric, context, selector); break;
            case phase_t::heal_k: reconnect_(step, metric, context); break;
            default: break;
            }
            return status_t::success_k;
        }

        /** The barrier between phases, one caller after every step of the current one ran:
         *  publishes a copied donor and opens its pass, publishes a split's parts, opens the next
         *  walking or sweeping tile, or picks the next phase, and answers whether one follows; a
         *  finished pass copies the next donor in, if any is pending. */
        bool advance() noexcept {
            if (!index_) return false;
            // The first donor's pass audits; every later one carries its levels, the donor's nodes
            // unreached, and only the last verifies and heals.
            if (phase_ == phase_t::copy_k) {
                bool const opening = !absorbed_donors_;
                publish_copied_();
                if (opening) {
                    audit.rebind(*index_, nodes_.bound);
                    open_pass_();
                }
                else {
                    audit.widen(nodes_.bound);
                    resume_pass_();
                }
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
            if (phase_ == phase_t::audit_k) {
                if (!audit.advance_level()) converge_();
            }
            // The walking tiles merge their nodes' picks as they go; then the sweep's tiles, over
            // every node, collect the links the walked nodes hold into each and merge them.
            else if (phase_ == phase_t::reselect_k && tile_.bound < nodes_.bound) open_tile_(tile_.bound);
            else if (phase_ == phase_t::reselect_k) open_sweep_(0);
            else if (phase_ == phase_t::count_back_links_k) {
                offset_back_links_();
                phase_ = phase_t::scatter_back_links_k;
            }
            else if (phase_ == phase_t::scatter_back_links_k) phase_ = phase_t::merge_back_links_k;
            else if (phase_ == phase_t::merge_back_links_k && sweep_.bound < audit.visible) open_sweep_(sweep_.bound);
            else if (phase_ == phase_t::merge_back_links_k && absorbed_donors_ != donors_.size())
                phase_ = phase_t::copy_k;
            else if (phase_ != phase_t::done_k) {
                audit.restart_sweep();
                swept_after_ = phase_;
                phase_ = phase_t::audit_k;
            }
            if (phase_ == phase_t::done_k && refined_receivers_ != receivers_.size()) settle_receiver_();
            if (phase_ == phase_t::done_k && absorbed_donors_ != donors_.size()) phase_ = phase_t::copy_k;
            return phase_ != phase_t::done_k;
        }

        /** Ends the planned range with the open walking tile, so the sweep follows it and the pass
         *  then verifies, and the receivers of a split still to bind walk nothing; a sweep under
         *  way finishes. A no-op outside the walking tiles, the copies included. */
        void stop() noexcept {
            if (phase_ != phase_t::reselect_k) return;
            nodes_.bound = tile_.bound;
            for (std::size_t receiver = refined_receivers_ + 1; receiver < receivers_.size(); ++receiver)
                part_extents_[receiver].picked_count = 0;
            taken = steps().size();
        }

        /** The walks' knobs, which every context stepping the plan is reserved for alongside
         *  @ref connectivity. */
        insert_config_t walk() const noexcept { return walk_; }

        /** The budget planning reserved every neighbor list for. */
        std::size_t connectivity() const noexcept { return connectivity_; }

        /** Whether the current phase copies a donor or moves a split's nodes - the phases no stop
         *  skips. */
        bool moving() const noexcept {
            return phase_ == phase_t::copy_k || phase_ == phase_t::partition_k || phase_ == phase_t::scatter_k;
        }

        /** Whether the current phase re-selects the planned range, which a caller's progress
         *  counts. */
        bool reselecting() const noexcept { return phase_ == phase_t::reselect_k; }

        /** Nodes the re-selection steps over, the total a caller's progress counts against: the
         *  planned range, or every donor's nodes. */
        std::size_t planned_nodes() const noexcept { return planned_; }

        /** The nodes the open tile spans - what a re-selection phase steps over. */
        iota_view<node_id_t> tile() const noexcept { return tile_; }

        /**
         *  The pass so far: complete once @ref advance answered false, otherwise @c cancelled_k.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is bound to no index.
         *  - @c cancelled_k The pass stopped before @ref advance answered false.
         */
        expected<refine_report_t> finish() const noexcept {
            if (!index_) return {{}, status_t::uninitialized_k};
            return {tally_, phase_ == phase_t::done_k ? status_t::success_k : status_t::cancelled_k};
        }

      private:
        friend ram_index;

        enum class phase_t : u8_t {
            copy_k,
            partition_k,
            scatter_k,
            reselect_k,
            audit_k,
            count_back_links_k,
            scatter_back_links_k,
            merge_back_links_k,
            reselect_islands_k,
            heal_k,
            done_k,
        };

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

        /** The nodes the current phase steps over: a donor's, a split donor's, every audited
         *  one in the heal, the walked ones whose links the sweep reads, its tile's, or the
         *  open walking tile's. */
        iota_view<node_id_t> phase_nodes_() const noexcept {
            if (phase_ == phase_t::copy_k) return copied_nodes_();
            if (phase_ == phase_t::partition_k || phase_ == phase_t::scatter_k) return {0, split_nodes_()};
            if (phase_ == phase_t::reselect_islands_k || phase_ == phase_t::heal_k) return {0, audit.visible};
            if (phase_ == phase_t::count_back_links_k || phase_ == phase_t::scatter_back_links_k) return nodes_;
            if (phase_ == phase_t::merge_back_links_k) return sweep_;
            return tile_;
        }

        /** Nodes one step of the current phase spans: the whole partition, a word of nodes whose
         *  links are read, or one node walked or written. */
        std::size_t phase_width_() const noexcept {
            if (phase_ == phase_t::partition_k) return max_of<std::size_t>(split_nodes_(), 1);
            if (phase_ == phase_t::count_back_links_k || phase_ == phase_t::scatter_back_links_k)
                return health_plan_t::lanes_k;
            return 1;
        }

        /** Opens the walking tile at @p first: up to @c tile_nodes_ nodes of the planned range. */
        void open_tile_(node_id_t first) noexcept {
            tile_ = {first, static_cast<node_id_t>(min_of<std::size_t>(first + tile_nodes_, nodes_.bound))};
        }

        /** Opens the sweep's tile of target nodes from @p first, up to @c tile_nodes_ of them, none
         *  linked yet. */
        void open_sweep_(node_id_t first) noexcept {
            sweep_ = {first, static_cast<node_id_t>(min_of<std::size_t>(first + tile_nodes_, audit.visible))};
            std::memset(incoming_offsets_.data(), 0, sweep_.size() * sizeof(node_id_t));
            ++tally_.swept_tiles;
            phase_ = phase_t::count_back_links_k;
        }

        /** Nodes one tile of a pass over @p nodes nodes re-selecting @p forward_nodes of them
         *  takes: as many as @p memory_limit holds past the audit and the changed bits, at 8 B per
         *  neighbor slot plus an offset each, or at zero the widest storage chunk; zero when the
         *  limit holds not one. */
        std::size_t tile_nodes_for_(std::size_t nodes, std::size_t forward_nodes,
                                    std::size_t memory_limit) const noexcept {
            std::size_t const capped = min_of<std::size_t>(forward_nodes, missing_k / connectivity_);
            if (!memory_limit) return min_of(widest_window_below_(forward_nodes), capped);
            std::size_t const words = divide_round_up(nodes, health_plan_t::lanes_k);
            std::size_t const carried = (health_plan_t::bitsets_k + 1) * words * sizeof(u1x64_t) + 3 * nodes;
            std::size_t const word_bytes = health_plan_t::lanes_k *
                                           (connectivity_ * sizeof(scored_node_t) + sizeof(node_id_t));
            if (memory_limit <= carried) return 0;
            return min_of((memory_limit - carried) / word_bytes * health_plan_t::lanes_k, capped);
        }

        /** Nodes the widest storage chunk below @p rows holds: a pass's tile under a zero limit. */
        static std::size_t widest_window_below_(std::size_t rows) noexcept {
            return rows ? min_of(rows, nodes_schedule_t::nodes_in(nodes_schedule_t::locate(rows - 1).chunk)) : 0;
        }

        /** Nodes the split's donor published at planning: one slot each. */
        node_id_t split_nodes_() const noexcept { return static_cast<node_id_t>(part_slots_.size()); }

        /** The most nodes one part takes: the donor's nodes over the receivers, rounded up. */
        std::size_t part_capacity_() const noexcept {
            return divide_round_up<std::size_t>(part_slots_.size(), receivers_.size());
        }

        /** Opens the pass over @c nodes_ below the audit's watermark: its changed bits, and the
         *  re-selection, or straight the verifying sweep when it walks nothing or a stop cut it. */
        void open_pass_() noexcept {
            open_merge_();
            islands_before_ = 0;
            // The opening audit steers the merges; a pass walking nothing verifies alone.
            swept_after_ = !nodes_.empty() ? phase_t::done_k : phase_t::merge_back_links_k;
            phase_ = phase_t::audit_k;
        }

        /** Turns the pass carried from the donors before to the nodes the current one appended,
         *  their first tile. */
        void resume_pass_() noexcept {
            open_tile_(nodes_.first);
            phase_ = phase_t::reselect_k;
        }

        /** The current donor's nodes, in its own ID space. */
        iota_view<node_id_t> copied_nodes_() const noexcept {
            return {0, static_cast<node_id_t>(donors_[absorbed_donors_]->size())};
        }

        /** Reserves one tile of @p tile_nodes nodes and the changed bits of @p nodes nodes. */
        status_t reserve_merge_(std::size_t nodes, std::size_t tile_nodes) noexcept {
            if (failed(back_links_.reserve_exactly(tile_nodes * connectivity_)) ||
                failed(incoming_offsets_.reserve_exactly(tile_nodes)) ||
                failed(changed_.reserve_exactly(divide_round_up(nodes, health_plan_t::lanes_k))))
                return status_t::bad_alloc_k;
            tile_nodes_ = tile_nodes;
            return status_t::success_k;
        }

        /** Sizes the tile within what @ref reserve_merge_ placed, every node the plan reserved for
         *  unchanged. */
        void open_merge_() noexcept {
            [[maybe_unused]] status_t const offset = incoming_offsets_.resize(tile_nodes_);
            [[maybe_unused]] status_t const uncounted = changed_.resize(changed_.capacity());
            if (!changed_.empty()) std::memset(changed_.data(), 0, changed_.size() * sizeof(u1x64_t));
        }

        /** Marks node @p node_id counted as changed, answering whether it was already. */
        bool count_changed_(node_id_t node_id) noexcept {
            return atomic_ref<u1x64_t>(changed_[node_id / health_plan_t::lanes_k])
                .fetch_test_and_set_bit(static_cast<unsigned>(node_id % health_plan_t::lanes_k),
                                        std::memory_order_relaxed);
        }

        /** Turns the sweep tile's counts into each node's first link's offset, which the scatter
         *  advances to its end, and sizes the tape to them: the tile ends before the first node
         *  whose links pass the tape, and a lone node past it keeps the tape's worth of them. */
        void offset_back_links_() noexcept {
            std::size_t const capacity = back_links_.capacity();
            std::size_t offset = 0, nodes = 0;
            for (; nodes != sweep_.size(); ++nodes) {
                std::size_t count = incoming_offsets_[nodes];
                if (offset + count > capacity) {
                    if (nodes) break;
                    count = capacity;
                }
                incoming_offsets_[nodes] = static_cast<node_id_t>(offset);
                offset += count;
            }
            sweep_.bound = static_cast<node_id_t>(sweep_.first + max_of<std::size_t>(nodes, 1));
            back_links_.clear();
            back_links_.extend_uninitialized(offset);
        }

        /** Counts every link of the walked nodes of @p step into the sweep's tile. No list is
         *  written meanwhile, so the scatter reads the same links. */
        void count_back_links_(step_t const& step) noexcept {
            for (node_id_t node_id = step.nodes.first; node_id < step.nodes.bound; ++node_id)
                for (node_id_t const target : index_->nodes_[node_id].neighbors)
                    if (target >= sweep_.first && target < sweep_.bound)
                        atomic_ref<node_id_t>(incoming_offsets_[target - sweep_.first])
                            .fetch_add(1u, std::memory_order_relaxed);
        }

        /** Scatters every link of the walked nodes of @p step into the sweep's tile onto its
         *  target's run, the linking node unscored; a link the tape has no room for is left out. */
        void scatter_back_links_(step_t const& step) noexcept {
            std::size_t scattered = 0;
            std::size_t const tape = back_links_.size();
            for (node_id_t node_id = step.nodes.first; node_id < step.nodes.bound; ++node_id)
                for (node_id_t const target : index_->nodes_[node_id].neighbors) {
                    if (target < sweep_.first || target >= sweep_.bound) continue;
                    std::size_t const position = atomic_ref<node_id_t>(incoming_offsets_[target - sweep_.first])
                                                     .fetch_add(1u, std::memory_order_relaxed);
                    if (position < tape) back_links_[position] = scored_node_t {node_id, distance_t {}}, ++scattered;
                }
            add_(tally_.swept_links, scattered);
        }

        /** Merges every node of @p step in the sweep's tile with the nodes linking to it, scored
         *  against it first; a node counts as changed once, whichever of its merges first gains
         *  it a member. */
        template <typename metric_type_, typename context_type_, typename selector_type_>
        void merge_back_links_(step_t const& step, metric_type_& metric, context_type_& context,
                               selector_type_ const& selector) noexcept {
            traversal_stats_t const before = context.telemetry;
            std::size_t changed = 0;
            auto nearer = [](scored_node_t const& left, scored_node_t const& right) noexcept {
                return left.distance < right.distance ||
                       (left.distance == right.distance && left.node_id < right.node_id);
            };
            std::size_t const tape = back_links_.size();
            for (node_id_t node_id = step.nodes.first; node_id < step.nodes.bound; ++node_id) {
                std::size_t const node = node_id - sweep_.first;
                std::size_t const first = node ? min_of<std::size_t>(incoming_offsets_[node - 1], tape) : 0;
                std::size_t const bound = min_of<std::size_t>(incoming_offsets_[node], tape);
                std::span<scored_node_t> const incoming {back_links_.data() + first, bound - first};
                if (incoming.empty()) continue;
                index_->score_run_(node_id, incoming, metric, context);
                // The scatter filled each run in arrival order; sorted, its fold is reproducible.
                partial_sort(incoming, incoming.size(), nearer);
                bool const gained = index_->merge_back_links_(node_id, {}, incoming, audit, metric, context, selector);
                // An absorb's links also reach nodes it never walked, which the report leaves out.
                bool const walked = node_id >= nodes_.first && node_id < nodes_.bound;
                changed += gained && walked && !count_changed_(node_id);
            }
            add_(tally_.changed_nodes, changed);
            add_spent_(context.telemetry - before);
        }

        /** Moves the nodes of @p step from the current donor into the records past the watermark,
         *  every neighbor ID shifted by it; the nodes swap buffers, so nothing allocates. */
        void copy_(step_t const& step) noexcept {
            ram_index& donor = *donors_[absorbed_donors_];
            node_id_t const base = static_cast<node_id_t>(index_->size());
            for (node_id_t donor_node = step.nodes.first; donor_node < step.nodes.bound; ++donor_node) {
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
            // The pass that follows walks the copied nodes alone, below the watermark they moved.
            nodes_ = {base, static_cast<node_id_t>(base + copied)};
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
            // Until the picked nodes replace them, these count each part's expanded members.
            node_id_t& expanded = part_extents_[part].picked_count;
            std::size_t const capacity = part_capacity_();
            if (expanded == part_extents_[part].nodes || part_extents_[part].nodes == capacity) return false;
            node_id_t const node_id = part_members_[part * capacity + expanded++];
            for (node_id_t const target : split_donor_->nodes_[node_id].neighbors)
                if (target < split_nodes_() && part_slots_[target] == missing_k &&
                    part_extents_[part].nodes != capacity)
                    enlist_(target, part);
            return true;
        }

        /** Moves donor node @p node into its part's picked nodes, the prefix of its members, unless
         *  it is there already, keeping both swapped nodes' slots. */
        void enlist_picked_(node_id_t node) noexcept {
            std::size_t const capacity = part_capacity_();
            std::size_t const part = part_slots_[node] / capacity, position = part_slots_[node] % capacity;
            node_id_t& picked_count = part_extents_[part].picked_count;
            if (position < picked_count) return;
            node_id_t* const members = part_members_.data() + part * capacity;
            std::swap(members[position], members[picked_count]);
            part_slots_[members[position]] = static_cast<node_id_t>(part * capacity + position);
            part_slots_[members[picked_count]] = static_cast<node_id_t>(part * capacity + picked_count);
            ++picked_count;
        }

        /**
         *  @brief Grows one part per receiver from a root the seed picks, breadth first over the
         *      source's links and one expansion per part in turn, each capped at the part capacity;
         *      every node left unplaced then seeds a growth of the smallest part.
         *
         *  Each part's picked nodes, the ones at either end of a link into another part, then move
         *  ahead of the rest, and every node's slot names its part and its ID in the receiver.
         *  Reads the donor without locks, since its unique side is held.
         */
        void partition_() noexcept {
            std::size_t const parts = receivers_.size(), capacity = part_capacity_();
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
            for (std::size_t part = 0; part != parts; ++part) part_extents_[part].picked_count = 0;
            for (node_id_t node = 0; node != split_nodes; ++node)
                for (node_id_t const target : split_donor_->nodes_[node].neighbors) {
                    if (target >= split_nodes || part_slots_[target] / capacity == part_slots_[node] / capacity)
                        continue;
                    enlist_picked_(node);
                    enlist_picked_(target);
                }
        }

        /** Moves the donor nodes of @p step into their receivers' records, the links inside each
         *  part rebased through its slots and the links across dropped; the nodes swap buffers, so
         *  nothing allocates. */
        void scatter_(step_t const& step) noexcept {
            std::size_t const capacity = part_capacity_();
            node_id_t const split_nodes = split_nodes_();
            for (node_id_t donor_node = step.nodes.first; donor_node < step.nodes.bound; ++donor_node) {
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

        /** Turns the pass to the next receiver: its audit, its picked nodes as the range,
         *  and the re-selection, or straight the verifying sweep when it has none or a stop
         *  cut the pass. */
        void bind_receiver_() noexcept {
            std::size_t const receiver = refined_receivers_;
            index_ = receivers_[receiver];
            audit.rebind(*index_, part_extents_[receiver].nodes);
            nodes_ = {0, part_extents_[receiver].picked_count};
            open_pass_();
        }

        /** Folds a healed receiver's audit and islands into the split's, then binds the next one,
         *  or reports the sums once the last healed. */
        void settle_receiver_() noexcept {
            receivers_reachability_ += tally_.reachability;
            receivers_islands_ += std::exchange(tally_.islands_found, 0);
            if (++refined_receivers_ != receivers_.size()) return bind_receiver_();
            tally_.reachability = receivers_reachability_;
            tally_.islands_found = receivers_islands_;
        }

        /** Picks what follows a converged audit, from the phase the audit followed: the merge
         *  after the re-selection, whose levels it now holds, and the heal after the merge. */
        void converge_() noexcept {
            audited_rows_ = audit.visible;
            auto const [health, status] = audit.finish();
            std::size_t const left = succeeded(status) ? audit.visible - health.reachable_nodes : 0;
            tally_.reachability = health;
            // The opening audit, whose levels steer every merge, leads to the first tile.
            if (swept_after_ == phase_t::done_k) {
                open_tile_(nodes_.first);
                phase_ = phase_t::reselect_k;
                return;
            }
            if (swept_after_ == phase_t::merge_back_links_k) {
                tally_.islands_found += left;
                phase_ = left ? phase_t::reselect_islands_k : phase_t::done_k;
            }
            else if (swept_after_ == phase_t::reselect_islands_k) { phase_ = left ? phase_t::heal_k : phase_t::done_k; }
            // A round freeing no island frees none on the next: the picks it needs are unreached.
            else { phase_ = left && left < islands_before_ ? phase_t::reselect_islands_k : phase_t::done_k; }
            if (swept_after_ != phase_t::reselect_islands_k) islands_before_ = left;
        }

        /** Re-selects every planned node of @p step and merges its picks into its list,
         *  or, once the heal rounds began, only its islands, each written and back-linked
         *  as an insertion is. */
        template <typename metric_type_, typename context_type_, typename selector_type_>
        void reselect_(step_t const& step, metric_type_& metric, context_type_& context,
                       selector_type_ const& selector) noexcept {
            traversal_stats_t const before = context.telemetry;
            std::size_t reselected = 0, changed = 0;
            // Only the planned pass walks across; the heal rounds re-select as a plain pass does.
            bool const planned = phase_ == phase_t::reselect_k;
            bool const across = planned && part_bases_.size() > 1;
            for (node_id_t node_id = step.nodes.first; node_id < step.nodes.bound; ++node_id) {
                if (!planned && audit.reached(node_id) && audit.reaches_back(node_id)) continue;
                // Everything absorbed before the node's part: the resident nodes, and the parts
                // moved in ahead of its own.
                iota_view<node_id_t> const other =
                    across ? iota_view<node_id_t> {part_bases_[0], part_bases_[part_of(to_span(part_bases_), node_id)]}
                           : iota_view<node_id_t> {};
                std::span<node_id_t const> const across_seeds = other.empty()
                                                                    ? std::span<node_id_t const> {}
                                                                    : stage_across_seeds_(node_id, other, context);
                index_->refine_one_(node_id, audit.visible, other, across_seeds, walk_, metric, context, selector);
                ++reselected;
                if (planned) {
                    // The merge writes the selection, so the picks compete from a copy.
                    auto& picks = context.picks;
                    picks.clear();
                    if (failed(picks.reserve(context.selected_neighbors.size()))) continue;
                    for (scored_node_t const pick : context.selected_neighbors) picks.push_back_reserved(pick);
                    bool const gained = index_->merge_back_links_(node_id, to_span(picks), {}, audit, metric, context,
                                                                  selector);
                    changed += gained && !count_changed_(node_id);
                    continue;
                }
                changed += index_->rewrite_list_(node_id, selector.connectivity, context);
                index_->template back_link_<synchronization_t::concurrent_k>(node_id, metric, selector, context);
            }
            add_(tally_.reselected_nodes, reselected);
            add_(tally_.changed_nodes, changed);
            add_spent_(context.telemetry - before);
        }

        /** Stages into @c context.hinted_ids the neighbors of @p node_id 's neighbors inside
         *  @p other, which a neighbor that walked already merged in: where its stitched neighbors
         *  landed, so its own walk across starts beside them. One list of seeds, as many as a walk
         *  draws landmarks, taken neighbor by neighbor. */
        template <typename context_type_>
        std::span<node_id_t const> stage_across_seeds_(node_id_t node_id, iota_view<node_id_t> other,
                                                       context_type_& context) noexcept {
            context.staged_ids.clear();
            auto const [degree, copy_status] = index_->template copy_neighbors_<synchronization_t::concurrent_k>(
                node_id, context);
            context.hinted_ids.clear();
            if (failed(copy_status) || failed(context.hinted_ids.reserve(connectivity_))) return {};
            for (std::size_t member = 0; member != degree && context.hinted_ids.size() != connectivity_; ++member) {
                node_t& neighbor = index_->nodes_[context.staged_ids[member]];
                neighbor.lock();
                for (node_id_t const pick : neighbor.neighbors)
                    if (pick >= other.first && pick < other.bound && context.hinted_ids.size() != connectivity_)
                        context.hinted_ids.push_back_reserved(pick);
                neighbor.unlock();
            }
            return to_span(context.hinted_ids);
        }

        /** Folds one step's walk and merge spend into the tally. */
        void add_spent_(traversal_stats_t const& spent) noexcept {
            add_(tally_.traversal_stats.computed_distances, spent.computed_distances);
            add_(tally_.traversal_stats.visited_members, spent.visited_members);
            add_(tally_.traversal_stats.restarts, spent.restarts);
            add_(tally_.traversal_stats.saturated_walks, spent.saturated_walks);
            add_(tally_.traversal_stats.computed_distances_in_refines, spent.computed_distances_in_refines);
            add_(tally_.traversal_stats.computed_distances_in_reverse_refines,
                 spent.computed_distances_in_reverse_refines);
        }

        /** Reconnects every island in @p step, a node the audit left unreached or not reaching
         *  back, to the nearest node it proved both ways, found by a repair walk: that node's
         *  neighbor list links an unreached island, and the island's own neighbor list links that
         *  node when it has no way back. */
        template <typename metric_type_, typename context_type_>
        void reconnect_(step_t const& step, metric_type_& metric, context_type_& context) noexcept {
            std::size_t reconnected = 0;
            typename health_plan_t::membership_t members {audit};
            dummy_predicate_t unfiltered;
            for (node_id_t island = step.nodes.first; island < step.nodes.bound; ++island) {
                bool const reached = audit.reached(island), reaches_back = audit.reaches_back(island);
                if (reached && reaches_back) continue;
                // The walk routes through every visible node and admits only the proved ones, so a
                // sparse membership never starves its landmarks.
                context.results.clear();
                context.frontier.clear();
                context.visited.clear();
                context.admitted.clear();
                predicate_fold fold {members, unfiltered, context.admitted, walk_.expansion_factor};
                index_->template search_around_<synchronization_t::concurrent_k>(
                    index_->published_value_of(island), metric, snapshot_admission_t {audit.visible},
                    walk_.expansion_factor, connectivity_, context.admitted, context, fold);
                if (context.admitted.empty()) continue;
                // An unreached island reaches the nearest proved node that can take it.
                if (!reached)
                    for (scored_node_t const& candidate : context.admitted)
                        if (link_island_(candidate.node_id, island)) {
                            ++reconnected;
                            break;
                        }
                if (!reaches_back) reconnected += link_island_(island, context.admitted[0].node_id);
            }
            add_(tally_.islands_reconnected, reconnected);
        }

        /** Links @p island from @p owner 's neighbor list under its lock: a free slot when there is
         *  one, otherwise a slot spread across the tail, so islands sharing one neighbor list heal
         *  in one round; answers whether it wrote. */
        bool link_island_(node_id_t owner, node_id_t island) noexcept {
            node_t& node = index_->nodes_[owner];
            node.lock();
            neighbors_buffer_t& neighbors = node.neighbors;
            bool present = false, linked = false;
            for (node_id_t const resident : neighbors) present = present || resident == island;
            if (!present && neighbors.size() < connectivity_ && neighbors.size() < neighbors.capacity()) {
                neighbors.push_back_reserved(island);
                linked = true;
            }
            // Never the owner's last way back, and in a reached neighbor list, whose unreached
            // residents are this round's links, never one of those nor a resident's last in-link
            // from an earlier level; an unreached neighbor list proves no in-link.
            else if (!present && audit.release_out_link(owner)) {
                std::size_t const width = min_of(heal_spread(connectivity_), neighbors.size());
                for (std::size_t attempt = 0; attempt != width && !linked; ++attempt) {
                    std::size_t const slot = neighbors.size() - 1 - (island + attempt) % width;
                    node_id_t const resident = neighbors[slot];
                    if (resident >= audit.visible ||
                        (audit.reached(owner) && (!audit.reached(resident) || !audit.release_link(resident))))
                        continue;
                    neighbors[slot] = island;
                    linked = true;
                }
            }
            node.unlock();
            return linked;
        }

        /** The refined index, whose shared side @c hold_ occupies and whose claims stay closed;
         *  null in an unbound plan. */
        ram_index* index_ = nullptr;

        /** The shared hold spanning the plan's whole lifetime. */
        shared_lock_t hold_;

        /** The IDs the pass re-selects, clamped to the audit's watermark: the planned range, or the
         *  nodes the pending donor appended. */
        iota_view<node_id_t> nodes_;

        /** The ascending first nodes of the parts the re-selection walks across; empty for none. */
        growing_buffer<node_id_t, allocator_t> part_bases_;

        /** The sweep tile's tape: every link into its nodes from the walked nodes, each a
         *  linking node and, once its merge scored it, their distance, the links of one node
         *  back to back. */
        growing_buffer<scored_node_t, allocator_t> back_links_;

        /** Per node of the sweep's tile: the links it drew, then where they start on the tape, and
         *  once scattered where they end. */
        growing_buffer<node_id_t, allocator_t> incoming_offsets_;

        /** The nodes walking now, or the last that walked. */
        iota_view<node_id_t> tile_;

        /** The target nodes the sweep collects links into now. */
        iota_view<node_id_t> sweep_;

        /** Nodes one walking or sweeping tile takes at most. */
        std::size_t tile_nodes_ = 0;

        /** Per node, whether a merge counted it changed already. */
        growing_buffer<u1x64_t, allocator_t> changed_;

        /** Nodes every pass of the plan re-selects in all. */
        std::size_t planned_ = 0;

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

        /** One part's nodes, and the picked count within them: the nodes before it lost a link
         *  across. */
        struct part_extent_t {
            node_id_t nodes = 0;
            node_id_t picked_count = 0;
        };

        /** Per part, its extent. */
        growing_buffer<part_extent_t, allocator_t> part_extents_;

        /** What picks the parts' roots. */
        u64_t seed_ = 0;

        /** Receivers whose passes healed so far; the bound one is the next. */
        std::size_t refined_receivers_ = 0;

        /** The audits of the receivers healed so far, summed. */
        reachability_t receivers_reachability_ {};

        /** The islands the receivers healed so far found, summed. */
        std::size_t receivers_islands_ = 0;

        /** Breadth and fan-out of every re-selection walk. */
        insert_config_t walk_ {};

        /** The planned neighbors-per-node budget every neighbor list was reserved for. */
        std::size_t connectivity_ = 0;

        /** The phase whose steps run now. */
        phase_t phase_ = phase_t::done_k;

        /** The phase the current audit follows - what decides the phase after it converges. */
        phase_t swept_after_ = phase_t::reselect_k;

        /** Islands the last round started from - the shrinking the rounds require. */
        std::size_t islands_before_ = 0;

        /** Rows the last converged sweep covered; those appended since are neither reached nor
         *  islands yet. */
        node_id_t audited_rows_ = 0;

        /** Counts summed through @c atomic_ref by the steps, and set by @ref advance. */
        refine_report_t tally_;
    };

    /**
     *  @brief Opens a refinement pass over @p nodes, clamped to the published IDs, with a fresh
     *      audit seeded like @ref plan_health; an empty range heals alone.
     *
     *  Every fallible step happens here: the audit's bitsets from @p context 's allocator, the
     *  walks' scratch in @p context, and every published neighbor list reserved to @p selector 's
     *  budget, so the steps never allocate; a caller fanning the steps out reserves its other
     *  contexts alike. Enters the shared side and closes the claims, refusing with @c contended_k
     *  while a batch is in flight.
     *
     *  @param[in] nodes The ID range the first round re-selects.
     *  @param[in] config Optional tuning knobs; zeros mean "derive from the selector's budget".
     *  @param[in] selector Carries the budget every neighbor list is reserved for; the steps take a
     *      selector of the same budget.
     *  @param[in,out] context The planning thread's scratch, whose allocator places the plan.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The selector's budget is zero.
     *  - @c overflow_k A scratch capacity of @p context wraps when rounded.
     *  - @c contended_k A batch is in flight, or another refine holds the claims closed.
     */
    template <typename selector_type_>
    expected<refine_plan_t> plan_refine(iota_view<node_id_t> nodes, refine_config_t config,
                                        selector_type_ const& selector, traversal_context_t& context) noexcept {
        return plan_pass_(nodes, {}, context, selector.connectivity, config);
    }

    /**
     *  @brief Opens an absorb of @p donors, whose plan's first phase moves every node of each into
     *      this index, one donor after another, as a part of its own.
     *
     *  Later phases re-select the appended nodes and, round by round, the residents their
     *  back-links edited across parts, and heal.
     *
     *  Every fallible step of every copy happens here, after the checks of @c plan_refine: each
     *  donor's unique side taken, the records and the key multimap past the watermark covered, and
     *  every donor neighbor list reserved to the budget. A copy step swaps a donor node's value and
     *  neighbors in with every ID shifted, so nothing allocates, and the phase boundary files the
     *  donor's keys and publishes its nodes at once, empties the donor and nulls its slot; at most
     *  one donor's nodes are ever unpublished. Donors must share this index's allocator, and
     *  nothing may search them meanwhile.
     *
     *  @param[in,out] donors The indexes giving their nodes up, each nulled once moved; the slots
     *      must outlive the plan.
     *  @param[in] config The refinement budget the plan runs under.
     *  @param[in] selector Picks the neighbors each node keeps when the plan re-selects.
     *  @param[in,out] context The traversal context the plan's walks draw their buffers from.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k A null donor, this index itself, or a zero budget.
     *  - @c contended_k A batch in flight on this index or on a donor, or another refine.
     *  - @c overflow_k Nodes past the reserved ceiling.
     */
    template <typename selector_type_>
    expected<refine_plan_t> plan_absorb(std::span<ram_index*> donors, refine_config_t config,
                                        selector_type_ const& selector, traversal_context_t& context) noexcept {
        return plan_pass_({}, donors, context, selector.connectivity, config);
    }

    /**
     *  @brief Opens a split of this index into @p receivers - the inverse of @c plan_absorb:
     *      its plan's first phases move every node into one graph-local part per receiver, then
     *      refine every receiver in turn over the nodes that lost a link, and heal.
     *
     *  The one partition step grows the parts breadth first over the links, one root per part
     *  through @ref seeded_bijection under the config's seed, each part capped at the nodes over
     *  the receivers rounded up; nodes no growth reached seed the smallest part. Scatter steps move
     *  each node's value, key and neighbors into its receiver, the links inside its part rebased
     *  and the links across dropped, and the phase boundary files every receiver's keys and
     *  publishes its nodes at once, then empties this index and releases it, so its claims reopen
     *  at zero. Every receiver then re-selects its picked nodes, the prefix of nodes that lost a
     *  link, sweeps and heals as @ref plan_refine does, and the result sums over the receivers.
     *
     *  Every fallible step happens here: this index's unique side taken, each receiver's shared
     *  side entered and its claims closed, the records, key multimap and audit of a full part
     *  covered in every receiver, and every neighbor list reserved to the budget. Receivers must
     *  share this index's allocator, the slots must outlive the plan, and nothing may search this
     *  index meanwhile.
     *
     *  @param[in] receivers The empty indexes taking the parts, at least two, all distinct.
     *  @param[in] config The refinement budget the plan runs under, and the seed picking the parts'
     *      roots, so equal seeds over equal graphs split alike.
     *  @param[in] selector Picks the neighbors each node keeps when the plan re-selects.
     *  @param[in,out] context The traversal context the plan's walks draw their buffers from.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k Fewer than two receivers, a null, repeated or non-empty one, this
     *    index itself, or a zero budget.
     *  - @c contended_k A batch in flight on this index or on a receiver, or another refine.
     *  - @c overflow_k Part slots past the node ID type, or a part past a receiver's ceiling.
     */
    template <typename selector_type_>
    expected<refine_plan_t> plan_split(std::span<ram_index*> receivers, split_config_t config,
                                       selector_type_ const& selector, traversal_context_t& context) noexcept {
        return plan_split_(receivers, context, selector.connectivity, config.refine, config.seed);
    }

    /**
     *  @brief Re-selects the neighbors of every published node in @p nodes and heals the islands
     *      that leaves - one @ref refine_plan_t drained on the calling thread.
     *
     *  @param[in] nodes The ID range to re-select; IDs at or past the watermark are skipped.
     *  @param[in] config Optional tuning knobs; zeros mean "derive from the selector's budget".
     *  @param[in] selector The link-selection heuristic and its budget.
     *  @param[in,out] metric This thread's scoring instance.
     *  @param[in,out] context The calling thread's scratch & telemetry, which also places the plan.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The selector's budget is zero.
     *  - @c overflow_k A scratch capacity of @p context wraps when rounded.
     *  - @c contended_k A batch is in flight, or another refine holds the claims closed.
     */
    template <typename metric_type_, typename selector_type_>
        requires metric<metric_type_, value_t, distance_t> && selector<selector_type_, ram_index, metric_type_>
    expected<refine_report_t> refine(iota_view<node_id_t> nodes, refine_config_t config, selector_type_ const& selector,
                                     metric_type_& metric, traversal_context_t& context) noexcept {
        auto [plan, status] = plan_refine(nodes, config, selector, context);
        if (failed(status)) return {{}, status};
        do
            for (typename refine_plan_t::step_t const step : plan.steps()) [[maybe_unused]]
                status_t const ran = plan.run(step, metric, context, selector);
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
     *  each expanded node to copy its neighbor list - while a certified @c immutable_k or
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
     *  - @c contended_k A certified walk meets a mutator or a batch yet to publish.
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
     *  - @c contended_k A certified walk meets a mutator or a batch yet to publish.
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
     *  - @c contended_k A certified walk meets a mutator or a batch yet to publish.
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

        auto [audit, audit_status] = plan_audit_(visible, allocator_);
        if (failed(audit_status)) {
            report.status = audit_status;
            return report;
        }
        do
            for (typename health_plan_t::step_t const step : audit.steps()) [[maybe_unused]]
                std::size_t const expanded = audit.sweep(step);
        while (audit.advance_level());
        report.reachability = audit.finish().value;
        return report;
    }

#pragma endregion Keys and Introspection

#pragma region Serialization

    /** One step-parallel save in flight: the caller's five columns, the covered-node
     *  tally, and the shared mutex's unique hold - released when the plan dies or @c finish
     *  pronounces the columns complete. */
    struct [[nodiscard]] save_plan_t {

        /** The caller-placed columns every step writes disjoint slices of. */
        columns_t columns {};

        /** Nodes the steps covered so far, mutated through @c atomic_ref. */
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

        /** Copies the node range's keys, neighbor IDs & encoded values into the columns -
         *  disjoint slices, so any number of caller threads run their steps concurrently;
         *  every slice starts where the plan's prefix sums say. A refusal of @p codec passes
         *  through.
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

        /** Pronounces the columns complete & releases the hold: @c config_mismatch_k
         *  unless the steps' sizes sum to the node count - overlapping steps that still
         *  sum right are the caller's own lie. A success spends the plan, so a second call
         *  answers @c uninitialized_k instead of unlocking a hold it no longer owns.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is empty or already spent.
         *  - @c config_mismatch_k The steps covered other than every node.
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
     *  @brief Opens a step-parallel save.
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
     *      side for the tally; live mutators and open plans answer @c contended_k at entry.
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
     *  @brief Serializes the snapshot into @p columns as one plan with a single step.
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

    /** One step-parallel load in flight - the mirror of @c save_plan_t: the vetted columns, the
     *  covered-node tally, and the shared mutex's unique hold, released when the plan dies or
     *  @c finish files the keys and publishes the loaded graph. */
    struct [[nodiscard]] load_plan_t {

        /** The vetted columns being decoded. */
        columns_view_t columns {};

        /** Nodes the steps covered so far, mutated through @c atomic_ref. */
        std::size_t covered = 0;

        load_plan_t() noexcept = default;
        load_plan_t(ram_index* index, columns_view_t columns, unique_lock_t&& hold) noexcept
            : columns(columns), index_(index), hold_(std::move(hold)) {}
        load_plan_t(load_plan_t&& other) noexcept
            : columns(other.columns), covered(other.covered), index_(std::exchange(other.index_, nullptr)),
              hold_(std::move(other.hold_)) {}
        load_plan_t& operator=(load_plan_t&&) = delete;

        /** Decodes the node range's keys, neighbor lists & values into its pre-constructed records,
         *  which no other step touches, so any number of caller threads run their steps; the keys
         *  enter the multimap in @c finish. A refusal of @p codec passes through.
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

        /** Publishes the loaded graph in the tier-wide order - keys, claims, identity,
         *  watermark last - and releases the hold: @c config_mismatch_k unless the
         *  steps' sizes sum to the node count. The hold makes it the map's one writer.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is empty or already published.
         *  - @c config_mismatch_k The steps covered other than every node.
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
     *  @brief Opens a step-parallel load into this empty index: takes the shared mutex's unique
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
        // The gate excludes mutators & flips certified probes to refusal; the exclusivity
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
     *  @brief Loads a snapshot into this empty index as one plan with a single step -
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
        // The steps run between this add and the turn, so a slow scorer delays the batches
        // dispensed after this one.
        node_id_t const first_id = atomic_ref<node_id_t>(next_dispensed_id_)
                                       .fetch_add(static_cast<node_id_t>(count), std::memory_order_acq_rel);
        iota_view<node_id_t> const range {first_id, static_cast<node_id_t>(first_id + count)};
        return {this, range, std::move(claim.hold_)};
    }

    /** An audit over the IDs below @p visible, its bitsets placed by @p allocator. */
    template <typename bits_allocator_type_>
    expected<health_plan<bits_allocator_type_>> plan_audit_(node_id_t visible,
                                                            bits_allocator_type_ const& allocator) noexcept {
        using audit_t = health_plan<bits_allocator_type_>;
        using levels_allocator_t = typename audit_t::levels_allocator_t;
        // An empty graph is audited, not refused: no words to sweep, no steps, zero reached.
        std::size_t const words = divide_round_up<std::size_t>(visible, audit_t::lanes_k);
        typename audit_t::bits_t bits {allocator};
        typename audit_t::levels_t levels {levels_allocator_t {allocator}},
            back_levels {levels_allocator_t {allocator}}, parents {levels_allocator_t {allocator}};
        if (failed(bits.resize(audit_t::bitsets_k * words)) || failed(levels.resize(visible)) ||
            failed(back_levels.resize(visible)) || failed(parents.resize(visible)))
            return {audit_t {allocator}, status_t::bad_alloc_k};
        return {
            audit_t {*this, visible, std::move(bits), std::move(levels), std::move(back_levels), std::move(parents)},
            status_t::success_k};
    }

    /** What @ref plan_refine and @ref plan_absorb share: every check, hold, claim and reservation
     *  of a pass over @p nodes, or over the appended ones once @p donors move in, every neighbor
     *  list reserved to @p connectivity. */
    expected<refine_plan_t> plan_pass_(iota_view<node_id_t> nodes, std::span<ram_index*> donors,
                                       traversal_context_t& context, std::size_t connectivity,
                                       refine_config_t config) noexcept {
        using phase_t = typename refine_plan_t::phase_t;
        refine_plan_t plan {context.allocator};
        auto refuse = [&](status_t status) noexcept {
            plan.release_();
            return expected<refine_plan_t> {std::move(plan), status};
        };
        if (!connectivity) return refuse(status_t::config_mismatch_k);
        for (ram_index const* donor : donors)
            if (!donor || donor == this) return refuse(status_t::config_mismatch_k);
        if (status_t const sized = reserve_traversal(context, connectivity, {config.expansion_factor}); failed(sized))
            return refuse(sized);

        // From here the plan reopens the claims on any refusal, and at its death.
        plan.hold_ = shared_lock_t {shared_mutex_};
        if (status_t const closed = close_(); failed(closed)) return refuse(closed);
        plan.index_ = this;
        if (failed(plan.donor_holds_.reserve_exactly(donors.size()))) return refuse(status_t::bad_alloc_k);
        node_id_t const visible = static_cast<node_id_t>(size());
        std::size_t entries = visible, parts = visible != 0;
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
            parts += donor_entries != 0;
        }
        if (status_t const covered = nodes_.cover(entries); failed(covered)) return refuse(covered);
        if (status_t const keyed = key_map_.reserve(entries); failed(keyed)) return refuse(keyed);
        if (parts > 1) {
            if (failed(plan.part_bases_.reserve_exactly(parts))) return refuse(status_t::bad_alloc_k);
            if (visible) plan.part_bases_.push_back_reserved(0);
            std::size_t base = visible;
            for (ram_index const* donor : donors) {
                if (donor->size()) plan.part_bases_.push_back_reserved(static_cast<node_id_t>(base));
                base += donor->size();
            }
        }
        auto [audit, audit_status] = plan_audit_(static_cast<node_id_t>(entries), context.allocator);
        if (failed(audit_status)) return refuse(audit_status);
        plan.audit = std::move(audit);

        // Neighbor lists last, under their locks, as searches copy them meanwhile; the donors are
        // held.
        for (node_id_t node_id = 0; node_id != visible; ++node_id) {
            node_t& node = nodes_[node_id];
            node.lock();
            status_t const grown = node.neighbors.reserve_exactly(connectivity);
            node.unlock();
            if (failed(grown)) return refuse(grown);
        }
        for (ram_index* donor : donors)
            for (std::size_t node = 0; node != donor->size(); ++node)
                if (status_t const grown = donor->nodes_[node].neighbors.reserve_exactly(connectivity); failed(grown))
                    return refuse(grown);

        widen_(connectivity);
        for (ram_index const* donor : donors) widen_(donor->connectivity());
        if (!donors.empty()) nodes = {visible, static_cast<node_id_t>(entries)};
        node_id_t const bound = min_of(nodes.bound, static_cast<node_id_t>(entries));
        plan.nodes_ = {min_of(nodes.first, bound), bound};
        plan.donors_ = donors;
        plan.planned_ = plan.nodes_.size();
        plan.walk_ = resolved_config_({config.expansion_factor}, connectivity);
        plan.connectivity_ = connectivity;
        // One tile serves every pass: of the planned range, or of the largest donor.
        std::size_t forward = donors.empty() ? plan.nodes_.size() : 0;
        for (ram_index const* donor : donors) forward = max_of(forward, donor->size());
        std::size_t const tile_nodes = plan.tile_nodes_for_(entries, forward, config.memory_limit);
        if ((forward && !tile_nodes) || failed(plan.reserve_merge_(entries, tile_nodes)))
            return refuse(status_t::bad_alloc_k);
        if (donors.empty()) plan.open_pass_();
        else plan.phase_ = phase_t::copy_k;
        return {std::move(plan), status_t::success_k};
    }

    /** What @ref plan_split does: every check, hold, claim, cover and reservation of a split of
     *  this index into @p receivers, every neighbor list reserved to @p connectivity. */
    expected<refine_plan_t> plan_split_(std::span<ram_index*> receivers, traversal_context_t& context,
                                        std::size_t connectivity, refine_config_t config, u64_t seed) noexcept {
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
        if (status_t const sized = reserve_traversal(context, connectivity, {config.expansion_factor}); failed(sized))
            return refuse(sized);

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
        // One audit over a full part sweeps every receiver in turn, rebound as its turn comes, and
        // one tile over a full part serves every receiver's sweep.
        auto [audit, audit_status] = plan_audit_(static_cast<node_id_t>(capacity), context.allocator);
        if (failed(audit_status)) return refuse(audit_status);
        plan.audit = std::move(audit);
        plan.connectivity_ = connectivity;
        std::size_t const tile_nodes = plan.tile_nodes_for_(capacity, capacity, config.memory_limit);
        if ((capacity && !tile_nodes) || failed(plan.reserve_merge_(capacity, tile_nodes)))
            return refuse(status_t::bad_alloc_k);

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
        // Neighbor lists last; each moves its buffer into a receiver, where re-selecting allocates
        // nothing.
        for (node_id_t node_id = 0; node_id != entries; ++node_id)
            if (status_t const grown = nodes_[node_id].neighbors.reserve_exactly(connectivity); failed(grown))
                return refuse(grown);

        for (ram_index* receiver : receivers) {
            receiver->widen_(connectivity);
            receiver->widen_(this->connectivity());
        }
        plan.index_ = receivers[0];
        plan.seed_ = seed;
        plan.walk_ = resolved_config_({config.expansion_factor}, connectivity);
        plan.connectivity_ = connectivity;
        plan.phase_ = phase_t::partition_k;
        return {std::move(plan), status_t::success_k};
    }

    /**
     *  @brief One node's re-selection: a walk across into @p across alone fills the candidates
     *      first, seeded by @p across_seeds ahead of its random landmarks, then the node's own
     *      neighbor list seeds a walk over every ID below @p visible and enters the candidates
     *      past that walk's breadth; @p selector picks from them into
     *      @c context.selected_neighbors, which the caller merges or writes.
     *
     *  No metric verb runs under a lock, and no neighbor list allocates.
     */
    template <typename metric_type_, typename context_type_, typename selector_type_>
    void refine_one_(node_id_t node_id, node_id_t visible, iota_view<node_id_t> across,
                     std::span<node_id_t const> across_seeds, insert_config_t walk, metric_type_& metric,
                     context_type_& context, selector_type_ const& selector) noexcept {
        constexpr synchronization_t concurrent_k = synchronization_t::concurrent_k;
        value_t const& value = published_value_of(node_id);
        context.results.clear();
        context.frontier.clear();
        context.visited.clear();
        // Visited before anything stages, so the node never enters its own candidates.
        [[maybe_unused]] membership_t const self_marked = context.visited.insert(node_id);
        part_admission_t const other {across.first, min_of(across.bound, visible)};
        if (other.first_id < other.first_invisible_id) {
            search_around_<concurrent_k>(value, metric, other, walk.expansion_factor, selector.connectivity,
                                         context.results, context, dummy_fold_t {}, across_seeds);
            context.frontier.clear();
        }
        snapshot_admission_t const every {visible};
        seed_from_neighbors_(node_id, value, metric, every, walk.expansion_factor, context);
        search_around_<concurrent_k>(value, metric, every, walk.expansion_factor, selector.connectivity,
                                     context.results, context, dummy_fold_t {});
        add_neighbor_candidates_(node_id, value, metric, every, context);
        selector(*this, metric, context);
    }

    /**
     *  @brief Writes @c context.selected_neighbors, at most @p connectivity of them, into
     *      @p node_id 's list under its lock, and snapshots the list into
     *      @c context.reverse_ids for the back-links.
     *
     *  The set gained a member only when a pick was absent before; a shrink alone gains none.
     *
     *  @return Whether the list gained a member.
     */
    template <typename context_type_>
    bool rewrite_list_(node_id_t node_id, std::size_t connectivity, context_type_& context) noexcept {
        node_t& node = nodes_[node_id];
        context.reverse_ids.clear();
        node.lock();
        std::size_t const kept_count = min_of(min_of(context.selected_neighbors.size(), connectivity),
                                              node.neighbors.capacity());
        bool changed = false;
        for (std::size_t kept = 0; kept != kept_count && !changed; ++kept) {
            bool present = false;
            for (node_id_t const neighbor : node.neighbors)
                present = present || neighbor == context.selected_neighbors[kept].node_id;
            changed = !present;
        }
        node.neighbors.clear();
        for (std::size_t kept = 0; kept != kept_count; ++kept) {
            node.neighbors.push_back_reserved(context.selected_neighbors[kept].node_id);
            context.reverse_ids.push_back_reserved(context.selected_neighbors[kept].node_id);
        }
        node.unlock();
        return changed;
    }

    /** Scores every proposal of @p run against @p node_id in place, in as many rounds as the
     *  metric's staging takes; a metric refusing an empty round leaves the rest unbounded. */
    template <typename metric_type_, typename context_type_>
    void score_run_(node_id_t node_id, std::span<scored_node_t> run, metric_type_& metric,
                    context_type_& context) noexcept {
        value_t const& value = published_value_of(node_id);
        for (std::size_t first = 0; first != run.size();) {
            metric.reset(value);
            std::size_t staged_count = 0;
            while (first + staged_count != run.size() &&
                   metric.push(published_value_of(run[first + staged_count].node_id)) == metric_staging_t::staged_k)
                ++staged_count;
            if (!staged_count) {
                for (; first != run.size(); ++first) run[first].distance = unbounded_radius<distance_t>();
                return;
            }
            drain_round(metric, staged_count, context, unbounded_radius<distance_t>());
            for (std::size_t lane = 0; lane != staged_count; ++lane)
                run[first + lane].distance = context.staged_distances[lane];
            first += staged_count;
        }
    }

    /** Adds the scored candidates of @p scored the results lack, in order, until the
     *  results' reservation is full. @return How many of them were consumed, the ones
     *  listed already included. */
    template <typename results_type_>
    static std::size_t add_scored_candidates_(std::span<scored_node_t const> scored, results_type_& results) noexcept {
        std::size_t consumed = 0;
        for (; consumed != scored.size() && results.size() != results.capacity(); ++consumed) {
            scored_node_t const candidate = scored[consumed];
            bool listed = false;
            for (std::size_t position = 0; position != results.size() && !listed; ++position)
                listed = results[position].node_id == candidate.node_id;
            if (!listed) results.insert(scored_node_t {candidate}, results.size() + 1);
        }
        return consumed;
    }

    /**
     *  @brief One node's merge: its live neighbors, its walk's @p picks and the @p incoming
     *      proposals compete, @p selector fills the list, the audit's levels keep its
     *      reachability, and it is written once.
     *
     *  The picks and the proposals carry their distances, so only the live neighbors are scored.
     *  The sources fold in rounds within the candidates' reservation, each round's selection
     *  carrying into the next, so no proposal is dropped however many a hub draws.
     *
     *  @return Whether the list gained a member.
     */
    template <typename metric_type_, typename context_type_, typename selector_type_>
    bool merge_back_links_(node_id_t node_id, std::span<scored_node_t const> picks,
                           std::span<scored_node_t const> incoming, health_plan_t& audit, metric_type_& metric,
                           context_type_& context, selector_type_ const& selector) noexcept {
        value_t const& value = published_value_of(node_id);
        snapshot_admission_t const every {audit.visible};
        typename context_type_::results_array_t& results = context.results;
        // The live neighbors scored first and kept apart, with their distances, for the levels.
        results.clear();
        add_neighbor_candidates_(node_id, value, metric, every, context);
        typename context_type_::results_array_t& live = context.admitted;
        live.clear();
        for (std::size_t position = 0; position != results.size(); ++position)
            live.insert(scored_node_t {results[position]}, min_of(live.size() + 1, live.capacity()));
        results.clear();

        std::span<scored_node_t const> members = to_span(live);
        std::size_t const refines_before = context.telemetry.computed_distances_in_refines;
        for (bool folding = true; folding;) {
            members = members.subspan(add_scored_candidates_(members, results));
            picks = picks.subspan(add_scored_candidates_(picks, results));
            incoming = incoming.subspan(add_scored_candidates_(incoming, results));
            selector(*this, metric, context);
            folding = !members.empty() || !picks.empty() || !incoming.empty();
            if (!folding) break;
            results.clear();
            for (std::size_t position = 0; position != context.selected_neighbors.size(); ++position)
                results.insert(scored_node_t {context.selected_neighbors[position]}, results.size() + 1);
        }
        // The selector counts a refine; re-selecting a list from its proposals is a reverse refine.
        std::size_t const reselecting = context.telemetry.computed_distances_in_refines - refines_before;
        context.telemetry.computed_distances_in_refines = refines_before;
        context.telemetry.computed_distances_in_reverse_refines += reselecting;
        keep_levels_(node_id, live, context.selected_neighbors, selector.connectivity, audit);
        return rewrite_list_(node_id, selector.connectivity, context);
    }

    /**
     *  @brief Puts back what @p selected dropped of the @p live neighbors that reachability needs:
     *      a neighbor whose last parent @p node_id is, and, when no pick stands one reverse level
     *      closer to node zero, the node's nearest live neighbor that does. A list short of
     *      @p connectivity grows by each; a full one gives up its farthest pick that is kept by
     *      neither, a live neighbor only once another parent keeps it.
     *
     *  By induction over the levels every node reached both ways stays so throughout the merge: a
     *  node at a level keeps one in-link from the level before, and one out-link to a lower
     *  reverse level, and node zero never moves.
     */
    template <typename live_type_, typename selected_type_>
    void keep_levels_(node_id_t node_id, live_type_ const& live, selected_type_& selected, std::size_t connectivity,
                      health_plan_t& audit) noexcept {
        constexpr u8_t unknown_level_k = health_plan_t::unknown_level_k;
        u8_t const level = audit.level(node_id), back_level = audit.back_level(node_id);
        // The saturated level counts no parents, so its children are never released.
        u8_t const child_level = level < unknown_level_k - 1 ? static_cast<u8_t>(level + 1) : unknown_level_k;
        auto position_in = []<typename among_type_>(among_type_ const& among, node_id_t member) noexcept {
            std::size_t position = 0;
            while (position != among.size() && among[position].node_id != member) ++position;
            return position;
        };
        auto releasable = [&](node_id_t member) noexcept {
            return child_level == unknown_level_k || audit.level(member) != child_level ||
                   position_in(live, member) == live.size() || audit.release_parent(member);
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
        if (child_level != unknown_level_k)
            for (std::size_t position = 0; position != live.size(); ++position) {
                scored_node_t const member = live[position];
                if (audit.level(member.node_id) != child_level ||
                    position_in(selected, member.node_id) != selected.size())
                    continue;
                if (!audit.release_parent(member.node_id)) keep(member);
            }
        if (back_level != unknown_level_k && back_level != 0) {
            std::size_t standing = 0;
            while (standing != selected.size() && audit.back_level(selected[standing].node_id) >= back_level)
                ++standing;
            scored_node_t way_back = standing != selected.size()
                                         ? selected[standing]
                                         : scored_node_t {missing_k, unbounded_radius<distance_t>()};
            for (std::size_t position = 0; position != live.size() && standing == selected.size(); ++position)
                if (audit.back_level(live[position].node_id) < back_level &&
                    live[position].distance < way_back.distance)
                    way_back = live[position];
            if (way_back.node_id != missing_k) keep(way_back);
        }
    }

    /** Seeds a re-selection's walk with @p node_id 's own neighbor list: its admitted, unvisited
     *  neighbors score into the candidates and the frontier, in as many rounds as the metric's
     *  staging takes, so every link already wired competes. A refused copy seeds nothing, and the
     *  walk samples landmarks instead. */
    template <typename metric_type_, typename admission_type_, typename context_type_>
    void seed_from_neighbors_(node_id_t node_id, value_t const& value, metric_type_& metric,
                              admission_type_ const& admission, std::size_t expansion_factor,
                              context_type_& context) noexcept {
        context.staged_ids.clear();
        auto const [degree, copy_status] = copy_neighbors_<synchronization_t::concurrent_k>(node_id, context);
        if (failed(copy_status) || failed(context.visited.reserve(context.visited.size() + degree))) {
            ++context.telemetry.saturated_walks;
            return;
        }
        // Each round compacts its IDs into the prefix of the neighbor list copy it has already read
        // past.
        for (std::size_t position = 0; position != degree;) {
            metric.reset(value);
            std::size_t staged_count = 0;
            for (; position != degree; ++position) {
                node_id_t const neighbor_id = context.staged_ids[position];
                if (!admission.admits(neighbor_id) || context.visited.contains(neighbor_id)) continue;
                if (metric.push(published_value_of(neighbor_id)) == metric_staging_t::refused_k) break;
                [[maybe_unused]] membership_t const marked = context.visited.insert(neighbor_id);
                context.staged_ids[staged_count++] = neighbor_id;
            }
            if (!staged_count) return;
            drain_round(metric, staged_count, context, unbounded_radius<distance_t>());
            for (std::size_t lane = 0; lane != staged_count; ++lane) {
                scored_node_t const scored {context.staged_ids[lane], context.staged_distances[lane]};
                [[maybe_unused]] bool const ranked = context.results.insert(scored_node_t {scored}, expansion_factor);
                if (failed(context.frontier.insert(scored_node_t {scored}))) {
                    ++context.telemetry.saturated_walks;
                    return;
                }
            }
        }
    }

    /** Adds @p node_id 's current neighbor list to the candidates a re-selection picks from, past
     *  the walk's breadth: the walk's nearer finds evict the far members, the long links among
     *  them, which must still compete. Scored in as many rounds as the metric's staging takes; a
     *  refused copy or growth, or a metric refusing an empty round, counts a saturated walk. */
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
        // walk wires a thinner node rather than abandoning it.
        // The first node of a virgin index finds nothing to walk to and publishes with no
        // neighbors; a later node walking before any is wired waits for node zero and walks again.
        node_t& first_node = nodes_[0];
        while (new_node_id) {
            std::size_t const saturated_before = context.telemetry.saturated_walks;
            node_id_t const first_invisible_id =
                atomic_ref<node_id_t>(last_visible_id_).load(std::memory_order_acquire);
            search_around_<synchronization_>(
                incoming, metric, dispensed_admission_t {this, first_invisible_id, new_node_id}, walk.expansion_factor,
                connectivity, context.results, context, dummy_fold_t {});
            if (first_invisible_id || !context.results.empty() || context.telemetry.saturated_walks != saturated_before)
                break;
            u32_t const observed = atomic_ref<u32_t>(first_node.state).load(std::memory_order_acquire);
            if (!(observed & node_t::wired_k)) wait_t {}(&first_node.state, observed, 0u);
        }
        selector(*this, metric, context);

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

    /** What a refine's walk across walks: one part's published IDs alone, so the nodes of other
     *  parts, which the pass rewires meanwhile, never route it out. */
    struct part_admission_t {

        /** The part's first ID. */
        node_id_t first_id = 0;

        /** The part's bound, clamped to the watermark. */
        node_id_t first_invisible_id = 0;

        /** Whether the walk may locate @p node_id. */
        bool admits(node_id_t node_id) const noexcept { return first_id <= node_id && node_id < first_invisible_id; }

        /** Landmarks come from the part alone. */
        node_id_t sampling_first() const noexcept { return first_id; }
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
     *  bounds the whole walk; a certified mode proves its own certification through the shared
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
     *  A certified @p synchronization_ compiles the node locks out, its caller proving it. */
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

    /** The watermark at unique acquisition; @c contended_k when the dispenser has
     *  run ahead of the watermark, which only an exclusive-certified mutator lying mid-turn
     *  can do with the gate held. */
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
     *  monotone over their element columns, every neighbor ID naming a node - so the step
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

    /** The world gate: mutators hold it shared, stop-the-world verbs try-unique, certified
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
 *  concurrent mode and as plain copies on the certified ones. A staged list may name IDs at or past
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
