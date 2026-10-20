/**
 *  @file include/usearch/basic/ram_index.hpp
 *  @author Ash Vardanian
 *  @date August 15, 2026
 *  @brief Similarity Search baseline over arbitrary payloads & caller-supplied metrics.
 *
 *  The baseline tier's RAM residency: the readable reference implementation. Arbitrary
 *  payload objects, expensive caller-supplied batched metrics, caller-visible keys,
 *  growable per-node neighbor rows. Standard-library & OS headers only - together with
 *  @c utilities/ this tier ships as a standalone header-only library. Every specialized
 *  optimization - lock-free readers, SIMD kernel kits - belongs to the dense family, not here.
 */
#pragma once

#include <cassert> // `assert` - an unfinished insertion plan's death
#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint64_t`

#include <array>        // `std::array` - a filtered walk's quads
#include <atomic>       // `std::memory_order`
#include <bit>          // `std::popcount`, `std::bit_width` - an audit's reached count, a refine's rounds
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
#include "usearch/utilities/ram_tables.hpp"

namespace unum::usearch::basic {
using namespace unum::usearch::utilities;

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
 *  A round's rows carry no order: results land by slot, and ties break by node ID.
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
 *  all the rounds that follow, and each @c drain writes @b [queries,rows] distances, query @c q to
 *  staged row @c r at `distances[q * rows + r]`, exact within the one radius they share. */
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
 *  @brief Scores rows @c [first_row,end_row) of @p view_of against every query of @p queries into
 *      @p distances, @b [queries,rows] over the rows scored, and answers where those rows end:
 *      @p first_row when the metric refused the first of them.
 *
 *  A metric scoring one query at a time reopens each query over every row, in as many rounds as its
 *  staging takes, each within its own of @p thresholds.
 */
template <typename metric_type_, typename query_type_, typename view_of_type_, typename distance_type_>
std::size_t score_queries(metric_type_& metric, std::span<query_type_ const> queries, view_of_type_&& view_of,
                          std::size_t first_row, std::size_t end_row, std::span<distance_type_ const> thresholds,
                          std::span<distance_type_> distances) noexcept {
    std::size_t const rows = end_row - first_row;
    for (std::size_t query = 0; query != queries.size(); ++query)
        for (std::size_t round_first = first_row; round_first != end_row;) {
            metric.reset(queries[query]);
            std::size_t round_end = round_first;
            while (round_end != end_row && metric.push(view_of(round_end)) == metric_staging_t::staged_k) ++round_end;
            if (round_end == round_first) return first_row;
            metric.drain(distances.subspan(query * rows + round_first - first_row, round_end - round_first),
                         thresholds[query]);
            round_first = round_end;
        }
    return end_row;
}

/** A @c batched_metric opened by @ref open_queries scores every query in one round, within the
 *  loosest of @p thresholds, so the rows end wherever its staging refused. */
template <typename metric_type_, typename query_type_, typename view_of_type_, typename distance_type_>
    requires batched_metric<metric_type_, query_type_>
std::size_t score_queries(metric_type_& metric, std::span<query_type_ const> queries, view_of_type_&& view_of,
                          std::size_t first_row, std::size_t end_row, std::span<distance_type_ const> thresholds,
                          std::span<distance_type_> distances) noexcept {
    std::size_t round_end = first_row;
    while (round_end != end_row && metric.push(view_of(round_end)) == metric_staging_t::staged_k) ++round_end;
    if (round_end == first_row) return first_row;
    distance_type_ loosest = thresholds.front();
    for (distance_type_ const threshold : thresholds) loosest = max_of(loosest, threshold);
    metric.drain(distances.first(queries.size() * (round_end - first_row)), loosest);
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

/** The per-walk counters every traversal accumulates; results report one call's share as the
 *  difference of two snapshots. */
struct traversal_telemetry_t {

    /** Metric invocations, batched rounds counted once per scored candidate: the walks' and the two
     *  subsets below. */
    std::size_t computed_distances = 0;

    /** Frontier candidates whose neighbor lists were expanded. */
    std::size_t visited_members = 0;

    /** Re-seedings after the frontier drained prematurely - a connectivity signal. */
    std::size_t restarts = 0;

    /** Walks a scratch refusal ended early - ranked short, and on insertion wired thinner. */
    std::size_t saturated_walks = 0;

    /** Subset of @c computed_distances ranking a full row and the newcomer against the row's owner
     *  while back-linking. */
    std::size_t computed_distances_in_rankings = 0;

    /** Subset of @c computed_distances a selector spent pruning candidates to a row: an inserted
     *  or refined node's own, and every full row a back-link re-selects. */
    std::size_t computed_distances_in_selections = 0;

    traversal_telemetry_t operator-(traversal_telemetry_t const& before) const noexcept {
        return {computed_distances - before.computed_distances,
                visited_members - before.visited_members,
                restarts - before.restarts,
                saturated_walks - before.saturated_walks,
                computed_distances_in_rankings - before.computed_distances_in_rankings,
                computed_distances_in_selections - before.computed_distances_in_selections};
    }
};

/** Outcome of one refinement or heal pass, counted as the dense tier counts it, with the walks'
 *  spend as every basic verb reports it. */
struct [[nodiscard]] refine_result_t {

    /** Nodes re-selected: the planned range, then the islands of every round. */
    std::size_t visited_nodes = 0;

    /** Of those, the nodes whose neighbors differ as a set from before - never more than
     *  @c visited_nodes. */
    std::size_t changed_nodes = 0;

    /** Nodes the first audit left unreached, counted once. */
    std::size_t islands_found = 0;

    /** Links actually written between an island and a node the audit proved both ways. */
    std::size_t islands_reconnected = 0;

    /** The audit closing the heal, as a fresh @c plan_health sweep right after would take it. */
    health_result_t health;

    /** The re-selection walks' and back-links' spend. */
    traversal_telemetry_t telemetry;
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
 *      both admit joins @c admitted, at most @c breadth, while the walk routes through all.
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
        // A refused growth leaves the backfill alone to fill the row.
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
                context.telemetry.computed_distances_in_selections += staged_count;
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

/** A reference spelling the verbs the index posts past the standard's - the no-return add,
 *  subtract and bit clear, and the bounded add - as ForkUnion's do. For the standard's own
 *  reference, which does not, the functions below spell them; ForkUnion's references carry the
 *  same names as free functions in their namespace, reached through argument lookup, so a site
 *  reads the same whichever reference the index was given. */
template <typename reference_type_, typename value_type_>
concept extended_atomic_ref = requires(reference_type_ reference, value_type_ value) {
    reference.add(value, std::memory_order_relaxed);
    reference.sub(value, std::memory_order_relaxed);
    reference.clear_bits(value, std::memory_order_relaxed);
    reference.fetch_add_if_at_most(value, value, std::memory_order_relaxed);
};

/** A reference spelling only the standard's verbs, such as @c std::atomic_ref, for which the
 *  functions below spell the rest. */
template <typename reference_type_, typename value_type_>
concept plain_atomic_ref = !extended_atomic_ref<reference_type_, value_type_>;

template <typename reference_type_, typename value_type_>
    requires plain_atomic_ref<reference_type_, value_type_>
void atomic_add(reference_type_ reference, value_type_ operand, std::memory_order order) noexcept {
    reference.fetch_add(operand, order);
}
template <typename reference_type_, typename value_type_>
    requires plain_atomic_ref<reference_type_, value_type_>
void atomic_sub(reference_type_ reference, value_type_ operand, std::memory_order order) noexcept {
    reference.fetch_sub(operand, order);
}
template <typename reference_type_, typename value_type_>
    requires plain_atomic_ref<reference_type_, value_type_>
void atomic_clear_bits(reference_type_ reference, value_type_ bits, std::memory_order order) noexcept {
    reference.fetch_and(static_cast<value_type_>(~bits), order);
}

/** Adds @p operand only if the sum stays at most @p limit; returns the value held before. */
template <typename reference_type_, typename value_type_>
    requires plain_atomic_ref<reference_type_, value_type_>
value_type_ atomic_fetch_add_if_at_most(reference_type_ reference, value_type_ operand, value_type_ limit,
                                        std::memory_order order) noexcept {
    value_type_ observed = reference.load(std::memory_order_relaxed);
    while (observed <= limit && limit - observed >= operand &&
           !reference.compare_exchange_weak(observed, observed + operand, order, std::memory_order_relaxed)) {}
    return observed;
}

/** Raises the word to @p operand if it is larger; returns the value held before. */
template <typename reference_type_, typename value_type_>
    requires plain_atomic_ref<reference_type_, value_type_>
value_type_ atomic_fetch_max(reference_type_ reference, value_type_ operand, std::memory_order order) noexcept {
    value_type_ observed = reference.load(std::memory_order_relaxed);
    while (observed < operand &&
           !reference.compare_exchange_weak(observed, operand, order, std::memory_order_relaxed)) {}
    return observed;
}
template <typename reference_type_, typename value_type_>
    requires extended_atomic_ref<reference_type_, value_type_>
value_type_ atomic_fetch_max(reference_type_ reference, value_type_ operand, std::memory_order order) noexcept {
    return reference.fetch_max(operand, order);
}

/**
 *  @brief Shared among mutators, try-unique for stop-the-world, wait-free to probe - one
 *      @c std::atomic word, so every reading is a consistent joint snapshot.
 *
 *  The word packs forty-one generation bits on top, a dedicated unique bit, and twenty-two
 *  holder bits - the OS pid ceiling bounds simultaneous holders well below that. The
 *  generation steps on every shared @b and unique acquisition: a whole acquisition that
 *  opens and closes between two probes still moves it, and wrapping it inside one probe
 *  bracket would need days of saturated acquisitions. With the generation on top its carry
 *  leaves the word, so the shared side is one blind add of a holder and a step, backed off
 *  when the unique bit was already set; only the unique side compare-exchanges, from a
 *  free word.
 */
template <typename wait_type_ = dummy_waiter_t, template <typename> class atomic_ref_template_ = standard_atomic_ref>
struct shared_mutex {
    template <typename word_type_>
    using atomic_ref = atomic_ref_template_<word_type_>;

    u64_t word {0};

    /** Bits counting holders: Linux's @c PID_MAX_LIMIT, 2^22, bounds the threads alive at once. */
    inline static constexpr std::size_t holder_bits_k = 22;

    inline static constexpr u64_t holders_mask_k = (u64_t {1} << holder_bits_k) - 1;
    inline static constexpr u64_t unique_bit_k = u64_t {1} << holder_bits_k;
    inline static constexpr u64_t generation_unit_k = unique_bit_k << 1;
    inline static constexpr u64_t generation_mask_k = ~(generation_unit_k - 1);

    /** One consistent reading: how the gate is held & how often it has been. */
    struct snapshot_t {
        u64_t generation = 0;
        std::size_t holders = 0;
        bool unique = false;

        bool unlocked() const noexcept { return !holders && !unique; }
    };

    /** Joins the shared side, waiting out a unique holder through the waiter. */
    void lock_shared() noexcept {
        annotate_lock_shared_started(&word);
        // A masked compare-exchange kept the fields apart while the generation sat under the
        // unique bit; it retried under contention where this one add does not.
        while (true) {
            u64_t const observed = atomic_ref<u64_t>(word).fetch_add(generation_unit_k + 1, std::memory_order_acquire);
            if (!(observed & unique_bit_k)) break;
            // Joined behind a unique holder: hand the holder back, then wait for the word to move.
            u64_t const backed_off = atomic_ref<u64_t>(word).fetch_sub(1, std::memory_order_release) - 1;
            wait_type_ {}(&word, backed_off, 0u);
        }
        // Pairs with the acquire fence before a certified probe's exit snapshot: a row read that
        // returned a store made under this hold proves the acquisition to that snapshot, which
        // can then no longer read the word from before it. Arm and POWER need the fence as much as
        // the C++ model does: an acquire orders the add's read before the row stores, not its write.
        std::atomic_thread_fence(std::memory_order_release);
        annotate_lock_shared_acquired(&word);
    }

    void unlock_shared() noexcept {
        // The raw release precedes the bracket: inside it TSan ignores the word's own
        // atomics, and the certified probes' acquire would otherwise pair with nothing.
        annotate_release(&word);
        annotate_unlock_shared_started(&word);
        atomic_sub(atomic_ref<u64_t>(word), u64_t {1}, std::memory_order_release);
        annotate_unlock_shared_finished(&word);
    }

    /** Takes the unique side only from a genuinely free word - generation churn
     *  retries, it is not contention - and never waits: a refusal is the answer. */
    [[nodiscard]] bool try_lock() noexcept {
        annotate_try_lock_started(&word);
        u64_t observed = atomic_ref<u64_t>(word).load(std::memory_order_relaxed);
        while ((observed & (holders_mask_k | unique_bit_k)) == 0)
            if (atomic_ref<u64_t>(word).compare_exchange_weak(observed, (observed + generation_unit_k) | unique_bit_k,
                                                              std::memory_order_acquire, std::memory_order_relaxed)) {
                annotate_try_lock_acquired(&word);
                return true;
            }
        annotate_try_lock_refused(&word);
        return false;
    }

    void unlock() noexcept {
        annotate_release(&word);
        annotate_unlock_started(&word);
        atomic_clear_bits(atomic_ref<u64_t>(word), unique_bit_k, std::memory_order_release);
        annotate_unlock_finished(&word);
    }

    /** The wait-free probe the certified readers bracket their walks with. */
    snapshot_t snapshot() noexcept {
        u64_t const observed = atomic_ref<u64_t const>(word).load(std::memory_order_acquire);
        annotate_acquire(&word);
        return {observed & generation_mask_k, static_cast<std::size_t>(observed & holders_mask_k),
                (observed & unique_bit_k) != 0};
    }
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

    /** IDs of one staged fan-out - the batch a single metric round scores; also
     *  the flood stack of `health` and the row copies of back-linking. */
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

    /** Running per-walk counters; calls report their share as snapshot deltas. */
    traversal_telemetry_t telemetry;

    /** The allocator the containers above rebind; a plan opened through the context places its own
     *  state with it. */
    allocator_t allocator;

    /** Every container gets a rebound copy of @p allocator, so scratch lands where
     *  the caller places it; seed distinctly per thread to decorrelate the walks. */
    explicit traversal_context(u64_t seed = 0, allocator_t allocator = {}) noexcept
        : results(scored_allocator_t(allocator)), frontier(scored_allocator_t(allocator)),
          visited(visits_allocator_t(allocator)), random_generator {seed}, staged_ids(allocator),
          staged_distances(allocator), selected_neighbors(allocator), reverse_ids(allocator),
          admitted(scored_allocator_t(allocator)), allocator(allocator) {}
};

/** Row & element counts of the five columns a snapshot needs - what a container sizes. */
struct columns_extents_t {

    /** Published nodes, the row count of every node-sized column. */
    std::size_t nodes = 0;

    /** Neighbor IDs across every row. */
    std::size_t neighbors = 0;

    /** Codec bytes across every value. */
    std::size_t value_bytes = 0;
};

/** The five caller-placed columns a basic-tier snapshot is: keys, neighbor prefix sums,
 *  neighbor IDs, value prefix sums, and codec bytes - each an independent buffer, so a
 *  container lays them out wherever it likes. Both prefix-sum columns hold @b nodes+1 sums
 *  from zero, so row @c i owns the slice between sums @c i and @c i+1. */
template <typename key_type_, typename node_id_type_, typename byte_type_, typename offset_type_ = u64_t>
struct columns {

    /** One key per row. */
    std::span<key_type_> keys;

    /** @b nodes+1 sums of neighbor IDs, from zero. */
    std::span<offset_type_> neighbor_offsets;

    /** Every row's neighbor IDs, back to back. */
    std::span<node_id_type_> neighbor_ids;

    /** @b nodes+1 sums of codec bytes, from zero. */
    std::span<offset_type_> value_offsets;

    /** Every row's codec bytes, back to back. */
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
 *  The design optimizes for being read, and its storage is a column of node records in
 *  chunks that grow on demand and never move, plus the key multimap. Each record owns its
 *  neighbors row, which starts empty and grows with wiring - automatic through @c insert's
 *  selector, which carries the neighbors-per-node budget, or manual through @c link. Metric
 *  calls cost microseconds to milliseconds here, so brief per-node locks are noise -
 *  the only hard rule is that no metric verb ever runs while any row lock is held - the
 *  shared mutex is deliberately held across whole batches, so a parked metric starves
 *  stop-the-world verbs, and delays the publication of the batches dispensed after its own.
 *
 *  Concurrency runs on plain primitives: batch capacity is claimed by a bounded add and
 *  covered before anything is dispensed, one more add dispenses the batch's whole ID range,
 *  any number of threads land its positions, and one turn of a watermark advanced in
 *  dispensed order publishes that range at once, so a search, a cursor or an ID lookup sees
 *  all of a batch or none of it. A batch's keys land in the multimap only after its turn, so
 *  a key an insert made findable names a published node; a load registers its keys before
 *  @c finish publishes, while the caller keeps readers out. Keys need not be unique: every
 *  holder of a key stays addressable through @c find_each, and ID → key stays authoritative.
 *  Node IDs are never recycled before a @c clear and removals are deleted, keeping every
 *  reader's stale ID valid until then.
 *
 *  @tparam value_type_ any payload satisfying @c basic::value; immovable once published.
 *  @tparam key_type_ trivially-copyable caller-visible identity.
 *  @tparam node_id_type_ graph-local identity, dense from zero; @c u32_t or @c u64_t.
 *  @tparam distance_type_ the metrics' result element.
 *  @tparam wait_type_ the monitored-wait engaged on contended words.
 *  @tparam atomic_ref_template_ the atomic reference owning the plain synchronization words;
 *      @c standard_atomic_ref by default, ForkUnion's ISA-specific references plug in; either binds a const word
 *      for the const readers' loads.
 *  @tparam allocator_type_ places the node records, neighbor buffers & key multimap, and builds
 *      every stored value through uses-allocator construction.
 *  @tparam cache_hints_type_ the line hints streaming the record a walk expands next toward the
 *      core.
 */
template <                                                                //
    typename value_type_,                                                 //
    typename key_type_ = uuid_t,                                          //
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

    using neighbors_buffer_t = growing_buffer<node_id_t, allocator_t>;

    /** One node record: its state, the payload, its key and the neighbors it links to. The value &
     *  key are written before the row is wired and never change until a @c clear; only @c neighbors
     *  mutates after - grown or re-selected by later inserts' back-links, grown or trimmed by
     *  manual linking - with its own @c size() as the live degree. */
    struct node_t {

        /** Set with a release once an insert wired the row, ahead of its batch's turn, so another
         *  batch's walk may admit it early. A load never sets it, since its rows are reached below
         *  the watermark alone, and only @c clear, which recycles the IDs, resets it. */
        inline static constexpr u32_t wired_k = 1;

        /** Held by whoever edits @c neighbors once the row is reachable. */
        inline static constexpr u32_t held_k = 2;

        /** The row's wiring and its lock in one word, through @c atomic_ref_template_. */
        u32_t state {0};
        value_t value;
        key_t key {};
        neighbors_buffer_t neighbors;

        // No default form: it would seed the row with the default allocator, silently
        // escaping whatever allocator the index was constructed with.
        node_t() = delete;
        explicit node_t(allocator_t const& allocator) noexcept
            : value(std::make_obj_using_allocator<value_t>(allocator)), neighbors(allocator) {}

        /** Waits out a holder, then holds the row for an edit of @c neighbors. */
        void lock() noexcept {
            annotate_lock_started(&state);
            // Claim only after observing the row free: on Arm a losing claim still takes the line
            // exclusive, stealing it from the holder; the peek keeps contenders in Shared state.
            atomic_ref<u32_t> word {state};
            while (true) {
                u32_t observed = word.load(std::memory_order_relaxed);
                for (; observed & held_k; observed = word.load(std::memory_order_relaxed))
                    wait_t {}(&state, observed, 0u);
                if (word.compare_exchange_weak(observed, observed | held_k, std::memory_order_acquire,
                                               std::memory_order_relaxed))
                    break;
            }
            annotate_lock_acquired(&state);
        }

        void unlock() noexcept {
            annotate_unlock_started(&state);
            atomic_clear_bits(atomic_ref<u32_t>(state), held_k, std::memory_order_release);
            annotate_unlock_finished(&state);
        }

        /** Publishes the row to other batches' walks; nothing else reaches it before this. */
        void wire() noexcept { atomic_ref<u32_t>(state).store(wired_k, std::memory_order_release); }

        /** Whether an insert wired the row, carrying its value and neighbors with the acquire. */
        bool wired() const noexcept { return atomic_ref<u32_t const>(state).load(std::memory_order_acquire) & wired_k; }
    };

  public:

    /** Ceiling on dispensable node IDs: every ID the type holds but the all-ones sentinel, and no
     *  more records than the address space holds. */
    inline static constexpr std::size_t max_node_count_k = static_cast<std::size_t>(
        min_of<u64_t>(min_of<u64_t>(std::numeric_limits<node_id_t>::max(), std::numeric_limits<std::size_t>::max()),
                      (u64_t {1} << address_bits_k) / sizeof(node_t)));

  private:

    /** The chunks spanning @c max_node_count_k; the first one's rows are also one plan step's. */
    using nodes_schedule_t = chunk_schedule_spanning<max_node_count_k>;

    /** The node records in chunks on @c nodes_schedule_t: a cover builds every record of a fresh
     *  chunk from the index allocator, so slots exist before publication with no per-slot
     *  lifecycle tracking, and none ever moves. */
    using nodes_column_t =
        chunked_column<node_t, nodes_schedule_t, max_node_count_k, allocator_t, atomic_ref_template_>;

    /** The claim count while a refine holds the claims closed: past every ceiling, so the bounded
     *  add of every batch refuses it. */
    inline static constexpr std::size_t closed_claims_k = std::numeric_limits<std::size_t>::max();

    /** The all-ones ID no node ever takes: a heal's answer for no target. */
    inline static constexpr node_id_t missing_k = std::numeric_limits<node_id_t>::max();

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

    /** The allocator placing every record, neighbor row and the key multimap, and building every
     *  stored value - the one a caller builds the values it inserts with. */
    allocator_t get_allocator() const noexcept { return allocator_; }

    /** Covers the node records & the key multimap for @p capacity entries ahead of the inserts,
     *  which cover what they claim on their own, so this is a hint callable any time beside every
     *  other verb. Neighbor rows start empty and grow with wiring. Answers the node capacity
     *  covered so far: @c overflow_k past @c max_node_count_k, and @c bad_alloc_k keeps the chunks
     *  it did cover, so the same call retries. */
    expected<std::size_t> reserve(std::size_t capacity) noexcept {
        if (status_t const covered = nodes_.cover(capacity); failed(covered)) return {nodes_.capacity(), covered};
        if (status_t const keyed = key_map_.reserve(capacity); failed(keyed)) return {nodes_.capacity(), keyed};
        return {nodes_.capacity(), status_t::success_k};
    }

    /** Forgets every node and key and keeps every covered chunk for the next build: each record
     *  below the dispenser gets a fresh value from the index allocator, an empty row and no key.
     *  Takes the shared mutex's unique side - live mutators and open plans answer @c contended_k -
     *  while the caller keeps readers out, as for @c load, since the IDs restart from zero. */
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
        traversal_telemetry_t telemetry;
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
     *  add dispenses the batch's whole ID range, one step lands it, and @c finish publishes. Per
     *  value: walk the graph scoring against the caller's still-unmoved value, move the value & key
     *  into its record, wire the selected neighbors, mark the row wired, then back-link - locking
     *  one neighbor at a time, never during a metric verb. The walk admits the published nodes and
     *  the rows other inserts wired ahead of their turns, this batch's earlier rows among them. One
     *  watermark turn then publishes the whole range, and the keys land last. A full reverse row
     *  re-selects down to the selector's budget - manual over-wiring is advisory here and may be
     *  pruned back.
     *
     *  @param[in] incoming_values The batch to consume - moved from on success only.
     *  @param[in] keys Caller identities, parallel to @p incoming_values.
     *  @param[inout] metric This thread's scoring instance; weights may be shared,
     *      staging scratch must be thread-private.
     *  @param[out] assigned_ids Caller-owned output: either empty, or exactly one slot per
     *      incoming value, receiving that value's dispensed node ID - one contiguous range
     *      per batch, in batch order.
     *  @param[inout] context The calling thread's scratch & telemetry, reused across batches, over
     *      any allocator.
     *  @param[in] selector The link-selection heuristic, carrying its own
     *      neighbors-per-node budget - a per-call value, so heuristics & budgets swap per
     *      batch without re-instantiating the index.
     *  @param[in] config Optional tuning knobs; zeros mean "derive from connectivity".
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

        traversal_telemetry_t const telemetry_before = context.telemetry;
        plan.template insert<synchronization_>(0, incoming_values, keys, metric, context, selector, config);
        for (std::size_t position = 0; position != assigned_ids.size(); ++position)
            assigned_ids[position] = static_cast<node_id_t>(plan.range.first + position);
        // The one step above landed every position, so the finish cannot refuse.
        [[maybe_unused]] status_t const finished = plan.finish();

        insert_result_t result;
        result.telemetry = context.telemetry - telemetry_before;
        return {result, status_t::success_k};
    }

    /**
     *  @brief One batched insertion in flight: the shared hold, the key seats and the dispensed
     *      range, so its positions can only land, and @ref finish publishes them all at once.
     *
     *  Positions map onto @c range in order. Every position must land before @c finish: the range
     *  holds every later turn until then, so a caller fanning the steps out over threads lands a
     *  refused fan-out on the calling thread, and nothing cancels a plan once dispensed. A plan
     *  that dies unfinished only asserts, since publishing its range would expose rows nobody
     *  wrote.
     */
    struct [[nodiscard]] insert_plan_t {

        /** The IDs the plan dispensed, one per position, in position order. */
        iota_view<node_id_t> range {};

        /** Positions the steps landed so far, mutated through @c atomic_ref. */
        std::size_t landed = 0;

        insert_plan_t() noexcept = default;
        insert_plan_t(ram_index* index, iota_view<node_id_t> range, typename key_map_t::seats_t seats,
                      shared_lock_t&& hold) noexcept
            : range(range), index_(index), hold_(std::move(hold)), seats_(seats) {}
        insert_plan_t(insert_plan_t&& other) noexcept
            : range(other.range), landed(other.landed), index_(std::exchange(other.index_, nullptr)),
              hold_(std::move(other.hold_)), seats_(other.seats_) {}
        insert_plan_t& operator=(insert_plan_t&&) = delete;
        ~insert_plan_t() noexcept { assert((!index_ || range.empty()) && "A dispensed plan must finish"); }

        /**
         *  @brief Lands positions @p first onward, one per value of @p values under its key in
         *      @p keys; cannot fail, since a refused walk wires a thinner row.
         *
         *  Per value: a walk scoring against the still-unmoved value, the value & key moved into
         *  the record, the selector's neighbors wired, the row marked wired, then the back-links.
         *  Any number of threads step one plan at once over disjoint positions, each with its own
         *  @p metric and a @p context that @ref reserve_traversal sized for the selector's
         *  connectivity and @p config. The walks admit the published nodes and every row wired
         *  ahead of its turn, the plan's own among them. Positions outside the plan, keys of
         *  another length, or a zero connectivity land nothing, and @c finish then refuses.
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
                index_->template land_<synchronization_>(static_cast<node_id_t>(range.first + first + position),
                                                         values[position], keys[position], metric, context, selector,
                                                         walk);
            // Releases this step's rows to the acquire in `finish`, whose turn carries them on.
            atomic_add(atomic_ref<std::size_t>(landed), count, std::memory_order_release);
        }

        /** Publishes every position at once: @c config_mismatch_k while any has yet to land,
         *  leaving the plan whole; then the range's one watermark turn, the keys in their seats,
         *  and the hold dropped. A success spends the plan, so a second call answers
         *  @c uninitialized_k. */
        status_t finish() noexcept {
            if (!index_) return status_t::uninitialized_k;
            if (atomic_ref<std::size_t>(landed).load(std::memory_order_acquire) != range.size())
                return status_t::config_mismatch_k;
            if (!range.empty()) {
                // Take the turn strictly in dispensed order: every range dispensed ahead of this
                // one has published, which keeps the visible prefix hole-free. Holds no row lock
                // while waiting.
                node_id_t& last_visible_id = index_->last_visible_id_;
                for (node_id_t watermark = atomic_ref<node_id_t>(last_visible_id).load(std::memory_order_acquire);
                     watermark != range.first;
                     watermark = atomic_ref<node_id_t>(last_visible_id).load(std::memory_order_acquire))
                    wait_t {}(&last_visible_id, watermark, 0u);
                atomic_ref<node_id_t>(last_visible_id).store(range.bound, std::memory_order_release);

                // Post-turn: keys become findable only for published nodes, in seats claimed first.
                for (node_id_t const node_id : range)
                    index_->key_map_.insert(seats_, index_->nodes_[node_id].key, node_id);
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

        /** One key seat per position, claimed before the dispense. */
        typename key_map_t::seats_t seats_ {};
    };

    /**
     *  @brief Opens a batched insertion of @p count values: joins the shared side, claims the key
     *      seats, claims and covers the node records, then dispenses the whole ID range in one add.
     *
     *  Every refusal precedes the dispense: @c overflow_k past @c max_node_count_k, or a refused
     *  cover's status once the claim is given back; a refused seat claim or cover strands its
     *  seats, which only leaves their chunks emptier. Everything else fallible belongs before this
     *  call, since nothing may refuse once the range is dispensed: building the values, sizing each
     *  stepping thread's context through @ref reserve_traversal, a metric's own setup. A @p count
     *  of zero dispenses nothing, and a refine plan holding the claims closed refuses with
     *  @c contended_k. Joining the shared side waits out a unique holder, so a call from a thread
     *  holding a save or load plan deadlocks on its own hold, by construction.
     */
    expected<insert_plan_t> plan_insert(std::size_t count) noexcept {
        // The gate opens before the claim, so a batch parked behind a unique holder never
        // sits on committed capacity, spuriously overflowing its peers.
        shared_lock_t hold {shared_mutex_};
        if (!count) return {insert_plan_t {this, {}, {}, std::move(hold)}, status_t::success_k};
        // Refused before the seats, so batches retried against a refine strand none.
        if (atomic_ref<std::size_t>(claimed_count_).load(std::memory_order_relaxed) == closed_claims_k)
            return {{}, status_t::contended_k};

        // Key seats come first, so a refused seat claim leaves no capacity claimed.
        auto [seats, seat_status] = key_map_.claim(count);
        if (failed(seat_status)) return {{}, seat_status};

        // Claims convert one-to-one into dispensed IDs once covered, and commit only within the
        // ID ceiling, so peers never observe an overshoot. Covering to this claim's end suffices:
        // of the batches dispensed up to this one, the last to claim covered past this range, and
        // the dispenser's acquire-release carries its cover along.
        std::size_t const claimed = atomic_fetch_add_if_at_most(atomic_ref<std::size_t>(claimed_count_), count,
                                                                max_node_count_k, std::memory_order_acq_rel);
        if (claimed == closed_claims_k) return {{}, status_t::contended_k};
        if (max_node_count_k - claimed < count) return {{}, status_t::overflow_k};
        if (status_t const covered = nodes_.cover(claimed + count); failed(covered)) {
            atomic_sub(atomic_ref<std::size_t>(claimed_count_), count, std::memory_order_release);
            return {{}, covered};
        }

        // One add dispenses the range one turn will publish, so nothing past it may fail; the steps
        // run between the two, so a slow scorer delays the batches dispensed after this one.
        node_id_t const first_id = atomic_ref<node_id_t>(next_dispensed_id_)
                                       .fetch_add(static_cast<node_id_t>(count), std::memory_order_acq_rel);
        iota_view<node_id_t> const range {first_id, static_cast<node_id_t>(first_id + count)};
        return {insert_plan_t {this, range, seats, std::move(hold)}, status_t::success_k};
    }

    /** Sizes @p context for walks under @p connectivity and @p config - the fallible half of an
     *  insertion, hoisted ahead of @ref plan_insert since no step may allocate it. The candidates,
     *  the staging surfaces and the selections never allocate again, and the frontier holds the
     *  full candidates' unexpanded members plus one round's admissions, which shedding past their
     *  radius keeps it within, so only ties at that radius grow it. The visited set starts at one
     *  row per candidate and alone re-tables as a walk widens - once per doubling per context
     *  lifetime, which a context kept across walks amortizes. Idempotent; a zero @p connectivity
     *  answers @c config_mismatch_k. */
    template <typename context_allocator_type_>
    static status_t reserve_traversal(traversal_context_over<context_allocator_type_>& context,
                                      std::size_t connectivity, insert_config_t config) noexcept {
        if (!connectivity) return status_t::config_mismatch_k;
        insert_config_t const walk = resolved_config_(config, connectivity);
        std::size_t const round_capacity = ram_index::round_capacity(connectivity, config);
        // A re-selection's candidates: the walk's breadth plus the node's own row, which also
        // covers a back-link's full row plus the newcomer.
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
        return context.reverse_ids.reserve(connectivity);
    }

    /** Candidates one metric round of a walk under @p connectivity and @p config stages: the rows
     *  whose neighbors fill a walk's candidates, of the widest row, or a full row plus the newcomer
     *  a back-link re-selects - what a metric's staging holds to score every round at once. */
    static std::size_t round_capacity(std::size_t connectivity, insert_config_t config) noexcept {
        return max_of(
            divide_round_up(resolved_config_(config, connectivity).expansion_factor, connectivity) * connectivity,
            connectivity + 1);
    }

    /** Appends one directed edge @p from → @p to - purely structural: no metric,
     *  no selector, no pruning, and the row grows past any prior extent when needed.
     *  Idempotent: an existing edge answers @c present_k. Both IDs must be published;
     *  unpublished IDs & self-links answer @c malformed_k. Safe concurrently with
     *  @c insert, @c search, readers & other manual wiring. */
    template <synchronization_t synchronization_ = synchronization_t::concurrent_k>
    expected<membership_t> link(node_id_t from, node_id_t to) noexcept {
        static_assert(synchronization_ != synchronization_t::immutable_k,
                      "Certified immutability contradicts wiring - certify exclusive_k or stay concurrent_k");
        constexpr bool locked_k = synchronization_ == synchronization_t::concurrent_k;
        std::size_t const visible = size();
        if (static_cast<std::size_t>(from) >= visible || static_cast<std::size_t>(to) >= visible || from == to)
            return {membership_t::absent_k, status_t::malformed_k};

        // Joins the mutator class: manual wiring never moves the watermark, so the shared
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

    /** Removes one directed edge @p from → @p to; @c malformed_k when the edge or
     *  either ID is unpublished or absent. The surviving row's order is unspecified.
     *  Safe concurrently with @c insert, @c search, readers & other manual wiring. */
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

    /** Optional tuning knobs of a refine's walks, an insertion's own; zeros mean "derive from the
     *  selector's budget". */
    using refine_config_t = insert_config_t;

    /**
     *  @brief One reachability audit in flight: a sweep from node zero over the published nodes,
     *      forward along the rows and back against them, one level per sweep of every step, until
     *      a level settles nothing.
     *
     *  Read-only on the graph, so concurrent insertions merely leave fresh nodes uncounted. Steps
     *  are disjoint first-chunk-sized ID ranges, so any number of threads may @ref sweep different
     *  steps of one level at once; @ref advance_level is the level boundary, one caller. Reached
     *  bits only turn on and a node expands once, so the sweep converges to the exact reachable
     *  set. A node reaches back once a neighbor in its row does, so no reverse adjacency is built;
     *  every node both reached and reaching back is reached from any other such node.
     */
    struct health_plan_t {

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

            /** Linked from a row settled before the level that reached it. */
            linked_once_k,

            /** Linked from two such rows, so one link may go to a reconnection. */
            linked_twice_k,

            /** Its neighbors already swept. */
            expanded_k,

            /** Reached before the current level began. */
            settled_k,

            /** Reaches node zero through some neighbor in its row. */
            reaches_back_k,

            /** Reached node zero back before the current level began. */
            back_settled_k,

            /** Linked to two nodes that reached back before it did, so one link may go to a
             *  reconnection. */
            back_linked_twice_k,

            /** The bitsets in all. */
            bitsets_k,
        };

        using bits_t = growing_buffer<u1x64_t, allocator_t>;

        /** Lanes one word carries - what an ID splits on. */
        inline static constexpr std::size_t lanes_k = std::numeric_limits<u64_t>::digits;

        /** The audit's relation for a repair walk: admits the nodes reached from node zero and
         *  reaching it back. */
        struct membership_t {
            health_plan_t const& audit;

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
        explicit health_plan_t(allocator_t const& allocator) noexcept : bits_(allocator) {}
        health_plan_t(ram_index& index, node_id_t visible, bits_t&& bits) noexcept
            : visible(visible), index_(&index), bits_(std::move(bits)) {
            restart_sweep();
        }
        health_plan_t(health_plan_t&& other) noexcept
            : visible(other.visible), index_(std::exchange(other.index_, nullptr)), bits_(std::move(other.bits_)),
              expanded_in_level_(other.expanded_in_level_), reached_back_in_level_(other.reached_back_in_level_),
              sweeps_(other.sweeps_), back_sweeps_(other.back_sweeps_) {}
        health_plan_t& operator=(health_plan_t&& other) noexcept {
            visible = other.visible;
            index_ = std::exchange(other.index_, nullptr);
            bits_ = std::move(other.bits_);
            expanded_in_level_ = other.expanded_in_level_;
            reached_back_in_level_ = other.reached_back_in_level_;
            sweeps_ = other.sweeps_;
            back_sweeps_ = other.back_sweeps_;
            return *this;
        }
        health_plan_t(health_plan_t const&) = delete;
        health_plan_t& operator=(health_plan_t const&) = delete;

        /** Step @p position of the plan; an empty step past the end. */
        step_t operator[](std::size_t position) const noexcept {
            step_t step;
            if (position >= steps().size()) return step;
            std::size_t const first = position * nodes_schedule_t::rows_in(0);
            std::size_t const bound = min_of(first + nodes_schedule_t::rows_in(0), static_cast<std::size_t>(visible));
            step.position = position;
            step.nodes = {static_cast<node_id_t>(first), static_cast<node_id_t>(bound)};
            return step;
        }

        /** The plan's steps, positionally; every level sweeps all of them. */
        indexed_range<health_plan_t> steps() const noexcept {
            return {this, index_ ? divide_round_up<std::size_t>(visible, nodes_schedule_t::rows_in(0)) : 0u};
        }

        /** Turns the audit onto @p index 's IDs below @p visible, which its bitsets must span, and
         *  restarts the sweep. */
        void rebind(ram_index& index, node_id_t visible) noexcept {
            index_ = &index;
            this->visible = visible;
            restart_sweep();
        }

        /** Clears the bitsets and marks node zero both ways, so the next @ref sweep starts level
         *  one. */
        void restart_sweep() noexcept {
            for (u1x64_t& lanes : bits_) lanes = 0;
            expanded_in_level_ = 0;
            reached_back_in_level_ = 0;
            sweeps_ = 0;
            back_sweeps_ = 0;
            if (!visible) return;
            insert_(reached_k, node_id_t {0});
            insert_(reaches_back_k, node_id_t {0});
            snapshot_levels_();
        }

        /**
         *  @brief Expands every node of @p step settled before this level and not yet expanded,
         *      its neighbors below the watermark turning reached, and marks every node of it a
         *      neighbor of which reached back before this level.
         *  @return The nodes this call expanded or found reaching back.
         *
         *  Safe on any thread for disjoint steps - the expanded bits are the step's own, the
         *  reached bits are set through relaxed atomics, and the settled bits stay fixed for the
         *  level.
         */
        std::size_t sweep(step_t const& step) noexcept {
            std::size_t expanded_count = 0, reached_back_count = 0;
            for (node_id_t node_id = step.nodes.first; node_id < step.nodes.bound; ++node_id) {
                bool const expanding = contains_(settled_k, node_id) && !contains_(expanded_k, node_id);
                bool const pulling = !contains_(reaches_back_k, node_id);
                if (!expanding && !pulling) continue;
                // Bits only, never a metric verb, so the row lock spans the whole row.
                node_t& node = index_->nodes_[node_id];
                node.lock();
                if (expanding) {
                    insert_exclusive_(expanded_k, node_id);
                    ++expanded_count;
                    // Only settled rows expand, so every link to a target the level reaches counts.
                    for (node_id_t const target : node.neighbors) {
                        if (target >= visible || target == node_id || contains_(settled_k, target)) continue;
                        if (!contains_(reached_k, target)) insert_(reached_k, target);
                        if (insert_(linked_once_k, target) && !contains_(linked_twice_k, target))
                            insert_(linked_twice_k, target);
                    }
                }
                if (pulling) {
                    std::size_t settled_neighbors = 0;
                    for (node_id_t const target : node.neighbors)
                        settled_neighbors += target < visible && target != node_id && contains_(back_settled_k, target);
                    if (settled_neighbors) {
                        insert_(reaches_back_k, node_id);
                        ++reached_back_count;
                        if (settled_neighbors > 1) insert_exclusive_(back_linked_twice_k, node_id);
                    }
                }
                node.unlock();
            }
            atomic_add(atomic_ref<std::size_t>(expanded_in_level_), expanded_count, std::memory_order_relaxed);
            atomic_add(atomic_ref<std::size_t>(reached_back_in_level_), reached_back_count, std::memory_order_relaxed);
            return expanded_count + reached_back_count;
        }

        /** The level boundary, one caller after every step swept: counts the level in each
         *  direction that moved, settles what it marked, and returns what it marked; zero means
         *  the sweep converged. */
        std::size_t advance_level() noexcept {
            std::size_t const expanded = std::exchange(expanded_in_level_, 0);
            std::size_t const reached_back = std::exchange(reached_back_in_level_, 0);
            sweeps_ += expanded != 0;
            back_sweeps_ += reached_back != 0;
            if (expanded || reached_back) snapshot_levels_();
            return expanded + reached_back;
        }

        /** Whether @p node_id was reached - readable by any thread between levels. */
        bool reached(node_id_t node_id) const noexcept { return contains_(reached_k, node_id); }

        /** Whether @p node_id reaches node zero - readable by any thread between levels. */
        bool reaches_back(node_id_t node_id) const noexcept { return contains_(reaches_back_k, node_id); }

        /** Takes one of @p node_id 's in-links for a reconnection, answering whether one from a row
         *  settled a level earlier remains; a refusal leaves it untouched. Sound by induction on
         *  levels: the kept link's source keeps a link from an earlier level in turn. */
        bool release_link(node_id_t node_id) noexcept { return release_(linked_twice_k, node_id); }

        /** Takes one of @p node_id 's out-links for a reconnection, answering whether its row still
         *  reaches back through a node that did a level earlier; a row with no way back has none to
         *  lose. */
        bool release_out_link(node_id_t node_id) noexcept {
            return !contains_(reaches_back_k, node_id) || release_(back_linked_twice_k, node_id);
        }

        /** The report: the nodes reached and reaching back, and the levels each direction took.
         *  @c internal_k when a level is still pending, since the count would then be a lower
         *  bound. */
        expected<health_result_t> finish() const noexcept {
            if (!index_) return {{}, status_t::uninitialized_k};
            if (expanded_in_level_ || reached_back_in_level_) return {{}, status_t::internal_k};
            health_result_t result;
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

        /** The audited index; null on a moved-from plan. */
        ram_index* index_ = nullptr;

        /** Every bitset, one bit per swept node each, laid out as @ref bitset_t orders them. */
        bits_t bits_;

        /** Nodes the current level expanded so far, summed through @c atomic_ref. */
        std::size_t expanded_in_level_ = 0;

        /** Nodes the current level found reaching back so far, summed through @c atomic_ref. */
        std::size_t reached_back_in_level_ = 0;

        /** Levels since the last @ref restart_sweep that expanded a node. */
        std::size_t sweeps_ = 0;

        /** Levels since the last @ref restart_sweep that found a node reaching back. */
        std::size_t back_sweeps_ = 0;
    };

    /** Opens a reachability audit over the published nodes, from node zero both ways, so the
     *  nodes it reaches both ways are what a walk from any of them reaches. The caller sweeps
     *  every step per level until a level marks nothing, then reads @c finish(). The bitsets come
     *  from @p context 's allocator. */
    expected<health_plan_t> plan_health(traversal_context_t& context) noexcept {
        return plan_audit_(static_cast<node_id_t>(size()), context.allocator);
    }

    /**
     *  @brief One refinement pass in flight: donors landed first when it absorbs, then the planned
     *      range re-selected, reachability swept level by level, and the islands healed.
     *
     *  Re-selection runs once per round of walks across while the index holds several parts, and
     *  the islands are re-selected and reconnected round by round while their count shrinks. An
     *  empty range heals.
     *
     *  A split runs it in reverse: the source's rows partition into one graph-local part per
     *  receiver, scatter into the receivers with the links inside each part rebased and the links
     *  across dropped, and publish; then every receiver is refined in turn over its seams, the
     *  prefix of rows that lost a link, and heals, the tallies summing over the receivers.
     *
     *  The plan holds the shared side and the claims closed from planning to its death, so batches
     *  answer @c contended_k and stop-the-world verbs refuse meanwhile, while searches run
     *  throughout. Round @c r walks every node across the neighboring group of `2^r` parts, which
     *  @ref part_across names, so @c k parts merge in `bit_width(k - 1)` rounds and two parts in
     *  one. The first round starts at the planned range's first node, every later one at node zero.
     *
     *  Steps of the current phase are disjoint ID ranges, so any number of threads may @ref run
     *  them at once, each with its own metric and a context @ref reserve_traversal sized for the
     *  planned budget; @ref advance is the single-caller barrier between phases and answers whether
     *  another phase follows. Re-selection and reconnection edit under the per-node locks like an
     *  insertion and allocate nothing, since planning reserved every row to the budget, and a pass
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
            : audit(allocator), part_bases_(allocator), donor_holds_(allocator), receiver_holds_(allocator),
              receiver_seats_(allocator), part_slots_(allocator), part_members_(allocator), part_extents_(allocator) {}
        refine_plan_t(refine_plan_t&& other) noexcept
            : audit(std::move(other.audit)), taken(other.taken), ticked(other.ticked),
              index_(std::exchange(other.index_, nullptr)), hold_(std::move(other.hold_)), nodes_(other.nodes_),
              part_bases_(std::move(other.part_bases_)), donors_(other.donors_),
              donor_holds_(std::move(other.donor_holds_)), landed_donors_(other.landed_donors_), seats_(other.seats_),
              source_(std::exchange(other.source_, nullptr)), source_hold_(std::move(other.source_hold_)),
              receivers_(other.receivers_), receiver_holds_(std::move(other.receiver_holds_)),
              receiver_seats_(std::move(other.receiver_seats_)), part_slots_(std::move(other.part_slots_)),
              part_members_(std::move(other.part_members_)), part_extents_(std::move(other.part_extents_)),
              seed_(other.seed_), refined_receivers_(other.refined_receivers_),
              receivers_health_(other.receivers_health_), receivers_islands_(other.receivers_islands_),
              rounds_(other.rounds_), round_(other.round_), walk_(other.walk_), connectivity_(other.connectivity_),
              phase_(other.phase_), swept_after_(other.swept_after_), islands_before_(other.islands_before_),
              tally_(other.tally_) {}
        refine_plan_t(refine_plan_t const&) = delete;
        refine_plan_t& operator=(refine_plan_t const&) = delete;
        ~refine_plan_t() noexcept { release_(); }

        /** Step @p position of the current phase; an empty step past the end. */
        step_t operator[](std::size_t position) const noexcept {
            if (audited_()) return audit[position];
            step_t step;
            if (position >= steps().size()) return step;
            iota_view<node_id_t> const range = phase_rows_();
            std::size_t const width = phase_ == phase_t::partition_k ? max_of<std::size_t>(range.size(), 1)
                                                                     : nodes_schedule_t::rows_in(0);
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
            if (audited_()) return {this, audit.steps().size()};
            return {this, divide_round_up<std::size_t>(phase_rows_().size(), nodes_schedule_t::rows_in(0))};
        }

        /**
         *  @brief Runs @p step of the current phase on the calling thread - safe for disjoint steps
         *      of one phase, and never failing on a bound plan, since planning claimed, covered and
         *      reserved everything a step writes.
         *
         *  A landing step moves a donor's rows in, the one partition step grows a split's parts, a
         *  scatter step moves source rows into their receivers, a re-selection step walks from each
         *  node and wires what @p selector picks, a sweep step expands the audit, and a
         *  reconnection step links islands to the nodes the audit proved both ways; @p metric and
         *  @p context serve the walks alone.
         */
        template <typename metric_type_, typename context_allocator_type_, typename selector_type_ = keep_diverse_t>
            requires metric<metric_type_, value_t, distance_t> && selector<selector_type_, ram_index, metric_type_>
        status_t run(step_t const& step, metric_type_& metric, traversal_context_over<context_allocator_type_>& context,
                     selector_type_ const& selector = {}) noexcept {
            if (!index_) return status_t::uninitialized_k;
            switch (phase_) {
            case phase_t::land_k: land_(step); break;
            case phase_t::partition_k: partition_(); break;
            case phase_t::scatter_k: scatter_(step); break;
            case phase_t::reselect_k:
            case phase_t::reselect_islands_k: reselect_(step, metric, context, selector); break;
            case phase_t::sweep_k: audit.sweep(step); break;
            case phase_t::reconnect_k: reconnect_(step, metric, context); break;
            default: break;
            }
            return status_t::success_k;
        }

        /** The barrier between phases, one caller after every step of the current one ran:
         *  publishes a landed donor or a split's parts, or picks the next phase, and answers
         *  whether one follows. */
        bool advance() noexcept {
            if (!index_) return false;
            if (phase_ == phase_t::land_k) {
                publish_landed_();
                if (landed_donors_ == donors_.size()) phase_ = first_pass_phase_();
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
            if (phase_ == phase_t::sweep_k) {
                if (!audit.advance_level()) converge_();
            }
            else if (phase_ != phase_t::done_k) {
                if (phase_ == phase_t::reselect_k && ++round_ < rounds_) return true;
                audit.restart_sweep();
                swept_after_ = phase_;
                phase_ = phase_t::sweep_k;
            }
            if (phase_ == phase_t::done_k && refined_receivers_ != receivers_.size()) settle_receiver_();
            return phase_ != phase_t::done_k;
        }

        /** Stops the re-selection where its steps ran, so the next @ref advance turns to the heal;
         *  a no-op in any other phase, the landings included. */
        void stop() noexcept {
            if (phase_ != phase_t::reselect_k) return;
            rounds_ = 0;
            taken = steps().size();
        }

        /** The walks' knobs, which every context stepping the plan is reserved for alongside
         *  @ref connectivity. */
        insert_config_t walk() const noexcept { return walk_; }

        /** The budget planning reserved every row for. */
        std::size_t connectivity() const noexcept { return connectivity_; }

        /** Whether the current phase lands a donor or moves a split's rows - the phases no stop
         *  skips. */
        bool landing() const noexcept {
            return phase_ == phase_t::land_k || phase_ == phase_t::partition_k || phase_ == phase_t::scatter_k;
        }

        /** Whether the current phase re-selects the planned range, which a caller's progress
         *  counts. */
        bool reselecting() const noexcept { return phase_ == phase_t::reselect_k; }

        /** The pass so far: complete once @ref advance answered false, otherwise @c cancelled_k. */
        expected<refine_result_t> finish() const noexcept {
            if (!index_) return {{}, status_t::uninitialized_k};
            return {tally_, phase_ == phase_t::done_k ? status_t::success_k : status_t::cancelled_k};
        }

      private:
        friend ram_index;

        enum class phase_t : u8_t {
            land_k,
            partition_k,
            scatter_k,
            reselect_k,
            sweep_k,
            reselect_islands_k,
            reconnect_k,
            done_k,
        };

        static void add_(std::size_t& into, std::size_t value) noexcept {
            atomic_add(atomic_ref<std::size_t>(into), value, std::memory_order_relaxed);
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
            source_hold_ = {};
        }

        /** Whether the current phase steps over the audit's ranges rather than rows of its own. */
        bool audited_() const noexcept {
            return phase_ == phase_t::sweep_k || phase_ == phase_t::reselect_islands_k ||
                   phase_ == phase_t::reconnect_k;
        }

        /** The rows the current phase steps over: a donor's, a split source's, or the round's. */
        iota_view<node_id_t> phase_rows_() const noexcept {
            if (phase_ == phase_t::land_k) return landing_rows_();
            if (phase_ == phase_t::partition_k || phase_ == phase_t::scatter_k) return {0, source_rows_()};
            return round_nodes_();
        }

        /** Rows the split's source published at planning: one slot each. */
        node_id_t source_rows_() const noexcept { return static_cast<node_id_t>(part_slots_.size()); }

        /** The most rows one part takes: the source's rows over the receivers, rounded up. */
        std::size_t part_capacity_() const noexcept {
            return divide_round_up<std::size_t>(part_slots_.size(), receivers_.size());
        }

        /** The phase the pass starts with once no donor is pending. */
        phase_t first_pass_phase_() const noexcept {
            return nodes_.empty() && rounds_ == 1 ? phase_t::sweep_k : phase_t::reselect_k;
        }

        /** The current donor's rows, in its own ID space. */
        iota_view<node_id_t> landing_rows_() const noexcept {
            return {0, static_cast<node_id_t>(donors_[landed_donors_]->size())};
        }

        /** The nodes the current round of re-selection walks: the planned range in the first round,
         *  every node below its bound in a later one. */
        iota_view<node_id_t> round_nodes_() const noexcept {
            return round_ ? iota_view<node_id_t> {0, nodes_.bound} : nodes_;
        }

        /** Moves the rows of @p step from the current donor into the records past the watermark,
         *  every neighbor ID shifted by it; the rows swap buffers, so nothing allocates. */
        void land_(step_t const& step) noexcept {
            ram_index& donor = *donors_[landed_donors_];
            node_id_t const base = static_cast<node_id_t>(index_->size());
            for (node_id_t source = step.nodes.first; source < step.nodes.bound; ++source) {
                node_t& from = donor.nodes_[source];
                node_t& into = index_->nodes_[base + source];
                into.value = std::move(from.value);
                into.key = from.key;
                into.neighbors = std::move(from.neighbors);
                for (node_id_t& neighbor : into.neighbors) neighbor += base;
            }
        }

        /** Publishes the current donor's rows at once, then their keys in the seats claimed for
         *  them, empties the donor and releases it, and nulls its slot. */
        void publish_landed_() noexcept {
            ram_index& donor = *donors_[landed_donors_];
            node_id_t const rows = static_cast<node_id_t>(donor.size());
            node_id_t const base = static_cast<node_id_t>(index_->size());
            if (rows) {
                atomic_ref<node_id_t>(index_->next_dispensed_id_).store(base + rows, std::memory_order_relaxed);
                atomic_ref<node_id_t>(index_->last_visible_id_).store(base + rows, std::memory_order_release);
                for (node_id_t node_id = base; node_id != base + rows; ++node_id)
                    index_->key_map_.insert(seats_, index_->nodes_[node_id].key, node_id);
            }
            donor.forget_(rows);
            donor_holds_[landed_donors_].unlock();
            donors_[landed_donors_] = nullptr;
            ++landed_donors_;
        }

        /** Places source row @p row last in @p part, its slot naming the part and the position. */
        void enlist_(node_id_t row, std::size_t part) noexcept {
            std::size_t const position = part_extents_[part].rows++, capacity = part_capacity_();
            part_members_[part * capacity + position] = row;
            part_slots_[row] = static_cast<node_id_t>(part * capacity + position);
        }

        /** Expands the oldest unexpanded member of @p part, enlisting its unplaced neighbors while
         *  the part has room; answers whether the part could expand at all. */
        bool expand_(std::size_t part) noexcept {
            // Until the seams replace them, these count each part's expanded members.
            node_id_t& expanded = part_extents_[part].seam;
            std::size_t const capacity = part_capacity_();
            if (expanded == part_extents_[part].rows || part_extents_[part].rows == capacity) return false;
            node_id_t const node_id = part_members_[part * capacity + expanded++];
            for (node_id_t const target : source_->nodes_[node_id].neighbors)
                if (target < source_rows_() && part_slots_[target] == missing_k && part_extents_[part].rows != capacity)
                    enlist_(target, part);
            return true;
        }

        /** Moves source row @p row into its part's seams, the prefix of its members, unless it is
         *  there already, keeping both swapped rows' slots. */
        void enlist_seam_(node_id_t row) noexcept {
            std::size_t const capacity = part_capacity_();
            std::size_t const part = part_slots_[row] / capacity, position = part_slots_[row] % capacity;
            node_id_t& seam = part_extents_[part].seam;
            if (position < seam) return;
            node_id_t* const members = part_members_.data() + part * capacity;
            std::swap(members[position], members[seam]);
            part_slots_[members[position]] = static_cast<node_id_t>(part * capacity + position);
            part_slots_[members[seam]] = static_cast<node_id_t>(part * capacity + seam);
            ++seam;
        }

        /**
         *  @brief Grows one part per receiver from a root the seed picks, breadth first over the
         *      source's links and one expansion per part in turn, each capped at the part capacity;
         *      every row left unplaced then seeds a growth of the smallest part.
         *
         *  Each part's seams, the rows at either end of a link into another part, then move ahead
         *  of the rest, and every row's slot names its part and its row in the receiver. Reads the
         *  source without locks, since its unique side is held.
         */
        void partition_() noexcept {
            std::size_t const parts = receivers_.size(), capacity = part_capacity_();
            node_id_t const source_rows = source_rows_();
            for (node_id_t& slot : part_slots_) slot = missing_k;
            for (std::size_t part = 0; part != parts; ++part) part_extents_[part] = {};
            for (std::size_t part = 0; part != parts && part != source_rows; ++part)
                enlist_(static_cast<node_id_t>(seeded_bijection(part, source_rows, seed_)), part);
            for (std::size_t expanded = parts; expanded;) {
                expanded = 0;
                for (std::size_t part = 0; part != parts; ++part) expanded += expand_(part);
            }
            for (node_id_t row = 0; row != source_rows; ++row) {
                if (part_slots_[row] != missing_k) continue;
                std::size_t smallest = 0;
                for (std::size_t part = 1; part != parts; ++part)
                    if (part_extents_[part].rows < part_extents_[smallest].rows) smallest = part;
                enlist_(row, smallest);
                while (expand_(smallest)) {}
            }
            for (std::size_t part = 0; part != parts; ++part) part_extents_[part].seam = 0;
            for (node_id_t row = 0; row != source_rows; ++row)
                for (node_id_t const target : source_->nodes_[row].neighbors) {
                    if (target >= source_rows || part_slots_[target] / capacity == part_slots_[row] / capacity)
                        continue;
                    enlist_seam_(row);
                    enlist_seam_(target);
                }
        }

        /** Moves the source rows of @p step into their receivers' records, the links inside each
         *  part rebased through its slots and the links across dropped; the rows swap buffers, so
         *  nothing allocates. */
        void scatter_(step_t const& step) noexcept {
            std::size_t const capacity = part_capacity_();
            node_id_t const source_rows = source_rows_();
            for (node_id_t source = step.nodes.first; source < step.nodes.bound; ++source) {
                std::size_t const slot = part_slots_[source], part = slot / capacity;
                node_t& from = source_->nodes_[source];
                node_t& into = receivers_[part]->nodes_[slot % capacity];
                into.value = std::move(from.value);
                into.key = from.key;
                into.neighbors = std::move(from.neighbors);
                std::size_t kept = 0;
                for (node_id_t const target : into.neighbors)
                    if (target < source_rows && part_slots_[target] / capacity == part)
                        into.neighbors[kept++] = static_cast<node_id_t>(part_slots_[target] % capacity);
                into.neighbors.resize_reserved(kept);
            }
        }

        /** Publishes every receiver's rows at once, then their keys in the seats claimed for them,
         *  and empties the source and releases it. */
        void publish_parts_() noexcept {
            for (std::size_t part = 0; part != receivers_.size(); ++part) {
                ram_index& into = *receivers_[part];
                node_id_t const rows = part_extents_[part].rows;
                if (!rows) continue;
                atomic_ref<node_id_t>(into.next_dispensed_id_).store(rows, std::memory_order_relaxed);
                atomic_ref<node_id_t>(into.last_visible_id_).store(rows, std::memory_order_release);
                for (node_id_t node_id = 0; node_id != rows; ++node_id)
                    into.key_map_.insert(receiver_seats_[part], into.nodes_[node_id].key, node_id);
            }
            source_->forget_(source_rows_());
            source_hold_.unlock();
            source_ = nullptr;
        }

        /** Turns the pass to the next receiver: its audit, its seams as the range, and the
         *  re-selection, or straight the sweep when it has no seams or a stop cut the pass. */
        void bind_receiver_() noexcept {
            std::size_t const receiver = refined_receivers_;
            index_ = receivers_[receiver];
            audit.rebind(*index_, part_extents_[receiver].rows);
            nodes_ = {0, part_extents_[receiver].seam};
            round_ = 0;
            swept_after_ = phase_t::reselect_k;
            islands_before_ = 0;
            phase_ = !rounds_ || nodes_.empty() ? phase_t::sweep_k : phase_t::reselect_k;
        }

        /** Folds a healed receiver's audit and islands into the split's, then binds the next one,
         *  or reports the sums once the last healed. */
        void settle_receiver_() noexcept {
            receivers_health_ += tally_.health;
            receivers_islands_ += tally_.islands_found;
            if (++refined_receivers_ != receivers_.size()) return bind_receiver_();
            tally_.health = receivers_health_;
            tally_.islands_found = receivers_islands_;
        }

        /** Picks what follows a converged audit, from the phase the audit followed. */
        void converge_() noexcept {
            auto const [health, status] = audit.finish();
            std::size_t const left = succeeded(status) ? audit.visible - health.reachable_nodes : 0;
            tally_.health = health;
            if (swept_after_ == phase_t::reselect_k) {
                tally_.islands_found = left;
                phase_ = left ? phase_t::reselect_islands_k : phase_t::done_k;
            }
            else if (swept_after_ == phase_t::reselect_islands_k) {
                phase_ = left ? phase_t::reconnect_k : phase_t::done_k;
            }
            // A round freeing no island frees none on the next: the picks it needs are unreached.
            else { phase_ = left && left < islands_before_ ? phase_t::reselect_islands_k : phase_t::done_k; }
            if (swept_after_ != phase_t::reselect_islands_k) islands_before_ = left;
        }

        /** Re-selects every node of @p step, or only its islands once the heal rounds began. */
        template <typename metric_type_, typename context_type_, typename selector_type_>
        void reselect_(step_t const& step, metric_type_& metric, context_type_& context,
                       selector_type_ const& selector) noexcept {
            traversal_telemetry_t const before = context.telemetry;
            std::size_t visited = 0, changed = 0;
            // Only the planned rounds walk across; the heal rounds re-select as a plain pass does.
            bool const across = phase_ == phase_t::reselect_k && part_bases_.size() > 1;
            for (node_id_t node_id = step.nodes.first; node_id < step.nodes.bound; ++node_id) {
                if (phase_ == phase_t::reselect_islands_k && audit.reached(node_id) && audit.reaches_back(node_id))
                    continue;
                iota_view<node_id_t> const other = across ? part_across(part_bases_.data(), part_bases_.size(), round_,
                                                                        audit.visible, node_id)
                                                          : iota_view<node_id_t> {};
                changed += index_->refine_one_(node_id, audit.visible, other, walk_, metric, context, selector);
                ++visited;
            }
            traversal_telemetry_t const spent = context.telemetry - before;
            add_(tally_.visited_nodes, visited);
            add_(tally_.changed_nodes, changed);
            add_(tally_.telemetry.computed_distances, spent.computed_distances);
            add_(tally_.telemetry.visited_members, spent.visited_members);
            add_(tally_.telemetry.restarts, spent.restarts);
            add_(tally_.telemetry.saturated_walks, spent.saturated_walks);
            add_(tally_.telemetry.computed_distances_in_rankings, spent.computed_distances_in_rankings);
            add_(tally_.telemetry.computed_distances_in_selections, spent.computed_distances_in_selections);
        }

        /** Reconnects every island in @p step, a node the audit left unreached or not reaching
         *  back, to the nearest node it proved both ways, found by a repair walk: that node's row
         *  links an unreached island, and the island's own row links that node when it has no way
         *  back. */
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
                // An unreached island lands in the nearest proved row that can take it.
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

        /** Links @p island from @p owner 's row under its lock: a free slot when there is one,
         *  otherwise a landing spread across the tail, so islands sharing one row heal in one
         *  round; answers whether it wrote. */
        bool link_island_(node_id_t owner, node_id_t island) noexcept {
            node_t& node = index_->nodes_[owner];
            node.lock();
            neighbors_buffer_t& row = node.neighbors;
            bool present = false, linked = false;
            for (node_id_t const resident : row) present = present || resident == island;
            if (!present && row.size() < connectivity_ && row.size() < row.capacity()) {
                row.push_back_reserved(island);
                linked = true;
            }
            // Never the owner's last way back, and in a reached row, whose unreached residents are
            // this round's landings, never one of those nor a resident's last in-link from an
            // earlier level; an unreached row proves no in-link.
            else if (!present && audit.release_out_link(owner)) {
                std::size_t const width = min_of(heal_spread(connectivity_), row.size());
                for (std::size_t attempt = 0; attempt != width && !linked; ++attempt) {
                    std::size_t const slot = row.size() - 1 - (island + attempt) % width;
                    node_id_t const resident = row[slot];
                    if (resident >= audit.visible ||
                        (audit.reached(owner) && (!audit.reached(resident) || !audit.release_link(resident))))
                        continue;
                    row[slot] = island;
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

        /** The IDs the first round re-selects, clamped to the audit's watermark. */
        iota_view<node_id_t> nodes_;

        /** The ascending first rows of the parts the rounds walk across, below the audit's
         *  watermark; empty for none. */
        growing_buffer<node_id_t, allocator_t> part_bases_;

        /** The caller's donor slots, each nulled once landed. */
        std::span<ram_index*> donors_;

        /** One unique hold per donor, each released once its donor landed. */
        growing_buffer<unique_lock_t, allocator_t> donor_holds_;

        /** Donors landed so far. */
        std::size_t landed_donors_ = 0;

        /** One key seat per donor row, claimed at planning. */
        typename key_map_t::seats_t seats_ {};

        /** The index a split empties, whose unique side @c source_hold_ occupies until its rows
         *  publish in the receivers; null otherwise. */
        ram_index* source_ = nullptr;

        /** The unique hold on a split's source. */
        unique_lock_t source_hold_;

        /** The caller's receiver slots of a split, one part each; empty otherwise. */
        std::span<ram_index*> receivers_;

        /** One shared hold per receiver whose claims the split closed, in receiver order. */
        growing_buffer<shared_lock_t, allocator_t> receiver_holds_;

        /** One run of key seats per receiver, as many as the part capacity. */
        growing_buffer<typename key_map_t::seats_t, allocator_t> receiver_seats_;

        /** Per source row: its part times the part capacity, plus its row in that receiver. */
        growing_buffer<node_id_t, allocator_t> part_slots_;

        /** Per part, a run of the part capacity: its source rows in receiver order. */
        growing_buffer<node_id_t, allocator_t> part_members_;

        /** One part's rows, and the seam within them: the rows before it lost a link across. */
        struct part_extent_t {
            node_id_t rows = 0;
            node_id_t seam = 0;
        };

        /** Per part, its extent. */
        growing_buffer<part_extent_t, allocator_t> part_extents_;

        /** What picks the parts' roots. */
        u64_t seed_ = 0;

        /** Receivers whose passes healed so far; the bound one is the next. */
        std::size_t refined_receivers_ = 0;

        /** The audits of the receivers healed so far, summed. */
        health_result_t receivers_health_ {};

        /** The islands the receivers healed so far found, summed. */
        std::size_t receivers_islands_ = 0;

        /** Rounds of re-selection: one with fewer than three parts, none once @ref stop cut it. */
        std::size_t rounds_ = 1;

        /** The round the re-selection runs, or, once it left, the rounds it finished. */
        std::size_t round_ = 0;

        /** Breadth and fan-out of every re-selection walk. */
        insert_config_t walk_ {};

        /** The planned neighbors-per-node budget every row was reserved for. */
        std::size_t connectivity_ = 0;

        /** The phase whose steps run now. */
        phase_t phase_ = phase_t::done_k;

        /** The phase the current audit follows - what decides the phase after it converges. */
        phase_t swept_after_ = phase_t::reselect_k;

        /** Islands the last round started from - the shrinking the rounds require. */
        std::size_t islands_before_ = 0;

        /** Counts summed through @c atomic_ref by the steps, and set by @ref advance. */
        refine_result_t tally_;
    };

    /**
     *  @brief Opens a refinement pass over @p nodes, clamped to the published IDs, with a fresh
     *      audit seeded like @ref plan_health; an empty range heals alone.
     *
     *  Every fallible step happens here: the audit's bitsets from @p context 's allocator, the
     *  walks' scratch in @p context, and every published row reserved to @p selector 's budget, so
     *  the steps never allocate; a caller fanning the steps out reserves its other contexts alike.
     *  Joins the shared side and closes the claims, refusing with @c contended_k while a batch is
     *  in flight.
     *
     *  @param[in] nodes The ID range the first round re-selects.
     *  @param[inout] context The planning thread's scratch, whose allocator places the plan.
     *  @param[in] selector Carries the budget every row is reserved for; the steps take a selector
     *      of the same budget.
     *  @param[in] config Optional tuning knobs; zeros mean "derive from the selector's budget".
     */
    template <typename selector_type_ = keep_diverse_t>
    expected<refine_plan_t> plan_refine(iota_view<node_id_t> nodes, traversal_context_t& context,
                                        selector_type_ const& selector = {}, refine_config_t config = {}) noexcept {
        return plan_pass_(nodes, {}, context, selector.connectivity, config);
    }

    /**
     *  @brief Opens an absorb of @p donors, whose plan's first phase moves every row of each into
     *      this index, one donor after another, as a part of its own, then re-selects every node
     *      walking across the parts round by round, and heals.
     *
     *  Every fallible step of every landing happens here, after the checks of @ref plan_refine:
     *  each donor's unique side taken, the records past the watermark covered, a key seat claimed
     *  per donor row, and every donor row reserved to the budget. A landing step swaps a donor
     *  row's value and neighbors in with every ID shifted, so nothing allocates, and the phase
     *  boundary publishes the donor's rows at once, their keys right after, empties the donor and
     *  nulls its slot; at most one donor's rows are ever unpublished. Donors must share this
     *  index's allocator, and nothing may search them meanwhile.
     *
     *  @param[inout] donors The indexes giving their rows up, each nulled once landed; the slots
     *      must outlive the plan.
     *  @retval config_mismatch_k A null donor, this index itself, or a zero budget.
     *  @retval contended_k A batch in flight on this index or on a donor, or another refine.
     *  @retval overflow_k Rows past @c max_node_count_k.
     */
    template <typename selector_type_ = keep_diverse_t>
    expected<refine_plan_t> plan_absorb(std::span<ram_index*> donors, traversal_context_t& context,
                                        selector_type_ const& selector = {}, refine_config_t config = {}) noexcept {
        return plan_pass_({0, static_cast<node_id_t>(max_node_count_k)}, donors, context, selector.connectivity,
                          config);
    }

    /**
     *  @brief Opens a split of this index into @p receivers - the inverse of @ref plan_absorb:
     *      its plan's first phases move every row into one graph-local part per receiver, then
     *      refine every receiver in turn over the rows that lost a link, and heal.
     *
     *  The one partition step grows the parts breadth first over the links, one root per part
     *  through @ref seeded_bijection under @p seed, each part capped at the rows over the receivers
     *  rounded up; rows no growth reached seed the smallest part. Scatter steps move each row's
     *  value, key and neighbors into its receiver, the links inside its part rebased and the links
     *  across dropped, and the phase boundary publishes every receiver's rows at once, their keys
     *  right after, then empties this index and releases it, so its claims reopen at zero. Every
     *  receiver then re-selects its seams, the prefix of rows that lost a link, sweeps and heals as
     *  @ref plan_refine does, and the result sums over the receivers.
     *
     *  Every fallible step happens here: this index's unique side taken, each receiver's shared
     *  side joined and its claims closed, the records, key seats and audit of a full part covered
     *  in every receiver, and every row reserved to the budget. Receivers must share this index's
     *  allocator, the slots must outlive the plan, and nothing may search this index meanwhile.
     *
     *  @param[in] receivers The empty indexes taking the parts, at least two, all distinct.
     *  @param[in] seed Picks the parts' roots, so equal seeds over equal graphs split alike.
     *  @retval config_mismatch_k Fewer than two receivers, a null, repeated or non-empty one, this
     *      index itself, or a zero budget.
     *  @retval contended_k A batch in flight on this index or on a receiver, or another refine.
     *  @retval overflow_k Part slots past the node ID type.
     */
    template <typename selector_type_ = keep_diverse_t>
    expected<refine_plan_t> plan_split(std::span<ram_index*> receivers, traversal_context_t& context,
                                       selector_type_ const& selector = {}, refine_config_t config = {},
                                       u64_t seed = 0) noexcept {
        return plan_split_(receivers, context, selector.connectivity, config, seed);
    }

    /**
     *  @brief Re-selects the neighbors of every published node in @p nodes and heals whatever that
     *      strands - one @ref refine_plan_t drained on the calling thread.
     *
     *  @param[in] nodes The ID range to re-select; IDs at or past the watermark are skipped.
     *  @param[inout] metric This thread's scoring instance.
     *  @param[inout] context The calling thread's scratch & telemetry, which also places the plan.
     *  @param[in] selector The link-selection heuristic and its budget.
     *  @param[in] config Optional tuning knobs; zeros mean "derive from the selector's budget".
     */
    template <typename metric_type_, typename selector_type_ = keep_diverse_t>
        requires metric<metric_type_, value_t, distance_t> && selector<selector_type_, ram_index, metric_type_>
    expected<refine_result_t> refine(iota_view<node_id_t> nodes, metric_type_& metric, traversal_context_t& context,
                                     selector_type_ const& selector = {}, refine_config_t config = {}) noexcept {
        auto [plan, status] = plan_refine(nodes, context, selector, config);
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
        traversal_telemetry_t telemetry;
    };

    /**
     *  @brief Finds the nearest published neighbors of @p query; matches land in
     *      @c context.results ascending by distance - the caller reranks or trims.
     *
     *  The leading template parameter is the caller's synchronization contract: the
     *  default assumes live mutators - the walk reads only nodes below the watermark
     *  snapshot and briefly locks each expanded node to copy its row - while a certified
     *  @c immutable_k or @c exclusive_k instantiation compiles the locks out entirely,
     *  bracketed by the shared mutex's wait-free probes: a violated certification answers
     *  @c contended_k rather than a silently wrong result - detected, not memory-safe, since
     *  the plain reads may meet a row a lying mutator is regrowing.
     *  Scoring runs off-lock, one @c reset / @c push / @c drain per round, and the metric need not
     *  match the one that built the graph. Once the candidates are full, their furthest member's
     *  distance drains each round as the radius, so a metric may stop scoring a candidate that
     *  cannot come under it. The query is a stored-value reference by default; a
     *  @c transparent_metric accepts other query forms.
     *
     *  @param[in] query The similarity target, alive for the whole call.
     *  @param[inout] metric This thread's scoring instance.
     *  @param[inout] context The calling thread's scratch & telemetry over any allocator; holds the
     *      matches.
     *  @param[in] config Optional tuning knobs; zeros mean "derive from connectivity".
     */
    template <synchronization_t synchronization_ = synchronization_t::concurrent_k, //
              typename metric_type_, typename context_allocator_type_, typename query_type_ = value_t>
        requires metric<metric_type_, value_t, distance_t> && query<query_type_, metric_type_, value_t>
    expected<search_result_t> search(query_type_ const& query, metric_type_& metric, //
                                     traversal_context_over<context_allocator_type_>& context,
                                     search_config_t config = {}) noexcept {
        return search<synchronization_>(query, metric, dummy_fold_t {}, context, config);
    }

    /**
     *  @brief Walks exactly like @c search while @p fold reads every round the walk scores - the
     *      entry point for verbs that reduce a walk to something other than its nearest matches.
     *
     *  @param[in] query The similarity target, alive for the whole call.
     *  @param[inout] metric This thread's scoring instance.
     *  @param[inout] fold The reader of every scored round - see @c basic::fold. A @c sated_k ends
     *      the walk successfully, before that round reaches the candidates.
     *  @param[inout] context The calling thread's scratch & telemetry over any allocator; holds the
     *      matches.
     *  @param[in] config Optional tuning knobs; zeros mean "derive from connectivity".
     */
    template <synchronization_t synchronization_ = synchronization_t::concurrent_k, //
              typename metric_type_, typename fold_type_, typename context_allocator_type_,
              typename query_type_ = value_t>
        requires metric<metric_type_, value_t, distance_t> && query<query_type_, metric_type_, value_t> &&
                 fold<std::remove_cvref_t<fold_type_>, node_id_t, distance_t>
    expected<search_result_t> search(query_type_ const& query, metric_type_& metric, fold_type_&& fold, //
                                     traversal_context_over<context_allocator_type_>& context,
                                     search_config_t config = {}) noexcept {
        return fold_walk_<synchronization_>(query, metric, fold, context.results, context, config);
    }

    /**
     *  @brief Walks like @c search, keeping only the scored nodes @p relation and @p predicate both
     *      admit, ascending in @c context.results. The walk routes through every node it scores and
     *      ends once the admitted ones fill the expansion or the graph runs out.
     *
     *  @param[in] query The similarity target, alive for the whole call.
     *  @param[inout] metric This thread's scoring instance.
     *  @param[inout] relation The engine's own filter over node IDs, or @c dummy_predicate_t.
     *  @param[inout] predicate The caller's filter over node IDs, asked only about the contenders
     *      @p relation kept: a quad at a time, or once per round when it aggregates.
     *  @param[inout] context The calling thread's scratch & telemetry over any allocator; holds the
     *      matches.
     *  @param[in] config Optional tuning knobs; zeros mean "derive from connectivity".
     */
    template <synchronization_t synchronization_ = synchronization_t::concurrent_k, //
              typename metric_type_, typename relation_type_, typename predicate_type_,
              typename context_allocator_type_, typename query_type_ = value_t>
        requires metric<metric_type_, value_t, distance_t> && query<query_type_, metric_type_, value_t>
    expected<search_result_t> search_if(query_type_ const& query, metric_type_& metric, relation_type_& relation,
                                        predicate_type_& predicate,
                                        traversal_context_over<context_allocator_type_>& context,
                                        search_config_t config = {}) noexcept {
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
     *  @brief The exact sibling of @c search: scores @p query against every row published when the
     *      scan starts and keeps the nearest @p wanted that @p relation and @p predicate both
     *      admit, ascending in @c context.results.
     *
     *  Once @p wanted are held, every round drains within the worst of them, so a metric may stop
     *  scoring a row that cannot come under it. Only a row that would enter the matches is asked
     *  about, which makes the scan the fallback for filters too selective for a walk to cross.
     *
     *  @param[in] query The similarity target, alive for the whole call.
     *  @param[inout] metric This thread's scoring instance.
     *  @param[inout] relation The engine's own filter over node IDs, or @c dummy_predicate_t.
     *  @param[inout] predicate The caller's filter over node IDs, or @c dummy_predicate_t.
     *  @param[inout] context The calling thread's scratch & telemetry; holds the matches.
     *  @param[in] wanted How many matches to keep.
     */
    template <typename metric_type_, typename relation_type_, typename predicate_type_,
              typename context_allocator_type_, typename query_type_ = value_t>
        requires metric<metric_type_, value_t, distance_t> && query<query_type_, metric_type_, value_t>
    expected<search_result_t> scan(query_type_ const& query, metric_type_& metric, relation_type_& relation,
                                   predicate_type_& predicate, traversal_context_over<context_allocator_type_>& context,
                                   std::size_t wanted) noexcept {
        context.results.clear();
        if (!wanted) return {};
        auto const radius = [&]() noexcept {
            return context.results.size() == wanted ? context.results.top().distance : unbounded_radius<distance_t>();
        };
        predicate_fold fold {relation, predicate, context.results, wanted};
        return scan_rounds_(query, metric, radius, fold, context, wanted);
    }

    /**
     *  @brief Scores @p query against every row published when the scan starts, handing every round
     *      to @p fold: the exact sibling of the folding @c search, for verbs that reduce all rows
     *      to something other than the nearest few, such as containment.
     *
     *  @param[inout] fold The reader of every scored round - see @c basic::fold. A @c sated_k ends
     *      the scan successfully.
     */
    template <typename metric_type_, typename fold_type_, typename context_allocator_type_,
              typename query_type_ = value_t>
        requires metric<metric_type_, value_t, distance_t> && query<query_type_, metric_type_, value_t> &&
                 fold<std::remove_cvref_t<fold_type_>, node_id_t, distance_t>
    expected<search_result_t> scan(query_type_ const& query, metric_type_& metric, fold_type_&& fold,
                                   traversal_context_over<context_allocator_type_>& context) noexcept {
        context.results.clear();
        auto const radius = []() noexcept { return unbounded_radius<distance_t>(); };
        return scan_rounds_(query, metric, radius, fold, context, 1);
    }

#pragma endregion Search

#pragma region Keys and Introspection

    /** Number of published nodes - the visibility watermark; safe any time. */
    std::size_t size() const noexcept {
        return static_cast<std::size_t>(atomic_ref<node_id_t const>(last_visible_id_).load(std::memory_order_acquire));
    }

    /** Rows the covered node chunks hold, as inserts, loads and @c reserve grow them; safe any
     *  time. */
    std::size_t capacity() const noexcept { return nodes_.capacity(); }

    /** The widest neighbor row any wiring may have written - the largest budget an insert or a
     *  refine wired with, a manually linked row, a loaded or a moved-in one - and never under one,
     *  so an edgeless graph still draws a landmark. A search derives its unset knobs from it; safe
     *  any time. */
    std::size_t connectivity() const noexcept {
        return max_of<std::size_t>(atomic_ref<std::size_t const>(widest_row_).load(std::memory_order_relaxed), 1);
    }

    /** The walk breadth @p config asks for, or the one a search derives from @ref connectivity
     *  when it asks for none. */
    std::size_t non_zero_expansion_factor(search_config_t const& config) const noexcept {
        return resolved_config_({config.expansion_factor}, connectivity()).expansion_factor;
    }

    /** Bytes the node records' chunks and the key multimap span. The neighbor rows and whatever the
     *  values allocate on their own are left out, since counting them would walk every record. */
    std::size_t memory_usage() const noexcept { return nodes_.memory_usage() + key_map_.memory_usage(); }

    /** The key stored under @p node_id - the authoritative direction, immutable
     *  after publication. One acquire load of the watermark gates the ID, so any input
     *  is safe any time; an unpublished ID answers @c malformed_k. */
    expected<key_t> key_of(node_id_t node_id) const noexcept {
        if (static_cast<std::size_t>(node_id) >= size()) return {key_t {}, status_t::malformed_k};
        return {nodes_[node_id].key, status_t::success_k};
    }

    /** The payload stored under @p node_id - never relocated, valid until teardown,
     *  immutable after publication. The same watermark gate as @c key_of, so any input
     *  is safe any time; an unpublished ID answers @c malformed_k. */
    expected<value_t const&> value_of(node_id_t node_id) const noexcept {
        if (static_cast<std::size_t>(node_id) >= size()) return {nullptr, status_t::malformed_k};
        return {&nodes_[node_id].value, status_t::success_k};
    }

    /** The payload under a @b published or wired @p node_id, unchecked: for traversal-contract
     *  callers - selectors and the walks themselves - whose IDs come from the walk's admitted
     *  candidates, where the watermark gate of @c value_of would re-prove a known fact or refuse a
     *  row another insert wired ahead of its turn. */
    value_t const& published_value_of(node_id_t node_id) const noexcept { return nodes_[node_id].value; }

    /** The out-degree of @p node_id: the same watermark gate as @c key_of, then the
     *  row lock - every post-publication row edit holds it, so the read is one
     *  consistent instant even while back-linking appends. */
    expected<std::size_t> degree_of(node_id_t node_id) noexcept {
        if (static_cast<std::size_t>(node_id) >= size()) return {0, status_t::malformed_k};
        node_t& node = nodes_[node_id];
        node.lock();
        std::size_t const degree = node.neighbors.size();
        node.unlock();
        return {degree, status_t::success_k};
    }

    /** One published node under @p key: @c malformed_k when absent. Duplicates surface in an
     *  unspecified order - use @c find_each to see them all. Wait-free, concurrent
     *  with everything. */
    expected<node_id_t> find(key_t const& key) const noexcept { return key_map_.find(key); }

    /** Invokes @p callback with the ID of every published node under @p key -
     *  wait-free, concurrent with everything. */
    template <typename callback_type_>
    void find_each(key_t const& key, callback_type_&& callback) const noexcept {
        key_map_.find_each(key, std::forward<callback_type_>(callback));
    }

    /** Whether at least one published node exists under @p key. */
    bool contains(key_t const& key) const noexcept { return key_map_.contains(key); }

    /** Number of published nodes under @p key. */
    std::size_t count(key_t const& key) const noexcept { return key_map_.count(key); }

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
        health_result_t reachability;

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
     *  @param[inout] context The calling thread's scratch over any allocator; its visited set and
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
        // mutator; the row-ID tripwires stay as a corruption net, never an overflow.
        report.degree_min = (std::numeric_limits<std::size_t>::max)();
        std::size_t total_neighbors = 0;
        for (std::size_t node = 0; node != node_count; ++node) {
            neighbors_buffer_t const& row = nodes_[node].neighbors;
            report.degree_min = min_of(report.degree_min, row.size());
            report.degree_max = max_of(report.degree_max, row.size());
            total_neighbors += row.size();
            for (std::size_t link = 0; link != row.size(); ++link) {
                if (static_cast<std::size_t>(row[link]) >= node_count) {
                    report.status = status_t::internal_k;
                    continue;
                }
                neighbors_buffer_t const& reverse = nodes_[row[link]].neighbors;
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
                neighbors_buffer_t const& row = nodes_[node].neighbors;
                for (std::size_t link = 0; link != row.size(); ++link) {
                    // The same corruption net as the edge sweep above.
                    if (static_cast<std::size_t>(row[link]) >= node_count) {
                        report.status = status_t::internal_k;
                        continue;
                    }
                    if (context.visited.insert(row[link]) == membership_t::absent_k)
                        context.staged_ids.push_back_reserved(row[link]);
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
                std::size_t const marked = audit.sweep(step);
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
         *  every slice starts where the plan's prefix sums say. */
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
                neighbors_buffer_t const& row = index_->nodes_[node].neighbors;
                for (std::size_t link = 0; link != row.size(); ++link) columns.neighbor_ids[ids_cursor++] = row[link];
            }

            // Each value fills exactly the slice its sum reserved; a codec needing more refuses itself.
            for (node_id_t const node : node_range) {
                std::size_t const first = columns.value_offsets[node];
                std::size_t const bound = columns.value_offsets[node + 1];
                status_t const encoded = codec.save(index_->nodes_[node].value,
                                                    columns.values.subspan(first, bound - first));
                if (failed(encoded)) return encoded;
            }
            atomic_add(atomic_ref<std::size_t>(covered), node_range.size(), std::memory_order_release);
            return status_t::success_k;
        }

        /** Pronounces the columns complete & releases the hold: @c config_mismatch_k
         *  unless the steps' sizes sum to the row count - overlapping steps that still
         *  sum right are the caller's own lie. A success spends the plan, so a second call
         *  answers @c uninitialized_k instead of unlocking a hold it no longer owns. */
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
     *  @c config_mismatch_k, so the key column's length always names the row count. */
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
     *  @brief Row & element counts a save needs right now. Takes the shared mutex's unique
     *      side for the tally; live mutators and open plans answer @c contended_k at entry.
     *
     *  @param[in] codec Sizes each stored value's encoded form.
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
     *  @param[in] codec Encodes each stored value into its slice. */
    template <typename value_codec_type_>
        requires value_codec<value_codec_type_, value_t>
    status_t save(columns_t columns, value_codec_type_ const& codec) noexcept {
        auto [plan, status] = plan_save(columns, codec);
        if (failed(status)) return status;
        node_id_t const node_count = static_cast<node_id_t>(plan.columns.keys.size());
        if (status_t const copied = plan.save({0, node_count}, codec); failed(copied)) return copied;
        return plan.finish();
    }

    /** One step-parallel load in flight - the mirror of @ref save_plan_t: the
     *  vetted columns, the covered-node tally, and the shared mutex's unique hold,
     *  released when the plan dies or @c finish publishes the loaded graph. */
    struct [[nodiscard]] load_plan_t {

        /** The vetted columns being decoded. */
        columns_view_t columns {};

        /** Nodes the steps covered so far, mutated through @c atomic_ref. */
        std::size_t covered = 0;

        load_plan_t() noexcept = default;
        load_plan_t(ram_index* index, columns_view_t columns, std::size_t first_seat, unique_lock_t&& hold) noexcept
            : columns(columns), index_(index), first_seat_(first_seat), hold_(std::move(hold)) {}
        load_plan_t(load_plan_t&& other) noexcept
            : columns(other.columns), covered(other.covered), index_(std::exchange(other.index_, nullptr)),
              first_seat_(other.first_seat_), hold_(std::move(other.hold_)) {}
        load_plan_t& operator=(load_plan_t&&) = delete;

        /** A plan that dies before @c finish wipes the keys its steps registered, still
         *  under its hold: an insert on the empty index would otherwise dispense IDs the
         *  unpublished attempt's keys still name, and spend the map's seats on the leftovers. */
        ~load_plan_t() noexcept {
            if (index_) index_->key_map_.clear();
        }

        /** Decodes the node range's keys, neighbor rows & values into its
         *  pre-constructed records and registers the keys - the multimap inserts are
         *  concurrency-safe, so any number of caller threads run their steps. */
        template <typename value_codec_type_>
            requires value_codec<value_codec_type_, value_t>
        status_t load(iota_view<node_id_t> node_range, value_codec_type_ const& codec) noexcept {
            if (!index_) return status_t::uninitialized_k;
            std::size_t const node_count = columns.keys.size();
            if (node_range.bound < node_range.first || static_cast<std::size_t>(node_range.bound) > node_count)
                return status_t::config_mismatch_k;

            // Each node takes the seat its position names, so concurrent steps never share one.
            typename key_map_t::seats_t seats {first_seat_ + node_range.first, first_seat_ + node_range.bound};
            // The plan vetted both prefix-sum columns and every ID, and reserved every row, so the
            // slices are trusted and the rows fit.
            std::size_t widest_row = 0;
            for (node_id_t const node : node_range) {
                node_t& record = index_->nodes_[node];
                record.key = columns.keys[node];

                std::size_t const ids_first = columns.neighbor_offsets[node];
                std::size_t const ids_bound = columns.neighbor_offsets[node + 1];
                neighbors_buffer_t& row = record.neighbors;
                row.clear();
                for (std::size_t link = ids_first; link != ids_bound; ++link)
                    row.push_back_reserved(columns.neighbor_ids[link]);
                widest_row = max_of(widest_row, row.size());

                std::size_t const bytes_first = columns.value_offsets[node];
                std::size_t const bytes_bound = columns.value_offsets[node + 1];
                if (status_t const decoded = codec.load(columns.values.subspan(bytes_first, bytes_bound - bytes_first),
                                                        record.value);
                    failed(decoded))
                    return decoded;

                index_->key_map_.insert(seats, record.key, node);
            }
            index_->widen_(widest_row);
            atomic_add(atomic_ref<std::size_t>(covered), node_range.size(), std::memory_order_release);
            return status_t::success_k;
        }

        /** Publishes the loaded graph in the tier-wide order - claims, identity,
         *  watermark last - and releases the hold: @c config_mismatch_k unless the
         *  steps' sizes sum to the row count. */
        status_t finish() noexcept {
            if (!index_) return status_t::uninitialized_k;
            std::size_t const node_count = columns.keys.size();
            if (atomic_ref<std::size_t>(covered).load(std::memory_order_acquire) != node_count)
                return status_t::config_mismatch_k;
            atomic_ref<std::size_t>(index_->claimed_count_).store(node_count, std::memory_order_relaxed);
            atomic_ref<node_id_t>(index_->next_dispensed_id_)
                .store(static_cast<node_id_t>(node_count), std::memory_order_relaxed);
            atomic_ref<node_id_t>(index_->last_visible_id_)
                .store(static_cast<node_id_t>(node_count), std::memory_order_release);
            hold_.unlock();
            index_ = nullptr; // Published: the plan is spent, and its death wipes nothing.
            return status_t::success_k;
        }

      private:

        /** The index being repopulated, whose unique side @c hold_ occupies; null in the
         *  empty plan a refused @ref plan_load returns, and once @c finish published. */
        ram_index* index_ = nullptr;

        /** The seat node zero's key takes, claimed with every other row's at planning. */
        std::size_t first_seat_ = 0;

        /** The unique hold spanning the plan's whole lifetime. */
        unique_lock_t hold_;
    };

    /**
     *  @brief Opens a step-parallel load into this empty index: takes the shared mutex's unique
     *      side - held inside the returned plan until its @c finish or its death - vets both
     *      prefix-sum columns and every neighbor ID in full, and wipes the key multimap.
     *
     *  The vetted rows' records are then covered, their neighbor rows reserved and their key seats
     *  claimed, so the plan's own @c load calls decode the per-node bulk and land no link or key
     *  that needs memory; nothing publishes until its @c finish.
     *
     *  @param[in] columns The five columns a prior @c save filled; the key column's length
     *      is the row count. */
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
        // A dying plan wipes the keys it registered, so the map arrives whole here.
        auto [seats, seat_status] = key_map_.claim(node_count);
        if (failed(seat_status)) return {{}, seat_status};
        return {load_plan_t {this, columns, seats.next, std::move(hold)}, status_t::success_k};
    }

    /**
     *  @brief Loads a snapshot into this empty index as one plan with a single step -
     *      exclusive: decodes each value into its pre-constructed record, rebuilds the key
     *      multimap, and publishes the watermark last.
     *
     *  The key map is wiped at planning, so a failed attempt may be retried with corrected columns.
     *
     *  @param[in] columns The five columns a prior @c save filled.
     *  @param[in] codec Decodes each value slice into its record. */
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

    /** Raises @ref connectivity to @p row, a row some wiring may write; a relaxed peek first, so a
     *  graph at its budget never writes the shared line. */
    void widen_(std::size_t row) noexcept {
        atomic_ref<std::size_t> const widest {widest_row_};
        if (widest.load(std::memory_order_relaxed) < row) atomic_fetch_max(widest, row, std::memory_order_relaxed);
    }

    /** Forgets every record below @p visible, every key and every counter: what @c clear does under
     *  the unique side, and what a landed donor becomes under its plan's. */
    void forget_(node_id_t visible) noexcept {
        for (std::size_t node = 0; node != static_cast<std::size_t>(visible); ++node) {
            node_t& record = nodes_[node];
            record.value = std::make_obj_using_allocator<value_t>(allocator_);
            record.key = key_t {};
            record.neighbors.clear();
            record.state = 0;
        }
        key_map_.clear();
        atomic_ref<std::size_t>(widest_row_).store(0, std::memory_order_relaxed);
        atomic_ref<std::size_t>(claimed_count_).store(0, std::memory_order_relaxed);
        atomic_ref<node_id_t>(next_dispensed_id_).store(0, std::memory_order_relaxed);
        atomic_ref<node_id_t>(last_visible_id_).store(0, std::memory_order_release);
    }

    /** Closes the claims at the watermark: one compare-exchange that succeeds only while every
     *  claimed ID is published, after which every batch refuses until @ref reopen_. */
    status_t close_() noexcept {
        std::size_t visible = size();
        return atomic_ref<std::size_t>(claimed_count_)
                       .compare_exchange_strong(visible, closed_claims_k, std::memory_order_acq_rel,
                                                std::memory_order_relaxed)
                   ? status_t::success_k
                   : status_t::contended_k;
    }

    /** Reopens closed claims at the watermark, past every row a landing published meanwhile. */
    void reopen_() noexcept { atomic_ref<std::size_t>(claimed_count_).store(size(), std::memory_order_release); }

    /** An audit over the IDs below @p visible, its bitsets placed by @p allocator. */
    expected<health_plan_t> plan_audit_(node_id_t visible, allocator_t const& allocator) noexcept {
        // An empty graph is audited, not refused: no words to sweep, no steps, zero reached.
        std::size_t const words = divide_round_up<std::size_t>(visible, health_plan_t::lanes_k);
        typename health_plan_t::bits_t bits {allocator};
        if (failed(bits.resize(health_plan_t::bitsets_k * words)))
            return {health_plan_t {allocator}, status_t::bad_alloc_k};
        return {health_plan_t {*this, visible, std::move(bits)}, status_t::success_k};
    }

    /** What @ref plan_refine and @ref plan_absorb share: every check, hold, claim and reservation
     *  of a pass over @p nodes once @p donors land, every row reserved to @p connectivity. */
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
        if (status_t const sized = reserve_traversal(context, connectivity, config); failed(sized))
            return refuse(sized);

        // From here the plan reopens the claims on any refusal, and at its death.
        plan.hold_ = shared_lock_t {shared_mutex_};
        if (status_t const closed = close_(); failed(closed)) return refuse(closed);
        plan.index_ = this;
        if (failed(plan.donor_holds_.reserve_exactly(donors.size()))) return refuse(status_t::bad_alloc_k);
        node_id_t const visible = static_cast<node_id_t>(size());
        std::size_t rows = visible, parts = visible != 0;
        for (ram_index* donor : donors) {
            unique_lock_t donor_hold {donor->shared_mutex_, std::try_to_lock};
            if (!donor_hold) return refuse(status_t::contended_k);
            auto const [donor_rows, watermark_status] = donor->watermark_();
            if (failed(watermark_status)) return refuse(watermark_status);
            plan.donor_holds_.push_back_reserved(std::move(donor_hold));
            if (static_cast<std::size_t>(donor_rows) > max_node_count_k - rows) return refuse(status_t::overflow_k);
            rows += donor_rows;
            parts += donor_rows != 0;
        }
        if (status_t const covered = nodes_.cover(rows); failed(covered)) return refuse(covered);
        if (rows != visible) {
            auto const [seats, seat_status] = key_map_.claim(rows - visible);
            if (failed(seat_status)) return refuse(seat_status);
            plan.seats_ = seats;
        }
        if (parts > 1) {
            if (failed(plan.part_bases_.reserve_exactly(parts))) return refuse(status_t::bad_alloc_k);
            if (visible) plan.part_bases_.push_back_reserved(0);
            std::size_t base = visible;
            for (ram_index const* donor : donors) {
                if (donor->size()) plan.part_bases_.push_back_reserved(static_cast<node_id_t>(base));
                base += donor->size();
            }
        }
        auto [audit, audit_status] = plan_audit_(static_cast<node_id_t>(rows), context.allocator);
        if (failed(audit_status)) return refuse(audit_status);
        plan.audit = std::move(audit);

        // Rows last, under their locks, as searches copy them meanwhile; the donors are held.
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
        node_id_t const bound = min_of(nodes.bound, static_cast<node_id_t>(rows));
        plan.nodes_ = {min_of(nodes.first, bound), bound};
        plan.donors_ = donors;
        plan.rounds_ = parts > 1 ? static_cast<std::size_t>(std::bit_width(parts - 1)) : 1u;
        plan.walk_ = resolved_config_(config, connectivity);
        plan.connectivity_ = connectivity;
        plan.phase_ = donors.empty() ? plan.first_pass_phase_() : phase_t::land_k;
        return {std::move(plan), status_t::success_k};
    }

    /** What @ref plan_split does: every check, hold, claim, cover and reservation of a split of
     *  this index into @p receivers, every row reserved to @p connectivity. */
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
        if (status_t const sized = reserve_traversal(context, connectivity, config); failed(sized))
            return refuse(sized);

        // The source is held as a donor is, so no batch lands in it while its rows move out.
        unique_lock_t source_hold {shared_mutex_, std::try_to_lock};
        if (!source_hold) return refuse(status_t::contended_k);
        auto const [rows, watermark_status] = watermark_();
        if (failed(watermark_status)) return refuse(watermark_status);
        plan.source_ = this;
        plan.source_hold_ = std::move(source_hold);
        plan.receivers_ = receivers;
        std::size_t const capacity = divide_round_up<std::size_t>(rows, parts);
        if (parts * capacity > static_cast<std::size_t>(missing_k)) return refuse(status_t::overflow_k);
        if (failed(plan.receiver_holds_.reserve_exactly(parts)) ||
            failed(plan.receiver_seats_.reserve_exactly(parts)) || failed(plan.part_extents_.resize(parts)) ||
            failed(plan.part_slots_.resize(rows)) || failed(plan.part_members_.resize(parts * capacity)))
            return refuse(status_t::bad_alloc_k);
        // One audit over a full part sweeps every receiver in turn, rebound as its turn comes.
        auto [audit, audit_status] = plan_audit_(static_cast<node_id_t>(capacity), context.allocator);
        if (failed(audit_status)) return refuse(audit_status);
        plan.audit = std::move(audit);

        // From here the plan reopens the claims of every receiver it closed on any refusal, and at
        // its death; the claims closed at an empty watermark prove no batch is in flight.
        for (ram_index* receiver : receivers) {
            shared_lock_t receiver_hold {receiver->shared_mutex_};
            if (status_t const closed = receiver->close_(); failed(closed)) return refuse(closed);
            plan.receiver_holds_.push_back_reserved(std::move(receiver_hold));
            if (receiver->size()) return refuse(status_t::config_mismatch_k);
        }
        for (ram_index* receiver : receivers) {
            if (status_t const covered = receiver->nodes_.cover(capacity); failed(covered)) return refuse(covered);
            auto [seats, seat_status] = receiver->key_map_.claim(capacity);
            if (failed(seat_status)) return refuse(seat_status);
            plan.receiver_seats_.push_back_reserved(seats);
        }
        // Rows last; each moves its buffer into a receiver, where re-selecting allocates nothing.
        for (node_id_t node_id = 0; node_id != rows; ++node_id)
            if (status_t const grown = nodes_[node_id].neighbors.reserve_exactly(connectivity); failed(grown))
                return refuse(grown);

        for (ram_index* receiver : receivers) {
            receiver->widen_(connectivity);
            receiver->widen_(this->connectivity());
        }
        plan.index_ = receivers[0];
        plan.seed_ = seed;
        plan.walk_ = resolved_config_(config, connectivity);
        plan.connectivity_ = connectivity;
        plan.phase_ = phase_t::partition_k;
        return {std::move(plan), status_t::success_k};
    }

    /**
     *  @brief One node's re-selection: a walk across into @p across alone fills the candidates
     *      first, then the node's own row seeds a walk over every ID below @p visible and joins the
     *      candidates past that walk's breadth.
     *  @return Whether the new neighbors differ as a set from the ones replaced.
     *
     *  @p selector picks from those candidates, the rewrite runs under the node's lock, and every
     *  kept neighbor is back-linked. No metric verb runs under a lock, and no row allocates.
     */
    template <typename metric_type_, typename context_type_, typename selector_type_>
    bool refine_one_(node_id_t node_id, node_id_t visible, iota_view<node_id_t> across, insert_config_t walk,
                     metric_type_& metric, context_type_& context, selector_type_ const& selector) noexcept {
        constexpr synchronization_t concurrent_k = synchronization_t::concurrent_k;
        value_t const& value = published_value_of(node_id);
        context.results.clear();
        context.frontier.clear();
        context.visited.clear();
        // Visited before anything stages, so the node never joins its own candidates.
        [[maybe_unused]] membership_t const self_marked = context.visited.insert(node_id);
        part_admission_t const other {across.first, min_of(across.bound, visible)};
        if (other.first_id < other.first_invisible_id) {
            search_around_<concurrent_k>(value, metric, other, walk.expansion_factor, selector.connectivity,
                                         context.results, context, dummy_fold_t {});
            context.frontier.clear();
        }
        snapshot_admission_t const every {visible};
        seed_from_row_(node_id, value, metric, every, walk.expansion_factor, context);
        search_around_<concurrent_k>(value, metric, every, walk.expansion_factor, selector.connectivity,
                                     context.results, context, dummy_fold_t {});
        add_row_candidates_(node_id, value, metric, every, context);
        selector(*this, metric, context);

        // Neighbors are distinct, so equal sizes and every pick present mean the same set.
        node_t& node = nodes_[node_id];
        context.reverse_ids.clear();
        node.lock();
        std::size_t const kept_count = min_of(min_of(context.selected_neighbors.size(), selector.connectivity),
                                              node.neighbors.capacity());
        bool changed = kept_count != node.neighbors.size();
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
        back_link_<concurrent_k>(node_id, metric, selector, context);
        return changed;
    }

    /** Seeds a re-selection's walk with @p node_id 's own row: its admitted, unvisited neighbors
     *  score into the candidates and the frontier, in as many rounds as the metric's staging takes,
     *  so every link already wired competes. A refused copy seeds nothing, and the walk samples
     *  landmarks instead. */
    template <typename metric_type_, typename admission_type_, typename context_type_>
    void seed_from_row_(node_id_t node_id, value_t const& value, metric_type_& metric, admission_type_ const& admission,
                        std::size_t expansion_factor, context_type_& context) noexcept {
        context.staged_ids.clear();
        auto const [degree, copy_status] = copy_row_<synchronization_t::concurrent_k>(node_id, context);
        if (failed(copy_status) || failed(context.visited.reserve(context.visited.size() + degree))) {
            ++context.telemetry.saturated_walks;
            return;
        }
        // Each round compacts its IDs into the prefix of the row copy it has already read past.
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

    /** Adds @p node_id 's current row to the candidates a re-selection picks from, past the walk's
     *  breadth: the walk's nearer finds evict the far members, the long links among them, which
     *  must still compete. Scored in as many rounds as the metric's staging takes; a refused copy
     *  or growth, or a metric refusing an empty round, counts a saturated walk. */
    template <typename metric_type_, typename admission_type_, typename context_type_>
    void add_row_candidates_(node_id_t node_id, value_t const& value, metric_type_& metric,
                             admission_type_ const& admission, context_type_& context) noexcept {
        context.staged_ids.clear();
        auto const [degree, copy_status] = copy_row_<synchronization_t::concurrent_k>(node_id, context);
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
     *  the value & key moved into the record, the row wired and marked, then the back-links. */
    template <synchronization_t synchronization_, typename metric_type_, typename context_type_,
              typename selector_type_>
    void land_(node_id_t new_node_id, value_t& incoming, key_t const& key, metric_type_& metric, context_type_& context,
               selector_type_ const& selector, insert_config_t walk) noexcept {
        std::size_t const connectivity = selector.connectivity;
        context.results.clear();
        context.frontier.clear();
        context.visited.clear();

        // Walk against the caller's still-unmoved value; the first node of a virgin index has
        // nothing to walk to and publishes with an empty neighbors list. The ID is already
        // dispensed, so a refused walk wires a thinner node rather than abandoning it.
        if (new_node_id) {
            node_id_t const first_invisible_id =
                atomic_ref<node_id_t>(last_visible_id_).load(std::memory_order_acquire);
            search_around_<synchronization_>(
                incoming, metric, dispensed_admission_t {this, first_invisible_id, new_node_id}, walk.expansion_factor,
                connectivity, context.results, context, dummy_fold_t {});
        }
        selector(*this, metric, context);

        node_t& record = nodes_[new_node_id];
        record.value = std::move(incoming);
        record.key = key;
        context.reverse_ids.clear();
        std::size_t const selected_count = min_of(context.selected_neighbors.size(), connectivity);
        // A failed load leaves its unpublished rows filled; this one starts from nothing.
        record.neighbors.clear();
        // A refused row growth wires a thinner node, counted like a refused walk, since the
        // dispensed ID must still publish.
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
     *  @brief Appends @p node_id's live neighbors into @c context.staged_ids - under the row's
     *      lock on the concurrent contract, as a plain copy otherwise - topping up the staging
     *      surfaces first, since manual linking grows rows past any estimate.
     *
     *  Bounded copies only, never metric verbs.
     *  @return The copied degree, or the refusal that kept the row out of staging. */
    template <synchronization_t synchronization_, typename context_type_>
    expected<std::size_t> copy_row_(node_id_t node_id, context_type_& context) noexcept {
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

    /** What a refine's walk across walks: one part's published IDs alone, so the rows of other
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

    /** What an insertion walks: the IDs below its snapshot, and past it every row whose state shows
     *  it wired. The acquire of that state carries the row's value and neighbors, so batches wired
     *  side by side link to each other rather than forming islands. */
    struct dispensed_admission_t {

        /** The index the walk reads. */
        ram_index const* index = nullptr;

        /** The watermark snapshot. */
        node_id_t first_invisible_id = 0;

        /** The walking row. */
        node_id_t node_id = 0;

        /** Whether the walk may locate @p candidate_id. */
        bool admits(node_id_t candidate_id) const noexcept {
            return candidate_id < first_invisible_id || index->nodes_[candidate_id].wired();
        }

        /** Landmarks come from the IDs from this one, the first. */
        node_id_t sampling_first() const noexcept { return 0; }

        /** Landmarks come from the IDs below this bound: each was dispensed ahead of the walking row,
         *  so most are wired already. */
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
     *  Draws stay within the IDs @p admission samples from. Past the snapshot a row may stay
     *  unwired for the whole walk, so the re-rolls of unadmitted draws stop once they outnumber the
     *  IDs there.
     */
    template <typename metric_type_, typename query_type_, typename admission_type_, typename context_type_,
              typename fold_type_>
    expected<walk_step_t> sample_landmarks_(query_type_ const& query, metric_type_& metric,
                                            admission_type_ const& admission, std::size_t wanted,
                                            std::size_t expansion_factor, context_type_& context,
                                            fold_type_&& fold) const noexcept {
        if (status_t const seen = context.visited.reserve(context.visited.size() + wanted); failed(seen)) {
            ++context.telemetry.saturated_walks;
            return {walk_step_t::exhausted_k, seen};
        }
        context.staged_ids.clear();
        metric.reset(query);
        // `visited` may also hold IDs no draw names, like a later batch's rows, so the size check
        // can only under-sample, never let the probe loop spin.
        node_id_t const sampling_first = admission.sampling_first(), sampling_bound = admission.sampling_bound();
        node_id_t unwired_draws = 0;
        while (context.staged_ids.size() != wanted &&
               context.visited.size() < static_cast<std::size_t>(sampling_bound - sampling_first)) {
            node_id_t draw = sampling_first +
                             uniform_below<node_id_t> {sampling_bound - sampling_first}(context.random_generator);
            while (context.visited.contains(draw)) draw = draw + 1 == sampling_bound ? sampling_first : draw + 1;
            if (!admission.admits(draw)) {
                if (++unwired_draws == sampling_bound - admission.first_invisible_id) break;
                continue;
            }
            if (metric.push(published_value_of(draw)) == metric_staging_t::refused_k)
                break; // Unmarked, so a later round may retry it.
            [[maybe_unused]] membership_t const marked = context.visited.insert(draw);
            context.staged_ids.push_back_reserved(draw);
        }
        std::size_t const staged_count = context.staged_ids.size();
        if (!staged_count) return {walk_step_t::exhausted_k, status_t::success_k};
        drain_round(metric, staged_count, context, unbounded_radius<distance_t>());
        walk_demand_t const demand = fold.absorb(std::span<node_id_t const>(context.staged_ids),
                                                 std::span<distance_t const>(context.staged_distances));
        settle_fold(fold);
        if (demand == walk_demand_t::sated_k) return {walk_step_t::sated_k, status_t::success_k};
        for (std::size_t lane = 0; lane != staged_count; ++lane) {
            scored_node_t const scored {context.staged_ids[lane], context.staged_distances[lane]};
            [[maybe_unused]] bool const ranked = context.results.insert(scored_node_t {scored}, expansion_factor);
            if (status_t const queued = context.frontier.insert(scored_node_t {scored}); failed(queued)) {
                ++context.telemetry.saturated_walks;
                return {walk_step_t::advanced_k, queued};
            }
        }
        return {walk_step_t::advanced_k, status_t::success_k};
    }

    /** One round's expansion half: peels up to @p filling_rows frontier entries within the stopping
     *  radius, copies their rows, and pushes the admission- and visited-filtered union into the
     *  metric, compacted in @c context.staged_ids. A full staging ends the round early, with the
     *  entry it cut short back on the frontier. */
    template <synchronization_t synchronization_, typename metric_type_, typename query_type_, typename admission_type_,
              typename stopping_type_, typename context_type_>
    walk_step_t expand_frontier_(query_type_ const& query, metric_type_& metric, admission_type_ const& admission,
                                 std::size_t expansion_factor, std::size_t filling_rows, stopping_type_ const& stopping,
                                 context_type_& context) noexcept {
        context.staged_ids.clear();
        metric.reset(query);
        for (std::size_t expanded_count = 0; expanded_count != filling_rows && !context.frontier.empty();
             ++expanded_count) {
            scored_node_t const best = context.frontier.top();
            bool const stopping_full = stopping.size() == expansion_factor;
            if (stopping_full && best.distance > stopping.top().distance) {
                if (expanded_count == 0) return walk_step_t::exhausted_k;
                break;
            }

            std::size_t const segment_start = context.staged_ids.size();
            auto const [degree, status] = copy_row_<synchronization_>(best.node_id, context);
            if (failed(status) || failed(context.visited.reserve(context.visited.size() + degree))) {
                // Drop the raw row copy: only pushed, filtered IDs may reach the drain.
                // The entry stays on the frontier - nothing is lost, only postponed.
                context.staged_ids.resize_reserved(segment_start);
                ++context.telemetry.saturated_walks;
                return walk_step_t::saturated_k;
            }
            context.frontier.pop();
            // The next entry's record, not its row, which a writer may move until a copy locks it.
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
                    context.staged_ids.resize_reserved(kept);
                    // A full staging ends the round: the entry goes back on the frontier, and
                    // the next round stages the rest of its row past the IDs this one marked.
                    // Only a metric refusing an empty round saturates the walk.
                    if (kept && succeeded(context.frontier.insert(scored_node_t {best})))
                        return walk_step_t::advanced_k;
                    ++context.telemetry.saturated_walks;
                    return walk_step_t::saturated_k;
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

    /** The rounds of both scans: every row below the watermark snapshot staged as many at a time as
     *  the metric takes, drained within @p radius, and handed to @p fold, which may end the scan
     *  with @c sated_k. A metric refusing an empty round can never finish, which a scan reports as
     *  a refused allocation. */
    template <typename query_type_, typename metric_type_, typename radius_type_, typename fold_type_,
              typename context_type_>
    expected<search_result_t> scan_rounds_(query_type_ const& query, metric_type_& metric, radius_type_ const& radius,
                                           fold_type_& fold, context_type_& context, std::size_t wanted) noexcept {
        node_id_t const first_invisible_id = atomic_ref<node_id_t>(last_visible_id_).load(std::memory_order_acquire);
        if (!first_invisible_id) return {};
        // A scan stages rounds and ranks, and never touches the frontier or the visited set.
        std::size_t const staging = round_capacity(connectivity(), {wanted});
        if (failed(context.staged_ids.reserve(staging)) || failed(context.staged_distances.reserve(staging)) ||
            failed(context.results.reserve(wanted)))
            return {{}, status_t::bad_alloc_k};
        traversal_telemetry_t const telemetry_before = context.telemetry;
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
     *  walk stops on @p stopping, its own candidates or the admitted set. */
    template <synchronization_t synchronization_, typename metric_type_, typename fold_type_, typename stopping_type_,
              typename context_allocator_type_, typename query_type_>
    expected<search_result_t> fold_walk_(query_type_ const& query, metric_type_& metric, fold_type_&& fold,
                                         stopping_type_ const& stopping,
                                         traversal_context_over<context_allocator_type_>& context,
                                         search_config_t config) noexcept {
        std::size_t const widest_row = connectivity();
        insert_config_t const walk = resolved_config_({config.expansion_factor}, widest_row);
        if (status_t const sized = reserve_traversal(context, widest_row, walk); failed(sized)) return {{}, sized};
        std::size_t const expansion_factor = walk.expansion_factor;
        context.results.clear();
        context.frontier.clear();
        context.visited.clear();

        // One watermark snapshot bounds the whole walk; a certified mode proves its own
        // certification through the shared mutex's wait-free probes instead of trusting it.
        node_id_t first_invisible_id = 0;
        [[maybe_unused]] u64_t entry_generation = 0;
        if constexpr (synchronization_ == synchronization_t::concurrent_k)
            first_invisible_id = atomic_ref<node_id_t>(last_visible_id_).load(std::memory_order_acquire);
        else {
            auto const entry_probe = shared_mutex_.snapshot();
            if (!entry_probe.unlocked()) return {{}, status_t::contended_k};
            entry_generation = entry_probe.generation;
            auto const [watermark, watermark_status] = watermark_();
            if (failed(watermark_status)) return {{}, watermark_status};
            first_invisible_id = watermark;
        }
        if (!first_invisible_id) return {};

        traversal_telemetry_t const telemetry_before = context.telemetry;
        search_around_<synchronization_>(query, metric, snapshot_admission_t {first_invisible_id}, expansion_factor,
                                         widest_row, stopping, context, fold);

        // A refused walk still ranked whatever it reached; the telemetry says the ranking is
        // short because scratch ran out, not because the graph had nothing better.
        search_result_t result;
        status_t status = status_t::success_k;
        result.count = context.results.size();
        result.telemetry = context.telemetry - telemetry_before;
        if constexpr (synchronization_ != synchronization_t::concurrent_k) {
            // The fence pairs with the release fence in `lock_shared`: a walk read that met a
            // mutator's write proves its acquisition to the reload, which then reads a moved
            // word. The watermark equality is the belt against generation wrap driven by inserts.
            std::atomic_thread_fence(std::memory_order_acquire);
            auto const exit_probe = shared_mutex_.snapshot();
            if (!exit_probe.unlocked() || exit_probe.generation != entry_generation ||
                atomic_ref<node_id_t>(last_visible_id_).load(std::memory_order_acquire) != first_invisible_id)
                status = status_t::contended_k;
        }
        return {result, status};
    }

    /**
     *  @brief The shared walk of @c search & @c insert: seed, then per round expand the
     *      nearest frontier entries and score the staged union in one builder round - every
     *      scored round, seeds included, reaching the fold before the candidates.
     *
     *  The walk locates only the IDs @p admission admits: a search's snapshot, or an insertion's
     *  snapshot widened by the rows wired ahead of their turns. It never fails hard: an exhausted
     *  admitted set, a refused staging or a sated fold ends it normally, and a refused re-table ends
     *  it early with the ranking it has and counts a saturated walk - an insertion wires the thinner
     *  row regardless, and a search reports the count so the caller can widen the scratch. Frontier
     *  starvation with unfilled candidates re-seeds and counts a restart. The frontier-admission
     *  rule: a candidate enters the frontier iff it improved the results, or the frontier would
     *  otherwise be empty - the seedings.
     */
    template <synchronization_t synchronization_, typename metric_type_, typename query_type_, typename admission_type_,
              typename stopping_type_, typename context_type_, typename fold_type_>
    void search_around_(query_type_ const& query, metric_type_& metric, admission_type_ const& admission,
                        std::size_t expansion_factor, std::size_t connectivity, stopping_type_ const& stopping,
                        context_type_& context, fold_type_&& fold) noexcept {
        // A round expands the rows whose neighbors fill a walk's candidates, as `round_capacity`
        // sized.
        std::size_t const filling_rows = divide_round_up(expansion_factor, connectivity);
        while (true) {
            // An empty frontier seeds: the first round, or a restart after it drained before the
            // candidates filled. Nothing left to seed from ends the walk.
            if (context.frontier.empty()) {
                if (stopping.size() >= expansion_factor) return;
                bool const restarting = !context.results.empty();
                expected<walk_step_t> const seeded = sample_landmarks_(query, metric, admission, connectivity,
                                                                       expansion_factor, context, fold);
                if (failed(seeded.status) || seeded.value != walk_step_t::advanced_k) return;
                context.telemetry.restarts += restarting;
                continue;
            }
            walk_step_t const expanded = expand_frontier_<synchronization_>(query, metric, admission, expansion_factor,
                                                                            filling_rows, stopping, context);
            if (expanded == walk_step_t::exhausted_k) return;
            walk_step_t const ranked = rank_round_(metric, expansion_factor, stopping, context, fold);
            if (expanded == walk_step_t::saturated_k || ranked != walk_step_t::advanced_k) return;
        }
    }

    /** One round's scoring half: drains the staged union through the metric, hands it to the fold,
     *  and pushes what improved the candidates into the frontier, shedding the frontier past the
     *  full candidates' radius first so the pushes fit. */
    template <typename metric_type_, typename stopping_type_, typename context_type_, typename fold_type_>
    walk_step_t rank_round_(metric_type_& metric, std::size_t expansion_factor, stopping_type_ const& stopping,
                            context_type_& context, fold_type_&& fold) const noexcept {
        std::size_t const staged_count = context.staged_ids.size();
        if (!staged_count) return walk_step_t::advanced_k;
        // Full stopping candidates only admit what beats their furthest member, so that member's
        // distance is the radius, which a filtered walk takes from its admitted set.
        bool const stopping_full = stopping.size() == expansion_factor;
        drain_round(metric, staged_count, context,
                    stopping_full ? stopping.top().distance : unbounded_radius<distance_t>());
        walk_demand_t const demand = fold.absorb(std::span<node_id_t const>(context.staged_ids),
                                                 std::span<distance_t const>(context.staged_distances));
        settle_fold(fold);
        if (demand == walk_demand_t::sated_k) return walk_step_t::sated_k;
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
     *  metric cost; otherwise copy the full row, unlock, score the copy and the newcomer off-lock
     *  in as many rounds as the metric's staging takes, re-select through the selector, relock, and
     *  apply only if the row is still the one copied - a lost race starts over from the lock, so
     *  the newcomer competes for every row it selected. A row already linking back, as a refined
     *  node's neighbor may, is left as it is. Only a refused allocation, or a metric refusing an
     *  empty round, gives a link up, counted as a saturated walk. */
    template <synchronization_t synchronization_, typename metric_type_, typename selector_type_,
              typename context_type_>
    void back_link_(node_id_t new_node_id, metric_type_& metric, selector_type_ const& selector,
                    context_type_& context) noexcept {
        for (std::size_t reverse = 0; reverse != context.reverse_ids.size(); ++reverse)
            while (!back_link_one_<synchronization_>(context.reverse_ids[reverse], new_node_id, metric, selector,
                                                     context)) {}
    }

    /** One reverse link of @ref back_link_ into @p neighbor_id 's row; @c false when the row raced
     *  the re-selection and must be tried again from the lock. */
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

        // Off-lock: candidates = the copied row plus the newcomer, scored against the neighbor's
        // own stored value; every member is ranked, so nothing is too far.
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
            context.telemetry.computed_distances_in_rankings += staged_count;
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
        selector(*this, metric, context);

        // Relock & revalidate: apply the re-selection only onto the exact row it was computed
        // from; a raced row goes round again.
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
     *  overflows, a long one disagrees about the row count. For the three node-sized columns,
     *  whose length is what names the row count. */
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

    /** Vets untrusted columns in full - both prefix-sum columns sized to the rows and
     *  monotone over their element columns, every neighbor ID naming a row - so the step
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

    /** Batch capacity claims - the all-or-nothing admission, separate from
     *  identity: claims convert one-to-one into dispensed IDs, so dispensing never
     *  overflows, and a refused cover gives its claim back before anything is dispensed.
     *  The bounded add commits only within @c max_node_count_k, so a rejected batch
     *  never writes and the count never overshoots. */
    alignas(default_alignment_k) std::size_t claimed_count_ {0};

    /** Node-ID dispenser; one add hands a batch its whole range before the first walk.
     *  Its per-batch RMW gets its own line, so it never invalidates the watermark line
     *  every search snapshots. */
    alignas(default_alignment_k) node_id_t next_dispensed_id_ {0};

    /** The visibility watermark: every ID strictly below it is published. */
    alignas(default_alignment_k) node_id_t last_visible_id_ {0};

    /** What @ref connectivity answers, only ever raised through @ref widen_; beside the watermark,
     *  since every search reads both and this one stops moving once the graph's budget is met. */
    std::size_t widest_row_ {0};

    /** Key → ID multimap, covered by each batch's seats and by @c reserve. */
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
 *      copy of the current node's row.
 *
 *  Strictly read-only under every contract, so readers never stall writers. One watermark snapshot,
 *  captured at construction and widened by @c refresh, bounds everything: keys & values below it
 *  are immutable and read wait-free, rows are staged under the row lock on the concurrent mode and
 *  as plain copies on the certified ones. A staged row may name IDs at or past this cursor's
 *  snapshot - fresh back-links - and @c follow refuses those with @c malformed_k until a @c refresh
 *  admits them. The cursor starts past the end; @c seek(0) enters the graph.
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
     *  staged-row buffer grows on the first @c seek through @p allocator. */
    explicit ram_cursor(index_t& index, allocator_t allocator = {}) noexcept
        : index_(index), staged_row_(allocator),
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
    std::span<node_id_t const> neighbors() const noexcept { return {staged_row_.begin(), staged_row_.end()}; }

    /** Positions on @p node & stages its row: @c malformed_k at or past the
     *  snapshot, @c bad_alloc_k when the staging buffer refused to grow - the cursor
     *  stays where it was on either refusal. */
    status_t seek(node_id_t node) noexcept {
        if (static_cast<std::size_t>(node) >= static_cast<std::size_t>(visible_count_)) return status_t::malformed_k;
        typename index_t::node_t& record = index_.nodes_[node];
        if constexpr (synchronization_ == synchronization_t::concurrent_k) record.lock();
        std::size_t const degree = record.neighbors.size();
        status_t const staged = staged_row_.reserve(degree);
        if (succeeded(staged)) {
            staged_row_.clear();
            for (std::size_t link = 0; link != degree; ++link) staged_row_.push_back_reserved(record.neighbors[link]);
        }
        if constexpr (synchronization_ == synchronization_t::concurrent_k) record.unlock();
        if (failed(staged)) return staged;
        current_ = node;
        return status_t::success_k;
    }

    /** Advances to the next published ID, staging its row; past the last one the
     *  cursor turns false. A staging refusal parks on the new ID with an empty row -
     *  a later @c seek to the same ID retries the staging. */
    ram_cursor& operator++() noexcept {
        node_id_t const next = current_ + 1;
        if (static_cast<std::size_t>(next) >= static_cast<std::size_t>(visible_count_)) {
            current_ = visible_count_;
            staged_row_.clear();
            return *this;
        }
        if (failed(seek(next))) {
            current_ = next;
            staged_row_.clear();
        }
        return *this;
    }

    /** Deleted: it would copy the staged row - an allocation in disguise. */
    ram_cursor operator++(int) = delete;

    /** Hops to the staged row's @p position -th neighbor: @c malformed_k past the
     *  staged degree or when the neighbor sits at or past the snapshot. */
    status_t follow(std::size_t position) noexcept {
        if (position >= staged_row_.size()) return status_t::malformed_k;
        return seek(staged_row_[position]);
    }

  private:

    /** Certified contracts already ordered every publication, so no acquire. */
    inline static constexpr std::memory_order watermark_order_k = synchronization_ == synchronization_t::concurrent_k
                                                                      ? std::memory_order_acquire
                                                                      : std::memory_order_relaxed;

    index_t& index_;

    /** The private copy of the current node's row - @c follow's ground truth. */
    growing_buffer<node_id_t, allocator_t> staged_row_;

    /** This cursor's watermark snapshot; IDs strictly below it are readable. */
    node_id_t visible_count_;

    /** The current ID, or @c visible_count_ when past the end. */
    node_id_t current_;
};

} // namespace unum::usearch::basic
