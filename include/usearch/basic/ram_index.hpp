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

#include <atomic>       // `std::memory_order`
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

    /** Orders scored nodes nearest-first; the comparator of the sorted results pool. */
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

/** The four per-walk counters every traversal accumulates; results report one
 *  call's share as the difference of two snapshots. */
struct traversal_telemetry_t {

    /** Metric invocations, batched rounds counted once per scored candidate. */
    std::size_t computed_distances = 0;

    /** Frontier candidates whose neighbor lists were expanded. */
    std::size_t visited_members = 0;

    /** Re-seedings after the frontier drained prematurely - a connectivity signal. */
    std::size_t restarts = 0;

    /** Walks a scratch refusal ended early - ranked short, and on insertion wired thinner. */
    std::size_t saturated_walks = 0;

    traversal_telemetry_t operator-(traversal_telemetry_t const& before) const noexcept {
        return {computed_distances - before.computed_distances, visited_members - before.visited_members,
                restarts - before.restarts, saturated_walks - before.saturated_walks};
    }
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

/** Relative-neighborhood pruning: a candidate survives only if closer to the
 *  query than to every already-kept neighbor - pair distances drawn through the same
 *  metric in shrinking rounds, exiting each candidate on its first violation. The slots
 *  the rule leaves empty take the nearest rejected candidates, after the diverse picks. */
struct keep_diverse_t {

    /** The neighbors-per-node budget this selector prunes to. */
    std::size_t connectivity = connectivity_32_k;

    template <typename index_type_, typename metric_type_, typename context_type_>
    void operator()(index_type_ const& index, metric_type_& metric, context_type_& context) const noexcept {
        auto& selected = context.selected_neighbors;
        selected.clear();
        for (std::size_t position = 0; position != context.results.size() && selected.size() != connectivity;
             ++position) {
            auto const candidate = context.results[position];
            std::size_t const kept_count = selected.size();
            bool diverse = true;
            if (kept_count) {
                // One round per candidate: its own value queries the kept neighbors, and only pairs
                // closer than the candidate itself can disqualify it.
                metric.reset(index.published_value_of(candidate.node_id));
                std::size_t staged_count = 0;
                for (; staged_count != kept_count; ++staged_count)
                    if (metric.push(index.published_value_of(selected[staged_count].node_id)) ==
                        metric_staging_t::refused_k)
                        break;
                drain_round(metric, staged_count, context, candidate.distance);
                if (staged_count != kept_count) break; // Staging refused: the backfill completes it.
                for (std::size_t pair = 0; pair != staged_count && diverse; ++pair)
                    diverse = !(context.staged_distances[pair] < candidate.distance);
            }
            if (diverse) selected.push_back_reserved(candidate);
        }

        // The kept picks are a subsequence of the ascending pool, so one merge-walk tells a kept
        // candidate from a rejected one.
        std::size_t const diverse_count = selected.size();
        for (std::size_t position = 0, matched = 0;
             position != context.results.size() && selected.size() != connectivity; ++position) {
            if (matched != diverse_count && selected[matched].node_id == context.results[position].node_id) ++matched;
            else selected.push_back_reserved(context.results[position]);
        }
    }
};

/** Any link-selection heuristic: carries its own neighbors-per-node budget as
 *  @c connectivity, reads the ascending pool in @c context.results, resolves values
 *  through the index's @c published_value_of - every pooled ID is published or wired - scores
 *  pairs through the @p metric, and fills @c context.selected_neighbors. A selector
 *  may also write @c context.staged_distances - and nothing else: @c staged_ids and
 *  @c reverse_ids belong to the calling traversal. The context's allocator is the caller's,
 *  so a selector takes any @c traversal_context; the concept checks the index's own. */
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
            wait_type_ {}(&word, backed_off, 0u, wait_capped_t {});
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
    snapshot_t snapshot() const noexcept {
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

    /** Running per-walk counters; calls report their share as snapshot deltas. */
    traversal_telemetry_t telemetry;

    /** Every container gets a rebound copy of @p allocator, so scratch lands where
     *  the caller places it; seed distinctly per thread to decorrelate the walks. */
    explicit traversal_context(u64_t seed = 0, allocator_t allocator = {}) noexcept
        : results(scored_allocator_t(allocator)), frontier(scored_allocator_t(allocator)),
          visited(visits_allocator_t(allocator)), random_generator {seed}, staged_ids(allocator),
          staged_distances(allocator), selected_neighbors(allocator), reverse_ids(allocator) {}
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
 */
template <                                                                //
    typename value_type_,                                                 //
    typename key_type_ = uuid_t,                                          //
    typename node_id_type_ = u64_t,                                       //
    typename distance_type_ = f64_t,                                      //
    typename wait_type_ = dummy_waiter_t,                                 //
    template <typename> class atomic_ref_template_ = standard_atomic_ref, //
    typename allocator_type_ = std::allocator<byte_t>                     //
    >
    requires value<value_type_> && (std::same_as<node_id_type_, u32_t> || std::same_as<node_id_type_, u64_t>)
struct ram_index {

    using value_t = value_type_;
    using key_t = key_type_;
    using node_id_t = node_id_type_;
    using distance_t = distance_type_;
    using wait_t = wait_type_;
    using allocator_t = allocator_type_;
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
                    wait_t {}(&state, observed, 0u, wait_capped_t {});
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

    /** The node records in chunks spanning @c max_node_count_k: a cover builds every record of a
     *  fresh chunk from the index allocator, so slots exist before publication with no per-slot
     *  lifecycle tracking, and none ever moves. */
    using nodes_column_t = chunked_column<node_t, chunk_schedule_spanning<max_node_count_k>, max_node_count_k,
                                          allocator_t, atomic_ref_template_>;

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

    /** Constructs an empty index; allocates nothing until the first @c reserve or insert. */
    ram_index(allocator_t allocator = {}) noexcept
        : nodes_(allocator), key_map_(allocator), allocator_(std::move(allocator)) {}

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
        for (std::size_t node = 0; node != static_cast<std::size_t>(visible); ++node) {
            node_t& record = nodes_[node];
            record.value = std::make_obj_using_allocator<value_t>(allocator_);
            record.key = key_t {};
            record.neighbors.clear();
            record.state = 0;
        }
        key_map_.clear();
        atomic_ref<std::size_t>(claimed_count_).store(0, std::memory_order_relaxed);
        atomic_ref<node_id_t>(next_dispensed_id_).store(0, std::memory_order_relaxed);
        atomic_ref<node_id_t>(last_visible_id_).store(0, std::memory_order_release);
        return status_t::success_k;
    }

#pragma endregion Lifecycle

#pragma region Mutation

    /** Optional tuning knobs for @c insert(); zeros mean "derive from the
     *  selector's budget". */
    struct insert_config_t {

        /** Breadth of the candidate pool during wiring. */
        std::size_t expansion_factor = 0;

        /** Frontier entries expanded per metric round. */
        std::size_t fan_out = 0;
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
                    wait_t {}(&last_visible_id, watermark, 0u, wait_uncapped_t {});
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
     *  of zero dispenses nothing. Joining the shared side waits out a unique holder, so a call from
     *  a thread holding a save or load plan deadlocks on its own hold, by construction.
     */
    expected<insert_plan_t> plan_insert(std::size_t count) noexcept {
        // The gate opens before the claim, so a batch parked behind a unique holder never
        // sits on committed capacity, spuriously overflowing its peers.
        shared_lock_t hold {shared_mutex_};
        if (!count) return {insert_plan_t {this, {}, {}, std::move(hold)}, status_t::success_k};

        // Key seats come first, so a refused seat claim leaves no capacity claimed.
        auto [seats, seat_status] = key_map_.claim(count);
        if (failed(seat_status)) return {{}, seat_status};

        // Claims convert one-to-one into dispensed IDs once covered, and commit only within the
        // ID ceiling, so peers never observe an overshoot. Covering to this claim's end suffices:
        // of the batches dispensed up to this one, the last to claim covered past this range, and
        // the dispenser's acquire-release carries its cover along.
        std::size_t const claimed = atomic_fetch_add_if_at_most(atomic_ref<std::size_t>(claimed_count_), count,
                                                                max_node_count_k, std::memory_order_acq_rel);
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
     *  insertion, hoisted ahead of @ref plan_insert since no step may allocate it. The pools, the
     *  staging surfaces and the selections never allocate again, the frontier sheds within its
     *  reservation, and the visited set alone may re-table as a walk widens - once per doubling per
     *  context lifetime, which a context kept across walks amortizes. Idempotent; a zero
     *  @p connectivity answers @c config_mismatch_k. */
    template <typename context_allocator_type_>
    static status_t reserve_traversal(traversal_context_over<context_allocator_type_>& context,
                                      std::size_t connectivity, insert_config_t config) noexcept {
        if (!connectivity) return status_t::config_mismatch_k;
        insert_config_t const walk = resolved_config_(config, connectivity);
        std::size_t const round_capacity = max_of(walk.fan_out * connectivity, connectivity + 1);
        if (status_t const sized = context.results.reserve(max_of(walk.expansion_factor, connectivity + 1));
            failed(sized))
            return sized;
        if (status_t const sized = context.frontier.reserve(walk.expansion_factor * 4); failed(sized)) return sized;
        if (status_t const sized = context.visited.reserve(walk.expansion_factor * 4); failed(sized)) return sized;
        if (status_t const sized = context.staged_ids.reserve(round_capacity); failed(sized)) return sized;
        if (status_t const sized = context.staged_distances.reserve(round_capacity); failed(sized)) return sized;
        if (status_t const sized = context.selected_neighbors.reserve(connectivity + 1); failed(sized)) return sized;
        return context.reverse_ids.reserve(connectivity);
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
        if constexpr (locked_k) node.unlock();
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

#pragma region Search

    /** Optional tuning knobs for @c search(); zeros mean "derive from
     *  @c connectivity_32_k" - the searcher does not know the budgets the graph was
     *  wired with. */
    struct search_config_t {

        /** Breadth of the candidate pool; raise to trade latency for recall. */
        std::size_t expansion_factor = 0;

        /** Frontier entries expanded per round; their unioned rows form one
         *  metric round - the lever that feeds tiled engines their batch width. */
        std::size_t fan_out = 0;
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
     *  Scoring runs off-lock, one @c reset / @c push / @c drain round per fan-out, and the
     *  metric need not match the one that built the graph. Once the pool is full, its furthest
     *  member's distance drains each round as the radius, so a metric may stop scoring a
     *  candidate that cannot come under it. The query is a stored-value reference by default;
     *  a @c transparent_metric accepts other query forms.
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
     *      the walk successfully, before that round reaches the pool.
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
        insert_config_t const walk = resolved_config_({config.expansion_factor, config.fan_out}, connectivity_32_k);
        if (status_t const sized = reserve_traversal(context, connectivity_32_k, walk); failed(sized))
            return {{}, sized};
        std::size_t const expansion_factor = walk.expansion_factor;
        std::size_t const fan_out = walk.fan_out;
        context.results.clear();
        context.frontier.clear();
        context.visited.clear();

        // One watermark snapshot bounds the whole walk; a certified arm proves its own
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
                                         fan_out, connectivity_32_k, context, fold);

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

#pragma endregion Search

#pragma region Keys and Introspection

    /** Number of published nodes - the visibility watermark; safe any time. */
    std::size_t size() const noexcept {
        return static_cast<std::size_t>(atomic_ref<node_id_t const>(last_visible_id_).load(std::memory_order_acquire));
    }

    /** Rows the covered node chunks hold, as inserts, loads and @c reserve grow them; safe any
     *  time. */
    std::size_t capacity() const noexcept { return nodes_.capacity(); }

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
     *  pool, where the watermark gate of @c value_of would re-prove a known fact or refuse a
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

        /** Nodes no flood from the stochastic entry points reached - the
         *  searcher's-eye blind spot. */
        std::size_t unreachable_from_seeds = 0;

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
     *      reachability from stochastic seeds.
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

        // Reachability from as many stochastic landmarks as one search draws - the searcher's view.
        // The coverage guard is load-bearing twice over: it ends the probe once the graph
        // is covered, and it keeps every seed provably unvisited, which bounds the stack.
        context.visited.clear();
        std::size_t const seed_count = min_of<std::size_t>(connectivity_32_k, node_count);
        for (std::size_t seed = 0; seed != seed_count && context.visited.size() != node_count; ++seed) {
            node_id_t draw = uniform_below<node_id_t> {visible}(context.random_generator);
            while (context.visited.contains(draw)) draw = draw + 1 == visible ? 0 : draw + 1;
            flood(draw);
        }
        report.unreachable_from_seeds = node_count - context.visited.size();
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
        load_plan_t(ram_index* index, columns_view_t columns, unique_lock_t&& hold) noexcept
            : columns(columns), index_(index), hold_(std::move(hold)) {}
        load_plan_t(load_plan_t&& other) noexcept
            : columns(other.columns), covered(other.covered), index_(std::exchange(other.index_, nullptr)),
              hold_(std::move(other.hold_)) {}
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

            // Seats first, so recording a landed row's key cannot fail.
            auto [seats, seat_status] = index_->key_map_.claim(node_range.size());
            if (failed(seat_status)) return seat_status;
            // The plan vetted both prefix-sum columns and every ID, so the slices are trusted.
            for (node_id_t const node : node_range) {
                node_t& record = index_->nodes_[node];
                record.key = columns.keys[node];

                std::size_t const ids_first = columns.neighbor_offsets[node];
                std::size_t const ids_bound = columns.neighbor_offsets[node + 1];
                neighbors_buffer_t& row = record.neighbors;
                row.clear();
                if (status_t const grown = row.reserve(ids_bound - ids_first); failed(grown)) return grown;
                for (std::size_t link = ids_first; link != ids_bound; ++link)
                    row.push_back_reserved(columns.neighbor_ids[link]);

                std::size_t const bytes_first = columns.value_offsets[node];
                std::size_t const bytes_bound = columns.value_offsets[node + 1];
                if (status_t const decoded = codec.load(columns.values.subspan(bytes_first, bytes_bound - bytes_first),
                                                        record.value);
                    failed(decoded))
                    return decoded;

                index_->key_map_.insert(seats, record.key, node);
            }
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

        /** The unique hold spanning the plan's whole lifetime. */
        unique_lock_t hold_;
    };

    /**
     *  @brief Opens a step-parallel load into this empty index: takes the shared mutex's unique
     *      side - held inside the returned plan until its @c finish or its death - vets both
     *      prefix-sum columns and every neighbor ID in full, and wipes the key multimap.
     *
     *  The vetted rows' records are then covered, and the plan's own @c load calls decode the
     *  per-node bulk; nothing publishes until its @c finish.
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
        if (status_t const covered = nodes_.cover(columns.keys.size()); failed(covered)) return {{}, covered};
        // A dying plan wipes the keys it registered, so the map arrives whole here.
        return {load_plan_t {this, columns, std::move(hold)}, status_t::success_k};
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

    /** @p config with its zeros derived from @p connectivity: a pool twice the budget, and enough
     *  expanded rows per round for one metric round to stage a full pool. */
    static insert_config_t resolved_config_(insert_config_t config, std::size_t connectivity) noexcept {
        if (!config.expansion_factor) config.expansion_factor = connectivity * 2;
        if (!config.fan_out)
            config.fan_out = max_of<std::size_t>(divide_round_up(config.expansion_factor, connectivity), 1);
        return config;
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
                walk.fan_out, connectivity, context, dummy_fold_t {});
        }
        selector(*this, metric, context);

        node_t& record = nodes_[new_node_id];
        record.value = std::move(incoming);
        record.key = key;
        context.reverse_ids.clear();
        std::size_t const selected_count = min_of(context.selected_neighbors.size(), connectivity);
        // A failed load leaves its unpublished rows filled; this one starts from nothing.
        record.neighbors.clear();
        // A refused row growth wires a thinner node - the dispensed ID must still publish.
        std::size_t const wired_count = succeeded(record.neighbors.reserve(selected_count)) ? selected_count : 0;
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

        /** Landmarks come from the IDs below this bound, the snapshot. */
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
     *  Draws stay below the bound @p admission sets. Past the snapshot a row may stay unwired for the
     *  whole walk, so the re-rolls of unadmitted draws stop once they outnumber the IDs there.
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
        node_id_t const sampling_bound = admission.sampling_bound();
        node_id_t unwired_draws = 0;
        while (context.staged_ids.size() != wanted &&
               context.visited.size() < static_cast<std::size_t>(sampling_bound)) {
            node_id_t draw = uniform_below<node_id_t> {sampling_bound}(context.random_generator);
            while (context.visited.contains(draw)) draw = draw + 1 == sampling_bound ? 0 : draw + 1;
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
        if (fold.absorb(std::span<node_id_t const>(context.staged_ids),
                        std::span<distance_t const>(context.staged_distances)) == walk_demand_t::sated_k)
            return {walk_step_t::sated_k, status_t::success_k};
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

    /** One round's expansion half: peels up to @p fan_out frontier entries within
     *  the stopping radius, copies their rows, and pushes the admission- and
     *  visited-filtered union into the metric, compacted in @c context.staged_ids. */
    template <synchronization_t synchronization_, typename metric_type_, typename query_type_, typename admission_type_,
              typename context_type_>
    walk_step_t expand_frontier_(query_type_ const& query, metric_type_& metric, admission_type_ const& admission,
                                 std::size_t expansion_factor, std::size_t fan_out, context_type_& context) noexcept {
        context.staged_ids.clear();
        metric.reset(query);
        for (std::size_t expanded_count = 0; expanded_count != fan_out && !context.frontier.empty(); ++expanded_count) {
            scored_node_t const best = context.frontier.top();
            bool const results_full = context.results.size() == expansion_factor;
            if (results_full && best.distance > context.results.top().distance) {
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
            ++context.telemetry.visited_members;

            std::size_t kept = segment_start;
            for (std::size_t position = segment_start; position != segment_start + degree; ++position) {
                node_id_t const neighbor_id = context.staged_ids[position];
                // The translation frontier: an ID the admission refuses has no
                // happens-before edge to its record, so a later query picks it up.
                if (!admission.admits(neighbor_id)) continue;
                if (context.visited.contains(neighbor_id)) continue;
                if (metric.push(published_value_of(neighbor_id)) == metric_staging_t::refused_k) {
                    context.staged_ids.resize_reserved(kept);
                    return walk_step_t::saturated_k;
                }
                [[maybe_unused]] membership_t const marked = context.visited.insert(neighbor_id);
                context.staged_ids[kept] = neighbor_id;
                ++kept;
            }
            context.staged_ids.resize_reserved(kept);
        }
        return walk_step_t::advanced_k;
    }

    /**
     *  @brief The shared walk of @c search & @c insert: seed, then per round expand the
     *      nearest frontier entries and score the staged union in one builder round - every
     *      scored round, seeds included, reaching the fold before the pool.
     *
     *  The walk locates only the IDs @p admission admits: a search's snapshot, or an insertion's
     *  snapshot widened by the rows wired ahead of their turns. It never fails hard: an exhausted
     *  admitted set, a refused staging or a sated fold ends it normally, and a refused re-table ends
     *  it early with the ranking it has and counts a saturated walk - an insertion wires the thinner
     *  row regardless, and a search reports the count so the caller can widen the scratch. Frontier
     *  starvation with an unfilled pool re-seeds and counts a restart. The frontier-admission rule: a
     *  candidate enters the frontier iff it improved the results pool, or the frontier would
     *  otherwise be empty - the seedings.
     */
    template <synchronization_t synchronization_, typename metric_type_, typename query_type_, typename admission_type_,
              typename context_type_, typename fold_type_>
    void search_around_(query_type_ const& query, metric_type_& metric, admission_type_ const& admission,
                        std::size_t expansion_factor, std::size_t fan_out, std::size_t connectivity,
                        context_type_& context, fold_type_&& fold) noexcept {
        while (true) {
            // An empty frontier seeds: the first round, or a restart after it drained before the
            // pool filled. Nothing left to seed from ends the walk.
            if (context.frontier.empty()) {
                if (context.results.size() >= expansion_factor) return;
                bool const restarting = !context.results.empty();
                expected<walk_step_t> const seeded = sample_landmarks_(query, metric, admission, connectivity,
                                                                       expansion_factor, context, fold);
                if (failed(seeded.status) || seeded.value != walk_step_t::advanced_k) return;
                context.telemetry.restarts += restarting;
                continue;
            }
            walk_step_t const expanded = expand_frontier_<synchronization_>(query, metric, admission, expansion_factor,
                                                                            fan_out, context);
            if (expanded == walk_step_t::exhausted_k) return;
            walk_step_t const ranked = rank_round_(metric, expansion_factor, context, fold);
            if (expanded == walk_step_t::saturated_k || ranked != walk_step_t::advanced_k) return;
        }
    }

    /** One round's scoring half: drains the staged union through the metric, hands it to the
     *  fold, and pushes what improved the pool into the frontier, shedding the frontier past a
     *  full pool's radius first so the pushes fit. */
    template <typename metric_type_, typename context_type_, typename fold_type_>
    walk_step_t rank_round_(metric_type_& metric, std::size_t expansion_factor, context_type_& context,
                            fold_type_&& fold) const noexcept {
        std::size_t const staged_count = context.staged_ids.size();
        if (!staged_count) return walk_step_t::advanced_k;
        // A full pool only admits what beats its furthest member, so that member's distance is the radius.
        bool const results_full = context.results.size() == expansion_factor;
        drain_round(metric, staged_count, context,
                    results_full ? context.results.top().distance : unbounded_radius<distance_t>());
        if (fold.absorb(std::span<node_id_t const>(context.staged_ids),
                        std::span<distance_t const>(context.staged_distances)) == walk_demand_t::sated_k)
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

    /** Wires the reverse edges of the freshly wired node before its batch's turn, walking
     *  the links snapshotted in @c context.reverse_ids: per neighbor - lock, append when room
     *  remains at zero metric cost; otherwise copy the full row, unlock, re-select
     *  off-lock through the selector, relock, and apply only if the row is still the
     *  one copied - a lost race falls back to append-if-room, never a re-score. */
    template <synchronization_t synchronization_, typename metric_type_, typename selector_type_,
              typename context_type_>
    void back_link_(node_id_t new_node_id, metric_type_& metric, selector_type_ const& selector,
                    context_type_& context) noexcept {
        constexpr bool locked_k = synchronization_ == synchronization_t::concurrent_k;
        std::size_t const connectivity = selector.connectivity;
        value_t const& new_value = published_value_of(new_node_id);
        // `link` names only published IDs and the newcomer publishes after this pass, so no row holds it yet.
        for (std::size_t reverse = 0; reverse != context.reverse_ids.size(); ++reverse) {
            node_id_t const neighbor_id = context.reverse_ids[reverse];
            node_t& node = nodes_[neighbor_id];

            if constexpr (locked_k) node.lock();
            if (node.neighbors.size() < connectivity) {
                // A refused growth loses one reverse link - tolerated, like refused staging.
                if (succeeded(node.neighbors.reserve(node.neighbors.size() + 1)))
                    node.neighbors.push_back_reserved(new_node_id);
                if constexpr (locked_k) node.unlock();
                continue;
            }
            std::size_t const degree = node.neighbors.size();
            context.staged_ids.clear();
            if (failed(context.staged_ids.reserve(degree))) {
                if constexpr (locked_k) node.unlock();
                continue; // A refused staging loses one reverse link.
            }
            for (std::size_t position = 0; position != degree; ++position)
                context.staged_ids.push_back_reserved(node.neighbors[position]);
            if constexpr (locked_k) node.unlock();
            if (failed(context.staged_distances.reserve(degree + 1)) || failed(context.results.reserve(degree + 1)))
                continue; // A refused staging loses one reverse link.

            // Off-lock: pool = the copied row plus the newcomer, scored against the
            // neighbor's own stored value in one builder round; every member is ranked, so nothing is too far.
            metric.reset(published_value_of(neighbor_id));
            std::size_t staged_count = 0;
            for (; staged_count != degree; ++staged_count)
                if (metric.push(published_value_of(context.staged_ids[staged_count])) == metric_staging_t::refused_k)
                    break;
            bool const complete = staged_count == degree && metric.push(new_value) == metric_staging_t::staged_k;
            staged_count += complete;
            drain_round(metric, staged_count, context, unbounded_radius<distance_t>());
            if (!complete) continue; // Staging refused - keep the old row, lose one reverse link.

            context.results.clear();
            for (std::size_t position = 0; position != degree; ++position)
                context.results.insert(scored_node_t {context.staged_ids[position], //
                                                      context.staged_distances[position]},
                                       degree + 1);
            context.results.insert(scored_node_t {new_node_id, context.staged_distances[degree]}, degree + 1);
            selector(*this, metric, context);

            // Relock & revalidate: apply the re-selection only onto the exact row it was
            // computed from; a raced row takes the cheap monotone fallback instead.
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
            else if (node.neighbors.size() < connectivity &&
                     succeeded(node.neighbors.reserve(node.neighbors.size() + 1))) {
                node.neighbors.push_back_reserved(new_node_id);
            }
            if constexpr (locked_k) node.unlock();
        }
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

    /** Key → ID multimap, covered by each batch's seats and by @c reserve. */
    key_map_t key_map_;

    /** The world gate: mutators hold it shared, stop-the-world verbs try-unique, certified
     *  readers probe it wait-free. On its own line, away from the watermark, which turn loops
     *  spin on. */
    alignas(default_alignment_k) shared_mutex_t shared_mutex_;

    [[no_unique_address]] allocator_t allocator_;

#pragma endregion Internals and State
};

/**
 *  @brief External reader's cursor over the watermark of one @c basic::ram_index:
 *      ID-order iteration, random seeks, and edge-following through a privately staged
 *      copy of the current node's row.
 *
 *  Strictly read-only under every contract, so readers never stall writers. One watermark snapshot,
 *  captured at construction and widened by @c refresh, bounds everything: keys & values below it
 *  are immutable and read wait-free, rows are staged under the row lock on the concurrent arm and
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
