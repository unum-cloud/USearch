/**
 *  @brief Similarity Search baseline over arbitrary payloads & caller-supplied metrics.
 *  @author Ash Vardanian
 *  @file include/usearch/basic/ram_index.hpp
 *  @date August 15, 2026
 *
 *  The baseline tier's RAM residency: the readable reference implementation. Arbitrary
 *  payload objects, expensive caller-supplied batched metrics, caller-visible keys,
 *  growable per-node neighbor rows. Standard-library & OS headers only - together with
 *  @c utilities/ this tier ships as a standalone header-only library. Every specialized
 *  optimization - packed state words, growth schedules, lock-free readers - belongs to
 *  the dense family, not here.
 */
#pragma once

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint64_t`

#include <atomic>       // `std::atomic_ref`
#include <limits>       // `std::numeric_limits` - the id-space ceiling in `reserve`
#include <memory>       // `std::allocator`, `std::allocator_traits`
#include <mutex>        // `std::unique_lock` - the try-only unique side's guard
#include <shared_mutex> // `std::shared_lock` - the mutators' guard
#include <span>         // `std::span`
#include <type_traits>  // `std::is_nothrow_move_assignable_v`
#include <utility>      // `std::move`, `std::forward`

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/sanitizers.hpp"
#include "usearch/utilities/ram_buffers.hpp"
#include "usearch/utilities/ram_tables.hpp"

namespace unum {
namespace usearch {

/** @brief One traversal candidate; both halves are the caller's chosen widths. */
template <typename node_id_type_ = u64_t, typename distance_type_ = f64_t>
struct basic_scored_node {
    node_id_type_ node_id;
    distance_type_ distance;

    /** @brief Orders scored nodes nearest-first; the comparator of the sorted results pool. */
    struct closer_t {
        bool operator()(basic_scored_node first, basic_scored_node second) const noexcept {
            return first.distance < second.distance;
        }
    };
    /** @brief Orders scored nodes furthest-first; the comparator of the frontier heap. */
    struct further_t {
        bool operator()(basic_scored_node first, basic_scored_node second) const noexcept {
            return first.distance > second.distance;
        }
    };
};

/** @brief Stored payloads live in records default-constructed by @c reserve, receive their
 *      content through one move-assignment at publication, are read as @c const& after, and
 *      are destroyed only at teardown - that is the whole lifecycle. */
template <typename value_type_>
concept basic_value = std::is_nothrow_default_constructible_v<value_type_> && //
                      std::is_nothrow_move_assignable_v<value_type_> &&       //
                      std::is_nothrow_destructible_v<value_type_>;

/** @brief A batched metric's answer to one @c push: whether the candidate entered the
 *      staging of the current scoring round - a capacity answer, never a failure. */
enum class basic_metric_staging_t : bool {
    /** @brief The staging is full; the round should drain before anything else stages. */
    refused_k = false,
    /** @brief The candidate is staged, and the round's @c drain will score it. */
    staged_k = true,
};

/**
 *  @brief Any batched similarity measure, driven as a builder: @c reset opens a scoring
 *      round for one query, @c push stages one candidate - the metric owns its staging,
 *      packing or uploading eagerly if it wishes, answering whether the candidate staged -
 *      and @c drain scores everything staged, writing distances in push order and clearing
 *      the staging.
 *
 *  Every verb runs with no locks held, so a @c drain may cost milliseconds without
 *  stalling any other thread; pushed references stay valid until the round's @c drain
 *  returns, never longer. Every requirement is a call-validity check, so a metric is free
 *  to take converting parameters - @c std::string_view for @c std::string values.
 */
template <typename metric_type_, typename value_type_, typename distance_type_>
concept basic_metric = basic_value<value_type_> &&
                       requires(metric_type_& metric, value_type_ const& value, std::span<distance_type_> distances) {
                           { metric.reset(value) } noexcept;
                           { metric.push(value) } noexcept -> std::same_as<basic_metric_staging_t>;
                           { metric.drain(distances) } noexcept;
                       };

/** @brief Marks a metric whose @c reset also accepts query forms other than the stored
 *      value - the standard @c is_transparent convention. */
template <typename metric_type_>
concept transparent_metric = requires { typename metric_type_::is_transparent; };

/** @brief A query form @p metric_type_ opens rounds for: the stored value itself, or -
 *      on a @c transparent_metric - any other type its @c reset accepts. */
template <typename query_type_, typename metric_type_, typename value_type_>
concept basic_query = (std::same_as<query_type_, value_type_> || transparent_metric<metric_type_>) &&
                      requires(metric_type_& metric, query_type_ const& query) {
                          { metric.reset(query) } noexcept;
                      };

/** @brief Serializes stored values into caller-owned buffers and back - the payload half
 *      of @c save & @c load, supplied by the caller because values are arbitrary objects. */
template <typename codec_type_, typename value_type_>
concept basic_value_codec = requires(codec_type_ const& codec, value_type_ const& immutable_value,
                                     value_type_& mutable_value, bytes_span_t output, bytes_view_t input) {
    { codec.storage_requirements(immutable_value) } noexcept -> std::same_as<std::size_t>;
    { codec.save(immutable_value, output) } noexcept -> std::same_as<status_t>;
    { codec.load(input, mutable_value) } noexcept -> std::same_as<status_t>;
};

/** @brief The four per-walk counters every traversal accumulates; results report one
 *      call's share as the difference of two snapshots. */
struct basic_traversal_telemetry_t {
    /** @brief Metric invocations, batched rounds counted once per scored candidate. */
    std::size_t computed_distances = 0;
    /** @brief Frontier candidates whose neighbor lists were expanded. */
    std::size_t visited_members = 0;
    /** @brief Re-seedings after the frontier drained prematurely - a connectivity signal. */
    std::size_t restarts = 0;
    /** @brief Walks a scratch refusal ended early - ranked short, and on insertion wired thinner. */
    std::size_t saturated_walks = 0;

    basic_traversal_telemetry_t operator-(basic_traversal_telemetry_t const& before) const noexcept {
        return {computed_distances - before.computed_distances, visited_members - before.visited_members,
                restarts - before.restarts, saturated_walks - before.saturated_walks};
    }
};

/** @brief How one frontier-expansion round of a batched traversal ended - the interface
 *      between the expansion that stages one metric round and the caller draining it. */
enum class basic_expansion_round_t : u32_t {
    /** @brief The nearest frontier entry exceeds a full pool's radius - the walk is done. */
    converged_k,
    /** @brief Scratch or metric staging refused mid-round - drain what staged, then stop. */
    saturated_k,
    /** @brief A normal round: whatever staged is ready to drain. */
    advanced_k,
};

/** @brief One round's scoring epilogue: sizes the distances to the staging fill, drains
 *      the metric, and counts the round into the telemetry. */
template <typename metric_type_, typename context_type_>
void drain_round(metric_type_& metric, std::size_t staged_count, context_type_& context) noexcept {
    context.staged_distances.resize_reserved(staged_count);
    metric.drain({context.staged_distances.data(), staged_count});
    context.telemetry.computed_distances += staged_count;
}

/** @brief Plain NSW selection: the nearest @c connectivity candidates, no pruning. */
struct basic_keep_closest_t {
    /** @brief The neighbors-per-node budget this selector prunes to. */
    std::size_t connectivity = connectivity_32_k;

    template <typename index_type_, typename metric_type_>
    void operator()(index_type_ const&, metric_type_&,
                    typename index_type_::traversal_context_t& context) const noexcept {
        context.selected_neighbors.clear();
        std::size_t const kept_count = min_of(connectivity, context.results.size());
        for (std::size_t position = 0; position != kept_count; ++position)
            context.selected_neighbors.push_back_reserved(context.results[position]);
    }
};

/** @brief Relative-neighborhood pruning: a candidate survives only if closer to the
 *      query than to every already-kept neighbor - pair distances drawn through the same
 *      metric in shrinking rounds, exiting each candidate on its first violation. */
struct basic_keep_diverse_t {
    /** @brief The neighbors-per-node budget this selector prunes to. */
    std::size_t connectivity = connectivity_32_k;

    template <typename index_type_, typename metric_type_>
    void operator()(index_type_ const& index, metric_type_& metric,
                    typename index_type_::traversal_context_t& context) const noexcept {
        auto& selected = context.selected_neighbors;
        selected.clear();
        for (std::size_t position = 0; position != context.results.size() && selected.size() != connectivity;
             ++position) {
            auto const candidate = context.results[position];
            std::size_t const kept_count = selected.size();
            bool diverse = true;
            if (kept_count) {
                // One round per candidate: its own value queries the kept neighbors.
                metric.reset(index.published_value_of(candidate.node_id));
                std::size_t staged_count = 0;
                for (; staged_count != kept_count; ++staged_count)
                    if (metric.push(index.published_value_of(selected[staged_count].node_id)) ==
                        basic_metric_staging_t::refused_k)
                        break;
                drain_round(metric, staged_count, context);
                if (staged_count != kept_count) return; // Staging refused - keep what survived so far.
                for (std::size_t pair = 0; pair != staged_count && diverse; ++pair)
                    diverse = !(context.staged_distances[pair] < candidate.distance);
            }
            if (diverse) selected.push_back_reserved(candidate);
        }
    }
};

/** @brief Any link-selection heuristic: carries its own neighbors-per-node budget as
 *      @c connectivity, reads the ascending pool in @c context.results, resolves values
 *      through the index's @c published_value_of - every pooled id is published - scores
 *      pairs through the @p metric, and fills @c context.selected_neighbors. A selector
 *      may also write @c context.staged_distances - and nothing else: @c staged_ids and
 *      @c reverse_ids belong to the calling traversal. */
template <typename selector_type_, typename index_type_, typename metric_type_>
concept basic_selector = requires(selector_type_ const& selector, index_type_ const& index, metric_type_& metric,
                                  typename index_type_::traversal_context_t& context) {
    { selector(index, metric, context) } noexcept;
    { selector.connectivity } -> std::convertible_to<std::size_t>;
};

/** @brief Minimal test-and-test-and-set lock over one plain @c bool, owned through the
 *      atomic reference. The waiter policy monitors the flag's address, so hardware waits
 *      plug in without this header naming them; the claiming exchange is the loop's only
 *      store, so a monitored waiter wakes once per @c unlock rather than on every attempt. */
template <typename wait_type_ = dummy_waiter_t, template <typename> class atomic_ref_template_ = std::atomic_ref>
struct basic_mutex {
    template <typename word_type_>
    using atomic_ref = atomic_ref_template_<word_type_>;

    bool flag {false};

    void lock() noexcept {
        annotate_lock_started(&flag);
        // Claim only after observing the flag free: on Arm a `SWPAL` is a write even when it
        // just reads `true` back, stealing the line from the holder; the peek keeps contenders
        // in Shared state, and the exchange stays the loop's only store.
        while (true) {
            while (atomic_ref<bool>(flag).load(std::memory_order_relaxed))
                wait_type_ {}(&flag, true, 0u, wait_capped_t {});
            if (!atomic_ref<bool>(flag).exchange(true, std::memory_order_acquire)) break;
        }
        annotate_lock_acquired(&flag);
    }
    /** @brief Takes the lock only if it is free right now - a contention question, never a
     *      failure. Named for the standard @c Lockable requirement, which @c lock, @c unlock
     *      and this together satisfy, so the STL's lock adapters accept this type. */
    [[nodiscard]] bool try_lock() noexcept {
        annotate_try_lock_started(&flag);
        bool const acquired = !atomic_ref<bool>(flag).exchange(true, std::memory_order_acquire);
        if (acquired) annotate_try_lock_acquired(&flag);
        else annotate_try_lock_refused(&flag);
        return acquired;
    }
    void unlock() noexcept {
        annotate_unlock_started(&flag);
        atomic_ref<bool>(flag).store(false, std::memory_order_release);
        annotate_unlock_finished(&flag);
    }
};

/**
 *  @brief Shared among mutators, try-unique for stop-the-world, wait-free to probe - one
 *      @c std::atomic word, so every reading is a consistent joint snapshot.
 *
 *  The word packs a dedicated unique bit, twenty-two holder bits - the OS pid ceiling
 *  bounds simultaneous holders well below that - and forty-one generation bits, bumped on
 *  every successful shared @b and unique acquisition: a whole acquisition that opens and
 *  closes between two probes still moves the generation, and wrapping it inside one
 *  probe bracket would need days of saturated acquisitions. Every update is a masked
 *  compare-exchange, never a blind add, so the fields cannot bleed into each other.
 */
template <typename wait_type_ = dummy_waiter_t, template <typename> class atomic_ref_template_ = std::atomic_ref>
struct basic_shared_mutex {
    template <typename word_type_>
    using atomic_ref = atomic_ref_template_<word_type_>;

    mutable u64_t word {0};

    inline static constexpr u64_t holders_mask_k = (u64_t {1} << 22) - 1;
    inline static constexpr u64_t generation_unit_k = u64_t {1} << 22;
    inline static constexpr u64_t generation_mask_k = ((u64_t {1} << 41) - 1) << 22;
    inline static constexpr u64_t unique_bit_k = u64_t {1} << 63;

    /** @brief One consistent reading: how the gate is held & how often it has been. */
    struct snapshot_t {
        u64_t generation = 0;
        std::size_t holders = 0;
        bool unique = false;

        bool unlocked() const noexcept { return !holders && !unique; }
    };

    /** @brief Joins the shared side, waiting out a unique holder through the waiter. */
    void lock_shared() noexcept {
        annotate_lock_shared_started(&word);
        u64_t observed = atomic_ref<u64_t>(word).load(std::memory_order_relaxed);
        while (true) {
            if (observed & unique_bit_k) {
                wait_type_ {}(&word, observed, 0u, wait_capped_t {});
                observed = atomic_ref<u64_t>(word).load(std::memory_order_relaxed);
                continue;
            }
            if (atomic_ref<u64_t>(word).compare_exchange_weak(observed, advanced_(observed) + 1,
                                                              std::memory_order_acquire, std::memory_order_relaxed)) {
                annotate_lock_shared_acquired(&word);
                return;
            }
        }
    }

    void unlock_shared() noexcept {
        // The raw release precedes the bracket: inside it TSan ignores the word's own
        // atomics, and the certified probes' acquire would otherwise pair with nothing.
        annotate_release(&word);
        annotate_unlock_shared_started(&word);
        atomic_ref<u64_t>(word).fetch_sub(1, std::memory_order_release);
        annotate_unlock_shared_finished(&word);
    }

    /** @brief Takes the unique side only from a genuinely free word - generation churn
     *      retries, it is not contention - and never waits: a refusal is the answer. */
    [[nodiscard]] bool try_lock() noexcept {
        annotate_try_lock_started(&word);
        u64_t observed = atomic_ref<u64_t>(word).load(std::memory_order_relaxed);
        while ((observed & (holders_mask_k | unique_bit_k)) == 0)
            if (atomic_ref<u64_t>(word).compare_exchange_weak(observed, advanced_(observed) | unique_bit_k,
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
        atomic_ref<u64_t>(word).fetch_and(~unique_bit_k, std::memory_order_release);
        annotate_unlock_finished(&word);
    }

    /** @brief The wait-free probe the certified readers bracket their walks with. */
    snapshot_t snapshot() const noexcept {
        u64_t const observed = atomic_ref<u64_t>(word).load(std::memory_order_acquire);
        annotate_acquire(&word);
        return {observed & generation_mask_k, static_cast<std::size_t>(observed & holders_mask_k),
                (observed & unique_bit_k) != 0};
    }

  private:
    /** @brief Advances the generation within its own field - wrap stays contained. */
    static constexpr u64_t advanced_(u64_t observed) noexcept {
        return (observed & ~generation_mask_k) | ((observed + generation_unit_k) & generation_mask_k);
    }
};

/**
 *  @brief Per-thread traversal scratch: the candidate queues, the exact visited set,
 *      the staging surfaces of batched metric rounds, and running telemetry.
 */
template <typename node_id_type_ = u64_t, typename distance_type_ = f64_t,
          typename allocator_type_ = std::allocator<byte_t>>
struct basic_traversal_context {

    using scored_node_t = basic_scored_node<node_id_type_, distance_type_>;
    using allocator_t = allocator_type_;
    using scored_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<scored_node_t>;
    using visits_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<std::byte>;

    using results_array_t = sorted_buffer<scored_node_t, typename scored_node_t::closer_t, scored_allocator_t>;
    using frontier_queue_t = max_heap_dary<8, scored_node_t, typename scored_node_t::further_t, scored_allocator_t>;
    using visited_set_t = growing_hash_set<node_id_type_, std::hash<node_id_type_>, visits_allocator_t>;
    /** @brief The buffer rebinds to its element type itself, so no alias spells it here. */
    template <typename element_type_>
    using scratch_buffer = growing_buffer<element_type_, allocator_t>;

    /** @brief Nodes already scored, kept ascending as the best results so far. */
    results_array_t results;
    /** @brief Nodes considered promising, to be expanded next. */
    frontier_queue_t frontier;
    /** @brief Nodes already checked - exact, never lossy. */
    visited_set_t visited;
    /** @brief Pseudo-random generator for picking traversal starting points. */
    splitmix64_t random_generator;

    /** @brief Ids of one staged fan-out - the batch a single metric round scores; also
     *      the flood stack of `health` and the row copies of back-linking. */
    scratch_buffer<node_id_type_> staged_ids;
    /** @brief Distances output of one batched metric round, parallel to @c staged_ids. */
    scratch_buffer<distance_type_> staged_distances;
    /** @brief The selector's output: surviving neighbors, ascending by query distance. */
    scratch_buffer<scored_node_t> selected_neighbors;
    /** @brief The freshly wired node's neighbors, snapshotted for the reverse pass - the
     *      re-selections there reuse @c selected_neighbors, so the pass walks its own copy. */
    scratch_buffer<node_id_type_> reverse_ids;

    /** @brief Running per-walk counters; calls report their share as snapshot deltas. */
    basic_traversal_telemetry_t telemetry;

    /** @brief Every container gets a rebound copy of @p allocator, so scratch lands where
     *      the caller places it; seed distinctly per thread to decorrelate the walks. */
    explicit basic_traversal_context(u64_t seed = 0, allocator_t allocator = {}) noexcept
        : results(scored_allocator_t(allocator)), frontier(scored_allocator_t(allocator)),
          visited(visits_allocator_t(allocator)), random_generator {seed}, staged_ids(allocator),
          staged_distances(allocator), selected_neighbors(allocator), reverse_ids(allocator) {}
};

/** @brief Row & element counts of the five columns a snapshot needs - what a container sizes. */
struct basic_columns_extents_t {
    /** @brief Published nodes, the row count of every node-sized column. */
    std::size_t nodes = 0;
    /** @brief Neighbor ids across every row. */
    std::size_t neighbors = 0;
    /** @brief Codec bytes across every value. */
    std::size_t value_bytes = 0;
};

/**
 *  @brief The five caller-placed columns a basic-tier snapshot is: keys, neighbor prefix sums,
 *      neighbor ids, value prefix sums, and codec bytes - each an independent buffer, so a
 *      container lays them out wherever it likes. Both prefix-sum columns hold @b nodes+1 sums
 *      from zero, so row @c i owns the slice between sums @c i and @c i+1.
 */
template <typename key_type_, typename node_id_type_, typename byte_type_, typename offset_type_ = u64_t>
struct basic_columns {
    /** @brief One key per row. */
    std::span<key_type_> keys;
    /** @brief @b nodes+1 sums of neighbor ids, from zero. */
    std::span<offset_type_> neighbor_offsets;
    /** @brief Every row's neighbor ids, back to back. */
    std::span<node_id_type_> neighbor_ids;
    /** @brief @b nodes+1 sums of codec bytes, from zero. */
    std::span<offset_type_> value_offsets;
    /** @brief Every row's codec bytes, back to back. */
    std::span<byte_type_> values;

    /** @brief The read-only view of the same columns, so a saved set loads as it is. */
    basic_columns<key_type_ const, node_id_type_ const, byte_type_ const, offset_type_ const> view() const noexcept {
        return {keys, neighbor_offsets, neighbor_ids, value_offsets, values};
    }
};

template <typename index_type_, synchronization_t synchronization_ = synchronization_t::concurrent_k,
          typename allocator_type_ = typename index_type_::allocator_t>
struct basic_ram_cursor;

/**
 *  @brief Similarity Search index baseline for arbitrary values & similarity measures.
 *
 *  The design optimizes for being read, and its storage is one array of node records
 *  sized once by @c reserve, plus the key multimap. Each record owns its neighbors row,
 *  which starts empty and grows with wiring - automatic through @c insert's selector,
 *  which carries the neighbors-per-node budget, or manual through @c link. Metric
 *  calls cost microseconds to milliseconds here, so brief per-node locks are noise -
 *  the only hard rule is that no metric verb ever runs while any row lock is held - the
 *  shared mutex is deliberately held across whole batches, so a parked metric starves
 *  stop-the-world verbs, never peers.
 *
 *  Concurrency runs on plain primitives: batch capacity is claimed by @c fetch_add,
 *  identity is dispensed per node right before its watermark turn, nodes become visible
 *  through a watermark advanced in dispensed order, and a node's key lands in the
 *  multimap only after its watermark turn - so a findable key always names a published
 *  node. Keys need not be unique: every holder of a key stays addressable through
 *  @c find_each, and id → key stays authoritative. Node ids are never recycled and
 *  removals are deleted, keeping every reader's stale id valid forever.
 *
 *  @tparam value_type_ any payload satisfying @c basic_value; immovable once published.
 *  @tparam key_type_ trivially-copyable caller-visible identity.
 *  @tparam node_id_type_ graph-local identity, dense from zero; @c u32_t or @c u64_t.
 *  @tparam distance_type_ the metrics' result element.
 *  @tparam wait_type_ the monitored-wait engaged on contended words.
 *  @tparam atomic_ref_template_ the atomic reference owning the plain synchronization words;
 *      @c std::atomic_ref by default, ForkUnion's ISA-specific references plug in.
 *  @tparam allocator_type_ places the node records, neighbor buffers & key multimap.
 */
template <                                                            //
    typename value_type_,                                             //
    typename key_type_ = uuid_t,                                      //
    typename node_id_type_ = u64_t,                                   //
    typename distance_type_ = f64_t,                                  //
    typename wait_type_ = dummy_waiter_t,                             //
    template <typename> class atomic_ref_template_ = std::atomic_ref, //
    typename allocator_type_ = std::allocator<byte_t>                 //
    >
    requires basic_value<value_type_> && (std::same_as<node_id_type_, u32_t> || std::same_as<node_id_type_, u64_t>)
struct basic_ram_index {

    using value_t = value_type_;
    using key_t = key_type_;
    using node_id_t = node_id_type_;
    using distance_t = distance_type_;
    using wait_t = wait_type_;
    using allocator_t = allocator_type_;
    template <typename word_type_>
    using atomic_ref = atomic_ref_template_<word_type_>;

    using scored_node_t = basic_scored_node<node_id_t, distance_t>;
    using traversal_context_t = basic_traversal_context<node_id_t, distance_t, allocator_t>;
    /** @brief The five caller-placed columns a save fills. */
    using columns_t = basic_columns<key_t, node_id_t, byte_t, u64_t>;
    /** @brief The read-only columns a load consumes. */
    using columns_view_t = basic_columns<key_t const, node_id_t const, byte_t const, u64_t const>;

  private:
    template <typename cursor_index_type_, synchronization_t cursor_synchronization_, typename cursor_allocator_type_>
    friend struct basic_ram_cursor;

    using neighbors_buffer_t = growing_buffer<node_id_t, allocator_t>;

    /** @brief One node record: the payload, its key, the neighbors it links to, and the
     *      mutex guarding those neighbors. The value & key are written before publication
     *      and never change; only @c neighbors mutates after - wired by inserts, grown or
     *      trimmed by manual linking - with its own @c size() as the live degree. */
    struct node_t {
        /** @brief Mutable so id-addressed const readers, like @c degree_of, may briefly
         *      hold it - locking never mutates the record's logical state. */
        mutable basic_mutex<wait_t, atomic_ref_template_> mutex;
        value_t value {};
        key_t key {};
        neighbors_buffer_t neighbors;

        // No default form: it would seed the row with the default allocator, silently
        // escaping whatever allocator the index was constructed with.
        node_t() = delete;
        explicit node_t(allocator_t const& allocator) noexcept : neighbors(allocator) {}
        node_t(node_t&& other) noexcept : key(other.key), neighbors(std::move(other.neighbors)) {
            value = std::move(other.value);
        }
    };

    /** @brief All node records in one array: @c reserve sizes it once and constructs every
     *      record up front - each seeded with the index allocator - so slots exist before
     *      publication with no per-slot lifecycle tracking. */
    using node_array_t = growing_buffer<node_t, allocator_t>;

    using key_map_t = growing_multimap<key_t, node_id_t, std::hash<key_t>, allocator_t>;

    using shared_mutex_t = basic_shared_mutex<wait_t, atomic_ref_template_>;
    using shared_lock_t = std::shared_lock<shared_mutex_t>;
    using unique_lock_t = std::unique_lock<shared_mutex_t>;

  public:
#pragma region Lifecycle

    basic_ram_index(basic_ram_index const&) noexcept = delete;
    basic_ram_index& operator=(basic_ram_index const&) noexcept = delete;
    basic_ram_index(basic_ram_index&&) noexcept = delete;
    basic_ram_index& operator=(basic_ram_index&&) noexcept = delete;

    /** @brief Constructs an empty index; allocates nothing - storage arrives in @c reserve. */
    basic_ram_index(allocator_t allocator = {}) noexcept
        : nodes_array_(allocator), key_map_(allocator), allocator_(std::move(allocator)) {}

    /** @brief Exclusive: the record array destroys every value in place, the key multimap
     *      frees its planes wholesale. */
    ~basic_ram_index() noexcept = default;

    /** @brief Sizes the node records & the key multimap for @p capacity entries - once:
     *      the capacity is frozen by the first success, and later calls answer
     *      @c overflow_k past it with zero side effects. Neighbor rows start empty and
     *      grow with wiring. An allocation failure leaves the shape unfrozen and the same
     *      call retryable. Not safe concurrently with any other operation. */
    expected<std::size_t> reserve(std::size_t capacity) noexcept {
        if (std::size_t const frozen = atomic_ref<std::size_t>(capacity_).load(std::memory_order_acquire); frozen)
            return {frozen, capacity > frozen ? status_t::overflow_k : status_t::success_k};
        if (!capacity) return {0, status_t::config_mismatch_k};
        if (capacity > static_cast<std::size_t>((std::numeric_limits<node_id_t>::max)()))
            return {0, status_t::overflow_k};

        // The record array's extent is frozen right after this, so it is sized exactly;
        // both fallible steps precede any construction, so a failed call builds nothing.
        if (status_t const sized = nodes_array_.reserve_exactly(capacity); failed(sized)) return {0, sized};
        if (status_t const keyed = key_map_.reserve(capacity); failed(keyed)) return {0, keyed};
        for (std::size_t position = 0; position != capacity; ++position)
            nodes_array_.push_back_reserved(node_t {allocator_});

        // The release store freezing the capacity doubles as the publication edge for the
        // whole record array & key multimap - entry points pair it with one acquire load.
        atomic_ref<std::size_t>(capacity_).store(capacity, std::memory_order_release);
        return {capacity, status_t::success_k};
    }

#pragma endregion Lifecycle

#pragma region Mutation

    /** @brief Optional tuning knobs for @c insert(); zeros mean "derive from the
     *      selector's budget". */
    struct insert_config_t {
        /** @brief Breadth of the candidate pool during wiring. */
        std::size_t expansion_factor = 0;
        /** @brief Frontier entries expanded per metric round. */
        std::size_t fan_out = 0;
    };

    /** @brief Outcome & telemetry of one @c insert() batch. */
    struct [[nodiscard]] insert_result_t {
        /** @brief @c success_k, or why the whole batch was rejected before any id was claimed. */
        status_t status = status_t::success_k;
        /** @brief This batch's share of the context's running counters. */
        basic_traversal_telemetry_t telemetry;

        explicit operator bool() const noexcept { return succeeded(status); }
    };

    /**
     *  @brief Moves a batch of values into the index under their keys, walking the graph
     *      for each and wiring the selector's chosen neighbors. All-or-nothing: capacity &
     *      scratch for the whole batch are secured before the first value moves, so a
     *      non-success status means the caller's batch is untouched. Safe concurrently
     *      with @c insert & @c search; never with @c reserve, @c save, @c load, @c health,
     *      or teardown. An @c exclusive_k instantiation compiles the per-node locks out
     *      for single-threaded builds; an @c immutable_k one is rejected at compile time.
     *
     *  Per value: walk the graph scoring against the caller's still-unmoved value, claim
     *  an id from the dispenser, move the value & key into the claimed record, wire the
     *  selected neighbors, take the watermark turn, publish the key, then back-link -
     *  locking one neighbor at a time, never during a metric verb. A full reverse row
     *  re-selects down to the selector's budget - manual over-wiring is advisory here
     *  and may be pruned back.
     *
     *  @param[in] incoming_values The batch to consume - moved from on success only.
     *  @param[in] keys Caller identities, parallel to @p incoming_values.
     *  @param[inout] metric This thread's scoring instance; weights may be shared,
     *      staging scratch must be thread-private.
     *  @param[out] assigned_ids Caller-owned output: either empty, or exactly one slot per
     *      incoming value, receiving that value's dispensed node id - generally
     *      non-contiguous across concurrently inserting threads.
     *  @param[inout] context The calling thread's scratch & telemetry, reused across batches.
     *  @param[in] selector The link-selection heuristic, carrying its own
     *      neighbors-per-node budget - a per-call value, so heuristics & budgets swap per
     *      batch without re-instantiating the index.
     *  @param[in] config Optional tuning knobs; zeros mean "derive from connectivity".
     */
    template <synchronization_t synchronization_ = synchronization_t::concurrent_k, typename metric_type_,
              typename selector_type_ = basic_keep_diverse_t>
        requires basic_metric<metric_type_, value_t, distance_t> &&
                 basic_selector<selector_type_, basic_ram_index, metric_type_>
    insert_result_t insert(std::span<value_t> incoming_values, std::span<key_t const> keys, metric_type_& metric,
                           std::span<node_id_t> assigned_ids, traversal_context_t& context,
                           selector_type_ selector = {}, insert_config_t config = {}) noexcept {
        static_assert(synchronization_ != synchronization_t::immutable_k,
                      "Certified immutability contradicts insertion - certify exclusive_k or stay concurrent_k");
        std::size_t const incoming_count = incoming_values.size();
        if (!incoming_count) return {};
        std::size_t const capacity = atomic_ref<std::size_t>(capacity_).load(std::memory_order_acquire);
        if (!capacity) return {status_t::uninitialized_k, {}};
        std::size_t const connectivity = selector.connectivity;
        if (!connectivity) return {status_t::config_mismatch_k, {}};
        if (keys.size() != incoming_count) return {status_t::config_mismatch_k, {}};
        bool const report_assigned_ids = !assigned_ids.empty();
        if (report_assigned_ids && assigned_ids.size() != incoming_count) return {status_t::config_mismatch_k, {}};

        std::size_t const expansion_factor = config.expansion_factor ? config.expansion_factor : connectivity * 2;
        std::size_t const fan_out = config.fan_out ? config.fan_out : derived_fan_out_(expansion_factor, connectivity);
        if (status_t const sized = reserve_scratch_(context, expansion_factor, fan_out, connectivity); failed(sized))
            return {sized, {}};

        // The gate opens before the claim, so a batch parked behind a unique holder never
        // sits on committed capacity, spuriously overflowing its peers.
        shared_lock_t shared_lock {shared_mutex_};

        // Claims convert one-to-one into dispensed ids, so dispensing, wiring into
        // pre-reserved rows, and the quota-matched key publication can never fail. The
        // claim commits only within capacity, so peers never observe an overshoot.
        std::size_t claimed = atomic_ref<std::size_t>(claimed_count_).load(std::memory_order_relaxed);
        do {
            if (claimed + incoming_count > capacity) return {status_t::overflow_k, {}};
        } while (!atomic_ref<std::size_t>(claimed_count_)
                      .compare_exchange_weak(claimed, claimed + incoming_count, //
                                             std::memory_order_acq_rel, std::memory_order_relaxed));

        basic_traversal_telemetry_t const telemetry_before = context.telemetry;
        for (std::size_t incoming_index = 0; incoming_index != incoming_count; ++incoming_index) {
            value_t& incoming = incoming_values[incoming_index];
            context.results.clear();
            context.frontier.clear();
            context.visited.clear();

            // Walk against the caller's still-unmoved value; a virgin index skips the walk
            // and the first node publishes with an empty neighbors list.
            node_id_t const first_invisible_id =
                atomic_ref<node_id_t>(last_visible_id_).load(std::memory_order_acquire);
            // Capacity for this row is already claimed, so a refused walk wires a thinner
            // node rather than abandoning an id nothing will ever publish.
            if (first_invisible_id)
                search_around_<synchronization_>(incoming, metric, first_invisible_id, expansion_factor, fan_out,
                                                 connectivity, context);
            selector(*this, metric, context);

            // Identity is dispensed here, after the metric-bound walk, so one slow scorer
            // never convoys other publishers through the watermark.
            node_id_t const new_node_id =
                atomic_ref<node_id_t>(next_dispensed_id_).fetch_add(1, std::memory_order_acq_rel);
            node_t& record = nodes_array_[new_node_id];
            record.value = std::move(incoming);
            record.key = keys[incoming_index];
            context.reverse_ids.clear();
            std::size_t const selected_count = min_of(context.selected_neighbors.size(), connectivity);
            // A refused row growth wires a thinner node - the claimed id must still publish.
            std::size_t const wired_count = succeeded(record.neighbors.reserve(selected_count)) ? selected_count : 0;
            for (std::size_t link = 0; link != wired_count; ++link) {
                record.neighbors.push_back_reserved(context.selected_neighbors[link].node_id);
                context.reverse_ids.push_back_reserved(context.selected_neighbors[link].node_id);
            }
            if (report_assigned_ids) assigned_ids[incoming_index] = new_node_id;

            // Take the watermark turn strictly in dispensed order - each thread holds at
            // most one unpublished id, so the wait is bounded by one wiring per peer.
            for (node_id_t watermark = atomic_ref<node_id_t>(last_visible_id_).load(std::memory_order_acquire);
                 watermark != new_node_id;
                 watermark = atomic_ref<node_id_t>(last_visible_id_).load(std::memory_order_acquire))
                wait_t {}(&last_visible_id_, watermark, 0u, wait_uncapped_t {});
            atomic_ref<node_id_t>(last_visible_id_).store(new_node_id + 1, std::memory_order_release);

            // Post-turn: the key becomes findable only for a published node, and the
            // quota matches the capacity claim, so this insert cannot fail.
            [[maybe_unused]] status_t const keyed = key_map_.insert(record.key, new_node_id);

            back_link_<synchronization_>(new_node_id, metric, selector, context);
        }

        insert_result_t result;
        result.telemetry = context.telemetry - telemetry_before;
        return result;
    }

    /** @brief Appends one directed edge @p from → @p to - purely structural: no metric,
     *      no selector, no pruning, and the row grows past any prior extent when needed.
     *      Idempotent: an existing edge answers @c present_k. Both ids must be published;
     *      unpublished ids & self-links answer @c malformed_k. Safe concurrently with
     *      @c insert, @c search, readers & other manual wiring. */
    template <synchronization_t synchronization_ = synchronization_t::concurrent_k>
    expected<membership_t> link(node_id_t from, node_id_t to) noexcept {
        static_assert(synchronization_ != synchronization_t::immutable_k,
                      "Certified immutability contradicts wiring - certify exclusive_k or stay concurrent_k");
        constexpr bool locked_k = synchronization_ == synchronization_t::concurrent_k;
        std::size_t const visible = size();
        if (static_cast<std::size_t>(from) >= visible || static_cast<std::size_t>(to) >= visible || from == to)
            return {membership_t::fresh_k, status_t::malformed_k};

        // Joins the mutator class: manual wiring never moves the watermark, so the shared
        // mutex is what makes it visible to stop-the-world verbs & the certified probes.
        shared_lock_t shared_lock {shared_mutex_};
        node_t& node = nodes_array_[from];
        if constexpr (locked_k) node.mutex.lock();
        membership_t membership = membership_t::fresh_k;
        for (std::size_t position = 0; position != node.neighbors.size() && membership == membership_t::fresh_k;
             ++position)
            if (node.neighbors[position] == to) membership = membership_t::present_k;
        status_t status = status_t::success_k;
        if (membership == membership_t::fresh_k) {
            status = node.neighbors.reserve(node.neighbors.size() + 1);
            if (succeeded(status)) node.neighbors.push_back_reserved(to);
        }
        if constexpr (locked_k) node.mutex.unlock();
        return {membership, status};
    }

    /** @brief Removes one directed edge @p from → @p to; @c malformed_k when the edge or
     *      either id is unpublished or absent. The surviving row's order is unspecified.
     *      Safe concurrently with @c insert, @c search, readers & other manual wiring. */
    template <synchronization_t synchronization_ = synchronization_t::concurrent_k>
    status_t unlink(node_id_t from, node_id_t to) noexcept {
        static_assert(synchronization_ != synchronization_t::immutable_k,
                      "Certified immutability contradicts wiring - certify exclusive_k or stay concurrent_k");
        constexpr bool locked_k = synchronization_ == synchronization_t::concurrent_k;
        std::size_t const visible = size();
        if (static_cast<std::size_t>(from) >= visible || static_cast<std::size_t>(to) >= visible)
            return status_t::malformed_k;

        shared_lock_t shared_lock {shared_mutex_};
        node_t& node = nodes_array_[from];
        if constexpr (locked_k) node.mutex.lock();
        status_t status = status_t::malformed_k;
        for (std::size_t position = 0; position != node.neighbors.size(); ++position)
            if (node.neighbors[position] == to) {
                node.neighbors[position] = node.neighbors.back();
                [[maybe_unused]] node_id_t const dropped = node.neighbors.pop_back();
                status = status_t::success_k;
                break;
            }
        if constexpr (locked_k) node.mutex.unlock();
        return status;
    }

    /** @brief Removals are deleted, not pending: id immortality is what keeps every
     *      reader's stale id valid. */
    status_t remove(std::span<key_t const>, traversal_context_t&) noexcept = delete;

#pragma endregion Mutation

#pragma region Search

    /** @brief Optional tuning knobs for @c search(); zeros mean "derive from
     *      @c connectivity_32_k" - the searcher does not know the budgets the graph was
     *      wired with. */
    struct search_config_t {
        /** @brief Breadth of the candidate pool; raise to trade latency for recall. */
        std::size_t expansion_factor = 0;
        /** @brief Frontier entries expanded per round; their unioned rows form one
         *      metric round - the lever that feeds tiled engines their batch width. */
        std::size_t fan_out = 0;
    };

    /** @brief Outcome & telemetry of one @c search(); matches stay in the caller's context. */
    struct [[nodiscard]] search_result_t {
        /** @brief @c success_k, or why the search could not run at all. */
        status_t status = status_t::success_k;
        /** @brief Matches left in @c context.results, ascending by distance. */
        std::size_t count = 0;
        /** @brief This walk's share of the context's running counters. */
        basic_traversal_telemetry_t telemetry;

        explicit operator bool() const noexcept { return succeeded(status); }
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
     *  @c contended_k rather than corrupting silently.
     *  Scoring runs off-lock, one @c reset / @c push / @c drain round per fan-out, and the
     *  metric need not match the one that built the graph. The query is a stored-value
     *  reference by default; a @c transparent_metric accepts other query forms.
     *
     *  @param[in] query The similarity target, alive for the whole call.
     *  @param[inout] metric This thread's scoring instance.
     *  @param[inout] context The calling thread's scratch & telemetry; holds the matches.
     *  @param[in] config Optional tuning knobs; zeros mean "derive from connectivity".
     */
    template <synchronization_t synchronization_ = synchronization_t::concurrent_k, //
              typename metric_type_, typename query_type_ = value_t>
        requires basic_metric<metric_type_, value_t, distance_t> && basic_query<query_type_, metric_type_, value_t>
    search_result_t search(query_type_ const& query, metric_type_& metric, //
                           traversal_context_t& context, search_config_t config = {}) noexcept {
        if (!capacity()) return {status_t::uninitialized_k, 0, {}};
        std::size_t const expansion_factor = config.expansion_factor ? config.expansion_factor : connectivity_32_k * 2;
        std::size_t const fan_out = config.fan_out ? config.fan_out
                                                   : derived_fan_out_(expansion_factor, connectivity_32_k);
        if (status_t const sized = reserve_scratch_(context, expansion_factor, fan_out, connectivity_32_k);
            failed(sized))
            return {sized, 0, {}};
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
            if (!entry_probe.unlocked()) return {status_t::contended_k, 0, {}};
            entry_generation = entry_probe.generation;
            auto const [published, settled] = published_prefix_();
            if (failed(settled)) return {settled, 0, {}};
            first_invisible_id = published;
        }
        if (!first_invisible_id) return {};

        basic_traversal_telemetry_t const telemetry_before = context.telemetry;
        search_around_<synchronization_>(query, metric, first_invisible_id, expansion_factor, fan_out,
                                         connectivity_32_k, context);

        // A refused walk still ranked whatever it reached; the telemetry says the ranking is
        // short because scratch ran out, not because the graph had nothing better.
        search_result_t result;
        result.count = context.results.size();
        result.telemetry = context.telemetry - telemetry_before;
        if constexpr (synchronization_ != synchronization_t::concurrent_k) {
            // The fence keeps the walk's plain reads from sinking below the reload; the
            // watermark equality is the belt against generation wrap driven by inserts.
            std::atomic_thread_fence(std::memory_order_acquire);
            auto const exit_probe = shared_mutex_.snapshot();
            if (!exit_probe.unlocked() || exit_probe.generation != entry_generation ||
                atomic_ref<node_id_t>(last_visible_id_).load(std::memory_order_acquire) != first_invisible_id)
                result.status = status_t::contended_k;
        }
        return result;
    }

#pragma endregion Search

#pragma region Keys and Introspection

    /** @brief Number of published nodes - the visibility watermark; safe any time. */
    std::size_t size() const noexcept {
        return static_cast<std::size_t>(atomic_ref<node_id_t>(last_visible_id_).load(std::memory_order_acquire));
    }

    /** @brief Total entries the frozen storage can host; zero before the first @c reserve
     *      froze it - the acquire pairs with that freeze, so this is safe any time. */
    std::size_t capacity() const noexcept { return atomic_ref<std::size_t>(capacity_).load(std::memory_order_acquire); }

    /** @brief The key stored under @p node_id - the authoritative direction, immutable
     *      after publication. One acquire load of the watermark gates the id, so any input
     *      is safe any time; an unpublished id answers @c malformed_k. */
    expected<key_t> key_of(node_id_t node_id) const noexcept {
        if (static_cast<std::size_t>(node_id) >= size()) return {key_t {}, status_t::malformed_k};
        return {nodes_array_[node_id].key, status_t::success_k};
    }

    /** @brief The payload stored under @p node_id - never relocated, valid until teardown,
     *      immutable after publication. The same watermark gate as @c key_of, so any input
     *      is safe any time; an unpublished id answers @c malformed_k. */
    expected<value_t const&> value_of(node_id_t node_id) const noexcept {
        if (static_cast<std::size_t>(node_id) >= size()) return {nullptr, status_t::malformed_k};
        return {&nodes_array_[node_id].value, status_t::success_k};
    }

    /** @brief The payload under a @b published @p node_id, unchecked: for traversal-contract
     *      callers - selectors and the walks themselves - whose ids come from the published
     *      pool, where the watermark gate of @c value_of would re-prove a known fact. */
    value_t const& published_value_of(node_id_t node_id) const noexcept { return nodes_array_[node_id].value; }

    /** @brief The out-degree of @p node_id: the same watermark gate as @c key_of, then the
     *      row mutex - every post-publication row edit holds it, so the read is one
     *      consistent instant even while back-linking appends. */
    expected<std::size_t> degree_of(node_id_t node_id) const noexcept {
        if (static_cast<std::size_t>(node_id) >= size()) return {0, status_t::malformed_k};
        node_t const& node = nodes_array_[node_id];
        node.mutex.lock();
        std::size_t const degree = node.neighbors.size();
        node.mutex.unlock();
        return {degree, status_t::success_k};
    }

    /** @brief One published node under @p key: @c malformed_k when absent,
     *      @c uninitialized_k before the first @c reserve. Duplicates surface in an
     *      unspecified order - use @c find_each to see them all. Wait-free, concurrent
     *      with everything. */
    expected<node_id_t> find(key_t const& key) const noexcept {
        if (!capacity()) return {node_id_t {}, status_t::uninitialized_k};
        return key_map_.find(key);
    }

    /** @brief Invokes @p callback with the id of every published node under @p key -
     *      wait-free, concurrent with everything; a no-op before the first @c reserve. */
    template <typename callback_type_>
    void find_each(key_t const& key, callback_type_&& callback) const noexcept {
        if (!capacity()) return;
        key_map_.find_each(key, std::forward<callback_type_>(callback));
    }

    /** @brief Whether at least one published node exists under @p key; @c false before the
     *      first @c reserve. */
    bool contains(key_t const& key) const noexcept {
        if (!capacity()) return false;
        return key_map_.contains(key);
    }

    /** @brief Number of published nodes under @p key; zero before the first @c reserve. */
    std::size_t count(key_t const& key) const noexcept {
        if (!capacity()) return 0;
        return key_map_.count(key);
    }

    /** @brief Post-construction graph report - the quality signal a recall number hides.
     *      Meaningful only on @c success_k. */
    struct [[nodiscard]] graph_health_t {
        /** @brief @c success_k, or why the report could not be taken. */
        status_t status = status_t::success_k;
        /** @brief Forward-reachability components, flooded in id order - equal to weak
         *      components whenever back-linking has kept the edges reciprocal. */
        std::size_t connected_components = 0;
        /** @brief Nodes in the largest such component. */
        std::size_t largest_component = 0;
        /** @brief Nodes no flood from the stochastic entry points reached - the
         *      searcher's-eye blind spot. */
        std::size_t unreachable_from_seeds = 0;
        /** @brief The smallest published out-degree. */
        std::size_t degree_min = 0;
        /** @brief The largest published out-degree. */
        std::size_t degree_max = 0;
        /** @brief Directed edges whose reverse edge also exists. */
        std::size_t reciprocal_edges = 0;
        /** @brief Average published out-degree. */
        f64_t mean_degree = 0;

        explicit operator bool() const noexcept { return succeeded(status); }
    };

    /** @brief Walks the whole graph, reporting components, degree spread, edge
     *      reciprocity, and reachability from stochastic seeds. Takes the shared mutex's
     *      unique side for the whole walk - live mutators answer @c contended_k at entry,
     *      and none can start until the report returns.
     *
     *  @param[inout] context The calling thread's scratch; its visited set and staged
     *      ids grow to the full graph size for the duration of the walk. */
    graph_health_t health(traversal_context_t& context) const noexcept {
        graph_health_t report;
        unique_lock_t unique_lock {shared_mutex_, std::try_to_lock};
        if (!unique_lock) {
            report.status = status_t::contended_k;
            return report;
        }
        auto const [visible, settled] = published_prefix_();
        if (failed(settled)) {
            report.status = settled;
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
        // mutator; the row-id tripwires stay as a corruption net, never an overflow.
        report.degree_min = (std::numeric_limits<std::size_t>::max)();
        std::size_t total_neighbors = 0;
        for (std::size_t node = 0; node != node_count; ++node) {
            neighbors_buffer_t const& row = nodes_array_[node].neighbors;
            report.degree_min = min_of(report.degree_min, row.size());
            report.degree_max = max_of(report.degree_max, row.size());
            total_neighbors += row.size();
            for (std::size_t link = 0; link != row.size(); ++link) {
                if (static_cast<std::size_t>(row[link]) >= node_count) {
                    report.status = status_t::internal_k;
                    continue;
                }
                neighbors_buffer_t const& reverse = nodes_array_[row[link]].neighbors;
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
            [[maybe_unused]] membership_t const seeded = context.visited.set(seed);
            while (!context.staged_ids.empty()) {
                node_id_t const node = context.staged_ids.pop_back();
                ++flooded;
                neighbors_buffer_t const& row = nodes_array_[node].neighbors;
                for (std::size_t link = 0; link != row.size(); ++link) {
                    // The same corruption net as the edge sweep above.
                    if (static_cast<std::size_t>(row[link]) >= node_count) {
                        report.status = status_t::internal_k;
                        continue;
                    }
                    if (context.visited.set(row[link]) == membership_t::fresh_k)
                        context.staged_ids.push_back_reserved(row[link]);
                }
            }
            return flooded;
        };

        context.visited.clear();
        for (std::size_t node = 0; node != node_count; ++node) {
            if (context.visited.test(static_cast<node_id_t>(node))) continue;
            ++report.connected_components;
            report.largest_component = max_of(report.largest_component, flood(static_cast<node_id_t>(node)));
        }

        // Reachability from a handful of stochastic entry points - the searcher's view.
        // The coverage guard is load-bearing twice over: it ends the probe once the graph
        // is covered, and it keeps every seed provably unvisited, which bounds the stack.
        context.visited.clear();
        std::size_t const seed_count = min_of<std::size_t>(health_seeds_k, node_count);
        for (std::size_t seed = 0; seed != seed_count && context.visited.size() != node_count; ++seed) {
            node_id_t draw = uniform_below<node_id_t> {visible}(context.random_generator);
            while (context.visited.test(draw)) draw = draw + 1 == visible ? 0 : draw + 1;
            flood(draw);
        }
        report.unreachable_from_seeds = node_count - context.visited.size();
        return report;
    }

#pragma endregion Keys and Introspection

#pragma region Serialization

    /** @brief One region-parallel save in flight: the caller's five columns, the covered-node
     *      tally, and the shared mutex's unique hold - released when the plan dies or @c finish
     *      pronounces the columns complete. */
    struct [[nodiscard]] save_plan_t {
        /** @brief The index being serialized, whose unique side @c hold occupies; null in the
         *      empty plan a refused @ref plan_save hands back. */
        basic_ram_index const* index = nullptr;
        /** @brief The caller-placed columns every region writes disjoint slices of. */
        columns_t columns {};
        /** @brief Nodes the regions covered so far, mutated through @c atomic_ref. */
        std::size_t covered = 0;
        /** @brief The unique hold spanning the plan's whole lifetime. */
        unique_lock_t hold;

        /** @brief Copies one id region's keys, neighbor ids & encoded values into the columns -
         *      disjoint slices, so any number of caller threads run their regions concurrently;
         *      every slice starts where the plan's prefix sums say. */
        template <typename value_codec_type_>
            requires basic_value_codec<value_codec_type_, value_t>
        status_t save(iota_view<node_id_t> region, value_codec_type_ const& codec) noexcept {
            if (!index) return status_t::uninitialized_k;
            std::size_t const node_count = columns.keys.size();
            if (region.bound < region.first || static_cast<std::size_t>(region.bound) > node_count)
                return status_t::config_mismatch_k;

            for (node_id_t const node : region) columns.keys[node] = index->nodes_array_[node].key;

            std::size_t ids_cursor = columns.neighbor_offsets[region.first];
            for (node_id_t const node : region) {
                neighbors_buffer_t const& row = index->nodes_array_[node].neighbors;
                for (std::size_t link = 0; link != row.size(); ++link) columns.neighbor_ids[ids_cursor++] = row[link];
            }

            // Each value fills exactly the slice its sum reserved; a codec needing more refuses itself.
            for (node_id_t const node : region) {
                std::size_t const first = columns.value_offsets[node];
                std::size_t const bound = columns.value_offsets[node + 1];
                status_t const encoded = codec.save(index->nodes_array_[node].value,
                                                    columns.values.subspan(first, bound - first));
                if (failed(encoded)) return encoded;
            }
            atomic_ref<std::size_t>(covered).fetch_add(region.size(), std::memory_order_acq_rel);
            return status_t::success_k;
        }

        /** @brief Pronounces the columns complete & releases the hold: @c config_mismatch_k
         *      unless the regions' sizes sum to the row count - overlapping regions that still
         *      sum right are the caller's own lie. */
        status_t finish() noexcept {
            if (!index) return status_t::uninitialized_k;
            if (atomic_ref<std::size_t>(covered).load(std::memory_order_acquire) != columns.keys.size())
                return status_t::config_mismatch_k;
            hold.unlock();
            return status_t::success_k;
        }
    };

    /** @brief Opens a region-parallel save: takes the shared mutex's unique side - held
     *      inside the returned plan until its @c finish or its death - sizes the columns
     *      against the published prefix, and writes both prefix-sum columns up front, so the
     *      plan's own @c save calls copy the per-node bulk into slices already named. Live
     *      mutators answer @c contended_k; a mutator call from the plan-holding thread
     *      deadlocks on its own hold, by construction.
     *
     *  A column shorter than its count answers @c overflow_k and a longer one
     *  @c config_mismatch_k, so the key column's length always names the row count. */
    template <typename value_codec_type_>
        requires basic_value_codec<value_codec_type_, value_t>
    expected<save_plan_t> plan_save(columns_t columns, value_codec_type_ const& codec) const noexcept {
        unique_lock_t hold {shared_mutex_, std::try_to_lock};
        if (!hold) return {{}, status_t::contended_k};
        auto const [visible, settled] = published_prefix_();
        if (failed(settled)) return {{}, settled};
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
            node_t const& record = nodes_array_[node];
            running_neighbors += record.neighbors.size();
            running_values += codec.storage_requirements(record.value);
            columns.neighbor_offsets[node + 1] = running_neighbors;
            columns.value_offsets[node + 1] = running_values;
        }
        if (status_t const sized = sized_at_least_(columns.neighbor_ids.size(), running_neighbors); failed(sized))
            return {{}, sized};
        if (status_t const sized = sized_at_least_(columns.values.size(), running_values); failed(sized))
            return {{}, sized};
        return {save_plan_t {this, columns, 0, std::move(hold)}, status_t::success_k};
    }

    /** @brief Row & element counts a save needs right now. Takes the shared mutex's unique
     *      side for the tally; live mutators and open plans answer @c contended_k at entry.
     *
     *  @param[in] codec Sizes each stored value's encoded form. */
    template <typename value_codec_type_>
        requires basic_value_codec<value_codec_type_, value_t>
    expected<basic_columns_extents_t> columns_extents(value_codec_type_ const& codec) const noexcept {
        unique_lock_t unique_lock {shared_mutex_, std::try_to_lock};
        if (!unique_lock) return {{}, status_t::contended_k};
        auto const [visible, settled] = published_prefix_();
        if (failed(settled)) return {{}, settled};
        basic_columns_extents_t extents {static_cast<std::size_t>(visible), 0, 0};
        for (std::size_t node = 0; node != extents.nodes; ++node) {
            node_t const& record = nodes_array_[node];
            extents.neighbors += record.neighbors.size();
            extents.value_bytes += codec.storage_requirements(record.value);
        }
        return {extents, status_t::success_k};
    }

    /** @brief Serializes the snapshot into @p columns as one plan with a single region.
     *      Takes the shared mutex's unique side for the whole copy - live mutators answer
     *      @c contended_k at entry, none can start until the columns are complete, and
     *      searches keep running throughout.
     *
     *  @param[out] columns Caller-placed destinations sized from @ref columns_extents.
     *  @param[in] codec Encodes each stored value into its slice. */
    template <typename value_codec_type_>
        requires basic_value_codec<value_codec_type_, value_t>
    status_t save(columns_t columns, value_codec_type_ const& codec) const noexcept {
        expected<save_plan_t> planned = plan_save(columns, codec);
        if (failed(planned.status)) return planned.status;
        node_id_t const node_count = static_cast<node_id_t>(planned.value.columns.keys.size());
        if (status_t const copied = planned.value.save({0, node_count}, codec); failed(copied)) return copied;
        return planned.value.finish();
    }

    /** @brief One region-parallel load in flight - the mirror of @ref save_plan_t: the
     *      vetted columns, the covered-node tally, and the shared mutex's unique hold,
     *      released when the plan dies or @c finish publishes the loaded graph. */
    struct [[nodiscard]] load_plan_t {
        /** @brief The index being repopulated, whose unique side @c hold occupies; null in the
         *      empty plan a refused @ref plan_load hands back. */
        basic_ram_index* index = nullptr;
        /** @brief The vetted columns being decoded. */
        columns_view_t columns {};
        /** @brief Nodes the regions covered so far, mutated through @c atomic_ref. */
        std::size_t covered = 0;
        /** @brief The unique hold spanning the plan's whole lifetime. */
        unique_lock_t hold;

        /** @brief Decodes one id region's keys, neighbor rows & values into its
         *      pre-constructed records and registers the keys - the multimap inserts are
         *      concurrency-safe, so any number of caller threads run their regions. */
        template <typename value_codec_type_>
            requires basic_value_codec<value_codec_type_, value_t>
        status_t load(iota_view<node_id_t> region, value_codec_type_ const& codec) noexcept {
            if (!index) return status_t::uninitialized_k;
            std::size_t const node_count = columns.keys.size();
            if (region.bound < region.first || static_cast<std::size_t>(region.bound) > node_count)
                return status_t::config_mismatch_k;

            // The plan vetted both prefix-sum columns and every id, so the slices are trusted.
            for (node_id_t const node : region) {
                node_t& record = index->nodes_array_[node];
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

                if (status_t const keyed = index->key_map_.insert(record.key, node); failed(keyed)) return keyed;
            }
            atomic_ref<std::size_t>(covered).fetch_add(region.size(), std::memory_order_acq_rel);
            return status_t::success_k;
        }

        /** @brief Publishes the loaded graph in the tier-wide order - claims, identity,
         *      watermark last - and releases the hold: @c config_mismatch_k unless the
         *      regions' sizes sum to the row count. */
        status_t finish() noexcept {
            if (!index) return status_t::uninitialized_k;
            std::size_t const node_count = columns.keys.size();
            if (atomic_ref<std::size_t>(covered).load(std::memory_order_acquire) != node_count)
                return status_t::config_mismatch_k;
            atomic_ref<std::size_t>(index->claimed_count_).store(node_count, std::memory_order_relaxed);
            atomic_ref<node_id_t>(index->next_dispensed_id_)
                .store(static_cast<node_id_t>(node_count), std::memory_order_relaxed);
            atomic_ref<node_id_t>(index->last_visible_id_)
                .store(static_cast<node_id_t>(node_count), std::memory_order_release);
            hold.unlock();
            return status_t::success_k;
        }
    };

    /** @brief Opens a region-parallel load into this reserved, empty index: takes the
     *      shared mutex's unique side - held inside the returned plan until its
     *      @c finish or its death - vets both prefix-sum columns and every neighbor id
     *      in full, and wipes the key multimap. The plan's own @c load calls then decode
     *      the per-node bulk, and nothing publishes until its @c finish.
     *
     *  @param[in] columns The five columns a prior @c save filled; the key column's length
     *      is the row count. */
    expected<load_plan_t> plan_load(columns_view_t columns) noexcept {
        std::size_t const capacity = atomic_ref<std::size_t>(capacity_).load(std::memory_order_acquire);
        if (!capacity) return {{}, status_t::uninitialized_k};
        // The gate excludes mutators & flips certified probes to refusal; the exclusivity
        // contract against readers stays the caller's - key lookups race the map rebuild.
        unique_lock_t hold {shared_mutex_, std::try_to_lock};
        if (!hold) return {{}, status_t::contended_k};
        if (atomic_ref<node_id_t>(last_visible_id_).load(std::memory_order_acquire) != 0)
            return {{}, status_t::config_mismatch_k};
        if (columns.keys.size() > capacity) return {{}, status_t::overflow_k};
        if (status_t const vetted = vetted_columns_(columns); failed(vetted)) return {{}, vetted};
        // A failed prior attempt burned key-map quota; wiping first keeps every entry
        // unique and the quota whole, so failed loads stay genuinely retryable.
        key_map_.clear();
        return {load_plan_t {this, columns, 0, std::move(hold)}, status_t::success_k};
    }

    /** @brief Loads a snapshot into this reserved, empty index as one plan with a single
     *      region - exclusive: decodes each value into its pre-constructed record,
     *      rebuilds the key multimap, and publishes the watermark last. The key map is
     *      wiped at planning, so a failed attempt may be retried with corrected columns.
     *
     *  @param[in] columns The five columns a prior @c save filled.
     *  @param[in] codec Decodes each value slice into its record. */
    template <typename value_codec_type_>
        requires basic_value_codec<value_codec_type_, value_t>
    status_t load(columns_view_t columns, value_codec_type_ const& codec) noexcept {
        expected<load_plan_t> planned = plan_load(columns);
        if (failed(planned.status)) return planned.status;
        node_id_t const node_count = static_cast<node_id_t>(planned.value.columns.keys.size());
        if (status_t const decoded = planned.value.load({0, node_count}, codec); failed(decoded)) return decoded;
        return planned.value.finish();
    }

#pragma endregion Serialization

  private:
#pragma region Internals

    /** @brief Stochastic entry points the health report floods from. */
    inline static constexpr std::size_t health_seeds_k = 8;

    /** @brief The default frontier fan-out: enough expanded rows for one metric round to
     *      stage a full candidate pool. */
    static std::size_t derived_fan_out_(std::size_t expansion_factor, std::size_t connectivity) noexcept {
        return max_of<std::size_t>(divide_round_up(expansion_factor, connectivity), 1);
    }

    /** @brief Sizes every context container for one walk from the frozen geometry: the pools,
     *      the staging surfaces and the selections never allocate again, the frontier sheds
     *      within its reservation, and the visited set alone may re-table as a walk widens -
     *      once per doubling per context lifetime, which a context kept across walks amortizes. */
    static status_t reserve_scratch_(traversal_context_t& context, std::size_t expansion_factor, std::size_t fan_out,
                                     std::size_t connectivity) noexcept {
        std::size_t const round_capacity = max_of(fan_out * connectivity, connectivity + 1);
        if (status_t const sized = context.results.reserve(max_of(expansion_factor, connectivity + 1)); failed(sized))
            return sized;
        if (status_t const sized = context.frontier.reserve(expansion_factor * 4); failed(sized)) return sized;
        if (status_t const sized = context.visited.reserve(expansion_factor * 4); failed(sized)) return sized;
        if (status_t const sized = context.staged_ids.reserve(round_capacity); failed(sized)) return sized;
        if (status_t const sized = context.staged_distances.reserve(round_capacity); failed(sized)) return sized;
        if (status_t const sized = context.selected_neighbors.reserve(connectivity + 1); failed(sized)) return sized;
        return context.reverse_ids.reserve(connectivity);
    }

    /** @brief Appends @p node_id's live neighbors into @c context.staged_ids - under the
     *      node's mutex on the concurrent contract, as a plain copy otherwise - topping up
     *      the staging surfaces first, since manual linking grows rows past any estimate.
     *      Bounded copies only, never metric verbs.
     *  @return The copied degree, or the refusal that kept the row out of staging. */
    template <synchronization_t synchronization_>
    expected<std::size_t> copy_row_(node_id_t node_id, traversal_context_t& context) const noexcept {
        constexpr bool locked_k = synchronization_ == synchronization_t::concurrent_k;
        node_t const& node = nodes_array_[node_id];
        if constexpr (locked_k) node.mutex.lock();
        std::size_t const degree = node.neighbors.size();
        std::size_t const staged_bound = context.staged_ids.size() + degree;
        if (failed(context.staged_ids.reserve(staged_bound)) ||
            failed(context.staged_distances.reserve(staged_bound))) {
            if constexpr (locked_k) node.mutex.unlock();
            return {0, status_t::bad_alloc_k};
        }
        for (std::size_t position = 0; position != degree; ++position)
            context.staged_ids.push_back_reserved(node.neighbors[position]);
        if constexpr (locked_k) node.mutex.unlock();
        return {degree, status_t::success_k};
    }

    /** @brief Scores up to @p wanted fresh stochastic landmarks in one builder round,
     *      seeding the results & frontier. Visited collisions probe forward cyclically,
     *      staying exact.
     *  @return The landmarks scored; zero means the published set is exhausted, staging was
     *      refused, or scratch could not grow - the last counted as a saturated walk - and is
     *      the reason to stop. */
    template <typename metric_type_, typename query_type_>
    std::size_t sample_landmarks_(query_type_ const& query, metric_type_& metric, node_id_t first_invisible_id,
                                  std::size_t wanted, std::size_t expansion_factor,
                                  traversal_context_t& context) const noexcept {
        if (failed(context.visited.reserve(context.visited.size() + wanted)))
            return ++context.telemetry.saturated_walks, 0;
        context.staged_ids.clear();
        metric.reset(query);
        while (context.staged_ids.size() != wanted &&
               context.visited.size() < static_cast<std::size_t>(first_invisible_id)) {
            node_id_t draw = uniform_below<node_id_t> {first_invisible_id}(context.random_generator);
            while (context.visited.test(draw)) draw = draw + 1 == first_invisible_id ? 0 : draw + 1;
            if (metric.push(published_value_of(draw)) == basic_metric_staging_t::refused_k)
                break; // Unmarked, so a later round may retry it.
            [[maybe_unused]] membership_t const marked = context.visited.set(draw);
            context.staged_ids.push_back_reserved(draw);
        }
        std::size_t const staged_count = context.staged_ids.size();
        if (!staged_count) return 0;
        drain_round(metric, staged_count, context);
        for (std::size_t lane = 0; lane != staged_count; ++lane) {
            scored_node_t const scored {context.staged_ids[lane], context.staged_distances[lane]};
            [[maybe_unused]] bool const ranked = context.results.insert(scored_node_t {scored}, expansion_factor);
            if (failed(context.frontier.insert(scored_node_t {scored})))
                return ++context.telemetry.saturated_walks, lane;
        }
        return staged_count;
    }

    /** @brief One round's expansion half: peels up to @p fan_out frontier entries within
     *      the stopping radius, copies their rows, and pushes the watermark- and
     *      visited-filtered union into the metric, compacted in @c context.staged_ids. */
    template <synchronization_t synchronization_, typename metric_type_, typename query_type_>
    basic_expansion_round_t expand_frontier_(query_type_ const& query, metric_type_& metric,
                                             node_id_t first_invisible_id, std::size_t expansion_factor,
                                             std::size_t fan_out, traversal_context_t& context) const noexcept {
        context.staged_ids.clear();
        metric.reset(query);
        for (std::size_t expanded_count = 0; expanded_count != fan_out && !context.frontier.empty(); ++expanded_count) {
            scored_node_t const best = context.frontier.top();
            bool const results_full = context.results.size() == expansion_factor;
            if (results_full && best.distance > context.results.top().distance) {
                if (expanded_count == 0) return basic_expansion_round_t::converged_k;
                break;
            }

            std::size_t const segment_start = context.staged_ids.size();
            expected<std::size_t> const copied = copy_row_<synchronization_>(best.node_id, context);
            if (failed(copied.status) || failed(context.visited.reserve(context.visited.size() + copied.value))) {
                // Drop the raw row copy: only pushed, filtered ids may reach the drain.
                // The entry stays on the frontier - nothing is lost, only postponed.
                context.staged_ids.resize_reserved(segment_start);
                ++context.telemetry.saturated_walks;
                return basic_expansion_round_t::saturated_k;
            }
            std::size_t const degree = copied.value;
            context.frontier.pop();
            ++context.telemetry.visited_members;

            std::size_t kept = segment_start;
            for (std::size_t position = segment_start; position != segment_start + degree; ++position) {
                node_id_t const neighbor_id = context.staged_ids[position];
                // The translation frontier: an id at or above the snapshot has no
                // happens-before edge to its record, so a later query picks it up.
                if (neighbor_id >= first_invisible_id) continue;
                if (context.visited.test(neighbor_id)) continue;
                if (metric.push(published_value_of(neighbor_id)) == basic_metric_staging_t::refused_k) {
                    context.staged_ids.resize_reserved(kept);
                    return basic_expansion_round_t::saturated_k;
                }
                [[maybe_unused]] membership_t const marked = context.visited.set(neighbor_id);
                context.staged_ids[kept] = neighbor_id;
                ++kept;
            }
            context.staged_ids.resize_reserved(kept);
        }
        return basic_expansion_round_t::advanced_k;
    }

    /**
     *  @brief The shared walk of @c search & @c insert: seed, then per round expand the
     *      nearest frontier entries and score the staged union in one builder round.
     *
     *  The walk never fails hard: an exhausted published set or a refused staging ends it
     *  normally, and a refused re-table ends it early with the ranking it has and counts a
     *  saturated walk - an insertion wires the thinner row regardless, and a search reports
     *  the count so the caller can widen the scratch. Frontier starvation with an unfilled
     *  pool re-seeds and counts a restart. The frontier-admission rule: a candidate enters
     *  the frontier iff it improved the results pool, or the frontier would otherwise be
     *  empty - the seedings.
     */
    template <synchronization_t synchronization_, typename metric_type_, typename query_type_>
    void search_around_(query_type_ const& query, metric_type_& metric, node_id_t first_invisible_id,
                        std::size_t expansion_factor, std::size_t fan_out, std::size_t connectivity,
                        traversal_context_t& context) const noexcept {
        while (true) {
            // An empty frontier seeds: the first round, or a restart after it drained before the
            // pool filled. Nothing left to seed from ends the walk.
            if (context.frontier.empty()) {
                if (context.results.size() >= expansion_factor) return;
                bool const restarting = !context.results.empty();
                if (!sample_landmarks_(query, metric, first_invisible_id, connectivity, expansion_factor, context))
                    return;
                context.telemetry.restarts += restarting;
                continue;
            }
            basic_expansion_round_t const expanded = expand_frontier_<synchronization_>(
                query, metric, first_invisible_id, expansion_factor, fan_out, context);
            if (expanded == basic_expansion_round_t::converged_k) return;
            basic_expansion_round_t const ranked = rank_round_(metric, expansion_factor, context);
            if (expanded == basic_expansion_round_t::saturated_k || ranked == basic_expansion_round_t::saturated_k)
                return;
        }
    }

    /** @brief One round's scoring half: drains the staged union through the metric and pushes what
     *      improved the pool into the frontier, shedding the frontier past a full pool's radius
     *      first so the pushes fit. */
    template <typename metric_type_>
    basic_expansion_round_t rank_round_(metric_type_& metric, std::size_t expansion_factor,
                                        traversal_context_t& context) const noexcept {
        std::size_t const staged_count = context.staged_ids.size();
        if (!staged_count) return basic_expansion_round_t::advanced_k;
        drain_round(metric, staged_count, context);
        if (context.results.size() == expansion_factor &&
            context.frontier.capacity() - context.frontier.size() < staged_count)
            context.frontier.erase_below(context.results.top());
        for (std::size_t lane = 0; lane != staged_count; ++lane) {
            scored_node_t const scored {context.staged_ids[lane], context.staged_distances[lane]};
            if (!context.results.insert(scored_node_t {scored}, expansion_factor)) continue;
            if (failed(context.frontier.insert(scored_node_t {scored}))) {
                ++context.telemetry.saturated_walks;
                return basic_expansion_round_t::saturated_k;
            }
        }
        return basic_expansion_round_t::advanced_k;
    }

    /**
     *  @brief Wires the reverse edges of the freshly published node, walking the links
     *      snapshotted in @c context.reverse_ids: per neighbor - lock, append when room
     *      remains at zero metric cost; otherwise copy the full row, unlock, re-select
     *      off-lock through the selector, relock, and apply only if the row is still the
     *      one copied - a lost race falls back to append-if-room, never a re-score.
     */
    template <synchronization_t synchronization_, typename metric_type_, typename selector_type_>
    void back_link_(node_id_t new_node_id, metric_type_& metric, selector_type_ const& selector,
                    traversal_context_t& context) noexcept {
        constexpr bool locked_k = synchronization_ == synchronization_t::concurrent_k;
        std::size_t const connectivity = selector.connectivity;
        value_t const& new_value = published_value_of(new_node_id);
        // The newcomer published before this pass, so a racing manual link may have wired
        // it already - every append re-checks under the same lock, or edges duplicate.
        auto already_linked = [new_node_id](node_t const& node) noexcept {
            for (std::size_t position = 0; position != node.neighbors.size(); ++position)
                if (node.neighbors[position] == new_node_id) return true;
            return false;
        };
        for (std::size_t reverse = 0; reverse != context.reverse_ids.size(); ++reverse) {
            node_id_t const neighbor_id = context.reverse_ids[reverse];
            node_t& node = nodes_array_[neighbor_id];

            if constexpr (locked_k) node.mutex.lock();
            if (node.neighbors.size() < connectivity) {
                // A refused growth loses one reverse link - tolerated, like refused staging.
                if (!already_linked(node) && succeeded(node.neighbors.reserve(node.neighbors.size() + 1)))
                    node.neighbors.push_back_reserved(new_node_id);
                if constexpr (locked_k) node.mutex.unlock();
                continue;
            }
            std::size_t const degree = node.neighbors.size();
            context.staged_ids.clear();
            if (failed(context.staged_ids.reserve(degree))) {
                if constexpr (locked_k) node.mutex.unlock();
                continue; // A refused staging loses one reverse link.
            }
            for (std::size_t position = 0; position != degree; ++position)
                context.staged_ids.push_back_reserved(node.neighbors[position]);
            if constexpr (locked_k) node.mutex.unlock();
            if (failed(context.staged_distances.reserve(degree + 1)) || failed(context.results.reserve(degree + 1)))
                continue; // A refused staging loses one reverse link.

            // Off-lock: pool = the copied row plus the newcomer, scored against the
            // neighbor's own stored value in one builder round.
            metric.reset(published_value_of(neighbor_id));
            std::size_t staged_count = 0;
            for (; staged_count != degree; ++staged_count)
                if (metric.push(published_value_of(context.staged_ids[staged_count])) ==
                    basic_metric_staging_t::refused_k)
                    break;
            bool const complete = staged_count == degree && metric.push(new_value) == basic_metric_staging_t::staged_k;
            staged_count += complete;
            drain_round(metric, staged_count, context);
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
            if constexpr (locked_k) node.mutex.lock();
            bool unchanged = node.neighbors.size() == degree;
            for (std::size_t position = 0; position != degree && unchanged; ++position)
                unchanged = node.neighbors[position] == context.staged_ids[position];
            if (unchanged) {
                node.neighbors.clear();
                std::size_t const kept_count = min_of(context.selected_neighbors.size(), connectivity);
                for (std::size_t kept = 0; kept != kept_count; ++kept)
                    node.neighbors.push_back_reserved(context.selected_neighbors[kept].node_id);
            }
            else if (node.neighbors.size() < connectivity && !already_linked(node) &&
                     succeeded(node.neighbors.reserve(node.neighbors.size() + 1))) {
                node.neighbors.push_back_reserved(new_node_id);
            }
            if constexpr (locked_k) node.mutex.unlock();
        }
    }

    /** @brief The published prefix at unique acquisition; @c contended_k when the dispenser has
     *      run ahead of the watermark, which only an exclusive-certified mutator lying mid-turn
     *      can do with the gate held. */
    expected<node_id_t> published_prefix_() const noexcept {
        node_id_t const visible = atomic_ref<node_id_t>(last_visible_id_).load(std::memory_order_acquire);
        return {visible, visible == atomic_ref<node_id_t>(next_dispensed_id_).load(std::memory_order_acquire) //
                             ? status_t::success_k
                             : status_t::contended_k};
    }

    /** @brief Refuses a column of @p size that is not exactly @p wanted elements: a short one
     *      overflows, a long one disagrees about the row count. For the three node-sized columns,
     *      whose length is what names the row count. */
    static status_t sized_exactly_(std::size_t size, std::size_t wanted) noexcept {
        if (size < wanted) return status_t::overflow_k;
        if (size > wanted) return status_t::config_mismatch_k;
        return status_t::success_k;
    }

    /** @brief Refuses a column of @p size shorter than @p wanted elements. For the two bulk
     *      columns, whose content the prefix sums delimit, so spare room is unused rather than
     *      wrong - and a graph that sheds edges between sizing and planning still saves. */
    static status_t sized_at_least_(std::size_t size, std::size_t wanted) noexcept {
        return size < wanted ? status_t::overflow_k : status_t::success_k;
    }

    /** @brief Whether @p sums start at zero, never fall, never pass @p total, and end at it. */
    static status_t monotone_sums_(std::span<u64_t const> sums, std::size_t total) noexcept {
        if (sums.empty() || sums.front() != 0) return status_t::malformed_k;
        u64_t previous = 0;
        for (u64_t const next : sums.subspan(1)) {
            if (next < previous || next > total) return status_t::malformed_k;
            previous = next;
        }
        return previous == total ? status_t::success_k : status_t::malformed_k;
    }

    /** @brief Vets untrusted columns in full - both prefix-sum columns sized to the rows and
     *      monotone over their element columns, every neighbor id naming a row - so the region
     *      decoders trust every slice they read. */
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

#pragma endregion Internals

#pragma region State

    /** @brief The node records, sized by @c reserve, addresses stable forever. */
    node_array_t nodes_array_;
    /** @brief Frozen entry capacity; its release store is @c reserve's last act, doubling
     *      as the publication edge for the record array & key multimap. Public entry
     *      points take one acquire load into a local, so no loop reads it. */
    mutable std::size_t capacity_ {0}; // ? `mutable` throughout: the pre-P3323 `std::atomic_ref` binds no
    // const word, not even to load it, and the const readers only load

    /** @brief Batch capacity claims - the all-or-nothing admission, separate from
     *      identity: claims convert one-to-one into dispensed ids, so dispensing never
     *      overflows. Rejected batches roll their claim back, so the count transiently
     *      overshoots under contention - kept a full word wide, so no overshoot can ever
     *      wrap it around the bounds check. */
    alignas(default_alignment_k) mutable std::size_t claimed_count_ {0};
    /** @brief Node-id dispenser; ids are claimed after the walk, right before publication.
     *      Its per-insert RMW gets its own line, so it never invalidates the watermark line
     *      every search snapshots. */
    alignas(default_alignment_k) mutable node_id_t next_dispensed_id_ {0};
    /** @brief The visibility watermark: every id strictly below it is published. */
    alignas(default_alignment_k) mutable node_id_t last_visible_id_ {0};
    /** @brief Key → ID multimap, sized by @c reserve. */
    key_map_t key_map_;

    /** @brief The world gate: mutators hold it shared, stop-the-world verbs try-unique,
     *      certified readers probe it wait-free. Mutable for the const cold readers, like
     *      the row mutexes; on its own line, away from the watermark, which turn loops spin
     *      on. */
    alignas(default_alignment_k) mutable shared_mutex_t shared_mutex_;

    [[no_unique_address]] allocator_t allocator_;

#pragma endregion State
};

/**
 *  @brief External reader's cursor over the published prefix of one @c basic_ram_index:
 *      id-order iteration, random seeks, and edge-following through a privately staged
 *      copy of the current node's row.
 *
 *  Strictly read-only under every contract, so readers never stall writers. One watermark
 *  snapshot, captured at construction and widened by @c refresh, bounds everything: keys
 *  & values below it are immutable and read wait-free, rows are staged under the row
 *  mutex on the concurrent arm and as plain copies on the certified ones. A staged row
 *  may name ids at or past this cursor's snapshot - fresh back-links - and @c follow
 *  refuses those with @c malformed_k until a @c refresh admits them. The cursor starts
 *  past the end; @c seek(0) enters the graph.
 */
template <typename index_type_, synchronization_t synchronization_, typename allocator_type_>
struct basic_ram_cursor {
    using index_t = index_type_;
    template <typename word_type_>
    using atomic_ref = typename index_t::template atomic_ref<word_type_>;
    using node_id_t = typename index_t::node_id_t;
    using key_t = typename index_t::key_t;
    using value_t = typename index_t::value_t;
    using allocator_t = allocator_type_;

    basic_ram_cursor(basic_ram_cursor const&) = delete;
    basic_ram_cursor& operator=(basic_ram_cursor const&) = delete;
    basic_ram_cursor(basic_ram_cursor&&) noexcept = default;
    basic_ram_cursor& operator=(basic_ram_cursor&&) = delete;

    /** @brief Binds the index & captures the watermark snapshot; allocates nothing - the
     *      staged-row buffer grows on the first @c seek through @p allocator. */
    explicit basic_ram_cursor(index_t const& index, allocator_t allocator = {}) noexcept
        : index_(index), staged_row_(allocator),
          visible_count_(atomic_ref<node_id_t>(index.last_visible_id_).load(watermark_order_k)),
          current_(visible_count_) {}

    /** @brief Ids strictly below this snapshot are readable through this cursor. */
    node_id_t visible_count() const noexcept { return visible_count_; }

    /** @brief Widens the snapshot to admit nodes published since; a cursor standing past the
     *      end stays past the end - entering the admitted region takes an explicit @c seek. */
    void refresh() noexcept {
        bool const past_end = current_ == visible_count_;
        visible_count_ = atomic_ref<node_id_t>(index_.last_visible_id_).load(watermark_order_k);
        if (past_end) current_ = visible_count_;
    }

    /** @brief Whether the cursor stands on a readable node. */
    explicit operator bool() const noexcept { return current_ < visible_count_; }

    node_id_t id() const noexcept { return current_; }
    key_t key() const noexcept { return index_.nodes_array_[current_].key; }
    value_t const& value() const noexcept { return index_.nodes_array_[current_].value; }
    std::span<node_id_t const> neighbors() const noexcept { return {staged_row_.begin(), staged_row_.end()}; }

    /** @brief Positions on @p node & stages its row: @c malformed_k at or past the
     *      snapshot, @c bad_alloc_k when the staging buffer refused to grow - the cursor
     *      stays where it was on either refusal. */
    status_t seek(node_id_t node) noexcept {
        if (static_cast<std::size_t>(node) >= static_cast<std::size_t>(visible_count_)) return status_t::malformed_k;
        typename index_t::node_t const& record = index_.nodes_array_[node];
        if constexpr (synchronization_ == synchronization_t::concurrent_k) record.mutex.lock();
        std::size_t const degree = record.neighbors.size();
        status_t const staged = staged_row_.reserve(degree);
        if (succeeded(staged)) {
            staged_row_.clear();
            for (std::size_t link = 0; link != degree; ++link) staged_row_.push_back_reserved(record.neighbors[link]);
        }
        if constexpr (synchronization_ == synchronization_t::concurrent_k) record.mutex.unlock();
        if (failed(staged)) return staged;
        current_ = node;
        return status_t::success_k;
    }

    /** @brief Advances to the next published id, staging its row; past the last one the
     *      cursor turns false. A staging refusal parks on the new id with an empty row -
     *      a later @c seek to the same id retries the staging. */
    basic_ram_cursor& operator++() noexcept {
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
    /** @brief Deleted: it would copy the staged row - an allocation in disguise. */
    basic_ram_cursor operator++(int) = delete;

    /** @brief Hops to the staged row's @p position -th neighbor: @c malformed_k past the
     *      staged degree or when the neighbor sits at or past the snapshot. */
    status_t follow(std::size_t position) noexcept {
        if (position >= staged_row_.size()) return status_t::malformed_k;
        return seek(staged_row_[position]);
    }

  private:
    /** @brief Certified contracts already ordered every publication, so no acquire. */
    inline static constexpr std::memory_order watermark_order_k = synchronization_ == synchronization_t::concurrent_k
                                                                      ? std::memory_order_acquire
                                                                      : std::memory_order_relaxed;

    index_t const& index_;
    /** @brief The private copy of the current node's row - @c follow's ground truth. */
    growing_buffer<node_id_t, allocator_t> staged_row_;
    /** @brief This cursor's watermark snapshot; ids strictly below it are readable. */
    node_id_t visible_count_;
    /** @brief The current id, or @c visible_count_ when past the end. */
    node_id_t current_;
};

} // namespace usearch
} // namespace unum
