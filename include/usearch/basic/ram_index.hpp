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

#include <array>       // `std::array` - the byte images of serialized words
#include <atomic>      // `std::atomic`
#include <bit>         // `std::bit_cast` - unaligned-safe snapshot reads & writes
#include <limits>      // `std::numeric_limits` - the id-space ceiling in `reserve`
#include <memory>      // `std::allocator`, `std::allocator_traits`
#include <span>        // `std::span`
#include <type_traits> // `std::is_nothrow_move_assignable_v`
#include <utility>     // `std::move`, `std::forward`

#include "usearch/utilities/types.hpp"
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
                                     value_type_& mutable_value, std::span<std::byte> output,
                                     std::span<std::byte const> input) {
    { codec.saved_bytes(immutable_value) } noexcept -> std::same_as<std::size_t>;
    { codec.save(immutable_value, output) } noexcept -> std::same_as<status_t>;
    { codec.load(input, mutable_value) } noexcept -> std::same_as<status_t>;
};

/** @brief The three per-walk counters every traversal accumulates; results report one
 *      call's share as the difference of two snapshots. */
struct basic_traversal_telemetry_t {
    /** @brief Metric invocations, batched rounds counted once per scored candidate. */
    std::size_t computed_distances = 0;
    /** @brief Frontier candidates whose neighbor lists were expanded. */
    std::size_t visited_members = 0;
    /** @brief Re-seedings after the frontier drained prematurely - a connectivity signal. */
    std::size_t restarts = 0;

    basic_traversal_telemetry_t operator-(basic_traversal_telemetry_t const& before) const noexcept {
        return {computed_distances - before.computed_distances, visited_members - before.visited_members,
                restarts - before.restarts};
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

/** @brief Minimal test-and-test-and-set lock over one @c std::atomic<bool>. The waiter
 *      policy monitors the flag's address, so hardware waits plug in without this header
 *      naming them; the claiming exchange is the loop's only store, so a monitored waiter
 *      wakes once per @c unlock rather than on every contender's attempt. */
template <typename wait_type_ = dummy_waiter_t>
struct basic_mutex {
    std::atomic<bool> flag {false};

    void lock() noexcept {
        while (flag.exchange(true, std::memory_order_acquire))
            while (flag.load(std::memory_order_relaxed)) wait_type_ {}(flag, true, 0u, wait_capped_t {});
    }
    /** @brief Takes the lock only if it is free right now - a contention question, never a
     *      failure. Named for the standard @c Lockable requirement, which @c lock, @c unlock
     *      and this together satisfy, so the STL's lock adapters accept this type. */
    [[nodiscard]] bool try_lock() noexcept { return !flag.exchange(true, std::memory_order_acquire); }
    void unlock() noexcept { flag.store(false, std::memory_order_release); }
};

/**
 *  @brief Per-thread traversal scratch: the candidate queues, the exact visited set,
 *      the staging surfaces of batched metric rounds, and running telemetry.
 */
template <typename node_id_type_ = u64_t, typename distance_type_ = f64_t,
          typename allocator_type_ = std::allocator<std::byte>>
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

/**
 *  @brief Fixed descriptor fronting every basic-tier snapshot. Sections follow in order -
 *      keys, neighbor prefix sums, neighbor ids, value prefix sums, codec value bytes -
 *      each starting on an @c alignment boundary, so an interchange adapter re-frames the
 *      payload without relocating a single buffer.
 */
struct basic_snapshot_header_t {
    /** @brief The shared USearch payload magic. @ref usearch_magic_k */
    inline static constexpr u64_t magic_k = usearch_magic_k;
    /** @brief "bsic" read as a little-endian word - discriminates this layout from the
     *      dense payloads, whose word after the magic is a small version number. */
    inline static constexpr u32_t layout_k = 0x63697362u;
    /** @brief The layout revision this implementation writes and understands. */
    inline static constexpr u32_t version_k = 1;
    /** @brief The smallest legal section alignment. */
    inline static constexpr u32_t alignment_floor_k = 64u;

    /** @brief Identifies USearch payloads; must equal @c magic_k. */
    u64_t magic = magic_k;
    /** @brief Identifies the basic-tier layout; must equal @c layout_k. */
    u32_t layout = layout_k;
    /** @brief Layout revision of the sections that follow. */
    u32_t version = version_k;
    /** @brief Section alignment in bytes - a power of two, at least the floor. */
    u32_t alignment = 4096;
    /** @brief The constant per-node degree, or zero when degrees vary - the interchange
     *      convention for non-constant connectivity. */
    u32_t connectivity = 0;
    /** @brief Width of one serialized node id; must match the loading instantiation. */
    u32_t node_id_bytes = 0;
    /** @brief Width of one serialized key; must match the loading instantiation. */
    u32_t key_bytes = 0;
    /** @brief Published nodes in the payload. */
    u64_t node_count = 0;
    /** @brief Total neighbor ids across all rows. */
    u64_t neighbors_count = 0;
    /** @brief Total codec-encoded value bytes. */
    u64_t value_bytes = 0;
};
static_assert(sizeof(basic_snapshot_header_t) == 56 && std::is_trivially_copyable_v<basic_snapshot_header_t>,
              "The header is the serialized preamble, so its size & layout are contractual");

/**
 *  @brief Similarity Search index baseline for arbitrary values & similarity measures.
 *
 *  The design optimizes for being read, and its storage is one array of node records
 *  sized once by @c reserve, plus the key multimap. Each record owns its neighbors row,
 *  which starts empty and grows with wiring - automatic through @c insert's selector,
 *  which carries the neighbors-per-node budget, or manual through @c link. Metric
 *  calls cost microseconds to milliseconds here, so brief per-node locks are noise -
 *  the only hard rule is that no metric verb ever runs while any lock is held.
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
 *  @tparam allocator_type_ places the node records, neighbor buffers & key multimap.
 */
template <                                               //
    typename value_type_,                                //
    typename key_type_ = uuid_t,                         //
    typename node_id_type_ = u64_t,                      //
    typename distance_type_ = f64_t,                     //
    typename wait_type_ = dummy_waiter_t,                //
    typename allocator_type_ = std::allocator<std::byte> //
    >
    requires basic_value<value_type_> && (std::same_as<node_id_type_, u32_t> || std::same_as<node_id_type_, u64_t>)
struct basic_ram_index {

    using value_t = value_type_;
    using key_t = key_type_;
    using node_id_t = node_id_type_;
    using distance_t = distance_type_;
    using wait_t = wait_type_;
    using allocator_t = allocator_type_;

    using scored_node_t = basic_scored_node<node_id_t, distance_t>;
    using traversal_context_t = basic_traversal_context<node_id_t, distance_t, allocator_t>;

  private:
    using neighbors_buffer_t = growing_buffer<node_id_t, allocator_t>;

    /** @brief One node record: the payload, its key, the neighbors it links to, and the
     *      mutex guarding those neighbors. The value & key are written before publication
     *      and never change; only @c neighbors mutates after - wired by inserts, grown or
     *      trimmed by manual linking - with its own @c size() as the live degree. */
    struct node_t {
        /** @brief Mutable so id-addressed const readers, like @c degree_of, may briefly
         *      hold it - locking never mutates the record's logical state. */
        mutable basic_mutex<wait_t> mutex;
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
        if (std::size_t const frozen = capacity_.load(std::memory_order_acquire); frozen)
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
        capacity_.store(capacity, std::memory_order_release);
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
        std::size_t const capacity = capacity_.load(std::memory_order_acquire);
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

        // Claims convert one-to-one into dispensed ids, so dispensing, wiring into
        // pre-reserved rows, and the quota-matched key publication can never fail. The
        // claim commits only within capacity, so peers never observe an overshoot.
        std::size_t claimed = claimed_count_.load(std::memory_order_relaxed);
        do {
            if (claimed + incoming_count > capacity) return {status_t::overflow_k, {}};
        } while (!claimed_count_.compare_exchange_weak(claimed, claimed + incoming_count, //
                                                       std::memory_order_acq_rel, std::memory_order_relaxed));
        // Counted before any dispense of this batch, so a guard reading equal counters and
        // a zero in-flight half through its acquires is reading a truly settled past.
        mutation_census_.fetch_add(mutation_started_k + 1, std::memory_order_acq_rel);

        basic_traversal_telemetry_t const telemetry_before = context.telemetry;
        for (std::size_t incoming_index = 0; incoming_index != incoming_count; ++incoming_index) {
            value_t& incoming = incoming_values[incoming_index];
            context.results.clear();
            context.frontier.clear();
            context.visited.clear();

            // Walk against the caller's still-unmoved value; a virgin index skips the walk
            // and the first node publishes with an empty neighbors list.
            node_id_t const first_invisible_id = last_visible_id_.load(std::memory_order_acquire);
            // Capacity for this row is already claimed, so a refused walk wires a thinner
            // node rather than abandoning an id nothing will ever publish.
            if (first_invisible_id) [[maybe_unused]]
                status_t const walked = search_around_<synchronization_>(
                    incoming, metric, first_invisible_id, expansion_factor, fan_out, connectivity, context);
            selector(*this, metric, context);

            // Identity is dispensed here, after the metric-bound walk, so one slow scorer
            // never convoys other publishers through the watermark.
            node_id_t const new_node_id = next_dispensed_id_.fetch_add(1, std::memory_order_acq_rel);
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
            for (node_id_t watermark = last_visible_id_.load(std::memory_order_acquire); watermark != new_node_id;
                 watermark = last_visible_id_.load(std::memory_order_acquire))
                wait_t {}(last_visible_id_, watermark, 0u, wait_uncapped_t {});
            last_visible_id_.store(new_node_id + 1, std::memory_order_release);

            // Post-turn: the key becomes findable only for a published node, and the
            // quota matches the capacity claim, so this insert cannot fail.
            [[maybe_unused]] status_t const keyed = key_map_.insert(record.key, new_node_id);

            back_link_<synchronization_>(new_node_id, metric, selector, context);
        }
        // The release orders every back-link of this batch before a guard's acquire.
        mutation_census_.fetch_sub(1, std::memory_order_release);

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

        // Joins the mutator census: manual wiring never moves the watermark, so the
        // census word - its started-half in particular - is what lets the settled-past
        // guard see even a link that finishes before the guard's second probe.
        mutation_census_.fetch_add(mutation_started_k + 1, std::memory_order_acq_rel);
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
        mutation_census_.fetch_sub(1, std::memory_order_release);
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

        mutation_census_.fetch_add(mutation_started_k + 1, std::memory_order_acq_rel);
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
        mutation_census_.fetch_sub(1, std::memory_order_release);
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
     *  bracketed by the settled-past guard: a violated certification answers
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
        if (!capacity_.load(std::memory_order_acquire)) return {status_t::uninitialized_k, 0, {}};
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
        // certification through the settled-past guard instead of trusting it.
        node_id_t first_invisible_id = 0;
        [[maybe_unused]] u64_t settled_census = 0;
        if constexpr (synchronization_ == synchronization_t::concurrent_k)
            first_invisible_id = last_visible_id_.load(std::memory_order_acquire);
        else if (!settled_past_(first_invisible_id, settled_census)) return {status_t::contended_k, 0, {}};
        if (!first_invisible_id) return {};

        basic_traversal_telemetry_t const telemetry_before = context.telemetry;
        status_t const walked = search_around_<synchronization_>(query, metric, first_invisible_id, expansion_factor,
                                                                 fan_out, connectivity_32_k, context);

        // A refused walk still ranked whatever it reached; the status says the ranking is
        // short because scratch ran out, not because the graph had nothing better.
        search_result_t result;
        result.status = walked;
        result.count = context.results.size();
        result.telemetry = context.telemetry - telemetry_before;
        if constexpr (synchronization_ != synchronization_t::concurrent_k)
            if (!still_settled_(first_invisible_id, settled_census)) result.status = status_t::contended_k;
        return result;
    }

#pragma endregion Search

#pragma region Keys and Introspection

    /** @brief Number of published nodes - the visibility watermark; safe any time. */
    std::size_t size() const noexcept {
        return static_cast<std::size_t>(last_visible_id_.load(std::memory_order_acquire));
    }

    /** @brief Total entries the frozen storage can host; zero before the first @c reserve
     *      froze it - the acquire pairs with that freeze, so this is safe any time. */
    std::size_t capacity() const noexcept { return capacity_.load(std::memory_order_acquire); }

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
        if (!capacity_.load(std::memory_order_acquire)) return {node_id_t {}, status_t::uninitialized_k};
        return key_map_.find(key);
    }

    /** @brief Invokes @p callback with the id of every published node under @p key -
     *      wait-free, concurrent with everything; a no-op before the first @c reserve. */
    template <typename callback_type_>
    void find_each(key_t const& key, callback_type_&& callback) const noexcept {
        if (!capacity_.load(std::memory_order_acquire)) return;
        key_map_.find_each(key, std::forward<callback_type_>(callback));
    }

    /** @brief Whether at least one published node exists under @p key; @c false before the
     *      first @c reserve. */
    bool contains(key_t const& key) const noexcept {
        if (!capacity_.load(std::memory_order_acquire)) return false;
        return key_map_.contains(key);
    }

    /** @brief Number of published nodes under @p key; zero before the first @c reserve. */
    std::size_t count(key_t const& key) const noexcept {
        if (!capacity_.load(std::memory_order_acquire)) return 0;
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
     *      reciprocity, and reachability from stochastic seeds. Demands a mutation-free
     *      phase, proven by the settled-past guard re-checked on return - concurrent
     *      mutation yields @c contended_k, never a torn report.
     *
     *  @param[inout] context The calling thread's scratch; its visited set and staged
     *      ids grow to the full graph size for the duration of the walk. */
    graph_health_t health(traversal_context_t& context) const noexcept {
        graph_health_t report;
        node_id_t visible = 0;
        u64_t settled_census = 0;
        if (!settled_past_(visible, settled_census)) {
            report.status = status_t::contended_k;
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

        // Degrees & reciprocity in one edge sweep. Every row is copied under its own
        // mutex - one lock at a time, so the reciprocity hop cannot deadlock - and a
        // violated mutation-free contract degrades the stats, never the memory safety.
        report.degree_min = (std::numeric_limits<std::size_t>::max)();
        std::size_t total_neighbors = 0;
        for (std::size_t node = 0; node != node_count; ++node) {
            node_t const& record = nodes_array_[node];
            record.mutex.lock();
            std::size_t const degree = record.neighbors.size();
            bool const staged = succeeded(context.staged_ids.reserve(degree));
            context.staged_ids.clear();
            if (staged)
                for (std::size_t link = 0; link != degree; ++link)
                    context.staged_ids.push_back_reserved(record.neighbors[link]);
            record.mutex.unlock();
            if (!staged) {
                report.status = status_t::bad_alloc_k;
                return report;
            }
            report.degree_min = min_of(report.degree_min, degree);
            report.degree_max = max_of(report.degree_max, degree);
            total_neighbors += degree;
            for (std::size_t link = 0; link != degree; ++link) {
                // An out-of-range id is a violated mutation-free contract: report it as
                // contention & degrade the stats, never read past the record array.
                if (static_cast<std::size_t>(context.staged_ids[link]) >= node_count) {
                    report.status = status_t::contended_k;
                    continue;
                }
                node_t const& reverse_record = nodes_array_[context.staged_ids[link]];
                reverse_record.mutex.lock();
                for (std::size_t back = 0; back != reverse_record.neighbors.size(); ++back)
                    if (reverse_record.neighbors[back] == static_cast<node_id_t>(node)) {
                        ++report.reciprocal_edges;
                        break;
                    }
                reverse_record.mutex.unlock();
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
                node_t const& record = nodes_array_[node];
                record.mutex.lock(); // The only lock held: pushes below take no other.
                for (std::size_t link = 0; link != record.neighbors.size(); ++link) {
                    node_id_t const neighbor_id = record.neighbors[link];
                    // The same violated-contract net as the edge sweep above.
                    if (static_cast<std::size_t>(neighbor_id) >= node_count) {
                        report.status = status_t::contended_k;
                        continue;
                    }
                    if (context.visited.set(neighbor_id) == membership_t::fresh_k)
                        context.staged_ids.push_back_reserved(neighbor_id);
                }
                record.mutex.unlock();
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

        if (!still_settled_(visible, settled_census)) report.status = status_t::contended_k;
        return report;
    }

#pragma endregion Keys and Introspection

#pragma region Serialization

    /** @brief Bytes a @c save would emit right now - the raw snapshot adapters build
     *      interchange formats from. Demands the same mutation-free phase as @c save;
     *      concurrent mutation yields @c contended_k.
     *
     *  @param[in] codec Sizes each stored value's encoded form. */
    template <typename value_codec_type_>
        requires basic_value_codec<value_codec_type_, value_t>
    expected<std::size_t> snapshot_bytes(value_codec_type_ const& codec) const noexcept {
        basic_snapshot_header_t header;
        u64_t settled_census = 0;
        status_t const tallied = tally_(codec, header, settled_census);
        if (tallied != status_t::success_k) return {0, tallied};
        return {snapshot_layout_(header).total_bytes, status_t::success_k};
    }

    /** @brief Serializes the snapshot into @p output. Demands a mutation-free phase,
     *      proven by the settled-past guard re-checked after the copy - concurrent
     *      mutation yields @c contended_k, never a torn payload.
     *
     *  @param[out] output Caller-owned destination, at least @c snapshot_bytes large.
     *  @param[in] codec Encodes each stored value into its section slice. */
    template <typename value_codec_type_>
        requires basic_value_codec<value_codec_type_, value_t>
    status_t save(std::span<std::byte> output, value_codec_type_ const& codec) const noexcept {
        basic_snapshot_header_t header;
        u64_t settled_census = 0;
        status_t const tallied = tally_(codec, header, settled_census);
        if (tallied != status_t::success_k) return tallied;
        snapshot_layout_t const layout = snapshot_layout_(header);
        if (output.size() < layout.total_bytes) return status_t::overflow_k;
        std::size_t const node_count = static_cast<std::size_t>(header.node_count);

        std::size_t cursor = 0;
        write_bytes_(output, cursor, header);
        pad_until_(output, cursor, layout.keys_offset);
        for (std::size_t node = 0; node != node_count; ++node) write_bytes_(output, cursor, nodes_array_[node].key);

        pad_until_(output, cursor, layout.neighbor_sums_offset);
        // One pass writes each row's prefix sum & ids under one lock hold, so the two
        // sections always agree on the row they describe; the copy is capped by the
        // tallied extent, so a save racing a violated certification writes bounded
        // bytes - and the census re-check on return still reports the contention.
        std::size_t sums_cursor = layout.neighbor_sums_offset;
        std::size_t ids_cursor = layout.neighbor_ids_offset;
        i64_t running_neighbors = 0;
        write_bytes_(output, sums_cursor, running_neighbors);
        for (std::size_t node = 0; node != node_count; ++node) {
            node_t const& record = nodes_array_[node];
            record.mutex.lock();
            std::size_t const budget = static_cast<std::size_t>(header.neighbors_count) -
                                       static_cast<std::size_t>(running_neighbors);
            std::size_t const degree = min_of(record.neighbors.size(), budget);
            for (std::size_t link = 0; link != degree; ++link) write_bytes_(output, ids_cursor, record.neighbors[link]);
            record.mutex.unlock();
            running_neighbors += static_cast<i64_t>(degree);
            write_bytes_(output, sums_cursor, running_neighbors);
        }
        pad_until_(output, sums_cursor, layout.neighbor_ids_offset);
        cursor = ids_cursor;

        pad_until_(output, cursor, layout.value_sums_offset);
        i64_t running_values = 0;
        write_bytes_(output, cursor, running_values);
        for (std::size_t node = 0; node != node_count; ++node) {
            running_values += static_cast<i64_t>(codec.saved_bytes(nodes_array_[node].value));
            write_bytes_(output, cursor, running_values);
        }

        pad_until_(output, cursor, layout.values_offset);
        for (std::size_t node = 0; node != node_count; ++node) {
            std::size_t const encoded = codec.saved_bytes(nodes_array_[node].value);
            status_t const status = codec.save(nodes_array_[node].value, output.subspan(cursor, encoded));
            if (status != status_t::success_k) return status;
            cursor += encoded;
        }

        node_id_t const visible = static_cast<node_id_t>(header.node_count);
        if (!still_settled_(visible, settled_census)) return status_t::contended_k;
        return status_t::success_k;
    }

    /** @brief Loads a snapshot into this reserved, empty index - exclusive: decodes each
     *      value into its pre-constructed record, rebuilds the key multimap
     *      single-threaded, and publishes the watermark last. Rows are cleared on entry,
     *      so a failed attempt may be retried with a corrected payload.
     *
     *  @param[in] payload The bytes a prior @c save emitted.
     *  @param[in] codec Decodes each value slice into its record. */
    template <typename value_codec_type_>
        requires basic_value_codec<value_codec_type_, value_t>
    status_t load(std::span<std::byte const> payload, value_codec_type_ const& codec) noexcept {
        std::size_t const capacity = capacity_.load(std::memory_order_acquire);
        if (!capacity) return status_t::uninitialized_k;
        if (last_visible_id_.load(std::memory_order_acquire) != 0) return status_t::config_mismatch_k;
        if (payload.size() < sizeof(basic_snapshot_header_t)) return status_t::malformed_k;

        std::size_t cursor = 0;
        auto const header = read_bytes_<basic_snapshot_header_t>(payload, cursor);
        if (header.magic != basic_snapshot_header_t::magic_k || header.layout != basic_snapshot_header_t::layout_k ||
            header.version != basic_snapshot_header_t::version_k)
            return status_t::malformed_k;
        if (header.alignment < basic_snapshot_header_t::alignment_floor_k ||
            (header.alignment & (header.alignment - 1)) != 0)
            return status_t::malformed_k;
        if (header.node_id_bytes != sizeof(node_id_t) || header.key_bytes != sizeof(key_t))
            return status_t::config_mismatch_k;
        std::size_t const node_count = static_cast<std::size_t>(header.node_count);
        if (node_count > capacity) return status_t::overflow_k;
        // Bound the untrusted extents before any size arithmetic, so nothing can wrap; a
        // zero header connectivity means variable degrees, bounded by the node count.
        u64_t const degree_ceiling = header.connectivity ? header.connectivity : static_cast<u64_t>(node_count);
        if (degree_ceiling > static_cast<u64_t>(node_count) ||
            header.neighbors_count > static_cast<u64_t>(node_count) * degree_ceiling ||
            header.value_bytes > payload.size())
            return status_t::malformed_k;
        snapshot_layout_t const layout = snapshot_layout_(header);
        if (payload.size() < layout.total_bytes) return status_t::malformed_k;

        cursor = layout.keys_offset;
        for (std::size_t node = 0; node != node_count; ++node)
            nodes_array_[node].key = read_bytes_<key_t>(payload, cursor);

        std::size_t sums_cursor = layout.neighbor_sums_offset;
        std::size_t ids_cursor = layout.neighbor_ids_offset;
        i64_t previous_neighbors = read_bytes_<i64_t>(payload, sums_cursor);
        if (previous_neighbors != 0) return status_t::malformed_k;
        for (std::size_t node = 0; node != node_count; ++node) {
            i64_t const next_neighbors = read_bytes_<i64_t>(payload, sums_cursor);
            if (next_neighbors < previous_neighbors ||
                next_neighbors - previous_neighbors > static_cast<i64_t>(degree_ceiling))
                return status_t::malformed_k;
            neighbors_buffer_t& row = nodes_array_[node].neighbors;
            row.clear();
            if (status_t const grown = row.reserve(static_cast<std::size_t>(next_neighbors - previous_neighbors));
                failed(grown))
                return grown;
            for (i64_t link = previous_neighbors; link != next_neighbors; ++link) {
                node_id_t const neighbor_id = read_bytes_<node_id_t>(payload, ids_cursor);
                if (static_cast<std::size_t>(neighbor_id) >= node_count) return status_t::malformed_k;
                row.push_back_reserved(neighbor_id);
            }
            previous_neighbors = next_neighbors;
        }
        if (previous_neighbors != static_cast<i64_t>(header.neighbors_count)) return status_t::malformed_k;

        sums_cursor = layout.value_sums_offset;
        i64_t previous_values = read_bytes_<i64_t>(payload, sums_cursor);
        if (previous_values != 0) return status_t::malformed_k;
        for (std::size_t node = 0; node != node_count; ++node) {
            i64_t const next_values = read_bytes_<i64_t>(payload, sums_cursor);
            if (next_values < previous_values || next_values > static_cast<i64_t>(header.value_bytes))
                return status_t::malformed_k;
            std::span<std::byte const> const slice = payload.subspan(
                layout.values_offset + static_cast<std::size_t>(previous_values),
                static_cast<std::size_t>(next_values - previous_values));
            status_t const decoded = codec.load(slice, nodes_array_[node].value);
            if (decoded != status_t::success_k) return decoded;
            previous_values = next_values;
        }
        if (previous_values != static_cast<i64_t>(header.value_bytes)) return status_t::malformed_k;

        // A failed prior attempt burned key-map quota; wiping first keeps every entry
        // unique and the quota whole, so failed loads stay genuinely retryable.
        key_map_.clear();
        for (std::size_t node = 0; node != node_count; ++node) {
            status_t const keyed = key_map_.insert(nodes_array_[node].key, static_cast<node_id_t>(node));
            if (keyed != status_t::success_k) return keyed;
        }

        // The tier-wide publication order: claims, then identity, then the watermark last.
        claimed_count_.store(node_count, std::memory_order_relaxed);
        next_dispensed_id_.store(static_cast<node_id_t>(node_count), std::memory_order_relaxed);
        last_visible_id_.store(static_cast<node_id_t>(node_count), std::memory_order_release);
        return status_t::success_k;
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

    /** @brief Sizes every context container for one traversal, so the staging surfaces &
     *      selections never allocate mid-walk; the visited set and frontier may still
     *      grow past deep expansions, degrading the walk gracefully when they cannot. */
    [[nodiscard]] static status_t reserve_scratch_(traversal_context_t& context, std::size_t expansion_factor,
                                                   std::size_t fan_out, std::size_t connectivity) noexcept {
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
     *  @return The landmarks scored; @c 0 on @c success_k means the published set is
     *      exhausted or staging was refused, which is a reason to stop restarting rather
     *      than a failure. A failed status means scratch could not grow, which is. */
    template <typename metric_type_, typename query_type_>
    [[nodiscard]] expected<std::size_t> sample_landmarks_(query_type_ const& query, metric_type_& metric,
                                                          node_id_t first_invisible_id, std::size_t wanted,
                                                          std::size_t expansion_factor,
                                                          traversal_context_t& context) const noexcept {
        if (status_t const seen = context.visited.reserve(context.visited.size() + wanted); failed(seen))
            return {0, seen};
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
        if (!staged_count) return {0, status_t::success_k};
        drain_round(metric, staged_count, context);
        for (std::size_t lane = 0; lane != staged_count; ++lane) {
            scored_node_t const scored {context.staged_ids[lane], context.staged_distances[lane]};
            [[maybe_unused]] bool const ranked = context.results.insert(scored_node_t {scored}, expansion_factor);
            if (status_t const queued = context.frontier.insert(scored_node_t {scored}); failed(queued))
                return {lane, queued};
        }
        return {staged_count, status_t::success_k};
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
     *  The walk itself never fails hard: an exhausted published set or a refused staging
     *  ends it normally, and a scratch refusal ends it early but leaves whatever was
     *  already ranked intact. The status says which happened, so a read-only caller can
     *  report the refusal while an insertion caller - already holding claimed capacity -
     *  finishes regardless. Frontier starvation with an unfilled pool re-seeds and counts
     *  a restart. The frontier-admission rule: a candidate enters the frontier iff it
     *  improved the results pool, or the frontier would otherwise be empty - the seedings.
     */
    template <synchronization_t synchronization_, typename metric_type_, typename query_type_>
    [[nodiscard]] status_t search_around_(query_type_ const& query, metric_type_& metric, node_id_t first_invisible_id,
                                          std::size_t expansion_factor, std::size_t fan_out, std::size_t connectivity,
                                          traversal_context_t& context) const noexcept {
        expected<std::size_t> seeded = sample_landmarks_(query, metric, first_invisible_id, connectivity,
                                                         expansion_factor, context);
        if (failed(seeded.status)) return seeded.status;
        if (!seeded.value) return status_t::success_k; // Nothing published to walk from.

        while (true) {
            if (context.frontier.empty()) {
                if (context.results.size() >= expansion_factor) return status_t::success_k;
                seeded = sample_landmarks_(query, metric, first_invisible_id, connectivity, expansion_factor, context);
                if (failed(seeded.status)) return seeded.status;
                if (!seeded.value) return status_t::success_k;
                ++context.telemetry.restarts;
                continue;
            }

            basic_expansion_round_t const round = expand_frontier_<synchronization_>(query, metric, first_invisible_id,
                                                                                     expansion_factor, fan_out, //
                                                                                     context);
            if (round == basic_expansion_round_t::converged_k) return status_t::success_k;

            std::size_t const staged_count = context.staged_ids.size();
            if (staged_count) {
                drain_round(metric, staged_count, context);
                for (std::size_t lane = 0; lane != staged_count; ++lane) {
                    scored_node_t const scored {context.staged_ids[lane], context.staged_distances[lane]};
                    if (context.results.insert(scored_node_t {scored}, expansion_factor))
                        if (status_t const queued = context.frontier.insert(scored_node_t {scored}); failed(queued))
                            return queued;
                }
            }
            if (round == basic_expansion_round_t::saturated_k) return status_t::success_k;
        }
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

    /** @brief Opens the settled-past guard: the watermark loads before the dispenser, so
     *      their equality - with a zero in-flight half in the census word - proves no
     *      publication, back-link, or manual wiring is live. */
    bool settled_past_(node_id_t& visible, u64_t& census) const noexcept {
        visible = last_visible_id_.load(std::memory_order_acquire);
        node_id_t const dispensed = next_dispensed_id_.load(std::memory_order_acquire);
        census = mutation_census_.load(std::memory_order_acquire);
        return visible == dispensed && (census & (mutation_started_k - 1)) == 0;
    }

    /** @brief Re-checks the guard after a copy: any movement - including a mutation that
     *      both started and finished inside the bracket, which only the census word's
     *      monotone started-half can reveal - means contention. */
    bool still_settled_(node_id_t visible, u64_t census) const noexcept {
        return last_visible_id_.load(std::memory_order_acquire) == visible &&
               next_dispensed_id_.load(std::memory_order_acquire) == visible &&
               mutation_census_.load(std::memory_order_acquire) == census;
    }

    /** @brief Walks the settled graph once, filling the header's extents; @c contended_k
     *      when the settled-past guard refuses. */
    template <typename value_codec_type_>
    status_t tally_(value_codec_type_ const& codec, basic_snapshot_header_t& header,
                    u64_t& settled_census) const noexcept {
        node_id_t visible = 0;
        if (!settled_past_(visible, settled_census)) return status_t::contended_k;
        header.node_id_bytes = sizeof(node_id_t);
        header.key_bytes = sizeof(key_t);
        header.node_count = static_cast<u64_t>(visible);
        std::size_t degree_min = (std::numeric_limits<std::size_t>::max)();
        std::size_t degree_max = 0;
        for (std::size_t node = 0; node != static_cast<std::size_t>(visible); ++node) {
            node_t const& record = nodes_array_[node];
            record.mutex.lock(); // Sizes race manual wiring; the value itself is immutable.
            std::size_t const degree = record.neighbors.size();
            record.mutex.unlock();
            degree_min = min_of(degree_min, degree);
            degree_max = max_of(degree_max, degree);
            header.neighbors_count += degree;
            header.value_bytes += codec.saved_bytes(record.value);
        }
        // The constant degree when there is one; zero - the interchange convention - when
        // degrees vary or nothing is published.
        header.connectivity = visible && degree_min == degree_max ? static_cast<u32_t>(degree_max) : 0u;
        return status_t::success_k;
    }

    /** @brief Byte offsets of the five aligned snapshot sections. @ref basic_snapshot_header_t */
    struct snapshot_layout_t {
        std::size_t keys_offset = 0;
        std::size_t neighbor_sums_offset = 0;
        std::size_t neighbor_ids_offset = 0;
        std::size_t value_sums_offset = 0;
        std::size_t values_offset = 0;
        std::size_t total_bytes = 0;
    };

    /** @brief Lays the sections out on the header's alignment; callers bound the header's
     *      extents first, so none of the arithmetic can wrap. */
    static snapshot_layout_t snapshot_layout_(basic_snapshot_header_t const& header) noexcept {
        std::size_t const alignment = header.alignment;
        std::size_t const node_count = static_cast<std::size_t>(header.node_count);
        snapshot_layout_t layout;
        layout.keys_offset = divide_round_up<std::size_t>(sizeof(basic_snapshot_header_t), alignment) * alignment;
        layout.neighbor_sums_offset =
            divide_round_up<std::size_t>(layout.keys_offset + node_count * sizeof(key_t), alignment) * alignment;
        layout.neighbor_ids_offset = divide_round_up<std::size_t>(
                                         layout.neighbor_sums_offset + (node_count + 1) * sizeof(i64_t), alignment) *
                                     alignment;
        layout.value_sums_offset = divide_round_up<std::size_t>(
                                       layout.neighbor_ids_offset +
                                           static_cast<std::size_t>(header.neighbors_count) * sizeof(node_id_t),
                                       alignment) *
                                   alignment;
        layout.values_offset = divide_round_up<std::size_t>(layout.value_sums_offset + (node_count + 1) * sizeof(i64_t),
                                                            alignment) *
                               alignment;
        layout.total_bytes = layout.values_offset + static_cast<std::size_t>(header.value_bytes);
        return layout;
    }

    /** @brief Zero-fills @p output up to @p target, so section pads are deterministic. */
    static void pad_until_(std::span<std::byte> output, std::size_t& cursor, std::size_t target) noexcept {
        for (; cursor != target; ++cursor) output[cursor] = std::byte {};
    }

    /** @brief Appends @p object's bytes at @p cursor - unaligned-safe - and advances it. */
    template <typename object_type_>
    static void write_bytes_(std::span<std::byte> output, std::size_t& cursor, object_type_ const& object) noexcept {
        auto const bytes = std::bit_cast<std::array<std::byte, sizeof(object_type_)>>(object);
        for (std::size_t position = 0; position != bytes.size(); ++position)
            output[cursor + position] = bytes[position];
        cursor += bytes.size();
    }

    /** @brief Reads one @p object_type_ at @p cursor - unaligned-safe - and advances it. */
    template <typename object_type_>
    static object_type_ read_bytes_(std::span<std::byte const> input, std::size_t& cursor) noexcept {
        std::array<std::byte, sizeof(object_type_)> bytes;
        for (std::size_t position = 0; position != bytes.size(); ++position) bytes[position] = input[cursor + position];
        cursor += bytes.size();
        return std::bit_cast<object_type_>(bytes);
    }

#pragma endregion Internals

#pragma region State

    /** @brief The node records, sized by @c reserve, addresses stable forever. */
    node_array_t nodes_array_;
    /** @brief Frozen entry capacity; its release store is @c reserve's last act, doubling
     *      as the publication edge for the record array & key multimap. Public entry
     *      points take one acquire load into a local, so no loop reads it. */
    std::atomic<std::size_t> capacity_ {0};

    /** @brief Batch capacity claims - the all-or-nothing admission, separate from
     *      identity: claims convert one-to-one into dispensed ids, so dispensing never
     *      overflows. Rejected batches roll their claim back, so the count transiently
     *      overshoots under contention - kept a full word wide, so no overshoot can ever
     *      wrap it around the bounds check. */
    std::atomic<std::size_t> claimed_count_ {0};
    /** @brief Node-id dispenser; ids are claimed after the walk, right before publication. */
    std::atomic<node_id_t> next_dispensed_id_ {0};
    /** @brief The visibility watermark: every id strictly below it is published. */
    std::atomic<node_id_t> last_visible_id_ {0};
    /** @brief The mutation census, two halves in one word so a single load snapshots
     *      both: the high half counts mutations ever started - monotone, so a guard can
     *      prove nothing started inside its bracket - and the low half counts mutations
     *      in flight. Two separate atomics could not serve: no probe order gives a
     *      consistent joint read, so a wiring that starts & finishes between two probes
     *      would slip through. Packed by necessity, not optimization. */
    std::atomic<u64_t> mutation_census_ {0};
    /** @brief One unit of the census word's started-half. The split is derived, not
     *      picked: the in-flight half must hold every simultaneously-live mutator, and
     *      the OS pid ceiling near 2^22 rounds up to the even 32-bit split; the started
     *      half then wraps only if 2^32 mutations begin inside one guard bracket and land
     *      on an exact multiple - saturated wiring for tens of seconds inside a
     *      microseconds-long copy. */
    inline static constexpr u64_t mutation_started_k = u64_t {1} << 32;

    /** @brief Key → ID multimap, sized by @c reserve. */
    key_map_t key_map_;

    [[no_unique_address]] allocator_t allocator_;

#pragma endregion State
};

} // namespace usearch
} // namespace unum
