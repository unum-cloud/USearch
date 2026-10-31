/**
 *  @file include/usearch/utilities/ram_tables.hpp
 *  @author Ash Vardanian
 *  @date August 18, 2026
 *  @brief Associative open-addressing hash tables for Graph traversals & key lookups.
 *
 *  Standalone header providing baseline associative structures, like:
 *
 *  - Exact per-thread visited @b hash-sets
 *  - Insert-only @b hash-multimaps of nodes under keys, growing in chunks, read alongside their
 *    writer
 *  - Keyed nodes: every node's key beside every key's nodes
 */
#pragma once

#include <cassert> // `assert` - every `insert` follows a `reserve`, and never inserts the sentinel
#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint64_t` - batched membership masks
#include <cstring> // `std::memset` - the all-ones sentinel fill

#include <array>       // `std::array` - batched membership inserts
#include <bit>         // `std::countr_zero` - the first matching lane of a group scan
#include <functional>  // `std::hash`
#include <limits>      // `std::numeric_limits` - the all-ones sentinel
#include <memory>      // `std::allocator`
#include <type_traits> // `std::is_integral`, `std::is_trivially_copyable_v`, `std::is_unsigned_v`
#include <utility>     // `std::swap`, `std::move`

#include "usearch/utilities/ram_buffers.hpp" // `exchange`, `ceil2`
#include "usearch/utilities/types.hpp"       // `status_t`, `expected`, `USEARCH_ARCH_X8664_`

namespace unum::usearch::utilities {

/**
 *  @brief Returns the sentinel that marks an unused slot in
 *      @ref unum::usearch::utilities::growing_hash_set.
 *
 *  For integral types this is the all-ones bit pattern, which matches the
 *  @c 0xFF byte-fill emitted by @c growing_hash_set::clear.
 */
template <typename element_type_>
inline element_type_ unused_hash_slot_value() noexcept {
    static_assert(std::is_integral<element_type_>::value,
                  "`growing_hash_set` requires an integral element type for its sentinel convention");
    return (std::numeric_limits<element_type_>::max)();
}

/** Whether a membership structure already held the element an insert offered it -
 *  an answer, never a failure, so it never travels as a @c status_t. */
enum class membership_t : bool {

    /** The element was absent, and this insert is the one that added it. */
    absent_k = false,

    /** The element was already there, and the insert changed nothing. */
    present_k = true,
};

/** The slot walk every set takes at the portable floor: one slot per step, any element
 *  width. Returns the slot holding @p element, or the empty slot where it belongs; @c contains,
 *  @c insert and the rehash all place and find through this one walk, which is what keeps
 *  placement and lookup coherent. */
struct serial_visits_hash_set_kit_t {
    template <typename element_type_, typename hasher_type_>
    static inline element_type_* locate_slot(element_type_* slots, std::size_t capacity, hasher_type_ const& hasher,
                                             element_type_ element) noexcept {
        std::size_t index = hasher(element) & (capacity - 1);
        while (slots[index] != unused_hash_slot_value<element_type_>() && slots[index] != element)
            index = (index + 1) & (capacity - 1);
        return slots + index;
    }
};

/**
 *  @brief Minimalistic open-addressing hash-set, used to track visited nodes during graph
 *      traversal - a sparse alternative to a bit-set in our primary use case.
 *
 *  It doesn't support deleting individual elements, but supports @c clear -ing all at once.
 *  Every @c insert follows a @c reserve for the same count, so no mid-insert resize is ever
 *  needed and a full linear probe never runs: a refused @c reserve leaves the table as it
 *  was, and the caller stops marking. All-ones slots are unused, to simplify the design; the
 *  slot count stays a power of two, and bucket collisions resolve by the walk its kit spells -
 *  the serial one here, a wider group scan where a dense class carries it.
 */
template <                                             //
    typename element_type_,                            //
    typename hasher_type_ = std::hash<element_type_>,  //
    typename allocator_type_ = std::allocator<byte_t>, //
    typename hash_set_kit_type_ = serial_visits_hash_set_kit_t>
class growing_hash_set {

    using element_t = element_type_;
    using hasher_t = hasher_type_;
    using hash_set_kit_t = hash_set_kit_type_;

    using allocator_t = allocator_type_;
    using byte_t = typename allocator_t::value_type;
    static_assert(sizeof(byte_t) == 1, "Allocator must allocate separate addressable bytes");

    element_t* slots_ {};

    /** Number of slots. */
    std::size_t capacity_ {};

    /** Number of populated slots. */
    std::size_t count_ {};
    hasher_t hasher_ {};
    allocator_t allocator_ {};

  public:

    /** Every set names its allocator; nothing is placed by default. */
    growing_hash_set() = delete;

    /** The process-wide allocator, for a set no arena or domain owns. */
    explicit growing_hash_set(global_allocator_t) noexcept
        requires std::default_initializable<allocator_t>
        : growing_hash_set(allocator_t {}) {}
    explicit growing_hash_set(allocator_t allocator) noexcept : allocator_(std::move(allocator)) {}
    ~growing_hash_set() noexcept { reset(); }

    explicit operator bool() const noexcept { return slots_; }
    std::size_t size() const noexcept { return count_; }

    /** Empties the set; one already empty, as @c reserve leaves a fresh table, is left as it is. */
    void clear() noexcept {
        if (count_) std::memset((void*)slots_, 0xFF, capacity_ * sizeof(element_t));
        count_ = 0;
    }

    void reset() noexcept {
        if (slots_) allocator_.deallocate((byte_t*)slots_, capacity_ * sizeof(element_t));
        slots_ = nullptr;
        capacity_ = 0;
        count_ = 0;
    }

    /** Slots @p capacity elements probe over; sized to the bare count a table fills, and a
     *  full linear probe never terminates. */
    static std::size_t slots_for(std::size_t capacity) noexcept { return ceil2(capacity * 5u / 3u); }

    /** Bytes @p capacity elements occupy, the load headroom included. */
    static std::size_t memory_usage(std::size_t capacity) noexcept { return slots_for(capacity) * sizeof(element_t); }

    /** Bytes this set currently occupies. */
    std::size_t memory_usage() const noexcept { return capacity_ * sizeof(element_t); }

    growing_hash_set(growing_hash_set&& other) noexcept {
        slots_ = exchange(other.slots_, nullptr);
        capacity_ = exchange(other.capacity_, 0);
        count_ = exchange(other.count_, 0);
        allocator_ = std::move(other.allocator_);
    }

    growing_hash_set& operator=(growing_hash_set&& other) noexcept {
        std::swap(slots_, other.slots_);
        std::swap(capacity_, other.capacity_);
        std::swap(count_, other.count_);
        std::swap(allocator_, other.allocator_);
        return *this;
    }

    growing_hash_set(growing_hash_set const&) = delete;
    growing_hash_set& operator=(growing_hash_set const&) = delete;

    /**
     *  @brief Checks if the element is already in the hash-set.
     *  @return @c true if the element is present.
     */
    inline bool contains(element_t const& element) const noexcept {
        return *hash_set_kit_t::locate_slot(slots_, capacity_, hasher_, element) == element;
    }

    /**
     *  @brief Inserts an element into the hash-set.
     *  @return The membership the element had @b before this call.
     */
    [[nodiscard]] inline membership_t insert(element_t const& element) noexcept {
        assert(slots_ && (count_ + 1) * 5u / 3u <= capacity_);
        assert(element != unused_hash_slot_value<element_t>() &&
               "The sentinel marks empty slots, so it can never be a member");
        element_t* slot = hash_set_kit_t::locate_slot(slots_, capacity_, hasher_, element);
        if (*slot == element) return membership_t::present_k;
        *slot = element;
        ++count_;
        return membership_t::absent_k;
    }

    /**
     *  @brief Inserts up to @c batch_size_ elements sequentially.
     *  @return 64-bit mask - bit @c i is set iff @c batch[i] was newly inserted, i.e. iff the
     *      single-element @ref insert would have answered @c absent_k.
     *
     *  Scalar baseline just calls @ref insert in a loop; the measured win over single-element
     *  inserts (~1.6× on the microbench) comes from the compiler interleaving independent
     *  load-use chains through two load ports. Future SIMD variants can overlap wider
     *  bucket compares without changing the call shape.
     */
    template <std::size_t batch_size_>
    inline u64_t insert(std::array<element_t, batch_size_> const& batch) noexcept {
        static_assert(batch_size_ > 0 && batch_size_ <= 64, "Batch size must fit the u64 return mask");
        u64_t out = 0;
        for (std::size_t i = 0; i < batch_size_; ++i)
            if (insert(batch[i]) == membership_t::absent_k) out |= u64_t {1} << i;
        return out;
    }

    /**
     *  @brief Extends the capacity of the hash-set, rehashing every present element.
     */
    status_t reserve(std::size_t new_capacity) noexcept {
        new_capacity = (new_capacity * 5u) / 3u;
        if (new_capacity <= capacity_) return status_t::success_k;

        new_capacity = ceil2(new_capacity);
        element_t* new_slots = (element_t*)allocator_.allocate(new_capacity * sizeof(element_t));
        if (!new_slots) return status_t::bad_alloc_k;

        std::memset((void*)new_slots, 0xFF, new_capacity * sizeof(element_t));
        std::size_t new_count = count_;
        // Re-place through the very same probe walk `contains` & `insert` use - the target slot
        // of a not-yet-present element is exactly where the kit's `locate_slot` points.
        if (count_)
            for (std::size_t old_index = 0; old_index != capacity_; ++old_index) {
                if (slots_[old_index] == unused_hash_slot_value<element_t>()) continue;
                *hash_set_kit_t::locate_slot(new_slots, new_capacity, hasher_, slots_[old_index]) = slots_[old_index];
            }

        reset();
        slots_ = new_slots;
        capacity_ = new_capacity;
        count_ = new_count;
        return status_t::success_k;
    }
};

/**
 *  @brief Insert-only, linear-probe hash multimap of nodes under their keys, growing without moving
 *      an entry: its chunks are fixed tables appended on a doubling @ref chunk_schedule, so inserts
 *      and lookups run while it grows.
 *
 *  Node @c n goes in the chunk the schedule locates it in, whose slots are twice its nodes, so
 *  every probe chain terminates. Each chunk has one writer at a time, the one filing the nodes it
 *  holds, so an insert claims nothing: it finds an empty slot, writes the key and node, then
 *  releases the slot's probe tag. Duplicate keys coexist as separate slots. The chunks are two
 *  @ref chunked_column s - control bytes and slots - that @ref reserve covers.
 *
 *  Slot states live in the control-byte column the probes scan - 64 states per cache line - so a
 *  slot's key & node are touched only on a published slot whose tag matches, and no key or node
 *  bit-pattern is ever reserved. Readers are wait-free: each passes the watermark it acquired,
 *  probes the published chunks holding nodes below it with acquire loads, stops at @c empty_k, and
 *  skips every node at or above the watermark.
 */
template <typename key_type_, typename node_id_type_,                           //
          typename hash_type_ = std::hash<key_type_>,                           //
          typename allocator_type_ = std::allocator<byte_t>,                    //
          template <typename> class atomic_ref_template_ = standard_atomic_ref> //
class chunked_multimap {
    static_assert(std::is_trivially_copyable_v<key_type_> && std::is_unsigned_v<node_id_type_>,
                  "A node is placed by its ID and ordered against a watermark");

  public:
    using key_t = key_type_;
    using node_id_t = node_id_type_;
    using hash_t = hash_type_;
    using allocator_t = allocator_type_;
    template <typename word_type_>
    using atomic_ref = atomic_ref_template_<word_type_>;

    /** Nodes per chunk, doubling from 1'024; the slab phase lies past any table a machine holds. */
    using nodes_schedule_t = chunk_schedule<10, 40>;

    /** Slots per chunk, twice its nodes. */
    using slots_schedule_t = chunk_schedule<11, 41>;

    /** A slot never filled, the probe chain terminator and what a fresh chunk starts as; a filled
     *  slot holds its key's @c probe_tag_of instead, and its key & node are immutable from then
     *  on. */
    enum class slot_state_t : u8_t { empty_k = 0 };

    chunked_multimap(chunked_multimap const&) noexcept = delete;
    chunked_multimap& operator=(chunked_multimap const&) noexcept = delete;

    /** Every map names its allocator; nothing is placed by default. */
    chunked_multimap() = delete;

    /** The process-wide allocator, for a map no arena or domain owns. */
    explicit chunked_multimap(global_allocator_t) noexcept
        requires std::default_initializable<allocator_t>
        : chunked_multimap(allocator_t {}) {}
    explicit chunked_multimap(allocator_t const& allocator) noexcept : states_(allocator), slots_(allocator) {}

    chunked_multimap(chunked_multimap&& other) noexcept
        : states_(std::move(other.states_)), slots_(std::move(other.slots_)) {}

    /** Swap-based, so the displaced chunks die with @p other rather than here. Single-threaded, like @ref clear. */
    chunked_multimap& operator=(chunked_multimap&& other) noexcept {
        states_ = std::move(other.states_);
        slots_ = std::move(other.slots_);
        return *this;
    }

    /** Forgets every entry, keeping the chunks - no writer may run alongside, and a reader running
     *  alongside must discard what it found, as one bracketed by an epoch the caller moves does. */
    void clear() noexcept {
        for (std::size_t index = 0; index != max_chunks_k; ++index) {
            std::span<slot_state_t> const states = states_.chunk(index);
            if (states.empty()) break;
            std::memset(states.data(), 0, states.size_bytes());
        }
    }

    /** Frees every chunk - single-threaded, like @ref clear. */
    void reset() noexcept {
        states_.reset();
        slots_.reset();
    }

    /** Bytes the published chunks span. */
    std::size_t memory_usage() const noexcept { return states_.memory_usage() + slots_.memory_usage(); }

    /**
     *  Publishes the chunks holding the first @p nodes nodes - callable any time, alongside inserts
     *  and lookups.
     *
     *  @par Statuses:
     *  - @c overflow_k @p nodes passes the nodes the chunks can ever hold.
     */
    status_t reserve(std::size_t nodes) noexcept {
        if (nodes > max_slots_k / 2) return status_t::overflow_k;
        // The control bytes first: a reader acquires the slots before the states, and stops at the
        // first missing.
        if (status_t const covered = states_.cover(nodes * 2); failed(covered)) return covered;
        return slots_.cover(nodes * 2);
    }

    /** Files @p node under @p key in the chunk holding @p node, which a @ref reserve covered; the
     *  caller is that chunk's one writer, so nothing is claimed. */
    void insert(key_t const& key, node_id_t node) noexcept {
        std::size_t const index = nodes_schedule_t::locate(node).chunk;
        std::span<slot_t> const slots = slots_.chunk(index);
        std::span<slot_state_t> const states = states_.chunk(index);
        assert(!slots.empty() && "Every insert follows a reserve covering its node");
        u64_t const hash = mixed_hash_of(key);
        std::size_t position = first_probe_of(hash, states.size());
        while (atomic_ref<slot_state_t>(states[position]).load(std::memory_order_relaxed) != slot_state_t::empty_k)
            position = (position + 1) & (states.size() - 1);
        slots[position] = {key, node};
        atomic_ref<slot_state_t>(states[position]).store(probe_tag_of(hash), std::memory_order_release);
    }

    /** Invokes @p callback for every node below @p visible filed under @p key - wait-free. */
    template <typename callback_type_>
    void find_each(key_t const& key, node_id_t visible, callback_type_&& callback) const noexcept {
        any_of_(key, visible, [&](node_id_t node) noexcept {
            callback(node);
            return false;
        });
    }

    /** Nodes below @p visible filed under @p key. */
    std::size_t count(key_t const& key, node_id_t visible) const noexcept {
        std::size_t matches = 0;
        find_each(key, visible, [&](node_id_t) noexcept { ++matches; });
        return matches;
    }

    /** Whether any node below @p visible is filed under @p key, stopping at the first. */
    bool contains(key_t const& key, node_id_t visible) const noexcept {
        return any_of_(key, visible, [](node_id_t) noexcept { return true; });
    }

    /** Nodes the published chunks hold. */
    std::size_t capacity() const noexcept { return slots_.capacity() / 2; }

  private:

    /** One probe slot: the key beside its node; its lifecycle lives in the control-byte column, so
     *  no payload bit-pattern is ever reserved. */
    struct slot_t {
        key_t key;
        node_id_t node;
    };

    /** Slots the columns hold - every chunk of the doubling phase, past any table a machine holds,
     *  capped at what @c std::size_t addresses. */
    inline static constexpr std::size_t max_slots_k = static_cast<std::size_t>(
        min_of<u64_t>(slots_schedule_t::doubling_nodes_k, std::numeric_limits<std::size_t>::max()));
    inline static constexpr std::size_t max_chunks_k = slots_schedule_t::chunks_for(max_slots_k);

    /** Whether any node below @p visible filed under @p key satisfies @p accepts, stopping at the
     *  first; only the chunks holding such nodes are probed, and a chunk whose slots are not
     *  published yet ends the probe. */
    template <typename predicate_type_>
    bool any_of_(key_t const& key, node_id_t visible, predicate_type_&& accepts) const noexcept {
        u64_t const hash = mixed_hash_of(key);
        slot_state_t const tag = probe_tag_of(hash);
        for (std::size_t index = 0, chunks = nodes_schedule_t::chunks_for(visible); index != chunks; ++index) {
            std::span<slot_t const> const slots = slots_.chunk(index);
            if (slots.empty()) return false;
            std::span<slot_state_t const> const states = states_.chunk(index);
            for (std::size_t position = first_probe_of(hash, states.size());;
                 position = (position + 1) & (states.size() - 1)) {
                slot_state_t const state =
                    atomic_ref<slot_state_t const>(states[position]).load(std::memory_order_acquire);
                if (state == slot_state_t::empty_k) break;
                if (state == tag && slots[position].key == key && slots[position].node < visible &&
                    accepts(slots[position].node))
                    return true;
            }
        }
        return false;
    }

    /** Avalanches the hash through @ref mix_bits, so the identity hashes of sequential integer
     *  keys still scatter across the table: its low bits pick the first probe, its top seven the
     *  probe tag. */
    u64_t mixed_hash_of(key_t const& key) const noexcept { return mix_bits(static_cast<u64_t>(hash_(key))); }

    /** The first slot the chain of @p mixed_hash probes in a chunk of @p slots_count. */
    static std::size_t first_probe_of(u64_t mixed_hash, std::size_t slots_count) noexcept {
        return static_cast<std::size_t>(mixed_hash) & (slots_count - 1);
    }

    /** The state a slot of @p mixed_hash publishes: the top bit, above @c empty_k, then the hash's
     *  top seven bits, so a probe skips most other keys without reading their slots. */
    static slot_state_t probe_tag_of(u64_t mixed_hash) noexcept {
        return static_cast<slot_state_t>(0x80u | (mixed_hash >> 57));
    }

    chunked_column<slot_state_t, slots_schedule_t, max_slots_k, allocator_t, atomic_ref_template_> states_;
    chunked_column<slot_t, slots_schedule_t, max_slots_k, allocator_t, atomic_ref_template_> slots_;

    [[no_unique_address]] hash_t hash_;
};

/**
 *  @brief The key of every node, and the nodes under every key: a column addressed by node ID
 *      beside a @ref chunked_multimap placed by node ID, so neither ever claims a slot.
 *
 *  A node's claimer @ref assign s its key; the one writer of a published range @ref publish es it
 *  into the map before moving the watermark. Readers pass the watermark they acquired as @c visible
 *  and never see a node at or above it, so a key is found exactly when its node is.
 */
template <typename key_type_, typename node_id_type_, typename schedule_type_, std::size_t max_nodes_,
          typename allocator_type_, template <typename> class atomic_ref_template_,
          typename key_hash_type_ = std::hash<key_type_>>
class keyed_nodes {
  public:
    using key_t = key_type_;
    using node_id_t = node_id_type_;
    using allocator_t = allocator_type_;
    using keys_column_t = chunked_column<key_t, schedule_type_, max_nodes_, allocator_t, atomic_ref_template_>;
    using nodes_map_t = chunked_multimap<key_t, node_id_t, key_hash_type_, allocator_t, atomic_ref_template_>;

    /** Every table names its allocator; nothing is placed by default. */
    keyed_nodes() = delete;

    /** The process-wide allocator, for a table no arena or domain owns. */
    explicit keyed_nodes(global_allocator_t) noexcept
        requires std::default_initializable<allocator_t>
        : keyed_nodes(allocator_t {}) {}
    explicit keyed_nodes(allocator_t const& allocator) noexcept : keys_(allocator), nodes_(allocator) {}

    /**
     *  Covers the column and the map for the first @p nodes nodes - safe on any thread, beside
     *  every other column the grower covers.
     *
     *  @par Statuses:
     *  - @c overflow_k @p nodes passes @c max_nodes_.
     */
    status_t cover(std::size_t nodes) noexcept {
        if (status_t const covered = keys_.cover(nodes); failed(covered)) return covered;
        return nodes_.reserve(nodes);
    }

    /** Stores the key of @p node, which only its claimer writes; a cover already reached it. */
    void assign(node_id_t node, key_t const& key) noexcept { keys_[node] = key; }

    /** Files every node of @p nodes under its key - their one writer, before their watermark
     *  moves. */
    void publish(iota_view<node_id_t> nodes) noexcept {
        for (node_id_t const node : nodes) nodes_.insert(keys_[node], node);
    }

    /** Invokes @p callback for every node below @p visible stored under @p key - wait-free. */
    template <typename callback_type_>
    void find_each(key_t const& key, node_id_t visible, callback_type_&& callback) const noexcept {
        nodes_.find_each(key, visible, callback);
    }

    /** Nodes below @p visible stored under @p key. */
    std::size_t count(key_t const& key, node_id_t visible) const noexcept { return nodes_.count(key, visible); }

    /** Whether any node below @p visible is stored under @p key, stopping at the first. */
    bool contains(key_t const& key, node_id_t visible) const noexcept { return nodes_.contains(key, visible); }

    /** Empties the map, keeping its chunks and the column - only inside the owner's window, once
     *  the dispenser closed, so the next @ref publish starts from node zero. */
    void forget() noexcept { nodes_.clear(); }

    /** The key of @p node, which a cover already reached. */
    key_t const& operator[](node_id_t node) const noexcept { return keys_[node]; }

    /** The node-to-key column, for bulk copies out. */
    keys_column_t const& column() const noexcept { return keys_; }

    /** Bytes the published chunks of the column and the map span. */
    std::size_t memory_usage() const noexcept { return keys_.memory_usage() + nodes_.memory_usage(); }

  private:
    keys_column_t keys_;
    nodes_map_t nodes_;
};

} // namespace unum::usearch::utilities
