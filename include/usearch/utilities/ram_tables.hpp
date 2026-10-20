/**
 *  @file include/usearch/utilities/ram_tables.hpp
 *  @author Ash Vardanian
 *  @date August 18, 2026
 *  @brief Associative open-addressing hash tables for Graph traversals & key lookups.
 *
 *  Standalone header providing baseline associative structures, like:
 *
 *  - Exact per-thread visited @b hash-sets
 *  - Concurrent insert-only @b hash-multimaps, growing in chunks
 */
#pragma once

#include <cassert> // `assert` - every `insert` follows a `reserve`, and never inserts the sentinel
#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint64_t` - batched membership masks
#include <cstring> // `std::memset` - the all-ones sentinel fill

#include <array>       // `std::array` - batched membership inserts
#include <atomic>      // `std::memory_order` - the slot states and the claimed seats
#include <bit>         // `std::countr_zero` - the first matching lane of a group scan
#include <functional>  // `std::hash`
#include <limits>      // `std::numeric_limits` - the all-ones sentinel
#include <memory>      // `std::allocator`
#include <type_traits> // `std::is_integral`, `std::is_trivially_copyable_v`
#include <utility>     // `std::swap`, `std::move`

#include "usearch/utilities/ram_buffers.hpp" // `exchange`, `ceil2`
#include "usearch/utilities/types.hpp"       // `status_t`, `expected`, `USEARCH_ARCH_X8664_`

namespace unum::usearch::utilities {

/**
 *  @brief Returns the sentinel that marks an unused slot in @ref growing_hash_set.
 *
 *  For integral types this is the all-ones bit pattern, which matches the
 *  @c 0xFF byte-fill emitted by @ref growing_hash_set::clear.
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
     *  @return @c success_k when the capacity is available, @c bad_alloc_k otherwise.
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
        // of a not-yet-present element is exactly where the kit's `locate_slot` lands.
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
 *  @brief Insert-only, linear-probe hash multimap that grows without moving an entry: its chunks are fixed tables
 *      appended on a doubling @ref chunk_schedule, so inserts and lookups run while it grows.
 *
 *  Duplicate keys coexist as separate slots, which is what makes concurrent insertion CAS-once instead of
 *  lock-and-chain: a slot's state advances @c empty_k → @c building_k → @c published_k exactly once.
 *
 *  Every entry takes one seat, a global index a @ref claim hands out and covers before the entry exists, so
 *  recording it never fails; seat @c s lands in the chunk the schedule locates it in, whose slots are twice its
 *  seats, so every probe chain terminates. The chunks are two @ref chunked_column s - control bytes and slots - that
 *  cover publishes like any other column, so the claimed seats are the one word the map owns.
 *
 *  Slot states live in the control-byte column the probes scan - 64 states per cache line - so a slot's key & value
 *  are touched only on a published candidate, and no key or value bit-pattern is ever reserved. Readers are
 *  wait-free: they probe every published chunk with acquire loads, stop at @c empty_k, and treat @c building_k as
 *  occupied-but-mismatched - never as end-of-chain - so a half-published entry is merely invisible, never wrong.
 */
template <typename key_type_, typename value_type_,                             //
          typename hash_type_ = std::hash<key_type_>,                           //
          typename allocator_type_ = std::allocator<byte_t>,                    //
          template <typename> class atomic_ref_template_ = standard_atomic_ref> //
class chunked_multimap {
    static_assert(std::is_trivially_copyable_v<key_type_> && std::is_trivially_copyable_v<value_type_>);

  public:
    using key_t = key_type_;
    using value_t = value_type_;
    using hash_t = hash_type_;
    using allocator_t = allocator_type_;
    template <typename word_type_>
    using atomic_ref = atomic_ref_template_<word_type_>;

    /** Seats per chunk, doubling from 1'024; the slab phase lies past any table a machine holds. */
    using seats_schedule_t = chunk_schedule<10, 40>;

    /** Slots per chunk, twice its seats. */
    using slots_schedule_t = chunk_schedule<11, 41>;

    /** Lifecycle of one slot, advanced monotonically by its one claiming inserter. */
    enum class slot_state_t : u8_t {

        /** Never claimed - the probe chain terminator, and what a fresh chunk starts as. */
        empty_k = 0,

        /** Claimed; the key & value stores have not been released yet. */
        building_k,

        /** Fully published; the key & value are immutable from here on. */
        published_k,
    };

    /** Seats one @ref claim handed out: the next to fill and the bound past the last. */
    struct seats_t {
        std::size_t next = 0;
        std::size_t bound = 0;
    };

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
        : states_(std::move(other.states_)), slots_(std::move(other.slots_)),
          claimed_seats_(std::exchange(other.claimed_seats_, 0)) {}

    /** Swap-based, so the displaced chunks die with @p other rather than here. Single-threaded, like @ref clear. */
    chunked_multimap& operator=(chunked_multimap&& other) noexcept {
        states_ = std::move(other.states_);
        slots_ = std::move(other.slots_);
        std::swap(claimed_seats_, other.claimed_seats_);
        return *this;
    }

    /** Forgets every entry and seat, keeping the chunks - single-threaded: the caller certifies no concurrent
     *  readers or writers for the call. */
    void clear() noexcept {
        for (std::size_t index = 0; index != max_chunks_k; ++index) {
            std::span<slot_state_t> const states = states_.chunk(index);
            if (states.empty()) break;
            std::memset(states.data(), 0, states.size_bytes());
        }
        claimed_seats_ = 0;
    }

    /** Frees every chunk - single-threaded, like @ref clear. */
    void reset() noexcept {
        states_.reset();
        slots_.reset();
        claimed_seats_ = 0;
    }

    /** Bytes the published chunks span. */
    std::size_t memory_usage() const noexcept { return states_.memory_usage() + slots_.memory_usage(); }

    /** Publishes chunks for @p capacity seats - a hint, callable any time, alongside inserts. */
    status_t reserve(std::size_t capacity) noexcept { return cover_(capacity); }

    /** Seats for @p count later inserts, covering their chunks - the only fallible step, so the inserts that consume
     *  the seats never fail. A refused cover strands its seats, which only leaves their chunks emptier. */
    expected<seats_t> claim(std::size_t count) noexcept {
        std::size_t const first = atomic_ref<std::size_t>(claimed_seats_).fetch_add(count, std::memory_order_relaxed);
        if (status_t const covered = cover_(first + count); failed(covered)) return {seats_t {}, covered};
        return {seats_t {first, first + count}, status_t::success_k};
    }

    /** Records @p value under @p key in the next of @p seats - safe under arbitrary concurrent inserts & lookups,
     *  and never fails, never waits. */
    void insert(seats_t& seats, key_t const& key, value_t value) noexcept {
        assert(seats.next != seats.bound && "Every insert consumes a claimed seat");
        std::size_t const index = seats_schedule_t::locate(seats.next++).chunk;
        std::span<slot_state_t> const states = states_.chunk(index);
        std::span<slot_t> const slots = slots_.chunk(index);
        std::size_t position = first_probe_of(key, states.size());
        while (true) {
            atomic_ref<slot_state_t> state(states[position]);
            slot_state_t expected_empty = slot_state_t::empty_k;
            if (state.load(std::memory_order_relaxed) == slot_state_t::empty_k &&
                state.compare_exchange_strong(expected_empty, slot_state_t::building_k, std::memory_order_acquire,
                                              std::memory_order_relaxed)) {
                slots[position] = {key, value};
                state.store(slot_state_t::published_k, std::memory_order_release);
                return;
            }
            position = (position + 1) & (states.size() - 1);
        }
    }

    /** One value stored under @p key, or @c malformed_k when absent. Duplicates surface in an unspecified order -
     *  use @c find_each to see them all. */
    expected<value_t> find(key_t const& key) const noexcept {
        expected<value_t> result {value_t {}, status_t::malformed_k};
        find_each(key, [&](value_t value) noexcept { result = {value, status_t::success_k}; });
        return result;
    }

    /** The first value stored under @p key in chunk and probe order, stopping there, or
     *  @c malformed_k when absent; the same value every call once the key's inserts published. */
    expected<value_t> find_first(key_t const& key) const noexcept {
        for (std::size_t index = 0; index != max_chunks_k; ++index) {
            std::span<slot_state_t const> const states = states_.chunk(index);
            std::span<slot_t const> const slots = slots_.chunk(index);
            if (slots.empty()) break;
            for (std::size_t position = first_probe_of(key, states.size());;
                 position = (position + 1) & (states.size() - 1)) {
                slot_state_t const state =
                    atomic_ref<slot_state_t const>(states[position]).load(std::memory_order_acquire);
                if (state == slot_state_t::empty_k) break;
                if (state == slot_state_t::published_k && slots[position].key == key)
                    return {slots[position].value, status_t::success_k};
            }
        }
        return {value_t {}, status_t::malformed_k};
    }

    /** Invokes @p callback for every value stored under @p key, across every published chunk - wait-free. A chunk
     *  whose slots are not published yet holds no entry, and neither does any after it. */
    template <typename callback_type_>
    void find_each(key_t const& key, callback_type_&& callback) const noexcept {
        for (std::size_t index = 0; index != max_chunks_k; ++index) {
            std::span<slot_state_t const> const states = states_.chunk(index);
            std::span<slot_t const> const slots = slots_.chunk(index);
            if (slots.empty()) return;
            for (std::size_t position = first_probe_of(key, states.size());;
                 position = (position + 1) & (states.size() - 1)) {
                slot_state_t const state =
                    atomic_ref<slot_state_t const>(states[position]).load(std::memory_order_acquire);
                if (state == slot_state_t::empty_k) break;
                // A building slot's key is still in flight - occupied, but never a match.
                if (state == slot_state_t::published_k && slots[position].key == key) callback(slots[position].value);
            }
        }
    }

    /** Number of values stored under @p key. */
    std::size_t count(key_t const& key) const noexcept {
        std::size_t matches = 0;
        find_each(key, [&](value_t) noexcept { ++matches; });
        return matches;
    }

    /** Whether at least one value is stored under @p key. */
    bool contains(key_t const& key) const noexcept { return static_cast<bool>(find(key)); }

    /** Seats the published chunks hold. */
    std::size_t capacity() const noexcept { return slots_.capacity() / 2; }

  private:

    /** One probe slot: the key beside its value; its lifecycle lives in the control-byte column, so no payload
     *  bit-pattern is ever reserved. */
    struct slot_t {
        key_t key;
        value_t value;
    };

    /** Slots the columns hold - every chunk of the doubling phase, past any table a machine holds. */
    inline static constexpr std::size_t max_slots_k = slots_schedule_t::doubling_rows_k;
    inline static constexpr std::size_t max_chunks_k = slots_schedule_t::chunks_for(max_slots_k);

    status_t cover_(std::size_t seats) noexcept {
        if (seats > max_slots_k / 2) return status_t::overflow_k;
        // The control bytes first: a reader stops at the first chunk whose slots are missing.
        if (status_t const covered = states_.cover(seats * 2); failed(covered)) return covered;
        return slots_.cover(seats * 2);
    }

    /** Avalanches the hash through @ref mix_bits before masking, so the identity hashes of sequential integer keys
     *  still scatter across the table. */
    std::size_t first_probe_of(key_t const& key, std::size_t slots_count) const noexcept {
        return static_cast<std::size_t>(mix_bits(static_cast<u64_t>(hash_(key)))) & (slots_count - 1);
    }

    chunked_column<slot_state_t, slots_schedule_t, max_slots_k, allocator_t, atomic_ref_template_> states_;
    chunked_column<slot_t, slots_schedule_t, max_slots_k, allocator_t, atomic_ref_template_> slots_;

    /** Seats every @ref claim so far handed out. */
    std::size_t claimed_seats_ = 0;

    [[no_unique_address]] hash_t hash_;
};

} // namespace unum::usearch::utilities
