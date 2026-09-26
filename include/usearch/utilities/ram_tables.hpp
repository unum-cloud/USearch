/**
 *  @file include/usearch/utilities/ram_tables.hpp
 *  @author Ash Vardanian
 *  @date August 18, 2026
 *  @brief Associative open-addressing hash tables for Graph traversals & key lookups.
 *
 *  Standalone header providing baseline associative structures, like:
 *
 *  - Exact per-thread visited @b hash-sets
 *  - Concurrent insert-only @b hash-multimaps
 */
#pragma once

#include <cassert> // `assert` - every `insert` follows a `reserve`, and never inserts the sentinel
#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint64_t` - batched membership masks
#include <cstring> // `std::memset` - the all-ones sentinel fill

#include <array>       // `std::array` - batched membership inserts
#include <atomic>      // `std::atomic`, `std::atomic_ref`
#include <bit>         // `std::countr_zero` - the first matching lane of a group scan
#include <functional>  // `std::hash`
#include <limits>      // `std::numeric_limits` - the all-ones sentinel
#include <memory>      // `std::allocator`
#include <type_traits> // `std::is_integral`, `std::is_trivially_copyable_v`
#include <utility>     // `std::swap`, `std::move`

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h> // `_mm512_cmpeq_epi32_mask` - the sixteen-slot group scan
#endif

#include "usearch/utilities/ram_buffers.hpp" // `exchange`, `ceil2`
#include "usearch/utilities/types.hpp"       // `status_t`, `expected`

namespace unum {
namespace usearch {

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

/** The group scan for u32 slots, defined below wherever the intrinsics exist. Declared
 *  unconditionally, so a capability chain can name it on every target and its own tier flag
 *  alone decides whether a class takes it - an unselected layer never needs a body. */
struct u32_skylake_visits_hash_set_kit_t;

#if defined(__x86_64__) || defined(_M_X64)
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx512f"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx512f")
#endif

/**
 *  @brief The group scan for u32 slots on AVX-512 units: one 512-bit load compares sixteen slots
 *      against the element and against the empty sentinel, so a probe touches cache lines rather
 *      than slots.
 *
 *  Placement is group-ordered: an element may land @b before its exact hashed slot inside its
 *  starting group, and every reader replicates that order, so a group holding any empty lane ends
 *  the walk. Groups are 16-aligned and the capacity is a power of two, so once it reaches 16 no
 *  load straddles the table's end; smaller tables take the serial walk. The scan carries its own
 *  ISA, so any unit compiles it whatever flags that unit was given.
 */
struct u32_skylake_visits_hash_set_kit_t {
    template <typename hasher_type_>
    static inline u32_t* locate_slot(u32_t* slots, std::size_t capacity, hasher_type_ const& hasher,
                                     u32_t element) noexcept {
        if (capacity < 16) return serial_visits_hash_set_kit_t::locate_slot(slots, capacity, hasher, element);
        __m512i const key_u32x16 = _mm512_set1_epi32(static_cast<int>(element));
        __m512i const empty_u32x16 = _mm512_set1_epi32(static_cast<int>(unused_hash_slot_value<u32_t>()));
        std::size_t group = (hasher(element) & (capacity - 1)) & ~std::size_t {15};
        while (true) {
            __m512i const slots_u32x16 = _mm512_loadu_si512(slots + group);
            __mmask16 const key_mask_m16 = _mm512_cmpeq_epi32_mask(slots_u32x16, key_u32x16);
            if (key_mask_m16) return slots + group + std::countr_zero(static_cast<unsigned>(key_mask_m16));
            __mmask16 const empty_mask_m16 = _mm512_cmpeq_epi32_mask(slots_u32x16, empty_u32x16);
            if (empty_mask_m16) return slots + group + std::countr_zero(static_cast<unsigned>(empty_mask_m16));
            group = (group + 16) & (capacity - 1);
        }
    }
};

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // defined(__x86_64__) || defined(_M_X64)

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

    void clear() noexcept {
        if (slots_) std::memset((void*)slots_, 0xFF, capacity_ * sizeof(element_t));
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
 *  @brief Insert-only, linear-probe hash multimap. Duplicate keys coexist as separate
 *      slots, which is what makes concurrent insertion CAS-once instead of lock-and-chain:
 *      a slot's state advances @c empty_k → @c building_k → @c published_k exactly once.
 *
 *  Slot states live in a parallel control-byte array the probes scan - 64 states per cache
 *  line - so a slot's key & value are touched only on a published candidate, and no key or
 *  value bit-pattern is ever reserved. Readers are wait-free: they probe with acquire
 *  loads, stop at @c empty_k, and treat @c building_k as occupied-but-mismatched - never
 *  as end-of-chain - so a half-published entry is merely invisible, never wrong. Growth is
 *  explicit: @c reserve may be called repeatedly to rehash into a larger array, but it
 *  demands a mutation-free phase - concurrent inserts never trigger a hidden migration,
 *  they fail with @c overflow_k when the ceiling is spent. The slot count stays at least
 *  twice the entry ceiling, so every probe chain terminates.
 */
template <typename key_type_, typename value_type_,          //
          typename hash_type_ = std::hash<key_type_>,        //
          typename allocator_type_ = std::allocator<byte_t>> //
class growing_multimap {
    static_assert(std::is_trivially_copyable_v<key_type_> && std::is_trivially_copyable_v<value_type_>);

  public:
    using key_t = key_type_;
    using value_t = value_type_;
    using hash_t = hash_type_;
    using allocator_t = allocator_type_;

    /** Lifecycle of one slot, advanced monotonically by its one claiming inserter. */
    enum class slot_state_t : u8_t {

        /** Never claimed - the probe chain terminator. */
        empty_k = 0,

        /** Claimed; the key & value stores have not been released yet. */
        building_k,

        /** Fully published; the key & value are immutable from here on. */
        published_k,
    };

    growing_multimap(growing_multimap const&) noexcept = delete;
    growing_multimap& operator=(growing_multimap const&) noexcept = delete;

    /** Every map names its allocator; nothing is placed by default. */
    growing_multimap() = delete;

    /** The process-wide allocator, for a map no arena or domain owns. */
    explicit growing_multimap(global_allocator_t) noexcept
        requires std::default_initializable<allocator_t>
        : growing_multimap(allocator_t {}) {}
    explicit growing_multimap(allocator_t allocator) noexcept : allocator_(std::move(allocator)) {}
    ~growing_multimap() noexcept { reset(); }

    growing_multimap(growing_multimap&& other) noexcept
        : slots_(exchange(other.slots_, nullptr)), states_(exchange(other.states_, nullptr)),
          slots_count_(exchange(other.slots_count_, 0)), entries_ceiling_(exchange(other.entries_ceiling_, 0)),
          inserted_count_(other.inserted_count_.load(std::memory_order_relaxed)), hash_(std::move(other.hash_)),
          allocator_(std::move(other.allocator_)) {
        other.inserted_count_.store(0, std::memory_order_relaxed);
    }

    /** Swap-based, so the displaced table dies with @p other rather than here; the count is
     *  moved by hand, since an atomic is not swappable. Single-threaded, like @ref clear. */
    growing_multimap& operator=(growing_multimap&& other) noexcept {
        std::swap(slots_, other.slots_);
        std::swap(states_, other.states_);
        std::swap(slots_count_, other.slots_count_);
        std::swap(entries_ceiling_, other.entries_ceiling_);
        std::size_t const inserted = inserted_count_.load(std::memory_order_relaxed);
        inserted_count_.store(other.inserted_count_.load(std::memory_order_relaxed), std::memory_order_relaxed);
        other.inserted_count_.store(inserted, std::memory_order_relaxed);
        std::swap(hash_, other.hash_);
        std::swap(allocator_, other.allocator_);
        return *this;
    }

    /** Forgets every entry, keeping the slot array & ceiling - single-threaded, like
     *  @c reserve: the caller certifies no concurrent readers or writers for the call. */
    void clear() noexcept {
        for (std::size_t position = 0; position != slots_count_; ++position) states_[position] = slot_state_t::empty_k;
        inserted_count_.store(0, std::memory_order_relaxed);
    }

    /** Bytes @p slots_count slots occupy - one allocation carrying both arrays: the
     *  slots, then the control bytes. */
    static std::size_t memory_usage(std::size_t slots_count) noexcept {
        return slots_count * (sizeof(slot_t) + sizeof(slot_state_t));
    }

    /** Bytes this multimap currently occupies. */
    std::size_t memory_usage() const noexcept { return memory_usage(slots_count_); }

    void reset() noexcept {
        if (slots_)
            bytes_allocator_t(allocator_).deallocate(reinterpret_cast<std::byte*>(slots_), memory_usage(slots_count_));
        slots_ = nullptr;
        states_ = nullptr;
        slots_count_ = 0;
        entries_ceiling_ = 0;
        inserted_count_.store(0, std::memory_order_relaxed);
    }

    /** Sizes the slot array for @p capacity entries at a load factor of at most one
     *  half. Repeatable: a larger @p capacity rehashes into a fresh array, single-
     *  threaded - the caller certifies no concurrent readers or writers for the call. */
    status_t reserve(std::size_t capacity) noexcept {
        if (capacity <= entries_ceiling_) return status_t::success_k;

        std::size_t const new_slots_count = ceil2(max_of<std::size_t>(capacity * 2u, 16u));
        std::byte* const new_bytes = bytes_allocator_t(allocator_).allocate(memory_usage(new_slots_count));
        if (!new_bytes) return status_t::bad_alloc_k;
        slot_t* const new_slots = reinterpret_cast<slot_t*>(new_bytes);
        slot_state_t* const new_states = reinterpret_cast<slot_state_t*>(new_bytes + //
                                                                         new_slots_count * sizeof(slot_t));
        for (std::size_t position = 0; position != new_slots_count; ++position)
            new_states[position] = slot_state_t::empty_k;

        // Re-place settled entries through the same probe walk `insert` uses; the phase is
        // mutation-free, so no building slots can exist and plain stores suffice.
        for (std::size_t position = 0; position != slots_count_; ++position) {
            if (states_[position] != slot_state_t::published_k) continue;
            std::size_t chain = first_probe_of(slots_[position].key, new_slots_count);
            while (new_states[chain] != slot_state_t::empty_k) chain = (chain + 1) & (new_slots_count - 1);
            new_slots[chain] = slots_[position];
            new_states[chain] = slot_state_t::published_k;
        }

        if (slots_)
            bytes_allocator_t(allocator_).deallocate(reinterpret_cast<std::byte*>(slots_), memory_usage(slots_count_));
        slots_ = new_slots;
        states_ = new_states;
        slots_count_ = new_slots_count;
        entries_ceiling_ = capacity;
        return status_t::success_k;
    }

    /** Claims one slot along the probe path of @p key - safe under arbitrary
     *  concurrent inserts & lookups; fails with @c overflow_k once the ceiling is spent. */
    status_t insert(key_t const& key, value_t value) noexcept {
        if (!slots_count_) return status_t::uninitialized_k;
        // Claim a seat under the ceiling before touching any slot, so probe chains stay bounded by construction
        // and `size()` never passes `capacity()`, not even while a refusal is in flight.
        std::size_t inserted = inserted_count_.load(std::memory_order_relaxed);
        while (inserted < entries_ceiling_ &&
               !inserted_count_.compare_exchange_weak(inserted, inserted + 1, std::memory_order_relaxed)) {}
        if (inserted >= entries_ceiling_) return status_t::overflow_k;
        std::size_t chain = first_probe_of(key, slots_count_);
        while (true) {
            std::atomic_ref<slot_state_t> state(states_[chain]);
            slot_state_t expected_empty = slot_state_t::empty_k;
            if (state.load(std::memory_order_relaxed) == slot_state_t::empty_k &&
                state.compare_exchange_strong(expected_empty, slot_state_t::building_k, std::memory_order_acquire,
                                              std::memory_order_relaxed)) {
                slots_[chain].key = key;
                slots_[chain].value = value;
                state.store(slot_state_t::published_k, std::memory_order_release);
                return status_t::success_k;
            }
            chain = (chain + 1) & (slots_count_ - 1);
        }
    }

    /** One value stored under @p key, or @c malformed_k when absent. Duplicates
     *  surface in an unspecified order - use @c find_each to see them all. */
    expected<value_t> find(key_t const& key) const noexcept {
        expected<value_t> result {value_t {}, status_t::malformed_k};
        find_each(key, [&](value_t value) noexcept { result = {value, status_t::success_k}; });
        return result;
    }

    /** Invokes @p callback for every value stored under @p key - wait-free. */
    template <typename callback_type_>
    void find_each(key_t const& key, callback_type_&& callback) const noexcept {
        if (!slots_count_) return;
        std::size_t chain = first_probe_of(key, slots_count_);
        while (true) {
            slot_state_t const state = std::atomic_ref<slot_state_t>(states_[chain]).load(std::memory_order_acquire);
            if (state == slot_state_t::empty_k) return;
            // A building slot's key is still in flight - occupied, but never a match.
            if (state == slot_state_t::published_k && slots_[chain].key == key) callback(slots_[chain].value);
            chain = (chain + 1) & (slots_count_ - 1);
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

    /** Total inserted entries across all keys. */
    std::size_t size() const noexcept { return inserted_count_.load(std::memory_order_relaxed); }

    /** Total entries the current slot array can host. */
    std::size_t capacity() const noexcept { return entries_ceiling_; }

  private:

    /** One probe slot: the key beside its value; its lifecycle lives in the
     *  parallel control-byte array, so no payload bit-pattern is ever reserved. */
    struct slot_t {
        key_t key;
        value_t value;
    };

    using bytes_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<std::byte>;

    /** Avalanches the hash through @ref mix_bits before masking, so the identity
     *  hashes of sequential integer keys still scatter across the table. */
    std::size_t first_probe_of(key_t const& key, std::size_t slots_count) const noexcept {
        return static_cast<std::size_t>(mix_bits(static_cast<u64_t>(hash_(key)))) & (slots_count - 1);
    }

    slot_t* slots_ = nullptr;
    slot_state_t* states_ = nullptr;
    std::size_t slots_count_ = 0;
    std::size_t entries_ceiling_ = 0;
    std::atomic<std::size_t> inserted_count_ {0};

    [[no_unique_address]] hash_t hash_;
    [[no_unique_address]] allocator_t allocator_;
};

} // namespace usearch
} // namespace unum
