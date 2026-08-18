/**
 *  @brief Associative open-addressing hash tables for Graph traversals & key lookups.
 *  @author Ash Vardanian
 *  @file include/usearch/utilities/ram_tables.hpp
 *  @date August 18, 2026
 *
 *  Standalone header providing baseline associative structures, like:
 *
 *  - Exact per-thread visited @b hash-sets
 *  - Concurrent insert-only @b hash-multimaps
 */
#pragma once

#include <array>       // `std::array` - batched membership inserts
#include <bit>         // `std::countr_zero` - first-empty lane of a probe group
#include <cstddef>     // `std::size_t`
#include <cstdint>     // `std::uint64_t` - batched membership masks
#include <cstring>     // `std::memset` - the all-ones sentinel fill
#include <atomic>      // `std::atomic`, `std::atomic_ref`
#include <functional>  // `std::hash`
#include <limits>      // `std::numeric_limits` - the all-ones sentinel
#include <memory>      // `std::allocator`
#include <type_traits> // `std::is_integral`, `std::is_trivially_copyable_v`
#include <utility>     // `std::swap`, `std::move`

#if defined(__AVX512F__)
#include <immintrin.h> // 512-bit group probes of the visited hash-set
#endif

#include "usearch/utilities/ram_buffers.hpp" // `exchange`, `ceil2`
#include "usearch/utilities/types.hpp"       // `status_t`, `expected`

namespace unum {
namespace usearch {

/**
 *  @brief Single-byte scalar used as the default allocation unit for @ref growing_hash_set.
 */
using byte_t = char;

/**
 *  @brief Returns the sentinel that marks an unused slot in @ref growing_hash_set.
 *
 *  For integral types this is the all-ones bit pattern, which matches the
 *  @c 0xFF byte-fill emitted by @ref growing_hash_set::clear.
 */
template <typename element_type_>
inline element_type_ default_free_value() noexcept {
    static_assert(std::is_integral<element_type_>::value,
                  "`growing_hash_set` requires an integral element type for its sentinel convention");
    return (std::numeric_limits<element_type_>::max)();
}

/**
 *  @brief Minimalistic open-addressing hash-set, used to track visited nodes during graph
 *      traversal - a sparse alternative to a bit-set in our primary use case.
 *
 *  It doesn't support deleting individual elements, but supports @c clear -ing all at once.
 *  It expects @c reserve to be called ahead of all insertions, so no mid-insert resizes are
 *  needed, and it treats all-ones slots as unused, to simplify the design. The slot count
 *  stays a power of two, and bucket collisions resolve by linear probing.
 */
template <                                            //
    typename element_type_,                           //
    typename hasher_type_ = std::hash<element_type_>, //
    typename allocator_type_ = std::allocator<byte_t>>
class growing_hash_set {

    using element_t = element_type_;
    using hasher_t = hasher_type_;

    using allocator_t = allocator_type_;
    using byte_t = typename allocator_t::value_type;
    static_assert(sizeof(byte_t) == 1, "Allocator must allocate separate addressable bytes");

    element_t* slots_ {};
    /** @brief Number of slots. */
    std::size_t capacity_ {};
    /** @brief Number of populated slots. */
    std::size_t count_ {};
    hasher_t hasher_ {};
    allocator_t allocator_ {};

  public:
    growing_hash_set(allocator_t allocator = {}) noexcept : allocator_(std::move(allocator)) {}
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

    growing_hash_set(std::size_t capacity, allocator_t allocator = {}) noexcept : allocator_(std::move(allocator)) {
        slots_ = (element_t*)allocator_.allocate(ceil2(capacity) * sizeof(element_t));
        capacity_ = slots_ ? ceil2(capacity) : 0u;
        count_ = 0u;
        clear();
    }

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

  private:
#if defined(__AVX512F__)
    /** @brief Sixteen 4-byte slots fill one 512-bit register, so the probe advances a whole
     *      cache line per step; wider or narrower elements keep the classic scalar walk. */
    static constexpr bool simd_groups_k = sizeof(element_t) == 4 && std::is_integral_v<element_t>;
#else
    static constexpr bool simd_groups_k = false;
#endif

    /**
     *  @brief The one probe walk behind @c test, @c set, and the rehash: returns the slot
     *      holding @p elem, or the empty slot where it belongs. A single shared algorithm is
     *      what keeps placement and lookup coherent - a rehash placing elements any other
     *      way would strand them unreachable.
     */
    static inline element_t* locate_(element_t* slots, std::size_t capacity, hasher_t const& hasher,
                                     element_t elem) noexcept
        requires(!simd_groups_k)
    {
        std::size_t index = hasher(elem) & (capacity - 1);
        while (slots[index] != default_free_value<element_t>() && slots[index] != elem)
            index = (index + 1) & (capacity - 1);
        return slots + index;
    }

#if defined(__AVX512F__)
    /**
     *  @brief The group sibling of the scalar @c locate_: one unaligned 512-bit load compares
     *      sixteen slots at once - one mask for the key, one for the empty sentinel - so a
     *      probe touches cache lines, not slots, and the mispredicted probe-continue branch
     *      of the scalar walk disappears.
     *
     *  Placement is group-ordered: an element may land @b before its exact hashed slot inside
     *  its starting group, and every reader replicates that order, so a group containing any
     *  empty lane terminates the walk. Groups are 16-aligned and the capacity is a power of
     *  two, so once it reaches 16 no load ever straddles the table's end; smaller tables take
     *  the scalar walk below.
     */
    static inline element_t* locate_(element_t* slots, std::size_t capacity, hasher_t const& hasher,
                                     element_t elem) noexcept
        requires(simd_groups_k)
    {
        if (capacity < 16) {
            std::size_t index = hasher(elem) & (capacity - 1);
            while (slots[index] != default_free_value<element_t>() && slots[index] != elem)
                index = (index + 1) & (capacity - 1);
            return slots + index;
        }
        __m512i const key = _mm512_set1_epi32(static_cast<int>(elem));
        __m512i const empty = _mm512_set1_epi32(static_cast<int>(default_free_value<element_t>()));
        std::size_t group = (hasher(elem) & (capacity - 1)) & ~std::size_t {15};
        while (true) {
            __m512i const block = _mm512_loadu_si512(slots + group);
            __mmask16 const key_lanes = _mm512_cmpeq_epi32_mask(block, key);
            if (key_lanes) return slots + group + std::countr_zero(static_cast<unsigned>(key_lanes));
            __mmask16 const empty_lanes = _mm512_cmpeq_epi32_mask(block, empty);
            if (empty_lanes) return slots + group + std::countr_zero(static_cast<unsigned>(empty_lanes));
            group = (group + 16) & (capacity - 1);
        }
    }
#endif

  public:
    /**
     *  @brief Checks if the element is already in the hash-set.
     *  @return @c true if the element is present.
     */
    inline bool test(element_t const& elem) const noexcept {
        return *locate_(slots_, capacity_, hasher_, elem) == elem;
    }

    /**
     *  @brief Inserts an element into the hash-set.
     *  @return The previous membership: @c true if the element was already present,
     *      @c false when it was newly inserted.
     */
    inline bool set(element_t const& elem) noexcept {
        element_t* slot = locate_(slots_, capacity_, hasher_, elem);
        if (*slot == elem) return true;
        *slot = elem;
        ++count_;
        return false;
    }

    /**
     *  @brief Inserts up to @c batch_size_ elements sequentially.
     *  @return 64-bit mask - bit @c i is set iff @c batch[i] was newly inserted (matching
     *      the single-element @c set convention where a newly-inserted element returns @c false).
     *
     *  Scalar baseline just calls @ref set in a loop; the measured win over single-element
     *  inserts (~1.6× on the microbench) comes from the compiler interleaving independent
     *  load-use chains through two load ports. Future SIMD variants can overlap wider
     *  bucket compares without changing the call shape.
     */
    template <std::size_t batch_size_>
    inline std::uint64_t set(std::array<element_t, batch_size_> const& batch) noexcept {
        static_assert(batch_size_ > 0 && batch_size_ <= 64, "Batch size must fit the uint64 return mask");
        std::uint64_t out = 0;
        for (std::size_t i = 0; i < batch_size_; ++i)
            if (!set(batch[i])) out |= std::uint64_t {1} << i;
        return out;
    }

    /**
     *  @brief Extends the capacity of the hash-set, rehashing every present element.
     *  @return @c true if enough capacity is available, @c false if memory allocation failed.
     */
    bool reserve(std::size_t new_capacity) noexcept {
        new_capacity = (new_capacity * 5u) / 3u;
        if (new_capacity <= capacity_) return true;

        new_capacity = ceil2(new_capacity);
        element_t* new_slots = (element_t*)allocator_.allocate(new_capacity * sizeof(element_t));
        if (!new_slots) return false;

        std::memset((void*)new_slots, 0xFF, new_capacity * sizeof(element_t));
        std::size_t new_count = count_;
        // Re-place through the very same probe walk `test` & `set` use - the target slot of
        // a not-yet-present element is exactly where `locate_` lands.
        if (count_)
            for (std::size_t old_index = 0; old_index != capacity_; ++old_index) {
                if (slots_[old_index] == default_free_value<element_t>()) continue;
                *locate_(new_slots, new_capacity, hasher_, slots_[old_index]) = slots_[old_index];
            }

        reset();
        slots_ = new_slots;
        capacity_ = new_capacity;
        count_ = new_count;
        return true;
    }
};

/**
 *  @brief Insert-only, linear-probe hash multimap. Duplicate keys coexist as separate
 *      slots, which is what makes concurrent insertion CAS-once instead of lock-and-chain:
 *      a slot's value word flips from the reserved all-ones sentinel to its mapped value
 *      exactly once.
 *
 *  Readers are wait-free: they probe with acquire loads and treat an in-flight slot -
 *  claimed, value not yet visible - as occupied-but-mismatched, never as end-of-chain.
 *  Growth is explicit: @c reserve may be called repeatedly to rehash into a larger slot
 *  array, but it demands a mutation-free phase - concurrent inserts never trigger a
 *  hidden migration, they fail with @c overflow_k when the table is full. The all-ones
 *  @p value_type_ bit-pattern is reserved as the empty sentinel and must never be inserted.
 */
template <typename key_type_, typename value_type_,             //
          typename hash_type_ = std::hash<key_type_>,           //
          typename allocator_type_ = std::allocator<std::byte>> //
class growing_multimap {
    static_assert(std::is_trivially_copyable_v<key_type_> && std::is_trivially_copyable_v<value_type_>);

  public:
    using key_t = key_type_;
    using value_t = value_type_;
    using hash_t = hash_type_;
    using allocator_t = allocator_type_;

    growing_multimap(growing_multimap const&) noexcept = delete;
    growing_multimap& operator=(growing_multimap const&) noexcept = delete;

    explicit growing_multimap(allocator_t allocator = {}) noexcept;
    ~growing_multimap() noexcept;

    /** @brief Sizes the slot array for @p capacity entries at a bounded load factor.
     *      Repeatable: a larger @p capacity rehashes into a fresh array, single-threaded -
     *      the caller certifies no concurrent readers or writers for the growth call. */
    status_t reserve(std::size_t capacity) noexcept;

    /** @brief Claims one slot along the probe path of @p key - safe under arbitrary
     *      concurrent inserts & lookups; fails with @c overflow_k when full. */
    status_t insert(key_t const& key, value_t value) noexcept;

    /** @brief One value stored under @p key, or @c malformed_k when absent. Duplicates
     *      surface in an unspecified order - use @c find_each to see them all. */
    expected<value_t> find(key_t const& key) const noexcept;

    /** @brief Invokes @p callback for every value stored under @p key. */
    template <typename callback_type_>
    void find_each(key_t const& key, callback_type_&& callback) const noexcept;

    /** @brief Number of values stored under @p key. */
    std::size_t count(key_t const& key) const noexcept;

    /** @brief Whether at least one value is stored under @p key. */
    bool contains(key_t const& key) const noexcept;

    /** @brief Total inserted entries across all keys. */
    std::size_t size() const noexcept;

    /** @brief Total entries the current slot array can host. */
    std::size_t capacity() const noexcept;

  private:
    /** @brief One probe slot: the key beside its value; the value doubles as the claim
     *      word, release-stored after the key lands. */
    struct slot_t {
        key_t key;
        value_t value;
    };

    slot_t* slots_ = nullptr;
    std::size_t slots_count_ = 0;
    std::atomic<std::size_t> inserted_count_ {0};

    [[no_unique_address]] hash_t hash_;
    [[no_unique_address]] allocator_t allocator_;
};

} // namespace usearch
} // namespace unum
