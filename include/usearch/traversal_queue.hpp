/**
 *  @brief SIMD-accelerated dynamic data-structures for Graph traversals.
 *  @author Ash Vardanian
 *  @file traversal_queue.hpp
 *  @date April 21, 2026
 *
 *  Standalone header providing baseline dynamic data-structures for Graph traversals, like:
 *
 *  - Implicit @b binary-heaps
 *  - @b Sorted-arrays
 *  - Resizable flat @b hash-sets
 */
#pragma once

#include <algorithm>   // `std::lower_bound`, `std::sort_heap`, `std::min`, `std::max`
#include <cstddef>     // `std::size_t`
#include <cstdint>     // `SIZE_MAX`
#include <cstring>     // `std::memcpy`, `std::memset`
#include <functional>  // `std::less`, `std::hash`
#include <limits>      // `std::numeric_limits`
#include <memory>      // `std::allocator`
#include <new>         // `new (ptr) element_t(value)` for "placement-new"
#include <type_traits> // `std::is_trivially_destructible`, `std::is_integral`
#include <utility>     // `std::swap`, `std::move`, `std::forward`

namespace unum {
namespace usearch {

/**
 *  @brief Single-byte scalar used as the default allocation unit for @ref growing_hash_set.
 */
using byte_t = char;

/**
 *  @brief Assigns @p new_value to @p old_value and returns its previous contents.
 *
 *  Local re-implementation of C++14's `std::exchange`, matching the rest of the
 *  USearch codebase (see `index.hpp`) so the header stays usable in projects
 *  pinned to C++11.
 */
template <typename value_type_, typename other_type_ = value_type_>
inline value_type_ exchange(value_type_& old_value, other_type_&& new_value) noexcept {
    value_type_ previous = std::move(old_value);
    old_value = std::forward<other_type_>(new_value);
    return previous;
}

/**
 *  @brief Rounds @p value up to the next power of two.
 *  @return The smallest power of two greater than or equal to @p value, or `0` for `0`.
 *
 *  Keeps container capacities power-of-two so that `index & (capacity_ - 1)` can
 *  substitute for the otherwise costlier `index % capacity_`. Branchless bit trick
 *  from Henry S. Warren Jr., "Hacker's Delight", 2nd ed., §3-2.
 */
inline std::size_t ceil2(std::size_t value) noexcept {
    value--;
    value |= value >> 1;
    value |= value >> 2;
    value |= value >> 4;
    value |= value >> 8;
    value |= value >> 16;
#if SIZE_MAX > 0xFFFFFFFFu
    value |= value >> 32;
#endif
    value++;
    return value;
}

/**
 *  @brief Returns the sentinel that marks an unused slot in @ref growing_hash_set.
 *
 *  For integral types this is the all-ones bit pattern, which matches the
 *  `0xFF` byte-fill emitted by @ref growing_hash_set::clear.
 */
template <typename element_type_>
inline element_type_ default_free_value() noexcept {
    static_assert(std::is_integral<element_type_>::value,
                  "`growing_hash_set` requires an integral element type for its sentinel convention");
    return (std::numeric_limits<element_type_>::max)();
}

/**
 *  @brief Similar to @b `std::priority_queue`, but allows raw access to underlying
 *      memory, in case you want to shuffle it or sort. Good for collections
 *      from 100s to 10'000s elements.
 *
 *  In a max-heap, the heap property ensures that the value of each node is greater
 *  than or equal to the values of its children. This means that the largest element
 *  is always at the root of the heap.
 *
 *  @section Heap Structures
 *
 *  There are several designs of heaps. Binary heaps are the simplest & most common
 *  variant, that is easy to implement as a succint array. However, they are not the
 *  most efficient for all operations. Most importantly, @b melding (merging) of
 *  two heaps has linear complexity in time.
 *
 *  +-----------------+---------+-----------+---------+--------------+---------+
 *  | Operation       | find-max| delete-max| insert  | increase-key | meld    |
 *  +-----------------+---------+-----------+---------+--------------+---------+
 *  | Binary          | Θ(1)    | Θ(log n)  | O(log n)| O(log n)     | Θ(n)    |
 *  | Leftist         | Θ(1)    | Θ(log n)  | O(log n)| Θ(log n)     | Θ(log n)|
 *  | Binomial        | Θ(1)    | Θ(log n)  | Θ(1)    | Θ(log n)     | O(log n)|
 *  | Skew binomial   | Θ(1)    | Θ(log n)  | Θ(1)    | O(log n)     | O(log n)|
 *  | Pairing         | Θ(1)    | O(log n)  | Θ(1)    | o(log n)     | Θ(1)    |
 *  | Rank-pairing    | Θ(1)    | O(log n)  | Θ(1)    | Θ(1)         | Θ(1)    |
 *  | Fibonacci       | Θ(1)    | O(log n)  | Θ(1)    | Θ(1)         | Θ(1)    |
 *  | Strict Fibonacci| Θ(1)    | O(log n)  | Θ(1)    | Θ(1)         | Θ(1)    |
 *  | Brodal          | Θ(1)    | Θ(log n)  | Θ(1)    | Θ(1)         | Θ(1)    |
 *  | 2–3 heap        | Θ(1)    | O(log n)  | Θ(1)    | Θ(1)         | O(log n)|
 *  +-----------------+---------+-----------+---------+--------------+---------+
 *
 *  It's well known, that improved priority queue structures translate into better
 *  graph-transversal algorithms. For example, Dijkstra's algorithm can be sped up
 *  by using a Fibonacci heap for arbitrary weights. For integer weight bounded
 *  by L, Schrijver reported following time complexities in 2004:
 *
 *  +------------+-------------------------------------+----------------------------+--------------------------+
 *  | Weights    | Algorithm                           | Time complexity            | Author                   |
 *  +------------+-------------------------------------+----------------------------+--------------------------+
 *  | R          |                                     | O(V^2 EL)                  | Ford 1956                |
 *  | R          | Bellman–Ford algorithm              | O(VE)                      | Shimbel 1955, Bellman    |
 *  |            |                                     |                            | 1958, Moore 1959         |
 *  | R          |                                     | O(V^2 log V)               | Dantzig 1960             |
 *  | R          | Dijkstra's with list                | O(V^2)                     | Leyzorek et al. 1957,    |
 *  |            |                                     |                            | Dijkstra 1959...         |
 *  | R          | Dijkstra's with binary heap         | O((E + V) log V)           | Johnson 1977             |
 *  | R          | Dijkstra's with Fibonacci heap      | O(E + V log V)             | Fredman & Tarjan 1984,   |
 *  |            |                                     |                            | Fredman & Tarjan 1987    |
 *  | R          | Quantum Dijkstra                    | O(√VE log^2 V)             | Dürr et al. 2006         |
 *  | R          | Dial's algorithm (Dijkstra's using  | O(E + LV)                  | Dial 1969                |
 *  |            | a bucket queue with L buckets)      |                            |                          |
 *  | N          |                                     | O(E log log L)             | Johnson 1981, Karlsson & |
 *  |            |                                     |                            | Poblete 1983             |
 *  | N          | Gabow's algorithm                   | O(E log_E/V L)             | Gabow 1983, Gabow 1985   |
 *  | N          |                                     | O(E + V √log L)            | Ahuja et al. 1990        |
 *  | N          | Thorup                              | O(E + V log log V)         | Thorup 2004              |
 *  +------------+-------------------------------------+----------------------------+--------------------------+
 *
 *  Possible improvements:
 *  - Randomized meldable heaps: https://en.wikipedia.org/wiki/Randomized_meldable_heap
 *  - D-ary heaps: https://en.wikipedia.org/wiki/D-ary_heap
 *  - B-heap: https://en.wikipedia.org/wiki/B-heap
 */
template <                                                    //
    typename element_type_,                                   //
    typename comparator_type_ = std::less<void>,              //
    typename allocator_type_ = std::allocator<element_type_>> //
class max_heap {
  public:
    using element_t = element_type_;
    using comparator_t = comparator_type_;
    using allocator_t = allocator_type_;

    using value_type = element_t;

    static_assert(std::is_trivially_destructible<element_t>(), "This heap is designed for trivial structs");
    static_assert(std::is_trivially_copy_constructible<element_t>(), "This heap is designed for trivial structs");

  private:
    element_t* elements_;
    std::size_t size_;
    std::size_t capacity_;

  public:
    max_heap() noexcept : elements_(nullptr), size_(0), capacity_(0) {}

    max_heap(max_heap&& other) noexcept
        : elements_(exchange(other.elements_, nullptr)), size_(exchange(other.size_, 0)),
          capacity_(exchange(other.capacity_, 0)) {}

    max_heap& operator=(max_heap&& other) noexcept {
        std::swap(elements_, other.elements_);
        std::swap(size_, other.size_);
        std::swap(capacity_, other.capacity_);
        return *this;
    }

    max_heap(max_heap const&) = delete;
    max_heap& operator=(max_heap const&) = delete;

    ~max_heap() noexcept { reset(); }

    void reset() noexcept {
        if (elements_) allocator_t {}.deallocate(elements_, capacity_);
        elements_ = nullptr;
        capacity_ = 0;
        size_ = 0;
    }

    inline bool empty() const noexcept { return !size_; }
    inline std::size_t size() const noexcept { return size_; }
    inline std::size_t capacity() const noexcept { return capacity_; }
    inline element_t* data() noexcept { return elements_; }
    inline element_t const* data() const noexcept { return elements_; }
    inline void clear() noexcept { size_ = 0; }
    inline void shrink(std::size_t n) noexcept { size_ = (std::min<std::size_t>)(n, size_); }

    /**
     *  @brief Selects the largest element in the heap.
     *  @return Reference to the stored element.
     */
    inline element_t const& top() const noexcept { return elements_[0]; }

    /**
     *  @brief Invalidates the "max-heap" property, transforming into ascending range.
     */
    inline void sort_ascending() noexcept { std::sort_heap(elements_, elements_ + size_, &less); }

    /**
     *  @brief Ensures the heap has enough capacity for the specified number of elements.
     *  @param new_capacity The desired minimum capacity.
     *  @return True if the capacity was successfully increased, false otherwise.
     */
    bool reserve(std::size_t new_capacity) noexcept {
        if (new_capacity <= capacity_) return true;

        new_capacity = ceil2(new_capacity);
        if (new_capacity == 0) return false;
        new_capacity = (std::max<std::size_t>)(new_capacity, (std::max<std::size_t>)(capacity_ * 2u, 16u));
        auto allocator = allocator_t {};
        auto new_elements = allocator.allocate(new_capacity);
        if (!new_elements) return false;

        if (elements_) {
            std::memcpy(new_elements, elements_, size_ * sizeof(element_t));
            allocator.deallocate(elements_, capacity_);
        }
        elements_ = new_elements;
        capacity_ = new_capacity;
        return new_elements;
    }

    /**
     *  @brief Inserts an element into the heap.
     *  @param element The element to be inserted.
     *  @return True if the element was successfully inserted, false otherwise.
     */
    bool insert(element_t&& element) noexcept {
        if (!reserve(size_ + 1)) return false;

        insert_reserved(std::move(element));
        return true;
    }

    /**
     *  @brief Inserts an element into the heap without reserving additional space.
     *  @param element The element to be inserted.
     */
    void insert_reserved(element_t&& element) noexcept {
        new (&elements_[size_]) element_t(element);
        size_++;
        shift_up(size_ - 1);
    }

    /**
     *  @brief Inserts multiple elements into the heap.
     *  @param elements Pointer to the elements to be inserted.
     *  @return True if the elements were successfully inserted, false otherwise.
     */
    inline bool insert_many(element_t const* elements) noexcept {
        // Wikipedia describes a procedure, due to Floyd, which constructs a heap from an array in linear time.
        // It also mentions a procedure for merging two heaps, of sizes 𝑛 and 𝑘, in time 𝑂(𝑘+log𝑘log𝑛).
        // Altogether, we can add 𝑘 elements to a heap of length 𝑛 in time 𝑂(𝑘+log𝑘log𝑛): first build a heap containing
        // 𝑘 elements to be inserted (takes 𝑂(𝑘) time), then merge that with the heap of size 𝑛 (takes 𝑂(𝑘+log𝑘log𝑛)
        // time). Compare this to repeated insertion, which would run in time 𝑂(𝑘log𝑛).
        return false;
    }

    element_t pop() noexcept {
        element_t result = top();
        std::swap(elements_[0], elements_[size_ - 1]);
        size_--;
        elements_[size_].~element_t();
        shift_down(0);
        return result;
    }

  private:
    static std::size_t parent_idx(std::size_t i) noexcept { return (i - 1u) / 2u; }
    static std::size_t left_child_idx(std::size_t i) noexcept { return (i * 2u) + 1u; }
    static std::size_t right_child_idx(std::size_t i) noexcept { return (i * 2u) + 2u; }
    static bool less(element_t const& a, element_t const& b) noexcept { return comparator_t {}(a, b); }

    /**
     *  @brief Shifts an element up to maintain the heap property.
     *  @param i Index of the element to be shifted up.
     *
     *  This operation is called when a new element is @b added at the end of the heap.
     *  The element is moved up until the heap property is restored.
     */
    void shift_up(std::size_t i) noexcept {
        for (; i && less(elements_[parent_idx(i)], elements_[i]); i = parent_idx(i))
            std::swap(elements_[parent_idx(i)], elements_[i]);
    }

    /**
     *  @brief Shifts an element down to maintain the heap property.
     *  @param i Index of the element to be shifted down.
     *
     *  This operation is called when the root element is @b removed and the last element is moved to the root.
     *  The element is moved down until the heap property is restored.
     */
    void shift_down(std::size_t i) noexcept {
        std::size_t max_idx = i;

        std::size_t left = left_child_idx(i);
        if (left < size_ && less(elements_[max_idx], elements_[left])) max_idx = left;

        std::size_t right = right_child_idx(i);
        if (right < size_ && less(elements_[max_idx], elements_[right])) max_idx = right;

        if (i != max_idx) {
            std::swap(elements_[i], elements_[max_idx]);
            shift_down(max_idx);
        }
    }
};

/**
 *  @brief Similar to @b `std::priority_queue`, but allows raw access to underlying
 *      memory and always keeps the data sorted. Ideal for small collections
 *      under 128 elements.
 */
template <                                                    //
    typename element_type_,                                   //
    typename comparator_type_ = std::less<void>,              // <void> is needed before C++14.
    typename allocator_type_ = std::allocator<element_type_>> //
class sorted_buffer {
  public:
    using element_t = element_type_;
    using comparator_t = comparator_type_;
    using allocator_t = allocator_type_;

    static_assert(std::is_trivially_destructible<element_t>(), "This heap is designed for trivial structs");
    static_assert(std::is_trivially_copy_constructible<element_t>(), "This heap is designed for trivial structs");

    using value_type = element_t;

  private:
    element_t* elements_;
    std::size_t size_;
    std::size_t capacity_;

  public:
    sorted_buffer() noexcept : elements_(nullptr), size_(0), capacity_(0) {}

    sorted_buffer(sorted_buffer&& other) noexcept
        : elements_(exchange(other.elements_, nullptr)), size_(exchange(other.size_, 0)),
          capacity_(exchange(other.capacity_, 0)) {}

    sorted_buffer& operator=(sorted_buffer&& other) noexcept {
        std::swap(elements_, other.elements_);
        std::swap(size_, other.size_);
        std::swap(capacity_, other.capacity_);
        return *this;
    }

    sorted_buffer(sorted_buffer const&) = delete;
    sorted_buffer& operator=(sorted_buffer const&) = delete;

    ~sorted_buffer() noexcept { reset(); }

    void reset() noexcept {
        if (elements_) allocator_t {}.deallocate(elements_, capacity_);
        elements_ = nullptr;
        capacity_ = 0;
        size_ = 0;
    }

    inline bool empty() const noexcept { return !size_; }
    inline std::size_t size() const noexcept { return size_; }
    inline std::size_t capacity() const noexcept { return capacity_; }
    inline element_t const& top() const noexcept { return elements_[size_ - 1]; }
    inline void clear() noexcept { size_ = 0; }

    bool reserve(std::size_t new_capacity) noexcept {
        if (new_capacity <= capacity_) return true;

        new_capacity = ceil2(new_capacity);
        if (new_capacity == 0) return false;
        new_capacity = (std::max<std::size_t>)(new_capacity, (std::max<std::size_t>)(capacity_ * 2u, 16u));
        auto allocator = allocator_t {};
        auto new_elements = allocator.allocate(new_capacity);
        if (!new_elements) return false;

        if (size_) std::memcpy(new_elements, elements_, size_ * sizeof(element_t));
        if (elements_) allocator.deallocate(elements_, capacity_);

        elements_ = new_elements;
        capacity_ = new_capacity;
        return true;
    }

    inline void insert_reserved(element_t&& element) noexcept {
        std::size_t slot = size_ ? std::lower_bound(elements_, elements_ + size_, element, &less) - elements_ : 0;
        std::size_t to_move = size_ - slot;
        element_t* source = elements_ + size_ - 1;
        for (; to_move; --to_move, --source) source[1] = source[0];
        elements_[slot] = element;
        size_++;
    }

    /**
     *  @retval `true` if the entry was added, `false` if it wasn't relevant enough.
     */
    inline bool insert(element_t&& element, std::size_t limit) noexcept {
        std::size_t slot = size_ ? std::lower_bound(elements_, elements_ + size_, element, &less) - elements_ : 0;
        if (slot == limit) return false;
        std::size_t to_move = size_ - slot - (size_ == limit);
        element_t* source = elements_ + size_ - 1 - (size_ == limit);
        for (; to_move; --to_move, --source) source[1] = source[0];
        elements_[slot] = element;
        size_ += size_ != limit;
        return true;
    }

    inline element_t pop() noexcept {
        size_--;
        element_t result = elements_[size_];
        elements_[size_].~element_t();
        return result;
    }

    void sort_ascending() noexcept {}
    inline void shrink(std::size_t n) noexcept { size_ = (std::min<std::size_t>)(n, size_); }

    inline element_t* data() noexcept { return elements_; }
    inline element_t const* data() const noexcept { return elements_; }

  private:
    static bool less(element_t const& a, element_t const& b) noexcept { return comparator_t {}(a, b); }
};

/**
 *  @brief Minimalistic hash-set similar to @b `std::unordered_set`, used to track visited nodes during graph
 *      traversal. In our primary usecase, its a sparse alternative to a bit-set.
 *
 *  It doesn't support deletion of separate objects, but supports `clear`-ing all at once.
 *  It expects `reserve` to be called ahead of all insertions, so no resizes are needed.
 *  It also assumes `0xFF...FF` slots to be unused, to simplify the design.
 *  It uses linear probing, the number of slots is always a power of two, and it uses linear-probing
 *  in case of bucket collisions.
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
    /// @brief Number of slots.
    std::size_t capacity_ {};
    /// @brief Number of populated.
    std::size_t count_ {};
    hasher_t hasher_ {};

  public:
    growing_hash_set() noexcept {}
    ~growing_hash_set() noexcept { reset(); }

    explicit operator bool() const noexcept { return slots_; }
    std::size_t size() const noexcept { return count_; }

    void clear() noexcept {
        if (slots_) std::memset((void*)slots_, 0xFF, capacity_ * sizeof(element_t));
        count_ = 0;
    }

    void reset() noexcept {
        if (slots_) allocator_t {}.deallocate((byte_t*)slots_, capacity_ * sizeof(element_t));
        slots_ = nullptr;
        capacity_ = 0;
        count_ = 0;
    }

    growing_hash_set(std::size_t capacity) noexcept
        : slots_((element_t*)allocator_t {}.allocate(ceil2(capacity) * sizeof(element_t))),
          capacity_(slots_ ? ceil2(capacity) : 0u), count_(0u) {
        clear();
    }

    growing_hash_set(growing_hash_set&& other) noexcept {
        slots_ = exchange(other.slots_, nullptr);
        capacity_ = exchange(other.capacity_, 0);
        count_ = exchange(other.count_, 0);
    }

    growing_hash_set& operator=(growing_hash_set&& other) noexcept {
        std::swap(slots_, other.slots_);
        std::swap(capacity_, other.capacity_);
        std::swap(count_, other.count_);
        return *this;
    }

    growing_hash_set(growing_hash_set const&) = delete;
    growing_hash_set& operator=(growing_hash_set const&) = delete;

    /**
     *  @brief Checks if the element is already in the hash-set.
     *  @return `true` if the element is already in the hash-set.
     */
    inline bool test(element_t const& elem) const noexcept {
        std::size_t index = hasher_(elem) & (capacity_ - 1);
        while (slots_[index] != default_free_value<element_t>()) {
            if (slots_[index] == elem) return true;

            index = (index + 1) & (capacity_ - 1);
        }
        return false;
    }

    /**
     *  @brief Inserts an element into the hash-set.
     *  @return Similar to `bitset_gt`, returns the previous value.
     */
    inline bool set(element_t const& elem) noexcept {
        std::size_t index = hasher_(elem) & (capacity_ - 1);
        while (slots_[index] != default_free_value<element_t>()) {
            // Already exists
            if (slots_[index] == elem) return true;

            index = (index + 1) & (capacity_ - 1);
        }
        slots_[index] = elem;
        ++count_;
        return false;
    }

    /**
     *  @brief Extends the capacity of the hash-set.
     *  @return `true` if enough capacity is available, `false` if memory allocation failed.
     */
    bool reserve(std::size_t new_capacity) noexcept {
        new_capacity = (new_capacity * 5u) / 3u;
        if (new_capacity <= capacity_) return true;

        new_capacity = ceil2(new_capacity);
        element_t* new_slots = (element_t*)allocator_t {}.allocate(new_capacity * sizeof(element_t));
        if (!new_slots) return false;

        std::memset((void*)new_slots, 0xFF, new_capacity * sizeof(element_t));
        std::size_t new_count = count_;
        if (count_) {
            for (std::size_t old_index = 0; old_index != capacity_; ++old_index) {
                if (slots_[old_index] == default_free_value<element_t>()) continue;

                std::size_t new_index = hasher_(slots_[old_index]) & (new_capacity - 1);
                while (new_slots[new_index] != default_free_value<element_t>())
                    new_index = (new_index + 1) & (new_capacity - 1);
                new_slots[new_index] = slots_[old_index];
            }
        }

        reset();
        slots_ = new_slots;
        capacity_ = new_capacity;
        count_ = new_count;
        return true;
    }
};

#if defined(__AVX2__)

#endif // __AVX2__

#if defined(__AVX512F__) && defined(__AVX512DQ__) && defined(__AVX512BW__) && defined(__BMI2__)

#endif // AVX-512 feature set

} // namespace usearch
} // namespace unum
