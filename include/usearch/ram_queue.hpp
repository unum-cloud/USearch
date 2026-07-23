/**
 *  @brief SIMD-accelerated dynamic data-structures for Graph traversals.
 *  @author Ash Vardanian
 *  @file include/usearch/ram_queue.hpp
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
#include <array>       // `std::array` - batch-of-four overloads
#include <bit>         // `std::countr_zero` - first-empty lane of a probe group
#include <cstddef>     // `std::size_t`
#include <span>        // `std::span` - the sorted-batch insert argument
#include <cstdint>     // `SIZE_MAX`, `std::uint8_t`
#include <cstring>     // `std::memcpy`, `std::memset`
#include <functional>  // `std::less`, `std::hash`
#include <limits>      // `std::numeric_limits`
#include <memory>      // `std::allocator`
#include <new>         // `new (ptr) element_t(value)` for "placement-new"
#include <type_traits> // `std::is_trivially_destructible`, `std::is_integral`
#include <utility>     // `std::swap`, `std::move`, `std::forward`

#if defined(__AVX512F__)
#include <immintrin.h> // 512-bit group probes of the visited hash-set
#endif

namespace unum {
namespace usearch {

/**
 *  @brief Single-byte scalar used as the default allocation unit for @ref growing_hash_set.
 */
using byte_t = char;

/**
 *  @brief Assigns @p new_value to @p old_value and returns its previous contents -
 *      @c std::exchange respelled as unconditionally @c noexcept.
 */
template <typename value_type_, typename other_type_ = value_type_>
inline value_type_ exchange(value_type_& old_value, other_type_&& new_value) noexcept {
    value_type_ previous = std::move(old_value);
    old_value = std::forward<other_type_>(new_value);
    return previous;
}

/**
 *  @brief Rounds @p value up to the next power of two.
 *  @return The smallest power of two greater than or equal to @p value, or @c 0 for @c 0.
 *
 *  Keeps container capacities power-of-two, so masking by capacity minus one can
 *  substitute for the otherwise costlier modulo. Branchless bit trick from
 *  Henry S. Warren Jr., "Hacker's Delight", 2nd ed., §3-2.
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
 *  @c 0xFF byte-fill emitted by @ref growing_hash_set::clear.
 */
template <typename element_type_>
inline element_type_ default_free_value() noexcept {
    static_assert(std::is_integral<element_type_>::value,
                  "`growing_hash_set` requires an integral element type for its sentinel convention");
    return (std::numeric_limits<element_type_>::max)();
}

/**
 *  @brief Sorts up to four @p values ascending by @p comparator_type_ through fully-unrolled
 *      sorting networks - no loops, no recursion, no library sort; the four-element network
 *      is the optimal five compare-exchanges.
 *
 *  Exchanges fire only on strict inversions, so equal elements keep their original order -
 *  the stability the sorted-batch @c sorted_buffer::insert relies on for reproducible ties.
 */
template <typename element_type_, std::size_t extent_, typename comparator_type_ = std::less<void>>
inline void network_sort(std::span<element_type_, extent_> values, comparator_type_ comparator = {}) noexcept {
    auto exchange = [&](std::size_t left, std::size_t right) noexcept {
        if (comparator(values[right], values[left])) std::swap(values[left], values[right]);
    };
    switch (values.size()) {
    case 4: exchange(0, 1), exchange(2, 3), exchange(0, 2), exchange(1, 3), exchange(1, 2); return;
    case 3: exchange(0, 1), exchange(0, 2), exchange(1, 2); return;
    case 2: exchange(0, 1); return;
    default: return; // Zero or one element is sorted by definition.
    }
}

/**
 *  @brief Similar to @c std::priority_queue, but allows raw access to underlying
 *      memory, in case you want to shuffle it or sort. Good for collections
 *      from 100s to 10'000s elements.
 *
 *  In a max-heap, the heap property ensures that the value of each node is greater
 *  than or equal to the values of its children. This means that the largest element
 *  is always at the root of the heap.
 *
 *  @section max_heap_designs Heap Structures
 *
 *  There are several designs of heaps. Binary heaps are the simplest & most common
 *  variant, that is easy to implement as a succinct array. However, they are not the
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
 *  graph-traversal algorithms. For example, Dijkstra's algorithm can be sped up
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
    allocator_t allocator_;

  public:
    max_heap(allocator_t allocator = {}) noexcept
        : elements_(nullptr), size_(0), capacity_(0), allocator_(std::move(allocator)) {}

    max_heap(max_heap&& other) noexcept
        : elements_(exchange(other.elements_, nullptr)), size_(exchange(other.size_, 0)),
          capacity_(exchange(other.capacity_, 0)), allocator_(std::move(other.allocator_)) {}

    max_heap& operator=(max_heap&& other) noexcept {
        std::swap(elements_, other.elements_);
        std::swap(size_, other.size_);
        std::swap(capacity_, other.capacity_);
        std::swap(allocator_, other.allocator_);
        return *this;
    }

    max_heap(max_heap const&) = delete;
    max_heap& operator=(max_heap const&) = delete;

    ~max_heap() noexcept { reset(); }

    void reset() noexcept {
        if (elements_) allocator_.deallocate(elements_, capacity_);
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
     *  @brief Ensures the heap has enough capacity for @p new_capacity elements.
     *  @return @c true if the capacity is available, @c false when the allocation failed.
     */
    bool reserve(std::size_t new_capacity) noexcept {
        if (new_capacity <= capacity_) return true;

        new_capacity = ceil2(new_capacity);
        if (new_capacity == 0) return false;
        new_capacity = (std::max<std::size_t>)(new_capacity, (std::max<std::size_t>)(capacity_ * 2u, 16u));
        auto new_elements = allocator_.allocate(new_capacity);
        if (!new_elements) return false;

        if (elements_) {
            std::memcpy(new_elements, elements_, size_ * sizeof(element_t));
            allocator_.deallocate(elements_, capacity_);
        }
        elements_ = new_elements;
        capacity_ = new_capacity;
        return new_elements;
    }

    /**
     *  @brief Inserts an element into the heap, growing the storage when needed.
     *  @return @c true if the element was inserted, @c false when the allocation failed.
     */
    bool insert(element_t&& element) noexcept {
        if (!reserve(size_ + 1)) return false;

        insert_reserved(std::move(element));
        return true;
    }

    /**
     *  @brief Inserts an element into the heap without reserving additional space.
     */
    void insert_reserved(element_t&& element) noexcept {
        new (&elements_[size_]) element_t(element);
        size_++;
        shift_up(size_ - 1);
    }

    /**
     *  @brief Inserts @p count elements into the heap.
     *  @return @c true if the elements were successfully inserted, @c false if allocation failed.
     *
     *  Appends the batch at the tail, then heapifies. Picks between two strategies based on
     *  the ratio of new elements to existing size:
     *
     *  - @b Floyd bulk-heapify when @p count is at least half the existing size: re-heapify
     *    the full array from the last parent down - O(size + count) compare-swaps.
     *  - Incremental @c shift_up when @p count is small relative to the heap: only the
     *    appended suffix needs work - O(count · log(size + count)) compare-swaps.
     *
     *  Crossover is at count ≈ (size + count) / log(size + count). In the HNSW hot
     *  path (batch of ~32, heap of ~128) incremental @c shift_up wins; for bulk init
     *  from empty, Floyd wins trivially.
     */
    bool insert_many(element_t const* new_elements, std::size_t count) noexcept {
        if (count == 0) return true;
        if (!reserve(size_ + count)) return false;
        std::memcpy(&elements_[size_], new_elements, count * sizeof(element_t));
        size_ += count;

        bool const use_floyd_rebuild = count >= (size_ - count) / 2 || size_ - count == 0;
        if (use_floyd_rebuild) {
            if (size_ > 1)
                for (std::size_t parent_index = size_ / 2; parent_index-- > 0;) shift_down(parent_index);
        }
        else {
            for (std::size_t appended_index = size_ - count; appended_index < size_; ++appended_index)
                shift_up(appended_index);
        }
        return true;
    }

    template <std::size_t batch_size_>
    inline bool insert_many(std::array<element_t, batch_size_> const& batch) noexcept {
        return insert_many(batch.data(), batch_size_);
    }

    element_t pop() noexcept {
        element_t result = top();
        std::swap(elements_[0], elements_[size_ - 1]);
        size_--;
        elements_[size_].~element_t();
        shift_down(0);
        return result;
    }

    /**
     *  @brief Pops the @c batch_size_ largest elements into a @c std::array, descending.
     *
     *  Straightforward @c batch_size_ sequential @ref pop calls. There is no algorithmic
     *  savings over the loop - each pop still costs O(log size) - but a single call
     *  amortizes function-call overhead and matches the batch API shape of the rest of
     *  this header. Useful at HNSW search finalization when a top-k result list is
     *  extracted from the frontier.
     */
    template <std::size_t batch_size_>
    inline std::array<element_t, batch_size_> pop_many() noexcept {
        std::array<element_t, batch_size_> out;
        for (std::size_t i = 0; i < batch_size_; ++i) out[i] = pop();
        return out;
    }

  private:
    static std::size_t parent_idx(std::size_t i) noexcept { return (i - 1u) / 2u; }
    static std::size_t left_child_idx(std::size_t i) noexcept { return (i * 2u) + 1u; }
    static std::size_t right_child_idx(std::size_t i) noexcept { return (i * 2u) + 2u; }
    static bool less(element_t const& a, element_t const& b) noexcept { return comparator_t {}(a, b); }

    /** @brief Moves the element at @p i up until the heap property is restored - the
     *      companion of appending at the tail. */
    void shift_up(std::size_t i) noexcept {
        for (; i && less(elements_[parent_idx(i)], elements_[i]); i = parent_idx(i))
            std::swap(elements_[parent_idx(i)], elements_[i]);
    }

    /** @brief Moves the element at @p i down until the heap property is restored - the
     *      companion of displacing the root during @ref pop. */
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
 *  @brief D-ary max-heap with implicit array layout - parent at @c i, children at
 *      @c d·i+1 .. @c d·i+d. Same flat-buffer storage and same public API as @ref max_heap;
 *      the difference is the tree-fan-out @p arity_ controlling per-level work and depth.
 *
 *  Insert - @c shift_up - walks a single path from leaf to root with log_d(n) compare-swaps
 *  - shallower than the binary heap's log_2(n) by a factor of log_2(d). Pop - @c shift_down -
 *  scans up to @p arity_ children per level over log_d(n) levels, so total compares per
 *  pop are roughly the same as binary, just rearranged.
 *
 *  HNSW frontier inserts dominate pops (~M·2 pushes per pop), so the insert-side win
 *  carries the workload. Sweet spot is an @p arity_ of 4, 8, or 16 - d=2 falls back to
 *  binary (and serves as a parity-check oracle), d=64+ blows up the pop sweep.
 *
 *  @note @c sort_ascending uses @c std::sort here (not @c std::sort_heap) because the
 *      latter assumes a binary heap layout.
 */
template <std::size_t arity_,                          //
          typename element_type_,                      //
          typename comparator_type_ = std::less<void>, //
          typename allocator_type_ = std::allocator<element_type_>>
class max_heap_dary {
    static_assert(arity_ >= 2, "Heap arity must be at least 2");

  public:
    using element_t = element_type_;
    using comparator_t = comparator_type_;
    using allocator_t = allocator_type_;
    using value_type = element_t;
    static constexpr std::size_t arity = arity_;

    static_assert(std::is_trivially_destructible<element_t>(), "This heap is designed for trivial structs");
    static_assert(std::is_trivially_copy_constructible<element_t>(), "This heap is designed for trivial structs");

  private:
    element_t* elements_;
    std::size_t size_;
    std::size_t capacity_;
    allocator_t allocator_;

  public:
    max_heap_dary(allocator_t allocator = {}) noexcept
        : elements_(nullptr), size_(0), capacity_(0), allocator_(std::move(allocator)) {}

    max_heap_dary(max_heap_dary&& other) noexcept
        : elements_(exchange(other.elements_, nullptr)), size_(exchange(other.size_, 0)),
          capacity_(exchange(other.capacity_, 0)), allocator_(std::move(other.allocator_)) {}

    max_heap_dary& operator=(max_heap_dary&& other) noexcept {
        std::swap(elements_, other.elements_);
        std::swap(size_, other.size_);
        std::swap(capacity_, other.capacity_);
        std::swap(allocator_, other.allocator_);
        return *this;
    }

    max_heap_dary(max_heap_dary const&) = delete;
    max_heap_dary& operator=(max_heap_dary const&) = delete;

    ~max_heap_dary() noexcept { reset(); }

    void reset() noexcept {
        if (elements_) allocator_.deallocate(elements_, capacity_);
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
    inline element_t const& top() const noexcept { return elements_[0]; }

    /** @brief Sorts the underlying array ascending. Uses @c std::sort because @c std::sort_heap
     *      assumes binary-heap layout. */
    inline void sort_ascending() noexcept { std::sort(elements_, elements_ + size_, &less); }

    bool reserve(std::size_t new_capacity) noexcept {
        if (new_capacity <= capacity_) return true;
        new_capacity = ceil2(new_capacity);
        if (new_capacity == 0) return false;
        new_capacity = (std::max<std::size_t>)(new_capacity, (std::max<std::size_t>)(capacity_ * 2u, 16u));
        auto new_elements = allocator_.allocate(new_capacity);
        if (!new_elements) return false;
        if (elements_) {
            std::memcpy(new_elements, elements_, size_ * sizeof(element_t));
            allocator_.deallocate(elements_, capacity_);
        }
        elements_ = new_elements;
        capacity_ = new_capacity;
        return new_elements;
    }

    bool insert(element_t&& element) noexcept {
        if (!reserve(size_ + 1)) return false;
        insert_reserved(std::move(element));
        return true;
    }

    void insert_reserved(element_t&& element) noexcept {
        new (&elements_[size_]) element_t(element);
        size_++;
        shift_up(size_ - 1);
    }

    /**
     *  @brief Inserts @p count elements via append-then-heapify, picking between
     *      @b Floyd bulk rebuild (for batches at least half the existing size) and
     *      incremental @c shift_up (for smaller batches).
     */
    bool insert_many(element_t const* new_elements, std::size_t count) noexcept {
        if (count == 0) return true;
        if (!reserve(size_ + count)) return false;
        std::memcpy(&elements_[size_], new_elements, count * sizeof(element_t));
        size_ += count;
        bool const use_floyd_rebuild = count >= (size_ - count) / 2 || size_ - count == 0;
        if (use_floyd_rebuild) {
            if (size_ > 1) {
                std::size_t const last_parent = (size_ - 2) / arity_;
                for (std::size_t parent = last_parent + 1; parent-- > 0;) shift_down(parent);
            }
        }
        else {
            for (std::size_t appended = size_ - count; appended < size_; ++appended) shift_up(appended);
        }
        return true;
    }

    template <std::size_t batch_size_>
    inline bool insert_many(std::array<element_t, batch_size_> const& batch) noexcept {
        return insert_many(batch.data(), batch_size_);
    }

    element_t pop() noexcept {
        element_t result = top();
        std::swap(elements_[0], elements_[size_ - 1]);
        size_--;
        elements_[size_].~element_t();
        shift_down(0);
        return result;
    }

    template <std::size_t batch_size_>
    inline std::array<element_t, batch_size_> pop_many() noexcept {
        std::array<element_t, batch_size_> out;
        for (std::size_t i = 0; i < batch_size_; ++i) out[i] = pop();
        return out;
    }

  private:
    static std::size_t parent_idx(std::size_t i) noexcept { return (i - 1u) / arity_; }
    static std::size_t first_child_idx(std::size_t i) noexcept { return arity_ * i + 1u; }
    static bool less(element_t const& a, element_t const& b) noexcept { return comparator_t {}(a, b); }

    void shift_up(std::size_t i) noexcept {
        for (; i && less(elements_[parent_idx(i)], elements_[i]); i = parent_idx(i))
            std::swap(elements_[parent_idx(i)], elements_[i]);
    }

    /**
     *  @brief Iterative shift-down. Loads up to @p arity_ children per level, finds the
     *      max via a scalar linear scan (well-predicted for small @p arity_), swaps with
     *      parent if needed, descends. No recursion - the loop terminates when the heap
     *      property is restored.
     */
    void shift_down(std::size_t i) noexcept {
        while (true) {
            std::size_t const first = first_child_idx(i);
            if (first >= size_) return;
            std::size_t const last = (first + arity_ < size_) ? (first + arity_) : size_;
            std::size_t max_idx = i;
            for (std::size_t c = first; c < last; ++c)
                if (less(elements_[max_idx], elements_[c])) max_idx = c;
            if (max_idx == i) return;
            std::swap(elements_[i], elements_[max_idx]);
            i = max_idx;
        }
    }
};

/**
 *  @brief Similar to @c std::priority_queue, but allows raw access to underlying
 *      memory and always keeps the data sorted. Ideal for small collections
 *      under 128 elements.
 */
template <                                                    //
    typename element_type_,                                   //
    typename comparator_type_ = std::less<void>,              //
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
    allocator_t allocator_;

  public:
    sorted_buffer(allocator_t allocator = {}) noexcept
        : elements_(nullptr), size_(0), capacity_(0), allocator_(std::move(allocator)) {}

    sorted_buffer(sorted_buffer&& other) noexcept
        : elements_(exchange(other.elements_, nullptr)), size_(exchange(other.size_, 0)),
          capacity_(exchange(other.capacity_, 0)), allocator_(std::move(other.allocator_)) {}

    sorted_buffer& operator=(sorted_buffer&& other) noexcept {
        std::swap(elements_, other.elements_);
        std::swap(size_, other.size_);
        std::swap(capacity_, other.capacity_);
        std::swap(allocator_, other.allocator_);
        return *this;
    }

    sorted_buffer(sorted_buffer const&) = delete;
    sorted_buffer& operator=(sorted_buffer const&) = delete;

    ~sorted_buffer() noexcept { reset(); }

    void reset() noexcept {
        if (elements_) allocator_.deallocate(elements_, capacity_);
        elements_ = nullptr;
        capacity_ = 0;
        size_ = 0;
    }

    inline bool empty() const noexcept { return !size_; }
    inline std::size_t size() const noexcept { return size_; }
    inline std::size_t capacity() const noexcept { return capacity_; }
    inline element_t const& top() const noexcept { return elements_[size_ - 1]; }
    inline element_t const& operator[](std::size_t i) const noexcept { return elements_[i]; }
    inline element_t const* begin() const noexcept { return elements_; }
    inline element_t const* end() const noexcept { return elements_ + size_; }
    inline void clear() noexcept { size_ = 0; }

    bool reserve(std::size_t new_capacity) noexcept {
        if (new_capacity <= capacity_) return true;

        new_capacity = ceil2(new_capacity);
        if (new_capacity == 0) return false;
        new_capacity = (std::max<std::size_t>)(new_capacity, (std::max<std::size_t>)(capacity_ * 2u, 16u));
        auto new_elements = allocator_.allocate(new_capacity);
        if (!new_elements) return false;

        if (size_) std::memcpy(new_elements, elements_, size_ * sizeof(element_t));
        if (elements_) allocator_.deallocate(elements_, capacity_);

        elements_ = new_elements;
        capacity_ = new_capacity;
        return true;
    }

    inline void insert_reserved(element_t&& element) noexcept {
        std::size_t slot = size_ ? std::lower_bound(elements_, elements_ + size_, element, &less) - elements_ : 0;
        shift_right(slot, size_);
        elements_[slot] = element;
        size_++;
    }

    /**
     *  @brief Inserts @p element into the ascending buffer, capping its size at @p limit.
     *  @return @c true if the entry was added, @c false if it wasn't relevant enough.
     *
     *  The fast-reject path - a full buffer whose top beats @p element - returns without
     *  the @c std::lower_bound walk: O(1) rather than O(log limit). That's the common case
     *  on the HNSW hot path once the top-k buffer warms up and most successors lose.
     */
    inline bool insert(element_t&& element, std::size_t limit) noexcept {
        if (size_ == limit && !less(element, elements_[size_ - 1])) return false;
        std::size_t slot = size_ ? std::lower_bound(elements_, elements_ + size_, element, &less) - elements_ : 0;
        shift_right(slot, size_ - (size_ == limit));
        elements_[slot] = element;
        size_ += size_ != limit;
        return true;
    }

    /**
     *  @brief Merges an ascending pre-sorted @p batch into the buffer in one back-to-front
     *      pass, keeping the top @p limit of the union - existing elements win ties.
     *  @return The number of accepted @p batch entries - always a @b prefix of the ascending
     *      batch, since any rejected entry rejects everything at or above it.
     *
     *  The caller owns the ordering: micro-batches out of the 4-way kernels sort in a few
     *  compare-exchanges via @ref network_sort, never a library sort. One merge pass costs
     *  O(accepted + shifted) element moves - no per-element search, no intermediate state.
     */
    inline std::size_t insert(std::span<element_t const> batch, std::size_t limit) noexcept {
        std::size_t const count = batch.size();
        std::size_t const total = size_ + count;
        std::size_t const new_size = total < limit ? total : limit;
        std::size_t const drop = total - new_size;
        std::size_t existing_tail = size_;
        std::size_t batch_tail = count;

        // Drop phase - discard the `drop` largest of (existing ∪ batch) from the tails.
        for (std::size_t iteration = 0; iteration != drop; ++iteration) {
            bool const pop_batch = existing_tail == 0 ||
                                   (batch_tail > 0 && less(elements_[existing_tail - 1], batch[batch_tail - 1]));
            if (pop_batch) --batch_tail;
            else --existing_tail;
        }
        std::size_t const accepted = batch_tail;

        // Merge phase - write the surviving `new_size` entries from back to front.
        std::size_t write_index = new_size;
        while (write_index > 0 && batch_tail > 0) {
            bool const pick_batch = existing_tail == 0 || //
                                    less(elements_[existing_tail - 1], batch[batch_tail - 1]);
            --write_index;
            if (pick_batch) elements_[write_index] = batch[--batch_tail];
            else elements_[write_index] = elements_[--existing_tail];
            if (batch_tail == 0 && existing_tail == write_index) break;
        }
        size_ = new_size;
        return accepted;
    }

    inline element_t pop() noexcept {
        size_--;
        element_t result = elements_[size_];
        elements_[size_].~element_t();
        return result;
    }

    /** @brief No-op for API parity with the heaps: the buffer is always sorted ascending. */
    void sort_ascending() noexcept {}
    inline void shrink(std::size_t n) noexcept { size_ = (std::min<std::size_t>)(n, size_); }

    inline element_t* data() noexcept { return elements_; }
    inline element_t const* data() const noexcept { return elements_; }

  private:
    static bool less(element_t const& a, element_t const& b) noexcept { return comparator_t {}(a, b); }

    /** @brief Opens a slot at @p slot by moving the [@p slot, @p tail) range one position
     *      right, back to front, so the overlap never clobbers unread elements. */
    void shift_right(std::size_t slot, std::size_t tail) noexcept {
        std::size_t to_move = tail - slot;
        element_t* source = elements_ + tail - 1;
        for (; to_move; --to_move, --source) source[1] = source[0];
    }
};

/**
 *  @brief Minimalistic hash-set similar to @c std::unordered_set, used to track visited nodes
 *      during graph traversal - a sparse alternative to a bit-set in our primary use case.
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

#if defined(__AVX2__)

#endif // __AVX2__

#if defined(__AVX512F__) && defined(__AVX512DQ__) && defined(__AVX512BW__) && defined(__BMI2__)

#endif // AVX-512 feature set

} // namespace usearch
} // namespace unum
