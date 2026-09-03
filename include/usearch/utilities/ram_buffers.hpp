/**
 *  @brief Flat sequential containers for Graph traversals - buffers with an ordering discipline.
 *  @author Ash Vardanian
 *  @file include/usearch/utilities/ram_buffers.hpp
 *  @date April 21, 2026
 *
 *  Standalone header providing baseline sequential structures for Graph traversals, like:
 *
 *  - Implicit @b binary-heaps
 *  - @b Sorted-arrays
 *  - Unsorted flat @b scratch-arrays
 *
 *  Plus the @b arena allocator they wear when their storage is a slice of one shared run rather
 *  than an allocation of their own, and the concepts pinning both faces.
 */
#pragma once

#include <cstddef> // `std::size_t`
#include <cstdint> // `SIZE_MAX`, `std::uint8_t`
#include <cstring> // `std::memcpy`

#include <algorithm>   // `std::lower_bound`, `std::sort_heap`, `std::min`, `std::max`
#include <array>       // `std::array` - batch-of-four overloads
#include <concepts>    // `std::same_as` - the buffer-allocator concept
#include <span>        // `std::span` - the sorted-batch insert argument
#include <functional>  // `std::less`
#include <memory>      // `std::allocator`, `std::unique_ptr` - the arena's chunk chain
#include <new>         // `new (ptr) element_t(value)` for "placement-new"
#include <type_traits> // `std::is_trivially_destructible`, `std::is_integral`
#include <utility>     // `std::swap`, `std::move`, `std::forward`

#include "usearch/utilities/types.hpp"      // `status_t`, `succeeded`, `failed`
#include "usearch/utilities/sanitizers.hpp" // `annotate_container`

namespace unum {
namespace usearch {

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
constexpr std::size_t ceil2(std::size_t value) noexcept {
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

/** @brief The capacity a fresh reservation of @p capacity entries rounds up to: a power of two, sixteen at least. */
constexpr std::size_t reserved_capacity(std::size_t capacity) noexcept {
    return max_of<std::size_t>(ceil2(capacity), 16u);
}

/** @brief The allocator surface these containers use; narrower than the standard's, since none of
 *      them constructs or destroys through it. */
template <typename allocator_>
concept buffer_allocator = requires(allocator_ allocator, typename allocator_::value_type* pointer, std::size_t count) {
    typename allocator_::value_type;
    { allocator.allocate(count) } -> std::same_as<typename allocator_::value_type*>;
    allocator.deallocate(pointer, count);
};

static_assert(buffer_allocator<std::allocator<std::byte>>);

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
     *  @return @c success_k, @c overflow_k when the rounded capacity wraps, @c bad_alloc_k otherwise.
     */
    status_t reserve(std::size_t new_capacity) noexcept {
        if (new_capacity <= capacity_) return status_t::success_k;

        if (ceil2(new_capacity) == 0) return status_t::overflow_k;
        new_capacity = max_of<std::size_t>(reserved_capacity(new_capacity), capacity_ * 2u);
        auto new_elements = allocator_.allocate(new_capacity);
        if (!new_elements) return status_t::bad_alloc_k;

        if (elements_) {
            std::memcpy(new_elements, elements_, size_ * sizeof(element_t));
            allocator_.deallocate(elements_, capacity_);
        }
        elements_ = new_elements;
        capacity_ = new_capacity;
        return status_t::success_k;
    }

    /**
     *  @brief Inserts an element into the heap, growing the storage when needed.
     *  @return @c success_k, or whatever @ref reserve refused with.
     */
    status_t insert(element_t&& element) noexcept {
        if (status_t const grown = reserve(size_ + 1); failed(grown)) return grown;

        insert_reserved(std::move(element));
        return status_t::success_k;
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
     *  @return @c success_k, or whatever @ref reserve refused with.
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
    status_t insert_many(element_t const* new_elements, std::size_t count) noexcept {
        if (count == 0) return status_t::success_k;
        if (status_t const grown = reserve(size_ + count); failed(grown)) return grown;
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
        return status_t::success_k;
    }

    template <std::size_t batch_size_>
    inline status_t insert_many(std::array<element_t, batch_size_> const& batch) noexcept {
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

    /** @brief Bytes a fresh reservation of @p capacity entries occupies. */
    static constexpr std::size_t memory_usage(std::size_t capacity) noexcept {
        return reserved_capacity(capacity) * sizeof(element_t);
    }

    /** @brief Bytes this heap currently occupies. */
    std::size_t memory_usage() const noexcept { return memory_usage(capacity_); }

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

    /** @return @c success_k, @c overflow_k when the rounded capacity wraps, @c bad_alloc_k otherwise. */
    status_t reserve(std::size_t new_capacity) noexcept {
        if (new_capacity <= capacity_) return status_t::success_k;
        if (ceil2(new_capacity) == 0) return status_t::overflow_k;
        new_capacity = max_of<std::size_t>(reserved_capacity(new_capacity), capacity_ * 2u);
        auto new_elements = allocator_.allocate(new_capacity);
        if (!new_elements) return status_t::bad_alloc_k;
        if (elements_) {
            std::memcpy(new_elements, elements_, size_ * sizeof(element_t));
            allocator_.deallocate(elements_, capacity_);
        }
        elements_ = new_elements;
        capacity_ = new_capacity;
        return status_t::success_k;
    }

    status_t insert(element_t&& element) noexcept {
        if (status_t const grown = reserve(size_ + 1); failed(grown)) return grown;
        insert_reserved(std::move(element));
        return status_t::success_k;
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
    status_t insert_many(element_t const* new_elements, std::size_t count) noexcept {
        if (count == 0) return status_t::success_k;
        if (status_t const grown = reserve(size_ + count); failed(grown)) return grown;
        std::memcpy(&elements_[size_], new_elements, count * sizeof(element_t));
        size_ += count;
        bool const use_floyd_rebuild = count >= (size_ - count) / 2 || size_ - count == 0;
        if (use_floyd_rebuild) rebuild_();
        else
            for (std::size_t appended = size_ - count; appended < size_; ++appended) shift_up(appended);
        return status_t::success_k;
    }

    template <std::size_t batch_size_>
    inline status_t insert_many(std::array<element_t, batch_size_> const& batch) noexcept {
        return insert_many(batch.data(), batch_size_);
    }

    /** @brief Drops every entry the comparator orders below @p bar and re-heapifies what remains in
     *      one pass - a frontier keyed further-first sheds everything further than @p bar. */
    std::size_t erase_below(element_t const& bar) noexcept {
        auto below = [&bar](element_t const& entry) noexcept { return less(entry, bar); };
        std::size_t const kept = static_cast<std::size_t>(std::remove_if(elements_, elements_ + size_, below) - elements_);
        std::size_t const erased = size_ - kept;
        size_ = kept;
        rebuild_();
        return erased;
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

    /** @brief Floyd's bottom-up heapify over the whole fill. */
    void rebuild_() noexcept {
        if (size_ < 2) return;
        for (std::size_t parent = (size_ - 2) / arity_ + 1; parent-- > 0;) shift_down(parent);
    }

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

    /** @brief Bytes a fresh reservation of @p capacity entries occupies. */
    static constexpr std::size_t memory_usage(std::size_t capacity) noexcept {
        return reserved_capacity(capacity) * sizeof(element_t);
    }

    /** @brief Bytes this buffer currently occupies. */
    std::size_t memory_usage() const noexcept { return memory_usage(capacity_); }

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

    /** @return @c success_k, @c overflow_k when the rounded capacity wraps, @c bad_alloc_k otherwise. */
    status_t reserve(std::size_t new_capacity) noexcept {
        if (new_capacity <= capacity_) return status_t::success_k;

        if (ceil2(new_capacity) == 0) return status_t::overflow_k;
        new_capacity = max_of<std::size_t>(reserved_capacity(new_capacity), capacity_ * 2u);
        auto new_elements = allocator_.allocate(new_capacity);
        if (!new_elements) return status_t::bad_alloc_k;

        if (size_) std::memcpy(new_elements, elements_, size_ * sizeof(element_t));
        if (elements_) allocator_.deallocate(elements_, capacity_);

        elements_ = new_elements;
        capacity_ = new_capacity;
        return status_t::success_k;
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
 *  @brief Exception-free flat dynamic array: the unsorted sibling of @c sorted_buffer,
 *      serving both trivial per-thread scratch and arrays of heavier objects, as long as
 *      moves & destruction never throw.
 *
 *  Two roles share one type, and the @c _reserved suffix is what tells them apart. The
 *  @b sized-once role - @c resize, @c resize_uninitialized, @c reserve, @c reserve_exactly
 *  and @c push_back - allocates and answers a @c status_t. The @b scratch role -
 *  @c resize_reserved, @c push_back_reserved, @c pop_back and @c clear - never allocates,
 *  never fails, and never touches capacity, so a hot loop refills a buffer it reserved once.
 *
 *  @c reserve rounds up to a power of two and at least doubles, which amortizes a growing
 *  loop but over-allocates a table whose extent is known; @c reserve_exactly is the sizing
 *  form for those, and @c resize is built on it.
 */
template <                                                    //
    typename element_type_,                                   //
    typename allocator_type_ = std::allocator<element_type_>> //
class growing_buffer {
  public:
    using element_t = element_type_;
    using allocator_t = typename std::allocator_traits<allocator_type_>::template rebind_alloc<element_type_>;

    static_assert(std::is_nothrow_move_constructible<element_t>() && std::is_nothrow_destructible<element_t>(),
                  "Elements relocate on growth and die in `reset`, so neither may throw");

    using value_type = element_t;

  private:
    element_t* elements_;
    std::size_t size_;
    std::size_t capacity_;
    allocator_t allocator_;

  public:
    /** @brief Takes the caller's allocator in its own element type; the rebind to @c element_t
     *      happens here, so no consumer spells @c allocator_traits itself. */
    growing_buffer(allocator_type_ const& allocator = {}) noexcept
        : elements_(nullptr), size_(0), capacity_(0), allocator_(allocator) {}

    /** @brief Qualified, since an element type from @c std drags @c std::exchange into the
     *      overload set through argument-dependent lookup and makes the call ambiguous. */
    growing_buffer(growing_buffer&& other) noexcept
        : elements_(usearch::exchange(other.elements_, nullptr)), size_(usearch::exchange(other.size_, 0)),
          capacity_(usearch::exchange(other.capacity_, 0)), allocator_(std::move(other.allocator_)) {}

    growing_buffer& operator=(growing_buffer&& other) noexcept {
        std::swap(elements_, other.elements_);
        std::swap(size_, other.size_);
        std::swap(capacity_, other.capacity_);
        std::swap(allocator_, other.allocator_);
        return *this;
    }

    growing_buffer(growing_buffer const&) = delete;
    growing_buffer& operator=(growing_buffer const&) = delete;

    ~growing_buffer() noexcept { reset(); }

    /** @brief Bytes this buffer currently occupies. */
    std::size_t memory_usage() const noexcept { return capacity_ * sizeof(element_t); }

    void reset() noexcept {
        clear();
        // The annotation contract: full addressability returns before the storage does.
        annotate_container(elements_, elements_ + capacity_, elements_, elements_ + capacity_);
        if (elements_) allocator_.deallocate(elements_, capacity_);
        elements_ = nullptr;
        capacity_ = 0;
    }

    inline void clear() noexcept {
        if constexpr (!std::is_trivially_destructible<element_t>())
            for (std::size_t position = 0; position != size_; ++position) elements_[position].~element_t();
        annotate_container(elements_, elements_ + capacity_, elements_ + size_, elements_);
        size_ = 0;
    }

    inline bool empty() const noexcept { return !size_; }
    inline std::size_t size() const noexcept { return size_; }
    inline std::size_t capacity() const noexcept { return capacity_; }

    inline element_t& operator[](std::size_t position) noexcept { return elements_[position]; }
    inline element_t const& operator[](std::size_t position) const noexcept { return elements_[position]; }
    inline element_t& front() noexcept { return elements_[0]; }
    inline element_t const& front() const noexcept { return elements_[0]; }
    inline element_t& back() noexcept { return elements_[size_ - 1]; }
    inline element_t const& back() const noexcept { return elements_[size_ - 1]; }
    inline element_t* begin() noexcept { return elements_; }
    inline element_t* end() noexcept { return elements_ + size_; }
    inline element_t const* begin() const noexcept { return elements_; }
    inline element_t const* end() const noexcept { return elements_ + size_; }
    inline element_t* data() noexcept { return elements_; }
    inline element_t const* data() const noexcept { return elements_; }

    /** @brief Grows capacity to at least @p new_capacity, rounding up to a power of two and
     *      at least doubling, so a repeatedly-growing loop amortizes. Live elements survive.
     *  @return @c success_k, @c overflow_k when the rounded capacity wraps, @c bad_alloc_k otherwise. */
    status_t reserve(std::size_t new_capacity) noexcept {
        if (new_capacity <= capacity_) return status_t::success_k;

        if (ceil2(new_capacity) == 0) return status_t::overflow_k;
        return reserve_exactly(max_of<std::size_t>(reserved_capacity(new_capacity), capacity_ * 2u));
    }

    /** @brief Grows capacity to exactly @p new_capacity, without the rounding @ref reserve applies -
     *      the sizing form for a table whose extent the caller already knows and then freezes. */
    status_t reserve_exactly(std::size_t new_capacity) noexcept {
        if (new_capacity <= capacity_) return status_t::success_k;
        auto new_elements = allocator_.allocate(new_capacity);
        if (!new_elements) return status_t::bad_alloc_k;

        // A fresh region arrives fully addressable, so poison its slack first; the used
        // prefix stays open for the relocating constructions below.
        annotate_container(new_elements, new_elements + new_capacity, new_elements + new_capacity,
                           new_elements + size_);
        for (std::size_t position = 0; position != size_; ++position) {
            new (new_elements + position) element_t(std::move(elements_[position]));
            elements_[position].~element_t();
        }
        annotate_container(elements_, elements_ + capacity_, elements_ + size_, elements_ + capacity_);
        if (elements_) allocator_.deallocate(elements_, capacity_);

        elements_ = new_elements;
        capacity_ = new_capacity;
        return status_t::success_k;
    }

    /** @brief Reallocates to exactly @p count value-initialized elements, discarding any prior
     *      contents - the sized-once form. Trivial elements are zero-filled in one @c memset. */
    status_t resize(std::size_t count) noexcept {
        reset();
        if (count == 0) return status_t::success_k;
        if (status_t const sized = reserve_exactly(count); failed(sized)) return sized;
        annotate_container(elements_, elements_ + capacity_, elements_ + size_, elements_ + count);
        if constexpr (std::is_trivially_default_constructible<element_t>())
            std::memset(static_cast<void*>(elements_), 0, count * sizeof(element_t));
        else
            for (std::size_t position = 0; position != count; ++position) new (elements_ + position) element_t();
        size_ = count;
        return status_t::success_k;
    }

    /** @brief Like @ref resize, but skips the zero fill so the caller owns the first touch.
     *  @note Trivial element types only - nothing is constructed, so every element must be
     *      written before it is read. */
    status_t resize_uninitialized(std::size_t count) noexcept {
        reset();
        if (count == 0) return status_t::success_k;
        if (status_t const sized = reserve_exactly(count); failed(sized)) return sized;
        extend_uninitialized(count);
        return status_t::success_k;
    }

    /** @brief Moves the fill to @p count within reserved capacity, leaving the bytes untouched -
     *      never allocates, never fails. This is the slice an arena hands out, where value-
     *      constructing the run would cost a pass over every slice handed out. */
    inline void extend_uninitialized(std::size_t count) noexcept {
        static_assert(
            std::is_trivially_default_constructible<element_t>() && std::is_trivially_destructible<element_t>(),
            "Uninitialized storage is only safe for trivial element types");
        annotate_container(elements_, elements_ + capacity_, elements_ + size_, elements_ + count);
        size_ = count;
    }

    /** @brief Appends @p element into pre-reserved capacity - the hot-path form. */
    inline void push_back_reserved(element_t element) noexcept {
        annotate_container(elements_, elements_ + capacity_, elements_ + size_, elements_ + size_ + 1);
        new (elements_ + size_) element_t(std::move(element));
        ++size_;
    }

    /** @brief Appends @p element, doubling capacity when full. @ref push_back_reserved is the
     *      form for a loop that reserved its whole extent up front. */
    inline status_t push_back(element_t element) noexcept {
        if (size_ == capacity_)
            if (status_t const grown = reserve(size_ + 1); failed(grown)) return grown;
        push_back_reserved(std::move(element));
        return status_t::success_k;
    }

    inline element_t pop_back() noexcept {
        element_t popped = std::move(elements_[--size_]);
        elements_[size_].~element_t();
        annotate_container(elements_, elements_ + capacity_, elements_ + size_ + 1, elements_ + size_);
        return popped;
    }

    /** @brief Moves the live fill to @p count within reserved capacity, value-constructing
     *      growth and destroying shrinkage; never allocates. Instantiates only for
     *      default-constructible elements. */
    inline void resize_reserved(std::size_t count) noexcept {
        std::size_t const previous = size_;
        for (; size_ > count; --size_) elements_[size_ - 1].~element_t();
        annotate_container(elements_, elements_ + capacity_, elements_ + previous, elements_ + count);
        for (; size_ < count; ++size_) new (elements_ + size_) element_t();
    }
};

/** @brief Per-thread bump arena: reservations bump inside the newest chunk, and one that cannot
 *      fit them stays alive under its slices while a chunk at least twice as wide opens - so
 *      addresses never move and slices outlive growth. The rewind keeps the widest chunk and
 *      drops the ones it outgrew; releasing a slice does nothing, the rewind is the release. */
template <typename upstream_allocator_ = std::allocator<std::byte>>
class growing_arena {
    using upstream_t = typename std::allocator_traits<upstream_allocator_>::template rebind_alloc<std::byte>;
    using run_t = growing_buffer<std::byte, upstream_t>;

    /** @brief One chunk: its run, and the chunk it outgrew. */
    struct chunk_t {
        run_t run;
        std::unique_ptr<chunk_t> previous;
    };

    /** @brief The newest chunk, the one reservations bump; null before the first reservation. */
    std::unique_ptr<chunk_t> newest_;
    /** @brief Places every run - the thread's memory domain. */
    upstream_t upstream_;

  public:
    explicit growing_arena(upstream_t upstream = {}) noexcept : upstream_(upstream) {}
    growing_arena(growing_arena&&) noexcept = default;
    growing_arena& operator=(growing_arena&&) noexcept = default;

    /** @brief Bytes across chunks. */
    std::size_t capacity() const noexcept {
        std::size_t bytes = 0;
        for (chunk_t const* chunk = newest_.get(); chunk; chunk = chunk->previous.get()) bytes += chunk->run.capacity();
        return bytes;
    }

    /** @brief Opens the seed chunk at @p bytes, the width every later chunk doubles from; a
     *      seeded arena answers in one compare. */
    status_t reserve_exactly(std::size_t bytes) noexcept {
        if (newest_) return status_t::success_k;
        return grow_(bytes);
    }

    /** @brief Hands out @p bytes on an @p alignment boundary from the newest chunk, growing by one
     *      at least twice as wide when it cannot fit them; null when the upstream refuses. */
    std::byte* allocate(std::size_t bytes, std::size_t alignment) noexcept {
        if (newest_)
            if (std::byte* const slice = slice_(newest_->run, bytes, alignment)) return slice;
        std::size_t const last_width = newest_ ? newest_->run.capacity() : 0;
        if (failed(grow_(max_of(last_width * 2u, bytes + alignment)))) return nullptr;
        return slice_(newest_->run, bytes, alignment);
    }

    /** @brief Rewinds the newest chunk and drops the ones it outgrew, so the next call starts at
     *      the high-water mark. */
    void clear() noexcept {
        if (!newest_) return;
        newest_->previous.reset();
        newest_->run.clear();
    }

    /** @brief Releases every chunk. */
    void reset() noexcept { newest_.reset(); }

  private:
    /** @brief Grows by a chunk of @p width bytes, the newest from now on, the old one chained behind. */
    status_t grow_(std::size_t width) noexcept {
        std::unique_ptr<chunk_t> grown {new (std::nothrow) chunk_t {run_t {upstream_}, nullptr}};
        if (!grown) return status_t::bad_alloc_k;
        if (status_t const sized = grown->run.reserve_exactly(width); failed(sized)) return sized;
        grown->previous = std::move(newest_);
        newest_ = std::move(grown);
        return status_t::success_k;
    }

    /** @brief The address-aligned slice at the end of @p run, or null when it cannot fit @p bytes. */
    static std::byte* slice_(run_t& run, std::size_t bytes, std::size_t alignment) noexcept {
        // Aligned by address, not by offset: the upstream only promises what `malloc` does.
        std::uintptr_t const cursor = reinterpret_cast<std::uintptr_t>(run.data() + run.size());
        std::size_t const padding =
            static_cast<std::size_t>(divide_round_up<std::uintptr_t>(cursor, alignment) * alignment - cursor);
        if (run.capacity() - run.size() < padding + bytes) return nullptr;
        run.extend_uninitialized(run.size() + padding + bytes);
        return run.data() + run.size() - bytes;
    }
};

/** @brief Hands out slices of one @ref growing_arena it points at but does not own. Copies share
 *      the arena by pointer, so containers built from one reserve disjoint slices; releasing a
 *      slice does nothing - the arena's rewind is the release. Every slice starts on an
 *      @p alignment_ boundary, since the arena's raw bytes promise only what @c malloc does and
 *      the containers built over a slice issue the widest loads their kernels know. */
template <typename value_type_ = std::byte, typename upstream_allocator_ = std::allocator<std::byte>,
          std::size_t alignment_ = default_alignment_k>
struct arena_allocator {
    using value_type = value_type_;
    using size_type = std::size_t;
    using arena_t = growing_arena<upstream_allocator_>;

    /** @brief Spelled out because the alignment's non-type parameter defeats the traits'
     *      automatic first-argument replacement; the upstream and the alignment ride along. */
    template <typename other_type_>
    struct rebind {
        using other = arena_allocator<other_type_, upstream_allocator_, alignment_>;
    };

    /** @brief The boundary every reservation starts on; echoes the template argument. */
    inline static constexpr std::size_t alignment_k = alignment_;

    /** @brief Bytes a reservation of @p bytes occupies here, the next one starting on its own boundary. */
    static constexpr std::size_t reserved_bytes(std::size_t bytes) noexcept {
        return round_up_to_multiple(bytes, alignment_k);
    }

    /** @brief The arena every rebound copy shares; null refuses every request. */
    arena_t* arena = nullptr;

    constexpr arena_allocator() noexcept = default;
    explicit constexpr arena_allocator(arena_t& arena) noexcept : arena(&arena) {}

    /** @brief The rebinding conversion @c std::allocator_traits performs - the upstream rides
     *      along, and the arena is shared rather than copied. */
    template <typename other_type_>
    constexpr arena_allocator(arena_allocator<other_type_, upstream_allocator_, alignment_> const& other) noexcept
        : arena(other.arena) {}

    value_type_* allocate(std::size_t count) noexcept {
        if (!arena) return nullptr;
        return reinterpret_cast<value_type_*>(arena->allocate(count * sizeof(value_type_), alignment_k));
    }
    void deallocate(value_type_*, std::size_t) noexcept {}
    bool operator==(arena_allocator const& other) const noexcept { return arena == other.arena; }
};

static_assert(buffer_allocator<arena_allocator<std::byte>>);
static_assert(std::same_as<std::allocator_traits<arena_allocator<std::byte>>::rebind_alloc<int>, //
                           arena_allocator<int>>);

} // namespace usearch
} // namespace unum
