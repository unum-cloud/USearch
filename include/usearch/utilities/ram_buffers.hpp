/**
 *  @file include/usearch/utilities/ram_buffers.hpp
 *  @author Ash Vardanian
 *  @date April 21, 2026
 *  @brief Flat sequential containers for Graph traversals - buffers with an ordering discipline.
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
#include <cstring> // `std::memcpy`, `std::memset` - a fresh chunk starts zeroed

#include <array>       // `std::array` - batch-of-four overloads, the chunk directories, the arena's runs
#include <bit>         // `std::bit_width`, `std::bit_floor` - the closed-form chunk location
#include <concepts>    // `std::same_as` - the buffer-allocator concept
#include <span>        // `std::span` - the sorted-batch insert argument
#include <functional>  // `std::less`
#include <limits>      // `std::numeric_limits` - the arena's chunk bound
#include <memory>      // `std::allocator`, `std::destroy_n` - built rows
#include <new>         // `new (ptr) element_t(value)` for "placement-new"
#include <type_traits> // `std::is_trivially_destructible`, `std::is_trivially_copyable`, `std::is_nothrow_*`
#include <utility>     // `std::swap`, `std::move`, `std::forward`

#include "usearch/utilities/types.hpp"      // `status_t`, `succeeded`, `failed`
#include "usearch/utilities/sanitizers.hpp" // `annotate_container`

namespace unum::usearch::utilities {

/** Assigns @p new_value to @p old_value and returns its previous contents -
 *  @c std::exchange respelled as unconditionally @c noexcept. */
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

/** The capacity a fresh reservation of @p capacity entries rounds up to: a power of two, sixteen at least. */
constexpr std::size_t reserved_capacity(std::size_t capacity) noexcept {
    return max_of<std::size_t>(ceil2(capacity), 16u);
}

/** The allocator surface these containers use; narrower than the standard's, since none of
 *  them constructs or destroys through it. */
template <typename allocator_>
concept buffer_allocator = requires(allocator_ allocator, typename allocator_::value_type* pointer, std::size_t count) {
    typename allocator_::value_type;
    { allocator.allocate(count) } -> std::same_as<typename allocator_::value_type*>;
    allocator.deallocate(pointer, count);
};

static_assert(buffer_allocator<std::allocator<byte_t>>);

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

    /** Every heap names its allocator; nothing is placed by default. */
    max_heap() = delete;

    /** The process-wide allocator, for a heap no arena or domain owns. */
    explicit max_heap(global_allocator_t) noexcept
        requires std::default_initializable<allocator_t>
        : max_heap(allocator_t {}) {}
    explicit max_heap(allocator_t allocator) noexcept
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
    inline void shrink(std::size_t n) noexcept { size_ = min_of(n, size_); }

    /**
     *  @brief Selects the largest element in the heap.
     *  @return Reference to the stored element.
     */
    inline element_t const& top() const noexcept { return elements_[0]; }

    /** Invalidates the "max-heap" property, transforming into ascending range: each step parks the
     *  maximum behind the shrinking heap. */
    inline void sort_ascending() noexcept {
        std::size_t const size = size_;
        while (size_ > 1) {
            std::swap(elements_[0], elements_[--size_]);
            shift_down(0);
        }
        size_ = size;
    }

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

    /** Inserts an element into the heap without reserving additional space. */
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
    static bool less(element_t const& first, element_t const& second) noexcept {
        return comparator_t {}(first, second);
    }

    /** Moves the element at @p i up until the heap property is restored - the
     *  companion of appending at the tail. */
    void shift_up(std::size_t i) noexcept {
        for (; i && less(elements_[parent_idx(i)], elements_[i]); i = parent_idx(i))
            std::swap(elements_[parent_idx(i)], elements_[i]);
    }

    /** Moves the element at @p i down until the heap property is restored - the
     *  companion of displacing the root during @ref pop. */
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

    /** Every heap names its allocator; nothing is placed by default. */
    max_heap_dary() = delete;

    /** The process-wide allocator, for a heap no arena or domain owns. */
    explicit max_heap_dary(global_allocator_t) noexcept
        requires std::default_initializable<allocator_t>
        : max_heap_dary(allocator_t {}) {}
    explicit max_heap_dary(allocator_t allocator) noexcept
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

    /** Bytes a fresh reservation of @p capacity entries occupies. */
    static constexpr std::size_t memory_usage(std::size_t capacity) noexcept {
        return reserved_capacity(capacity) * sizeof(element_t);
    }

    /** Bytes this heap currently occupies. */
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
    inline void shrink(std::size_t n) noexcept { size_ = min_of(n, size_); }
    inline element_t const& top() const noexcept { return elements_[0]; }

    /** Invalidates the "max-heap" property, transforming into ascending range: each step parks the
     *  maximum behind the shrinking heap, whatever its arity. */
    inline void sort_ascending() noexcept {
        std::size_t const size = size_;
        while (size_ > 1) {
            std::swap(elements_[0], elements_[--size_]);
            shift_down(0);
        }
        size_ = size;
    }

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

    /** Inserts @p count elements via append-then-heapify, picking between
     *  @b Floyd bulk rebuild (for batches at least half the existing size) and
     *  incremental @c shift_up (for smaller batches). */
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

    /** Drops every entry the comparator orders below @p bar and re-heapifies what remains in
     *  one pass - a frontier keyed further-first sheds everything further than @p bar. */
    std::size_t erase_below(element_t const& bar) noexcept {
        std::size_t kept = 0;
        for (std::size_t index = 0; index != size_; ++index)
            if (!less(elements_[index], bar)) elements_[kept++] = elements_[index];
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
    static bool less(element_t const& first, element_t const& second) noexcept {
        return comparator_t {}(first, second);
    }

    /** Floyd's bottom-up heapify over the whole fill. */
    void rebuild_() noexcept {
        if (size_ < 2) return;
        for (std::size_t parent = (size_ - 2) / arity_ + 1; parent-- > 0;) shift_down(parent);
    }

    void shift_up(std::size_t i) noexcept {
        for (; i && less(elements_[parent_idx(i)], elements_[i]); i = parent_idx(i))
            std::swap(elements_[parent_idx(i)], elements_[i]);
    }

    /** Iterative shift-down. Loads up to @p arity_ children per level, finds the
     *  max via a scalar linear scan (well-predicted for small @p arity_), swaps with
     *  parent if needed, descends. No recursion - the loop terminates when the heap
     *  property is restored. */
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

/** Similar to @c std::priority_queue, but allows raw access to underlying
 *  memory and always keeps the data sorted. Ideal for small collections
 *  under 128 elements. */
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

    /** Every buffer names its allocator; nothing is placed by default. */
    sorted_buffer() = delete;

    /** The process-wide allocator, for a buffer no arena or domain owns. */
    explicit sorted_buffer(global_allocator_t) noexcept
        requires std::default_initializable<allocator_t>
        : sorted_buffer(allocator_t {}) {}
    explicit sorted_buffer(allocator_t allocator) noexcept
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

    /** Bytes a fresh reservation of @p capacity entries occupies. */
    static constexpr std::size_t memory_usage(std::size_t capacity) noexcept {
        return reserved_capacity(capacity) * sizeof(element_t);
    }

    /** Bytes this buffer currently occupies. */
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
        std::size_t const slot = lower_bound(std::span(elements_, size_), element, comparator_t {}) - elements_;
        shift_right(slot, size_);
        elements_[slot] = element;
        size_++;
    }

    /**
     *  @brief Inserts @p element into the ascending buffer, capping its size at @p limit.
     *  @return @c true if the entry was added, @c false if it wasn't relevant enough.
     *
     *  The fast-reject path - a full buffer whose top beats @p element - returns without the
     *  @c lower_bound walk: O(1) rather than O(log limit). That's the common case on the HNSW hot
     *  path once the top-k buffer warms up and most successors lose.
     */
    inline bool insert(element_t&& element, std::size_t limit) noexcept {
        if (size_ == limit && !less(element, elements_[size_ - 1])) return false;
        std::size_t const slot = lower_bound(std::span(elements_, size_), element, comparator_t {}) - elements_;
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

    /** No-op for API parity with the heaps: the buffer is always sorted ascending. */
    void sort_ascending() noexcept {}
    inline void shrink(std::size_t n) noexcept { size_ = min_of(n, size_); }

    inline element_t* data() noexcept { return elements_; }
    inline element_t const* data() const noexcept { return elements_; }

  private:
    static bool less(element_t const& first, element_t const& second) noexcept {
        return comparator_t {}(first, second);
    }

    /** Opens a slot at @p slot by moving the [@p slot, @p tail) range one position
     *  right, back to front, so the overlap never clobbers unread elements. */
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

    /** Every buffer names its allocator; nothing is placed by default. */
    growing_buffer() = delete;

    /** The process-wide allocator, for a buffer no arena or domain owns. */
    explicit growing_buffer(global_allocator_t) noexcept
        requires std::default_initializable<allocator_type_>
        : growing_buffer(allocator_type_ {}) {}

    /** Takes the caller's allocator in its own element type; the rebind to @c element_t
     *  happens here, so no consumer spells @c allocator_traits itself. */
    explicit growing_buffer(allocator_type_ const& allocator) noexcept
        : elements_(nullptr), size_(0), capacity_(0), allocator_(allocator) {}

    /** Qualified, since an element type from @c std drags @c std::exchange into the
     *  overload set through argument-dependent lookup and makes the call ambiguous. */
    growing_buffer(growing_buffer&& other) noexcept
        : elements_(utilities::exchange(other.elements_, nullptr)), size_(utilities::exchange(other.size_, 0)),
          capacity_(utilities::exchange(other.capacity_, 0)), allocator_(std::move(other.allocator_)) {}

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

    /** Bytes this buffer currently occupies. */
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

    /**
     *  @brief Grows capacity to at least @p new_capacity, rounding up to a power of two and
     *      at least doubling, so a repeatedly-growing loop amortizes. Live elements survive.
     *  @return @c success_k, @c overflow_k when the rounded capacity wraps, @c bad_alloc_k otherwise.
     */
    status_t reserve(std::size_t new_capacity) noexcept {
        if (new_capacity <= capacity_) return status_t::success_k;

        if (ceil2(new_capacity) == 0) return status_t::overflow_k;
        return reserve_exactly(max_of<std::size_t>(reserved_capacity(new_capacity), capacity_ * 2u));
    }

    /** Grows capacity to exactly @p new_capacity, without the rounding @ref reserve applies -
     *  the sizing form for a table whose extent the caller already knows and then freezes. */
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

    /** Sizes to exactly @p count value-initialized elements, discarding any prior contents and
     *  growing past the current capacity only. Trivial elements are zero-filled in one
     *  @c memset. */
    status_t resize(std::size_t count) noexcept {
        clear();
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

    /**
     *  @brief Like @ref resize, but skips the zero fill so the caller owns the first touch.
     *  @note Implicit-lifetime element types only - nothing is constructed, so every element must be
     *      written before it is read.
     */
    status_t resize_uninitialized(std::size_t count) noexcept {
        clear();
        if (count == 0) return status_t::success_k;
        if (status_t const sized = reserve_exactly(count); failed(sized)) return sized;
        extend_uninitialized(count);
        return status_t::success_k;
    }

    /** Moves the fill to @p count within reserved capacity, leaving the bytes untouched -
     *  never allocates, never fails. This is the slice an arena hands out, where value-
     *  constructing the run would cost a pass over every slice handed out. */
    inline void extend_uninitialized(std::size_t count) noexcept {
        // Trivially copyable types begin their lifetime with the first write, even NumKong's scalars, whose
        // default constructor zeroes.
        static_assert(std::is_trivially_copyable<element_t>(),
                      "Uninitialized storage is only safe for implicit-lifetime element types");
        annotate_container(elements_, elements_ + capacity_, elements_ + size_, elements_ + count);
        size_ = count;
    }

    /** Appends @p element into pre-reserved capacity - the hot-path form. */
    inline void push_back_reserved(element_t element) noexcept {
        annotate_container(elements_, elements_ + capacity_, elements_ + size_, elements_ + size_ + 1);
        new (elements_ + size_) element_t(std::move(element));
        ++size_;
    }

    /** Appends @p element, doubling capacity when full. @ref push_back_reserved is the
     *  form for a loop that reserved its whole extent up front. */
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

    /** Moves the live fill to @p count within reserved capacity, value-constructing
     *  growth and destroying shrinkage; never allocates. Instantiates only for
     *  default-constructible elements. */
    inline void resize_reserved(std::size_t count) noexcept {
        std::size_t const previous = size_;
        for (; size_ > count; --size_) elements_[size_ - 1].~element_t();
        annotate_container(elements_, elements_ + capacity_, elements_ + previous, elements_ + count);
        for (; size_ < count; ++size_) new (elements_ + size_) element_t();
    }
};

/**
 *  @brief The one growth schedule every chunked collection follows: chunks double from 2^first_rows_log2_ rows up
 *      to 2^slab_rows_log2_, then stay slab-sized, so a row's chunk and its offset in it are closed-form and never
 * move.
 *
 *  Doubling chunks give closed-form prefix sums, so @ref locate is one @c bit_width plus a few adds - cheap enough
 *  for every traversal hop.
 */
template <std::size_t first_rows_log2_, std::size_t slab_rows_log2_>
struct chunk_schedule {
    static_assert(first_rows_log2_ <= slab_rows_log2_ && slab_rows_log2_ < address_bits_k);

    /** Rows in every mature slab-sized chunk, as a power-of-two exponent. */
    inline static constexpr std::size_t slab_rows_log2_k = slab_rows_log2_;

    /** Chunks in the doubling phase: the first chunk plus one per doubling. */
    inline static constexpr std::size_t doubling_chunks_k = slab_rows_log2_ - first_rows_log2_ + 1;

    /** Rows the doubling phase covers: 2^(slab_rows_log2_ + 1) - 2^first_rows_log2_, in 64 bits,
     *  since a schedule may span more rows than a 32-bit @c std::size_t addresses. */
    inline static constexpr u64_t doubling_rows_k = (u64_t {1} << (slab_rows_log2_ + 1)) -
                                                    (u64_t {1} << first_rows_log2_);

    /** Where a row lives: its chunk, and its offset within that chunk. */
    struct location_t {
        std::size_t chunk = 0;
        std::size_t offset = 0;
    };

    /** The chunk and offset of @p row. */
    static constexpr location_t locate(std::size_t row) noexcept {
        if (row < doubling_rows_k) {
            // Biasing by the first chunk's size makes the leading bit of the row encode the chunk.
            u64_t const biased = row + (u64_t {1} << first_rows_log2_);
            return {static_cast<std::size_t>(std::bit_width(biased)) - 1 - first_rows_log2_,
                    static_cast<std::size_t>(biased - std::bit_floor(biased))};
        }
        u64_t const flat = row - doubling_rows_k;
        return {doubling_chunks_k + static_cast<std::size_t>(flat >> slab_rows_log2_),
                static_cast<std::size_t>(flat & ((u64_t {1} << slab_rows_log2_) - 1))};
    }

    /** Rows across the first @p chunks chunks - the prefix sums @ref locate inverts. */
    static constexpr std::size_t rows_before(std::size_t chunks) noexcept {
        if (chunks <= doubling_chunks_k)
            return static_cast<std::size_t>((u64_t {1} << (first_rows_log2_ + chunks)) -
                                            (u64_t {1} << first_rows_log2_));
        return static_cast<std::size_t>(doubling_rows_k + (u64_t {chunks - doubling_chunks_k} << slab_rows_log2_));
    }

    /** Rows in chunk @p chunk, capped at what @c std::size_t addresses,
     *  which no allocation reaches. */
    static constexpr std::size_t rows_in(std::size_t chunk) noexcept {
        u64_t const rows = u64_t {1} << (chunk < doubling_chunks_k ? first_rows_log2_ + chunk : slab_rows_log2_);
        return static_cast<std::size_t>(min_of<u64_t>(rows, std::numeric_limits<std::size_t>::max()));
    }

    /** Chunks that hold the first @p rows rows. */
    static constexpr std::size_t chunks_for(std::size_t rows) noexcept { return rows ? locate(rows - 1).chunk + 1 : 0; }
};

/** The schedule spanning @p max_rows_ rows in about 2^12 chunks: 2^10-row chunks doubling up to
 *  slabs as wide as that takes - at 2^32 rows, the dense index's @c chunk_schedule<10, 20>. */
template <std::size_t max_rows_>
using chunk_schedule_spanning = chunk_schedule<10, max_of<std::size_t>(std::bit_width(max_rows_), 10 + 12) - 12>;

/**
 *  @brief Rows addressed by ID that never move: chunks on @p schedule_type_, a missing one allocated, zeroed and
 *      published by whichever thread asks for it first, so readers index while writers grow the column.
 *
 *  @ref cover is the only fallible call, and it may run on any thread. Covers publish chunks in order, so the
 *  published ones are always a prefix. A reader indexes only rows some cover already reached, and learns of them
 *  through the owner's own publication, which orders the chunk before the read. The directory entries are plain
 *  pointers owned through @p atomic_ref_template_.
 *
 *  A row type that is not trivially copyable is built instead of zeroed: every row of a fresh
 *  chunk is constructed from @p allocator_type_ before the chunk publishes, and destroyed before
 *  the chunk is returned.
 */
template <typename value_type_, typename schedule_type_, std::size_t max_rows_, typename allocator_type_,
          template <typename> class atomic_ref_template_ = standard_atomic_ref>
class chunked_column {
    static_assert(std::is_trivially_copyable_v<value_type_> ||
                      std::is_nothrow_constructible_v<value_type_, allocator_type_ const&>,
                  "Rows are zeroed, or built from the column's allocator");

  public:
    using value_t = value_type_;
    using schedule_t = schedule_type_;
    using location_t = typename schedule_t::location_t;
    using allocator_t = typename std::allocator_traits<allocator_type_>::template rebind_alloc<value_t>;
    template <typename word_type_>
    using atomic_ref = atomic_ref_template_<word_type_>;

    /** Chunks the directory holds - enough for @p max_rows_ rows. */
    inline static constexpr std::size_t max_chunks_k = schedule_t::chunks_for(max_rows_);

    explicit chunked_column(allocator_t allocator) noexcept : allocator_(std::move(allocator)) {}
    chunked_column(chunked_column&& other) noexcept : chunks_(other.chunks_), allocator_(std::move(other.allocator_)) {
        other.chunks_.fill(nullptr);
    }
    chunked_column& operator=(chunked_column&& other) noexcept {
        std::swap(chunks_, other.chunks_);
        std::swap(allocator_, other.allocator_);
        return *this;
    }
    chunked_column(chunked_column const&) = delete;
    chunked_column& operator=(chunked_column const&) = delete;
    ~chunked_column() noexcept { reset(); }

    /** Allocates, zeroes or builds, and publishes every chunk the first @p rows rows need and is
     *  missing - safe on any thread. */
    status_t cover(std::size_t rows) noexcept {
        if (rows > max_rows_) return status_t::overflow_k;
        for (std::size_t chunk = 0, chunks = schedule_t::chunks_for(rows); chunk != chunks; ++chunk) {
            atomic_ref<value_t*> slot {chunks_[chunk]};
            if (slot.load(std::memory_order_acquire)) continue;
            value_t* const fresh = allocator_.allocate(schedule_t::rows_in(chunk));
            if (!fresh) return status_t::bad_alloc_k;
            if constexpr (std::is_trivially_copyable_v<value_t>)
                std::memset(static_cast<void*>(fresh), 0, schedule_t::rows_in(chunk) * sizeof(value_t));
            else {
                allocator_type_ const rows_allocator(allocator_);
                for (std::size_t row = 0; row != schedule_t::rows_in(chunk); ++row)
                    new (fresh + row) value_t(rows_allocator);
            }
            value_t* expected_empty = nullptr;
            // A peer covering the same chunk first keeps its own; this one goes back.
            if (!slot.compare_exchange_strong(expected_empty, fresh, std::memory_order_release,
                                              std::memory_order_acquire))
                release_(fresh, schedule_t::rows_in(chunk));
        }
        return status_t::success_k;
    }

    /** The row at @p location, which a @ref cover already reached - for callers that located it
     *  once for several columns. */
    value_t& operator[](location_t location) noexcept {
        value_t* const rows = atomic_ref<value_t*>(chunks_[location.chunk]).load(std::memory_order_relaxed);
        return rows[location.offset];
    }
    value_t const& operator[](location_t location) const noexcept {
        value_t* const rows = atomic_ref<value_t* const>(chunks_[location.chunk]).load(std::memory_order_relaxed);
        return rows[location.offset];
    }

    /** Row @p row, which a @ref cover already reached. */
    value_t& operator[](std::size_t row) noexcept { return (*this)[schedule_t::locate(row)]; }
    value_t const& operator[](std::size_t row) const noexcept { return (*this)[schedule_t::locate(row)]; }

    /** The rows of chunk @p index, acquired with its contents; empty until a cover publishes it. */
    std::span<value_t> chunk(std::size_t index) noexcept {
        value_t* const rows = atomic_ref<value_t*>(chunks_[index]).load(std::memory_order_acquire);
        return {rows, rows ? schedule_t::rows_in(index) : 0};
    }
    std::span<value_t const> chunk(std::size_t index) const noexcept {
        value_t* const rows = atomic_ref<value_t* const>(chunks_[index]).load(std::memory_order_acquire);
        return {rows, rows ? schedule_t::rows_in(index) : 0};
    }

    /** Rows the published chunks hold. */
    std::size_t capacity() const noexcept {
        std::size_t chunks = 0;
        while (chunks != max_chunks_k && !chunk(chunks).empty()) ++chunks;
        return schedule_t::rows_before(chunks);
    }

    /** Hands @p callback the @p rows as contiguous spans, one per chunk they cross - the shape a bulk copy in or out
     *  takes. */
    template <typename callback_type_>
    void for_each_span(iota_view<std::size_t> rows, callback_type_&& callback) {
        for (std::size_t row = rows.first; row != rows.bound;) {
            auto const at = schedule_t::locate(row);
            std::size_t const run = min_of(rows.bound - row, schedule_t::rows_in(at.chunk) - at.offset);
            callback(chunk(at.chunk).subspan(at.offset, run));
            row += run;
        }
    }
    template <typename callback_type_>
    void for_each_span(iota_view<std::size_t> rows, callback_type_&& callback) const {
        for (std::size_t row = rows.first; row != rows.bound;) {
            auto const at = schedule_t::locate(row);
            std::size_t const run = min_of(rows.bound - row, schedule_t::rows_in(at.chunk) - at.offset);
            callback(chunk(at.chunk).subspan(at.offset, run));
            row += run;
        }
    }

    /** Bytes the published chunks span. */
    std::size_t memory_usage() const noexcept { return capacity() * sizeof(value_t); }

    /** Frees every chunk - single-threaded: the caller certifies no concurrent readers or writers. */
    void reset() noexcept {
        for (std::size_t index = 0; index != max_chunks_k; ++index)
            if (value_t* const rows = std::exchange(chunks_[index], nullptr))
                release_(rows, schedule_t::rows_in(index));
    }

  private:

    /** Destroys the @p count rows a cover built, if it built any, and returns their chunk. */
    void release_(value_t* rows, std::size_t count) noexcept {
        if constexpr (!std::is_trivially_copyable_v<value_t>) std::destroy_n(rows, count);
        allocator_.deallocate(rows, count);
    }

    /** One pointer per chunk, sized once and never reallocated; a null entry is a chunk no cover reached yet. */
    std::array<value_t*, max_chunks_k> chunks_ {};

    [[no_unique_address]] allocator_t allocator_;
};

/**
 *  @brief The matrix twin of @ref chunked_column: rows of one width, addressed by ID, that never
 *      move, each chunk allocated, filled and published by the first @ref cover reaching it, under
 *      the column's contract.
 *
 *  The first @ref cover fixes the width, before any reader can learn of a row, and every later one
 *  repeats it.
 */
template <typename value_type_, typename schedule_type_, std::size_t max_rows_, typename allocator_type_,
          template <typename> class atomic_ref_template_ = standard_atomic_ref>
class chunked_matrix {
    static_assert(std::is_trivially_copyable_v<value_type_>, "Rows are written in place, never constructed");

  public:
    using value_t = value_type_;
    using schedule_t = schedule_type_;
    using location_t = typename schedule_t::location_t;
    using allocator_t = typename std::allocator_traits<allocator_type_>::template rebind_alloc<value_t>;
    template <typename word_type_>
    using atomic_ref = atomic_ref_template_<word_type_>;

    /** Chunks the directory holds - enough for @p max_rows_ rows. */
    inline static constexpr std::size_t max_chunks_k = schedule_t::chunks_for(max_rows_);

    explicit chunked_matrix(allocator_t allocator) noexcept : allocator_(std::move(allocator)) {}
    chunked_matrix(chunked_matrix const&) = delete;
    chunked_matrix& operator=(chunked_matrix const&) = delete;
    ~chunked_matrix() noexcept { reset(); }

    /** Allocates every chunk the first @p rows rows of @p columns values need and is missing, sets
     *  each of its bytes to @p fill, and publishes it - safe on any thread. */
    status_t cover(std::size_t rows, std::size_t columns, std::uint8_t fill = 0) noexcept {
        if (rows > max_rows_) return status_t::overflow_k;
        if (columns_ != columns) columns_ = columns;
        for (std::size_t chunk = 0, chunks = schedule_t::chunks_for(rows); chunk != chunks; ++chunk) {
            atomic_ref<value_t*> slot {chunks_[chunk]};
            if (slot.load(std::memory_order_acquire)) continue;
            std::size_t const values = schedule_t::rows_in(chunk) * columns;
            value_t* const fresh = allocator_.allocate(values);
            if (!fresh) return status_t::bad_alloc_k;
            std::memset(static_cast<void*>(fresh), fill, values * sizeof(value_t));
            value_t* expected_empty = nullptr;
            // A peer covering the same chunk first keeps its own; this one goes back.
            if (!slot.compare_exchange_strong(expected_empty, fresh, std::memory_order_release,
                                              std::memory_order_acquire))
                allocator_.deallocate(fresh, values);
        }
        return status_t::success_k;
    }

    /** Values per row, fixed by the first @ref cover; zero before it. */
    std::size_t columns() const noexcept { return columns_; }

    /** The first of the @ref columns values of the row at @p location, which a @ref cover already
     *  reached. */
    value_t* operator[](location_t location) noexcept {
        value_t* const rows = atomic_ref<value_t*>(chunks_[location.chunk]).load(std::memory_order_relaxed);
        return rows + location.offset * columns_;
    }
    value_t const* operator[](location_t location) const noexcept {
        value_t* const rows = atomic_ref<value_t* const>(chunks_[location.chunk]).load(std::memory_order_relaxed);
        return rows + location.offset * columns_;
    }

    /** Frees every chunk - single-threaded: the caller certifies no concurrent readers or
     *  writers. */
    void reset() noexcept {
        for (std::size_t index = 0; index != max_chunks_k; ++index)
            if (value_t* const rows = std::exchange(chunks_[index], nullptr))
                allocator_.deallocate(rows, schedule_t::rows_in(index) * columns_);
    }

  private:

    /** One pointer per chunk, sized once and never reallocated; a null entry is a chunk no cover
     *  reached yet. */
    std::array<value_t*, max_chunks_k> chunks_ {};

    /** Values per row, fixed by the first cover; zero before it. */
    std::size_t columns_ = 0;

    [[no_unique_address]] allocator_t allocator_;
};

/** Per-thread bump arena: reservations bump inside the newest chunk, and one that cannot
 *  fit them stays alive under its slices while a chunk at least twice as wide opens - so
 *  addresses never move and slices outlive growth. The rewind keeps the widest chunk and
 *  drops the ones it outgrew; releasing a slice does nothing, the rewind is the release. */
template <typename upstream_allocator_ = std::allocator<byte_t>>
class growing_arena {
    using upstream_t = typename std::allocator_traits<upstream_allocator_>::template rebind_alloc<std::byte>;
    using run_t = growing_buffer<std::byte, upstream_t>;

    /** Each chunk at least doubles the one it outgrew, so no more are alive at once than a byte
     *  count has bits. */
    static constexpr std::size_t max_chunks_k = std::numeric_limits<std::size_t>::digits;

    /** The live chunks, oldest first, the last of the first @c chunks_ being the newest, the one
     *  reservations bump; the rest hold nothing, so no chunk takes a record from any heap. */
    std::array<run_t, max_chunks_k> runs_;

    /** How many of @c runs_ hold a chunk. */
    std::size_t chunks_ = 0;

    /** Places every run - the thread's memory domain. */
    upstream_t upstream_;

  public:

    /** Every arena names its upstream allocator; nothing is placed by default. */
    growing_arena() = delete;

    /** The process-wide allocator, for an arena no memory domain owns. */
    explicit growing_arena(global_allocator_t) noexcept
        requires std::default_initializable<upstream_t>
        : growing_arena(upstream_t {}) {}
    explicit growing_arena(upstream_t upstream) noexcept
        : runs_(empty_runs_(upstream, std::make_index_sequence<max_chunks_k> {})), upstream_(upstream) {}

    /** The allocator every chunk of this arena comes from. */
    upstream_t const& upstream() const noexcept { return upstream_; }
    growing_arena(growing_arena&& other) noexcept
        : runs_(std::move(other.runs_)), chunks_(utilities::exchange(other.chunks_, 0)), upstream_(other.upstream_) {}
    growing_arena& operator=(growing_arena&& other) noexcept {
        std::swap(runs_, other.runs_);
        std::swap(chunks_, other.chunks_);
        std::swap(upstream_, other.upstream_);
        return *this;
    }

    /** Bytes across chunks. */
    std::size_t capacity() const noexcept {
        std::size_t bytes = 0;
        for (std::size_t chunk = 0; chunk != chunks_; ++chunk) bytes += runs_[chunk].capacity();
        return bytes;
    }

    /** Bytes handed out across chunks, the padding before every slice included. */
    std::size_t size() const noexcept {
        std::size_t bytes = 0;
        for (std::size_t chunk = 0; chunk != chunks_; ++chunk) bytes += runs_[chunk].size();
        return bytes;
    }

    /** Opens the seed chunk at @p bytes, the width every later chunk doubles from; a
     *  seeded arena answers in one compare. */
    status_t reserve_exactly(std::size_t bytes) noexcept {
        if (chunks_) return status_t::success_k;
        return grow_(bytes);
    }

    /** Hands out @p bytes on an @p alignment boundary from the newest chunk, growing by one
     *  at least twice as wide when it cannot fit them; null when the upstream refuses. */
    std::byte* allocate(std::size_t bytes, std::size_t alignment) noexcept {
        if (chunks_)
            if (std::byte* const slice = slice_(runs_[chunks_ - 1], bytes, alignment)) return slice;
        std::size_t const last_width = chunks_ ? runs_[chunks_ - 1].capacity() : 0;
        if (failed(grow_(max_of(last_width * 2u, bytes + alignment)))) return nullptr;
        return slice_(runs_[chunks_ - 1], bytes, alignment);
    }

    /** Rewinds the newest chunk and drops the ones it outgrew, so the next call starts at
     *  the high-water mark. */
    void clear() noexcept {
        if (!chunks_) return;
        if (chunks_ > 1) std::swap(runs_[0], runs_[chunks_ - 1]);
        for (std::size_t chunk = 1; chunk != chunks_; ++chunk) runs_[chunk].reset();
        chunks_ = 1;
        runs_[0].clear();
    }

    /** Releases every chunk. */
    void reset() noexcept {
        for (std::size_t chunk = 0; chunk != chunks_; ++chunk) runs_[chunk].reset();
        chunks_ = 0;
    }

    /** Whether @p pointer lies inside any chunk, which the next rewind reuses or frees. */
    bool contains(void const* pointer) const noexcept {
        std::uintptr_t const address = reinterpret_cast<std::uintptr_t>(pointer);
        for (std::size_t chunk = 0; chunk != chunks_; ++chunk) {
            std::uintptr_t const first = reinterpret_cast<std::uintptr_t>(runs_[chunk].data());
            if (address >= first && address < first + runs_[chunk].capacity()) return true;
        }
        return false;
    }

  private:

    /** Every run empty over @p upstream, one per index. */
    template <std::size_t... indices_>
    static std::array<run_t, max_chunks_k> empty_runs_(upstream_t const& upstream,
                                                       std::index_sequence<indices_...>) noexcept {
        return {{(static_cast<void>(indices_), run_t {upstream})...}};
    }

    /** Grows by a chunk of @p width bytes, the newest from now on, the old ones kept behind. */
    status_t grow_(std::size_t width) noexcept {
        if (chunks_ == max_chunks_k) return status_t::bad_alloc_k;
        if (status_t const sized = runs_[chunks_].reserve_exactly(width); failed(sized)) return sized;
        ++chunks_;
        return status_t::success_k;
    }

    /** The address-aligned slice at the end of @p run, or null when it cannot fit @p bytes. */
    static std::byte* slice_(run_t& run, std::size_t bytes, std::size_t alignment) noexcept {
        // Aligned by address, not by offset: the upstream only promises what `malloc` does.
        std::uintptr_t const cursor = reinterpret_cast<std::uintptr_t>(run.data() + run.size());
        std::size_t const padding = static_cast<std::size_t>(
            divide_round_up<std::uintptr_t>(cursor, alignment) * alignment - cursor);
        if (run.capacity() - run.size() < padding + bytes) return nullptr;
        run.extend_uninitialized(run.size() + padding + bytes);
        return run.data() + run.size() - bytes;
    }
};

/** Hands out the one span its caller lends, to a request of exactly its extent, and refuses every
 *  other; releasing does nothing. The upstream of a @ref growing_arena whose only chunk is borrowed
 *  memory: seeded by @ref growing_arena::reserve_exactly over the whole span, any later chunk would
 *  at least double it, so the span is never handed out twice. */
template <typename value_type_ = std::byte>
struct lent_allocator {
    using value_type = value_type_;

    /** The lent memory; empty refuses every request. */
    std::span<value_type_> lent;

    value_type_* allocate(std::size_t count) noexcept { return count == lent.size() ? lent.data() : nullptr; }
    void deallocate(value_type_*, std::size_t) noexcept {}
};

static_assert(buffer_allocator<lent_allocator<std::byte>>);

/** Hands out slices of one @ref growing_arena it points at but does not own. Copies share
 *  the arena by pointer, so containers built from one reserve disjoint slices; releasing a
 *  slice does nothing - the arena's rewind is the release. Every slice starts on an
 *  @p alignment_ boundary, since the arena's raw bytes promise only what @c malloc does and
 *  the containers built over a slice issue the widest loads their kernels know. */
template <typename value_type_ = std::byte, typename upstream_allocator_ = std::allocator<byte_t>,
          std::size_t alignment_ = default_alignment_k>
struct arena_allocator {
    using value_type = value_type_;
    using size_type = std::size_t;
    using arena_t = growing_arena<upstream_allocator_>;

    /** Spelled out because the alignment's non-type parameter defeats the traits'
     *  automatic first-argument replacement; the upstream and the alignment ride along. */
    template <typename other_type_>
    struct rebind {
        using other = arena_allocator<other_type_, upstream_allocator_, alignment_>;
    };

    /** The boundary every reservation starts on; echoes the template argument. */
    inline static constexpr std::size_t alignment_k = alignment_;

    /** Bytes a reservation of @p bytes occupies here, the next one starting on its own boundary. */
    static constexpr std::size_t reserved_bytes(std::size_t bytes) noexcept {
        return round_up_to_multiple(bytes, alignment_k);
    }

    /** The arena every rebound copy shares; null refuses every request. */
    arena_t* arena = nullptr;

    constexpr arena_allocator() noexcept = default;
    explicit constexpr arena_allocator(arena_t& arena) noexcept : arena(&arena) {}

    /** The rebinding conversion @c std::allocator_traits performs - the upstream rides
     *  along, and the arena is shared rather than copied. */
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

} // namespace unum::usearch::utilities
