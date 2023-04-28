/**
 *  @file usearch.hpp
 *  @author Ashot Vardanian
 *  @brief Single-header Vector Search.
 *  @date 2023-04-26
 *
 *  @copyright Copyright (c) 2023
 */
#pragma once
#include <atomic>
#include <bitset>  // `std::bitset`
#include <climits> // `CHAR_BIT`
#include <cmath>   // `std::sqrt`
#include <cstring> // `std::memset`
#include <mutex>
#include <random>
#include <utility> // `std::exchange`
#include <vector>

#include <fcntl.h>    // `fallocate`
#include <stdlib.h>   // `posix_memalign`
#include <sys/mman.h> // `mmap`
#include <sys/stat.h> // `fstat` for file size
#include <unistd.h>   // `open`, `close`

#if defined(__GNUC__)
// https://gcc.gnu.org/onlinedocs/gcc/Other-Builtins.html
// Zero means we are only going to read from that memory.
// Three means high temporal locality and suggests to keep
// the data in all layers of cache.
#define prefetch_m(ptr) __builtin_prefetch((void*)(ptr), 0, 3)
#elif defined(__x86_64__)
#define prefetch_m(ptr) _mm_prefetch((void*)(ptr), _MM_HINT_T0)
#else
#define prefetch_m(ptr)
#endif

#ifdef NDEBUG
#define assert_m(must_be_true, message)
#else
#define assert_m(must_be_true, message)                                                                                \
    if (!(must_be_true)) {                                                                                             \
        throw std::runtime_error(message);                                                                             \
    }
#endif

#define usearch_align_m __attribute__((aligned(64)))
#define usearch_pack_m __attribute__((packed))

namespace unum {
namespace usearch {

using label_t = std::size_t;
using dim_t = std::int32_t;
using level_t = std::int32_t;

using f64_t = double;
using f32_t = float;

/**
 *  GCC supports half-precision types.
 *  https://gcc.gnu.org/onlinedocs/gcc/Half-Precision.html
 */
enum class f16_t : std::uint16_t {};

template <typename at> constexpr at angle_to_radians(at angle) noexcept { return angle * M_PI / 180; }

template <typename at> constexpr at square(at value) noexcept { return value * value; }

inline std::size_t ceil2(std::size_t v) noexcept {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v |= v >> 32;
    v++;
    return v;
}

template <typename scalar_at> struct ip_gt {
    using scalar_t = scalar_at;
    using result_type = scalar_t;

    inline result_type operator()(scalar_t const* a, scalar_t const* b, dim_t dim, dim_t = 0) const noexcept {
        result_type ab = 0;
#if defined(_OPENMP)
#pragma omp simd reduction(+ : ab)
#elif defined(__GNUC__)
#pragma GCC ivdep
#elif defined(__clang__)
#pragma clang loop vectorize(enable)
#endif
        for (dim_t i = 0; i != dim; ++i)
            ab += a[i] * b[i];
        return 1 - ab;
    }
};

template <typename scalar_at> struct cos_gt {
    using scalar_t = scalar_at;
    using result_type = scalar_t;

    inline result_type operator()(scalar_t const* a, scalar_t const* b, dim_t dim, dim_t = 0) const noexcept {
        result_type ab = 0, a2 = 0, b2 = 0;
#if defined(_OPENMP)
#pragma omp simd reduction(+ : ab, a2, b2)
#elif defined(__GNUC__)
#pragma GCC ivdep
#elif defined(__clang__)
#pragma clang loop vectorize(enable)
#endif
        for (dim_t i = 0; i != dim; ++i)
            ab += a[i] * b[i], a2 += a[i] * a[i], b2 += b[i] * b[i];
        return ab / (std::sqrt(a2) * std::sqrt(b2));
    }
};

template <typename scalar_at> struct l2_squared_gt {
    using scalar_t = scalar_at;
    using result_type = scalar_t;

    inline result_type operator()(scalar_t const* a, scalar_t const* b, dim_t dim, dim_t = 0) const noexcept {
        result_type ab_deltas_sq = 0;
#if defined(_OPENMP)
#pragma omp simd reduction(+ : ab_deltas_sq)
#elif defined(__GNUC__)
#pragma GCC ivdep
#elif defined(__clang__)
#pragma clang loop vectorize(enable)
#endif
        for (dim_t i = 0; i != dim; ++i)
            ab_deltas_sq += (a[i] - b[i]) * (a[i] - b[i]);
        return ab_deltas_sq;
    }
};

/**
 *  @brief  Hamming distance computes the number of differing bits in
 *          two arrays of integers. An example would be a textual document,
 *          tokenized and hashed into a fixed-capacity bitset.
 */
template <typename scalar_at> struct bit_hamming_gt {
    using scalar_t = scalar_at;
    using result_type = dim_t;
    static_assert(!std::is_floating_point<scalar_t>::value && std::is_unsigned<scalar_t>::value,
                  "Hamming distance requires unsigned integral words");
    static constexpr std::size_t bits_per_word_k = sizeof(scalar_t) * CHAR_BIT;

    inline result_type operator()(scalar_t const* a, scalar_t const* b, dim_t dim_words, dim_t = 0) const noexcept {
        dim_t matches = 0;
#if defined(_OPENMP)
#pragma omp simd reduction(+ : matches)
#elif defined(__GNUC__)
#pragma GCC ivdep
#elif defined(__clang__)
#pragma clang loop vectorize(enable)
#endif
        for (dim_t i = 0; i != dim_words; ++i)
            matches += std::bitset<bits_per_word_k>(a[i] ^ b[i]).count();
        return matches;
    }
};

/**
 *  @brief  Counts the number of matching elements in two unique sorted sets.
 *          Can be used to compute the similarity between two textual documents
 *          using the IDs of tokens present in them.
 */
template <typename scalar_at> struct jaccard_gt {
    using scalar_t = scalar_at;
    using result_type = f32_t;
    static_assert(!std::is_floating_point<scalar_t>::value, "Jaccard distance requires integral scalars");

    inline result_type operator()(scalar_t const* a, scalar_t const* b, dim_t a_dim, dim_t b_dim) const noexcept {
        dim_t intersection = 0;
        dim_t i = 0;
        dim_t j = 0;
        while (i != a_dim && j != b_dim) {
            intersection += a[i] == b[j];
            i += a[i] < b[j];
            j += a[i] >= b[j];
        }
        return 1.f - intersection * 1.f / (a_dim + b_dim - intersection);
    }
};

/**
 *  @brief  Counts the number of matching elements in two unique sorted sets.
 *          Can be used to compute the similarity between two textual documents
 *          using the IDs of tokens present in them.
 */
template <typename scalar_at> struct pearson_correlation_gt {
    using scalar_t = scalar_at;
    using result_type = scalar_t;

    inline result_type operator()(scalar_t const* a, scalar_t const* b, dim_t dim, dim_t = 0) const noexcept {
        scalar_t a_sum = 0, b_sum = 0, ab_sum = 0;
        scalar_t a_sq_sum = 0, b_sq_sum = 0;
#if defined(_OPENMP)
#pragma omp simd reduction(+ : a_sum, b_sum, ab_sum, a_sq_sum, b_sq_sum)
#elif defined(__GNUC__)
#pragma GCC ivdep
#elif defined(__clang__)
#pragma clang loop vectorize(enable)
#endif
        for (dim_t i = 0; i != dim; ++i) {
            a_sum += a[i];
            b_sum += b[i];
            ab_sum += a[i] * b[i];
            a_sq_sum += a[i] * a[i];
            b_sq_sum += b[i] * b[i];
        }
        result_type denom = std::sqrt((dim * a_sq_sum - a_sum * a_sum) * (dim * b_sq_sum - b_sum * b_sum));
        result_type corr = (dim * ab_sum - a_sum * b_sum) / denom;
        return -corr;
    }
};

/**
 *  @brief  Haversine distance for the shortest distance between two points on
 *          the surface of a 3D sphere, defined with latitude and longitude.
 */
template <typename scalar_at> struct haversine_gt {
    using scalar_t = scalar_at;
    using result_type = scalar_t;
    static_assert(std::is_floating_point<scalar_t>::value, "Latitude and longtitude must be floating-point");

    inline result_type operator()(scalar_t const* a, scalar_t const* b, dim_t = 2, dim_t = 2) const noexcept {
        scalar_t lat_a = a[0], lon_a = a[1];
        scalar_t lat_b = b[0], lon_b = b[1];

        scalar_t lat_delta = angle_to_radians(lat_b - lat_a);
        scalar_t lon_delta = angle_to_radians(lon_b - lon_a);

        scalar_t converted_lat_a = angle_to_radians(lat_a);
        scalar_t converted_lat_b = angle_to_radians(lat_b);

        scalar_t x = square(std::sin(lat_delta / 2)) +
                     std::cos(converted_lat_a) * std::cos(converted_lat_b) * square(std::sin(lon_delta / 2));

        return std::atan2(std::sqrt(x), std::sqrt(1 - x));
    }
};

template <std::size_t multiple_ak> inline std::size_t divide_round_up(std::size_t num) noexcept {
    return (num + multiple_ak - 1) / multiple_ak;
}

template <typename allocator_at = std::allocator<char>> class visits_bitset_gt {
    using allocator_t = allocator_at;
    using byte_t = typename allocator_t::value_type;
    static_assert(sizeof(byte_t) == 1, "Allocator must allocate separate addressable bytes");

    using slot_t = std::uint64_t;

    std::uint64_t* u64s_{};
    std::size_t slots_{};

  public:
    visits_bitset_gt() noexcept {}
    visits_bitset_gt(std::size_t capacity) noexcept {
        slots_ = divide_round_up<64>(capacity);
        u64s_ = (slot_t*)allocator_t{}.allocate(slots_ * sizeof(slot_t));
    }
    ~visits_bitset_gt() noexcept { allocator_t{}.deallocate((byte_t*)u64s_, slots_ * sizeof(slot_t)), u64s_ = nullptr; }
    void clear() noexcept { std::memset(u64s_, 0, slots_ * sizeof(slot_t)); }
    inline bool test(std::size_t i) const noexcept { return u64s_[i / 64] & (1ul << (i & 63ul)); }
    inline void set(std::size_t i) noexcept { u64s_[i / 64] |= (1ul << (i & 63ul)); }

    visits_bitset_gt(visits_bitset_gt&& other) noexcept {
        u64s_ = other.u64s_;
        slots_ = other.slots_;
        other.u64s_ = nullptr;
        other.slots_ = 0;
    }

    visits_bitset_gt& operator=(visits_bitset_gt&& other) noexcept {
        std::swap(u64s_, other.u64s_);
        std::swap(slots_, other.slots_);
        return *this;
    }
};

using visits_bitset_t = visits_bitset_gt<>;

/**
 *  @brief  Similar to `std::priority_queue`, but allows raw access to underlying
 *          memory, in case you want to shuffle it or sort.
 */
template <typename element_at,                                //
          typename comparator_at = std::less<void>,           // <void> is needed before C++14.
          typename allocator_at = std::allocator<element_at>> //
class max_heap_gt {
  public:
    using element_t = element_at;
    using comparator_t = comparator_at;
    using allocator_t = allocator_at;

    using value_type = element_t;

  private:
    element_t* elements_{};
    std::size_t size_{};
    std::size_t capacity_{};
    std::size_t max_capacity_{};

  public:
    max_heap_gt() noexcept {}
    max_heap_gt(std::size_t max_capacity) noexcept : max_capacity_(max_capacity) {}
    bool empty() const noexcept { return !size_; }
    std::size_t size() const noexcept { return size_; }
    std::size_t capacity() const noexcept { return capacity_; }
    element_t const& top() const noexcept { return elements_[0]; }
    element_t const& max() const noexcept { return elements_[0]; }

    void clear() noexcept {
        while (size_) {
            size_--;
            elements_[size_].~element_t();
        }
    }

    bool reserve(std::size_t n) noexcept {
        if (n < capacity_)
            return true;
        if (max_capacity_ && capacity_ == max_capacity_)
            return false;

        auto new_capacity = std::max<std::size_t>(capacity_ * 2u, 16u);
        if (max_capacity_)
            new_capacity = std::min(new_capacity, max_capacity_);

        auto allocator = allocator_t{};
        auto new_elements = allocator.allocate(new_capacity);
        if (!new_elements)
            return false;

        std::uninitialized_copy_n(elements_, size_, new_elements);
        allocator.deallocate(elements_, capacity_);
        elements_ = new_elements;
        capacity_ = new_capacity;
        return new_elements;
    }

    template <typename... args_at> //
    bool emplace(args_at&&... args) noexcept {
        if (!reserve(size_ + 1))
            return false;

        new (&elements_[size_]) element_t({std::forward<args_at>(args)...});
        size_++;
        shift_up(size_ - 1);
        return true;
    }

    element_t pop() noexcept {
        element_t result = top();
        std::swap(elements_[0], elements_[size_ - 1]);
        size_--;
        elements_[size_].~element_t();
        shift_down(0);
        return result;
    }

    /** @brief Invalidates the "max-heap" property, transforming into ascending range. */
    void sort_ascending() noexcept { std::sort_heap(elements_, elements_ + size_, &less); }
    /** @brief Invalidates the "max-heap" property, transforming into descending range. */
    void sort_descending() noexcept { sort_ascending(), std::reverse(elements_, elements_ + size_); }
    void sort_heap() noexcept { std::make_heap(elements_, elements_ + size_, &less); }
    void shrink(std::size_t n) noexcept { size_ = n; }

    element_t* data() noexcept { return elements_; }
    element_t const* data() const noexcept { return elements_; }

  private:
    inline std::size_t parent_idx(std::size_t i) const noexcept { return (i - 1u) / 2u; }
    inline std::size_t left_child_idx(std::size_t i) const noexcept { return (i * 2u) + 1u; }
    inline std::size_t right_child_idx(std::size_t i) const noexcept { return (i * 2u) + 2u; }
    static bool less(element_t const& a, element_t const& b) noexcept { return comparator_t{}(a, b); }

    void shift_up(std::size_t i) noexcept {
        for (; i && less(elements_[parent_idx(i)], elements_[i]); i = parent_idx(i))
            std::swap(elements_[parent_idx(i)], elements_[i]);
    }

    void shift_down(std::size_t i) noexcept {
        std::size_t max_idx = i;

        std::size_t left = left_child_idx(i);
        if (left < size_ && less(elements_[max_idx], elements_[left]))
            max_idx = left;

        std::size_t right = right_child_idx(i);
        if (right < size_ && less(elements_[max_idx], elements_[right]))
            max_idx = right;

        if (i != max_idx) {
            std::swap(elements_[i], elements_[max_idx]);
            shift_down(max_idx);
        }
    }
};

#if defined(__GNUC__) or defined(__clang__)

class mutex_t {
    using slot_t = std::int32_t;
    slot_t flag_;

  public:
    inline mutex_t(slot_t flag = 0) noexcept : flag_(flag) {}
    inline ~mutex_t() noexcept {}

    inline bool try_lock() noexcept {
        slot_t raw = 0;
        return __atomic_compare_exchange_n(&flag_, &raw, 1, true, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED);
    }

    inline void lock() noexcept {
        slot_t raw;
    lock_again:
        raw = 0;
        if (!__atomic_compare_exchange_n(&flag_, &raw, 1, true, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED))
            goto lock_again;
    }

    inline void unlock() noexcept { __atomic_store_n(&flag_, 0, __ATOMIC_RELEASE); }
};

static_assert(sizeof(mutex_t) == sizeof(std::int32_t), "Mutex is larger than expected");

#else
using mutex_t = std::mutex;
#endif

using lock_t = std::unique_lock<mutex_t>;

/**
 *  @brief Implements append-only linked arenas, great for rapid point-cloud construction.
 */
class append_only_allocator_t {
    std::uintptr_t address;
    struct arena_head_t {
        std::uintptr_t next_address;
    };
};

template <typename element_at = char, std::size_t alignment_ak = 64> class aligned_allocator_gt {
  public:
    using value_type = element_at;
    using size_type = std::size_t;
    using pointer = element_at*;
    using const_pointer = element_at const*;
    template <typename other_element_at> struct rebind {
        using other = aligned_allocator_gt<other_element_at>;
    };

    pointer allocate(size_type length) const noexcept {
        void* result = nullptr;
        int status = posix_memalign(&result, alignment_ak, ceil2(length * sizeof(value_type)));
        return status == 0 ? (pointer)result : nullptr;
    }

    void deallocate(pointer begin, size_type) const noexcept { std::free(begin); }
};

using aligned_allocator_t = aligned_allocator_gt<>;

/**
 *  @brief Five-byte integer type to address point clouds with over 4B entries.
 */
class usearch_pack_m uint40_t {
    unsigned char octets[5];

  public:
    inline uint40_t() noexcept { std::memset(octets, 0, 5); }
    inline uint40_t(std::uint32_t n) noexcept { std::memcpy(octets + 1, (char*)&n, 4), octets[0] = 0; }
    inline uint40_t(std::uint64_t n) noexcept { std::memcpy(octets, (char*)&n + 3, 5); }
#if defined(__clang__)
    inline uint40_t(std::size_t n) noexcept { std::memcpy(octets, (char*)&n + 3, 5); }
#endif

    uint40_t(uint40_t&&) = default;
    uint40_t(uint40_t const&) = default;
    uint40_t& operator=(uint40_t&&) = default;
    uint40_t& operator=(uint40_t const&) = default;

    inline uint40_t& operator+=(std::uint32_t i) noexcept {
        std::uint32_t& tail = *reinterpret_cast<std::uint32_t*>(octets + 1);
        octets[0] += static_cast<unsigned char>((tail + i) < tail);
        tail += i;
        return *this;
    }

    inline operator std::size_t() const noexcept {
        std::size_t result = 0;
        std::memcpy((char*)&result + 3, octets, 5);
        return result;
    }

    inline uint40_t& operator++() noexcept { return *this += 1; }

    inline uint40_t operator++(int) noexcept {
        uint40_t old = *this;
        *this += 1;
        return old;
    }
};

static_assert(sizeof(uint40_t) == 5, "uint40_t must be exactly 5 bytes");

struct config_t {
    std::size_t max_elements = 0;
    std::size_t connectivity = 16;
    std::size_t expansion_construction = 200;
    std::size_t expansion_search = 100;
    std::size_t max_threads_add = 64;
    std::size_t max_threads_search = 64;
    dim_t dim = 0;
};

/**
 *  @brief
 *      Approximate Nearest Neighbors Search index using the
 *      Hierarchical Navigable Small World graph algorithm.
 *
 *  @tparam distance_function_at
 *      A function object resposible for computing the distance
 *      between two vectors. Must define:
 *          - `distance_function_at::scalar_t`
 *          - `distance_function_at::result_type`
 *      Must overload the call operator with the following signature:
 *          - `result_type (*) (scalar_t const *, scalar_t const *, dim_t, dim_t)`
 *
 *  @tparam label_at
 *      The type of unique labels to assign to vectors.
 *
 *  @tparam id_at
 *      The smallest unsigned integer type to address indexed elements.
 *      Can be a built-in `uint32_t`, `uint64_t`, or our custom `uint40_t`.
 *
 *  @tparam allocator_at
 *      Dynamic memory allocator to
 */
template <typename distance_function_at = ip_gt<float>, //
          typename label_at = std::size_t,              //
          typename id_at = std::uint32_t,               //
          typename scalar_at = float,                   //
          typename allocator_at = std::allocator<char>> //
class index_gt {
  public:
    using distance_function_t = distance_function_at;
    using scalar_t = scalar_at;
    using label_t = label_at;
    using id_t = id_at;
    using allocator_t = allocator_at;

    using distance_t = typename distance_function_t::result_type;
    using neighbors_count_t = id_t;

  private:
    using allocator_traits_t = std::allocator_traits<allocator_t>;
    using byte_t = typename allocator_t::value_type;
    static_assert(sizeof(byte_t) == 1, "Allocator must allocate separate addressable bytes");
    static constexpr std::size_t base_level_multiple_k = 2;

    using visits_bitset_t = visits_bitset_gt<allocator_t>;

    struct precomputed_constants_t {
        double inverse_log_connectivity{};
        std::size_t connectivity_max_base{};
        std::size_t bytes_per_neighbors{};
        std::size_t bytes_per_neighbors_base{};
        std::size_t bytes_per_mutex{};
    };
    struct distance_and_id_t {
        distance_t first;
        id_t second;
    };
    struct compare_by_distance_t {
        inline bool operator()(distance_and_id_t a, distance_and_id_t b) const noexcept { return a.first < b.first; }
    };

    using distances_and_ids_allocator_t = typename allocator_traits_t::template rebind_alloc<distance_and_id_t>;
    using distances_and_ids_t = max_heap_gt<distance_and_id_t, compare_by_distance_t, distances_and_ids_allocator_t>;

    struct neighbors_ref_t {
        neighbors_count_t& count;
        id_t* neighbors{};

        inline neighbors_ref_t(byte_t* tape) noexcept
            : count(*(neighbors_count_t*)tape), neighbors((neighbors_count_t*)tape + 1) {}
    };

    struct usearch_pack_m point_head_t {
        label_t label;
        dim_t dim;
        level_t level;
        // Variable length array, that has multiple similarly structured segments.
        // Each starts with a `neighbors_count_t` and is followed by such number of `id_t`s.
        byte_t neighbors[1];
    };
    static constexpr std::size_t bytes_per_head_k = sizeof(label_t) + sizeof(dim_t) + sizeof(level_t);

    struct point_and_vector_t {
        byte_t* point_;
        scalar_t* vector_;
    };

    class point_ref_t {
        mutex_t* mutex_{};

      public:
        point_head_t& head{};
        scalar_t* vector{};

        inline point_ref_t(mutex_t& m, point_head_t& h, scalar_t* s) noexcept : mutex_(&m), head(h), vector(s) {}
        inline lock_t lock() const noexcept { return mutex_ ? lock_t{*mutex_} : lock_t{}; }
        inline operator point_and_vector_t() const noexcept {
            return {mutex_ ? (byte_t*)mutex_ : (byte_t*)&head, vector};
        }
    };

    struct usearch_align_m thread_context_t {
        distances_and_ids_t top_candidates;
        distances_and_ids_t candidates_set;
        distances_and_ids_t candidates_filter;
        visits_bitset_t visits;
        std::default_random_engine level_generator;
        distance_function_t distance;
    };

    config_t config_{};
    precomputed_constants_t pre_;
    std::size_t capacity_{0};

    usearch_align_m mutable std::atomic<std::size_t> size_{};
    int viewed_file_descriptor_{};

    mutex_t global_mutex_{};
    level_t max_level_{};
    id_t entry_id_{};

    using point_and_vector_allocator_t = typename allocator_traits_t::template rebind_alloc<point_and_vector_t>;
    std::vector<point_and_vector_t, point_and_vector_allocator_t> points_and_vectors_{};

    using thread_context_allocator_t = typename allocator_traits_t::template rebind_alloc<thread_context_t>;
    mutable std::vector<thread_context_t, thread_context_allocator_t> thread_contexts_;

  public:
    std::size_t dim() const noexcept { return config_.dim; }
    std::size_t connectivity() const noexcept { return config_.connectivity; }
    std::size_t capacity() const noexcept { return capacity_; }
    std::size_t size() const noexcept { return size_; }
    bool is_immutable() const noexcept { return viewed_file_descriptor_ != 0; }
    bool synchronize() const noexcept { return config_.max_threads_add > 1; }

    index_gt(config_t config = {}) : config_(config) {

        // Externally defined hyper-parameters:
        config_.expansion_construction = std::max(config_.expansion_construction, config_.connectivity);
        pre_ = precompute(config);

        // Configure initial empty state:
        size_ = 0u;
        max_level_ = -1;
        entry_id_ = 0u;
        viewed_file_descriptor_ = 0;

        // Dynamic memory:
        thread_contexts_.resize(std::max(config.max_threads_search, config.max_threads_add));
        reserve(config.max_elements);
    }

    ~index_gt() noexcept { clear(); }

#pragma region Adjusting Configuration

    void clear() noexcept {
        std::size_t n = size_;
        for (std::size_t i = 0; i != n; ++i)
            point_free(i);
        size_ = 0;
        max_level_ = -1;
        entry_id_ = 0u;
    }

    void reserve(std::size_t new_capacity) noexcept(false) {

        assert_m(new_capacity >= size_, "Can't drop existing values");
        points_and_vectors_.resize(new_capacity);
        for (thread_context_t& context : thread_contexts_)
            context.visits = visits_bitset_t(new_capacity);

        capacity_ = new_capacity;
    }

    void adjust_dimensions(std::size_t n) noexcept { config_.dim = n; }

    template <typename... args_at> //
    void adjust_metric(args_at&&... args) noexcept {
        for (thread_context_t& context : thread_contexts_)
            context.distance = distance_function_t(std::forward<args_at>(args)...);
    }

    static config_t optimize(config_t const& config) noexcept {
        precomputed_constants_t pre = precompute(config);
        std::size_t bytes_per_point_base = bytes_per_head_k + pre.bytes_per_neighbors_base + pre.bytes_per_mutex;
        std::size_t rounded_size = divide_round_up<64>(bytes_per_point_base) * 64;
        std::size_t added_connections = (rounded_size - rounded_size) / sizeof(id_t);
        config_t result = config;
        result.connectivity = config.connectivity + added_connections / base_level_multiple_k;
        return result;
    }

#pragma endregion

#pragma region Construction and Search

    id_t add(label_t new_label, scalar_t const* new_vector, dim_t new_dim, //
             std::size_t thread_idx = 0, bool store_vector = true) {

        assert_m(!is_immutable(), "Can't add to an immutable index");
        assert_m(size_ < capacity_, "Inserting beyond capacity, reserve first");
        id_t new_id = static_cast<id_t>(size_.fetch_add(1));

        // Determining how much memory to allocate depends on the target level.
        lock_t new_level_lock(global_mutex_);
        level_t max_level = max_level_;
        thread_context_t& context = thread_contexts_[thread_idx];
        level_t new_target_level = choose_random_level(context.level_generator);
        if (new_target_level <= max_level)
            new_level_lock.unlock();

        // Allocate the neighbors
        new_dim = new_dim ? new_dim : static_cast<dim_t>(config_.dim);
        point_ref_t new_point = point_malloc(new_label, new_vector, new_dim, new_target_level, store_vector);
        lock_t new_lock = new_point.lock();
        points_and_vectors_[new_id] = new_point;

        // Do nothing for the first element
        if (!new_id) {
            max_level_ = new_target_level;
            return new_id;
        }

        // Go down the level, tracking only the closest match
        id_t closest_id = entry_id_;
        distance_t closest_dist =
            context.distance(new_vector, point(closest_id).vector, new_dim, point(closest_id).head.dim);
        for (level_t level = max_level; level > new_target_level; level--) {
            bool changed;
            do {
                changed = false;
                point_ref_t closest_point = point(closest_id);
                lock_t closest_lock = closest_point.lock();
                neighbors_ref_t closest_header = neighbors_non_base(closest_point, level);
                iterate_through_neighbors(closest_header, [&](id_t candidate_id) noexcept {
                    point_ref_t candidate_point = point(candidate_id);
                    distance_t candidate_dist =
                        context.distance(new_vector, candidate_point.vector, new_dim, candidate_point.head.dim);
                    if (candidate_dist < closest_dist) {
                        closest_dist = candidate_dist;
                        closest_id = candidate_id;
                        changed = true;
                    }
                });
            } while (changed);
        }

        // From `new_target_level` down perform proper extensive search.
        for (level_t level = std::min(new_target_level, max_level); level >= 0; level--) {
            search_to_insert(closest_id, new_vector, new_dim, level, context);
            closest_id = connect_new_element(new_id, level, context);
        }

        // Releasing lock for the maximum level
        if (new_target_level > max_level) {
            entry_id_ = new_id;
            max_level_ = new_target_level;
        }
        return new_id;
    }

    template <typename label_and_distance_callback_at>
    void search(scalar_t const* query_vec, dim_t query_dim, std::size_t k, std::size_t thread_idx,
                label_and_distance_callback_at&& callback) const {

        if (!size_)
            return;

        // Go down the level, tracking only the closest match
        thread_context_t& context = thread_contexts_[thread_idx];
        id_t closest_id = entry_id_;
        distance_t closest_dist =
            context.distance(query_vec, point(closest_id).vector, query_dim, point(closest_id).head.dim);
        for (level_t level = max_level_; level > 0; level--) {
            bool changed;
            do {
                changed = false;
                point_ref_t closest_point = point(closest_id);
                neighbors_ref_t closest_header = neighbors_non_base(closest_point, level);
                iterate_through_neighbors(closest_header, [&](id_t candidate_id) noexcept {
                    point_ref_t candidate_point = point(candidate_id);
                    distance_t candidate_dist =
                        context.distance(query_vec, candidate_point.vector, query_dim, candidate_point.head.dim);
                    if (candidate_dist < closest_dist) {
                        closest_dist = candidate_dist;
                        closest_id = candidate_id;
                        changed = true;
                    }
                });

            } while (changed);
        }

        // For bottom layer we need a more optimized procedure
        search_to_find_in_base(closest_id, query_vec, query_dim, std::max(config_.expansion_search, k), context);
        while (context.top_candidates.size() > k)
            context.top_candidates.pop();

        while (context.top_candidates.size()) {
            distance_and_id_t top = context.top_candidates.top();
            callback(point(top.second).head.label, top.first);
            context.top_candidates.pop();
        }
    }

#pragma endregion

#pragma region Serialization

    struct state_t {
        std::uint64_t connectivity = 16;
        std::uint64_t dim = 0;
        std::uint64_t size = 0;
        std::uint64_t entry_id = 0;
        std::uint64_t max_level = 0;
    };

    /**
     *  Compatibility: Linux, MacOS, Windows.
     */
    void save(char const* file_path) const noexcept(false) {

        state_t state;
        state.connectivity = config_.connectivity;
        state.dim = config_.dim;
        state.size = size_;
        state.entry_id = entry_id_;
        state.max_level = max_level_;

        std::FILE* file = std::fopen(file_path, "w");
        if (!file)
            throw std::runtime_error(std::strerror(errno));

        // Write the header
        {
            std::size_t written = std::fwrite(&state, sizeof(state), 1, file);
            if (!written) {
                std::fclose(file);
                throw std::runtime_error(std::strerror(errno));
            }
        }

        // Serialize points one by one
        for (std::size_t i = 0; i != state.size; ++i) {
            point_ref_t point_ref = point(static_cast<id_t>(i));
            std::size_t bytes_to_dump = point_dump_size(point_ref.head.dim, point_ref.head.level);
            std::size_t bytes_in_vector = point_ref.head.dim * sizeof(scalar_t);
            std::size_t written = std::fwrite(&point_ref.head, bytes_to_dump - bytes_in_vector, 1, file);
            if (!written) {
                std::fclose(file);
                throw std::runtime_error(std::strerror(errno));
            }
            written = std::fwrite(point_ref.vector, bytes_in_vector, 1, file);
            if (!written) {
                std::fclose(file);
                throw std::runtime_error(std::strerror(errno));
            }
        }

        std::fclose(file);
    }

    /**
     *  Compatibility: Linux, MacOS, Windows.
     */
    void load(char const* file_path) noexcept(false) {
        state_t state;
        std::FILE* file = std::fopen(file_path, "r");
        if (!file)
            throw std::runtime_error(std::strerror(errno));

        // Read the header
        {
            std::size_t read = std::fread(&state, sizeof(state), 1, file);
            if (!read) {
                std::fclose(file);
                throw std::runtime_error(std::strerror(errno));
            }
            config_.dim = state.dim;
            config_.connectivity = state.connectivity;
            config_.max_elements = state.size;
            pre_ = precompute(config_);
            reserve(state.size);
            size_ = state.size;
            max_level_ = state.max_level;
            entry_id_ = state.entry_id;
        }

        // Load points one by one
        for (std::size_t i = 0; i != state.size; ++i) {
            point_head_t head;
            std::size_t read = std::fread(&head, bytes_per_head_k, 1, file);
            if (!read) {
                std::fclose(file);
                throw std::runtime_error(std::strerror(errno));
            }

            std::size_t bytes_to_dump = point_dump_size(head.dim, head.level);
            point_ref_t point_ref = point_malloc(head.label, nullptr, head.dim, head.level, true);
            read = std::fread((byte_t*)&point_ref.head + bytes_per_head_k, bytes_to_dump - bytes_per_head_k, 1, file);
            if (!read) {
                std::fclose(file);
                throw std::runtime_error(std::strerror(errno));
            }

            points_and_vectors_[i] = point_ref;
        }

        std::fclose(file);
        viewed_file_descriptor_ = 0;
    }

    /**
     *  Compatibility: Linux, MacOS.
     */
    void view(char const* file_path) noexcept(false) {
        state_t state;
        int open_flags = O_RDONLY;
#if __linux__
        open_flags |= O_NOATIME;
#endif
        int descriptor = open(file_path, open_flags);

        // Estimate the file size
        struct stat file_stat;
        int fstat_status = fstat(descriptor, &file_stat);
        if (fstat_status < 0) {
            close(descriptor);
            throw std::runtime_error(std::strerror(errno));
        }

        // Map the entire file
        byte_t* file = (byte_t*)mmap(NULL, file_stat.st_size, PROT_READ, MAP_PRIVATE, descriptor, 0);
        if (file == MAP_FAILED) {
            close(descriptor);
            throw std::runtime_error(std::strerror(errno));
        }

        // Read the header
        {
            std::memcpy(&state, file, sizeof(state));
            config_.dim = state.dim;
            config_.connectivity = state.connectivity;
            config_.max_elements = state.size;
            config_.max_threads_add = 0;
            pre_ = precompute(config_);
            reserve(state.size);
            size_ = state.size;
            max_level_ = state.max_level;
            entry_id_ = state.entry_id;
        }

        // Locate every point packed into file
        std::size_t progress = sizeof(state);
        for (std::size_t i = 0; i != state.size; ++i) {
            point_head_t const& head = *(point_head_t const*)(file + progress);
            std::size_t bytes_to_dump = point_dump_size(head.dim, head.level);
            points_and_vectors_[i].point_ = (byte_t*)(file + progress);
            points_and_vectors_[i].vector_ = (scalar_t*)(file + progress + bytes_to_dump - sizeof(scalar_t) * head.dim);
            progress += bytes_to_dump;
            max_level_ = std::max(max_level_, head.level);
        }

        viewed_file_descriptor_ = descriptor;
    }

#pragma endregion

  private:
    inline static precomputed_constants_t precompute(config_t const& config) noexcept {
        precomputed_constants_t pre;
        pre.connectivity_max_base = config.connectivity * base_level_multiple_k;
        pre.inverse_log_connectivity = 1.0 / std::log(static_cast<double>(config.connectivity));
        pre.bytes_per_neighbors = config.connectivity * sizeof(id_t) + sizeof(neighbors_count_t);
        pre.bytes_per_neighbors_base = pre.connectivity_max_base * sizeof(id_t) + sizeof(neighbors_count_t);
        pre.bytes_per_mutex = sizeof(mutex_t) * (config.max_threads_add > 1);
        return pre;
    }

    inline std::size_t point_dump_size(dim_t dim, level_t level) const noexcept {
        return bytes_per_head_k + pre_.bytes_per_neighbors_base + pre_.bytes_per_neighbors * level +
               sizeof(scalar_t) * dim;
    }

    void point_free(std::size_t id) noexcept {

        if (viewed_file_descriptor_)
            return;

        // This function is rarely called and can be as expensive as needed for higher space-efficiency.
        point_and_vector_t& pair = points_and_vectors_[id];
        if (!pair.point_)
            return;

        point_head_t const& head = *(point_head_t const*)(pair.point_ + pre_.bytes_per_mutex);
        std::size_t size_levels = pre_.bytes_per_neighbors_base + pre_.bytes_per_neighbors * head.level;
        bool store_vector = (byte_t*)(pair.point_ + pre_.bytes_per_mutex + bytes_per_head_k + size_levels) == //
                            (byte_t*)(pair.vector_);
        std::size_t size_point =                       //
            pre_.bytes_per_mutex +                     // Optional concurrency-control
            bytes_per_head_k + size_levels +           // Obligatory neighborhood index
            sizeof(scalar_t) * head.dim * store_vector // Optional vector copy
            ;

        allocator_t{}.deallocate(pair.point_, size_point);
        pair = {};
    }

    point_ref_t point_malloc(                             //
        label_t label, scalar_t const* vector, dim_t dim, //
        level_t level, bool store_vector = true) noexcept(false) {

        // This function is rarely called and can be as expensive as needed for higher space-efficiency.
        std::size_t size_levels = pre_.bytes_per_neighbors_base + pre_.bytes_per_neighbors * level;
        std::size_t size_point =                  //
            pre_.bytes_per_mutex +                // Optional concurrency-control
            bytes_per_head_k + size_levels +      // Obligatory neighborhood index
            sizeof(scalar_t) * dim * store_vector // Optional vector copy
            ;

        byte_t* data = (byte_t*)allocator_t{}.allocate(size_point);
        assert_m(data, "Not enough memory for links");

        mutex_t* mutex = synchronize() ? (mutex_t*)data : nullptr;
        scalar_t* scalars = store_vector //
                                ? (scalar_t*)(data + pre_.bytes_per_mutex + bytes_per_head_k + size_levels)
                                : (scalar_t*)(vector);

        std::memset(data, 0, size_point);
        std::memcpy(scalars, vector, sizeof(scalar_t) * dim * (store_vector && vector));

        point_head_t& head = *(point_head_t*)(data + pre_.bytes_per_mutex);
        head.label = label;
        head.dim = dim;
        head.level = level;

        return {*mutex, head, scalars};
    }

    inline point_ref_t point(id_t id) const noexcept {

        point_and_vector_t pair = points_and_vectors_[id];
        byte_t* data = pair.point_;
        mutex_t* mutex = synchronize() ? (mutex_t*)data : nullptr;
        point_head_t& head = *(point_head_t*)(data + pre_.bytes_per_mutex);
        scalar_t* scalars = pair.vector_;

        return {*mutex, head, scalars};
    }

    inline neighbors_ref_t neighbors_base(point_ref_t point) const noexcept { return {point.head.neighbors}; }

    inline neighbors_ref_t neighbors_non_base(point_ref_t point, level_t level) const noexcept {
        return {point.head.neighbors + pre_.bytes_per_neighbors_base + (level - 1) * pre_.bytes_per_neighbors};
    }

    inline neighbors_ref_t neighbors(point_ref_t point, level_t level) const noexcept {
        return level ? neighbors_non_base(point, level) : neighbors_base(point);
    }

    id_t connect_new_element(id_t new_id, level_t level, thread_context_t& context) noexcept(false) {

        point_ref_t new_point = point(new_id);
        distances_and_ids_t& top_candidates = context.top_candidates;
        std::size_t connectivity_max = level ? config_.connectivity : pre_.connectivity_max_base;
        filter_top_candidates_with_heuristic(top_candidates, context.candidates_set, config_.connectivity, context);

        distance_and_id_t const* const top_unordered = top_candidates.data();
        std::size_t const top_count = top_candidates.size();
        id_t next_closest_entry_id = top_unordered[0].second;
        distance_t next_closest_distance = top_unordered[0].first;

        // Outgoing links from `new_id`:
        {
            neighbors_ref_t new_neighbors = neighbors(new_point, level);
            assert_m(!new_neighbors.count, "The newly inserted element should have blank link list");

            new_neighbors.count = static_cast<neighbors_count_t>(top_count);
            for (std::size_t idx = 0; idx < top_count; idx++) {
                assert_m(!new_neighbors.neighbors[idx], "Possible memory corruption");
                assert_m(level <= point(top_unordered[idx].second).head.level, "Linking to missing level");

                new_neighbors.neighbors[idx] = top_unordered[idx].second;
                if (top_unordered[idx].first < next_closest_distance) {
                    next_closest_entry_id = top_unordered[idx].second;
                    next_closest_distance = top_unordered[idx].first;
                }
            }
        }

        // Reverse links from the neighbors:
        for (std::size_t idx = 0; idx < top_count; idx++) {
            id_t close_id = top_unordered[idx].second;
            point_ref_t close_point = point(close_id);
            lock_t close_lock = close_point.lock();

            neighbors_ref_t close_header = neighbors(close_point, level);
            assert_m(close_header.count <= connectivity_max, "Possible corruption");
            assert_m(close_id != new_id, "Self-loops are impossible");
            assert_m(level <= close_point.head.level, "Linking to missing level");

            // If `new_id` is already present in the neighboring connections of `close_id`
            // then no need to modify any connections or run the heuristics.
            if (close_header.count < connectivity_max) {
                close_header.neighbors[close_header.count] = new_id;
                close_header.count++;
                continue;
            }

            // To fit a new connection we need to drop an existing one.
            distances_and_ids_t& candidates = context.candidates_filter;
            candidates.clear();
            candidates.emplace(
                context.distance(new_point.vector, close_point.vector, new_point.head.dim, close_point.head.dim),
                new_id);
            iterate_through_neighbors(close_header, [&](id_t successor_id) noexcept {
                point_ref_t successor_point = point(successor_id);
                candidates.emplace(context.distance(successor_point.vector, close_point.vector,
                                                    successor_point.head.dim, close_point.head.dim),
                                   successor_id);
            });
            filter_top_candidates_with_heuristic(candidates, context.candidates_set, connectivity_max, context);

            // Export the results:
            close_header.count = 0u;
            while (candidates.size()) {
                close_header.neighbors[close_header.count] = candidates.top().second;
                close_header.count++;
                candidates.pop();
            }
        }

        return next_closest_entry_id;
    }

    level_t choose_random_level(std::default_random_engine& level_generator) const noexcept {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        double r = -std::log(distribution(level_generator)) * pre_.inverse_log_connectivity;
        return (level_t)r;
    }

    void search_to_insert(                                         //
        id_t start_id, scalar_t const* query_vec, dim_t query_dim, //
        level_t level, thread_context_t& context) noexcept(false) {

        visits_bitset_t& visits = context.visits;
        distances_and_ids_t& top_candidates = context.top_candidates;
        distances_and_ids_t& candidates_set = context.candidates_set;

        top_candidates.clear();
        candidates_set.clear();
        visits.clear();

        distance_t closest_dist =
            context.distance(query_vec, point(start_id).vector, query_dim, point(start_id).head.dim);
        top_candidates.emplace(closest_dist, start_id);
        candidates_set.emplace(-closest_dist, start_id);
        visits.set(start_id);

        while (!candidates_set.empty()) {

            distance_and_id_t candidacy = candidates_set.top();
            if ((-candidacy.first) > closest_dist && top_candidates.size() == config_.expansion_construction)
                break;

            candidates_set.pop();
            id_t candidate_id = candidacy.second;
            point_ref_t candidate_point = point(candidate_id);
            lock_t candidate_lock = candidate_point.lock();
            neighbors_ref_t candidate_header = neighbors(candidate_point, level);

            iterate_through_neighbors(candidate_header, [&](id_t successor_id) noexcept {
                if (visits.test(successor_id))
                    return;

                visits.set(successor_id);
                point_ref_t successor_point = point(successor_id);
                distance_t successor_dist =
                    context.distance(query_vec, successor_point.vector, query_dim, successor_point.head.dim);
                if (top_candidates.size() < config_.expansion_construction || closest_dist > successor_dist) {
                    candidates_set.emplace(-successor_dist, successor_id);

                    top_candidates.emplace(successor_dist, successor_id);
                    if (top_candidates.size() > config_.expansion_construction)
                        top_candidates.pop();
                    if (!top_candidates.empty())
                        closest_dist = top_candidates.top().first;
                }
            });
        }
    }

    void search_to_find_in_base(                                   //
        id_t start_id, scalar_t const* query_vec, dim_t query_dim, //
        std::size_t expansion, thread_context_t& context) const noexcept(false) {

        visits_bitset_t& visits = context.visits;
        distances_and_ids_t& top_candidates = context.top_candidates;
        distances_and_ids_t& candidates_set = context.candidates_set;

        visits.clear();
        top_candidates.clear();
        candidates_set.clear();

        distance_t closest_dist =
            context.distance(query_vec, point(start_id).vector, query_dim, point(start_id).head.dim);
        top_candidates.emplace(closest_dist, start_id);
        candidates_set.emplace(-closest_dist, start_id);
        visits.set(start_id);

        while (!candidates_set.empty()) {

            distance_and_id_t current_node_pair = candidates_set.top();
            if ((-current_node_pair.first) > closest_dist)
                break;

            candidates_set.pop();

            id_t candidate_id = current_node_pair.second;
            neighbors_ref_t candidate_header = neighbors_base(point(candidate_id));

            iterate_through_neighbors(candidate_header, [&](id_t successor_id) noexcept {
                if (visits.test(successor_id))
                    return;

                visits.set(successor_id);
                point_ref_t successor_point = point(successor_id);
                distance_t successor_dist =
                    context.distance(query_vec, successor_point.vector, query_dim, successor_point.head.dim);

                if (top_candidates.size() < expansion || closest_dist > successor_dist) {
                    candidates_set.emplace(-successor_dist, successor_id);

                    top_candidates.emplace(successor_dist, successor_id);
                    if (top_candidates.size() > expansion)
                        top_candidates.pop();
                    if (!top_candidates.empty())
                        closest_dist = top_candidates.top().first;
                }
            });
        }
    }

    /**
     *  @brief A simple `for`-loop that prefetches vectors of neighbors.
     */
    template <typename neighbor_id_callback_at>
    inline void iterate_through_neighbors(neighbors_ref_t head, neighbor_id_callback_at&& callback) const noexcept {
        std::size_t n = head.count;
        for (std::size_t j = 0; j != n; j++)
            callback(head.neighbors[j]);
    }

    void filter_top_candidates_with_heuristic( //
        distances_and_ids_t& top_candidates, distances_and_ids_t& temporary, std::size_t needed,
        thread_context_t& context) const noexcept(false) {

        if (top_candidates.size() < needed)
            return;

        // TODO: Sort ascending, then run an inplace triangular reduction.
        temporary.clear();
        while (top_candidates.size()) {
            temporary.emplace(-top_candidates.top().first, top_candidates.top().second);
            top_candidates.pop();
        }

        while (temporary.size() && top_candidates.size() < needed) {

            distance_and_id_t best = temporary.top();
            distance_t dist_to_query = -best.first;
            temporary.pop();
            bool good = true;

            distance_and_id_t const* const top_unordered = top_candidates.data();
            std::size_t const top_count = top_candidates.size();
            for (std::size_t idx = 0; idx < top_count; idx++) {
                distance_and_id_t other = top_unordered[idx];
                point_ref_t other_point = point(other.second);
                point_ref_t best_point = point(best.second);
                distance_t inter_result_dist =
                    context.distance(other_point.vector, best_point.vector, other_point.head.dim, best_point.head.dim);
                if (inter_result_dist < dist_to_query) {
                    good = false;
                    break;
                }
            }

            if (good)
                top_candidates.emplace(-best.first, best.second);
        }
    }
};

} // namespace usearch
} // namespace unum
