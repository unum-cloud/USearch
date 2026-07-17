/**
 *  @brief Vector Search index for a single uniform latency memory space.
 *  @author Ash Vardanian
 *  @file include/usearch/inner_volatile_index.hpp
 *  @date July 26, 2023
 */
#pragma once

#include <bit> // `std::bit_width`, `std::bit_floor`

#include <forkunion/types.hpp>

#include "usearch/types.hpp"
#include "usearch/inner_metric.hpp"
#include "usearch/traversal_queue.hpp"

namespace fu = ashvardanian::forkunion;

namespace unum {
namespace usearch {

using inner_node_id_t = u32_t;
using inner_distance_t = f32_t;

struct inner_scored_node_t {
    inner_node_id_t node_id;
    inner_distance_t distance;
};

struct inner_closer_scored_node_t {
    inline bool operator()(inner_scored_node_t a, inner_scored_node_t b) const noexcept {
        return a.distance < b.distance;
    }
};

struct inner_further_scored_node_t {
    inline bool operator()(inner_scored_node_t a, inner_scored_node_t b) const noexcept {
        return a.distance > b.distance;
    }
};

/**
 *  @brief Minimal 8-byte SplitMix64 PRNG for stochastic NSW start-point selection.
 *
 *  Picked over `std::mt19937` (5 KB of per-thread state on libstdc++) and `std::minstd_rand`
 *  (a per-call 64-bit modulo). Quality is ample for spreading entry points; each traversal
 *  context owns one, seeded distinctly per thread so the streams decorrelate.
 */
struct splitmix_u32_t {
    using seed_t = u64_t;
    seed_t state_;

    inline explicit splitmix_u32_t(seed_t seed = 0) noexcept : state_(seed) {}

    /** @brief One SplitMix64 step. */
    inline seed_t unbounded() noexcept {
        seed_t z = (state_ += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }

    /** @brief Index in `[0, bound)` via Lemire's multiply-shift - one multiply, no division.
     *      Bias is ~`bound / 2^32`, negligible for entry-point selection. */
    inline u32_t bounded(u32_t bound) noexcept { return static_cast<u32_t>(unbounded() * bound >> 32); }
};

/**
 *  @brief Vector Search index for a single uniform latency memory space.
 *
 *  @section Supported Metrics: Only "Inner Product class"
 *
 *  This template is designed exclusively for dense representations in the most common metric spaces:
 *  @b Inner-Product, Angular (Cosine), Euclidean (L2), Hamming (Manhattan), and Jaccard (Tanimoto).
 *  All of them can be trivially inferred in a single pass of vector-vector products if L1 and L2
 *  norms of the vectors are known ahead of time and stored locally.
 *
 *  @section inner_volatile_index_growth Growth Limits
 *
 *  This index is designed exclusively for @b up-to-4-billion entries, all stored @b in-memory.
 *  Float32 distance representations are used to compare entries which may result in recall regressions for
 *  small K queries. The typical mode of use is to retrieve significantly more entries from the index of
 *  quantized vectors and rerank them with either pre-quantized vectors or original data.
 *
 *  @section inner_volatile_index_concurrency Concurrency & Transactions
 *
 *  This class is designed to allow simutanous construction and traversal in heavily multi-threaded environments.
 *  One design previously considered is to have a small number of locks proportional to the CPU core count and
 *  graph connectivity, and hash the node IDs to pick the target lock.
 *
 *  Instead, we reuse the per-node neigbors lists to also contain per-node locks. This conveniently places them
 *  at the start of independent cache lines. Besides those, we use a minimal number of "atomics" for book-keeping,
 *  avoiding expensive Compare-And-Swap exchange operations.
 *
 *  @section inner_volatile_index_persistence Persistence
 *
 *  This index is not expected to be accessed from disk. For that, explore io_uring and SPDK reimplementations.
 *
 *  @section inner_volatile_index_malloc Memory Allocation Strategy
 *
 *  Instead of re-allocating one huge matrix and paying the padding memory costs right after the growth phase,
 *  we implement a custom memory allocation strategy. On small-ish indexes we progressively grow through power-of-two
 *  sizes, until we reach a threashold that easily amortizes memory allocations between insertions.
 *  Aiming for @b ~4GB/allocation slabs and compatibility with CXL-backed memory allcoators.
 *
 *
 */
template <                                                          //
    nk_dtype_t vector_dtype_ = nk_bf16_k,                           //
    nk_kernel_kind_t kernel_kind_ = nk_kernel_angular_k,            //
    nk_capability_t capability_ = nk_cap_icelake_k,                 //
    typename vectors_allocator_type_ = std::allocator<std::byte>,   //
    typename neighbors_allocator_type_ = std::allocator<std::byte>, //
    typename traversals_allocator_type_ = std::allocator<std::byte> //
    >
struct inner_volatile_index;

/**
 *  @brief Separate class for handling queues and other dynamic memory structures.
 */
template <                                               //
    nk_capability_t capability_ = nk_cap_icelake_k,      //
    typename allocator_type_ = std::allocator<std::byte> //
    >
struct inner_traversal_context;

template <nk_capability_t capability_, typename allocator_type_>
struct inner_traversal_context {

    using allocator_t = allocator_type_;
    using candidates_allocator_t =
        typename std::allocator_traits<allocator_t>::template rebind_alloc<inner_scored_node_t>;
    using visits_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<inner_node_id_t>;

    using results_array_t = sorted_buffer<inner_scored_node_t, inner_closer_scored_node_t, candidates_allocator_t>;
    using frontier_queue_t = max_heap<inner_scored_node_t, inner_further_scored_node_t, candidates_allocator_t>;
    using visited_set_t = growing_hash_set<inner_node_id_t, std::less<inner_node_id_t>, visits_allocator_t>;

    /** @brief Nodes we've already seen, forming a heap of best results so far. */
    results_array_t results;
    /** @brief Nodes we've already re-evaluated for post-insertion refinement. */
    results_array_t refining_candidates;
    /** @brief Nodes we consider promising and want to evaluate next. */
    frontier_queue_t frontier;
    /** @brief Nodes we've already checked and should avoid visiting again. */
    visited_set_t visited;
    /** @brief Pseudo-random generator for picking traversal starting points. */
    splitmix_u32_t random_generator;

    inner_traversal_context() noexcept = delete;
    inner_traversal_context(inner_traversal_context const&) noexcept = delete;
    inner_traversal_context& operator=(inner_traversal_context const&) noexcept = delete;

    inner_traversal_context(allocator_t alloc) noexcept
        : results(alloc), refining_candidates(alloc), frontier(alloc), visited(alloc), random_generator() {}
    inner_traversal_context(inner_traversal_context&&) = default;
    inner_traversal_context& operator=(inner_traversal_context&&) = default;
};

template <                                                                                //
    nk_dtype_t vector_dtype_, nk_kernel_kind_t kernel_kind_, nk_capability_t capability_, //
    typename vectors_allocator_type_, typename neighbors_allocator_type_,
    typename traversals_allocator_type_> //
struct inner_volatile_index {

    inline static constexpr nk_dtype_t vector_dtype_k = vector_dtype_;

    using scalar_t = typename nk::type_for<vector_dtype_>::type;
    using vector_product_result_t = typename scalar_t::dot_result_t;
    using distance_t = f32_t;
    using candidate_t = inner_scored_node_t;
    using key_t = std::uint64_t;

    /** Default integer type to use for loops and indexing at sub-index scale. */
    using small_size_t = std::uint32_t;

    using vectors_allocator_t =
        typename std::allocator_traits<vectors_allocator_type_>::template rebind_alloc<scalar_t>;
    using neighbors_allocator_t =
        typename std::allocator_traits<neighbors_allocator_type_>::template rebind_alloc<u32_t>;
    using candidates_allocator_t =
        typename std::allocator_traits<traversals_allocator_type_>::template rebind_alloc<candidate_t>;

    using vector_view_t = nk::vector_view<scalar_t>;
    using vector_span_t = nk::vector_span<scalar_t>;
    using vector_norms_t = vector_norms<vector_product_result_t>;

    using keys_view_t = nk::vector_view<key_t>;
    using vectors_view_t = nk::matrix_view<scalar_t>;
    using node_ids_span_t = nk::vector_span<node_id_t>;
    using node_ids_view_t = nk::vector_view<node_id_t>;

    inline static constexpr nk_kernel_kind_t kernel_kind_k = kernel_kind_;
    static_assert( //
        kernel_kind_k == nk_kernel_dot_k || kernel_kind_k == nk_kernel_angular_k ||
        kernel_kind_k == nk_kernel_euclidean_k || kernel_kind_k == nk_kernel_sqeuclidean_k ||
        kernel_kind_k == nk_kernel_hamming_k || kernel_kind_k == nk_kernel_jaccard_k);

    /**
     *  @brief Each thread aiming to perform scans over the graph needs to have its own object like this.
     */
    using inner_traversal_context_t = inner_traversal_context<capability_, candidates_allocator_t>;
    using results_array_t = typename inner_traversal_context_t::results_array_t;
    using frontier_queue_t = typename inner_traversal_context_t::frontier_queue_t;
    using visited_set_t = typename inner_traversal_context_t::visited_set_t;

    using norms_allocator_t =
        typename std::allocator_traits<vectors_allocator_type_>::template rebind_alloc<vector_norms_t>;

    using vectors_chunk_t = nk::matrix<scalar_t, vectors_allocator_t>;
    using norms_chunk_t = nk::tensor<vector_norms_t, norms_allocator_t, 1>;
    using neighbors_chunk_t = nk::matrix<u32_t, neighbors_allocator_t>;

    /** @brief Rows in the very first chunk, as a power-of-two exponent. @ref inner_volatile_index_malloc */
    inline static constexpr small_size_t first_chunk_rows_log2_k = 10; // 1'024 rows
    /** @brief Rows in every mature slab-sized chunk, as a power-of-two exponent. @ref inner_volatile_index_malloc */
    inline static constexpr small_size_t slab_rows_log2_k = 20; // 1'048'576 rows
    /** @brief Chunks in the doubling phase of the growth schedule: the first chunk plus one per doubling. */
    inline static constexpr small_size_t doubling_phase_chunks_k = slab_rows_log2_k - first_chunk_rows_log2_k + 1;
    /** @brief Node ids covered by the doubling phase: 2^(slab_rows_log2_k + 1) - 2^first_chunk_rows_log2_k. */
    inline static constexpr node_id_t doubling_phase_capacity_k = power_of_two<node_id_t>(slab_rows_log2_k + 1) -
                                                                  power_of_two<node_id_t>(first_chunk_rows_log2_k);
    /** @brief Upper bound on the chunk count, addressing the entire 2^32 id space. @ref inner_volatile_index_growth */
    inline static constexpr small_size_t max_chunks_k =
        doubling_phase_chunks_k +
        static_cast<small_size_t>(((u64_t(1) << 32) - doubling_phase_capacity_k) >> slab_rows_log2_k);

    /**
     *  @brief Separately allocated matrix slabs, where each row is one vector entry. Slot indexes are global,
     *      monotonically transparently growing across individual matrices.
     *
     *  Each directory is sized once to @p max_chunks_k entries in @c reserve() and never reallocates
     *  after that, so readers can dereference chunk headers with zero synchronization while a single
     *  writer default-constructs the next one in place. The stride pads every header to
     *  @c fu::default_alignment_k, keeping the writer's half-built entry off the readers' cache lines.
     *  @ref inner_volatile_index_malloc
     */
    fu::dynamic_padded_array<vectors_chunk_t, vectors_allocator_type_> vectors_chunks_;

    /**
     *  @brief Separately allocated array slabs, where each entry holds the pre-computed norms of one vector.
     *      Slot indexes are global, monotonically transparently growing across individual arrays.
     *  @ref inner_volatile_index_malloc
     */
    fu::dynamic_padded_array<norms_chunk_t, vectors_allocator_type_> norms_chunks_;

    /**
     *  @brief Separately allocated matrix slabs, where each row is the fixed-capacity neighbors list of one vector.
     *      Slot indexes are global, monotonically transparently growing across individual matrices.
     *  @ref inner_volatile_index_malloc
     */
    fu::dynamic_padded_array<neighbors_chunk_t, neighbors_allocator_type_> neighbors_chunks_;

    /**
     *  @brief Signed integer describing "unsigned capacity" to simplify underflow/overflow detection from concurrent
     *      thread updates. @b Never assign values to it, always use increment & decrement to move it up and down.
     */
    std::atomic<std::ptrdiff_t> remaining_capacity_ {0};

    /**
     *  @brief Unsigned integer containing the identifier of the last "allocated" node. It defines the lower
     *      bound for the number of initialized or empty rows across @p vectors_chunks_ and @p neighbors_chunks_
     *      chunks. It's typically incremented in batches.
     */
    std::atomic<node_id_t> last_allocated_id_ {0};

    /**
     *  @brief Unsigned integer containing the identifier of the last "filled" node. It defines the lower
     *      bound for the number of initialized non-empty rows across in @p vectors_chunks_ chunks, that can
     *      be used for randomly picked landmarks. It's typically incremented one by one.
     */
    std::atomic<node_id_t> next_visible_id_ {0};
    /**
     *  @brief Unsigned integer containing the identifier of the last "filled" node. It defines the lower
     *      bound for the number of initialized non-empty rows across in @p vectors_chunks_ chunks, that can
     *      be used for randomly picked landmarks. It's typically incremented one by one.
     */
    std::atomic<node_id_t> last_visible_id_ {0};

    /** @brief Scalar dimensions per vector row; frozen by @c reserve() before the first insertion. */
    small_size_t dimensions_ = 0;

    /** @brief Neighbor ids per node's list - a power of two, tiling whole cache lines exactly.
     *      The leading slot doubles as the lock @b without sacrificing a neighbor slot.
     *      Frozen by @c reserve() before the first insertion. */
    connectivity_t connectivity_ = connectivity_unknown_k;

    /** @brief Constructed chunks per directory. Deliberately non-atomic: only the one "responsible"
     *      grower thread mutates it, and successive growers are ordered by the acquire/release pair
     *      on @p remaining_capacity_. @ref create_next_chunk */
    small_size_t chunks_count_ = 0;

    /** @brief Allocator instance handed to every vectors & norms chunk. */
    [[no_unique_address]] vectors_allocator_t vectors_allocator_;

    /** @brief Allocator instance handed to every neighbors chunk. */
    [[no_unique_address]] neighbors_allocator_t neighbors_allocator_;

    inner_volatile_index(inner_volatile_index const&) noexcept = delete;
    inner_volatile_index& operator=(inner_volatile_index const&) noexcept = delete;

    inner_volatile_index(inner_volatile_index&&) noexcept = delete;
    inner_volatile_index& operator=(inner_volatile_index&&) noexcept = delete;

    /**
     *  @brief Constructs an empty index, remembering the allocators to hand to every future chunk.
     *
     *  The chunk directories receive a stride padding every chunk header to a multiple of
     *  @c fu::default_alignment_k, so the grower's half-built header never shares a cache line
     *  with the ones readers dereference. Their storage is allocated lazily, ahead of the first chunk.
     */
    inner_volatile_index(vectors_allocator_type_ const& vectors_allocator = {},
                         neighbors_allocator_type_ const& neighbors_allocator = {}) noexcept
        : vectors_chunks_(vectors_allocator,
                          fu::round_up_to_multiple(sizeof(vectors_chunk_t), fu::default_alignment_k)),
          norms_chunks_(vectors_allocator, fu::round_up_to_multiple(sizeof(norms_chunk_t), fu::default_alignment_k)),
          neighbors_chunks_(neighbors_allocator,
                            fu::round_up_to_multiple(sizeof(neighbors_chunk_t), fu::default_alignment_k)),
          vectors_allocator_(vectors_allocator), neighbors_allocator_(neighbors_allocator) {}

  private:
    /** @brief Trampoline for @c usearch::inner_norms with hardened types. */
    inline static vector_norms_t inner_norms(vector_view_t vec) noexcept {
        return usearch::inner_norms<capability_, scalar_t>(vec);
    }
    /** @brief Trampoline for @c usearch::distance with hardened types. */
    inline static distance_t distance(                   //
        vector_view_t query, vector_norms_t query_norms, //
        vector_view_t candidate, vector_norms_t candidate_norms) noexcept {
        return usearch::distance<kernel_kind_, capability_, scalar_t>( //
            query, query_norms, candidate, candidate_norms);
    }
    /** @brief Trampoline for @c usearch::distances with hardened types. */
    inline static std::array<distance_t, 4> distances(   //
        vector_view_t query, vector_norms_t query_norms, //
        std::array<vector_view_t, 4> candidates, std::array<vector_norms_t, 4> candidates_norms) noexcept {
        return usearch::distances<kernel_kind_, capability_, scalar_t>( //
            query, query_norms, candidates, candidates_norms);
    }

    /** @brief Describes in which chunk and at which row inside the chunk a certain entry is located. */
    struct chunked_location_t {
        small_size_t chunk_index = 0;
        small_size_t index_in_chunk = 0;
    };

    /** @brief Maps the global @p node_id onto a chunk and a row within it via a @b closed-form
     *      analytical solution - O(1) time and memory, no per-chunk metadata to walk or binary-search.
     *
     *  Chunk sizes follow the growth schedule from @ref inner_volatile_index_malloc: the first chunk
     *  holds 2^ @p first_chunk_rows_log2_k rows, every next chunk doubles, until reaching the slab
     *  size of 2^ @p slab_rows_log2_k rows - all chunks after that keep that fixed slab size.
     *  Doubling gives closed-form prefix sums, so the lookup is one `lzcnt` plus a few adds -
     *  cheap enough to call on every traversal hop without caching locations.
     */
    inline chunked_location_t chunked_location(node_id_t node_id) const noexcept {
        // The doubling phase spans chunks [0, slab_rows_log2_k - first_chunk_rows_log2_k], where chunk `j`
        // covers ids [2^(first_chunk_rows_log2_k + j) - 2^first_chunk_rows_log2_k, twice that upper bound).
        if (node_id < doubling_phase_capacity_k) {
            // Biasing by the first chunk's size makes the leading bit of the id encode the chunk index.
            node_id_t const biased_id = node_id + power_of_two<node_id_t>(first_chunk_rows_log2_k);
            small_size_t const biased_id_log2 = std::bit_width(biased_id) - 1;
            return {biased_id_log2 - first_chunk_rows_log2_k, biased_id - std::bit_floor(biased_id)};
        }
        node_id_t const id_in_flat_phase = node_id - doubling_phase_capacity_k;
        return {doubling_phase_chunks_k + static_cast<small_size_t>(id_in_flat_phase >> slab_rows_log2_k),
                static_cast<small_size_t>(id_in_flat_phase & power_of_two_mask<node_id_t>(slab_rows_log2_k))};
    }

    /** @brief Total rows across the first @p chunks_count chunks of the growth schedule -
     *      the same closed-form prefix sums that @ref chunked_location inverts. */
    inline static constexpr std::size_t capacity_of_chunks(small_size_t chunks_count) noexcept {
        if (chunks_count <= doubling_phase_chunks_k)
            return power_of_two<std::size_t>(first_chunk_rows_log2_k + chunks_count) -
                   power_of_two<std::size_t>(first_chunk_rows_log2_k);
        return static_cast<std::size_t>(doubling_phase_capacity_k) +
               (static_cast<std::size_t>(chunks_count - doubling_phase_chunks_k) << slab_rows_log2_k);
    }

    /** @brief Immutable view of the vector stored at @p location. */
    inline vector_view_t vector_for(chunked_location_t const& location) const noexcept {
        return vectors_chunks_[location.chunk_index].row(location.index_in_chunk).as_vector();
    }

    /** @brief Mutable span over the vector slot at @p location - only for the slot's unpublished owner. */
    inline vector_span_t vector_span_for(chunked_location_t const& location) noexcept {
        return vectors_chunks_[location.chunk_index].row(location.index_in_chunk).as_vector();
    }

    /** @brief Copy of the pre-computed norms of the vector stored at @p location. */
    inline vector_norms_t norm_for(chunked_location_t const& location) const noexcept {
        return norms_chunks_[location.chunk_index][location.index_in_chunk];
    }

    /** @brief Mutable reference to the norms slot at @p location - only for the slot's unpublished owner. */
    inline vector_norms_t& norm_ref_for(chunked_location_t const& location) noexcept {
        return norms_chunks_[location.chunk_index][location.index_in_chunk];
    }

    /** @brief Mutable span over the fixed-capacity neighbors list at @p location, whose leading
     *      slot doubles as the lock sentinel. @ref inner_volatile_index_concurrency */
    inline node_ids_span_t neighbors_for(chunked_location_t const& location) noexcept {
        return neighbors_chunks_[location.chunk_index].row(location.index_in_chunk).as_vector();
    }
    /** @brief Immutable view of the fixed-capacity neighbors list at @p location. */
    inline node_ids_view_t neighbors_for(chunked_location_t const& location) const noexcept {
        return neighbors_chunks_[location.chunk_index].row(location.index_in_chunk).as_vector();
    }

    /**
     *  @brief Allocates & publishes chunks until at least @p minimal_capacity fresh rows are added,
     *      following the growth schedule of @ref inner_volatile_index_malloc.
     *  @return The number of added rows - possibly non-zero even when the status is a failure,
     *      in which case the caller must still credit them back to @p remaining_capacity_.
     *
     *  Only one thread may run this at a time: the "responsible" one that dropped
     *  @p remaining_capacity_ below zero. Successive responsible threads are ordered by the
     *  acquire/release pair on that counter, which is also what publishes the non-atomic
     *  @p chunks_count_ and the directory writes. Readers never touch a chunk before some node
     *  inside it becomes visible, and that visibility chains through the same atomics.
     */
    expected<std::size_t> create_next_chunk(std::size_t minimal_capacity) noexcept {
        // The never-reallocating chunk directories are sized once, by `reserve()`;
        // inserting without a prior successful reservation is a usage error, not an allocation failure.
        if (!vectors_chunks_.size()) return {0, status_t::uninitialized_k};

        std::size_t added_capacity = 0;
        while (added_capacity < minimal_capacity) {
            if (chunks_count_ == max_chunks_k) return {added_capacity, status_t::overflow_k};

            // Chunk `c` holds 2^(first_chunk_rows_log2_k + c) rows through the doubling phase,
            // and a constant 2^slab_rows_log2_k rows afterwards.
            small_size_t const unclamped_rows_log2 = first_chunk_rows_log2_k + chunks_count_;
            small_size_t const chunk_rows_log2 = unclamped_rows_log2 < slab_rows_log2_k //
                                                     ? unclamped_rows_log2
                                                     : slab_rows_log2_k;
            std::size_t const chunk_rows = power_of_two<std::size_t>(chunk_rows_log2);

            vectors_chunk_t vectors_chunk = vectors_chunk_t::try_zeros( //
                {chunk_rows, dimensions_}, vectors_allocator_);
            norms_chunk_t norms_chunk = norms_chunk_t::try_zeros( //
                {chunk_rows}, norms_allocator_t(vectors_allocator_));
            // Filling the neighbors with the all-ones sentinel is load-bearing: a sentinel leading
            // slot reads as "unpublished" to every traversal, and sentinel tails terminate shorter
            // lists - so no explicit neighbor counts are stored anywhere.
            neighbors_chunk_t neighbors_chunk = neighbors_chunk_t::try_full( //
                {chunk_rows, static_cast<std::size_t>(connectivity_)}, inner_node_id_unknown_k, neighbors_allocator_);
            if (vectors_chunk.empty() || norms_chunk.empty() || neighbors_chunk.empty())
                return {added_capacity, status_t::bad_alloc_k};

            vectors_chunks_[chunks_count_] = std::move(vectors_chunk);
            norms_chunks_[chunks_count_] = std::move(norms_chunk);
            neighbors_chunks_[chunks_count_] = std::move(neighbors_chunk);
            ++chunks_count_;
            added_capacity += chunk_rows;
        }
        return {added_capacity, status_t::success_k};
    }

    /** @brief Atomic reference to the leading slot of a node's @p neighbors list - its combined
     *      lifecycle indicator & lock: zero while unpublished or locked for editing, otherwise the
     *      neighbors count plus one. The slot sits at the start of the row, so on chunk row widths
     *      matching @ref connectivity_t it lands on its own cache-line boundary.
     *      @ref inner_volatile_index_concurrency */
    inline static std::atomic_ref<node_id_t> neighbors_lock_ref(node_ids_span_t neighbors) noexcept {
        return std::atomic_ref<node_id_t>(*neighbors.data());
    }

  public:
    /**
     *  @brief Returns the number of keys & vectors in the index.
     */
    std::size_t size() const noexcept { return last_visible_id_.load(); }

    /** @brief Slots per neighbors list, including the leading lock/indicator slot. */
    std::size_t connectivity() const noexcept { return static_cast<std::size_t>(connectivity_); }

    struct reserve_result_t {
        std::size_t new_capacity = 0;
        connectivity_t connectivity = connectivity_unknown_k;
        status_t status = status_t::success_k;
    };

    struct reserve_capacity_t {
        std::size_t capacity = 0;
        connectivity_t connectivity = connectivity_32_k;
        /** @brief Scalar dimensions per vector; required before the first insertion. */
        std::size_t dimensions = 0;
    };

    /**
     *  @brief Pre-allocates the chunk directories & enough chunks to fit @p config.capacity entries.
     *
     *  Constructors deliberately allocate nothing, matching the design of ForkUnion's dynamic
     *  structures - this is the only entry point that creates directory storage, and it reports
     *  failures in the returned @c reserve_result_t instead of leaving half-built state behind.
     *
     *  Unlike the rest of the interface, this call is @b not safe to run concurrently with
     *  insertions or searches - invoke it once, right after construction. The first successful
     *  call freezes the vector @p dimensions and @p connectivity; later calls can only grow
     *  capacity and echo the frozen connectivity back in the result.
     */
    reserve_result_t reserve(reserve_capacity_t config) noexcept {
        // The very first reservation freezes the geometry and sizes the never-reallocating
        // directories to their lifetime maximum - one allocation each, ~4'105 padded headers.
        if (chunks_count_ == 0) {
            connectivity_ = config.connectivity;
            dimensions_ = static_cast<small_size_t>(config.dimensions);
            if (!vectors_chunks_.size() &&
                (!vectors_chunks_.try_resize(max_chunks_k) || !norms_chunks_.try_resize(max_chunks_k) ||
                 !neighbors_chunks_.try_resize(max_chunks_k)))
                return {0, connectivity_, status_t::bad_alloc_k};
        }
        std::size_t const current_capacity = capacity_of_chunks(chunks_count_);
        if (config.capacity <= current_capacity) return {current_capacity, connectivity_, status_t::success_k};
        auto [added_capacity, chunks_status] = create_next_chunk(config.capacity - current_capacity);
        remaining_capacity_.fetch_add(added_capacity, std::memory_order_release);
        return {current_capacity + added_capacity, connectivity_, chunks_status};
    }

    struct insert_result_t {
        status_t status = status_t::success_k;
    };

    struct insert_config_t {
        small_size_t expansion_factor;
    };

    /**
     *  @brief Inserts a batch of @p new_keys mapping to @p new_values.
     */
    status_t insert(keys_view_t new_keys, vectors_view_t new_vectors, inner_traversal_context_t& context,
                    insert_config_t config = {}) noexcept {

        std::size_t const incoming_vectors_count = new_vectors.size();
        if (!incoming_vectors_count) return success_k;
        std::size_t const node_connectivity = connectivity();

        // NSW and HNSW use an "ef" expansion factor controlling the breadth of the candidate pool,
        // tunable independently at insertion and search time. Default it to twice the connectivity -
        // enough slack for the neighbor-selection heuristic to prune - unless the caller overrides it.
        std::size_t const expansion_factor = config.expansion_factor ? config.expansion_factor //
                                                                     : node_connectivity * 2;

        // Check if we need to perform new memory allocations.
        // Keep in mind, another thread may be also inserting, searching, or deleting entries,
        // so we need a "polling" loop to cover all of the state machine corner cases.
        node_id_t incoming_first_id = node_id_unknown_k;
        while (true) {
            std::ptrdiff_t const old_remaining_capacity = remaining_capacity_.fetch_sub( //
                incoming_vectors_count, std::memory_order_acquire);
            std::ptrdiff_t const new_remaining_capacity = old_remaining_capacity -
                                                          static_cast<std::ptrdiff_t>(incoming_vectors_count);
            // If this is the thread that moved the counter under the watermark,
            // it's responsible for the next allocation
            if (old_remaining_capacity > 0 && new_remaining_capacity < 0) {
                auto [added_capacity, chunk_status] = create_next_chunk(incoming_vectors_count);
                // Check if the chunk allocation has failed
                if (chunk_status != status_t::success_k) {
                    // Revert our decrement - crediting any partially added chunks along the way -
                    // so that a smaller insertion may succeed later
                    remaining_capacity_.fetch_add(incoming_vectors_count + added_capacity, std::memory_order_release);
                    return chunk_status;
                }
                // Chunk allocation succeeded!
                else {
                    incoming_first_id = last_allocated_id_.fetch_add(incoming_vectors_count, std::memory_order_acquire);
                    remaining_capacity_.fetch_add(added_capacity, std::memory_order_release);
                }
                // Break out of the capacity "polling" loop.
                break;
            }
            // We may be lucky and already have enough space reserved for all of those vector
            else if (new_remaining_capacity >= 0) {
                incoming_first_id = last_allocated_id_.fetch_add(incoming_vectors_count, std::memory_order_acquire);
                // Break out of the capacity "polling" loop.
                break;
            }
            // We may face a corner case, when index has 10 slots, another thread Y tries to insert 20 entries,
            // and this thread X - is looking to insert a 100. So even if Y has just allocated some memory,
            // it may not be enough for us.
            else {
                // Let's revert the counter and "poll" again!
                remaining_capacity_.fetch_add(incoming_vectors_count, std::memory_order_release);
                std::this_thread::yield();
                continue;
            }
        }

        // At the very beginning, when the index is empty, we only need to wire the incoming points with each other
        if (incoming_first_id == 0) {}

        // Go through the vectors and inject them one-by-one.
        for (small_size_t incoming_index = 0; incoming_index != incoming_vectors_count; ++incoming_index) {
            node_id_t const new_node_id = next_visible_id_.fetch_add(1);
            vector_view_t const new_vector = new_vectors.row(incoming_index);
            vector_norms_t const new_norms = inner_norms(new_vector);
            top_candidates_array_t& results = context.results;
            top_candidates_array_t& refining_candidates = context.refining_candidates;
            frontier_queue_t& frontier = context.frontier;
            visited_set_t& visited = context.visited;

            // Let's first copy the `new_vector` into the owned buffers, so we

            // Let's random-sample a few "landmark" points in our graph to traverse from there onwards.
            // We want at least as many landmarks as the connectivity of our graph and prefer to unroll
            // those "gathers" into 4-way batches.
            for (small_size_t landmarks_batch = 0; landmarks_batch != node_connectivity; ++landmarks_batch) {

                std::array<node_id_t, 4> landmarks_ids;
                std::array<vector_view_t, 4> landmarks_vectors;
                std::array<vector_norms_t, 4> landmarks_norms;
                small_size_t landmarks_in_batch = 0;

                // It's possible, that the node ID has been allocated, but the node hasn't been filled yet.
                while (landmarks_in_batch != landmarks_ids.capacity()) {
                    node_id_t const landmark_node_id = context.random_generator.bounded(incoming_first_id);
                    chunked_location_t const landmark_location = chunked_location(landmark_node_id);
                    node_ids_span_t landmark_neighbors = neighbors_for(landmark_location);
                    std::atomic_ref<node_id_t> landmark_indicator = neighbors_lock_ref(landmark_neighbors);

                    // If the node is currently uninitialized or locked, loop again, until we end up with
                    // a randomly chosen landmark that's not yet booked
                    if (landmark_indicator.load(std::memory_order_acquire) == 0) continue;
                    landmarks_ids[landmarks_in_batch] = landmark_node_id;
                    landmarks_vectors[landmarks_in_batch] = vector_for(landmark_location);
                    landmarks_norms[landmarks_in_batch] = norm_for(landmark_location);
                    ++landmarks_in_batch;
                }

                // Compute the distances and export them into our queues
                std::array<distance_t, 4> const landmark_distances = distances( //
                    new_vector, new_norms, landmarks_vectors, landmarks_norms);
                visited.insert(landmarks_ids);

                std::array<inner_scored_node_t, 4> landmarks_scored = {
                    inner_scored_node_t(landmarks_ids[0], landmark_distances[0]),
                    inner_scored_node_t(landmarks_ids[1], landmark_distances[1]),
                    inner_scored_node_t(landmarks_ids[2], landmark_distances[2]),
                    inner_scored_node_t(landmarks_ids[3], landmark_distances[3]),
                };
                frontier.insert(landmarks_scored);
                results.insert(landmarks_scored);
            }

            // Greedily walk the frontier nearest-first, keeping the best `expansion_factor` in `results`.
            while (!frontier.empty()) {
                // Peek - don't pop yet: the chosen node's neighbor list may be locked, in which case
                // we retry the selection rather than dropping it from the frontier.
                inner_scored_node_t const candidate = frontier.top();
                // `results` stays sorted ascending, so its back is the furthest kept - our search radius.
                distance_t const radius = results.top().distance;
                // Once the nearest unexpanded node is past that radius and `results` is full,
                // nothing deeper can improve the answer.
                if (candidate.distance > radius && results.size() == expansion_factor) break;

                chunked_location_t const candidate_location = chunked_location(candidate.node_id);
                node_ids_span_t candidate_neighbors = neighbors_for(candidate_location);
                std::atomic_ref<node_id_t> candidate_indicator = neighbors_lock_ref(candidate_neighbors);

                // If the node is currently uninitialized or locked, loop again. It will either free up by that
                // time or the probabilitic rank-weighted will pick a different `candidate` that's available.
                if (candidate_indicator.load(std::memory_order_acquire) == 0) continue;

                // The node is live and readable - commit to it and drop it from the frontier.
                frontier.pop();

                // Let's gather the neighbors of our candidate
                for (small_size_t next_batch = 0; next_batch != node_connectivity; ++next_batch) {}
            }
        }
    }

    /**
     *  @brief Entry removals are explicitly banned to simplify concurrency management.
     */
    status_t remove(keys_view_t, inner_traversal_context_t&) noexcept = delete;

    status_t search(vectors_view_t query_vectors, inner_traversal_context_t& context) noexcept {}
};

} // namespace usearch
} // namespace unum
