/**
 *  @brief Vector Search index for a single uniform latency memory space.
 *  @author Ash Vardanian
 *  @file include/usearch/inner_volatile_index.hpp
 *  @date July 26, 2023
 */
#pragma once

#include "usearch/inner_metric.hpp"
#include "usearch/traversal_queue.hpp"

namespace unum {
namespace usearch {

using u32_t = std::uint32_t;
using inner_node_id_t = u32_t;
using inner_distance_t = f32_t;

struct inner_candidate_t {
    inner_node_id_t slot;
    inner_distance_t distance;
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
 *  The number of locks (userspace mutexes) is proportion to the number of configured threads, not the index size.
 *  All threads are expected to have the same access latency to the entries in this index. As for @b Visibility
 *  contracts in this class, insertions follow a two-stage commit semantic, where changes to the graph are not
 *  made visible to other threads until the change is committed.
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
    using candidates_allocator_t =
        typename std::allocator_traits<allocator_type_>::template rebind_alloc<inner_candidate_t>;
    using visits_allocator_t = typename std::allocator_traits<allocator_type_>::template rebind_alloc<inner_node_id_t>;

    using top_candidates_array_t =
        sorted_buffer<inner_candidate_t, std::less<inner_candidate_t>, candidates_allocator_t>;
    using next_candidates_heap_t = max_heap<inner_candidate_t, std::less<inner_candidate_t>, candidates_allocator_t>;
    using visits_hash_set_t = growing_hash_set<inner_node_id_t, std::less<inner_node_id_t>, visits_allocator_t>;

    /** @brief Nodes we've already seen, forming a heap of best results so far. */
    top_candidates_array_t found_candidates;
    /** @brief Nodes we've already re-evaluated for post-insertion refinement. */
    top_candidates_array_t refined_candidates;
    /** @brief Nodes we consider promising and want to evaluate next. */
    next_candidates_heap_t next_candidates {};
    /** @brief Nodes we've already checked and should avoid visiting again. */
    visits_hash_set_t visits {};
};

template <                                                                                //
    nk_dtype_t vector_dtype_, nk_kernel_kind_t kernel_kind_, nk_capability_t capability_, //
    typename vectors_allocator_type_, typename neighbors_allocator_type_,
    typename traversals_allocator_type_> //
struct inner_volatile_index {

    inline static constexpr nk_dtype_t vector_dtype_k = vector_dtype_;
    using vector_dtype_t = typename nk::type_for<vector_dtype_>::type;
    using vector_product_result_t = typename vector_dtype_t::dot_result_t;
    using distance_t = f32_t;
    using candidate_t = inner_candidate_t;

    using vectors_allocator_t =
        typename std::allocator_traits<vectors_allocator_type_>::template rebind_alloc<vector_dtype_t>;
    using neighbors_allocator_t =
        typename std::allocator_traits<neighbors_allocator_type_>::template rebind_alloc<u32_t>;
    using candidates_allocator_t =
        typename std::allocator_traits<traversals_allocator_type_>::template rebind_alloc<candidate_t>;

    using vector_view_t = nk::vector_view<vector_dtype_t>;
    using vector_norms_t = vector_norms<vector_product_result_t>;

    inline static constexpr nk_kernel_kind_t kernel_kind_k = kernel_kind_;
    static_assert( //
        kernel_kind_k == nk_kernel_dot_k || kernel_kind_k == nk_kernel_angular_k ||
        kernel_kind_k == nk_kernel_euclidean_k || kernel_kind_k == nk_kernel_sqeuclidean_k ||
        kernel_kind_k == nk_kernel_hamming_k || kernel_kind_k == nk_kernel_jaccard_k);

    /**
     *  @brief Each thread aiming to perform scans over the graph needs to have its own object like this.
     */
    using inner_traversal_context_t = inner_traversal_context<capability_, candidates_allocator_t>;

    using vectors_chunk_t = nk::tensor<vector_dtype_t, 2>;

    /**
     *  @brief Separately allocated matrix slabs, where each row is one vector entry. Slot indexes are global,
     *      monotonically transperantly growing across individual matrices.
     *  @ref inner_volatile_index_malloc
     */
    padded_array<vectors_chunk_t> vectors_chunks_;

    padded_array<vector_norms_t> vectors_norms_;

    using neighbors_chunk_t = nk::tensor<u32_t, 2>;

    /**
     *  @brief Separately allocated matrix slabs, where each row is the fixed-capacity neighbors list of one vector.
     *      Slot indexes are global, monotonically transperantly growing across individual matrices.
     *  @ref inner_volatile_index_malloc
     */
    padded_array<neighbors_chunk_t> neighbors_chunks_;

    /**
     *  @brief Stops competing transactions from simultaneusly updating the same graph entries.
     *  @ref inner_volatile_index_concurrency
     */
    bitset_t locks_;

    /** @brief Trampoline for @c usearch::inner_norms with hardened types. */
    inline static vector_norms_t inner_norms(vector_view_t vec) noexcept {
        return usearch::inner_norms<capability_, vector_dtype_t>(vec);
    }
    /** @brief Trampoline for @c usearch::distance with hardened types. */
    inline static distance_t distance(                   //
        vector_view_t query, vector_norms_t query_norms, //
        vector_view_t candidate, vector_norms_t candidate_norms) noexcept {
        return usearch::distance<kernel_kind_, capability_, vector_dtype_t>( //
            query, query_norms, candidate, candidate_norms);
    }
    /** @brief Trampoline for @c usearch::distances with hardened types. */
    inline static std::array<distance_t, 4> distances(   //
        vector_view_t query, vector_norms_t query_norms, //
        std::array<vector_view_t, 4> candidates, std::array<vector_norms_t, 4> candidates_norms) noexcept {
        return usearch::distances<kernel_kind_, capability_, vector_dtype_t>( //
            query, query_norms, candidates, candidates_norms);
    }

    /** @brief Given the @p node_id, finds which chunk in @p vectors_chunks_ and @p neighbors_chunks_
     *      contains relevant data. */
    inline std::size_t chunk_for_node(node_id_t node_id) const noexcept {}
};

} // namespace usearch
} // namespace unum
