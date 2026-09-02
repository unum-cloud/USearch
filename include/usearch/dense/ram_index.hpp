/**
 *  @brief Vector Search index for a single uniform latency memory space.
 *  @author Ash Vardanian
 *  @file include/usearch/dense/ram_index.hpp
 *  @date July 26, 2023
 */
#pragma once

#include <bit>         // `std::bit_width`, `std::bit_floor`
#include <type_traits> // `std::has_unique_object_representations_v` - the geometry word's guard

#include <forkunion/capabilities.hpp> // `fu::preferred_yield_t` - the default waiter
#include <forkunion/types.hpp>
#include <numkong/tensor.hpp>

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/ram_buffers.hpp"
#include "usearch/utilities/ram_tables.hpp"
#include "usearch/dense/ram_metric.hpp"

namespace unum {
namespace usearch {

/** @brief External reader's cursor over one @c dense_ram_index - declared here, defined
 *      after the index it befriends. The allocator defaults to the index's own traversals-side
 *      family, so staging inherits the same NUMA-aware placement as the traversal contexts;
 *      external readers pinned elsewhere override it. */
template <typename index_type_,                                                 //
          synchronization_t synchronization_ = synchronization_t::concurrent_k, //
          typename allocator_type_ = typename index_type_::candidates_allocator_t>
struct dense_ram_cursor;

/**
 *  @brief Vector Search index for a single uniform latency memory space.
 *
 *  @section dense_ram_index_metrics Supported Metrics
 *
 *  This template is designed exclusively for dense representations in the most common metric spaces:
 *  @b Inner-Product, Angular (Cosine), Euclidean (L2), Hamming (Manhattan), and Jaccard (Tanimoto).
 *  All of them can be trivially inferred in a single pass of vector-vector products if L1 and L2
 *  norms of the vectors are known ahead of time and stored locally.
 *
 *  @section dense_ram_index_growth Growth Limits
 *
 *  This index is designed exclusively for @b up-to-4-billion entries, all stored @b in-memory.
 *  Float32 distance representations are used to compare entries which may result in recall regressions for
 *  small K queries. The typical mode of use is to retrieve significantly more entries from the index of
 *  quantized vectors and rerank them with either pre-quantized vectors or original data.
 *
 *  @section dense_ram_index_concurrency Concurrency & Transactions
 *
 *  This class is designed to allow simultaneous construction and traversal in heavily multi-threaded environments.
 *  One design previously considered is to have a small number of locks proportional to the CPU core count and
 *  graph connectivity, and hash the node IDs to pick the target lock.
 *
 *  Instead, we reuse the per-node neighbors lists to also contain per-node locks. This conveniently places them
 *  at the start of independent cache lines. Besides those, we use a minimal number of "atomics" for book-keeping,
 *  avoiding expensive Compare-And-Swap exchange operations.
 *
 *  @section dense_ram_index_persistence Persistence
 *
 *  This index is not expected to be accessed from disk. For that, explore io_uring and SPDK reimplementations.
 *
 *  @section dense_ram_index_malloc Memory Allocation Strategy
 *
 *  Instead of re-allocating one huge matrix and paying the padding memory costs right after the growth phase,
 *  we implement a custom memory allocation strategy. On small-ish indexes we progressively grow through power-of-two
 *  sizes, until we reach a threshold that easily amortizes memory allocations between insertions.
 *  Aiming for @b ~4GB/allocation slabs and compatibility with CXL-backed memory allocators.
 */
template <                                                          //
    nk_dtype_t vector_dtype_ = nk_bf16_k,                           //
    nk_kernel_kind_t kernel_kind_ = nk_kernel_angular_k,            //
    nk_capability_t capability_ = nk_cap_icelake_k,                 //
    typename wait_type_ = fu::preferred_yield_t,                    //
    typename vectors_allocator_type_ = std::allocator<std::byte>,   //
    typename neighbors_allocator_type_ = std::allocator<std::byte>, //
    typename traversals_allocator_type_ = std::allocator<std::byte> //
    >
struct dense_ram_index;

/** @brief One quad's worth of predicate-admitted candidates - the output buffer handed to
 *      batched @c search_if predicates, bounded by the distance kernels' fan-out. */
using dense_admissions_t = fu::limited_array<dense_scored_node_t, 4>;

/**
 *  @brief Runs @p predicate over one scored quad, filling @p admissions with what it admits.
 *
 *  Callables come in two shapes, detected by signature: batched functors receive the whole
 *  @p scored span and push admitted entries themselves - amortizing expensive external
 *  checks across the 4-way gathers the scan already stages, with distances in view - while
 *  per-node @c admission_t(dense_node_id_t) functors are walked lane by lane, the named verdict
 *  leaving no doubt which way the answer cuts. @c dummy_predicate_t never reaches here: its
 *  callers are compiled out whole.
 */
template <typename predicate_type_>
inline void invoke_admission(predicate_type_& predicate, std::span<dense_scored_node_t const> scored,
                             dense_admissions_t& admissions) noexcept {
    if constexpr (requires { predicate(scored, admissions); }) predicate(scored, admissions);
    else
        for (dense_scored_node_t const candidate : scored)
            if (predicate(candidate.node_id) == admission_t::admit_k) {
                // One quad in, four slots of capacity - the buffer cannot refuse.
                [[maybe_unused]] fu::status_t const admitted = admissions.push_back(candidate);
            }
}

/** @brief Per-thread traversal scratch: the candidate queues, the visited set, the link
 *      selection buffers, and the running telemetry counters - one instance per scanning thread. */
template <                                          //
    nk_dtype_t vector_dtype_ = nk_bf16_k,           //
    nk_capability_t capability_ = nk_cap_icelake_k, //
    typename stored_norms_type_ =                   //
    usearch::vector_norms<typename nk::type_for<vector_dtype_>::type::dot_result_t>,
    typename allocator_type_ = std::allocator<std::byte>>
struct dense_traversal_context;

template <nk_dtype_t vector_dtype_, nk_capability_t capability_, typename stored_norms_type_, typename allocator_type_>
struct dense_traversal_context {

    using scalar_t = typename nk::type_for<vector_dtype_>::type;
    using vector_view_t = nk::vector_view<scalar_t>;
    using vector_norms_t = stored_norms_type_;

    using allocator_t = allocator_type_;
    using candidates_allocator_t =
        typename std::allocator_traits<allocator_t>::template rebind_alloc<dense_scored_node_t>;
    /** @brief The hash-set allocates raw bytes and places ids itself, so it demands a byte allocator. */
    using visits_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<std::byte>;
    using ids_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<dense_node_id_t>;
    using views_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<vector_view_t>;
    using norms_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<vector_norms_t>;
    using distances_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<dense_distance_t>;
    using bytes_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<std::byte>;

    using ids_tensor_t = nk::tensor<dense_node_id_t, ids_allocator_t, 1>;
    using views_tensor_t = nk::tensor<vector_view_t, views_allocator_t, 1>;
    using norms_tensor_t = nk::tensor<vector_norms_t, norms_allocator_t, 1>;
    using distances_matrix_t = nk::tensor<dense_distance_t, distances_allocator_t, 2>;
    using bytes_tensor_t = nk::tensor<std::byte, bytes_allocator_t, 1>;
    /** @brief One selection's output: @b [kept] scored nodes - the id and its distance to the
     *      selection's own query. The tensor's live shape is the fill; @c capacity() is the
     *      reservation, so no separate count exists anywhere. */
    using selection_t = nk::tensor<dense_scored_node_t, candidates_allocator_t, 1>;

    using results_array_t = sorted_buffer<dense_scored_node_t, dense_closer_scored_node_t, candidates_allocator_t>;
    /** @brief 8-ary: eight 8-byte scored nodes fill one cache line per level scan, and the
     *      insert-dominated frontier (~2x connectivity pushes per pop) rides the shallower
     *      log_8 @c shift_up depth. */
    using frontier_queue_t = max_heap_dary<8, dense_scored_node_t, dense_further_scored_node_t, candidates_allocator_t>;
    using visited_set_t = growing_hash_set<dense_node_id_t, std::hash<dense_node_id_t>, visits_allocator_t>;

    /** @brief Nodes we've already scored, kept ascending as the best results so far. */
    results_array_t results;
    /** @brief Nodes we've already re-evaluated for post-insertion refinement. */
    results_array_t refining_candidates;
    /** @brief Nodes we consider promising and want to evaluate next. */
    frontier_queue_t frontier;
    /** @brief Nodes we've already checked and should avoid visiting again. */
    visited_set_t visited;
    /** @brief Pseudo-random generator for picking traversal starting points. */
    splitmix64_t random_generator;

    /** @brief Links chosen for the freshly inserted node, with their query distances -
     *      reusable on the reverse side, since every supported metric is symmetric. */
    selection_t selected_links;
    /** @brief Re-selection output for one saturated neighbor's list. */
    selection_t refined_links;

    /** @brief Ids of one neighbor's staged list - the input of a re-selection, never its output. */
    ids_tensor_t staged_ids;
    /** @brief The one staging surface both engines read: contiguous views of the staged rows
     *      during scoring, then the kept prefix of whichever selection is running. The two
     *      lives never overlap - scoring drains the staged rows before any selection writes. */
    views_tensor_t staged_vectors;
    /** @brief Stored norms, parallel to @c staged_vectors. */
    norms_tensor_t staged_norms;
    /** @brief Pair-matrix positions of the kept links, parallel to the running selection's fill. */
    ids_tensor_t kept_positions;

    /** @brief Tile scratch of the staged Gram: the packed B-tiles plus one span of A-tiles. */
    bytes_tensor_t tiles_scratch;
    /** @brief The @b [pool,pool] pairwise distances; its row stride is its own trailing extent,
     *      pre-padded to whole tiles so tail tiles always write in bounds. */
    distances_matrix_t pair_distances;

    /** @brief Total one-to-one distance computations, padding lanes excluded. Like all the
     *      telemetry below, it's a running counter: operations snapshot it on entry and report
     *      the difference, so nested and repeated calls compose without resets. */
    std::size_t computed_distances = 0;
    /** @brief Subset of @c computed_distances spent in insertion-side diversity selection. */
    std::size_t computed_distances_in_refines = 0;
    /** @brief Subset of @c computed_distances spent re-selecting full lists while back-linking. */
    std::size_t computed_distances_in_reverse_refines = 0;
    /** @brief Frontier candidates whose neighbor lists were expanded during traversals. */
    std::size_t visited_members = 0;
    /** @brief Starvation restarts: re-seedings after the frontier drained prematurely -
     *      a graph-connectivity health signal unique to the flat NSW design. */
    std::size_t restarts = 0;

    dense_traversal_context() noexcept = delete;
    dense_traversal_context(dense_traversal_context const&) noexcept = delete;
    dense_traversal_context& operator=(dense_traversal_context const&) noexcept = delete;

    /** @brief Constructs every traversal container with a rebound copy of @p alloc - and keeps
     *      the allocator for the lazily-sized link scratches - so all scratch lands wherever the
     *      caller's allocator places it, like a NUMA domain local to the owning thread. */
    dense_traversal_context(allocator_t alloc) noexcept
        : results(candidates_allocator_t(alloc)), refining_candidates(candidates_allocator_t(alloc)),
          frontier(candidates_allocator_t(alloc)), visited(visits_allocator_t(alloc)), random_generator(),
          allocator_(alloc) {}
    dense_traversal_context(dense_traversal_context&&) = default;
    dense_traversal_context& operator=(dense_traversal_context&&) = default;

    /** @brief Sizes every link scratch for @p capacity entries - the frozen connectivity, the
     *      largest selection any path stages - through the context's own allocator. Geometry
     *      freezes once, so the tensors allocate once. */
    status_t reserve_links(std::size_t capacity) noexcept {
        if (selected_links.capacity() >= capacity) return status_t::success_k;
        selected_links = selection_t::try_zeros({capacity}, candidates_allocator_t(allocator_));
        refined_links = selection_t::try_zeros({capacity}, candidates_allocator_t(allocator_));
        staged_ids = ids_tensor_t::try_zeros({capacity}, ids_allocator_t(allocator_));
        staged_vectors = views_tensor_t::try_zeros({capacity}, views_allocator_t(allocator_));
        staged_norms = norms_tensor_t::try_zeros({capacity}, norms_allocator_t(allocator_));
        kept_positions = ids_tensor_t::try_zeros({capacity}, ids_allocator_t(allocator_));
        bool const sized = !selected_links.empty() && !refined_links.empty() && !staged_ids.empty() &&
                           !staged_vectors.empty() && !staged_norms.empty() && !kept_positions.empty();
        return sized ? status_t::success_k : status_t::bad_alloc_k;
    }

    /** @brief Bytes @c reserve_links reserves for @p capacity links, each tensor on its own arena line. */
    static std::size_t links_memory_usage(std::size_t capacity) noexcept {
        using arena_t = arena_allocator<std::byte>;
        return 2u * arena_t::reserved_bytes(capacity * sizeof(dense_scored_node_t)) +
               2u * arena_t::reserved_bytes(capacity * sizeof(dense_node_id_t)) +
               arena_t::reserved_bytes(capacity * sizeof(vector_view_t)) +
               arena_t::reserved_bytes(capacity * sizeof(vector_norms_t));
    }

    /** @brief Sizes the pair matrix for a @p capacity -row pool - arriving pre-padded to whole
     *      tiles by the caller, which knows its kit's shapes - plus @p scratch_bytes of
     *      kit-specific tile scratch. The matrix's own trailing extent is the row stride, so
     *      tail tiles always write in bounds. The context itself stays agnostic of every
     *      engine's geometry. */
    status_t reserve_tiles(std::size_t capacity, std::size_t scratch_bytes) noexcept {
        if (pair_distances.extent(0) >= capacity) return status_t::success_k;
        tiles_scratch = bytes_tensor_t::try_zeros({scratch_bytes}, bytes_allocator_t(allocator_));
        pair_distances = distances_matrix_t::try_zeros({capacity, capacity}, distances_allocator_t(allocator_));
        return !tiles_scratch.empty() && !pair_distances.empty() ? status_t::success_k : status_t::bad_alloc_k;
    }

    /** @brief Allocator instance behind every link scratch; the queues hold rebound copies. */
    [[no_unique_address]] allocator_t allocator_;
};

template <                                                                                //
    nk_dtype_t vector_dtype_, nk_kernel_kind_t kernel_kind_, nk_capability_t capability_, //
    typename wait_type_,                                                                  //
    typename vectors_allocator_type_, typename neighbors_allocator_type_,
    typename traversals_allocator_type_> //
struct dense_ram_index {

    inline static constexpr nk_dtype_t vector_dtype_k = vector_dtype_;

    using scalar_t = typename nk::type_for<vector_dtype_>::type;
    using vector_product_result_t = typename scalar_t::dot_result_t;
    using distance_t = f32_t;
    using candidate_t = dense_scored_node_t;

    /** @brief Default integer type to use for loops and indexing at sub-index scale. */
    using small_size_t = u32_t;

    /** @brief Node identifier within this sub-index - the class-local spelling of @c dense_node_id_t. */
    using node_id_t = dense_node_id_t;

    using vectors_allocator_t =
        typename std::allocator_traits<vectors_allocator_type_>::template rebind_alloc<scalar_t>;
    using neighbors_allocator_t =
        typename std::allocator_traits<neighbors_allocator_type_>::template rebind_alloc<u32_t>;
    using candidates_allocator_t =
        typename std::allocator_traits<traversals_allocator_type_>::template rebind_alloc<candidate_t>;

    using vector_view_t = nk::vector_view<scalar_t>;
    using vector_span_t = nk::vector_span<scalar_t>;
    /** @brief What the chunks store per row: real moments for norm-consuming finalizes, the
     *      zero-size stand-in for dot - the second random gather per candidate disappears.
     *      The metric layer accepts either record natively, widening internally. */
    using vector_norms_t = stored_norms<kernel_kind_, scalar_t>;

    using vectors_view_t = nk::matrix_view<scalar_t>;
    using node_ids_span_t = nk::vector_span<node_id_t>;
    using node_ids_view_t = nk::vector_view<node_id_t>;

    /** @brief The hardware wait monitor engaged on contended words - the class-local spelling of
     *      @c wait_type_, reused by external readers waiting out transient lock sentinels. */
    using wait_t = wait_type_;

    /** @brief The strongest @c synchronization_t contract this design honors: arbitrary
     *      readers and writers may be live, and even snapshots run optimistically. */
    inline static constexpr synchronization_t synchronization_ceiling_k = synchronization_t::concurrent_k;

    template <typename, synchronization_t, typename>
    friend struct dense_ram_cursor;

    inline static constexpr nk_kernel_kind_t kernel_kind_k = kernel_kind_;

    /** @brief The capability bit of the distance kit the chain actually picked - lower than
     *      @c capability_ when no SIMD kit is adapted for this dtype & metric pair yet. */
    inline static constexpr nk_capability_t picked_quads_capability_k =
        dense_quads_chain<kernel_kind_, scalar_t, capability_>::picked_k;

    /** @brief The tile backend's capability, on its own independent axis - one machine
     *      runs icelake 1-to-1 kernels and sapphireamx tiles simultaneously. */
    inline static constexpr nk_capability_t picked_tiles_capability_k =
        dense_tiles_chain<kernel_kind_, scalar_t, capability_>::picked_k;
    /** @brief The tile kit behind the tiled refinement paths; @c no_tiles_kit_t when none compiled. */
    using tiles_kit_t = typename dense_tiles_chain<kernel_kind_, scalar_t, capability_>::kit_t;
    /** @brief Whether a tile engine is compiled in for this (kernel, dtype) pair. */
    inline static constexpr bool has_tiles_k = dense_tiles_chain<kernel_kind_, scalar_t, capability_>::has_tiles_k;
    static_assert( //
        kernel_kind_k == nk_kernel_dot_k || kernel_kind_k == nk_kernel_angular_k ||
        kernel_kind_k == nk_kernel_euclidean_k || kernel_kind_k == nk_kernel_sqeuclidean_k ||
        kernel_kind_k == nk_kernel_hamming_k || kernel_kind_k == nk_kernel_jaccard_k);

    /**
     *  @brief Each thread aiming to perform scans over the graph needs to have its own object like this.
     */
    using dense_traversal_context_t =
        dense_traversal_context<vector_dtype_, capability_, vector_norms_t, candidates_allocator_t>;
    using results_array_t = typename dense_traversal_context_t::results_array_t;
    using frontier_queue_t = typename dense_traversal_context_t::frontier_queue_t;
    using visited_set_t = typename dense_traversal_context_t::visited_set_t;
    using selection_t = typename dense_traversal_context_t::selection_t;
    using ids_tensor_t = typename dense_traversal_context_t::ids_tensor_t;

    using norms_allocator_t =
        typename std::allocator_traits<vectors_allocator_type_>::template rebind_alloc<vector_norms_t>;

    using vectors_chunk_t = nk::matrix<scalar_t, vectors_allocator_t>;
    using norms_chunk_t = nk::tensor<vector_norms_t, norms_allocator_t, 1>;
    using neighbors_chunk_t = nk::matrix<u32_t, neighbors_allocator_t>;

    /** @brief Rows in the very first chunk, as a power-of-two exponent. @ref dense_ram_index_malloc */
    inline static constexpr small_size_t first_chunk_rows_log2_k = 10; // 1'024 rows
    /** @brief Rows in every mature slab-sized chunk, as a power-of-two exponent. @ref dense_ram_index_malloc */
    inline static constexpr small_size_t slab_rows_log2_k = 20; // 1'048'576 rows
    /** @brief Chunks in the doubling phase of the growth schedule: the first chunk plus one per doubling. */
    inline static constexpr small_size_t doubling_phase_chunks_k = slab_rows_log2_k - first_chunk_rows_log2_k + 1;
    /** @brief Node ids covered by the doubling phase: 2^(slab_rows_log2_k + 1) - 2^first_chunk_rows_log2_k. */
    inline static constexpr node_id_t doubling_phase_capacity_k = power_of_two<node_id_t>(slab_rows_log2_k + 1) -
                                                                  power_of_two<node_id_t>(first_chunk_rows_log2_k);
    /** @brief Upper bound on the chunk count, addressing the entire 2^32 id space. @ref dense_ram_index_growth */
    inline static constexpr small_size_t max_chunks_k =
        doubling_phase_chunks_k +
        static_cast<small_size_t>(((u64_t(1) << 32) - doubling_phase_capacity_k) >> slab_rows_log2_k);

    /** @brief Ceiling on dispensable node ids: the all-ones id is the sentinel and must never be
     *      claimed, so at most 2^32 - 1 rows are ever accounted - the final chunk physically holds
     *      one extra row that stays forever unclaimed. @ref dense_ram_index_growth */
    inline static constexpr std::size_t max_node_count_k = (u64_t(1) << 32) - 1u;

    /**
     *  @brief Once-frozen geometry: vector dimensions in the high half, neighbor-list
     *      connectivity in the low one; zero bits means "not reserved yet". Two fields ride
     *      one word so competing first reservations agree on both through a single cold-path
     *      compare-exchange - the only CAS in the whole index - and every reader unpacks a
     *      local acquire-loaded copy, so a torn or half-frozen geometry is unrepresentable.
     *      Connectivity stays a power of two, tiling whole cache lines, with the leading slot
     *      of every neighbor list doubling as the lock without sacrificing a slot.
     */
    struct geometry_t {
        u64_t packed_bits = 0;

        constexpr geometry_t() noexcept = default;
        /** @brief Packs @p dimensions & @p connectivity - the inverse of the accessors. */
        constexpr geometry_t(std::size_t dimensions, connectivity_t connectivity) noexcept
            : packed_bits((static_cast<u64_t>(dimensions) << 32u) | static_cast<u64_t>(connectivity)) {}

        /** @brief Scalar dimensions per vector. */
        constexpr small_size_t dimensions() const noexcept { return static_cast<small_size_t>(packed_bits >> 32u); }
        /** @brief Neighbor ids per node's list. */
        constexpr small_size_t connectivity() const noexcept { return static_cast<small_size_t>(packed_bits); }
        /** @brief @c true once frozen by the first successful @c reserve(). */
        constexpr explicit operator bool() const noexcept { return packed_bits != 0; }
        constexpr bool operator==(geometry_t const&) const noexcept = default;
    };
    static_assert(std::has_unique_object_representations_v<geometry_t>,
                  "The compare-exchange compares object representations, never operator==: " //
                  "padding would make value-equal geometries spuriously lose the freeze race");
    static_assert(std::atomic<geometry_t>::is_always_lock_free,
                  "Every geometry load must stay one plain instruction, never a library call");

    /** @brief Allocator instance handed to every vectors & norms chunk. */
    [[no_unique_address]] vectors_allocator_t vectors_allocator_;

    /** @brief Allocator instance handed to every neighbors chunk. */
    [[no_unique_address]] neighbors_allocator_t neighbors_allocator_;

#pragma region Placement

    dense_ram_index(dense_ram_index const&) noexcept = delete;
    dense_ram_index& operator=(dense_ram_index const&) noexcept = delete;

    dense_ram_index(dense_ram_index&&) noexcept = delete;
    dense_ram_index& operator=(dense_ram_index&&) noexcept = delete;

    /**
     *  @brief Constructs an empty index, remembering the allocators to hand to every future chunk.
     *
     *  The chunk directories receive a stride padding every chunk header to a multiple of
     *  @c default_alignment_k, so the grower's half-built header never shares a cache line
     *  with the ones readers dereference. Their storage is allocated lazily, ahead of the first chunk.
     */
    dense_ram_index(vectors_allocator_type_ const& vectors_allocator = {},
                    neighbors_allocator_type_ const& neighbors_allocator = {}) noexcept
        : vectors_allocator_(vectors_allocator), neighbors_allocator_(neighbors_allocator),
          vectors_chunks_(vectors_allocator, round_up_to_multiple(sizeof(vectors_chunk_t), default_alignment_k)),
          norms_chunks_(vectors_allocator, round_up_to_multiple(sizeof(norms_chunk_t), default_alignment_k)),
          neighbors_chunks_(neighbors_allocator, round_up_to_multiple(sizeof(neighbors_chunk_t), default_alignment_k)) {
    }

#pragma endregion Placement

#pragma region Measures

    /** @brief Number of published vectors - the visibility watermark; the index itself stores no keys. */
    std::size_t size() const noexcept { return last_visible_id_.load(); }

    /** @brief Neighbor ids per node's list - its full fixed capacity, none lost to bookkeeping. */
    std::size_t connectivity() const noexcept { return geometry_.load(std::memory_order_acquire).connectivity(); }

    /** @brief Scalar dimensions per stored vector; zero until the first reservation freezes it. */
    std::size_t dimensions() const noexcept { return geometry_.load(std::memory_order_acquire).dimensions(); }

    /** @brief Rows hosted without growing again - claimed plus unclaimed, the invariant tying the
     *      two counters together. A transient dip while a grower is mid-flight only makes the
     *      estimate conservative, which is what @c reserve compares against. */
    std::size_t capacity() const noexcept {
        std::size_t const claimed_rows = last_allocated_id_.load(std::memory_order_acquire);
        std::ptrdiff_t const unclaimed_rows = remaining_capacity_.load(std::memory_order_acquire);
        return claimed_rows + static_cast<std::size_t>(unclaimed_rows > 0 ? unclaimed_rows : 0);
    }

    /** @brief Bytes the materialized chunks span - vectors, norms & neighbor lists - derived from
     *      the hosted capacity and the growth schedule rather than read from the chunk
     *      directories, so it stays safe under concurrent growth. A reservation materializes its
     *      chunks up front, so reserved-but-empty rows count from the moment they are granted. */
    std::size_t memory_usage() const noexcept {
        geometry_t const geometry = geometry_.load(std::memory_order_acquire);
        if (!geometry) return sizeof(dense_ram_index);
        std::size_t const hosted_rows = capacity();
        small_size_t chunks = 0;
        while (chunks != max_chunks_k && capacity_of_chunks(chunks) < hosted_rows) ++chunks;
        std::size_t const rows = capacity_of_chunks(chunks);
        std::size_t bytes = sizeof(dense_ram_index);
        bytes += rows * stored_row_values(geometry) * sizeof(scalar_t);
        if constexpr (!std::is_same_v<vector_norms_t, no_norms_t>) bytes += rows * sizeof(vector_norms_t);
        bytes += rows * geometry.connectivity() * sizeof(node_id_t);
        return bytes;
    }

    /** @brief The stored bytes of one node's vector - the id-addressed read a key lookup or an
     *      export resolves through; ids at or above @ref size() name unpublished rows. */
    vector_view_t vector(node_id_t node) const noexcept { return vector_for(chunked_location(node)); }

    /**
     *  @brief Storage values per row under the frozen @p geometry - what a caller's matrix must be
     *      as wide as, since the copy width comes from here rather than from the matrix.
     */
    static std::size_t stored_row_values(geometry_t geometry) noexcept {
        return divide_round_up<std::size_t>(geometry.dimensions(), nk::dimensions_per_value<scalar_t>());
    }

    /** @brief The matrix shapes @c save() fills - the caller sizes its buffers from these.
     *      All extents are logical: the tensors' own strides account for sub-byte packing.
     *      Monotone counters make the two loads safe in this order under any contract. */
    struct saved_shape_t {
        /** @brief Rows of both matrices - the published node count. */
        std::size_t nodes = 0;
        /** @brief Logical dimensions per vector - the vectors matrix's trailing extent. */
        std::size_t dimensions = 0;
        /** @brief Neighbor ids per neighbors-matrix row - the frozen connectivity. */
        std::size_t connectivity = 0;
    };

    /** @brief The shapes a @c save() of the current contents fills. */
    saved_shape_t saved_shape() const noexcept {
        geometry_t const geometry = geometry_.load(std::memory_order_acquire);
        std::size_t const count = last_visible_id_.load(std::memory_order_acquire);
        return {count, geometry.dimensions(), geometry.connectivity()};
    }

#pragma endregion Measures

#pragma region Reservation

    /** @brief Optional tuning knobs for @c insert(); zeros mean "derive from connectivity". */
    struct insert_config_t {
        /** @brief Breadth of the candidate pool during wiring; defaults to twice the connectivity. */
        small_size_t expansion_factor = 0;
    };

    /** @brief Outcome of a @c reserve() call: the capacity & connectivity actually in effect. */
    struct reserve_result_t {
        /** @brief Total rows the index can now host without growing again. */
        std::size_t new_capacity = 0;
        /** @brief The frozen neighbors-list width - echoed back even on a geometry mismatch. */
        connectivity_t connectivity = connectivity_unknown_k;
        /** @brief @c success_k, or why the reservation failed. */
        status_t status = status_t::success_k;
    };

    /** @brief Requested geometry & capacity for a @c reserve() call; frozen by the first success. */
    struct reserve_capacity_t {
        /** @brief Minimal number of entries the index must be able to host. */
        std::size_t capacity = 0;
        /** @brief Neighbors-list width per node; a power of two tiling whole cache lines. */
        connectivity_t connectivity = connectivity_32_k;
        /** @brief Scalar dimensions per vector; required before the first insertion. */
        std::size_t dimensions = 0;
    };

    /**
     *  @brief Ensures the index can host at least @p config.capacity entries, growing when needed.
     *
     *  Safe to call concurrently with insertions, searches, and other reservations: capacity
     *  growth is a thin wrapper over the very same grower-election protocol that powers @c insert,
     *  and the geometry (vector dimensions & neighbor connectivity) freezes exactly once through
     *  a single cold-path compare-exchange. Later calls must pass a matching geometry, or they
     *  receive @c status_t::config_mismatch_k with the frozen connectivity echoed back.
     *
     *  Constructors deliberately allocate nothing, matching the design of ForkUnion's dynamic
     *  structures - every storage failure surfaces right here, in the returned @c reserve_result_t.
     */
    reserve_result_t reserve(reserve_capacity_t config) noexcept {

        // Freeze the geometry exactly once; competing first-reservations are decided by one CAS.
        geometry_t const desired_geometry {config.dimensions, config.connectivity};
        geometry_t expected_geometry {};
        bool const froze_now = geometry_.compare_exchange_strong( //
            expected_geometry, desired_geometry, std::memory_order_acq_rel, std::memory_order_acquire);
        geometry_t const frozen_geometry = froze_now ? desired_geometry : expected_geometry;
        connectivity_t const frozen_connectivity = static_cast<connectivity_t>(frozen_geometry.connectivity());
        if (frozen_geometry != desired_geometry) return {0, frozen_connectivity, status_t::config_mismatch_k};

        while (true) {
            // Physical rows = claimed + unclaimed - the invariant tying the two counters together.
            // Transient dips while a grower is mid-flight only make this estimate conservative.
            std::size_t const claimed_rows = last_allocated_id_.load(std::memory_order_acquire);
            std::ptrdiff_t const unclaimed_rows = remaining_capacity_.load(std::memory_order_acquire);
            std::size_t const total_rows = claimed_rows +
                                           static_cast<std::size_t>(unclaimed_rows > 0 ? unclaimed_rows : 0);
            if (total_rows >= config.capacity) return {total_rows, frozen_connectivity, status_t::success_k};

            // Grow through the same grower-election protocol that powers `insert`: claim the
            // missing rows, then hand the whole claim back. The revert order is load-bearing -
            // deflating `last_allocated_id_` before re-crediting `remaining_capacity_` keeps the
            // physical-rows sum conservative for concurrent observers, so a racing reservation
            // can only over-grow, never report capacity that does not exist.
            std::size_t const deficit_rows = config.capacity - total_rows;
            auto const [first_row, granted] = claim_capacity(deficit_rows);
            if (failed(granted)) return {total_rows, frozen_connectivity, granted};
            release_capacity(deficit_rows);
        }
    }

    /**
     *  @brief Bytes one context reserves for @p config under the frozen geometry - what an arena
     *      backing a traversal must hold, since an arena never grows and a walk never allocates
     *      twice.
     *
     *  Mirrors @ref reserve_traversal reservation for reservation, and sits beside it so the two
     *  drift only under a reviewer's eye. Zero before the geometry freezes.
     */
    std::size_t traversal_memory_usage(insert_config_t config = {}) const noexcept {
        geometry_t const geometry = geometry_.load(std::memory_order_acquire);
        if (!geometry) return 0;
        std::size_t const node_connectivity = geometry.connectivity();
        std::size_t const expansion_factor = config.expansion_factor ? config.expansion_factor //
                                                                     : node_connectivity * 2;
        // One term per reservation, each occupying what the arena will hand out for it.
        using context_t = dense_traversal_context_t;
        using arena_t = arena_allocator<std::byte>;
        std::size_t const links = node_connectivity;
        std::size_t bytes = arena_t::reserved_bytes(context_t::results_array_t::memory_usage(expansion_factor)) +
                            arena_t::reserved_bytes(context_t::results_array_t::memory_usage(links + 1)) +
                            arena_t::reserved_bytes(context_t::frontier_queue_t::memory_usage(expansion_factor * 4u)) +
                            arena_t::reserved_bytes(context_t::visited_set_t::memory_usage(expansion_factor * 4u)) +
                            context_t::links_memory_usage(links);

        if constexpr (has_tiles_k) {
            std::size_t const pool = tiles_pool_(expansion_factor, links);
            bytes += arena_t::reserved_bytes(tiles_kit_t::scratch_bytes(pool, geometry.dimensions())) +
                     arena_t::reserved_bytes(pool * pool * sizeof(dense_distance_t));
        }
        return bytes;
    }

    /**
     *  @brief Sizes @p context 's scratch for the frozen geometry and @p config, so no walk
     *      allocates mid-flight - the other fallible half of an insertion, hoistable ahead of a
     *      fan-out. Idempotent and monotone: a context already wide enough answers in six compares.
     */
    status_t reserve_traversal(dense_traversal_context_t& context, insert_config_t config = {}) noexcept {
        geometry_t const geometry = geometry_.load(std::memory_order_acquire);
        if (!geometry) return status_t::uninitialized_k;
        std::size_t const node_connectivity = geometry.connectivity();
        std::size_t const expansion_factor = config.expansion_factor ? config.expansion_factor //
                                                                     : node_connectivity * 2;

        // `sorted_buffer::insert(.., limit)` never grows on its own, and `growing_hash_set::set`
        // never rehashes, so these reservations are load-bearing rather than an optimization.
        // `refining_candidates` must fit a full neighbor list plus the newcomer during back-linking.
        if (failed(context.results.reserve(expansion_factor)) ||                  //
            failed(context.refining_candidates.reserve(node_connectivity + 1)) || //
            failed(context.frontier.reserve(expansion_factor * 4u)) ||            //
            failed(context.visited.reserve(expansion_factor * 4u)) ||             //
            failed(context.reserve_links(node_connectivity)))
            return status_t::bad_alloc_k;

        // The tile scratch sizes from the same frozen geometry and the kit's own shapes - padded to
        // whole tiles, so the padded pool doubles as the pair-matrix stride.
        if constexpr (has_tiles_k) {
            std::size_t const pool = tiles_pool_(expansion_factor, node_connectivity);
            if (failed(context.reserve_tiles(pool, tiles_kit_t::scratch_bytes(pool, geometry.dimensions()))))
                return status_t::bad_alloc_k;
        }
        return status_t::success_k;
    }

#pragma endregion Reservation

#pragma region Insertion

    /** @brief Outcome & telemetry of one @c insert() batch. */
    struct insert_result_t {
        /** @brief @c success_k, or why the whole batch was rejected. A batch is atomic: every
         *      fallible step runs before the first id is claimed, so a failure publishes no row
         *      and there is no partial count to report. The residual hole is a thread dying
         *      between a claim and its publication, which no return value can carry. */
        status_t status = status_t::success_k;
        /** @brief Index size visible after this batch fully published - the watermark. A
         *      rejected batch publishes nothing, so this reports the size it left untouched. */
        std::size_t new_size = 0;
        /** @brief Frontier candidates expanded across all per-node walks of the batch. */
        std::size_t visited_members = 0;
        /** @brief Total distance computations, with padding lanes of 4-way kernels excluded. */
        std::size_t computed_distances = 0;
        /** @brief Subset of @c computed_distances spent in insertion-side diversity selection. */
        std::size_t computed_distances_in_refines = 0;
        /** @brief Subset of @c computed_distances spent re-selecting full lists while back-linking. */
        std::size_t computed_distances_in_reverse_refines = 0;
        /** @brief Starvation restarts across the batch - a graph-connectivity health signal. */
        std::size_t restarts = 0;

        explicit operator bool() const noexcept { return succeeded(status); }
    };

    /**
     *  @brief Inserts a batch of @p new_vectors, reporting each one's assigned node id.
     *
     *  Per claimed node the flow is: land the payload, walk the graph for candidates, wire & publish.
     *  All fallible scratch allocations happen before the id claim - once ids are claimed, this
     *  thread must roll forward to publication. @ref dense_ram_index_concurrency
     *
     *  @param[in] new_vectors The batch to insert, one row per vector, in the frozen geometry.
     *  @param[out] assigned_ids Caller-owned output: either empty, or exactly one slot per incoming
     *      vector, receiving that vector's assigned node id. The index stores no keys - ids are
     *      dense, so callers park keys in a flat id-indexed array of the outer layer. One batch's
     *      ids are generally @b not contiguous: identity is assigned per node, interleaved
     *      across concurrently inserting threads.
     *  @param[inout] context The calling thread's scratch & telemetry, reused across batches.
     *  @param[in] config Optional tuning knobs; zeros mean "derive from connectivity".
     */
    insert_result_t insert(vectors_view_t new_vectors, node_ids_span_t assigned_ids, dense_traversal_context_t& context,
                           insert_config_t config = {}) noexcept {

        std::size_t const incoming_vectors_count = new_vectors.extent(0); // rows, not scalars
        if (!incoming_vectors_count) return {status_t::success_k, size()};
        bool const report_assigned_ids = assigned_ids.size() != 0;
        if (report_assigned_ids && assigned_ids.size() != incoming_vectors_count)
            return {status_t::config_mismatch_k, size()};

        // Snapshot the context's running telemetry, to report this batch's share on return.
        std::size_t const visited_members_before = context.visited_members;
        std::size_t const computed_distances_before = context.computed_distances;
        std::size_t const refines_before = context.computed_distances_in_refines;
        std::size_t const reverse_refines_before = context.computed_distances_in_reverse_refines;
        std::size_t const restarts_before = context.restarts;

        // Geometry is frozen by the first successful `reserve()` - a hard prerequisite here:
        // without it, neither the copy width nor the neighbor-list width is defined.
        geometry_t const geometry = geometry_.load(std::memory_order_acquire);
        if (!geometry) return {status_t::uninitialized_k, size()};

        // Rows narrower than the frozen geometry would be over-read by the copy below, which takes
        // its width from the geometry rather than from the caller's matrix; a packed matrix
        // reports its logical width, like the geometry does.
        if (new_vectors.extent(1) != geometry.dimensions()) return {status_t::config_mismatch_k, size()};

        if (status_t const sized = reserve_traversal(context, config); failed(sized)) return {sized, size()};

        // Claim storage for the whole batch: from here on enough sentinel-filled rows exist for
        // all our nodes, and this thread must roll forward to publishing every id it claims.
        auto [batch_claim, granted] = claim(incoming_vectors_count);
        if (failed(granted)) return {granted, size()};
        insert_reserved(new_vectors, assigned_ids, context, std::move(batch_claim), config);

        insert_result_t result;
        result.new_size = size();
        result.visited_members = context.visited_members - visited_members_before;
        result.computed_distances = context.computed_distances - computed_distances_before;
        result.computed_distances_in_refines = context.computed_distances_in_refines - refines_before;
        result.computed_distances_in_reverse_refines = context.computed_distances_in_reverse_refines -
                                                       reverse_refines_before;
        result.restarts = context.restarts - restarts_before;
        return result;
    }

    /**
     *  @brief Entry removals are explicitly banned to simplify concurrency management.
     */
    status_t remove(node_ids_view_t, dense_traversal_context_t&) noexcept = delete;

    /**
     *  @brief Storage claimed in this index and not yet inserted against - the fallible half of an
     *      insertion, hoisted so a caller spanning several indexes commits or refuses as a whole.
     *
     *  Releasing is automatic and two-sided, so an abandoned claim leaves the physical-rows sum as
     *  it found it. Rows an id was dispensed against never come back, which is why consuming a
     *  claim and dispensing against it are the same step. @ref dense_ram_index_concurrency
     */
    struct capacity_claim_t {
        capacity_claim_t() noexcept = default;
        /** @brief The granting index and the rows it granted; only @ref claim ever calls it. */
        capacity_claim_t(dense_ram_index& index, std::size_t rows) noexcept : index_(&index), rows_(rows) {}
        capacity_claim_t(capacity_claim_t&& other) noexcept { swap(other); }
        capacity_claim_t& operator=(capacity_claim_t&& other) noexcept { return swap(other), *this; }
        capacity_claim_t(capacity_claim_t const&) = delete;
        capacity_claim_t& operator=(capacity_claim_t const&) = delete;
        /** @brief Credits back whatever was never consumed; a consumed claim releases nothing. */
        ~capacity_claim_t() noexcept {
            if (index_ && rows_) index_->release_capacity(rows_);
        }

        void swap(capacity_claim_t& other) noexcept {
            std::swap(index_, other.index_);
            std::swap(rows_, other.rows_);
        }

        /** @brief Rows this claim still holds; zero once an insertion has taken it. */
        std::size_t rows() const noexcept { return rows_; }
        explicit operator bool() const noexcept { return rows_ != 0; }

        /** @brief Marks the rows as taken, so destruction credits nothing back - what an insertion
         *      spread over several threads calls once every one of its rows has claimed an id. */
        void consume() noexcept { rows_ = 0; }

      private:
        /** @brief The granting index, reached only to credit back; null marks the empty state a
         *      default construction and a move-out both leave behind. */
        dense_ram_index* index_ = nullptr;
        /** @brief Rows still held; zero once consumed, so destruction credits nothing. */
        std::size_t rows_ = 0;
    };

    /** @brief Claims storage for @p count rows without assigning any identity - the step a batch
     *      spanning several indexes runs on every one of them before touching any. */
    expected<capacity_claim_t> claim(std::size_t count) noexcept {
        auto const [first_row, granted] = claim_capacity(count);
        if (failed(granted)) return {capacity_claim_t {}, granted};
        return {capacity_claim_t {*this, count}, status_t::success_k};
    }

    /** @brief Returns @p count rows no id was dispensed against. Deflating the claim watermark
     *      before crediting the capacity is load-bearing: it keeps the physical-rows sum
     *      conservative, so a concurrent reservation can only over-grow. */
    void release_capacity(std::size_t count) noexcept {
        last_allocated_id_.fetch_sub(static_cast<node_id_t>(count), std::memory_order_relaxed);
        remaining_capacity_.fetch_add(static_cast<std::ptrdiff_t>(count), std::memory_order_release);
    }

    /**
     *  @brief Inserts against capacity @p claim already obtained and scratch already reserved - the
     *      half of an insertion that cannot fail, so a batch spanning several indexes commits or is
     *      refused as a whole. Consumes @p claim; telemetry accrues in @p context.
     *
     *  @p on_assigned fires per row with the assigned node id, before the walk and before
     *  publication - so an outer layer's per-id state, like a key plane, is written ahead of the
     *  watermark turn and no reachable node can be observed without it.
     */
    template <typename callback_type_ = dummy_callback_t>
    void insert_reserved(vectors_view_t new_vectors, node_ids_span_t assigned_ids, dense_traversal_context_t& context,
                         capacity_claim_t&& claim, insert_config_t config = {},
                         callback_type_&& on_assigned = {}) noexcept {
        // Consumed: from here every row dispenses an id, and a dispensed id is never returnable.
        claim.consume();
        std::size_t const incoming_vectors_count = new_vectors.extent(0);
        bool const report_assigned_ids = assigned_ids.size() != 0;
        geometry_t const geometry = geometry_.load(std::memory_order_acquire);
        std::size_t const node_connectivity = geometry.connectivity();
        std::size_t const expansion_factor = config.expansion_factor ? config.expansion_factor //
                                                                     : node_connectivity * 2;

        // All slots share one contiguous packed row layout, so the copy width is loop-invariant.
        // Sub-byte types pack several dimensions per storage value - count storage bytes, not dimensions.
        std::size_t const vector_bytes = stored_row_values(geometry) * sizeof(scalar_t);

        // Go through the vectors and inject them one-by-one.
        for (small_size_t incoming_index = 0; incoming_index != incoming_vectors_count; ++incoming_index) {
            // Identity is dispensed per node, never derived from the batched capacity claim: the
            // claim only proves rows exist. One-by-one dispensing assigns ids in global
            // wiring-start order, so each thread holds at most one unpublished id and the
            // publication gap stays bounded by the thread count rather than the batch size.
            // The `acq_rel` order chains a happens-before edge from the growth of the hosting
            // chunk through the all-RMW dispenser to this thread.
            node_id_t const new_node_id = next_visible_id_.fetch_add(1, std::memory_order_acq_rel);
            chunked_location_t const new_location = chunked_location(new_node_id);
            if (report_assigned_ids) assigned_ids[incoming_index] = new_node_id;
            on_assigned(new_node_id);

            // The scratch structures persist across iterations and calls - reset them for this node.
            context.results.clear();
            context.refining_candidates.clear();
            context.frontier.clear();
            context.visited.clear();

            // Copy the incoming vector into its owned slot first, and derive the norms from the owned
            // representation - the exact bytes that all future distance computations will stream.
            // Both sides share `scalar_t`, so this is a raw copy of packed storage bytes,
            // not a numeric conversion - `nk::cast` is reserved for genuine dtype changes.
            vector_view_t const incoming_vector = new_vectors.row(incoming_index).as_vector();
            vector_span_t const owned_vector = vector_span_for(new_location);
            std::memcpy(owned_vector.byte_data(), incoming_vector.byte_data(), vector_bytes);
            vector_norms_t const new_norms = dense_norms(owned_vector);
            if constexpr (!std::is_same_v<vector_norms_t, no_norms_t>) norm_ref_for(new_location) = new_norms;

            // Snapshot the contiguous visibility watermark: every id strictly below it is published,
            // making it the safe bound for random landmark sampling. On a virgin index there is
            // nothing to traverse - the node publishes with an empty neighbors list, and the
            // following entries wire back to it as soon as the watermark advances.
            node_id_t const first_invisible_id = last_visible_id_.load(std::memory_order_acquire);
            // The row is already claimed, so a refused walk wires a thinner node rather than
            // abandoning an id nothing will ever publish.
            if (first_invisible_id != 0) [[maybe_unused]]
                status_t const walked = search_around(owned_vector, new_norms, first_invisible_id, expansion_factor,
                                                      context);

            // Select up to `connectivity()` diverse links from the search results and write them
            // into our own still-unpublished list. No locks are needed: no reader can arrive
            // before slot 0 turns non-sentinel, so relaxed stores into slots 1.. suffice, and the
            // untouched tail keeps its sentinel termination.
            selection_t& selected = context.selected_links;
            std::size_t const refine_distances = select_diverse(context.results, node_connectivity, selected, context);
            context.computed_distances += refine_distances;
            context.computed_distances_in_refines += refine_distances;
            neighbors_list_ref_t const new_neighbors = neighbors_for(new_location);
            small_size_t const selected_count = static_cast<small_size_t>(selected.size());
            for (small_size_t link = 1; link < selected_count; ++link)
                new_neighbors.store_slot(link, selected[link].node_id);

            // Publish the node: slot 0 receives its nearest diverse neighbor, or the node's own id
            // when the graph offered no candidates - the "published, zero neighbors" self-loop
            // encoding, distinguishable from the all-ones "unpublished or locked" sentinel.
            // The release store is what makes the payload visible to any thread that later
            // discovers this node.
            node_id_t const first_link_id = selected_count ? selected[0].node_id : new_node_id;
            new_neighbors.unlock(first_link_id);

            // Wire the reverse edges before taking the watermark turn: once the watermark
            // passes this id, no thread edits any published row on its behalf - the
            // settlement probe in the snapshot rests on exactly that - and the own row
            // stays unreachable below the watermark, so unlocking it above is harmless.
            // Every neighbor lock is released inside, so the spin below holds none.
            back_link_neighbors(new_node_id, owned_vector, new_norms, selected, context);

            // Take our watermark turn, strictly in dispensed order, keeping the visible prefix
            // hole-free. Each thread holds at most one unpublished id, so this wait is bounded by
            // one wiring per other in-flight thread - the "publication convoy" is the accepted
            // cost of the CAS-free design; a helping-CAS advancement is the known alternative.
            for (node_id_t watermark = last_visible_id_.load(std::memory_order_acquire); watermark != new_node_id;
                 watermark = last_visible_id_.load(std::memory_order_acquire))
                // Uncapped monitored wait: the watermark is the only word we wait on, and every
                // predecessor's publication store wakes us to re-check.
                wait_type_ {}(last_visible_id_, watermark, 0u, fu::wait_uncapped_t {});
            last_visible_id_.store(new_node_id + 1, std::memory_order_release);
        }
    }

#pragma endregion Insertion

#pragma region Search

    /** @brief Optional tuning knobs for @c search(); zeros mean "derive from connectivity". */
    struct search_config_t {
        /** @brief Breadth of the candidate pool; defaults to twice the connectivity.
         *      Raise it to trade latency for recall - the caller reranks anyway. */
        small_size_t expansion_factor = 0;
        /** @brief Reader contract; @c immutable_k - or stronger - drops every atomic in the
         *      traversal when the caller certifies no concurrent mutators. */
        synchronization_t synchronization = synchronization_t::concurrent_k;
    };

    /** @brief Outcome & telemetry of one @c search() call; matches stay in the caller's context. */
    struct search_result_t {
        /** @brief @c success_k, or why the search could not run at all. */
        status_t status = status_t::success_k;
        /** @brief Matches left in `context.results`, ascending by distance to the query. */
        std::size_t count = 0;
        /** @brief Frontier candidates expanded during the walk. */
        std::size_t visited_members = 0;
        /** @brief Total distance computations, with padding lanes of 4-way kernels excluded. */
        std::size_t computed_distances = 0;
        /** @brief Starvation restarts - a graph-connectivity health signal. */
        std::size_t restarts = 0;

        explicit operator bool() const noexcept { return succeeded(status); }
    };

    /**
     *  @brief Finds the nearest published neighbors of @p query_vector - the caller reranks or
     *      trims the prefix it needs.
     *
     *  Under the default @c concurrent_k contract it is safe to call concurrently with
     *  insertions and other searches: the traversal reads only nodes below the visibility
     *  watermark, acquire-loads every slot-0 lock before scanning a list, retries transiently
     *  locked nodes, and never writes to the graph. Freshly inserted nodes become reachable the
     *  moment their publisher advances the watermark; reverse-edge ids above this query's
     *  watermark snapshot are skipped by the translation-frontier invariant, and picked up by
     *  any later query with a fresher snapshot. Under @c immutable_k - or stronger - the caller
     *  certifies no mutators and the traversal runs atomics-free. @ref dense_ram_index_concurrency
     *
     *  @param[in] query_vector The query, in the index's own dtype & frozen dimensionality.
     *  @param[inout] context The calling thread's scratch & telemetry: the matches end up in
     *      @c context.results, ascending by distance - the nearest first.
     *  @param[in] config Optional tuning knobs; zeros mean "derive from connectivity".
     */
    search_result_t search(vector_view_t query_vector, dense_traversal_context_t& context,
                           search_config_t config = {}) noexcept {
        if (config.synchronization != synchronization_t::concurrent_k)
            return search_typed<synchronization_t::immutable_k>(query_vector, context, config);
        return search_typed<synchronization_t::concurrent_k>(query_vector, context, config);
    }

    /**
     *  @brief Finds the nearest published neighbors of @p query_vector that satisfy
     *      @p predicate - the filtered sibling of @c search, sharing its whole contract.
     *
     *  Non-matching nodes still score and route the walk - filtering navigation itself would
     *  strand it in disconnected islands - but only matching nodes enter the reported pool,
     *  whose radius governs termination once it holds @c expansion_factor entries; until then
     *  the ordinary navigation radius bounds the frontier. Predicates come in two shapes,
     *  detected by signature: per-node @c admission_t(dense_node_id_t), or batched - receiving
     *  the scored quad and an admissions buffer, pushing what it admits:
     *
     *  @code{.cpp}
     *  index.search_if(query, [&](std::span<dense_scored_node_t const> scored, auto& admitted) {
     *      for (dense_scored_node_t candidate : scored)
     *          if (labels[candidate.node_id] == wanted) [[maybe_unused]] auto kept = admitted.push_back(candidate);
     *  }, context);
     *  @endcode
     *
     *  The batch amortizes expensive external checks across the 4-way gathers the scan
     *  already stages, with distances in view - so radius cut-offs are just predicates.
     *  Each node is tested at most once per query: the exact visited set is the predicate's
     *  memoization. With a trivially-true predicate the results match @c search exactly.
     *
     *  @param[in] query_vector The query, in the index's own dtype & frozen dimensionality.
     *  @param[in] predicate The admission filter, called with published node ids only.
     *  @param[inout] context The calling thread's scratch & telemetry: the matches end up in
     *      @c context.results, ascending by distance - the nearest first.
     *  @param[in] config Optional tuning knobs; zeros mean "derive from connectivity".
     */
    template <typename predicate_type_>
    search_result_t search_if(vector_view_t query_vector, predicate_type_&& predicate,
                              dense_traversal_context_t& context, search_config_t config = {}) noexcept {
        if (config.synchronization != synchronization_t::concurrent_k)
            return search_typed<synchronization_t::immutable_k>(query_vector, context, config,
                                                                std::forward<predicate_type_>(predicate));
        return search_typed<synchronization_t::concurrent_k>(query_vector, context, config,
                                                             std::forward<predicate_type_>(predicate));
    }

    /** @brief The compile-time form of @c search() & @c search_if(): one whole-traversal
     *      instantiation per @p synchronization_ contract, so the exclusive form carries no
     *      atomics at all, and per predicate type, so @c dummy_predicate_t compiles every
     *      admission check out of the unfiltered paths. */
    template <synchronization_t synchronization_, typename predicate_type_ = dummy_predicate_t>
    search_result_t search_typed(vector_view_t query_vector, dense_traversal_context_t& context,
                                 search_config_t config = {},
                                 predicate_type_&& predicate = predicate_type_ {}) noexcept {
        constexpr bool predicated_k = !std::is_same_v<std::remove_cvref_t<predicate_type_>, dummy_predicate_t>;

        std::size_t const expansion_factor = config.expansion_factor ? config.expansion_factor //
                                                                     : connectivity() * 2;

        // Same load-bearing reservations as in `insert`: the scratch structures never grow on
        // their own mid-traversal. A predicated walk also owns the admitted pool - the
        // re-selection buffer, idle during any search - sized to the same expansion.
        if (failed(context.results.reserve(expansion_factor)) ||       //
            failed(context.frontier.reserve(expansion_factor * 4u)) || //
            failed(context.visited.reserve(expansion_factor * 4u)))
            return {status_t::bad_alloc_k};
        if constexpr (predicated_k) {
            if (failed(context.refining_candidates.reserve(expansion_factor))) return {status_t::bad_alloc_k};
            context.refining_candidates.clear();
        }
        context.results.clear();
        context.frontier.clear();
        context.visited.clear();

        // Snapshot the watermark once: everything strictly below it is published and readable.
        // An empty index legitimately yields an empty result set. A mutation-free contract needs
        // no acquire - the caller's own synchronization already ordered every publication first.
        constexpr std::memory_order watermark_order_k = synchronization_ == synchronization_t::concurrent_k
                                                            ? std::memory_order_acquire
                                                            : std::memory_order_relaxed;
        node_id_t const first_invisible_id = last_visible_id_.load(watermark_order_k);
        if (first_invisible_id == 0) return {status_t::success_k};

        // Snapshot the context's running telemetry, to report this query's share on return.
        std::size_t const visited_members_before = context.visited_members;
        std::size_t const computed_distances_before = context.computed_distances;
        std::size_t const restarts_before = context.restarts;

        vector_norms_t const query_norms = dense_norms(query_vector);
        status_t const walked = search_around<synchronization_>(query_vector, query_norms, first_invisible_id,
                                                                expansion_factor, context, //
                                                                std::forward<predicate_type_>(predicate));

        // The result contract is one pool: matches always land in `context.results`, so a
        // predicated walk swaps its admitted pool into place - both are the same buffer type,
        // and the navigation leftovers get cleared by whichever call reuses that scratch next.
        if constexpr (predicated_k) std::swap(context.results, context.refining_candidates);

        // A refused walk still ranked whatever it reached; the status says the ranking is short
        // because scratch ran out, not because the graph had nothing better.
        search_result_t result;
        result.status = walked;
        result.count = context.results.size();
        result.visited_members = context.visited_members - visited_members_before;
        result.computed_distances = context.computed_distances - computed_distances_before;
        result.restarts = context.restarts - restarts_before;
        return result;
    }

#pragma endregion Search

#pragma region Checkpoints

    /** @brief The class-local spelling of the shared, template-independent
     *      @c dense_index_header_t. The index itself never writes it - checkpoint
     *      writers prefix their own containers with it. */
    using header_t = dense_index_header_t;

    /**
     *  @brief Copies the published prefix into two caller-owned matrices - vectors and
     *      sentinel-padded neighbor lists - one copy-loop instantiation per contract.
     *      Norms are never serialized: they re-derive from the vectors on @c load(). The
     *      neighbor rows keep the positional sentinel convention - any id at or past the
     *      node count names no node - so the matrix is interchange-ready as a fixed-degree
     *      column without rewriting.
     *
     *  A checkpointing reader, not a mutator: concurrent searches keep running on every arm.
     *  The concurrent arm is a speculative copy staging neighbor rows through relaxed
     *  atomics - formally race-free against live back-linkers - that answers @c contended_k
     *  whenever any insert overlapped the bracket and @c success_k only over a proven-
     *  quiescent interval. The certified arms keep plain vectorizable copies, and a violated
     *  certification still surfaces as @c contended_k instead of a silently torn snapshot.
     *
     *  @param[out] vectors Destination @b [nodes,dimensions] matrix, per @c saved_shape().
     *  @param[out] neighbors Destination @b [nodes,connectivity] matrix, per @c saved_shape().
     *  @return @c success_k, or why the snapshot was not produced.
     */
    template <synchronization_t synchronization_>
    status_t save_typed(nk::matrix_span<scalar_t> vectors, nk::matrix_span<node_id_t> neighbors) const noexcept {
        // The settlement probe, trailing counter first: the pair is monotone with
        // visible ≤ dispensed at every instant, so equal reads in this order prove the
        // pair rested at that value across the whole probe, not one instant. A thread
        // edits published rows only between its dispense & its watermark turn - reverse
        // edges land before the turn - so equality means no such thread exists.
        node_id_t const visible_before = last_visible_id_.load(std::memory_order_acquire);
        node_id_t const dispensed_before = next_visible_id_.load(std::memory_order_acquire);
        node_id_t const allocated_before = last_allocated_id_.load(std::memory_order_acquire);
        if (visible_before != dispensed_before) return status_t::contended_k;

        geometry_t const geometry = geometry_.load(std::memory_order_acquire);
        std::size_t const nodes = static_cast<std::size_t>(visible_before);
        if (vectors.extent(0) != nodes || vectors.extent(1) != geometry.dimensions() || neighbors.extent(0) != nodes ||
            neighbors.extent(1) != geometry.connectivity())
            return status_t::config_mismatch_k;

        // Vector rows are immutable once published and copy plain on every arm; only the
        // neighbor rows follow the arm's staging discipline.
        save_rows_<synchronization_t::immutable_k>(vectors, vectors_chunks_, nodes);
        save_rows_<synchronization_>(neighbors, neighbors_chunks_, nodes);

        // Orders the staged reads before the re-reads below; pairs with the release fence
        // heading @c back_link_neighbors, so any row edit the copy overlapped surfaces here.
        // The non-monotone allocation counter rides along as a bonus growth sensor.
        std::atomic_thread_fence(std::memory_order_acquire);
        bool const settled = last_visible_id_.load(std::memory_order_relaxed) == visible_before &&
                             next_visible_id_.load(std::memory_order_relaxed) == dispensed_before &&
                             last_allocated_id_.load(std::memory_order_relaxed) == allocated_before;
        return settled ? status_t::success_k : status_t::contended_k;
    }

    /** @brief The runtime face of @ref save_typed - @c exclusive_k shares the immutable
     *      instantiation, exactly as @c search collapses the certified arms. */
    status_t save(nk::matrix_span<scalar_t> vectors, nk::matrix_span<node_id_t> neighbors,
                  synchronization_t synchronization) const noexcept {
        if (synchronization == synchronization_t::concurrent_k)
            return save_typed<synchronization_t::concurrent_k>(vectors, neighbors);
        return save_typed<synchronization_t::immutable_k>(vectors, neighbors);
    }

    /**
     *  @brief Restores contents previously written by @c save() from two typed matrices,
     *      re-deriving every vector's norms - they are never serialized.
     *
     *  Demands sole ownership of an empty index whose geometry is already frozen: the caller
     *  reads its own container's header, calls @c reserve() with the recorded geometry, and
     *  then loads - so shapes validate against the frozen geometry instead of trusting bytes.
     *  A failed load leaves the structure unusable - destroy it.
     *
     *  @param[in] vectors Source @b [nodes,dimensions] matrix of stored scalars.
     *  @param[in] neighbors Source @b [nodes,connectivity] matrix of sentinel-padded ids.
     *  @param[in] synchronization The caller's certified environment; only @c exclusive_k loads.
     *  @return @c success_k, or why nothing was restored.
     */
    status_t load(nk::matrix_view<scalar_t> vectors, nk::matrix_view<node_id_t> neighbors,
                  synchronization_t synchronization) noexcept {
        if (synchronization != synchronization_t::exclusive_k) return status_t::config_mismatch_k;
        geometry_t const geometry = geometry_.load(std::memory_order_acquire);
        if (!geometry || last_visible_id_.load(std::memory_order_acquire) != 0) return status_t::config_mismatch_k;

        std::size_t const count = vectors.extent(0);
        if (vectors.extent(1) != geometry.dimensions() || neighbors.extent(0) != count ||
            neighbors.extent(1) != geometry.connectivity())
            return status_t::config_mismatch_k;
        if (!count) return status_t::success_k;

        // Claim the rows through the ordinary capacity protocol: a preceding `reserve` is
        // consumed instead of duplicated, and chunks grow only on a genuine deficit.
        auto const [first_row, granted] = claim_capacity(count);
        if (failed(granted)) return granted;

        load_rows_(vectors, vectors_chunks_, count);
        load_rows_(neighbors, neighbors_chunks_, count);

        // Re-derive the norms in node id order - one reduction pass over the freshly copied
        // vectors, through the same kernels the insertion path uses.
        if constexpr (!std::is_same_v<vector_norms_t, no_norms_t>)
            for (std::size_t node = 0; node != count; ++node) {
                chunked_location_t const location = chunked_location(static_cast<node_id_t>(node));
                norms_chunks_[location.chunk_index][location.index_in_chunk] = dense_norms(vector_for(location));
            }

        // Publish: identity first, and the watermark last with a release - the same order
        // every publication in this design follows.
        next_visible_id_.fetch_add(static_cast<node_id_t>(count), std::memory_order_relaxed);
        last_visible_id_.store(static_cast<node_id_t>(count), std::memory_order_release);
        return status_t::success_k;
    }

#pragma endregion Checkpoints

  private:
    /** @brief The tile pool one refinement scores: the wider candidate set, padded to whole tiles. */
    static std::size_t tiles_pool_(std::size_t expansion_factor, std::size_t node_connectivity) noexcept {
        return round_up_to_multiple(max_of<std::size_t>(expansion_factor, node_connectivity + 1), tile_span_k);
    }

#pragma region Storage

    /** @brief Trampoline for the kernel-aware @c usearch::dense_norms with hardened types -
     *      computed moments for norm-consuming kernels, the empty stand-in for dot. */
    inline static vector_norms_t dense_norms(vector_view_t vec) noexcept {
        return usearch::dense_norms<kernel_kind_, capability_, scalar_t>(vec);
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

    /** @brief Scores @p query against four consecutive entries of the cached @p vectors &
     *      @p norms arrays starting at @p start; the gather cursor sticks at @p count - 1, so
     *      short tails repeat the last entry and the kernel always streams four valid lanes. */
    inline static std::array<distance_t, 4> distances_quad_of(     //
        vector_view_t query, vector_norms_t query_norms,           //
        vector_view_t const* vectors, vector_norms_t const* norms, //
        small_size_t start, small_size_t count) noexcept {
        std::array<vector_view_t, 4> lane_vectors;
        std::array<vector_norms_t, 4> lane_norms;
        for (small_size_t lane = 0, source = start; lane != lane_vectors.size(); ++lane) {
            lane_vectors[lane] = vectors[source];
            lane_norms[lane] = norms[source];
            source += source + 1 != count;
        }
        return distances(query, query_norms, lane_vectors, lane_norms);
    }

    /** @brief Describes in which chunk and at which row inside the chunk a certain entry is located. */
    struct chunked_location_t {
        small_size_t chunk_index = 0;
        small_size_t index_in_chunk = 0;
    };

    /** @brief Maps the global @p node_id onto a chunk and a row within it via a @b closed-form
     *      analytical solution - O(1) time and memory, no per-chunk metadata to walk or binary-search.
     *
     *  Chunk sizes follow the growth schedule from @ref dense_ram_index_malloc: the first chunk
     *  holds 2^ @c first_chunk_rows_log2_k rows, every next chunk doubles, until reaching the slab
     *  size of 2^ @c slab_rows_log2_k rows - all chunks after that keep that fixed slab size.
     *  Doubling gives closed-form prefix sums, so the lookup is one @c lzcnt plus a few adds -
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

    /**
     *  @brief Typed view over one node's neighbor row - the only legal way to touch a list that
     *      concurrent scanners may be reading. Copyable value type with reference semantics.
     *
     *  Slot 0 is the combined lock & first-neighbor word: the all-ones @c dense_node_id_unknown_k
     *  sentinel while unpublished, empty, or locked for editing - otherwise the plain id of the
     *  first neighbor. Ids are stored raw and no row bytes are spent on bookkeeping, so the
     *  neighbor capacity stays a power of two, tiling whole cache lines. Slots 1.. are accessed
     *  as relaxed atomics on both sides, making the "stale-but-valid mix" a scanner can observe
     *  during a concurrent re-link defined behavior instead of a data race.
     *  @ref dense_ram_index_concurrency
     *
     *  Under a mutation-free contract - @c immutable_k or stronger - every atomic degrades to a
     *  plain load: with no writers alive there is nothing to order, and plain reads free the
     *  compiler to unroll & prefetch the row scan. Mutating members belong to the fully
     *  concurrent form alone.
     */
    template <synchronization_t synchronization_ = synchronization_t::concurrent_k>
    struct neighbors_list_ref {
        node_ids_span_t row_;

        /** @brief Total slots in the row, every one usable for a neighbor id. */
        inline small_size_t capacity() const noexcept { return static_cast<small_size_t>(row_.size()); }

        /** @brief Relaxed atomic read of slot @p index; slot 0 belongs to @c peek_first / @c lock. */
        inline node_id_t load_slot(small_size_t index) const noexcept {
            if constexpr (synchronization_ == synchronization_t::concurrent_k)
                return std::atomic_ref<node_id_t>(row_[index]).load(std::memory_order_relaxed);
            else return row_[index];
        }
        /** @brief Relaxed atomic write of slot @p index - only for the unpublished owner or the
         *      current lock holder. */
        inline void store_slot(small_size_t index, node_id_t value) const noexcept {
            std::atomic_ref<node_id_t>(row_[index]).store(value, std::memory_order_relaxed);
        }
        /** @brief Acquire-load of the lock word: the sentinel means unpublished, empty, or locked;
         *      any other value is the first neighbor id, with everything its last (un)locker wrote
         *      now visible to us. */
        inline node_id_t peek_first() const noexcept {
            if constexpr (synchronization_ == synchronization_t::concurrent_k)
                return std::atomic_ref<node_id_t>(row_[0]).load(std::memory_order_acquire);
            else return row_[0];
        }
        /** @brief Spins until the lock is ours; returns the displaced first-neighbor value.
         *      An exchange that receives the sentinel back did @b not acquire the lock - that rule
         *      lives here, in one place, instead of at every call site. Only meaningful on
         *      published nodes: locking an unpublished row would spin until its owner publishes. */
        inline node_id_t lock() const noexcept {
            node_id_t first_value = std::atomic_ref<node_id_t>(row_[0]).exchange( //
                dense_node_id_unknown_k, std::memory_order_acq_rel);
            while (first_value == dense_node_id_unknown_k) {
                // Monitored capped wait on the lock word itself: the holder's unlock store wakes
                // us; the cap guards the arming race. The address form exists precisely for such
                // in-place words that are not `std::atomic` objects.
                wait_type_ {}(row_.data(), dense_node_id_unknown_k, 0u, fu::wait_capped_t {});
                first_value = std::atomic_ref<node_id_t>(row_[0]).exchange( //
                    dense_node_id_unknown_k, std::memory_order_acq_rel);
            }
            return first_value;
        }
        /** @brief Release-stores @p first_value into slot 0 - unlocking and simultaneously
         *      (re)publishing the edited list to every future acquiring reader. */
        inline void unlock(node_id_t first_value) const noexcept {
            std::atomic_ref<node_id_t>(row_[0]).store(first_value, std::memory_order_release);
        }
    };

    /** @brief The writer-side spelling: mutating paths are always fully synchronized. */
    using neighbors_list_ref_t = neighbors_list_ref<synchronization_t::concurrent_k>;

    /** @brief Immutable view of the vector stored at @p location. */
    inline vector_view_t vector_for(chunked_location_t const& location) const noexcept {
        return vectors_chunks_[location.chunk_index].row(location.index_in_chunk).as_vector();
    }

    /** @brief Mutable span over the vector slot at @p location - only for the slot's unpublished owner. */
    inline vector_span_t vector_span_for(chunked_location_t const& location) noexcept {
        return vectors_chunks_[location.chunk_index].row(location.index_in_chunk).as_vector();
    }

    /** @brief Copy of the pre-computed norms of the vector stored at @p location - the empty
     *      stand-in without touching memory when the kernel never reads norms. */
    inline vector_norms_t norm_for(chunked_location_t const& location) const noexcept {
        if constexpr (std::is_same_v<vector_norms_t, no_norms_t>) return {};
        else return norms_chunks_[location.chunk_index][location.index_in_chunk];
    }

    /** @brief Mutable reference to the norms slot at @p location - only for the slot's unpublished owner. */
    inline vector_norms_t& norm_ref_for(chunked_location_t const& location) noexcept {
        return norms_chunks_[location.chunk_index][location.index_in_chunk];
    }

    /** @brief Typed view over the fixed-capacity neighbors list at @p location - atomic by
     *      default, plain under a mutation-free contract. @ref dense_ram_index_concurrency */
    template <synchronization_t synchronization_ = synchronization_t::concurrent_k>
    inline neighbors_list_ref<synchronization_> neighbors_for(chunked_location_t const& location) noexcept {
        return {neighbors_chunks_[location.chunk_index].row(location.index_in_chunk).as_vector()};
    }
    /** @brief Immutable view of the fixed-capacity neighbors list at @p location. */
    inline node_ids_view_t neighbors_for(chunked_location_t const& location) const noexcept {
        return neighbors_chunks_[location.chunk_index].row(location.index_in_chunk).as_vector();
    }

    /**
     *  @brief Allocates & publishes chunks until at least @p minimal_capacity fresh rows are added,
     *      following the growth schedule of @ref dense_ram_index_malloc.
     *  @return The number of added rows - possibly non-zero even when the status is a failure,
     *      in which case the caller must still credit them back to @c remaining_capacity_.
     *
     *  Only one thread may run this at a time: the "responsible" one that dropped
     *  @c remaining_capacity_ below zero. Successive responsible threads are ordered by the
     *  acquire/release pair on that counter, which is also what publishes the non-atomic
     *  @c chunks_count_ and the directory writes. Readers never touch a chunk before some node
     *  inside it becomes visible, and that visibility chains through the same atomics.
     */
    expected<std::size_t> create_next_chunk(std::size_t minimal_capacity) noexcept {
        // Growing before the geometry is frozen is a usage error, not an allocation failure.
        geometry_t const geometry = geometry_.load(std::memory_order_acquire);
        if (!geometry) return {0, status_t::uninitialized_k};
        std::size_t const chunk_dimensions = geometry.dimensions();
        std::size_t const chunk_connectivity = geometry.connectivity();

        // Size the never-reallocating chunk directories ahead of the very first chunk - safe
        // without any locks, because growth is provably single-threaded via the grower election.
        if (!vectors_chunks_.size() &&
            (fu::failed(vectors_chunks_.resize(max_chunks_k)) || fu::failed(norms_chunks_.resize(max_chunks_k)) ||
             fu::failed(neighbors_chunks_.resize(max_chunks_k))))
            return {0, status_t::bad_alloc_k};

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
                {chunk_rows, chunk_dimensions}, vectors_allocator_);
            // Norms chunks exist only for the kernels whose finalize reads them - dot never
            // pays this pool, in storage or in the per-candidate gather.
            norms_chunk_t norms_chunk;
            if constexpr (!std::is_same_v<vector_norms_t, no_norms_t>)
                norms_chunk = norms_chunk_t::try_zeros({chunk_rows}, norms_allocator_t(vectors_allocator_));
            // Filling the neighbors with the all-ones sentinel is load-bearing: a sentinel leading
            // slot reads as "unpublished" to every traversal, and sentinel tails terminate shorter
            // lists - so no explicit neighbor counts are stored anywhere.
            neighbors_chunk_t neighbors_chunk = neighbors_chunk_t::try_full( //
                {chunk_rows, chunk_connectivity}, dense_node_id_unknown_k, neighbors_allocator_);
            bool const norms_missing = !std::is_same_v<vector_norms_t, no_norms_t> && norms_chunk.empty();
            if (vectors_chunk.empty() || norms_missing || neighbors_chunk.empty())
                return {added_capacity, status_t::bad_alloc_k};

            vectors_chunks_[chunks_count_] = std::move(vectors_chunk);
            norms_chunks_[chunks_count_] = std::move(norms_chunk);
            neighbors_chunks_[chunks_count_] = std::move(neighbors_chunk);
            ++chunks_count_;
            added_capacity += chunk_rows;
            // The final chunk closes the id space at 2^32 rows, but the all-ones id is the sentinel:
            // account one row less, so no claim can ever dispense it and the dispenser never wraps.
            if (chunks_count_ == max_chunks_k) added_capacity -= 1;
        }
        return {added_capacity, status_t::success_k};
    }

    /**
     *  @brief Claims storage capacity for @p count new nodes, growing chunks when needed.
     *  @return The first row of the claim - only meaningful on @c success_k, and rarely needed:
     *      identity is dispensed separately at wiring time.
     *
     *  Spins while other threads are mid-growth, covering all interleavings of the capacity state
     *  machine. Claims @b storage only, never identity: node ids are dispensed one-by-one from
     *  @c next_visible_id_ at wiring time. On success, enough sentinel-filled rows exist for
     *  @p count dispenses, and the caller must roll forward to wiring & publishing them all -
     *  claimed capacity cannot be handed back once ids are dispensed against it, or the
     *  visibility watermark would stall forever. @ref dense_ram_index_concurrency
     */
    expected<node_id_t> claim_capacity(std::size_t count) noexcept {
        // A batch beyond the id-space ceiling can never succeed: reject it before the counter
        // dance, so it neither grows the index to its maximum in vain nor overflows the signed
        // capacity arithmetic below.
        if (count > max_node_count_k) return {dense_node_id_unknown_k, status_t::overflow_k};
        while (true) {
            std::ptrdiff_t const old_remaining_capacity = remaining_capacity_.fetch_sub( //
                count, std::memory_order_acquire);
            std::ptrdiff_t const new_remaining_capacity = old_remaining_capacity - static_cast<std::ptrdiff_t>(count);

            // The thread that moved the counter from non-negative to negative is responsible for
            // the next allocation; the counter can't recover until this thread credits it, so
            // growth stays single-threaded. Accepting `old == 0` is load-bearing: capacity may
            // rest at exactly zero, and `old > 0` would then elect nobody - a proven livelock.
            if (old_remaining_capacity >= 0 && new_remaining_capacity < 0) {
                auto [added_capacity, chunk_status] = create_next_chunk(count);
                if (chunk_status != status_t::success_k) {
                    // Revert our claim - crediting any partially added chunks along the way -
                    // so that a smaller insertion may succeed later.
                    remaining_capacity_.fetch_add(count + added_capacity, std::memory_order_release);
                    return {dense_node_id_unknown_k, chunk_status};
                }
                node_id_t const first_id = last_allocated_id_.fetch_add(count, std::memory_order_acquire);
                remaining_capacity_.fetch_add(added_capacity, std::memory_order_release);
                return {first_id, status_t::success_k};
            }
            // We may be lucky and the space is already reserved for the whole batch.
            else if (new_remaining_capacity >= 0) {
                return {last_allocated_id_.fetch_add(count, std::memory_order_acquire), status_t::success_k};
            }
            // Corner case: the index has 10 free slots, thread Y is growing it for its 20 entries,
            // and we want a 100 - even freshly added capacity may not be enough for us.
            else {
                // Revert the counter, then wait for it to move - any credit or claim wakes the
                // capped monitored wait, and we re-poll regardless once the cap expires.
                std::ptrdiff_t const reverted_capacity = remaining_capacity_.fetch_add(count,
                                                                                       std::memory_order_release) +
                                                         static_cast<std::ptrdiff_t>(count);
                wait_type_ {}(remaining_capacity_, reverted_capacity, 0u, fu::wait_capped_t {});
            }
        }
    }

#pragma endregion Storage

#pragma region Insertion

    /**
     *  @brief Greedy diversity selection over the ascending candidates - the classic NSW/HNSW
     *      pruning heuristic: walking nearest-first, a candidate is kept only when it is closer
     *      to the query than to every already-kept link, so links spread across directions.
     *
     *  Reads the graph without locks. Cost is O(kept · considered) one-to-one distances;
     *  4-way batching is a known follow-up.
     *
     *  @param[in] candidates Search results, ascending by distance to the query.
     *  @param[in] limit Ceiling on the kept links - typically the connectivity.
     *  @param[out] out Receives the kept scored nodes; its live shape is the fill.
     *  @return The number of distance computations performed, for the caller's telemetry.
     */
    std::size_t select_diverse(results_array_t const& candidates, std::size_t limit, selection_t& out,
                               dense_traversal_context_t& context) const noexcept {
        // The thin pick between the engine siblings: pools spanning at least one tile take
        // the span-by-kept-tiles rectangles, keyed purely off `tile_span_k`, never a tuned
        // threshold.
        if constexpr (has_tiles_k)
            if (candidates.size() >= tile_span_k) return select_diverse_via_tiles(candidates, limit, out, context);
        return select_diverse_via_quads(candidates, limit, out, context);
    }

    /**
     *  @brief The vector-engine sibling of @c select_diverse: pair distances computed on the
     *      fly with the 4-way kernels, the classic greedy scan deciding as they arrive. The
     *      kept prefix's views & norms ride the staging surface - free during any selection.
     */
    std::size_t select_diverse_via_quads(results_array_t const& candidates, std::size_t limit, selection_t& out,
                                         dense_traversal_context_t& context) const noexcept {
        if (!out.try_resize({limit})) return 0;
        vector_view_t* const kept_vectors = context.staged_vectors.data();
        vector_norms_t* const kept_norms = context.staged_norms.data();
        small_size_t kept_count = 0;
        std::size_t computed_distances = 0;
        dense_scored_node_t const* candidates_data = candidates.data();
        for (std::size_t index = 0; index != candidates.size() && kept_count != limit; ++index) {
            dense_scored_node_t const candidate = candidates_data[index];
            chunked_location_t const candidate_location = chunked_location(candidate.node_id);
            vector_view_t const candidate_vector = vector_for(candidate_location);
            vector_norms_t const candidate_norms = norm_for(candidate_location);
            // Compare against the kept links in 4-way batches over their cached views; short
            // tails repeat the last link. A batch may score a few links past the first
            // violation, but one 4-way dispatch is still far cheaper than four serial ones.
            small_size_t kept = 0;
            while (kept != kept_count) {
                std::array<distance_t, 4> const kept_distances = distances_quad_of( //
                    candidate_vector, candidate_norms, kept_vectors, kept_norms, kept, kept_count);
                small_size_t const lanes_count = min_of<small_size_t>(kept_count - kept, 4u);
                computed_distances += lanes_count;
                small_size_t losses = 0;
                for (small_size_t lane = 0; lane != lanes_count; ++lane)
                    losses += kept_distances[lane] < candidate.distance;
                if (losses) break; // Some kept link is closer than the query - not diverse.
                kept += lanes_count;
            }
            // Surviving every batch is the diversity verdict itself - no flag to thread through.
            if (kept != kept_count) continue;
            out[kept_count] = candidate;
            kept_vectors[kept_count] = candidate_vector;
            kept_norms[kept_count] = candidate_norms;
            ++kept_count;
        }
        // Shrinking within the reservation never fails - the fill becomes the live shape.
        [[maybe_unused]] bool const shrunk = out.try_resize({kept_count});
        return computed_distances;
    }

    /**
     *  @brief The tile-engine sibling of @c select_diverse: candidates advance one span of
     *      @c tile_span_k rows at a time - the A-side of the tile kit - each span paying one
     *      span × kept-tiles rectangle before the greedy scan decides from precomputed rows,
     *      the standard gather → pack → init → update → finalize walk against the kept links
     *      instead of a full square.
     *
     *  Only the kept tiles frozen at the span's start ride the matrix; links accepted after
     *  that - the sub-tile fringe and any kept tile completed mid-span - score on the 4-way
     *  kernels straight off the staging surface, so no tile is ever recomputed. A kept tile
     *  packs exactly once, when its sixteenth link lands, and serves every later span.
     *  Decisions match @c select_diverse_via_quads including the first-strict-violation
     *  rule, and the shared finalize kit keeps both engines' distances bit-identical - only
     *  the compute schedule differs.
     */
    std::size_t select_diverse_via_tiles(results_array_t const& candidates, std::size_t limit, selection_t& out,
                                         dense_traversal_context_t& context) const noexcept {
        constexpr std::size_t rows_k = tiles_kit_t::tile_rows_k;
        constexpr std::size_t columns_k = tiles_kit_t::tile_columns_k;
        constexpr std::size_t depth_k = tiles_kit_t::tile_depth_k;
        if (!out.try_resize({limit})) return 0;

        vector_view_t* const kept_vectors = context.staged_vectors.data();
        vector_norms_t* const kept_norms = context.staged_norms.data();
        distance_t* const pairs = context.pair_distances.data();
        std::size_t const stride = context.pair_distances.extent(1);
        std::size_t const count = candidates.size();
        dense_scored_node_t const* candidates_data = candidates.data();

        // The packed kept tiles first, then the span's A-tiles, on the tiles' own alignment.
        std::size_t const dimensions = vector_for(chunked_location(candidates_data[0].node_id)).size();
        std::size_t const depth_tiles = tiles_kit_t::depth_tiles(dimensions);
        std::size_t const kept_tiles_capacity = tiles_kit_t::column_tiles(limit);
        typename tiles_kit_t::scratch_t const scratch = tiles_kit_t::scratch_of(context.tiles_scratch.data(),
                                                                                kept_tiles_capacity * depth_tiles);
        typename tiles_kit_t::b_tile_t* const kept_tiles = scratch.b_tiles;
        typename tiles_kit_t::a_tile_t* const a_tiles = scratch.a_tiles;
        tiles_kit_t::configure_tiles();

        typename tiles_kit_t::accumulator_tile_t accumulator;
        std::array<vector_view_t, rows_k> span_vectors;
        std::array<vector_norms_t, rows_k> span_norms;
        std::array<vector_view_t, columns_k> kept_tile_vectors;
        std::array<vector_norms_t, columns_k> kept_tile_norms;
        small_size_t kept_count = 0;
        std::size_t computed_distances = 0;
        for (std::size_t span_begin = 0; span_begin < count && kept_count != limit; span_begin += rows_k) {
            std::size_t const span_count = min_of<std::size_t>(count - span_begin, rows_k);
            for (std::size_t row = 0; row != rows_k; ++row) {
                chunked_location_t const location = chunked_location( //
                    candidates_data[min_of(span_begin + row, count - 1)].node_id);
                span_vectors[row] = vector_for(location);
                span_norms[row] = norm_for(location);
            }

            // The rectangle: this span against every kept tile frozen before it started,
            // finalized into the pair matrix at the kept tile's column offset, span-local rows.
            std::size_t const kept_tiles_frozen = kept_count / columns_k;
            if (kept_tiles_frozen) {
                for (std::size_t depth_tile = 0; depth_tile != depth_tiles; ++depth_tile)
                    tiles_kit_t::gather_a_tile(a_tiles[depth_tile], span_vectors, depth_tile * depth_k);
                for (std::size_t kept_tile = 0; kept_tile != kept_tiles_frozen; ++kept_tile) {
                    for (std::size_t column = 0; column != columns_k; ++column)
                        kept_tile_norms[column] = kept_norms[kept_tile * columns_k + column];
                    tiles_kit_t::init_tile(accumulator);
                    for (std::size_t depth_tile = 0; depth_tile != depth_tiles; ++depth_tile)
                        tiles_kit_t::update_tile(accumulator, a_tiles[depth_tile],
                                                 kept_tiles[kept_tile * depth_tiles + depth_tile]);
                    tiles_kit_t::finalize_tile(accumulator, span_norms, kept_tile_norms, //
                                               pairs + kept_tile * columns_k, stride);
                }
                computed_distances += span_count * kept_tiles_frozen * columns_k;
            }

            for (std::size_t row = 0; row != span_count && kept_count != limit; ++row) {
                dense_scored_node_t const candidate = candidates_data[span_begin + row];
                distance_t const* matrix_row = pairs + row * stride;
                std::size_t const frozen_links = kept_tiles_frozen * columns_k;
                small_size_t losses = 0;
                for (std::size_t kept = 0; kept != frozen_links; ++kept)
                    losses += matrix_row[kept] < candidate.distance;
                // Links accepted after the rectangle froze score on the 4-way path, exactly
                // like the vector sibling - the staging surface already holds them.
                small_size_t fringe = static_cast<small_size_t>(frozen_links);
                while (!losses && fringe != kept_count) {
                    std::array<distance_t, 4> const fringe_distances = distances_quad_of( //
                        span_vectors[row], span_norms[row], kept_vectors, kept_norms, fringe, kept_count);
                    small_size_t const lanes_count = min_of<small_size_t>(kept_count - fringe, 4u);
                    computed_distances += lanes_count;
                    for (small_size_t lane = 0; lane != lanes_count; ++lane)
                        losses += fringe_distances[lane] < candidate.distance;
                    fringe += lanes_count;
                }
                if (losses) continue;
                out[kept_count] = candidate;
                kept_vectors[kept_count] = span_vectors[row];
                kept_norms[kept_count] = span_norms[row];
                ++kept_count;
                // A sixteenth link completes a kept tile: gather → pack it once, through the
                // same staging A-tile, and every later span streams it straight into updates.
                if (kept_count % columns_k == 0) {
                    std::size_t const kept_tile = kept_count / columns_k - 1;
                    for (std::size_t column = 0; column != columns_k; ++column)
                        kept_tile_vectors[column] = kept_vectors[kept_tile * columns_k + column];
                    for (std::size_t depth_tile = 0; depth_tile != depth_tiles; ++depth_tile) {
                        tiles_kit_t::gather_a_tile(a_tiles[0], kept_tile_vectors, depth_tile * depth_k);
                        tiles_kit_t::pack_b_tile(a_tiles[0], kept_tiles[kept_tile * depth_tiles + depth_tile]);
                    }
                }
            }
        }
        // Shrinking within the reservation never fails - the fill becomes the live shape.
        [[maybe_unused]] bool const shrunk = out.try_resize({kept_count});
        return computed_distances;
    }

    /**
     *  @brief The compute half of the tile engine: the full pairwise distance matrix of
     *      @p rows through the tile kit, into the context's @c pair_distances. The finalize
     *      kit is the very one the vector path picked, so both engines produce bit-identical
     *      distances for the deciding siblings to read.
     *
     *  @param[in] rows Scattered views of the pool - the staged list, or a candidate pool.
     *  @param[in] norms The stored norms, parallel to @p rows.
     *  @param[inout] context Supplies @c pair_distances & @c tiles_scratch from @c reserve_tiles.
     *  @return The pair count - tiled refinement spend, honestly counted.
     */
    std::size_t pair_distances_via_tiles(std::span<vector_view_t const> rows, std::span<vector_norms_t const> norms,
                                         dense_traversal_context_t& context) const noexcept {
        constexpr std::size_t rows_k = tiles_kit_t::tile_rows_k;
        constexpr std::size_t columns_k = tiles_kit_t::tile_columns_k;
        constexpr std::size_t depth_k = tiles_kit_t::tile_depth_k;
        std::size_t const count = rows.size();
        std::size_t const stride = context.pair_distances.extent(1);

        // The engine-agnostic tile composition: pack each row-tile once - the Gram symmetry
        // turns its gathered A-strips straight into B-tiles - then init → update over depth →
        // finalize per tile pair. Tail tiles pad by repeating the last valid row, like the
        // 4-way tails; their lanes finalize into the padded stride and are never read.
        std::size_t const dimensions = rows[0].size();
        std::size_t const row_tiles = tiles_kit_t::row_tiles(count);
        std::size_t const depth_tiles = tiles_kit_t::depth_tiles(dimensions);
        typename tiles_kit_t::scratch_t const scratch = tiles_kit_t::scratch_of(context.tiles_scratch.data(),
                                                                                row_tiles * depth_tiles);
        typename tiles_kit_t::b_tile_t* const b_tiles = scratch.b_tiles;
        typename tiles_kit_t::a_tile_t* const a_tiles = scratch.a_tiles;
        tiles_kit_t::configure_tiles();

        std::array<vector_view_t, rows_k> tile_rows;
        for (std::size_t column_tile = 0; column_tile != row_tiles; ++column_tile) {
            for (std::size_t row = 0; row != rows_k; ++row)
                tile_rows[row] = rows[min_of(column_tile * rows_k + row, count - 1)];
            for (std::size_t depth_tile = 0; depth_tile != depth_tiles; ++depth_tile) {
                tiles_kit_t::gather_a_tile(a_tiles[0], tile_rows, depth_tile * depth_k);
                tiles_kit_t::pack_b_tile(a_tiles[0], b_tiles[column_tile * depth_tiles + depth_tile]);
            }
        }

        typename tiles_kit_t::accumulator_tile_t accumulator;
        std::array<vector_norms_t, rows_k> row_norms;
        std::array<vector_norms_t, columns_k> column_norms;
        for (std::size_t row_tile = 0; row_tile != row_tiles; ++row_tile) {
            for (std::size_t row = 0; row != rows_k; ++row) {
                std::size_t const source = min_of(row_tile * rows_k + row, count - 1);
                tile_rows[row] = rows[source];
                row_norms[row] = norms[source];
            }
            for (std::size_t depth_tile = 0; depth_tile != depth_tiles; ++depth_tile)
                tiles_kit_t::gather_a_tile(a_tiles[depth_tile], tile_rows, depth_tile * depth_k);
            // The Gram never reads these rows again, so their lines leave the private caches now.
            [[maybe_unused]] typename tiles_kit_t::demote_candidate_t const demote_candidate {};
            for (std::size_t row = 0; row != rows_k; ++row)
                for (std::size_t depth_tile = 0; depth_tile != depth_tiles; ++depth_tile)
                    demote_candidate(tile_rows[row].data() + depth_tile * depth_k);
            for (std::size_t column_tile = 0; column_tile != row_tiles; ++column_tile) {
                for (std::size_t column = 0; column != columns_k; ++column)
                    column_norms[column] = norms[min_of(column_tile * columns_k + column, count - 1)];
                tiles_kit_t::init_tile(accumulator);
                for (std::size_t depth_tile = 0; depth_tile != depth_tiles; ++depth_tile)
                    tiles_kit_t::update_tile(accumulator, a_tiles[depth_tile],
                                             b_tiles[column_tile * depth_tiles + depth_tile]);
                tiles_kit_t::finalize_tile(
                    accumulator, row_norms, column_norms,
                    context.pair_distances.data() + row_tile * rows_k * stride + column_tile * columns_k, stride);
            }
        }
        return count * count;
    }

    /**
     *  @brief The pair-matrix sibling of @c select_diverse: a pure decision scan over
     *      precomputed distances - no kernels anywhere. The @p pool entries carry matrix
     *      @b positions in their @c node_id; a kept position at the staged fill is the
     *      newcomer, everything below translates through the staged ids.
     */
    std::size_t select_diverse_via_pairs(results_array_t const& pool, std::size_t limit, selection_t& out,
                                         node_id_t newcomer_id, dense_traversal_context_t& context) const noexcept {
        ids_tensor_t const& staged_ids = context.staged_ids;
        distance_t const* pairs = context.pair_distances.data();
        std::size_t const stride = context.pair_distances.extent(1);
        dense_scored_node_t const* pool_data = pool.data();
        if (!out.try_resize({limit})) return 0;
        small_size_t kept_count = 0;
        for (std::size_t index = 0; index != pool.size() && kept_count != limit; ++index) {
            small_size_t const position = static_cast<small_size_t>(pool_data[index].node_id);
            distance_t const* row = pairs + std::size_t(position) * stride;
            small_size_t losses = 0;
            for (small_size_t kept = 0; kept != kept_count; ++kept)
                losses += row[context.kept_positions[kept]] < pool_data[index].distance;
            if (losses) continue;
            bool const is_newcomer = position == staged_ids.size();
            node_id_t const kept_id = is_newcomer ? newcomer_id : staged_ids[position];
            out[kept_count] = dense_scored_node_t(kept_id, pool_data[index].distance);
            context.kept_positions[kept_count] = position;
            ++kept_count;
        }
        // Shrinking within the reservation never fails - the fill becomes the live shape.
        [[maybe_unused]] bool const shrunk = out.try_resize({kept_count});
        return 0; // Decision only - every distance here was already paid for.
    }

    /**
     *  @brief The vector-engine scoring of one staged list against its owning @p neighbor:
     *      4-way sweeps feeding the sorted re-selection pool with real node ids, the
     *      newcomer entering on its reused symmetric score.
     *  @return The distances computed, for the caller's telemetry.
     */
    std::size_t stage_scores_via_quads(vector_view_t neighbor_vector, vector_norms_t neighbor_norms,
                                       node_id_t new_node_id, distance_t newcomer_score,
                                       dense_traversal_context_t& context) const noexcept {
        results_array_t& refining_candidates = context.refining_candidates;
        small_size_t const count = static_cast<small_size_t>(context.staged_ids.size());
        small_size_t const pool_limit = count + 1;
        refining_candidates.insert(dense_scored_node_t(new_node_id, newcomer_score), pool_limit);
        std::size_t computed = 0;
        for (small_size_t scored = 0; scored != count;) {
            std::array<distance_t, 4> const staged_distances = distances_quad_of( //
                neighbor_vector, neighbor_norms, context.staged_vectors.data(), context.staged_norms.data(), scored,
                count);
            small_size_t const lanes_count = min_of<small_size_t>(count - scored, 4u);
            for (small_size_t lane = 0; lane != lanes_count; ++lane)
                refining_candidates.insert(
                    dense_scored_node_t(context.staged_ids[scored + lane], staged_distances[lane]), pool_limit);
            scored += lanes_count;
            computed += lanes_count;
        }
        return computed;
    }

    /**
     *  @brief The tile-engine sibling of @c stage_scores_via_quads: one staged Gram plus
     *      two 4-way rows - the neighbor's scores ordering the pool, and the newcomer's
     *      symmetric row & column - so re-selection decides entirely from precomputed pairs.
     *      Pool entries carry matrix @b positions, ready for @c select_diverse_via_pairs.
     *  @return The distances computed, Gram pairs included - tiled spend, honestly counted.
     */
    std::size_t stage_scores_via_tiles(vector_view_t neighbor_vector, vector_norms_t neighbor_norms,
                                       vector_view_t new_vector, vector_norms_t new_norms, distance_t newcomer_score,
                                       dense_traversal_context_t& context) const noexcept {
        results_array_t& refining_candidates = context.refining_candidates;
        small_size_t const count = static_cast<small_size_t>(context.staged_ids.size());
        small_size_t const pool_limit = count + 1;
        std::size_t const stride = context.pair_distances.extent(1);
        vector_view_t const* staged_vectors = context.staged_vectors.data();
        vector_norms_t const* staged_norms = context.staged_norms.data();
        std::size_t computed = pair_distances_via_tiles(std::span<vector_view_t const> {staged_vectors, count},
                                                        std::span<vector_norms_t const> {staged_norms, count}, context);

        // The neighbor's scores - the pool's ordering - through the vector engine.
        for (small_size_t scored = 0; scored != count;) {
            std::array<distance_t, 4> const scores = distances_quad_of( //
                neighbor_vector, neighbor_norms, staged_vectors, staged_norms, scored, count);
            small_size_t const lanes_count = min_of<small_size_t>(count - scored, 4u);
            for (small_size_t lane = 0; lane != lanes_count; ++lane)
                refining_candidates.insert(dense_scored_node_t(scored + lane, scores[lane]), pool_limit);
            scored += lanes_count;
            computed += lanes_count;
        }

        // The newcomer's row & column, so it competes uniformly inside the matrix at
        // position @c count; its own score is the reused symmetric one - never recomputed.
        distance_t* pairs = context.pair_distances.data();
        for (small_size_t scored = 0; scored != count;) {
            std::array<distance_t, 4> const row = distances_quad_of( //
                new_vector, new_norms, staged_vectors, staged_norms, scored, count);
            small_size_t const lanes_count = min_of<small_size_t>(count - scored, 4u);
            for (small_size_t lane = 0; lane != lanes_count; ++lane) {
                pairs[std::size_t(count) * stride + scored + lane] = row[lane];
                pairs[std::size_t(scored + lane) * stride + count] = row[lane];
            }
            scored += lanes_count;
            computed += lanes_count;
        }
        refining_candidates.insert(dense_scored_node_t(static_cast<node_id_t>(count), newcomer_score), pool_limit);
        return computed;
    }

    /**
     *  @brief Wires the reverse edges: for every selected link, locks that neighbor's list via the
     *      slot-0 sentinel exchange and inserts the fresh node - appending when there is room,
     *      otherwise re-selecting the diverse best among the existing links plus the newcomer.
     *
     *  Locking discipline: at most one neighbor lock is held at any
     *  moment, and never while spinning on the watermark - callers invoke this after wiring
     *  their own row and before their watermark turn, so every lock is released by the time
     *  the turn's spin begins. An exchange that returns the sentinel did @b not acquire the lock -
     *  another writer owns it - so we politely retry. All slot writes happen under the lock, and
     *  the unlocking release-store of slot 0 is what republishes the edited list.
     *
     *  @param[in] selected The fresh node's chosen links with their recorded query distances -
     *      those score the newcomer for free, since the metrics are symmetric.
     *  @param[inout] context Scratch for the re-selection, plus the telemetry counters.
     */
    void back_link_neighbors(node_id_t new_node_id, vector_view_t new_vector, vector_norms_t new_norms,
                             selection_t const& selected, dense_traversal_context_t& context) noexcept {
        // Pairs with the acquire fence in the snapshot probe: a save reading any row word
        // written below - the relaxed tail stores included - proves this node's dispense to
        // the probe's re-check, so an overlapped copy always reports contention.
        std::atomic_thread_fence(std::memory_order_release);

        results_array_t& refining_candidates = context.refining_candidates;
        std::size_t const node_connectivity = connectivity();

        for (small_size_t link = 0; link != selected.size(); ++link) {
            node_id_t const neighbor_id = selected[link].node_id;
            chunked_location_t const neighbor_location = chunked_location(neighbor_id);
            neighbors_list_ref_t const neighbor_list = neighbors_for(neighbor_location);

            // Lock by exchanging the sentinel in; the displaced slot-0 value is ours to restore.
            node_id_t const first_slot_value = neighbor_list.lock();

            // An empty list is encoded as a self-loop: the fresh node becomes the only neighbor.
            if (first_slot_value == neighbor_id) {
                neighbor_list.unlock(new_node_id);
                continue;
            }

            // Fast path: room to append - no distance math at all.
            small_size_t const slots_count = neighbor_list.capacity();
            small_size_t existing_count = 1;
            while (existing_count != slots_count && neighbor_list.load_slot(existing_count) != dense_node_id_unknown_k)
                ++existing_count;
            if (existing_count != slots_count) {
                neighbor_list.store_slot(existing_count, new_node_id);
                neighbor_list.unlock(first_slot_value);
                continue;
            }

            // Slow path: the list is full. Score the existing links & the newcomer by distance to
            // the neighbor itself, and re-run the diversity heuristic to keep the best spread.
            // Distances are computed under the lock for now; an optimistic precompute-and-revalidate
            // scheme shortening this critical section is a known follow-up.
            vector_view_t const neighbor_vector = vector_for(neighbor_location);
            vector_norms_t const neighbor_norms = norm_for(neighbor_location);

            // Stage the list at its own slot positions - the lock-displaced slot-0 value first.
            // The newcomer never enters the staged rows: mixing it in would hand the tile
            // kernels an irregular connectivity + 1 operand, and its score is already known -
            // every supported metric is symmetric, so d(neighbor, newcomer) was recorded by
            // the very selection that kept this link. The staging stays whole tiles.
            if (!context.staged_ids.try_resize({slots_count})) {
                neighbor_list.unlock(first_slot_value);
                continue;
            }
            chunked_location_t const first_location = chunked_location(first_slot_value);
            context.staged_ids[0] = first_slot_value;
            context.staged_vectors[0] = vector_for(first_location);
            context.staged_norms[0] = norm_for(first_location);
            for (small_size_t slot = 1; slot != slots_count; ++slot) {
                node_id_t const existing_id = neighbor_list.load_slot(slot);
                chunked_location_t const existing_location = chunked_location(existing_id);
                context.staged_ids[slot] = existing_id;
                context.staged_vectors[slot] = vector_for(existing_location);
                context.staged_norms[slot] = norm_for(existing_location);
            }

            // The re-selection pool is still the whole list plus the newcomer - that "+ 1"
            // lives in the sorted pool alone, never in any operand shape. The engine pick:
            // tiles Gram the whole staged list and decide from precomputed pairs; vectors
            // sweep and decide as distances arrive. The newcomer's own score is reused in
            // both - symmetric metrics recorded it at selection time, never recomputed.
            refining_candidates.clear();
            selection_t& refined = context.refined_links;
            std::size_t refine_distances = 0;
            if constexpr (has_tiles_k) {
                refine_distances = stage_scores_via_tiles(neighbor_vector, neighbor_norms, new_vector, new_norms,
                                                          selected[link].distance, context);
                select_diverse_via_pairs(refining_candidates, node_connectivity, refined, new_node_id, context);
            }
            else {
                refine_distances = stage_scores_via_quads(neighbor_vector, neighbor_norms, new_node_id,
                                                          selected[link].distance, context);
                refine_distances += select_diverse_via_quads(refining_candidates, node_connectivity, refined, context);
            }
            context.computed_distances += refine_distances;
            context.computed_distances_in_reverse_refines += refine_distances;

            // Rewrite the tail slots, sentinel-terminate any shrink, and unlock with the new head.
            // A non-empty refined fill always holds: the heuristic never rejects the nearest candidate.
            small_size_t const refined_count = static_cast<small_size_t>(refined.size());
            for (small_size_t slot = 1; slot < refined_count; ++slot)
                neighbor_list.store_slot(slot, refined[slot].node_id);
            for (small_size_t slot = refined_count; slot < slots_count; ++slot)
                neighbor_list.store_slot(slot, dense_node_id_unknown_k);
            neighbor_list.unlock(refined[0].node_id);
        }
    }

#pragma endregion Insertion

#pragma region Search

    /**
     *  @brief Samples one 4-way batch of distinct, unvisited, published landmarks, scores them
     *      against the @p query, and queues them into the frontier & the result pool.
     *  @return The landmarks queued; @c 0 on @c success_k means every published node is already
     *      visited, which ends the walk rather than failing it. A failed status means scratch
     *      could not grow, which is a different answer the caller may want to report.
     *
     *  Uniform draws that collide with @c visited fall back to cyclic linear probing over the id
     *  space, so every call makes strict progress without retry budgets - the sampler is exact,
     *  not heuristic. Only transiently locked nodes are re-rolled, mirroring the traversal policy.
     */
    template <synchronization_t synchronization_ = synchronization_t::concurrent_k,
              typename predicate_type_ = dummy_predicate_t>
    expected<std::size_t> sample_landmarks(vector_view_t query, vector_norms_t query_norms,
                                           node_id_t first_invisible_id, std::size_t expansion_factor,
                                           dense_traversal_context_t& context,
                                           predicate_type_&& predicate = predicate_type_ {}) noexcept {
        constexpr bool predicated_k = !std::is_same_v<std::remove_cvref_t<predicate_type_>, dummy_predicate_t>;

        results_array_t& results = context.results;
        frontier_queue_t& frontier = context.frontier;
        visited_set_t& visited = context.visited;

        std::array<node_id_t, 4> landmarks_ids;
        std::array<vector_view_t, 4> landmarks_vectors;
        std::array<vector_norms_t, 4> landmarks_norms;
        small_size_t landmarks_in_batch = 0;

        if (status_t const seen = visited.reserve(visited.size() + landmarks_ids.size()); failed(seen))
            return {0, seen};
        // `visited` may also hold above-watermark ids discovered through fresh back-links, so the
        // size check is conservative: it can only under-sample, never let the probe loop spin.
        while (landmarks_in_batch != landmarks_ids.size() && visited.size() < first_invisible_id) {
            // Draw uniformly; on collision with an already-visited id, probe forward cyclically -
            // deterministic progress even when the published set is almost fully explored.
            node_id_t landmark_node_id = uniform_below<node_id_t> {first_invisible_id}(context.random_generator);
            while (visited.test(landmark_node_id))
                landmark_node_id = landmark_node_id + 1 == first_invisible_id ? 0 : landmark_node_id + 1;

            chunked_location_t const landmark_location = chunked_location(landmark_node_id);
            // Published nodes may be transiently locked for re-linking - re-roll those. Under a
            // mutation-free contract every sampled node is published & unlocked - no probe needed.
            if constexpr (synchronization_ == synchronization_t::concurrent_k)
                if (neighbors_for(landmark_location).peek_first() == dense_node_id_unknown_k) continue;

            [[maybe_unused]] membership_t const marked = visited.set(landmark_node_id);
            landmarks_ids[landmarks_in_batch] = landmark_node_id;
            landmarks_vectors[landmarks_in_batch] = vector_for(landmark_location);
            landmarks_norms[landmarks_in_batch] = norm_for(landmark_location);
            ++landmarks_in_batch;
        }
        if (!landmarks_in_batch) return {0, status_t::success_k};

        // Pad the tail lanes with lane zero, so the 4-way kernel streams valid memory;
        // the padded outputs are simply never consumed.
        for (small_size_t lane = landmarks_in_batch; lane != landmarks_ids.size(); ++lane) {
            landmarks_ids[lane] = landmarks_ids[0];
            landmarks_vectors[lane] = landmarks_vectors[0];
            landmarks_norms[lane] = landmarks_norms[0];
        }
        std::array<distance_t, 4> const landmark_distances = distances( //
            query, query_norms, landmarks_vectors, landmarks_norms);
        context.computed_distances += landmarks_in_batch;
        std::array<dense_scored_node_t, 4> landmarks_scored;
        for (small_size_t lane = 0; lane != landmarks_in_batch; ++lane)
            landmarks_scored[lane] = dense_scored_node_t(landmarks_ids[lane], landmark_distances[lane]);
        std::span const scored = std::span(landmarks_scored).first(landmarks_in_batch);
        network_sort(scored, dense_closer_scored_node_t {});

        // Matching landmarks enter the admitted pool right here: they are marked visited, so
        // the walk never re-scores them and this is their one admission opportunity.
        if constexpr (predicated_k) {
            dense_admissions_t admissions;
            invoke_admission(predicate, scored, admissions);
            network_sort(std::span(admissions), dense_closer_scored_node_t {});
            context.refining_candidates.insert(std::span(admissions), expansion_factor);
        }
        results.insert(scored, expansion_factor);
        for (small_size_t lane = 0; lane != landmarks_in_batch; ++lane)
            if (status_t const queued = frontier.insert(dense_scored_node_t(scored[lane])); failed(queued))
                return {lane, queued};
        return {landmarks_in_batch, status_t::success_k};
    }

    /**
     *  @brief Greedy best-first NSW walk from random published landmarks toward the @p query.
     *
     *  Read-only on the graph: takes no locks, writes nothing, retries transiently locked nodes.
     *  Scratch allocation failures degrade the walk - keeping the best candidates found so far -
     *  rather than fail it: callers may already own claimed node ids and must roll forward
     *  regardless. The returned status says which kind of stop happened, so a read-only caller
     *  can report a refusal that an insertion caller must swallow.
     *
     *  @param[in] first_invisible_id The caller's watermark snapshot, bounding the readable id space.
     *  @param[in] expansion_factor Breadth of the candidate pool - the walk's stopping radius.
     *  @param[inout] context Scratch & telemetry; must arrive pre-reserved & pre-cleared, and ends
     *      with up to @p expansion_factor nearest published candidates in @c context.results.
     *  @param[in] predicate Optional admission filter - see @c search_if. Non-matching nodes
     *      still score and route the walk; only matching ones enter @c context.refining_candidates,
     *      whose radius governs termination once that pool fills.
     */
    template <synchronization_t synchronization_ = synchronization_t::concurrent_k,
              typename predicate_type_ = dummy_predicate_t>
    status_t search_around(vector_view_t query, vector_norms_t query_norms, node_id_t first_invisible_id,
                           std::size_t expansion_factor, dense_traversal_context_t& context,
                           predicate_type_&& predicate = predicate_type_ {}) noexcept {
        constexpr bool predicated_k = !std::is_same_v<std::remove_cvref_t<predicate_type_>, dummy_predicate_t>;

        results_array_t& results = context.results;
        frontier_queue_t& frontier = context.frontier;
        visited_set_t& visited = context.visited;
        std::size_t const node_connectivity = connectivity();

        // Seed with about as many landmarks as the graph connectivity - a stochastic multi-start,
        // gathered in 4-way batches to amortize the loop overhead over unrolled "gathers".
        small_size_t const landmark_batches = divide_round_up<small_size_t>(node_connectivity, 4);
        for (small_size_t landmarks_batch = 0; landmarks_batch != landmark_batches; ++landmarks_batch) {
            expected<std::size_t> const seeded = sample_landmarks<synchronization_>(
                query, query_norms, first_invisible_id, expansion_factor, context, predicate);
            if (failed(seeded.status)) return seeded.status;
            if (!seeded.value) break; // Everything published is already visited.
        }

        // Greedily walk the frontier nearest-first, keeping the best `expansion_factor` in `results`.
        while (true) {
            // Starvation restart, knob-free: a drained frontier before a full result pool means
            // the walk exhausted everything reachable from its seeds - a sparse early graph or a
            // disconnected component - so re-seed and continue, like restarting a stochastic
            // descent. Termination needs no budget: every successful batch strictly grows both
            // `visited` (bounded by the published set) and `results` (bounded by the expansion
            // factor), so restarts are self-limiting by construction.
            if (frontier.empty()) {
                // A predicated walk re-seeds until the admitted pool fills - matches may hide in
                // regions the drained frontier never reached - still self-limited by the visited
                // set's growth toward the published id space.
                std::size_t const filled = predicated_k ? context.refining_candidates.size() : results.size();
                if (filled >= expansion_factor) break;
                expected<std::size_t> const seeded = sample_landmarks<synchronization_>(
                    query, query_norms, first_invisible_id, expansion_factor, context, predicate);
                if (failed(seeded.status)) return seeded.status;
                if (!seeded.value) break; // Nothing fresh left to restart from.
                ++context.restarts;
                continue;
            }

            // Peek - don't pop yet: the chosen node's neighbor list may be locked, in which case
            // we retry the selection rather than dropping it from the frontier.
            dense_scored_node_t const candidate = frontier.top();
            if constexpr (predicated_k) {
                // A predicated walk switches to the admitted pool's radius once that pool
                // fills: matches are sparser than nodes, so that radius is the wider one and
                // the walk runs deeper; until it fills, the navigation radius keeps the
                // frontier honest.
                results_array_t const& matches = context.refining_candidates;
                bool const matches_full = matches.size() == expansion_factor;
                if (matches_full && candidate.distance > matches.top().distance) break;
                if (!matches_full && results.size() == expansion_factor && candidate.distance > results.top().distance)
                    break;
            }
            else {
                // `results` stays sorted ascending, so its back is the furthest kept - our search radius.
                distance_t const radius = results.top().distance;
                // Once the nearest unexpanded node is past that radius and `results` is full,
                // nothing deeper can improve the answer.
                if (candidate.distance > radius && results.size() == expansion_factor) break;
            }

            chunked_location_t const candidate_location = chunked_location(candidate.node_id);
            neighbors_list_ref<synchronization_> const candidate_neighbors = neighbors_for<synchronization_>( //
                candidate_location);

            // Slot 0 holds either the lock sentinel or the first neighbor's id. If the node is
            // locked for re-linking, spin: it will either free up, or we keep re-evaluating the
            // same nearest candidate until it does - dropping it would silently degrade recall.
            node_id_t const first_neighbor_id = candidate_neighbors.peek_first();
            if constexpr (synchronization_ == synchronization_t::concurrent_k)
                if (first_neighbor_id == dense_node_id_unknown_k) continue;

            // The node is live and readable - commit to it and drop it from the frontier.
            frontier.pop();
            ++context.visited_members;

            // Scan the fixed-capacity sentinel-terminated list. Slot 0 comes from the acquiring load
            // above - never re-read it, a writer may have locked the node since. The remaining
            // slots are relaxed atomic reads that may interleave with a concurrent re-linker, so
            // the worst case is a stale-but-valid mix of neighbors, never a torn read.
            small_size_t const slots_count = candidate_neighbors.capacity();
            small_size_t neighbor_slot = 0;
            bool list_ended = false;
            while (!list_ended && neighbor_slot != slots_count) {

                // Gather the unvisited neighbors in 4-way batches, score them against the query,
                // and push the promising ones into the frontier & result pool.
                std::array<node_id_t, 4> gathered_ids;
                std::array<vector_view_t, 4> gathered_vectors;
                std::array<vector_norms_t, 4> gathered_norms;
                small_size_t gathered_count = 0;
                for (; gathered_count != gathered_ids.size() && neighbor_slot != slots_count; ++neighbor_slot) {
                    node_id_t const neighbor_id = neighbor_slot ? candidate_neighbors.load_slot(neighbor_slot) //
                                                                : first_neighbor_id;
                    if (neighbor_id == dense_node_id_unknown_k) {
                        list_ended = true;
                        break;
                    }
                    // Translation-frontier invariant: only ids below our acquire-fenced watermark
                    // snapshot may be located. A fresher id read from a concurrently re-linked slot
                    // has no happens-before edge to its hosting chunk yet, so skip it; a later
                    // traversal with a fresher snapshot will meet it safely.
                    if (neighbor_id >= first_invisible_id) continue;
                    if (visited.test(neighbor_id)) continue;
                    chunked_location_t const neighbor_location = chunked_location(neighbor_id);
                    gathered_ids[gathered_count] = neighbor_id;
                    gathered_vectors[gathered_count] = vector_for(neighbor_location);
                    gathered_norms[gathered_count] = norm_for(neighbor_location);
                    ++gathered_count;
                }
                if (!gathered_count) continue;

                // Pad the tail lanes with lane zero, so the 4-way kernel streams valid memory;
                // the padded outputs are simply never consumed.
                for (small_size_t lane = gathered_count; lane != gathered_ids.size(); ++lane) {
                    gathered_ids[lane] = gathered_ids[0];
                    gathered_vectors[lane] = gathered_vectors[0];
                    gathered_norms[lane] = gathered_norms[0];
                }
                std::array<distance_t, 4> const gathered_distances = distances( //
                    query, query_norms, gathered_vectors, gathered_norms);
                context.computed_distances += gathered_count;
                std::array<dense_scored_node_t, 4> scored_batch;
                for (small_size_t lane = 0; lane != gathered_count; ++lane)
                    scored_batch[lane] = dense_scored_node_t(gathered_ids[lane], gathered_distances[lane]);
                std::span const scored = std::span(scored_batch).first(gathered_count);
                network_sort(scored, dense_closer_scored_node_t {});

                // Predicate admission is independent of navigation: a matching node enters the
                // admitted pool even when the navigation pool rejects it, and a non-matching
                // node keeps routing - filtering the walk itself would strand it in islands.
                // The re-sort only normalizes what a reordering predicate may hand back - for
                // order-preserving ones the network fires no exchanges.
                if constexpr (predicated_k) {
                    dense_admissions_t admissions;
                    invoke_admission(predicate, scored, admissions);
                    network_sort(std::span(admissions), dense_closer_scored_node_t {});
                    context.refining_candidates.insert(std::span(admissions), expansion_factor);
                }

                // Grow the visited set ahead of the marks - `set()` itself never rehashes.
                // Full quads - the hot case - mark all four lanes in one interleaved probe sweep.
                if (status_t const seen = visited.reserve(visited.size() + gathered_count); failed(seen)) return seen;
                if (gathered_count == gathered_ids.size()) {
                    [[maybe_unused]] u64_t const fresh = visited.set(gathered_ids);
                }
                else
                    for (small_size_t lane = 0; lane != gathered_count; ++lane) {
                        [[maybe_unused]] membership_t const marked = visited.set(gathered_ids[lane]);
                    }

                // One merge folds the ascending quad into the pool; the accepted entries are
                // its prefix - the top-expansion-of-the-union truth - and only they enter the
                // frontier.
                std::size_t const accepted_count = results.insert(scored, expansion_factor);
                for (small_size_t lane = 0; lane != accepted_count; ++lane)
                    if (status_t const queued = frontier.insert(dense_scored_node_t(scored[lane])); failed(queued))
                        return queued;
            }
        }
        return status_t::success_k;
    }

#pragma endregion Search

#pragma region Checkpoints

    /** @brief Copies the first @p count rows of every chunk in @p chunks into the dense rows
     *      of @p output - typed element copies, one bulk run per chunk. The row width in
     *      storage values derives from the stride, so sub-byte packing needs no special case.
     *      Templated because the chunk directories are distinct @c fu::dynamic_padded_array
     *      instantiations. Concurrent back-linkers write neighbor words through
     *      @c std::atomic_ref, so the speculative concurrent arm reads them the same way;
     *      certified contracts keep the vectorizable bulk copy. */
    template <synchronization_t synchronization_, typename chunks_type_, typename value_type_>
    static void save_rows_(nk::tensor_span<value_type_, 2> output, chunks_type_ const& chunks,
                           std::size_t count) noexcept {
        std::size_t const row_values = static_cast<std::size_t>(output.stride_bytes(0)) / sizeof(value_type_);
        value_type_* destination = output.data();
        for (small_size_t chunk = 0; count; ++chunk) {
            std::size_t const chunk_rows = capacity_of_chunks(chunk + 1) - capacity_of_chunks(chunk);
            std::size_t const rows = min_of(chunk_rows, count);
            std::size_t const values = rows * row_values;
            value_type_ const* source = chunks[chunk].data();
            // The `const_cast` is the pre-P3323 currency for atomics living in const memory.
            if constexpr (synchronization_ == synchronization_t::concurrent_k)
                for (std::size_t value = 0; value != values; ++value)
                    destination[value] = std::atomic_ref<value_type_>(const_cast<value_type_&>(source[value]))
                                             .load(std::memory_order_relaxed);
            else std::copy_n(source, values, destination);
            destination += values;
            count -= rows;
        }
    }

    /** @brief Fills the first @p count rows of every chunk in @p chunks from the dense rows of
     *      @p input. Templated for the same reason as @ref save_rows_. */
    template <typename chunks_type_, typename value_type_>
    static void load_rows_(nk::tensor_view<value_type_, 2> input, chunks_type_& chunks, std::size_t count) noexcept {
        std::size_t const row_values = static_cast<std::size_t>(input.stride_bytes(0)) / sizeof(value_type_);
        value_type_ const* source = input.data();
        for (small_size_t chunk = 0; count; ++chunk) {
            std::size_t const chunk_rows = capacity_of_chunks(chunk + 1) - capacity_of_chunks(chunk);
            std::size_t const rows = min_of(chunk_rows, count);
            std::copy_n(source, rows * row_values, chunks[chunk].data());
            source += rows * row_values;
            count -= rows;
        }
    }

#pragma endregion Checkpoints

    /**
     *  @brief Separately allocated matrix slabs, where each row is one vector entry. Slot indexes are global,
     *      monotonically transparently growing across individual matrices.
     *
     *  Each directory is sized once to @c max_chunks_k entries in @c reserve() and never reallocates
     *  after that, so readers can dereference chunk headers with zero synchronization while a single
     *  writer default-constructs the next one in place. The stride pads every header to
     *  @c default_alignment_k, keeping the writer's half-built entry off the readers' cache lines.
     *  @ref dense_ram_index_malloc
     */
    fu::dynamic_padded_array<vectors_chunk_t, vectors_allocator_type_> vectors_chunks_;

    /**
     *  @brief Separately allocated array slabs, where each entry holds the pre-computed norms of one vector.
     *      Slot indexes are global, monotonically transparently growing across individual arrays.
     *  @ref dense_ram_index_malloc
     */
    fu::dynamic_padded_array<norms_chunk_t, vectors_allocator_type_> norms_chunks_;

    /**
     *  @brief Separately allocated matrix slabs, where each row is the fixed-capacity neighbors list of one vector.
     *      Slot indexes are global, monotonically transparently growing across individual matrices.
     *  @ref dense_ram_index_malloc
     */
    fu::dynamic_padded_array<neighbors_chunk_t, neighbors_allocator_type_> neighbors_chunks_;

    /**
     *  @brief Signed integer describing "unsigned capacity" to simplify underflow/overflow detection from concurrent
     *      thread updates. @b Never assign values to it, always use increment & decrement to move it up and down.
     */
    std::atomic<std::ptrdiff_t> remaining_capacity_ {0};

    /**
     *  @brief Unsigned integer containing the identifier of the last "allocated" node. It defines the lower
     *      bound for the number of initialized or empty rows across @c vectors_chunks_ and @c neighbors_chunks_
     *      chunks. It's typically incremented in batches.
     */
    std::atomic<node_id_t> last_allocated_id_ {0};

    /**
     *  @brief The id dispenser: the identifier the next wired node receives. Incremented one-by-one
     *      at wiring time, so ids follow the global wiring-start order and each thread holds at most
     *      one unpublished id. @ref dense_ram_index_concurrency
     */
    std::atomic<node_id_t> next_visible_id_ {0};
    /**
     *  @brief The visibility watermark: every id strictly below it is published, its vector, norms &
     *      neighbors readable - the safe bound for random landmark sampling. Advanced strictly in
     *      dispensed order by each publisher taking its turn. @ref dense_ram_index_concurrency
     */
    std::atomic<node_id_t> last_visible_id_ {0};

    std::atomic<geometry_t> geometry_ {};

    /** @brief Constructed chunks per directory. Deliberately non-atomic: only the one "responsible"
     *      grower thread mutates it, and successive growers are ordered by the acquire/release pair
     *      on @c remaining_capacity_. @ref create_next_chunk */
    small_size_t chunks_count_ = 0;
};

/**
 *  @brief External reader's cursor over the published prefix of one @c dense_ram_index:
 *      random access & id-order iteration over nodes, plus edge-following through a privately
 *      staged copy of the current node's neighbor list.
 *
 *  Strictly read-only under every contract - the index is never written through, so readers
 *  can never stall writers or convoy behind them. Under @c concurrent_k the staging reads
 *  neighbor words through relaxed atomics - a re-linker may interleave with the copy, and the
 *  worst case is the same stale-but-valid mix the traversal itself tolerates; exact lists come
 *  from the @c immutable_k contract, which reads plain memory. The cursor captures the frozen
 *  geometry once at construction, starts past the end, and @c seek(0) enters the graph.
 *
 *  The staged list may contain ids at or above this cursor's watermark snapshot, discovered
 *  through fresh back-links; @c seek & @c follow reject those, and @c refresh widens the
 *  snapshot to admit them. A cursor standing past the end stays past the end through a
 *  @c refresh - entering the admitted region takes an explicit @c seek.
 */
template <typename index_type_, synchronization_t synchronization_, typename allocator_type_>
struct dense_ram_cursor {

    using index_t = index_type_;
    using allocator_t = allocator_type_;
    using scalar_t = typename index_t::scalar_t;
    using vector_view_t = typename index_t::vector_view_t;
    using vector_norms_t = typename index_t::vector_norms_t;
    using node_id_t = typename index_t::node_id_t;
    using small_size_t = typename index_t::small_size_t;
    using node_ids_view_t = typename index_t::node_ids_view_t;
    using location_t = typename index_t::chunked_location_t;

    using ids_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<node_id_t>;
    using ids_tensor_t = nk::tensor<node_id_t, ids_allocator_t, 1>;

    /** @brief The watermark ordering behind this cursor's snapshots: acquiring while writers
     *      may be live, plain when the caller certified a mutation-free environment. */
    inline static constexpr std::memory_order watermark_order_k = synchronization_ == synchronization_t::concurrent_k
                                                                      ? std::memory_order_acquire
                                                                      : std::memory_order_relaxed;

    /** @brief The traversed index - never written through, under any contract. */
    index_t const& index_;
    /** @brief The staged @b [connectivity] neighbor copy of the current node. */
    ids_tensor_t staged_neighbors_;
    /** @brief Chunk & row of the current node, derived once per @c seek. */
    location_t location_ {};
    /** @brief Decoded live prefix of @c staged_neighbors_ - self-loops & sentinels stripped. */
    small_size_t staged_count_ = 0;
    /** @brief This cursor's watermark snapshot: ids strictly below it are readable. */
    node_id_t visible_count_;
    /** @brief The current node id, or the watermark snapshot when past the end. */
    node_id_t current_;
    /** @brief The frozen neighbors-list width, captured once at construction. */
    small_size_t connectivity_;

    /** @brief Binds the cursor & captures its snapshots; allocates nothing - call @c reserve(). */
    explicit dense_ram_cursor(index_t const& index) noexcept
        : index_(index), visible_count_(index.last_visible_id_.load(watermark_order_k)), current_(visible_count_),
          connectivity_(static_cast<small_size_t>(index.connectivity())) {}

    dense_ram_cursor(dense_ram_cursor const&) = delete;
    dense_ram_cursor& operator=(dense_ram_cursor const&) = delete;
    dense_ram_cursor(dense_ram_cursor&&) = default;
    dense_ram_cursor& operator=(dense_ram_cursor&&) = delete;

    /** @brief Sizes the staging for the frozen connectivity through @p allocator - the index's
     *      traversals-side family by default; call once before any @c seek. */
    status_t reserve(allocator_t allocator = {}) noexcept {
        if (staged_neighbors_.size() >= connectivity_) return status_t::success_k;
        staged_neighbors_ = ids_tensor_t::try_zeros({connectivity_}, ids_allocator_t(allocator));
        return !staged_neighbors_.empty() ? status_t::success_k : status_t::bad_alloc_k;
    }

    /** @brief Nodes visible to this cursor; ids strictly below are readable. */
    node_id_t visible_count() const noexcept { return visible_count_; }

    /** @brief Widens the snapshot to admit nodes published since; a cursor standing past the
     *      end stays past the end - entering the admitted region takes an explicit @c seek. */
    void refresh() noexcept {
        bool const past_end = current_ == visible_count_;
        visible_count_ = index_.last_visible_id_.load(watermark_order_k);
        if (past_end) current_ = visible_count_;
    }

    /** @brief The current node's id; equals @c visible_count() past the end. */
    node_id_t id() const noexcept { return current_; }
    /** @brief @c true while the cursor points at a readable node. */
    explicit operator bool() const noexcept { return current_ < visible_count_; }
    /** @brief Immutable view of the current node's vector; meaningless past the end. */
    vector_view_t vector() const noexcept { return index_.vector_for(location_); }
    /** @brief The current node's pre-computed norms; meaningless past the end. */
    vector_norms_t norms() const noexcept { return index_.norm_for(location_); }
    /** @brief The staged neighbor ids of the current node - decoded, stable until the next move. */
    std::span<node_id_t const> neighbors() const noexcept { return to_span(staged_neighbors_).first(staged_count_); }

    /** @brief Positions on @p node & stages a copy of its neighbor list: @c malformed_k at or
     *      past the snapshot, @c uninitialized_k before @c reserve - the cursor stays where it
     *      was on either refusal. Read-only under every contract. */
    status_t seek(node_id_t node) noexcept {
        if (node >= visible_count_) return status_t::malformed_k;
        if (staged_neighbors_.size() < connectivity_) return status_t::uninitialized_k;
        current_ = node;
        location_ = index_.chunked_location(node);
        node_ids_view_t const row = index_.neighbors_for(location_);
        // Slot 0 is the lock word: acquire it exactly once - the sentinel means a writer is
        // mid-edit, waited out through the index's own monitor - and never re-read it, so a
        // fresh lock sentinel cannot slip into the staging. The `const_cast` is the
        // pre-P3323 currency for atomics living in const memory.
        node_id_t first_value = row[0];
        if constexpr (synchronization_ == synchronization_t::concurrent_k) {
            first_value = std::atomic_ref<node_id_t>(const_cast<node_id_t&>(row[0])).load(std::memory_order_acquire);
            while (first_value == dense_node_id_unknown_k) {
                typename index_t::wait_t {}(row.data(), dense_node_id_unknown_k, small_size_t(0), fu::wait_capped_t {});
                first_value =
                    std::atomic_ref<node_id_t>(const_cast<node_id_t&>(row[0])).load(std::memory_order_acquire);
            }
        }
        staged_neighbors_[0] = first_value;
        // Tail slots may interleave with a live re-linker: relaxed atomics make the
        // stale-but-valid mix defined; certified contracts read plain memory.
        if constexpr (synchronization_ == synchronization_t::concurrent_k)
            for (small_size_t slot = 1; slot != connectivity_; ++slot)
                staged_neighbors_[slot] =
                    std::atomic_ref<node_id_t>(const_cast<node_id_t&>(row[slot])).load(std::memory_order_relaxed);
        else
            for (small_size_t slot = 1; slot != connectivity_; ++slot) staged_neighbors_[slot] = row[slot];
        // Decode the row's encodings: a self-loop is "published, zero neighbors", and the
        // all-ones sentinel terminates shorter lists.
        if (staged_neighbors_[0] == node) { staged_count_ = 0; }
        else {
            small_size_t live = 0;
            while (live != connectivity_ && staged_neighbors_[live] != dense_node_id_unknown_k) ++live;
            staged_count_ = live;
        }
        return status_t::success_k;
    }

    /** @brief Moves to the next id below the snapshot, or past the end; a staging refusal
     *      parks on the new id with an empty row - a later @c seek to the same id retries. */
    dense_ram_cursor& operator++() noexcept {
        node_id_t const next = current_ + 1;
        if (next >= visible_count_) {
            current_ = visible_count_;
            staged_count_ = 0;
            return *this;
        }
        if (failed(seek(next))) {
            current_ = next;
            staged_count_ = 0;
        }
        return *this;
    }
    /** @brief Deleted: it would copy the staging tensor - an allocation in disguise. */
    dense_ram_cursor operator++(int) = delete;

    /** @brief Follows edge @p slot of the staged list: @c malformed_k when the slot is beyond
     *      the live prefix or its target is at or past this cursor's snapshot. */
    status_t follow(small_size_t slot) noexcept {
        if (slot >= staged_count_) return status_t::malformed_k;
        return seek(staged_neighbors_[slot]);
    }
};

} // namespace usearch
} // namespace unum
