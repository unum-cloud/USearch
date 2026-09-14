/**
 *  @brief Vector Search index for a single uniform latency memory space.
 *  @author Ash Vardanian
 *  @file include/usearch/dense/ram_index.hpp
 *  @date July 26, 2023
 */
#pragma once

#include <cstring> // `std::memcpy` - one incoming vector into its published row

#include <bit>         // `std::bit_width`, `std::bit_floor`
#include <limits>      // `std::numeric_limits` - the audit's bits per word
#include <span>        // `std::span` - the neighbor run a load rebases
#include <type_traits> // `std::has_unique_object_representations_v` - the geometry word's guard

#include <forkunion/atomics.hpp> // `fu::standard_atomic_ref` - the default atomic reference
#include <forkunion/types.hpp>   // `fu::standard_yield_t` - the default waiter
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
 *  Instead, we reuse the per-node neighbors to also contain per-node locks. This conveniently places them
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
template <                                                                    //
    nk_dtype_t vector_dtype_ = nk_bf16_k,                                     //
    nk_kernel_kind_t kernel_kind_ = nk_kernel_angular_k,                      //
    nk_capability_t capability_ = nk_cap_icelake_k,                           //
    typename wait_type_ = fu::standard_yield_t,                               //
    template <typename> class atomic_ref_template_ = fu::standard_atomic_ref, //
    typename vectors_allocator_type_ = std::allocator<byte_t>,                //
    typename neighbors_allocator_type_ = std::allocator<byte_t>,              //
    typename traversals_allocator_type_ = std::allocator<byte_t>              //
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

/** @brief Per-thread traversal scratch: the candidate queues, the visited set, the neighbor
 *      selection buffers, and the running telemetry counters - one instance per scanning thread. */
template <                                          //
    nk_dtype_t vector_dtype_ = nk_bf16_k,           //
    nk_capability_t capability_ = nk_cap_icelake_k, //
    typename stored_norms_type_ =                   //
    usearch::vector_norms<typename nk::type_for<vector_dtype_>::type::dot_result_t>,
    typename allocator_type_ = std::allocator<byte_t>>
struct dense_traversal_context;

template <nk_dtype_t vector_dtype_, nk_capability_t capability_, typename stored_norms_type_, typename allocator_type_>
struct dense_traversal_context {

    using scalar_t = typename nk::type_for<vector_dtype_>::type;
    using vector_view_t = nk::vector_view<scalar_t>;
    using vector_norms_t = stored_norms_type_;

    using allocator_t = allocator_type_;
    using candidates_allocator_t =
        typename std::allocator_traits<allocator_t>::template rebind_alloc<dense_scored_node_t>;
    /** @brief The hash-set allocates raw bytes and places IDs itself, so it demands a byte allocator. */
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
    /** @brief One selection's output: @b [kept] scored nodes - the ID and its distance to the
     *      selection's own query. The tensor's live shape is the fill; @c capacity() is the
     *      reservation, so no separate count exists anywhere. */
    using selection_t = nk::tensor<dense_scored_node_t, candidates_allocator_t, 1>;

    using results_array_t = sorted_buffer<dense_scored_node_t, dense_closer_scored_node_t, candidates_allocator_t>;
    /** @brief 8-ary: eight 8-byte scored nodes fill one cache line per level scan, and the
     *      insert-dominated frontier (~2x connectivity pushes per pop) rides the shallower
     *      log_8 @c shift_up depth. */
    using frontier_queue_t = max_heap_dary<8, dense_scored_node_t, dense_further_scored_node_t, candidates_allocator_t>;
    /** @brief The visited set walks its slots with the kit the class's mask earns: sixteen at a
     *      time on the AVX-512 classes, one at a time below them. */
    using visited_set_t = growing_hash_set<dense_node_id_t, std::hash<dense_node_id_t>, visits_allocator_t,
                                           typename dense_visits_hash_set_chain<capability_>::kit_t>;

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

    /** @brief Neighbors chosen for the freshly inserted node, with their query distances -
     *      reusable while back-linking, since every supported metric is symmetric. */
    selection_t selected_neighbors;
    /** @brief Re-selection output for one saturated neighbor's neighbors. */
    selection_t refined_neighbors;

    /** @brief Neighbor IDs of one staged node - the input of a re-selection, never its output. */
    ids_tensor_t staged_ids;
    /** @brief The one staging surface both engines read: contiguous views of the staged rows
     *      during scoring, then the kept prefix of whichever selection is running. The two
     *      lives never overlap - scoring drains the staged rows before any selection writes. */
    views_tensor_t staged_vectors;
    /** @brief Stored norms, parallel to @c staged_vectors. */
    norms_tensor_t staged_norms;
    /** @brief Pair-matrix positions of the kept neighbors, parallel to the running selection's fill. */
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
    /** @brief Subset of @c computed_distances spent re-selecting a full neighbor set while back-linking. */
    std::size_t computed_distances_in_reverse_refines = 0;
    /** @brief Frontier candidates whose neighbors were expanded during traversals. */
    std::size_t visited_members = 0;
    /** @brief Starvation restarts: re-seedings after the frontier drained prematurely -
     *      a graph-connectivity health signal unique to the flat NSW design. */
    std::size_t restarts = 0;
    /** @brief Walks a scratch refusal ended early - ranked short, and on insertion wired thinner. */
    std::size_t saturated_walks = 0;

    dense_traversal_context() noexcept = delete;
    dense_traversal_context(dense_traversal_context const&) noexcept = delete;
    dense_traversal_context& operator=(dense_traversal_context const&) noexcept = delete;

    /** @brief Constructs every traversal container with a rebound copy of @p alloc - and keeps
     *      the allocator for the lazily-sized neighbor scratches - so all scratch lands wherever the
     *      caller's allocator places it, like a NUMA domain local to the owning thread. */
    dense_traversal_context(allocator_t alloc) noexcept
        : results(candidates_allocator_t(alloc)), refining_candidates(candidates_allocator_t(alloc)),
          frontier(candidates_allocator_t(alloc)), visited(visits_allocator_t(alloc)), random_generator(),
          allocator_(alloc) {}
    dense_traversal_context(dense_traversal_context&&) = default;
    dense_traversal_context& operator=(dense_traversal_context&&) = default;

    /** @brief Sizes every neighbor scratch for @p capacity entries - the frozen connectivity, the
     *      largest selection any path stages - through the context's own allocator. Geometry
     *      freezes once, so the tensors allocate once. */
    status_t reserve_neighbors(std::size_t capacity) noexcept {
        if (selected_neighbors.capacity() >= capacity) return status_t::success_k;
        selected_neighbors = selection_t::try_zeros({capacity}, candidates_allocator_t(allocator_));
        refined_neighbors = selection_t::try_zeros({capacity}, candidates_allocator_t(allocator_));
        staged_ids = ids_tensor_t::try_zeros({capacity}, ids_allocator_t(allocator_));
        staged_vectors = views_tensor_t::try_zeros({capacity}, views_allocator_t(allocator_));
        staged_norms = norms_tensor_t::try_zeros({capacity}, norms_allocator_t(allocator_));
        kept_positions = ids_tensor_t::try_zeros({capacity}, ids_allocator_t(allocator_));
        bool const sized = !selected_neighbors.empty() && !refined_neighbors.empty() && !staged_ids.empty() &&
                           !staged_vectors.empty() && !staged_norms.empty() && !kept_positions.empty();
        return sized ? status_t::success_k : status_t::bad_alloc_k;
    }

    /** @brief Bytes @c reserve_neighbors reserves for @p capacity neighbors, each tensor on its own arena line. */
    static std::size_t neighbors_memory_usage(std::size_t capacity) noexcept {
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

    /** @brief Allocator instance behind every neighbor scratch; the queues hold rebound copies. */
    [[no_unique_address]] allocator_t allocator_;
};

template <                                                                                //
    nk_dtype_t vector_dtype_, nk_kernel_kind_t kernel_kind_, nk_capability_t capability_, //
    typename wait_type_, template <typename> class atomic_ref_template_,                  //
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
    /** @brief The atomic reference owning the index's plain synchronization words - the class-local
     *      spelling of @c atomic_ref_template_, reused by cursors reading published rows. */
    template <typename word_type_>
    using atomic_ref = atomic_ref_template_<word_type_>;

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
    /** @brief Node IDs covered by the doubling phase: 2^(slab_rows_log2_k + 1) - 2^first_chunk_rows_log2_k. */
    inline static constexpr node_id_t doubling_phase_capacity_k = power_of_two<node_id_t>(slab_rows_log2_k + 1) -
                                                                  power_of_two<node_id_t>(first_chunk_rows_log2_k);
    /** @brief Upper bound on the chunk count, addressing the entire 2^32 ID space. @ref dense_ram_index_growth */
    inline static constexpr small_size_t max_chunks_k =
        doubling_phase_chunks_k +
        static_cast<small_size_t>(((u64_t(1) << 32) - doubling_phase_capacity_k) >> slab_rows_log2_k);

    /** @brief Ceiling on dispensable node IDs: the all-ones ID is the sentinel and must never be
     *      claimed, so at most 2^32 - 1 rows are ever accounted - the final chunk physically holds
     *      one extra row that stays forever unclaimed. @ref dense_ram_index_growth */
    inline static constexpr std::size_t max_node_count_k = (u64_t(1) << 32) - 1u;

    /**
     *  @brief Once-frozen geometry: vector dimensions in the high half, the neighbor
     *      connectivity in the low one; zero bits means "not reserved yet". Two fields ride
     *      one word so competing first reservations agree on both through a single cold-path
     *      compare-exchange - the only CAS in the whole index - and every reader unpacks a
     *      local acquire-loaded copy, so a torn or half-frozen geometry is unrepresentable.
     *      Connectivity stays a power of two, tiling whole cache lines, with the leading slot
     *      of every node's neighbors doubling as the lock without sacrificing a slot.
     */
    struct geometry_t {
        u64_t packed_bits = 0;

        constexpr geometry_t() noexcept = default;
        /** @brief Packs @p dimensions & @p connectivity - the inverse of the accessors. */
        constexpr geometry_t(std::size_t dimensions, connectivity_t connectivity) noexcept
            : packed_bits((static_cast<u64_t>(dimensions) << 32u) | static_cast<u64_t>(connectivity)) {}

        /** @brief Scalar dimensions per vector. */
        constexpr small_size_t dimensions() const noexcept { return static_cast<small_size_t>(packed_bits >> 32u); }
        /** @brief Neighbor slots per node. */
        constexpr small_size_t connectivity() const noexcept { return static_cast<small_size_t>(packed_bits); }
        /** @brief @c true once frozen by the first successful @c reserve(). */
        constexpr explicit operator bool() const noexcept { return packed_bits != 0; }
        constexpr bool operator==(geometry_t const&) const noexcept = default;
    };
    static_assert(std::has_unique_object_representations_v<geometry_t>,
                  "The compare-exchange compares object representations, never operator==: " //
                  "padding would make value-equal geometries spuriously lose the freeze race");
    static_assert(std::atomic_ref<geometry_t>::is_always_lock_free,
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
    std::size_t size() const noexcept {
        return atomic_ref<node_id_t>(last_visible_id_).load(std::memory_order_acquire);
    }

    /** @brief The dimensions & connectivity together, as one acquire load of the frozen word - empty
     *      until the first successful @c reserve, and never a mix of two freezes. */
    geometry_t geometry() const noexcept { return atomic_ref<geometry_t>(geometry_).load(std::memory_order_acquire); }

    /** @brief Neighbor slots per node - the full fixed capacity, none lost to bookkeeping. */
    std::size_t connectivity() const noexcept { return geometry().connectivity(); }

    /** @brief Scalar dimensions per stored vector; zero until the first reservation freezes it. */
    std::size_t dimensions() const noexcept { return geometry().dimensions(); }

    /** @brief Rows hosted without growing again - unclaimed plus claimed, the invariant tying the
     *      two counters together. The release is read first: @ref release_capacity deflates the
     *      claim before it releases, so a release seen here carries its deflation, and one missed
     *      only dips the sum. A transient dip while a grower is mid-flight is the same
     *      conservative side, which is what @c reserve compares against. */
    std::size_t capacity() const noexcept {
        std::ptrdiff_t const unclaimed_rows =
            atomic_ref<std::ptrdiff_t>(remaining_capacity_).load(std::memory_order_acquire);
        std::size_t const claimed_rows = atomic_ref<node_id_t>(last_allocated_id_).load(std::memory_order_acquire);
        return claimed_rows + static_cast<std::size_t>(unclaimed_rows > 0 ? unclaimed_rows : 0);
    }

    /** @brief Bytes the materialized chunks span - vectors, norms & neighbors - derived from
     *      the hosted capacity and the growth schedule rather than read from the chunk
     *      directories, so it stays safe under concurrent growth. A reservation materializes its
     *      chunks up front, so reserved-but-empty rows count from the moment they are granted. */
    std::size_t memory_usage() const noexcept {
        geometry_t const frozen = geometry();
        if (!frozen) return sizeof(dense_ram_index);
        std::size_t const hosted_rows = capacity();
        small_size_t chunks = 0;
        while (chunks != max_chunks_k && capacity_of_chunks(chunks) < hosted_rows) ++chunks;
        std::size_t const rows = capacity_of_chunks(chunks);
        std::size_t bytes = sizeof(dense_ram_index);
        bytes += rows * stored_row_values(frozen) * sizeof(scalar_t);
        if constexpr (!std::is_same_v<vector_norms_t, no_norms_t>) bytes += rows * sizeof(vector_norms_t);
        bytes += rows * frozen.connectivity() * sizeof(node_id_t);
        return bytes;
    }

    /** @brief The stored bytes of one node's vector - the ID-addressed read a key lookup or an
     *      export resolves through; IDs at or above @ref size() name unpublished rows. */
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
        /** @brief Neighbor IDs per neighbors-matrix row - the frozen connectivity. */
        std::size_t connectivity = 0;
    };

    /** @brief The shapes a @c save() of the current contents fills. */
    saved_shape_t saved_shape() const noexcept {
        geometry_t const frozen = geometry();
        std::size_t const count = atomic_ref<node_id_t>(last_visible_id_).load(std::memory_order_acquire);
        return {count, frozen.dimensions(), frozen.connectivity()};
    }

    /** @brief Where a save lands: two row-major planes the caller sized from @ref saved_shape,
     *      handed over at planning so no step names a destination of its own. */
    struct columns_t {
        /** @brief The vectors plane, one packed row per node. */
        bytes_span_t vectors;
        /** @brief The neighbors plane, @c connectivity IDs per node. */
        bytes_span_t neighbors;
    };

    /** @brief Where a load reads from - the read-only twin of @ref columns_t. */
    struct columns_view_t {
        /** @brief The vectors plane, one packed row per node. */
        bytes_view_t vectors;
        /** @brief The neighbors plane, @c connectivity IDs per node. */
        bytes_view_t neighbors;
    };

#pragma endregion Measures

#pragma region Reservation

    /** @brief Optional tuning knobs for @c insert(); zeros mean "derive from connectivity". */
    struct insert_config_t {
        /** @brief Breadth of the candidate pool during wiring; defaults to twice the connectivity. */
        small_size_t expansion_factor = 0;
    };

    /** @brief Requested geometry & capacity for a @c reserve() call; frozen by the first success. */
    struct reserve_capacity_t {
        /** @brief Minimal number of entries the index must be able to host. */
        std::size_t capacity = 0;
        /** @brief Neighbor slots per node; a power of two tiling whole cache lines. */
        connectivity_t connectivity = connectivity_32_k;
        /** @brief Scalar dimensions per vector; required before the first insertion. */
        std::size_t dimensions = 0;
    };

    /**
     *  @brief Ensures the index can host at least @p config.capacity entries, growing when needed,
     *      and answers the capacity now in effect.
     *
     *  Safe to call concurrently with insertions, searches, and other reservations: capacity
     *  growth is a thin wrapper over the very same grower-election protocol that powers @c insert,
     *  and the geometry (vector dimensions & neighbor connectivity) freezes exactly once through
     *  a single cold-path compare-exchange. Later calls must pass a matching geometry, or they
     *  receive @c status_t::config_mismatch_k.
     *
     *  Constructors deliberately allocate nothing, matching the design of ForkUnion's dynamic
     *  structures - every storage failure surfaces right here, in the returned status.
     */
    expected<std::size_t> reserve(reserve_capacity_t config) noexcept {

        // Freeze the geometry exactly once; competing first-reservations are decided by one CAS.
        geometry_t const desired_geometry {config.dimensions, config.connectivity};
        geometry_t expected_geometry {};
        bool const froze_now = atomic_ref<geometry_t>(geometry_).compare_exchange_strong( //
            expected_geometry, desired_geometry, std::memory_order_acq_rel, std::memory_order_acquire);
        geometry_t const frozen_geometry = froze_now ? desired_geometry : expected_geometry;
        if (frozen_geometry != desired_geometry) return {0, status_t::config_mismatch_k};

        while (true) {
            std::size_t const total_rows = capacity();
            if (total_rows >= config.capacity) return {total_rows, status_t::success_k};

            // Claim from the claimed rows up to the request, not the deficit alone: a deficit that
            // fits in the free rows is granted without a growth and released, forever. Only a claim
            // crossing the free rows elects a grower, and one growth covers it.
            std::size_t const claimed_rows = atomic_ref<node_id_t>(last_allocated_id_).load(std::memory_order_acquire);
            std::size_t const claim_rows = config.capacity - claimed_rows;
            auto const [first_row, status] = claim_capacity(claim_rows);
            if (failed(status)) return {capacity(), status};
            release_capacity(claim_rows);
        }
    }

    /** @brief The four queue reservations a walk starts from, for one expansion factor and connectivity. */
    struct queue_sizes_t {
        /** @brief The navigation pool - the expansion factor. */
        std::size_t results;
        /** @brief Every neighbor plus the newcomer while back-linking, or the admitted pool while filtering. */
        std::size_t refining_candidates;
        /** @brief Four expansions' worth, refilled by shedding rather than growth. */
        std::size_t frontier;
        /** @brief One node's neighbors per expansion - the first size a re-table doubles from. */
        std::size_t visited;
    };

    static constexpr queue_sizes_t queue_sizes_(std::size_t expansion_factor, std::size_t connectivity) noexcept {
        return {expansion_factor, max_of<std::size_t>(connectivity + 1, expansion_factor), expansion_factor * 4u,
                expansion_factor * connectivity};
    }

    static status_t reserve_queues_(dense_traversal_context_t& context, queue_sizes_t sizes) noexcept {
        if (status_t const sized = context.results.reserve(sizes.results); failed(sized)) return sized;
        if (status_t const sized = context.refining_candidates.reserve(sizes.refining_candidates); failed(sized))
            return sized;
        if (status_t const sized = context.frontier.reserve(sizes.frontier); failed(sized)) return sized;
        return context.visited.reserve(sizes.visited);
    }

    /**
     *  @brief Bytes one context reserves for @p config under the frozen geometry - the first size
     *      of every structure a walk touches. The visited set may still re-table as a walk widens,
     *      which a context kept across walks amortizes, and which the arena behind an executor
     *      grows for.
     *
     *  Mirrors @ref reserve_traversal reservation for reservation, and sits beside it so the two
     *  drift only under a reviewer's eye. Zero before the geometry freezes.
     */
    std::size_t traversal_memory_usage(insert_config_t config = {}) const noexcept {
        geometry_t const frozen = geometry();
        if (!frozen) return 0;
        std::size_t const node_connectivity = frozen.connectivity();
        std::size_t const expansion_factor = config.expansion_factor ? config.expansion_factor //
                                                                     : node_connectivity * 2;
        // One term per reservation, each occupying what the arena will hand out for it.
        using context_t = dense_traversal_context_t;
        using arena_t = arena_allocator<std::byte>;
        queue_sizes_t const sizes = queue_sizes_(expansion_factor, node_connectivity);
        std::size_t bytes = arena_t::reserved_bytes(context_t::results_array_t::memory_usage(sizes.results)) +
                            arena_t::reserved_bytes(
                                context_t::results_array_t::memory_usage(sizes.refining_candidates)) +
                            arena_t::reserved_bytes(context_t::frontier_queue_t::memory_usage(sizes.frontier)) +
                            arena_t::reserved_bytes(context_t::visited_set_t::memory_usage(sizes.visited)) +
                            context_t::neighbors_memory_usage(node_connectivity);

        if constexpr (has_tiles_k) {
            std::size_t const pool = tiles_pool_(expansion_factor, node_connectivity);
            bytes += arena_t::reserved_bytes(tiles_kit_t::scratch_bytes(pool, frozen.dimensions())) +
                     arena_t::reserved_bytes(pool * pool * sizeof(dense_distance_t));
        }
        return bytes;
    }

    /**
     *  @brief Sizes @p context 's scratch for the frozen geometry and @p config - the other fallible
     *      half of an insertion, hoistable ahead of a fan-out. Idempotent and monotone: a context
     *      already wide enough answers in six compares.
     *
     *  Every container is sized for one walk from the frozen geometry: the pools and the neighbor
     *  scratch never allocate again, the frontier sheds within its reservation, and the visited
     *  set alone may re-table as a walk widens - once per doubling per context lifetime, which a
     *  context kept across walks amortizes and the arena behind an executor grows for.
     */
    status_t reserve_traversal(dense_traversal_context_t& context, insert_config_t config = {}) noexcept {
        geometry_t const frozen = geometry();
        if (!frozen) return status_t::uninitialized_k;
        std::size_t const node_connectivity = frozen.connectivity();
        std::size_t const expansion_factor = config.expansion_factor ? config.expansion_factor //
                                                                     : node_connectivity * 2;

        // `sorted_buffer::insert(.., limit)` never grows on its own, and `growing_hash_set::set`
        // never rehashes, so these reservations are load-bearing rather than an optimization.
        if (status_t const sized = reserve_queues_(context, queue_sizes_(expansion_factor, node_connectivity));
            failed(sized))
            return sized;
        if (status_t const sized = context.reserve_neighbors(node_connectivity); failed(sized)) return sized;

        // The tile scratch sizes from the same frozen geometry and the kit's own shapes - padded to
        // whole tiles, so the padded pool doubles as the pair-matrix stride.
        if constexpr (has_tiles_k) {
            std::size_t const pool = tiles_pool_(expansion_factor, node_connectivity);
            if (failed(context.reserve_tiles(pool, tiles_kit_t::scratch_bytes(pool, frozen.dimensions()))))
                return status_t::bad_alloc_k;
        }
        return status_t::success_k;
    }

#pragma endregion Reservation

#pragma region Insertion

    /** @brief Telemetry of one @c insert() batch; a batch is atomic, so a failed status leaves no
     *      partial count to report. */
    struct [[nodiscard]] insert_result_t {
        /** @brief Index size visible after this batch fully published - the watermark. A
         *      rejected batch publishes nothing, so this reports the size it left untouched. */
        std::size_t new_size = 0;
        /** @brief Frontier candidates expanded across all per-node walks of the batch. */
        std::size_t visited_members = 0;
        /** @brief Total distance computations, with padding lanes of 4-way kernels excluded. */
        std::size_t computed_distances = 0;
        /** @brief Subset of @c computed_distances spent in insertion-side diversity selection. */
        std::size_t computed_distances_in_refines = 0;
        /** @brief Subset of @c computed_distances spent re-selecting a full neighbor set while back-linking. */
        std::size_t computed_distances_in_reverse_refines = 0;
        /** @brief Starvation restarts across the batch - a graph-connectivity health signal. */
        std::size_t restarts = 0;
        /** @brief Walks a scratch refusal ended early; their nodes published with fewer neighbors. */
        std::size_t saturated_walks = 0;
    };

    /**
     *  @brief Inserts a batch of @p new_vectors, reporting each one's assigned node ID.
     *
     *  Per claimed node the flow is: land the payload, walk the graph for candidates, wire & publish.
     *  All fallible scratch allocations happen before the ID claim - once IDs are claimed, this
     *  thread must roll forward to publication. @ref dense_ram_index_concurrency
     *
     *  @param[in] new_vectors The batch to insert, one row per vector, in the frozen geometry.
     *  @param[out] assigned_ids Caller-owned output: either empty, or exactly one slot per incoming
     *      vector, receiving that vector's assigned node ID. The index stores no keys - IDs are
     *      dense, so callers park keys in a flat ID-indexed array of the outer layer. One batch's
     *      IDs are generally @b not contiguous: identity is assigned per node, interleaved
     *      across concurrently inserting threads.
     *  @param[inout] context The calling thread's scratch & telemetry, reused across batches.
     *  @param[in] config Optional tuning knobs; zeros mean "derive from connectivity".
     */
    expected<insert_result_t> insert(vectors_view_t new_vectors, node_ids_span_t assigned_ids,
                                     dense_traversal_context_t& context, insert_config_t config = {}) noexcept {

        std::size_t const incoming_vectors_count = new_vectors.extent(0); // rows, not scalars
        if (!incoming_vectors_count) return {{size()}, status_t::success_k};
        bool const report_assigned_ids = assigned_ids.size() != 0;
        if (report_assigned_ids && assigned_ids.size() != incoming_vectors_count)
            return {{size()}, status_t::config_mismatch_k};

        // Snapshot the context's running telemetry, to report this batch's share on return.
        std::size_t const visited_members_before = context.visited_members;
        std::size_t const computed_distances_before = context.computed_distances;
        std::size_t const refines_before = context.computed_distances_in_refines;
        std::size_t const reverse_refines_before = context.computed_distances_in_reverse_refines;
        std::size_t const restarts_before = context.restarts;
        std::size_t const saturated_walks_before = context.saturated_walks;

        // Geometry is frozen by the first successful `reserve()` - a hard prerequisite here:
        // without it, neither the copy width nor the connectivity is defined.
        geometry_t const frozen = geometry();
        if (!frozen) return {{size()}, status_t::uninitialized_k};

        // Rows narrower than the frozen geometry would be over-read by the copy below, which takes
        // its width from the geometry rather than from the caller's matrix; a packed matrix
        // reports its logical width, like the geometry does.
        if (new_vectors.extent(1) != frozen.dimensions()) return {{size()}, status_t::config_mismatch_k};

        if (status_t const sized = reserve_traversal(context, config); failed(sized)) return {{size()}, sized};

        // Claim storage for the whole batch: from here on enough sentinel-filled rows exist for
        // all our nodes, and this thread must roll forward to publishing every ID it claims.
        auto [batch_claim, status] = claim(incoming_vectors_count);
        if (failed(status)) return {{size()}, status};
        insert(std::move(batch_claim), new_vectors, assigned_ids, context, config);

        insert_result_t result;
        result.new_size = size();
        result.visited_members = context.visited_members - visited_members_before;
        result.computed_distances = context.computed_distances - computed_distances_before;
        result.computed_distances_in_refines = context.computed_distances_in_refines - refines_before;
        result.computed_distances_in_reverse_refines = context.computed_distances_in_reverse_refines -
                                                       reverse_refines_before;
        result.restarts = context.restarts - restarts_before;
        result.saturated_walks = context.saturated_walks - saturated_walks_before;
        return {result, status_t::success_k};
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
     *  it found it. Rows an ID was dispensed against never come back, which is why consuming a
     *  claim and dispensing against it are the same step. @ref dense_ram_index_concurrency
     */
    struct [[nodiscard]] capacity_claim_t {
        capacity_claim_t() noexcept = default;
        /** @brief The granting index, the rows it granted and the first of them; only @ref claim
         *      ever calls it. */
        capacity_claim_t(dense_ram_index& index, std::size_t rows, node_id_t base) noexcept
            : index_(&index), rows_(rows), base_(base) {}
        capacity_claim_t(capacity_claim_t&& other) noexcept { swap(other); }
        capacity_claim_t& operator=(capacity_claim_t&& other) noexcept { return swap(other), *this; }
        capacity_claim_t(capacity_claim_t const&) = delete;
        capacity_claim_t& operator=(capacity_claim_t const&) = delete;
        /** @brief Releases whatever was never consumed; a consumed claim releases nothing. */
        ~capacity_claim_t() noexcept {
            if (index_ && rows_) index_->release_capacity(rows_);
        }

        void swap(capacity_claim_t& other) noexcept {
            std::swap(index_, other.index_);
            std::swap(rows_, other.rows_);
            std::swap(base_, other.base_);
        }

        /** @brief Rows this claim still holds; zero once an insertion has taken it. */
        std::size_t rows() const noexcept { return rows_; }
        /** @brief The first granted row - where a load plan appends. */
        node_id_t base() const noexcept { return base_; }
        explicit operator bool() const noexcept { return rows_ != 0; }

        /** @brief Marks the rows as taken, so destruction releases nothing - what an insertion
         *      spread over several threads calls once every one of its rows has claimed an ID. */
        void consume() noexcept { rows_ = 0; }

      private:
        /** @brief The granting index, reached only to release; null marks the empty state a
         *      default construction and a move-out both leave behind. */
        dense_ram_index* index_ = nullptr;
        /** @brief Rows still held; zero once consumed, so destruction releases nothing. */
        std::size_t rows_ = 0;
        /** @brief The first granted row. */
        node_id_t base_ = 0;
    };

    /** @brief Claims storage for @p count rows without assigning any identity - the step a batch
     *      spanning several indexes runs on every one of them before touching any. */
    expected<capacity_claim_t> claim(std::size_t count) noexcept {
        auto const [first_row, status] = claim_capacity(count);
        if (failed(status)) return {capacity_claim_t {}, status};
        return {capacity_claim_t {*this, count, first_row}, status_t::success_k};
    }

    /** @brief Returns @p count rows no ID was dispensed against. Deflating the claim watermark
     *      before releasing the capacity is load-bearing: a reader taking the release with its
     *      acquire sees the deflation too, so the physical-rows sum it forms is conservative and
     *      a concurrent reservation can only over-grow. The deflation releases as well: a posted,
     *      order-free subtract could land after the release.
     *
     *  The release lands only on a whole counter. A negative one has a grower inside, and a release
     *  lifting it back across zero would elect a second grower off the next claim, against the
     *  plain chunk directory; every growth ends by releasing at least its deficit, so the wait is
     *  as long as one growth. */
    void release_capacity(std::size_t count) noexcept {
        atomic_ref<node_id_t>(last_allocated_id_).sub(static_cast<node_id_t>(count), std::memory_order_release);
        std::ptrdiff_t observed = atomic_ref<std::ptrdiff_t>(remaining_capacity_).load(std::memory_order_acquire);
        while (true) {
            if (observed < 0) {
                wait_type_ {}(&remaining_capacity_, observed, 0u, fu::wait_capped_t {});
                observed = atomic_ref<std::ptrdiff_t>(remaining_capacity_).load(std::memory_order_acquire);
            }
            else if (atomic_ref<std::ptrdiff_t>(remaining_capacity_)
                         .compare_exchange_weak(observed, observed + static_cast<std::ptrdiff_t>(count),
                                                std::memory_order_release, std::memory_order_acquire))
                return;
        }
    }

    /**
     *  @brief Inserts against capacity @p claim already obtained and scratch already reserved - the
     *      half of an insertion that cannot fail, so a batch spanning several indexes commits or is
     *      refused as a whole. Consumes @p claim; telemetry accrues in @p context.
     *
     *  @p on_assigned fires per row with the assigned node ID, before the walk and before
     *  publication - so an outer layer's per-ID state, like a key column, is written ahead of the
     *  watermark turn and no reachable node can be observed without it.
     */
    template <typename callback_type_ = dummy_callback_t>
    void insert(capacity_claim_t&& claim, vectors_view_t new_vectors, node_ids_span_t assigned_ids,
                dense_traversal_context_t& context, insert_config_t config = {},
                callback_type_&& on_assigned = {}) noexcept {
        // Consumed: from here every row dispenses an ID, and a dispensed ID is never returnable.
        claim.consume();
        std::size_t const incoming_vectors_count = new_vectors.extent(0);
        bool const report_assigned_ids = assigned_ids.size() != 0;
        geometry_t const frozen = geometry();
        std::size_t const node_connectivity = frozen.connectivity();
        std::size_t const expansion_factor = config.expansion_factor ? config.expansion_factor //
                                                                     : node_connectivity * 2;

        // All slots share one contiguous packed row layout, so the copy width is loop-invariant.
        // Sub-byte types pack several dimensions per storage value - count storage bytes, not dimensions.
        std::size_t const vector_bytes = stored_row_values(frozen) * sizeof(scalar_t);

        // Go through the vectors and inject them one-by-one.
        for (small_size_t incoming_index = 0; incoming_index != incoming_vectors_count; ++incoming_index) {
            // Identity is dispensed per node, never derived from the batched capacity claim: the
            // claim only proves rows exist. One-by-one dispensing assigns IDs in global
            // wiring-start order, so each thread holds at most one unpublished ID and the
            // publication gap stays bounded by the thread count rather than the batch size.
            // The `acq_rel` order chains a happens-before edge from the growth of the hosting
            // chunk through the all-RMW dispenser to this thread.
            node_id_t const new_node_id =
                atomic_ref<node_id_t>(next_dispensed_id_).fetch_add(1, std::memory_order_acq_rel);
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

            // Snapshot the contiguous visibility watermark: every ID strictly below it is published,
            // making it the safe bound for random landmark sampling. On a virgin index there is
            // nothing to traverse - the node publishes with no neighbors, and the
            // following entries wire back to it as soon as the watermark advances.
            node_id_t const first_invisible_id =
                atomic_ref<node_id_t>(last_visible_id_).load(std::memory_order_acquire);
            // The row is already claimed, so a refused walk wires a thinner node rather than
            // abandoning an ID nothing will ever publish.
            if (first_invisible_id != 0 &&
                failed(search_around(owned_vector, new_norms, first_invisible_id, expansion_factor, context)))
                ++context.saturated_walks;

            // Select up to `connectivity()` diverse neighbors from the search results and write them
            // into our own still-unpublished slots. No locks are needed: no reader can arrive
            // before slot 0 turns non-sentinel, so relaxed stores into slots 1.. suffice, and the
            // untouched tail keeps its sentinel termination.
            selection_t& selected = context.selected_neighbors;
            std::size_t const refine_distances = select_diverse(context.results, node_connectivity, selected, context);
            context.computed_distances += refine_distances;
            context.computed_distances_in_refines += refine_distances;
            neighbors_ref_t const new_neighbors = neighbors_for(new_location);
            small_size_t const selected_count = static_cast<small_size_t>(selected.size());
            for (small_size_t slot = 1; slot < selected_count; ++slot)
                new_neighbors.store_slot(slot, selected[slot].node_id);

            // Publish the node: slot 0 receives its nearest diverse neighbor, or the node's own ID
            // when the graph offered no candidates - the "published, zero neighbors" self-loop
            // encoding, distinguishable from the all-ones "unpublished or locked" sentinel.
            // The release store is what makes the payload visible to any thread that later
            // discovers this node.
            node_id_t const first_link_id = selected_count ? selected[0].node_id : new_node_id;
            new_neighbors.unlock(first_link_id);

            // Back-link the neighbors before taking the watermark turn: once the watermark
            // passes this ID, no thread edits any published row on its behalf - the
            // settlement probe in the snapshot rests on exactly that - and the own row
            // stays unreachable below the watermark, so unlocking it above is harmless.
            // Every neighbor lock is released inside, so the spin below holds none.
            back_link_neighbors(new_node_id, owned_vector, new_norms, selected, context);

            // Take our watermark turn, strictly in dispensed order, keeping the visible prefix
            // hole-free. Each thread holds at most one unpublished ID, so this wait is bounded by
            // one wiring per other in-flight thread - the "publication convoy" is the accepted
            // cost of the CAS-free design; a helping-CAS advancement is the known alternative.
            for (node_id_t watermark = atomic_ref<node_id_t>(last_visible_id_).load(std::memory_order_acquire);
                 watermark != new_node_id;
                 watermark = atomic_ref<node_id_t>(last_visible_id_).load(std::memory_order_acquire))
                // Uncapped monitored wait: the watermark is the only word we wait on, and every
                // predecessor's publication store wakes us to re-check.
                wait_type_ {}(&last_visible_id_, watermark, 0u, fu::wait_uncapped_t {});
            atomic_ref<node_id_t>(last_visible_id_).store(new_node_id + 1, std::memory_order_release);
        }
    }

#pragma endregion Insertion

#pragma region Refinement

    /** @brief Optional tuning knobs for @c refine(); zeros mean "derive from connectivity". */
    struct refine_config_t {
        /** @brief Breadth of the re-selection pool; defaults to twice the connectivity, like every walk. */
        small_size_t expansion_factor = 0;
    };

    /** @brief Telemetry of one @c refine() sweep; a sweep that started never fails. */
    struct [[nodiscard]] refine_result_t {
        /** @brief Published nodes the sweep re-selected - the progress unit. */
        std::size_t visited_nodes = 0;
        /** @brief Nodes whose neighbors differ as a set from before - the settling signal. */
        std::size_t changed_nodes = 0;
        /** @brief Frontier candidates expanded across all per-node walks of the sweep. */
        std::size_t visited_members = 0;
        /** @brief Total distance computations, with padding lanes of 4-way kernels excluded. */
        std::size_t computed_distances = 0;
        /** @brief Subset of @c computed_distances spent in the forward diversity selection. */
        std::size_t computed_distances_in_refines = 0;
        /** @brief Subset of @c computed_distances spent re-selecting a full neighbor set while back-linking. */
        std::size_t computed_distances_in_reverse_refines = 0;
        /** @brief Starvation restarts across the sweep - a graph-connectivity health signal. */
        std::size_t restarts = 0;
        /** @brief Walks a scratch refusal ended early; their rows re-selected from a shorter pool. */
        std::size_t saturated_walks = 0;
    };

    /**
     *  @brief Re-selects the neighbors of every published node in @p nodes against the graph as
     *      it stands and back-links every one kept - an insertion's wiring, replayed over stored
     *      nodes with each node's own vector as the query.
     *
     *  Per node: the neighbors as read seed the walk, so every one already wired competes; the
     *  selection is the insertion's own; the slots are rewritten under the node's slot-0 lock;
     *  every kept neighbor is back-linked. The watermark never moves, so searches run throughout.
     *  Threads may sweep disjoint ranges concurrently: when a peer back-links into a node between
     *  its read and its rewrite, that write is dropped and the next sweep restores it. Insertions
     *  may not overlap - the same drop would orphan a fresh node.
     *  @ref dense_ram_index_concurrency
     *
     *  @param[in] nodes The ID range to sweep; IDs at or past the watermark are skipped.
     *  @param[inout] context The calling thread's scratch & telemetry, reused across sweeps.
     *  @param[in] config Optional tuning knobs; zeros mean "derive from connectivity".
     */
    expected<refine_result_t> refine(iota_view<node_id_t> nodes, dense_traversal_context_t& context,
                                     refine_config_t config = {}) noexcept {
        geometry_t const frozen = geometry();
        if (!frozen) return {{}, status_t::uninitialized_k};
        std::size_t const node_connectivity = frozen.connectivity();
        std::size_t const expansion_factor = config.expansion_factor ? config.expansion_factor //
                                                                     : node_connectivity * 2;
        if (status_t const sized = reserve_traversal(context, insert_config_t {config.expansion_factor}); failed(sized))
            return {{}, sized};

        // Snapshot the context's running telemetry, to report this sweep's share on return.
        std::size_t const visited_members_before = context.visited_members;
        std::size_t const computed_distances_before = context.computed_distances;
        std::size_t const refines_before = context.computed_distances_in_refines;
        std::size_t const reverse_refines_before = context.computed_distances_in_reverse_refines;
        std::size_t const restarts_before = context.restarts;
        std::size_t const saturated_walks_before = context.saturated_walks;

        // IDs at or past the watermark are unpublished, so the sweep ends at the lower bound.
        node_id_t const first_invisible_id = atomic_ref<node_id_t>(last_visible_id_).load(std::memory_order_acquire);
        node_id_t const bound = min_of(nodes.bound, first_invisible_id);
        refine_result_t result;
        for (node_id_t node_id = nodes.first; node_id < bound; ++node_id) {
            result.changed_nodes += refine_one_(node_id, first_invisible_id, expansion_factor, context);
            ++result.visited_nodes;
        }

        result.visited_members = context.visited_members - visited_members_before;
        result.computed_distances = context.computed_distances - computed_distances_before;
        result.computed_distances_in_refines = context.computed_distances_in_refines - refines_before;
        result.computed_distances_in_reverse_refines = context.computed_distances_in_reverse_refines -
                                                       reverse_refines_before;
        result.restarts = context.restarts - restarts_before;
        result.saturated_walks = context.saturated_walks - saturated_walks_before;
        return {result, status_t::success_k};
    }

#pragma endregion Refinement

#pragma region Health

    /** @brief Outcome of one reachability audit. */
    struct health_result_t {
        /** @brief Nodes forward-reachable from the landmark seeds - the searcher's view. */
        std::size_t reachable_nodes = 0;
        /** @brief Level-synchronous sweeps to convergence - a diameter proxy. */
        std::size_t sweeps = 0;
    };

    /**
     *  @brief One reachability audit in flight: a landmark sweep over the published nodes, one
     *      level per sweep of every step, until a level expands nothing. Read-only on the graph,
     *      so concurrent insertions merely leave fresh nodes uncounted.
     *
     *  Steps are first-chunk-sized ID ranges, disjoint, so any number of threads may @ref sweep
     *  different steps of one level at once; @ref advance_level is the level boundary, one caller.
     *  Reached bits only turn on and a node expands once, so re-sweeping until a level expands
     *  nothing converges to the exact reachable set.
     */
    struct health_plan_t {
        /** @brief One unit of a level's fan-out. */
        struct step_t {
            /** @brief The step's position in the plan - what a fan-out hands to a thread. */
            std::size_t position = 0;
            /** @brief The IDs this step sweeps. */
            iota_view<node_id_t> nodes;
        };

        using bitset_allocator_t =
            typename std::allocator_traits<candidates_allocator_t>::template rebind_alloc<u1x64_t>;
        using growing_bitset_t = growing_buffer<u1x64_t, bitset_allocator_t>;

        /** @brief Lanes one word carries - what an ID splits on. */
        inline static constexpr std::size_t lanes_k = std::numeric_limits<u64_t>::digits;

        /** @brief The watermark snapshot the audit covers; IDs strictly below it are swept. */
        node_id_t visible = 0;
        /** @brief Landmark seeds, stride-spread across the published IDs. */
        std::size_t seed_count = 0;

        /** @brief An unbound plan - a refused audit's answer; a plan owns bitsets, so it names
         *      an allocator like every owning structure here. */
        explicit health_plan_t(bitset_allocator_t const& allocator) noexcept
            : reached_(allocator), expanded_(allocator) {}
        health_plan_t(dense_ram_index const& index, node_id_t visible, std::size_t seed_count,
                      growing_bitset_t&& reached, growing_bitset_t&& expanded) noexcept
            : visible(visible), seed_count(seed_count), index_(&index), reached_(std::move(reached)),
              expanded_(std::move(expanded)) {
            restart_sweep();
        }
        health_plan_t(health_plan_t&& other) noexcept
            : visible(other.visible), seed_count(other.seed_count), index_(std::exchange(other.index_, nullptr)),
              reached_(std::move(other.reached_)), expanded_(std::move(other.expanded_)),
              expanded_in_level_(other.expanded_in_level_), sweeps_(other.sweeps_) {}
        health_plan_t(health_plan_t const&) = delete;
        health_plan_t& operator=(health_plan_t const&) = delete;

        /** @brief Step @p position of the plan; an empty step past the end. */
        step_t operator[](std::size_t position) const noexcept {
            step_t step;
            if (position >= steps().size()) return step;
            std::size_t const first = position << first_chunk_rows_log2_k;
            std::size_t const bound = min_of(first + power_of_two<std::size_t>(first_chunk_rows_log2_k),
                                             static_cast<std::size_t>(visible));
            step.position = position;
            step.nodes = {static_cast<node_id_t>(first), static_cast<node_id_t>(bound)};
            return step;
        }

        /** @brief The plan's steps, positionally; every level sweeps all of them. */
        indexed_range<health_plan_t> steps() const noexcept {
            std::size_t const count =
                index_ ? divide_round_up<std::size_t>(visible, power_of_two<std::size_t>(first_chunk_rows_log2_k)) : 0u;
            return {this, count};
        }

        /** @brief Clears the bitsets and re-marks the seeds: the next @ref sweep starts level one. */
        void restart_sweep() noexcept {
            for (u1x64_t& lanes : reached_) lanes = 0;
            for (u1x64_t& lanes : expanded_) lanes = 0;
            for (std::size_t seed = 0; seed != seed_count; ++seed)
                insert_(reached_, static_cast<node_id_t>(seed * static_cast<std::size_t>(visible) / seed_count));
            expanded_in_level_ = 0;
            sweeps_ = 0;
        }

        /**
         *  @brief Expands every reached, not yet expanded node of @p step: its neighbors below
         *      the watermark turn reached. Safe on any thread for disjoint steps - the expanded
         *      bits are the step's own, the reached bits are set through relaxed atomics.
         *  @return The nodes this call expanded.
         */
        std::size_t sweep(step_t const& step) noexcept {
            std::size_t expanded_count = 0;
            for (node_id_t node_id = step.nodes.first; node_id < step.nodes.bound; ++node_id) {
                if (!contains_(reached_, node_id) || contains_(expanded_, node_id)) continue;
                insert_exclusive_(expanded_, node_id);
                ++expanded_count;
                // Slot 0 is the lock word, acquired once and waited out like a cursor does; the
                // tail slots are relaxed reads, tolerating a re-linker's stale-but-valid mix. The
                // `const_cast` is the pre-P3323 currency for atomics living in const memory.
                node_ids_view_t const row = index_->neighbors_for(index_->chunked_location(node_id));
                node_id_t first_value =
                    atomic_ref<node_id_t>(const_cast<node_id_t&>(row[0])).load(std::memory_order_acquire);
                while (first_value == dense_node_id_unknown_k) {
                    wait_type_ {}(row.data(), dense_node_id_unknown_k, small_size_t(0), fu::wait_capped_t {});
                    first_value = atomic_ref<node_id_t>(const_cast<node_id_t&>(row[0])).load(std::memory_order_acquire);
                }
                if (first_value == node_id) continue; // Published with no neighbors.
                for (std::size_t slot = 0; slot != row.size(); ++slot) {
                    node_id_t const target =
                        slot ? atomic_ref<node_id_t>(const_cast<node_id_t&>(row[slot])).load(std::memory_order_relaxed)
                             : first_value;
                    if (target == dense_node_id_unknown_k) break;
                    if (target < visible && target != node_id) insert_(reached_, target);
                }
            }
            atomic_ref<std::size_t>(expanded_in_level_).fetch_add(expanded_count, std::memory_order_relaxed);
            return expanded_count;
        }

        /** @brief The level boundary, one caller after every step swept: counts the level and
         *      returns what it expanded; zero means the sweep converged. */
        std::size_t advance_level() noexcept {
            ++sweeps_;
            return std::exchange(expanded_in_level_, 0);
        }

        /** @brief The nodes of @p step the sweep never reached, written into @p scratch in ID
         *      order; the filled prefix is returned, truncated to the scratch's length. */
        std::span<node_id_t const> islands(step_t const& step, node_ids_span_t scratch) const noexcept {
            std::size_t count = 0;
            for (node_id_t node_id = step.nodes.first; node_id < step.nodes.bound && count != scratch.size(); ++node_id)
                if (!contains_(reached_, node_id)) scratch[count++] = node_id;
            return {scratch.data(), count};
        }

        /** @brief The landmark whose share of the id space holds @p node_id - the inverse of the
         *      stride the seeds are spread on, so it is the nearest one below it in id order.
         *      Reached by construction, since every sweep starts from it. */
        node_id_t landmark_covering(node_id_t node_id) const noexcept {
            if (!seed_count || !visible) return dense_node_id_unknown_k;
            std::size_t const seed = static_cast<std::size_t>(node_id) * seed_count / visible;
            return static_cast<node_id_t>(seed * static_cast<std::size_t>(visible) / seed_count);
        }

        /** @brief Whether @p node_id was reached - readable by any thread between levels. */
        bool reached(node_id_t node_id) const noexcept { return contains_(reached_, node_id); }

        /** @brief The report: the reached population and the levels it took. @c internal_k when a
         *      level is still pending, since the count would then be a lower bound. */
        expected<health_result_t> finish() const noexcept {
            if (!index_) return {{}, status_t::uninitialized_k};
            if (expanded_in_level_) return {{}, status_t::internal_k};
            health_result_t result;
            for (u64_t const word : reached_) result.reachable_nodes += static_cast<std::size_t>(std::popcount(word));
            result.sweeps = sweeps_;
            return {result, status_t::success_k};
        }

      private:
        static bool contains_(growing_bitset_t const& words, node_id_t node_id) noexcept {
            u1x64_t const lanes =
                atomic_ref<u1x64_t>(const_cast<u1x64_t&>(words[node_id / lanes_k])).load(std::memory_order_relaxed);
            return (lanes >> (node_id % lanes_k)) & 1u;
        }
        static void insert_(growing_bitset_t& words, node_id_t node_id) noexcept {
            atomic_ref<u1x64_t>(words[node_id / lanes_k])
                .fetch_or(u1x64_t {1} << (node_id % lanes_k), std::memory_order_relaxed);
        }
        static void insert_exclusive_(growing_bitset_t& words, node_id_t node_id) noexcept {
            words[node_id / lanes_k] |= u1x64_t {1} << (node_id % lanes_k);
        }

        /** @brief The audited index; null on a moved-from plan. */
        dense_ram_index const* index_ = nullptr;
        /** @brief One bit per published node: reached by the sweep. */
        growing_bitset_t reached_;
        /** @brief One bit per published node: its neighbors already swept. */
        growing_bitset_t expanded_;
        /** @brief Nodes the current level expanded so far, summed through @c atomic_ref. */
        std::size_t expanded_in_level_ = 0;
        /** @brief Levels completed since the last @ref restart_sweep. */
        std::size_t sweeps_ = 0;
    };

    /**
     *  @brief Opens a reachability audit over the published nodes, seeded from as many
     *      stride-spread landmarks as one search draws - the connectivity - so the reachable set
     *      is one search's reachable world. The caller sweeps every step per level and advances
     *      until a level expands nothing, then reads @c finish(). The bitsets come from
     *      @p context 's allocator, so the plan lands wherever the calling thread's scratch does.
     */
    expected<health_plan_t> plan_health(dense_traversal_context_t& context) const noexcept {
        using growing_bitset_t = typename health_plan_t::growing_bitset_t;
        using bitset_allocator_t = typename health_plan_t::bitset_allocator_t;
        node_id_t const visible = atomic_ref<node_id_t>(last_visible_id_).load(std::memory_order_acquire);
        // An empty graph is audited, not refused: it takes the same path with no words to sweep,
        // holds no steps, and finishes at zero reached - the answer every residency gives.
        bitset_allocator_t const allocator {context.allocator_};
        std::size_t const words = divide_round_up<std::size_t>(visible, health_plan_t::lanes_k);
        growing_bitset_t reached {allocator}, expanded {allocator};
        if (failed(reached.resize(words)) || failed(expanded.resize(words)))
            return {health_plan_t {allocator}, status_t::bad_alloc_k};
        std::size_t const seed_count = min_of<std::size_t>(connectivity(), visible);
        return {health_plan_t {*this, visible, seed_count, std::move(reached), std::move(expanded)},
                status_t::success_k};
    }

    /** @brief Outcome of one healing pass. */
    struct heal_result_t {
        /** @brief The converged audit the pass ended on. */
        health_result_t health;
        /** @brief Nodes the first audit found unreachable - all re-selected. */
        std::size_t islands_found = 0;
        /** @brief Nodes the re-selection left unreached, each then reconnected into a reachable pick. */
        std::size_t islands_reconnected = 0;
        /** @brief Frontier candidates expanded across the islands' re-selection walks. */
        std::size_t visited_members = 0;
        /** @brief Distances computed across the islands' re-selection. */
        std::size_t computed_distances = 0;
    };

    /**
     *  @brief One healing pass in flight over an audit: the islands the sweep leaves are
     *      re-selected through the refinement, the bitsets restart, and whatever the diversity
     *      judgment still drops is reconnected into a reachable forward pick, round by round.
     *
     *  Every phase fans out over the audit's steps on any threads; @ref reselect_islands and
     *  @ref reconnect_islands edit under the per-node locks like an insertion. The round structure
     *  is the caller's: sweep to convergence, re-select, restart and sweep, reconnect, restart and
     *  sweep, while @ref islands_left keeps shrinking. Insertions may not overlap.
     */
    struct heal_plan_t {
        using step_t = typename health_plan_t::step_t;

        /** @brief The audit the pass heals against; @ref restart_sweep it between phases. */
        health_plan_t audit;

        /** @brief An unbound pass over @p allocator - what a refused heal answers with. */
        explicit heal_plan_t(typename health_plan_t::bitset_allocator_t const& allocator) noexcept : audit(allocator) {}
        heal_plan_t(dense_ram_index& index, health_plan_t&& audit) noexcept : audit(std::move(audit)), index_(&index) {}
        heal_plan_t(heal_plan_t&& other) noexcept
            : audit(std::move(other.audit)), index_(std::exchange(other.index_, nullptr)),
              islands_found_(other.islands_found_), islands_reconnected_(other.islands_reconnected_),
              visited_members_(other.visited_members_), computed_distances_(other.computed_distances_) {}
        heal_plan_t(heal_plan_t const&) = delete;
        heal_plan_t& operator=(heal_plan_t const&) = delete;

        indexed_range<health_plan_t> steps() const noexcept { return audit.steps(); }
        std::size_t sweep(step_t const& step) noexcept { return audit.sweep(step); }
        std::size_t advance_level() noexcept { return audit.advance_level(); }
        void restart_sweep() noexcept { audit.restart_sweep(); }

        /** @brief Re-selects the neighbors of every island in @p step - the refinement's own
         *      per-node pass, so the islands gain the best forward picks the graph offers and
         *      back-link into them. Counts the islands as found on the first call per pass. */
        expected<refine_result_t> reselect_islands(step_t const& step, dense_traversal_context_t& context) noexcept {
            if (!index_) return {{}, status_t::uninitialized_k};
            std::size_t const expansion_factor = index_->connectivity() * 2;
            if (status_t const sized = index_->reserve_traversal(context); failed(sized)) return {{}, sized};
            refine_result_t result;
            std::size_t const visited_before = context.visited_members;
            std::size_t const computed_before = context.computed_distances;
            for (node_id_t node_id = step.nodes.first; node_id < step.nodes.bound; ++node_id) {
                if (audit.reached(node_id)) continue;
                result.changed_nodes += index_->refine_one_(node_id, audit.visible, expansion_factor, context);
                ++result.visited_nodes;
            }
            result.visited_members = context.visited_members - visited_before;
            result.computed_distances = context.computed_distances - computed_before;
            atomic_ref<std::size_t>(islands_found_).fetch_add(result.visited_nodes, std::memory_order_relaxed);
            atomic_ref<std::size_t>(visited_members_).fetch_add(result.visited_members, std::memory_order_relaxed);
            atomic_ref<std::size_t>(computed_distances_)
                .fetch_add(result.computed_distances, std::memory_order_relaxed);
            return {result, status_t::success_k};
        }

        /**
         *  @brief Grafts every island in @p step into its nearest reached neighbor: the
         *      island's ID lands among that neighbor's slots, in a free one when there is one and
         *      otherwise spread across the tail, so island clusters sharing one target heal in one
         *      round rather than overwriting each other. An island whose neighbors are all islands
         *      waits for a round in which one of them got connected.
         *  @return The islands this call reconnected.
         */
        std::size_t reconnect_islands(step_t const& step, dense_traversal_context_t& context) noexcept {
            if (!index_) return 0;
            std::size_t reconnected = 0;
            small_size_t const connectivity = static_cast<small_size_t>(index_->connectivity());
            small_size_t const spread = min_of<small_size_t>(16u, connectivity / 2);
            for (node_id_t island = step.nodes.first; island < step.nodes.bound; ++island) {
                if (audit.reached(island)) continue;
                neighbors_ref_t const own_neighbors = index_->neighbors_for(index_->chunked_location(island));
                small_size_t const staged_count = index_->stage_neighbors_(own_neighbors, island, context);
                node_id_t target = dense_node_id_unknown_k;
                for (small_size_t staged = 0; staged != staged_count && target == dense_node_id_unknown_k; ++staged) {
                    node_id_t const pick = context.staged_ids[staged];
                    if (pick < audit.visible && audit.reached(pick)) target = pick;
                }
                // A cluster of islands picks only its own, and no round frees any of them, so the
                // last resort is the landmark covering this id: a walk starts there by definition,
                // which is what makes it reached and what makes the edge worth its length.
                if (target == dense_node_id_unknown_k) target = audit.landmark_covering(island);
                if (target == dense_node_id_unknown_k || target == island) continue;

                // Under the target's lock: already wired, a free slot, or the spread landing.
                neighbors_ref_t const target_neighbors = index_->neighbors_for(index_->chunked_location(target));
                node_id_t const displaced = target_neighbors.lock();
                if (displaced == target) { // Published with no neighbors: the island becomes the only one.
                    target_neighbors.unlock(island);
                    ++reconnected;
                    continue;
                }
                small_size_t landing = connectivity - 1 - static_cast<small_size_t>(island % spread);
                bool present = displaced == island;
                for (small_size_t slot = 1; slot != connectivity && !present; ++slot) {
                    node_id_t const resident = target_neighbors.load_slot(slot);
                    present = resident == island;
                    if (resident == dense_node_id_unknown_k) {
                        landing = slot;
                        break;
                    }
                }
                if (!present) {
                    target_neighbors.store_slot(landing, island);
                    ++reconnected;
                }
                target_neighbors.unlock(displaced);
            }
            atomic_ref<std::size_t>(islands_reconnected_).fetch_add(reconnected, std::memory_order_relaxed);
            return reconnected;
        }

        /** @brief Islands left after the last converged sweep - the caller's stopping test. */
        std::size_t islands_left() const noexcept {
            auto const [health, status] = audit.finish();
            return succeeded(status) ? static_cast<std::size_t>(audit.visible) - health.reachable_nodes : 0;
        }

        /** @brief The report of the whole pass, the converged audit inside it. */
        expected<heal_result_t> finish() const noexcept {
            if (!index_) return {{}, status_t::uninitialized_k};
            auto const [health, status] = audit.finish();
            if (failed(status)) return {{}, status};
            heal_result_t result;
            result.health = health;
            result.islands_found = islands_found_;
            result.islands_reconnected = islands_reconnected_;
            result.visited_members = visited_members_;
            result.computed_distances = computed_distances_;
            return {result, status_t::success_k};
        }

      private:
        /** @brief The healed index; null on a moved-from plan. */
        dense_ram_index* index_ = nullptr;
        /** @brief Islands re-selected across the pass, summed through @c atomic_ref. */
        std::size_t islands_found_ = 0;
        /** @brief Islands reconnected across the pass, summed through @c atomic_ref. */
        std::size_t islands_reconnected_ = 0;
        /** @brief Frontier candidates the re-selection walks expanded, summed through @c atomic_ref. */
        std::size_t visited_members_ = 0;
        /** @brief Distances the re-selection computed, summed through @c atomic_ref. */
        std::size_t computed_distances_ = 0;
    };

    /** @brief Opens a healing pass over a fresh audit - @ref plan_health with the same seeds. */
    expected<heal_plan_t> plan_heal(dense_traversal_context_t& context) noexcept {
        auto [audit, status] = plan_health(context);
        if (failed(status))
            return {heal_plan_t {typename health_plan_t::bitset_allocator_t {context.allocator_}}, status};
        return {heal_plan_t {*this, std::move(audit)}, status_t::success_k};
    }

#pragma endregion Health

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

    /** @brief Telemetry of one @c search() call; matches stay in the caller's context. */
    struct [[nodiscard]] search_result_t {
        /** @brief Matches left in `context.results`, ascending by distance to the query. */
        std::size_t count = 0;
        /** @brief Frontier candidates expanded during the walk. */
        std::size_t visited_members = 0;
        /** @brief Total distance computations, with padding lanes of 4-way kernels excluded. */
        std::size_t computed_distances = 0;
        /** @brief Starvation restarts - a graph-connectivity health signal. */
        std::size_t restarts = 0;
        /** @brief One when a scratch refusal ended the walk early, leaving the ranking short. */
        std::size_t saturated_walks = 0;
    };

    /**
     *  @brief Finds the nearest published neighbors of @p query_vector - the caller reranks or
     *      trims the prefix it needs.
     *
     *  Under the default @c concurrent_k contract it is safe to call concurrently with
     *  insertions and other searches: the traversal reads only nodes below the visibility
     *  watermark, acquire-loads every slot-0 lock before scanning a node's neighbors, retries
     *  transiently locked nodes, and never writes to the graph. Freshly inserted nodes become
     *  reachable the moment their publisher advances the watermark; back-linked IDs above this query's
     *  watermark snapshot are skipped by the translation-frontier invariant, and picked up by
     *  any later query with a fresher snapshot. Under @c immutable_k - or stronger - the caller
     *  certifies no mutators and the traversal runs atomics-free. @ref dense_ram_index_concurrency
     *
     *  @param[in] query_vector The query, in the index's own dtype & frozen dimensionality.
     *  @param[inout] context The calling thread's scratch & telemetry: the matches end up in
     *      @c context.results, ascending by distance - the nearest first.
     *  @param[in] config Optional tuning knobs; zeros mean "derive from connectivity".
     */
    expected<search_result_t> search(vector_view_t query_vector, dense_traversal_context_t& context,
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
     *  @param[in] predicate The admission filter, called with published node IDs only.
     *  @param[inout] context The calling thread's scratch & telemetry: the matches end up in
     *      @c context.results, ascending by distance - the nearest first.
     *  @param[in] config Optional tuning knobs; zeros mean "derive from connectivity".
     */
    template <typename predicate_type_>
    expected<search_result_t> search_if(vector_view_t query_vector, predicate_type_&& predicate,
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
    expected<search_result_t> search_typed(vector_view_t query_vector, dense_traversal_context_t& context,
                                           search_config_t config = {},
                                           predicate_type_&& predicate = predicate_type_ {}) noexcept {
        constexpr bool predicated_k = !std::is_same_v<std::remove_cvref_t<predicate_type_>, dummy_predicate_t>;

        std::size_t const expansion_factor = config.expansion_factor ? config.expansion_factor //
                                                                     : connectivity() * 2;

        // Same reservations as in `insert`, the first size of each structure; the visited set may
        // still re-table as the walk widens. A predicated walk also owns the admitted pool - the
        // re-selection buffer, idle during any search - which the same sizes cover.
        if (status_t const sized = reserve_queues_(context, queue_sizes_(expansion_factor, connectivity()));
            failed(sized))
            return {{}, sized};
        if constexpr (predicated_k) context.refining_candidates.clear();
        context.results.clear();
        context.frontier.clear();
        context.visited.clear();

        // Snapshot the watermark once: everything strictly below it is published and readable.
        // An empty index legitimately yields an empty result set. A mutation-free contract needs
        // no acquire - the caller's own synchronization already ordered every publication first.
        constexpr std::memory_order watermark_order_k = synchronization_ == synchronization_t::concurrent_k
                                                            ? std::memory_order_acquire
                                                            : std::memory_order_relaxed;
        node_id_t const first_invisible_id = atomic_ref<node_id_t>(last_visible_id_).load(watermark_order_k);
        if (first_invisible_id == 0) return {{}, status_t::success_k};

        // Snapshot the context's running telemetry, to report this query's share on return.
        std::size_t const visited_members_before = context.visited_members;
        std::size_t const computed_distances_before = context.computed_distances;
        std::size_t const restarts_before = context.restarts;
        std::size_t const saturated_walks_before = context.saturated_walks;

        vector_norms_t const query_norms = dense_norms(query_vector);
        status_t const walked = search_around<synchronization_>(query_vector, query_norms, first_invisible_id,
                                                                expansion_factor, context, //
                                                                std::forward<predicate_type_>(predicate));

        // The result contract is one pool: matches always land in `context.results`, so a
        // predicated walk swaps its admitted pool into place - both are the same buffer type,
        // and the navigation leftovers get cleared by whichever call reuses that scratch next.
        if constexpr (predicated_k) std::swap(context.results, context.refining_candidates);

        // A refused walk still ranked whatever it reached; the counter says the ranking is short
        // because scratch ran out, not because the graph had nothing better.
        if (failed(walked)) ++context.saturated_walks;
        search_result_t result;
        result.count = context.results.size();
        result.visited_members = context.visited_members - visited_members_before;
        result.computed_distances = context.computed_distances - computed_distances_before;
        result.restarts = context.restarts - restarts_before;
        result.saturated_walks = context.saturated_walks - saturated_walks_before;
        return {result, status_t::success_k};
    }

#pragma endregion Search

#pragma region Checkpoints

    /**
     *  @brief One save in flight: the watermark it took, the contract it runs under, and the storage
     *      chunks it hands out to any number of threads. Vectors and sentinel-padded neighbor rows
     *      are what a snapshot holds; norms re-derive on load.
     *
     *  A checkpointing reader, not a mutator: concurrent searches keep running on every arm.
     *  Under @c concurrent_k the neighbor words are read through relaxed atomics and @ref finish
     *  probes for settlement across the whole plan, since a later insert back-linking itself
     *  edits an earlier row; the certified arms read plain, and a violated certification still
     *  surfaces as @c contended_k instead of a silently torn snapshot.
     */
    struct [[nodiscard]] save_plan_t {
        /** @brief Where the rows land, vetted against the watermark at planning. */
        columns_t columns {};
        /** @brief What an ID under the watermark moves by - the row this shard starts at. */
        node_id_t id_shift = 0;
        /** @brief The contract the caller certified. */
        synchronization_t synchronization = synchronization_t::concurrent_k;
        /** @brief Rows under the watermark at planning. */
        node_id_t visible = 0;
        /** @brief The dispenser at planning; equal to @c visible, else no plan was made. */
        node_id_t dispensed = 0;
        /** @brief The allocation counter at planning - the growth sensor the probe re-reads. */
        node_id_t allocated = 0;
        /** @brief Rows whose vectors landed, tallied through @c atomic_ref. */
        std::size_t covered_vectors = 0;
        /** @brief Rows whose neighbors landed, tallied through @c atomic_ref. */
        std::size_t covered_neighbors = 0;

        save_plan_t() noexcept = default;
        /** @brief The index being snapshotted into @p columns under @p synchronization, its IDs
         *      shifted by @p id_shift; only @ref plan_save calls it. */
        save_plan_t(dense_ram_index const& index, columns_t columns, synchronization_t synchronization,
                    node_id_t id_shift) noexcept
            : columns(columns), id_shift(id_shift), synchronization(synchronization), index_(&index) {}
        save_plan_t(save_plan_t&& other) noexcept
            : columns(other.columns), id_shift(other.id_shift), synchronization(other.synchronization),
              visible(other.visible), dispensed(other.dispensed), allocated(other.allocated),
              covered_vectors(other.covered_vectors), covered_neighbors(other.covered_neighbors),
              index_(std::exchange(other.index_, nullptr)) {}
        save_plan_t& operator=(save_plan_t&&) = delete;
        // The tallies are this plan's own: a copy would tally into a twin `finish` never reads.
        save_plan_t(save_plan_t const&) = delete;
        save_plan_t& operator=(save_plan_t const&) = delete;
        /** @brief A save owns nothing - no rows were claimed and no lock taken - so a dying plan
         *      leaves the index as it found it. */
        ~save_plan_t() noexcept = default;

        /** @brief One step of the plan: one storage chunk's share of the rows under the watermark. */
        struct step_t {
            /** @brief The step's position in the plan - what a fan-out hands to a thread. */
            std::size_t position = 0;
            /** @brief Where the rows sit in the index, in its own ID space. */
            iota_view<node_id_t> index_rows;
            /** @brief Where they sit in the caller's buffer, counted from its first row. */
            iota_view<node_id_t> source_rows;
            /** @brief Bytes this step's vector rows occupy inside the plan's vectors column. */
            std::size_t vectors_bytes = 0;
            /** @brief Bytes this step's neighbor rows occupy inside the plan's neighbors column. */
            std::size_t neighbors_bytes = 0;
        };

        /** @brief Step @p position of the plan; an empty one past the last. */
        step_t operator[](std::size_t position) const noexcept {
            step_t step;
            if (position >= steps().size()) return step;
            std::size_t const first = capacity_of_chunks(static_cast<small_size_t>(position));
            std::size_t const bound = min_of(capacity_of_chunks(static_cast<small_size_t>(position + 1)),
                                             static_cast<std::size_t>(visible));
            geometry_t const frozen = index_->geometry();
            step.position = position;
            step.index_rows = {static_cast<node_id_t>(first), static_cast<node_id_t>(bound)};
            step.source_rows = step.index_rows;
            step.vectors_bytes = step.index_rows.size() * stored_row_values(frozen) * sizeof(scalar_t);
            step.neighbors_bytes = step.index_rows.size() * frozen.connectivity() * sizeof(node_id_t);
            return step;
        }

        /** @brief The steps the save spans - one per storage chunk it reaches; none once spent. */
        indexed_range<save_plan_t> steps() const noexcept {
            std::size_t const count =
                index_ && visible ? index_->chunked_location(static_cast<node_id_t>(visible - 1)).chunk_index + 1u : 0u;
            return {this, count};
        }

        /** @brief Copies step @p step 's rows into the plan's own columns: the vectors straight
         *      from chunk memory, since published vectors never change, and the neighbors rebased
         *      through @p scratch - every ID under the watermark shifted by @c id_shift, every other
         *      folded to the sentinel.
         *
         *  @param[in] step The step to copy, as @ref steps hands them out.
         *  @param[inout] scratch One step of IDs the caller reserved, where the rebase happens.
         */
        status_t save(step_t const& step, nk::matrix_span<node_id_t> scratch) noexcept {
            if (!index_) return status_t::uninitialized_k;
            iota_view<node_id_t> const rows = step.index_rows;
            if (rows.empty()) return status_t::config_mismatch_k;
            geometry_t const frozen = index_->geometry();
            std::size_t const row_values = stored_row_values(frozen);
            std::size_t const connectivity = frozen.connectivity();
            std::size_t const step_scalars = rows.size() * row_values;
            std::size_t const step_slots = rows.size() * connectivity;
            std::size_t const vectors_first = step.source_rows.first * row_values * sizeof(scalar_t);
            std::size_t const neighbors_first = step.source_rows.first * connectivity * sizeof(node_id_t);
            if (step.vectors_bytes != step_scalars * sizeof(scalar_t) ||
                step.neighbors_bytes != step_slots * sizeof(node_id_t))
                return status_t::config_mismatch_k;
            if (vectors_first > columns.vectors.size() || neighbors_first > columns.neighbors.size() ||
                columns.vectors.size() - vectors_first < step.vectors_bytes ||
                columns.neighbors.size() - neighbors_first < step.neighbors_bytes)
                return status_t::config_mismatch_k;
            if (scratch.extent(0) < rows.size() || scratch.extent(1) != connectivity || !scratch.is_contiguous())
                return status_t::config_mismatch_k;

            chunked_location_t const location = index_->chunked_location(rows.first);
            vectors_view_t const stored = index_->vectors_chunks_[location.chunk_index].view()(
                nk::range {location.index_in_chunk, location.index_in_chunk + rows.size()}, nk::slice);
            [[maybe_unused]] status_t const copied_vectors = copy_bytes(
                columns.vectors.subspan(vectors_first, step.vectors_bytes), bytes_of(stored.data(), step_scalars));
            atomic_ref<std::size_t>(covered_vectors).add(rows.size(), std::memory_order_release);

            node_id_t const* source =
                index_->neighbors_chunks_[location.chunk_index].view().row(location.index_in_chunk).data();
            node_id_t* staged = scratch.data();
            // The `const_cast` is the pre-P3323 currency for atomics living in const memory.
            if (synchronization == synchronization_t::concurrent_k)
                for (std::size_t word = 0; word != step_slots; ++word)
                    staged[word] =
                        atomic_ref<node_id_t>(const_cast<node_id_t&>(source[word])).load(std::memory_order_relaxed);
            else std::copy_n(source, step_slots, staged);
            for (std::size_t word = 0; word != step_slots; ++word)
                staged[word] = staged[word] < visible ? static_cast<node_id_t>(staged[word] + id_shift)
                                                      : dense_node_id_unknown_k;
            [[maybe_unused]] status_t const copied_neighbors = copy_bytes(
                columns.neighbors.subspan(neighbors_first, step.neighbors_bytes), bytes_of(staged, step_slots));
            atomic_ref<std::size_t>(covered_neighbors).add(rows.size(), std::memory_order_release);
            return status_t::success_k;
        }

        /** @brief Pronounces the snapshot complete: every chunk's vectors and neighbors landed,
         *      and under @c concurrent_k the counters rested across the whole plan. A success
         *      spends the plan; a refusal leaves it whole, so the missing steps can still land. */
        status_t finish() noexcept {
            if (!index_) return status_t::uninitialized_k;
            std::size_t const nodes = static_cast<std::size_t>(visible);
            if (atomic_ref<std::size_t>(covered_vectors).load(std::memory_order_acquire) != nodes ||
                atomic_ref<std::size_t>(covered_neighbors).load(std::memory_order_acquire) != nodes)
                return status_t::config_mismatch_k;
            if (synchronization == synchronization_t::concurrent_k) {
                // Orders the staged reads before the re-reads below; pairs with the release fence
                // heading @c back_link_neighbors, so any row edit the copies overlapped surfaces here.
                // The non-monotone allocation counter rides along as a bonus growth sensor.
                std::atomic_thread_fence(std::memory_order_acquire);
                bool const settled =
                    atomic_ref<node_id_t>(index_->last_visible_id_).load(std::memory_order_relaxed) == visible &&
                    atomic_ref<node_id_t>(index_->next_dispensed_id_).load(std::memory_order_relaxed) == dispensed &&
                    atomic_ref<node_id_t>(index_->last_allocated_id_).load(std::memory_order_relaxed) == allocated;
                if (!settled) return status_t::contended_k;
            }
            index_ = nullptr;
            return status_t::success_k;
        }

      private:
        /** @brief The index being snapshotted; null in the empty plan a refused @ref plan_save
         *      returns, and once @c finish succeeded. */
        dense_ram_index const* index_ = nullptr;
    };

    /**
     *  @brief Opens a save into @p columns under @p synchronization: takes the watermark and, under
     *      @c concurrent_k, refuses while an insert is mid-flight - the settlement probe, trailing
     *      counter first, since the pair is monotone with visible ≤ dispensed at every instant.
     *      @c exclusive_k reads like @c immutable_k. Columns too short for the rows under it are refused
     *      here, so no step ever writes past them.
     *
     *  @param[out] columns Where the rows land, sized from @ref saved_shape.
     *  @param[in] synchronization The contract the caller certifies.
     *  @param[in] id_shift What an ID under the watermark moves by - the row this shard starts at.
     */
    expected<save_plan_t> plan_save(columns_t columns, synchronization_t synchronization,
                                    node_id_t id_shift = 0) const noexcept {
        geometry_t const frozen = geometry();
        if (!frozen) return {{}, status_t::config_mismatch_k};
        save_plan_t plan {*this, columns, synchronization, id_shift};
        plan.visible = atomic_ref<node_id_t>(last_visible_id_).load(std::memory_order_acquire);
        plan.dispensed = atomic_ref<node_id_t>(next_dispensed_id_).load(std::memory_order_acquire);
        plan.allocated = atomic_ref<node_id_t>(last_allocated_id_).load(std::memory_order_acquire);
        if (plan.visible != plan.dispensed) return {{}, status_t::contended_k};
        std::size_t const rows = static_cast<std::size_t>(plan.visible);
        if (columns.vectors.size() < rows * stored_row_values(frozen) * sizeof(scalar_t) ||
            columns.neighbors.size() < rows * frozen.connectivity() * sizeof(node_id_t))
            return {{}, status_t::config_mismatch_k};
        return {std::move(plan), status_t::success_k};
    }

    /** @brief Serializes every row under the watermark into @p columns on the calling thread - the one-shot over
     *      @ref plan_save for a caller with no threads to fan the steps out over. */
    status_t save(columns_t columns, synchronization_t synchronization,
                  nk::matrix_span<node_id_t> scratch) const noexcept {
        auto [plan, status] = plan_save(columns, synchronization);
        if (failed(status)) return status;
        for (typename save_plan_t::step_t const step : plan.steps())
            if (status_t const saved = plan.save(step, scratch); failed(saved)) return saved;
        return plan.finish();
    }

    /**
     *  @brief One load in flight: the rows claimed after the watermark and the storage
     *      chunks they span, filled by any number of threads; @ref finish publishes them.
     *
     *  Demands sole ownership of an index whose geometry is already frozen. A source's IDs are
     *  local to the rows it delivers, so every chunk shifts them onto the appended range and
     *  folds every other ID onto the sentinel; norms re-derive as the vectors land. Repeated
     *  plans append, which is how several parts land in one shard. The plan owns the claim its
     *  rows ride in: a plan that dies before @c finish releases them, so the next plan appends
     *  at the same base and the rows it half-filled are never published.
     */
    struct [[nodiscard]] load_plan_t {
        /** @brief Where the rows come from, vetted against them at planning. */
        columns_view_t columns {};
        /** @brief The ID space the source's rows speak. */
        iota_view<node_id_t> source_range {};
        /** @brief The first appended ID - the row the claim landed at. */
        node_id_t base = 0;
        /** @brief Rows the plan appends. */
        std::size_t rows = 0;
        /** @brief Rows whose vectors landed, tallied through @c atomic_ref. */
        std::size_t covered_vectors = 0;
        /** @brief Rows whose neighbors landed, tallied through @c atomic_ref. */
        std::size_t covered_neighbors = 0;

        load_plan_t() noexcept = default;
        /** @brief The index appended to at @p base, the @p columns the rows come from under
         *      @p source_range, and the claim they ride in - empty for a plan of no rows; only
         *      @ref plan_load calls it. */
        load_plan_t(dense_ram_index& index, columns_view_t columns, iota_view<node_id_t> source_range, node_id_t base,
                    capacity_claim_t&& claim) noexcept
            : columns(columns), source_range(source_range), base(base), rows(claim.rows()), index_(&index),
              claim_(std::move(claim)) {}
        load_plan_t(load_plan_t&& other) noexcept
            : columns(other.columns), source_range(other.source_range), base(other.base),
              rows(std::exchange(other.rows, 0)), covered_vectors(other.covered_vectors),
              covered_neighbors(other.covered_neighbors), index_(std::exchange(other.index_, nullptr)),
              claim_(std::move(other.claim_)) {}
        load_plan_t& operator=(load_plan_t&&) = delete;
        // The tallies are this plan's own: a copy would tally into a twin `finish` never reads.
        load_plan_t(load_plan_t const&) = delete;
        load_plan_t& operator=(load_plan_t const&) = delete;
        /** @brief The claim's own death releases the rows, unless @c finish consumed them. */
        ~load_plan_t() noexcept = default;

        /** @brief One step of the plan: one storage chunk's share of the appended rows. */
        struct step_t {
            /** @brief The step's position in the plan - what a fan-out hands to a thread. */
            std::size_t position = 0;
            /** @brief Where the rows land in the index, in its own ID space, from @c base. */
            iota_view<node_id_t> index_rows;
            /** @brief Where they come from in the caller's buffer, from its first row. */
            iota_view<node_id_t> source_rows;
            /** @brief Bytes this step's vector rows occupy inside the plan's vectors column. */
            std::size_t vectors_bytes = 0;
            /** @brief Bytes this step's neighbor rows occupy inside the plan's neighbors column. */
            std::size_t neighbors_bytes = 0;
        };

        /** @brief Step @p position of the plan; an empty one past the last. */
        step_t operator[](std::size_t position) const noexcept {
            step_t step;
            if (position >= steps().size()) return step;
            small_size_t const first_chunk = index_->chunked_location(base).chunk_index;
            std::size_t const first = max_of(capacity_of_chunks(static_cast<small_size_t>(first_chunk + position)),
                                             static_cast<std::size_t>(base));
            std::size_t const bound = min_of(capacity_of_chunks(static_cast<small_size_t>(first_chunk + position + 1)),
                                             static_cast<std::size_t>(base) + rows);
            geometry_t const frozen = index_->geometry();
            step.position = position;
            step.index_rows = {static_cast<node_id_t>(first), static_cast<node_id_t>(bound)};
            step.source_rows = {static_cast<node_id_t>(first - base), static_cast<node_id_t>(bound - base)};
            step.vectors_bytes = step.index_rows.size() * stored_row_values(frozen) * sizeof(scalar_t);
            step.neighbors_bytes = step.index_rows.size() * frozen.connectivity() * sizeof(node_id_t);
            return step;
        }

        /** @brief The steps the append spans - one per storage chunk it reaches; none once spent. */
        indexed_range<load_plan_t> steps() const noexcept {
            std::size_t const count =
                index_ && rows ? index_->chunked_location(static_cast<node_id_t>(base + rows - 1)).chunk_index -
                                     index_->chunked_location(base).chunk_index + 1u
                               : 0u;
            return {this, count};
        }

        /** @brief Copies step @p step 's rows out of the plan's own columns straight into chunk
         *      memory: the vectors with their norms derived as they land, then the neighbors with
         *      every ID inside @c source_range folded onto the appended range and every other onto
         *      the sentinel.
         *
         *  @param[in] step The step to fill, as @ref steps hands them out.
         */
        status_t load(step_t const& step) noexcept { return load(step, columns, source_range); }

        /** @brief The same copy out of @p columns under @p source_range rather than the plan's
         *      own - what a sharded plan drives, one part's slice at a time, its step naming the
         *      rows in that part's row and ID spaces.
         *
         *  @param[in] step The step to fill; its source rows address @p columns.
         *  @param[in] columns The part's own planes.
         *  @param[in] source_range The ID space the part's rows speak.
         */
        status_t load(step_t const& step, columns_view_t columns, iota_view<node_id_t> source_range) noexcept {
            if (!index_) return status_t::uninitialized_k;
            iota_view<node_id_t> const range = step.index_rows;
            if (range.empty()) return status_t::config_mismatch_k;
            geometry_t const frozen = index_->geometry();
            std::size_t const row_values = stored_row_values(frozen);
            std::size_t const connectivity = frozen.connectivity();
            std::size_t const step_scalars = range.size() * row_values;
            std::size_t const step_slots = range.size() * connectivity;
            std::size_t const vectors_first = step.source_rows.first * row_values * sizeof(scalar_t);
            std::size_t const neighbors_first = step.source_rows.first * connectivity * sizeof(node_id_t);
            if (step.vectors_bytes != step_scalars * sizeof(scalar_t) ||
                step.neighbors_bytes != step_slots * sizeof(node_id_t))
                return status_t::config_mismatch_k;
            if (vectors_first > columns.vectors.size() || neighbors_first > columns.neighbors.size() ||
                columns.vectors.size() - vectors_first < step.vectors_bytes ||
                columns.neighbors.size() - neighbors_first < step.neighbors_bytes)
                return status_t::config_mismatch_k;

            chunked_location_t const location = index_->chunked_location(range.first);
            nk::matrix_span<scalar_t> const vectors = index_->vectors_chunks_[location.chunk_index].span()(
                nk::range {location.index_in_chunk, location.index_in_chunk + range.size()}, nk::slice);
            [[maybe_unused]] status_t const copied_vectors = copy_bytes(
                bytes_of(vectors.data(), step_scalars), columns.vectors.subspan(vectors_first, step.vectors_bytes));
            // Norms come from the same kernels the insertion path uses.
            if constexpr (!std::is_same_v<vector_norms_t, no_norms_t>)
                for (std::size_t offset_in_chunk = 0; offset_in_chunk != range.size(); ++offset_in_chunk) {
                    chunked_location_t const row {location.chunk_index,
                                                  static_cast<small_size_t>(location.index_in_chunk + offset_in_chunk)};
                    index_->norm_ref_for(row) = dense_norms(index_->vector_for(row));
                }
            atomic_ref<std::size_t>(covered_vectors).add(range.size(), std::memory_order_release);

            nk::matrix_span<node_id_t> const neighbors = index_->neighbors_chunks_[location.chunk_index].span()(
                nk::range {location.index_in_chunk, location.index_in_chunk + range.size()}, nk::slice);
            [[maybe_unused]] status_t const copied_neighbors = copy_bytes(
                bytes_of(neighbors.data(), step_slots),
                columns.neighbors.subspan(neighbors_first, step.neighbors_bytes));
            std::span<node_id_t> const landed {neighbors.data(), step_slots};
            // The part's rows land at this step's own index rows, so an ID's shift follows from
            // the step rather than from the plan's base - the two agree for a one-part plan.
            node_id_t const shift = static_cast<node_id_t>(range.first - step.source_rows.first + source_range.first);
            rebase_neighbors(landed, source_range, shift, dense_node_id_unknown_k);
            compact_neighbors(landed, connectivity, range.first, dense_node_id_unknown_k);
            atomic_ref<std::size_t>(covered_neighbors).add(range.size(), std::memory_order_release);
            return status_t::success_k;
        }

        /** @brief Publishes the appended rows once every chunk's vectors and neighbors landed:
         *      the claim consumed first - a published row never comes back - then identity, then
         *      the watermark last with a release, as every publication here. The identity
         *      releases too: a posted, order-free add could land after the watermark. A success
         *      spends the plan, so a second call answers @c uninitialized_k instead of publishing
         *      twice; a refusal leaves it whole, so the missing steps can still land. */
        status_t finish() noexcept {
            if (!index_) return status_t::uninitialized_k;
            if (atomic_ref<std::size_t>(covered_vectors).load(std::memory_order_acquire) != rows ||
                atomic_ref<std::size_t>(covered_neighbors).load(std::memory_order_acquire) != rows)
                return status_t::config_mismatch_k;
            if (rows) {
                claim_.consume();
                atomic_ref<node_id_t>(index_->next_dispensed_id_)
                    .add(static_cast<node_id_t>(rows), std::memory_order_release);
                atomic_ref<node_id_t>(index_->last_visible_id_)
                    .store(static_cast<node_id_t>(base + rows), std::memory_order_release);
            }
            index_ = nullptr;
            return status_t::success_k;
        }

      private:
        /** @brief The index appended to; null in the empty plan a refused @ref plan_load hands
         *      back, and once @c finish published. */
        dense_ram_index* index_ = nullptr;
        /** @brief The rows this plan appends, released by their own death unless consumed. */
        capacity_claim_t claim_;
    };

    /**
     *  @brief Opens a load of @p rows from @p columns after the watermark: claims them
     *      through the ordinary capacity protocol, so a preceding @c reserve is consumed rather
     *      than duplicated and chunks grow only on a genuine deficit. Refuses an unfrozen
     *      geometry, columns shorter than the rows, an insert mid-flight, and rows past the ID
     *      space.
     *
     *  @param[in] columns Where the rows come from, both planes sized for @p rows.
     *  @param[in] rows Rows to append.
     *  @param[in] source_range The ID space the source's rows speak.
     */
    expected<load_plan_t> plan_load(columns_view_t columns, std::size_t rows,
                                    iota_view<node_id_t> source_range) noexcept {
        geometry_t const frozen = geometry();
        if (!frozen) return {{}, status_t::config_mismatch_k};
        if (columns.vectors.size() < rows * stored_row_values(frozen) * sizeof(scalar_t) ||
            columns.neighbors.size() < rows * frozen.connectivity() * sizeof(node_id_t))
            return {{}, status_t::config_mismatch_k};
        auto [plan, status] = plan_load(rows);
        if (failed(status)) return {{}, status};
        plan.columns = columns;
        plan.source_range = source_range;
        return {std::move(plan), status};
    }

    /** @brief Opens a load of @p rows the caller sources per step - what a sharded plan drives,
     *      its parts arriving one slice at a time rather than as one pair of columns. */
    expected<load_plan_t> plan_load(std::size_t rows) noexcept {
        if (!geometry()) return {{}, status_t::config_mismatch_k};
        node_id_t const visible = atomic_ref<node_id_t>(last_visible_id_).load(std::memory_order_acquire);
        if (!rows) return {load_plan_t {*this, {}, {}, visible, {}}, status_t::success_k};
        if (visible != atomic_ref<node_id_t>(next_dispensed_id_).load(std::memory_order_acquire))
            return {{}, status_t::contended_k};
        if (rows > max_node_count_k - visible) return {{}, status_t::overflow_k};
        auto [capacity_claim, status] = claim(rows);
        if (failed(status)) return {{}, status};
        if (capacity_claim.base() != visible) return {{}, status_t::contended_k}; // The claim dies and releases.
        return {load_plan_t {*this, {}, {}, visible, std::move(capacity_claim)}, status_t::success_k};
    }

    /** @brief Appends the whole source on the calling thread - the one-shot over @ref plan_load
     *      for a caller with no threads to fan the steps out over. */
    status_t load(columns_view_t columns, std::size_t rows, iota_view<node_id_t> source_range) noexcept {
        auto [plan, status] = plan_load(columns, rows, source_range);
        if (failed(status)) return status;
        for (typename load_plan_t::step_t const step : plan.steps())
            if (status_t const loaded = plan.load(step); failed(loaded)) return loaded;
        return plan.finish();
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
        // covers IDs [2^(first_chunk_rows_log2_k + j) - 2^first_chunk_rows_log2_k, twice that upper bound).
        if (node_id < doubling_phase_capacity_k) {
            // Biasing by the first chunk's size makes the leading bit of the ID encode the chunk index.
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
     *  @brief Typed view over one node's neighbors - the only legal way to touch slots that
     *      concurrent scanners may be reading. Copyable value type with reference semantics.
     *
     *  Slot 0 is the combined lock & first-neighbor word: the all-ones @c dense_node_id_unknown_k
     *  sentinel while unpublished, empty, or locked for editing - otherwise the plain ID of the
     *  first neighbor. IDs are stored raw and no row bytes are spent on bookkeeping, so the
     *  neighbor capacity stays a power of two, tiling whole cache lines. Slots 1.. are accessed
     *  as relaxed atomics on both sides, making the "stale-but-valid mix" a scanner can observe
     *  during concurrent re-linking defined behavior instead of a data race.
     *  @ref dense_ram_index_concurrency
     *
     *  Under a mutation-free contract - @c immutable_k or stronger - every atomic degrades to a
     *  plain load: with no writers alive there is nothing to order, and plain reads free the
     *  compiler to unroll & prefetch the row scan. Mutating members belong to the fully
     *  concurrent form alone.
     */
    template <synchronization_t synchronization_ = synchronization_t::concurrent_k>
    struct neighbors_ref {
        node_ids_span_t row_;

        /** @brief Total slots in the row, every one usable for a neighbor ID. */
        inline small_size_t capacity() const noexcept { return static_cast<small_size_t>(row_.size()); }

        /** @brief Relaxed atomic read of slot @p index; slot 0 belongs to @c peek_first / @c lock. */
        inline node_id_t load_slot(small_size_t index) const noexcept {
            if constexpr (synchronization_ == synchronization_t::concurrent_k)
                return atomic_ref<node_id_t>(row_[index]).load(std::memory_order_relaxed);
            else return row_[index];
        }
        /** @brief Relaxed atomic write of slot @p index - only for the unpublished owner or the
         *      current lock holder. */
        inline void store_slot(small_size_t index, node_id_t value) const noexcept {
            atomic_ref<node_id_t>(row_[index]).store(value, std::memory_order_relaxed);
        }
        /** @brief Acquire-load of the lock word: the sentinel means unpublished, empty, or locked;
         *      any other value is the first neighbor ID, with everything its last (un)locker wrote
         *      now visible to us. */
        inline node_id_t peek_first() const noexcept {
            if constexpr (synchronization_ == synchronization_t::concurrent_k)
                return atomic_ref<node_id_t>(row_[0]).load(std::memory_order_acquire);
            else return row_[0];
        }
        /** @brief Spins until the lock is ours; returns the displaced first-neighbor value.
         *      An exchange that receives the sentinel back did @b not acquire the lock - that rule
         *      lives here, in one place, instead of at every call site. Only meaningful on
         *      published nodes: locking an unpublished row would spin until its owner publishes. */
        inline node_id_t lock() const noexcept {
            // Claim only after observing the slot free: on Arm a `SWPAL` is a write even when it
            // just reads the sentinel back, pulling the line into Modified state and stealing it
            // from the holder mid-rewrite; the acquiring peek keeps contenders in Shared state.
            // On x86 the peek is one plain load.
            while (true) {
                while (peek_first() == dense_node_id_unknown_k) wait_while_locked();
                node_id_t const displaced = atomic_ref<node_id_t>(row_[0]).exchange( //
                    dense_node_id_unknown_k, std::memory_order_acq_rel);
                if (displaced != dense_node_id_unknown_k) return displaced;
            }
        }
        /** @brief Monitored capped wait on the lock word itself until it stops reading as the
         *      sentinel: the holder's unlock store wakes us, the cap guards the arming race. The
         *      address form exists precisely for such in-place words that are not `std::atomic`
         *      objects. Readers use it too: a bare re-peek loop would spin acquiring loads for the
         *      writer's whole critical section, burning issue slots and cross-cluster fabric. */
        inline void wait_while_locked() const noexcept {
            wait_type_ {}(row_.data(), dense_node_id_unknown_k, 0u, fu::wait_capped_t {});
        }
        /** @brief Release-stores @p first_value into slot 0 - unlocking and simultaneously
         *      (re)publishing the edited neighbors to every future acquiring reader. */
        inline void unlock(node_id_t first_value) const noexcept {
            atomic_ref<node_id_t>(row_[0]).store(first_value, std::memory_order_release);
        }
    };

    /** @brief The writer-side spelling: mutating paths are always fully synchronized. */
    using neighbors_ref_t = neighbors_ref<synchronization_t::concurrent_k>;

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

    /** @brief Typed view over the fixed-capacity neighbors at @p location - atomic by
     *      default, plain under a mutation-free contract. @ref dense_ram_index_concurrency */
    template <synchronization_t synchronization_ = synchronization_t::concurrent_k>
    inline neighbors_ref<synchronization_> neighbors_for(chunked_location_t const& location) noexcept {
        return {neighbors_chunks_[location.chunk_index].row(location.index_in_chunk).as_vector()};
    }
    /** @brief Immutable view of the fixed-capacity neighbors at @p location. */
    inline node_ids_view_t neighbors_for(chunked_location_t const& location) const noexcept {
        return neighbors_chunks_[location.chunk_index].row(location.index_in_chunk).as_vector();
    }

    /**
     *  @brief Allocates & publishes chunks until at least @p minimal_capacity fresh rows are added,
     *      following the growth schedule of @ref dense_ram_index_malloc.
     *  @return The number of added rows - possibly non-zero even when the status is a failure,
     *      in which case the caller must still release them back into @c remaining_capacity_.
     *
     *  Only one thread may run this at a time: the "responsible" one that dropped
     *  @c remaining_capacity_ below zero. Successive responsible threads are ordered by the
     *  acquire/release pair on that counter, which is also what publishes the non-atomic
     *  @c chunks_count_ and the directory writes. Readers never touch a chunk before some node
     *  inside it becomes visible, and that visibility chains through the same atomics.
     */
    expected<std::size_t> create_next_chunk(std::size_t minimal_capacity) noexcept {
        // Growing before the geometry is frozen is a usage error, not an allocation failure.
        geometry_t const frozen = geometry();
        if (!frozen) return {0, status_t::uninitialized_k};
        std::size_t const chunk_dimensions = frozen.dimensions();
        std::size_t const chunk_connectivity = frozen.connectivity();

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
            // slot reads as "unpublished" to every traversal, and sentinel tails terminate nodes
            // with fewer neighbors - so no explicit neighbor counts are stored anywhere.
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
            // The final chunk closes the ID space at 2^32 rows, but the all-ones ID is the sentinel:
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
     *  machine. Claims @b storage only, never identity: node IDs are dispensed one-by-one from
     *  @c next_dispensed_id_ at wiring time. On success, enough sentinel-filled rows exist for
     *  @p count dispenses, and the caller must roll forward to wiring & publishing them all -
     *  claimed capacity cannot be released once IDs are dispensed against it, or the
     *  visibility watermark would stall forever. @ref dense_ram_index_concurrency
     */
    expected<node_id_t> claim_capacity(std::size_t count) noexcept {
        // A batch beyond the ID-space ceiling can never succeed: reject it before the counter
        // dance, so it neither grows the index to its maximum in vain nor overflows the signed
        // capacity arithmetic below.
        if (count > max_node_count_k) return {dense_node_id_unknown_k, status_t::overflow_k};
        while (true) {
            // One unconditional subtract, and its sign says what happened. A bounded subtract
            // with a separate election was considered: it never leaves a deficit behind, but
            // costs a load and a compare-exchange loop where this is one add, and the deficit is
            // exactly the signal the waiters below watch.
            std::ptrdiff_t const old_remaining_capacity = atomic_ref<std::ptrdiff_t>(remaining_capacity_)
                                                              .fetch_sub( //
                                                                  count, std::memory_order_acquire);
            std::ptrdiff_t const new_remaining_capacity = old_remaining_capacity - static_cast<std::ptrdiff_t>(count);

            // The thread that moved the counter from non-negative to negative is responsible for
            // the next allocation. Accepting `old == 0` is load-bearing: capacity may rest at
            // exactly zero, and `old > 0` would then elect nobody - a proven livelock. Nothing lifts
            // a negative counter but this thread's own release: `release_capacity` waits for a whole
            // one, so no second crossing happens while a grower is inside.
            if (old_remaining_capacity >= 0 && new_remaining_capacity < 0) {
                auto [added_capacity, chunk_status] = create_next_chunk(count);
                if (chunk_status != status_t::success_k) {
                    // Revert our claim - releasing any partially added chunks along the way -
                    // so that a smaller insertion may succeed later.
                    atomic_ref<std::ptrdiff_t>(remaining_capacity_)
                        .add(count + added_capacity, std::memory_order_release);
                    return {dense_node_id_unknown_k, chunk_status};
                }
                // The take releases too: a claimer that fit on the counter before the growth but whose rows this
                // take pushes into the new chunk reaches the directory through this word alone.
                node_id_t const first_id =
                    atomic_ref<node_id_t>(last_allocated_id_).fetch_add(count, std::memory_order_acq_rel);
                atomic_ref<std::ptrdiff_t>(remaining_capacity_).add(added_capacity, std::memory_order_release);
                return {first_id, status_t::success_k};
            }
            // We may be lucky and the space is already reserved for the whole batch.
            else if (new_remaining_capacity >= 0) {
                return {atomic_ref<node_id_t>(last_allocated_id_).fetch_add(count, std::memory_order_acquire),
                        status_t::success_k};
            }
            // Corner case: the index has 10 free slots, thread Y is growing it for its 20 entries,
            // and we want a 100 - even freshly added capacity may not be enough for us.
            else {
                // Revert the counter, then wait for it to move - any release or claim wakes the
                // capped monitored wait, and we re-poll regardless once the cap expires.
                std::ptrdiff_t const reverted_capacity =
                    atomic_ref<std::ptrdiff_t>(remaining_capacity_).fetch_add(count, std::memory_order_release) +
                    static_cast<std::ptrdiff_t>(count);
                wait_type_ {}(&remaining_capacity_, reverted_capacity, 0u, fu::wait_capped_t {});
            }
        }
    }

#pragma endregion Storage

#pragma region Insertion

    /**
     *  @brief Greedy diversity selection over the ascending candidates - the classic NSW/HNSW
     *      pruning heuristic: walking nearest-first, a candidate is kept only when it is closer
     *      to the query than to every already-kept neighbor, so neighbors spread across directions.
     *      The slots the heuristic leaves empty take the nearest rejected candidates, so the
     *      selection is thin only when the pool itself is.
     *
     *  Reads the graph without locks. Cost is O(kept · considered) one-to-one distances;
     *  4-way batching is a known follow-up.
     *
     *  @param[in] candidates Search results, ascending by distance to the query.
     *  @param[in] limit Ceiling on the kept neighbors - typically the connectivity.
     *  @param[out] out Receives the kept scored nodes, diverse picks first; its live shape is the fill.
     *  @return The number of distance computations performed, for the caller's telemetry.
     */
    std::size_t select_diverse(results_array_t const& candidates, std::size_t limit, selection_t& out,
                               dense_traversal_context_t& context) const noexcept {
        // The thin pick between the engine siblings: pools spanning at least one tile take
        // the span-by-kept-tiles rectangles, keyed purely off `tile_span_k`, never a tuned
        // threshold.
        std::size_t computed_distances;
        if constexpr (has_tiles_k)
            computed_distances = candidates.size() >= tile_span_k
                                     ? select_diverse_via_tiles(candidates, limit, out, context)
                                     : select_diverse_via_quads(candidates, limit, out, context);
        else computed_distances = select_diverse_via_quads(candidates, limit, out, context);
        fill_selection(to_span(candidates), out, limit);
        return computed_distances;
    }

    /** @brief Appends the nearest rejected candidates of the ascending @p pool to the diverse
     *      prefix in @p out until @p limit or the pool runs out. The prefix is a subsequence of
     *      the pool, so one merge-walk tells a kept candidate from a rejected one. */
    static void fill_selection(std::span<dense_scored_node_t const> pool, selection_t& out,
                               std::size_t limit) noexcept {
        std::size_t const diverse_count = out.size();
        if (diverse_count == limit || !out.try_resize({limit})) return;
        std::size_t filled_count = diverse_count;
        std::size_t matched = 0;
        for (dense_scored_node_t const candidate : pool) {
            if (filled_count == limit) break;
            if (matched != diverse_count && out[matched].node_id == candidate.node_id) ++matched;
            else out[filled_count++] = candidate;
        }
        // Shrinking within the reservation never fails - the fill becomes the live shape.
        [[maybe_unused]] bool const shrunk = out.try_resize({filled_count});
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
            // Compare against the kept neighbors in 4-way batches over their cached views; short
            // tails repeat the last one. A batch may score a few neighbors past the first
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
                if (losses) break; // Some kept neighbor is closer than the query - not diverse.
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
     *      the standard gather → pack → init → update → finalize walk against the kept neighbors
     *      instead of a full square.
     *
     *  Only the kept tiles frozen at the span's start ride the matrix; neighbors accepted after
     *  that - the sub-tile fringe and any kept tile completed mid-span - score on the 4-way
     *  kernels straight off the staging surface, so no tile is ever recomputed. A kept tile
     *  packs exactly once, when its sixteenth neighbor lands, and serves every later span.
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
                // Neighbors accepted after the rectangle froze score on the 4-way path, exactly
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
                // A sixteenth neighbor completes a kept tile: gather → pack it once, through the
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
     *  @param[in] rows Scattered views of the pool - the staged neighbors, or a candidate pool.
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
     *      newcomer, everything below translates through the staged IDs.
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
        // The nearest rejected positions fill the rest, translated like the kept ones.
        small_size_t const diverse_count = kept_count;
        small_size_t matched = 0;
        for (std::size_t index = 0; index != pool.size() && kept_count != limit; ++index) {
            small_size_t const position = static_cast<small_size_t>(pool_data[index].node_id);
            if (matched != diverse_count && context.kept_positions[matched] == position) {
                ++matched;
                continue;
            }
            bool const is_newcomer = position == staged_ids.size();
            node_id_t const filled_id = is_newcomer ? newcomer_id : staged_ids[position];
            out[kept_count++] = dense_scored_node_t(filled_id, pool_data[index].distance);
        }
        // Shrinking within the reservation never fails - the fill becomes the live shape.
        [[maybe_unused]] bool const shrunk = out.try_resize({kept_count});
        return 0; // Decision only - every distance here was already paid for.
    }

    /**
     *  @brief The vector-engine scoring of one node's staged neighbors against the node itself:
     *      4-way sweeps feeding the sorted re-selection pool with real node IDs, the
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
     *  @brief Back-links the node: for every selected neighbor, locks that neighbor's slots via the
     *      slot-0 sentinel exchange and inserts the node - appending when there is room,
     *      otherwise re-selecting the diverse best among the existing neighbors plus the newcomer.
     *      A neighbor already holding the node is left as it stands, so a refinement keeping an
     *      existing neighbor never duplicates it.
     *
     *  Locking discipline: at most one neighbor lock is held at any
     *  moment, and never while spinning on the watermark - callers invoke this after wiring
     *  their own row and before their watermark turn, so every lock is released by the time
     *  the turn's spin begins. An exchange that returns the sentinel did @b not acquire the lock -
     *  another writer owns it - so we politely retry. All slot writes happen under the lock, and
     *  the unlocking release-store of slot 0 is what republishes the edited neighbors.
     *
     *  @param[in] selected The node's chosen neighbors with their recorded query distances -
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

        for (small_size_t selected_index = 0; selected_index != selected.size(); ++selected_index) {
            node_id_t const neighbor_id = selected[selected_index].node_id;
            chunked_location_t const neighbor_location = chunked_location(neighbor_id);
            neighbors_ref_t const neighbors = neighbors_for(neighbor_location);

            // Lock by exchanging the sentinel in; the displaced slot-0 value is ours to restore.
            node_id_t const first_slot_value = neighbors.lock();

            // No neighbors is encoded as a self-loop: the node becomes the only neighbor.
            if (first_slot_value == neighbor_id) {
                neighbors.unlock(new_node_id);
                continue;
            }

            // One scan of the live prefix finds both its extent and a node already wired.
            small_size_t const slots_count = neighbors.capacity();
            small_size_t existing_count = 1;
            bool present = first_slot_value == new_node_id;
            while (existing_count != slots_count) {
                node_id_t const existing_id = neighbors.load_slot(existing_count);
                if (existing_id == dense_node_id_unknown_k) break;
                present |= existing_id == new_node_id;
                ++existing_count;
            }
            if (present) {
                neighbors.unlock(first_slot_value);
                continue;
            }

            // Fast path: room to append - no distance math at all.
            if (existing_count != slots_count) {
                neighbors.store_slot(existing_count, new_node_id);
                neighbors.unlock(first_slot_value);
                continue;
            }

            // Slow path: every slot is taken. Score the existing neighbors & the newcomer by distance to
            // the neighbor itself, and re-run the diversity heuristic to keep the best spread.
            // Distances are computed under the lock for now; an optimistic precompute-and-revalidate
            // scheme shortening this critical section is a known follow-up.
            vector_view_t const neighbor_vector = vector_for(neighbor_location);
            vector_norms_t const neighbor_norms = norm_for(neighbor_location);

            // Stage the neighbors at their own slot positions - the lock-displaced slot-0 value first.
            // The newcomer never enters the staged rows: mixing it in would hand the tile
            // kernels an irregular connectivity + 1 operand, and its score is already known -
            // every supported metric is symmetric, so d(neighbor, newcomer) was recorded by
            // the very selection that kept this neighbor. The staging stays whole tiles.
            if (!context.staged_ids.try_resize({slots_count})) {
                neighbors.unlock(first_slot_value);
                continue;
            }
            chunked_location_t const first_location = chunked_location(first_slot_value);
            context.staged_ids[0] = first_slot_value;
            context.staged_vectors[0] = vector_for(first_location);
            context.staged_norms[0] = norm_for(first_location);
            for (small_size_t slot = 1; slot != slots_count; ++slot) {
                node_id_t const existing_id = neighbors.load_slot(slot);
                chunked_location_t const existing_location = chunked_location(existing_id);
                context.staged_ids[slot] = existing_id;
                context.staged_vectors[slot] = vector_for(existing_location);
                context.staged_norms[slot] = norm_for(existing_location);
            }

            // The re-selection pool is still every neighbor plus the newcomer - that "+ 1"
            // lives in the sorted pool alone, never in any operand shape. The engine pick:
            // tiles Gram the whole staging and decide from precomputed pairs; vectors
            // sweep and decide as distances arrive. The newcomer's own score is reused in
            // both - symmetric metrics recorded it at selection time, never recomputed.
            refining_candidates.clear();
            selection_t& refined = context.refined_neighbors;
            std::size_t refine_distances = 0;
            if constexpr (has_tiles_k) {
                refine_distances = stage_scores_via_tiles(neighbor_vector, neighbor_norms, new_vector, new_norms,
                                                          selected[selected_index].distance, context);
                select_diverse_via_pairs(refining_candidates, node_connectivity, refined, new_node_id, context);
            }
            else {
                refine_distances = stage_scores_via_quads(neighbor_vector, neighbor_norms, new_node_id,
                                                          selected[selected_index].distance, context);
                refine_distances += select_diverse_via_quads(refining_candidates, node_connectivity, refined, context);
            }
            context.computed_distances += refine_distances;
            context.computed_distances_in_reverse_refines += refine_distances;

            // Rewrite the tail slots, sentinel-terminate any shrink, and unlock with the new head.
            // A non-empty refined fill always holds: the heuristic never rejects the nearest candidate.
            small_size_t const refined_count = static_cast<small_size_t>(refined.size());
            for (small_size_t slot = 1; slot < refined_count; ++slot) neighbors.store_slot(slot, refined[slot].node_id);
            for (small_size_t slot = refined_count; slot < slots_count; ++slot)
                neighbors.store_slot(slot, dense_node_id_unknown_k);
            neighbors.unlock(refined[0].node_id);
        }
    }

#pragma endregion Insertion

#pragma region Refinement

    /** @brief Stages the live prefix of @p neighbors into @c context.staged_ids, waiting out a
     *      lock holder like any traversal does; a published-empty self-loop stages nothing.
     *  @return The staged count - the tensor's live shape. */
    small_size_t stage_neighbors_(neighbors_ref_t neighbors, node_id_t node_id,
                                  dense_traversal_context_t& context) const noexcept {
        ids_tensor_t& staged_ids = context.staged_ids;
        small_size_t const slots_count = neighbors.capacity();
        if (!staged_ids.try_resize({slots_count})) return 0;
        node_id_t first_value = neighbors.peek_first();
        while (first_value == dense_node_id_unknown_k) {
            neighbors.wait_while_locked();
            first_value = neighbors.peek_first();
        }
        small_size_t staged_count = 0;
        if (first_value != node_id) {
            staged_ids[staged_count++] = first_value;
            while (staged_count != slots_count) {
                node_id_t const slot_value = neighbors.load_slot(staged_count);
                if (slot_value == dense_node_id_unknown_k) break;
                staged_ids[staged_count++] = slot_value;
            }
        }
        // Shrinking within the reservation never fails - the fill becomes the live shape.
        [[maybe_unused]] bool const shrunk = staged_ids.try_resize({staged_count});
        return staged_count;
    }

    /**
     *  @brief One node's re-selection: its neighbors seed the walk, the insertion's own selection
     *      picks the new ones, the rewrite runs under the node's lock, and every kept neighbor is
     *      back-linked. No distance math runs under a lock.
     *  @return Whether the new neighbors differ as a set from the ones read.
     */
    bool refine_one_(node_id_t node_id, node_id_t first_invisible_id, std::size_t expansion_factor,
                     dense_traversal_context_t& context) noexcept {
        std::size_t const node_connectivity = connectivity();
        chunked_location_t const location = chunked_location(node_id);
        vector_view_t const own_vector = vector_for(location);
        vector_norms_t const own_norms = norm_for(location);
        neighbors_ref_t const own_neighbors = neighbors_for(location);

        // The neighbors as read seed the walk, so every one already wired competes; the node
        // itself is visited before anything stages, so it never joins its own pool.
        context.results.clear();
        context.refining_candidates.clear();
        context.frontier.clear();
        context.visited.clear();
        [[maybe_unused]] membership_t const self_marked = context.visited.insert(node_id);
        small_size_t const staged_count = stage_neighbors_(own_neighbors, node_id, context);
        std::span<node_id_t const> const seeds {context.staged_ids.data(), staged_count};
        if (failed(search_around(own_vector, own_norms, first_invisible_id, expansion_factor, context,
                                 dummy_predicate_t {}, seeds)))
            ++context.saturated_walks;

        selection_t& selected = context.selected_neighbors;
        std::size_t const refine_distances = select_diverse(context.results, node_connectivity, selected, context);
        context.computed_distances += refine_distances;
        context.computed_distances_in_refines += refine_distances;
        small_size_t const selected_count = static_cast<small_size_t>(selected.size());

        // The settling signal compares neighbor sets against the ones read: a node's neighbors are
        // distinct, so equal sizes and one match per kept neighbor mean the same set. The slots
        // are then rewritten ascending, sentinel-terminated, and republished through slot 0.
        small_size_t matches = 0;
        for (small_size_t selected_index = 0; selected_index != selected_count; ++selected_index)
            for (node_id_t const staged_id : seeds) matches += staged_id == selected[selected_index].node_id;
        bool const changed = selected_count != staged_count || matches != selected_count;
        small_size_t const slots_count = own_neighbors.capacity();
        [[maybe_unused]] node_id_t const displaced = own_neighbors.lock();
        for (small_size_t slot = 1; slot < selected_count; ++slot)
            own_neighbors.store_slot(slot, selected[slot].node_id);
        for (small_size_t slot = max_of<small_size_t>(selected_count, 1); slot < slots_count; ++slot)
            own_neighbors.store_slot(slot, dense_node_id_unknown_k);
        own_neighbors.unlock(selected_count ? selected[0].node_id : node_id);

        back_link_neighbors(node_id, own_vector, own_norms, selected, context);
        return changed;
    }

#pragma endregion Refinement

#pragma region Search

    /**
     *  @brief Samples one 4-way batch of distinct, unvisited, published landmarks, scores them
     *      against the @p query, and queues them into the frontier & the result pool.
     *  @return The landmarks queued; @c 0 on @c success_k means every published node is already
     *      visited, which ends the walk rather than failing it. A failed status means scratch
     *      could not grow, which is a different answer the caller may want to report.
     *
     *  Uniform draws that collide with @c visited fall back to cyclic linear probing over the ID
     *  space, so every call makes strict progress without retry budgets - the sampler is exact,
     *  not heuristic. Only transiently locked nodes are re-rolled, mirroring the traversal policy.
     */
    template <synchronization_t synchronization_ = synchronization_t::concurrent_k,
              typename predicate_type_ = dummy_predicate_t>
    expected<std::size_t> sample_landmarks(vector_view_t query, vector_norms_t query_norms,
                                           node_id_t first_invisible_id, std::size_t expansion_factor,
                                           dense_traversal_context_t& context, std::span<node_id_t const>& seeds,
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
        // `visited` may also hold above-watermark IDs discovered through fresh back-linking, so the
        // size check is conservative: it can only under-sample, never let the probe loop spin.
        while (landmarks_in_batch != landmarks_ids.size() && visited.size() < first_invisible_id) {
            // Caller-staged seeds go first, each taken once and skipped when unpublished or
            // already visited; past them, draw uniformly and on a collision probe forward
            // cyclically - deterministic progress even when the published set is almost fully explored.
            node_id_t landmark_node_id;
            if (!seeds.empty()) {
                landmark_node_id = seeds.front();
                seeds = seeds.subspan(1);
                if (landmark_node_id >= first_invisible_id || visited.contains(landmark_node_id)) continue;
            }
            else {
                landmark_node_id = uniform_below<node_id_t> {first_invisible_id}(context.random_generator);
                while (visited.contains(landmark_node_id))
                    landmark_node_id = landmark_node_id + 1 == first_invisible_id ? 0 : landmark_node_id + 1;
            }

            chunked_location_t const landmark_location = chunked_location(landmark_node_id);
            // Published nodes may be transiently locked for re-linking - re-roll those. Under a
            // mutation-free contract every sampled node is published & unlocked - no probe needed.
            if constexpr (synchronization_ == synchronization_t::concurrent_k)
                if (neighbors_for(landmark_location).peek_first() == dense_node_id_unknown_k) continue;

            [[maybe_unused]] membership_t const marked = visited.insert(landmark_node_id);
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

    /** @brief The pool whose radius governs a walk's termination: the admitted one once it fills,
     *      since matches are sparser than nodes and that radius is the wider one, the navigation
     *      one until then. */
    template <typename predicate_type_>
    static results_array_t const& governing_pool_(dense_traversal_context_t const& context,
                                                  std::size_t expansion_factor) noexcept {
        constexpr bool predicated_k = !std::is_same_v<std::remove_cvref_t<predicate_type_>, dummy_predicate_t>;
        if constexpr (predicated_k)
            if (context.refining_candidates.size() == expansion_factor) return context.refining_candidates;
        return context.results;
    }

    /**
     *  @brief Greedy best-first NSW walk from random published landmarks toward the @p query.
     *
     *  Read-only on the graph: takes no locks, writes nothing, retries transiently locked nodes.
     *  The walk never fails hard: a refused re-table ends it early with the ranking it has, and
     *  the returned status says so, which the callers count as a saturated walk - an insertion
     *  wires the node with fewer neighbors regardless, and a search reports the count so the
     *  caller can widen the scratch.
     *
     *  @param[in] first_invisible_id The caller's watermark snapshot, bounding the readable ID space.
     *  @param[in] expansion_factor Breadth of the candidate pool - the walk's stopping radius.
     *  @param[inout] context Scratch & telemetry; must arrive pre-reserved & pre-cleared, and ends
     *      with up to @p expansion_factor nearest published candidates in @c context.results.
     *  @param[in] predicate Optional admission filter - see @c search_if. Non-matching nodes
     *      still score and route the walk; only matching ones enter @c context.refining_candidates,
     *      whose radius governs termination once that pool fills.
     *  @param[in] seeds Optional landmarks to start from ahead of the random draws - a
     *      re-selection hands in the node's own neighbors, so every one already wired competes.
     */
    template <synchronization_t synchronization_ = synchronization_t::concurrent_k,
              typename predicate_type_ = dummy_predicate_t>
    status_t search_around(vector_view_t query, vector_norms_t query_norms, node_id_t first_invisible_id,
                           std::size_t expansion_factor, dense_traversal_context_t& context,
                           predicate_type_&& predicate = predicate_type_ {},
                           std::span<node_id_t const> seeds = {}) noexcept {
        constexpr bool predicated_k = !std::is_same_v<std::remove_cvref_t<predicate_type_>, dummy_predicate_t>;

        results_array_t& results = context.results;
        frontier_queue_t& frontier = context.frontier;
        visited_set_t& visited = context.visited;
        std::size_t const node_connectivity = connectivity();

        // Seed with about as many landmarks as the graph connectivity - a stochastic multi-start,
        // gathered in 4-way batches to amortize the loop overhead over unrolled "gathers".
        small_size_t const landmark_batches = divide_round_up<small_size_t>(node_connectivity, 4);
        for (small_size_t landmarks_batch = 0; landmarks_batch != landmark_batches; ++landmarks_batch) {
            auto const [landmarks, seed_status] = sample_landmarks<synchronization_>(
                query, query_norms, first_invisible_id, expansion_factor, context, seeds, predicate);
            if (failed(seed_status)) return seed_status;
            if (!landmarks) break; // Everything published is already visited.
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
                // set's growth toward the published ID space.
                std::size_t const filled = predicated_k ? context.refining_candidates.size() : results.size();
                if (filled >= expansion_factor) break;
                auto const [landmarks, seed_status] = sample_landmarks<synchronization_>(
                    query, query_norms, first_invisible_id, expansion_factor, context, seeds, predicate);
                if (failed(seed_status)) return seed_status;
                if (!landmarks) break; // Nothing fresh left to restart from.
                ++context.restarts;
                continue;
            }

            // Peek - don't pop yet: the chosen node's neighbors may be locked, in which case
            // we retry the selection rather than dropping it from the frontier.
            dense_scored_node_t const candidate = frontier.top();
            results_array_t const& pool = governing_pool_<predicate_type_>(context, expansion_factor);
            bool const pool_full = pool.size() == expansion_factor;
            // Once the nearest unexpanded node is past a full pool's radius, nothing deeper can
            // improve the answer.
            if (pool_full && candidate.distance > pool.top().distance) break;
            // Every other entry past that radius is never popped either: shed them when the coming
            // expansion, at most one node's neighbors, would not fit.
            if (pool_full && frontier.capacity() - frontier.size() < node_connectivity)
                frontier.erase_below(pool.top());

            chunked_location_t const candidate_location = chunked_location(candidate.node_id);
            neighbors_ref<synchronization_> const candidate_neighbors = neighbors_for<synchronization_>( //
                candidate_location);

            // Slot 0 holds either the lock sentinel or the first neighbor's ID. If the node is
            // locked for re-linking, wait it out: it will either free up, or we keep re-evaluating
            // the same nearest candidate until it does - dropping it would silently degrade recall.
            node_id_t const first_neighbor_id = candidate_neighbors.peek_first();
            if constexpr (synchronization_ == synchronization_t::concurrent_k)
                if (first_neighbor_id == dense_node_id_unknown_k) {
                    candidate_neighbors.wait_while_locked();
                    continue;
                }

            // The node is live and readable - commit to it and drop it from the frontier.
            frontier.pop();
            ++context.visited_members;

            // Scan the fixed-capacity sentinel-terminated slots. Slot 0 comes from the acquiring load
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
                    // Translation-frontier invariant: only IDs below our acquire-fenced watermark
                    // snapshot may be located. A fresher ID read from a concurrently re-linked slot
                    // has no happens-before edge to its hosting chunk yet, so skip it; a later
                    // traversal with a fresher snapshot will meet it safely.
                    if (neighbor_id >= first_invisible_id) continue;
                    if (visited.contains(neighbor_id)) continue;
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
                // The re-sort only normalizes what a reordering predicate may return - for
                // order-preserving ones the network fires no exchanges.
                if constexpr (predicated_k) {
                    dense_admissions_t admissions;
                    invoke_admission(predicate, scored, admissions);
                    network_sort(std::span(admissions), dense_closer_scored_node_t {});
                    context.refining_candidates.insert(std::span(admissions), expansion_factor);
                }

                // One merge folds the ascending quad into the pool; the accepted entries are
                // its prefix - the top-expansion-of-the-union truth - and only they enter the
                // frontier. The merge precedes the marks, so a refused re-table keeps the quad.
                std::size_t const accepted_count = results.insert(scored, expansion_factor);

                // Grow the visited set ahead of the marks - `set()` itself never rehashes.
                // Full quads - the hot case - mark all four lanes in one interleaved probe sweep.
                if (status_t const seen = visited.reserve(visited.size() + gathered_count); failed(seen)) return seen;
                if (gathered_count == gathered_ids.size()) {
                    [[maybe_unused]] u64_t const fresh = visited.insert(gathered_ids);
                }
                else
                    for (small_size_t lane = 0; lane != gathered_count; ++lane) {
                        [[maybe_unused]] membership_t const marked = visited.insert(gathered_ids[lane]);
                    }

                for (small_size_t lane = 0; lane != accepted_count; ++lane)
                    if (status_t const queued = frontier.insert(dense_scored_node_t(scored[lane])); failed(queued))
                        return queued;
            }
        }
        return status_t::success_k;
    }

#pragma endregion Search

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
     *  @brief Separately allocated matrix slabs, where each row holds the fixed-capacity neighbors of one vector.
     *      Slot indexes are global, monotonically transparently growing across individual matrices.
     *  @ref dense_ram_index_malloc
     */
    fu::dynamic_padded_array<neighbors_chunk_t, neighbors_allocator_type_> neighbors_chunks_;

    /**
     *  @brief Signed integer describing "unsigned capacity" to simplify underflow/overflow detection from concurrent
     *      thread updates. @b Never assign values to it, always use increment & decrement to move it up and down.
     *
     *  Every RMW'd atomic below sits on its own @c default_alignment_k line: the claim counters move per
     *  batch, the dispenser per insert, and the watermark is acquire-read by every search - co-located,
     *  each dispenser RMW would invalidate the line every concurrent search snapshots. Read-mostly members
     *  stay grouped: a line every core holds Shared is the cheap kind.
     *
     *  Every one of them is @c mutable: the pre-P3323 @c std::atomic_ref binds no const word, not even
     *  to load it, and the const readers only load.
     */
    alignas(default_alignment_k) mutable std::ptrdiff_t remaining_capacity_ {0};

    /**
     *  @brief Unsigned integer containing the identifier of the last "allocated" node. It defines the lower
     *      bound for the number of initialized or empty rows across @c vectors_chunks_ and @c neighbors_chunks_
     *      chunks. It's typically incremented in batches.
     */
    alignas(default_alignment_k) mutable node_id_t last_allocated_id_ {0};

    /**
     *  @brief The ID dispenser: the identifier the next wired node receives. Incremented one-by-one
     *      at wiring time, so IDs follow the global wiring-start order and each thread holds at most
     *      one unpublished ID. @ref dense_ram_index_concurrency
     */
    alignas(default_alignment_k) mutable node_id_t next_dispensed_id_ {0};
    /**
     *  @brief The visibility watermark: every ID strictly below it is published, its vector, norms &
     *      neighbors readable - the safe bound for random landmark sampling. Advanced strictly in
     *      dispensed order by each publisher taking its turn. @ref dense_ram_index_concurrency
     */
    alignas(default_alignment_k) mutable node_id_t last_visible_id_ {0};

    /** @brief Read-mostly after the freeze - kept off the watermark's line with the other cold members. */
    alignas(default_alignment_k) mutable geometry_t geometry_ {};

    /** @brief Constructed chunks per directory. Deliberately non-atomic: only the one elected grower
     *      mutates it, and successive growers are ordered by the acquire/release pair on
     *      @c remaining_capacity_, which no release lifts while one is inside. @ref create_next_chunk */
    small_size_t chunks_count_ = 0;
};

/**
 *  @brief External reader's cursor over the watermark of one @c dense_ram_index:
 *      random access & ID-order iteration over nodes, plus neighbor-following through a privately
 *      staged copy of the current node's neighbors.
 *
 *  Strictly read-only under every contract - the index is never written through, so readers
 *  can never stall writers or convoy behind them. Under @c concurrent_k the staging reads
 *  neighbor words through relaxed atomics - a re-linker may interleave with the copy, and the
 *  worst case is the same stale-but-valid mix the traversal itself tolerates; exact neighbors come
 *  from the @c immutable_k contract, which reads plain memory. The cursor captures the frozen
 *  geometry once at construction, starts past the end, and @c seek(0) enters the graph.
 *
 *  The staged neighbors may contain IDs at or above this cursor's watermark snapshot, discovered
 *  through fresh back-linking; @c seek & @c follow reject those, and @c refresh widens the
 *  snapshot to admit them. A cursor standing past the end stays past the end through a
 *  @c refresh - entering the admitted region takes an explicit @c seek.
 */
template <typename index_type_, synchronization_t synchronization_, typename allocator_type_>
struct dense_ram_cursor {

    using index_t = index_type_;
    template <typename word_type_>
    using atomic_ref = typename index_t::template atomic_ref<word_type_>;
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
    /** @brief This cursor's watermark snapshot: IDs strictly below it are readable. */
    node_id_t visible_count_;
    /** @brief The current node ID, or the watermark snapshot when past the end. */
    node_id_t current_;
    /** @brief The frozen connectivity, captured once at construction. */
    small_size_t connectivity_;

    /** @brief Binds the cursor & captures its snapshots; allocates nothing - call @c reserve(). */
    explicit dense_ram_cursor(index_t const& index) noexcept
        : index_(index), visible_count_(atomic_ref<node_id_t>(index.last_visible_id_).load(watermark_order_k)),
          current_(visible_count_), connectivity_(static_cast<small_size_t>(index.connectivity())) {}

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

    /** @brief Nodes visible to this cursor; IDs strictly below are readable. */
    node_id_t visible_count() const noexcept { return visible_count_; }

    /** @brief Widens the snapshot to admit nodes published since; a cursor standing past the
     *      end stays past the end - entering the admitted region takes an explicit @c seek. */
    void refresh() noexcept {
        bool const past_end = current_ == visible_count_;
        visible_count_ = atomic_ref<node_id_t>(index_.last_visible_id_).load(watermark_order_k);
        if (past_end) current_ = visible_count_;
    }

    /** @brief The current node's ID; equals @c visible_count() past the end. */
    node_id_t id() const noexcept { return current_; }
    /** @brief @c true while the cursor points at a readable node. */
    explicit operator bool() const noexcept { return current_ < visible_count_; }
    /** @brief Immutable view of the current node's vector; meaningless past the end. */
    vector_view_t vector() const noexcept { return index_.vector_for(location_); }
    /** @brief The current node's pre-computed norms; meaningless past the end. */
    vector_norms_t norms() const noexcept { return index_.norm_for(location_); }
    /** @brief The staged neighbor IDs of the current node - decoded, stable until the next move. */
    std::span<node_id_t const> neighbors() const noexcept { return to_span(staged_neighbors_).first(staged_count_); }

    /** @brief Positions on @p node & stages a copy of its neighbors: @c malformed_k at or
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
        node_id_t first_value;
        if constexpr (synchronization_ == synchronization_t::concurrent_k) {
            first_value = atomic_ref<node_id_t>(const_cast<node_id_t&>(row[0])).load(std::memory_order_acquire);
            while (first_value == dense_node_id_unknown_k) {
                typename index_t::wait_t {}(row.data(), dense_node_id_unknown_k, small_size_t(0), fu::wait_capped_t {});
                first_value = atomic_ref<node_id_t>(const_cast<node_id_t&>(row[0])).load(std::memory_order_acquire);
            }
        }
        else first_value = row[0];
        staged_neighbors_[0] = first_value;
        // Tail slots may interleave with a live re-linker: relaxed atomics make the
        // stale-but-valid mix defined; certified contracts read plain memory.
        if constexpr (synchronization_ == synchronization_t::concurrent_k)
            for (small_size_t slot = 1; slot != connectivity_; ++slot)
                staged_neighbors_[slot] =
                    atomic_ref<node_id_t>(const_cast<node_id_t&>(row[slot])).load(std::memory_order_relaxed);
        else
            for (small_size_t slot = 1; slot != connectivity_; ++slot) staged_neighbors_[slot] = row[slot];
        // Decode the row's encodings: a self-loop is "published, zero neighbors", and the
        // all-ones sentinel terminates nodes with fewer neighbors.
        if (staged_neighbors_[0] == node) { staged_count_ = 0; }
        else {
            small_size_t live = 0;
            while (live != connectivity_ && staged_neighbors_[live] != dense_node_id_unknown_k) ++live;
            staged_count_ = live;
        }
        return status_t::success_k;
    }

    /** @brief Moves to the next ID below the snapshot, or past the end; a staging refusal
     *      parks on the new ID with no neighbors staged - a later @c seek to the same ID retries. */
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

    /** @brief Follows the staged neighbor at @p slot: @c malformed_k when the slot is beyond
     *      the live prefix or its target is at or past this cursor's snapshot. */
    status_t follow(small_size_t slot) noexcept {
        if (slot >= staged_count_) return status_t::malformed_k;
        return seek(staged_neighbors_[slot]);
    }
};

} // namespace usearch
} // namespace unum
