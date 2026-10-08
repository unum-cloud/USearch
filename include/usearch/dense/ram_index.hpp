/**
 *  @file include/usearch/dense/ram_index.hpp
 *  @author Ash Vardanian
 *  @date July 26, 2023
 *  @brief Vector Search index for a single uniform latency memory space.
 */
#pragma once

#include <cassert> // `assert`
#include <cmath>   // `std::exp` - the softmax epilogue
#include <cstring> // `std::memcpy`, `std::memset`

#include <array>       // `std::array`
#include <bit>         // `std::bit_width`, `std::bit_floor`, `std::has_single_bit`
#include <functional>  // `std::identity` - the single index's resolver
#include <limits>      // `std::numeric_limits`
#include <span>        // `std::span`
#include <type_traits> // `std::has_unique_object_representations_v`

#include <forkunion/atomics.hpp> // `fu::standard_atomic_ref`
#include <forkunion/types.hpp>   // `fu::standard_yield_t`, `fu::standard_cache_hints_t`
#include <numkong/tensor.hpp>

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/ram_buffers.hpp"
#include "usearch/utilities/ram_sorts.hpp" // `network_sort` - the scored quads
#include "usearch/utilities/ram_tables.hpp"
#include "usearch/dense/types.hpp"
#include "usearch/dense/ram_metric.hpp"

namespace unum {
namespace usearch {

/** External reader's cursor over one @c dense_ram_index, defined after the index it befriends.
 *  The allocator defaults to the index's traversals-side family, so staging inherits the same
 *  NUMA-aware placement; readers pinned elsewhere override it. */
template <typename index_type_,                                                 //
          synchronization_t synchronization_ = synchronization_t::concurrent_k, //
          typename allocator_type_ = typename index_type_::candidates_allocator_t>
struct dense_ram_cursor;

/**
 *  @brief Vector Search index for a single uniform latency memory space.
 *
 *  @section dense_ram_index_metrics Supported Metrics
 *
 *  Dense representations under @b Inner-Product, Angular (Cosine), Euclidean (L2), Hamming
 *  (Manhattan) and Jaccard (Tanimoto) - each inferable from one pass of vector-vector products
 *  once the L1 and L2 norms are stored alongside the rows.
 *
 *  @section dense_ram_index_growth Growth Limits
 *
 *  Up to 4 billion entries, all @b in-memory. Distances are f32, which can cost recall on small
 *  K: retrieve widely from the quantized index and rerank against the originals.
 *
 *  @section dense_ram_index_concurrency Concurrency & Transactions
 *
 *  Construction and traversal run simultaneously. Every node's neighbors carry that node's own
 *  lock in their leading slot, placing locks at the start of independent cache lines; the rest
 *  of the book-keeping is a handful of atomics, none of them compare-and-swap.
 *
 *  @section dense_ram_index_persistence Persistence
 *
 *  Not meant to be reached from disk - io_uring and SPDK reimplementations cover that.
 *
 *  @section dense_ram_index_malloc Memory Allocation Strategy
 *
 *  Chunks grow through power-of-two sizes up to @b ~4GB slabs, amortizing allocations between
 *  insertions instead of re-padding one huge matrix. Compatible with CXL-backed allocators.
 */
template <                                                                    //
    nk_dtype_t vector_dtype_ = nk_bf16_k,                                     //
    nk_kernel_kind_t kernel_kind_ = nk_kernel_angular_k,                      //
    nk_capability_t capability_ = nk_cap_icelake_k,                           //
    typename wait_type_ = fu::standard_yield_t,                               //
    template <typename> class atomic_ref_template_ = fu::standard_atomic_ref, //
    typename cache_hints_type_ = fu::standard_cache_hints_t,                  //
    typename vectors_allocator_type_ = std::allocator<byte_t>,                //
    typename neighbors_allocator_type_ = std::allocator<byte_t>,              //
    typename traversals_allocator_type_ = std::allocator<byte_t>              //
    >
struct dense_ram_index;

/** One quad's worth of predicate-admitted candidates - the buffer handed to batched
 *  @c search_if predicates, bounded by the distance kernels' fan-out. */
using dense_admissions_t = fu::limited_array<dense_scored_node_t, 4>;

/**
 *  @brief Runs @p predicate over one scored quad, filling @p admissions with what it admits.
 *
 *  Two shapes, detected by signature: a batched functor takes the whole @p scored span and
 *  pushes what it admits, a per-node @c admission_t(dense_node_id_t) functor is walked lane by
 *  lane. @c dummy_predicate_t never reaches here - its callers compile out whole.
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

/** Filtered search as a fold: each quad's matches merge into the admitted pool, while the walk
 *  keeps routing through whatever the predicate rejects. */
template <typename predicate_type_, typename pool_type_>
struct dense_filtering_fold {

    /** The caller's filter, per-node or batched - see @c invoke_admission. */
    predicate_type_& predicate;

    /** The pool matches merge into, ascending by distance. */
    pool_type_& admitted;

    /** The breadth @c admitted is capped at. */
    std::size_t expansion_factor;

    template <typename dot_result_type_>
    walk_demand_t absorb(std::span<dense_scored_node_t const> scored_nodes,
                         std::span<dot_result_type_ const>) noexcept {
        dense_admissions_t admissions;
        invoke_admission(predicate, scored_nodes, admissions);
        network_sort(std::span(admissions), dense_closer_scored_node_t {});
        admitted.insert(std::span(admissions), expansion_factor);
        return walk_demand_t::continue_k;
    }
};

/** Reads every scored quad - live lanes in kernel order beside their native dots, both valid only
 *  for the call - and answers whether the walk should go on. */
template <typename fold_type_, typename dot_result_type_>
concept dense_fold = requires(fold_type_& fold, std::span<dense_scored_node_t const> scored_nodes,
                              std::span<dot_result_type_ const> dots) {
    { fold.absorb(scored_nodes, dots) } noexcept -> std::same_as<walk_demand_t>;
};

/** Graph products of one left-hand row as a fold: each scored pair appends its resolved ID and native
 *  dot until the row is full, while @c scored keeps counting past it. */
template <typename id_type_, typename dot_result_type_, typename resolve_type_ = std::identity>
struct dense_multiply_fold {

    /** The row's ID output. */
    std::span<id_type_> ids;

    /** The row's product output, positionally paired with @c ids. */
    std::span<dot_result_type_> products;

    /** Maps a walked node ID to the ID the row reports. */
    resolve_type_ resolve = {};

    /** Pairs written so far. */
    std::size_t written = 0;

    /** Pairs scored, written or not - above the row's width when it was truncated. */
    std::size_t scored = 0;

    walk_demand_t absorb(std::span<dense_scored_node_t const> scored_nodes,
                         std::span<dot_result_type_ const> dots) noexcept {
        std::size_t const writable = min_of<std::size_t>(scored_nodes.size(), ids.size() - written);
        for (std::size_t lane = 0; lane != writable; ++lane) {
            ids[written + lane] = resolve(scored_nodes[lane].node_id);
            products[written + lane] = dots[lane];
        }
        written += writable;
        scored += scored_nodes.size();
        return walk_demand_t::continue_k;
    }
};

/** Prefetches the value row of every scored node that beats the pool's current k-th best, so the attention
 *  epilogue gathers its picks from cache. The pool is read before each quad merges into it. */
template <nk::numeric_dtype scalar_type_, typename cache_hints_type_, typename pool_type_,
          typename row_of_type_ = std::identity>
struct dense_attend_prefetch_fold {

    /** The value rows the picks address. */
    nk::matrix_view<scalar_type_> values;

    /** The walk's pool, ascending by distance. */
    pool_type_ const& pool;

    /** The picks the epilogue weighs - the "k". */
    std::size_t picks;

    /** Maps a walked node ID to its value row. */
    row_of_type_ row_of = {};

    template <typename dot_result_type_>
    walk_demand_t absorb(std::span<dense_scored_node_t const> scored_nodes,
                         std::span<dot_result_type_ const>) noexcept {
        if constexpr (!std::is_same_v<cache_hints_type_, fu::standard_cache_hints_t>) {
            cache_hints_type_ const hints;
            bool const bounded = picks && pool.size() >= picks;
            std::size_t const row_bytes = values.extent(1) * nk_dtype_bits(scalar_type_::dtype()) / 8;
            for (dense_scored_node_t const scored : scored_nodes) {
                if (bounded && !(scored.distance < pool[picks - 1].distance)) continue;
                std::size_t const row = row_of(scored.node_id);
                if (row >= values.extent(0)) continue;
                char const* const bytes = values.row(row).byte_data();
                for (std::size_t offset = 0; offset < row_bytes; offset += fu::default_alignment_k)
                    hints(bytes + offset, fu::prefetch_line_k);
            }
        }
        return walk_demand_t::continue_k;
    }
};

/**
 *  @brief The softmax over @p count picks, written into @p output already divided by its partition.
 *
 *  @p logit_of names each pick's logit and @p row_of its value row, gathered once. A row past
 *  @c values.extent(0) refuses with @c shape_mismatch_k before anything is written.
 */
template <nk::numeric_dtype scalar_type_, nk_capability_t capability_, typename logit_of_type_, typename row_of_type_>
expected<dense_softmax_state_t> dense_attend_picks(std::size_t count, logit_of_type_&& logit_of, row_of_type_&& row_of,
                                                   nk::matrix_view<scalar_type_> values,
                                                   std::span<f32_t> output) noexcept {
    dense_softmax_state_t state {std::numeric_limits<f32_t>::lowest(), 0};
    for (std::size_t pick = 0; pick != count; ++pick) {
        if (row_of(pick) >= values.extent(0)) return {state, status_t::shape_mismatch_k};
        state.maximum_score = max_of(state.maximum_score, logit_of(pick));
    }
    std::memset(output.data(), 0, output.size_bytes());
    constexpr std::size_t batch_k = 16;
    std::array<char const*, batch_k> rows;
    std::array<f32_t, batch_k> weights;
    for (std::size_t first = 0; first < count; first += batch_k) {
        std::size_t const batch = min_of(batch_k, count - first);
        for (std::size_t lane = 0; lane != batch; ++lane) {
            weights[lane] = std::exp(logit_of(first + lane) - state.maximum_score);
            rows[lane] = values.row(row_of(first + lane)).byte_data();
            state.partition += weights[lane];
        }
        dense_values_step<scalar_type_, capability_>(output.data(), output.size(), rows.data(), weights.data(), batch,
                                                     1);
    }
    f32_t const normalizer = state.partition > 0 ? 1 / state.partition : 0, zero = 0;
    [[maybe_unused]] nk_status_t const scaled = dense_elementwise_chain<capability_>::kit_t::scale(
        output.data(), output.size(), &normalizer, &zero, output.data(), nullptr);
    assert(scaled == nk_success_k && "The serial tier scales on every CPU");
    return {state, status_t::success_k};
}

/** Per-thread traversal scratch: candidate queues, visited set, neighbor selection buffers and
 *  the running telemetry counters - one instance per scanning thread. */
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

    /** The hash set allocates raw bytes and places IDs itself, so it demands a byte allocator. */
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

    /** One selection's output: @b [kept] scored nodes. The tensor's live shape is the fill and
     *  @c capacity() the reservation, so no separate count exists. */
    using selection_t = nk::tensor<dense_scored_node_t, candidates_allocator_t, 1>;

    using results_array_t = sorted_buffer<dense_scored_node_t, dense_closer_scored_node_t, candidates_allocator_t>;

    /** 8-ary: eight 8-byte scored nodes fill one cache line per level scan, and the
     *  insert-dominated frontier rides the shallower log_8 @c shift_up depth. */
    using frontier_queue_t = max_heap_dary<8, dense_scored_node_t, dense_further_scored_node_t, candidates_allocator_t>;

    /** The visited set walks its slots with the kit the capability mask earns: sixteen at a
     *  time on the AVX-512 classes, one at a time below them. */
    using visited_set_t = growing_hash_set<dense_node_id_t, std::hash<dense_node_id_t>, visits_allocator_t,
                                           typename dense_visits_hash_set_chain<capability_>::kit_t>;

    /** Nodes we've already scored, kept ascending as the best results so far. */
    results_array_t results;

    /** Nodes we've already re-evaluated for post-insertion refinement. */
    results_array_t refining_candidates;

    /** Nodes we consider promising and want to evaluate next. */
    frontier_queue_t frontier;

    /** Nodes we've already checked and should avoid visiting again. */
    visited_set_t visited;

    /** Pseudo-random generator for picking traversal starting points. */
    splitmix64_t random_generator;

    /** Neighbors chosen for the freshly inserted node, with their query distances - reusable
     *  while back-linking, since every supported metric is symmetric. */
    selection_t selected_neighbors;

    /** Re-selection output for one saturated neighbor's neighbors. */
    selection_t refined_neighbors;

    /** Neighbor IDs of one staged node - the input of a re-selection, never its output. */
    ids_tensor_t staged_ids;

    /** The one staging surface both engines read: the staged rows during scoring, then the kept
     *  prefix of the running selection - scoring drains the rows before any selection writes. */
    views_tensor_t staged_vectors;

    /** Stored norms, parallel to @c staged_vectors. */
    norms_tensor_t staged_norms;

    /** Pair-matrix positions of the kept neighbors, parallel to the running selection's fill. */
    ids_tensor_t kept_positions;

    /** Tile scratch of the staged Gram: the packed B-tiles plus one span of A-tiles. */
    bytes_tensor_t tiles_scratch;

    /** The @b [pool,pool] pairwise distances; its row stride is its own trailing extent,
     *  pre-padded to whole tiles so tail tiles always write in bounds. */
    distances_matrix_t pair_distances;

    /** Total one-to-one distance computations, padding lanes excluded. Every counter here runs
     *  across calls: an operation snapshots on entry and reports the difference. */
    std::size_t computed_distances = 0;

    /** Subset of @c computed_distances spent in insertion-side diversity selection. */
    std::size_t computed_distances_in_refines = 0;

    /** Subset of @c computed_distances spent re-selecting a full neighbor set while back-linking. */
    std::size_t computed_distances_in_reverse_refines = 0;

    /** Frontier candidates whose neighbors were expanded during traversals. */
    std::size_t visited_members = 0;

    /** Re-seedings after the frontier drained early - a graph-connectivity health signal. */
    std::size_t restarts = 0;

    /** Walks a scratch refusal ended early - ranked short, and on insertion wired thinner. */
    std::size_t saturated_walks = 0;

    dense_traversal_context() noexcept = delete;
    dense_traversal_context(dense_traversal_context const&) noexcept = delete;
    dense_traversal_context& operator=(dense_traversal_context const&) noexcept = delete;

    /** Constructs every container with a rebound copy of @p alloc, and keeps the allocator for
     *  the lazily-sized neighbor scratches, so all scratch lands where the caller places it. */
    dense_traversal_context(allocator_t alloc) noexcept
        : results(candidates_allocator_t(alloc)), refining_candidates(candidates_allocator_t(alloc)),
          frontier(candidates_allocator_t(alloc)), visited(visits_allocator_t(alloc)), random_generator(),
          allocator_(alloc) {}
    dense_traversal_context(dense_traversal_context&&) = default;
    dense_traversal_context& operator=(dense_traversal_context&&) = default;

    /** Sizes every neighbor scratch for @p capacity entries through the context's own allocator;
     *  geometry freezes once, so the tensors allocate once. */
    status_t reserve_neighbors(std::size_t capacity) noexcept {
        if (selected_neighbors.capacity() >= capacity) return status_t::success_k;
        auto selected = selection_t::zeros({capacity}, candidates_allocator_t(allocator_));
        auto refined = selection_t::zeros({capacity}, candidates_allocator_t(allocator_));
        auto ids = ids_tensor_t::zeros({capacity}, ids_allocator_t(allocator_));
        auto views = views_tensor_t::zeros({capacity}, views_allocator_t(allocator_));
        auto norms = norms_tensor_t::zeros({capacity}, norms_allocator_t(allocator_));
        auto positions = ids_tensor_t::zeros({capacity}, ids_allocator_t(allocator_));
        if (!selected || !refined || !ids || !views || !norms || !positions) return status_t::bad_alloc_k;
        selected_neighbors = std::move(selected.value);
        refined_neighbors = std::move(refined.value);
        staged_ids = std::move(ids.value);
        staged_vectors = std::move(views.value);
        staged_norms = std::move(norms.value);
        kept_positions = std::move(positions.value);
        return status_t::success_k;
    }

    /** Bytes @c reserve_neighbors reserves for @p capacity neighbors, each tensor on its own arena line. */
    static std::size_t neighbors_memory_usage(std::size_t capacity) noexcept {
        using arena_t = arena_allocator<std::byte>;
        return 2u * arena_t::reserved_bytes(capacity * sizeof(dense_scored_node_t)) +
               2u * arena_t::reserved_bytes(capacity * sizeof(dense_node_id_t)) +
               arena_t::reserved_bytes(capacity * sizeof(vector_view_t)) +
               arena_t::reserved_bytes(capacity * sizeof(vector_norms_t));
    }

    /** Sizes the pair matrix for a @p capacity -row pool - pre-padded to whole tiles by the
     *  caller - plus @p scratch_bytes of tile scratch. The matrix's own trailing extent is the
     *  row stride, so tail tiles always write in bounds. */
    status_t reserve_tiles(std::size_t capacity, std::size_t scratch_bytes) noexcept {
        if (pair_distances.extent(0) >= capacity) return status_t::success_k;
        auto scratch = bytes_tensor_t::zeros({scratch_bytes}, bytes_allocator_t(allocator_));
        auto distances = distances_matrix_t::zeros({capacity, capacity}, distances_allocator_t(allocator_));
        if (!scratch || !distances) return status_t::bad_alloc_k;
        tiles_scratch = std::move(scratch.value);
        pair_distances = std::move(distances.value);
        return status_t::success_k;
    }

    /** Allocator instance behind every neighbor scratch; the queues hold rebound copies. */
    [[no_unique_address]] allocator_t allocator_;
};

template <                                                                                //
    nk_dtype_t vector_dtype_, nk_kernel_kind_t kernel_kind_, nk_capability_t capability_, //
    typename wait_type_, template <typename> class atomic_ref_template_,                  //
    typename cache_hints_type_, typename vectors_allocator_type_, typename neighbors_allocator_type_,
    typename traversals_allocator_type_> //
struct dense_ram_index {

    inline static constexpr nk_dtype_t vector_dtype_k = vector_dtype_;

    using scalar_t = typename nk::type_for<vector_dtype_>::type;
    using dot_result_t = typename scalar_t::dot_result_t;
    using distance_t = f32_t;
    using candidate_t = dense_scored_node_t;

    /** Default integer type to use for loops and indexing at sub-index scale. */
    using small_size_t = u32_t;

    /** Node identifier within this sub-index - the class-local spelling of @c dense_node_id_t. */
    using node_id_t = dense_node_id_t;

    using vectors_allocator_t =
        typename std::allocator_traits<vectors_allocator_type_>::template rebind_alloc<scalar_t>;
    using neighbors_allocator_t =
        typename std::allocator_traits<neighbors_allocator_type_>::template rebind_alloc<u32_t>;
    using candidates_allocator_t =
        typename std::allocator_traits<traversals_allocator_type_>::template rebind_alloc<candidate_t>;

    using vector_view_t = nk::vector_view<scalar_t>;
    using vector_span_t = nk::vector_span<scalar_t>;

    /** What the chunks store per row: real moments for norm-consuming finalizes, the zero-size
     *  stand-in for dot, which spares the second random gather per candidate. */
    using vector_norms_t = stored_norms<kernel_kind_, scalar_t>;

    using vectors_view_t = nk::matrix_view<scalar_t>;
    using node_ids_span_t = nk::vector_span<node_id_t>;
    using node_ids_view_t = nk::vector_view<node_id_t>;

    /** The hardware wait monitor engaged on contended words - the class-local spelling of
     *  @c wait_type_, reused by external readers waiting out transient lock sentinels. */
    using wait_t = wait_type_;

    /** The atomic reference owning the index's plain synchronization words - the class-local
     *  spelling of @c atomic_ref_template_, reused by cursors reading published rows. */
    template <typename word_type_>
    using atomic_ref = atomic_ref_template_<word_type_>;

    /** The strongest @c synchronization_t contract this design honors: arbitrary readers and
     *  writers may be live, and even snapshots run optimistically. */
    inline static constexpr synchronization_t synchronization_ceiling_k = synchronization_t::concurrent_k;

    template <typename, synchronization_t, typename>
    friend struct dense_ram_cursor;

    inline static constexpr nk_kernel_kind_t kernel_kind_k = kernel_kind_;

    /** The capability bit of the distance kit the chain actually picked - lower than
     *  @c capability_ when no SIMD kit is adapted for this dtype & metric pair yet. */
    inline static constexpr nk_capability_t picked_quads_capability_k =
        dense_quads_chain<kernel_kind_, scalar_t, capability_>::picked_k;

    /** The tile backend's capability, on its own independent axis - one machine runs icelake
     *  1-to-1 kernels and sapphireamx tiles simultaneously. */
    inline static constexpr nk_capability_t picked_tiles_capability_k =
        dense_tiles_chain<kernel_kind_, scalar_t, capability_>::picked_k;

    /** The tile kit behind the tiled refinement paths; @c no_tiles_kit_t when none compiled. */
    using tiles_kit_t = typename dense_tiles_chain<kernel_kind_, scalar_t, capability_>::kit_t;

    /** The distance kit the chain picked; its step is the row pitch the chunks round to. */
    using quads_kit_t = typename dense_quads_chain<kernel_kind_, scalar_t, capability_>::kit_t;

    /** Whether a tile engine is compiled in for this kernel & dtype pair. */
    inline static constexpr bool has_tiles_k = dense_tiles_chain<kernel_kind_, scalar_t, capability_>::has_tiles_k;
    static_assert( //
        kernel_kind_k == nk_kernel_dot_k || kernel_kind_k == nk_kernel_angular_k ||
        kernel_kind_k == nk_kernel_euclidean_k || kernel_kind_k == nk_kernel_sqeuclidean_k ||
        kernel_kind_k == nk_kernel_hamming_k || kernel_kind_k == nk_kernel_jaccard_k);

    /** Each thread scanning the graph needs its own object like this. */
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

    /** The growth schedule of every chunk directory - 1'024 rows first, doubling up to 1'048'576-row slabs - shared
     *  with the owner columns that grow beside the index. @ref dense_ram_index_malloc */
    using chunk_schedule_t = chunk_schedule<10, 20>;

    /** A row's chunk and its offset within that chunk. */
    using location_t = typename chunk_schedule_t::location_t;

    /** Ceiling on dispensable node IDs: the all-ones ID is the sentinel and must never be
     *  claimed, so the final chunk holds one row that stays forever unclaimed.
     *  @ref dense_ram_index_growth */
    inline static constexpr std::size_t max_node_count_k = (u64_t(1) << 32) - 1u;

    /** Upper bound on the chunk count: every chunk before the one the first undispensable ID lands in.
     *  @ref dense_ram_index_growth */
    inline static constexpr small_size_t max_chunks_k = static_cast<small_size_t>(
        chunk_schedule_t::locate(max_node_count_k).chunk);

    /** Once-frozen geometry: vector dimensions in the high half, neighbor connectivity in the
     *  low one; zero bits means not reserved yet. Both ride one word so competing first
     *  reservations agree on both through a single cold-path compare-exchange - the only CAS in
     *  the index - and every reader unpacks a local acquire-loaded copy, so a torn or
     *  half-frozen geometry is unrepresentable. */
    struct geometry_t {
        u64_t packed_bits = 0;

        constexpr geometry_t() noexcept = default;

        /** Packs @p dimensions & @p connectivity - the inverse of the accessors. */
        constexpr geometry_t(std::size_t dimensions, connectivity_t connectivity) noexcept
            : packed_bits((static_cast<u64_t>(dimensions) << 32u) | static_cast<u64_t>(connectivity)) {}

        /** Scalar dimensions per vector. */
        constexpr small_size_t dimensions() const noexcept { return static_cast<small_size_t>(packed_bits >> 32u); }

        /** Neighbor slots per node. */
        constexpr small_size_t connectivity() const noexcept { return static_cast<small_size_t>(packed_bits); }

        /** @c true once frozen by the first successful @c reserve(). */
        constexpr explicit operator bool() const noexcept { return packed_bits != 0; }
        constexpr bool operator==(geometry_t const&) const noexcept = default;
    };
    static_assert(std::has_unique_object_representations_v<geometry_t>,
                  "The compare-exchange compares object representations, never operator==: " //
                  "padding would make value-equal geometries spuriously lose the freeze race");
    static_assert(std::atomic_ref<geometry_t>::is_always_lock_free,
                  "Every geometry load must stay one plain instruction, never a library call");

    /** Allocator instance handed to every vectors & norms chunk. */
    [[no_unique_address]] vectors_allocator_t vectors_allocator_;

    /** Allocator instance handed to every neighbors chunk. */
    [[no_unique_address]] neighbors_allocator_t neighbors_allocator_;

#pragma region Placement

    dense_ram_index(dense_ram_index const&) noexcept = delete;
    dense_ram_index& operator=(dense_ram_index const&) noexcept = delete;

    dense_ram_index(dense_ram_index&&) noexcept = delete;
    dense_ram_index& operator=(dense_ram_index&&) noexcept = delete;

    /**
     *  @brief Constructs an empty index, remembering the allocators to hand to every future chunk.
     *
     *  The chunk directories pad every header to @c default_alignment_k, so a grower's
     *  half-built header never shares a cache line with the ones readers dereference. Their
     *  storage is allocated lazily, ahead of the first chunk.
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

    /** Number of published vectors - the visibility watermark; the index itself stores no keys. */
    std::size_t size() const noexcept {
        return atomic_ref<node_id_t const>(last_visible_id_).load(std::memory_order_acquire);
    }

    /** The dimensions & connectivity together, as one acquire load of the frozen word - empty
     *  until the first successful @c reserve, and never a mix of two freezes. */
    geometry_t geometry() const noexcept {
        return atomic_ref<geometry_t const>(geometry_).load(std::memory_order_acquire);
    }

    /** Neighbor slots per node - the full fixed capacity, none lost to bookkeeping. */
    std::size_t connectivity() const noexcept { return geometry().connectivity(); }

    /** The walk breadth @p config asks for, or twice the connectivity when it asks for none. */
    template <typename config_type_>
    std::size_t non_zero_expansion_factor(config_type_ const& config) const noexcept {
        return config.expansion_factor ? config.expansion_factor : connectivity() * 2;
    }

    /** Scalar dimensions per stored vector; zero until the first reservation freezes it. */
    std::size_t dimensions() const noexcept { return geometry().dimensions(); }

    /** Rows hosted without growing again - unclaimed plus claimed. The release is read first:
     *  @ref release_capacity deflates the claim before it releases, so the sum can only dip, which
     *  is the conservative side @c reserve compares against. */
    std::size_t capacity() const noexcept {
        std::ptrdiff_t const unclaimed_rows =
            atomic_ref<std::ptrdiff_t const>(remaining_capacity_).load(std::memory_order_acquire);
        std::size_t const claimed_rows =
            atomic_ref<node_id_t const>(last_allocated_id_).load(std::memory_order_acquire);
        return claimed_rows + static_cast<std::size_t>(unclaimed_rows > 0 ? unclaimed_rows : 0);
    }

    /** Bytes the materialized chunks span - vectors, norms & neighbors - derived from the hosted
     *  capacity and the growth schedule rather than from the chunk directories, so it stays safe
     *  under concurrent growth. Reserved-but-empty rows count from the moment they are granted. */
    std::size_t memory_usage() const noexcept {
        geometry_t const frozen = geometry();
        if (!frozen) return sizeof(dense_ram_index);
        std::size_t const hosted_rows = capacity();
        small_size_t chunks = 0;
        while (chunks != max_chunks_k && chunk_schedule_t::rows_before(chunks) < hosted_rows) ++chunks;
        std::size_t const rows = chunk_schedule_t::rows_before(chunks);
        std::size_t bytes = sizeof(dense_ram_index);
        bytes += rows * padded_row_values(frozen) * sizeof(scalar_t);
        if constexpr (!std::is_same_v<vector_norms_t, no_norms_t>) bytes += rows * sizeof(vector_norms_t);
        bytes += rows * frozen.connectivity() * sizeof(node_id_t);
        return bytes;
    }

    /** The stored bytes of one node's vector - the ID-addressed read a key lookup or an export
     *  resolves through; IDs at or above @ref size() name unpublished rows. */
    vector_view_t vector(node_id_t node) const noexcept {
        return vector_view_t {vector_for(chunk_schedule_t::locate(node)).data(), dimensions()};
    }

    /** Storage values per row under the frozen @p geometry - the copy width every caller's
     *  matrix must be as wide as. */
    static std::size_t stored_row_values(geometry_t geometry) noexcept {
        return divide_round_up<std::size_t>(geometry.dimensions(), nk::dimensions_per_value<scalar_t>());
    }

    /** Storage values per row as the chunks hold it: whole steps of the picked quads kit, zero-tailed like the GPU's
     *  fragments, so every step load stays aligned. Checkpoints and copies keep the logical @ref stored_row_values. */
    static std::size_t padded_row_values(geometry_t geometry) noexcept {
        std::size_t const step_values = quads_kit_t::step_dimensions / nk::dimensions_per_value<scalar_t>();
        return round_up_to_multiple(stored_row_values(geometry), step_values);
    }

    /** The matrix shapes @c save() fills. Every extent is logical - the tensors' own strides
     *  account for sub-byte packing - and monotone counters make the loads safe in this order
     *  under any contract. */
    struct saved_shape_t {

        /** Rows of both matrices - the published node count. */
        std::size_t nodes = 0;

        /** Logical dimensions per vector - the vectors matrix's trailing extent. */
        std::size_t dimensions = 0;

        /** Neighbor IDs per neighbors-matrix row - the frozen connectivity. */
        std::size_t connectivity = 0;
    };

    /** The shapes a @c save() of the current contents fills. */
    saved_shape_t saved_shape() const noexcept {
        geometry_t const frozen = geometry();
        std::size_t const count = atomic_ref<node_id_t const>(last_visible_id_).load(std::memory_order_acquire);
        return {count, frozen.dimensions(), frozen.connectivity()};
    }

    /** Where a save lands: two row-major planes the caller sized from @ref saved_shape, handed
     *  over at planning so no step names a destination of its own. */
    struct columns_t {

        /** The vectors plane, one packed row per node. */
        bytes_span_t vectors;

        /** The neighbors plane, @c connectivity IDs per node. */
        bytes_span_t neighbors;
    };

    /** Where a load reads from - the read-only twin of @ref columns_t. */
    struct columns_view_t {

        /** The vectors plane, one packed row per node. */
        bytes_view_t vectors;

        /** The neighbors plane, @c connectivity IDs per node. */
        bytes_view_t neighbors;
    };

#pragma endregion Measures

#pragma region Reservation

    /** Optional tuning knobs for @c insert(); zeros mean "derive from connectivity". */
    struct insert_config_t {

        /** Breadth of the candidate pool during wiring; defaults to twice the connectivity. */
        small_size_t expansion_factor = 0;
    };

    /** Requested geometry & capacity for a @c reserve() call; frozen by the first success. */
    struct reserve_capacity_t {

        /** Minimal number of entries the index must be able to host. */
        std::size_t capacity = 0;

        /** Neighbor slots per node; a power of two tiling whole cache lines. */
        connectivity_t connectivity = connectivity_32_k;

        /** Scalar dimensions per vector; required before the first insertion. */
        std::size_t dimensions = 0;
    };

    /**
     *  @brief Ensures the index can host at least @p config.capacity entries, growing when
     *      needed, and answers the capacity now in effect.
     *
     *  Safe to call concurrently with insertions, searches and other reservations: growth runs
     *  the same grower-election protocol as @c insert, and the geometry freezes exactly once
     *  through a single cold-path compare-exchange. A later call naming a different geometry
     *  receives @c status_t::config_mismatch_k, as does a geometry with no dimensions, with sub-byte
     *  dimensions that leave a storage value partly filled, or with a connectivity that is not a
     *  power of two. Constructors allocate nothing, so every storage
     *  failure surfaces here, in the returned status.
     */
    expected<std::size_t> reserve(reserve_capacity_t config) noexcept {
        // Sub-byte rows fill whole storage values: a ragged tail would be dropped by the kernels' tail loads.
        if (!config.dimensions || config.dimensions % nk::dimensions_per_value<scalar_t>() ||
            !std::has_single_bit(static_cast<u32_t>(config.connectivity)))
            return {0, status_t::config_mismatch_k};

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

    /** The four queue reservations a walk starts from, for one expansion factor and connectivity. */
    struct queue_sizes_t {

        /** The navigation pool - the expansion factor. */
        std::size_t results;

        /** Every neighbor plus the newcomer while back-linking, or the admitted pool while filtering. */
        std::size_t refining_candidates;

        /** Four expansions' worth, refilled by shedding rather than growth. */
        std::size_t frontier;

        /** One node's neighbors per expansion - the first size a re-table doubles from. */
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
     *  @brief Bytes one context reserves for @p config under the frozen geometry; zero before
     *      the geometry freezes.
     *
     *  The first size of every structure a walk touches, reservation for reservation with
     *  @ref reserve_traversal. The visited set may still re-table as a walk widens.
     */
    std::size_t traversal_memory_usage(insert_config_t config = {}) const noexcept {
        geometry_t const frozen = geometry();
        if (!frozen) return 0;
        std::size_t const node_connectivity = frozen.connectivity();
        std::size_t const expansion_factor = non_zero_expansion_factor(config);
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
     *  @brief Sizes @p context 's scratch for the frozen geometry and @p config - the other
     *      fallible half of an insertion, hoistable ahead of a fan-out.
     *
     *  Idempotent and monotone. The pools and the neighbor scratch never allocate again and the
     *  frontier sheds within its reservation; the visited set alone may re-table as a walk widens.
     */
    status_t reserve_traversal(dense_traversal_context_t& context, insert_config_t config = {}) noexcept {
        geometry_t const frozen = geometry();
        if (!frozen) return status_t::uninitialized_k;
        std::size_t const node_connectivity = frozen.connectivity();
        std::size_t const expansion_factor = non_zero_expansion_factor(config);

        // `sorted_buffer::insert(.., limit)` never grows on its own, and `growing_hash_set::set`
        // never rehashes, so these reservations are load-bearing rather than an optimization.
        if (status_t const sized = reserve_queues_(context, queue_sizes_(expansion_factor, node_connectivity));
            failed(sized))
            return sized;
        if (status_t const sized = context.reserve_neighbors(node_connectivity); failed(sized)) return sized;

        // Padded to whole tiles, so the padded pool doubles as the pair-matrix stride.
        if constexpr (has_tiles_k) {
            std::size_t const pool = tiles_pool_(expansion_factor, node_connectivity);
            if (failed(context.reserve_tiles(pool, tiles_kit_t::scratch_bytes(pool, frozen.dimensions()))))
                return status_t::bad_alloc_k;
        }
        return status_t::success_k;
    }

#pragma endregion Reservation

#pragma region Insertion

    /** Telemetry of one @c insert() batch; a batch is atomic, so a failed status leaves no
     *  partial count to report. */
    struct [[nodiscard]] insert_result_t {

        /** Index size visible after this batch fully published; a rejected batch reports the
         *  size it left untouched. */
        std::size_t new_size = 0;

        /** Frontier candidates expanded across all per-node walks of the batch. */
        std::size_t visited_members = 0;

        /** Total distance computations, with padding lanes of 4-way kernels excluded. */
        std::size_t computed_distances = 0;

        /** Subset of @c computed_distances spent in insertion-side diversity selection. */
        std::size_t computed_distances_in_refines = 0;

        /** Subset of @c computed_distances spent re-selecting a full neighbor set while back-linking. */
        std::size_t computed_distances_in_reverse_refines = 0;

        /** Starvation restarts across the batch - a graph-connectivity health signal. */
        std::size_t restarts = 0;

        /** Walks a scratch refusal ended early; their nodes published with fewer neighbors. */
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
     *  @param[out] assigned_ids Either empty, or one slot per incoming vector receiving its node
     *      ID. One batch's IDs are generally @b not contiguous - identity is assigned per node,
     *      interleaved across concurrently inserting threads.
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

        // Without a frozen geometry neither the copy width nor the connectivity is defined.
        geometry_t const frozen = geometry();
        if (!frozen) return {{size()}, status_t::uninitialized_k};

        // The copy below takes its width from the geometry, not from the caller's matrix, so a
        // narrower row would be over-read.
        if (new_vectors.extent(1) != frozen.dimensions()) return {{size()}, status_t::config_mismatch_k};

        if (status_t const sized = reserve_traversal(context, config); failed(sized)) return {{size()}, sized};

        // From here enough sentinel-filled rows exist for every node, and this thread must roll
        // forward to publishing every ID it claims.
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

    /** Entry removals are banned to simplify concurrency management. */
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

        /** The granting index, the rows it granted and the first of them; only @ref claim calls it. */
        capacity_claim_t(dense_ram_index& index, std::size_t rows, node_id_t base) noexcept
            : index_(&index), rows_(rows), base_(base) {}
        capacity_claim_t(capacity_claim_t&& other) noexcept { swap(other); }
        capacity_claim_t& operator=(capacity_claim_t&& other) noexcept { return swap(other), *this; }
        capacity_claim_t(capacity_claim_t const&) = delete;
        capacity_claim_t& operator=(capacity_claim_t const&) = delete;

        /** Releases whatever was never consumed; a consumed claim releases nothing. */
        ~capacity_claim_t() noexcept {
            if (index_ && rows_) index_->release_capacity(rows_);
        }

        void swap(capacity_claim_t& other) noexcept {
            std::swap(index_, other.index_);
            std::swap(rows_, other.rows_);
            std::swap(base_, other.base_);
        }

        /** Rows this claim still holds; zero once an insertion has taken it. */
        std::size_t rows() const noexcept { return rows_; }

        /** The first granted row - where a load plan appends. */
        node_id_t base() const noexcept { return base_; }
        explicit operator bool() const noexcept { return rows_ != 0; }

        /** Marks the rows as taken, so destruction releases nothing - what an insertion
         *  spread over several threads calls once every row has claimed an ID. */
        void consume() noexcept { rows_ = 0; }

      private:

        /** The granting index, reached only to release; null marks the empty state. */
        dense_ram_index* index_ = nullptr;

        /** Rows still held; zero once consumed, so destruction releases nothing. */
        std::size_t rows_ = 0;

        /** The first granted row. */
        node_id_t base_ = 0;
    };

    /** Claims storage for @p count rows without assigning any identity - the step a batch
     *  spanning several indexes runs on every one of them before touching any. */
    expected<capacity_claim_t> claim(std::size_t count) noexcept {
        auto const [first_row, status] = claim_capacity(count);
        if (failed(status)) return {capacity_claim_t {}, status};
        return {capacity_claim_t {*this, count, first_row}, status_t::success_k};
    }

    /**
     *  @brief Returns @p count rows no ID was dispensed against.
     *
     *  Deflating the claim watermark before releasing the capacity is load-bearing: a reader
     *  taking the release with its acquire sees the deflation too, so the physical-rows sum it
     *  forms stays conservative. The deflation releases as well - a posted, order-free subtract
     *  could land after the release. The release lands only on a non-negative counter: a negative
     *  one has a grower inside, and lifting it back across zero would elect a second.
     */
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
        std::size_t const expansion_factor = non_zero_expansion_factor(config);

        // Sub-byte types pack several dimensions per storage value - count storage bytes.
        std::size_t const vector_bytes = stored_row_values(frozen) * sizeof(scalar_t);

        for (small_size_t incoming_index = 0; incoming_index != incoming_vectors_count; ++incoming_index) {
            // Identity is dispensed per node, so each thread holds at most one unpublished ID and
            // the publication gap stays bounded by the thread count. The `acq_rel` order chains
            // the hosting chunk's growth through the all-RMW dispenser to this thread.
            node_id_t const new_node_id =
                atomic_ref<node_id_t>(next_dispensed_id_).fetch_add(1, std::memory_order_acq_rel);
            location_t const new_location = chunk_schedule_t::locate(new_node_id);
            if (report_assigned_ids) assigned_ids[incoming_index] = new_node_id;
            on_assigned(new_node_id);

            // The scratch structures persist across iterations and calls - reset them for this node.
            context.results.clear();
            context.refining_candidates.clear();
            context.frontier.clear();
            context.visited.clear();

            // Norms derive from the owned copy - the exact bytes future distance computations
            // stream. Both sides share `scalar_t`, so this is a byte move, never a conversion.
            vector_view_t const incoming_vector = new_vectors.row(incoming_index).as_vector();
            vector_span_t const owned_vector = vector_span_for(new_location);
            std::memcpy(owned_vector.byte_data(), incoming_vector.byte_data(), vector_bytes);
            vector_norms_t const new_norms = dense_norms(owned_vector);
            if constexpr (!std::is_same_v<vector_norms_t, no_norms_t>) norm_ref_for(new_location) = new_norms;

            // Every ID strictly below the watermark is published - the safe bound for landmark
            // sampling. On a virgin index the node publishes with no neighbors, and later
            // entries wire back to it.
            node_id_t const first_invisible_id =
                atomic_ref<node_id_t>(last_visible_id_).load(std::memory_order_acquire);
            // The row is already claimed, so a refused walk wires a thinner node rather than
            // abandoning an ID nothing will ever publish.
            if (first_invisible_id != 0 &&
                failed(search_around(vector_view_t(owned_vector), new_norms, first_invisible_id, expansion_factor,
                                     context, context.results)))
                ++context.saturated_walks;

            // No reader can arrive before slot 0 turns non-sentinel, so relaxed stores into slots
            // 1.. need no lock and the untouched tail keeps its sentinel termination.
            selection_t& selected = context.selected_neighbors;
            std::size_t const refine_distances = select_diverse(context.results, node_connectivity, selected, context);
            context.computed_distances += refine_distances;
            context.computed_distances_in_refines += refine_distances;
            neighbors_ref_t const new_neighbors = neighbors_for(new_location);
            small_size_t const selected_count = static_cast<small_size_t>(selected.size());
            for (small_size_t slot = 1; slot < selected_count; ++slot)
                new_neighbors.store_slot(slot, selected[slot].node_id);

            // Slot 0 takes the nearest diverse neighbor, or the node's own ID when the graph
            // offered none - the self-loop encoding of "published, zero neighbors", distinct from
            // the all-ones sentinel. The release store is what makes the payload visible.
            node_id_t const first_link_id = selected_count ? selected[0].node_id : new_node_id;
            new_neighbors.unlock(first_link_id);

            // Back-link before the watermark turn: past this ID no thread edits a published row on
            // its behalf, which the snapshot's settlement probe rests on. Every neighbor lock is
            // released inside, so the spin below holds none.
            back_link_neighbors(new_node_id, owned_vector, new_norms, selected, context);

            // The turn runs strictly in dispensed order, keeping the visible prefix hole-free;
            // each thread holds at most one unpublished ID, so the wait is bounded by one wiring
            // per other in-flight thread.
            for (node_id_t watermark = atomic_ref<node_id_t>(last_visible_id_).load(std::memory_order_acquire);
                 watermark != new_node_id;
                 watermark = atomic_ref<node_id_t>(last_visible_id_).load(std::memory_order_acquire))
                // Uncapped: every predecessor's publication store wakes us to re-check.
                wait_type_ {}(&last_visible_id_, watermark, 0u, fu::wait_uncapped_t {});
            atomic_ref<node_id_t>(last_visible_id_).store(new_node_id + 1, std::memory_order_release);
        }
    }

#pragma endregion Insertion

#pragma region Refinement

    /** Optional tuning knobs for @c refine(); zeros mean "derive from connectivity". */
    struct refine_config_t {

        /** Breadth of the re-selection pool; defaults to twice the connectivity, like every walk. */
        small_size_t expansion_factor = 0;
    };

    using refine_result_t = dense_refine_result_t;
    using health_result_t = dense_health_result_t;

    /**
     *  @brief One reachability audit in flight: a landmark sweep over the published nodes, one
     *      level per sweep of every step, until a level expands nothing. Read-only on the graph,
     *      so concurrent insertions merely leave fresh nodes uncounted.
     *
     *  Steps are disjoint first-chunk-sized ID ranges, so any number of threads may @ref sweep
     *  different steps of one level at once; @ref advance_level is the level boundary, one caller.
     *  Reached bits only turn on and a node expands once, so the sweep converges to the exact
     *  reachable set.
     */
    struct health_plan_t {

        /** One unit of a level's fan-out. */
        struct step_t {

            /** The step's position in the plan - what a fan-out hands to a thread. */
            std::size_t position = 0;

            /** The IDs this step sweeps. */
            iota_view<node_id_t> nodes;
        };

        using bitset_allocator_t =
            typename std::allocator_traits<candidates_allocator_t>::template rebind_alloc<u1x64_t>;
        using growing_bitset_t = growing_buffer<u1x64_t, bitset_allocator_t>;

        /** Lanes one word carries - what an ID splits on. */
        inline static constexpr std::size_t lanes_k = std::numeric_limits<u64_t>::digits;

        /** The watermark snapshot the audit covers; IDs strictly below it are swept. */
        node_id_t visible = 0;

        /** Landmark seeds, stride-spread across the published IDs. */
        std::size_t seed_count = 0;

        /** An unbound plan - a refused audit's answer; it owns bitsets, so it names an allocator. */
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

        /** Step @p position of the plan; an empty step past the end. */
        step_t operator[](std::size_t position) const noexcept {
            step_t step;
            if (position >= steps().size()) return step;
            std::size_t const first = position * chunk_schedule_t::rows_in(0);
            std::size_t const bound = min_of(first + chunk_schedule_t::rows_in(0), static_cast<std::size_t>(visible));
            step.position = position;
            step.nodes = {static_cast<node_id_t>(first), static_cast<node_id_t>(bound)};
            return step;
        }

        /** The plan's steps, positionally; every level sweeps all of them. */
        indexed_range<health_plan_t> steps() const noexcept {
            std::size_t const count = index_ ? divide_round_up<std::size_t>(visible, chunk_schedule_t::rows_in(0)) : 0u;
            return {this, count};
        }

        /** Clears the bitsets and re-marks the seeds: the next @ref sweep starts level one. */
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
                // Slot 0 is the lock word, acquired once and waited out; the tail slots are
                // relaxed reads, tolerating a re-linker's stale-but-valid mix.
                node_ids_view_t const row = index_->neighbors_for(chunk_schedule_t::locate(node_id));
                node_id_t first_value = atomic_ref<node_id_t const>(row[0]).load(std::memory_order_acquire);
                while (first_value == dense_node_id_unknown_k) {
                    wait_type_ {}(row.data(), dense_node_id_unknown_k, small_size_t(0), fu::wait_capped_t {});
                    first_value = atomic_ref<node_id_t const>(row[0]).load(std::memory_order_acquire);
                }
                if (first_value == node_id) continue; // Published with no neighbors.
                for (std::size_t slot = 0; slot != row.size(); ++slot) {
                    node_id_t const target =
                        slot ? atomic_ref<node_id_t const>(row[slot]).load(std::memory_order_relaxed) : first_value;
                    if (target == dense_node_id_unknown_k) break;
                    if (target < visible && target != node_id) insert_(reached_, target);
                }
            }
            atomic_ref<std::size_t>(expanded_in_level_).fetch_add(expanded_count, std::memory_order_relaxed);
            return expanded_count;
        }

        /** The level boundary, one caller after every step swept: counts the level and returns
         *  what it expanded; zero means the sweep converged. */
        std::size_t advance_level() noexcept {
            ++sweeps_;
            return std::exchange(expanded_in_level_, 0);
        }

        /** The nodes of @p step the sweep never reached, written into @p scratch in ID order;
         *  the filled prefix is returned, truncated to the scratch's length. */
        std::span<node_id_t const> islands(step_t const& step, node_ids_span_t scratch) const noexcept {
            std::size_t count = 0;
            for (node_id_t node_id = step.nodes.first; node_id < step.nodes.bound && count != scratch.size(); ++node_id)
                if (!contains_(reached_, node_id)) scratch[count++] = node_id;
            return {scratch.data(), count};
        }

        /** The landmark whose share of the ID space holds @p node_id - the nearest seed below it
         *  in ID order, reached by construction since every sweep starts from it. */
        node_id_t landmark_covering(node_id_t node_id) const noexcept {
            if (!seed_count || !visible) return dense_node_id_unknown_k;
            std::size_t const seed = static_cast<std::size_t>(node_id) * seed_count / visible;
            return static_cast<node_id_t>(seed * static_cast<std::size_t>(visible) / seed_count);
        }

        /** Whether @p node_id was reached - readable by any thread between levels. */
        bool reached(node_id_t node_id) const noexcept { return contains_(reached_, node_id); }

        /** The report: the reached population and the levels it took. @c internal_k when a level
         *  is still pending, since the count would then be a lower bound. */
        expected<health_result_t> finish() const noexcept {
            if (!index_) return {{}, status_t::uninitialized_k};
            if (expanded_in_level_) return {{}, status_t::internal_k};
            health_result_t result;
            result.audited_nodes = visible;
            for (u64_t const word : reached_) result.reachable_nodes += static_cast<std::size_t>(std::popcount(word));
            result.sweeps = sweeps_;
            return {result, status_t::success_k};
        }

      private:
        static bool contains_(growing_bitset_t const& words, node_id_t node_id) noexcept {
            u1x64_t const lanes = atomic_ref<u1x64_t const>(words[node_id / lanes_k]).load(std::memory_order_relaxed);
            return (lanes >> (node_id % lanes_k)) & 1u;
        }
        static void insert_(growing_bitset_t& words, node_id_t node_id) noexcept {
            atomic_ref<u1x64_t>(words[node_id / lanes_k])
                .fetch_or(u1x64_t {1} << (node_id % lanes_k), std::memory_order_relaxed);
        }
        static void insert_exclusive_(growing_bitset_t& words, node_id_t node_id) noexcept {
            words[node_id / lanes_k] |= u1x64_t {1} << (node_id % lanes_k);
        }

        /** The audited index; null on a moved-from plan. */
        dense_ram_index const* index_ = nullptr;

        /** One bit per published node: reached by the sweep. */
        growing_bitset_t reached_;

        /** One bit per published node: its neighbors already swept. */
        growing_bitset_t expanded_;

        /** Nodes the current level expanded so far, summed through @c atomic_ref. */
        std::size_t expanded_in_level_ = 0;

        /** Levels completed since the last @ref restart_sweep. */
        std::size_t sweeps_ = 0;
    };

    /** Opens a reachability audit over the published nodes, seeded from as many stride-spread
     *  landmarks as one search draws, so the reachable set is one search's reachable world. The
     *  caller sweeps every step per level until a level expands nothing, then reads @c finish().
     *  The bitsets come from @p context 's allocator. */
    expected<health_plan_t> plan_health(dense_traversal_context_t& context) const noexcept {
        using growing_bitset_t = typename health_plan_t::growing_bitset_t;
        using bitset_allocator_t = typename health_plan_t::bitset_allocator_t;
        node_id_t const visible = atomic_ref<node_id_t const>(last_visible_id_).load(std::memory_order_acquire);
        // An empty graph is audited, not refused: no words to sweep, no steps, zero reached.
        bitset_allocator_t const allocator {context.allocator_};
        std::size_t const words = divide_round_up<std::size_t>(visible, health_plan_t::lanes_k);
        growing_bitset_t reached {allocator}, expanded {allocator};
        if (failed(reached.resize(words)) || failed(expanded.resize(words)))
            return {health_plan_t {allocator}, status_t::bad_alloc_k};
        std::size_t const seed_count = min_of<std::size_t>(connectivity(), visible);
        return {health_plan_t {*this, visible, seed_count, std::move(reached), std::move(expanded)},
                status_t::success_k};
    }

    /**
     *  @brief One refinement pass in flight: the planned range is re-selected, reachability swept level by level, and
     *      the islands re-selected and reconnected round by round while their count shrinks. An empty range heals.
     *
     *  Steps of the current phase are disjoint ID ranges, so any number of threads may @ref run them at once;
     *  @ref advance is the single-caller barrier between phases and answers whether another phase follows.
     *  Re-selection and reconnection edit under the per-node locks like an insertion, so searches run throughout and a
     * pass stopped between steps leaves the graph correct. Insertions may not overlap.
     */
    struct refine_plan_t {
        using step_t = typename health_plan_t::step_t;

        /** The audit the heal phases sweep. */
        health_plan_t audit;

        /** An unbound pass over @p audit - what a refused plan answers with. */
        explicit refine_plan_t(health_plan_t&& audit) noexcept : audit(std::move(audit)) {}
        refine_plan_t(dense_ram_index& index, health_plan_t&& audit, iota_view<node_id_t> nodes,
                      std::size_t expansion_factor) noexcept
            : audit(std::move(audit)), index_(&index), nodes_(nodes), expansion_factor_(expansion_factor),
              phase_(nodes.empty() ? phase_t::sweep_k : phase_t::reselect_k) {}
        refine_plan_t(refine_plan_t&& other) noexcept
            : audit(std::move(other.audit)), index_(std::exchange(other.index_, nullptr)), nodes_(other.nodes_),
              expansion_factor_(other.expansion_factor_), phase_(other.phase_), swept_after_(other.swept_after_),
              islands_before_(other.islands_before_), tally_(other.tally_) {}
        refine_plan_t(refine_plan_t const&) = delete;
        refine_plan_t& operator=(refine_plan_t const&) = delete;

        /** Step @p position of the current phase; an empty step past the end. */
        step_t operator[](std::size_t position) const noexcept {
            if (phase_ != phase_t::reselect_k) return audit[position];
            step_t step;
            if (position >= steps().size()) return step;
            std::size_t const first = nodes_.first + (position * chunk_schedule_t::rows_in(0));
            std::size_t const bound = min_of(first + chunk_schedule_t::rows_in(0),
                                             static_cast<std::size_t>(nodes_.bound));
            step.position = position;
            step.nodes = {static_cast<node_id_t>(first), static_cast<node_id_t>(bound)};
            return step;
        }

        /** The current phase's steps, positionally; none once the pass is over. */
        indexed_range<refine_plan_t> steps() const noexcept {
            if (!index_ || phase_ == phase_t::done_k) return {this, 0u};
            if (phase_ != phase_t::reselect_k) return {this, audit.steps().size()};
            return {this, divide_round_up<std::size_t>(nodes_.size(), chunk_schedule_t::rows_in(0))};
        }

        /** Runs @p step of the current phase on the calling thread - safe for disjoint steps of one phase. */
        status_t run(step_t const& step, dense_traversal_context_t& context) noexcept {
            if (!index_) return status_t::uninitialized_k;
            switch (phase_) {
            case phase_t::reselect_k:
            case phase_t::reselect_islands_k: return reselect_(step, context);
            case phase_t::sweep_k: audit.sweep(step); return status_t::success_k;
            case phase_t::reconnect_k: reconnect_(step, context); return status_t::success_k;
            default: return status_t::success_k;
            }
        }

        /** The barrier between phases, one caller after every step of the current one ran: picks the next phase
         *  and answers whether one follows. */
        bool advance() noexcept {
            if (!index_) return false;
            if (phase_ == phase_t::sweep_k) {
                ++tally_.sweeps;
                if (!audit.advance_level()) converge_();
            }
            else if (phase_ != phase_t::done_k) {
                audit.restart_sweep();
                swept_after_ = phase_;
                phase_ = phase_t::sweep_k;
            }
            return phase_ != phase_t::done_k;
        }

        /** Whether the current phase re-selects the planned range - the phase a caller's progress counts. */
        bool reselecting() const noexcept { return phase_ == phase_t::reselect_k; }

        /** The pass so far: complete once @ref advance answered false, otherwise @c cancelled_k. */
        expected<refine_result_t> finish() const noexcept {
            if (!index_) return {{}, status_t::uninitialized_k};
            return {tally_, phase_ == phase_t::done_k ? status_t::success_k : status_t::cancelled_k};
        }

      private:
        enum class phase_t : u8_t { reselect_k, sweep_k, reselect_islands_k, reconnect_k, done_k };

        static void add_(std::size_t& into, std::size_t value) noexcept {
            atomic_ref<std::size_t>(into).fetch_add(value, std::memory_order_relaxed);
        }

        /** Picks what follows a converged audit, from the phase the audit followed. */
        void converge_() noexcept {
            auto const [health, status] = audit.finish();
            std::size_t const left = succeeded(status) ? audit.visible - health.reachable_nodes : 0;
            tally_.reachable_nodes = health.reachable_nodes;
            if (swept_after_ == phase_t::reselect_k) {
                tally_.islands_found = left;
                phase_ = left ? phase_t::reselect_islands_k : phase_t::done_k;
            }
            else if (swept_after_ == phase_t::reselect_islands_k) {
                phase_ = left ? phase_t::reconnect_k : phase_t::done_k;
            }
            // A round that frees no island frees none on the next either, since the picks it needs are unreached.
            else { phase_ = left && left < islands_before_ ? phase_t::reselect_islands_k : phase_t::done_k; }
            if (swept_after_ != phase_t::reselect_islands_k) islands_before_ = left;
        }

        /** Re-selects every node of @p step, or only its islands once the heal rounds began. */
        status_t reselect_(step_t const& step, dense_traversal_context_t& context) noexcept {
            if (status_t const sized = index_->reserve_traversal(
                    context, insert_config_t {static_cast<small_size_t>(expansion_factor_)});
                failed(sized))
                return sized;
            std::size_t const visited_members_before = context.visited_members;
            std::size_t const computed_distances_before = context.computed_distances;
            std::size_t const refines_before = context.computed_distances_in_refines;
            std::size_t const reverse_refines_before = context.computed_distances_in_reverse_refines;
            std::size_t const restarts_before = context.restarts;
            std::size_t const saturated_walks_before = context.saturated_walks;
            std::size_t visited = 0, changed = 0;
            for (node_id_t node_id = step.nodes.first; node_id < step.nodes.bound; ++node_id) {
                if (phase_ == phase_t::reselect_islands_k && audit.reached(node_id)) continue;
                changed += index_->refine_one_(node_id, audit.visible, expansion_factor_, context);
                ++visited;
            }
            add_(tally_.visited_nodes, visited);
            add_(tally_.changed_nodes, changed);
            add_(tally_.visited_members, context.visited_members - visited_members_before);
            add_(tally_.computed_distances, context.computed_distances - computed_distances_before);
            add_(tally_.computed_distances_in_refines, context.computed_distances_in_refines - refines_before);
            add_(tally_.computed_distances_in_reverse_refines,
                 context.computed_distances_in_reverse_refines - reverse_refines_before);
            add_(tally_.restarts, context.restarts - restarts_before);
            add_(tally_.saturated_walks, context.saturated_walks - saturated_walks_before);
            return status_t::success_k;
        }

        /** Reconnects every island in @p step to its nearest reached neighbor - a free slot when there is one,
         *  otherwise a landing spread across the tail, so islands sharing one target heal in one round. */
        void reconnect_(step_t const& step, dense_traversal_context_t& context) noexcept {
            std::size_t reconnected = 0;
            small_size_t const connectivity = static_cast<small_size_t>(index_->connectivity());
            small_size_t const spread = min_of<small_size_t>(16u, connectivity / 2);
            for (node_id_t island = step.nodes.first; island < step.nodes.bound; ++island) {
                if (audit.reached(island)) continue;
                neighbors_ref_t const own_neighbors = index_->neighbors_for(chunk_schedule_t::locate(island));
                small_size_t const staged_count = index_->stage_neighbors_(own_neighbors, island, context);
                node_id_t target = dense_node_id_unknown_k;
                for (small_size_t staged = 0; staged != staged_count && target == dense_node_id_unknown_k; ++staged) {
                    node_id_t const pick = context.staged_ids[staged];
                    if (pick < audit.visible && audit.reached(pick)) target = pick;
                }
                // When a cluster picks only its own, the last resort is the landmark covering this
                // ID: a walk starts there by definition, which is what makes it reached.
                if (target == dense_node_id_unknown_k) target = audit.landmark_covering(island);
                if (target == dense_node_id_unknown_k || target == island) continue;

                // Under the target's lock: already wired, a free slot, or the spread landing.
                neighbors_ref_t const target_neighbors = index_->neighbors_for(chunk_schedule_t::locate(target));
                node_id_t const displaced = target_neighbors.lock();
                if (displaced == target) { // Published with no neighbors: the island becomes the only one.
                    target_neighbors.unlock(island);
                    ++reconnected;
                    continue;
                }
                small_size_t landing = connectivity;
                bool present = displaced == island;
                for (small_size_t slot = 1; slot != connectivity && !present; ++slot) {
                    node_id_t const resident = target_neighbors.load_slot(slot);
                    present = resident == island;
                    if (resident == dense_node_id_unknown_k) {
                        landing = slot;
                        break;
                    }
                }
                // No free slot: a tail slot spread by ID, never one a sibling island landed in this round.
                for (small_size_t attempt = 0; landing == connectivity && attempt != spread; ++attempt) {
                    small_size_t const slot = connectivity - 1 - static_cast<small_size_t>((island + attempt) % spread);
                    node_id_t const resident = target_neighbors.load_slot(slot);
                    if (resident < audit.visible && audit.reached(resident)) landing = slot;
                }
                if (landing == connectivity) {
                    target_neighbors.unlock(displaced);
                    continue;
                }
                if (!present) {
                    target_neighbors.store_slot(landing, island);
                    ++reconnected;
                }
                target_neighbors.unlock(displaced);
            }
            add_(tally_.islands_reconnected, reconnected);
        }

        /** The refined index; null on a moved-from plan. */
        dense_ram_index* index_ = nullptr;

        /** The IDs the first phase re-selects, clamped to the audit's watermark. */
        iota_view<node_id_t> nodes_;

        /** Breadth of every re-selection walk. */
        std::size_t expansion_factor_ = 0;

        /** The phase whose steps run now. */
        phase_t phase_ = phase_t::done_k;

        /** The phase the current audit follows - what decides the phase after it converges. */
        phase_t swept_after_ = phase_t::reselect_k;

        /** Islands the last round started from - the shrinking the rounds require. */
        std::size_t islands_before_ = 0;

        /** Counts summed through @c atomic_ref by the steps, and set by @ref advance. */
        refine_result_t tally_;
    };

    /** Opens a refinement pass over @p nodes, clamped to the published IDs, with a fresh audit seeded like
     *  @ref plan_health; an empty range heals alone. The bitsets come from @p context 's allocator. */
    expected<refine_plan_t> plan_refine(iota_view<node_id_t> nodes, dense_traversal_context_t& context,
                                        refine_config_t config = {}) noexcept {
        auto [audit, status] = plan_health(context);
        if (failed(status)) return {refine_plan_t {std::move(audit)}, status};
        if (!geometry()) return {refine_plan_t {std::move(audit)}, status_t::uninitialized_k};
        node_id_t const bound = min_of(nodes.bound, audit.visible);
        iota_view<node_id_t> const planned {min_of(nodes.first, bound), bound};
        return {refine_plan_t {*this, std::move(audit), planned, non_zero_expansion_factor(config)},
                status_t::success_k};
    }

    /**
     *  @brief Re-selects the neighbors of every published node in @p nodes and heals whatever that strands - one
     *      @ref refine_plan_t drained on the calling thread. An empty range heals alone.
     *
     *  @param[in] nodes The ID range to re-select; IDs at or past the watermark are skipped.
     *  @param[inout] context The calling thread's scratch & telemetry, reused across passes.
     *  @param[in] config Optional tuning knobs; zeros mean "derive from connectivity".
     */
    expected<refine_result_t> refine(iota_view<node_id_t> nodes, dense_traversal_context_t& context,
                                     refine_config_t config = {}) noexcept {
        auto [plan, status] = plan_refine(nodes, context, config);
        if (failed(status)) return {{}, status};
        do
            for (typename refine_plan_t::step_t const step : plan.steps())
                if (status_t const ran = plan.run(step, context); failed(ran)) return {plan.finish().value, ran};
        while (plan.advance());
        return plan.finish();
    }

#pragma endregion Refinement

#pragma region Search

    /** Optional tuning knobs for @c search(); zeros mean "derive from connectivity". */
    struct search_config_t {

        /** Breadth of the candidate pool; defaults to twice the connectivity. Raise it to trade
         *  latency for recall - the caller reranks anyway. */
        small_size_t expansion_factor = 0;

        /** Reader contract; @c immutable_k - or stronger - drops every atomic in the traversal
         *  when the caller certifies no concurrent mutators. */
        synchronization_t synchronization = synchronization_t::concurrent_k;
    };

    /** Telemetry of one @c search() call; matches stay in the caller's context. */
    struct [[nodiscard]] search_result_t {

        /** Matches left in @c context.results, ascending by distance to the query. */
        std::size_t count = 0;

        /** Frontier candidates expanded during the walk. */
        std::size_t visited_members = 0;

        /** Total distance computations, with padding lanes of 4-way kernels excluded. */
        std::size_t computed_distances = 0;

        /** Starvation restarts - a graph-connectivity health signal. */
        std::size_t restarts = 0;

        /** One when a scratch refusal ended the walk early, leaving the ranking short. */
        std::size_t saturated_walks = 0;
    };

    /**
     *  @brief Finds the nearest published neighbors of @p query_vector - the caller reranks or
     *      trims the prefix it needs.
     *
     *  Under the default @c concurrent_k contract it is safe to call concurrently with insertions
     *  and other searches: the traversal reads only nodes below the visibility watermark,
     *  acquire-loads every slot-0 lock before scanning a node's neighbors, retries transiently
     *  locked nodes and never writes. IDs above this query's watermark snapshot are skipped by the
     *  translation-frontier invariant and picked up by any later query with a fresher one. Under
     *  @c immutable_k - or stronger - the caller certifies no mutators and the traversal runs
     *  atomics-free. @ref dense_ram_index_concurrency
     *
     *  @param[in] query_vector The query, in the index's own dtype & frozen dimensionality.
     *  @param[inout] context The calling thread's scratch & telemetry: the matches end up in
     *      @c context.results, ascending by distance - the nearest first.
     *  @param[in] config Optional tuning knobs; zeros mean "derive from connectivity".
     */
    expected<search_result_t> search(vector_view_t query_vector, dense_traversal_context_t& context,
                                     search_config_t config = {}) noexcept {
        return search(query_vector, dummy_fold_t {}, context, config);
    }

    /**
     *  @brief Finds the published nodes a lightning indexer scores highest for one query row - the
     *      @c search contract, with every distance `1 −` the node's indexer score.
     *
     *  @param[in] query_vector The row the walk navigates by, conventionally `Σₕ wₕ·qₕ`.
     *  @param[in] heads The indexer heads of this one query row.
     *  @param[inout] context The calling thread's scratch & telemetry: the matches end up in
     *      @c context.results, ascending by distance - the highest score first.
     *  @param[in] config Optional tuning knobs; zeros mean "derive from connectivity".
     */
    expected<search_result_t> search(vector_view_t query_vector, dense_indexer_heads<scalar_t> const& heads,
                                     dense_traversal_context_t& context, search_config_t config = {}) noexcept
        requires(kernel_kind_ == nk_kernel_dot_k && !nk::is_integral_dtype<scalar_t>())
    {
        if (config.synchronization != synchronization_t::concurrent_k)
            return search_typed<synchronization_t::immutable_k>(query_vector, context, config, dummy_fold_t {},
                                                                context.results, heads);
        return search_typed<synchronization_t::concurrent_k>(query_vector, context, config, dummy_fold_t {},
                                                             context.results, heads);
    }

    /**
     *  @brief Walks exactly like @c search while @p fold reads every quad the walk scores - the
     *      entry point for verbs that reduce a walk to something other than its nearest matches.
     *
     *  @param[in] query_vector The query, in the index's own dtype & frozen dimensionality.
     *  @param[inout] fold The reader of every scored quad - see @c dense_fold.
     *  @param[inout] context The calling thread's scratch & telemetry: the nearest published
     *      candidates end up in @c context.results, ascending by distance.
     *  @param[in] config Optional tuning knobs; zeros mean "derive from connectivity".
     */
    template <typename fold_type_>
        requires dense_fold<std::remove_cvref_t<fold_type_>, dot_result_t>
    expected<search_result_t> search(vector_view_t query_vector, fold_type_&& fold, dense_traversal_context_t& context,
                                     search_config_t config = {}) noexcept {
        if (config.synchronization != synchronization_t::concurrent_k)
            return search_typed<synchronization_t::immutable_k>(query_vector, context, config, fold, context.results,
                                                                dummy_heads_t {});
        return search_typed<synchronization_t::concurrent_k>(query_vector, context, config, fold, context.results,
                                                             dummy_heads_t {});
    }

    /**
     *  @brief Finds the nearest published neighbors of @p query_vector that satisfy
     *      @p predicate - the filtered sibling of @c search, sharing its whole contract.
     *
     *  Non-matching nodes still score and route the walk - filtering navigation itself would
     *  strand it in disconnected islands - but only matching nodes enter the reported pool, whose
     *  radius governs termination once it holds @c expansion_factor entries; until then the
     *  ordinary navigation radius bounds the frontier. Predicates come in two shapes, detected by
     *  signature: per-node @c admission_t(dense_node_id_t), or batched - the scored quad and an
     *  admissions buffer, pushing what it admits. Each node is tested at most once per query, so
     *  the exact visited set is the predicate's memoization.
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
        // Matches admit into the re-selection buffer, idle during any search and reserved beside the
        // navigation pool; it bounds the walk once full, then swaps into `context.results`.
        context.refining_candidates.clear();
        dense_filtering_fold<std::remove_reference_t<predicate_type_>, results_array_t> filtering {
            predicate, context.refining_candidates, non_zero_expansion_factor(config)};
        expected<search_result_t> const result =
            config.synchronization != synchronization_t::concurrent_k
                ? search_typed<synchronization_t::immutable_k>(query_vector, context, config, filtering,
                                                               context.refining_candidates, dummy_heads_t {})
                : search_typed<synchronization_t::concurrent_k>(query_vector, context, config, filtering,
                                                                context.refining_candidates, dummy_heads_t {});
        std::swap(context.results, context.refining_candidates);
        return result;
    }

    /**
     *  @brief Graph products of one left-hand row: every pair the walk scores, node IDs beside native
     *      dots, written until the spans fill.
     *
     *  @param[in] left The left-hand row, in the index's own dtype & frozen dimensionality.
     *  @param[out] node_ids The node ID of each kept pair, in walk order.
     *  @param[out] products The dot of each kept pair, parallel to @p node_ids.
     *  @param[inout] context The calling thread's scratch & telemetry.
     *  @param[in] config Optional tuning knobs; zeros mean "derive from connectivity".
     *  @return The pairs scored in @c count - above the spans' width when they were truncated.
     */
    expected<search_result_t> multiply(vector_view_t left, std::span<dense_node_id_t> node_ids,
                                       std::span<dot_result_t> products, dense_traversal_context_t& context,
                                       search_config_t config = {}) noexcept
        requires(kernel_kind_ == nk_kernel_dot_k)
    {
        if (products.size() != node_ids.size()) return {{}, status_t::config_mismatch_k};
        dense_multiply_fold<dense_node_id_t, dot_result_t> fold {node_ids, products};
        expected<search_result_t> walked = search(left, fold, context, config);
        walked.value.count = fold.scored;
        return walked;
    }

    /**
     *  @brief Softmax attention of one query row over its most relevant published nodes: @c search's walk
     *      pools them, and the softmax weighs the best @p attended.size() of the pool, each value row
     *      gathered once.
     *
     *  @param[in] query The query row, in the index's own dtype & frozen dimensionality.
     *  @param[inout] context The calling thread's scratch & telemetry; the pool stays in @c context.results.
     *  @param[in] values One value row per node ID, stored like the keys, at least as many rows as the index holds.
     *  @param[in] score_scale The factor turning a dot into a logit.
     *  @param[out] attended The picks' node IDs, nearest first; its size is the "k".
     *  @param[out] output The attended row, @c values.extent(1) wide.
     *  @param[out] state The row's softmax carry, for recombining with other walks.
     *  @param[in] config Optional tuning knobs; zeros mean "derive from connectivity".
     *  @return The picks landed in @p attended, in @c count - fewer than k only when the pool holds fewer.
     */
    expected<search_result_t> attend(vector_view_t query, dense_traversal_context_t& context, vectors_view_t values,
                                     f32_t score_scale, std::span<dense_node_id_t> attended, std::span<f32_t> output,
                                     dense_softmax_state_t& state, search_config_t config = {}) noexcept
        requires(kernel_kind_ == nk_kernel_dot_k)
    {
        if (output.size() != values.extent(1)) return {{}, status_t::config_mismatch_k};
        if (values.extent(0) < size()) return {{}, status_t::shape_mismatch_k};
        dense_attend_prefetch_fold<scalar_t, cache_hints_type_, results_array_t> fold {values, context.results,
                                                                                       attended.size()};
        expected<search_result_t> walked = search(query, fold, context, config);
        if (!walked) return walked;
        results_array_t const& pool = context.results;
        std::size_t const count = min_of(attended.size(), pool.size());
        // The dot kernel's distance is `1 − dot`, so the logit inverts it rather than re-scoring the key.
        auto const [softmax, status] = dense_attend_picks<scalar_t, capability_>(
            count, [&](std::size_t pick) noexcept { return score_scale * (1 - pool[pick].distance); },
            [&](std::size_t pick) noexcept { return std::size_t {pool[pick].node_id}; }, values, output);
        if (failed(status)) return {walked.value, status};
        for (std::size_t pick = 0; pick != count; ++pick) attended[pick] = pool[pick].node_id;
        state = softmax;
        walked.value.count = count;
        return walked;
    }

    /** The compile-time form of @c search() & @c search_if(): one whole-traversal instantiation per
     *  @p synchronization_ contract, per fold type and per heads type, so a certified contract carries
     *  no atomics and @c dummy_fold_t & @c dummy_heads_t inline away. The walk ends against @p bound
     *  once it fills - @c context.results itself for every walk but a filtered one. */
    template <synchronization_t synchronization_, typename fold_type_, typename heads_type_>
        requires dense_fold<std::remove_cvref_t<fold_type_>, dot_result_t>
    expected<search_result_t> search_typed(vector_view_t query_vector, dense_traversal_context_t& context,
                                           search_config_t config, fold_type_&& fold, results_array_t const& bound,
                                           heads_type_ const& heads) noexcept {
        // Every distance kernel takes its width from the geometry, so a narrower query would be over-read.
        geometry_t const frozen = geometry();
        if (frozen && query_vector.size() != frozen.dimensions()) return {{}, status_t::config_mismatch_k};
        std::size_t const expansion_factor = non_zero_expansion_factor(config);

        // The same reservations `insert` makes, which also size the buffer `search_if` admits into.
        if (status_t const sized = reserve_queues_(context, queue_sizes_(expansion_factor, connectivity()));
            failed(sized))
            return {{}, sized};
        context.results.clear();
        context.frontier.clear();
        context.visited.clear();

        // Everything strictly below the watermark is published. A mutation-free contract needs no
        // acquire - the caller's own synchronization already ordered every publication.
        constexpr std::memory_order watermark_order_k = synchronization_ == synchronization_t::concurrent_k
                                                            ? std::memory_order_acquire
                                                            : std::memory_order_relaxed;
        node_id_t const first_invisible_id = atomic_ref<node_id_t>(last_visible_id_).load(watermark_order_k);
        // An unfrozen geometry read before a racing first insert left the query width unchecked.
        if (first_invisible_id == 0 || !frozen) return {{}, status_t::success_k};

        // Snapshot the context's running telemetry, to report this query's share on return.
        std::size_t const visited_members_before = context.visited_members;
        std::size_t const computed_distances_before = context.computed_distances;
        std::size_t const restarts_before = context.restarts;
        std::size_t const saturated_walks_before = context.saturated_walks;

        vector_norms_t const query_norms = dense_norms(query_vector);
        status_t const walked = search_around<synchronization_>(query_vector, query_norms, first_invisible_id,
                                                                expansion_factor, context, bound, fold, {}, heads);

        // A refused walk still ranked what it reached; the counter says the ranking is short
        // because scratch ran out, not because the graph had nothing better.
        if (failed(walked)) ++context.saturated_walks;
        search_result_t result;
        result.count = bound.size();
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
     *  edits an earlier row; the certified arms read plain, and a violated certification surfaces
     *  as @c contended_k rather than a silently torn snapshot.
     */
    struct [[nodiscard]] save_plan_t {

        /** Where the rows land, vetted against the watermark at planning. */
        columns_t columns {};

        /** What an ID under the watermark moves by - the row this shard starts at. */
        node_id_t id_shift = 0;

        /** The contract the caller certified. */
        synchronization_t synchronization = synchronization_t::concurrent_k;

        /** Rows under the watermark at planning. */
        node_id_t visible = 0;

        /** The dispenser at planning; equal to @c visible, else no plan was made. */
        node_id_t dispensed = 0;

        /** The allocation counter at planning - the growth sensor the probe re-reads. */
        node_id_t allocated = 0;

        /** Rows whose vectors landed, tallied through @c atomic_ref. */
        std::size_t covered_vectors = 0;

        /** Rows whose neighbors landed, tallied through @c atomic_ref. */
        std::size_t covered_neighbors = 0;

        save_plan_t() noexcept = default;

        /** The index being snapshotted into @p columns under @p synchronization, its IDs shifted
         *  by @p id_shift; only @ref plan_save calls it. */
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

        /** A save owns nothing - no rows claimed, no lock taken - so a dying plan leaves the index
         *  as it found it. */
        ~save_plan_t() noexcept = default;

        /** One step of the plan: one storage chunk's share of the rows under the watermark. */
        struct step_t {

            /** The step's position in the plan - what a fan-out hands to a thread. */
            std::size_t position = 0;

            /** Where the rows sit in the index, in its own ID space. */
            iota_view<node_id_t> index_rows;

            /** Where they sit in the caller's buffer, counted from its first row. */
            iota_view<node_id_t> source_rows;

            /** Bytes this step's vector rows occupy inside the plan's vectors column. */
            std::size_t vectors_bytes = 0;

            /** Bytes this step's neighbor rows occupy inside the plan's neighbors column. */
            std::size_t neighbors_bytes = 0;
        };

        /** Step @p position of the plan; an empty one past the last. */
        step_t operator[](std::size_t position) const noexcept {
            step_t step;
            if (position >= steps().size()) return step;
            std::size_t const first = chunk_schedule_t::rows_before(static_cast<small_size_t>(position));
            std::size_t const bound = min_of(chunk_schedule_t::rows_before(static_cast<small_size_t>(position + 1)),
                                             static_cast<std::size_t>(visible));
            geometry_t const frozen = index_->geometry();
            step.position = position;
            step.index_rows = {static_cast<node_id_t>(first), static_cast<node_id_t>(bound)};
            step.source_rows = step.index_rows;
            step.vectors_bytes = step.index_rows.size() * stored_row_values(frozen) * sizeof(scalar_t);
            step.neighbors_bytes = step.index_rows.size() * frozen.connectivity() * sizeof(node_id_t);
            return step;
        }

        /** The steps the save spans - one per storage chunk it reaches; none once spent. */
        indexed_range<save_plan_t> steps() const noexcept {
            std::size_t const count = index_ && visible
                                          ? chunk_schedule_t::locate(static_cast<node_id_t>(visible - 1)).chunk + 1u
                                          : 0u;
            return {this, count};
        }

        /**
         *  @brief Copies step @p step 's rows into the plan's own columns: the vectors straight
         *      from chunk memory, since published vectors never change, and the neighbors rebased
         *      through @p scratch.
         *
         *  Every ID under the watermark is shifted by @c id_shift, every other folded to the sentinel.
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

            location_t const location = chunk_schedule_t::locate(rows.first);
            vectors_view_t const vectors = index_->vectors_chunks_[location.chunk](
                nk::range(location.offset, location.offset + rows.size()), nk::slice);
            // Chunk rows are padded to whole kit steps, while checkpoints keep logical rows.
            std::size_t const row_bytes = row_values * sizeof(scalar_t);
            for (std::size_t row = 0; row != rows.size(); ++row) {
                [[maybe_unused]] status_t const copied = copy_bytes(
                    columns.vectors.subspan(vectors_first + row * row_bytes, row_bytes),
                    bytes_of(vectors.row(row).data(), row_values));
            }
            atomic_ref<std::size_t>(covered_vectors).add(rows.size(), std::memory_order_release);

            node_id_t const* source = index_->neighbors_chunks_[location.chunk].view().row(location.offset).data();
            node_id_t* staged = scratch.data();
            if (synchronization == synchronization_t::concurrent_k)
                for (std::size_t word = 0; word != step_slots; ++word)
                    staged[word] = atomic_ref<node_id_t const>(source[word]).load(std::memory_order_relaxed);
            else std::memcpy(staged, source, step_slots * sizeof(node_id_t));
            for (std::size_t word = 0; word != step_slots; ++word)
                staged[word] = staged[word] < visible ? static_cast<node_id_t>(staged[word] + id_shift)
                                                      : dense_node_id_unknown_k;
            [[maybe_unused]] status_t const copied_neighbors = copy_bytes(
                columns.neighbors.subspan(neighbors_first, step.neighbors_bytes), bytes_of(staged, step_slots));
            atomic_ref<std::size_t>(covered_neighbors).add(rows.size(), std::memory_order_release);
            return status_t::success_k;
        }

        /** Pronounces the snapshot complete: every chunk's vectors and neighbors landed, and
         *  under @c concurrent_k the counters rested across the whole plan. A success spends the
         *  plan; a refusal leaves it whole, so the missing steps can still land. */
        status_t finish() noexcept {
            if (!index_) return status_t::uninitialized_k;
            std::size_t const nodes = static_cast<std::size_t>(visible);
            if (atomic_ref<std::size_t>(covered_vectors).load(std::memory_order_acquire) != nodes ||
                atomic_ref<std::size_t>(covered_neighbors).load(std::memory_order_acquire) != nodes)
                return status_t::config_mismatch_k;
            if (synchronization == synchronization_t::concurrent_k) {
                // Orders the staged reads before the re-reads below; pairs with the release fence
                // heading `back_link_neighbors`, so an overlapped row edit surfaces here.
                std::atomic_thread_fence(std::memory_order_acquire);
                bool const settled =
                    atomic_ref<node_id_t const>(index_->last_visible_id_).load(std::memory_order_relaxed) == visible &&
                    atomic_ref<node_id_t const>(index_->next_dispensed_id_).load(std::memory_order_relaxed) ==
                        dispensed &&
                    atomic_ref<node_id_t const>(index_->last_allocated_id_).load(std::memory_order_relaxed) ==
                        allocated;
                if (!settled) return status_t::contended_k;
            }
            index_ = nullptr;
            return status_t::success_k;
        }

      private:

        /** The index being snapshotted; null in the empty plan a refused @ref plan_save returns,
         *  and once @c finish succeeded. */
        dense_ram_index const* index_ = nullptr;
    };

    /**
     *  @brief Opens a save into @p columns under @p synchronization: takes the watermark and, under
     *      @c concurrent_k, refuses while an insert is mid-flight.
     *
     *  The settlement probe reads the trailing counter first, since visible ≤ dispensed at every
     *  instant. @c exclusive_k reads like @c immutable_k. Columns too short for the rows under the
     *  watermark are refused here, so no step ever writes past them.
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
        plan.visible = atomic_ref<node_id_t const>(last_visible_id_).load(std::memory_order_acquire);
        plan.dispensed = atomic_ref<node_id_t const>(next_dispensed_id_).load(std::memory_order_acquire);
        plan.allocated = atomic_ref<node_id_t const>(last_allocated_id_).load(std::memory_order_acquire);
        if (plan.visible != plan.dispensed) return {{}, status_t::contended_k};
        std::size_t const rows = static_cast<std::size_t>(plan.visible);
        if (columns.vectors.size() < rows * stored_row_values(frozen) * sizeof(scalar_t) ||
            columns.neighbors.size() < rows * frozen.connectivity() * sizeof(node_id_t))
            return {{}, status_t::config_mismatch_k};
        return {std::move(plan), status_t::success_k};
    }

    /** Serializes every row under the watermark into @p columns on the calling thread - the
     *  one-shot over @ref plan_save for a caller with no threads to fan the steps out over. */
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
     *  rows ride in: one that dies before @c finish releases them, so the next plan appends at
     *  the same base and the rows it half-filled are never published.
     */
    struct [[nodiscard]] load_plan_t {

        /** Where the rows come from, vetted against them at planning. */
        columns_view_t columns {};

        /** The ID space the source's rows speak. */
        iota_view<node_id_t> source_range {};

        /** The first appended ID - the row the claim landed at. */
        node_id_t base = 0;

        /** Rows the plan appends. */
        std::size_t rows = 0;

        /** Rows whose vectors landed, tallied through @c atomic_ref. */
        std::size_t covered_vectors = 0;

        /** Rows whose neighbors landed, tallied through @c atomic_ref. */
        std::size_t covered_neighbors = 0;

        load_plan_t() noexcept = default;

        /** The index appended to at @p base, the @p columns the rows come from under @p source_range,
         *  and the claim they ride in - empty for a plan of no rows; only @ref plan_load calls it. */
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

        /** The claim's own death releases the rows, unless @c finish consumed them. */
        ~load_plan_t() noexcept = default;

        /** One step of the plan: one storage chunk's share of the appended rows. */
        struct step_t {

            /** The step's position in the plan - what a fan-out hands to a thread. */
            std::size_t position = 0;

            /** Where the rows land in the index, in its own ID space, from @c base. */
            iota_view<node_id_t> index_rows;

            /** Where they come from in the caller's buffer, from its first row. */
            iota_view<node_id_t> source_rows;

            /** Bytes this step's vector rows occupy inside the plan's vectors column. */
            std::size_t vectors_bytes = 0;

            /** Bytes this step's neighbor rows occupy inside the plan's neighbors column. */
            std::size_t neighbors_bytes = 0;
        };

        /** Step @p position of the plan; an empty one past the last. */
        step_t operator[](std::size_t position) const noexcept {
            step_t step;
            if (position >= steps().size()) return step;
            small_size_t const first_chunk = chunk_schedule_t::locate(base).chunk;
            std::size_t const first = max_of(
                chunk_schedule_t::rows_before(static_cast<small_size_t>(first_chunk + position)),
                static_cast<std::size_t>(base));
            std::size_t const bound = min_of(
                chunk_schedule_t::rows_before(static_cast<small_size_t>(first_chunk + position + 1)),
                static_cast<std::size_t>(base) + rows);
            geometry_t const frozen = index_->geometry();
            step.position = position;
            step.index_rows = {static_cast<node_id_t>(first), static_cast<node_id_t>(bound)};
            step.source_rows = {static_cast<node_id_t>(first - base), static_cast<node_id_t>(bound - base)};
            step.vectors_bytes = step.index_rows.size() * stored_row_values(frozen) * sizeof(scalar_t);
            step.neighbors_bytes = step.index_rows.size() * frozen.connectivity() * sizeof(node_id_t);
            return step;
        }

        /** The steps the append spans - one per storage chunk it reaches; none once spent. */
        indexed_range<load_plan_t> steps() const noexcept {
            std::size_t const count = index_ && rows
                                          ? chunk_schedule_t::locate(static_cast<node_id_t>(base + rows - 1)).chunk -
                                                chunk_schedule_t::locate(base).chunk + 1u
                                          : 0u;
            return {this, count};
        }

        /**
         *  @brief Copies step @p step 's rows out of the plan's own columns straight into chunk
         *      memory: the vectors with their norms derived as they land, then the neighbors.
         *
         *  Every ID inside @c source_range is folded onto the appended range and every other onto
         *  the sentinel.
         *
         *  @param[in] step The step to fill, as @ref steps hands them out.
         */
        status_t load(step_t const& step) noexcept { return load(step, columns, source_range); }

        /**
         *  @brief The same copy out of @p columns under @p source_range rather than the plan's
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
            // The part's rows land at this step's own index rows, so an ID's shift follows from
            // the step rather than from the plan's base - the two agree for a one-part plan.
            node_id_t const shift = static_cast<node_id_t>(range.first - step.source_rows.first + source_range.first);
            // A rebased ID past this plan would name whichever row a later insert fills.
            if (static_cast<std::size_t>(shift) + source_range.size() > static_cast<std::size_t>(base) + rows)
                return status_t::config_mismatch_k;

            location_t const location = chunk_schedule_t::locate(range.first);
            nk::matrix_span<scalar_t> const vectors = index_->vectors_chunks_[location.chunk](
                nk::range(location.offset, location.offset + range.size()), nk::slice);
            // Checkpoints keep logical rows, while chunk rows are padded to whole kit steps.
            std::size_t const row_bytes = row_values * sizeof(scalar_t);
            for (std::size_t row = 0; row != range.size(); ++row) {
                [[maybe_unused]] status_t const copied = copy_bytes(
                    bytes_of(vectors.row(row).data(), row_values),
                    columns.vectors.subspan(vectors_first + row * row_bytes, row_bytes));
            }
            if constexpr (!std::is_same_v<vector_norms_t, no_norms_t>)
                for (std::size_t offset_in_chunk = 0; offset_in_chunk != range.size(); ++offset_in_chunk) {
                    location_t const row {location.chunk, location.offset + offset_in_chunk};
                    index_->norm_ref_for(row) = dense_norms(index_->vector_for(row));
                }
            atomic_ref<std::size_t>(covered_vectors).add(range.size(), std::memory_order_release);

            nk::matrix_span<node_id_t> const neighbors = index_->neighbors_chunks_[location.chunk](
                nk::range(location.offset, location.offset + range.size()), nk::slice);
            [[maybe_unused]] status_t const copied_neighbors = copy_bytes(
                bytes_of(neighbors.data(), step_slots),
                columns.neighbors.subspan(neighbors_first, step.neighbors_bytes));
            std::span<node_id_t> const landed {neighbors.data(), step_slots};
            rebase_neighbors(landed, source_range, shift, dense_node_id_unknown_k);
            compact_neighbors(landed, connectivity, range.first, dense_node_id_unknown_k);
            atomic_ref<std::size_t>(covered_neighbors).add(range.size(), std::memory_order_release);
            return status_t::success_k;
        }

        /** Publishes the appended rows once every chunk's vectors and neighbors landed: the claim
         *  consumed first, then identity, then the watermark last with a release. The identity
         *  releases too - a posted, order-free add could land after the watermark. A success
         *  spends the plan; a refusal leaves it whole, so the missing steps can still land. */
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

        /** The index appended to; null in the empty plan a refused @ref plan_load hands back,
         *  and once @c finish published. */
        dense_ram_index* index_ = nullptr;

        /** The rows this plan appends, released by their own death unless consumed. */
        capacity_claim_t claim_;
    };

    /**
     *  @brief Opens a load of @p rows from @p columns after the watermark: claims them through the
     *      ordinary capacity protocol, so a preceding @c reserve is consumed rather than duplicated
     *      and chunks grow only on a genuine deficit.
     *
     *  Refuses an unfrozen geometry, columns shorter than the rows, an insert mid-flight, and rows
     *  past the ID space.
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

    /** Opens a load of @p rows the caller sources per step - what a sharded plan drives, its parts
     *  arriving one slice at a time rather than as one pair of columns. */
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

    /** Appends the whole source on the calling thread - the one-shot over @ref plan_load for a
     *  caller with no threads to fan the steps out over. */
    status_t load(columns_view_t columns, std::size_t rows, iota_view<node_id_t> source_range) noexcept {
        auto [plan, status] = plan_load(columns, rows, source_range);
        if (failed(status)) return status;
        for (typename load_plan_t::step_t const step : plan.steps())
            if (status_t const loaded = plan.load(step); failed(loaded)) return loaded;
        return plan.finish();
    }

#pragma endregion Checkpoints

  private:

    /** The tile pool one refinement scores: the wider candidate set, padded to whole tiles. */
    static std::size_t tiles_pool_(std::size_t expansion_factor, std::size_t node_connectivity) noexcept {
        return round_up_to_multiple(max_of<std::size_t>(expansion_factor, node_connectivity + 1), tile_span_k);
    }

#pragma region Storage

    /** Trampoline for the kernel-aware @c usearch::dense_norms with hardened types - computed
     *  moments for norm-consuming kernels, the empty stand-in for dot. */
    inline static vector_norms_t dense_norms(vector_view_t vec) noexcept {
        return usearch::dense_norms<kernel_kind_, capability_, scalar_t>(vec);
    }

    /** Trampoline for @c usearch::distance with hardened types. */
    inline static distance_t distance(                   //
        vector_view_t query, vector_norms_t query_norms, //
        vector_view_t candidate, vector_norms_t candidate_norms) noexcept {
        return usearch::distance<kernel_kind_, capability_, scalar_t>( //
            query, query_norms, candidate, candidate_norms);
    }

    /** Trampoline for @c usearch::distances with hardened types. */
    inline static std::array<distance_t, 4> distances(   //
        vector_view_t query, vector_norms_t query_norms, //
        std::array<vector_view_t, 4> candidates, std::array<vector_norms_t, 4> candidates_norms) noexcept {
        return usearch::distances<kernel_kind_, capability_, scalar_t>( //
            query, query_norms, candidates, candidates_norms);
    }

    /** Trampoline for @c usearch::scores with hardened types - the walk's scoring, whose dots a fold reads. */
    inline static dense_scores<dot_result_t, 4> scores(  //
        vector_view_t query, vector_norms_t query_norms, //
        std::array<vector_view_t, 4> candidates, std::array<vector_norms_t, 4> candidates_norms) noexcept {
        return usearch::scores<kernel_kind_, capability_, scalar_t>( //
            query, query_norms, candidates, candidates_norms);
    }

    /** As @c scores, streaming @p next_candidates in under this CPU class's cache hints. */
    inline static dense_scores<dot_result_t, 4> scores(                                          //
        vector_view_t query, vector_norms_t query_norms,                                         //
        std::array<vector_view_t, 4> candidates, std::array<vector_norms_t, 4> candidates_norms, //
        std::array<vector_view_t, 4> next_candidates) noexcept {
        return usearch::scores<kernel_kind_, capability_, scalar_t>( //
            query, query_norms, candidates, candidates_norms, next_candidates, cache_hints_type_ {});
    }

    /** Scores @p query against four consecutive entries of the cached @p vectors & @p norms
     *  arrays starting at @p start; the gather cursor sticks at @p count - 1, so short tails
     *  repeat the last entry and the kernel always streams four valid lanes. */
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

    /**
     *  @brief Typed view over one node's neighbors - the only legal way to touch slots that
     *      concurrent scanners may be reading. Copyable value type with reference semantics.
     *
     *  Slot 0 is the combined lock & first-neighbor word: the all-ones @c dense_node_id_unknown_k
     *  sentinel while unpublished, empty or locked for editing, otherwise the first neighbor's
     *  plain ID. IDs are stored raw and no row bytes go to bookkeeping, so the capacity stays a
     *  power of two tiling whole cache lines. Slots 1.. are relaxed atomics on both sides, making
     *  the stale-but-valid mix a scanner can observe during re-linking defined behavior instead
     *  of a data race. Under a mutation-free contract every atomic degrades to a plain load, and
     *  mutating members belong to the fully concurrent form alone.
     *  @ref dense_ram_index_concurrency
     */
    template <synchronization_t synchronization_ = synchronization_t::concurrent_k>
    struct neighbors_ref {
        node_ids_span_t row_;

        /** Total slots in the row, every one usable for a neighbor ID. */
        inline small_size_t capacity() const noexcept { return static_cast<small_size_t>(row_.size()); }

        /** Relaxed atomic read of slot @p index; slot 0 belongs to @c peek_first / @c lock. */
        inline node_id_t load_slot(small_size_t index) const noexcept {
            if constexpr (synchronization_ == synchronization_t::concurrent_k)
                return atomic_ref<node_id_t>(row_[index]).load(std::memory_order_relaxed);
            else return row_[index];
        }

        /** Relaxed atomic write of slot @p index - only for the unpublished owner or the current
         *  lock holder. */
        inline void store_slot(small_size_t index, node_id_t value) const noexcept {
            atomic_ref<node_id_t>(row_[index]).store(value, std::memory_order_relaxed);
        }

        /** Acquire-load of the lock word: the sentinel means unpublished, empty or locked; any
         *  other value is the first neighbor ID, with everything its last unlocker wrote now
         *  visible to us. */
        inline node_id_t peek_first() const noexcept {
            if constexpr (synchronization_ == synchronization_t::concurrent_k)
                return atomic_ref<node_id_t>(row_[0]).load(std::memory_order_acquire);
            else return row_[0];
        }

        /** Spins until the lock is ours; returns the displaced first-neighbor value. An exchange
         *  that receives the sentinel back did @b not acquire the lock. Only meaningful on
         *  published nodes - locking an unpublished row spins until its owner publishes. */
        inline node_id_t lock() const noexcept {
            // Claim only after observing the slot free: on Arm a `SWPAL` writes even when it reads
            // the sentinel back, stealing the line from the holder mid-rewrite; the acquiring peek
            // keeps contenders in Shared state.
            while (true) {
                while (peek_first() == dense_node_id_unknown_k) wait_while_locked();
                node_id_t const displaced = atomic_ref<node_id_t>(row_[0]).exchange( //
                    dense_node_id_unknown_k, std::memory_order_acq_rel);
                if (displaced != dense_node_id_unknown_k) return displaced;
            }
        }

        /** Monitored capped wait on the lock word until it stops reading as the sentinel: the
         *  holder's unlock store wakes us, and the cap guards the arming race. Readers use it too
         *  - a bare re-peek loop spins acquiring loads for the whole critical section. */
        inline void wait_while_locked() const noexcept {
            wait_type_ {}(row_.data(), dense_node_id_unknown_k, 0u, fu::wait_capped_t {});
        }

        /** Release-stores @p first_value into slot 0 - unlocking and simultaneously republishing
         *  the edited neighbors to every future acquiring reader. */
        inline void unlock(node_id_t first_value) const noexcept {
            assert(holds_well_formed_row_(first_value) && "A published row names distinct nodes, then sentinels");
            atomic_ref<node_id_t>(row_[0]).store(first_value, std::memory_order_release);
        }

        /** Whether the held row, headed by @p first_value, names distinct nodes with sentinels only as
         *  its tail; slots 1.. are read plain, since their one writer is the lock holder. */
        inline bool holds_well_formed_row_(node_id_t first_value) const noexcept {
            small_size_t live = 1;
            while (live != capacity() && row_[live] != dense_node_id_unknown_k) ++live;
            for (small_size_t slot = live; slot != capacity(); ++slot)
                if (row_[slot] != dense_node_id_unknown_k) return false;
            for (small_size_t slot = 1; slot != live; ++slot)
                for (small_size_t earlier = 0; earlier != slot; ++earlier)
                    if (row_[slot] == (earlier ? row_[earlier] : first_value)) return false;
            return first_value != dense_node_id_unknown_k;
        }
    };

    /** The writer-side spelling: mutating paths are always fully synchronized. */
    using neighbors_ref_t = neighbors_ref<synchronization_t::concurrent_k>;

    /** Immutable view of the vector stored at @p location. */
    inline vector_view_t vector_for(location_t const& location) const noexcept {
        return vectors_chunks_[location.chunk].row(location.offset).as_vector();
    }

    /** The vectors behind the four neighbor slots from @p start, for the current quad's scoring to stream in;
     *  the lanes the list leaves short keep @p lane_vectors, the current quad's own. */
    template <typename neighbors_type_>
    inline std::array<vector_view_t, 4> vectors_quad_of(neighbors_type_ const& neighbors, small_size_t start,
                                                        node_id_t first_invisible_id,
                                                        std::array<vector_view_t, 4> lane_vectors) const noexcept {
        if constexpr (!std::is_same_v<cache_hints_type_, fu::standard_cache_hints_t>) {
            small_size_t const slots_limit = min_of<small_size_t>(start + 4, neighbors.capacity());
            small_size_t lane = 0;
            for (small_size_t slot = start; slot < slots_limit; ++slot) {
                node_id_t const neighbor_id = neighbors.load_slot(slot);
                if (neighbor_id == dense_node_id_unknown_k) break;
                if (neighbor_id >= first_invisible_id) continue;
                lane_vectors[lane++] = vector_for(chunk_schedule_t::locate(neighbor_id));
            }
        }
        return lane_vectors;
    }

    /** Mutable span over the vector slot at @p location - only for the slot's unpublished owner. */
    inline vector_span_t vector_span_for(location_t const& location) noexcept {
        return vectors_chunks_[location.chunk].row(location.offset).as_vector();
    }

    /** Copy of the pre-computed norms of the vector stored at @p location - the empty stand-in,
     *  without touching memory, when the kernel never reads norms. */
    inline vector_norms_t norm_for(location_t const& location) const noexcept {
        if constexpr (std::is_same_v<vector_norms_t, no_norms_t>) return {};
        else return norms_chunks_[location.chunk][location.offset];
    }

    /** Mutable reference to the norms slot at @p location - only for the slot's unpublished owner. */
    inline vector_norms_t& norm_ref_for(location_t const& location) noexcept {
        return norms_chunks_[location.chunk][location.offset];
    }

    /** Typed view over the fixed-capacity neighbors at @p location - atomic by default, plain
     *  under a mutation-free contract. @ref dense_ram_index_concurrency */
    template <synchronization_t synchronization_ = synchronization_t::concurrent_k>
    inline neighbors_ref<synchronization_> neighbors_for(location_t const& location) noexcept {
        return {neighbors_chunks_[location.chunk].row(location.offset).as_vector()};
    }

    /** Immutable view of the fixed-capacity neighbors at @p location. */
    inline node_ids_view_t neighbors_for(location_t const& location) const noexcept {
        return neighbors_chunks_[location.chunk].row(location.offset).as_vector();
    }

    /**
     *  @brief Allocates & publishes chunks until at least @p minimal_capacity fresh rows are added,
     *      following the growth schedule of @ref dense_ram_index_malloc.
     *  @return The number of added rows - possibly non-zero even when the status is a failure,
     *      in which case the caller must still release them back into @c remaining_capacity_.
     *
     *  Only one thread may run this at a time - the one that dropped @c remaining_capacity_ below
     *  zero. Successive growers are ordered by the acquire/release pair on that counter, which is
     *  also what publishes the non-atomic @c chunks_count_ and the directory writes. Readers reach
     *  a chunk only once a node inside it becomes visible, which chains through the same atomics.
     */
    expected<std::size_t> create_next_chunk(std::size_t minimal_capacity) noexcept {
        // Growing before the geometry is frozen is a usage error, not an allocation failure.
        geometry_t const frozen = geometry();
        if (!frozen) return {0, status_t::uninitialized_k};
        std::size_t const chunk_dimensions = padded_row_values(frozen) * nk::dimensions_per_value<scalar_t>();
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

            std::size_t const chunk_rows = chunk_schedule_t::rows_in(chunks_count_);

            auto vectors_chunk = vectors_chunk_t::zeros({chunk_rows, chunk_dimensions}, vectors_allocator_);
            // Norms chunks exist only for kernels whose finalize reads them; dot pays nothing.
            nk::expected<norms_chunk_t> norms_chunk {{}, nk::status_t::success_k};
            if constexpr (!std::is_same_v<vector_norms_t, no_norms_t>)
                norms_chunk = norms_chunk_t::zeros({chunk_rows}, norms_allocator_t(vectors_allocator_));
            // The all-ones fill is load-bearing: a sentinel leading slot reads as unpublished, and
            // sentinel tails terminate short rows, so no neighbor counts are stored anywhere.
            auto neighbors_chunk = neighbors_chunk_t::full( //
                {chunk_rows, chunk_connectivity}, dense_node_id_unknown_k, neighbors_allocator_);
            if (!vectors_chunk || !norms_chunk || !neighbors_chunk) return {added_capacity, status_t::bad_alloc_k};

            vectors_chunks_[chunks_count_] = std::move(vectors_chunk.value);
            norms_chunks_[chunks_count_] = std::move(norms_chunk.value);
            neighbors_chunks_[chunks_count_] = std::move(neighbors_chunk.value);
            ++chunks_count_;
            added_capacity += chunk_rows;
            // The all-ones ID is the sentinel: account one row less in the final chunk, so no
            // claim can dispense it and the dispenser never wraps.
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
        // Rejected before the counter dance, so it neither grows the index in vain nor overflows
        // the signed capacity arithmetic below.
        if (count > max_node_count_k) return {dense_node_id_unknown_k, status_t::overflow_k};
        while (true) {
            // One unconditional subtract, and its sign says what happened; the deficit it can
            // leave behind is exactly the signal the waiters below watch.
            std::ptrdiff_t const old_remaining_capacity = atomic_ref<std::ptrdiff_t>(remaining_capacity_)
                                                              .fetch_sub( //
                                                                  count, std::memory_order_acquire);
            std::ptrdiff_t const new_remaining_capacity = old_remaining_capacity - static_cast<std::ptrdiff_t>(count);

            // The thread that moved the counter from non-negative to negative grows next.
            // Accepting `old == 0` is load-bearing - capacity may rest at exactly zero, and
            // `old > 0` would elect nobody; only this thread's release lifts a negative counter.
            if (old_remaining_capacity >= 0 && new_remaining_capacity < 0) {
                auto [added_capacity, chunk_status] = create_next_chunk(count);
                if (chunk_status != status_t::success_k) {
                    // Revert the claim, releasing any partially added chunks along the way.
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
            // Even freshly added capacity may fall short of our own batch.
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
     *  @brief Greedy diversity selection over the ascending candidates: walking nearest-first, a
     *      candidate is kept only when it is closer to the query than to every already-kept
     *      neighbor, so neighbors spread across directions.
     *
     *  The slots the heuristic leaves empty take the nearest rejected candidates.
     *
     *  Reads the graph without locks. Cost is O(kept · considered) one-to-one distances.
     *
     *  @param[in] candidates Search results, ascending by distance to the query.
     *  @param[in] limit Ceiling on the kept neighbors - typically the connectivity.
     *  @param[out] out Receives the kept scored nodes, diverse picks first; its live shape is the fill.
     *  @return The number of distance computations performed, for the caller's telemetry.
     */
    std::size_t select_diverse(results_array_t const& candidates, std::size_t limit, selection_t& out,
                               dense_traversal_context_t& context) const noexcept {
        // Pools spanning at least one tile take the rectangles, keyed off `tile_span_k` alone
        // rather than a tuned threshold.
        std::size_t computed_distances;
        if constexpr (has_tiles_k)
            computed_distances = candidates.size() >= tile_span_k
                                     ? select_diverse_via_tiles(candidates, limit, out, context)
                                     : select_diverse_via_quads(candidates, limit, out, context);
        else computed_distances = select_diverse_via_quads(candidates, limit, out, context);
        fill_selection(to_span(candidates), out, limit);
        return computed_distances;
    }

    /** Appends the nearest rejected candidates of the ascending @p pool to the diverse prefix in
     *  @p out until @p limit or the pool runs out. The prefix is a subsequence of the pool, so
     *  one merge-walk tells a kept candidate from a rejected one. */
    static void fill_selection(std::span<dense_scored_node_t const> pool, selection_t& out,
                               std::size_t limit) noexcept {
        std::size_t const diverse_count = out.size();
        if (diverse_count == limit || nk::failed(out.resize({limit}))) return;
        std::size_t filled_count = diverse_count;
        std::size_t matched = 0;
        for (dense_scored_node_t const candidate : pool) {
            if (filled_count == limit) break;
            if (matched != diverse_count && out[matched].node_id == candidate.node_id) ++matched;
            else out[filled_count++] = candidate;
        }
        // Shrinking within the reservation never fails - the fill becomes the live shape.
        [[maybe_unused]] nk::status_t const shrunk = out.resize({filled_count});
    }

    /** The vector-engine sibling of @c select_diverse: pair distances computed on the fly with
     *  the 4-way kernels, the greedy scan deciding as they arrive. The kept prefix's views &
     *  norms ride the staging surface, free during any selection. */
    std::size_t select_diverse_via_quads(results_array_t const& candidates, std::size_t limit, selection_t& out,
                                         dense_traversal_context_t& context) const noexcept {
        if (nk::failed(out.resize({limit}))) return 0;
        vector_view_t* const kept_vectors = context.staged_vectors.data();
        vector_norms_t* const kept_norms = context.staged_norms.data();
        small_size_t kept_count = 0;
        std::size_t computed_distances = 0;
        dense_scored_node_t const* candidates_data = candidates.data();
        for (std::size_t index = 0; index != candidates.size() && kept_count != limit; ++index) {
            dense_scored_node_t const candidate = candidates_data[index];
            location_t const candidate_location = chunk_schedule_t::locate(candidate.node_id);
            vector_view_t const candidate_vector = vector_for(candidate_location);
            vector_norms_t const candidate_norms = norm_for(candidate_location);
            // 4-way batches over the kept neighbors' cached views; short tails repeat the last
            // one. A batch may score past the first violation, still cheaper than four serial.
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
        [[maybe_unused]] nk::status_t const shrunk = out.resize({kept_count});
        return computed_distances;
    }

    /**
     *  @brief The tile-engine sibling of @c select_diverse.
     *
     *  Candidates advance one span of @c tile_span_k rows at a time - the A-side of the tile kit -
     *  each span paying one span × kept-tiles rectangle before the greedy scan decides from
     *  precomputed rows, the standard gather → pack → init → update → finalize walk against the
     *  kept neighbors instead of a full square.
     *
     *  Only the kept tiles frozen at the span's start ride the matrix; neighbors accepted after
     *  that score on the 4-way kernels straight off the staging surface, so no tile is ever
     *  recomputed. Decisions match @c select_diverse_via_quads, and the shared finalize kit keeps
     *  both engines' distances bit-identical - only the compute schedule differs.
     */
    std::size_t select_diverse_via_tiles(results_array_t const& candidates, std::size_t limit, selection_t& out,
                                         dense_traversal_context_t& context) const noexcept {
        constexpr std::size_t rows_k = tiles_kit_t::tile_rows_k;
        constexpr std::size_t columns_k = tiles_kit_t::tile_columns_k;
        constexpr std::size_t depth_k = tiles_kit_t::tile_depth_k;
        if (nk::failed(out.resize({limit}))) return 0;

        vector_view_t* const kept_vectors = context.staged_vectors.data();
        vector_norms_t* const kept_norms = context.staged_norms.data();
        distance_t* const pairs = context.pair_distances.data();
        std::size_t const stride = context.pair_distances.extent(1);
        std::size_t const count = candidates.size();
        dense_scored_node_t const* candidates_data = candidates.data();

        // The packed kept tiles first, then the span's A-tiles, on the tiles' own alignment.
        std::size_t const dimensions = vector_for(chunk_schedule_t::locate(candidates_data[0].node_id)).size();
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
                location_t const location = chunk_schedule_t::locate( //
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
                // Neighbors accepted after the rectangle froze score on the 4-way path.
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
                // A completed kept tile packs once, and every later span streams it into updates.
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
        [[maybe_unused]] nk::status_t const shrunk = out.resize({kept_count});
        return computed_distances;
    }

    /**
     *  @brief The compute half of the tile engine: the full pairwise distance matrix of @p rows
     *      through the tile kit, into the context's @c pair_distances.
     *
     *  The finalize kit is the very one the vector path picked, so both engines produce
     *  bit-identical distances for the deciding siblings to read.
     *
     *  @param[in] rows Scattered views of the pool - the staged neighbors, or a candidate pool.
     *  @param[in] norms The stored norms, parallel to @p rows.
     *  @param[inout] context Supplies @c pair_distances & @c tiles_scratch from @c reserve_tiles.
     *  @return The pair count - the tiled refinement's full spend.
     */
    std::size_t pair_distances_via_tiles(std::span<vector_view_t const> rows, std::span<vector_norms_t const> norms,
                                         dense_traversal_context_t& context) const noexcept {
        constexpr std::size_t rows_k = tiles_kit_t::tile_rows_k;
        constexpr std::size_t columns_k = tiles_kit_t::tile_columns_k;
        constexpr std::size_t depth_k = tiles_kit_t::tile_depth_k;
        std::size_t const count = rows.size();
        std::size_t const stride = context.pair_distances.extent(1);

        // Pack each row-tile once - the Gram symmetry turns its A-strips straight into B-tiles -
        // then init → update over depth → finalize per tile pair. Tail tiles repeat the last
        // valid row; their lanes finalize into the padded stride and are never read.
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
            cache_hints_type_ const hints;
            for (std::size_t row = 0; row != rows_k; ++row)
                for (std::size_t depth_tile = 0; depth_tile != depth_tiles; ++depth_tile)
                    hints(tile_rows[row].data() + depth_tile * depth_k, fu::demote_line_t {});
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

    /** The pair-matrix sibling of @c select_diverse: a pure decision scan over precomputed
     *  distances, no kernels anywhere. The @p pool entries carry matrix @b positions in their
     *  @c node_id; a kept position at the staged fill is the newcomer, everything below
     *  translates through the staged IDs. */
    std::size_t select_diverse_via_pairs(results_array_t const& pool, std::size_t limit, selection_t& out,
                                         node_id_t newcomer_id, dense_traversal_context_t& context) const noexcept {
        ids_tensor_t const& staged_ids = context.staged_ids;
        distance_t const* pairs = context.pair_distances.data();
        std::size_t const stride = context.pair_distances.extent(1);
        dense_scored_node_t const* pool_data = pool.data();
        if (nk::failed(out.resize({limit}))) return 0;
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
        [[maybe_unused]] nk::status_t const shrunk = out.resize({kept_count});
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
     *  @brief The tile-engine sibling of @c stage_scores_via_quads: one staged Gram plus two 4-way
     *      rows - the neighbor's scores ordering the pool, and the newcomer's symmetric row &
     *      column - so re-selection decides entirely from precomputed pairs.
     *
     *  Pool entries carry matrix @b positions, ready for @c select_diverse_via_pairs.
     *  @return The distances computed, Gram pairs included.
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

        // The newcomer's row & column, so it competes inside the matrix at position `count`;
        // its own score is the reused symmetric one, never recomputed.
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
     *      slot-0 sentinel exchange and inserts the node - appending when there is room, otherwise
     *      re-selecting the diverse best among the existing neighbors plus the newcomer.
     *
     *  A neighbor already holding the node is left as it stands, so a refinement keeping an
     *  existing neighbor never duplicates it.
     *
     *  Locking discipline: at most one neighbor lock is held at any moment, and never while
     *  spinning on the watermark - callers invoke this before their watermark turn, so every lock
     *  is released by the time the turn's spin begins. All slot writes happen under the lock, and
     *  the unlocking release-store of slot 0 is what republishes the edited neighbors.
     *
     *  @param[in] selected The node's chosen neighbors with their recorded query distances -
     *      those score the newcomer for free, since the metrics are symmetric.
     *  @param[inout] context Scratch for the re-selection, plus the telemetry counters.
     */
    void back_link_neighbors(node_id_t new_node_id, vector_view_t new_vector, vector_norms_t new_norms,
                             selection_t const& selected, dense_traversal_context_t& context) noexcept {
        // Pairs with the acquire fence in the snapshot probe: a save reading any row word written
        // below proves this node's dispense, so an overlapped copy always reports contention.
        std::atomic_thread_fence(std::memory_order_release);

        results_array_t& refining_candidates = context.refining_candidates;
        std::size_t const node_connectivity = connectivity();

        for (small_size_t selected_index = 0; selected_index != selected.size(); ++selected_index) {
            node_id_t const neighbor_id = selected[selected_index].node_id;
            location_t const neighbor_location = chunk_schedule_t::locate(neighbor_id);
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

            // Slow path: every slot is taken, so re-run the diversity heuristic over the existing
            // neighbors and the newcomer, scored against the neighbor itself, under the lock.
            vector_view_t const neighbor_vector = vector_for(neighbor_location);
            vector_norms_t const neighbor_norms = norm_for(neighbor_location);

            // Staged at their own slot positions, the lock-displaced slot-0 value first. The
            // newcomer stays out: it would hand the tile kernels an irregular connectivity + 1
            // operand, and its symmetric score was already recorded at selection time.
            if (nk::failed(context.staged_ids.resize({slots_count}))) {
                neighbors.unlock(first_slot_value);
                continue;
            }
            location_t const first_location = chunk_schedule_t::locate(first_slot_value);
            context.staged_ids[0] = first_slot_value;
            context.staged_vectors[0] = vector_for(first_location);
            context.staged_norms[0] = norm_for(first_location);
            for (small_size_t slot = 1; slot != slots_count; ++slot) {
                node_id_t const existing_id = neighbors.load_slot(slot);
                location_t const existing_location = chunk_schedule_t::locate(existing_id);
                context.staged_ids[slot] = existing_id;
                context.staged_vectors[slot] = vector_for(existing_location);
                context.staged_norms[slot] = norm_for(existing_location);
            }

            // The pool is every neighbor plus the newcomer; that "+ 1" lives in the sorted pool
            // alone, never in an operand shape.
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

    /**
     *  @brief Stages the live prefix of @p neighbors into @c context.staged_ids, waiting out a
     *      lock holder like any traversal does; a published-empty self-loop stages nothing.
     *  @return The staged count - the tensor's live shape.
     */
    small_size_t stage_neighbors_(neighbors_ref_t neighbors, node_id_t node_id,
                                  dense_traversal_context_t& context) const noexcept {
        ids_tensor_t& staged_ids = context.staged_ids;
        small_size_t const slots_count = neighbors.capacity();
        if (nk::failed(staged_ids.resize({slots_count}))) return 0;
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
        [[maybe_unused]] nk::status_t const shrunk = staged_ids.resize({staged_count});
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
        location_t const location = chunk_schedule_t::locate(node_id);
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
        if (failed(search_around(own_vector, own_norms, first_invisible_id, expansion_factor, context, context.results,
                                 dummy_fold_t {}, seeds)))
            ++context.saturated_walks;

        selection_t& selected = context.selected_neighbors;
        std::size_t const refine_distances = select_diverse(context.results, node_connectivity, selected, context);
        context.computed_distances += refine_distances;
        context.computed_distances_in_refines += refine_distances;
        small_size_t const selected_count = static_cast<small_size_t>(selected.size());

        // Neighbors are distinct, so equal sizes and one match per kept neighbor mean the same
        // set. The slots are then rewritten, sentinel-terminated and republished through slot 0.
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
     *  @return @c advanced_k, @c exhausted_k once every published node is already visited, or
     *      @c sated_k once the fold asks the walk to end - the last two end the walk rather than
     *      failing it. A failed status means scratch could not grow.
     *
     *  Uniform draws that collide with @c visited fall back to cyclic linear probing over the ID
     *  space, so every call makes strict progress without retry budgets. Only transiently locked
     *  nodes are re-rolled, like the traversal itself.
     */
    template <synchronization_t synchronization_ = synchronization_t::concurrent_k, typename fold_type_ = dummy_fold_t,
              typename heads_type_ = dummy_heads_t>
    expected<walk_step_t> sample_landmarks(vector_view_t query, vector_norms_t query_norms,
                                           node_id_t first_invisible_id, std::size_t expansion_factor,
                                           dense_traversal_context_t& context, std::span<node_id_t const>& seeds,
                                           fold_type_&& fold = fold_type_ {}, heads_type_ const& heads = {}) noexcept {
        results_array_t& results = context.results;
        frontier_queue_t& frontier = context.frontier;
        visited_set_t& visited = context.visited;

        std::array<node_id_t, 4> landmarks_ids;
        std::array<vector_view_t, 4> landmarks_vectors;
        std::array<vector_norms_t, 4> landmarks_norms;
        small_size_t landmarks_in_batch = 0;

        if (status_t const seen = visited.reserve(visited.size() + landmarks_ids.size()); failed(seen))
            return {walk_step_t::exhausted_k, seen};
        // `visited` may also hold above-watermark IDs discovered through fresh back-linking, so the
        // size check is conservative: it can only under-sample, never let the probe loop spin.
        while (landmarks_in_batch != landmarks_ids.size() && visited.size() < first_invisible_id) {
            // Caller-staged seeds go first, each taken once and skipped when unpublished or
            // already visited; past them, draw uniformly and probe forward cyclically.
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

            location_t const landmark_location = chunk_schedule_t::locate(landmark_node_id);
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
        if (!landmarks_in_batch) return {walk_step_t::exhausted_k, status_t::success_k};

        // Pad the tail lanes with lane zero; the padded outputs are never consumed.
        for (small_size_t lane = landmarks_in_batch; lane != landmarks_ids.size(); ++lane) {
            landmarks_ids[lane] = landmarks_ids[0];
            landmarks_vectors[lane] = landmarks_vectors[0];
            landmarks_norms[lane] = landmarks_norms[0];
        }
        dense_scores<dot_result_t, 4> landmark_scores = scores( //
            query, query_norms, landmarks_vectors, landmarks_norms);
        dense_rescore<capability_>(heads, query, landmarks_ids, landmarks_vectors, landmark_scores, landmarks_in_batch);
        context.computed_distances += landmarks_in_batch;
        std::array<dense_scored_node_t, 4> landmarks_scored;
        for (small_size_t lane = 0; lane != landmarks_in_batch; ++lane)
            landmarks_scored[lane] = dense_scored_node_t(landmarks_ids[lane], landmark_scores.distances[lane]);
        std::span const scored = std::span(landmarks_scored).first(landmarks_in_batch);
        // Landmarks fold right here, in lane order: they are marked visited, so the walk never
        // re-scores them and this is their one chance to be absorbed.
        if (fold.absorb(scored, std::span<dot_result_t const>(landmark_scores.dots).first(landmarks_in_batch)) ==
            walk_demand_t::sated_k)
            return {walk_step_t::sated_k, status_t::success_k};
        network_sort(scored, dense_closer_scored_node_t {});
        results.insert(scored, expansion_factor);
        for (small_size_t lane = 0; lane != landmarks_in_batch; ++lane)
            if (status_t const queued = frontier.insert(dense_scored_node_t(scored[lane])); failed(queued))
                return {walk_step_t::advanced_k, queued};
        return {walk_step_t::advanced_k, status_t::success_k};
    }

    /**
     *  @brief Greedy best-first NSW walk from random published landmarks toward the @p query.
     *
     *  Read-only on the graph: takes no locks, writes nothing, retries transiently locked nodes.
     *  A refused re-table ends the walk early with the ranking it has and says so in the status,
     *  which the callers count as a saturated walk rather than a failure.
     *
     *  @param[in] first_invisible_id The caller's watermark snapshot, bounding the readable ID space.
     *  @param[in] expansion_factor Breadth of the candidate pool - the walk's stopping radius.
     *  @param[inout] context Scratch & telemetry; must arrive pre-reserved & pre-cleared, and ends
     *      with up to @p expansion_factor nearest published candidates in @c context.results.
     *  @param[in] bound The pool whose fill ends restarts and whose radius bounds the walk once full:
     *      @c context.results itself, or a filtered walk's admitted pool.
     *  @param[in] fold Optional reader of every scored quad - see @c dense_fold. It never steers the
     *      walk, except that a @c sated_k from it ends the walk successfully.
     *  @param[in] seeds Optional landmarks to start from ahead of the random draws - a
     *      re-selection hands in the node's own neighbors, so every one already wired competes.
     *  @param[in] heads Optional indexer heads of this one query row, rescoring every scored node.
     */
    template <synchronization_t synchronization_ = synchronization_t::concurrent_k, typename fold_type_ = dummy_fold_t,
              typename heads_type_ = dummy_heads_t>
    status_t search_around(vector_view_t query, vector_norms_t query_norms, node_id_t first_invisible_id,
                           std::size_t expansion_factor, dense_traversal_context_t& context,
                           results_array_t const& bound, fold_type_&& fold = fold_type_ {},
                           std::span<node_id_t const> seeds = {}, heads_type_ const& heads = {}) noexcept {
        results_array_t& results = context.results;
        frontier_queue_t& frontier = context.frontier;
        visited_set_t& visited = context.visited;
        std::size_t const node_connectivity = connectivity();
        assert(results.capacity() >= expansion_factor && "A pool short of the expansion never tightens the radius");

        // About as many landmarks as the connectivity - a stochastic multi-start in 4-way batches.
        small_size_t const landmark_batches = divide_round_up<small_size_t>(node_connectivity, 4);
        for (small_size_t landmarks_batch = 0; landmarks_batch != landmark_batches; ++landmarks_batch) {
            auto const [seeding, seed_status] = sample_landmarks<synchronization_>(
                query, query_norms, first_invisible_id, expansion_factor, context, seeds, fold, heads);
            if (failed(seed_status)) return seed_status;
            if (seeding == walk_step_t::sated_k) return status_t::success_k;
            if (seeding == walk_step_t::exhausted_k) break; // Everything published is already visited.
        }

        // Greedily walk the frontier nearest-first, keeping the best `expansion_factor` in `results`.
        while (true) {
            // A drained frontier before a full result pool means the walk exhausted everything
            // reachable from its seeds, so re-seed and continue. Every successful batch strictly
            // grows `visited` and `results`, so restarts are self-limiting without a budget.
            if (frontier.empty()) {
                // Re-seed until the bound fills - a filtered walk's matches may hide in regions the
                // drained frontier never reached.
                if (bound.size() >= expansion_factor) break;
                auto const [seeding, seed_status] = sample_landmarks<synchronization_>(
                    query, query_norms, first_invisible_id, expansion_factor, context, seeds, fold, heads);
                if (failed(seed_status)) return seed_status;
                if (seeding == walk_step_t::sated_k) return status_t::success_k;
                if (seeding == walk_step_t::exhausted_k) break; // Nothing fresh left to restart from.
                ++context.restarts;
                continue;
            }

            // Peek - don't pop yet: the chosen node's neighbors may be locked, in which case
            // we retry the selection rather than dropping it from the frontier.
            dense_scored_node_t const candidate = frontier.top();
            // The bound governs once it fills, since a filtered walk's radius is the wider one.
            results_array_t const& pool = bound.size() == expansion_factor ? bound : results;
            bool const pool_full = pool.size() == expansion_factor;
            // Once the nearest unexpanded node is past a full pool's radius, nothing deeper can
            // improve the answer.
            if (pool_full && candidate.distance > pool.top().distance) break;
            // Every other entry past that radius is never popped either: shed them when the coming
            // expansion, at most one node's neighbors, would not fit.
            if (pool_full && frontier.capacity() - frontier.size() < node_connectivity)
                frontier.erase_below(pool.top());

            location_t const candidate_location = chunk_schedule_t::locate(candidate.node_id);
            neighbors_ref<synchronization_> const candidate_neighbors = neighbors_for<synchronization_>( //
                candidate_location);

            // Slot 0 holds either the lock sentinel or the first neighbor's ID; a locked node is
            // waited out rather than dropped, since dropping it would silently degrade recall.
            node_id_t const first_neighbor_id = candidate_neighbors.peek_first();
            if constexpr (synchronization_ == synchronization_t::concurrent_k)
                if (first_neighbor_id == dense_node_id_unknown_k) {
                    candidate_neighbors.wait_while_locked();
                    continue;
                }

            // The node is live and readable - commit to it and drop it from the frontier.
            frontier.pop();
            ++context.visited_members;

            // Slot 0 comes from the acquiring load above - never re-read it, a writer may have
            // locked the node since. The remaining slots are relaxed reads, so the worst case is
            // a stale-but-valid mix of neighbors, never a torn read.
            small_size_t const slots_count = candidate_neighbors.capacity();
            small_size_t neighbor_slot = 0;
            bool list_ended = false;
            while (!list_ended && neighbor_slot != slots_count) {

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
                    // Translation-frontier invariant: only IDs below the watermark snapshot may be
                    // located. A fresher one has no happens-before edge to its hosting chunk yet,
                    // so skip it; a later traversal with a fresher snapshot meets it safely.
                    if (neighbor_id >= first_invisible_id) continue;
                    if (visited.contains(neighbor_id)) continue;
                    location_t const neighbor_location = chunk_schedule_t::locate(neighbor_id);
                    gathered_ids[gathered_count] = neighbor_id;
                    gathered_vectors[gathered_count] = vector_for(neighbor_location);
                    gathered_norms[gathered_count] = norm_for(neighbor_location);
                    ++gathered_count;
                }
                if (!gathered_count) continue;

                // Pad the tail lanes with lane zero; the padded outputs are never consumed.
                for (small_size_t lane = gathered_count; lane != gathered_ids.size(); ++lane) {
                    gathered_ids[lane] = gathered_ids[0];
                    gathered_vectors[lane] = gathered_vectors[0];
                    gathered_norms[lane] = gathered_norms[0];
                }
                dense_scores<dot_result_t, 4> gathered_scores = scores( //
                    query, query_norms, gathered_vectors, gathered_norms,
                    vectors_quad_of(candidate_neighbors, neighbor_slot, first_invisible_id, gathered_vectors));
                dense_rescore<capability_>(heads, query, gathered_ids, gathered_vectors, gathered_scores,
                                           gathered_count);
                context.computed_distances += gathered_count;
                std::array<dense_scored_node_t, 4> scored_batch;
                for (small_size_t lane = 0; lane != gathered_count; ++lane)
                    scored_batch[lane] = dense_scored_node_t(gathered_ids[lane], gathered_scores.distances[lane]);
                std::span const scored = std::span(scored_batch).first(gathered_count);
                // Folding is independent of navigation: a fold reads every scored quad in lane order,
                // including what the navigation pool rejects, and never decides where the walk routes.
                if (fold.absorb(scored, std::span<dot_result_t const>(gathered_scores.dots).first(gathered_count)) ==
                    walk_demand_t::sated_k)
                    return status_t::success_k;
                network_sort(scored, dense_closer_scored_node_t {});

                // One merge folds the ascending quad into the pool; only its accepted prefix
                // enters the frontier. The merge precedes the marks, so a refusal keeps the quad.
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
     *  @brief Separately allocated matrix slabs, one row per vector entry; slot indexes are global
     *      and grow transparently across the matrices.
     *
     *  Sized once to @c max_chunks_k entries and never reallocated, so readers dereference chunk
     *  headers with zero synchronization while one writer constructs the next in place.
     *  @ref dense_ram_index_malloc
     */
    fu::dynamic_padded_array<vectors_chunk_t, vectors_allocator_type_> vectors_chunks_;

    /** Separately allocated array slabs, one entry per vector's pre-computed norms; slot indexes
     *  are global and grow transparently across the arrays. @ref dense_ram_index_malloc */
    fu::dynamic_padded_array<norms_chunk_t, vectors_allocator_type_> norms_chunks_;

    /** Separately allocated matrix slabs, one row per vector's fixed-capacity neighbors; slot
     *  indexes are global and grow transparently across the matrices. @ref dense_ram_index_malloc */
    fu::dynamic_padded_array<neighbors_chunk_t, neighbors_allocator_type_> neighbors_chunks_;

    /**
     *  @brief Signed integer describing "unsigned capacity", so concurrent under- and overflow stay
     *      detectable. @b Never assign to it - increment and decrement only.
     *
     *  This and every RMW'd counter below sits on its own @c default_alignment_k line, so a
     *  dispenser RMW never invalidates the line every concurrent search snapshots.
     */
    alignas(default_alignment_k) std::ptrdiff_t remaining_capacity_ {0};

    /** Identifier of the last allocated node - the lower bound on initialized or empty rows across
     *  the chunks; typically incremented in batches. */
    alignas(default_alignment_k) node_id_t last_allocated_id_ {0};

    /** The ID dispenser: the identifier the next wired node receives. Incremented one-by-one at
     *  wiring time, so IDs follow the global wiring-start order and each thread holds at most one
     *  unpublished ID. @ref dense_ram_index_concurrency */
    alignas(default_alignment_k) node_id_t next_dispensed_id_ {0};

    /** The visibility watermark: every ID strictly below it is published, its vector, norms &
     *  neighbors readable - the safe bound for random landmark sampling. Advanced strictly in
     *  dispensed order by each publisher taking its turn. @ref dense_ram_index_concurrency */
    alignas(default_alignment_k) node_id_t last_visible_id_ {0};

    /** Read-mostly after the freeze - kept off the watermark's line with the other cold members. */
    alignas(default_alignment_k) geometry_t geometry_ {};

    /** Constructed chunks per directory. Deliberately non-atomic: only the elected grower mutates
     *  it, and successive growers are ordered by the acquire/release pair on
     *  @c remaining_capacity_, which no release lifts while one is inside. @ref create_next_chunk */
    small_size_t chunks_count_ = 0;
};

/**
 *  @brief External reader's cursor over the watermark of one @c dense_ram_index:
 *      random access & ID-order iteration over nodes, plus neighbor-following through a privately
 *      staged copy of the current node's neighbors.
 *
 *  Strictly read-only under every contract, so readers never stall writers. Under @c concurrent_k
 *  the staging reads neighbor words through relaxed atomics, tolerating the same stale-but-valid
 *  mix the traversal does; exact neighbors come from the @c immutable_k contract, which reads
 *  plain memory. The cursor captures the frozen geometry once at construction, starts past the
 *  end, and @c seek(0) enters the graph.
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
    using location_t = typename index_t::location_t;

    using ids_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<node_id_t>;
    using ids_tensor_t = nk::tensor<node_id_t, ids_allocator_t, 1>;

    /** The watermark ordering behind this cursor's snapshots: acquiring while writers may be
     *  live, plain when the caller certified a mutation-free environment. */
    inline static constexpr std::memory_order watermark_order_k = synchronization_ == synchronization_t::concurrent_k
                                                                      ? std::memory_order_acquire
                                                                      : std::memory_order_relaxed;

    /** The traversed index - never written through, under any contract. */
    index_t const& index_;

    /** The staged @b [connectivity] neighbor copy of the current node. */
    ids_tensor_t staged_neighbors_;

    /** Chunk & row of the current node, derived once per @c seek. */
    location_t location_ {};

    /** Decoded live prefix of @c staged_neighbors_ - self-loops & sentinels stripped. */
    small_size_t staged_count_ = 0;

    /** This cursor's watermark snapshot: IDs strictly below it are readable. */
    node_id_t visible_count_;

    /** The current node ID, or the watermark snapshot when past the end. */
    node_id_t current_;

    /** The frozen connectivity, captured once at construction. */
    small_size_t connectivity_;

    /** Binds the cursor & captures its snapshots; allocates nothing - call @c reserve(). */
    explicit dense_ram_cursor(index_t const& index) noexcept
        : index_(index), visible_count_(atomic_ref<node_id_t const>(index.last_visible_id_).load(watermark_order_k)),
          current_(visible_count_), connectivity_(static_cast<small_size_t>(index.connectivity())) {}

    dense_ram_cursor(dense_ram_cursor const&) = delete;
    dense_ram_cursor& operator=(dense_ram_cursor const&) = delete;
    dense_ram_cursor(dense_ram_cursor&&) = default;
    dense_ram_cursor& operator=(dense_ram_cursor&&) = delete;

    /** Sizes the staging for the frozen connectivity through @p allocator - the index's
     *  traversals-side family by default; call once before any @c seek. */
    status_t reserve(allocator_t allocator = {}) noexcept {
        if (staged_neighbors_.size() >= connectivity_) return status_t::success_k;
        auto staged = ids_tensor_t::zeros({connectivity_}, ids_allocator_t(allocator));
        if (!staged) return status_t::bad_alloc_k;
        staged_neighbors_ = std::move(staged.value);
        return status_t::success_k;
    }

    /** Nodes visible to this cursor; IDs strictly below are readable. */
    node_id_t visible_count() const noexcept { return visible_count_; }

    /** Widens the snapshot to admit nodes published since; a cursor standing past the end stays
     *  past the end - entering the admitted region takes an explicit @c seek. */
    void refresh() noexcept {
        bool const past_end = current_ == visible_count_;
        visible_count_ = atomic_ref<node_id_t const>(index_.last_visible_id_).load(watermark_order_k);
        if (past_end) current_ = visible_count_;
    }

    /** The current node's ID; equals @c visible_count() past the end. */
    node_id_t id() const noexcept { return current_; }

    /** @c true while the cursor points at a readable node. */
    explicit operator bool() const noexcept { return current_ < visible_count_; }

    /** Immutable view of the current node's vector; meaningless past the end. */
    vector_view_t vector() const noexcept { return index_.vector_for(location_); }

    /** The current node's pre-computed norms; meaningless past the end. */
    vector_norms_t norms() const noexcept { return index_.norm_for(location_); }

    /** The staged neighbor IDs of the current node - decoded, stable until the next move. */
    std::span<node_id_t const> neighbors() const noexcept { return to_span(staged_neighbors_).first(staged_count_); }

    /** Positions on @p node & stages a copy of its neighbors: @c malformed_k at or past the
     *  snapshot, @c uninitialized_k before @c reserve - the cursor stays where it was on either
     *  refusal. Read-only under every contract. */
    status_t seek(node_id_t node) noexcept {
        if (node >= visible_count_) return status_t::malformed_k;
        if (staged_neighbors_.size() < connectivity_) return status_t::uninitialized_k;
        current_ = node;
        location_ = index_t::chunk_schedule_t::locate(node);
        node_ids_view_t const row = index_.neighbors_for(location_);
        // Slot 0 is the lock word: acquire it exactly once and never re-read it, so a fresh lock
        // sentinel cannot slip into the staging.
        node_id_t first_value;
        if constexpr (synchronization_ == synchronization_t::concurrent_k) {
            first_value = atomic_ref<node_id_t const>(row[0]).load(std::memory_order_acquire);
            while (first_value == dense_node_id_unknown_k) {
                typename index_t::wait_t {}(row.data(), dense_node_id_unknown_k, small_size_t(0), fu::wait_capped_t {});
                first_value = atomic_ref<node_id_t const>(row[0]).load(std::memory_order_acquire);
            }
        }
        else first_value = row[0];
        staged_neighbors_[0] = first_value;
        // Relaxed atomics make a live re-linker's stale-but-valid mix defined; certified
        // contracts read plain memory.
        if constexpr (synchronization_ == synchronization_t::concurrent_k)
            for (small_size_t slot = 1; slot != connectivity_; ++slot)
                staged_neighbors_[slot] = atomic_ref<node_id_t const>(row[slot]).load(std::memory_order_relaxed);
        else
            for (small_size_t slot = 1; slot != connectivity_; ++slot) staged_neighbors_[slot] = row[slot];
        // A self-loop is "published, zero neighbors"; the all-ones sentinel terminates short rows.
        if (staged_neighbors_[0] == node) { staged_count_ = 0; }
        else {
            small_size_t live = 0;
            while (live != connectivity_ && staged_neighbors_[live] != dense_node_id_unknown_k) ++live;
            staged_count_ = live;
        }
        return status_t::success_k;
    }

    /** Moves to the next ID below the snapshot, or past the end; a staging refusal parks on the
     *  new ID with no neighbors staged - a later @c seek to the same ID retries. */
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

    /** Deleted: it would copy the staging tensor - an allocation in disguise. */
    dense_ram_cursor operator++(int) = delete;

    /** Follows the staged neighbor at @p slot: @c malformed_k when the slot is beyond the live
     *  prefix or its target is at or past this cursor's snapshot. */
    status_t follow(small_size_t slot) noexcept {
        if (slot >= staged_count_) return status_t::malformed_k;
        return seek(staged_neighbors_[slot]);
    }
};

} // namespace usearch
} // namespace unum
