/**
 *  @file include/usearch/dense/ram_index.hpp
 *  @author Ash Vardanian
 *  @date July 26, 2023
 *  @brief Vector Search index for a single uniform latency memory space.
 */
#pragma once

#include <cassert> // `assert`
#include <cstring> // `std::memcpy`, `std::memset`

#include <array>       // `std::array`
#include <bit>         // `std::bit_width`, `std::bit_floor`, `std::has_single_bit`
#include <functional>  // `std::identity` - the single index's resolver
#include <limits>      // `std::numeric_limits`
#include <numbers>     // `std::numbers::log2e_v`, `std::numbers::ln2_v` - the base-2 softmax
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

namespace unum::usearch::dense {

/** External reader's cursor over one @c ram_index, defined after the index it befriends.
 *  The allocator defaults to the index's traversals-side family, so staging inherits the same
 *  NUMA-aware placement; readers pinned elsewhere override it. */
template <typename index_type_,                                                 //
          synchronization_t synchronization_ = synchronization_t::concurrent_k, //
          typename allocator_type_ = typename index_type_::candidates_allocator_t>
struct ram_cursor;

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
 *  A batch publishes whole: one add dispenses its rows' IDs as one range, and one watermark turn
 *  makes every row visible, so a reader sees all of a batch or none of it. The turns run in
 *  dispensed order, so the publication gap is bounded by the batches in flight, and a small batch
 *  waits behind an earlier, larger batch's wiring before its turn.
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
    nk_capability_t capability_ = nk_cap_icelake_k,                           //
    typename wait_type_ = fu::standard_yield_t,                               //
    template <typename> class atomic_ref_template_ = fu::standard_atomic_ref, //
    typename cache_hints_type_ = fu::standard_cache_hints_t,                  //
    typename vectors_allocator_type_ = std::allocator<byte_t>,                //
    typename neighbors_allocator_type_ = std::allocator<byte_t>,              //
    typename traversals_allocator_type_ = std::allocator<byte_t>              //
    >
struct ram_index;

/** One quad's worth of predicate-admitted candidates - the buffer handed to batched
 *  @c search_if predicates, bounded by the distance kernels' fan-out. */
using admissions_t = fu::limited_array<scored_node_t, 4>;

/**
 *  @brief Runs @p predicate over one scored quad, filling @p admissions with what it admits.
 *
 *  Two shapes, detected by signature: a batched functor takes the whole @p scored span and
 *  pushes what it admits, a per-node @c admission_t(node_id_t) functor is walked lane by
 *  lane. @c dummy_predicate_t never reaches here - its callers compile out whole.
 */
template <typename predicate_type_>
inline void invoke_admission(predicate_type_& predicate, std::span<scored_node_t const> scored,
                             admissions_t& admissions) noexcept {
    if constexpr (requires { predicate(scored, admissions); }) predicate(scored, admissions);
    else
        for (scored_node_t const candidate : scored)
            if (predicate(candidate.node_id) == admission_t::admit_k) {
                // One quad in, four slots of capacity - the buffer cannot refuse.
                [[maybe_unused]] fu::status_t const admitted = admissions.push_back(candidate);
            }
}

/** Filtered search as a fold: each quad's matches merge into the admitted pool, while the walk
 *  keeps routing through whatever the predicate rejects. */
template <typename predicate_type_, typename pool_type_>
struct filtering_fold {

    /** The caller's filter, per-node or batched - see @c invoke_admission. */
    predicate_type_& predicate;

    /** The pool matches merge into, ascending by distance. */
    pool_type_& admitted;

    /** The breadth @c admitted is capped at. */
    std::size_t expansion_factor;

    /** Whether @p candidate could still enter @c admitted: always until it fills, then unless its
     *  worst match is closer, so the predicate never sees a lane the pool would drop; a tie still
     *  merges in. */
    bool contends(scored_node_t const& candidate) const noexcept {
        return admitted.size() < expansion_factor || !closer_scored_node_t {}(admitted.top(), candidate);
    }

    template <typename dot_result_type_>
    walk_demand_t absorb(std::span<scored_node_t const> scored_nodes, std::span<dot_result_type_ const>) noexcept {
        admissions_t contenders, admissions;
        for (scored_node_t const candidate : scored_nodes)
            if (contends(candidate)) {
                // One quad in, four slots of capacity, so the buffer cannot refuse.
                [[maybe_unused]] fu::status_t const pushed = contenders.push_back(candidate);
            }
        invoke_admission(predicate, std::span<scored_node_t const>(contenders), admissions);
        network_sort(std::span(admissions), closer_scored_node_t {});
        admitted.insert(std::span(admissions), expansion_factor);
        return walk_demand_t::continue_k;
    }

    /** The keyed form: @p keys hold each lane's key, so a per-key predicate filters a walk that has
     *  no key column to resolve node IDs through. */
    template <typename dot_result_type_, typename key_type_>
    walk_demand_t absorb(std::span<scored_node_t const> scored_nodes, std::span<dot_result_type_ const>,
                         std::span<key_type_ const> keys) noexcept {
        admissions_t admissions;
        for (std::size_t lane = 0; lane != scored_nodes.size(); ++lane)
            if (contends(scored_nodes[lane]) && predicate(keys[lane]) == admission_t::admit_k) {
                // One quad in, four slots of capacity, so the buffer cannot refuse.
                [[maybe_unused]] fu::status_t const pushed = admissions.push_back(scored_nodes[lane]);
            }
        network_sort(std::span(admissions), closer_scored_node_t {});
        admitted.insert(std::span(admissions), expansion_factor);
        return walk_demand_t::continue_k;
    }
};

/** Reads every scored quad - live lanes in kernel order beside their native dots, both valid only
 *  for the call - and answers whether the walk should go on. */
template <typename fold_type_, typename dot_result_type_>
concept fold = requires(fold_type_& fold, std::span<scored_node_t const> scored_nodes,
                        std::span<dot_result_type_ const> dots) {
    { fold.absorb(scored_nodes, dots) } noexcept -> std::same_as<walk_demand_t>;
};

/** Graph products of one left-hand row as a fold: each scored pair appends its resolved ID and native
 *  dot until the row is full, while @c scored keeps counting past it. */
template <typename id_type_, typename dot_result_type_, typename resolve_type_ = std::identity>
struct multiply_fold {

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

    walk_demand_t absorb(std::span<scored_node_t const> scored_nodes, std::span<dot_result_type_ const> dots) noexcept {
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

/** Whether @p admits lets @p node into an exact scan's matches - always, under the dummy
 *  predicate. */
template <typename admits_type_>
bool admitted(admits_type_& admits, node_id_t node) noexcept {
    if constexpr (std::is_same_v<std::decay_t<admits_type_>, dummy_predicate_t>) return true;
    else return admits(node) == admission_t::admit_k;
}

/**
 *  @brief Scores every row of @p rows, the first being node @p first, against the queries packed in
 *      @p packed, landing each row that beats some query's threshold and that @p admits lets in.
 *
 *  Rows stream through the matrix engine @c stream_rows_k at a time into @p dots, as deep as the
 *  logical width of @p rows, which must be the width the queries were packed at. Each row of dots
 *  finalizes against four queries at a time through @p quads_kit_type_, so the distances match a
 *  walk's; @p query_sumsqs and every row of @p dots are read in whole quads past the last query.
 *  @p row_sumsq maps a row of @p rows to its stored sum of squares, @p land takes a query's column
 *  and the scored node, and answers that query's new threshold.
 */
template <typename tiles_kit_type_, typename quads_kit_type_, typename row_sumsq_type_, typename admits_type_,
          typename land_type_>
void scan_tiles(nk_kernel_kind_t kernel, nk::matrix_view<typename quads_kit_type_::scalar_t> rows, node_id_t first,
                void const* packed, std::span<typename quads_kit_type_::scalar_t::dot_result_t const> query_sumsqs,
                std::span<f32_t> thresholds,
                nk::matrix_span<typename quads_kit_type_::scalar_t::dot_result_t::raw_t> dots,
                row_sumsq_type_&& row_sumsq, admits_type_& admits, land_type_& land) noexcept {
    using dot_result_t = typename quads_kit_type_::scalar_t::dot_result_t;
    using raw_dot_t = typename dot_result_t::raw_t;
    using finalized_vec_t = typename quads_kit_type_::finalized_vec_t;
    static_assert(sizeof(finalized_vec_t) == 4 * sizeof(raw_dot_t), "A finalizer takes four dot lanes");
    std::size_t const rows_count = rows.extent(0), dimensions = rows.extent(1), columns = thresholds.size();
    std::array<f32_t, tiles_kit_type_::pack_columns_k> row_distances;
    for (std::size_t offset = 0; offset < rows_count; offset += tiles_kit_type_::stream_rows_k) {
        std::size_t const step_rows = min_of(tiles_kit_type_::stream_rows_k, rows_count - offset);
        nk::matrix_view<typename quads_kit_type_::scalar_t> const step = rows(nk::range(offset, offset + step_rows),
                                                                              nk::slice);
        [[maybe_unused]] nk_status_t const scored = tiles_kit_type_::dots_packed(
            step.raw_values_data(), packed, dots.data(), step_rows, columns, dimensions,
            static_cast<std::size_t>(step.stride_bytes(0)), static_cast<std::size_t>(dots.stride_bytes(0)), nullptr);
        assert(scored == nk_success_k && "A matrix engine scores any call");
        for (std::size_t row = 0; row != step_rows; ++row) {
            // The finalizer's seats are symmetric, so the stored row takes the query's against four
            // queries.
            std::span<raw_dot_t const> const row_dots = to_span(dots.row(row).as_vector());
            dot_result_t const sumsq = row_sumsq(offset + row);
            for (std::size_t column = 0; column < columns; column += 4) {
                std::array<f32_t, 4> const quad = convert_to_distances<quads_kit_type_>(
                    kernel, *reinterpret_cast<finalized_vec_t const*>(&row_dots[column]), sumsq,
                    *reinterpret_cast<finalized_vec_t const*>(&query_sumsqs[column]));
                std::memcpy(&row_distances[column], quad.data(), sizeof(quad));
            }
            bool closer = false;
            for (std::size_t column = 0; column != columns; ++column)
                closer |= row_distances[column] < thresholds[column];
            node_id_t const node = static_cast<node_id_t>(first + offset + row);
            if (!closer || !admitted(admits, node)) continue;
            for (std::size_t column = 0; column != columns; ++column)
                if (row_distances[column] < thresholds[column])
                    thresholds[column] = land(column, scored_node_t {node, row_distances[column]});
        }
    }
}

#pragma region Rescorings

/** Indexer heads as a walk's rescoring: each scored dot becomes Σₕ wₕ·ReLU(⟨qₕ, k⟩) and its
 *  distance 1 − score. */
template <nk_capability_t capability_, typename scalar_type_>
struct indexer_rescoring {

    /** The heads of the one query row being walked. */
    indexer_heads<scalar_type_> heads;

    template <typename dot_type_, std::size_t fan_out_>
    void operator()(std::array<node_id_t, fan_out_> const&,
                    std::array<nk::vector_view<scalar_type_>, fan_out_> const& keys,
                    std::array<vector_norms<dot_type_>, fan_out_> const& norms,
                    scores_result<dot_type_, fan_out_>& scores, std::size_t count) const noexcept {
        static_assert(!nk::is_integral_dtype<scalar_type_>(), "Indexer scoring needs floating-point keys");
        std::array<dot_type_, fan_out_> indexed {};
        for (std::size_t head = 0; head != heads.heads.extent(0); ++head) {
            scores_result<dot_type_, fan_out_> const head_scores = dense::scores<capability_, scalar_type_>(
                nk_kernel_dot_k, heads.heads.row(head).as_vector(), {}, keys, norms);
            dot_type_ const weight = dot_type_(heads.weights(0, head));
            for (std::size_t lane = 0; lane != count; ++lane)
                indexed[lane] += weight * max_of(head_scores.dots[lane], dot_type_ {});
        }
        for (std::size_t lane = 0; lane != count; ++lane) {
            scores.dots[lane] = indexed[lane];
            scores.distances[lane] = 1 - indexed[lane];
        }
    }
};

/** One token's query heads as the walk's rescoring and its attention fold in one: a node ranks by
 *  its best head as 1 − maxₕ⟨qₕ, k⟩, and every scored batch folds into each head's running softmax,
 *  in base 2 until @c finish. Head zero navigates the walk; a node without a value row fails
 *  @c status. */
template <nk_capability_t capability_, typename scalar_type_, typename value_lookup_type_>
struct attention_rescoring {
    using vector_view_t = nk::vector_view<scalar_type_>;
    using dot_result_t = typename scalar_type_::dot_result_t;
    using vector_norms_t = vector_norms<dot_result_t>;

    /** The token's heads, one row each. */
    nk::matrix_view<scalar_type_> queries;

    /** Resolves a scored batch to its value rows, as wide as the keys. */
    value_lookup_type_& value_lookup;

    /** The factor turning a dot into a logit. */
    f32_t score_scale;

    /** The heads' attended rows, one per head and packed. */
    nk::matrix_span<f32_t> outputs;

    /** The heads' running softmax states. */
    std::span<softmax_state_t> carry;

    /** The first refusal, which every later batch skips on. */
    status_t& status;

    /** Empties every head's row and running state ahead of the token's first walk. */
    void clear() const noexcept {
        std::memset(outputs.data(), 0, outputs.numel() * sizeof(f32_t));
        for (softmax_state_t& state : carry) state = {std::numeric_limits<f32_t>::lowest(), 0};
    }

    /** Divides every head's row by its partition and turns its base-2 maximum into a natural-log
     *  one. */
    void finish() const noexcept {
        std::size_t const columns = queries.extent(1);
        for (std::size_t head = 0; head != carry.size(); ++head) {
            f32_t const normalizer = carry[head].partition > 0 ? 1 / carry[head].partition : 0, zero = 0;
            f32_t* const row = outputs.row(head).data();
            [[maybe_unused]] nk_status_t const scaled = elementwise_chain<capability_>::kit_t::scale(
                row, columns, &normalizer, &zero, row, nullptr);
            assert(scaled == nk_success_k && "The serial tier scales on every CPU");
            carry[head].maximum_score *= std::numbers::ln2_v<f32_t>;
        }
    }

    /** Folds one scored batch, its value rows resolved by node ID. */
    void operator()(std::array<node_id_t, 4> const& ids, std::array<vector_view_t, 4> const& keys,
                    std::array<vector_norms_t, 4> const& norms, scores_result<dot_result_t, 4>& scores,
                    std::size_t count) const noexcept {
        (*this)(ids, keys, norms, scores, count, std::span<node_id_t const>(ids).first(count));
    }

    /** Folds one scored batch, its value rows resolved by @p lookup_keys: the rows' own keys, read
     *  beside the vectors by a walk that holds no key column to resolve node IDs through. */
    template <typename lookup_key_type_>
    void operator()(std::array<node_id_t, 4> const&, std::array<vector_view_t, 4> const& keys,
                    std::array<vector_norms_t, 4> const& norms, scores_result<dot_result_t, 4>& scores,
                    std::size_t count, std::span<lookup_key_type_ const> lookup_keys) const noexcept {
        if (failed(status)) return;
        std::size_t const columns = queries.extent(1), head_count = carry.size();
        std::array<vector_view_t, 4> rows;
        std::array<char const*, 4> rows_bytes;
        value_lookup(lookup_keys.first(count), std::span(rows).first(count));
        for (std::size_t lane = 0; lane != count; ++lane) {
            if (rows[lane].size() != columns) {
                status = status_t::shape_mismatch_k;
                return;
            }
            rows_bytes[lane] = rows[lane].byte_data();
        }
        f32_t const logit_scale = score_scale * std::numbers::log2e_v<f32_t>;
        for (std::size_t first = 0; first < head_count; first += 8) {
            std::size_t const group = min_of<std::size_t>(8, head_count - first);
            // One exp2 tile per scored node across the group's heads; the zeros past them
            // exponentiate harmlessly.
            std::array<std::array<f32_t, 8>, 4> weights {};
            std::array<f32_t, 8> corrections {};
            for (std::size_t head = 0; head != group; ++head) {
                vector_view_t const head_row = queries.row(first + head).as_vector();
                // A compensated finalize reads the head's own row sum beside the keys' stored ones.
                vector_norms_t head_norms {};
                if constexpr (quads_compensated<typename quads_chain<scalar_type_, capability_>::kit_t>)
                    if (first + head) head_norms = dense::norms<capability_>(head_row);
                scores_result<dot_result_t, 4> const head_scores =
                    first + head
                        ? dense::scores<capability_, scalar_type_>(nk_kernel_dot_k, head_row, head_norms, keys, norms)
                        : scores;
                softmax_state_t& state = carry[first + head];
                f32_t maximum = state.maximum_score;
                for (std::size_t lane = 0; lane != count; ++lane) {
                    weights[lane][head] = logit_scale * static_cast<f32_t>(head_scores.dots[lane]);
                    maximum = max_of(maximum, weights[lane][head]);
                    scores.dots[lane] = max_of(scores.dots[lane], head_scores.dots[lane]);
                }
                for (std::size_t lane = 0; lane != count; ++lane) weights[lane][head] -= maximum;
                corrections[head] = state.maximum_score - maximum;
                state.maximum_score = maximum;
            }
            for (std::size_t lane = 0; lane != count; ++lane)
                elementwise_chain<capability_>::kit_t::exp2(weights[lane]);
            elementwise_chain<capability_>::kit_t::exp2(corrections);
            for (std::size_t head = 0; head != group; ++head) {
                softmax_state_t& state = carry[first + head];
                state.partition *= corrections[head];
                for (std::size_t lane = 0; lane != count; ++lane) state.partition += weights[lane][head];
            }
            values_step<scalar_type_, capability_>(outputs.row(first).data(), group, columns, rows_bytes.data(),
                                                   weights.data(), corrections, count);
        }
        for (std::size_t lane = 0; lane != count; ++lane)
            scores.distances[lane] = 1 - static_cast<distance_t>(scores.dots[lane]);
    }
};

#pragma endregion Rescorings

/** Per-thread traversal scratch: candidate queues, visited set, neighbor selection buffers and
 *  the running telemetry counters - one instance per scanning thread. */
template <                                          //
    nk_dtype_t vector_dtype_ = nk_bf16_k,           //
    nk_capability_t capability_ = nk_cap_icelake_k, //
    typename allocator_type_ = std::allocator<byte_t>>
struct traversal_context;

template <nk_dtype_t vector_dtype_, nk_capability_t capability_, typename allocator_type_>
struct traversal_context {

    using scalar_t = typename nk::type_for<vector_dtype_>::type;
    using vector_view_t = nk::vector_view<scalar_t>;
    using vector_norms_t = vector_norms<typename scalar_t::dot_result_t>;

    using allocator_t = allocator_type_;
    using candidates_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<scored_node_t>;

    /** The hash set allocates raw bytes and places IDs itself, so it demands a byte allocator. */
    using visits_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<std::byte>;
    using ids_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<node_id_t>;
    using views_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<vector_view_t>;
    using norms_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<vector_norms_t>;

    using ids_tensor_t = nk::tensor<node_id_t, ids_allocator_t, 1>;
    using views_tensor_t = nk::tensor<vector_view_t, views_allocator_t, 1>;
    using norms_tensor_t = nk::tensor<vector_norms_t, norms_allocator_t, 1>;

    using raw_dot_t = typename scalar_t::dot_result_t::raw_t;
    using dots_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<raw_dot_t>;
    using bytes_tensor_t = nk::tensor<std::byte, visits_allocator_t, 1>;
    using dots_tensor_t = nk::tensor<raw_dot_t, dots_allocator_t, 2>;

    /** One selection's output: @b [kept] scored nodes. The tensor's live shape is the fill and
     *  @c capacity() the reservation, so no separate count exists. */
    using selection_t = nk::tensor<scored_node_t, candidates_allocator_t, 1>;

    using results_array_t = sorted_buffer<scored_node_t, closer_scored_node_t, candidates_allocator_t>;

    /** 8-ary: eight 8-byte scored nodes fill one cache line per level scan, and the
     *  insert-dominated frontier rides the shallower log_8 @c shift_up depth. */
    using frontier_queue_t = max_heap_dary<8, scored_node_t, further_scored_node_t, candidates_allocator_t>;

    /** The visited set walks its slots with the kit the capability mask earns: sixteen at a
     *  time on the AVX-512 classes, one at a time below them. */
    using visited_set_t = growing_hash_set<node_id_t, std::hash<node_id_t>, visits_allocator_t,
                                           typename visits_hash_set_chain<capability_>::kit_t>;

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

    /** The staging surface: the staged rows during scoring, then the kept prefix of the running
     *  selection - scoring drains the rows before any selection writes. */
    views_tensor_t staged_vectors;

    /** Stored norms, parallel to @c staged_vectors. */
    norms_tensor_t staged_norms;

    /** One block of queries packed for a matrix engine - an exact scan's operand, idle
     *  during walks. */
    bytes_tensor_t packed_queries;

    /** The raw dots one step of stored rows scores against @c packed_queries, @b [rows,queries]. */
    dots_tensor_t step_dots;

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

    traversal_context() noexcept = delete;
    traversal_context(traversal_context const&) noexcept = delete;
    traversal_context& operator=(traversal_context const&) noexcept = delete;

    /** Constructs every container with a rebound copy of @p alloc, and keeps the allocator for
     *  the lazily-sized neighbor scratches, so all scratch lands where the caller places it. */
    traversal_context(allocator_t alloc) noexcept
        : results(candidates_allocator_t(alloc)), refining_candidates(candidates_allocator_t(alloc)),
          frontier(candidates_allocator_t(alloc)), visited(visits_allocator_t(alloc)), random_generator(),
          allocator_(alloc) {}
    traversal_context(traversal_context&&) = default;
    traversal_context& operator=(traversal_context&&) = default;

    /** Sizes every neighbor scratch for @p capacity entries through the context's own allocator;
     *  geometry freezes once, so the tensors allocate once. */
    status_t reserve_neighbors(std::size_t capacity) noexcept {
        if (selected_neighbors.capacity() >= capacity) return status_t::success_k;
        auto selected = selection_t::zeros({capacity}, candidates_allocator_t(allocator_));
        auto refined = selection_t::zeros({capacity}, candidates_allocator_t(allocator_));
        auto ids = ids_tensor_t::zeros({capacity}, ids_allocator_t(allocator_));
        auto views = views_tensor_t::zeros({capacity}, views_allocator_t(allocator_));
        auto norms = norms_tensor_t::zeros({capacity}, norms_allocator_t(allocator_));
        if (!selected || !refined || !ids || !views || !norms) return status_t::bad_alloc_k;
        selected_neighbors = std::move(selected.value);
        refined_neighbors = std::move(refined.value);
        staged_ids = std::move(ids.value);
        staged_vectors = std::move(views.value);
        staged_norms = std::move(norms.value);
        return status_t::success_k;
    }

    /** Sizes the exact scan's matrix-engine scratch: @p packed_bytes of packed queries and a
     *  @p rows by @p columns dots tile; geometry freezes once, so the tensors allocate once per
     *  context. */
    status_t reserve_tiles(std::size_t packed_bytes, std::size_t rows, std::size_t columns) noexcept {
        if (packed_queries.numel() >= packed_bytes && step_dots.numel() >= rows * columns) return status_t::success_k;
        auto packed = bytes_tensor_t::zeros({packed_bytes}, visits_allocator_t(allocator_));
        auto dots = dots_tensor_t::zeros({rows, columns}, dots_allocator_t(allocator_));
        if (!packed || !dots) return status_t::bad_alloc_k;
        packed_queries = std::move(packed.value);
        step_dots = std::move(dots.value);
        return status_t::success_k;
    }

    /** Bytes @c reserve_neighbors reserves for @p capacity neighbors, each tensor on its own arena line. */
    static std::size_t neighbors_memory_usage(std::size_t capacity) noexcept {
        using arena_t = arena_allocator<std::byte>;
        return 2u * arena_t::reserved_bytes(capacity * sizeof(scored_node_t)) +
               arena_t::reserved_bytes(capacity * sizeof(node_id_t)) +
               arena_t::reserved_bytes(capacity * sizeof(vector_view_t)) +
               arena_t::reserved_bytes(capacity * sizeof(vector_norms_t));
    }

    /** Allocator instance behind every neighbor scratch; the queues hold rebound copies. */
    [[no_unique_address]] allocator_t allocator_;
};

template <                                                               //
    nk_dtype_t vector_dtype_, nk_capability_t capability_,               //
    typename wait_type_, template <typename> class atomic_ref_template_, //
    typename cache_hints_type_, typename vectors_allocator_type_, typename neighbors_allocator_type_,
    typename traversals_allocator_type_> //
struct ram_index {

    inline static constexpr nk_dtype_t vector_dtype_k = vector_dtype_;

    using scalar_t = typename nk::type_for<vector_dtype_>::type;
    using dot_result_t = typename scalar_t::dot_result_t;
    using distance_t = f32_t;
    using candidate_t = scored_node_t;

    /** Default integer type to use for loops and indexing at sub-index scale. */
    using small_size_t = u32_t;

    /** Node identifier within this sub-index - the class-local spelling of @c dense::node_id_t. */
    using node_id_t = dense::node_id_t;

    using vectors_allocator_t =
        typename std::allocator_traits<vectors_allocator_type_>::template rebind_alloc<scalar_t>;
    using neighbors_allocator_t =
        typename std::allocator_traits<neighbors_allocator_type_>::template rebind_alloc<u32_t>;
    using candidates_allocator_t =
        typename std::allocator_traits<traversals_allocator_type_>::template rebind_alloc<candidate_t>;

    using vector_view_t = nk::vector_view<scalar_t>;
    using vector_span_t = nk::vector_span<scalar_t>;

    /** What the norms chunks store per row, for the metrics whose finalize reads them. */
    using vector_norms_t = vector_norms<dot_result_t>;

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
    friend struct ram_cursor;

    // The sharded index walks every shard under one token's attention heads, so one running state spans them.
    template <typename, nk_dtype_t, nk_capability_t, typename, template <typename> class, typename, typename>
    friend struct ram_indexes;

    /** The capability bit of the distance kit the chain actually picked - lower than
     *  @c capability_ when no SIMD kit is adapted for this dtype yet. */
    inline static constexpr nk_capability_t picked_quads_capability_k = quads_chain<scalar_t, capability_>::picked_k;

    /** The distance kit the chain picked; its step is the row pitch the chunks round to. */
    using quads_kit_t = typename quads_chain<scalar_t, capability_>::kit_t;

    /** The matrix-engine kits an exact scan streams through, @c no_tiles_kit_t where no engine
     *  multiplies the scalar. */
    using tiles_chain_t = tiles_chain<scalar_t, capability_>;

    /** Each thread scanning the graph needs its own object like this. */
    using traversal_context_t = traversal_context<vector_dtype_, capability_, candidates_allocator_t>;
    using results_array_t = typename traversal_context_t::results_array_t;
    using frontier_queue_t = typename traversal_context_t::frontier_queue_t;
    using visited_set_t = typename traversal_context_t::visited_set_t;
    using selection_t = typename traversal_context_t::selection_t;
    using ids_tensor_t = typename traversal_context_t::ids_tensor_t;

    /** The growth schedule of the vectors, norms & neighbors - 1'024 rows first, doubling up to
     *  1'048'576-row slabs - shared with the owner columns that grow beside the index.
     *  @ref dense_ram_index_malloc */
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

    /** The allocator the vectors & norms grow through, whose memory domain the sharded index
     *  places by. */
    [[no_unique_address]] vectors_allocator_t vectors_allocator_;

    ram_index(ram_index const&) noexcept = delete;
    ram_index& operator=(ram_index const&) noexcept = delete;

    ram_index(ram_index&&) noexcept = delete;
    ram_index& operator=(ram_index&&) noexcept = delete;

    /** Constructs an empty index walking by @p kernel, its vectors & norms growing through
     *  @p vectors_allocator and its neighbors through @p neighbors_allocator; a @p kernel the
     *  dtype's kits cannot finalize is refused at the first @c reserve. */
    explicit ram_index(nk_kernel_kind_t kernel, vectors_allocator_type_ const& vectors_allocator = {},
                       neighbors_allocator_type_ const& neighbors_allocator = {}) noexcept
        : vectors_allocator_(vectors_allocator), vectors_(vectors_allocator), norms_(vectors_allocator),
          neighbors_(neighbors_allocator), kernel_(kernel) {}

#pragma region Measures

    /** The distance kind every walk ranks by, fixed at construction. */
    nk_kernel_kind_t kernel() const noexcept { return kernel_; }

    /** Whether rows keep norms: every metric but dot finalizes through them, and so does dot on a
     *  kit whose finalize subtracts the candidates' row sums. Otherwise the norms chunks stay
     *  empty, and checkpoints carry no norms column. */
    bool stores_norms() const noexcept { return kernel_ != nk_kernel_dot_k || quads_compensated<quads_kit_t>; }

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
     *  capacity and the growth schedule rather than from the chunks themselves, so it stays safe
     *  under concurrent growth. Reserved-but-empty rows count from the moment they are granted. */
    std::size_t memory_usage() const noexcept {
        geometry_t const frozen = geometry();
        if (!frozen) return sizeof(ram_index);
        std::size_t const hosted_rows = capacity();
        small_size_t chunks = 0;
        while (chunks != max_chunks_k && chunk_schedule_t::rows_before(chunks) < hosted_rows) ++chunks;
        std::size_t const rows = chunk_schedule_t::rows_before(chunks);
        std::size_t bytes = sizeof(ram_index);
        bytes += rows * padded_row_values(frozen) * sizeof(scalar_t);
        if (stores_norms()) bytes += rows * sizeof(vector_norms_t);
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

    /** Where a save lands: row-major columns the caller sized from @ref saved_shape, handed
     *  over at planning so no step names a destination of its own. */
    struct columns_t {

        /** The vectors matrix, one packed row per node. */
        bytes_span_t vectors;

        /** The neighbors matrix, @c connectivity IDs per node. */
        bytes_span_t neighbors;

        /** The norms column, one @c vector_norms_t per node, or empty to write none; only an index
         *  that keeps norms takes one. */
        bytes_span_t norms;
    };

    /** Where a load reads from - the read-only twin of @ref columns_t. */
    struct columns_view_t {

        /** The vectors matrix, one packed row per node. */
        bytes_view_t vectors;

        /** The neighbors matrix, @c connectivity IDs per node. */
        bytes_view_t neighbors;

        /** The norms column, one @c vector_norms_t per node, or empty to derive them from the
         *  vectors; only an index that keeps norms takes one. */
        bytes_view_t norms;
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
     *  receives @c status_t::config_mismatch_k, as does a geometry with no dimensions, with
     *  sub-byte dimensions that leave a storage value partly filled, or with a connectivity that
     *  is not a power of two. Constructors allocate nothing, so every storage failure surfaces
     *  here, in the returned status, and so does a kernel the dtype's kits cannot finalize, as
     *  @c status_t::unknown_kind_k.
     */
    expected<std::size_t> reserve(reserve_capacity_t config) noexcept {
        if (!kernel_supported<quads_kit_t>(kernel_)) return {0, status_t::unknown_kind_k};
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

    static status_t reserve_queues_(traversal_context_t& context, queue_sizes_t sizes) noexcept {
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
        using context_t = traversal_context_t;
        using arena_t = arena_allocator<std::byte>;
        queue_sizes_t const sizes = queue_sizes_(expansion_factor, node_connectivity);
        std::size_t bytes = arena_t::reserved_bytes(context_t::results_array_t::memory_usage(sizes.results)) +
                            arena_t::reserved_bytes(
                                context_t::results_array_t::memory_usage(sizes.refining_candidates)) +
                            arena_t::reserved_bytes(context_t::frontier_queue_t::memory_usage(sizes.frontier)) +
                            arena_t::reserved_bytes(context_t::visited_set_t::memory_usage(sizes.visited)) +
                            context_t::neighbors_memory_usage(node_connectivity);
        return bytes;
    }

    /**
     *  @brief Sizes @p context 's scratch for the frozen geometry and @p config - the other
     *      fallible half of an insertion, hoistable ahead of a fan-out.
     *
     *  Idempotent and monotone. The pools and the neighbor scratch never allocate again and the
     *  frontier sheds within its reservation; the visited set alone may re-table as a walk widens.
     */
    status_t reserve_traversal(traversal_context_t& context, insert_config_t config = {}) noexcept {
        geometry_t const frozen = geometry();
        if (!frozen) return status_t::uninitialized_k;
        std::size_t const node_connectivity = frozen.connectivity();
        std::size_t const expansion_factor = non_zero_expansion_factor(config);

        // `sorted_buffer::insert(.., limit)` never grows on its own, and `growing_hash_set::set`
        // never rehashes, so these reservations are load-bearing rather than an optimization.
        if (status_t const sized = reserve_queues_(context, queue_sizes_(expansion_factor, node_connectivity));
            failed(sized))
            return sized;
        return context.reserve_neighbors(node_connectivity);
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
     *  One add dispenses the batch's IDs; every row lands its payload, walks the graph for candidates
     *  and is wired, then one watermark turn publishes them all. All fallible scratch allocations
     *  happen before the dispense; once IDs are dispensed, this thread must roll forward to
     *  publication. @ref dense_ram_index_concurrency
     *
     *  @param[in] new_vectors The batch to insert, one row per vector, in the frozen geometry.
     *  @param[out] assigned_ids Either empty, or one slot per incoming vector receiving its node
     *      ID. One batch's IDs are contiguous, in the order of its rows.
     *  @param[inout] context The calling thread's scratch & telemetry, reused across batches.
     *  @param[in] config Optional tuning knobs; zeros mean "derive from connectivity".
     */
    expected<insert_result_t> insert(vectors_view_t new_vectors, node_ids_span_t assigned_ids,
                                     traversal_context_t& context, insert_config_t config = {}) noexcept {

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
    status_t remove(node_ids_view_t, traversal_context_t&) noexcept = delete;

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
        capacity_claim_t(ram_index& index, std::size_t rows, node_id_t base) noexcept
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

        /** Marks the rows as taken, so destruction releases nothing; dispensing their IDs and a load
         *  plan's @c finish call it. */
        void consume() noexcept { rows_ = 0; }

      private:

        /** The granting index, reached only to release; null marks the empty state. */
        ram_index* index_ = nullptr;

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
     *  One add dispenses the claim's IDs, every row is wired, and one watermark turn publishes them
     *  together. @p on_assigned fires per row with the assigned node ID, before the walk and before
     *  publication - so an outer layer's per-ID state, like a key column, is written ahead of the
     *  watermark turn and no reachable node can be observed without it.
     */
    template <typename callback_type_ = dummy_callback_t>
    void insert(capacity_claim_t&& claim, vectors_view_t new_vectors, node_ids_span_t assigned_ids,
                traversal_context_t& context, insert_config_t config = {}, callback_type_&& on_assigned = {}) noexcept {
        assert(claim.rows() == new_vectors.extent(0) && "A claim holds exactly its batch's rows");
        // An empty range takes no turn: its first ID opens the next range too, which may publish first.
        if (!claim) return;
        bool const report_assigned_ids = assigned_ids.size() != 0;
        iota_view<node_id_t> const range = dispense_(std::move(claim));
        for (node_id_t const new_node_id : range) {
            std::size_t const row = new_node_id - range.first;
            if (report_assigned_ids) assigned_ids[row] = new_node_id;
            on_assigned(new_node_id);
            wire_(new_node_id, new_vectors.row(row).as_vector(), context, config);
        }
        await_turn_(range);
        publish_(range);
    }

#pragma endregion Insertion

#pragma region Refinement

    /** Optional tuning knobs for @c refine(); zeros mean "derive from connectivity". */
    struct refine_config_t {

        /** Breadth of the re-selection pool; defaults to twice the connectivity, like every walk. */
        small_size_t expansion_factor = 0;
    };

    using refine_result_t = dense::refine_result_t;
    using health_result_t = dense::health_result_t;

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
        health_plan_t(ram_index const& index, node_id_t visible, std::size_t seed_count, growing_bitset_t&& reached,
                      growing_bitset_t&& expanded) noexcept
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
                while (first_value == node_id_unknown_k) {
                    wait_type_ {}(row.data(), node_id_unknown_k, small_size_t(0), fu::wait_capped_t {});
                    first_value = atomic_ref<node_id_t const>(row[0]).load(std::memory_order_acquire);
                }
                if (first_value == node_id) continue; // Published with no neighbors.
                for (std::size_t slot = 0; slot != row.size(); ++slot) {
                    node_id_t const target =
                        slot ? atomic_ref<node_id_t const>(row[slot]).load(std::memory_order_relaxed) : first_value;
                    if (target == node_id_unknown_k) break;
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
            if (!seed_count || !visible) return node_id_unknown_k;
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
        ram_index const* index_ = nullptr;

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
    expected<health_plan_t> plan_health(traversal_context_t& context) const noexcept {
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
        refine_plan_t(ram_index& index, health_plan_t&& audit, iota_view<node_id_t> nodes,
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
        status_t run(step_t const& step, traversal_context_t& context) noexcept {
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
        status_t reselect_(step_t const& step, traversal_context_t& context) noexcept {
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
        void reconnect_(step_t const& step, traversal_context_t& context) noexcept {
            std::size_t reconnected = 0;
            small_size_t const connectivity = static_cast<small_size_t>(index_->connectivity());
            small_size_t const spread = heal_spread(connectivity);
            for (node_id_t island = step.nodes.first; island < step.nodes.bound; ++island) {
                if (audit.reached(island)) continue;
                neighbors_ref_t const own_neighbors = index_->neighbors_for(chunk_schedule_t::locate(island));
                small_size_t const staged_count = index_->stage_neighbors_(own_neighbors, island, context);
                node_id_t target = node_id_unknown_k;
                for (small_size_t staged = 0; staged != staged_count && target == node_id_unknown_k; ++staged) {
                    node_id_t const pick = context.staged_ids[staged];
                    if (pick < audit.visible && audit.reached(pick)) target = pick;
                }
                // When a cluster picks only its own, the last resort is the landmark covering this
                // ID: a walk starts there by definition, which is what makes it reached.
                if (target == node_id_unknown_k) target = audit.landmark_covering(island);
                if (target == node_id_unknown_k || target == island) continue;

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
                    if (resident == node_id_unknown_k) {
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
        ram_index* index_ = nullptr;

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
    expected<refine_plan_t> plan_refine(iota_view<node_id_t> nodes, traversal_context_t& context,
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
    expected<refine_result_t> refine(iota_view<node_id_t> nodes, traversal_context_t& context,
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

        /** How rows are reached; @c scan_k scores every published row exactly, keeping the
         *  breadth's nearest, and only @c search and @c search_if implement it. */
        traversal_t traversal = traversal_t::walk_k;
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
    expected<search_result_t> search(vector_view_t query_vector, traversal_context_t& context,
                                     search_config_t config = {}) noexcept {
        if (config.traversal == traversal_t::scan_k)
            return scan_if_(query_vector, dummy_predicate_t {}, context, config, static_cast<node_id_t>(size()));
        return search(query_vector, dummy_fold_t {}, context, config);
    }

    /**
     *  @brief Finds the published nodes a lightning indexer scores highest for one query row - the
     *      @c search contract, with every distance 1 − the node's indexer score.
     *
     *  @param[in] query_vector The row the walk navigates by, conventionally Σₕ wₕ·qₕ.
     *  @param[in] heads The indexer heads of this one query row.
     *  @param[inout] context The calling thread's scratch & telemetry: the matches end up in
     *      @c context.results, ascending by distance - the highest score first.
     *  @param[in] config Optional tuning knobs; zeros mean "derive from connectivity".
     *  @return @c config_mismatch_k unless the index walks by dot.
     */
    expected<search_result_t> search(vector_view_t query_vector, indexer_heads<scalar_t> const& heads,
                                     traversal_context_t& context, search_config_t config = {}) noexcept
        requires floating_scalar<scalar_t>
    {
        if (kernel_ != nk_kernel_dot_k) return {{}, status_t::config_mismatch_k};
        return walk_(query_vector, dummy_fold_t {}, indexer_rescoring<capability_, scalar_t> {heads}, context, config,
                     static_cast<node_id_t>(size()));
    }

    /**
     *  @brief Walks exactly like @c search while @p fold reads every quad the walk scores - the
     *      entry point for verbs that reduce a walk to something other than its nearest matches.
     *
     *  @param[in] query_vector The query, in the index's own dtype & frozen dimensionality.
     *  @param[inout] fold The reader of every scored quad - see @c fold.
     *  @param[inout] context The calling thread's scratch & telemetry: the nearest published
     *      candidates end up in @c context.results, ascending by distance.
     *  @param[in] config Optional tuning knobs; zeros mean "derive from connectivity".
     */
    template <typename fold_type_>
        requires fold<std::remove_cvref_t<fold_type_>, dot_result_t>
    expected<search_result_t> search(vector_view_t query_vector, fold_type_&& fold, traversal_context_t& context,
                                     search_config_t config = {}) noexcept {
        return walk_(query_vector, fold, dummy_rescoring_t {}, context, config, static_cast<node_id_t>(size()));
    }

    /**
     *  @brief Finds the nearest published neighbors of @p query_vector that satisfy
     *      @p predicate - the filtered sibling of @c search, sharing its whole contract.
     *
     *  Non-matching nodes still score and route the walk - filtering navigation itself would
     *  strand it in disconnected islands - but only matching nodes enter the reported pool, whose
     *  radius governs termination once it holds @c expansion_factor entries; until then the
     *  ordinary navigation radius bounds the frontier. Predicates come in two shapes, detected by
     *  signature: per-node @c admission_t(node_id_t), or batched - the scored quad and an
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
                                        traversal_context_t& context, search_config_t config = {}) noexcept {
        if (config.traversal == traversal_t::scan_k) {
            // A batched predicate meets each row that beats the worst match as a quad of one.
            auto admits = [&](node_id_t node) noexcept {
                scored_node_t const candidate {node, 0};
                admissions_t admissions;
                invoke_admission(predicate, std::span(&candidate, 1), admissions);
                return admissions.empty() ? admission_t::reject_k : admission_t::admit_k;
            };
            return scan_if_(query_vector, admits, context, config, static_cast<node_id_t>(size()));
        }
        return walk_if_(query_vector, predicate, context, config, static_cast<node_id_t>(size()));
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
     *  @return The pairs scored in @c count - above the spans' width when they were truncated - or
     *      @c config_mismatch_k unless the index walks by dot.
     */
    expected<search_result_t> multiply(vector_view_t left, std::span<node_id_t> node_ids,
                                       std::span<dot_result_t> products, traversal_context_t& context,
                                       search_config_t config = {}) noexcept {
        if (kernel_ != nk_kernel_dot_k || products.size() != node_ids.size()) return {{}, status_t::config_mismatch_k};
        multiply_fold<node_id_t, dot_result_t> fold {node_ids, products};
        expected<search_result_t> walked = search(left, fold, context, config);
        walked.value.count = fold.scored;
        return walked;
    }

    /**
     *  @brief Softmax attention of query heads over every node their token's walk scores, folded as Flash
     *      Attention folds key tiles: the @p head_count heads of one token share one walk.
     *
     *  The walk is @c search's own frontier and stopping rule, ranking a node by its best head as
     *  1 − maxₕ⟨qₕ, k⟩, so every head's own region is explored. Each scored batch meets all the token's
     *  heads at once: every head rescales its running maximum, partition and accumulator as its maximum rises,
     *  and each value row is resolved once and added into every head.
     *
     *  @param[in] queries Row t · @p head_count + h is head h of token t, in the index's own dtype.
     *  @param[in] head_count Query heads sharing this key-value head; it must divide the rows.
     *  @param[in] value_lookup Resolves a scored batch's node IDs to their value rows, as wide as
     *      the keys and stored like them:
     *      `void(std::span<node_id_t const>, std::span<vector_view_t>)`, one row per ID. A node
     *      without a row refuses with @c shape_mismatch_k, since a missing row is a wrong softmax.
     *  @param[in] score_scale The factor turning a dot into a logit.
     *  @param[out] outputs The attended rows, one per query row, as wide as the keys, packed and
     *      divided by their partitions.
     *  @param[out] carry Every query row's softmax state, which merges this walk with other branches.
     *  @param[inout] context The calling thread's scratch & telemetry.
     *  @param[in] config Optional tuning knobs; zeros mean "derive from connectivity".
     *  @return The query rows attended in @c count, or @c config_mismatch_k unless the index walks
     *      by dot.
     */
    template <typename value_lookup_type_>
    expected<search_result_t> attend(vectors_view_t queries, std::size_t head_count, value_lookup_type_&& value_lookup,
                                     f32_t score_scale, nk::matrix_span<f32_t> outputs,
                                     std::span<softmax_state_t> carry, traversal_context_t& context,
                                     search_config_t config = {}) noexcept {
        if (kernel_ != nk_kernel_dot_k) return {{}, status_t::config_mismatch_k};
        std::size_t const rows_count = queries.extent(0), columns = queries.extent(1);
        if (!head_count || rows_count % head_count) return {{}, status_t::shape_mismatch_k};
        if (outputs.extent(0) != rows_count || outputs.extent(1) != columns || carry.size() != rows_count)
            return {{}, status_t::config_mismatch_k};
        if (!outputs.is_contiguous()) return {{}, status_t::unsupported_layout_k};
        status_t status = status_t::success_k;
        search_result_t attended;
        for (std::size_t first = 0; first != rows_count; first += head_count) {
            attention_rescoring<capability_, scalar_t, value_lookup_type_> const rescoring {
                queries(nk::range(first, first + head_count), nk::slice),
                value_lookup,
                score_scale,
                outputs(nk::range(first, first + head_count), nk::slice),
                carry.subspan(first, head_count),
                status};
            rescoring.clear();
            expected<search_result_t> const walked = walk_(rescoring.queries.row(0).as_vector(), dummy_fold_t {},
                                                           rescoring, context, config, static_cast<node_id_t>(size()));
            if (!walked) return {attended, walked.status};
            if (failed(status)) return {attended, status};
            rescoring.finish();
            attended.visited_members += walked.value.visited_members;
            attended.computed_distances += walked.value.computed_distances;
            attended.restarts += walked.value.restarts;
            attended.saturated_walks += walked.value.saturated_walks;
        }
        attended.count = rows_count;
        return {attended, status_t::success_k};
    }

    /** The compile-time form of @c search() & @c search_if(): one whole-traversal instantiation per
     *  @p synchronization_ contract, per fold type and per rescoring type, so a certified contract carries
     *  no atomics and @c dummy_fold_t & @c dummy_rescoring_t inline away. The walk ends against @p bound
     *  once it fills - @c context.results itself for every walk but a filtered one. */
    template <synchronization_t synchronization_, typename fold_type_, typename rescoring_type_>
        requires fold<std::remove_cvref_t<fold_type_>, dot_result_t>
    expected<search_result_t> search_typed(vector_view_t query_vector, traversal_context_t& context,
                                           search_config_t config, fold_type_&& fold, results_array_t const& bound,
                                           rescoring_type_ const& rescoring) noexcept {
        // Everything strictly below the watermark is published. A mutation-free contract needs no
        // acquire - the caller's own synchronization already ordered every publication.
        constexpr std::memory_order watermark_order_k = synchronization_ == synchronization_t::concurrent_k
                                                            ? std::memory_order_acquire
                                                            : std::memory_order_relaxed;
        node_id_t const first_invisible_id = atomic_ref<node_id_t>(last_visible_id_).load(watermark_order_k);
        return search_below_<synchronization_>(query_vector, context, config, fold, bound, rescoring,
                                               first_invisible_id);
    }

#pragma endregion Search

#pragma region Checkpoints

    /**
     *  @brief One save in flight: the watermark it took, the contract it runs under, and the storage
     *      chunks it hands out to any number of threads. Vectors and sentinel-padded neighbor rows
     *      are what a snapshot holds, and the norms when the columns carry one for them.
     *
     *  A checkpointing reader, not a mutator: concurrent searches keep running on every arm.
     *  Under @c concurrent_k every row is copied under its slot-zero lock, as writers edit it, and
     *  @ref finish probes for settlement across the whole plan, since a later insert back-linking
     *  itself edits an earlier row; the certified arms read plain, and a violated certification
     *  surfaces as @c contended_k rather than a silently torn snapshot.
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
        save_plan_t(ram_index& index, columns_t columns, synchronization_t synchronization, node_id_t id_shift) noexcept
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

        /** A save owns nothing - no rows claimed, no lock held between steps - so a dying plan
         *  leaves the index as it found it. */
        ~save_plan_t() noexcept = default;

        /** One step of the plan: one storage chunk's share of the rows under the watermark. */
        struct step_t {

            /** The step's position in the plan - what a fan-out hands to a thread. */
            std::size_t position = 0;

            /** Where the rows sit in the index, in its own ID space. */
            iota_view<node_id_t> index_rows;

            /** Where they sit in the caller's buffer, counted from its first row. */
            iota_view<node_id_t> source_rows;
        };

        /** Step @p position of the plan; an empty one past the last. */
        step_t operator[](std::size_t position) const noexcept {
            step_t step;
            if (position >= steps().size()) return step;
            std::size_t const first = chunk_schedule_t::rows_before(static_cast<small_size_t>(position));
            std::size_t const bound = min_of(chunk_schedule_t::rows_before(static_cast<small_size_t>(position + 1)),
                                             static_cast<std::size_t>(visible));
            step.position = position;
            step.index_rows = {static_cast<node_id_t>(first), static_cast<node_id_t>(bound)};
            step.source_rows = step.index_rows;
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
         *  @brief Copies step @p step 's rows into the plan's own columns: the vectors and their
         *      norms straight from chunk memory, since neither changes once published, and the
         *      neighbors rebased through @p scratch.
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
            std::size_t const connectivity = frozen.connectivity();
            std::size_t const step_slots = rows.size() * connectivity;
            // Chunk rows are padded to whole kit steps, while checkpoints keep logical rows.
            std::size_t const row_bytes = stored_row_values(frozen) * sizeof(scalar_t);
            std::size_t const first = step.source_rows.first;
            bytes_span_t const vectors_rows = flat_rows(columns.vectors, first, rows.size(), row_bytes);
            bytes_span_t const neighbors_rows = flat_rows(columns.neighbors, first, rows.size(),
                                                          connectivity * sizeof(node_id_t));
            bytes_span_t const norms_rows = flat_rows(columns.norms, first, rows.size(), sizeof(vector_norms_t));
            if (vectors_rows.empty() || neighbors_rows.empty() || norms_rows.empty() != columns.norms.empty())
                return status_t::config_mismatch_k;
            if (scratch.extent(0) < rows.size() || scratch.extent(1) != connectivity || !scratch.is_contiguous())
                return status_t::config_mismatch_k;

            location_t const location = chunk_schedule_t::locate(rows.first);
            for (std::size_t row = 0; row != rows.size(); ++row)
                std::memcpy(vectors_rows.data() + row * row_bytes,
                            index_->vectors_[location_t {location.chunk, location.offset + row}], row_bytes);
            if (index_->stores_norms() && !norms_rows.empty())
                std::memcpy(norms_rows.data(), &index_->norms_[location], norms_rows.size());
            atomic_ref<std::size_t>(covered_vectors).add(rows.size(), std::memory_order_release);

            node_id_t const* source = index_->neighbors_[location];
            node_id_t* staged = scratch.data();
            if (synchronization == synchronization_t::concurrent_k)
                for (std::size_t row = 0; row != rows.size(); ++row) {
                    neighbors_ref_t const neighbors = index_->neighbors_for(
                        chunk_schedule_t::locate(static_cast<node_id_t>(rows.first + row)));
                    node_id_t* const staged_row = staged + row * connectivity;
                    // A refinement relocks published rows and moves no counter `finish` re-reads.
                    staged_row[0] = neighbors.lock();
                    for (small_size_t slot = 1; slot != neighbors.capacity(); ++slot)
                        staged_row[slot] = neighbors.load_slot(slot);
                    neighbors.unlock(staged_row[0]);
                }
            else std::memcpy(staged, source, step_slots * sizeof(node_id_t));
            for (std::size_t word = 0; word != step_slots; ++word)
                staged[word] = staged[word] < visible ? static_cast<node_id_t>(staged[word] + id_shift)
                                                      : node_id_unknown_k;
            std::memcpy(neighbors_rows.data(), staged, neighbors_rows.size());
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
        ram_index* index_ = nullptr;
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
                                    node_id_t id_shift = 0) noexcept {
        geometry_t const frozen = geometry();
        if (!frozen) return {{}, status_t::config_mismatch_k};
        save_plan_t plan {*this, columns, synchronization, id_shift};
        plan.visible = atomic_ref<node_id_t const>(last_visible_id_).load(std::memory_order_acquire);
        plan.dispensed = atomic_ref<node_id_t const>(next_dispensed_id_).load(std::memory_order_acquire);
        plan.allocated = atomic_ref<node_id_t const>(last_allocated_id_).load(std::memory_order_acquire);
        if (plan.visible != plan.dispensed) return {{}, status_t::contended_k};
        std::size_t const rows = static_cast<std::size_t>(plan.visible);
        if (columns.vectors.size() < rows * stored_row_values(frozen) * sizeof(scalar_t) ||
            columns.neighbors.size() < rows * frozen.connectivity() * sizeof(node_id_t) ||
            (!columns.norms.empty() && (!stores_norms() || columns.norms.size() < rows * sizeof(vector_norms_t))))
            return {{}, status_t::config_mismatch_k};
        return {std::move(plan), status_t::success_k};
    }

    /** Serializes every row under the watermark into @p columns on the calling thread - the
     *  one-shot over @ref plan_save for a caller with no threads to fan the steps out over. */
    status_t save(columns_t columns, synchronization_t synchronization, nk::matrix_span<node_id_t> scratch) noexcept {
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
     *  local to the rows it delivers, so every chunk shifts them onto the appended range and folds
     *  every other ID onto the sentinel; norms land beside the vectors, or re-derive from them
     *  where the source carries none. Repeated plans append, which is how several parts land in one
     *  shard. The plan owns the claim its rows ride in: one that dies before @c finish releases
     *  them, so the next plan appends at the same base and the rows it half-filled are never
     *  published.
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
        load_plan_t(ram_index& index, columns_view_t columns, iota_view<node_id_t> source_range, node_id_t base,
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
            step.position = position;
            step.index_rows = {static_cast<node_id_t>(first), static_cast<node_id_t>(bound)};
            step.source_rows = {static_cast<node_id_t>(first - base), static_cast<node_id_t>(bound - base)};
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
         *      memory: the vectors with their norms, copied or derived as they land, then the
         *      neighbors.
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
         *  @param[in] columns The part's own columns.
         *  @param[in] source_range The ID space the part's rows speak.
         */
        status_t load(step_t const& step, columns_view_t columns, iota_view<node_id_t> source_range) noexcept {
            if (!index_) return status_t::uninitialized_k;
            iota_view<node_id_t> const range = step.index_rows;
            if (range.empty()) return status_t::config_mismatch_k;
            geometry_t const frozen = index_->geometry();
            std::size_t const connectivity = frozen.connectivity();
            std::size_t const step_slots = range.size() * connectivity;
            // Checkpoints keep logical rows, while chunk rows are padded to whole kit steps.
            std::size_t const row_bytes = stored_row_values(frozen) * sizeof(scalar_t);
            std::size_t const first = step.source_rows.first;
            bytes_view_t const vectors_rows = flat_rows(columns.vectors, first, range.size(), row_bytes);
            bytes_view_t const neighbors_rows = flat_rows(columns.neighbors, first, range.size(),
                                                          connectivity * sizeof(node_id_t));
            bytes_view_t const norms_rows = flat_rows(columns.norms, first, range.size(), sizeof(vector_norms_t));
            if (vectors_rows.empty() || neighbors_rows.empty() || norms_rows.empty() != columns.norms.empty())
                return status_t::config_mismatch_k;
            // The part's rows land at this step's own index rows, so an ID's shift follows from
            // the step rather than from the plan's base - the two agree for a one-part plan.
            node_id_t const shift = static_cast<node_id_t>(range.first - step.source_rows.first + source_range.first);
            // A rebased ID past this plan would name whichever row a later insert fills.
            if (static_cast<std::size_t>(shift) + source_range.size() > static_cast<std::size_t>(base) + rows)
                return status_t::config_mismatch_k;

            location_t const location = chunk_schedule_t::locate(range.first);
            for (std::size_t row = 0; row != range.size(); ++row)
                std::memcpy(index_->vectors_[location_t {location.chunk, location.offset + row}],
                            vectors_rows.data() + row * row_bytes, row_bytes);
            if (index_->stores_norms()) {
                if (!norms_rows.empty()) std::memcpy(&index_->norms_[location], norms_rows.data(), norms_rows.size());
                else
                    for (std::size_t offset_in_chunk = 0; offset_in_chunk != range.size(); ++offset_in_chunk) {
                        location_t const row {location.chunk, location.offset + offset_in_chunk};
                        index_->norm_ref_for(row) = index_->norms(index_->vector_for(row));
                    }
            }
            atomic_ref<std::size_t>(covered_vectors).add(range.size(), std::memory_order_release);

            node_id_t* const neighbors = index_->neighbors_[location];
            std::memcpy(neighbors, neighbors_rows.data(), neighbors_rows.size());
            std::span<node_id_t> const landed {neighbors, step_slots};
            rebase_neighbors(landed, source_range, shift, node_id_unknown_k);
            compact_neighbors(landed, connectivity, range.first, node_id_unknown_k);
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
        ram_index* index_ = nullptr;

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
     *  @param[in] columns Where the rows come from, every column but an empty norms one sized for
     *      @p rows.
     *  @param[in] rows Rows to append.
     *  @param[in] source_range The ID space the source's rows speak.
     */
    expected<load_plan_t> plan_load(columns_view_t columns, std::size_t rows,
                                    iota_view<node_id_t> source_range) noexcept {
        geometry_t const frozen = geometry();
        if (!frozen) return {{}, status_t::config_mismatch_k};
        if (columns.vectors.size() < rows * stored_row_values(frozen) * sizeof(scalar_t) ||
            columns.neighbors.size() < rows * frozen.connectivity() * sizeof(node_id_t) ||
            (!columns.norms.empty() && (!stores_norms() || columns.norms.size() < rows * sizeof(vector_norms_t))))
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

#pragma region Storage

    /** Trampoline for @c dense::norms with hardened types - zeroed moments, without
     *  touching the vector, when the rows keep none. */
    inline vector_norms_t norms(vector_view_t vec) const noexcept {
        return stores_norms() ? dense::norms<capability_, scalar_t>(vec) : vector_norms_t {};
    }

    /** Trampoline for @c dense::distances with hardened types. */
    inline std::array<distance_t, 4> distances(          //
        vector_view_t query, vector_norms_t query_norms, //
        std::array<vector_view_t, 4> candidates, std::array<vector_norms_t, 4> candidates_norms) const noexcept {
        return dense::distances<capability_, scalar_t>(kernel_, query, query_norms, candidates, candidates_norms);
    }

    /** Trampoline for @c dense::scores with hardened types - the walk's scoring,
     *  whose dots a fold reads. */
    inline scores_result<dot_result_t, 4> scores(        //
        vector_view_t query, vector_norms_t query_norms, //
        std::array<vector_view_t, 4> candidates, std::array<vector_norms_t, 4> candidates_norms) const noexcept {
        return dense::scores<capability_, scalar_t>(kernel_, query, query_norms, candidates, candidates_norms);
    }

    /** As @c scores, streaming @p next_candidates in under this CPU class's cache hints. */
    inline scores_result<dot_result_t, 4> scores(                                                //
        vector_view_t query, vector_norms_t query_norms,                                         //
        std::array<vector_view_t, 4> candidates, std::array<vector_norms_t, 4> candidates_norms, //
        std::array<vector_view_t, 4> next_candidates) const noexcept {
        return dense::scores<capability_, scalar_t>( //
            kernel_, query, query_norms, candidates, candidates_norms, next_candidates, cache_hints_type_ {});
    }

    /** Scores @p query against four consecutive entries of the cached @p vectors & @p norms
     *  arrays starting at @p start; the gather cursor sticks at @p count - 1, so short tails
     *  repeat the last entry and the kernel always streams four valid lanes. */
    inline std::array<distance_t, 4> distances_via_quads(          //
        vector_view_t query, vector_norms_t query_norms,           //
        vector_view_t const* vectors, vector_norms_t const* norms, //
        small_size_t start, small_size_t count) const noexcept {
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
     *  Slot 0 is the combined lock & first-neighbor word: the all-ones @c node_id_unknown_k
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
                while (peek_first() == node_id_unknown_k) wait_while_locked();
                node_id_t const displaced = atomic_ref<node_id_t>(row_[0]).exchange( //
                    node_id_unknown_k, std::memory_order_acq_rel);
                if (displaced != node_id_unknown_k) return displaced;
            }
        }

        /** Monitored capped wait on the lock word until it stops reading as the sentinel: the
         *  holder's unlock store wakes us, and the cap guards the arming race. Readers use it too
         *  - a bare re-peek loop spins acquiring loads for the whole critical section. */
        inline void wait_while_locked() const noexcept {
            wait_type_ {}(row_.data(), node_id_unknown_k, 0u, fu::wait_capped_t {});
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
            while (live != capacity() && row_[live] != node_id_unknown_k) ++live;
            for (small_size_t slot = live; slot != capacity(); ++slot)
                if (row_[slot] != node_id_unknown_k) return false;
            for (small_size_t slot = 1; slot != live; ++slot)
                for (small_size_t earlier = 0; earlier != slot; ++earlier)
                    if (row_[slot] == (earlier ? row_[earlier] : first_value)) return false;
            return first_value != node_id_unknown_k;
        }
    };

    /** The writer-side spelling: mutating paths are always fully synchronized. */
    using neighbors_ref_t = neighbors_ref<synchronization_t::concurrent_k>;

    /** Immutable view of the vector stored at @p location. */
    inline vector_view_t vector_for(location_t const& location) const noexcept {
        return {vectors_[location], vectors_.columns() * nk::dimensions_per_value<scalar_t>()};
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
                if (neighbor_id == node_id_unknown_k) break;
                if (neighbor_id >= first_invisible_id) continue;
                lane_vectors[lane++] = vector_for(chunk_schedule_t::locate(neighbor_id));
            }
        }
        return lane_vectors;
    }

    /** Mutable span over the vector slot at @p location - only for the slot's unpublished owner. */
    inline vector_span_t vector_span_for(location_t const& location) noexcept {
        return {vectors_[location], vectors_.columns() * nk::dimensions_per_value<scalar_t>()};
    }

    /** Copy of the pre-computed norms of the vector stored at @p location - zeroed moments, without
     *  touching memory, when the rows keep none. */
    inline vector_norms_t norm_for(location_t const& location) const noexcept {
        if (!stores_norms()) return {};
        return norms_[location];
    }

    /** Mutable reference to the norms slot at @p location - only for the slot's unpublished owner. */
    inline vector_norms_t& norm_ref_for(location_t const& location) noexcept { return norms_[location]; }

    /** Typed view over the fixed-capacity neighbors at @p location - atomic by default, plain
     *  under a mutation-free contract. @ref dense_ram_index_concurrency */
    template <synchronization_t synchronization_ = synchronization_t::concurrent_k>
    inline neighbors_ref<synchronization_> neighbors_for(location_t const& location) noexcept {
        return {node_ids_span_t {neighbors_[location], neighbors_.columns()}};
    }

    /** Immutable view of the fixed-capacity neighbors at @p location. */
    inline node_ids_view_t neighbors_for(location_t const& location) const noexcept {
        return {neighbors_[location], neighbors_.columns()};
    }

    /**
     *  @brief Allocates & publishes chunks until at least @p minimal_capacity fresh rows are added,
     *      following the growth schedule of @ref dense_ram_index_malloc.
     *  @return The number of added rows - possibly non-zero even when the status is a failure,
     *      in which case the caller must still release them back into @c remaining_capacity_.
     *
     *  Only one thread may run this at a time - the one that dropped @c remaining_capacity_ below
     *  zero. Successive growers are ordered by the acquire/release pair on that counter, which is
     *  also what publishes the non-atomic @c chunks_count_. Readers reach a chunk only once a node
     *  inside it becomes visible, which chains through the same atomics.
     */
    expected<std::size_t> create_next_chunk(std::size_t minimal_capacity) noexcept {
        // Growing before the geometry is frozen is a usage error, not an allocation failure.
        geometry_t const frozen = geometry();
        if (!frozen) return {0, status_t::uninitialized_k};

        std::size_t added_capacity = 0;
        while (added_capacity < minimal_capacity) {
            if (chunks_count_ == max_chunks_k) return {added_capacity, status_t::overflow_k};

            std::size_t const covered_rows = chunk_schedule_t::rows_before(chunks_count_ + 1u);
            // Norms exist only when the finalize reads them, which a dot does on compensated kits
            // alone. The all-ones neighbors fill is load-bearing: a sentinel leading slot reads as
            // unpublished, and sentinel tails terminate short rows, so no counts are stored.
            static_assert(node_id_unknown_k == static_cast<node_id_t>(~node_id_t {0}));
            status_t covered = vectors_.cover(covered_rows, padded_row_values(frozen));
            if (succeeded(covered) && stores_norms()) covered = norms_.cover(covered_rows);
            if (succeeded(covered)) covered = neighbors_.cover(covered_rows, frozen.connectivity(), 0xFF);
            if (failed(covered)) return {added_capacity, covered};

            added_capacity += chunk_schedule_t::rows_in(chunks_count_);
            ++chunks_count_;
            // The all-ones ID is the sentinel: account one row less in the final chunk, so no
            // claim can dispense it and the dispenser never wraps.
            if (chunks_count_ == max_chunks_k) added_capacity -= 1;
        }
        return {added_capacity, status_t::success_k};
    }

    /**
     *  @brief Claims storage capacity for @p count new nodes, growing chunks when needed.
     *  @return The first row of the claim - only meaningful on @c success_k, and rarely needed:
     *      identity is dispensed separately.
     *
     *  Spins while other threads are mid-growth, covering all interleavings of the capacity state
     *  machine. Claims @b storage only, never identity: node IDs are dispensed per batch from
     *  @c next_dispensed_id_ by @ref dispense_. On success, enough sentinel-filled rows exist for
     *  @p count dispenses, and the caller must roll forward to wiring & publishing them all -
     *  claimed capacity cannot be released once IDs are dispensed against it, or the
     *  visibility watermark would stall forever. @ref dense_ram_index_concurrency
     */
    expected<node_id_t> claim_capacity(std::size_t count) noexcept {
        // Rejected before the counter dance, so it neither grows the index in vain nor overflows
        // the signed capacity arithmetic below.
        if (count > max_node_count_k) return {node_id_unknown_k, status_t::overflow_k};
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
                    return {node_id_unknown_k, chunk_status};
                }
                // The take releases too: a claimer that fit on the counter before the growth but whose rows this
                // take pushes into the new chunk reaches its pointer through this word alone.
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
                               traversal_context_t& context) const noexcept {
        std::size_t const computed_distances = select_diverse_via_quads(candidates, limit, out, context);
        fill_selection(to_span(candidates), out, limit);
        return computed_distances;
    }

    /** Appends the nearest rejected candidates of the ascending @p pool to the diverse prefix in
     *  @p out until @p limit or the pool runs out. The prefix is a subsequence of the pool, so
     *  one merge-walk tells a kept candidate from a rejected one. */
    static void fill_selection(std::span<scored_node_t const> pool, selection_t& out, std::size_t limit) noexcept {
        std::size_t const diverse_count = out.size();
        if (diverse_count == limit || nk::failed(out.resize({limit}))) return;
        std::size_t filled_count = diverse_count;
        std::size_t matched = 0;
        for (scored_node_t const candidate : pool) {
            if (filled_count == limit) break;
            if (matched != diverse_count && out[matched].node_id == candidate.node_id) ++matched;
            else out[filled_count++] = candidate;
        }
        // Shrinking within the reservation never fails - the fill becomes the live shape.
        [[maybe_unused]] nk::status_t const shrunk = out.resize({filled_count});
    }

    /** The diverse prefix of @c select_diverse: pair distances computed on the fly with the 4-way
     *  kernels, the greedy scan deciding as they arrive. The kept prefix's views & norms ride the
     *  staging surface, free during any selection. */
    std::size_t select_diverse_via_quads(results_array_t const& candidates, std::size_t limit, selection_t& out,
                                         traversal_context_t& context) const noexcept {
        if (nk::failed(out.resize({limit}))) return 0;
        vector_view_t* const kept_vectors = context.staged_vectors.data();
        vector_norms_t* const kept_norms = context.staged_norms.data();
        small_size_t kept_count = 0;
        std::size_t computed_distances = 0;
        scored_node_t const* candidates_data = candidates.data();
        for (std::size_t index = 0; index != candidates.size() && kept_count != limit; ++index) {
            scored_node_t const candidate = candidates_data[index];
            location_t const candidate_location = chunk_schedule_t::locate(candidate.node_id);
            vector_view_t const candidate_vector = vector_for(candidate_location);
            vector_norms_t const candidate_norms = norm_for(candidate_location);
            // 4-way batches over the kept neighbors' cached views; short tails repeat the last
            // one. A batch may score past the first violation, still cheaper than four serial.
            small_size_t kept = 0;
            while (kept != kept_count) {
                std::array<distance_t, 4> const kept_distances = distances_via_quads( //
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
     *  @brief The vector-engine scoring of one node's staged neighbors against the node itself:
     *      4-way sweeps feeding the sorted re-selection pool with real node IDs, the
     *      newcomer entering on its reused symmetric score.
     *  @return The distances computed, for the caller's telemetry.
     */
    std::size_t stage_scores_via_quads(vector_view_t neighbor_vector, vector_norms_t neighbor_norms,
                                       node_id_t new_node_id, distance_t newcomer_score,
                                       traversal_context_t& context) const noexcept {
        results_array_t& refining_candidates = context.refining_candidates;
        small_size_t const count = static_cast<small_size_t>(context.staged_ids.size());
        small_size_t const pool_limit = count + 1;
        refining_candidates.insert(scored_node_t(new_node_id, newcomer_score), pool_limit);
        std::size_t computed = 0;
        for (small_size_t scored = 0; scored != count;) {
            std::array<distance_t, 4> const staged_distances = distances_via_quads( //
                neighbor_vector, neighbor_norms, context.staged_vectors.data(), context.staged_norms.data(), scored,
                count);
            small_size_t const lanes_count = min_of<small_size_t>(count - scored, 4u);
            for (small_size_t lane = 0; lane != lanes_count; ++lane)
                refining_candidates.insert(scored_node_t(context.staged_ids[scored + lane], staged_distances[lane]),
                                           pool_limit);
            scored += lanes_count;
            computed += lanes_count;
        }
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
    void back_link_neighbors(node_id_t new_node_id, selection_t const& selected,
                             traversal_context_t& context) noexcept {
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
                if (existing_id == node_id_unknown_k) break;
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
            // newcomer stays out: its symmetric score was already recorded at selection time.
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
            refine_distances = stage_scores_via_quads(neighbor_vector, neighbor_norms, new_node_id,
                                                      selected[selected_index].distance, context);
            refine_distances += select_diverse_via_quads(refining_candidates, node_connectivity, refined, context);
            context.computed_distances += refine_distances;
            context.computed_distances_in_reverse_refines += refine_distances;

            // Rewrite the tail slots, sentinel-terminate any shrink, and unlock with the new head.
            // A non-empty refined fill always holds: the heuristic never rejects the nearest candidate.
            small_size_t const refined_count = static_cast<small_size_t>(refined.size());
            for (small_size_t slot = 1; slot < refined_count; ++slot) neighbors.store_slot(slot, refined[slot].node_id);
            for (small_size_t slot = refined_count; slot < slots_count; ++slot)
                neighbors.store_slot(slot, node_id_unknown_k);
            neighbors.unlock(refined[0].node_id);
        }
    }

    /** Dispenses one range of IDs for every row @p claim holds with one add, and consumes the claim;
     *  a batch publishes that range whole. Every ID it hands out must be published, so nothing after
     *  it may fail. */
    iota_view<node_id_t> dispense_(capacity_claim_t&& claim) noexcept {
        node_id_t const rows = static_cast<node_id_t>(claim.rows());
        claim.consume();
        // The `acq_rel` order chains the hosting chunks' growth through the all-RMW dispenser to this thread.
        node_id_t const first = atomic_ref<node_id_t>(next_dispensed_id_).fetch_add(rows, std::memory_order_acq_rel);
        return {first, static_cast<node_id_t>(first + rows)};
    }

    /**
     *  @brief Wires the dispensed row @p new_node_id: lands @p new_vector and its norms, walks the graph,
     *      writes the neighbors and unlocks slot zero, then back-links. Moves no watermark, so the
     *      row's range publishes whole through @ref publish_.
     *
     *  The walk admits the rows below its watermark snapshot and the rows dispensed before it began
     *  whose slot zero shows them published. The row is already dispensed, so a refused walk wires a
     *  thinner node rather than abandoning an ID nothing will ever publish.
     */
    void wire_(node_id_t new_node_id, vector_view_t new_vector, traversal_context_t& context,
               insert_config_t config) noexcept {
        geometry_t const frozen = geometry();
        location_t const new_location = chunk_schedule_t::locate(new_node_id);

        // The scratch structures persist across rows and calls - reset them for this one.
        context.results.clear();
        context.refining_candidates.clear();
        context.frontier.clear();
        context.visited.clear();

        // Norms derive from the owned copy - the exact bytes future distance computations
        // stream. Both sides share `scalar_t`, so this is a byte move, never a conversion.
        vector_span_t const owned_vector = vector_span_for(new_location);
        std::memcpy(owned_vector.byte_data(), new_vector.byte_data(), stored_row_values(frozen) * sizeof(scalar_t));
        vector_norms_t const new_norms = norms(owned_vector);
        if (stores_norms()) norm_ref_for(new_location) = new_norms;

        // The first row of a virgin index finds nothing to walk to and publishes with no neighbors;
        // later rows wire back to it.
        node_id_t const first_invisible_id = atomic_ref<node_id_t>(last_visible_id_).load(std::memory_order_acquire);
        node_id_t const first_undispensed_id =
            atomic_ref<node_id_t>(next_dispensed_id_).load(std::memory_order_acquire);
        dispensed_admission_t const admission {this, first_invisible_id, first_undispensed_id, new_node_id};
        if (failed(search_around(vector_view_t(owned_vector), new_norms, admission, non_zero_expansion_factor(config),
                                 context, context.results)))
            ++context.saturated_walks;

        // No reader can arrive before slot 0 turns non-sentinel, so relaxed stores into slots
        // 1.. need no lock and the untouched tail keeps its sentinel termination.
        selection_t& selected = context.selected_neighbors;
        std::size_t const refine_distances = select_diverse(context.results, frozen.connectivity(), selected, context);
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

        // Back-link before the range's turn: once it publishes, no thread edits a published row on
        // its behalf, which the snapshot's settlement probe rests on. Every neighbor lock is
        // released inside, so the turn's wait holds none.
        back_link_neighbors(new_node_id, selected, context);
    }

    /** Waits for @p range 's turn: every range dispensed ahead of it has published, which keeps the
     *  visible prefix hole-free. Holds no lock while waiting. */
    void await_turn_(iota_view<node_id_t> range) noexcept {
        for (node_id_t watermark = atomic_ref<node_id_t>(last_visible_id_).load(std::memory_order_acquire);
             watermark != range.first;
             watermark = atomic_ref<node_id_t>(last_visible_id_).load(std::memory_order_acquire))
            // Uncapped: every predecessor's publication store wakes us to re-check.
            wait_type_ {}(&last_visible_id_, watermark, 0u, fu::wait_uncapped_t {});
    }

    /** Publishes every row of @p range at once, its turn having come: one release store moves the
     *  watermark past them. */
    void publish_(iota_view<node_id_t> range) noexcept {
        atomic_ref<node_id_t>(last_visible_id_).store(range.bound, std::memory_order_release);
    }

#pragma endregion Insertion

#pragma region Refinement

    /**
     *  @brief Stages the live prefix of @p neighbors into @c context.staged_ids, waiting out a
     *      lock holder like any traversal does; a published-empty self-loop stages nothing.
     *  @return The staged count - the tensor's live shape.
     */
    small_size_t stage_neighbors_(neighbors_ref_t neighbors, node_id_t node_id,
                                  traversal_context_t& context) const noexcept {
        ids_tensor_t& staged_ids = context.staged_ids;
        small_size_t const slots_count = neighbors.capacity();
        if (nk::failed(staged_ids.resize({slots_count}))) return 0;
        node_id_t first_value = neighbors.peek_first();
        while (first_value == node_id_unknown_k) {
            neighbors.wait_while_locked();
            first_value = neighbors.peek_first();
        }
        small_size_t staged_count = 0;
        if (first_value != node_id) {
            staged_ids[staged_count++] = first_value;
            while (staged_count != slots_count) {
                node_id_t const slot_value = neighbors.load_slot(staged_count);
                if (slot_value == node_id_unknown_k) break;
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
                     traversal_context_t& context) noexcept {
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
        if (failed(search_around(own_vector, own_norms, snapshot_admission_t {first_invisible_id}, expansion_factor,
                                 context, context.results, dummy_fold_t {}, seeds)))
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
            own_neighbors.store_slot(slot, node_id_unknown_k);
        own_neighbors.unlock(selected_count ? selected[0].node_id : node_id);

        back_link_neighbors(node_id, selected, context);
        return changed;
    }

#pragma endregion Refinement

#pragma region Search

    /** The body of @c search_typed below the watermark snapshot @p first_invisible_id the caller took:
     *  the index's own, or one consistent across shards. */
    template <synchronization_t synchronization_, typename fold_type_, typename rescoring_type_>
    expected<search_result_t> search_below_(vector_view_t query_vector, traversal_context_t& context,
                                            search_config_t config, fold_type_&& fold, results_array_t const& bound,
                                            rescoring_type_ const& rescoring, node_id_t first_invisible_id) noexcept {
        if (config.traversal != traversal_t::walk_k) return {{}, status_t::unsupported_k};
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

        // An unfrozen geometry read before a racing first insert left the query width unchecked.
        if (first_invisible_id == 0 || !frozen) return {{}, status_t::success_k};

        // Snapshot the context's running telemetry, to report this query's share on return.
        std::size_t const visited_members_before = context.visited_members;
        std::size_t const computed_distances_before = context.computed_distances;
        std::size_t const restarts_before = context.restarts;
        std::size_t const saturated_walks_before = context.saturated_walks;

        vector_norms_t const query_norms = norms(query_vector);
        status_t const walked = search_around<synchronization_>(query_vector, query_norms,
                                                                snapshot_admission_t {first_invisible_id},
                                                                expansion_factor, context, bound, fold, {}, rescoring);

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

    /** The one dispatch of a walk under @p fold and @p rescoring onto @c search_below_, by the contract @p config
     *  names, below the watermark snapshot @p first_invisible_id and bounded by @c context.results. */
    template <typename fold_type_, typename rescoring_type_>
    expected<search_result_t> walk_(vector_view_t query_vector, fold_type_&& fold, rescoring_type_ const& rescoring,
                                    traversal_context_t& context, search_config_t config,
                                    node_id_t first_invisible_id) noexcept {
        if (config.synchronization != synchronization_t::concurrent_k)
            return search_below_<synchronization_t::immutable_k>(query_vector, context, config, fold, context.results,
                                                                 rescoring, first_invisible_id);
        return search_below_<synchronization_t::concurrent_k>(query_vector, context, config, fold, context.results,
                                                              rescoring, first_invisible_id);
    }

    /** The filtered walk behind @c search_if, below the watermark snapshot @p first_invisible_id. */
    template <typename predicate_type_>
    expected<search_result_t> walk_if_(vector_view_t query_vector, predicate_type_& predicate,
                                       traversal_context_t& context, search_config_t config,
                                       node_id_t first_invisible_id) noexcept {
        // Matches admit into the re-selection buffer, idle during any search and reserved beside the
        // navigation pool; it bounds the walk once full, then swaps into `context.results`.
        context.refining_candidates.clear();
        filtering_fold<predicate_type_, results_array_t> filtering {predicate, context.refining_candidates,
                                                                    non_zero_expansion_factor(config)};
        expected<search_result_t> const result =
            config.synchronization != synchronization_t::concurrent_k
                ? search_below_<synchronization_t::immutable_k>(query_vector, context, config, filtering,
                                                                context.refining_candidates, dummy_rescoring_t {},
                                                                first_invisible_id)
                : search_below_<synchronization_t::concurrent_k>(query_vector, context, config, filtering,
                                                                 context.refining_candidates, dummy_rescoring_t {},
                                                                 first_invisible_id);
        std::swap(context.results, context.refining_candidates);
        return result;
    }

    /** The exact sibling of @c walk_if_: one query against every row below @p first_invisible_id,
     *  the nearest breadth of rows that @p admits lets in kept in @c context.results. */
    template <typename admits_type_>
    expected<search_result_t> scan_if_(vector_view_t query_vector, admits_type_&& admits, traversal_context_t& context,
                                       search_config_t config, node_id_t first_invisible_id) noexcept {
        geometry_t const frozen = geometry();
        if (frozen && query_vector.size() != frozen.dimensions()) return {{}, status_t::config_mismatch_k};
        std::size_t const expansion_factor = non_zero_expansion_factor(config);
        if (status_t const sized = reserve_queues_(context, queue_sizes_(expansion_factor, connectivity()));
            failed(sized))
            return {{}, sized};
        context.results.clear();
        if (first_invisible_id == 0 || !frozen) return {{}, status_t::success_k};

        std::size_t const computed_distances_before = context.computed_distances;
        auto land = [&](std::size_t, scored_node_t candidate) noexcept {
            [[maybe_unused]] bool const kept = context.results.insert(std::move(candidate), expansion_factor);
            return context.results.size() == expansion_factor ? context.results.top().distance
                                                              : std::numeric_limits<distance_t>::max();
        };
        if (status_t const scanned = scan_quads_(query_vector, admits, land, context, first_invisible_id);
            failed(scanned))
            return {{}, scanned};
        search_result_t result;
        result.count = context.results.size();
        result.computed_distances = context.computed_distances - computed_distances_before;
        return {result, status_t::success_k};
    }

    /** Queries one exact-scan block takes: a whole number of tiles when a matrix engine multiplies
     *  the scalar and @p queries_count fills one, shared across @p threads up to the kit's pack;
     *  else one query. */
    static constexpr std::size_t scan_width(std::size_t queries_count, std::size_t threads) noexcept {
        if constexpr (!tiles_chain_t::has_tiles_k) return 1;
        else {
            using tiles_kit_t = typename tiles_chain_t::kit_t;
            if (queries_count < tiles_kit_t::tile_columns_k) return 1;
            std::size_t const share = divide_round_up(queries_count, max_of<std::size_t>(threads, 1));
            return min_of(round_up_to_multiple(share, tiles_kit_t::tile_columns_k), tiles_kit_t::pack_columns_k);
        }
    }

    /**
     *  @brief The exact sibling of @c walk_: scores every row below @p first_invisible_id against
     *      each query of @p block, landing each row that beats a query's worst match through
     *      @p land.
     *
     *  A block of many queries streams through the matrix engine, a single query through the 4-way
     *  kits; both finalize through @c quads_kit_t, so the distances match the walk's. @p admits
     *  sees a row only once it beats some query's worst match, and at most once per block.
     */
    template <typename admits_type_, typename land_type_>
    status_t scan_(vectors_view_t block, admits_type_&& admits, land_type_&& land, traversal_context_t& context,
                   node_id_t first_invisible_id) noexcept {
        if constexpr (tiles_chain_t::has_tiles_k)
            if (block.extent(0) > 1) return scan_tiles_(block, admits, land, context, first_invisible_id);
        return scan_quads_(block.row(0).as_vector(), admits, land, context, first_invisible_id);
    }

    /** One query against every row, four rows per kernel call, each landing as it beats the worst
     *  match. */
    template <typename admits_type_, typename land_type_>
    status_t scan_quads_(vector_view_t query, admits_type_& admits, land_type_& land, traversal_context_t& context,
                         node_id_t first_invisible_id) noexcept {
        vector_norms_t const query_norms = norms(query);
        distance_t threshold = std::numeric_limits<distance_t>::max();
        for (std::size_t chunk = 0, first = 0; first < first_invisible_id;
             first += chunk_schedule_t::rows_in(chunk++)) {
            std::size_t const rows_count = min_of(chunk_schedule_t::rows_in(chunk), first_invisible_id - first);
            for (std::size_t row = 0; row < rows_count; row += 4) {
                std::size_t const lanes = min_of<std::size_t>(4, rows_count - row);
                std::array<vector_view_t, 4> lane_rows;
                std::array<vector_norms_t, 4> lane_norms;
                for (std::size_t lane = 0; lane != 4; ++lane) {
                    location_t const location {chunk, row + min_of(lane, lanes - 1)};
                    lane_rows[lane] = vector_for(location);
                    lane_norms[lane] = norm_for(location);
                }
                std::array<distance_t, 4> const lane_distances = distances(query, query_norms, lane_rows, lane_norms);
                for (std::size_t lane = 0; lane != lanes; ++lane) {
                    node_id_t const node = static_cast<node_id_t>(first + row + lane);
                    if (lane_distances[lane] < threshold && admitted(admits, node))
                        threshold = land(0, {node, lane_distances[lane]});
                }
            }
        }
        context.computed_distances += first_invisible_id;
        return status_t::success_k;
    }

    /** A block of queries packed once, and every chunk streamed against the pack through
     *  @c scan_tiles. */
    template <typename admits_type_, typename land_type_>
    status_t scan_tiles_(vectors_view_t block, admits_type_& admits, land_type_& land, traversal_context_t& context,
                         node_id_t first_invisible_id) noexcept
        requires tiles_chain_t::has_tiles_k
    {
        using tiles_kit_t = typename tiles_chain_t::kit_t;
        std::size_t const columns = block.extent(0), dimensions = this->dimensions();
        nk_size_t packed_bytes = 0;
        if (tiles_kit_t::pack_size(tiles_kit_t::pack_columns_k, dimensions, &packed_bytes) != nk_success_k ||
            failed(context.reserve_tiles(packed_bytes, tiles_kit_t::stream_rows_k, tiles_kit_t::pack_columns_k)))
            return status_t::bad_alloc_k;
        [[maybe_unused]] nk_status_t const packed = tiles_kit_t::pack(
            block.raw_values_data(), columns, dimensions, static_cast<std::size_t>(block.stride_bytes(0)),
            context.packed_queries.data(), 0, columns, nullptr);
        assert(packed == nk_success_k && "A matrix engine packs any block");

        alignas(16) std::array<dot_result_t, tiles_kit_t::pack_columns_k> query_sumsqs {};
        std::array<distance_t, tiles_kit_t::pack_columns_k> thresholds;
        for (std::size_t column = 0; column != columns; ++column) {
            query_sumsqs[column] = norms(block.row(column).as_vector()).sumsq;
            thresholds[column] = std::numeric_limits<distance_t>::max();
        }
        for (std::size_t chunk = 0, first = 0; first < first_invisible_id;
             first += chunk_schedule_t::rows_in(chunk++)) {
            std::size_t const chunk_rows = min_of(chunk_schedule_t::rows_in(chunk), first_invisible_id - first);
            // Chunk rows are padded to whole kit steps, while the queries were packed at
            // the logical width.
            vectors_view_t const padded {vectors_[location_t {chunk, 0}], chunk_schedule_t::rows_in(chunk),
                                         vectors_.columns() * nk::dimensions_per_value<scalar_t>()};
            nk::shape_storage_<2> const logical {{chunk_rows, dimensions},
                                                 {padded.stride_bytes(0), padded.stride_bytes(1)},
                                                 2};
            scan_tiles<tiles_kit_t, quads_kit_t>(
                kernel_, vectors_view_t {padded.byte_data(), logical}, static_cast<node_id_t>(first),
                context.packed_queries.data(), std::span<dot_result_t const>(query_sumsqs).first(columns),
                std::span<distance_t>(thresholds).first(columns), context.step_dots.span(),
                [&](std::size_t row) noexcept { return norm_for({chunk, row}).sumsq; }, admits, land);
        }
        context.computed_distances += first_invisible_id * columns;
        return status_t::success_k;
    }

    /** What a search walks: the IDs below its watermark snapshot, published for every reader. */
    struct snapshot_admission_t {

        /** The watermark snapshot. */
        node_id_t first_invisible_id = 0;

        /** Whether the walk may locate @p node_id. */
        bool admits(node_id_t node_id) const noexcept { return node_id < first_invisible_id; }

        /** Landmarks come from the IDs below this bound, the snapshot. */
        node_id_t sampling_bound() const noexcept { return first_invisible_id; }
    };

    /** What an insertion walks: the IDs below its snapshot, and past it every row dispensed before the walk
     *  began whose slot zero shows it published. The acquire of the dispenser carried those rows' chunks to
     *  the walk, so batches wired side by side link to each other rather than forming islands. */
    struct dispensed_admission_t {

        /** The index the walk reads. */
        ram_index* index = nullptr;

        /** The watermark snapshot. */
        node_id_t first_invisible_id = 0;

        /** The dispenser as the walk began: no row at or past it is reachable. */
        node_id_t first_undispensed_id = 0;

        /** The walking row. */
        node_id_t node_id = 0;

        /** Whether the walk may locate @p candidate_id; the acquire of its slot zero carries its payload. */
        bool admits(node_id_t candidate_id) const noexcept {
            if (candidate_id < first_invisible_id) return true;
            if (candidate_id >= first_undispensed_id) return false;
            return index->neighbors_for(chunk_schedule_t::locate(candidate_id)).peek_first() != node_id_unknown_k;
        }

        /** Landmarks come from the IDs below this bound, every one dispensed ahead of the walking row, which
         *  rows wired in dispensed order have mostly published. */
        node_id_t sampling_bound() const noexcept { return node_id; }
    };

    /**
     *  @brief Samples one 4-way batch of distinct, unvisited, published landmarks, scores them
     *      against the @p query, and queues them into the frontier & the result pool.
     *  @return @c advanced_k, @c exhausted_k once every published node is already visited, or
     *      @c sated_k once the fold asks the walk to end - the last two end the walk rather than
     *      failing it. A failed status means scratch could not grow.
     *
     *  Uniform draws that collide with @c visited fall back to cyclic linear probing below the
     *  bound @p admission sets, so every call makes strict progress without retry budgets. Only
     *  transiently locked nodes are re-rolled, like the traversal itself; past the snapshot a row
     *  may stay unpublished for the whole walk, so those re-rolls stop once they outnumber the IDs
     *  there.
     */
    template <synchronization_t synchronization_ = synchronization_t::concurrent_k,
              typename admission_type_ = snapshot_admission_t, typename fold_type_ = dummy_fold_t,
              typename rescoring_type_ = dummy_rescoring_t>
    expected<walk_step_t> sample_landmarks(vector_view_t query, vector_norms_t query_norms,
                                           admission_type_ const& admission, std::size_t expansion_factor,
                                           traversal_context_t& context, std::span<node_id_t const>& seeds,
                                           fold_type_&& fold = fold_type_ {},
                                           rescoring_type_ const& rescoring = {}) noexcept {
        results_array_t& results = context.results;
        frontier_queue_t& frontier = context.frontier;
        visited_set_t& visited = context.visited;

        std::array<node_id_t, 4> landmarks_ids;
        std::array<vector_view_t, 4> landmarks_vectors;
        std::array<vector_norms_t, 4> landmarks_norms;
        small_size_t landmarks_in_batch = 0;

        if (status_t const seen = visited.reserve(visited.size() + landmarks_ids.size()); failed(seen))
            return {walk_step_t::exhausted_k, seen};
        // `visited` may also hold IDs no draw names, like an insertion's later siblings, so the size
        // check is conservative: it can only under-sample, never let the probe loop spin.
        node_id_t const sampling_bound = admission.sampling_bound();
        node_id_t unpublished_draws = 0;
        while (landmarks_in_batch != landmarks_ids.size() && visited.size() < sampling_bound) {
            // Caller-staged seeds go first, each taken once and skipped when unpublished or
            // already visited; past them, draw uniformly and probe forward cyclically.
            node_id_t landmark_node_id;
            if (!seeds.empty()) {
                landmark_node_id = seeds.front();
                seeds = seeds.subspan(1);
                if (!admission.admits(landmark_node_id) || visited.contains(landmark_node_id)) continue;
            }
            else {
                landmark_node_id = uniform_below<node_id_t> {sampling_bound}(context.random_generator);
                while (visited.contains(landmark_node_id))
                    landmark_node_id = landmark_node_id + 1 == sampling_bound ? 0 : landmark_node_id + 1;
            }

            location_t const landmark_location = chunk_schedule_t::locate(landmark_node_id);
            // Published nodes may be transiently locked for re-linking - re-roll those. Under a
            // mutation-free contract every sampled node is published & unlocked - no probe needed.
            if constexpr (synchronization_ == synchronization_t::concurrent_k)
                if (neighbors_for(landmark_location).peek_first() == node_id_unknown_k) {
                    if (landmark_node_id >= admission.first_invisible_id &&
                        ++unpublished_draws == sampling_bound - admission.first_invisible_id)
                        break;
                    continue;
                }

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
        scores_result<dot_result_t, 4> landmark_scores = scores( //
            query, query_norms, landmarks_vectors, landmarks_norms);
        rescoring(landmarks_ids, landmarks_vectors, landmarks_norms, landmark_scores, landmarks_in_batch);
        context.computed_distances += landmarks_in_batch;
        std::array<scored_node_t, 4> landmarks_scored;
        for (small_size_t lane = 0; lane != landmarks_in_batch; ++lane)
            landmarks_scored[lane] = scored_node_t(landmarks_ids[lane], landmark_scores.distances[lane]);
        std::span const scored = std::span(landmarks_scored).first(landmarks_in_batch);
        // Landmarks fold right here, in lane order: they are marked visited, so the walk never
        // re-scores them and this is their one chance to be absorbed.
        if (fold.absorb(scored, std::span<dot_result_t const>(landmark_scores.dots).first(landmarks_in_batch)) ==
            walk_demand_t::sated_k)
            return {walk_step_t::sated_k, status_t::success_k};
        network_sort(scored, closer_scored_node_t {});
        results.insert(scored, expansion_factor);
        for (small_size_t lane = 0; lane != landmarks_in_batch; ++lane)
            if (status_t const queued = frontier.insert(scored_node_t(scored[lane])); failed(queued))
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
     *  @param[in] admission What the walk may locate and draw landmarks from: the IDs below the
     *      caller's watermark snapshot, and for an insertion the published rows dispensed before it.
     *  @param[in] expansion_factor Breadth of the candidate pool - the walk's stopping radius.
     *  @param[inout] context Scratch & telemetry; must arrive pre-reserved & pre-cleared, and ends
     *      with up to @p expansion_factor nearest published candidates in @c context.results.
     *  @param[in] bound The pool whose fill ends restarts and whose radius bounds the walk once full:
     *      @c context.results itself, or a filtered walk's admitted pool.
     *  @param[in] fold Optional reader of every scored quad - see @c fold. It never steers the
     *      walk, except that a @c sated_k from it ends the walk successfully.
     *  @param[in] seeds Optional landmarks to start from ahead of the random draws - a
     *      re-selection hands in the node's own neighbors, so every one already wired competes.
     *  @param[in] rescoring Optional rescoring of every scored node in this one walk - an indexer's heads or
     *      one token's attention heads.
     */
    template <synchronization_t synchronization_ = synchronization_t::concurrent_k,
              typename admission_type_ = snapshot_admission_t, typename fold_type_ = dummy_fold_t,
              typename rescoring_type_ = dummy_rescoring_t>
    status_t search_around(vector_view_t query, vector_norms_t query_norms, admission_type_ const& admission,
                           std::size_t expansion_factor, traversal_context_t& context, results_array_t const& bound,
                           fold_type_&& fold = fold_type_ {}, std::span<node_id_t const> seeds = {},
                           rescoring_type_ const& rescoring = {}) noexcept {
        results_array_t& results = context.results;
        frontier_queue_t& frontier = context.frontier;
        visited_set_t& visited = context.visited;
        std::size_t const node_connectivity = connectivity();
        assert(results.capacity() >= expansion_factor && "A pool short of the expansion never tightens the radius");

        // About as many landmarks as the connectivity - a stochastic multi-start in 4-way batches.
        small_size_t const landmark_batches = divide_round_up<small_size_t>(node_connectivity, 4);
        for (small_size_t landmarks_batch = 0; landmarks_batch != landmark_batches; ++landmarks_batch) {
            auto const [seeding, seed_status] = sample_landmarks<synchronization_>(
                query, query_norms, admission, expansion_factor, context, seeds, fold, rescoring);
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
                    query, query_norms, admission, expansion_factor, context, seeds, fold, rescoring);
                if (failed(seed_status)) return seed_status;
                if (seeding == walk_step_t::sated_k) return status_t::success_k;
                if (seeding == walk_step_t::exhausted_k) break; // Nothing fresh left to restart from.
                ++context.restarts;
                continue;
            }

            // Peek - don't pop yet: the chosen node's neighbors may be locked, in which case
            // we retry the selection rather than dropping it from the frontier.
            scored_node_t const candidate = frontier.top();
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
                if (first_neighbor_id == node_id_unknown_k) {
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
                    if (neighbor_id == node_id_unknown_k) {
                        list_ended = true;
                        break;
                    }
                    // Translation-frontier invariant: only IDs below the watermark snapshot, or the published
                    // rows dispensed before an insertion's walk, may be located. A fresher one has no
                    // happens-before edge to its hosting chunk yet, so skip it; a later traversal meets it.
                    if (!admission.admits(neighbor_id)) continue;
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
                scores_result<dot_result_t, 4> gathered_scores = scores( //
                    query, query_norms, gathered_vectors, gathered_norms,
                    vectors_quad_of(candidate_neighbors, neighbor_slot, admission.first_invisible_id,
                                    gathered_vectors));
                rescoring(gathered_ids, gathered_vectors, gathered_norms, gathered_scores, gathered_count);
                context.computed_distances += gathered_count;
                std::array<scored_node_t, 4> scored_batch;
                for (small_size_t lane = 0; lane != gathered_count; ++lane)
                    scored_batch[lane] = scored_node_t(gathered_ids[lane], gathered_scores.distances[lane]);
                std::span const scored = std::span(scored_batch).first(gathered_count);
                // Folding is independent of navigation: a fold reads every scored quad in lane order,
                // including what the navigation pool rejects, and never decides where the walk routes.
                if (fold.absorb(scored, std::span<dot_result_t const>(gathered_scores.dots).first(gathered_count)) ==
                    walk_demand_t::sated_k)
                    return status_t::success_k;
                network_sort(scored, closer_scored_node_t {});

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
                    if (status_t const queued = frontier.insert(scored_node_t(scored[lane])); failed(queued))
                        return queued;
            }
        }
        return status_t::success_k;
    }

#pragma endregion Search

    /** One row per vector, padded to whole kit steps. @ref dense_ram_index_malloc */
    chunked_matrix<scalar_t, chunk_schedule_t, max_node_count_k, vectors_allocator_type_> vectors_;

    /** One entry per vector's pre-computed norms, covered only when @ref stores_norms. */
    chunked_column<vector_norms_t, chunk_schedule_t, max_node_count_k, vectors_allocator_type_> norms_;

    /** One row per vector's fixed-capacity neighbors, sentinel-filled. */
    chunked_matrix<node_id_t, chunk_schedule_t, max_node_count_k, neighbors_allocator_type_> neighbors_;

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

    /** The ID dispenser: the first identifier of the next batch's range. One add per batch moves
     *  it past every row of that batch, so a batch's IDs are contiguous. @ref dense_ram_index_concurrency */
    alignas(default_alignment_k) node_id_t next_dispensed_id_ {0};

    /** The visibility watermark: every ID strictly below it is published, its vector, norms &
     *  neighbors readable - the safe bound for random landmark sampling. Advanced strictly in
     *  dispensed order by each batch taking its turn. @ref dense_ram_index_concurrency */
    alignas(default_alignment_k) node_id_t last_visible_id_ {0};

    /** Read-mostly after the freeze - kept off the watermark's line with the other cold members. */
    alignas(default_alignment_k) geometry_t geometry_ {};

    /** The distance kind every walk ranks by; only the distance epilogue and the norms chunks
     *  read it. */
    nk_kernel_kind_t kernel_;

    /** Chunks every column covers. Deliberately non-atomic: only the elected grower mutates it,
     *  and successive growers are ordered by the acquire/release pair on @c remaining_capacity_,
     *  which no release lifts while one is inside. @ref create_next_chunk */
    small_size_t chunks_count_ = 0;
};

/**
 *  @brief External reader's cursor over the watermark of one @c ram_index:
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
struct ram_cursor {

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
    explicit ram_cursor(index_t const& index) noexcept
        : index_(index), visible_count_(atomic_ref<node_id_t const>(index.last_visible_id_).load(watermark_order_k)),
          current_(visible_count_), connectivity_(static_cast<small_size_t>(index.connectivity())) {}

    ram_cursor(ram_cursor const&) = delete;
    ram_cursor& operator=(ram_cursor const&) = delete;
    ram_cursor(ram_cursor&&) = default;
    ram_cursor& operator=(ram_cursor&&) = delete;

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
            while (first_value == node_id_unknown_k) {
                typename index_t::wait_t {}(row.data(), node_id_unknown_k, small_size_t(0), fu::wait_capped_t {});
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
            while (live != connectivity_ && staged_neighbors_[live] != node_id_unknown_k) ++live;
            staged_count_ = live;
        }
        return status_t::success_k;
    }

    /** Moves to the next ID below the snapshot, or past the end; a staging refusal parks on the
     *  new ID with no neighbors staged - a later @c seek to the same ID retries. */
    ram_cursor& operator++() noexcept {
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
    ram_cursor operator++(int) = delete;

    /** Follows the staged neighbor at @p slot: @c malformed_k when the slot is beyond the live
     *  prefix or its target is at or past this cursor's snapshot. */
    status_t follow(small_size_t slot) noexcept {
        if (slot >= staged_count_) return status_t::malformed_k;
        return seek(staged_neighbors_[slot]);
    }
};

} // namespace unum::usearch::dense
